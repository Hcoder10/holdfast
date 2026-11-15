"""State-program debugger: a language model debugs a world model by patching its state writes.

    cfg["debug"] = {"max": 3}

The fixed rules of repair.py are replaced by a model that works the way a person debugging the rollout would:

  1. run the episode with the current state program
  2. READ   the critic's verdict on the frames, and a state report measured from the rollout itself: how much each
            stretch of the video moves, how far the colours have drifted from the first frame, commanded against
            measured camera turns, which state writes fired (pins, registers)
  3. RECALL from past.dev what earlier debugging sessions found: symptom -> patch -> did the critic's score move
  4. PATCH  choose one edit to the state program from the write ports the controller has (below), with a
            one-sentence hypothesis
  5. run again, compare the critic's score, keep the better program, and WRITE the outcome back to past.dev, so
            the next scene with the same symptom starts from what worked and avoids what did not

Write ports the debugger may patch (each is a write into the model's state, see README):

  register_gain   0 | 4 | 8 | 16      attention to the request text, cross-attention logits (third-person requests)
  anchor          0 | 0.5 | 0.8       pull of each chunk's latent mean toward the first chunk's
  shadow          true | false        unseen view kept running in a forked state, pinned on return (first-person)
  landmarks       true | false        significant views kept and pinned on arrival (first-person)
  scene_action    none | impulse | timeline   Exa-derived action text in the register for still-camera scenes
  cam_gain        1 | 2 | 3           extrapolation of the camera conditioning when a commanded move is ignored
  new_noise       true | false        same program, different noise

The critic and the debugger see what the generator may see (caption, objective, perspective, frames, state reads);
never the benchmark rubric or a judge's score.
"""
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))

SYSTEM = (
    "You debug a video world model. You cannot change its weights; you can only patch its state program: a small "
    "set of writes into its attention, latents and conditioning. You are given the request, the critic's verdict "
    "on the last rollout, a state report measured from that rollout, the programs already tried with their "
    "scores, and notes from earlier debugging sessions. Choose ONE patch most likely to fix the worst defect. "
    "Prefer a patch that earlier notes say helped for the same symptom; do not repeat a patch that notes or this "
    "session show did not help. Reply with one JSON object and nothing else.")

MENU = """Patchable controls (give only the ones you change):
"register_gain": 0 | 4 | 8 | 16   attention to the request text. Raise when a visible third-person subject does not do
                                   what was asked; lower when the picture or subject breaks. Keep 0 for first-person
                                   or camera requests.
"anchor": 0 | 0.5 | 0.8           raise when colours drift over time; lower if the scene should change brightness.
"shadow": true | false            first-person only: keeps the view the camera left and writes it back on return.
"landmarks": true | false         first-person only: also keeps earlier significant views (start, ends of moves).
"scene_action": "none" | "impulse" | "timeline"   still camera only: what the subjects do, from a web answer, written
                                   into the register (impulse: first seconds only; timeline: early/middle/late).
"cam_gain": 1 | 2 | 3             raise when a commanded camera move (strafe, turn, walk) is not executed in the frames.
"new_noise": true                 same program, different noise; use when the program looks right and the defect
                                   looks like bad luck.
Also give "symptom": 3-8 words naming the defect class (e.g. "subject does not start walking", "scene differs on
return", "colours drift", "strafe ignored", "repeats in place"), and "hypothesis": one sentence.
JSON keys: symptom, hypothesis, and the changed controls."""


def state_report(res, frames):
    """Reads taken from the rollout itself, as text."""
    f = np.asarray(frames[:: max(1, len(frames) // 240)], dtype=np.float32)
    n = len(f)
    thirds = [f[i * n // 3:(i + 1) * n // 3] for i in range(3)]
    motion = [float(np.abs(np.diff(t, axis=0)).mean()) if len(t) > 1 else 0.0 for t in thirds]
    drift = float(np.abs(f[-1].mean(axis=(0, 1)) - f[0].mean(axis=(0, 1))).mean())
    back = float(np.abs(f[-1] - f[0]).mean())
    turns = [(r["label"], r.get("measured_yaw")) for r in res.get("records", []) if str(r.get("label", "")).startswith("yaw")]
    cmd = sum(float(str(l).split()[1]) for l, _ in turns) if turns else 0.0
    got = sum(float(m) for _, m in turns if m is not None) if turns else 0.0
    b = res.get("bank") or {}
    lines = [f"frame-to-frame motion by third of the video: {motion[0]:.1f}, {motion[1]:.1f}, {motion[2]:.1f} (below 2 is nearly frozen)",
             f"mean colour shift first to last frame: {drift:.1f} (above 12 is visible drift)",
             f"last frame vs first frame difference: {back:.1f} (below 20 means the video ends on the starting view)"]
    if turns:
        lines.append(f"camera turns commanded {cmd:+.0f} deg, measured {got:+.0f} deg over {len(turns)} chunks")
    if b:
        lines.append(f"state pins on return: {len(b.get('reads') or [])} (forks {b.get('forks')}, dropped {b.get('dropped')}, "
                     f"landmarks {b.get('landmarks')})")
    r = res.get("register") or {}
    if r:
        lines.append(f"request register gain {r.get('gain')} ({r.get('bias_hits')} attention calls biased)")
    return lines


def apply(cfg, patch, case):
    out = copy.deepcopy(cfg)
    third = (case.get("perspective") or "").strip().lower() == "third person"
    wait = (case.get("prompt") or "").lower().startswith("wait and observe")
    done = {}
    if "register_gain" in patch and third and not wait:
        g = min((0, 4, 8, 16), key=lambda c: abs(c - float(patch["register_gain"])))
        if g:
            out["register"] = {"gain": g, "when": "third person", "replace": True}
        else:
            out.pop("register", None)
        done["register_gain"] = g
    if "anchor" in patch:
        out["anchor_stats"] = min((0, 0.5, 0.8), key=lambda c: abs(c - float(patch["anchor"])))
        done["anchor"] = out["anchor_stats"]
    if ("shadow" in patch or "landmarks" in patch) and not third and not wait:
        bank = dict(out.get("bank") or {"via": "key", "tol": 18, "pos_tol": 1.2})
        if "shadow" in patch:
            bank["shadow"] = bool(patch["shadow"])
        if patch.get("landmarks"):
            bank["shadow"] = True
            bank["landmarks"] = {"tol": 12, "min_age": 5, "pos_tol": 0.5}
        elif "landmarks" in patch:
            bank.pop("landmarks", None)
        if bank.get("shadow"):
            out["bank"] = bank
        else:
            out.pop("bank", None)
        done.update({k: patch[k] for k in ("shadow", "landmarks") if k in patch})
    if "scene_action" in patch and wait:
        import planner
        sa = str(patch["scene_action"]).lower()
        if planner.SCENE_ACTION.get(sa):
            out["act_register"] = dict(planner.SCENE_ACTION[sa])
        else:
            out.pop("act_register", None)
        done["scene_action"] = sa
    if "cam_gain" in patch:
        g = min((1, 2, 3), key=lambda c: abs(c - float(patch["cam_gain"])))
        if g > 1:
            out["cam_gain"] = float(g)
        else:
            out.pop("cam_gain", None)
        done["cam_gain"] = g
    if patch.get("new_noise") or not done:
        out["_seed"] = out.get("_seed", 0) + 1
        done["new_noise"] = True
    return out, done


def recall(case, symptom=None):
    try:
        import mem
        q = f"debug session: symptom {symptom or ''}, patch tried and whether the critic score moved, for: {case.get('prompt', '')}"
        notes = []
        for h in mem.recall("controller", q, 8):
            for t in [h.get("text") or ""] + list(h.get("excerpts") or []):
                if "debug session" in t and t[:260] not in notes:
                    notes.append(" ".join(t.split())[:260])
        return notes[:6]
    except Exception as e:
        return [f"recall failed: {e!r}"[:100]]


def run(make_episode, stepper, case, image, seed, cfg, log=print):
    """Returns (result dict, frames) of the best attempt by the critic's score."""
    import critic
    import llm
    dbg = cfg.get("debug") or {}
    prog = {k: v for k, v in cfg.items() if k != "debug"}
    attempts, kept, notes0 = [], None, recall(case)
    for k in range(int(dbg.get("max", 3))):
        run_cfg = {key: val for key, val in prog.items() if key != "_seed"}
        ep = make_episode(run_cfg)
        res = ep.run(case, image, seed=seed + 101 * prog.get("_seed", 0))
        frames = stepper.video()
        try:
            v = critic.review(frames, case.get("image_caption", ""), case.get("prompt", ""),
                              case.get("perspective", ""), model=dbg.get("critic_model"))
        except Exception as e:
            v = {"score": -1.0, "note": "critic failed: " + repr(e)[:100], "action": 1, "subject": 1, "picture": 1, "alive": 1}
        report = state_report(res, frames)
        att = {"program": {kk: vv for kk, vv in prog.items() if kk != "prompt"}, "verdict": v, "state_report": report}
        if attempts and attempts[-1].get("patch"):
            prev = attempts[-1]
            delta = v["score"] - prev.get("base_score", prev["verdict"]["score"])
            att["delta_vs_previous"] = round(delta, 2)
            outcome = "helped" if delta > 0.4 else "hurt" if delta < -0.4 else "no change"
            try:
                import mem
                mem.write("controller", "program",
                          f"debug session on {case.get('task_id')} ({case.get('perspective', '')}; request: {case.get('prompt', '')[:90]}): "
                          f"symptom '{prev['patch'].get('symptom')}'; patch {json.dumps(prev['applied'])}; critic score "
                          f"{prev.get('base_score', prev['verdict']['score']):.1f} -> {v['score']:.1f} ({outcome}). Hypothesis was: {prev['patch'].get('hypothesis', '')[:140]}",
                          meta={"case": case.get("task_id"), "outcome": outcome, "patch": prev["applied"]})
            except Exception:
                pass
        attempts.append(att)
        log(f"{case.get('task_id')} debug attempt {k}: score {v['score']} note {v['note'][:80]}")
        if kept is None or v["score"] > attempts[kept[0]]["verdict"]["score"]:
            kept = (len(attempts) - 1, res, frames, copy.deepcopy(prog))
        if k + 1 >= int(dbg.get("max", 3)) or (v["action"] == 2 and v["subject"] == 2 and v["picture"] == 2 and v["alive"] == 2):
            break
        history = "\n".join(f"- attempt {i}: program {json.dumps(a['program'])[:300]} -> critic {a['verdict']['score']:.1f} "
                            f"(action {a['verdict']['action']}, subject {a['verdict']['subject']}, picture {a['verdict']['picture']}, "
                            f"alive {a['verdict']['alive']}): {a['verdict']['note']}" for i, a in enumerate(attempts))
        notes = recall(case, v["note"]) or notes0
        text = (f"Scene: {case.get('image_caption', '')}\nRequested: {case.get('prompt', '')}\nViewpoint: {case.get('perspective', '')}\n\n"
                f"Programs tried in this session:\n{history}\n\nState report of the last rollout:\n- " + "\n- ".join(report) +
                "\n\nNotes from earlier debugging sessions (past.dev):\n- " + ("\n- ".join(notes) if notes else "(none yet)") +
                "\n\n" + MENU)
        try:
            patch = llm.ask_json(SYSTEM, text, images=[np.asarray(x) for x in critic.sheets(frames)], max_tokens=400,
                                 model=dbg.get("model"))
        except Exception as e:
            patch = {"symptom": "debugger call failed", "hypothesis": repr(e)[:100], "new_noise": True}
        # continue from the best program so far, not from a patch that made things worse
        base = kept[3]
        prog, applied = apply(base, patch, case)
        attempts[-1].update({"patch": patch, "applied": applied, "recalled": notes,
                             "base_score": attempts[kept[0]]["verdict"]["score"]})
    best, res, frames, _ = kept
    res["debug"] = {"attempts": attempts, "kept": best}
    return res, frames
