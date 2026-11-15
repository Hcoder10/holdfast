"""Self-repair loop: generate, have the critic look, change the state program, try again.

A *program* is the set of hidden-state writes an episode runs with (see README, "state
program"). The loop starts from the configured program (or one recalled from past.dev
for a similar scene), and after each attempt the critic's verdict picks one edit:

  picture collapsed or subject lost   -> the write was too strong: halve the register gain
                                         (or, with no register left to lower, new noise)
  action not carried out              -> the objective is not getting through: double the
                                         register gain (third-person) or draw new noise
  world frozen                        -> add the gentle "alive" residual write
  nothing wrong                       -> stop

The best attempt by the critic's score is kept. Every attempt, verdict and edit goes into
the result, and the winning program is written to past.dev so that later episodes with a
similar scene start from it. The critic sees the case's caption, objective and
perspective and the generated frames; it never sees the benchmark rubric or judge scores.
"""
import copy
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))

# The residual write that stays clean: early sampling steps only. Written on the late steps
# (which set colour and detail) the same vector compounds into a colour collapse.
ALIVE = {"file": "alive.npz", "alpha": 0.03, "steps": [0, 1], "cache": False}
MAX_GAIN, MIN_GAIN = 16.0, 1.0


def summary(cfg):
    r = cfg.get("register") or {}
    return {"register_gain": r.get("gain"), "alive": bool(cfg.get("steer")), "seed_shift": cfg.get("_seed", 0)}


def good(v):
    return v["action"] == 2 and v["subject"] == 2 and v["picture"] >= 1 and v["alive"] >= 1


def revise(cfg, v, third_person):
    """Next program, and the reason, or (None, reason) to stop."""
    nxt = copy.deepcopy(cfg)
    reg = nxt.get("register")
    if good(v):
        return None, "critic found nothing to fix"
    if v["picture"] == 0 or v["subject"] == 0:
        if reg and reg.get("gain", 1) > MIN_GAIN:
            reg["gain"] = max(MIN_GAIN, reg["gain"] / 2)
            return nxt, "picture or subject lost: register gain halved"
        if nxt.get("steer"):
            nxt.pop("steer")
            return nxt, "picture or subject lost: residual write removed"
        nxt["_seed"] = nxt.get("_seed", 0) + 1
        return nxt, "picture or subject lost with no write to lower: new noise"
    if v["action"] < 2:
        if third_person and reg and reg.get("gain", 1) < MAX_GAIN:
            reg["gain"] = min(MAX_GAIN, reg["gain"] * 2)
            return nxt, "objective not carried out: register gain doubled"
        nxt["_seed"] = nxt.get("_seed", 0) + 1
        return nxt, "objective not carried out: new noise"
    if v["alive"] == 0 and not nxt.get("steer"):
        nxt["steer"] = dict(ALIVE)
        return nxt, "world frozen: gentle alive write added"
    nxt["_seed"] = nxt.get("_seed", 0) + 1
    return nxt, "minor defect: new noise"


def recall_program(case, cfg):
    """Start from the gain that worked on the most similar remembered scene, if any."""
    try:
        import mem
        q = f"state program that worked for: {case.get('prompt', '')} | {case.get('image_caption', '')[:160]}"
        for h in mem.recall("controller", q, 5):
            text = " ".join([h.get("text") or ""] + list(h.get("excerpts") or []))
            m = re.search(r"winning register gain ([0-9.]+)", text)
            if m and cfg.get("register"):
                out = copy.deepcopy(cfg)
                out["register"]["gain"] = float(m.group(1))
                return out, {"from_memory": text[:200], "gain": float(m.group(1))}
    except Exception as e:
        return cfg, {"from_memory": None, "error": repr(e)[:120]}
    return cfg, {"from_memory": None}


def remember(case, attempts, best):
    try:
        import mem
        b = attempts[best]
        gain = b["program"]["register_gain"]
        text = (f"Scene: {case.get('image_caption', '')[:200]} Objective: {case.get('prompt', '')} "
                f"({case.get('perspective', '')}). Tried {len(attempts)} programs. "
                + (f"winning register gain {gain}. " if gain else "no register. ")
                + f"Critic on the winner: action {b['verdict']['action']}/2, subject {b['verdict']['subject']}/2, "
                  f"picture {b['verdict']['picture']}/2. " + " | ".join(a["edit"] for a in attempts if a.get("edit")))
        mem.write("controller", "program", text, meta={"case": case.get("task_id"), "gain": gain,
                                                        "attempts": len(attempts)})
        worst = min(attempts, key=lambda a: a["verdict"]["score"])
        if worst is not b:
            mem.write("critic", "defect", f"{case.get('task_id')}: with program {worst['program']} the critic saw: "
                      f"{worst['verdict']['note']}", meta={"case": case.get("task_id")})
    except Exception:
        pass


def run(make_episode, stepper, case, image, seed, cfg, log=print):
    """make_episode(cfg) -> Episode. Returns (result dict, frames) of the best attempt."""
    import critic
    rep = cfg.get("repair") or {}
    third = (case.get("perspective") or "").strip().lower() == "third person"
    prog, mem_info = (recall_program(case, cfg) if rep.get("memory", True) else (cfg, {"from_memory": None}))
    attempts, kept = [], None
    for k in range(int(rep.get("max", 3))):
        run_cfg = {key: val for key, val in prog.items() if key not in ("repair", "_seed")}
        ep = make_episode(run_cfg)
        res = ep.run(case, image, seed=seed + 101 * prog.get("_seed", 0))
        frames = stepper.video()
        try:
            v = critic.review(frames, case.get("image_caption", ""), case.get("prompt", ""),
                              case.get("perspective", ""), model=rep.get("model"))
        except Exception as e:   # no critic, no repair: keep what we have
            attempts.append({"program": summary(prog), "verdict": {"score": -1.0, "note": repr(e)[:120],
                                                                    "action": 1, "subject": 1, "picture": 1, "alive": 1}})
            if kept is None:
                kept = (len(attempts) - 1, res, frames)
            break
        attempts.append({"program": summary(prog), "verdict": v})
        log(f"{case.get('task_id')} attempt {k}: {summary(prog)} -> {v}")
        if kept is None or v["score"] > attempts[kept[0]]["verdict"]["score"]:
            kept = (len(attempts) - 1, res, frames)
        nxt, why = revise(prog, v, third)
        attempts[-1]["edit"] = why
        if nxt is None:
            break
        prog = nxt
    best, res, frames = kept
    if rep.get("memory", True):
        remember(case, attempts, best)
    res["repair"] = {"attempts": attempts, "kept": best, "memory": mem_info}
    return res, frames
