"""Launch-time decision model: a small LLM chooses the state program for this task and scene.

Before any video is generated, Claude Haiku is shown the first frame, the caption, the objective and
the perspective (the fields the generator may read; never the rubric or a judge's score) and a menu of
the writes the controller can make, each with what it was measured to do and where it backfired. It
returns one setting per write. The controller then runs the episode with that program.

The menu is what a night of sweeps established (results/hillclimb_log.md):

  action_register  attention gain on the objective, in its own cross-attention segment
  alive            a residual-stream direction ("alive" minus "frozen", from recorded activations)
                   added on the first two sampling steps
  anchor           pull each chunk's latent mean back toward the first chunk's

Before deciding, the planner recalls from past.dev what the controller and the critic recorded about
programs on similar scenes, and afterwards it writes its decision there.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))

SYSTEM = (
    "You configure a video world model before it starts generating. You are given the first frame of a scene, "
    "its caption, what is requested, and the viewpoint. Choose one setting for each control. The model's known "
    "weaknesses: it under-follows the request, it often leaves a scene frozen (a subject that should start moving "
    "stays still, a process that should progress repeats in place or does not advance), and its colours drift "
    "over long videos. Reply with one JSON object and nothing else.")

MENU = """Controls (pick exactly one value for each):

"action_register": 0 | 4 | 8
  How strongly the model attends to the request text.
  8  when the request describes what a VISIBLE subject does (a person, animal, vehicle or robot seen from outside
     walks, climbs, enters, drives, picks something up). Measured: completes actions the plain prompt misses.
  4  the same, when the subject is small or fragile-looking in the frame and could be distorted.
  0  when the request is an instruction to the viewer or the camera (turn, look, tilt, walk forward, wait and
     observe) or the view is first-person. Measured: a high value there conjures objects from the instruction's
     words (a camera on a tripod from "turn the camera") and can erase a first-person vehicle interior.

"alive": 0 | 0.03 | 0.06
  Adds motion inside the model. Measured: about three times more frame-to-frame change at 0.03, picture intact.
  0.06 when the scene will very probably sit frozen without it: the subject is standing or sitting still in the
       first frame and must begin an action from rest; or a process must visibly advance over the video
       (cooking, pouring, drawing, writing, paying at a counter, a launch) and nothing in the frame is moving yet.
  0.03 when there is some risk of freezing: a calm scene whose people or objects should keep moving a little; a
       camera that waits for a long time; a subject already mid-stride that must keep going.
  0    when the frame already shows strong ongoing motion (traffic, a moving crowd, running water, a vehicle
       under way) or the scene is meant to be static architecture or still objects that must not change
       (a room, a sculpture hall, a street of buildings to look around).

"anchor": 0 | 0.5
  Pulls colours back toward the first frame over time. Measured: stops drift on long videos.
  0.5 normally.
  0   only if the lighting should legitimately change a great deal (the view will end on something much brighter
      or darker than it started, for example tilting from a dark room up to open sky).

Also give "why": one short sentence naming what in the frame or the request decided it.
JSON keys: action_register, alive, anchor, why."""

MENU_NEUTRAL = """Controls (pick exactly one value for each). What each control does to the judged score is NOT
stated here; rely on the evidence listed below.

"action_register": 0 | 4 | 8    attention gain on the request text, in its own cross-attention segment
"alive": 0 | 0.03 | 0.06        a residual-stream direction added on the first two sampling steps; raises
                                frame-to-frame motion
"anchor": 0 | 0.5               pulls each chunk's latent mean back toward the first chunk's

Also give "why": one short sentence naming the evidence that decided it.
JSON keys: action_register, alive, anchor, why."""
NL = chr(10)

ALLOWED = {"action_register": (0, 4, 8), "alive": (0, 0.03, 0.06), "anchor": (0, 0.5)}


def _snap(v, choices, default):
    try:
        v = float(v)
    except Exception:
        return default
    return min(choices, key=lambda c: abs(c - v))


EVIDENCE_Q = ("measured effect on judged scores, rules and hypotheses about the state-program controls: "
              "alive residual write, action register, latent re-anchoring")
KIND_RE = re.compile(r"[[](belief|measured|rule) from (.+?)[]] *(.*)", re.S)


def evidence(as_of=None):
    """What the shared memory holds about the controls, as the planner may read it: each note with its
    author, its kind (belief = untested, measured = judged result, rule = binding), its date, and whether
    a later statement has superseded it."""
    import mem
    notes = []
    for h in mem.recall("planner", EVIDENCE_Q, 20, as_of=as_of):
        texts = [h.get("text") or ""] + list(h.get("excerpts") or [])
        m = next((KIND_RE.search(t) for t in texts if KIND_RE.search(t)), None)
        if not m:
            continue
        notes.append({"kind": m.group(1), "author": m.group(2), "text": " ".join(m.group(3).split())[:420],
                      "occurred_at": h.get("occurred_at"), "superseded_at": h.get("superseded_at"),
                      "source_id": h.get("source_id"), "claim": (h.get("text") or "")[:200]})
    by = {}
    for n in notes:                       # one line per source; superseded if any of its claims is
        k = n["source_id"]
        if k not in by:
            by[k] = n
        elif n["superseded_at"] and not by[k]["superseded_at"]:
            by[k]["superseded_at"] = n["superseded_at"]
    return sorted(by.values(), key=lambda x: x["occurred_at"] or "")


def decide(case, image, use_memory=True, model=None, from_memory=False, as_of=None, write_back=True):
    """Returns {"decision": {...}, "cfg": the state-program keys to merge into the episode cfg,
    "recalled": what past.dev returned, "why": str}."""
    import llm
    recalled = []
    if use_memory:
        try:
            import mem
            q = f"state program results and defects for: {case.get('prompt', '')} | {case.get('image_caption', '')[:160]}"
            for h in mem.recall("planner", q, 6):
                if h.get("author") in ("controller", "critic", "planner"):
                    recalled.append(f"({h.get('author')}) {(h.get('text') or '')[:220]}")
        except Exception as e:
            recalled = [f"recall failed: {e!r}"[:120]]
    text = (f"Caption: {case.get('image_caption', '')}\nRequested: {case.get('prompt', '')}\n"
            f"Viewpoint: {case.get('perspective', '') or 'unknown'}\n\n" + MENU)
    ev = []
    if from_memory:
        # The menu's claims about what each control does are replaced by what the memory holds, so the
        # planner cannot mistake an untested belief for a result (which is what it did in sweep21b).
        try:
            ev = evidence(as_of)
        except Exception as e:
            ev = [{"kind": "error", "author": "", "text": repr(e)[:120], "occurred_at": None, "superseded_at": None}]
        text = text.replace(MENU, MENU_NEUTRAL)
        text += (NL + NL + "Evidence from the shared memory, oldest first. A MEASURED note is a judged result and "
                 "overrides any BELIEF (an untested claim). A RULE must be obeyed. A note marked SUPERSEDED "
                 "is void. If the evidence gives no measured gain for a control, set it to 0." + NL)
        for n in ev:
            text += (f"- {n['kind'].upper()}{' (SUPERSEDED)' if n.get('superseded_at') else ''} by {n['author']} "
                     f"at {n['occurred_at']}: {n['text']}" + NL)
        if not ev:
            text += "- (nothing recorded)" + NL
    if recalled and not recalled[0].startswith("recall failed"):
        text += "\n\nNotes recorded by the controller and critic on earlier scenes (may or may not apply):\n- " + \
                "\n- ".join(recalled[:4])
    try:
        v = llm.ask_json(SYSTEM, text, [image], max_tokens=300, model=model)
    except Exception as e:
        v = {"why": "planner call failed: " + repr(e)[:100]}
    third = (case.get("perspective") or "").strip().lower() == "third person"
    wait = (case.get("prompt") or "").lower().startswith("wait and observe")
    d = {"action_register": _snap(v.get("action_register"), ALLOWED["action_register"], 8 if third else 0),
         "alive": _snap(v.get("alive"), ALLOWED["alive"], 0),
         "anchor": _snap(v.get("anchor"), ALLOWED["anchor"], 0.5)}
    if wait:
        d["action_register"] = 0      # there is no action text to amplify
    cfg = {"anchor_stats": d["anchor"]}
    if d["action_register"] > 0:
        cfg["register"] = {"gain": d["action_register"], "replace": True}
    if d["alive"] > 0:
        cfg["steer"] = {"file": "alive.npz", "alpha": d["alive"], "steps": [0, 1], "cache": False}
    why = str(v.get("why", ""))[:200]
    if use_memory and write_back:
        try:
            import mem
            mem.write("planner", "program", f"{case.get('task_id')} launch decision: action_register {d['action_register']}, "
                      f"alive {d['alive']}, anchor {d['anchor']}. Scene: {case.get('image_caption', '')[:140]} "
                      f"Request: {case.get('prompt', '')}. Reason: {why}", meta={"case": case.get("task_id"), **d})
        except Exception:
            pass
    return {"decision": d, "cfg": cfg, "recalled": recalled, "why": why, "evidence": ev}
