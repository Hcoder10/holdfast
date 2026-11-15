"""past.dev demo 3: the memory decides how a cross-attention write is scheduled.

  python -I -X utf8 tools/act_memory_demo.py        # writes results/pastdev_act_demo.txt

What happened on the morning of 15 November, with real times (Pacific):

  10:10  proposer      belief    a constant Exa-derived action sentence in the cross-attention register at gain 8
                                 will make still-camera subjects act and raise insight scores
  10:58  experimenter  measured  judged, 15 cases, two passes: it lowers insight 2.86 -> 2.40; the judge's reasons
                                 are looping motion and frozen scenes
  11:00  experimenter  rule      never hold the action register constant; default none until a schedule is measured
  12:25  experimenter  measured  the judged results of the two time-varying schedules (sweep29, and sweep32 on held-out
                                 cases): timeline lowers scores, impulse has no measured gain; the rule is updated

The planner (Haiku, a different agent, member of wh-programs) chooses `scene_action` for still-camera cases three
times: reading the memory as of 10:30, as of 11:05, and now. Nothing is deleted between reads; past.dev's
`queryTimestamp` gives the earlier views. The choice becomes the `act_register` entry of the episode's state
program, i.e. it changes what is written into the model's cross-attention layers.
"""
import json
import os
import sys
import time

from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "adapters", "lingbot"))
import mem  # noqa: E402
import planner  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join(ROOT, "results", "pastdev_act_demo.txt")
lines = []


def say(*a):
    t = " ".join(str(x) for x in a)
    lines.append(t)
    print(t)


RECORDS = [
    ("proposer", "belief", "belief-act-register-constant", "2026-11-15T18:10:00Z",
     "Still-camera scenes ('wait and observe'): writing an Exa-derived action sentence into the cross-attention "
     "register at gain 8 for the whole episode (scene_action constant) will make subjects act and raise insight scores."),
    ("experimenter", "measured", "measured-act-register-constant", "2026-11-15T18:58:00Z",
     "Still-camera scenes: scene_action constant (action register at gain 8 all episode) LOWERS judged insight, "
     "2.86 -> 2.40 over 15 development cases, two Gemini passes (sweep27). Judge reasons: the subject repeats the same "
     "motion with no accumulated change, or the scene freezes in the described state. A constant sentence is "
     "re-satisfied every chunk; the score rewards accumulated change."),
    ("experimenter", "rule", "rule-act-register", "2026-11-15T19:00:00Z",
     "Rule for every agent that builds a program: on still-camera scenes scene_action must not be constant. The "
     "impulse and timeline schedules are untested; until one has a measured gain, scene_action is none."),
]
res29, res32 = os.path.join(ROOT, "results", "sweep29.json"), os.path.join(ROOT, "results", "sweep32.json")
if os.path.exists(res29) and os.path.exists(res32):
    r, h = json.load(open(res29)), json.load(open(res32))
    ie = {a: r["arms"][a]["IE"] for a in r["arms"]}
    hd = h["paired"]["final_minus_baseline"]
    RECORDS.append(("experimenter", "measured", "measured-act-register-schedules", "2026-11-15T20:25:00Z",
                    f"Still-camera scenes, two Gemini passes. Development cases (15, sweep29): no write {ie['baseline']:.2f}, "
                    f"scene_action impulse {ie['alive']:.2f}, scene_action timeline {ie['final']:.2f}. Held-out cases (15, sweep32): "
                    f"impulse minus no write {hd['overall']:+.2f}, interval {hd['ci95'][0]:+.2f} to {hd['ci95'][1]:+.2f}, "
                    f"{hd['wins']} better and {hd['losses']} worse. Pooled over 30 cases: +0.16, interval -0.23 to +0.56. "
                    "Conclusion: timeline lowers scores; impulse has no measured gain. scene_action stays none."))
    RECORDS.append(("experimenter", "rule", "rule-act-register", "2026-11-15T20:26:00Z",
                    "Rule for every agent that builds a program: on still-camera scenes scene_action is none. Constant and "
                    "timeline lower judged scores (measured); impulse was tested on 30 cases and has no measured gain."))
VIEWS = [("as of 10:30 (belief only)", "2026-11-15T18:30:00Z"), ("as of 11:05 (belief, measurement, rule)", "2026-11-15T19:05:00Z"),
         ("now", None)]

say("== records published to wh-programs (author, kind, time) ==")
for agent, kind, id_, ts, text in RECORDS:
    w = mem.write(agent, kind, text, id=id_, wait=True, timestamp=ts)
    say(f"{ts}  {agent:12} {kind:9} {id_}  [{w.get('status')}]")
time.sleep(10)

cases = {c["task_id"]: c for c in json.load(open(os.path.join(ROOT, "third_party", "pw_data", "insight", "data.json"), encoding="utf8"))}
import cv2  # noqa: E402
VID = os.path.join(ROOT, "runs", "view", "sweep27", "lingbot2_14b_open_anchor")
pick = [t for t in ("OE002", "OE005", "OE013", "OE066") if t in cases and os.path.exists(os.path.join(VID, t + ".mp4"))]


def first_frame(t):
    ok, f = cv2.VideoCapture(os.path.join(VID, t + ".mp4")).read()
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


results = {}
for label, as_of in VIEWS:
    ev = [n for n in planner.evidence(as_of)]
    say(f"\n== what the planner reads about still-camera scenes, {label} ==")
    for n in ev:
        if "act-register" in (n["source_id"] or ""):
            say(f"  {n['occurred_at']}  {n['kind']:9} by {n['author']:12} supersededAt={n['superseded_at']}  source={n['source_id']}")
            say(f"      {n['text'][:200]}")
    say(f"\n== planner decisions, {label} ==")
    results[label] = {}
    for t in pick:
        c = cases[t]
        img = first_frame(t)          # the scene's first frame, from the control clip
        case = {k: c[k] for k in ("task_id", "prompt", "image_caption", "perspective") if k in c}
        d = planner.decide(case, img, use_memory=True, from_memory=True, as_of=as_of, write_back=False)
        results[label][t] = d["decision"]["scene_action"]
        say(f"  {t}  scene_action {d['decision']['scene_action']:<9} -> act_register {json.dumps(d['cfg'].get('act_register'))} | {d['why'][:130]}")

say("\n== change ==")
for label, _ in VIEWS:
    v = list(results[label].values())
    say(f"{label}: " + ", ".join(f"{k} {v.count(k)}" for k in ("none", "constant", "impulse", "timeline") if v.count(k)) + f"  (of {len(v)} cases)")

open(OUT, "w", encoding="utf8", newline="\n").write("\n".join(lines) + "\n")
print("\nwrote", OUT)
