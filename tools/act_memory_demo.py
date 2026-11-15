"""past.dev demo 3: the memory decides how a cross-attention write is scheduled.

  python -I -X utf8 tools/act_memory_demo.py        # writes results/pastdev_act_demo.txt

What happened on the morning of 15 November, with real times (Pacific):

  10:10  proposer      belief    a constant Exa-derived action sentence in the cross-attention register at gain 8
                                 will make still-camera subjects act and raise insight scores
  10:58  experimenter  measured  judged, 15 cases, two passes: it lowers insight 2.86 -> 2.40; the judge's reasons
                                 are looping motion and frozen scenes
  11:00  experimenter  rule      never hold the action register constant; default none until a schedule is measured
  later  experimenter  measured  the judged result of the two time-varying schedules (results/sweep29.json), if present

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
res29 = os.path.join(ROOT, "results", "sweep29.json")
if os.path.exists(res29):
    r = json.load(open(res29))
    ie = {a: r["arms"][a]["IE"] for a in r["arms"]}
    best = max(("alive", "final"), key=lambda a: ie[a])
    name = {"alive": "impulse", "final": "timeline"}
    verdict = (f"scene_action {name[best]} is the best measured schedule and beats no write; use it"
               if ie[best] > ie["baseline"] + 0.1 else
               "neither schedule beats no write; scene_action stays none")
    RECORDS.append(("experimenter", "measured", "measured-act-register-schedules", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    f"Still-camera scenes, same 15 cases, two Gemini passes (sweep29): no write {ie['baseline']:.2f}, "
                    f"scene_action impulse {ie['alive']:.2f}, scene_action timeline {ie['final']:.2f}. Conclusion: {verdict}."))
VIEWS = [("as of 10:30 (belief only)", "2026-11-15T18:30:00Z"), ("as of 11:05 (belief, measurement, rule)", "2026-11-15T19:05:00Z"),
         ("now", None)]

say("== records published to wh-programs (author, kind, time) ==")
for agent, kind, id_, ts, text in RECORDS:
    w = mem.write(agent, kind, text, id=id_, wait=True, timestamp=ts)
    say(f"{ts}  {agent:12} {kind:9} {id_}  [{w.get('status')}]")
time.sleep(10)

cases = {c["task_id"]: c for c in json.load(open(os.path.join(ROOT, "third_party", "pw_data", "insight", "data.json"), encoding="utf8"))}
pick = [t for t in ("OE002", "OE005", "OE006", "OE011", "OE013", "OE040", "OE066", "OE072") if t in cases]
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
        img = None
        for cand in (os.path.join(ROOT, "third_party", "pw_data", "insight", c["image_path"]),
                     os.path.join(ROOT, "third_party", "pw_data", c["image_path"]),
                     os.path.join(ROOT, "third_party", "pw_data", "insight", "images", os.path.basename(c["image_path"]))):
            if os.path.exists(cand):
                img = Image.open(cand)
                break
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
