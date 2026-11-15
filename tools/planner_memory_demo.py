"""past.dev demo: a belief, its refutation, a rule, and a planner whose decisions change because of them.

  python -I -X utf8 tools/planner_memory_demo.py        # writes results/pastdev_planner_demo.txt

This replays, with their real times, four things that happened on the night of 14-15 November:

  00:50  critic        rule      the action register must be 0 for first-person or camera-instruction requests
                                 (at gain 4-8 the word "camera" produced a camera on a tripod)
  05:10  proposer      belief    the residual "alive" write fixes frozen scenes (from frame strips, not judged)
  06:12  experimenter  measured  judged on 21 cases: it does not lift frozen cases and lowers ones that were fine
  06:45  experimenter  measured  a planner that trusted the belief scored 1.59 against 2.19
  06:46  experimenter  rule      alive must be 0 in every program until a judged test shows a gain

The proposer and experimenter work in the lab layer (wh-lab); these five records are the conclusions they
publish to the layer the program-building agents read (wh-programs). The planner (a different agent) then
chooses a state program for eight cases twice: reading the memory as it stood at 05:30, when only the belief
and the critic's rule existed, and reading it now. Nothing is deleted between the two reads; past.dev's
`queryTimestamp` gives the earlier view.
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
OUT = os.path.join(ROOT, "results", "pastdev_planner_demo.txt")
lines = []


def say(*a):
    t = " ".join(str(x) for x in a)
    lines.append(t)
    print(t)


RECORDS = [
    ("critic", "rule", "wh-plan-rule-register-0050", "2026-11-15T08:50:00Z",
     "Rule for every agent that builds a program: action_register must be 0 when the request is first-person or an "
     "instruction to the camera. Measured in frames at gain 4 and 8: the word 'camera' produced a camera on a tripod, "
     "and a first-person car interior disappeared. Use 8 only when a visible third-person subject must act."),
    ("proposer", "belief", "wh-plan-belief-alive-0510", "2026-11-15T13:10:00Z",
     "Belief, not yet judged: the alive residual write fixes frozen scenes. It triples frame-to-frame motion with "
     "the picture intact. Use alive 0.06 when a subject must start moving from rest or a process must advance, "
     "0.03 for a calm scene, 0 only when there is already strong motion."),
    ("experimenter", "measured", "wh-plan-measured-alive-0612", "2026-11-15T14:12:00Z",
     "Measured with the benchmark judge (21 cases, two passes): the alive write always on at 0.03 does not lift "
     "frozen cases (1.87 to 1.93) and lowers cases that were fine (3.00 to 2.45). More motion is not the requested "
     "thing happening."),
    ("experimenter", "measured", "wh-plan-measured-alive-0645", "2026-11-15T14:45:00Z",
     "Measured with the benchmark judge (same 21 cases, two passes): programs with alive at 0.03 or 0.06 on every "
     "case scored 1.59 against 2.19 with alive 0. Dose-response: alive 0 gives 2.19, 0.03 gives 2.08, mostly 0.06 "
     "gives 1.59. The alive write lowers judged scores."),
    ("experimenter", "rule", "wh-plan-rule-alive-0646", "2026-11-15T14:46:00Z",
     "Rule for every agent that builds a program: alive must be 0 until a judged test shows a gain. This replaces "
     "the proposer's belief of 05:10 that the alive write fixes frozen scenes."),
    ("experimenter", "measured", "wh-plan-measured-anchor-0605", "2026-11-15T14:05:00Z",
     "Measured with the benchmark judge (171 cases): latent re-anchoring at 0.5 is the only change on the 30 "
     "wait-and-observe cases and they score 2.57 against 2.30. Keep anchor at 0.5."),
]
AS_OF_EARLY = "2026-11-15T13:30:00Z"       # 05:30 Pacific: belief and critic's rule only

say("== audiences ==")
for slug, rec in mem.setup_audiences().items():
    say(slug, "members:", rec.get("identities"))

say("\n== records published to wh-programs (author, kind, time) ==")
for agent, kind, id_, ts, text in RECORDS:
    r = mem.write(agent, kind, text, id=id_, wait=True, timestamp=ts)
    say(f"{ts}  {agent:12} {kind:9} {id_}  [{r.get('status')}]")
time.sleep(10)

cases = {}
for sp in ("gc", "if", "insight", "outsight"):
    for c in json.load(open(os.path.join(ROOT, "third_party", "pw_data", sp, "data.json"), encoding="utf8")):
        cases[c["task_id"]] = c
frames = {os.path.splitext(f)[0]: os.path.join(ROOT, "runs", "frames0", f)
          for f in os.listdir(os.path.join(ROOT, "runs", "frames0"))}

results = {}
for label, as_of in (("as the memory stood at 05:30 (belief and one rule)", AS_OF_EARLY), ("as it stands now", None)):
    ev = planner.evidence(as_of)
    say(f"\n== what the planner reads, {label} ==")
    for n in ev:
        say(f"  {n['occurred_at']}  {n['kind']:9} by {n['author']:12} supersededAt={n['superseded_at']}  source={n['source_id']}")
        say(f"      {n['text'][:170]}")
    say(f"\n== planner decisions, {label} ==")
    results[label] = {}
    for t in sorted(frames):
        c = cases[t]
        case = {k: c[k] for k in ("task_id", "prompt", "image_caption", "perspective") if k in c}
        d = planner.decide(case, Image.open(frames[t]), use_memory=True, from_memory=True, as_of=as_of, write_back=False)
        results[label][t] = d["decision"]
        say(f"  {t}  {str(c.get('perspective'))[:12]:12} register {d['decision']['action_register']:<3} alive "
            f"{d['decision']['alive']:<5} anchor {d['decision']['anchor']:<4} | {d['why'][:120]}")

a, b = list(results.values())
say("\n== change ==")
say("alive switched on:", sum(1 for t in a if a[t]["alive"] > 0), "of", len(a), "cases before;",
    sum(1 for t in b if b[t]["alive"] > 0), "of", len(b), "after the experimenter's measurements and rule")
say("register on a first-person case (the critic's rule):",
    sum(1 for t in a if a[t]["action_register"] > 0 and "first" in str(cases[t].get("perspective")).lower()), "before,",
    sum(1 for t in b if b[t]["action_register"] > 0 and "first" in str(cases[t].get("perspective")).lower()), "after")

say("\n== the same question read by an agent outside wh-programs ==")
for who in ("composer", "proposer"):
    hits = [h for h in mem.recall(who, planner.EVIDENCE_Q, 20) if (h.get("source_id") or "").startswith("wh-plan-")]
    say(f"  {who}: {len(hits)} of these records visible")

open(OUT, "w", encoding="utf8", newline="\n").write("\n".join(lines) + "\n")
print("\nwrote", OUT)
