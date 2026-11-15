"""past.dev demo for the challenge: read boundaries, one agent using another's data, a superseded fact.

  python -I -X utf8 tools/mem_demo.py            # writes results/pastdev_demo.txt

1. Creates the three audiences (tools/mem.py: AUDIENCES).
2. The controller records the register gain it uses for third-person cases at two times: 4 (00:45,
   after the first sweep) and 8 (01:10, after validation on fresh cases). Both statements are true
   to what happened; the later one should supersede the earlier.
3. The critic records a defect it saw at gain 16.
4. Recalls the same question as three identities and prints who can see what, with each result's
   date, its supersededAt, its author (from the source label) and the excerpts it rests on.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mem  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join(mem.ROOT, "results", "pastdev_demo.txt")
lines = []


def say(*a):
    t = " ".join(str(x) for x in a)
    lines.append(t)
    print(t)


say("== audiences ==")
for slug, rec in mem.setup_audiences().items():
    say(slug, "members:", rec.get("identities"), "| memberCount", rec.get("memberCount"))

say("\n== writes ==")
W = [("controller", "program", "Register gain for third-person interaction cases on lingbot2_14b: 4. "
      "Chosen after the first gain sweep (six cases): gain 4 and above completes the requested action.",
      "wh-demo-gain-0045", "2026-11-15T08:45:00Z"),
     ("controller", "program", "Register gain for third-person interaction cases on lingbot2_14b: 8. "
      "Changed from 4 after validation on seven further cases: gain 8 scored 4.0 against 3.5 for gain 4.",
      "wh-demo-gain-0110", "2026-11-15T09:10:00Z"),
     ("critic", "defect", "At register gain 16 on a first-person driving case the car interior disappears; "
      "the register is limited to third-person cases.", "wh-demo-defect-0050", "2026-11-15T08:50:00Z"),
     ("experimenter", "lesson", "Trial 5 (objective prompt plus rotation memory) scored 2.00 against 2.33: "
      "rotation memory does not add to the objective prompt.", "wh-demo-lesson-2320", "2026-11-15T07:20:00Z")]
for agent, kind, text, id_, ts in W:
    r = mem.write(agent, kind, text, id=id_, wait=True, timestamp=ts)
    say(agent, kind, "->", mem.KIND_AUDIENCE[kind], "|", id_, "|", r.get("status"))
time.sleep(8)

say("\n== the same question, read as three identities ==")
Q = "What register gain is used for third-person interaction cases, and what goes wrong at high gain?"
for who in ("controller", "composer", "proposer"):
    hits = mem.recall(who, Q, limit=12)
    mine = [h for h in hits if (h.get("source_id") or "").startswith("wh-demo-")]
    say(f"\n[{who}] audiences: {[a for a, m in mem.AUDIENCES.items() if who in m]}; "
        f"{len(hits)} results, {len(mine)} from this demo's sources")
    for h in mine:
        say("  -", (h["text"] or "")[:150])
        say("    author:", h["author"], "| occurredAt:", h["occurred_at"], "| supersededAt:", h["superseded_at"],
            "| source:", h["source_id"])
        for e in (h.get("excerpts") or [])[:1]:
            say("    excerpt:", e[:150])
open(OUT, "w", encoding="utf8", newline="\n").write("\n".join(lines) + "\n")
print("\nwrote", OUT)
