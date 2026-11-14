"""Turn judge answers into PlayWorld scores and append them to the ledger.

  python judge_collect.py --judged <dir> --playworld <PlayWorld checkout> \
      --judge "haiku-5.5 subagent" --player preset [--map blind_map.json] [--ledger results/ledger.jsonl]

<dir>/<system>/<task_id>/ must hold score_result.json and, for gated cases,
gate_result.json. Scoring follows PlayWorld's aggregate.py: a failed gate scores
1, otherwise `normalized_score` (weighted categories for GC, the judge's final
score elsewhere). --map renames blinded system folders back to run names.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--judged", type=Path, required=True)
ap.add_argument("--playworld", type=Path, required=True)
ap.add_argument("--judge", required=True)
ap.add_argument("--player", required=True)
ap.add_argument("--map", type=Path)
ap.add_argument("--ledger", type=Path)
ap.add_argument("--note", default="")
a = ap.parse_args()
sys.path.insert(0, str(a.playworld))
from metrics.aggregate import normalized_score  # noqa: E402

names = json.loads(a.map.read_text()) if a.map else {}
rows = []
for sysdir in sorted(p for p in a.judged.iterdir() if p.is_dir()):
    for case in sorted(p for p in sysdir.iterdir() if p.is_dir()):
        sr, gr = case / "score_result.json", case / "gate_result.json"
        if not sr.exists():
            continue
        result = json.loads(sr.read_text(encoding="utf8"))
        if isinstance(result, list):
            result = result[0]
        gate = None
        if gr.exists():
            g = json.loads(gr.read_text(encoding="utf8"))
            gate = (g[0] if isinstance(g, list) else g).get("verdict")
        raw = normalized_score(f"{sysdir.name}/{case.name}", result)
        score = 1.0 if gate == "fail" else raw
        rows.append({"run": names.get(sysdir.name, sysdir.name), "task_id": case.name,
                     "gate": gate, "rubric_score": raw, "score": score,
                     "judge": a.judge, "player": a.player, "note": a.note,
                     "judged_dir": str(case).replace("\\", "/"),
                     "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
for r in rows:
    print(f"{r['run']:28s} {r['task_id']}  gate={r['gate']}  rubric={r['rubric_score']}  score={r['score']}")
by = {}
for r in rows:
    if r["score"] is not None:
        by.setdefault(r["run"], []).append(r["score"])
for run, v in by.items():
    print(f"MEAN {run:28s} n={len(v)}  {sum(v)/len(v):.2f}")
if a.ledger:
    a.ledger.parent.mkdir(parents=True, exist_ok=True)
    with a.ledger.open("a", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
