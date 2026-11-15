"""Per-dimension and overall PlayWorld scores for a judged, blinded run, with paired differences.

  python eval/tally.py --judged runs/gem_half1 --map runs/jblind_half1/blind_map.json \
      --playworld third_party/PlayWorld --base lingbot2_14b_open --ours "lingbot2_14b_memory+online" \
      [--out results/half1_tally.json]

Scoring follows PlayWorld's aggregate.py: a failed instruction-following gate scores 1,
otherwise `normalized_score`; the overall score is the unweighted mean of the four
dimension means (Geometry, Interaction, Insight, Out-of-sight). The confidence
interval on the paired difference is a bootstrap over cases within each dimension.
"""
import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--judged", type=Path, required=True)
ap.add_argument("--map", type=Path, required=True)
ap.add_argument("--playworld", type=Path, required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--ours", required=True)
ap.add_argument("--out", type=Path)
a = ap.parse_args()
sys.path.insert(0, str(a.playworld))
from metrics.aggregate import OE_INSIGHT_TASKS, embedded_review, normalized_score  # noqa: E402


def load(p):
    d = json.loads(p.read_text(encoding="utf8"))
    return d[0] if isinstance(d, list) else d


def dim(task):
    if task.startswith("GC"):
        return "GC"
    if task.startswith("IF"):
        return "IF"
    return "IE" if task in OE_INSIGHT_TASKS else "OE"


names = json.loads(a.map.read_text())
scores = {}   # run -> task -> (score, gate)
for v, run in names.items():
    for case in (a.judged / v).iterdir():
        sp = case / "score_result.json"
        if not sp.exists():
            continue
        res = load(sp)
        gate = load(case / "gate_result.json").get("verdict") if (case / "gate_result.json").exists() else None
        if gate is None:
            rev = embedded_review(res, "OE" if case.name.startswith("OE") else case.name[:2], "third-person")
            gate = rev.get("verdict") if rev else None
        raw = normalized_score(f"{v}/{case.name}", res)
        scores.setdefault(run, {})[case.name] = (1.0 if gate == "fail" else raw, gate)

DIMS = ("GC", "IF", "IE", "OE")
rows, paired = {}, {d: [] for d in DIMS}
for run in (a.base, a.ours):
    by = {d: [] for d in DIMS}
    for task, (s, g) in scores.get(run, {}).items():
        if s is not None:
            by[dim(task)].append(s)
    rows[run] = {d: (round(st.mean(v), 2), len(v)) for d, v in by.items() if v}
    rows[run]["overall"] = round(st.mean(rows[run][d][0] for d in DIMS if d in rows[run]), 2)
for task in sorted(set(scores.get(a.base, {})) & set(scores.get(a.ours, {}))):
    b, o = scores[a.base][task][0], scores[a.ours][task][0]
    if b is not None and o is not None:
        paired[dim(task)].append((task, b, o))

print(f"{'':28s}" + "".join(f"{d:>12s}" for d in DIMS) + f"{'overall':>10s}")
for run in (a.base, a.ours):
    print(f"{run:28s}" + "".join(f"{rows[run][d][0]:>7.2f} (n{rows[run][d][1]:>2d})" if d in rows[run] else f"{'-':>12s}"
                                 for d in DIMS) + f"{rows[run]['overall']:>10.2f}")
diff = {d: round(st.mean(o - b for _, b, o in paired[d]), 2) for d in DIMS if paired[d]}
overall_diff = round(st.mean(diff.values()), 2)
rng = random.Random(0)
boots = []
for _ in range(5000):
    boots.append(st.mean(st.mean(o - b for _, b, o in [rng.choice(paired[d]) for _ in paired[d]])
                         for d in DIMS if paired[d]))
boots.sort()
lo, hi = round(boots[125], 2), round(boots[4874], 2)
print(f"{'paired difference':28s}" + "".join(f"{diff.get(d, float('nan')):>+12.2f}" for d in DIMS)
      + f"{overall_diff:>+10.2f}   95% CI [{lo:+.2f}, {hi:+.2f}]")
wins = {d: (sum(o > b for _, b, o in paired[d]), sum(o == b for _, b, o in paired[d]),
            sum(o < b for _, b, o in paired[d])) for d in DIMS if paired[d]}
print("wins/ties/losses by dimension:", wins)
gates = {run: {k: sum(1 for s, g in scores[run].values() if g == k) for k in ("pass", "partial", "fail")}
         for run in (a.base, a.ours)}
print("gates:", gates)
if a.out:
    a.out.write_text(json.dumps({"rows": rows, "paired_difference": diff, "overall_difference": overall_diff,
                                 "ci95": [lo, hi], "wins_ties_losses": wins, "gates": gates,
                                 "cases": {d: paired[d] for d in DIMS}}, indent=1))
