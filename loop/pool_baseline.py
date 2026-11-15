"""Score every hill-climb trial against a pooled baseline.

The judge scores an identical video up to 2-4 points apart on repeat (results/hillclimb_log.md),
so the baseline value of a development case is the mean of every judging of its baseline video:
the half-benchmark judging, plus each trial whose video for that case came out byte-identical
(same contact sheets). A trial's unchanged video takes the pooled baseline value (zero difference).

  python -I -X utf8 loop/pool_baseline.py
"""
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "PlayWorld"))
from metrics.aggregate import OE_INSIGHT_TASKS, normalized_score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
RUNS = ROOT / "runs"
DIMS = ("GC", "IF", "IE", "OE")


def sheet_hash(d):
    m = hashlib.md5()
    for f in sorted(p for p in Path(d).rglob("*") if p.is_file()):
        m.update(f.read_bytes())
    return m.hexdigest()


def dim(c):
    return c[:2] if c[:2] in ("GC", "IF") else "IE" if c in OE_INSIGHT_TASKS else "OE"


def load(p):
    d = json.loads(Path(p).read_text(encoding="utf8"))
    return d[0] if isinstance(d, list) else d


def judged(run):
    """{case: score} from runs/gem_<run>, and the mode folder name."""
    out, mode = {}, None
    bm = json.loads((RUNS / f"jblind_{run}" / "blind_map.json").read_text())
    for v, mode in bm.items():
        for case in (RUNS / f"gem_{run}" / v).iterdir():
            sp, gp = case / "score_result.json", case / "gate_result.json"
            if not sp.exists():
                continue
            raw = normalized_score(f"{v}/{case.name}", load(sp))
            if raw is not None:
                out[case.name] = 1.0 if gp.exists() and load(gp).get("verdict") == "fail" else raw
    return out, mode


base = {c: [v[0]] for c, v in json.loads((RUNS / "hillclimb" / "baseline_scores.json").read_text()).items()}
trials = {}
for td in sorted((RUNS / "hillclimb").glob("trial_*.txt")):
    t = int(td.stem.split("_")[1])
    run = f"hc{t:02d}"
    scores, mode = judged(run)
    rows = {}
    for c, s in scores.items():
        a, b = RUNS / f"jin_{run}" / mode / c / "sheets", RUNS / "jin_half1" / "lingbot2_14b_open" / c / "sheets"
        same = a.is_dir() and b.is_dir() and sheet_hash(a) == sheet_hash(b)
        if same:
            base[c].append(s)
        rows[c] = (s, same)
    aside = RUNS / f"jin_{run}_same"          # set aside before judging by loop/hillclimb.py
    for c in (p.name for p in aside.iterdir()) if aside.is_dir() else ():
        rows[c] = (None, True)
    trials[t] = (rows, td.read_text(encoding="utf8").splitlines()[0][:150])

pool = {c: st.mean(v) for c, v in base.items()}
(RUNS / "hillclimb" / "baseline_pooled.json").write_text(json.dumps(
    {c: {"judgings": v, "mean": round(pool[c], 2)} for c, v in sorted(base.items())}, indent=1))


def by_dim(sc):
    return {d: round(st.mean(s for c, s in sc.items() if dim(c) == d), 2) for d in DIMS}


ob = by_dim(pool)
print("repeat judgings of identical baseline videos:")
for c, v in sorted(base.items()):
    if len(v) > 1:
        print("  ", c, [round(x, 2) for x in v])
print(f"\n{'':34}" + "".join(f"{d:>7}" for d in DIMS) + "  overall  changed  W/L")
print(f"{'baseline (pooled)':34}" + "".join(f"{ob[d]:7.2f}" for d in DIMS) + f"  {st.mean(ob.values()):7.2f}")
out = {"baseline": ob}
for t, (rows, name) in trials.items():
    eff = {c: (pool[c] if same else s) for c, (s, same) in rows.items() if c in pool}
    o = by_dim(eff)
    ch = [(c, eff[c] - pool[c]) for c, (s, same) in rows.items() if c in pool and not same]
    w, l = sum(d > 0.25 for _, d in ch), sum(d < -0.25 for _, d in ch)
    print(f"{'trial ' + str(t):34}" + "".join(f"{o[d]:7.2f}" for d in DIMS) +
          f"  {st.mean(o.values()):7.2f}  {len(ch):7d}  {w}/{l}   n={len(eff)}")
    print("    ", name)
    out[f"trial_{t}"] = {"by_dim": o, "overall": round(st.mean(o.values()), 2), "changed": len(ch), "wins": w,
                         "losses": l, "pairs": sorted((c, round(pool[c], 2), round(eff[c], 2)) for c, _ in ch)}
(RUNS / "hillclimb" / "pooled_table.json").write_text(json.dumps(out, indent=1))
