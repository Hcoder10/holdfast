"""All 171 cases with the shadow world: results/full_benchmark_v2.json.

  python -I -X utf8 eval/full_v2.py

The shadow world only changes first-person cases, so it was generated and judged on those (46 held-out cases
in results/held_out_v2.json, 48 development cases in the gem_v2dev_p* passes). Every other case keeps its
judged score under the pre-registered program. The unmodified arm is the one in results/held_out.json and
results/dev_half.json. Dimension means first, then the mean of the four, as PlayWorld's aggregate.py does.
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT / "third_party" / "PlayWorld"))
from metrics.aggregate import OE_INSIGHT_TASKS, normalized_score  # noqa: E402

DIMS = ("GC", "IF", "IE", "OE")


def dim(t):
    return "GC" if t.startswith("GC") else "IF" if t.startswith("IF") else "IE" if t in OE_INSIGHT_TASKS else "OE"


def load(p):
    d = json.loads(Path(p).read_text(encoding="utf8"))
    return d[0] if isinstance(d, list) else d


def judged(prefix):
    """case -> [score per pass] over every blinded folder of the passes named gem_<prefix>_p*."""
    per = {}
    for pd in sorted(RUNS.glob(f"gem_{prefix}_p*")):
        for sp in pd.glob("*/*/score_result.json"):
            raw = normalized_score(f"{sp.parent.parent.name}/{sp.parent.name}", load(sp))
            if raw is None:
                continue
            gp = sp.parent / "gate_result.json"
            g = load(gp).get("verdict") if gp.exists() else None
            per.setdefault(sp.parent.name, []).append(1.0 if g == "fail" else raw)
    return per


def table(scores):
    by = {d: [s for c, s in scores.items() if dim(c) == d] for d in DIMS}
    m = {d: round(st.mean(v), 3) for d, v in by.items()}
    return dict(m, overall=round(st.mean(m.values()), 3), n=len(scores), n_by_dim={d: len(v) for d, v in by.items()})


def paired(a, b):
    common = sorted(set(a) & set(b))
    by = {d: [b[c] - a[c] for c in common if dim(c) == d] for d in DIMS}
    rng, boots = random.Random(0), []
    for _ in range(5000):
        boots.append(st.mean(st.mean(rng.choices(v, k=len(v))) for v in by.values() if v))
    boots.sort()
    return {"n": len(common), "by_dim": {d: round(st.mean(v), 3) for d, v in by.items() if v},
            "overall": round(st.mean(st.mean(v) for v in by.values() if v), 3), "ci95": [round(boots[125], 3), round(boots[4874], 3)],
            "wins": sum(b[c] > a[c] + 0.25 for c in common), "losses": sum(b[c] < a[c] - 0.25 for c in common)}


if __name__ == "__main__":
    held = json.loads((ROOT / "results" / "held_out.json").read_text())["per_case"]
    dev = json.loads((ROOT / "results" / "dev_half.json").read_text())["per_case"]
    base = {c: st.mean(v) for src in (held, dev) for c, v in src["baseline"].items()}
    final = {c: st.mean(v) for src in (held, dev) for c, v in src["final"].items()}
    v2h = {c: st.mean(v) for c, v in json.loads((ROOT / "results" / "held_out_v2.json").read_text())["per_case_v2"].items()}
    v2d = {c: st.mean(v) for c, v in judged("v2dev").items()}
    v2 = dict(final)
    v2.update(v2h)
    v2.update(v2d)
    out = {"note": "Shadow-world program on all 171 cases. First-person cases were regenerated and judged with the "
                   "shadow world (held-out half twice, development half twice); the rest keep their judged score "
                   "under the pre-registered program. The shadow-world arm was designed after the held-out result "
                   "of the pre-registered program was known.",
           "cases": len(v2), "shadow_cases": {"held_out": len(v2h), "development": len(v2d)},
           "arms": {"baseline": table(base), "final": table(final), "v2": table(v2)},
           "v2_minus_baseline": paired(base, v2), "v2_minus_final": paired(final, v2),
           "dev_shadow_cases_only": {"final": table({c: final[c] for c in v2d}) if len({dim(c) for c in v2d}) == 4 else
                                     {d: round(st.mean([final[c] for c in v2d if dim(c) == d]), 3) for d in DIMS if any(dim(c) == d for c in v2d)},
                                     "v2": {d: round(st.mean([v2d[c] for c in v2d if dim(c) == d]), 3) for d in DIMS if any(dim(c) == d for c in v2d)}},
           "genie3_published": {"GC": 2.74, "IF": 2.40, "IE": 1.51, "OE": 1.81, "overall": 2.12}}
    (ROOT / "results" / "full_benchmark_v2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
