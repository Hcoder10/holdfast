"""All 171 cases with the shadow world and landmarks: results/full_benchmark_v3.json.

  python -I -X utf8 eval/full_v3.py

Landmarks only change cases that come back to an earlier position by walking or sliding, so they were generated
and judged on those: 10 development cases (results/sweep31_full.json) and 9 held-out cases (results/sweep33.json).
In each of those batches the shadow-world arm was regenerated alongside and judged in the same blinded batch.
Two rows are built from results/full_benchmark_v2.json (the shadow-world program on all 171 cases):

  v2_same_batch   those 19 cases take the shadow-world arm's score from the landmark batches
  v3              those 19 cases take the landmark arm's score

so the two rows differ only where landmarks ran, and by scores from the same judging batch. Post hoc.
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import full_v2 as f2  # noqa: E402

if __name__ == "__main__":
    held = json.loads((ROOT / "results" / "held_out.json").read_text())["per_case"]
    dev = json.loads((ROOT / "results" / "dev_half.json").read_text())["per_case"]
    base = {c: st.mean(v) for src in (held, dev) for c, v in src["baseline"].items()}
    final = {c: st.mean(v) for src in (held, dev) for c, v in src["final"].items()}
    v2 = dict(final)
    v2.update({c: st.mean(v) for c, v in json.loads((ROOT / "results" / "held_out_v2.json").read_text())["per_case_v2"].items()})
    v2.update({c: st.mean(v) for c, v in f2.judged("v2dev").items()})
    ctrl, v3, used = dict(v2), dict(v2), {}
    for name in ("sweep31_full", "sweep33"):
        p = ROOT / "results" / f"{name}.json"
        if not p.exists():
            continue
        pc = json.loads(p.read_text())["per_case"]
        for c in pc["final"]:
            if c in pc["baseline"]:
                ctrl[c], v3[c] = st.mean(pc["baseline"][c]), st.mean(pc["final"][c])
                used[c] = name
    out = {"note": "Shadow world plus landmarks on all 171 cases. Landmarks ran on the cases listed in landmark_cases; "
                   "everything else keeps its score under the shadow-world program. Post hoc.",
           "landmark_cases": used,
           "arms": {"baseline": f2.table(base), "final": f2.table(final), "v2": f2.table(v2),
                    "v2_same_batch": f2.table(ctrl), "v3": f2.table(v3)},
           "v3_minus_v2_same_batch": f2.paired(ctrl, v3), "v3_minus_baseline": f2.paired(base, v3),
           "v3_minus_final": f2.paired(final, v3),
           "landmark_cases_only": f2.paired({c: ctrl[c] for c in used}, {c: v3[c] for c in used}) if len({f2.dim(c) for c in used}) else None,
           "genie3_published": {"GC": 2.74, "IF": 2.40, "IE": 1.51, "OE": 1.81, "overall": 2.12}}
    (ROOT / "results" / "full_benchmark_v3.json").write_text(json.dumps(out, indent=1))
    for a, t in out["arms"].items():
        print(a, {k: t[k] for k in ("GC", "IF", "IE", "OE", "overall")})
    for k in ("v3_minus_v2_same_batch", "v3_minus_baseline", "v3_minus_final", "landmark_cases_only"):
        print(k, out[k])
