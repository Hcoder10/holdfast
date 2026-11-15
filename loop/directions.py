"""Turn recorded activations and judge scores into steering directions.

  python loop/directions.py --acts runs/acts --out results/steer

Inputs: activations per chunk for every judged video (modal_app replay), and the
judge's score for that video (local runs/gem_* folders plus their blind maps).

Two kinds of direction are extracted, per layer:

  score_<DIM>   judge-derived. For each case that was run more than once with scores
                at least --gap apart, the difference between the mean activation of
                its higher-scoring run and its lower-scoring run; averaged over cases
                in the dimension. Pairing within a case removes "which scene is this".
  alive         event-derived. Mean activation of still-camera chunks whose latent
                motion is in the top third, minus those in the bottom third, with each
                episode's own mean removed first.
  stable        event-derived. Chunks with high continuity to the previous chunk minus
                chunks with the lowest (hard-reset-like), same normalisation.

Each direction is scaled to the typical activation norm at its layer, so strength
(alpha) has the same meaning everywhere: alpha 1.0 adds a vector as long as the
average activation. A report with pair counts and a held-out sign test is written
next to the vectors, because a direction from few pairs can be noise.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "PlayWorld"))
from metrics.aggregate import OE_INSIGHT_TASKS, normalized_score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--acts", type=Path, default=ROOT / "runs" / "acts")
ap.add_argument("--out", type=Path, default=ROOT / "results" / "steer")
ap.add_argument("--gap", type=float, default=1.5)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

# volume run name -> (judged dir, blind map dir); the blind map gives the mode folder
JUDGED = {"half1": ("gem_half1", "jblind_half1"), "dev2": ("gem_dev2", "jblind_dev2"),
          "freeze2": ("gem_freeze2", "jblind_freeze2"), "freeze3": ("gem_freeze3", "jblind_freeze3"),
          "hc01": ("gem_hc01", "jblind_hc01"), "hc02": ("gem_hc02", "jblind_hc02"),
          "hc03": ("gem_hc03", "jblind_hc03"), "hc04": ("gem_hc04", "jblind_hc04")}


def load(p):
    d = json.loads(Path(p).read_text(encoding="utf8"))
    return d[0] if isinstance(d, list) else d


def dim(task):
    return "GC" if task.startswith("GC") else "IF" if task.startswith("IF") else \
        "IE" if task in OE_INSIGHT_TASKS else "OE"


episodes = []   # dict(run, mode, task, score, acts [chunks, layers, C], motion, continuity)
for run, (gem, blind) in JUDGED.items():
    mp = ROOT / "runs" / blind / "blind_map.json"
    if not mp.exists():
        continue
    for v, mode in json.loads(mp.read_text()).items():
        vd = ROOT / "runs" / gem / v
        if not vd.is_dir():
            continue
        for case in vd.iterdir():
            sp = case / "score_result.json"
            f = a.acts / run / mode / f"{case.name}.npz"
            if not sp.exists() or not f.exists():
                continue
            gate = load(case / "gate_result.json").get("verdict") if (case / "gate_result.json").exists() else None
            raw = normalized_score(f"{v}/{case.name}", load(sp))
            if raw is None:
                continue
            z = np.load(f)
            episodes.append(dict(run=run, mode=mode, task=case.name, score=1.0 if gate == "fail" else raw,
                                 acts=z["acts"].astype(np.float32), motion=z["motion"], cont=z["continuity"]))
print(len(episodes), "judged episodes with activations")
if not episodes:
    raise SystemExit("nothing to do")
L = episodes[0]["acts"].shape[1]
from_layers = (4, 9, 14, 19, 24, 29, 34, 39)[:L]
norm = np.mean([np.linalg.norm(e["acts"], axis=2).mean(axis=0) for e in episodes], axis=0)   # [L]
report = {"episodes": len(episodes), "layers": list(from_layers), "typical_norm": [round(float(x), 2) for x in norm],
          "directions": {}}


def save(name, vec, info):
    """vec: [L, C]. Scale each layer's vector to that layer's typical activation norm."""
    out = {}
    for i, layer in enumerate(from_layers):
        n = np.linalg.norm(vec[i])
        out[str(layer)] = (vec[i] / n * norm[i]).astype(np.float32) if n > 0 else vec[i].astype(np.float32)
    np.savez_compressed(a.out / f"{name}.npz", **out)
    report["directions"][name] = info
    print(name, info)


# ---- judge-derived, within-case pairs -------------------------------------------
by_case = {}
for e in episodes:
    by_case.setdefault(e["task"], []).append(e)
for d in ("GC", "IF", "IE", "OE"):
    diffs, cases = [], []
    for task, runs in by_case.items():
        if dim(task) != d or len(runs) < 2:
            continue
        hi, lo = max(runs, key=lambda r: r["score"]), min(runs, key=lambda r: r["score"])
        if hi["score"] - lo["score"] >= a.gap:
            diffs.append(hi["acts"].mean(axis=0) - lo["acts"].mean(axis=0))
            cases.append(task)
    if len(diffs) < 4:
        report["directions"][f"score_{d}"] = {"pairs": len(diffs), "skipped": "too few pairs"}
        continue
    diffs = np.stack(diffs)   # [pairs, L, C]
    # held-out sign test: does a direction from half the pairs point the same way on the other half?
    rng, agree = random.Random(0), []
    for _ in range(200):
        idx = list(range(len(diffs)))
        rng.shuffle(idx)
        h = len(idx) // 2
        d1, d2 = diffs[idx[:h]].mean(axis=0), diffs[idx[h:]]
        proj = np.einsum("lc,plc->pl", d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-9), d2)
        agree.append((proj > 0).mean(axis=0))
    save(f"score_{d}", diffs.mean(axis=0),
         {"pairs": len(diffs), "cases": cases,
          "heldout_sign_agreement_by_layer": [round(float(x), 2) for x in np.mean(agree, axis=0)]})

# ---- event-derived, per chunk ------------------------------------------------------
chunks, mot, con = [], [], []
for e in episodes:
    centred = e["acts"] - e["acts"].mean(axis=0, keepdims=True)   # remove the episode's own mean
    for k in range(1, len(centred)):
        chunks.append(centred[k])
        mot.append(float(e["motion"][k]) / (float(np.median(e["motion"])) + 1e-9))
        con.append(float(e["cont"][k]))
chunks, mot, con = np.stack(chunks), np.array(mot), np.array(con)
for name, key in (("alive", mot), ("stable", con)):
    ok = np.isfinite(key)
    lo_t, hi_t = np.quantile(key[ok], [1 / 3, 2 / 3])
    hi, lo = chunks[ok & (key >= hi_t)], chunks[ok & (key <= lo_t)]
    save(name, hi.mean(axis=0) - lo.mean(axis=0),
         {"chunks_high": int(len(hi)), "chunks_low": int(len(lo)),
          "thresholds": [round(float(lo_t), 3), round(float(hi_t), 3)]})
(a.out / "report.json").write_text(json.dumps(report, indent=1))
