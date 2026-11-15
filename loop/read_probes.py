"""A trained read interface on the recorded hidden state: what can a linear probe decode?

  python -I -X utf8 loop/read_probes.py [--acts runs/acts] [--meta runs/read_meta]

Inputs
  runs/acts/<run>/<mode>/<task>.npz      acts [chunks, 8 blocks, 5120] float16: the residual stream,
                                         mean over image tokens, one vector per 0.75 s chunk and block
                                         (blocks 4, 9, ..., 39 of 40), recorded on a clean replay of the
                                         finished video (modal_app/wh_modal.py replay_shard). The replay
                                         feeds identity camera poses, so the action is NOT an input of
                                         the recorded pass: anything decoded here is read from what the
                                         model sees and remembers, not from its conditioning.
  runs/read_meta/<run>/<mode>/<task>.json the episode's result.json from the volume
                                         (modal volume get worldhold-runs <run>/<mode>/<task>/result.json);
                                         `actions` and `records` give one entry per chunk.

Probes: per block, standardise -> PCA 256 (fitted on the training folds only) -> ridge (regression,
alpha by leave-one-out inside the training folds) or logistic regression (classification).
Cases are held out: GroupKFold(5) on the task id, so every rerun and every arm of a case sits in
one fold. Two feature sets: "raw", and "centred" (the episode's own mean vector removed, which
strips scene identity; it uses no labels but does use the whole episode, so it is an offline read).

Outputs: results/read_probes.json and docs/read_probes.png
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
LAYERS = (4, 9, 14, 19, 24, 29, 34, 39)
KIND = {"LEFT": "turn_left", "RIGHT": "turn_right", "W": "forward", "S": "back", "A": "strafe_left",
        "D": "strafe_right", "WAIT": "wait", "UP": "tilt", "DOWN": "tilt"}
KINDS = sorted(set(KIND.values()))
AWAY_YAW, AWAY_PITCH = 60.0, 45.0     # adapters/lingbot/director.py: beyond this the start view has left the frame
PCS, FOLDS = 256, 5

ap = argparse.ArgumentParser()
ap.add_argument("--acts", type=Path, default=ROOT / "runs" / "acts")
ap.add_argument("--meta", type=Path, default=ROOT / "runs" / "read_meta")
ap.add_argument("--out", type=Path, default=ROOT / "results" / "read_probes.json")
ap.add_argument("--fig", type=Path, default=ROOT / "docs" / "read_probes.png")
a = ap.parse_args()
warnings.filterwarnings("ignore")


def wrap(y):
    return (y + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- load
X, Xc, rows, episodes = [], [], [], 0
for f in sorted(a.acts.glob("*/*/*.npz")):
    m = a.meta / f.relative_to(a.acts).with_suffix(".json")
    if not m.exists():
        continue
    d = json.loads(m.read_text(encoding="utf8"))
    acts = np.load(f)["acts"].astype(np.float32)                # [T, L, C]
    recs = [r for r in d["records"] if "chunk" in r]
    T = len(acts)
    if not (T == len(recs) == len(d["actions"])) or not np.isfinite(acts).all():
        print("skipped (chunks do not line up)", f)
        continue
    episodes += 1
    X.append(acts)
    Xc.append(acts - acts.mean(axis=0, keepdims=True))
    cum, moved, been_away = 0.0, False, False
    for t, (r, act) in enumerate(zip(recs, d["actions"])):
        kind = KIND[act]
        my = r["measured_yaw"]
        g = re.search(r"yaw (?:cmd |rewind )?([-+]?\d+\.?\d*)", r["label"])
        cmd = float(g.group(1)) if g else (0.0 if kind not in ("turn_left", "turn_right") else np.nan)
        cum += my or 0.0
        moved = moved or kind in ("forward", "back", "strafe_left", "strafe_right")
        # the episode's own tracked heading, valid as "relative to the start view" until the camera translates
        facing = abs(wrap(r["est_yaw"])) < AWAY_YAW and abs(r["est_pitch"]) < AWAY_PITCH
        home = np.nan
        if not moved and been_away:
            home = 1.0 if facing else 0.0          # after having been away: back on the start view, or still away
        early = 1.0 if (not moved and facing and not been_away) else 0.0
        if not moved and not facing:
            been_away = True
        rows.append(dict(task=d["task_id"], ep=episodes, t=t, T=T, kind=KINDS.index(kind), cmd_yaw=cmd,
                         meas_yaw=np.nan if my is None else my, cum_yaw=cum,
                         off_start=np.nan if moved else abs(wrap(r["est_yaw"])), home=home,
                         facing_first=early, rot_only=0.0 if moved else 1.0))
X, Xc = np.concatenate(X), np.concatenate(Xc)
col = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "task"}
tasks = np.array([r["task"] for r in rows])
groups = np.unique(tasks, return_inverse=True)[1]
print(f"{episodes} episodes, {len(rows)} chunks, {groups.max() + 1} cases, acts {X.shape}")

# "same view, different history": start-facing chunks before the first excursion (0) vs after one (1)
revisit = np.where(col["home"] == 1.0, 1.0, np.where(col["facing_first"] == 1.0, 0.0, np.nan))

# name -> (type, y, plain description)
TARGETS = {
    "action_kind": ("clf", col["kind"].astype(float),
                    "action of the chunk, 8 classes (turn left/right, forward, back, strafe left/right, wait, tilt)"),
    "commanded_yaw": ("reg", col["cmd_yaw"], "commanded yaw of the chunk in degrees, signed (0 for non-turn chunks; "
                      "agent-chosen turns with no logged angle left out)"),
    "measured_yaw": ("reg", col["meas_yaw"], "yaw measured from the frames of the chunk, degrees, signed"),
    "cumulative_heading": ("reg", col["cum_yaw"], "sum of measured yaw from the episode start, degrees, unwrapped"),
    "offset_from_start": ("reg", col["off_start"], "|heading| relative to the start view (0-180 degrees), from the "
                          "episode's tracked heading, chunks before any translation"),
    "home_after_away": ("bin", col["home"], "after the camera has been away (>60 degrees): is it back on the start "
                        "view (1) or still away (0); chunks before any translation"),
    "returned_vs_first_visit": ("bin", revisit, "among chunks facing the start view before any translation: "
                                "has the camera been away and come back (1) or never left yet (0)"),
    "chunk_index": ("reg", col["t"].astype(float), "chunk index in the episode (elapsed time / 0.75 s)"),
}
# a readout that needs no hidden state at all: chunk index and the action kind
META = np.column_stack([col["t"], col["t"] ** 2, np.eye(len(KINDS))[col["kind"]]])
META_TIME = np.column_stack([col["t"], col["t"] ** 2])


def score(kind, y, p):
    if kind == "reg":
        return {"r2": float(r2_score(y, p)), "r": float(np.corrcoef(y, p)[0, 1])}
    if kind == "bin":
        return {"auc": float(roc_auc_score(y, p)), "acc": float(accuracy_score(y, p > 0.5))}
    return {"acc": float(accuracy_score(y, p)), "balanced_acc": float(balanced_accuracy_score(y, p))}


def fit_predict(kind, Ztr, ytr, Zte):
    if kind == "reg":
        return RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(Ztr, ytr).predict(Zte)
    clf = LogisticRegression(C=0.1, max_iter=400, class_weight="balanced" if kind == "bin" else None).fit(Ztr, ytr)
    return clf.predict_proba(Zte)[:, 1] if kind == "bin" else clf.predict(Zte)


def baseline(kind, y, g, feats=None):
    """Held-out prediction with no hidden state: the training mean / majority class, or `feats` only."""
    p = np.zeros(len(y))
    for tr, te in GroupKFold(FOLDS).split(y, y, g):
        if feats is not None:
            mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-6
            p[te] = fit_predict(kind, (feats[tr] - mu) / sd, y[tr], (feats[te] - mu) / sd)
        elif kind == "clf":
            p[te] = np.bincount(y[tr].astype(int)).argmax()
        else:
            p[te] = y[tr].mean()
    return score(kind, y, p) if not (kind == "bin" and feats is None) else {"auc": 0.5, "acc": float(
        max(y.mean(), 1 - y.mean()))}


folds = list(GroupKFold(FOLDS).split(groups, groups, groups))
pred = {(fs, n): np.full((len(LAYERS), len(rows)), np.nan) for fs in ("raw", "centred") for n in TARGETS}
for fs, F in (("raw", X), ("centred", Xc)):
    for li, layer in enumerate(LAYERS):
        for tr, te in folds:
            mu, sd = F[tr, li].mean(0), F[tr, li].std(0) + 1e-6
            pca = PCA(PCS, svd_solver="randomized", random_state=0).fit((F[tr, li] - mu) / sd)
            Ztr, Zte = pca.transform((F[tr, li] - mu) / sd), pca.transform((F[te, li] - mu) / sd)
            s = Ztr.std(0) + 1e-6
            Ztr, Zte = Ztr / s, Zte / s
            for n, (kind, y, _) in TARGETS.items():
                ok_tr, ok_te = np.isfinite(y[tr]), np.isfinite(y[te])
                pred[fs, n][li, te[ok_te]] = fit_predict(kind, Ztr[ok_tr], y[tr][ok_tr], Zte[ok_te])
        print(fs, "block", layer, "done", flush=True)

report = {"episodes": episodes, "chunks": len(rows), "cases": int(groups.max() + 1), "layers": list(LAYERS),
          "folds": FOLDS, "pcs": PCS, "kinds": KINDS,
          "kind_counts": {k: int((col["kind"] == i).sum()) for i, k in enumerate(KINDS)},
          "note": "activations: mean over image tokens per chunk and block, recorded on a clean replay of the "
                  "finished video with identity camera poses; scores are held out by case (GroupKFold on task id)",
          "targets": {}}
MAIN = {"clf": "acc", "reg": "r2", "bin": "auc"}
for n, (kind, y, desc) in TARGETS.items():
    ok = np.isfinite(y)
    key = MAIN[kind]
    out = {"type": kind, "metric": key, "description": desc, "n_chunks": int(ok.sum()),
           "n_cases": int(len(np.unique(groups[ok]))), "n_episodes": int(len(np.unique(col["ep"][ok]))),
           "baseline": baseline(kind, y[ok], groups[ok]),
           "baseline_time_only": baseline(kind, y[ok], groups[ok], META_TIME[ok]),
           "baseline_time_and_action": None if n == "action_kind" else baseline(kind, y[ok], groups[ok], META[ok])}
    if kind == "bin":
        out["positive_rate"] = float(y[ok].mean())
    for fs in ("raw", "centred"):
        per = [score(kind, y[ok], pred[fs, n][li, ok]) for li in range(len(LAYERS))]
        best = int(np.argmax([s[key] for s in per]))
        # spread of the best block's score over cases resampled with replacement
        rng, gs, bs = np.random.default_rng(0), np.unique(groups[ok]), []
        idx = {g: np.flatnonzero(groups[ok] == g) for g in gs}
        for _ in range(300):
            take = np.concatenate([idx[g] for g in rng.choice(gs, len(gs))])
            try:
                bs.append(score(kind, y[ok][take], pred[fs, n][best, ok][take])[key])
            except ValueError:
                pass
        out[fs] = {"per_block": {str(l): s for l, s in zip(LAYERS, per)}, "best_block": LAYERS[best],
                   "best": per[best][key], "best_ci95_over_cases": [float(np.percentile(bs, 2.5)),
                                                                    float(np.percentile(bs, 97.5))]}
    report["targets"][n] = out
    b = out["baseline"][key]
    bm = out["baseline_time_and_action"] or out["baseline_time_only"]
    print(f"{n:26s} {key:4s} n={out['n_chunks']:5d} base {b:6.3f} time(+action) {bm[key]:6.3f} | "
          + " | ".join(f"{fs} best {out[fs]['best']:6.3f} @ block {out[fs]['best_block']}" for fs in ("raw", "centred")))
    for fs in ("raw", "centred"):
        print("   ", fs, " ".join(f"{out[fs]['per_block'][str(l)][key]:6.3f}" for l in LAYERS))

a.out.write_text(json.dumps(report, indent=1), encoding="utf8")

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TITLES = {"action_kind": "Action of the chunk (8 classes)", "commanded_yaw": "Commanded yaw of the chunk",
          "measured_yaw": "Measured yaw of the chunk", "cumulative_heading": "Cumulative heading since the start",
          "offset_from_start": "Angle away from the start view", "home_after_away": "Back home vs still away",
          "returned_vs_first_visit": "Start view: returned vs never left", "chunk_index": "Chunk index (elapsed time)"}
YLAB = {"acc": "held-out accuracy", "r2": "held-out R²", "auc": "held-out AUC"}
fig, axes = plt.subplots(2, 4, figsize=(17, 7.6), facecolor="white")
for ax, (n, t) in zip(axes.ravel(), report["targets"].items()):
    key = t["metric"]
    for fs, c, lab in (("raw", "#1f5fa8", "probe on activations"),
                       ("centred", "#d1701a", "probe, episode mean removed")):
        ax.plot(LAYERS, [t[fs]["per_block"][str(l)][key] for l in LAYERS], "-o", color=c, lw=2, ms=5, label=lab)
    ax.axhline(t["baseline"][key], color="#555555", ls="--", lw=1.3,
               label="majority class" if key == "acc" else "chance" if key == "auc" else "predict the mean")
    extra = t["baseline_time_and_action"] or t["baseline_time_only"]
    ax.axhline(extra[key], color="#2a9d5c", ls=":", lw=1.8,
               label="no hidden state: time" + (" + action" if t["baseline_time_and_action"] else " only"))
    lo = min(0.0, t["baseline"][key]) if key == "r2" else 0.0 if key == "acc" else 0.4
    ax.set_ylim(max(lo, -0.3) - 0.03, 1.03)
    ax.set_xticks(LAYERS)
    ax.set_title(f"{TITLES[n]}\n{t['n_chunks']} chunks, {t['n_cases']} cases", fontsize=11)
    ax.set_xlabel("transformer block (of 40)")
    ax.set_ylabel(YLAB[key])
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="best")
fig.suptitle("What a linear probe reads from the recorded hidden state, cases held out "
             f"({report['episodes']} episodes, {report['cases']} cases, 5 folds by case)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
a.fig.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.fig, dpi=140, facecolor="white")
print("wrote", a.out, a.fig)
