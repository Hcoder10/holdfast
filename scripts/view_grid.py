"""Side-by-side frame strips for one case across the variants of a sweep.

  python -I -X utf8 scripts/view_grid.py sweep1 IF011 [--frames 8] [--only substr,substr]

Downloads <run>/<variant folder>/<case>/video.mp4 from the worldhold-runs volume into
runs/view/<run>/ (kept, so a second call is instant) and writes
runs/view/<run>/<case>.jpg: one row per variant, frames evenly spaced in time, with the
variant name, frame count, mean frame-to-frame change and sharpness printed.
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("cases", nargs="+")
ap.add_argument("--frames", type=int, default=8)
ap.add_argument("--only", default="")
ap.add_argument("--width", type=int, default=240)
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
env = dict(os.environ, PYTHONIOENCODING="utf-8")


def ls(path):
    out = subprocess.run(["modal", "volume", "ls", "worldhold-runs", path], capture_output=True, text=True,
                         encoding="utf-8", errors="ignore", env=env).stdout
    return sorted({t.strip(" │|┃") for line in out.splitlines() for t in line.split() if t.startswith(path + "/")})


def read(p):
    cap, fr = cv2.VideoCapture(p), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    return fr


variants = [v for v in ls(a.run) if not a.only or any(s in v for s in a.only.split(","))]
for case in a.cases:
    rows = []
    for v in variants:
        name = v.split("/", 1)[1]
        local = os.path.join("runs", "view", a.run, name, case + ".mp4")
        if not os.path.exists(local):
            os.makedirs(os.path.dirname(local), exist_ok=True)
            subprocess.run(["modal", "volume", "get", "worldhold-runs", f"{v}/{case}/video.mp4", local],
                           capture_output=True, env=env)
        if not os.path.exists(local):
            continue
        fr = read(local)
        if len(fr) < 2:
            continue
        g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fr]).astype(np.float32)
        mot, sharp = np.abs(np.diff(g, axis=0)).mean(), np.abs(np.diff(g, axis=2)).mean()
        h = int(a.width * fr[0].shape[0] / fr[0].shape[1])
        idx = np.linspace(0, len(fr) - 1, a.frames).astype(int)
        strip = np.concatenate([cv2.resize(fr[i], (a.width, h)) for i in idx], axis=1)
        label = f"{name.replace('lingbot2_14b_', '')}  n={len(fr)}  motion={mot:.1f}  sharp={sharp:.1f}"
        cv2.rectangle(strip, (0, 0), (len(label) * 9 + 8, 18), (0, 0, 0), -1)
        cv2.putText(strip, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(strip)
        print(case, label)
    if rows:
        out = os.path.join("runs", "view", a.run, case + ".jpg")
        cv2.imwrite(out, np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("wrote", out)
