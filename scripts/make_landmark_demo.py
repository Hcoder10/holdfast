"""Landmarks demo with a visualizer of what the controller keeps and writes back.

  python -I -X utf8 scripts/make_landmark_demo.py            # demo/landmarks_demo.mp4 (benchmark cases)
  python -I -X utf8 scripts/make_landmark_demo.py --venue    # adds the venue corridor; results/venue/ (not in git)

Top: the same case without and with landmarks (same seed, same actions). Bottom: the camera's position along its
path over time, drawn from the run's own log (result.json): a flag where a checkpoint of the model's state was
kept (the start, the end of each move), and a green ring where the camera arrived back at one and that
checkpoint's frame was written into the model's image-condition input. Clips come from scripts/view_grid.py;
logs from the worldhold-runs volume (runs/lmdemo/<case>.json).
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_demo as m  # noqa: E402

STEP = {"forward": 1, "back": -1, "strafe_right": 1, "strafe_left": -1}
V31, VV = "runs/view/sweep31/lingbot2_14b_open_", "runs/view/"
CASES = [
    dict(case="GC025", asked="Walk into the room, walk back out, then walk in again.",
         left=("Shadow world only", V31 + "shadow/GC025.mp4"), right=("With landmarks", V31 + "lm/GC025.mp4"),
         note="Judged twice: 2.4 and 1.0 without, 3.2 and 3.8 with.", seconds=16),
    dict(case="OE043", asked="Move the camera left 5 steps, then move back to the starting point.",
         left=("Shadow world only", V31 + "shadow/OE043.mp4"), right=("With landmarks", V31 + "lm/OE043.mp4"),
         note="Judged twice: 1.0 and 1.0 without, 4.0 and 4.0 with.", seconds=12),
]
VENUE = dict(case="VENUE_corridor_long", asked="This room: walk 6 steps forward and back, twice.",
             left=("Unmodified model", VV + "venue2/lingbot2_14b_open_unmodified/VENUE_corridor_long.mp4"),
             right=("Holdfast with landmarks", VV + "venue3/lingbot2_14b_open_holdfast/VENUE_corridor_long.mp4"),
             note="Not a benchmark case: a photograph taken at the venue.", seconds=16)


def track(res):
    """Per chunk: position along the path (scripted moves only), and the events the log records."""
    pos, p, kinds = [0.0], 0.0, []
    for r in res["records"]:
        lab = str(r["label"])
        agent = lab.startswith("agent")
        if lab in STEP and not agent:
            p += STEP[lab]
        pos.append(p)
        kinds.append(None if agent else lab)
    flags, last = [(0, 0.0, "start")], None
    for i, k in enumerate(kinds):
        if k is None:
            continue
        if last is not None and k != last:
            flags.append((i, pos[i], "end of " + last.replace("_", " ")))
        last = k
    pins = []
    for r in (res.get("bank") or {}).get("reads") or []:
        if r.get("via") == "landmark":
            c = r["at_chunk"]
            src = 0.0 if r["from_chunk"] < 0 else pos[r["from_chunk"] + 1]
            name = "start" if r["from_chunk"] < 0 else next((w for fc, _, w in flags if fc == r["from_chunk"] + 1), "earlier")
            pins.append((c + 1, src, name))
    return pos, flags, pins


def segment(case, asked, left, right, note, seconds):
    a, b = m.read(os.path.join(m.ROOT, left[1])), m.read(os.path.join(m.ROOT, right[1]))
    res = json.load(open(os.path.join(m.ROOT, "runs", "lmdemo", case + ".json"), encoding="utf8"))
    pos, flags, pins = track(res)
    nch = len(pos) - 1
    n = int(seconds * m.FPS)
    pw, ph = 520, int(520 * a[0].shape[0] / a[0].shape[1])
    fa, fl, fs, ft = m.font(25, True), m.font(21, True), m.font(18), m.font(17)
    base = Image.new("RGB", (m.W, m.H), m.BG)
    d = ImageDraw.Draw(base)
    d.text((40, 16), "Asked: " + asked, font=fa, fill=m.FG)
    top = 92
    d.text((40, top - 30), left[0], font=fl, fill=m.BAD)
    d.text((600, top - 30), right[0], font=fl, fill=m.ACC)
    # right-hand legend
    lx = 1140
    for i, t in enumerate(("What the", "controller", "keeps and", "writes:")):
        d.text((lx, top + i * 24), t, font=fs, fill=m.DIM)
    d.polygon([(lx + 4, top + 118), (lx + 4, top + 96), (lx + 22, top + 104)], fill=(240, 200, 90))
    d.text((lx + 30, top + 96), "checkpoint", font=ft, fill=m.FG)
    d.text((lx + 30, top + 114), "kept", font=ft, fill=m.FG)
    d.ellipse([lx, top + 150, lx + 20, top + 170], outline=m.ACC, width=3)
    d.text((lx + 30, top + 144), "arrived:", font=ft, fill=m.FG)
    d.text((lx + 30, top + 162), "written", font=ft, fill=m.FG)
    d.text((lx + 30, top + 180), "back", font=ft, fill=m.FG)
    # plot frame
    x0, x1, y0, y1 = 110, 1230, top + ph + 44, m.H - 74
    lo, hi = min(pos) - 0.6, max(pos) + 0.6
    X = lambda c: x0 + (x1 - x0) * c / nch                      # noqa: E731
    Y = lambda p: y1 - (y1 - y0) * (p - lo) / (hi - lo)         # noqa: E731
    d.rectangle([x0, y0, x1, y1], outline=(70, 76, 84))
    d.text((x0, y0 - 26), "Camera position along its path (steps from the start), from the run's log", font=fs, fill=m.DIM)
    d.text((x0, y1 + 4), "time", font=ft, fill=m.DIM)
    d.text((x1 - 60, y1 + 4), f"{nch * 0.75:.0f} s", font=ft, fill=m.DIM)
    d.text((40, Y(0) - 10), "start", font=ft, fill=m.DIM)
    d.text((x0, m.H - 34), note, font=fs, fill=m.DIM)
    base = np.asarray(base)
    out = []
    for i in range(n):
        f = i / max(1, n - 1)
        c = f * nch
        im = Image.fromarray(base.copy())
        dr = ImageDraw.Draw(im)
        for fc, fp, why in flags:                 # checkpoint lines appear when the checkpoint is taken
            if c >= fc:
                yy = Y(fp)
                for xx in range(int(X(fc)), x1, 12):
                    dr.line([(xx, yy), (xx + 6, yy)], fill=(120, 104, 60), width=1)
                dr.polygon([(X(fc), yy), (X(fc), yy - 22), (X(fc) + 18, yy - 14)], fill=(240, 200, 90))
                dr.text((X(fc) + 22, yy - 26), "checkpoint: " + why, font=ft, fill=(240, 200, 90))
        pts = [(X(k), Y(pos[k])) for k in range(int(c) + 1)]
        k = int(c)
        if k < nch:
            pts.append((X(c), Y(pos[k] + (pos[k + 1] - pos[k]) * (c - k))))
        if len(pts) > 1:
            dr.line(pts, fill=m.FG, width=3)
        cx, cy = pts[-1]
        dr.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=m.FG)
        for pc, pp, src in pins:
            if c >= pc - 1:
                age = c - (pc - 1)
                rad = 12 + (8 * max(0.0, 1 - age / 2.0))
                dr.ellipse([X(pc) - rad, Y(pp) - rad, X(pc) + rad, Y(pp) + rad], outline=m.ACC, width=4)
                up = Y(pp) < (y0 + y1) / 2
                dr.text((X(pc) - 170, Y(pp) + (18 if up else -40)), f"arrived: '{src}' checkpoint written back", font=ft, fill=m.ACC)
        fr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        for clip, x in ((a, 40), (b, 600)):
            g = clip[min(len(clip) - 1, int(f * (len(clip) - 1)))]
            fr[top:top + ph, x:x + pw] = cv2.resize(g, (pw, ph))
        out.append(fr)
    out += [out[-1]] * int(1.5 * m.FPS)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", action="store_true")
    a = ap.parse_args()
    frames = m.card([("Landmarks", 62, True, m.FG),
                     ("checkpoints of the model's state, written back on arrival", 30, False, m.DIM),
                     ("", 14, False, m.DIM),
                     ("The model attends to three seconds. Leave a place and come back, and it is generated again.", 25, False, m.FG),
                     ("The controller keeps the latent of every view where the camera stopped or changed action,", 25, False, m.FG),
                     ("and pins the arriving frame to it through the model's own image-condition input.", 25, False, m.FG)], 6)
    for s in ([VENUE] if a.venue else []) + CASES:
        frames += segment(**s)
    frames += m.card([("Ten development cases, shadow world alone vs with landmarks, judged twice, blinded", 27, True, m.FG),
                      ("", 12, False, m.DIM),
                      ("Geometry 1.77 to 2.19    Out-of-sight 1.00 to 2.21", 30, False, m.ACC),
                      ("Paired difference +0.81, 95% interval +0.04 to +1.70 (5 better, 2 worse)", 25, False, m.FG),
                      ("", 12, False, m.DIM),
                      ("Ten cases and a noisy judge: a lead, not an established gain.", 23, False, m.DIM)], 6)
    out = os.path.join(m.ROOT, "results", "venue", "landmarks_demo_venue.mp4") if a.venue else os.path.join(m.ROOT, "demo", "landmarks_demo.mp4")
    raw = out.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), m.FPS, (m.W, m.H))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-movflags", "+faststart", out], check=True)
    os.remove(raw)
    print("wrote", out, f"{len(frames) / m.FPS:.0f} s")
