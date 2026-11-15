"""Assemble the two-minute demo video from clips already generated.

  python -I -X utf8 scripts/make_demo.py [--out demo/worldhold_demo.mp4]

Each segment plays the same benchmark case twice, side by side: the comparison on the left, the state
program's write on the right, with what was asked above and what to look for below. Clips are the ones
pulled for frame review (runs/view/<sweep>/<variant>/<case>.mp4); long clips are time-compressed to fit.
The closing card reads results/held_out.json if it exists; otherwise it says the judged run is pending.
"""
import argparse
import json
import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, FPS = 1280, 720, 24
BG, FG, DIM, ACC, BAD = (16, 18, 22), (238, 238, 232), (150, 156, 164), (120, 200, 150), (230, 140, 110)
F = "C:/Windows/Fonts/"


def font(size, bold=False):
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), ("arialbd.ttf" if bold else "arial.ttf")):
        if os.path.exists(F + name):
            return ImageFont.truetype(F + name, size)
    return ImageFont.load_default()


def wrap(draw, text, fnt, width):
    out, line = [], ""
    for word in text.split():
        t = (line + " " + word).strip()
        if draw.textlength(t, font=fnt) <= width:
            line = t
        else:
            out.append(line)
            line = word
    return out + [line]


def card(lines, seconds):
    """lines: [(text, size, bold, colour)]"""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    rows = []
    for text, size, bold, col in lines:
        f = font(size, bold)
        for t in (wrap(d, text, f, W - 160) if text else [""]):
            rows.append((t, f, col, int(size * 1.45)))
    y = (H - sum(r[3] for r in rows)) // 2
    for t, f, col, hgt in rows:
        d.text(((W - d.textlength(t, font=f)) // 2, y), t, font=f, fill=col)
        y += hgt
    fr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return [fr] * int(seconds * FPS)


def read(path):
    cap, fr = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    return fr


def segment(asked, left, right, look, seconds, left_bad=True):
    """left/right: (label, path)."""
    a, b = read(os.path.join(ROOT, left[1])), read(os.path.join(ROOT, right[1]))
    n = int(seconds * FPS)
    pw, ph = 600, int(600 * a[0].shape[0] / a[0].shape[1])
    base = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(base)
    fa, fl, fk = font(27, True), font(22, True), font(23)
    y = 34
    for t in wrap(d, "Asked: " + asked, fa, W - 120):
        d.text((60, y), t, font=fa, fill=FG)
        y += 38
    top = 150
    d.text((30, top - 34), left[0], font=fl, fill=BAD if left_bad else DIM)
    d.text((650, top - 34), right[0], font=fl, fill=ACC)
    y = top + ph + 26
    for t in wrap(d, look, fk, W - 120):
        d.text((60, y), t, font=fk, fill=DIM)
        y += 33
    base = cv2.cvtColor(np.asarray(base), cv2.COLOR_RGB2BGR)
    out = []
    for i in range(n):
        fr = base.copy()
        for clip, x in ((a, 30), (b, 650)):
            f = clip[min(len(clip) - 1, int(i / max(1, n - 1) * (len(clip) - 1)))]
            fr[top:top + ph, x:x + pw] = cv2.resize(f, (pw, ph))
        # progress bar
        cv2.rectangle(fr, (30, top + ph + 6), (30 + int(1220 * (i + 1) / n), top + ph + 10), (90, 96, 104), -1)
        out.append(fr)
    return out


V = "runs/view/"
SEGMENTS = [
    dict(asked="The character walks left into the pond.",
         left=("Unmodified model", V + "sweep2/lingbot2_14b_open_cap/IF006.mp4"),
         right=("Write 1: the request gets 8x the attention", V + "sweep2/lingbot2_14b_open_reg8/IF006.mp4"),
         look="Same model, same seed. The request sits in its own cross-attention segment and is attended to 8 times as "
              "strongly. Left: she walks past the pond. Right: she wades in.", seconds=11),
    dict(asked="The woman walks diagonally forward-right up the slope.",
         left=("Request in the prompt, normal strength", V + "sweep2/lingbot2_14b_open_obj/IF009.mp4"),
         right=("Write 1: the request gets 8x the attention", V + "sweep2/lingbot2_14b_open_reg8/IF009.mp4"),
         look="Saying it in the prompt is not enough: she walks into a bush at the foot. With the register she climbs.",
         seconds=10),
    dict(asked="Move along a rectangular path around the sculpture (30 seconds).",
         left=("Unmodified model", V + "sweep4/lingbot2_14b_open_none/GC002.mp4"),
         right=("Write 2: latents pulled back toward the first chunk", V + "sweep4/lingbot2_14b_open_anchor0.7/GC002.mp4"),
         look="Over 30 seconds the hall drifts from red to deep blue. Pulling each chunk's latent mean back toward the first "
              "chunk's holds the colours.", seconds=12),
    dict(asked="Walk forward, look around to the right, walk forward, look around again (a cabin interior).",
         left=("Without the shadow world", V + "sweep22/lingbot2_14b_open_base/GC015.mp4"),
         right=("Write 3: shadow world, written back on return", V + "sweep22/lingbot2_14b_open_shadow/GC015.mp4"),
         look="When the camera turns away, a copy of the model's whole state keeps running the view that was left. On return "
              "the arriving frame is pinned to the copy's frame through the model's own conditioning input.", seconds=14),
    dict(asked="Turn the camera right, then turn back to the original view (a western street).",
         left=("Without the shadow world", V + "sweep22/lingbot2_14b_open_base/GC010.mp4"),
         right=("Write 3: shadow world, written back on return", V + "sweep22/lingbot2_14b_open_shadow/GC010.mp4"),
         look="Geometry on the held-out half: 2.48 without this write, 2.82 with it.", seconds=10),
    dict(asked="The woman walks forward to the base of the Great Sphinx.",
         left=("Unmodified model", V + "sweep18/lingbot2_14b_open_base/IF011.mp4"),
         right=("Request at 8x attention", V + "sweep16/lingbot2_14b_open_reg8/IF011.mp4"),
         look="Where it still breaks: from a standing start on a real photograph, nothing we tried makes her walk. Attention "
              "gain to 16, camera guidance, residual steering, motion written into the noise: no advance.",
         seconds=9, left_bad=False),
]


def results_card():
    r = json.load(open(os.path.join(ROOT, "results", "full_benchmark.json")))
    a, d = r["arms"], r["final_minus_baseline"]
    v2 = json.load(open(os.path.join(ROOT, "results", "held_out_v2.json")))["arms"]["v2"]

    def row(name, t):
        return (f"{name}:  {t['overall']:.2f}   (geometry {t['GC']:.2f}, interaction {t['IF']:.2f}, "
                f"insight {t['IE']:.2f}, out-of-sight {t['OE']:.2f})")
    return card([("PlayWorld, all 171 cases. Judge: Gemini 3.1 Pro, blinded", 32, True, FG),
                 ("", 12, False, DIM),
                 (row("Unmodified model", a["baseline"]), 25, False, FG),
                 (row("Holdfast", a["final"]), 25, True, ACC),
                 ("Genie 3 (published):  2.12   (geometry 2.74, interaction 2.40, insight 1.51, out-of-sight 1.81)", 25, False, DIM),
                 ("", 12, False, DIM),
                 (f"Holdfast minus unmodified, paired: {d['overall']:+.2f}  (95% interval {d['ci95'][0]:+.2f} to {d['ci95'][1]:+.2f})",
                  25, False, FG),
                 (f"With the shadow world, on the 95 held-out cases: {v2['overall']:.2f} (geometry {v2['by_dim']['GC']:.2f}). "
                  "That arm was designed after the held-out result was known.", 23, False, DIM)], 11)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo/worldhold_demo.mp4")
    a = ap.parse_args()
    frames = card([("Holdfast", 66, True, FG),
                   ("state programs for a world model", 32, False, DIM),
                   ("", 16, False, DIM),
                   ("LingBot-World 2 on the PlayWorld benchmark. A controller writes into the model's state", 27, False, FG),
                   ("at three levels. No weights changed. Nothing trained.", 27, False, FG)], 6)
    for s in SEGMENTS:
        frames += segment(**s)
    frames += card([("Three writes, three levels of state", 36, True, FG),
                    ("attention to text   |   latents carried forward   |   the conditioning input", 27, False, DIM),
                    ("", 14, False, DIM),
                    ("Tested and dropped, with the evidence in the repo: residual steering, attention gain on the", 24, False, FG),
                    ("first frames, a view bank, camera guidance, an object ledger, a launch-time planner, noise flow.", 24, False, FG)], 7)
    frames += results_card()
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    raw = out.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-movflags", "+faststart", out], check=True)
    os.remove(raw)
    print("wrote", out, f"{len(frames) / FPS:.0f} s")
