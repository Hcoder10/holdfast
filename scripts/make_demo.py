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
         right=("Objective register, 8x attention", V + "sweep2/lingbot2_14b_open_reg8/IF006.mp4"),
         look="Same model, same seed. The objective gets its own cross-attention segment and 8x the attention. "
              "Left: she walks past the pond. Right: she wades in.", seconds=11),
    dict(asked="The woman walks diagonally forward-right up the slope.",
         left=("Objective in the prompt (normal strength)", V + "sweep2/lingbot2_14b_open_obj/IF009.mp4"),
         right=("Objective register, 8x attention", V + "sweep2/lingbot2_14b_open_reg8/IF009.mp4"),
         look="Saying it in the prompt is not enough: she walks into a bush at the foot. With the register she climbs.",
         seconds=11),
    dict(asked="Turn the camera right 180 degrees, turn back, wait, turn left 90 degrees, turn back.",
         left=("Unmodified model", V + "sweep19/lingbot2_14b_open_base/OE044.mp4"),
         right=("Caption gate", V + "sweep19/lingbot2_14b_open_gate/OE044.mp4"),
         look="The caption describes the starting view all episode, so the model paints tomatoes into every direction "
              "and the judge reads the camera as never leaving. Gated by measured heading, the away views are just kitchen "
              "and the cutting board comes back on return.", seconds=15),
    dict(asked="Turn right 180 degrees, wait two seconds, then turn back to the original view.",
         left=("Unmodified model", V + "sweep15/lingbot2_14b_open_base/OE053.mp4"),
         right=("Caption gate", V + "sweep15/lingbot2_14b_open_gate0.1/OE053.mp4"),
         look="Left: bakery in every direction, which reads as one endless pan. Right: a window and a wall while away, "
              "the counter again on return.", seconds=12),
    dict(asked="Walk forward, turn right 60 degrees, then turn back to the starting direction.",
         left=("Unmodified model", V + "sweep10/lingbot2_14b_open_base/OE019.mp4"),
         right=("Return register (Exa-sourced sentence)", V + "sweep10/lingbot2_14b_open_evo4/OE019.mp4"),
         look="An agent asks Exa how far a streetcar moves in ten seconds and writes one sentence for the view on return. "
              "It is silent while the camera is away and ramps in as it turns back. Left: the red tram is replaced by a modern one. "
              "Right: the red tram, closer.", seconds=13),
    dict(asked="Move along a rectangular path around the sculpture (30 seconds).",
         left=("Unmodified model", V + "sweep4/lingbot2_14b_open_none/GC002.mp4"),
         right=("Latent re-anchoring", V + "sweep4/lingbot2_14b_open_anchor0.7/GC002.mp4"),
         look="Over 30 seconds the hall drifts from red to deep blue. Pulling each chunk's latent mean back toward the first "
              "chunk's holds the colours.", seconds=12),
    dict(asked="The woman walks forward to the base of the Great Sphinx.",
         left=("Unmodified model", V + "sweep18/lingbot2_14b_open_base/IF011.mp4"),
         right=("Objective register, 8x attention", V + "sweep16/lingbot2_14b_open_reg8/IF011.mp4"),
         look="Where it still breaks: from a standing start on a real photograph, nothing we tried makes her walk. "
              "Register gain to 16, camera guidance, less attention to the first frames: no advance.",
         seconds=9, left_bad=False),
]


def results_card():
    p = os.path.join(ROOT, "results", "held_out.json")
    if not os.path.exists(p):
        return card([("Held-out half (95 cases, never used for development)", 34, True, FG),
                     ("Gemini 3.1 Pro judging in progress when this was rendered", 28, False, DIM)], 6)
    r = json.load(open(p))
    a = r["arms"]

    def row(name, t):
        return f"{name}:  overall {t['overall']:.2f}   (geometry {t['GC']:.2f}, interaction {t['IF']:.2f}, " \
               f"insight {t['IE']:.2f}, out-of-sight {t['OE']:.2f})"
    d = r["paired"]["final_minus_baseline"]
    return card([("Held-out half: 95 cases never used for development", 34, True, FG),
                 (f"Gemini 3.1 Pro, PlayWorld's prompts, blinded, {r['passes']} judging passes averaged", 24, False, DIM),
                 ("", 14, False, DIM),
                 (row("Unmodified model", a["baseline"]), 25, False, FG),
                 (row("Final state program", a["final"]), 25, True, ACC),
                 ("Genie 3 (published, 171 cases):  overall 2.12   (2.74, 2.40, 1.51, 1.81)", 25, False, DIM),
                 ("", 14, False, DIM),
                 (f"Paired difference, final minus unmodified: {d['overall']:+.2f}  (95% interval {d['ci95'][0]:+.2f} to {d['ci95'][1]:+.2f})",
                  26, False, FG)], 10)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo/worldhold_demo.mp4")
    a = ap.parse_args()
    frames = card([("worldhold", 64, True, FG),
                   ("How long can a generated world hold together while an agent plays in it?", 30, False, DIM),
                   ("", 16, False, DIM),
                   ("LingBot-World 2 on PlayWorld, with a state program written into its forward pass.", 28, False, FG),
                   ("No weights changed. Nothing trained.", 28, False, FG)], 6)
    for s in SEGMENTS:
        frames += segment(**s)
    frames += card([("What the controller writes, every 0.75 s", 36, True, FG),
                    ("cross-attention logits  ·  the caption's gain by measured heading  ·  latent statistics", 26, False, DIM),
                    ("", 14, False, DIM),
                    ("Exa supplies what should have changed out of sight; past.dev keeps the facts, the sentences", 26, False, FG),
                    ("and which program worked, behind three read boundaries.", 26, False, FG)], 7)
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
