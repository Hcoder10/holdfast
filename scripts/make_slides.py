"""Slides for the judges: python -I -X utf8 scripts/make_slides.py [out_dir]

Writes slide_01.png ... and holdfast_slides.pdf (1600x900). Frames come from benchmark clips in demo/script_clips
and runs/view; every number is from results/ (see README).
"""
import os
import sys

import cv2
from PIL import Image, ImageDraw, JpegImagePlugin, PdfImagePlugin  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_demo as m  # noqa: E402

W, H = 1600, 900
BG, FG, DIM, GOOD, BAD, WARN, PANEL = (14, 16, 20), (238, 238, 232), (150, 156, 164), (120, 200, 150), (230, 140, 110), (240, 200, 90), (23, 26, 32)
ROOT = m.ROOT
C = os.path.join(ROOT, "demo", "script_clips")


def frame(path, frac, size):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(n * frac) - 1))
    ok, f = cap.read()
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize(size)


def new(title, kicker=None):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    if kicker:
        d.text((80, 56), kicker.upper(), font=m.font(24, True), fill=GOOD)
    d.text((80, 92 if kicker else 70), title, font=m.font(58, True), fill=FG)
    return im, d


def para(d, x, y, text, size=32, col=FG, width=None, bold=False, gap=1.38):
    f = m.font(size, bold)
    for t in m.wrap(d, text, f, width or (W - x - 80)):
        d.text((x, y), t, font=f, fill=col)
        y += int(size * gap)
    return y


def bullets(d, x, y, items, size=32, width=None, gap=18):
    for it in items:
        col = FG
        if isinstance(it, tuple):
            it, col = it
        d.ellipse([x, y + size * 0.52, x + 10, y + size * 0.52 + 10], fill=GOOD)
        y = para(d, x + 30, y, it, size, col, width) + gap
    return y


def pair(im, d, y, before, after, lb, la, fb=0.92, fa=0.92, w=700):
    h = int(w * 480 / 832)
    im.paste(frame(before, fb, (w, h)), (80, y))
    im.paste(frame(after, fa, (w, h)), (W - 80 - w, y))
    d.text((80, y + h + 10), lb, font=m.font(28, True), fill=BAD)
    d.text((W - 80 - w, y + h + 10), la, font=m.font(28, True), fill=GOOD)
    return y + h + 60


def table(d, x, y, rows, widths, size=30, head=True):
    for i, row in enumerate(rows):
        cx = x
        hl = row and row[0].startswith("*")
        for j, cell in enumerate(row):
            cell = cell.lstrip("*")
            f = m.font(size, bold=(head and i == 0) or hl)
            col = DIM if (head and i == 0) else GOOD if hl else FG
            d.text((cx, y), cell, font=f, fill=col)
            cx += widths[j]
        y += int(size * 1.75)
        if head and i == 0:
            d.line([(x, y - 10), (x + sum(widths), y - 10)], fill=(60, 66, 76), width=2)
    return y


slides = []

# 1 title
im, d = new("")
d.text((80, 190), "Holdfast", font=m.font(150, True), fill=FG)
y = para(d, 86, 380, "Programming a world model's hidden state.", 54, (200, 205, 212))
y = para(d, 86, y + 4, "No training. Not one weight changed.", 54, GOOD)
para(d, 86, y + 60, "An open world model (LingBot-World 2) scores 2.31 on PlayWorld. Genie 3's published score is 2.12.", 34, FG, 1300)
d.text((86, H - 110), "Track 4, World Models   ·   Exa   ·   past.dev   ·   Modal", font=m.font(30), fill=DIM)
d.text((86, H - 66), "github.com/Hcoder10/holdfast", font=m.font(30), fill=DIM)
slides.append(im)

# 2 the idea
im, d = new("Weights are not the only thing you can program", "The idea")
y = para(d, 80, 200, "Weight-space programming edits a model's weights like code. A model also has a hidden state: the working memory it uses while it generates. In a world model that is where the scene lives.", 36, FG, 1400)
y = para(d, 80, y + 26, "Holdfast is a small controller around the model. Every 0.75 seconds it reads that state and writes a small correction back in.", 36, FG, 1400)
y += 36
for i, (a, b) in enumerate((("What it listens to", "attention to the text"), ("What it carries forward", "the latents passed between chunks"), ("What it is shown", "the image it is conditioned on"))):
    x = 80 + i * 490
    d.rounded_rectangle([x, y, x + 450, y + 170], radius=14, fill=PANEL, outline=(42, 47, 56))
    d.text((x + 26, y + 30), a, font=m.font(34, True), fill=GOOD)
    para(d, x + 26, y + 86, b, 28, DIM, 400)
para(d, 80, y + 220, "Three places to write. Switch the writes off and you have the original model.", 34, WARN, 1400)
slides.append(im)

# 3 problems
im, d = new("Three habits that break the illusion", "The problem")
bullets(d, 80, 220, ["It does not listen. Ask the character to walk into the pond and she walks past it.",
                     "It drifts. Each piece of video is built on the last, so small errors pile up and colours slide within 30 seconds.",
                     "It forgets. It only remembers about 3 seconds. Look away and back, and it makes up a new room."], 40, 1380, 34)
para(d, 80, 700, "Each fix below is one write, aimed at one of these.", 34, DIM)
slides.append(im)

# 4 fix 1
im, d = new("Fix 1: make it listen", "A write into attention")
y = pair(im, d, 190, os.path.join(C, "1a_request_ignored_pond_BEFORE.mp4"), os.path.join(C, "1a_request_ignored_pond_AFTER.mp4"),
         "Untouched: walks past the pond", "Holdfast: wades in")
y = para(d, 80, y + 6, "The request is a few words buried in a long scene description. We give it its own slot in cross-attention and 8 times the weight, in all 40 layers.", 32, FG, 1440)
para(d, 80, y + 14, "Interaction score: 2.05 to 2.30 (50 cases)", 40, GOOD, bold=True)
slides.append(im)

# 5 fix 2
im, d = new("Fix 2: stop the drift", "A write into the latents it carries forward")
y = pair(im, d, 190, os.path.join(C, "2_colour_drift_hall_BEFORE.mp4"), os.path.join(C, "2_colour_drift_hall_AFTER.mp4"),
         "Untouched: the hall turns blue", "Holdfast: colours hold", 0.95, 0.95)
y = para(d, 80, y + 6, "After each chunk we pull its overall latent statistics halfway back toward the first chunk's. Content is free to change; only the slow slide is corrected.", 32, FG, 1440)
para(d, 80, y + 14, "Insight score: 2.30 to 2.57 (30 cases)", 40, GOOD, bold=True)
slides.append(im)

# 6 fix 3
V = os.path.join(ROOT, "runs", "view", "sweep31")
im, d = new("Fix 3: give it a memory", "Checkpoints of its state, written back")
y = pair(im, d, 190, os.path.join(V, "lingbot2_14b_open_shadow", "GC025.mp4"), os.path.join(V, "lingbot2_14b_open_lm", "GC025.mp4"),
         "Without: a different hallway", "With: the same room again", 0.62, 0.62)
y = para(d, 80, y + 6, "Wherever the camera stops, the controller keeps a checkpoint of what the model was showing. When the camera comes back, the checkpoint is handed to the model through the input it uses for its first frame.", 32, FG, 1440)
para(d, 80, y + 14, "The same input the model uses for its very first frame, reused as a memory.", 34, GOOD, 1440)
slides.append(im)

# 7 results
im, d = new("Results: all 171 PlayWorld cases", "Judged by Gemini 3.1 Pro on blinded videos")
y = table(d, 80, 220, [["", "Geometry", "Interaction", "Insight", "Out-of-sight", "Overall"],
                       ["LingBot-World 2, untouched", "2.35", "2.05", "2.30", "1.68", "2.10"],
                       ["*Holdfast", "2.45", "2.30", "2.57", "1.94", "2.31"],
                       ["Genie 3 (published)", "2.74", "2.40", "1.51", "1.81", "2.12"]], [520, 190, 210, 170, 220, 150], 34)
bullets(d, 80, y + 30, ["Case by case against the untouched model: +0.22 (95% interval +0.03 to +0.40). 62 cases better, 40 worse.",
                        "Settings were chosen on half the benchmark and run once on the other half: 2.10 to 2.31 there too.",
                        "What makes up the gain: telling the model what was asked, Fix 1 and Fix 2."], 30, 1400, 14)
slides.append(im)

# 8 exa
im, d = new("Exa runs inside the world model", "Exa")
y = 210
for lab, txt, col in (("1. The controller looks at the first frame and asks Exa", "How far can a Labrador Retriever walk in 10 seconds at a typical walking pace?", FG),
                      ("2. Exa answers, with a source", "7.8 to 11.6 metres (25.6 to 38.1 feet). Source: American Journal of Veterinary Research, gait study.", WARN),
                      ("3. The answer is written into the model's cross-attention", "\"The yellow Labrador on the leash is now positioned further down the road, approximately twenty-five to thirty-eight feet ahead of where it started...\"", GOOD)):
    d.text((80, y), lab.upper(), font=m.font(22, True), fill=DIM)
    y = para(d, 80, y + 34, txt, 32, col, 1440) + 22
bullets(d, 80, y + 6, ["Ran in 47 of 52 first-person episodes of the scored run, each with a source link.",
                       "Also found 15 recent papers for our failure modes and verified 4 of 4 numbers we quote."], 30, 1400, 12)
slides.append(im)

# 9 past.dev
im, d = new("past.dev is the model's memory between runs", "past.dev")
y = bullets(d, 80, 210, ["Scene memory. The first time a scene is seen, one agent stores what Exa found. The next time, another recalls it in about 1 second instead of 7, with no search.",
                         "It decides what gets written into the model. One agent believed a write fixed frozen scenes. Another measured it: 2.19 fell to 1.59. past.dev marked the belief as replaced and the planner dropped that write on 8 of 8 scenes."], 32, 1400, 22)
table(d, 80, y + 14, [["How the planner learns what others measured", "Settings right", "Prompt size"],
                      ["No shared memory", "77%", "about 780 tokens"],
                      ["Whole log pasted into the prompt", "100%", "about 6,480 tokens"],
                      ["*past.dev recall", "92%", "about 1,350 tokens"]], [760, 280, 320], 30)
para(d, 80, H - 96, "Nine agents, three read boundaries.", 26, DIM, 1440)
slides.append(im)

# 10 modal + close
im, d = new("Every rollout ran on Modal", "Modal, and try it yourself")
y = bullets(d, 80, 210, ["About 35 sweeps over the weekend on H200s, up to 19 GPUs at once, one case per GPU.",
                         "Jobs launch on Modal's side, so they kept running with the laptop closed.",
                         "Live demo: upload any photo, and two GPUs generate the untouched model and Holdfast side by side in about 75 seconds."], 34, 1400, 24)
y = para(d, 80, y + 40, "Try it", 36, WARN, bold=True)
bullets(d, 80, y + 10, ["Live demo: carpediemhari--holdfast-live-web.us-east.modal.run",
                        "Code, every result and the full write-up: github.com/Hcoder10/holdfast"], 30, 1400, 12)
d.text((80, H - 80), "github.com/Hcoder10/holdfast      Hidden state is a programming surface.", font=m.font(32, True), fill=GOOD)
slides.append(im)

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "demo", "slides")
os.makedirs(out, exist_ok=True)
for i, s in enumerate(slides, 1):
    s.save(os.path.join(out, f"slide_{i:02d}.png"))
slides[0].save(os.path.join(out, "holdfast_slides.pdf"), save_all=True, append_images=slides[1:], resolution=150)
print("wrote", len(slides), "slides to", out)
