"""Make a labelled contact sheet from a video: python contact_sheet.py in.mp4 out.jpg [every_n] [cols]"""
import sys
import imageio
import numpy as np
from PIL import Image, ImageDraw

src, dst = sys.argv[1], sys.argv[2]
every = int(sys.argv[3]) if len(sys.argv) > 3 else 12
cols = int(sys.argv[4]) if len(sys.argv) > 4 else 5
frames = [f for i, f in enumerate(imageio.mimread(src, memtest=False)) if i % every == 0]
idx = [i * every for i in range(len(frames))]
h, w = frames[0].shape[:2]
tw = 416
th = int(h * tw / w)
rows = (len(frames) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tw, rows * th))
for k, f in enumerate(frames):
    im = Image.fromarray(f).resize((tw, th))
    ImageDraw.Draw(im).text((6, 4), f"f{idx[k]} t={idx[k]/16:.1f}s", fill=(255, 255, 0))
    sheet.paste(im, ((k % cols) * tw, (k // cols) * th))
sheet.save(dst, quality=85)
print(dst, len(frames), "frames")
