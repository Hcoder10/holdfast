"""Render the three whiteboard-style explainer GIFs for the README.

    python -I scripts/make_explainers.py

Writes docs/explainers/{register,reanchor,shadow}.gif. Deterministic (seeded).
Needs Pillow and numpy only. No network, no GPU.
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540
S = 2            # supersampling factor
FPS = 10
HOLD_S = 2.5     # hold on the final frame
TARGET_S = 11.0  # drawing time; total = TARGET_S + HOLD_S

BG = (250, 249, 244)
INK = (38, 38, 48)
GRAY = (135, 135, 145)
BLUE = (28, 92, 205)
RED = (214, 48, 44)
GREEN = (26, 140, 72)
DEEPBLUE = (24, 44, 150)

PALETTE_INKS = [INK, GRAY, BLUE, RED, GREEN, DEEPBLUE] + [
    tuple(int(round(RED[k] + (DEEPBLUE[k] - RED[k]) * a)) for k in range(3))
    for a in (0.05, 1 / 7, 2 / 7, 0.38, 3 / 7, 4 / 7, 5 / 7, 6 / 7)]

FONT_DIR = "C:/Windows/Fonts"
FONT_CANDIDATES = ["Inkfree.ttf", "segoepr.ttf", "comic.ttf"]


def find_font():
    for name in FONT_CANDIDATES:
        p = os.path.join(FONT_DIR, name)
        if os.path.exists(p):
            return p
    raise SystemExit("no handwriting font found in " + FONT_DIR)


FONT_PATH = find_font()
_fonts = {}


def font(size):
    key = int(round(size * S))
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT_PATH, key)
    return _fonts[key]


def lerp(c1, c2, a):
    return tuple(int(round(c1[i] + (c2[i] - c1[i]) * a)) for i in range(3))


def resample(pts, step=6.0):
    pts = np.asarray(pts, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(cum[-1] / step) + 1)
    s = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], 1)


def bezier(p0, c, p1, n=24):
    t = np.linspace(0, 1, n)[:, None]
    p0, c, p1 = (np.asarray(v, float) for v in (p0, c, p1))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1


def stroke(d, q, color, width):
    w = max(1, int(round(width * S)))
    xy = [(float(x) * S, float(y) * S) for x, y in q]
    d.line(xy, fill=color, width=w, joint="curve")
    r = w / 2.0
    for x, y in (xy[0], xy[-1]):
        d.ellipse([x - r, y - r, x + r, y + r], fill=color)


class Board:
    """A list of timed drawing elements. `self.t` is a cursor: each element
    starts at the cursor (or `at`) and advances it by its duration."""

    def __init__(self, name, seed):
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.items = []
        self.t = 0.0
        self.texts = []

    def _add(self, at, dur, fn):
        at = self.t if at is None else at
        self.items.append((at, dur, fn))
        self.t = at + dur
        return at

    def wait(self, s):
        self.t += s

    def wobble(self, pts, amp):
        n = len(pts)
        z = self.rng.normal(0, 1, (n + 8, 2))
        k = np.ones(5) / 5
        zs = np.stack([np.convolve(z[:, i], k, "valid")[:n] for i in range(2)], 1)
        return pts + zs * amp * 2.0

    # ---- strokes -------------------------------------------------------
    def path(self, pts, color=INK, width=2.5, dur=0.3, at=None, amp=1.2, dash=None):
        P = self.wobble(resample(pts), amp)
        n = len(P)

        def fn(d, p, t):
            k = max(2, int(math.ceil(p * n)))
            q = P[:k]
            if dash:
                on, off = dash
                for i in range(0, k - 1, on + off):
                    seg = q[i:i + on + 1]
                    if len(seg) >= 2:
                        stroke(d, seg, color, width)
            else:
                stroke(d, q, color, width)

        self._add(at, dur, fn)
        return P

    def line(self, p0, p1, **kw):
        return self.path([p0, p1], **kw)

    def rect(self, x, y, w, h, color=INK, width=2.5, dur=0.3, at=None):
        return self.path([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y - 2)],
                         color=color, width=width, dur=dur, at=at)

    def circle(self, c, r, color=INK, width=2.5, dur=0.25, at=None, dash=None, bumps=0, bump=0.0):
        a = np.linspace(-math.pi / 2, 1.5 * math.pi + 0.25, 40)
        rr = r + bump * np.sin(bumps * a)
        pts = np.stack([c[0] + rr * np.cos(a), c[1] + rr * np.sin(a)], 1)
        return self.path(pts, color=color, width=width, dur=dur, at=at, amp=0.7, dash=dash)

    def arrow(self, pts, color=INK, width=2.5, dur=0.3, at=None, head=11, dash=None, amp=1.0):
        pts = np.asarray(pts, float)
        P = self.path(pts, color=color, width=width, dur=dur, at=at, amp=amp, dash=dash)
        tip = P[-1]
        back = pts[-2] if len(pts) < 6 else pts[-4]
        ang = math.atan2(pts[-1][1] - back[1], pts[-1][0] - back[0])
        t0 = self.t
        for da in (0.45, -0.45):
            e = (tip[0] - head * math.cos(ang + da), tip[1] - head * math.sin(ang + da))
            self.path([tip, e], color=color, width=width, dur=0.1, at=t0, amp=0.4)
        return P

    def dot(self, c, r, color, dur=0.1, at=None, to=None, morph_at=None, morph_dur=0.4):
        def fn(d, p, t):
            col = color
            if to is not None and t > morph_at:
                col = lerp(color, to, min(1.0, (t - morph_at) / morph_dur))
            rad = r * (0.4 + 0.6 * p) * S
            x, y = c[0] * S, c[1] * S
            d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=col)

        self._add(at, dur, fn)

    # ---- text ----------------------------------------------------------
    def measure(self, s, size):
        x0, y0, x1, y1 = font(size).getbbox(s, stroke_width=1)
        return (x1 - x0) / S, (y1 - y0) / S

    def text(self, x, y, s, size=24, color=INK, at=None, dur=None, anchor="la", align="l"):
        """Word by word. `anchor` 'la' = y is the top, 'ls' = y is the baseline.
        `align` 'l', 'c' or 'r' says what x means."""
        f = font(size)
        bx0, by0, bx1, by1 = f.getbbox(s, anchor=anchor, stroke_width=1)
        w = (bx1 - bx0) / S
        if align == "c":
            x = x - w / 2
        elif align == "r":
            x = x - w
        box = (x + bx0 / S, y + by0 / S, x + bx1 / S, y + by1 / S)
        self.texts.append((s, box))
        words = s.split(" ")
        if dur is None:
            dur = 0.1 * len(words) + 0.05

        def fn(d, p, t):
            k = max(1, int(math.ceil(p * len(words))))
            d.text((x * S, y * S), " ".join(words[:k]), font=f, fill=color,
                   anchor=anchor, stroke_width=1, stroke_fill=color)

        self._add(at, dur, fn)
        return box

    def custom(self, fn, dur, at=None):
        self._add(at, dur, fn)

    # ---- checks and output ---------------------------------------------
    def check(self):
        ok = True
        m = 6
        for s, (x0, y0, x1, y1) in self.texts:
            if x0 < m or y0 < m or x1 > W - m or y1 > H - m:
                print("  [%s] text off canvas: %r %s" % (self.name, s, [round(v) for v in (x0, y0, x1, y1)]))
                ok = False
        for i in range(len(self.texts)):
            for j in range(i + 1, len(self.texts)):
                a, b = self.texts[i][1], self.texts[j][1]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    print("  [%s] text overlap: %r / %r" % (self.name, self.texts[i][0], self.texts[j][0]))
                    ok = False
        return ok

    def palette(self, mosaic):
        """Fixed palette: every marker colour at five strengths over the board
        colour (covers antialiased edges), topped up from the frames."""
        cols = [BG]
        for c in PALETTE_INKS:
            for a in (0.2, 0.4, 0.6, 0.8, 1.0):
                cols.append(lerp(BG, c, a))
        extra = mosaic.quantize(colors=48, method=Image.Quantize.MEDIANCUT,
                                dither=Image.Dither.NONE).getpalette()[:48 * 3]
        for i in range(0, len(extra), 3):
            cols.append(tuple(extra[i:i + 3]))
        cols = list(dict.fromkeys(cols))[:256]
        flat = [v for c in cols for v in c]
        flat += list(cols[0]) * (256 - len(cols))
        pal = Image.new("P", (1, 1))
        pal.putpalette(flat)
        return pal

    def frame(self, t):
        im = Image.new("RGB", (W * S, H * S), BG)
        d = ImageDraw.Draw(im)
        for t0, dur, fn in self.items:
            if t > t0:
                p = 1.0 if dur <= 0 else min(1.0, (t - t0) / dur)
                fn(d, p, t)
        return im.resize((W, H), Image.LANCZOS)

    def render(self, path):
        T = max(t0 + dur for t0, dur, _ in self.items) + 0.2
        scale = max(1.0, T / TARGET_S)       # compress time if the script runs long
        n = int(math.ceil(T / scale * FPS)) + 1
        frames = [self.frame(min(T, i / FPS * scale) + 1e-6) for i in range(n)]
        # one shared palette, taken from frames spread over the animation
        picks = [frames[int(round(k))] for k in np.linspace(0, n - 1, 6)]
        self.picks = picks
        mosaic = Image.new("RGB", (W, H * len(picks)))
        for i, f in enumerate(picks):
            mosaic.paste(f, (0, H * i))
        pal = self.palette(mosaic)
        q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
        durs = [int(1000 / FPS)] * n
        durs[-1] = int(HOLD_S * 1000)
        q[0].save(path, save_all=True, append_images=q[1:], duration=durs, loop=0,
                  optimize=False, disposal=1)
        total = sum(durs) / 1000.0
        print("%s: %d frames, %.1f s, %.2f MB" % (os.path.basename(path), n, total,
                                                 os.path.getsize(path) / 1e6))


# ---- shared pieces ----------------------------------------------------------
def title(b, s):
    box = b.text(40, 12, s, 42, INK, dur=0.9)
    b.line((40, 68), (box[2] + 4, 69), color=BLUE, width=3.5, dur=0.3)
    b.wait(0.2)


def result(b, label, old, new, note, base=508):
    """One centred line: label, old score, a drawn arrow, new score, note."""
    b.wait(0.3)
    wl = b.measure(label, 30)[0]
    wo = b.measure(old, 38)[0]
    wn = b.measure(new, 50)[0]
    wt = b.measure(note, 24)[0]
    arrow = 56
    total = wl + 18 + wo + 14 + arrow + 14 + wn + 22 + wt
    x = (W - total) / 2
    b.text(x, base, label, 30, INK, anchor="ls")
    x += wl + 18
    b.text(x, base, old, 38, GRAY, anchor="ls", dur=0.2)
    x += wo + 14
    b.arrow([(x, base - 13), (x + arrow, base - 13)], color=INK, width=3, dur=0.25)
    x += arrow + 14
    b.text(x, base, new, 50, GREEN, anchor="ls", dur=0.2)
    b.line((x - 2, base + 9), (x + wn + 2, base + 8), color=GREEN, width=3.5, dur=0.2)
    x += wn + 22
    b.text(x, base, note, 24, INK, anchor="ls")


def stick_figure(b, x, y, color=INK):
    b.circle((x, y - 52), 8, color=color, width=2.5, dur=0.15)
    b.line((x, y - 44), (x, y - 18), color=color, dur=0.08, amp=0.6)
    b.line((x - 10, y - 32), (x + 10, y - 35), color=color, dur=0.08, amp=0.6)
    b.line((x, y - 18), (x - 8, y), color=color, dur=0.08, amp=0.6)
    b.line((x, y - 18), (x + 8, y), color=color, dur=0.08, amp=0.6)


# ---- 1: register -------------------------------------------------------------
def make_register():
    b = Board("register", 11)
    title(b, "Write 1: attention to the request")

    # the failure
    b.text(40, 86, 'Asked: "the woman walks up the slope"', 26, INK)
    b.path([(565, 212), (725, 212), (915, 115)], INK, 3, dur=0.4)
    stick_figure(b, 705, 212)
    b.circle((600, 190), 20, color=GREEN, width=3, dur=0.3, bumps=6, bump=4)
    b.arrow([(728, 188), (885, 108)], color=BLUE, width=3, dur=0.35, dash=(2, 2))
    b.arrow([(688, 186), (634, 188)], color=RED, width=4, dur=0.3)
    b.text(40, 126, "Got: she walks into a bush", 26, RED)
    b.wait(0.4)

    # the mechanism
    xs = np.linspace(110, 470, 5)
    for cx in xs:
        b.rect(cx - 13, 392, 26, 26, INK, 2.5, dur=0.1)
    b.text(290, 422, "image tokens", 24, INK, align="c", dur=0.2)
    b.rect(60, 250, 190, 50, INK, 2.5, dur=0.25)
    b.text(155, 258, "caption", 26, INK, align="c", dur=0.15)
    for i, cx in enumerate(xs):
        b.arrow([(cx, 387), (155 + (i - 2) * 18, 305)], color=GRAY, width=1.5, dur=0.1, head=7, amp=0.6)
    b.rect(330, 250, 190, 50, BLUE, 4, dur=0.25)
    b.text(425, 258, "request", 26, BLUE, align="c", dur=0.15)
    notes = [
        ("request gets its own segment", None),
        ("attention to it: multiplied by 8", "arrows"),
        ("(log 8 added to the logits)", None),
        ("in all 40 blocks", None),
        ("caption stays the base prompt", None),
        ("for third-person cases", None),
    ]
    y = 244
    for s, what in notes:
        if what == "arrows":
            for i, cx in enumerate(xs):
                b.arrow([(cx, 387), (425 + (i - 2) * 18, 306)], color=BLUE, width=5, dur=0.16, head=12, amp=0.6)
            b.text(506, 318, "x8", 44, RED, dur=0.2)
        b.dot((582, y + 17), 3.5, BLUE, dur=0.05)
        b.text(594, y, s, 24, INK)
        y += 32

    result(b, "interaction score", "2.05", "2.30", "(171 cases)")
    return b


# ---- 2: reanchor ---------------------------------------------------------------
def make_reanchor():
    b = Board("reanchor", 22)
    title(b, "Write 2: the latents it carries forward")

    X0, BW, GAP, BH, N = 150, 68, 14, 54, 8

    def bx(i):
        return X0 + i * (BW + GAP)

    def cx(i):
        return bx(i) + BW / 2

    # the failure
    b.text(40, 84, "Over a 30 s video the picture drifts: a red hall turns deep blue.", 26, INK)
    y1 = 128
    b.text(40, y1 + 12, "no fix", 24, RED, dur=0.2)
    for i in range(N):
        b.rect(bx(i), y1, BW, BH, INK, 2.5, dur=0.12)
        b.dot((cx(i), y1 + BH / 2), 13, lerp(RED, DEEPBLUE, i / (N - 1)), dur=0.08)
    b.text(bx(N - 1) + BW + 12, y1 + 12, "30 s", 24, INK, dur=0.2)
    b.wait(0.3)

    # the mechanism
    b.text(40, 204, "After each 0.75 s chunk: pull the mean of its latents halfway back", 24, INK)
    b.text(40, 238, "to the first chunk's mean, before it is context for the next chunk.", 24, INK)
    y2 = 352
    b.text(40, y2 + 12, "with fix", 24, GREEN, dur=0.2)
    drifted = lerp(RED, DEEPBLUE, 0.38)
    held = lerp(RED, DEEPBLUE, 0.05)
    for i in range(N):
        b.rect(bx(i), y2, BW, BH, INK, 2.5, dur=0.12)
        if i == 0:
            b.dot((cx(i), y2 + BH / 2), 13, RED, dur=0.08)
            b.text(cx(0), y2 + BH + 9, "chunk 1", 24, INK, align="c", dur=0.2)
            continue
        t_arc = b.t + 0.08
        b.dot((cx(i), y2 + BH / 2), 13, drifted, dur=0.08, to=held, morph_at=t_arc + 0.1, morph_dur=0.3)
        h = 10 + 6.5 * i
        p0 = (cx(i), y2 - 5)
        p1 = (cx(0) + (i - 4) * 4, y2 - 5)
        ctrl = ((p0[0] + p1[0]) / 2, y2 - 5 - 2 * h)
        b.arrow(bezier(p0, ctrl, p1), color=BLUE, width=2.5, dur=0.25, head=8, amp=0.6)
    b.text(bx(N - 1) + BW + 12, y2 + 12, "30 s", 24, INK, dur=0.2)

    result(b, "insight score", "2.30", "2.57", "(30 cases)")
    return b


# ---- 3: shadow -------------------------------------------------------------------
def scene_glyph(b, x, y, color, dur=0.12):
    b.path([(x + 12, y + 42), (x + 24, y + 17), (x + 36, y + 42), (x + 11, y + 42)],
           color=color, width=2.5, dur=dur, amp=0.5)
    b.circle((x + 52, y + 31), 8, color=color, width=2.5, dur=dur)


def away_glyph(b, x, y, dur=0.12):
    xs = np.linspace(x + 12, x + 58, 14)
    ys = y + 30 + 7 * np.sin(np.linspace(0, 3 * math.pi, 14))
    b.path(np.stack([xs, ys], 1), color=GRAY, width=2, dur=dur, amp=0.4)


def make_shadow():
    b = Board("shadow", 33)
    title(b, "Write 3: what it sees when it looks back")

    # the failure
    b.text(40, 86, "The camera turns away and back.", 26, INK)
    for fx, bad in ((560, False), (800, True)):
        if bad:
            b.arrow(bezier((690, 112), (740, 80), (790, 112)), color=GRAY, width=2.5, dur=0.25, head=9)
        b.rect(fx, 80, 120, 74, INK, 2.5, dur=0.2)
        b.path([(fx + 16, 142), (fx + 31, 106), (fx + 46, 142), (fx + 15, 142)], INK, 2.5, dur=0.12, amp=0.5)
        if not bad:
            b.circle((fx + 68, 128), 11, color=INK, dur=0.12)
            b.rect(fx + 88, 118, 20, 22, INK, 2.5, dur=0.12)
        else:
            b.circle((fx + 68, 128), 11, color=RED, dur=0.15, dash=(1, 1))
            b.path([(fx + 98, 114), (fx + 110, 129), (fx + 98, 144), (fx + 86, 129), (fx + 99, 113)],
                   RED, 3, dur=0.15, amp=0.5)
        b.text(fx + 60, 160, "after" if bad else "before", 24, INK, align="c", dur=0.15)
    b.text(40, 124, "The scene is not the same.", 26, RED)
    b.wait(0.3)

    # the mechanism, left: top-down camera and scene
    b.rect(70, 222, 150, 42, INK, 2.5, dur=0.2)
    b.text(145, 227, "scene", 24, INK, align="c", dur=0.1)
    keys = [(b.t, 0.0)]
    cam = (145, 372)
    jit = b.rng.normal(0, 0.8, (8, 2))
    t_cam = b.t

    def angle(t):
        ts = [k[0] for k in keys]
        return float(np.interp(t, ts, [k[1] for k in keys]))

    def cam_fn(d, p, t):
        a = math.radians(angle(t))
        ca, sa = math.cos(a), math.sin(a)

        def rot(u, v):  # u = right, v = forward (toward the scene at angle 0)
            return (cam[0] + u * ca + v * sa, cam[1] + u * sa - v * ca)

        body = [rot(-14, -10), rot(14, -10), rot(14, 10), rot(-14, 10), rot(-14, -11)]
        stroke(d, np.asarray(body) + jit[:5], INK, 3)
        for k, u in enumerate((-30, 30)):
            stroke(d, np.asarray([rot(0, 10), rot(u, 78)]) + jit[5 + k:7 + k], BLUE, 3)

    b.custom(cam_fn, 0.2)
    b.text(116, 356, "camera", 24, INK, align="r", dur=0.1)

    # the mechanism, right: two lanes of chunks
    X0, BW, GAP, BH = 410, 70, 16, 56
    Y1, Y2 = 240, 348

    def bx(i):
        return X0 + i * (BW + GAP)

    b.text(X0, 204, "main video", 24, INK, dur=0.2)
    angles = [0, 50, 105, 105, 50, 0]
    step = 0.3
    for i in range(6):
        t0 = b.t
        last = i == 5
        b.rect(bx(i), Y1, BW, BH, GREEN if last else INK, 4 if last else 2.5, dur=0.12)
        if i == 0:
            scene_glyph(b, bx(i), Y1, INK)
        elif not last:
            away_glyph(b, bx(i), Y1)
        keys.append((t0 + step, float(angles[i])))
        t1 = b.t
        b.t = t0   # the copy's chunk is drawn alongside the main one
        if i == 1:
            b.arrow([(bx(0) + 35, Y1 + BH + 5), (bx(1) + 22, Y2 - 5)], color=BLUE, width=3, dur=0.2)
            b.text(398, 312, "copy", 24, BLUE, dur=0.1)
        if i >= 1:
            b.rect(bx(i), Y2, BW, BH, BLUE, 2.5, dur=0.12)
            scene_glyph(b, bx(i), Y2, BLUE)
        b.t = max(b.t, t1, t0 + step)
    keys.append((b.t, 0.0))
    b.text(bx(5) + BW, 204, "back within 18 degrees", 24, GREEN, align="r")
    b.arrow([(bx(5) + 35, Y2 - 5), (bx(5) + 35, Y1 + BH + 7)], color=GREEN, width=4.5, dur=0.25, head=12)
    b.text(bx(5) + 22, 306, "pinned", 26, GREEN, align="r", dur=0.15)
    scene_glyph(b, bx(5), Y1, GREEN, dur=0.15)
    b.text(930, 412, "copy of the state: keeps running the view you left", 24, BLUE, align="r")
    b.text(548, 308, "its camera held still", 24, BLUE)

    result(b, "geometry score", "2.45", "2.60", "(171 cases, post hoc; total unchanged)")
    return b


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "docs", "explainers")
    os.makedirs(out, exist_ok=True)
    print("font:", FONT_PATH)
    ok = True
    for make in (make_register, make_reanchor, make_shadow):
        b = make()
        ok = b.check() and ok
        b.render(os.path.join(out, b.name + ".gif"))
    if not ok:
        print("layout warnings above")
        sys.exit(1)


if __name__ == "__main__":
    main()
