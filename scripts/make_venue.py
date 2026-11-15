"""Side-by-side of the unmodified model and Holdfast on photographs of the venue.

  python -I -X utf8 scripts/view_grid.py venue1 VENUE_hall_right VENUE_hall_left VENUE_hall_wait VENUE_stage_right VENUE_stage_wait
  python -I -X utf8 scripts/make_venue.py

Reads runs/view/venue1/<variant>/<case>.mp4 and writes results/venue/venue_side_by_side.mp4. The photographs
show bystanders, so results/venue/ is git-ignored and the video stays out of the repository.
"""
import os
import subprocess
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_demo as m  # noqa: E402

U = "runs/view/venue2/lingbot2_14b_open_unmodified/"      # unmodified model
HF = "runs/view/venue3/lingbot2_14b_open_holdfast/"      # Holdfast (venue3: start view pinned as a later-slot latent)
CASES = [
    ("VENUE_corridor_walk", "Walk 8 steps down the corridor, then walk back to the start.",
     "Watch where each one ends. Holdfast keeps the starting view and writes it back on arrival.", 12),
    ("VENUE_piano_strafe", "Move the camera 5 steps right, then back to the start.",
     "Unmodified: the red curtain is gone when it comes back. Holdfast: same view as the first frame.", 11),
    ("VENUE_corridor_long", "Walk 6 steps forward and back, twice.",
     "Over 24 seconds the unmodified model invents a black cabinet and a drum. Holdfast ends where it began.", 13),
    ("VENUE_piano_walk", "Walk 6 steps toward the piano, then back to the start.", "Same request by the piano.", 10),
]


def ends_card(seconds=9):
    """First frame, then the last frame of each version, for every case: the comparison in one still."""
    import numpy as np
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (m.W, m.H), m.BG)
    d = ImageDraw.Draw(im)
    f = m.font(24, True)
    for j, t in enumerate(("First frame", "Unmodified: last frame", "Holdfast: last frame")):
        d.text((70 + j * 400, 20), t, font=f, fill=(m.FG, m.BAD, m.ACC)[j])
    for i, (case, _, _, _) in enumerate(CASES):
        a, b = m.read(os.path.join(m.ROOT, U + case + ".mp4")), m.read(os.path.join(m.ROOT, HF + case + ".mp4"))
        for j, fr in enumerate((a[0], a[-1], b[-1])):
            t = Image.fromarray(cv2.cvtColor(cv2.resize(fr, (270, 156)), cv2.COLOR_BGR2RGB))
            im.paste(t, (70 + j * 400, 60 + i * 164))
    fr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return [fr] * int(seconds * m.FPS)


if __name__ == "__main__":
    frames = m.card([("Holdfast, on this room", 60, True, m.FG),
                     ("", 14, False, m.DIM),
                     ("Two photographs taken here today. Same model, same seed, same camera path. No rotation:", 28, False, m.FG),
                     ("the camera walks or slides away and comes back.", 28, False, m.FG),
                     ("Left: LingBot-World 2 unmodified. Right: with Holdfast's state writes.", 28, False, m.FG)], 5)
    used = 0
    for case, asked, look, sec in CASES:
        a, b = U + case + ".mp4", HF + case + ".mp4"
        if not (os.path.exists(os.path.join(m.ROOT, a)) and os.path.exists(os.path.join(m.ROOT, b))):
            print("missing", case)
            continue
        frames += m.segment(asked, ("Unmodified model", a), ("Holdfast", b), look, sec)
        used += 1
    frames += ends_card()
    out = os.path.join(m.ROOT, "results", "venue", "venue_no_rotation.mp4")
    raw = out.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), m.FPS, (m.W, m.H))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-movflags", "+faststart", out], check=True)
    os.remove(raw)
    print("wrote", out, used, "segments", f"{len(frames) / m.FPS:.0f} s")
