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

V = "runs/view/venue1/lingbot2_14b_open_"
TURN = "Turn the camera {} 90 degrees, then turn back to the original view."
CASES = [
    ("VENUE_hall_right", TURN.format("right"), "The hall, this morning. Watch the banners and the tables when the camera comes back.", 12),
    ("VENUE_stage_right", TURN.format("right"), "The stage. Watch the piano, the player and the chalk wall on return.", 12),
    ("VENUE_hall_left", TURN.format("left"), "Same hall, turning the other way.", 10),
    ("VENUE_hall_wait", "Hold the camera still for 18 seconds.", "A still camera: does the room keep its colours and its people?", 9),
    ("VENUE_stage_wait", "Hold the camera still for 18 seconds.", "A still camera on the stage.", 9),
]

if __name__ == "__main__":
    frames = m.card([("Holdfast, on this room", 60, True, m.FG),
                     ("", 14, False, m.DIM),
                     ("Two photographs taken here today. Same model, same seed, same camera path.", 28, False, m.FG),
                     ("Left: LingBot-World 2 unmodified. Right: with Holdfast's state writes.", 28, False, m.FG)], 5)
    used = 0
    for case, asked, look, sec in CASES:
        a, b = V + "unmodified/" + case + ".mp4", V + "holdfast/" + case + ".mp4"
        if not (os.path.exists(os.path.join(m.ROOT, a)) and os.path.exists(os.path.join(m.ROOT, b))):
            print("missing", case)
            continue
        frames += m.segment(asked, ("Unmodified model", a), ("Holdfast", b), look, sec)
        used += 1
    out = os.path.join(m.ROOT, "results", "venue", "venue_side_by_side.mp4")
    raw = out.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), m.FPS, (m.W, m.H))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-movflags", "+faststart", out], check=True)
    os.remove(raw)
    print("wrote", out, used, "segments", f"{len(frames) / m.FPS:.0f} s")
