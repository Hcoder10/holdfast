"""Any photograph, live: unmodified model against Holdfast, with the landmark visualizer.

  set -a; . ./.env; set +a
  python -I -X utf8 scripts/live_demo.py path/to/photo.jpg [--name room] [--action "w*6 -> s*6"]
                                         [--ask "Walk forward 6 steps, then walk back to the start."]

Crops the photograph to the model's frame, has Claude Haiku caption it, launches the two versions on two Modal
GPUs at once (about five minutes including the cold start), fetches both clips and the run log, and writes
results/venue/live_<name>.mp4: the two clips side by side over a plot of where the controller kept a checkpoint
of the model's state and where it wrote one back. Photographs and clips stay in results/venue/, which is not in
git. Action syntax is PlayWorld's: w/s/a/d steps, arrows for turns, e.g. "d*4 -> a*4".
"""
import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HOLDFAST = {"prompt": "caption+objective", "register": {"gain": 8, "when": "third person", "replace": True},
            "anchor_stats": 0.5, "bank": {"via": "key", "shadow": True, "tol": 18, "pos_tol": 1.2,
                                          "landmarks": {"tol": 12, "min_age": 5, "pos_tol": 0.5}}}
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def sh(cmd):
    return subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", errors="ignore")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--name", default=time.strftime("live%H%M%S"))
    ap.add_argument("--action", default="w*6 -> s*6")
    ap.add_argument("--ask", default="Walk forward 6 steps, then walk back to the starting point.")
    ap.add_argument("--caption", default=None)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.join(ROOT, "results", "venue"), exist_ok=True)
    im = Image.open(a.photo).convert("RGB")
    w, h = im.size
    th = int(w * 480 / 832)
    if th <= h:
        top = int((h - th) * 0.55)
        im = im.crop((0, top, w, top + th))
    else:
        tw = int(h * 832 / 480)
        im = im.crop(((w - tw) // 2, 0, (w - tw) // 2 + tw, h))
    im = im.resize((832, 480), Image.LANCZOS)
    rel = f"venue/live_{a.name}.jpg"
    im.save(os.path.join(ROOT, "results", rel), quality=95)
    caption = a.caption
    if not caption:
        import llm
        caption = llm.ask_json("You caption a photograph for a video world model. Reply with one JSON object and nothing else.",
                               'Describe this scene in 45-70 words: the place, the main objects and where they are (left, '
                               'right, centre, background), materials and lighting. No people\'s identities. JSON: {"caption": "..."}',
                               [np.asarray(im)], max_tokens=300)["caption"]
    print("caption:", caption)
    case = f"LIVE_{a.name}"
    custom = {case: {"image": rel, "image_caption": caption, "perspective": "first person", "prompt": a.ask, "action": a.action}}
    procs = []
    for tag, cfg in (("unmodified", {}), ("holdfast", HOLDFAST)):
        job = {"cases": [case], "custom": custom, "modes": ["open"], "out": f"live_{a.name}", "player": 0,
               "wait_chunks": 24, "variants": {tag: cfg}}
        procs.append(subprocess.Popen(["modal", "run", "modal_app/wh_modal.py::generate", "--job-json", json.dumps(job),
                                       "--shards", "1"], cwd=ROOT, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    print("generating both versions on Modal ...")
    for p in procs:
        p.wait()
    view = os.path.join(ROOT, "runs", "view", f"live_{a.name}")
    for tag in ("unmodified", "holdfast"):
        d = os.path.join(view, f"lingbot2_14b_open_{tag}")
        os.makedirs(d, exist_ok=True)
        r = sh(["modal", "volume", "get", "worldhold-runs", f"live_{a.name}/lingbot2_14b_open_{tag}/{case}/video.mp4",
                os.path.join(d, case + ".mp4"), "--force"])
        if not os.path.exists(os.path.join(d, case + ".mp4")):
            sys.exit(f"no clip for {tag}: {r.stdout[-300:]} {r.stderr[-300:]}")
    os.makedirs(os.path.join(ROOT, "runs", "lmdemo"), exist_ok=True)
    sh(["modal", "volume", "get", "worldhold-runs", f"live_{a.name}/lingbot2_14b_open_holdfast/{case}/result.json",
        os.path.join(ROOT, "runs", "lmdemo", case + ".json"), "--force"])
    import make_demo as m
    import make_landmark_demo as lm
    res = json.load(open(os.path.join(ROOT, "runs", "lmdemo", case + ".json"), encoding="utf8"))
    reads = (res.get("bank") or {}).get("reads") or []
    v = f"runs/view/live_{a.name}/lingbot2_14b_open_"
    frames = lm.segment(case, a.ask, ("Unmodified model", v + f"unmodified/{case}.mp4"),
                        ("Holdfast", v + f"holdfast/{case}.mp4"),
                        f"Generated live from a photograph. State written back {len(reads)} time(s).",
                        max(8, res["chunks"] * 0.75))
    out = os.path.join(ROOT, "results", "venue", f"live_{a.name}.mp4")
    raw = out.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), m.FPS, (m.W, m.H))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-movflags", "+faststart", out], check=True)
    os.remove(raw)
    print(f"wrote {out} in {time.time() - t0:.0f} s")
