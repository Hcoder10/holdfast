"""Smoke test for the step-per-chunk driver: wait, turn right, turn back, walk.

Usage:
  python smoke_stepper.py --repo <lingbot repo> --ckpt <weights dir> --task i2v-1.3B \
      [--assets <14B dir>] --image <jpg> --prompt "..." --out <dir>
Writes <out>/smoke.mp4, <out>/smoke_stream.mp4 (frames as seen per step) and
<out>/smoke.json with per-step timing and peak memory.
"""
import argparse
import json
import os
import sys
import time

import imageio
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapters", "lingbot"))
from stepper import FPS, LingBotStepper, rel_pose  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--task", default="i2v-A14B")
ap.add_argument("--assets", default=None)
ap.add_argument("--image", required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--yaw_per_chunk", type=float, default=22.5)
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

t0 = time.time()
s = LingBotStepper(a.repo, a.ckpt, task=a.task, assets_dir=a.assets)
load_s = time.time() - t0

yaw = a.yaw_per_chunk / 3
plan = ([("wait", [rel_pose()] * 3)] * 2
        + [("right", [rel_pose(yaw_deg=yaw)] * 3)] * 4
        + [("left", [rel_pose(yaw_deg=-yaw)] * 3)] * 4
        + [("forward", [rel_pose(forward=a.speed)] * 3)] * 3)

t0 = time.time()
s.reset(Image.open(a.image), a.prompt, seed=a.seed)
reset_s = time.time() - t0
stream, steps = [], []
for name, rel in plan:
    torch.cuda.synchronize()
    t0 = time.time()
    frames = s.step(rel)
    torch.cuda.synchronize()
    steps.append({"action": name, "seconds": round(time.time() - t0, 2), "frames": len(frames)})
    stream.append(frames)
    print(steps[-1], flush=True)
t0 = time.time()
full = s.video()
decode_s = time.time() - t0
imageio.mimsave(os.path.join(a.out, "smoke.mp4"), full, fps=FPS, quality=8)
imageio.mimsave(os.path.join(a.out, "smoke_stream.mp4"), np.concatenate(stream), fps=FPS, quality=8)
info = {
    "task": a.task, "sdpa_shim": s.used_sdpa_shim, "size": [s.h, s.w],
    "load_seconds": round(load_s, 1), "reset_seconds": round(reset_s, 1),
    "full_decode_seconds": round(decode_s, 1),
    "mean_step_seconds": round(float(np.mean([x["seconds"] for x in steps[1:]])), 2),
    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
    "video_frames": int(full.shape[0]), "stream_frames": int(sum(len(f) for f in stream)),
    "steps": steps,
}
json.dump(info, open(os.path.join(a.out, "smoke.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != "steps"}), flush=True)
