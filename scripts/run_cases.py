"""Run PlayWorld cases on LingBot-World 2 from their preset actions.

  python run_cases.py --repo R --ckpt C [--assets A] --task i2v-A14B --data <PlayWorld data dir> \
      --cases GC010,OE014 --modes open,closed --out <dir> [--wait_chunks 40]

Writes <out>/<model>_<mode>/<task_id>/{video.mp4,result.json}. The model is loaded once.
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
from episode import Episode, load_case  # noqa: E402
from stepper import FPS, LingBotStepper  # noqa: E402

SPLIT = {"GC": "gc", "IF": "if"}

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--task", default="i2v-A14B")
ap.add_argument("--assets", default=None)
ap.add_argument("--data", required=True)
ap.add_argument("--cases", required=True)
ap.add_argument("--modes", default="open,closed")
ap.add_argument("--out", required=True)
ap.add_argument("--wait_chunks", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--guide_hard", type=int, default=1)
ap.add_argument("--guide_from", type=int, default=1)
ap.add_argument("--tag", default="")
a = ap.parse_args()


def find_case(task_id):
    splits = [SPLIT[task_id[:2]]] if task_id[:2] in SPLIT else ["insight", "outsight"]
    for sp in splits:
        try:
            return sp, load_case(os.path.join(a.data, sp, "data.json"), task_id)
        except KeyError:
            pass
    raise KeyError(task_id)


def image_for(split, case):
    rel = case["image_path"]
    for cand in (os.path.join(a.data, split, rel), os.path.join(a.data, split, "images", os.path.basename(rel)),
                 os.path.join(a.data, rel)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(rel)


stepper = LingBotStepper(a.repo, a.ckpt, task=a.task, assets_dir=a.assets)
model = "lingbot2_" + ("14b" if "14B" in a.task else "1p3b")
for task_id in a.cases.split(","):
    split, case = find_case(task_id)
    img = Image.open(image_for(split, case))
    for mode in a.modes.split(","):
        d = os.path.join(a.out, f"{model}_{mode}{a.tag}", task_id)
        os.makedirs(d, exist_ok=True)
        t0 = time.time()
        ep = Episode(stepper, mode=mode, wait_chunks=a.wait_chunks, log=lambda *x: None,
                     guide_from=a.guide_from, guide_hard=bool(a.guide_hard))
        res = ep.run(case, img, seed=a.seed)
        video = stepper.video()
        imageio.mimsave(os.path.join(d, "video.mp4"), video, fps=FPS, quality=8)
        res.update(model=model, wall_seconds=round(time.time() - t0, 1), prompt=case["prompt"],
                   action=case["action"], peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1, ensure_ascii=False)
        rc = res["return_check"]
        print(task_id, mode, f"{res['chunks']} chunks {res['seconds']}s video, {res['wall_seconds']}s wall",
              "return check:", None if rc is None else
              {k: (round(v, 1) if isinstance(v, float) else v) for k, v in rc.items() if k != "label"},
              flush=True)
