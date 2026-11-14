"""Keep the model loaded and run queued jobs (loading the 14B weights takes ~12 minutes).

  python serve_jobs.py --repo R --ckpt C [--assets A] --task i2v-A14B --data <PlayWorld data> --jobs <dir>

Drop a JSON file into <dir> to run it:
  {"cases": ["GC010", ...], "modes": ["open", "memory"], "out": "/path", "tag": "",
   "timelines": "/dir or null", "wait_chunks": 40, "seed": 0, "guide_hard": 1, "guide_from": 1}
Each job writes <name>.log beside it and is renamed to <name>.json.done when finished.
A file named STOP in <dir> ends the server.
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback

import imageio
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapters", "lingbot"))
import director  # noqa: E402
import episode  # noqa: E402
import odometry  # noqa: E402
import panorama  # noqa: E402
import stepper as stepper_mod  # noqa: E402
from stepper import FPS, LingBotStepper  # noqa: E402


def hot_reload(st):
    """Pick up edited adapter code without reloading the model weights."""
    import importlib
    import sdpa_shim
    for m in (sdpa_shim, odometry, panorama, director, stepper_mod, episode):
        importlib.reload(m)
    st.__class__ = stepper_mod.LingBotStepper
    st.used_sdpa_shim = sdpa_shim.install(force=True)

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--task", default="i2v-A14B")
ap.add_argument("--assets", default=None)
ap.add_argument("--data", required=True)
ap.add_argument("--jobs", required=True)
a = ap.parse_args()
os.makedirs(a.jobs, exist_ok=True)
SPLIT = {"GC": "gc", "IF": "if"}


def find_case(task_id):
    splits = [SPLIT[task_id[:2]]] if task_id[:2] in SPLIT else ["insight", "outsight"]
    for sp in splits:
        try:
            return sp, episode.load_case(os.path.join(a.data, sp, "data.json"), task_id)
        except KeyError:
            pass
    raise KeyError(task_id)


def image_for(split, case):
    rel = case["image_path"]
    for cand in (os.path.join(a.data, split, rel), os.path.join(a.data, rel),
                 os.path.join(a.data, split, "images", os.path.basename(rel))):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(rel)


def run_job(stepper, model, job, log):
    for task_id in job["cases"]:
        split, case = find_case(task_id)
        img = Image.open(image_for(split, case))
        for mode in job["modes"]:
            d = os.path.join(job["out"], f"{model}_{mode}{job.get('tag', '')}", task_id)
            if os.path.exists(os.path.join(d, "result.json")):
                continue
            os.makedirs(d, exist_ok=True)
            tl = None
            tdir = job.get("timelines")
            if tdir and os.path.exists(os.path.join(tdir, task_id + ".json")):
                tl = director.Timeline.load(os.path.join(tdir, task_id + ".json"))
            t0 = time.time()
            ep = episode.Episode(stepper, mode=mode, wait_chunks=job.get("wait_chunks", 40),
                         log=lambda *x: None, timeline=tl, anchor=job.get("anchor", 0),
                         guide_via=job.get("guide_via", "latent"),
                         guide_from=job.get("guide_from", 1), guide_hard=bool(job.get("guide_hard", 1)))
            res = ep.run(case, img, seed=job.get("seed", 0))
            imageio.mimsave(os.path.join(d, "video.mp4"), stepper.video(), fps=FPS, quality=8)
            res.update(directed=tl is not None, anchored=bool(ep.anchor), model=model, wall_seconds=round(time.time() - t0, 1),
                       prompt=case["prompt"], action=case["action"],
                       peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
            json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1, ensure_ascii=False)
            rc = res["return_check"]
            log(f"{task_id} {mode}{job.get('tag', '')} {res['chunks']} chunks {res['wall_seconds']}s "
                f"return check: " + str(None if rc is None else
                                        {k: (round(v, 1) if isinstance(v, float) else v)
                                         for k, v in rc.items() if k != 'label'}))


stepper = LingBotStepper(a.repo, a.ckpt, task=a.task, assets_dir=a.assets)
model = "lingbot2_" + ("14b" if "14B" in a.task else "1p3b")
open(os.path.join(a.jobs, "READY"), "w").write(model)
while not os.path.exists(os.path.join(a.jobs, "STOP")):
    pending = sorted(glob.glob(os.path.join(a.jobs, "*.json")))
    if not pending:
        time.sleep(2)
        continue
    path = pending[0]
    logf = open(path[:-5] + ".log", "a")

    def log(msg):
        logf.write(msg + "\n")
        logf.flush()
    try:
        hot_reload(stepper)
        run_job(stepper, model, json.load(open(path)), log)
        log("JOB DONE")
    except Exception:
        log("JOB FAILED\n" + traceback.format_exc())
    logf.close()
    os.rename(path, path + ".done")
