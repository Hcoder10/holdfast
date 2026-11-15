"""Run a job (a list of cases x modes) on a loaded stepper. Shared by the Modal app.

A job is a dict:
  {"cases": [...], "modes": ["open", "memory", "memory+anchor+memtok", ...], "out": dir,
   "tag": "", "timelines": dir or None, "wait_chunks": 40, "seed": 0}
Writes <out>/<model>_<mode><tag>/<task_id>/{video.mp4,result.json}; existing results are kept.
"""
import json
import os
import time

import imageio
import torch
from PIL import Image

import director
import episode
from stepper import FPS

SPLIT = {"GC": "gc", "IF": "if"}


def find_case(data, task_id):
    splits = [SPLIT[task_id[:2]]] if task_id[:2] in SPLIT else ["insight", "outsight"]
    for sp in splits:
        try:
            return sp, episode.load_case(os.path.join(data, sp, "data.json"), task_id)
        except KeyError:
            pass
    raise KeyError(task_id)


def image_for(data, split, case):
    rel = case["image_path"]
    for cand in (os.path.join(data, split, rel), os.path.join(data, rel),
                 os.path.join(data, split, "images", os.path.basename(rel))):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(rel)


def all_cases(data):
    out = []
    for sp in ("gc", "if", "insight", "outsight"):
        out += [c["task_id"] for c in json.load(open(os.path.join(data, sp, "data.json"), encoding="utf8"))]
    return out


def run_job(stepper, model, job, data, log=print, after_each=None):
    for task_id in job["cases"]:
        custom = (job.get("custom") or {}).get(task_id)
        if custom:        # a case that is not in the benchmark: our own image, caption, request and actions
            case = dict(custom, task_id=task_id)
            img = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", custom["image"]))
        else:
            split, case = find_case(data, task_id)
            img = Image.open(image_for(data, split, case))
        # "variants": {tag: cfg} runs several configurations in one job (one model load);
        # each lands in its own folder <model>_<mode>_<tag>.
        variants = job.get("variants") or {"": job.get("cfg")}
        for mode, (vtag, vcfg) in ((m, v) for m in job["modes"] for v in variants.items()):
            d = os.path.join(job["out"], f"{model}_{mode}{'_' + vtag if vtag else ''}{job.get('tag', '')}", task_id)
            if os.path.exists(os.path.join(d, "result.json")):
                continue
            os.makedirs(d, exist_ok=True)
            tl = None
            tdir = job.get("timelines")
            if tdir and os.path.exists(os.path.join(tdir, task_id + ".json")):
                tl = director.Timeline.load(os.path.join(tdir, task_id + ".json"))
            t0 = time.time()
            try:
                pl = None
                if job.get("player"):
                    import player
                    pl = player.Player()
                def make(c):
                    return episode.Episode(stepper, mode=mode, wait_chunks=job.get("wait_chunks", 40),
                                           log=lambda *x: None, timeline=tl, player=pl, cfg=c)
                frames = None
                if (vcfg or {}).get("repair"):
                    import repair
                    res, frames = repair.run(make, stepper, case, img, job.get("seed", 0), vcfg, log=log)
                else:
                    ep = make(vcfg)
                    res = ep.run(case, img, seed=job.get("seed", 0))
                    if getattr(ep, "ledger", None) and ep.ledger.get("objects"):
                        try:
                            import ledger
                            frames = stepper.video()
                            res["ledger_check"] = ledger.check(case, img, frames, ep.ledger["objects"])
                        except Exception as e:
                            res["ledger_check"] = {"error": repr(e)[:160]}
            except Exception as e:  # one bad case must not sink the shard
                import traceback
                log(f"{task_id} {mode} FAILED {e!r}")
                open(os.path.join(d, "error.txt"), "w").write(traceback.format_exc())
                continue
            imageio.mimsave(os.path.join(d, "video.mp4"), frames if frames is not None else stepper.video(),
                            fps=FPS, quality=8)
            res.update(directed=tl is not None, model=model, wall_seconds=round(time.time() - t0, 1),
                       prompt=case["prompt"], action=case["action"],
                       peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
            json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1, ensure_ascii=False)
            rc = res.get("return_check")
            log(f"{task_id} {mode}{job.get('tag', '')} {res['chunks']} chunks {res['wall_seconds']}s return: "
                + str(None if rc is None else {k: (round(v, 1) if isinstance(v, float) else v)
                                               for k, v in rc.items() if k != 'label'}))
            if after_each:
                after_each()
