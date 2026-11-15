"""Publish the rollouts of the 2.31 run as a Hugging Face dataset, straight from the Modal volume.

  hf auth login                      # once, or put HF_TOKEN in .env
  set -a; . ./.env; set +a
  modal run modal_app/hf_upload.py --repo <user>/holdfast-playworld-rollouts [--private] [--baseline]

Layout of the dataset:
  videos/holdfast/<task_id>.mp4        the Holdfast rollout (pre-registered program), all 171 PlayWorld cases
  logs/holdfast/<task_id>.json         the run log: actions taken, which state writes fired, Exa source if any
  videos/unmodified/<task_id>.mp4      with --baseline: the untouched model on the same cases
  metadata.jsonl                       one row per case: request, actions, dimension, split, judge scores of both arms
  README.md                            the dataset card
"""
import json
import os
import statistics as st

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = modal.App("holdfast-hf-upload")
runs = modal.Volume.from_name("worldhold-runs")
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub>=0.26", "hf_transfer").env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1"})
R = "/runs"
ARMS = {"holdfast": {"held_out": "final1/lingbot2_14b_open", "development": "final1dev/lingbot2_14b_open"},
        "unmodified": {"held_out": "base95/lingbot2_14b_open", "development": "half1/lingbot2_14b_open"}}

CARD = """---
license: other
license_name: see-card
task_categories:
- image-to-video
tags:
- world-model
- playworld
- lingbot-world
pretty_name: Holdfast rollouts on PlayWorld
---

# Holdfast rollouts on PlayWorld

The rollouts behind the 2.31 score reported in [Holdfast](https://github.com/Hcoder10/holdfast): LingBot-World 2
(14B, causal-fast) on all 171 cases of the [PlayWorld](https://github.com/kxding/PlayWorld) benchmark, with a
controller that writes into the model's hidden state while it runs. No weights were changed.

| All 171 cases | Geometry | Interaction | Insight | Out-of-sight | Overall |
|---|---|---|---|---|---|
| LingBot-World 2, untouched (our judging) | 2.35 | 2.05 | 2.30 | 1.68 | 2.10 |
| Holdfast | 2.45 | 2.30 | 2.57 | 1.94 | 2.31 |
| Genie 3 (published) | 2.74 | 2.40 | 1.51 | 1.81 | 2.12 |

Judge: Gemini 3.1 Pro with PlayWorld's own prompts, on blinded folders. The held-out half (95 cases) was judged
twice and averaged; the development half (76 cases) once.

## Files

- `videos/holdfast/<task_id>.mp4`: the rollout, 16 fps.
- `logs/holdfast/<task_id>.json`: the run log (actions taken, state writes that fired, the Exa question, answer
  and source where one was used).
- `metadata.jsonl`: one row per case with the request, camera actions, dimension, split and the judge's scores
  for Holdfast and for the untouched model.
{BASELINE}
## What produced these

Pre-registered state program: the request appended to the prompt; for third-person scenes the request in its own
cross-attention segment at 8 times the attention; each chunk's latent mean pulled halfway back toward the first
chunk's; on first-person scenes a caption gate and an Exa-sourced return sentence. Details and code in the
repository.

## Licence and sources

The first frame of every rollout is a PlayWorld benchmark image. PlayWorld's images come from several sources
with their own terms; see the PlayWorld repository and dataset for those. The generated continuations and the
logs are released for research use. Nothing here grants rights to the underlying source images.
"""


@app.function(image=image, volumes={R: runs}, timeout=3 * 3600, cpu=4, memory=8192)
def push(repo: str, token: str, meta: list, private: bool, baseline: bool):
    import shutil

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    stage = "/tmp/stage"
    failed = []

    def copy(src, dst):
        """Plain read and write, retried: a volume read can fail once and succeed after a reload."""
        for k in range(3):
            try:
                with open(src, "rb") as a, open(dst, "wb") as b:
                    while True:
                        buf = a.read(1 << 20)
                        if not buf:
                            break
                        b.write(buf)
                return True
            except OSError as e:
                err = repr(e)
                try:
                    runs.reload()
                except Exception:
                    pass
        failed.append((src, err))
        if os.path.exists(dst):
            os.remove(dst)
        return False
    arms = ["holdfast"] + (["unmodified"] if baseline else [])
    n = {a: 0 for a in arms}
    for row in meta:
        for arm in arms:
            src = f"{R}/{ARMS[arm][row['split']]}/{row['task_id']}"
            if not os.path.exists(f"{src}/video.mp4"):
                continue
            os.makedirs(f"{stage}/videos/{arm}", exist_ok=True)
            if not copy(f"{src}/video.mp4", f"{stage}/videos/{arm}/{row['task_id']}.mp4"):
                continue
            if arm == "holdfast" and os.path.exists(f"{src}/result.json"):
                os.makedirs(f"{stage}/logs/{arm}", exist_ok=True)
                copy(f"{src}/result.json", f"{stage}/logs/{arm}/{row['task_id']}.json")
            n[arm] += 1
    with open(f"{stage}/metadata.jsonl", "w", encoding="utf8") as f:
        for row in meta:
            row = dict(row, file_name=f"videos/holdfast/{row['task_id']}.mp4")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    base = ("- `videos/unmodified/<task_id>.mp4`: the untouched model on the same cases, same seed.\n" if baseline else "")
    open(f"{stage}/README.md", "w", encoding="utf8").write(CARD.replace("{BASELINE}", base))
    size = sum(os.path.getsize(os.path.join(d, x)) for d, _, fs in os.walk(stage) for x in fs)
    print("staged", n, f"{size / 1e9:.2f} GB", "failed:", failed, flush=True)
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=stage)
    else:
        for sub in ("README.md", "metadata.jsonl"):
            api.upload_file(path_or_fileobj=f"{stage}/{sub}", path_in_repo=sub, repo_id=repo, repo_type="dataset")
        for sub in ("logs", "videos"):
            api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=f"{stage}/{sub}", path_in_repo=sub,
                              commit_message=f"Add {sub}")
    return {"videos": n, "gb": round(size / 1e9, 2), "failed": failed, "url": f"https://huggingface.co/datasets/{repo}"}


@app.local_entrypoint()
def main(repo: str, private: bool = False, baseline: bool = False):
    token = os.environ.get("HF_TOKEN")
    if not token:
        from huggingface_hub import get_token
        token = get_token()
    assert token, "log in with `hf auth login` or set HF_TOKEN"
    import sys
    sys.path.insert(0, os.path.join(ROOT, "third_party", "PlayWorld"))
    from metrics.aggregate import OE_INSIGHT_TASKS
    cases = {}
    for sp in ("gc", "if", "insight", "outsight"):
        for c in json.load(open(os.path.join(ROOT, "third_party", "pw_data", sp, "data.json"), encoding="utf8")):
            cases[c["task_id"]] = c
    meta = []
    for split, name in (("held_out", "held_out.json"), ("development", "dev_half.json")):
        per = json.load(open(os.path.join(ROOT, "results", name)))["per_case"]
        for t in sorted(per["final"]):
            c = cases[t]
            dim = ("geometry" if t.startswith("GC") else "interaction" if t.startswith("IF")
                   else "insight" if t in OE_INSIGHT_TASKS else "out_of_sight")
            meta.append({"task_id": t, "dimension": dim, "split": split, "request": c.get("prompt"),
                         "camera_actions": c.get("action"), "perspective": c.get("perspective"),
                         "caption": c.get("image_caption"),
                         "judge_score_holdfast": round(st.mean(per["final"][t]), 2),
                         "judge_score_unmodified": round(st.mean(per["baseline"][t]), 2) if t in per["baseline"] else None,
                         "judge_passes": len(per["final"][t])})
    print(len(meta), "cases")
    print(push.remote(repo, token, meta, private, baseline))
