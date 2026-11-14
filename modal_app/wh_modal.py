"""worldhold on Modal: weights on a volume, generation fanned out over GPUs.

  modal run -d modal_app/wh_modal.py::download                 # once: weights + PlayWorld data
  modal run -d modal_app/wh_modal.py::generate --job-json '{"cases": "all", "modes": ["open"], "out": "full1"}' --shards 8
  modal run modal_app/wh_modal.py::judge_inputs --run full1    # PlayWorld judge prompts + contact sheets
  modal volume get worldhold-runs judge/full1 runs/            # then score locally (eval/openrouter_judge.py)

Each generate shard loads the 14B model once (about 12 minutes from a cold volume)
and runs its share of the cases. Results land on the `worldhold-runs` volume under
/<out>/<model>_<mode>/<task_id>/, and a shard skips cases that already have a result,
so an interrupted run can simply be started again.
"""
import json
import os

import modal

LINGBOT_COMMIT = "1895d30"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("opencv-python-headless>=4.9.0.80", "diffusers>=0.31.0", "transformers>=4.49.0,<=4.51.3",
                 "tokenizers>=0.20.3", "accelerate>=1.1.1", "tqdm", "imageio[ffmpeg]", "imageio-ffmpeg",
                 "easydict", "ftfy", "numpy>=1.23.5,<2", "scipy", "einops", "safetensors",
                 "huggingface_hub[cli]", "hf_transfer", "requests", "pillow", "google-genai>=1.0.0")
    .run_commands(
        "git clone https://github.com/Robbyant/lingbot-world-v2.git /opt/lingbot-world-v2"
        f" && cd /opt/lingbot-world-v2 && git checkout {LINGBOT_COMMIT}",
        "git clone https://github.com/kxding/PlayWorld.git /opt/PlayWorld",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DOWNLOAD_TIMEOUT": "60"})
    .add_local_dir(os.path.join(ROOT, "adapters"), remote_path="/wh/adapters")
    .add_local_dir(os.path.join(ROOT, "eval"), remote_path="/wh/eval")
    .add_local_dir(os.path.join(ROOT, "tools"), remote_path="/wh/tools")
    .add_local_dir(os.path.join(ROOT, "scripts"), remote_path="/wh/scripts")
    .add_local_dir(os.path.join(ROOT, "results"), remote_path="/wh/results")
)

app = modal.App("worldhold", image=image)
weights = modal.Volume.from_name("worldhold-weights", create_if_missing=True)
runs = modal.Volume.from_name("worldhold-runs", create_if_missing=True)
W, R = "/weights", "/runs"
CKPT = f"{W}/lingbot-world-v2-14b-causal-fast"
DATA = f"{W}/playworld-data"


@app.function(volumes={W: weights}, timeout=3 * 3600, cpu=8, memory=16384)
def download():
    from huggingface_hub import snapshot_download
    snapshot_download("robbyant/lingbot-world-v2-14b-causal-fast", local_dir=CKPT, max_workers=16)
    weights.commit()
    snapshot_download("jocelynd/playworld-bench", repo_type="dataset", local_dir=DATA, max_workers=8)
    weights.commit()
    n = sum(len(f) for _, _, f in os.walk(CKPT))
    print("files in checkpoint:", n, "data:", os.listdir(DATA))


@app.function(gpu="H200", volumes={W: weights, R: runs}, timeout=6 * 3600, cpu=8, memory=98304,
              secrets=[modal.Secret.from_dict({k: os.environ.get(k, "") for k in ("PAST_API_KEY",)})])
def generate_shard(job: dict, shard: int = 0):
    import sys
    import time
    sys.path.insert(0, "/wh/adapters/lingbot")
    t0 = time.time()
    from stepper import LingBotStepper
    import jobs
    stepper = LingBotStepper("/opt/lingbot-world-v2", CKPT, task="i2v-A14B")
    print(f"shard {shard}: model loaded in {time.time() - t0:.0f}s", flush=True)
    job = dict(job, out=f"{R}/{job['out']}")
    if job.get("timelines"):
        job["timelines"] = f"/wh/results/{job['timelines']}"
    jobs.run_job(stepper, "lingbot2_14b", job, DATA,
                 log=lambda m: print(f"[shard {shard}] {m}", flush=True), after_each=runs.commit)
    runs.commit()
    return shard


@app.function(volumes={W: weights, R: runs}, timeout=3600, cpu=8, memory=16384)
def judge_inputs_remote(run: str):
    import subprocess
    out = f"{R}/judge/{run}"
    subprocess.run(["python", "/wh/eval/judge_prep.py", "--data", DATA, "--runs", f"{R}/{run}",
                    "--out", out], cwd="/opt/PlayWorld", check=True)
    runs.commit()
    return out


@app.local_entrypoint()
def generate(job_json: str, shards: int = 4):
    job = json.loads(job_json)
    if job["cases"] == "all":
        job["cases"] = []
        for sp in ("gc", "if", "insight", "outsight"):
            p = os.path.join(ROOT, "third_party", "pw_data", sp, "data.json")
            job["cases"] += [c["task_id"] for c in json.load(open(p, encoding="utf8"))]
    cases = job["cases"]
    parts = [dict(job, cases=cases[i::shards]) for i in range(shards)]
    parts = [p for p in parts if p["cases"]]
    print(f"{len(cases)} cases over {len(parts)} shards")
    for s in generate_shard.starmap([(p, i) for i, p in enumerate(parts)]):
        print("shard done", s)


@app.local_entrypoint()
def judge_inputs(run: str):
    print(judge_inputs_remote.remote(run))
