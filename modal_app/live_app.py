"""Holdfast live: a web page where anyone with the access code uploads a photo and watches the unmodified model
and Holdfast generate the same camera path side by side.

  set -a; . ./.env; set +a
  LIVE_CODE=<pick a code> modal deploy modal_app/live_app.py        # prints the page's URL
  modal app stop holdfast-live                                       # when done: stops the page and the GPUs

Two H200 containers (one per version) each keep the 14B model loaded and stay up for 15 minutes after the last
request, so the first run pays the model load (several minutes; use "Warm up" beforehand) and later runs take
about as long as the clip does to generate. Photos and clips go to the `worldhold-runs` volume under live/.
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
    .add_local_dir(os.path.join(ROOT, "tools"), remote_path="/wh/tools")
)
web_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("fastapi[standard]", "pillow", "python-multipart", "requests")
             .add_local_file(os.path.join(ROOT, "modal_app", "live_page.html"), "/wh/live_page.html"))

app = modal.App("holdfast-live")
weights = modal.Volume.from_name("worldhold-weights")
runs = modal.Volume.from_name("worldhold-runs")
W, R = "/weights", "/runs"
CKPT = f"{W}/lingbot-world-v2-14b-causal-fast"
DATA = f"{W}/playworld-data"
SECRET = modal.Secret.from_dict({k: os.environ.get(k, "") for k in ("ANTHROPIC_API_KEY", "LIVE_CODE")})

HOLDFAST = {"prompt": "caption+objective", "register": {"gain": 8, "when": "third person", "replace": True},
            "anchor_stats": 0.5, "bank": {"via": "key", "shadow": True, "tol": 18, "pos_tol": 1.2,
                                          "landmarks": {"tol": 12, "min_age": 5, "pos_tol": 0.5}}}
VARIANTS = {"unmodified": {}, "holdfast": HOLDFAST}
PATHS = {"walk": ("w*6 -> s*6", "Walk forward 6 steps, then walk back to the starting point."),
         "slide": ("d*5 -> a*5", "Move the camera right 5 steps, then move back to the starting point."),
         "inout": ("w*5 -> s*5 -> w*5", "Walk forward, walk back out, then walk forward again."),
         "long": ("w*6 -> s*6 -> w*6 -> s*6", "Walk forward 6 steps and back to the start, twice."),
         "turn": ("→*4 -> ←*4", "Turn the camera right 90 degrees, then turn back to the original view."),
         "still": ("wait*1", "wait and observe")}


@app.cls(image=image, gpu="H200", volumes={W: weights, R: runs}, timeout=3600, cpu=8, memory=98304,
         scaledown_window=900, max_containers=2)
class Live:
    @modal.enter()
    def load(self):
        import sys
        import time
        sys.path.insert(0, "/wh/adapters/lingbot")
        t0 = time.time()
        from stepper import LingBotStepper
        self.stepper = LingBotStepper("/opt/lingbot-world-v2", CKPT, task="i2v-A14B", local_attn_size=18, sink_size=6)
        print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    @modal.method()
    def ping(self):
        return "warm"

    @modal.method()
    def run(self, job: dict):
        import jobs
        runs.reload()
        jobs.run_job(self.stepper, "lingbot2_14b", dict(job, out=f"{R}/{job['out']}"), DATA,
                     log=lambda m: print("[live]", m, flush=True), after_each=runs.commit)
        runs.commit()
        return "done"


@app.function(image=web_image, volumes={R: runs}, secrets=[SECRET], timeout=900, scaledown_window=600)
@modal.concurrent(max_inputs=30)
@modal.asgi_app()
def web():
    import base64
    import io
    import time
    import uuid

    import requests
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from PIL import Image

    api = FastAPI()
    code_ok = lambda c: bool(os.environ.get("LIVE_CODE")) and c == os.environ["LIVE_CODE"]    # noqa: E731

    def caption(im):
        buf = io.BytesIO()
        im.resize((640, 369)).save(buf, "JPEG", quality=85)
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=60,
                              headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                              json={"model": "claude-haiku-4-5-20251001", "max_tokens": 300, "messages": [{"role": "user", "content": [
                                  {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                               "data": base64.b64encode(buf.getvalue()).decode()}},
                                  {"type": "text", "text": "Describe this scene in 45-70 words for a video world model: the place, the "
                                                           "main objects and where they are (left, right, centre, background), materials "
                                                           "and lighting. Do not identify people. Reply with the description only."}]}]})
            text = r.json()["content"][0]["text"]
            return " ".join(line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")).replace("**", "")
        except Exception:
            return "A real photograph of a place, seen from a first-person viewpoint, with the objects and lighting as shown."

    @api.get("/", response_class=HTMLResponse)
    def page():
        return open("/wh/live_page.html", encoding="utf8").read()

    @api.post("/api/warm")
    def warm(code: str = Form("")):
        if not code_ok(code):
            raise HTTPException(403, "wrong access code")
        for _ in range(2):
            Live().ping.spawn()
        return {"ok": True}

    @api.post("/api/run")
    async def run(image: UploadFile = File(...), path: str = Form("walk"), code: str = Form("")):
        if not code_ok(code):
            raise HTTPException(403, "wrong access code")
        if path not in PATHS:
            raise HTTPException(400, "unknown path")
        im = Image.open(io.BytesIO(await image.read())).convert("RGB")
        w, h = im.size
        th = int(w * 480 / 832)
        if th <= h:
            top = int((h - th) * 0.55)
            im = im.crop((0, top, w, top + th))
        else:
            tw = int(h * 832 / 480)
            im = im.crop(((w - tw) // 2, 0, (w - tw) // 2 + tw, h))
        im = im.resize((832, 480), Image.LANCZOS)
        rid = time.strftime("%H%M%S") + uuid.uuid4().hex[:6]
        d = f"{R}/live/{rid}"
        os.makedirs(d, exist_ok=True)
        im.save(f"{d}/input.jpg", quality=95)
        cap = caption(im)
        action, ask = PATHS[path]
        case = f"LIVE_{rid}"
        calls = {}
        meta = {"id": rid, "path": path, "action": action, "ask": ask, "caption": cap, "case": case, "t0": time.time()}
        json.dump(meta, open(f"{d}/meta.json", "w"))
        runs.commit()
        custom = {case: {"image_abs": f"{d}/input.jpg", "image_caption": cap, "perspective": "first person",
                         "prompt": ask, "action": action}}
        for tag, cfg in VARIANTS.items():
            job = {"cases": [case], "custom": custom, "modes": ["open"], "out": f"live/{rid}", "player": 0,
                   "wait_chunks": 24, "variants": {tag: cfg}}
            calls[tag] = Live().run.spawn(job).object_id
        meta["calls"] = calls
        json.dump(meta, open(f"{d}/meta.json", "w"))
        runs.commit()
        return {"id": rid, "ask": ask, "caption": cap}

    @api.get("/api/status/{rid}")
    def status(rid: str):
        runs.reload()
        d = f"{R}/live/{rid}"
        if not os.path.exists(f"{d}/meta.json"):
            raise HTTPException(404, "no such run")
        meta = json.load(open(f"{d}/meta.json"))
        out = {"id": rid, "ask": meta["ask"], "caption": meta["caption"], "seconds": round(time.time() - meta["t0"]), "arms": {}}
        for tag in VARIANTS:
            cd = f"{d}/lingbot2_14b_open_{tag}/{meta['case']}"
            arm = {"ready": os.path.exists(f"{cd}/video.mp4") and os.path.exists(f"{cd}/result.json")}
            if arm["ready"]:
                res = json.load(open(f"{cd}/result.json"))
                reads = (res.get("bank") or {}).get("reads") or []
                arm.update(chunks=res.get("chunks"), wall=res.get("wall_seconds"),
                           writes=[{"at": round((r["at_chunk"] + 1) * 0.75, 1), "what": r.get("via")} for r in reads])
            elif os.path.exists(f"{cd}/error.txt"):
                arm["error"] = open(f"{cd}/error.txt").read()[-400:]
            else:
                try:
                    modal.FunctionCall.from_id(meta["calls"][tag]).get(timeout=0)
                except TimeoutError:
                    pass
                except Exception as e:
                    arm["error"] = repr(e)[:300]
            out["arms"][tag] = arm
        return JSONResponse(out)

    @api.get("/api/video/{rid}/{tag}")
    def video(rid: str, tag: str):
        if tag not in VARIANTS or not rid.isalnum():
            raise HTTPException(400, "bad request")
        runs.reload()
        meta = json.load(open(f"{R}/live/{rid}/meta.json"))
        p = f"{R}/live/{rid}/lingbot2_14b_open_{tag}/{meta['case']}/video.mp4"
        if not os.path.exists(p):
            raise HTTPException(404, "not ready")
        return FileResponse(p, media_type="video/mp4")

    @api.get("/api/input/{rid}")
    def inp(rid: str):
        if not rid.isalnum():
            raise HTTPException(400, "bad request")
        runs.reload()
        return FileResponse(f"{R}/live/{rid}/input.jpg", media_type="image/jpeg")

    return api
