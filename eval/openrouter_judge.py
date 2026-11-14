"""Judge prepared case folders with Gemini 3.1 Pro through OpenRouter.

Input is what eval/judge_prep.py (then eval/blind.py) wrote: per case folder,
PlayWorld's own gate_prompt.txt / score_prompt.txt and its contact sheets. The
message mirrors PlayWorld's `build_evidence_parts`: prompt, then each evidence
stream announced and its sheets in order. Temperature 0.1 and JSON output, as in
PlayWorld's Gemini call. Only the transport differs (OpenRouter chat completions
instead of the google-genai client).

  python eval/openrouter_judge.py --inputs runs/judge_x_blind --out runs/gem_x [--workers 4]
      [--modes gate,score] [--only v01,v02] [--model google/gemini-3.1-pro-preview]

Writes <out>/<system>/<task>/{gate_result.json,score_result.json} for
judge_collect.py, appends per-call cost to <out>/cost.jsonl, and skips work
already done. Needs OPENROUTER_API_KEY in the environment or .env.
"""
import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
URL = "https://openrouter.ai/api/v1/chat/completions"

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--modes", default="gate,score")
ap.add_argument("--only", default="")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--model", default="google/gemini-3.1-pro-preview")
ap.add_argument("--max_tokens", type=int, default=12000)
a = ap.parse_args()


def key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text(encoding="utf8").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                k = line.split("=", 1)[1].strip()
    if not k:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    return k


KEY = key()


def parse_json(text):
    v = text.strip()
    if v.startswith("```"):
        v = v.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(v)


def content_for(case, mode):
    prompt = (case / f"{mode}_prompt.txt").read_text(encoding="utf8")
    manifest = json.loads((case / "sheets" / "sampling_manifest.json").read_text(encoding="utf8"))
    parts = [{"type": "text", "text": prompt}]
    for stream in ("primary", "detail"):
        m = manifest[stream]
        sheets = sorted((case / "sheets" / stream).glob("*.jpg"))
        parts.append({"type": "text", "text": (
            f"BEGIN {stream.upper()} EVIDENCE: {m['fps']:g} FPS, {m['cell_size'][0]}x{m['cell_size'][1]} cells, "
            f"{m['grid'][0]}x{m['grid'][1]} chronological row-major grid.")})
        for i, (sheet, rng) in enumerate(zip(sheets, m["sheet_ranges"]), 1):
            parts.append({"type": "text", "text": (
                f"{stream.upper()} SHEET {i}/{len(sheets)}; sample indices "
                f"{rng['first_sample_index']}-{rng['last_sample_index']}; continue chronologically.")})
            b64 = base64.b64encode(sheet.read_bytes()).decode()
            parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}})
        parts.append({"type": "text", "text": f"END {stream.upper()} EVIDENCE."})
    return parts


def one(job):
    system, case, mode = job
    dst = a.out / system / case.name
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / f"{mode}_result.json"
    if target.exists():
        return f"{system} {case.name} {mode} cached"
    body = {"model": a.model, "temperature": 0.1, "max_tokens": a.max_tokens,
            "response_format": {"type": "json_object"}, "usage": {"include": True},
            "messages": [{"role": "user", "content": content_for(case, mode)}]}
    err = ""
    for attempt in range(3):
        try:
            r = requests.post(URL, headers={"Authorization": "Bearer " + KEY}, json=body, timeout=600)
            d = r.json()
            if "choices" not in d:
                err = json.dumps(d)[:500]
                time.sleep(4)
                continue
            result = parse_json(d["choices"][0]["message"]["content"])
            target.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf8")
            u = d.get("usage", {})
            with open(a.out / "cost.jsonl", "a", encoding="utf8") as f:
                f.write(json.dumps({"system": system, "task": case.name, "mode": mode, "model": a.model,
                                    "prompt_tokens": u.get("prompt_tokens"),
                                    "completion_tokens": u.get("completion_tokens"),
                                    "cost": u.get("cost")}) + "\n")
            return f"{system} {case.name} {mode} ok cost={u.get('cost')}"
        except Exception as e:  # network or malformed JSON: retry
            err = repr(e)[:300]
            time.sleep(4)
    (dst / f"{mode}_error.txt").write_text(err, encoding="utf8")
    return f"{system} {case.name} {mode} FAILED {err[:120]}"


only = set(filter(None, a.only.split(",")))
jobs = []
for sysdir in sorted(p for p in a.inputs.iterdir() if p.is_dir()):
    if only and sysdir.name not in only:
        continue
    for case in sorted(p for p in sysdir.iterdir() if p.is_dir()):
        for mode in a.modes.split(","):
            if (case / f"{mode}_prompt.txt").exists():
                jobs.append((sysdir.name, case, mode))
a.out.mkdir(parents=True, exist_ok=True)
with ThreadPoolExecutor(a.workers) as ex:
    for msg in ex.map(one, jobs):
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()
