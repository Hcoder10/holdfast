"""Score run folders with PlayWorld's own Rubric Verifier (Gemini), unchanged.

Run from the PlayWorld checkout with GEMINI_API_KEY (and optionally GEMINI_MODEL) set:
  python gemini_judge.py --data data --runs <dir of run folders> --out <dir> [--only run1,run2] [--workers 4] [--passes 1]

For each <run>/<task_id>/video.mp4 this calls `python -m metrics.vqa.score` in gate
mode (when the case has a gate) and score mode, and writes
<out>/<run>/<task_id>/{gate_result.json,score_result.json} in the layout
judge_collect.py reads. With --passes N the outputs go to <out>_p1 .. <out>_pN.
Existing results are kept, so the command can be re-run after a failure.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, ".")
from metrics.vqa.instruction_gate import gate_kind  # noqa: E402
from metrics.vqa.score import load_task  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--runs", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--only", default="")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--passes", type=int, default=1)
a = ap.parse_args()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")


def dataset_for(task_id):
    if task_id.startswith("GC"):
        return a.data / "gc" / "data.json"
    if task_id.startswith("IF"):
        return a.data / "if" / "data.json"
    for split in ("insight", "outsight"):
        try:
            load_task(a.data / split / "data.json", task_id)
            return a.data / split / "data.json"
        except KeyError:
            pass
    raise KeyError(task_id)


def one(job):
    run, case, mode, dst = job
    target = dst / f"{mode}_result.json"
    if target.exists():
        return f"{run} {case.name} {mode} cached"
    raw = dst / f"{mode}_raw.json"
    cmd = [sys.executable, "-m", "metrics.vqa.score", "--mode", mode,
           "--dataset", str(dataset_for(case.name)), "--task-id", case.name,
           "--video", str(case / "video.mp4"), "--world-model", "system_under_test",
           "--model", MODEL, "--output", str(raw)]
    for attempt in range(3):
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0 and raw.exists():
            result = json.loads(raw.read_text(encoding="utf8")).get("score")
            if result is not None:
                target.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf8")
                return f"{run} {case.name} {mode} ok"
    (dst / f"{mode}_error.txt").write_text((p.stderr or p.stdout)[-3000:], encoding="utf8")
    return f"{run} {case.name} {mode} FAILED"


only = set(filter(None, a.only.split(",")))
for n in range(a.passes):
    out = a.out if a.passes == 1 else Path(f"{a.out}_p{n + 1}")
    jobs = []
    for run in sorted(p for p in a.runs.iterdir() if p.is_dir()):
        if only and run.name not in only:
            continue
        for case in sorted(p for p in run.iterdir() if (p / "video.mp4").exists()):
            dst = out / run.name / case.name
            dst.mkdir(parents=True, exist_ok=True)
            task = load_task(dataset_for(case.name), case.name)
            if gate_kind(task):
                jobs.append((run.name, case, "gate", dst))
            jobs.append((run.name, case, "score", dst))
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(one, jobs):
            print(msg, flush=True)
