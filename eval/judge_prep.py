"""Build exactly what PlayWorld's Rubric Verifier would send to its judge, on disk.

For each video this writes, under <out>/<run>/<task_id>/:
  score_prompt.txt   the scoring prompt (PlayWorld's own build_scoring_prompt)
  gate_prompt.txt    the instruction-following gate prompt, when the case has one
  sheets/            the 10 FPS primary and 0.5 FPS detail contact sheets
  evidence.json      the order in which the judge is shown the sheets

Run from the PlayWorld checkout so `metrics` imports:
  python judge_prep.py --data data --runs <dir of run folders> --out <dir> [--only run1,run2]
This is the only place in worldhold that reads the rubric; nothing here feeds
back into generation.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
from metrics.vqa.instruction_gate import build_gate_prompt, gate_kind  # noqa: E402
from metrics.vqa.rubric import build_scoring_prompt  # noqa: E402
from metrics.vqa.sampling import export_artifacts, generate_dual_sampling  # noqa: E402
from metrics.vqa.score import load_task  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--runs", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--only", default="")
a = ap.parse_args()


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


only = set(filter(None, a.only.split(",")))
for run in sorted(p for p in a.runs.iterdir() if p.is_dir()):
    if only and run.name not in only:
        continue
    for case in sorted(p for p in run.iterdir() if (p / "video.mp4").exists()):
        task = load_task(dataset_for(case.name), case.name)
        metric_task = dict(task)
        metric_task["task_id"] = task.get("source_task_id") or task["task_id"]
        item = {"model": run.name, "task": metric_task["task_id"],
                "group": task.get("source_category") or task.get("category"),
                "view": task.get("perspective"), "caption": task.get("image_caption"),
                "status": "generated"}
        dst = a.out / run.name / case.name
        dst.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            art = generate_dual_sampling(case / "video.mp4", Path(tmp))
            meta = art.metadata()
            meta["contact_sheet_bytes"] = 0
            meta["reference_image_bytes"] = 0
            n = len(art.primary.frames)
            (dst / "score_prompt.txt").write_text(
                build_scoring_prompt(item, metric_task, n, fps=10, sampling_metadata=meta),
                encoding="utf-8")
            if gate_kind(task):
                (dst / "gate_prompt.txt").write_text(
                    build_gate_prompt(item, metric_task, meta), encoding="utf-8")
            export_artifacts(art, dst / "sheets")
            order = []
            for stream in (art.primary, art.detail):
                sp = stream.spec
                order.append({"stream": sp.name, "fps": sp.fps,
                              "grid": [sp.grid_columns, sp.grid_rows],
                              "sheets": sorted(str(p.relative_to(dst)).replace("\\", "/")
                                               for p in (dst / "sheets" / sp.name).glob("*.jpg"))})
            (dst / "evidence.json").write_text(json.dumps(order, indent=1), encoding="utf-8")
        print(run.name, case.name, "gate" if gate_kind(task) else "no gate", flush=True)
