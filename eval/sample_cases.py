"""Draw the stratified half-benchmark: a fixed random half of each PlayWorld dimension.

  python eval/sample_cases.py <PlayWorld data dir> [seed] > results/half_benchmark_cases.json

The draw depends only on the seed and the task ids, never on any result.
"""
import json
import random
import sys

data = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261114
rng = random.Random(seed)
out = {"seed": seed, "splits": {}, "cases": []}
for split in ("gc", "if", "insight", "outsight"):
    ids = sorted(c["task_id"] for c in json.load(open(f"{data}/{split}/data.json", encoding="utf8")))
    pick = sorted(rng.sample(ids, round(len(ids) / 2)))
    out["splits"][split] = pick
    out["cases"] += pick
print(json.dumps(out, indent=1))
