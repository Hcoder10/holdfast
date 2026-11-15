"""Shuffle and anonymise judge inputs so the judge cannot tell systems apart.

  python blind.py <judge inputs dir> <blinded dir> [seed]

Each <run>/<task_id> folder becomes <blinded>/vNN/<task_id>, the run name inside
the prompts is replaced by 'system_under_test', and blind_map.json records
vNN -> run for un-blinding after the answers are in.
"""
import json
import os
import random
import re
import shutil
import sys

src, dst = sys.argv[1], sys.argv[2]
random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
if os.path.exists(dst):
    shutil.rmtree(dst)
items = [(r, c) for r in sorted(os.listdir(src)) if os.path.isdir(os.path.join(src, r))
         for c in sorted(os.listdir(os.path.join(src, r)))]
random.shuffle(items)
mapping = {}
for i, (run, case) in enumerate(items):
    name = "v%02d" % (i + 1)
    d = os.path.join(dst, name, case)
    shutil.copytree(os.path.join(src, run, case), d)
    for f in ("score_prompt.txt", "gate_prompt.txt"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            t = open(p, encoding="utf8").read()
            # Only the model identifier is replaced. Replacing the run folder's name as a raw
            # string rewrote ordinary words when a folder was called "final" or "objective"
            # ("final score" became "system_under_test score"); see results/hillclimb_log.md.
            t = re.sub(r"lingbot2_[A-Za-z0-9_+]*", "system_under_test", t)
            open(p, "w", encoding="utf8", newline="\n").write(t)
    mapping[name] = run
json.dump(mapping, open(os.path.join(dst, "blind_map.json"), "w"), indent=1)
print(len(items), "videos blinded")
