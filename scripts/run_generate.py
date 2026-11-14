"""Run LingBot-World's stock generate.py, with the SDPA shim if flash-attn is absent.

Usage: python run_generate.py <lingbot repo dir> [generate.py args...]
Prints wall time and peak GPU memory at the end.
"""
import os
import runpy
import sys
import time

repo = os.path.abspath(sys.argv[1])
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo)
sys.path.insert(0, os.path.join(here, "..", "adapters", "lingbot"))
os.chdir(repo)

import torch  # noqa: E402
import wan  # noqa: E402,F401  (loads the modules the shim patches)
import sdpa_shim  # noqa: E402

print("sdpa shim installed:", sdpa_shim.install(), flush=True)
sys.argv = ["generate.py"] + sys.argv[2:]
t0 = time.time()
try:
    runpy.run_path(os.path.join(repo, "generate.py"), run_name="__main__")
finally:
    print(f"WALL_SECONDS {time.time() - t0:.1f}", flush=True)
    if torch.cuda.is_available():
        print(f"PEAK_VRAM_GB {torch.cuda.max_memory_allocated() / 2**30:.1f}", flush=True)
