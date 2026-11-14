"""World-state director: keeps the scene's ongoing process moving, seen or unseen.

A video world model given one fixed caption tends to freeze or loop a process,
and to bring a thing back exactly as it left when the camera looks away. The
director holds a small explicit state for the scene, a timeline of what the
process should look like at each moment, and hands the generator the text for
the current moment. Time is wall-clock episode time, so the state advances while
the target is off screen as well.

A timeline is JSON:
{
  "process": "one sentence naming what is happening",
  "target_box": [x0, y0, x1, y1] | null,   # normalised box of the evolving thing in frame 0
  "stages": [{"from_s": 0, "state": "short state label", "prompt": "full scene description"}, ...]
}
It is written by a planning model that sees only the first frame, the caption,
the objective and the planned duration (see director_prompt.md). It never sees
the benchmark rubric.
"""
import json

import numpy as np


class Timeline:
    def __init__(self, spec):
        self.spec = spec
        self.stages = sorted(spec["stages"], key=lambda s: s["from_s"])
        assert self.stages and self.stages[0]["from_s"] <= 0
        self.target_box = spec.get("target_box")

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path, encoding="utf8")))

    def stage_at(self, t):
        cur = self.stages[0]
        for s in self.stages:
            if s["from_s"] <= t:
                cur = s
        return cur

    def prompt_at(self, t):
        return self.stage_at(t)["prompt"]

    def target_mask(self, height, width, pad=0.04):
        """Pixel mask of the evolving target in the first frame, or None."""
        if not self.target_box:
            return None
        x0, y0, x1, y1 = self.target_box
        m = np.zeros((height, width), np.uint8)
        m[int(max(0, y0 - pad) * height):int(min(1, y1 + pad) * height),
          int(max(0, x0 - pad) * width):int(min(1, x1 + pad) * width)] = 255
        return m
