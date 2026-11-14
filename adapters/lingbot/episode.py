"""Play one PlayWorld case on LingBot-World 2 from its preset action sequence.

This is the development runner: no player agent, the dataset's basic action
sequence is executed directly. Two control modes:

  open    each action token becomes a fixed camera motion (what a naive adapter does)
  closed  turns are measured with visual odometry and corrected until the commanded
          angle is reached; a turn that should end at the starting view is finally
          re-anchored against the first frame
  memory  closed, plus rotational scene memory: views already seen from the current
          standing point are fed back into the denoiser, so turning back redraws
          the same scene instead of inventing a new one

Only `prompt`, `image_caption`, `perspective`, `action` and the image are read
from a case. The rubric (`questions` and the policy fields) is never loaded here.
"""
import json
import os
import re

import numpy as np

from odometry import intrinsics_matrix, relative_rotation
from panorama import RotationMemory
from stepper import CHUNK, FPS, rel_pose

ALLOWED_CASE_FIELDS = ("task_id", "category", "prompt", "image_path", "image_caption",
                       "perspective", "action")

YAW_PER_TOKEN = 22.5     # "turn 90 degrees" is written as 4 tokens in the dataset
PITCH_PER_TOKEN = 10.0   # "tilt 30 degrees" is written as 3 tokens
MAX_YAW_PER_CHUNK = 30.0
MIN_INLIERS = 30        # feature matches needed to trust a rotation measurement
ASSUMED_GAIN = 0.85     # achieved / commanded rotation when it cannot be measured
ARROWS = {"←": "left", "→": "right", "↑": "up", "↓": "down"}
KEYS = {"w": "forward", "s": "back", "a": "strafe_left", "d": "strafe_right",
        "u": "up", "j": "down", "i": "up", "k": "down", "l": "right"}


def load_case(data_json, task_id):
    """Read one case, keeping only the fields a world model is allowed to see."""
    for c in json.load(open(data_json, encoding="utf8")):
        if c["task_id"] == task_id:
            return {k: c[k] for k in ALLOWED_CASE_FIELDS if k in c}
    raise KeyError(task_id)


def parse_actions(action):
    """'w*3 -> <left>*4 -> wait*2' -> [('forward', 3), ('left', 4), ('wait', 2)]"""
    out = []
    for part in action.split("->"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(.*?)(?:\*\s*(\d+))?\s*s?$", part)
        name, n = m.group(1).strip().lower(), int(m.group(2) or 1)
        if name in ARROWS:
            kind = ARROWS[name]
        elif name in KEYS:
            kind = KEYS[name]
        else:  # wait, waits, "wait and observe", interact(...)
            kind = "wait"
        out.append((kind, n))
    return out


def _translate(kind, speed):
    return {"forward": rel_pose(forward=speed), "back": rel_pose(forward=-speed),
            "strafe_left": rel_pose(right=-speed), "strafe_right": rel_pose(right=speed)}[kind]


class Episode:
    def __init__(self, stepper, mode="closed", speed=1.0, wait_chunks=None, log=print,
                 guide_from=1, guide_hard=True, timeline=None):
        self.timeline = timeline
        self.s, self.mode, self.speed, self.log = stepper, mode, speed, log
        self.wait_chunks, self.guide_from, self.guide_hard = wait_chunks, guide_from, guide_hard
        self.records, self.frames = [], []
        self.yaw = self.pitch = 0.0   # estimated orientation since the last translation
        self.level_pitch = 0.0        # pitch the camera is meant to hold
        self.mem = None

    # -- low level ---------------------------------------------------------
    def _step(self, label, yaw=0.0, pitch=0.0, move=None):
        """One chunk: rotate by (yaw, pitch) degrees or translate. Returns the
        rotation the frames actually show, as (yaw, pitch); None if unmeasurable."""
        guide, reloc = None, None
        direction = int(np.sign(yaw)) if yaw else int(np.sign(pitch)) * 2
        if move is not None:
            rel = [_translate(move, self.speed)] * CHUNK
        else:
            rel = [rel_pose(yaw_deg=yaw / CHUNK, pitch_deg=pitch / CHUNK)] * CHUNK
            if self.mode == "memory" and self.mem is not None and (yaw or pitch) and self.frames:
                usable = True
                # Re-localise: where memory covers the current view, measure the offset
                # between what memory says we should see and what is on screen, and
                # correct the orientation estimate. If they cannot be matched, the
                # estimate is not trustworthy enough to paste from, so skip guidance.
                seen, cover = self.mem.render(self.yaw, self.pitch, chunk=self.s.chunk_id,
                                              direction=direction)
                if cover.mean() > 0.25:
                    dy, dp, n = relative_rotation(seen, self.frames[-1], self.K)
                    if dy is not None and n >= MIN_INLIERS and abs(dy) < 40 and abs(dp) < 40:
                        self.yaw, self.pitch = self.yaw + dy, self.pitch + dp
                        reloc = (round(dy, 1), round(dp, 1), n)
                    else:
                        usable, reloc = False, ("failed", n)
                if usable:
                    imgs, masks = zip(*[self.mem.render(self.yaw + yaw * (i + 1) / 12,
                                                        self.pitch + pitch * (i + 1) / 12,
                                                        chunk=self.s.chunk_id, direction=direction)
                                        for i in range(12)])
                    if np.mean(masks) > 0.10:
                        guide = (np.stack(imgs), np.stack(masks))
        prompt = stage = None
        if self.timeline is not None:
            t = len(self.frames) / FPS
            stage = self.timeline.stage_at(t)
            # heading at the middle of the chunk about to be generated
            prompt = self.timeline.prompt_at(t, self.yaw + yaw / 2, self.pitch + pitch / 2)
        new = self.s.step(rel, prompt=prompt, guide=guide, guide_from=self.guide_from,
                          guide_hard=self.guide_hard)
        prev = self.frames[-1] if self.frames else new[0]
        my, mp, n = relative_rotation(prev, new[-1], self.K)
        cover = self.s.last_guide_coverage if guide is not None else 0.0
        first = not self.frames
        self.frames.extend(list(new))
        if move is not None:
            # a new standing point: rotational memory starts over from here
            self.yaw = self.pitch = self.level_pitch = 0.0
            if self.mem is not None:
                self.mem.keys = []
                self.mem.add(new[-1], 0.0, 0.0, force=True, chunk=self.s.chunk_id - 1)
            got = (None, None)
        else:
            # where the view is pinned by memory the commanded turn is what happened
            if cover > 0.5:
                got = (yaw, pitch)
            elif my is None or n < MIN_INLIERS:
                # nothing to measure against (water, sky): assume the usual shortfall
                got = (yaw * ASSUMED_GAIN, pitch * ASSUMED_GAIN)
            else:
                got = (my, mp)
            y0, p0 = self.yaw, self.pitch
            self.yaw, self.pitch = y0 + got[0], p0 + got[1]
            if self.mem is not None:
                if first:
                    self.mem.add(new[0], 0.0, 0.0, force=True, chunk=-99)
                if cover < 0.5:
                    for j in (len(new) // 2, len(new) - 1):
                        f = (j + 1) / len(new)
                        self.mem.add(new[j], y0 + got[0] * f, p0 + got[1] * f,
                                     chunk=self.s.chunk_id - 1, direction=direction)
        self.records.append({"chunk": self.s.chunk_id - 1, "label": label,
                             "measured_yaw": my, "measured_pitch": mp, "inliers": n,
                             "guide_coverage": round(cover, 2), "reloc": reloc,
                             "state": None if stage is None else stage.get("state"),
                             "away": bool(stage is not None and prompt != stage["prompt"]),
                             "est_yaw": round(self.yaw, 1), "est_pitch": round(self.pitch, 1)})
        return got

    def _turn(self, axis, target_deg):
        """Rotate by target_deg about yaw or pitch."""
        per_token = YAW_PER_TOKEN if axis == "yaw" else PITCH_PER_TOKEN
        nominal = max(1, round(abs(target_deg) / per_token))
        if self.mode == "open":
            per = target_deg / nominal
            for _ in range(nominal):
                self._step(f"{axis} {per:+.1f}", **{axis: per})
            return
        start = self.yaw if axis == "yaw" else self.pitch
        for _ in range(2 * nominal + 2):
            now = self.yaw if axis == "yaw" else self.pitch
            remaining = target_deg - (now - start)
            if abs(remaining) < 4.0:
                break
            cmd = float(np.clip(remaining, -MAX_YAW_PER_CHUNK, MAX_YAW_PER_CHUNK))
            kw = {axis: cmd}
            if axis == "yaw" and abs(self.pitch - self.level_pitch) > 3.0:
                # the model tends to tilt during a sweep; steer back to the horizon
                kw["pitch"] = float(np.clip(self.level_pitch - self.pitch, -12.0, 12.0))
            self._step(f"{axis} cmd {cmd:+.1f}" + (f" pitch {kw['pitch']:+.1f}" if "pitch" in kw else ""), **kw)
        if axis == "pitch":
            self.level_pitch = self.pitch

    def _return_check(self):
        yaw, pitch, n = relative_rotation(self.frames[0], self.frames[-1], self.K)
        rec = {"label": "return check", "offset_yaw": yaw, "offset_pitch": pitch, "inliers": n}
        self.records.append(rec)
        return rec

    def _reanchor(self):
        """Closed mode only: measure the offset to the first frame and remove it."""
        for _ in range(3):
            r = self._return_check()
            if r["offset_yaw"] is None or r["inliers"] < 25 or \
                    (abs(r["offset_yaw"]) < 3.0 and abs(r["offset_pitch"]) < 3.0):
                return
            cy = float(np.clip(-r["offset_yaw"], -MAX_YAW_PER_CHUNK, MAX_YAW_PER_CHUNK))
            cp = float(np.clip(-r["offset_pitch"], -MAX_YAW_PER_CHUNK, MAX_YAW_PER_CHUNK))
            self._step(f"reanchor yaw {cy:+.1f} pitch {cp:+.1f}", yaw=cy, pitch=cp)

    # -- episode -----------------------------------------------------------
    def run(self, case, image, seed=0):
        s = self.s
        first_prompt = self.timeline.prompt_at(0) if self.timeline else case["image_caption"]
        s.reset(image, first_prompt, seed=seed)
        fx, fy, cx, cy = s.intrinsics_480
        k = s.w / 832
        self.K = intrinsics_matrix(fx * k, fy * k, s.w / 2, s.h / 2)
        if self.mode == "memory":
            self.mem = RotationMemory(self.K, s.w, s.h)
            if self.timeline is not None:
                self.mem.exclude0 = self.timeline.target_mask(s.h, s.w)
        plan = parse_actions(case["action"])
        net_yaw = net_pitch = 0.0
        moved = False
        for kind, n in plan:
            if kind in ("left", "right"):
                tgt = (1 if kind == "right" else -1) * n * YAW_PER_TOKEN
                self._turn("yaw", tgt)
                net_yaw += tgt
            elif kind in ("up", "down"):
                tgt = (1 if kind == "up" else -1) * n * PITCH_PER_TOKEN
                self._turn("pitch", tgt)
                net_pitch += tgt
            elif kind == "wait":
                reps = self.wait_chunks if (self.wait_chunks and len(plan) == 1) else n
                for _ in range(reps):
                    self._step("wait")
            else:
                moved = True
                for _ in range(n):
                    self._step(kind, move=kind)
        returns_to_start = (not moved and abs(net_yaw) % 360 < 1e-6 and abs(net_pitch) < 1e-6
                            and any(k in ("left", "right", "up", "down") for k, _ in plan))
        final = None
        if returns_to_start:
            if self.mode == "closed":
                self._reanchor()
            final = self._return_check()
        return {"task_id": case["task_id"], "mode": self.mode, "plan": plan,
                "chunks": s.chunk_id, "seconds": round(len(self.frames) / FPS, 2),
                "returns_to_start": returns_to_start, "return_check": final,
                "records": self.records}
