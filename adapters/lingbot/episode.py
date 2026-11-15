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

from director import facing_start
from odometry import intrinsics_matrix, relative_rotation
import cv2
from panorama import RotationMemory, rot, view_footprint
from player import FROM_KIND, TO_KIND
from stepper import CHUNK, FPS, rel_pose

ALLOWED_CASE_FIELDS = ("task_id", "category", "prompt", "image_path", "image_caption",
                       "perspective", "action")

YAW_PER_TOKEN = 22.5     # "turn 90 degrees" is written as 4 tokens in the dataset
PITCH_PER_TOKEN = 10.0   # "tilt 30 degrees" is written as 3 tokens
MAX_YAW_PER_CHUNK = 30.0
MAX_STEPS = 40          # PlayWorld's interaction budget
NEUTRAL_AWAY = ("The surrounding area of the same place, continuing naturally in every direction, "
                "in the same style, lighting and weather.")
MAX_AGENT_EXTRA = 2     # agent-added chunks allowed after each preset segment
MIN_INLIERS = 30        # feature matches needed to trust a rotation measurement
ASSUMED_GAIN = 1.0      # achieved / commanded rotation when nothing is measured or remembered
ARROWS = {"←": "left", "→": "right", "↑": "up", "↓": "down"}
KEYS = {"w": "forward", "s": "back", "a": "strafe_left", "d": "strafe_right",
        "u": "up", "j": "down", "i": "up", "k": "down", "l": "right"}


def recall_entries(task_id):
    """Ask shared memory (past.dev) what lies in each direction around this scene.
    Entries are the cartographer's facts of the form
    '<CASE> heading <deg>: <visual description>'; one per heading is used. This runs
    inside the generation process, at the start of the episode."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
    import mem
    hits = mem.recall("composer", f"{task_id} heading: what is visible in each direction "
                                  f"around the starting viewpoint", limit=20)
    out = {}
    for h in hits:
        for text in [h.get("text") or ""] + list(h.get("excerpts") or []):
            m = re.search(rf"{task_id} heading (-?\d+)(?: pitch (-?\d+))?: (.+)", text)
            if m and int(m.group(1)) not in out:
                out[int(m.group(1))] = {"heading": int(m.group(1)), "pitch": int(m.group(2) or 0),
                                        "text": m.group(3).strip().rstrip("."),
                                        "source_id": h.get("source_id"), "author": h.get("author")}
    return [out[k] for k in sorted(out)]


def recall_gain(model, default):
    """Controller reads the turn gain other episodes measured for this model."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
        import mem
        vals = []
        for h in mem.recall("controller", f"{model} measured yaw turn gain achieved over commanded", 10):
            for text in [h.get("text") or ""] + list(h.get("excerpts") or []):
                m = re.search(rf"{re.escape(model)} yaw gain ([0-9.]+)", text)
                if m:
                    vals.append(float(m.group(1)))
        return float(np.median(vals)) if vals else default
    except Exception:
        return default


def write_gain(model, gain, n, task_id):
    try:
        import mem
        mem.write("controller", "calibration",
                  f"{model} yaw gain {gain:.2f} (achieved over commanded, median of {n} measured chunks, case {task_id})",
                  {"model": model, "gain": round(gain, 3), "n": n, "case": task_id})
    except Exception:
        pass


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
                 guide_from=1, guide_hard=True, timeline=None, anchor=False,
                 guide_via="latent", memory_entries=None, player=None, cfg=None):
        # cfg: knobs the hill-climb varies. Defaults reproduce the frozen method.
        self.cfg = dict(cfg or {})
        self.max_yaw = float(self.cfg.get("max_yaw", MAX_YAW_PER_CHUNK))
        self.do_settle = bool(self.cfg.get("settle", True))
        self.brief = None
        self.player = player          # PlayWorld's agent player (player.Player) or None
        self.model_name = "lingbot2_14b"
        self.history, self.chunk_frames = [], []
        # memory_entries: [{"heading": deg, "pitch": deg, "text": visual description,
        #                   "source_id": past.dev id, "url": web source}], retrieved from
        # shared memory for this scene; each becomes its own cross-attention entry
        self.memory_entries = list(memory_entries or [])
        self.timeline = timeline
        # anchor: bind the subject's text to the starting view's footprint through
        # cross-attention instead of swapping the whole prompt when facing away
        # Options can ride on the mode string ("memory+anchor+memtok", "memory+sink") so a
        # job file alone selects them: anchor = world-anchored cross-attention,
        # memtok = recall direction-tagged entries from shared memory, sink = memory
        # through the attention cache instead of latent guidance.
        flags = mode.split("+")
        mode = flags[0]
        self.flags = flags[1:]
        anchor = anchor or "anchor" in flags
        if "sink" in flags:
            guide_via = "sink"
        self.s, self.mode, self.speed, self.log = stepper, mode, speed, log
        self.guide_via = guide_via   # "latent": inpaint remembered latents; "sink": attention only
        self.anchor = bool(anchor and timeline is not None and timeline.spec.get("away_prompt"))
        self.wait_chunks, self.guide_from, self.guide_hard = wait_chunks, guide_from, guide_hard
        self.records, self.frames = [], []
        self.yaw = self.pitch = 0.0   # estimated orientation since the last translation
        self.last_turn = None         # the turn just completed, kept for rewinding
        self.turn_log = None          # frames of the turn in progress
        self.trust_command = False
        self.gains = []               # reliable achieved/commanded yaw ratios this episode
        self.been_away = False        # has the starting view left the frame since we stood here
        self.shadow = None            # forked rollout still watching the starting view
        self.shadow_chunks = 0
        self.caption = ""
        self.prior_gain = ASSUMED_GAIN
        self.level_pitch = 0.0        # pitch the camera is meant to hold
        self.mem = None

    # -- low level ---------------------------------------------------------
    def _step(self, label, yaw=0.0, pitch=0.0, move=None, hold=False, rewind=None):
        """One chunk: rotate by (yaw, pitch) degrees or translate. Returns the
        rotation the frames actually show, as (yaw, pitch); None if unmeasurable."""
        guide, reloc = None, None
        direction = int(np.sign(yaw)) if yaw else int(np.sign(pitch)) * 2
        if move is not None:
            rel = [_translate(move, self.speed)] * CHUNK
        else:
            rel = [rel_pose(yaw_deg=yaw / CHUNK, pitch_deg=pitch / CHUNK)] * CHUNK
            if rewind is not None:
                guide = rewind   # the frames already seen on the way out, played back
            elif self.mode == "memory" and self.mem is not None and (yaw or pitch or hold) and self.frames:
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
        if "shadow" in self.flags and self.mode == "memory" and move is None and self.frames:
            guide = self._shadow(guide, yaw, pitch)
        prompt = stage = None
        if self.timeline is not None:
            t = len(self.frames) / FPS
            if hasattr(self.timeline, "tick"):
                # online planning: the agents update state and look up the next
                # directions while this chunk renders; use whatever is ready now
                if self.frames:
                    self.timeline.tick(t, self.s.chunk_id, self.frames[-1],
                                       facing_start(self.yaw, self.pitch))
                self.memory_entries = self.timeline.live_entries()
            stage = self.timeline.stage_at(t)
            # heading at the middle of the chunk about to be generated
            prompt = self.timeline.prompt_at(t, self.yaw + yaw / 2, self.pitch + pitch / 2)
        region = None
        if self.anchor and move is None:
            entries = [stage["prompt"]] + [m["text"] for m in self.memory_entries]
            self.s.set_anchored(self.timeline.spec["away_prompt"], entries)
            prompt = None
            heads = [(0.0, 0.0)] + [(float(m["heading"]), float(m.get("pitch", 0.0)))
                                    for m in self.memory_entries]
            region = [np.stack([view_footprint(self.K, self.s.w, self.s.h,
                                               self.yaw + yaw * (j + 1) / CHUNK,
                                               self.pitch + pitch * (j + 1) / CHUNK, hy, hp)
                                for j in range(CHUNK)]) for hy, hp in heads]
        if self.timeline is None and "anchor" in self.flags and move is None:
            # no language model in the loop: the caption is bound to the starting
            # view's footprint, and a neutral description covers everything else
            self.s.set_anchored(NEUTRAL_AWAY, [self.caption])
            region = [np.stack([view_footprint(self.K, self.s.w, self.s.h,
                                               self.yaw + yaw * (j + 1) / CHUNK,
                                               self.pitch + pitch * (j + 1) / CHUNK)
                                for j in range(CHUNK)])]
        new = self.s.step(rel, prompt=prompt, guide=guide, guide_from=self.guide_from,
                          guide_hard=self.guide_hard, region=region, guide_via=self.guide_via)
        prev = self.frames[-1] if self.frames else new[0]
        my, mp, n = relative_rotation(prev, new[-1], self.K)
        self.chunk_frames.append(new)
        self.history.append(FROM_KIND.get(move) if move else "RIGHT" if yaw > 0 else "LEFT" if yaw < 0
                            else "UP" if pitch > 0 else "DOWN" if pitch < 0 else "WAIT")
        cover = self.s.last_guide_coverage if guide is not None else 0.0
        first = not self.frames
        self.frames.extend(list(new))
        if move is not None:
            # a new standing point: rotational memory starts over from here
            self.last_turn = None
            self.been_away = False
            self.shadow = None
            self.yaw = self.pitch = self.level_pitch = 0.0
            if self.mem is not None:
                self.mem.keys = []
                self.mem.add(new[-1], 0.0, 0.0, force=True, chunk=self.s.chunk_id - 1)
            got = (None, None)
        else:
            # where the view is pinned by memory the commanded turn is what happened
            if rewind is not None or self.trust_command or (cover > 0.5 and self.guide_via == "latent"):
                got = (yaw, pitch)
            elif my is None or n < MIN_INLIERS:
                # nothing to measure against (bare wall, water, sky): dead-reckon with
                # the gain measured so far in this episode, else the one in memory
                g = self._gain()
                got = (yaw * g, pitch * g)
            else:
                got = (my, mp)
                if abs(yaw) >= 15.0 and abs(pitch) < 1.0 and 0.3 < my / yaw < 2.5:
                    self.gains.append(my / yaw)
            y0, p0 = self.yaw, self.pitch
            self.yaw, self.pitch = y0 + got[0], p0 + got[1]
            if self.turn_log is not None and rewind is None:
                for j in range(len(new)):
                    f = (j + 1) / len(new)
                    self.turn_log.append((new[j], y0 + got[0] * f, p0 + got[1] * f))
            if self.mem is not None:
                if first:
                    self.mem.add(new[0], 0.0, 0.0, force=True, chunk=-99)
                if cover < 0.5 and rewind is None:
                    for j in (len(new) // 2, len(new) - 1):
                        f = (j + 1) / len(new)
                        self.mem.add(new[j], y0 + got[0] * f, p0 + got[1] * f,
                                     chunk=self.s.chunk_id - 1, direction=direction)
        if move is None and not facing_start(self.yaw, self.pitch):
            self.been_away = True
        self.records.append({"chunk": self.s.chunk_id - 1, "label": label,
                             "measured_yaw": my, "measured_pitch": mp, "inliers": n,
                             "guide_coverage": round(cover, 2), "reloc": reloc,
                             "rewind": rewind is not None,
                             "state": None if stage is None else stage.get("state"),
                             "away": bool(stage is not None and prompt != stage["prompt"]),
                             "est_yaw": round(self.yaw, 1), "est_pitch": round(self.pitch, 1)})
        return got

    def _rewind(self, axis, target_deg, nominal):
        """Turn back along the arc just swept by replaying the views seen on the way
        out, in reverse. A return retraces the outbound path, so the correct picture
        for each moment already exists: it is the outbound frame at the mirrored
        position. Feeding those frames to the denoiser makes the way back identical
        to the way out, with no warping and no dependence on angle estimates, and
        it ends exactly on the view the turn started from."""
        lt = self.last_turn
        frames = lt["frames"]
        share = min(1.0, abs(target_deg) / max(abs(lt["achieved"]), 1e-6))
        L = len(frames)
        per = target_deg / nominal
        y_end = lt["yaw1"] + (lt["yaw0"] - lt["yaw1"]) * share
        p_end = lt["pitch1"] + (lt["pitch0"] - lt["pitch1"]) * share
        for c in range(nominal):
            imgs, masks = [], []
            for j in range(12):
                prog = (c * 12 + j + 1) / (nominal * 12)
                img, fy, fp = frames[int(round((1.0 - prog * share) * (L - 1)))]
                m = np.ones(img.shape[:2], bool)
                if self.mem is not None and self.mem.exclude0 is not None:
                    m &= self.mem.excluded(fy, fp) == 0   # the evolving target is redrawn
                imgs.append(img)
                masks.append(m)
            self._step(f"{axis} rewind {per:+.1f}", rewind=(np.stack(imgs), np.stack(masks)),
                       **{axis: per})
        self.yaw, self.pitch = y_end, p_end
        keep = int(round((1.0 - share) * (L - 1)))
        if keep > 12:   # part of the arc is still ahead: a further turn back can keep replaying it
            done = lt["achieved"] * share
            lt.update(frames=frames[:keep + 1], achieved=lt["achieved"] - done,
                      target=lt["target"] - done, yaw1=y_end, pitch1=p_end)
        else:
            self.last_turn = None

    def _turn(self, axis, target_deg):
        """Rotate by target_deg about yaw or pitch."""
        per_token = YAW_PER_TOKEN if axis == "yaw" else PITCH_PER_TOKEN
        nominal = max(1, round(abs(target_deg) / per_token))
        if self.mode == "open":
            per = target_deg / nominal
            for _ in range(nominal):
                self._step(f"{axis} {per:+.1f}", **{axis: per})
            return
        lt = self.last_turn
        if (self.mode == "memory" and lt is not None and lt["axis"] == axis
                and np.sign(target_deg) == -np.sign(lt["achieved"])
                and abs(target_deg) <= abs(lt["target"]) + 1.0 and len(lt["frames"]) > 12):
            self._rewind(axis, target_deg, nominal)
            return
        self.turn_log = []
        y_start, p_start = self.yaw, self.pitch
        self._turn_out(axis, target_deg, nominal)
        self.last_turn = {"axis": axis, "target": target_deg, "frames": self.turn_log,
                          "achieved": (self.yaw - y_start) if axis == "yaw" else (self.pitch - p_start),
                          "yaw0": y_start, "pitch0": p_start, "yaw1": self.yaw, "pitch1": self.pitch}
        self.turn_log = None

    def _turn_out(self, axis, target_deg, nominal):
        if axis == "pitch":
            # Tilts run open loop at the preset rate. Sky and ground give odometry
            # little to measure, and believing an unmeasured tilt had happened made
            # the camera skip it altogether (GC041).
            per = target_deg / nominal
            self.trust_command = True
            for _ in range(nominal):
                self._step(f"pitch {per:+.1f}", pitch=per)
            self.trust_command = False
            self.level_pitch = self.pitch
            return
        start = self.yaw if axis == "yaw" else self.pitch
        for _ in range(2 * nominal + 2):
            now = self.yaw if axis == "yaw" else self.pitch
            remaining = target_deg - (now - start)
            if abs(remaining) < 4.0:
                break
            cmd = float(np.clip(remaining, -self.max_yaw, self.max_yaw))
            kw = {axis: cmd}
            if axis == "yaw" and abs(self.pitch - self.level_pitch) > 3.0:
                # the model tends to tilt during a sweep; steer back to the horizon
                kw["pitch"] = float(np.clip(self.level_pitch - self.pitch, -12.0, 12.0))
            self._step(f"{axis} cmd {cmd:+.1f}" + (f" pitch {kw['pitch']:+.1f}" if "pitch" in kw else ""), **kw)
        if axis == "pitch":
            self.level_pitch = self.pitch

    def _agent_chunk(self, word):
        """One chunk chosen by the agent player (an extension or a correction)."""
        kind = TO_KIND[word]
        if kind in ("left", "right", "up", "down"):
            axis = "yaw" if kind in ("left", "right") else "pitch"
            deg = (YAW_PER_TOKEN if axis == "yaw" else PITCH_PER_TOKEN) * (1 if kind in ("right", "up") else -1)
            lt = self.last_turn
            if (self.mode == "memory" and lt is not None and lt["axis"] == axis
                    and np.sign(deg) == -np.sign(lt["achieved"]) and len(lt["frames"]) > 12):
                self._rewind(axis, float(np.sign(deg)) * min(abs(deg), abs(lt["achieved"])), 1)
                self._maybe_settle()
                return
            same = lt is not None and lt["axis"] == axis and np.sign(deg) == np.sign(lt["achieved"])
            if same:   # the agent is extending the turn just made: keep it rewindable
                self.turn_log = lt["frames"]
            else:
                self.last_turn = None
            before = self.yaw if axis == "yaw" else self.pitch
            self._step(f"agent {word}", **{axis: deg})
            if same:
                lt["achieved"] += (self.yaw if axis == "yaw" else self.pitch) - before
                lt["target"] += deg
                lt["yaw1"], lt["pitch1"] = self.yaw, self.pitch
                self.turn_log = None
            self._maybe_settle()
        elif kind == "wait":
            self._step("agent WAIT")
        else:
            self._step(f"agent {word}", move=kind)

    def _consult(self, case, allow_next, next_kind=None):
        """Let the agent player extend or correct after a segment. Returns True if it
        says the whole goal is complete. If the agent asks for the action the next
        preset segment is about to perform anyway, that is a NEXT, not an extra chunk
        (otherwise the turn back would be made twice)."""
        for _ in range(MAX_AGENT_EXTRA):
            if self.s.chunk_id >= MAX_STEPS:
                return False
            word = self.player.decide(case["prompt"], self.frames[0], self.chunk_frames,
                                      self.history, self.s.chunk_id, MAX_STEPS, allow_next)
            if word == "END":
                return True
            if word == "NEXT" or TO_KIND.get(word) == next_kind:
                return False
            self._agent_chunk(word)
        return False

    def _shadow(self, guide, yaw, pitch):
        """Shadow rollout: the starting view keeps running while the camera is away.

        When the camera begins to turn away from the starting view, the generation
        state is forked. From then on, for every chunk the main rollout makes, the
        fork makes one too with the camera held still on the starting view, so
        whatever was happening there carries on by the model's own dynamics. Wherever
        the starting view's footprint is in the main camera's frame, the main rollout
        is guided by the fork's frame for that same moment, warped into view. The
        camera therefore comes back to the scene as it is now, not as it was, and
        no language model has to guess what happened."""
        s = self.s
        y = (self.yaw + 180.0) % 360.0 - 180.0
        if self.shadow is None:
            if not (abs(y) < 8.0 and abs(self.pitch) < 8.0 and (yaw or pitch)):
                return guide
            self.shadow = s.fork()
            self.shadow_chunks = 0
        main = s.export()
        s.load(self.shadow)
        region = None
        if s.regional:
            ones = np.ones((CHUNK, s.h, s.w), np.uint8)
            region = [ones] + [np.zeros_like(ones)] * (len(s.spans) - 1)
        sf = s.step([rel_pose()] * CHUNK, region=region)
        self.shadow = s.export()
        s.load(main)
        self.shadow_chunks += 1
        imgs = list(guide[0]) if guide is not None else [np.zeros_like(sf[0])] * 12
        masks = list(guide[1]) if guide is not None else [np.zeros(sf[0].shape[:2], bool)] * 12
        out_i, out_m = [], []
        Kinv = np.linalg.inv(self.K)
        for i in range(12):
            hy, hp = self.yaw + yaw * (i + 1) / 12, self.pitch + pitch * (i + 1) / 12
            foot = view_footprint(self.K, s.w, s.h, hy, hp) > 0
            if foot.any():
                Hm = self.K @ rot(hy, hp).T @ Kinv
                warped = cv2.warpPerspective(sf[i], Hm, (s.w, s.h), flags=cv2.INTER_LINEAR)
                im = imgs[i].copy()
                im[foot] = warped[foot]
                out_i.append(im)
                out_m.append(masks[i] | foot)
            else:
                out_i.append(imgs[i])
                out_m.append(masks[i])
        if np.mean(out_m) > 0.10:
            return np.stack(out_i), np.stack(out_m)
        return guide

    def _maybe_settle(self):
        """If the camera has been away and its measured heading is back at the start,
        finish exactly on the starting orientation and hold. Triggered by geometry
        only: the generator is never told that an episode is meant to return."""
        y = (self.yaw + 180.0) % 360.0 - 180.0
        if (self.mode == "memory" and self.do_settle and self.been_away
                and abs(y) < 8.0 and abs(self.pitch) < 8.0):
            self._settle()
            self.been_away = False
            self.shadow = None   # the main rollout is back on the scene; the fork is no longer needed

    def _gain(self):
        """Achieved / commanded rotation to assume when it cannot be measured."""
        if len(self.gains) >= 2:
            return float(np.median(self.gains))
        return self.prior_gain

    def _settle(self):
        """Memory mode: finish exactly on the starting orientation and hold it, so a
        return ends on the first view instead of a few degrees past it."""
        y = (self.yaw + 180.0) % 360.0 - 180.0
        if abs(y) > 0.5 or abs(self.pitch) > 0.5:
            self._step(f"settle yaw {-y:+.1f} pitch {-self.pitch:+.1f}", yaw=-y, pitch=-self.pitch)
        self._step("hold start view", hold=True)

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
        self.caption = case["image_caption"]
        for k in ("erode", "feather"):          # guidance blend widths, read by the stepper
            if k in self.cfg:
                setattr(s, k, int(self.cfg[k]))
        if "guide_hard" in self.cfg:
            self.guide_hard = bool(self.cfg["guide_hard"])
        if "guide_from" in self.cfg:
            self.guide_from = int(self.cfg["guide_from"])
        if "speed" in self.cfg:
            self.speed = float(self.cfg["speed"])
        if self.cfg.get("prompt", "caption") == "caption+objective":
            # PlayWorld's released local adapter gives the model the objective as its
            # text; here it is appended to the caption. A bare "wait and observe" adds
            # nothing and is left out.
            obj = (case.get("prompt") or "").strip()
            if obj and not obj.lower().startswith("wait and observe"):
                case = dict(case, image_caption=case["image_caption"].rstrip().rstrip(".") + ". " + obj)
                self.caption = case["image_caption"]
        if self.cfg.get("brief", "none") != "none":
            import brief
            self.brief = brief.make(case, image, use_web=self.cfg.get("brief") != "llm")
            if self.brief["brief"]:
                # one text for the whole episode: the caption plus a sentence of motion
                case = dict(case, image_caption=case["image_caption"].rstrip() + " " + self.brief["brief"])
                self.caption = case["image_caption"]
        if self.mode != "open" and "nomem" not in self.flags:
            self.prior_gain = recall_gain(self.model_name, ASSUMED_GAIN)
        if "online" in self.flags:
            import online
            self.timeline = online.OnlineDirector(
                case, image, use_web="noweb" not in self.flags, use_memory="nomem" not in self.flags)
            self.anchor = "anchor" in self.flags
        if "memtok" in self.flags and not self.memory_entries:
            self.memory_entries = recall_entries(case["task_id"])
        first_prompt = self.timeline.prompt_at(0) if self.timeline else case["image_caption"]
        s.reset(image, first_prompt, seed=seed)
        if self.anchor:
            s.set_anchored(self.timeline.spec["away_prompt"],
                           [first_prompt] + [m["text"] for m in self.memory_entries])
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
        for seg_i, (kind, n) in enumerate(plan):
            if kind in ("left", "right"):
                tgt = (1 if kind == "right" else -1) * n * YAW_PER_TOKEN
                if abs(tgt) > 360.0 and "360" in case.get("prompt", ""):
                    # the preset pads a full turn with extra tokens for slow models;
                    # the objective is one revolution, so stop there
                    tgt = 360.0 if tgt > 0 else -360.0
                self._turn("yaw", tgt)
                net_yaw += tgt
                self._maybe_settle()
            elif kind in ("up", "down"):
                tgt = (1 if kind == "up" else -1) * n * PITCH_PER_TOKEN
                self._turn("pitch", tgt)
                net_pitch += tgt
                self._maybe_settle()
            elif kind == "wait":
                reps = self.wait_chunks if (self.wait_chunks and len(plan) == 1) else n
                for _ in range(reps):
                    self._step("wait")
            else:
                moved = True
                for _ in range(n):
                    self._step(kind, move=kind)
            if self.player is not None and not (len(plan) == 1 and kind == "wait"):
                last = seg_i == len(plan) - 1
                if self._consult(case, allow_next=not last,
                                 next_kind=None if last else plan[seg_i + 1][0]):
                    break
        returns_to_start = (not moved and abs(net_yaw) % 360 < 1e-6 and abs(net_pitch) < 1e-6
                            and any(k in ("left", "right", "up", "down") for k, _ in plan))
        final = None
        if returns_to_start:
            if self.mode == "closed":
                self._reanchor()
            final = self._return_check()
        if self.mode != "open" and "nomem" not in self.flags and len(self.gains) >= 3:
            write_gain(self.model_name, float(np.median(self.gains)), len(self.gains), case["task_id"])
        return {"task_id": case["task_id"], "mode": self.mode, "plan": plan,
                "turn_gain": {"prior": self.prior_gain, "measured": [round(g, 2) for g in self.gains]},
                "online": self.timeline.report() if hasattr(self.timeline, "report") else None,
                "player": None if self.player is None else {"model": self.player.model,
                                                            "trace": self.player.trace},
                "actions": self.history, "shadow_chunks": self.shadow_chunks,
                "cfg": self.cfg, "brief": self.brief,
                "memory_entries": self.memory_entries, "flags": self.flags,
                "anchored": self.anchor, "guide_via": self.guide_via,
                "chunks": s.chunk_id, "seconds": round(len(self.frames) / FPS, 2),
                "returns_to_start": returns_to_start, "return_check": final,
                "records": self.records}
