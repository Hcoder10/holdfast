"""PlayWorld's Agent Player for the LingBot runner ("Preset + Agent" control).

The benchmark does not replay a fixed action list: a vision-language agent (Claude
Haiku 4.5 in the paper) watches the generated frames and adapts the preset. This
module asks that agent the same question PlayWorld's released local adapter asks
(`Agent_player/SANA-WM/run_agent.py: decide_action`; the prompt wording below is
copied from it) through any OpenAI-compatible chat endpoint.

How it is used here: the runner executes each preset segment, then asks the agent
whether that segment is visually complete (NEXT), needs one more chunk of some action
(the action word), or the whole goal is done (END). Differences from the released
adapter, all to bound cost: the agent is consulted at segment boundaries and at the
end instead of after every chunk, and it is shown the initial frame plus samples from
the last three chunks instead of every chunk. The rubric is never shown to it.

Environment: PLAYER_API_URL (e.g. https://openrouter.ai/api/v1), PLAYER_API_KEY,
PLAYER_MODEL (e.g. anthropic/claude-haiku-4.5).
"""
import base64
import io
import os
import re

import requests
from PIL import Image

ACTIONS = ["END", "NEXT", "RIGHT", "LEFT", "DOWN", "WAIT", "UP", "W", "S", "A", "D"]
TO_KIND = {"RIGHT": "right", "LEFT": "left", "UP": "up", "DOWN": "down", "W": "forward",
           "S": "back", "A": "strafe_left", "D": "strafe_right", "WAIT": "wait"}
FROM_KIND = {v: k for k, v in TO_KIND.items()}


def _b64(frame, width=512):
    im = Image.fromarray(frame)
    im = im.resize((width, int(im.height * width / im.width)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Player:
    def __init__(self, url=None, key=None, model=None, log=None):
        self.url = (url or os.environ["PLAYER_API_URL"]).rstrip("/")
        self.key = key or os.environ["PLAYER_API_KEY"]
        self.model = model or os.environ.get("PLAYER_MODEL", "anthropic/claude-haiku-4.5")
        self.calls, self.trace, self.log = 0, [], log

    def decide(self, goal, first_frame, chunks, history, step, max_steps, allow_next=True):
        """chunks: list of [T,H,W,3] uint8 arrays, oldest first. Returns an action word."""
        images, labels = [_b64(first_frame)], ["initial frame at t=0.000s"]
        recent = chunks[-3:]
        base = len(chunks) - len(recent)
        for ci, ch in enumerate(recent):
            n = 8 if ci == len(recent) - 1 else 4
            for k in range(n):
                idx = min(len(ch) - 1, int(round((k + 1) * len(ch) / n)) - 1)
                images.append(_b64(ch[idx]))
                labels.append(f"chunk {base + ci + 1:02d} at local t={(idx + 1) / 16:.3f}s")
        choices = "W, S, A, D, LEFT, RIGHT, UP, DOWN, WAIT"
        prompt = (
            f"Our goal is: {goal}\n"
            f"We are now at step {step + 1} of at most {max_steps}. "
            f"We have completed {step} chunks and the actions already taken are: "
            f"{' -> '.join(history) if history else 'none'}.\n"
            f"The attached frames are chronological and have these timestamps in the same order: "
            f"{'; '.join(labels)}.\n"
            "Look at the actual visual progress in these frames and independently decide what action is needed now "
            "to accomplish the goal.\n"
            f"Choose one action from {choices}{', NEXT' if allow_next else ''}. "
            + ("Return NEXT only when the current action segment is visually complete and we should move to the next segment. "
               if allow_next else "")
            + "If the full goal in the prompt is already visually complete, return END. "
            "Return exactly one action word and nothing else.")
        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}} for b in images]
        content.append({"type": "text", "text": prompt})
        raw = ""
        for _ in range(3):
            try:
                r = requests.post(self.url + "/chat/completions",
                                  headers={"Authorization": "Bearer " + self.key},
                                  json={"model": self.model, "max_tokens": 16,
                                        "messages": [{"role": "user", "content": content}]}, timeout=120)
                raw = str(r.json()["choices"][0]["message"]["content"] or "").strip().upper()
                self.calls += 1
                for a in ACTIONS:
                    if (a != "NEXT" or allow_next) and re.search(rf"\b{a}\b", raw):
                        self.trace.append({"step": step, "answer": a})
                        return a
            except Exception as e:  # network or malformed reply: retry, then fall back
                raw = repr(e)[:120]
        self.trace.append({"step": step, "answer": "NEXT" if allow_next else "END", "fallback": raw})
        return "NEXT" if allow_next else "END"
