"""Minimal Anthropic client for the in-loop agents (JSON out, images in)."""
import base64
import io
import json
import os
import re

import requests

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL = os.environ.get("AGENT_MODEL", "claude-haiku-4-5-20251001")


def _key():
    k = os.environ.get("ANTHROPIC_API_KEY")
    env = os.path.join(ROOT, ".env")
    if os.path.exists(env):
        for line in open(env, encoding="utf8"):
            if line.startswith("ANTHROPIC_API_KEY="):
                k = line.split("=", 1)[1].strip()
    if not k:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return k


def image_block(frame, width=640):
    """uint8 HxWx3 array or PIL image -> an image content block."""
    from PIL import Image
    im = frame if hasattr(frame, "save") else Image.fromarray(frame)
    im = im.convert("RGB")
    im = im.resize((width, int(im.height * width / im.width)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(buf.getvalue()).decode()}}


def ask_json(system, text, images=(), max_tokens=1200, model=None):
    """One call; the reply must be a JSON object. Returns the parsed object."""
    content = [image_block(i) for i in images] + [{"type": "text", "text": text}]
    last = ""
    for _ in range(3):
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=90,
                              headers={"x-api-key": _key(), "anthropic-version": "2023-06-01"},
                              json={"model": model or MODEL, "max_tokens": max_tokens, "system": system,
                                    "messages": [{"role": "user", "content": content}]})
            last = r.json()["content"][0]["text"]
            m = re.search(r"\{.*\}", last, re.S)
            return json.loads(m.group(0))
        except Exception as e:  # network, rate limit or malformed JSON: retry
            last = repr(e)[:200]
    raise RuntimeError("agent call failed: " + last)
