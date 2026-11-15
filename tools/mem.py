"""Shared agent memory on past.dev.

Every agent in worldhold reads and writes through this one module (and CLI), so
there is a single place that shows what goes into the shared memory and who put it
there. past.dev holds *knowledge that outlives an episode*: how the world model
behaves, which prompt phrasings it obeys, sourced facts about real places and
processes. Per-frame state (keyframes, orientation) stays local in panorama.py.

CLI:
  python tools/mem.py write  --agent critic --kind lesson --text "..." [--meta '{"case":"OE014"}'] [--id my-id]
  python tools/mem.py recall --agent director --query "how to phrase progress so the model shows it" [--limit 8]
  python tools/mem.py log                      # local append-only record of every write and recall

Needs PAST_API_KEY in the environment or in .env at the repo root.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import requests

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
API = "https://api.past.dev/api/v1"
LOG = os.path.join(ROOT, "results", "memory_log.jsonl")
AGENTS = ("controller", "critic", "planner", "cartographer", "director", "composer", "proposer", "experimenter")

# Read boundaries. Every write carries one audience; a recall returns only what the reading
# identity's audiences reach.
#   wh-world     what the scene agents share: sourced facts and the state on return
#   wh-programs  what worked on the model: calibration, winning state programs, defects
#   wh-lab       the development loop's hypotheses and judged lessons
# The controller reads the world layer (it applies the sentence) and owns the programs layer;
# the scene agents never see programs or lab notes; the lab never reaches a running episode.
AUDIENCES = {
    "wh-world": ("cartographer", "director", "composer", "controller"),
    "wh-programs": ("controller", "critic", "planner"),
    "wh-lab": ("proposer", "experimenter"),
}
KIND_AUDIENCE = {"fact": "wh-world", "state": "wh-world", "view": "wh-world",
                 "calibration": "wh-programs", "program": "wh-programs", "defect": "wh-programs",
                 "hypothesis": "wh-lab", "lesson": "wh-lab", "result": "wh-lab",
                 # conclusions the lab publishes to the agents that build programs: a belief is an
                 # untested claim, a measurement is a judged result, a rule binds every later reader
                 "belief": "wh-programs", "measured": "wh-programs", "rule": "wh-programs"}
_AUD_READY = False


def setup_audiences():
    """Create or replace the three audiences (idempotent). Returns the API's records."""
    global _AUD_READY
    out = {}
    for slug, members in AUDIENCES.items():
        r = requests.put(f"{API}/audiences/{slug}", headers=_headers(), timeout=30,
                         json={"kind": "fixed", "name": slug, "identities": list(members)})
        r.raise_for_status()
        out[slug] = r.json()
    _AUD_READY = True
    return out


def _key():
    k = os.environ.get("PAST_API_KEY")
    if not k:
        env = os.path.join(ROOT, ".env")
        if os.path.exists(env):
            for line in open(env, encoding="utf8"):
                if line.startswith("PAST_API_KEY="):
                    k = line.split("=", 1)[1].strip()
    if not k:
        raise SystemExit("PAST_API_KEY is not set")
    return k


def _headers():
    return {"Authorization": "Bearer " + _key(), "Content-Type": "application/json"}


def _log(entry):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    entry["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(LOG, "a", encoding="utf8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write(agent, kind, text, meta=None, id=None, wait=False, timestamp=None):
    """Store one fact. `agent` is the author; `kind` selects the audience (KIND_AUDIENCE)."""
    assert agent in AGENTS, agent
    audience = KIND_AUDIENCE.get(kind, "wh-lab")
    if not _AUD_READY:
        try:
            setup_audiences()
        except Exception:
            audience = None      # audiences unavailable: fall back to project-visible, and say so in the log
    meta = dict(meta or {}, author=agent, kind=kind)
    # past.dev does not return the author on recall, so it is also in the text
    content = f"[{kind} from {agent}] {text}"
    body = {"id": id or f"{agent}-{kind}-{hashlib.sha1(content.encode()).hexdigest()[:12]}",
            "content": content, "timestamp": timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "label": f"{agent}/{kind}", "identity": agent, "metadata": meta}
    if audience:
        body["audience"] = audience
    r = requests.post(API + "/ingest", headers=_headers(), json=body, timeout=30)
    r.raise_for_status()
    out = r.json()
    _log({"op": "write", "agent": agent, "kind": kind, "text": text, "meta": meta,
          "id": body["id"], "status": out.get("status"), "audience": audience})
    if wait and out.get("ingestionId"):
        for _ in range(60):
            s = requests.get(API + "/ingest/" + out["ingestionId"], headers=_headers(), timeout=30).json()
            if s.get("status") != "processing" and s.get("status") != "queued":
                break
            time.sleep(2)
    return out


def recall(agent, query, limit=8, as_of=None):
    """Search the shared memory as `agent`. `as_of` (ISO time) reads the memory as it stood then.
    Returns [{text, author, kind, source_id, occurred_at, superseded_at, excerpts}]."""
    assert agent in AGENTS, agent
    body = {"query": query, "identity": agent, "limit": limit}
    if as_of:
        body["queryTimestamp"] = as_of
    r = requests.post(API + "/recall", headers=_headers(), json=body, timeout=60)
    r.raise_for_status()
    hits = []
    for res in r.json().get("results", []):
        src = (res.get("sources") or [{}])[0]
        meta = src.get("metadata") or {}
        label = src.get("label") or ""
        author = meta.get("author") or (label.split("/")[0] if "/" in label else None)
        hits.append({"text": res.get("content"), "author": author,
                     "kind": meta.get("kind") or (label.split("/")[1] if "/" in label else None),
                     "memory_kind": (res.get("artifact") or {}).get("kind"),
                     "occurred_at": res.get("occurredAt"),
                     "superseded_at": (res.get("artifact") or {}).get("supersededAt"),
                     "source_id": src.get("sourceId"), "excerpts": src.get("excerpts")})
    _log({"op": "recall", "agent": agent, "query": query, "n": len(hits),
          "authors": sorted({h["author"] for h in hits if h["author"]}),
          "source_ids": [h["source_id"] for h in hits]})
    return hits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--agent", required=True, choices=AGENTS)
    w.add_argument("--kind", required=True)
    w.add_argument("--text", required=True)
    w.add_argument("--meta", default="{}")
    w.add_argument("--id")
    w.add_argument("--wait", action="store_true")
    rc = sub.add_parser("recall")
    rc.add_argument("--agent", required=True, choices=AGENTS)
    rc.add_argument("--query", required=True)
    rc.add_argument("--limit", type=int, default=8)
    sub.add_parser("log")
    a = ap.parse_args()
    out = sys.stdout.buffer
    if a.cmd == "write":
        res = write(a.agent, a.kind, a.text, json.loads(a.meta), a.id, a.wait)
        out.write((json.dumps({"stored": res.get("sourceId"), "status": res.get("status")}) + "\n").encode())
    elif a.cmd == "recall":
        for h in recall(a.agent, a.query, a.limit):
            out.write((json.dumps(h, ensure_ascii=False) + "\n").encode("utf8"))
    else:
        out.write(open(LOG, encoding="utf8").read().encode("utf8") if os.path.exists(LOG) else b"")
