"""Web grounding through Exa.

Used by the cartographer agent to find what a real place actually looks like around
the camera, and how fast a real process actually runs, before the director writes
its timeline. Every call is appended to results/exa_log.jsonl with its sources, so
what the system learned from the web, and from where, can be checked.

CLI:
  python tools/exa_search.py search "what is south of the Taj Mahal mausoleum across the garden" [--n 4]
  python tools/exa_search.py answer "How fast does a typical escalator move, in metres per second?"

Needs EXA_API_KEY in the environment or in .env at the repo root.
"""
import argparse
import json
import os
import sys
import time

import requests

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
API = "https://api.exa.ai"
LOG = os.path.join(ROOT, "results", "exa_log.jsonl")


def _key():
    k = os.environ.get("EXA_API_KEY")
    if not k:
        env = os.path.join(ROOT, ".env")
        if os.path.exists(env):
            for line in open(env, encoding="utf8"):
                if line.startswith("EXA_API_KEY="):
                    k = line.split("=", 1)[1].strip()
    if not k:
        raise SystemExit("EXA_API_KEY is not set")
    return k


def _post(path, body):
    r = requests.post(API + path, headers={"x-api-key": _key(), "Content-Type": "application/json"},
                      json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def _log(entry):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    entry["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(LOG, "a", encoding="utf8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def search(query, n=4, chars=600):
    """Web search. Returns [{title, url, highlights}]."""
    d = _post("/search", {"query": query, "type": "auto", "numResults": n,
                          "contents": {"highlights": {"maxCharacters": chars}}})
    hits = [{"title": r.get("title"), "url": r.get("url"),
             "highlights": " ".join(r.get("highlights") or [])} for r in d.get("results", [])]
    _log({"op": "search", "query": query, "urls": [h["url"] for h in hits],
          "cost": (d.get("costDollars") or {}).get("total")})
    return hits


def answer(query):
    """A sourced short answer. Returns {answer, citations: [{title, url}]}."""
    d = _post("/answer", {"query": query})
    out = {"answer": d.get("answer"),
           "citations": [{"title": c.get("title"), "url": c.get("url")} for c in d.get("citations", [])]}
    _log({"op": "answer", "query": query, "answer": out["answer"],
          "urls": [c["url"] for c in out["citations"]], "cost": (d.get("costDollars") or {}).get("total")})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("search", "answer"))
    ap.add_argument("query")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    res = search(a.query, a.n) if a.cmd == "search" else answer(a.query)
    sys.stdout.buffer.write((json.dumps(res, ensure_ascii=False, indent=1) + "\n").encode("utf8"))
