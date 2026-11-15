"""Exa scout: when the system is stuck, search for what others have done about it, and check our own claims.

  python -I -X utf8 loop/scout.py methods     # failure modes -> Exa search -> proposals mapped to our write ports
  python -I -X utf8 loop/scout.py claims      # check the reference numbers we quote against the live source

`methods` is the self-improvement loop's search step. It reads how the final program still fails (the judge's
own summaries on the held-out half, grouped into failure modes with counts), asks Exa for recent work on each
mode, and has a small model turn each hit into a concrete proposal on one of the controller's write ports
(or say that it needs new code). Each proposal keeps its source URL and date. Proposals are written to past.dev
as beliefs (untested) by the `scout` identity; the experimenter's judged outcomes are what turn a belief into a
measurement, and the planner only acts on measurements (tools/planner_memory_demo.py).

`claims` fetches the pages our README cites and checks each quoted number against the text Exa returns.

Every Exa call is logged to results/exa_log.jsonl. Outputs: results/scout_methods.json, results/scout_claims.json
and results/scout_report.md.
"""
import collections
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import exa_search  # noqa: E402
import llm  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

# failure modes, how to recognise them in the judge's (Chinese or English) summaries, and what to search for
MODES = {
    "frozen": ("静止|冻结|没有移动|未移动|原地|无效的重复|没有任何.{0,6}(移动|进展|演变)|frozen|static|does not move",
               "training-free method to stop an autoregressive video diffusion world model from freezing: subject "
               "stays static or motion collapses during long rollout"),
    "replaced_on_return": ("消失|替换|取代|凭空|未能重现|disappear|replaced|vanish",
                           "method for object permanence and scene consistency when the camera returns to a "
                           "previously seen view in an interactive video world model, memory for revisits"),
    "hard_cut": ("硬切|硬重置|突变|瞬移|hard cut|reset|teleport",
                 "preventing abrupt scene resets or hard cuts in long autoregressive video generation with a "
                 "sliding attention window"),
    "trajectory_not_executed": ("没有执行|未能完成|未执行|did not follow|未能执行",
                                "improving camera control adherence in camera-conditioned video diffusion: model "
                                "ignores commanded camera translation or rotation"),
    "no_evolution_out_of_view": ("视野外|移出画面|out of view|out-of-sight|没有.{0,8}(积累|演变)",
                                 "world model state continues to evolve while off screen, out-of-view dynamics "
                                 "simulation in generative world models"),
}
PORTS = """The controller's existing write ports (a proposal is only testable today if it maps onto one of these):
- cross_attention_gain: attention gain on a text segment, optionally scheduled by camera heading
- latent_statistics: shift each chunk's latents toward reference statistics (re-anchoring)
- residual_direction: add or remove a direction in the residual stream at chosen blocks and sampling steps
- attention_cache: gain on, or contents of, the self-attention cache (the first frames and a rolling window);
  window length and number of permanent frames
- image_condition: pin a chosen frame of a chunk to a given latent (keyframe) through the model's conditioning input
- camera_modulation: extrapolate each block's camera scale and shift
- noise: correlate each chunk's starting noise with the previous chunk's; re-draw noise for a chunk
- state_fork: copy the whole generation state and run a second rollout from it (shadow world, best-of-N)"""
ASK = """You are helping improve an autoregressive, camera-controlled video world model WITHOUT training it.
Failure mode to fix: {mode} ({count} of {total} low-scoring cases on the held-out benchmark half).

Here are search results. For each one that describes a technique that could address this failure, give a proposal.
Reply with one JSON object: {{"proposals": [{{"url": the result's url, "technique": name of the technique in the
source (short), "idea": one sentence on what it does, "port": one of the port names below or "needs_new_code",
"setting": the concrete thing to try on that port (one sentence), "training_free": true/false}}]}}.
Skip results that are irrelevant or that require training. At most 3 proposals.

{ports}

Search results:
{hits}"""

# what we have already tested on each port, so a proposal can be marked as tried (with the judged or frame outcome)
TESTED = {
    "cross_attention_gain": "objective register (judged, in the final program, interaction +0.25) and caption gate (judged, no score gain)",
    "latent_statistics": "re-anchoring at 0.5 (judged, insight +0.27)",
    "residual_direction": "alive direction on sampling steps 0-1 (judged, lowers scores) and ablation (frames, colour cast)",
    "attention_cache": "gain 2-1000 on the first frames, a heading-indexed bank, windows of 30-42 frames (frames, no gain)",
    "image_condition": "keyframe arrival on the first frame (frames, clean, rarely triggers) and on a shadow-world frame (being judged)",
    "camera_modulation": "extrapolation w = 1.5-5 (frames, fixes ignored strafes, over-rotates other cases)",
    "noise": "correlation 0.5 between chunks (frames, no change); watchdog re-draw on resets and freezes (judged, no gain)",
    "state_fork": "shadow rollout with pixel pasting (judged, no gain); shadow world with keyframe write-back (being judged)",
}


def failure_digest():
    """Counts of each failure mode among final-arm held-out cases scoring 1.5 or less, from the judge's summaries."""
    sys.path.insert(0, str(ROOT / "third_party" / "PlayWorld"))
    held = json.loads((ROOT / "results" / "held_out.json").read_text())
    low = [c for c, v in held["per_case"]["final"].items() if sum(v) / len(v) <= 1.5]
    bm = json.loads((ROOT / "runs" / "jblind_held" / "blind_map.json").read_text())
    counts, examples = collections.Counter(), {}
    for v, arm in bm.items():
        if arm != "final":
            continue
        d = ROOT / "runs" / "gem_held_p1" / v
        for case in d.iterdir() if d.is_dir() else ():
            if case.name not in low or not (case / "score_result.json").exists():
                continue
            x = json.loads((case / "score_result.json").read_text(encoding="utf8"))
            x = x[0] if isinstance(x, list) else x
            t = str(x.get("summary_zh") or x.get("summary") or "")
            for mode, (pat, _) in MODES.items():
                if re.search(pat, t, re.I):
                    counts[mode] += 1
                    examples.setdefault(mode, f"{case.name}: {t[:140]}")
    return len(low), counts, examples


def search_recent(query, n=6):
    d = exa_search._post("/search", {"query": query, "type": "auto", "numResults": n,
                                     "startPublishedDate": "2025-01-01T00:00:00.000Z",
                                     "contents": {"highlights": {"maxCharacters": 700}}})
    hits = [{"title": r.get("title"), "url": r.get("url"), "published": (r.get("publishedDate") or "")[:10],
             "highlights": " ".join(r.get("highlights") or [])} for r in d.get("results", [])]
    exa_search._log({"op": "search", "query": query, "urls": [h["url"] for h in hits],
                     "cost": (d.get("costDollars") or {}).get("total"), "by": "scout"})
    return hits


def methods():
    total, counts, examples = failure_digest()
    out = {"low_scoring_cases": total, "modes": []}
    for mode, n in counts.most_common():
        hits = search_recent(MODES[mode][1])
        text = "\n".join(f"[{h['url']}] ({h['published']}) {h['title']}: {h['highlights'][:600]}" for h in hits)
        try:
            v = llm.ask_json("You map research findings onto a fixed set of interventions. Reply with JSON only.",
                             ASK.format(mode=mode.replace("_", " "), count=n, total=total, ports=PORTS, hits=text),
                             max_tokens=900)
        except Exception as e:
            v = {"proposals": [], "error": repr(e)[:120]}
        urls = {h["url"]: h for h in hits}
        props = []
        for p in (v.get("proposals") or [])[:3]:
            h = urls.get(p.get("url"))
            if not h:                     # a proposal must point at a result Exa actually returned
                continue
            port = p.get("port") if p.get("port") in TESTED else "needs_new_code"
            props.append({"technique": p.get("technique"), "idea": p.get("idea"), "port": port,
                          "setting": p.get("setting"), "source_url": h["url"], "source_title": h["title"],
                          "published": h["published"], "already_tested": TESTED.get(port)})
        out["modes"].append({"mode": mode, "cases": n, "example": examples.get(mode), "query": MODES[mode][1],
                             "results": [{k: h[k] for k in ("title", "url", "published")} for h in hits],
                             "proposals": props})
        print(f"{mode}: {n} cases, {len(hits)} results, {len(props)} proposals")
    (ROOT / "results" / "scout_methods.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")
    try:
        import mem
        if "scout" in mem.AGENTS:
            for m in out["modes"]:
                for p in m["proposals"]:
                    mem.write("scout", "hypothesis", f"Untested proposal for failure mode '{m['mode']}' ({m['cases']} cases): "
                              f"{p['technique']}: {p['idea']} Try on port {p['port']}: {p['setting']} "
                              f"Source: {p['source_url']} ({p['published']}).",
                              meta={"mode": m["mode"], "port": p["port"], "url": p["source_url"]})
    except Exception as e:
        print("memory write skipped:", repr(e)[:100])
    return out


PW = ["https://kxding.github.io/project/PlayWorld/", "https://arxiv.org/abs/2608.13552",
      "https://arxiv.org/html/2608.13552", "https://github.com/kxding/PlayWorld"]
CLAIMS = [
    ("PlayWorld reports Genie 3 at an overall score of 2.12", ["2.12"], PW),
    ("PlayWorld reports Matrix-Game-3.0 at an overall score of 1.14", ["1.14"], PW),
    ("PlayWorld has 171 cases across four dimensions", ["171"], PW),
    ("Sparks of Weight-Space Programming: a 0.016% weight edit took Qwen3.5-0.8B from 7.9% to 48.1% on MATH-500",
     ["0.016", "48.1"], ["https://rikiyat.github.io/blog/weight-space-programming/"]),
]


def claims():
    urls = sorted({u for c in CLAIMS for u in c[2]})
    pages = {}
    for u in urls:                      # one call per page, so one unreachable page does not sink the rest
        try:
            d = exa_search._post("/contents", {"urls": [u], "text": {"maxCharacters": 200000}})
            for r in d.get("results", []):
                pages[u] = r.get("text") or ""
            exa_search._log({"op": "contents", "query": "claim check", "urls": [u],
                             "cost": (d.get("costDollars") or {}).get("total"), "by": "scout"})
        except Exception as e:
            pages[u] = ""
            exa_search._log({"op": "contents", "query": "claim check", "urls": [u], "error": repr(e)[:120], "by": "scout"})
    out = []
    for claim, needles, cand in CLAIMS:
        best = {"url": cand[0], "verdict": "page not retrieved", "context": "", "page_chars": 0}
        for u in cand:
            text = pages.get(u, "")
            if not text:
                continue
            ok = all(re.search(r"(?<![0-9.])" + re.escape(n) + r"(?![0-9])", text) for n in needles)
            m = re.search(r".{0,100}(?<![0-9.])" + re.escape(needles[0]) + r"(?![0-9]).{0,100}", text, re.S)
            cur = {"url": u, "verdict": "supported" if ok else "not found on page",
                   "context": " ".join(m.group(0).split()) if m else "", "page_chars": len(text)}
            if ok or best["verdict"] == "page not retrieved":
                best = cur
            if ok:
                break
        out.append(dict(best, claim=claim, pages_tried=[u for u in cand if pages.get(u)]))
        print(best["verdict"], "|", claim, "|", best["url"])
    (ROOT / "results" / "scout_claims.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")
    return out


def report():
    L = ["# Exa scout report", "", "Generated by `loop/scout.py`. Every row links to the page Exa returned.", ""]
    p = ROOT / "results" / "scout_methods.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf8"))
        L += [f"## Methods: what to try next, from the web ({m['low_scoring_cases']} held-out cases still score 1.5 or less)", ""]
        for mode in m["modes"]:
            L += [f"### {mode['mode'].replace('_', ' ')} ({mode['cases']} cases)", "",
                  f"Judge, for example: {mode['example']}", "", f"Query: *{mode['query']}*", "",
                  "| Technique (source, date) | Idea | Maps to our port | What to try | Already tested here |", "|---|---|---|---|---|"]
            for q in mode["proposals"]:
                L.append(f"| [{q['technique']}]({q['source_url']}) ({q['published'] or 'n.d.'}) | {q['idea']} | "
                         f"`{q['port']}` | {q['setting']} | {q['already_tested'] or 'no: needs new code'} |")
            if not mode["proposals"]:
                L.append("| (no usable proposal) | | | | |")
            L += ["", "Other results returned: " + "; ".join(f"[{r['title']}]({r['url']})" for r in mode["results"][:6]), ""]
    p = ROOT / "results" / "scout_claims.json"
    if p.exists():
        L += ["## Claims we quote, checked against the live page", "", "| Claim | Verdict | Text found | Source |", "|---|---|---|---|"]
        for c in json.loads(p.read_text(encoding="utf8")):
            L.append(f"| {c['claim']} | {c['verdict']} | {c['context'][:160]} | {c['url']} |")
    (ROOT / "results" / "scout_report.md").write_text("\n".join(L) + "\n", encoding="utf8")
    print("wrote results/scout_report.md")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "methods"
    if cmd in ("methods", "all"):
        methods()
    if cmd in ("claims", "all"):
        claims()
    report()
