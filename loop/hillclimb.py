"""Hill-climb the method's knobs against the benchmark's judge, unattended.

Runs on this machine; generation runs on Modal. Each trial:
  proposer     reads what has been tried (local log + shared memory on past.dev), searches
               the web through Exa for a technique or setting worth trying, and picks the
               next configuration from the fixed knob space below, with a stated hypothesis
  experimenter generates the development cases with that configuration and the real player
  judge        scores them with PlayWorld's prompts (development judge model, see --judge)
  critic       reads the per-case differences and the judge's own words for the biggest wins
               and losses, writes a lesson and a verdict on the hypothesis to past.dev
A change becomes the incumbent only if it beats the incumbent on the paired development
cases by --margin with more wins than losses. The loop stops at --until or when the
OpenRouter balance falls to --reserve (kept back for the confirmation run on unseen cases).

  python loop/hillclimb.py --until 06:15 --reserve 24 [--judge google/gemini-3.1-pro-preview]

Everything is logged to runs/hillclimb/trials.jsonl and results/hillclimb_log.md.
Development cases come from the half of the benchmark already used; the other half is
never touched here.
"""
import argparse
import datetime as dt
import json
import os
import random
import shutil
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "third_party" / "PlayWorld"))
import exa_search  # noqa: E402
import llm  # noqa: E402
import mem  # noqa: E402
from metrics.aggregate import OE_INSIGHT_TASKS, normalized_score  # noqa: E402

KNOBS = {
    "mode": ["open", "memory", "memory+shadow"],     # rotation memory; shadow = forked rollout keeps the start view running
    "brief": ["none", "dynamics", "llm"],            # step-0 motion sentence (dynamics = Exa-grounded, llm = no web)
    "guide_hard": [True, False],                     # remembered latents replace the final prediction, or only steer it
    "guide_from": [1, 2],                            # sampling step from which remembered latents are imposed
    "erode": [2, 5, 8],                              # free band (latent cells) between remembered and new content
    "feather": [0, 2, 4],                            # ramp width at that band
    "max_yaw": [22.5, 30.0],                         # degrees per chunk when turning
    "settle": [True, False],                         # finish and hold exactly on the start view when back at it
    "speed": [0.7, 1.0],                             # translation speed (model units)
}
DEFAULT = {"mode": "open", "brief": "none", "guide_hard": True, "guide_from": 1, "erode": 5,
           "feather": 2, "max_yaw": 30.0, "settle": True, "speed": 1.0}
PER_DIM = 6
OUT = ROOT / "runs" / "hillclimb"
LOG = OUT / "trials.jsonl"
MD = ROOT / "results" / "hillclimb_log.md"

ap = argparse.ArgumentParser()
ap.add_argument("--until", default="06:15")
ap.add_argument("--reserve", type=float, default=24.0)
ap.add_argument("--margin", type=float, default=0.15)
ap.add_argument("--judge", default="google/gemini-3.1-pro-preview")
ap.add_argument("--shards", type=int, default=6)
ap.add_argument("--max_trials", type=int, default=30)
# Manual mode: one trial with a configuration chosen by whoever is driving (no proposer or
# critic model). Used when the orchestrating session makes the decisions itself.
ap.add_argument("--cfg", default=None, help="JSON of knobs differing from the defaults")
ap.add_argument("--hypothesis", default="")
ap.add_argument("--id", type=int, default=None)
ap.add_argument("--accept", type=int, default=None, help="mark trial N as the incumbent and exit")
ap.add_argument("--resume", action="store_true", help="the trial's generation is already running; only wait and judge")
a = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)


def env():
    e = dict(os.environ, PYTHONIOENCODING="utf-8")
    for line in (ROOT / ".env").read_text(encoding="utf8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            e[k] = v.strip()
    e.update(PLAYER_API_URL="https://api.anthropic.com/v1", PLAYER_API_KEY=e["ANTHROPIC_API_KEY"],
             PLAYER_MODEL="claude-haiku-4-5-20251001")
    return e


ENV = env()
os.environ.update({k: ENV[k] for k in ("ANTHROPIC_API_KEY", "EXA_API_KEY", "PAST_API_KEY", "OPENROUTER_API_KEY")
                   if k in ENV})


def say(msg):
    line = f"[{dt.datetime.now():%H:%M}] {msg}"
    print(line, flush=True)
    with open(OUT / "loop.log", "a", encoding="utf8") as f:
        f.write(line + "\n")


def sh(cmd, timeout=3600):
    p = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=timeout,
                       encoding="utf8", errors="replace")
    return p.stdout + p.stderr


def credit():
    import requests
    d = requests.get("https://openrouter.ai/api/v1/credits", timeout=30,
                     headers={"Authorization": "Bearer " + ENV["OPENROUTER_API_KEY"]}).json()["data"]
    return d["total_credits"] - d["total_usage"]


def dim(task):
    return "GC" if task.startswith("GC") else "IF" if task.startswith("IF") else \
        "IE" if task in OE_INSIGHT_TASKS else "OE"


def load(p):
    d = json.loads(Path(p).read_text(encoding="utf8"))
    return d[0] if isinstance(d, list) else d


def read_scores(gem, blind_map):
    """{task: (score, summary text)} for a judged single-arm run."""
    out = {}
    for v in json.loads(Path(blind_map).read_text()):
        for case in (Path(gem) / v).iterdir():
            sp = case / "score_result.json"
            if not sp.exists():
                continue
            res = load(sp)
            gate = load(case / "gate_result.json").get("verdict") if (case / "gate_result.json").exists() else None
            raw = normalized_score(f"{v}/{case.name}", res)
            if raw is not None:
                out[case.name] = (1.0 if gate == "fail" else raw, str(res.get("summary_zh") or "")[:400], gate)
    return out


def judge_dir(inputs, tag):
    """Blind, judge with the development judge, return scores."""
    blind, gem = ROOT / "runs" / f"jblind_{tag}", ROOT / "runs" / f"gem_{tag}"
    sh([sys.executable, "-I", "eval/blind.py", str(inputs), str(blind), str(random.randint(1, 10**6))])
    for workers in ("8", "3"):   # second pass picks up anything that failed
        sh([sys.executable, "-I", "eval/openrouter_judge.py", "--inputs", str(blind), "--out", str(gem),
            "--workers", workers, "--model", a.judge], timeout=5400)
    return read_scores(gem, blind / "blind_map.json")


def dev_cases():
    d = json.loads((ROOT / "results" / "half_benchmark_cases.json").read_text())
    return d["splits"]["gc"][:PER_DIM] + d["splits"]["if"][:PER_DIM] + \
        d["splits"]["insight"][:PER_DIM] + d["splits"]["outsight"][:PER_DIM]


def run_trial(tid, cfg, cases):
    run = f"hc{tid:02d}"
    knobs = {k: v for k, v in cfg.items() if k != "mode"}
    job = {"cases": cases, "modes": [cfg["mode"]], "out": run, "player": 1, "wait_chunks": 40, "cfg": knobs}
    if a.resume:
        say(f"trial {tid}: generation already launched, waiting for it: {cfg}")
    else:
        say(f"trial {tid}: launching {cfg}")
        sh(["modal", "run", "-d", "modal_app/wh_modal.py::generate_bg", "--job-json", json.dumps(job),
            "--shards", str(a.shards)], timeout=900)
    folder = f"{run}/lingbot2_14b_{cfg['mode']}"
    t0 = time.time()
    while time.time() - t0 < 75 * 60:
        time.sleep(120)
        n = sh(["modal", "volume", "ls", "worldhold-runs", folder], timeout=120).count(run + "/")
        if n >= len(cases):
            time.sleep(300)   # folders appear before their results
            break
        if "worldhold" not in "\n".join(l for l in sh(["modal", "app", "list"], timeout=120).splitlines()
                                        if "stopped" not in l):
            break
    sh(["modal", "run", "modal_app/wh_modal.py::judge_inputs", "--run", run], timeout=1800)
    inputs = ROOT / "runs" / f"jin_{run}"
    inputs.mkdir(exist_ok=True)
    sh(["modal", "volume", "get", "worldhold-runs", f"judge/{folder}", str(inputs) + "/"], timeout=1800)
    return judge_dir(inputs, run)


def compare(scores, base):
    pairs = [(t, base[t][0], scores[t][0]) for t in scores if t in base]
    by = {}
    for t, b, o in pairs:
        by.setdefault(dim(t), []).append(o - b)
    dims = {d: round(st.mean(v), 2) for d, v in by.items()}
    return {"n": len(pairs), "mean_diff": round(st.mean(dims.values()), 3) if dims else 0.0, "by_dim": dims,
            "wins": sum(o > b for _, b, o in pairs), "losses": sum(o < b for _, b, o in pairs),
            "score": round(st.mean(st.mean(scores[t][0] for t in scores if dim(t) == d)
                                   for d in {dim(t) for t in scores}), 2) if scores else None,
            "pairs": sorted(pairs, key=lambda x: x[2] - x[1])}


def history_text(trials):
    return "\n".join(f"- trial {t['id']}: {json.dumps(t['cfg'])} -> vs baseline mean {t['vs_base']['mean_diff']:+.2f} "
                     f"{t['vs_base']['by_dim']} (wins {t['vs_base']['wins']}, losses {t['vs_base']['losses']}); "
                     f"{'ACCEPTED' if t.get('accepted') else 'rejected'}; lesson: {t.get('lesson', '')[:220]}"
                     for t in trials) or "(none yet)"


def propose(trials, incumbent):
    tried = [t["cfg"] for t in trials]
    lessons = []
    try:
        lessons = [h["text"][:300] for h in mem.recall("proposer", "lessons and verdicts about the world model and "
                                                       "which settings helped or hurt the benchmark score", 10)]
    except Exception:
        pass
    q = llm.ask_json(
        "You are a research agent improving a video world model's long-horizon consistency without training. "
        "Reply with one JSON object: {\"query\": a web search query for a technique or finding that could guide "
        "the next setting to try}.",
        f"Knobs: {json.dumps({k: v for k, v in KNOBS.items()})}\nTrials so far:\n{history_text(trials)}", max_tokens=200)
    hits = []
    try:
        hits = exa_search.search(q["query"], n=4)
    except Exception:
        pass
    out = llm.ask_json(
        "You choose the next experiment for a video world model wrapper. Dimensions scored 1-5: GC geometry "
        "consistency (camera turns and returns), IF interaction fidelity (a character walks into things), IE insight "
        "evolution (still camera, the scene's process should progress, not freeze), OE out-of-sight evolution (look "
        "away and back: the subject should have progressed). Knob meanings: mode 'memory' feeds remembered views back "
        "when the camera revisits a heading; 'memory+shadow' also keeps a forked copy of the model watching the start "
        "view so it evolves while unseen; brief adds one motion sentence to the caption at step 0 ('dynamics' grounds "
        "it with a web search); guide_* and erode/feather control how strongly and where memory is imposed; max_yaw is "
        "turn rate; settle holds the start view when back at it; speed scales walking. Change one or two knobs from "
        "the incumbent based on the evidence. Do not repeat a configuration already tried. Reply with one JSON "
        "object: {\"hypothesis\": one sentence, \"cfg\": {all knobs}, \"source_url\": url of a search result you used "
        "or null}.",
        f"Knobs and allowed values: {json.dumps(KNOBS)}\nIncumbent: {json.dumps(incumbent)}\n"
        f"Trials so far:\n{history_text(trials)}\nShared-memory lessons:\n" + "\n".join("- " + x for x in lessons)
        + f"\nSearch results for '{q.get('query')}':\n" + "\n".join(f"[{h['url']}] {h['highlights'][:400]}" for h in hits),
        max_tokens=500)
    cfg = dict(incumbent)
    for k, v in (out.get("cfg") or {}).items():
        if k in KNOBS and v in KNOBS[k]:
            cfg[k] = v
    if cfg in tried or cfg == incumbent:   # fall back to an untried single-knob change
        opts = [dict(incumbent, **{k: v}) for k in KNOBS for v in KNOBS[k]]
        opts = [o for o in opts if o not in tried and o != incumbent]
        if not opts:
            return None
        cfg = random.choice(opts)
        out["hypothesis"] = (out.get("hypothesis") or "") + " (replaced by an untried single-knob change)"
    return {"cfg": cfg, "hypothesis": out.get("hypothesis", ""), "query": q.get("query"),
            "source_url": out.get("source_url"), "search_urls": [h["url"] for h in hits]}


def criticise(trial, scores):
    worst, best = trial["vs_inc"]["pairs"][:3], trial["vs_inc"]["pairs"][-3:]
    notes = "\n".join(f"{t} ({dim(t)}): incumbent {b}, this {o}; judge said: {scores[t][1]}" for t, b, o in worst + best)
    out = llm.ask_json(
        "You are the critic in a research loop on a video world model. Given a hypothesis, the configuration tried, "
        "the score differences and the judge's own words (Chinese) for the biggest losses and wins, reply with one "
        "JSON object: {\"verdict\": \"confirmed\" | \"refuted\" | \"inconclusive\", \"lesson\": one or two sentences "
        "in English a later experiment should know, naming the setting and the observed effect}.",
        f"Hypothesis: {trial['hypothesis']}\nConfiguration: {json.dumps(trial['cfg'])}\nVersus incumbent: "
        f"{json.dumps({k: v for k, v in trial['vs_inc'].items() if k != 'pairs'})}\n{notes}", max_tokens=400)
    return out.get("verdict", "inconclusive"), out.get("lesson", "")


# ---------------------------------------------------------------------- main
cases = dev_cases()
say(f"development cases ({len(cases)}): {cases}")
say(f"judge {a.judge}; OpenRouter balance ${credit():.2f}; reserve ${a.reserve:.2f}; until {a.until}")

# Baseline (and its scores under the development judge) from the half-benchmark's baseline videos.
base_file = OUT / "baseline_scores.json"
if base_file.exists():
    base = {k: tuple(v) for k, v in json.loads(base_file.read_text()).items()}
else:
    inp = ROOT / "runs" / "jin_hcbase" / "lingbot2_14b_open"
    inp.mkdir(parents=True, exist_ok=True)
    for c in cases:
        src = ROOT / "runs" / "jin_half1" / "lingbot2_14b_open" / c
        if src.exists() and not (inp / c).exists():
            shutil.copytree(src, inp / c)
    base = judge_dir(inp.parent, "hcbase")
    base_file.write_text(json.dumps(base))
# Check the development judge against the benchmark's judge on the same baseline videos.
PRO = "google/gemini-3.1-pro-preview"
if a.judge != PRO and not (OUT / "judge_check.json").exists():
    names = json.loads((ROOT / "runs" / "jblind_half1" / "blind_map.json").read_text())
    pro = {}
    for v, run in names.items():
        if run == "lingbot2_14b_open":
            for case in (ROOT / "runs" / "gem_half1" / v).iterdir():
                if case.name in base and (case / "score_result.json").exists():
                    gate = load(case / "gate_result.json").get("verdict") if (case / "gate_result.json").exists() else None
                    raw = normalized_score(f"{v}/{case.name}", load(case / "score_result.json"))
                    pro[case.name] = (1.0 if gate == "fail" else raw, "", gate)
    common = [t for t in base if t in pro and pro[t][0] is not None]

    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = ranks([base[t][0] for t in common]), ranks([pro[t][0] for t in common])
    ma, mb = st.mean(ra), st.mean(rb)
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    rho = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / den if den else 0.0
    check = {"dev_judge": a.judge, "n": len(common), "spearman": round(rho, 2),
             "mean_dev": round(st.mean(base[t][0] for t in common), 2),
             "mean_pro": round(st.mean(pro[t][0] for t in common), 2)}
    say(f"development judge check against {PRO}: {check}")
    if rho < 0.5:
        say("agreement too low: switching the loop to the benchmark's judge")
        a.judge, base = PRO, pro
        base_file.write_text(json.dumps(base))
        check["switched_to"] = PRO
    (OUT / "judge_check.json").write_text(json.dumps(check))
say(f"baseline on development cases: {len(base)} scored, mean by dimension "
    f"{ {d: round(st.mean(base[t][0] for t in base if dim(t) == d), 2) for d in ('GC', 'IF', 'IE', 'OE')} }")

trials = [json.loads(l) for l in LOG.read_text(encoding="utf8").splitlines()] if LOG.exists() else []
incumbent, inc_scores = dict(DEFAULT), base
for t in trials:
    if t.get("accepted"):
        incumbent, inc_scores = t["cfg"], {k: tuple(v) for k, v in t["scores"].items()}
if not MD.exists():
    MD.write_text("# Hill-climb log\n\nEach row is one trial on the development cases (the half of the benchmark "
                  "already used). Differences are paired, against the baseline and against the incumbent at the "
                  "time.\n\n| # | time | configuration | hypothesis | vs baseline | by dimension | W/L | verdict | lesson |\n"
                  "|---|---|---|---|---|---|---|---|---|\n", encoding="utf8")

if a.accept is not None:
    for t in trials:
        t["accepted"] = t.get("accepted") or t["id"] == a.accept
    LOG.write_text("".join(json.dumps(t, ensure_ascii=False) + chr(10) for t in trials), encoding="utf8")
    say(f"trial {a.accept} marked as incumbent")
    sys.exit(0)
if a.cfg:
    cfg = dict(DEFAULT, **json.loads(a.cfg))
    tid = a.id or (max([t["id"] for t in trials], default=0) + 1)
    try:
        mem.write("proposer", "hypothesis", f"trial {tid} hypothesis: {a.hypothesis} Configuration: {json.dumps(cfg)}",
                  {"trial": tid, "by": "orchestrating session"})
    except Exception:
        pass
    scores = run_trial(tid, cfg, cases)
    trial = dict(id=tid, time=f"{dt.datetime.now():%H:%M}", cfg=cfg, hypothesis=a.hypothesis, by="orchestrating session",
                 scores=scores, vs_base=compare(scores, base), vs_inc=compare(scores, inc_scores), accepted=False)
    with open(LOG, "a", encoding="utf8") as f:
        f.write(json.dumps(trial, ensure_ascii=False) + chr(10))
    lines =[f"trial {tid} {json.dumps(cfg)}", f"hypothesis: {a.hypothesis}",
             f"scored {len(scores)}/{len(cases)}; overall (mean of dimension means) {trial['vs_base']['score']}",
             f"vs baseline: {json.dumps({k: v for k, v in trial['vs_base'].items() if k != 'pairs'})}",
             f"vs incumbent: {json.dumps({k: v for k, v in trial['vs_inc'].items() if k != 'pairs'})}", ""]
    for t, b, o in trial["vs_base"]["pairs"]:
        lines.append(f"{t} ({dim(t)}) baseline {b} -> {o} gate={scores[t][2]} | {scores[t][1][:260]}")
    (OUT / f"trial_{tid:02d}.txt").write_text(chr(10).join(lines), encoding="utf8")
    say(f"trial {tid}: overall {trial['vs_base']['score']}; vs baseline {trial['vs_base']['mean_diff']:+.2f} "
        f"{trial['vs_base']['by_dim']} W/L {trial['vs_base']['wins']}/{trial['vs_base']['losses']}; "
        f"balance ${credit():.2f}")
    sys.exit(0)

end = dt.datetime.combine(dt.date.today() + dt.timedelta(days=1 if dt.datetime.now().hour >= 12 else 0),
                          dt.time(*map(int, a.until.split(":"))))
while len(trials) < a.max_trials and dt.datetime.now() < end:
    bal = credit()
    if bal < a.reserve + 5:
        say(f"stopping: OpenRouter balance ${bal:.2f} is at the reserve")
        break
    prop = propose(trials, incumbent)
    if prop is None:
        say("stopping: nothing untried left")
        break
    tid = len(trials) + 1
    try:
        mem.write("proposer", "hypothesis", f"trial {tid} hypothesis: {prop['hypothesis']} Configuration: "
                  f"{json.dumps(prop['cfg'])}", {"trial": tid, "url": prop.get("source_url")})
    except Exception:
        pass
    scores = run_trial(tid, prop["cfg"], cases)
    if len(scores) < len(cases) * 0.7:
        say(f"trial {tid}: only {len(scores)} cases scored; recording and moving on")
    trial = dict(id=tid, time=f"{dt.datetime.now():%H:%M}", **prop, scores=scores,
                 vs_base=compare(scores, base), vs_inc=compare(scores, inc_scores))
    verdict, lesson = ("inconclusive", "") if not scores else criticise(trial, scores)
    ok = bool(scores) and trial["vs_inc"]["mean_diff"] >= a.margin and trial["vs_inc"]["wins"] > trial["vs_inc"]["losses"]
    trial.update(verdict=verdict, lesson=lesson, accepted=ok, balance_after=round(credit(), 2))
    try:
        mem.write("critic", "lesson", f"trial {tid} verdict {verdict}: {lesson} (configuration {json.dumps(prop['cfg'])}; "
                  f"versus baseline {trial['vs_base']['mean_diff']:+.2f}, by dimension {trial['vs_base']['by_dim']})",
                  {"trial": tid, "accepted": ok})
    except Exception:
        pass
    if ok:
        incumbent, inc_scores = prop["cfg"], scores
    trials.append(trial)
    with open(LOG, "a", encoding="utf8") as f:
        f.write(json.dumps(trial, ensure_ascii=False) + "\n")
    with open(MD, "a", encoding="utf8") as f:
        f.write(f"| {tid} | {trial['time']} | `{json.dumps(prop['cfg'])}` | {prop['hypothesis']} | "
                f"{trial['vs_base']['mean_diff']:+.2f} | {trial['vs_base']['by_dim']} | "
                f"{trial['vs_base']['wins']}/{trial['vs_base']['losses']} | {verdict}"
                f"{' **accepted**' if ok else ''} | {lesson} |\n")
    say(f"trial {tid}: vs baseline {trial['vs_base']['mean_diff']:+.2f} {trial['vs_base']['by_dim']}, vs incumbent "
        f"{trial['vs_inc']['mean_diff']:+.2f}; {verdict}; {'ACCEPTED' if ok else 'rejected'}; balance ${trial['balance_after']}")
say(f"finished with incumbent {incumbent}")
(OUT / "incumbent.json").write_text(json.dumps(incumbent))
