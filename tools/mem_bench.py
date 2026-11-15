"""past.dev against two common ways of sharing what a swarm has learned: accuracy, latency, cost.

  python -I -X utf8 tools/mem_bench.py        # writes results/pastdev_bench.json and .md

Task: the launch-time planner (Claude Haiku) must choose the state program for a scene: attention gain on the
request, the residual "alive" write, re-anchoring, and the still-camera scene action. The right answer for each
control is what the benchmark judge measured over the weekend (results/hillclimb_log.md): alive 0 (it lowers
scores), re-anchoring 0.5, request gain 8 only for a third-person request and 0 otherwise, scene action none.
Those findings were made by other agents (experimenter, critic); the planner only knows them through memory.

Three ways of giving the planner that knowledge:

  none      no shared memory: the planner has its own menu, whose descriptions are the swarm's untested beliefs
  log       the usual approach: every note any agent ever wrote (results/memory_log.jsonl) pasted into the prompt
  past.dev  recall from the shared layer as the planner identity: a handful of notes, each with its author, kind,
            time and whether a later note superseded it

Measured per decision: controls set to the measured-best value (accuracy), wall-clock seconds including the
recall (latency), and prompt size in characters with the Haiku input cost that implies (cost).
"""
import json
import os
import sys
import time

import cv2
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "adapters", "lingbot"))
import llm  # noqa: E402
import planner  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
V = os.path.join(ROOT, "runs", "view")
CLIPS = {"OE013": "sweep27/lingbot2_14b_open_anchor", "OE066": "sweep27/lingbot2_14b_open_anchor",
         "OE002": "sweep27/lingbot2_14b_open_anchor", "OE005": "sweep27/lingbot2_14b_open_anchor",
         "IF006": "sweep2/lingbot2_14b_open_cap", "IF009": "sweep2/lingbot2_14b_open_obj",
         "IF011": "sweep18/lingbot2_14b_open_base", "GC002": "sweep4/lingbot2_14b_open_none",
         "GC015": "sweep22/lingbot2_14b_open_base", "GC010": "sweep22/lingbot2_14b_open_base",
         "GC025": "sweep31/lingbot2_14b_open_shadow", "OE043": "sweep31/lingbot2_14b_open_shadow"}
cases = {}
for sp in ("gc", "if", "insight", "outsight"):
    for c in json.load(open(os.path.join(ROOT, "third_party", "pw_data", sp, "data.json"), encoding="utf8")):
        cases[c["task_id"]] = c


def frame(t):
    ok, f = cv2.VideoCapture(os.path.join(V, CLIPS[t], t + ".mp4")).read()
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


def truth(c):
    third = (c.get("perspective") or "").strip().lower() == "third person"
    wait = c["prompt"].lower().startswith("wait and observe")
    return {"action_register": 8 if (third and not wait) else 0, "alive": 0, "anchor": 0.5, "scene_action": "none"}


sizes = []
_ask = llm.ask_json


def counted(system, text, images=(), max_tokens=1200, model=None):
    sizes.append(len(system) + len(text))
    return _ask(system, text, images, max_tokens=max_tokens, model=model)


llm.ask_json = counted
LOG = [json.loads(line) for line in open(os.path.join(ROOT, "results", "memory_log.jsonl"), encoding="utf8")]
NOTES = [f"({e.get('agent')}, {e.get('kind')}) {e.get('text')}" for e in LOG if e.get("op") == "write" and e.get("text")]


def decide_log(case, image):
    """The planner's prompt with the neutral menu and every note ever written, unfiltered and unranked."""
    text = (f"Caption: {case.get('image_caption', '')}\nRequested: {case.get('prompt', '')}\n"
            f"Viewpoint: {case.get('perspective', '') or 'unknown'}\n\n" + planner.MENU_NEUTRAL +
            "\n\nEverything the agents have written to the shared log, in the order written:\n- " + "\n- ".join(NOTES))
    v = llm.ask_json(planner.SYSTEM, text, [image], max_tokens=300)
    wait = (case.get("prompt") or "").lower().startswith("wait and observe")
    sa = str(v.get("scene_action", "none")).strip().lower()
    return {"action_register": 0 if wait else planner._snap(v.get("action_register"), planner.ALLOWED["action_register"], 0),
            "alive": planner._snap(v.get("alive"), planner.ALLOWED["alive"], 0),
            "anchor": planner._snap(v.get("anchor"), planner.ALLOWED["anchor"], 0.5),
            "scene_action": sa if (wait and sa in planner.SCENE_ACTION) else "none"}


ARMS = {"none": lambda c, im: planner.decide(c, im, use_memory=False, from_memory=False, write_back=False)["decision"],
        "log": decide_log,
        "past.dev": lambda c, im: planner.decide(c, im, use_memory=True, from_memory=True, write_back=False)["decision"]}
out = {"cases": sorted(CLIPS), "notes_in_log": len(NOTES), "arms": {}, "per_case": {}}
for arm, fn in ARMS.items():
    right = total = 0
    secs, chars, alive_on = [], [], 0
    for t in sorted(CLIPS):
        c = cases[t]
        case = {k: c[k] for k in ("task_id", "prompt", "image_caption", "perspective") if k in c}
        n0, t0 = len(sizes), time.time()
        try:
            d = fn(case, frame(t))
        except Exception as e:
            d = {"error": repr(e)[:120]}
        secs.append(time.time() - t0)
        chars.append(sum(sizes[n0:]))
        want = truth(c)
        ok = {k: d.get(k) == want[k] for k in want}
        right += sum(ok.values())
        total += len(ok)
        alive_on += int((d.get("alive") or 0) > 0)
        out["per_case"].setdefault(t, {})[arm] = {"decision": d, "correct": ok}
        print(arm, t, d, ok)
    tok = sum(chars) / len(chars) / 4
    out["arms"][arm] = {"accuracy": round(right / total, 3), "controls_right": right, "controls": total,
                        "alive_switched_on": alive_on, "seconds_per_decision": round(sum(secs) / len(secs), 2),
                        "prompt_chars_per_decision": round(sum(chars) / len(chars)),
                        "prompt_tokens_per_decision_est": round(tok),
                        "usd_per_1000_decisions_input_est": round(tok * 1000 / 1e6 * 1.0, 2)}
json.dump(out, open(os.path.join(ROOT, "results", "pastdev_bench.json"), "w", encoding="utf8"), indent=1, ensure_ascii=False)
lines = ["| How the planner learns what other agents measured | Controls set to the measured-best value | Alive write switched on (measured harmful) | Seconds per decision | Prompt size per decision | Input cost per 1,000 decisions |",
         "|---|---|---|---|---|---|"]
names = {"none": "No shared memory (own menu of beliefs)", "log": f"Whole shared log pasted into the prompt ({len(NOTES)} notes)",
         "past.dev": "past.dev recall (author, kind, time, supersededAt)"}
for arm, r in out["arms"].items():
    lines.append(f"| {names[arm]} | {r['controls_right']} of {r['controls']} ({100 * r['accuracy']:.0f}%) | {r['alive_switched_on']} of {len(CLIPS)} cases | "
                 f"{r['seconds_per_decision']:.1f} | about {r['prompt_tokens_per_decision_est']:,} tokens | ${r['usd_per_1000_decisions_input_est']:.2f} |")
open(os.path.join(ROOT, "results", "pastdev_bench.md"), "w", encoding="utf8", newline="\n").write("\n".join(lines) + "\n")
print("\n".join(lines))
