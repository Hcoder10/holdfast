"""Judge the held-out half: three arms on the same 95 cases, blinded together, judged twice.

  python -I -X utf8 eval/held_out.py fetch  <arm>     # export PlayWorld judge inputs on Modal and download them
  python -I -X utf8 eval/held_out.py blind             # shuffle all fetched arms into one anonymised set
  python -I -X utf8 eval/held_out.py judge  <pass>     # Gemini 3.1 Pro, direct API; pass = 1, 2, ...
  python -I -X utf8 eval/held_out.py tally             # results/held_out.json and results/held_out.md

Arms (Modal run name -> what it is):
  baseline   base95   the unmodified model through our driver
  objective  conf1    objective appended to the caption, nothing else
  final      final1   the final state program (results/final1_plan.json)

A case's score is the mean over judging passes (the judge moves a point or more on identical
inputs, see results/hillclimb_log.md). Scoring follows PlayWorld's aggregate.py: a failed gate
scores 1, otherwise `normalized_score`; overall is the mean of the four dimension means. The
interval on a paired difference is a bootstrap over cases within each dimension.
"""
import json
import os
import random
import shutil
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
# WH_SET selects the case set: "held" (default, the 95 held-out cases) or "dev" (the 76 development-half
# cases: the unmodified model's judge inputs from the earlier half-benchmark run, and the final program).
SET = os.environ.get("WH_SET", "held")
ARMS = {"held": {"baseline": "base95", "objective": "conf1", "final": "final1"},
        "dev": {"baseline": "half1", "final": "final1dev"},
        # 21 development-half cases (15 the judge called frozen, 6 that scored well): the cohesive base,
        # the alive write always on, and the launch-time planner choosing the program
        "sw21": {"baseline": "sweep21:lingbot2_14b_open_base", "alive": "sweep21:lingbot2_14b_open_alive03",
                 "final": "sweep21:lingbot2_14b_open_planner"},
        # the planner-chosen program on the 95 held-out cases, against the arms already judged there
        "v2": {"final": "final2"},
        # the planner arm of the 21-case test, run again after a bug left it switched off the first time
        "sw21b": {"final": "sweep21b:lingbot2_14b_open_planner"},
        # shadow world on 11 turn-and-return cases: base, static pin to the first frame ("alive" slot), shadow world
        # object ledger on the same 11 cases as sw22 (whose shadow arm is its control)
        "sw24": {"final": "sweep24:lingbot2_14b_open_ledger"},
        # 15 development insight cases: re-anchoring only, and with the Exa-derived action register at gain 8
        "sw27": {"baseline": "sweep27:lingbot2_14b_open_anchor", "final": "sweep27:lingbot2_14b_open_act8"},
        # shadow-world arm on the development half's first-person cases
        "v2dev": {"final": "final2dev"},
        # 14 turn-and-return out-of-sight cases: shadow world, and shadow world whose unseen copy gets the Exa sentence
        "sw26": {"baseline": "sweep26:lingbot2_14b_open_shadow", "final": "sweep26:lingbot2_14b_open_evolve"},
        "sw22": {"baseline": "sweep22:lingbot2_14b_open_base", "alive": "sweep22:lingbot2_14b_open_first",
                 "final": "sweep22:lingbot2_14b_open_shadow"}}[SET]
MODE = "lingbot2_14b_open"
INP, BLIND = RUNS / f"jin_{SET}", RUNS / f"jblind_{SET}"
OUTNAME = {"held": "held_out", "dev": "dev_half", "sw21": "sweep21", "v2": "held_out_v2", "sw21b": "sweep21b", "sw22": "sweep22", "sw24": "sweep24", "sw26": "sweep26", "sw27": "sweep27", "v2dev": "dev_half_v2"}[SET]
DIMS = ("GC", "IF", "IE", "OE")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def sh(cmd, timeout=3600):
    r = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", errors="ignore",
                       timeout=timeout)
    return (r.stdout or "") + (r.stderr or "")


def fetch(arm):
    run, _, mode = ARMS[arm].partition(":")
    mode = mode or MODE
    tmp = RUNS / f"jin_tmp_{run}"
    if run == "half1":                      # already exported and downloaded on Saturday
        tmp = RUNS / "jin_half1"
    else:
        if not (tmp / ".exported").exists():
            print(sh(["modal", "run", "modal_app/wh_modal.py::judge_inputs", "--run", run], 3000)[-300:])
            tmp.mkdir(exist_ok=True)
            (tmp / ".exported").write_text("1")
        sh(["modal", "volume", "get", "worldhold-runs", f"judge/{run}/{mode}", str(tmp) + "/", "--force"], 3000)
    dst = INP / arm
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for case in sorted((tmp / mode).iterdir()):
        if (case / "score_prompt.txt").exists() and not (dst / case.name).exists():
            shutil.copytree(case, dst / case.name)
            n += 1
    print(arm, "cases ready:", len(list(dst.iterdir())), f"(+{n})")


def blind():
    assert not BLIND.exists(), "already blinded; judging must use one fixed shuffle"
    print(sh([sys.executable, "-I", "eval/blind.py", str(INP), str(BLIND), "20261115"])[-200:])
    m = json.loads((BLIND / "blind_map.json").read_text())
    print("blinded folders:", len(m), {a: sum(1 for v in m.values() if v == a) for a in ARMS})


def judge(p):
    out = RUNS / f"gem_{SET}_p{p}"
    for workers in ("10", "4"):       # the second sweep picks up anything that failed
        r = sh([sys.executable, "-I", "-X", "utf8", "eval/openrouter_judge.py", "--inputs", str(BLIND), "--out",
                str(out), "--transport", os.environ.get("WH_TRANSPORT", "gemini"), "--workers", workers], 4 * 3600)
        print(r[-400:])
    cost = sum(json.loads(line).get("cost") or 0 for line in open(out / "cost.jsonl", encoding="utf8"))
    print(f"pass {p}: {len(list(out.glob('*/*/score_result.json')))} scored, ${cost:.2f}")


def tally():
    sys.path.insert(0, str(ROOT / "third_party" / "PlayWorld"))
    from metrics.aggregate import OE_INSIGHT_TASKS, normalized_score

    def load(pth):
        d = json.loads(Path(pth).read_text(encoding="utf8"))
        return d[0] if isinstance(d, list) else d

    def dim(t):
        return "GC" if t.startswith("GC") else "IF" if t.startswith("IF") else "IE" if t in OE_INSIGHT_TASKS else "OE"

    bm = json.loads((BLIND / "blind_map.json").read_text())
    passes = sorted(RUNS.glob(f"gem_{SET}_p*"))
    per = {a: {} for a in ARMS}       # arm -> case -> [score per pass]
    gates = {a: {} for a in ARMS}
    for pd in passes:
        for v, arm in bm.items():
            for case in (pd / v).iterdir() if (pd / v).is_dir() else ():
                sp, gp = case / "score_result.json", case / "gate_result.json"
                if not sp.exists():
                    continue
                raw = normalized_score(f"{v}/{case.name}", load(sp))
                if raw is None:
                    continue
                g = load(gp).get("verdict") if gp.exists() else None
                per[arm].setdefault(case.name, []).append(1.0 if g == "fail" else raw)
                gates[arm].setdefault(case.name, []).append(g)
    mean = {a: {c: st.mean(v) for c, v in per[a].items()} for a in ARMS}

    def table(scores):
        by = {d: [s for c, s in scores.items() if dim(c) == d] for d in DIMS}
        m = {d: (round(st.mean(v), 3) if v else None) for d, v in by.items()}
        ok = [x for x in m.values() if x is not None]
        return dict(m, overall=round(st.mean(ok), 3) if len(ok) == 4 else None, n=len(scores),
                    n_by_dim={d: len(v) for d, v in by.items()})

    out = {"passes": len(passes), "arms": {a: table(mean[a]) for a in ARMS}}
    if "alive" in ARMS:
        out.setdefault("paired_extra", {})
    # gate outcomes (first pass) for the two motion dimensions
    for a in ARMS:
        out["arms"][a]["gate"] = {d: {k: sum(1 for c, g in gates[a].items() if dim(c) == d and g and g[0] == k)
                                      for k in ("pass", "partial", "fail")} for d in ("GC", "OE")}
    # judge repeatability on this set: mean absolute difference between passes, same video
    if len(passes) >= 2:
        diffs = [abs(v[0] - v[1]) for a in ARMS for v in per[a].values() if len(v) >= 2]
        out["repeat_judging_mean_abs_diff"] = round(st.mean(diffs), 3) if diffs else None

    def paired(a, b):
        common = sorted(set(mean[a]) & set(mean[b]))
        by = {d: [(mean[b][c] - mean[a][c]) for c in common if dim(c) == d] for d in DIMS}
        point = st.mean(st.mean(v) for v in by.values() if v)
        rng, boots = random.Random(0), []
        for _ in range(5000):
            boots.append(st.mean(st.mean(rng.choices(v, k=len(v))) for v in by.values() if v))
        boots.sort()
        return {"n": len(common), "by_dim": {d: round(st.mean(v), 3) for d, v in by.items() if v},
                "overall": round(point, 3), "ci95": [round(boots[125], 3), round(boots[4874], 3)],
                "wins": sum(mean[b][c] > mean[a][c] + 0.25 for c in common),
                "losses": sum(mean[b][c] < mean[a][c] - 0.25 for c in common)}

    out["paired"] = {"final_minus_baseline": paired("baseline", "final")}
    if "alive" in ARMS:
        out["paired"]["alive_minus_baseline"] = paired("baseline", "alive")
    if "objective" in ARMS:
        out["paired"].update({"objective_minus_baseline": paired("baseline", "objective"),
                              "final_minus_objective": paired("objective", "final")})
    out["per_case"] = {a: {c: [round(x, 2) for x in v] for c, v in sorted(per[a].items())} for a in ARMS}
    (ROOT / "results" / f"{OUTNAME}.json").write_text(json.dumps(out, indent=1))
    lines = ["| Arm | Cases | Geometry | Interaction | Insight | Out-of-sight | Overall |", "|---|---|---|---|---|---|---|"]
    names = {"baseline": "Unmodified model", "objective": "Objective in the prompt only", "final": "Final state program",
             "alive": "Alive write always on"}
    if SET == "sw21":
        names.update(baseline="Cohesive base", final="Planner-chosen program")
    if SET == "sw22":
        names.update(baseline="Base", alive="Pin to the first frame on return", final="Shadow world")
    if SET == "v2":
        names.update(final="Planner-chosen program")
    for a in ARMS:
        t = out["arms"][a]
        lines.append(f"| {names[a]} | {t['n']} | " + " | ".join(
            "-" if t[d] is None else f"{t[d]:.2f}" for d in DIMS) + f" | {'-' if t['overall'] is None else format(t['overall'], '.2f')} |")
    lines.append("| Genie 3 (published, 171 cases) | 171 | 2.74 | 2.40 | 1.51 | 1.81 | 2.12 |")
    (ROOT / "results" / f"{OUTNAME}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({k: out[k] for k in ("passes", "paired", "repeat_judging_mean_abs_diff") if k in out}, indent=1))
    for a in ARMS:
        print(a, "gate:", out["arms"][a]["gate"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"fetch": lambda: fetch(sys.argv[2]), "blind": blind, "judge": lambda: judge(sys.argv[2]), "tally": tally}[cmd]()
