# worldhold

Hackathon Track 4 (World Models, Applied Reality), 14-15 November 2026.
Submission due Sunday 15 November, 14:30 PT.

**Question:** how long can a generated world hold together while an agent plays in it?

**Approach:** take an open world model (LingBot-World 2) and give it an explicit
world state outside the network, with no training:

1. **Step-per-chunk driver** (`adapters/lingbot/stepper.py`): the stock model takes
   a whole camera path and returns one video. The driver generates 0.75 s at a time
   with the camera motion and the text chosen at that moment, which is what an
   interactive benchmark needs.
2. **Rotation odometry** (`odometry.py`): a homography between frames measures how
   far the camera actually turned, so turns are closed-loop instead of hoped for.
3. **Rotational scene memory** (`panorama.py`): for a camera that only rotates, any
   earlier frame maps exactly onto a later view. Remembered views are fed back into
   the denoiser when the camera returns, so the model redraws the same place. Before
   each use the memory is matched against the screen (re-localisation); if they do
   not match, nothing is pasted.
4. **World-state director** (`director.py`, `director_prompt.md`): a planner that
   sees only the first frame, the caption and the camera plan writes a timeline of
   how the scene's process should progress. The text condition follows episode time,
   including while the target is off screen, and the evolving target is left out of
   what the memory pins.

## Status (Saturday 14:00 PT)

Working end to end on a rented RTX PRO 6000: driver, odometry, memory, director
wiring, and a judging path. Geometry results below are from the 1.3B model; the 14B
runs and the director experiments are in progress.

## Results so far

All scores use PlayWorld's released scorer prompts and aggregation, but with a
**Claude Haiku subagent standing in for Gemini 3.1 Pro as judge** and preset action
sequences instead of the player agent. They are valid for comparing our own runs
with each other and are **not comparable with published PlayWorld numbers**.
Every row is in `results/ledger.jsonl`.

10 pure-rotation Geometry Consistency cases, LingBot-World 2 1.3B, seed 0, one
blinded judge pass per video:

| | Open loop (baseline) | With memory |
|---|---|---|
| Trajectory gate: pass / partial / fail | 1 / 8 / 1 | 10 / 0 / 0 |
| Returns to the start view (homography check, 9 checkable cases) | 1 of 9 | 9 of 9 |
| Mean score, released scorer (partial counts as pass) | 3.60 | 3.88 |
| Mean score excluding the one failed gate | 3.89 | 3.87 |

What this does and does not show:

- The memory makes the camera come back to the same scene. Without it the model
  draws a different place in 8 of 9 cases.
- The released scorer barely rewards that. Its gate counts a partial return as a
  pass, and its rubric scores first-frame objects only when they reappear, so a run
  that never returns gets a default score. The whole 0.28 gap is one failed gate.
- Judge noise is large: the same baseline video scored 3.4 in one pass and 4.0 in
  another.
- The memory runs judged here had a defect (misaligned pasted strips when the
  angle estimate was wrong). Re-localisation was added afterwards and is not yet
  re-judged.

Published reference numbers (PlayWorld paper, Table 2; Gemini judge, agent player):

| Model | GC | IF | IE | OE | Overall |
|---|---|---|---|---|---|
| Genie 3 | 2.74 | 2.40 | 1.51 | 1.81 | 2.12 |
| LingBot-World2 | 2.04 | 2.13 | 1.95 | 1.16 | 1.82 |
| Matrix-Game-3.0 | 1.30 | 1.25 | 1.00 | 1.00 | 1.14 |

## Layout

```
adapters/lingbot/  driver, odometry, memory, director, episode runner
eval/              judge_prep (export PlayWorld's judge inputs), blind, judge_collect
scripts/           box setup, case runner, job server, small tools
results/           ledger.jsonl, director timelines
docs/              references and notes
```

## Reproduce

On a GPU box with about 96 GB VRAM (the 14B model peaks at 88 GB):

```bash
bash scripts/box_setup.sh                      # environment, weights, PlayWorld data
python scripts/run_cases.py --repo lingbot-world-v2 --ckpt lingbot-world-v2-14b-causal-fast \
    --task i2v-A14B --data PlayWorld/data --cases GC010,GC022 --modes open,memory --out out/
cd PlayWorld && python ../worldhold/eval/judge_prep.py --data data --runs ../out --out ../judge
python eval/blind.py judge judge_blind 0      # then have the judge answer each folder
python eval/judge_collect.py --judged judge_blind --playworld PlayWorld \
    --judge "<who judged>" --player "preset actions" --map judge_blind/blind_map.json
```

## Rules we hold ourselves to

- Generation code reads only a case's image, caption, objective, perspective and
  action sequence (`ALLOWED_CASE_FIELDS` in `episode.py`). The rubric is read only by
  `eval/judge_prep.py`.
- Judges see shuffled, anonymised folders.
- Nothing is claimed that is not in the ledger, and every ledger row names its judge
  and player.

## Known limits

- Memory covers rotation about a standing point. After the camera translates, it
  starts over from the new position; there is no depth-based memory.
- The driver's guidance leaves a visible join in some frames where remembered and
  new content meet.
- No scores yet with the real judge or the real player agent.

See `docs/references.md` for sources.
