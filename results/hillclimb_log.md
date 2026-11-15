# Hill-climb log

Each row is one trial on the development cases (the half of the benchmark already used). Differences are paired, against the baseline and against the incumbent at the time.

| # | time | configuration | hypothesis | vs baseline | by dimension | W/L | verdict | lesson |
|---|---|---|---|---|---|---|---|---|
| 2 | 21:40 | memory+shadow, brief=dynamics | shadow rollout plus motion sentence lifts OE and GC | -0.19 (2.11) | GC +0.03, IF +0.29, IE -0.47, OE -0.62 | 6/7 | rejected | shadow rollout does not help; brief may help IF |
| 3 | 22:14 | open, prompt=caption+objective | objective as text (as PlayWorld's own adapter does) lifts IF | +0.28 (2.58) | GC +1.27, IF +1.63, IE -1.17, OE -0.62 | 8/7 | incumbent | IF gain is large and mechanistic (IF004 1->4, IF006 1->5, IF013 2->5). The IE change is not real: see below |

## Judge test-retest noise (found in trial 3)

The six insight cases have the objective "wait and observe", which the caption+objective setting leaves out, so their
prompt in trial 3 is the baseline's prompt. Five of the six videos came out byte-identical to the baseline's (same
contact-sheet hashes). The judge, same model and prompt, scored those identical inputs:

| case | first judging | second judging |
|---|---|---|
| OE005 | 5.0 | 3.0 |
| OE010 | 5.0 | 3.0 |
| OE008 | 5.0 | 5.0 |
| OE006 | 2.0 | 1.0 |
| OE009 | 1.4 | 1.8 |

Mean absolute difference 1.08 on a 1-5 scale, with no change to the video. Consequences: a single judging of 6 cases
cannot resolve a dimension difference under about 1 point; the "IE -1.17" in trial 3 is this noise plus one changed
video (OE002); and any per-case swing between runs must be read with this in mind. Identical videos are now to be
treated as a zero difference rather than re-judged.

## Trial 4 and the pooled baseline (22:40)

| 4 | 22:38 | open, watchdog | rollback and reseed on latent resets and freezes lifts every dimension | -0.06 (2.24) | GC +0.40, IF +0.30, IE -1.07, OE +0.12 | 5/7 | rejected | 10 of 24 videos were byte-identical to the baseline's (the watchdog never fired); on the 14 it changed there is no gain |

Trial 4 gave more repeat judgings of identical baseline videos. Same video, same judge, same prompt:
GC010 1.0 / 4.2, IF004 1.0 / 3.0, OE005 5.0 / 3.0 / 1.0, OE010 5.0 / 3.0, OE006 2.0 / 1.0, OE002 3.4 / 4.0.
The raw "by dimension" figures for trial 4 above are therefore mostly judge noise on unchanged videos.

`loop/pool_baseline.py` averages all judgings of each baseline video (`runs/hillclimb/baseline_pooled.json`) and
gives an unchanged video the baseline's value. Against that:

| | GC | IF | IE | OE | overall |
|---|---|---|---|---|---|
| baseline (pooled) | 1.72 | 2.15 | 3.10 | 2.23 | 2.30 |
| 2: memory+shadow+brief | 1.43 | 2.29 | 3.17 | 1.54 | 2.11 |
| 3: caption+objective | 2.67 | 3.63 | 2.65 | 1.54 | 2.62 |
| 4: watchdog | 1.48 | 2.15 | 2.92 | 2.23 | 2.19 |

Only trial 3's interaction gain (+1.5) is clearly outside the judge's noise.

## Trials 1 and 5, all trials against the pooled baseline (23:20)

`python -I -X utf8 loop/pool_baseline.py` (table in `runs/hillclimb/pooled_table.json`). "Changed" is the number of
videos that differ from the baseline's; W/L counts changed videos more than 0.25 above or below it.

| | GC | IF | IE | OE | overall | changed | W/L | verdict |
|---|---|---|---|---|---|---|---|---|
| baseline (pooled) | 1.85 | 2.15 | 3.10 | 2.23 | 2.33 | | | |
| 1: step-0 Exa brief | 1.43 | 2.33 | 2.17 | 1.58 | 1.88 | 22 | 5/9 | rejected |
| 2: memory+shadow+brief | 1.43 | 2.29 | 3.17 | 1.54 | 2.11 | 23 | 5/8 | rejected |
| 3: caption+objective | 2.67 | 3.63 | 2.65 | 1.54 | 2.62 | 19 | 6/3 | incumbent |
| 4: watchdog | 1.62 | 2.15 | 2.92 | 2.23 | 2.23 | 14 | 0/3 | rejected |
| 5: caption+objective+memory | 1.83 | 2.60 | 2.58 | 1.00 | 2.00 | 20 | 6/7 | rejected |

Lessons:
- The step-0 brief does not help on its own; the interaction gain seen in trial 2 was not it.
- Rotation memory does not add to the objective prompt. Its earlier "+0.3 on geometry" is within the judge's noise.
- OE029 scored 5 in its single baseline judging and 1 in all four trials that changed its video. The baseline's
  out-of-sight mean (2.23) leans on that one judging; every changed configuration sits near 1.0-1.6.
- GC010's baseline video has now been judged 1.0, 5.0 and 4.2.
- The one effect outside the noise remains: the objective in the prompt lifts interaction (2.15 to 3.63 and 2.60).

## State-write sweeps, judged from frame strips by the orchestrating model (00:00-01:00, 15 Nov)

No Gemini calls. `scripts/view_grid.py <run> <case>` rebuilds every strip from the volume. Scores here are the
orchestrator's reading of the frames against the public rubric wording, not benchmark scores.

| sweep | write | cases | what the frames show | verdict |
|---|---|---|---|---|
| sweep1 | residual add / ablate, all sampling steps | 4 | add >= 0.03 and every ablation: colour cast growing into collapse. add 0.01 or one layer: clean, more motion | broken as written |
| sweep6 | same, limited to sampling steps 0-1, not in the cache pass | 4 | add 0.03 on steps 0-1: clean and sharp, motion x3. All steps without the cache pass: still collapses. So the late steps were the cause | fixed |
| sweep3 | forward translation x2, x4, x8 | 4 | x8 is noise in one chunk. Upstream normalises translation to a maximum of 1.0, so >1 is outside the model's input range | not a bug; keep 1.0 |
| sweep2 | objective in its own cross-attention register, attention gain 1-16 | 6 IF | gain >= 4 completes the action on 3 cases where the incumbent does not (slope, pond, river). Gain 1 = no effect. First-person driving case loses the car at gain >= 4 | keep, third-person only |
| sweep7 | same register on GC / OE (camera-instruction objectives) | 6 | at gain 4-8 the word "camera" materialises a camera on a tripod | off for first-person |
| win* | attention window 30/36/42 frames, sink 6/12 | 6 | runs, no visible gain on return-to-start | neutral |
| sweep4 | latent mean re-anchoring 0.3-0.7 | 6 | GC002: without it the hall drifts to deep blue and dark over 30 s; with it colours hold. OE028: no fade to black. No harm on a clean 30 s scene | keep (0.5) |
| sweep4 | sink attention gain | 6 | never applied: patched the wrong module (0 hits recorded). Fixed, not yet re-run | re-run |
| sweep4 | correlated chunk noise 0.5 | 6 | no clear change | drop |
| sweep5 | memory through the image-condition channel | 6 | returns to the start view where latent pasting did not, but with ghosting and block artifacts | needs work |

### Register validation on seven third-person interaction cases not used to choose it (sweep8, 01:10)

Orchestrator's scores from frame strips (1-5, interaction rubric wording):

| case | objective in prompt | + register 4 | + register 8 | register 8 alone |
|---|---|---|---|---|
| IF014 car turns right at the intersection | 3 (cuts to an interior view) | 2.5 (car changes shape) | 4 | 4 |
| IF015 woman walks to the hot dog stand | 3 | 3.5 | 4 | 4.5 |
| IF016 walks to the fence and on | 3 | 3 | 3 | 3 |
| IF021 waiter walks straight | 4 | 4 | 4 | 4 |
| IF023 walks the path to the sea | 4 | 4 | 4 | 4 |
| IF025 enters the tall corn | 2 (subject dissolves) | 4 | 4 | 4 |
| IF026 up the steps, hand on the rope | 3.5 | 3.5 | 4 | 4.5 |
| mean | 3.2 | 3.5 | 3.9 | 4.0 |

Chosen for third-person cases: the objective in its own register at gain 8, caption as the base prompt.
