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
