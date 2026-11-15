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

### Side memory sweeps (01:10-01:45)

| sweep | write | what the frames show | verdict |
|---|---|---|---|
| sweep9 | self-attention gain 2-16 on the sink (first frames), applied (3,200+ calls logged) | indistinguishable from the control | no effect; diagnostic running |
| sweep11 | heading-indexed latent bank written into half of the sink (29 views stored, 5 superseded, reads logged) | indistinguishable from the control | no effect through this route |
| sweep10 | return register: Exa-sourced sentence for the state on return, silent while away, gain 4 or 8 on return | OE019: the red tram is back, closer, same colours (control: replaced by a modern tram). OE036: one man and dog further along (control and sink variants: a duplicate man). OE017, OE029: no change, no harm | keep (gain 4) |
| sweep10 | caption re-asserted on return at gain 4 | duplicates: second dog and a new man, second locomotive, oversized apples | drop |
| sweep12 | keyframe arrival through the image-condition channel | fires at 20 and 35 degrees tolerance. 35: cross-fade ghost. 20: clean arrival on the easy case | hard cases pending |

Sentences written by the agents (cartographer: Exa question and fact; director: sentence), e.g. OE019: "The red
streetcar with gold trim has moved significantly closer to the camera's position, now much nearer along the
cobblestone street as it continues its approach down the tracks."

### Where the points are lost: the gate (02:00)

Baseline, 76-case half, real judge. The gate checks that the requested camera trajectory happened; a fail scores 1.

| dimension | cases | gate pass / partial / fail | mean | share of cases scoring <= 1.2 |
|---|---|---|---|---|
| Geometry | 24 | 16 / 6 / 2 | 2.03 | 42% |
| Out-of-sight | 22 | 8 / 7 / 7 | 1.51 | 82% |

Reading the nine failures against what was commanded and what the frames measured:
- translation ignored: the player pressed W/S or A/D and the camera stayed still (GC026, OE043, OE032, and the
  interaction case IF011 has the same symptom);
- the turn back shows new content, so it reads as "kept turning, never came back" (OE053: commanded and measured
  right x10, wait, left x10; judge: "continuously pans right");
- the first-frame subject stays in frame while the background turns, so it reads as "camera static" (OE046, OE044);
- a turn rendered as travel along a repeating structure (GC032).

### Sink diagnostic (02:00): the write is correct, and the sink is the model's return memory

Share of self-attention on the six first frames, measured inside the forward pass (explicit softmax on sampled
queries, 10-11 calls per run, two cases): unmodified 17-21%; gain 0.001 -> 0.0%; gain 16 -> 47-54%; gain 1000 -> 81%.
Frames, fixed actions, turn right and back: unmodified returns to the start view; with the sink blocked it does
not (ends on a different building); gain 16 looks like the control; gain 1000 smears the start view over everything.
So the earlier "no effect" was real, not a mislocated write: the model already uses those frames, and more is not better.

### Bug found: heading reset at every translation

`Episode.yaw` restarts at 0 after each translation chunk (by design, for the rotation memory). The return register,
caption gate and keyframe arrival read it as the heading from the start, so after "turn 90, walk" they believed
the camera faced the start. Fixed: they now use the heading accumulated from the episode's start.

### Trajectory legibility: caption gate and camera guidance (02:30)

| sweep | write | what the frames show | verdict |
|---|---|---|---|
| sweep15 | caption as a register whose gain falls to 0.3 / 0.1 / 0.02 as the camera turns away | OE044 (tomato cutting): control paints tomatoes and the boy into every direction; at 0.1 the away views are plain kitchen and the return shows the cutting again, twice. OE053 (bakery): control is bakery in all directions; gated, the away view is a window and wall and the return is the counter. OE046 (conveyor): at 0.1 a distinct away view appears. GC010: unchanged. The same register at constant gain 1 behaves like the control, so the gain is what matters | keep, away = 0.1 |
| sweep14 | camera guidance w = 1.5-5 (extrapolating each block's camera scale and shift away from a still camera's) | OE043: the ignored strafe is executed from w = 2, clearly at 3. GC032: "rotate in place" was rendered as travel along the bridge; at 3-5 it is a pan from a fixed spot, still sharp. GC026, OE032 and IF011 (walk forward toward something far): still no visible advance at any w | keep w = 3 for first-person; forward on distant scenes unsolved |

Ruled out as bugs while checking: a missing WASD input (upstream sets it to None; the camera layer takes pose rays
only); translation and rotation signs (derived from upstream's example trajectories: W +z, A -x, D +x, look right
+yaw, look up +pitch, matching ours); step size (upstream's normalised W step is about 0.75, ours 1.0); intrinsics.

### Modal budget from 02:40 (owner's limit: $250, assumed $4.50 per GPU-hour)

| item | GPU-h | est. cost |
|---|---|---|
| sweeps 17, 18, 19 (already running at 02:40) | 11 | $50 |

### Combined validation on twelve first-person cases (sweep19, 03:05) and what was rejected at the end

| write | what the frames show | verdict |
|---|---|---|
| caption gate (away 0.1) | OE044, OE053, OE046: distinct away views and a recognisable return (see sweep15). GC015: stays a coherent cabin interior where the control wanders outside. GC010, OE017: return as before | in the final configuration |
| + camera guidance w = 3 | helps the cases it was built for (OE019: the red tram is kept; GC032: pan in place) but breaks cases that were fine: GC010 ends pressed against a wall and blurrier (sharpness 4.5 -> 3.0), OE017 and GC015 leave the room and end outdoors, OE044 loses sharpness (3.8 -> 2.5) | rejected: turns already execute at w = 1 and the extra push over-rotates |
| + camera guidance w = 5 | chaotic on every case | rejected |
| first frames' attention cut to 0.25 / 0.05, with and without camera guidance, and step 0.75 (sweep18) | GC026 (shopfront) and IF011 (Sphinx): no advance in any variant | hypothesis rejected; walking forward from a standing start on these scenes is unsolved |
| keyframe arrival after the heading fix (sweep17) | fires on 2 of 6 cases (the player rarely returns to within a step of the start) | not in the final configuration |

Still to check before the final run: caption gate + return register without camera guidance (sweep20).

| Modal ledger | GPU-h | est. cost |
|---|---|---|
| sweep20 (8 cases, 4 GPUs) | 1.5 | $7 |

### Caption gate with and without the return register (sweep20, 03:18)

Seven out-of-sight cases, orchestrator's reading of frames: with the return register the subject on return is
better on three (OE036 one man and dog further along; OE046 the box is still on the conveyor, moved on; OE019 ends
on the street instead of inside a tram), worse on one (OE017 leaves the room during the turn), the same on three.
Kept, at gain 4.

### Final configuration (results/final1_plan.json), run once on the 95 held-out cases

`{"prompt": "caption+objective", "register": {"gain": 8, "when": "third person", "replace": true},
  "anchor_stats": 0.5, "cap_gate": {"away": 0.1}, "ret": {"gain": 4}}`

| Modal ledger | GPU-h | est. cost |
|---|---|---|
| final1 part A (49 cases, 4 GPUs) | 3 | $13 |
| final1 part B (46 cases, 6 GPUs) | 4 | $18 |

### Blinding bug in the held-out judging (04:50), found at the first tally

`eval/blind.py` replaced each run folder's name with "system_under_test" inside the prompts. The held-out arms were
named `baseline`, `objective` and `final`, so ordinary words were rewritten: every "final" in the final arm's scoring
prompts (about 62 per prompt; "final score" became "system_under_test score") and "objective" in the objective arm's
gate prompts. The baseline arm was untouched. Fixed: only the model identifier is replaced. The 140 affected prompts
were restored from the originals, the 190 scores and 90 gate checks made with them were set aside
(`*.misworded.json`, kept) and those calls were judged again in both passes.

| Modal ledger | GPU-h | est. cost |
|---|---|---|
| base95: unmodified model on the 95 held-out cases (5 GPUs) | 6 | $27 |
| final1dev: final configuration on the 76 development-half cases, for a full-benchmark figure (5 GPUs) | 5.5 | $25 |

### Held-out result (05:05): 95 cases, three arms, two passes, corrected prompts

| arm | GC | IF | IE | OE | overall |
|---|---|---|---|---|---|
| unmodified | 2.40 | 1.89 | 2.19 | 1.92 | 2.10 |
| objective in the prompt only | 2.66 | 2.04 | 2.11 | 2.13 | 2.24 |
| final state program | 2.48 | 2.24 | 2.47 | 2.06 | 2.31 |

Final minus unmodified +0.21 (95% interval -0.04 to +0.47). Judge cost for the held-out set: $125 on the direct key.
What the orchestrator's frame-reading got wrong: it expected interaction near 3.0 or more (2.24), and expected the
caption gate to lift out-of-sight above the objective-only arm (2.06 vs 2.13). What it got right: the gate passes
more out-of-sight trajectory checks (12 vs 8), and re-anchoring helps the 30 s insight cases (+0.28).

### Full benchmark (06:05): final program and unmodified model on all 171 cases

Development half (76 cases, one Gemini pass, both arms blinded together): unmodified 2.15 (GC 2.30, IF 2.43, IE 2.40,
OE 1.46), final 2.34 (2.42, 2.44, 2.67, 1.82); paired +0.19 (-0.10 to +0.48). Combined with the held-out half:
unmodified 2.10, final 2.31, paired +0.22 (95% interval +0.03 to +0.40) over 171 cases. `results/full_benchmark.json`.

### Frozen-case test of the residual "alive" write, judged (sweep21, 06:12)

21 development-half cases: 15 the judge had called frozen or static, 6 that scored well. Two Gemini passes averaged.

| arm | 15 frozen cases | 6 controls |
|---|---|---|
| cohesive base (objective prompt, third-person register, re-anchoring) | 1.87 | 3.00 |
| + alive write always on (0.03, sampling steps 0-1) | 1.93 | 2.45 |

The write that triples frame-to-frame motion does not make the judge score frozen cases higher (+0.06, inside the
0.67 repeat spread on this set), and switched on everywhere it costs half a point on scenes that were fine. More
motion is not the same as the requested thing happening.

The third arm, "planner", did not run the planner: it was given an empty settings object, which the code read as
off, so it ran as objective-only (1.67 and 3.30, i.e. no different from the base within noise). The switch is
fixed and the arm was run again as sweep21b.

The per-dimension "overall" printed by the tally for this set is not meaningful: geometry has one case in it.

| Modal ledger | GPU-h | est. cost |
|---|---|---|
| sweep21 (21 cases x 3, 7 GPUs) and sweep21b (planner arm again, 5 GPUs) | 8 | $36 |
| sweep22: shadow world vs first-frame pin vs base, 11 turn-and-return cases (6 GPUs) | 4.5 | $20 |
