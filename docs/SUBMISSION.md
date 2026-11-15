# Submission text

## Short description

**Holdfast: state programs for a world model.** Track 4, World Models.

We take an open world model (LingBot-World 2, 14B) and make it hold together longer on PlayWorld without
changing a weight. A thin controller reads the model's state every 0.75 seconds and writes small corrections
into its forward pass. Three writes earned their place, each at a different level of the model's state and each
fixing one named failure:

1. **Attention to text.** The request gets its own cross-attention segment and eight times the attention, so the
   subject does what was asked. Interaction 2.05 to 2.30.
2. **Latents carried forward.** Each chunk's latent mean is pulled halfway back toward the first chunk's, which
   stops long rollouts drifting. Insight 2.30 to 2.57.
3. **The conditioning input.** When the camera turns away, a copy of the whole generation state keeps running the
   view that was left; on return the arriving frame is pinned to the copy's frame. Geometry 2.48 to 2.82 on the
   held-out half.

**Result.** On all 171 PlayWorld cases, judged by Gemini 3.1 Pro with the benchmark's own prompts on blinded
folders: 2.31, against 2.10 for the unmodified model (paired +0.22, 95% interval +0.03 to +0.40) and 2.12
published for Genie 3. On the 95 cases never used for development, judged twice: 2.31, and 2.40 with the third
write (that arm was designed after the held-out result was known, and we say so).

**Rigour.** The configuration was chosen on half the benchmark and run once on the other half. We measured the
judge's own noise (the same video scored 5, 3 and 1 on repeats) and report intervals. A blinding bug of ours was
caught at the first tally, fixed and re-judged. Fifteen things we built did not work and are documented with
their evidence: residual-stream steering, attention gain on the first frames, a view bank, camera guidance, an
object ledger, a launch-time planner, motion written into the noise, and more.

**Where it still breaks.** A subject asked to walk forward from a standing start on a real photograph does not
move, whatever we write. Things that should progress while out of view (a sketch, a pour) come back unchanged.

**Sponsors.** Exa is called inside every first-person episode (its answer becomes a cross-attention register
that is silent until the camera returns), in a scout that searches recent work for each failure mode and maps
it onto our write ports, and in a checker that verifies the numbers we quote against live sources. past.dev
holds nine agent identities behind three read boundaries; our demo shows a belief published by one agent,
refuted by another's judged measurement, and a third agent's decisions changing from 8 of 8 to 0 of 8 as a
result. Modal ran every rollout.

## Links

- Repository: https://github.com/Hcoder10/worldhold
- Demo video: `demo/worldhold_demo.mp4` (90 seconds; rebuilt with `python -I -X utf8 scripts/make_demo.py`)
- Results: `results/full_benchmark.json`, `results/held_out.json`, `results/held_out_v2.json`
- Every trial and sweep: `results/hillclimb_log.md`
