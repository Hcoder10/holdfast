# References

Collected 14 November 2026 by web research. Nothing here was run by us; timings
are estimates. Items marked (unverified) were not checked against a primary source.

## PlayWorld

- Paper: https://arxiv.org/abs/2608.13552 ("PlayWorld: Benchmarking World Models
  with Agent Players over Long-Horizon Objectives", Ding et al.)
- Code: https://github.com/kxding/PlayWorld (no licence file)
- Data: https://huggingface.co/datasets/jocelynd/playworld-bench (264 MB, public)
- Leaderboard: https://huggingface.co/spaces/jocelynd/PlayWorld-Leaderboard
- Project page: https://kxding.github.io/project/PlayWorld/

Protocol: the model gets an initial frame and a text objective. A Claude Haiku 4.5
agent watches generated frames and picks from W/A/S/D, arrow keys and WAIT. An
episode ends on FINISH or at 40 steps (about 10-60 s of video).

Size: 171 cases, 820+ rubric questions.

| Dimension | Cases |
|---|---|
| Geometry Consistency (GC) | 48 |
| Interaction Fidelity (IF) | 50 |
| Insight Evolution (IE) | 30 |
| Out-of-sight Evolution (OE) | 43 |

Scoring: Gemini 3.1 Pro answers yes/no rubric questions over contact sheets (10 fps
and 0.5 fps streams). Per-dimension score = 1 + 4 x weighted yes-fraction. Overall =
unweighted mean of the four. GC, OE and third-person IF also pass through a
Pass/Partial/Fail instruction-following gate; a failed gate scores 1.

Released: agent harnesses for Genie 3, HappyOyster, HY-World2 (browser automation),
one local adapter (SANA-WM), a `your-models/` template, the scoring code.
Missing: the Matrix-Game 3.0 adapter (`metrics/aggregate.py` refers to
`matrixgame3_native`, which is not in the repo). `metrics/automatic/` is empty.
The SANA-WM adapter calls an endpoint with Kuaishou-internal headers and needs
light editing.

Judge noise: mean per-model sample variance 0.011 across two scoring passes.

## Matrix-Game 3.0

- Repo: https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-3
- Weights: https://huggingface.co/Skywork/Matrix-Game-3.0 (56.6 GB total)
- 5B, fine-tuned from Wan2.2-TI2V-5B. No training code.
- Generates 40 new frames per iteration; 3 steps distilled, 50 steps base.
- Memory: retrieves past frames by camera pose and field-of-view overlap, plus an
  optional "sink" first latent.
- Blackwell: requirements pin torch 2.10.0; a reporter ran torch 2.11+cu128 on
  RTX 5090s. FlashAttention 3 is Hopper-only, use FlashAttention 2.
- Known issues: #80 multi-GPU crash (`fa_version=` should be `version=` in
  `wan/distributed/ulysses.py`), #86 flash-attn mandatory, #72 multi-GPU hang with
  int8, #74 black frames after the first, #73 treats background dynamics as static,
  #95 sink-position change for long-run stability.
- Why it sits at the floor (inference): weights are for first-person generation in
  unreal scenes while PlayWorld images are largely real photos; only 37.5% of its GC
  rollouts pass the gate; camera rotation pass rate is 27.7%.

## ABot-World-0

- https://github.com/amap-cvlab/ABot-World,
  https://huggingface.co/acvlab/ABot-World-0-5B-LF (24.8 GB), arXiv 2607.19191
- 5B, 720p at up to 16 FPS, about 19 GB on one RTX 5090. Single GPU only.
- Third-party reproduction on an RTX PRO 6000 Blackwell exists (issue #7).
- Open issue #15: checkerboard artifact on every frame in the official Docker image.
- No training code. No published PlayWorld score.

## Other starting points

- SANA-WM: https://github.com/NVlabs/Sana (Apache-2.0, 2.6B). Score 1.48. The only
  model with a released local PlayWorld adapter; defaults to 60 denoising steps.
- LingBot-World / LingBot-World2: https://github.com/robbyant/lingbot-world,
  https://github.com/Robbyant/lingbot-world-v2. Scores 1.78 / 1.82, measured through
  the web interface, so local weights may differ. v2 size, licence and downloadable
  checkpoints (unverified).
- HY-WorldPlay: https://github.com/Tencent-Hunyuan/HY-WorldPlay. Score 1.21.

## Methods that fit in a day

- Matrix-Game 3.0's own knobs: number and selection of retrieved memory latents,
  the sink latent, the #95 sink-position change. Zero training.
- Keyframe re-anchoring in the adapter: re-inject the first frame or stored
  keyframes as memory when the camera pose returns. Targets GC and OE revisits.
- Deep Forcing: arXiv 2512.05081, https://github.com/cvlab-kaist/DeepForcing.
  Training-free deep sink plus context compression; built on Self-Forcing / Wan2.1,
  so it needs porting.
- VMem: https://github.com/runjiali-rl/vmem. Surfel-indexed view memory; needs a
  depth/pose pipeline.
- Need training, ideas only: WorldMem, Stable Video Infinity, LongLive.

## Time estimates (inferred, not measured)

| Stage | Estimate |
|---|---|
| Environment, weights, first stock rollout | 1.5-3 h |
| Matrix-Game 3.0 PlayWorld adapter | 3-6 h |
| Generation for one full evaluation, 171 episodes, 3 GPUs | 1.5-3 h |
| Judging, about 290 Gemini calls in parallel | 15-40 min |
