# worldhold

Hackathon Track 4 (World Models, Applied Reality), 14-15 November 2026.
Submission due Sunday 15 November, 14:30 PT.

**Question:** how long can a generated world hold together while an agent plays in it?

**Goal:** take an open world model and raise its PlayWorld score with inference-side
methods (memory, state tracking, re-anchoring). No training from scratch.

## Scoreboard

PlayWorld scores are on a 1-5 scale; overall is the mean of four dimensions.
Published numbers, from the PlayWorld paper (arXiv 2608.13552), Table 2:

| Model | GC | IF | IE | OE | Overall | Open weights |
|---|---|---|---|---|---|---|
| Genie 3 | 2.74 | 2.40 | 1.51 | 1.81 | 2.12 | no |
| LingBot-World2 | 2.04 | 2.13 | 1.95 | 1.16 | 1.82 | yes (unverified) |
| LingBot-World | 2.11 | 2.23 | 1.33 | 1.43 | 1.78 | yes |
| SANA-WM | 1.72 | 1.89 | 1.13 | 1.16 | 1.48 | yes |
| Matrix-Game-3.0 | 1.30 | 1.25 | 1.00 | 1.00 | 1.14 | yes |

GC = Geometry Consistency, IF = Interaction Fidelity, IE = Insight Evolution,
OE = Out-of-sight Evolution.

Our own runs go in `results/ledger.jsonl`, one line per evaluation. Nothing is
claimed here until it is in the ledger.

| Run | Model | Method | Cases | Overall | Notes |
|---|---|---|---|---|---|
| _none yet_ | | | | | |

## Layout

```
adapters/   PlayWorld adapters, one per world model (agent actions in, frames out)
methods/    consistency methods applied on top of a model (memory, re-anchoring, ...)
eval/       subset definitions, run launcher, judge-output cache, ledger writer
scripts/    environment setup and one-off tools
results/    ledger.jsonl and per-run summaries (videos stay out of git)
docs/       references and notes
```

## Plan

1. **Baseline first.** Stand up PlayWorld and get a number from our own run before
   changing anything. SANA-WM has a released adapter, so it is the fastest route to
   a first number; Matrix-Game 3.0 needs an adapter written.
2. **Fixed subset for iteration.** A 30-40 case subset weighted to GC and OE (the
   dimensions that test memory), with fixed seeds and cached judge outputs. Full
   171-case runs only to confirm.
3. **Cheap gains before clever ones.** Action mapping and the instruction-following
   gate first (a failed gate scores 1), then memory and re-anchoring.
4. **Replicate before claiming.** Judge noise is about 0.011 variance between
   scoring passes, so a subset gain is rerun before it goes in the table.
5. **Show where it still breaks.** Keep failure rollouts; the track judges on rigour.

## Open decisions

- Starting model: Matrix-Game 3.0 (named by the organisers, score 1.14, no adapter)
  versus SANA-WM (adapter exists, 1.48) versus LingBot-World (highest open score).
- Ask the organisers: do they score with their own harness or a held-out split, and
  is a starting model other than Matrix-Game 3.0 or ABot-World-0 allowed?

## Keys and compute

- Gemini 3.1 Pro (judge) and Claude Haiku 4.5 (player agent) API keys, in `.env`
  (not committed).
- GPUs are leased through landlord before any run.

See `docs/references.md` for links and the details behind the numbers above.
