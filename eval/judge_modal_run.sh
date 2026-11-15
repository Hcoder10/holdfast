#!/usr/bin/env bash
# Score a Modal run with the real judge: export PlayWorld judge inputs, fetch, blind, judge, collect.
#   bash eval/judge_modal_run.sh <run name on the worldhold-runs volume> <blind seed> "<ledger note>"
set -euo pipefail
RUN=$1; SEED=${2:-0}; NOTE=${3:-$RUN}
export PYTHONIOENCODING=utf-8
modal run modal_app/wh_modal.py::judge_inputs --run "$RUN" > "runs/modal_logs/judge_$RUN.log" 2>&1
rm -rf "runs/judge_$RUN" "runs/judge_${RUN}_blind"
modal volume get worldhold-runs "judge/$RUN" "runs/judge_$RUN" --force > /dev/null 2>&1
python -I eval/blind.py "runs/judge_$RUN/$RUN" "runs/judge_${RUN}_blind" "$SEED"
python -I eval/openrouter_judge.py --inputs "runs/judge_${RUN}_blind" --out "runs/gem_$RUN" --workers 8 | grep -c " ok" || true
python -I eval/judge_collect.py --judged "runs/gem_$RUN" --playworld third_party/PlayWorld \
  --judge "gemini-3.1-pro-preview via OpenRouter, PlayWorld prompts, 1 pass, blinded" \
  --player "claude-haiku-4-5 agent player (segment-boundary consults)" \
  --map "runs/judge_${RUN}_blind/blind_map.json" --ledger results/ledger.jsonl --note "$NOTE"
