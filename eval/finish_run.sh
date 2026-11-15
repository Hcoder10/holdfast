#!/usr/bin/env bash
# Wait for a Modal run to finish, then score it with the real judge and append to the ledger.
#   bash eval/finish_run.sh <run> <cases expected per mode> <blind seed> "<ledger note>" <mode> [<mode> ...]
RUN=$1; N=$2; SEED=$3; NOTE=$4; shift 4
export PYTHONIOENCODING=utf-8
count() { modal volume ls worldhold-runs "$RUN/lingbot2_14b_$1" 2>/dev/null | tr -cd '\11\12\15\40-\176' | grep -c "$RUN/"; }
for i in $(seq 1 60); do
  done_all=1
  for m in "$@"; do [ "$(count "$m")" -ge "$N" ] || done_all=0; done
  # a case folder appears before its result; give stragglers time, and stop waiting if the app is gone
  [ $done_all = 1 ] && sleep 240 && break
  modal app list 2>/dev/null | tr -cd '\11\12\15\40-\176' | grep worldhold | grep -qv stopped || { echo "app no longer running"; break; }
  sleep 120
done
for m in "$@"; do echo "$m: $(count "$m") case folders"; done
modal run modal_app/wh_modal.py::judge_inputs --run "$RUN" > "runs/modal_logs/judge_$RUN.log" 2>&1
mkdir -p "runs/jin_$RUN"
for m in "$@"; do modal volume get worldhold-runs "judge/$RUN/lingbot2_14b_$m" "runs/jin_$RUN/" > /dev/null 2>&1; done
echo "judge inputs: $(find "runs/jin_$RUN" -name score_prompt.txt | wc -l)"
python -I eval/blind.py "runs/jin_$RUN" "runs/jblind_$RUN" "$SEED"
python -I eval/openrouter_judge.py --inputs "runs/jblind_$RUN" --out "runs/gem_$RUN" --workers 8 | grep -c " ok"
python -I eval/openrouter_judge.py --inputs "runs/jblind_$RUN" --out "runs/gem_$RUN" --workers 4 | grep -c "FAILED"
python -I eval/judge_collect.py --judged "runs/gem_$RUN" --playworld third_party/PlayWorld \
  --judge "gemini-3.1-pro-preview via OpenRouter, PlayWorld prompts, 1 pass, blinded" \
  --player "claude-haiku-4-5 agent player (segment-boundary consults)" \
  --map "runs/jblind_$RUN/blind_map.json" --ledger results/ledger.jsonl --note "$NOTE" > "runs/gem_$RUN/collect.txt"
tail -3 "runs/gem_$RUN/collect.txt"
