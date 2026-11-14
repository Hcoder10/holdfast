#!/usr/bin/env bash
# Set up LingBot-World 2 on a fresh GPU box. Idempotent: safe to re-run.
# Logs and markers go to $OUT so progress can be checked from outside.
set -euo pipefail

OUT=${OUT:-/root/worldhold_out}
WORK=${WORK:-/workspace/w}
LINGBOT_COMMIT=1895d30
mkdir -p "$OUT" "$WORK"
cd "$WORK"
step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/setup_steps.log"; }

export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60

step "tools"
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git)
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) || true
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

step "repos"
[ -d lingbot-world-v2 ] || git clone -q https://github.com/Robbyant/lingbot-world-v2.git
git -C lingbot-world-v2 checkout -q "$LINGBOT_COMMIT"
[ -d PlayWorld ] || git clone -q https://github.com/kxding/PlayWorld.git

step "venv"
[ -x venv/bin/python ] || uv venv -q --python 3.11 venv
PY="$WORK/venv/bin/python"
uv pip install -q --python "$PY" torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
grep -v -i '^flash_attn' lingbot-world-v2/requirements.txt > "$OUT/requirements.noflash.txt"
uv pip install -q --python "$PY" -r "$OUT/requirements.noflash.txt" \
  einops "huggingface_hub[cli]" hf_transfer safetensors pillow

step "torch check"
"$PY" - <<'EOF' | tee -a "$OUT/setup_steps.log"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "archs", torch.cuda.get_arch_list())
print("gpu", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GB")
x = torch.randn(1024, 1024, device="cuda")
print("matmul ok", float((x @ x).abs().mean()) > 0)
EOF

step "flash-attn (optional; the runner falls back to SDPA without it)"
if uv pip install -q --python "$PY" flash-attn --no-build-isolation >"$OUT/flash_install.log" 2>&1 && \
   "$PY" - <<'EOF' >>"$OUT/flash_install.log" 2>&1
import torch, flash_attn
q = torch.randn(2, 16, 4, 32, device="cuda", dtype=torch.bfloat16)
flash_attn.flash_attn_func(q, q, q)
EOF
then echo yes > "$OUT/FLASH_OK"; step "flash-attn works"
else rm -f "$OUT/FLASH_OK"; uv pip uninstall -q --python "$PY" flash-attn || true; step "flash-attn unavailable"
fi

HF="$WORK/venv/bin/hf"
[ -x "$HF" ] || HF="$WORK/venv/bin/huggingface-cli"

step "weights: shared assets (text encoder, VAE, tokenizer)"
"$HF" download robbyant/lingbot-world-v2-14b-causal-fast --exclude "transformers/*" \
  --local-dir "$WORK/lingbot-world-v2-14b-causal-fast" >/dev/null
step "weights: 1.3B"
"$HF" download robbyant/lingbot-world-v2-1.3b-causal-fast \
  --local-dir "$WORK/lingbot-world-v2-1.3b-causal-fast/transformers" >/dev/null
touch "$OUT/SMALL_READY"
step "weights: 14B"
"$HF" download robbyant/lingbot-world-v2-14b-causal-fast \
  --local-dir "$WORK/lingbot-world-v2-14b-causal-fast" >/dev/null
touch "$OUT/BIG_READY"

step "playworld data"
"$HF" download jocelynd/playworld-bench --repo-type dataset --local-dir "$WORK/PlayWorld/data" >/dev/null

df -h "$WORK" | tail -1 | tee -a "$OUT/setup_steps.log"
touch "$OUT/SETUP_DONE"
step "done"
