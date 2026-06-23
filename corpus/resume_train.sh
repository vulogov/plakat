#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — verify --resume + complete training
# ===================================================================
# Trains a SHORT run with numbered checkpoints, then RESUMES from a checkpoint
# (simulating an interrupted run) and continues — proving the resumed LoRA keeps
# learning and renders. Works on any base; sd15 is the fastest to verify.
#
#   Usage:  ./resume_train.sh [sd15|sdxl|sd35]      (default: sd15)
#
# GPU + a few minutes. The trained LoRA goes to a scratch dir (not committed);
# the render lands in corpus/images/train/ as the committed proof. ⬜ until run.
#
# What to look for in the logs:
#   • run 2 prints "resuming from …-step10.safetensors at step 10/30"
#   • the loss continues DOWN from where run 1 left off (not reset)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
BASE="${1:-sd15}"
STYLE="$ROOT/corpus/style/watercolour"
TRIG="wcstyle watercolour painting illustration"
WORK="${WORK:-/tmp/plakat-resume-demo}"; mkdir -p "$WORK"
LORA="$WORK/resume-demo.safetensors"

case "$BASE" in
  # 256² (not 512²): this proof verifies the --resume MECHANISM, not max style
  # quality — and full SD-UNet back-prop at 512² rank 32 spikes ~20 GB (the
  # autograd graph), which thrashes/aborts on a 24 GB box. 256² rank 16 trains
  # comfortably + is reproducible anywhere. Override with SIZE=512 on a big box.
  sd15) SIZE="${SIZE:-256}"; LR=2e-4;   RANK="${RANK:-16}" ;;
  sdxl) SIZE="${SIZE:-512}"; LR=1.5e-4; RANK="${RANK:-16}" ;;
  sd35) SIZE="${SIZE:-256}"; LR=1.5e-4; RANK="${RANK:-16}" ;;
  *) echo "base must be sd15 | sdxl | sd35"; exit 1 ;;
esac
COMMON=(--from-dir "$STYLE" --base "$BASE" --trigger "$TRIG" \
        --rank "$RANK" --size "$SIZE" --lr "$LR" --checkpoint-every 10)

# 1) Initial run: 20 steps → writes resume-demo-step10.safetensors + final (step 20).
"$PLAKAT" style train "${COMMON[@]}" --out "$LORA" --steps 20

# 2) RESUME from the step-10 checkpoint and continue to 30 (so 20 more steps).
#    --resume loads those weights back into the adapters and continues the counter.
"$PLAKAT" style train "${COMMON[@]}" --out "$LORA" --steps 30 \
  --resume "$WORK/resume-demo-step10.safetensors"

# 3) Render with the resumed LoRA — proves the continued training produced a
#    usable LoRA (the watercolour style should be visible). Explicit 512² — the
#    default render size spikes Metal's single-buffer allocation on SD 1.5 and
#    can OOM a 24 GB box (the known large-SD1.5-on-Metal gotcha); 512² is modest.
"$PLAKAT" generate "$TRIG, a quiet mountain village beside a lake at dawn" \
  --model "$BASE" --lora "$LORA" --seed 42 --steps 28 --size "${RENDER_SIZE:-512x512}" \
  --out "$ROOT/corpus/images/train"

echo "✓ resume verified ($BASE): trained 0→20, resumed 10→30, rendered → corpus/images/train/"
