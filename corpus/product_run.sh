#!/usr/bin/env bash
#
# PRODUCT-1 feature-demonstration driver (RFC PRODUCT-1, plakat 6.9).
#
# A subject → a studio product-shot: a sweep + a grounded contact shadow + floor reflection. The
# weight-free half (lint / show / render / sheet from a cutout) needs no GPU; `RENDER=1` also generates a
# subject from a prompt and relights a cutout to a rig (needs a model). Everything lands under
# corpus/images/product/.
#
# Usage:   corpus/product_run.sh                         # everything, release binary
#          RENDER=0 corpus/product_run.sh                # weight-free only (no GPU)
#          PLAKAT=./target/debug/plakat corpus/product_run.sh
#
# Build the RELEASE binary first (debug inference is ~50x slower): cargo build --release --features metal
set -u

PLAKAT="${PLAKAT:-./target/release/plakat}"
OUT="corpus/images/product"
RENDER="${RENDER:-1}"
export PLAKAT_OOM_GUARD_GB="${PLAKAT_OOM_GUARD_GB:-0}"

mkdir -p "$OUT"
run() { echo "+ $*"; "$@" || echo "  ! step failed — continuing"; echo; }

echo "==================== PRODUCT-1 corpus driver ===================="
echo "binary : $PLAKAT"
echo "out    : $OUT/"
echo

# 1. Weight-free: validate + show the resolved plan.
run "$PLAKAT" product lint corpus/product_bottle.hjson
run "$PLAKAT" product show corpus/product_bottle.hjson

# 2. Weight-free: a grounded packshot from a cutout (sweep + contact shadow + reflection).
run "$PLAKAT" product render corpus/product_bottle.hjson --out "$OUT/bottle.png"

# 3. Weight-free: a catalog contact sheet (main + a side angle, same rig).
run "$PLAKAT" product sheet corpus/product_catalog.hjson --out "$OUT/catalog_sheet.png"

# 4. The model half (needs a model): a subject generated from a prompt, matted + grounded.
if [ "$RENDER" = "1" ]; then
  cat > "$OUT/_gen.hjson" <<'EOF'
{ subject: { prompt: "a glass perfume bottle with a gold cap", scale: 0.6 },
  canvas: { px: 768, bg: "grey-sweep" }, ground: { shadow: "soft", reflection: "gloss" },
  model: "sdxl", seed: 5, steps: 24 }
EOF
  run "$PLAKAT" product render "$OUT/_gen.hjson" --out "$OUT/generated.png"
else
  echo "· RENDER=0 — skipping the model step (weight-free packshots are at $OUT/bottle.png + $OUT/catalog_sheet.png)"
fi

echo "==================== done → $OUT/ ===================="
