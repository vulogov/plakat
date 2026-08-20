#!/usr/bin/env bash
#
# QUALITY comprehensive driver (RFC QUALITY-1 + QUALITY-2, plakat 6.10–6.11) — EVERY quality feature.
#
# `naturalize` reduces the "AI-generated" fingerprint of an image. The analog half (presets, content
# focuses, ghost-signature removal, AI-tell score + `rank --ai-tells`) is weight-free — no GPU. The
# corrective focuses (geometry / anatomy / no-twins), the `generate` integration (`--quality`,
# `--naturalize`), the 6.11 hi-res fix (`--hires`), and `--keep-best --ai-tells` need a model; run with
# RENDER=1 to include them. All outputs land under corpus/images/quality/.
#
# Usage:   corpus/naturalize_run.sh                 # weight-free demos (no GPU)
#          RENDER=1 corpus/naturalize_run.sh        # + the model-backed demos
#          PLAKAT=./target/debug/plakat SRC=my.png corpus/naturalize_run.sh
#
# Build the RELEASE binary first for the model-backed part: cargo build --release --features metal
set -u

PLAKAT="${PLAKAT:-./target/release/plakat}"
OUT="corpus/images/quality"
SRC="${SRC:-$OUT/forest_before.png}"
DEVICE="${DEVICE:-auto}"
RENDER="${RENDER:-0}"

mkdir -p "$OUT"
run() { echo "+ $*"; "$@" || echo "  ! step failed — continuing"; echo; }

echo "==================== QUALITY-1 comprehensive driver ===================="
echo "binary : $PLAKAT"
echo "src    : $SRC"
echo "out    : $OUT/    (RENDER=$RENDER)"
echo

# ---- 1. Presets (weight-free): subtle | photo | painting. Each prints its AI-tell score. ----
run "$PLAKAT" naturalize "$SRC" --out "$OUT/preset_subtle.png"   --preset subtle
run "$PLAKAT" naturalize "$SRC" --out "$OUT/preset_photo.png"    --preset photo
run "$PLAKAT" naturalize "$SRC" --out "$OUT/preset_painting.png" --preset painting

# ---- 2. Every analog content focus (weight-free): pre-tune the pass to a subject's tell. ----
for f in people sky vegetation cityscape landscape sea river mechanics household; do
  run "$PLAKAT" naturalize "$SRC" --out "$OUT/focus_${f}.png" --"$f" 1
done

# ---- 3. Combined focuses (weight-free): qualifiers stack. ----
run "$PLAKAT" naturalize "$SRC" --out "$OUT/combo_forest.png"   --preset photo --vegetation 1 --sky 1
run "$PLAKAT" naturalize "$SRC" --out "$OUT/combo_portrait.png" --preset photo --people 1.5 --household 0.5

# ---- 4. Manual per-stage tuning (weight-free): every analog knob. ----
run "$PLAKAT" naturalize "$SRC" --out "$OUT/manual.png" \
  --grain 0.3 --aberration 0.2 --vignette 0.1 --bloom 0.1 --desaturate 0.15 --warm 0.05 --defocus 0.05

# ---- 5. Ghost-signature removal (weight-free): dissolve a foreign corner smudge (keeps plakat etch). ----
run "$PLAKAT" naturalize "$SRC" --out "$OUT/designature_br.png" --designature br --preset photo

# ---- 6. Clean output (weight-free): drop plakat provenance instead of carrying it. ----
run "$PLAKAT" naturalize "$SRC" --out "$OUT/clean_noreetch.png" --preset photo --no-reetch

# ---- 6b. AI-tell ranking (weight-free, 6.11): least-AI-looking first over everything produced above. ----
run "$PLAKAT" rank "$OUT" --ai-tells

# ---- 7. Model-backed features (RENDER=1). ----
if [ "$RENDER" = "1" ]; then
  # 7a. Corrective focuses — img2img / inpaint fix what grain can't (structure / anatomy / lookalikes).
  run "$PLAKAT" naturalize "$SRC" --out "$OUT/corrective_geometry.png" --geometry 1 --device "$DEVICE"
  run "$PLAKAT" naturalize "$SRC" --out "$OUT/corrective_anatomy.png"  --anatomy 1  --device "$DEVICE"
  run "$PLAKAT" naturalize "$SRC" --out "$OUT/corrective_notwins.png"  --no-twins 1 --device "$DEVICE"
  # 7b. generate integration — the --quality preset (implies --hires 1.5) and the --naturalize post-pass.
  run "$PLAKAT" generate "an autumn forest path, soft light" --out "$OUT/gen_quality"    --quality high --device "$DEVICE"
  run "$PLAKAT" generate "an autumn forest path, soft light" --out "$OUT/gen_naturalize" --naturalize "photo vegetation=1 sky=0.5" --device "$DEVICE"
  # 7c. Hi-res fix (6.11) — explicit --hires factor: inject real coherent detail after generation.
  run "$PLAKAT" generate "an autumn forest path, soft light" --out "$OUT/gen_hires" --hires 2 --device "$DEVICE"
  # 7d. AI-tell selection (6.11) — keep the most human-looking of a batch (aesthetic − λ·ai_tell).
  run "$PLAKAT" generate "an autumn forest path, soft light" --out "$OUT/gen_keepbest" --count 4 --keep-best 2 --ai-tells --device "$DEVICE"
else
  echo "· RENDER=0 — skipping model-backed features (corrective geometry/anatomy/no-twins · generate --quality/--naturalize/--hires · --keep-best --ai-tells)."
  echo "  run 'RENDER=1 corpus/naturalize_run.sh' with a release build to include them."
fi

echo "==================== done → $OUT/ ===================="
