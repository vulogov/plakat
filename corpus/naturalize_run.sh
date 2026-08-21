#!/usr/bin/env bash
#
# QUALITY comprehensive driver (RFC QUALITY-1 + QUALITY-2, plakat 6.10–6.11) — EVERY quality feature.
#
# `naturalize` reduces the "AI-generated" fingerprint of an image. This driver runs on two PHOTOREAL
# samples (a portrait + a landscape) so the de-AI effect is actually visible and the subject-specific
# content focuses land on a matching subject:
#   - portrait_before.png — a "portrait photograph" with the classic waxy/too-smooth skin tell (--people),
#   - steppe_before.png   — a photoreal sunrise landscape, oversaturated sky + smooth clouds (--sky /
#                           --vegetation / --landscape).
# (For naturalizing AI ART/illustration rather than photos, see naturalize_art_run.sh.)
#
# The analog half (presets, content focuses, ghost-signature removal, AI-tell score + `rank --ai-tells`,
# and the etch provenance — fresh-etch + re-etch) is weight-free — no GPU. The corrective focuses
# (geometry / anatomy / no-twins), the `generate` integration (`--quality`, `--naturalize`), the hi-res
# fix (`--hires`), and `--keep-best --ai-tells` need a model; run with RENDER=1 to include them. All
# outputs land under corpus/images/quality/.
#
# Usage:   corpus/naturalize_run.sh                 # weight-free demos (no GPU)
#          RENDER=1 corpus/naturalize_run.sh        # + the model-backed demos
#          PLAKAT=./target/debug/plakat corpus/naturalize_run.sh
#
# Build the RELEASE binary first for the model-backed part: cargo build --release --features metal
set -u

PLAKAT="${PLAKAT:-./target/release/plakat}"
OUT="corpus/images/quality"
PORTRAIT="${PORTRAIT:-$OUT/portrait_before.png}"   # people / skin tell
SCENE="${SCENE:-$OUT/steppe_before.png}"           # sky / vegetation / landscape tells
DEVICE="${DEVICE:-auto}"
RENDER="${RENDER:-0}"

mkdir -p "$OUT"
run() { echo "+ $*"; "$@" || echo "  ! step failed — continuing"; echo; }

echo "==================== QUALITY comprehensive driver ===================="
echo "binary  : $PLAKAT"
echo "portrait: $PORTRAIT"
echo "scene   : $SCENE"
echo "out     : $OUT/    (RENDER=$RENDER)"
echo

# ---- 0. Baseline AI-tell of each photoreal input (weight-free ranking heuristic). ----
run "$PLAKAT" rank "$PORTRAIT" "$SCENE" --ai-tells

# ---- 1. Presets (weight-free) on the scene: subtle | photo | painting. Each prints its AI-tell score. ----
run "$PLAKAT" naturalize "$SCENE" --out "$OUT/preset_subtle.png"   --preset subtle
run "$PLAKAT" naturalize "$SCENE" --out "$OUT/preset_photo.png"    --preset photo
run "$PLAKAT" naturalize "$SCENE" --out "$OUT/preset_painting.png" --preset painting

# ---- 2. Content focuses (weight-free), each on a MATCHING subject: pre-tune the pass to a subject's tell. ----
run "$PLAKAT" naturalize "$PORTRAIT" --out "$OUT/focus_people.png"     --people 1.5      # waxy skin
run "$PLAKAT" naturalize "$SCENE"    --out "$OUT/focus_sky.png"        --sky 1           # gradient banding
run "$PLAKAT" naturalize "$SCENE"    --out "$OUT/focus_vegetation.png" --vegetation 1    # cloud-foliage mush
run "$PLAKAT" naturalize "$SCENE"    --out "$OUT/focus_landscape.png"  --landscape 1
# The remaining scenic focuses share the profile machinery (demoed on the scene for completeness):
for f in cityscape sea river mechanics household; do
  run "$PLAKAT" naturalize "$SCENE" --out "$OUT/focus_${f}.png" --"$f" 1
done

# ---- 3. Combined focuses (weight-free): qualifiers stack. ----
run "$PLAKAT" naturalize "$SCENE"    --out "$OUT/combo_scene.png"    --preset photo --vegetation 1 --sky 1
run "$PLAKAT" naturalize "$PORTRAIT" --out "$OUT/combo_portrait.png" --preset photo --people 1.5 --household 0.5

# ---- 4. Manual per-stage tuning (weight-free): every analog knob, strong enough to read the delta. ----
run "$PLAKAT" naturalize "$SCENE" --out "$OUT/manual.png" \
  --grain 0.4 --aberration 0.3 --vignette 0.2 --bloom 0.15 --desaturate 0.35 --warm 0.05 --defocus 0.05

# ---- 5. Ghost-signature removal (weight-free): dissolve a foreign corner smudge (keeps plakat etch). ----
run "$PLAKAT" naturalize "$SCENE" --out "$OUT/designature_br.png" --designature br --preset photo

# ---- 6. Quality-improvement isolation (weight-free): --polish alone vs. full pass. ----
# --polish is the "make a BETTER picture" pass (white balance + auto-levels + vibrance + unsharp); it runs
# even with every analog knob at 0, so you can see the pure correction.
run "$PLAKAT" naturalize "$SCENE" --out "$OUT/polish_only.png" \
  --polish 0.8 --grain 0 --aberration 0 --vignette 0 --bloom 0 --desaturate 0 --defocus 0 --micro 0
# Skin de-plasticiser (micro-texture): pores / micro-wrinkles added ONLY to the too-smooth regions.
run "$PLAKAT" naturalize "$PORTRAIT" --out "$OUT/skin_micro.png" --people 1.5 --micro 1
# NOTE ON ETCH: `--etch` is provenance (RFC ETCH-1), NOT a quality feature — its L1 pixel mark adds a
# faint DCT texture (a robustness tradeoff, visible on smooth gradients), so it's intentionally OFF here.
# Demonstrate it separately with `naturalize ... --etch` + `doctor --if-plakat` if you need provenance.

# ---- 7. AI-tell ranking (weight-free, 6.11): least-AI-looking first over everything produced above. ----
run "$PLAKAT" rank "$OUT" --ai-tells

# ---- 8. Model-backed features (RENDER=1). ----
if [ "$RENDER" = "1" ]; then
  # 8a. Corrective focuses — img2img / inpaint fix what grain can't (structure / anatomy / lookalikes).
  run "$PLAKAT" naturalize "$SCENE"    --out "$OUT/corrective_geometry.png" --geometry 1 --device "$DEVICE"
  run "$PLAKAT" naturalize "$PORTRAIT" --out "$OUT/corrective_anatomy.png"  --anatomy 1  --device "$DEVICE"
  run "$PLAKAT" naturalize "$PORTRAIT" --out "$OUT/corrective_notwins.png"  --no-twins 1 --device "$DEVICE"
  # 8b. generate integration — the --quality preset (implies --hires 1.5) and the --naturalize post-pass.
  run "$PLAKAT" generate "an autumn forest path, soft light" --out "$OUT/gen_quality"    --quality high --device "$DEVICE"
  run "$PLAKAT" generate "an autumn forest path, soft light" --out "$OUT/gen_naturalize" --naturalize "photo vegetation=1 sky=0.5" --device "$DEVICE"
  # 8c. Hi-res fix (6.11) — explicit --hires factor: inject real coherent detail after generation.
  run "$PLAKAT" generate "an autumn forest path, soft light" --out "$OUT/gen_hires" --hires 2 --device "$DEVICE"
  # 8d. AI-tell selection (6.11) — keep the most human-looking of a batch (aesthetic − λ·ai_tell).
  run "$PLAKAT" generate "an autumn forest path, soft light" --out "$OUT/gen_keepbest" --count 4 --keep-best 2 --ai-tells --device "$DEVICE"
else
  echo "· RENDER=0 — skipping model-backed features (corrective geometry/anatomy/no-twins · generate --quality/--naturalize/--hires · --keep-best --ai-tells)."
  echo "  run 'RENDER=1 corpus/naturalize_run.sh' with a release build to include them."
fi

echo "==================== done → $OUT/ ===================="
