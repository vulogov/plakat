#!/usr/bin/env bash
#
# PERSONA-1 feature-demonstration driver (RFC PERSONA-1, plakat 5.0.0).
#
# Runs the whole persona pipeline over the corpus specs (mira, idris) on two families — sd15 and
# sd35 — and writes every artefact under corpus/out/. It exercises, in order:
#
#   lint · show · geometry (maps + calibration pre-distort) · interview replay · edit-diff        [weights-free]
#   cast (candidates → coherence-checked reference set) · render (Tier-B swap bridge) · verify
#   (scorecard) · composite (details) · repair (targeted in-place fix) · multiperson render        [weights]
#
# Weights-free steps always run. The per-model steps download the model on first use and render on
# the GPU (Metal) or CPU. sd35 is heavy — LOWMEM keeps the T5-XXL + MMDiT from being co-resident.
#
# Usage:   corpus/persona_run.sh                 # both personas, sd15 + sd35, release binary
#          MODELS="sd15" corpus/persona_run.sh   # one family
#          PLAKAT=./target/debug/plakat CAST_COUNT=3 STEPS=16 corpus/persona_run.sh
#
# Build the RELEASE binary first (debug inference is ~50x slower):  cargo build --release
set -u

PLAKAT="${PLAKAT:-./target/release/plakat}"
MODELS="${MODELS:-sd15 sd35}"
CAST_COUNT="${CAST_COUNT:-6}"
KEEP_BEST="${KEEP_BEST:-3}"
STEPS="${STEPS:-24}"
OUT="corpus/images/persona"

# The OOM guard reads macOS `free` pages (near-zero by design) and mis-fires under a tight render
# loop; disable it here. LOWMEM lets sd35 run on a 24 GB Metal box.
export PLAKAT_OOM_GUARD_GB="${PLAKAT_OOM_GUARD_GB:-0}"
export PLAKAT_SD3_LOWMEM="${PLAKAT_SD3_LOWMEM:-1}"

mkdir -p "$OUT"
run() { echo "+ $*"; "$@" || echo "  ! step failed — continuing"; echo; }

echo "==================== PERSONA-1 corpus driver ===================="
echo "binary : $PLAKAT"
echo "models : $MODELS"
echo "out    : $OUT/"
echo

# ---------------------------------------------------------------------------
# 1. Weights-free, deterministic steps (run once, not per-model).
# ---------------------------------------------------------------------------
echo "---- lint + geometry + interview + diff (no weights) ----"
for P in mira idris; do
  SPEC="corpus/$P.hjson"
  run "$PLAKAT" persona lint "$SPEC"
  # geometry maps, pre-distorted through the sdxl calibration curves (§13.2).
  run "$PLAKAT" persona geometry "$SPEC" --out "$OUT/$P/geometry" --calibrate sdxl --size 512
done

# interview: replay a scripted answer log into a spec (§17.12, reproducible).
run "$PLAKAT" persona interview "$OUT/mira-from-answers.hjson" --answers corpus/mira-answers.hjson

# edit-diff: an eye-colour change is a cheap surface repair; a jaw change forces a re-cast (§6.5).
sed -e 's/color: "hazel"/color: "green"/' -e 's/spacing: 0.58/spacing: 0.80/' corpus/mira.hjson > "$OUT/mira-edited.hjson"
run "$PLAKAT" persona diff corpus/mira.hjson "$OUT/mira-edited.hjson"

# ---------------------------------------------------------------------------
# 2. Per-model generative pipeline.
# ---------------------------------------------------------------------------
for M in $MODELS; do
  case "$M" in
    sd35) SZ=768 ;;   # DiT family; larger native size, heavier
    *)    SZ=512 ;;
  esac
  echo "======== model: $M (size ${SZ}) ========"
  for P in mira idris; do
    SPEC="corpus/$P.hjson"
    D="$OUT/$P/$M"
    mkdir -p "$D"

    # the compiled prompt for this family (encoder-class-aware).
    run "$PLAKAT" persona show "$SPEC" --model "$M"

    # cast: render candidates, composite details, score, keep the best, validate coherence.
    run "$PLAKAT" persona cast "$SPEC" --model "$M" --count "$CAST_COUNT" --keep-best "$KEEP_BEST" \
        --size "$SZ" --steps "$STEPS" --aesthetic --out "$D/cast"

    # render the persona into a scene (Tier-B swap → restore → detail composite).
    run "$PLAKAT" persona render "$D/cast" --model "$M" --size "$SZ" --steps "$STEPS" \
        --scene "a colour portrait photograph, natural light, natural skin tones, plain background" --out "$D/render.png"

    # measure the render against the spec (scorecard: geometric scalars, colour, details, detect).
    run "$PLAKAT" persona verify "$SPEC" --image "$D/render.png" --model "$M"

    # composite the persona's details directly onto the render (standalone §8.4 pass).
    run "$PLAKAT" persona composite "$SPEC" --image "$D/render.png" --mask "$D/detail_mask.png" \
        --out "$D/composited.png"

    # targeted in-place repair of a surface attribute, kept only on improvement (§12.4).
    run "$PLAKAT" persona repair "$SPEC" --image "$D/render.png" --attr eyes.color \
        --model "$M" --out "$D/repaired_eyes.png"
  done

  # multiperson: both personas in one scene, each swapped into its own attributed figure (§14.2).
  run "$PLAKAT" persona render "$OUT/mira/$M/cast" --with "$OUT/idris/$M/cast" --model "$M" \
      --size "$SZ" --steps "$STEPS" --scene "two people at a cafe table, colour photograph, natural light" \
      --out "$OUT/multiperson_$M.png"
done

echo "==================== done → $OUT/ ===================="
