#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — IC-Light relighting (`plakat relight`)
# ===================================================================
#   ./corpus/relight.sh
#   plakat gallery corpus/images --recursive --out corpus/GALLERY.md
#
# IC-Light ("Imposing Consistent Light"): relight a foreground SUBJECT under
# a lighting described in text. SD 1.5-based — the UNet input conv is widened
# 4→8 channels and the lllyasviel/ic-light offset is merged over a base SD 1.5
# UNet; the matted subject latent is concatenated to the noisy latent each
# denoise step. The subject's identity is preserved while the light + scene
# change to match the prompt.
#
# A reused subject (the cached sage-captain portrait) is relit under three
# lighting moods. GPU SD 1.5 output is non-deterministic across runs/backends,
# so these are committed SHOWCASES (a gallery row), not byte-checks — the same
# convention as the other SD corpus drivers.
#
# Build with a GPU backend for speed: `cargo build --release --features metal`
# (Apple Silicon) / `--features cuda` (NVIDIA). SD 1.5 fits comfortably in 24 GB.
#
# Override with env vars: PLAKAT, SUBJECT, SIZE, STEPS, GUIDANCE.
# ===================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
SUBJECT="${SUBJECT:-$ROOT/corpus/images/portrait/sage_captain/plakat-portrait-43.png}"
OUT="$ROOT/corpus/images/relight"
SIZE="${SIZE:-512}"
STEPS="${STEPS:-25}"
GUIDANCE="${GUIDANCE:-2.0}"   # IC-Light wants LOW cfg (1.5–3); higher washes the subject out
mkdir -p "$OUT"

relight() { # <name> <seed> <prompt>
  "$PLAKAT" relight "$SUBJECT" \
    --prompt "$3" --negative "flat lighting, washed out" \
    --size "$SIZE" --steps "$STEPS" --guidance "$GUIDANCE" --seed "$2" \
    --out "$OUT/$1.png"
}

# Three lighting moods over the same subject — the relight should keep the
# captain's face/beard/garb while transforming the light + backdrop.
relight sunset      42 "dramatic sunset light from the left, warm golden rim light, cinematic, magic hour"
relight blue_hour    7 "cool blue twilight, soft moonlight from the right, misty, cinematic"
relight hearth      19 "warm orange firelight from below, cozy tavern interior glow, flickering hearth"

echo "✓ IC-Light relight: 3 moods (sunset / blue_hour / hearth) → corpus/images/relight/"
echo "  add to the gallery:  plakat gallery corpus/images --recursive --out corpus/GALLERY.md"
