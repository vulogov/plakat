#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — SDXL tiled hi-res scripting driver
# ===================================================================
# Runs corpus/tiled_script.bund via `plakat run`, proving the v1.0
# `plakat.tiled.enable` word: SDXL renders at 1536² (above native 1024²) from
# 1024 tiles via the pipeline's `generate_tiled` denoise — the scripting
# counterpart of the CLI `--tiled`. Ungated SDXL, Metal-safe.
# Output: corpus/images/script/tiled-valley.png.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

mkdir -p "$ROOT/corpus/images/script"
cd "$ROOT"
"$PLAKAT" run corpus/tiled_script.bund

# plakat.save writes under the run's output dir (./out); relocate into the corpus.
mv -f "$ROOT/out/tiled-valley.png" "$ROOT/corpus/images/script/tiled-valley.png"
rmdir "$ROOT/out" 2>/dev/null || true

echo "✓ wrote corpus/images/script/tiled-valley.png (1536² SDXL, tiled)"
