#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — Bund scripting driver
# ===================================================================
# Runs the script.bund chain (load -> generate -> upscale -> save) via
# `plakat run`, proving the stack-based scripting surface. Ungated SD 1.5,
# Metal-safe. Output: corpus/images/script/fox-2x.png.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

mkdir -p "$ROOT/corpus/images/script"
cd "$ROOT"
"$PLAKAT" run corpus/script.bund

# plakat.save writes under the run's output dir (./out); relocate the
# result into the corpus tree.
mv -f "$ROOT/out/fox-2x.png" "$ROOT/corpus/images/script/fox-2x.png"
rmdir "$ROOT/out" 2>/dev/null || true

echo "✓ wrote corpus/images/script/fox-2x.png"
