#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — compose with INLINE layer sources (v1.1)
# ===================================================================
# `plakat compose` layers used to be `load`-only (pre-made assets). Now a layer's
# pixels can also come from:
#   * generate: "<prompt>"  — rendered inline (t2i; optional model/seed/steps/gen_size)
#   * matte:    "<image>"   — a raw photo's subject cut out on the fly (U2Net)
# This scene builds a beach backdrop with `generate:` and drops a `matte:`d
# astronaut onto it — neither layer exists on disk beforehand.
#
# Light: sd15 @ 512² + U2Net matte — runs even on CPU (no SD memory wall).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

"$PLAKAT" compose "$ROOT/corpus/compose_generate_scene.hjson"

echo "✓ compose generate:/matte: → corpus/images/compose/beach-generate-matte.png"
