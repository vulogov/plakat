#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — compose (layered scene)
# ===================================================================
# `plakat compose` stacks image layers (z-order = array order) onto a canvas:
# a landscape background + RGBA cut-outs placed by 9-grid position, scaled, and
# alpha-composited. NO GPU — it composes existing committed assets, so it runs
# anywhere in seconds. (generate: / inline matte: layers are a GPU follow-up.)
# Output: corpus/images/compose/valley-scene.png.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

"$PLAKAT" compose "$ROOT/corpus/compose_scene.hjson"

echo "✓ composed → corpus/images/compose/valley-scene.png"
