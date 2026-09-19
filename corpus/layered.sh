#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — layered generation (RFC LAYERED-1)
# ===================================================================
# `plakat layers` builds a complex, multi-subject image from a PLAN: a backdrop
# plus independent subject layers, each with its own full-detail prompt and a
# box. Each layer is drafted ALONE (binding is trivial with one subject), the
# drafts become a low-frequency GUIDE, and the finish is ONE anchored denoising
# trajectory of the finish model — steered toward the guide's low frequencies
# inside each subject's box during the early steps. Layers are constraints on
# layout, never pixels copied to the output.
#
# The DEFAULT run is offline + deterministic (no GPU, no network): it lints the
# committed plan and draws the layer-box diagram. The GPU stages (planner LLM,
# and the draft -> guide -> finish render + verify + diff) are gated behind env
# flags so this proof stays CI-safe.
#
#   LAYERED_PLAN=1     also decompose prose -> a plan via an LLM (downloads a model)
#   LAYERED_PROVIDER=  planner provider, e.g. ollama:qwen2.5-coder:14b (else local GGUF)
#   LAYERED_RENDER=1   also run the full layered render + verify + diff (GPU + models)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
PLAN="$ROOT/corpus/layered/night-market.hjson"
OUT="$ROOT/corpus/images/layered"
mkdir -p "$OUT"

# 1) Lint the plan — schema + the per-layer SIZE CLASS for the finish model.
#    Big boxes are `anchored` (guide + verify + repair); mid boxes `hinted`
#    (prompt-named, verified); tiny boxes `lifted` (structure via the S5 lift).
"$PLAKAT" layers lint "$PLAN" --model sdxl

# 2) Show what the plan RESOLVES to (per-layer box / depth / class) and draw the
#    boxes, coloured by class, onto a blank canvas. Pure geometry — no GPU.
"$PLAKAT" layers show "$PLAN" --model sdxl --boxes "$OUT/night-market-plan.png"

# 3) (optional) Author a plan from PROSE with an LLM. `--provider ollama:<model>`
#    uses a local Ollama server (a bigger model than the in-process GGUF aliases);
#    omit it for the default local GGUF. Then lint the generated plan.
if [ "${LAYERED_PLAN:-0}" = 1 ]; then
  "$PLAKAT" layers plan \
    "a lively night market lane: a food vendor at a steaming cart, a customer in a red coat ordering, a glowing red paper lantern above, warm string lights and soft fog, cinematic poster" \
    -o "$OUT/planned.hjson" --model sdxl ${LAYERED_PROVIDER:+--provider "$LAYERED_PROVIDER"}
  "$PLAKAT" layers lint "$OUT/planned.hjson" --model sdxl
fi

# 4) (optional) The full pipeline: S1 drafts -> S2 guide -> S3 anchored finish.
#    A real diffusion job. `--keep` also writes the drafts, the composed guide,
#    and the weight/window anchor maps for inspection.
if [ "${LAYERED_RENDER:-0}" = 1 ]; then
  "$PLAKAT" layers render "$PLAN" -o "$OUT/night-market.png" --model sdxl \
    --draft-model sdxl --draft-steps 8 --steps 36 --guidance 6.5 --ramp 0.2 --seed 11 --keep "$OUT/stages"

  # S4 verify: did each anchored subject render inside its box? (OWL-ViT; no diffusion.)
  "$PLAKAT" layers verify "$PLAN" --image "$OUT/night-market.png" --model sdxl || true

  # How well did the finish track the guide's low-frequency layout? (Model-free.)
  "$PLAKAT" layers diff "$PLAN" --a "$OUT/stages/__guide.png" --b "$OUT/night-market.png" \
    --model sdxl -o "$OUT/guide-vs-finish.png"
fi

echo "✓ layered demo complete → $OUT"
