#!/usr/bin/env bash
# 07_scenario.sh — batch generation from the scenario.hjson.
#
# Produces several images using the HJSON form, which unlocks
# per-artefact overrides (offset, anchor, flip, alpha) that the CLI
# shorthand doesn't expose. See scenario.hjson for the configuration.
#
# Requires DEEPSEEK_API_KEY in the environment (scenarios always
# run task prompts through an LLM enhancer). To switch providers,
# edit `enhancer:` in scenario.hjson and export GEMINI_API_KEY
# instead.
#
# To validate the HJSON without generating images or calling the
# enhancer, append `--dry-run`:
#   plakat scenario scenario.hjson --dry-run
#
# Output: out/zones-tutorial/scenario/<task>/plakat-*.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

if [[ -z "${DEEPSEEK_API_KEY:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
    echo "warn: neither DEEPSEEK_API_KEY nor GEMINI_API_KEY is set." >&2
    echo "      Scenarios require one — export it before running. e.g.:" >&2
    echo "        export DEEPSEEK_API_KEY=sk-..." >&2
    exit 1
fi

"$PLAKAT" scenario scenario.hjson

echo
echo "Wrote outputs into out/zones-tutorial/scenario/"
echo "Each task has its own subdirectory."
