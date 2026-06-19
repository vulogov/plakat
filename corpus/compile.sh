#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — compile (prose prompts.txt → scenario HJSON)
# ===================================================================
# `plakat compile` turns a natural-language prompts.txt (paragraphs + optional
# `key: value` commands) into a ready-to-run scenario HJSON: one task per block,
# global→scene inheritance, model-family-aware prompt profiles, auto-negatives.
#
# This proof runs the DETERMINISTIC path (--no-enhance --no-negative → no LLM),
# so the committed basic.hjson is byte-stable, then validates it round-trips
# through the same parser `plakat scenario` uses. No GPU, no network, no API key.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

# 1) Compile prose → scenario HJSON, deterministically (no LLM calls).
"$PLAKAT" compile "$ROOT/corpus/compile/basic.txt" \
  --no-enhance --no-negative --out "$ROOT/corpus/compile/basic.hjson"

# 2) Validate the compiled scenario parses + dry-runs (2 tasks, per-task seed/count).
"$PLAKAT" scenario "$ROOT/corpus/compile/basic.hjson" --dry-run >/dev/null

# 3) And the pipe form (compile → scenario over stdin) works end to end.
"$PLAKAT" compile "$ROOT/corpus/compile/basic.txt" --no-enhance --no-negative --out - \
  | "$PLAKAT" scenario - --dry-run >/dev/null

echo "✓ compile: basic.txt → basic.hjson (2 tasks; validated via scenario --dry-run + pipe)"
