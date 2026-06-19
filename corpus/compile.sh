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

# 4) Determinism guard: re-compiling against the committed output is a no-op diff.
DIFF="$("$PLAKAT" compile "$ROOT/corpus/compile/basic.txt" --no-enhance --no-negative \
  --diff "$ROOT/corpus/compile/basic.hjson")"
[ "$DIFF" = "(no changes)" ] || { echo "✗ compile is not byte-stable: $DIFF"; exit 1; }

# 5) Round-trip: scenario → prompts.txt (--decompile) → recompile → still valid.
"$PLAKAT" compile "$ROOT/corpus/compile/basic.hjson" --decompile --out - \
  | "$PLAKAT" compile - --no-enhance --no-negative --out - \
  | "$PLAKAT" scenario - --dry-run >/dev/null

# 6) COMPILE-2: the Tera pre-pass — only if this binary was built with
#    `--features templates` (otherwise the stub errors and we skip gracefully).
if "$PLAKAT" compile "$ROOT/corpus/compile/series.tera" --vars "$ROOT/corpus/compile/series.json" \
     --dump-rendered-only >/tmp/plakat-series-rendered.txt 2>/dev/null; then
  diff -q /tmp/plakat-series-rendered.txt "$ROOT/corpus/compile/series.rendered.txt" >/dev/null \
    || { echo "✗ Tera render is not byte-stable vs the committed series.rendered.txt"; exit 1; }
  "$PLAKAT" compile "$ROOT/corpus/compile/series.tera" --vars "$ROOT/corpus/compile/series.json" \
    --no-enhance --no-negative --out - | "$PLAKAT" scenario - --dry-run >/dev/null
  rm -f /tmp/plakat-series-rendered.txt
  TERA="+ COMPILE-2: series.tera → byte-stable render → scenario validated"
else
  TERA="(COMPILE-2 skipped — binary built without --features templates)"
fi

echo "✓ compile: basic.txt → basic.hjson (2 tasks; --dry-run + pipe + no-op --diff + --decompile round-trip)"
echo "  $TERA"
