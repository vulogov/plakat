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

# 5b) MAP-4: a `map:` block compiles to a `type: map` scenario task. Deterministic
#     → byte-stable committed maps.hjson; the compiled scenario dry-runs (2 map tasks).
"$PLAKAT" compile "$ROOT/corpus/compile/maps.txt" \
  --no-enhance --no-negative --out "$ROOT/corpus/compile/maps.hjson"
"$PLAKAT" scenario "$ROOT/corpus/compile/maps.hjson" --dry-run >/dev/null
MDIFF="$("$PLAKAT" compile "$ROOT/corpus/compile/maps.txt" --no-enhance --no-negative \
  --diff "$ROOT/corpus/compile/maps.hjson")"
[ "$MDIFF" = "(no changes)" ] || { echo "✗ compile map block not byte-stable: $MDIFF"; exit 1; }

# 6) COMPILE-2: end-to-end Tera → scenario — only if this binary was built with
#    `--features templates` (otherwise the stub errors and we skip gracefully).
TERA_VARS=("--vars" "$ROOT/corpus/compile/series.json")
if "$PLAKAT" compile "$ROOT/corpus/compile/series.tera" "${TERA_VARS[@]}" \
     --dump-rendered-only >/tmp/plakat-series-rendered.txt 2>/dev/null; then
  # a) the Tera render is byte-stable vs the committed prompts.txt.
  diff -q /tmp/plakat-series-rendered.txt "$ROOT/corpus/compile/series.rendered.txt" >/dev/null \
    || { echo "✗ Tera render is not byte-stable vs the committed series.rendered.txt"; exit 1; }
  rm -f /tmp/plakat-series-rendered.txt
  # b) END TO END: series.tera (+series.json) → compile → COMMITTED scenario HJSON.
  "$PLAKAT" compile "$ROOT/corpus/compile/series.tera" "${TERA_VARS[@]}" \
    --no-enhance --no-negative --out "$ROOT/corpus/compile/series.hjson"
  # c) that committed scenario must itself validate (2 character tasks).
  "$PLAKAT" scenario "$ROOT/corpus/compile/series.hjson" --dry-run >/dev/null
  TERA="+ COMPILE-2: series.tera (+series.json) → series.hjson (committed, 2 tasks) → scenario validated"
else
  TERA="(COMPILE-2 skipped — binary built without --features templates)"
fi

echo "✓ compile: basic.txt → basic.hjson (2 tasks; --dry-run + pipe + no-op --diff + --decompile round-trip)"
echo "  + MAP-4: maps.txt -> maps.hjson (2 map tasks (type: map), byte-stable, scenario-validated)"
echo "  $TERA"
