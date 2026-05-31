# RFC v0.33 — Production polish bundle

**Status:** decisions locked 2026-05-31 — phase 0 in flight.

**Predecessors:**
- [`RFC_v0.32_ANIMATE_LITE_DIVERSIFY_3.md`](RFC_v0.32_ANIMATE_LITE_DIVERSIFY_3.md) — FreeNoise, vendored CLIP rollout, SDXL VAE cache.

## 1. TL;DR

After five diversify-flavored cycles (v0.28 / v0.29 productivity →
v0.30 / v0.31 / v0.32 carry-closure-driven), v0.33 picks a fresh
theme that nothing has touched in any cycle — **workflow polish for
users running plakat in production**. Four phases plus close-out:

1. **Better metadata fields.** Extend `GenerationMetadata` with
   structured LoRA / embedding / ControlNet stacks, look + genre
   preset names, prompt enhancement details. Additive-only —
   existing PNG tEXt sidecars from v0.32 must still parse.
2. **Improved error UX.** Actionable diagnostics for the top 5
   user-facing failures (OOM, gated repo 401, missing model,
   malformed prompt, scenario schema). Anyhow context decorators
   per failure mode.
3. **JSON sidecar mode.** `--json-summary` for scenarios writes
   a single structured-run JSON; `--json-sidecar` for `plakat
   generate` writes per-image structured metadata.
4. **Reproducibility audit.** Walk seed plumbing across every
   pipeline; document gaps via new `plakat doctor
   --reproducibility-check` mode.

Five phases including close-out, ~7-8 sessions. No carry closures
in v0.33 — the v0.32 carries defer to v0.34 to keep this cycle
thematically focused.

## 2. Why this is the v0.33 cycle

1. **Animate quality items can wait one more cycle.** v0.32
   shipped FreeNoise (the most user-visible). Remaining animate
   items (per-layer splice, HotShot-XL) are research-grade.
   AnimateLCM-SDXL is externally blocked.

2. **The biggest user-pulling work right now is workflow polish.**
   Plakat's surface is feature-complete for the core SD / Flux /
   SD3 + animate flows. The friction users hit is at the edges —
   what does this error mean, how do I script around the
   metadata, can I trust this generation is reproducible.

3. **Nothing has touched this surface in any cycle.** Fresh angle
   after five cycles of "close a deferral list item."

4. **Phase 0 metadata is foundational** for phase 2 JSON sidecar
   mode. Splitting these (metadata model first, JSON output
   second) keeps each phase tight.

## 3. Phase plan

### Phase 0 — Better metadata fields

**Goal:** richer structured metadata in `GenerationMetadata` so
downstream tooling (Civitai upload, sidecar inspection,
reproducibility audits) has the data it needs.

**New types (in `imaging::metadata`):**
- `LoraEntry { display, scale, source?, revision? }`
- `EmbeddingEntry { trigger, embed_dim, num_tokens, source? }`
- `ControlEntry { kind, image?, from?, video?, strength, start, end }`
- `EnhancementMetadata { provider, system_prompt_name?, cache_hit, original_prompt }`

**New fields on `GenerationMetadata`:**
- `look: Option<String>` — v0.25 `--look` name
- `genre: Option<String>` — v0.25 `--genre` name
- `negative_preset: Option<String>` — v0.19 `--negative-preset`
- `lora_stack: Option<Vec<LoraEntry>>` — structured companion to
  the existing flat `loras: Vec<String>`
- `embedding_stack: Option<Vec<EmbeddingEntry>>` — v0.30 TI details
- `embeddings: Vec<String>` — flat A1111-style mirror of triggers
- `control_stack: Option<Vec<ControlEntry>>` — structured companion
  to `controls: Vec<String>`
- `enhancement: Option<EnhancementMetadata>` — v0.19 details
- `free_noise: Option<bool>` — v0.32 phase 0 opt-in flag

**Schema compatibility constraint:** ADDITIVE only — every existing
PNG tEXt sidecar from v0.32 must still parse. Existing tests pin
the schema; extend their fixtures without removing or renaming.

The A1111 `parameters` string extends with:
- `Look: <name>` when present
- `Genre: <name>` when present
- `Enhancer: <provider>` when present
- `FreeNoise: on` when true

~2 sessions.

### Phase 1 — Improved error UX

Top 5 user-facing failure modes get actionable diagnostics:

1. **OOM (candle/CUDA/Metal)** — detect the error shape; append
   "try `--size 768x768` or `--quant-level Q4_K_S` (Flux)".
2. **Gated repo 401** — HF returns 401 → "set `HF_TOKEN` (see
   `plakat doctor` for current status)".
3. **Missing model alias** — `--model foo` doesn't resolve →
   excerpt of `plakat models aliases` showing closest match.
4. **Malformed prompt** — A1111 syntax errors point at the
   line/column of the bad token.
5. **Scenario schema** — HJSON parse errors point at the task
   name + the offending field.

Implementation: extend anyhow context chains with helpers per
failure mode. No new error type taxonomy.

~2 sessions.

### Phase 2 — JSON sidecar mode

- `plakat generate --json-sidecar` — writes `<image>.json` next to
  each PNG with the phase-0-extended `GenerationMetadata`.
- `plakat scenario --json-summary PATH` — single JSON at PATH
  with `ScenarioRunSummary` (per-task entries + aggregate stats).

`GenerationMetadata` already has `to_json_pretty()`; we wire the
flags + add `ScenarioRunSummary` as a new struct.

~2 sessions.

### Phase 3 — Reproducibility audit

`plakat doctor --reproducibility-check` walks every RNG-touching
code path and dumps a table:

| Pipeline | Code path | Determinism guarantee | Notes |
|---|---|---|---|

Surfaces gaps as warnings; deep fixes defer to v0.34.

~1-2 sessions.

### Phase 4 — Cycle close-out

Standard 7-step release.

~0.5 session.

## 4. Decisions locked

1. **Cycle shape: production polish bundle.** Four feature phases
   + close-out. Locked 2026-05-31.

## 5. What's NOT in v0.33

Deferred to v0.34+:
- Per-layer motion splice (RFC v0.27 §3.2).
- HotShot-XL integration.
- AnimateLCM-SDXL (externally blocked).
- INT8 SDXL UNet (blocked on candle quantized Conv2d).
- Plakat server mode (needs library API first).
- PixArt Sigma / Stable Cascade.
- v0.32 carries:
  - AnimateDiff load fn VAE cache passthrough.
  - Scripting `plakat.load` VAE cache.
  - Auto1111 two-separate-files SDXL TI convention.
- Deep fixes to determinism gaps surfaced by phase 3.

## 6. Related

- v0.18 phase 6 release notes — origin of `with_animate_lerp`
  metadata pattern that phase 0 extends.
- v0.25 release notes — origin of `--look` / `--genre` that phase
  0 captures in metadata.
- v0.30 phase 0 release notes — origin of embedding TI runtime
  that phase 0 records in `embedding_stack`.
- v0.32 phase 0 release notes — origin of `--free-noise` flag
  that phase 0 records.
