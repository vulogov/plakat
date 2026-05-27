# RFC v0.25 — Looks & Genres (art-style presets)

**Status:** decisions locked 2026-05-27 — ready for phase 0.

**Predecessors:**
- [`RFC_v0.21_BUND_SCRIPTING.md`](RFC_v0.21_BUND_SCRIPTING.md) — the 7-word MVP.
- [`RFC_v0.22_BUND_WORDS_EXPANSION.md`](RFC_v0.22_BUND_WORDS_EXPANSION.md) — the 28-word expansion.
- [`RFC_v0.23_BUND_DEFERRALS.md`](RFC_v0.23_BUND_DEFERRALS.md) — six v0.22 deferrals closed.
- [`RFC_v0.24_SCRIPT_SURFACE_COMPLETION.md`](RFC_v0.24_SCRIPT_SURFACE_COMPLETION.md) — persona depth + script surface completion.

## 1. TL;DR

plakat is heavily used to generate computer-rendered art — ink
wash, watercolor, oil painting, charcoal, anime, chalk, pencil
work, linocut, gouache. Today users hand-craft prompt prefix +
LoRA + scheduler + steps + negative_extras for every run. v0.25
ships two independent **preset axes** that bundle these tuned
parameters into reusable, override-only modifiers:

- **`--look`** — the *medium*. `ink-wash`, `watercolor`,
  `oil-painting`, `charcoal`, `pencil`, `chalk-pastel`, `linocut`,
  `gouache` (8 built-in). Each carries a prompt prefix/suffix,
  recommended sampler/steps/guidance, negative_extras, and a
  `lora_query` that drives automatic LoRA discovery.
- **`--genre`** — the *subject domain*. `anime` built-in;
  user-extensible via `$CONFIG_DIR/genres/*.json`.

Both axes are **optional**, **compose** with each other and with
existing `--style` / `--fast` / `--negative-preset` / `--lora`,
and follow the v0.14 fast-preset rule: override only what the
user didn't already pass.

Net surface: 42 → ~48 host words (`plakat.look.*` + `plakat.genre.*`);
new CLI flags on six subcommands; new scenario fields; full
docs/tutorial pass.

## 2. Why this is the v0.25 cycle

1. **User need.** plakat's primary art use case is "generate me a
   watercolor of X" — and today every such command is a 10-flag
   incantation. The look/genre axes condense the common case to a
   single flag while leaving the existing flag surface intact for
   power users.

2. **The v0.23 style system is detection-flavored.** CLIP-H
   exemplar matching, confidence margins, top3-mean aggregation —
   built for "detect what style this reference photo is in." A
   *prescriptive* selector ("I want watercolor, period") needs a
   different abstraction. The v0.14 `--fast` preset pattern is
   the right template: kebab-case name → bundled tuple → override
   only what the user didn't pass.

3. **Scripting layer is done.** v0.21/22/23/24 closed the
   scripting arc. The 42-word surface is comprehensive enough
   that v0.25 can add to it without rebalancing.

4. **Automatic LoRA discovery is a force multiplier.** Civitai's
   API exposes rich per-LoRA metadata (tags, triggers, base
   model). Matching looks to LoRAs at runtime — gated on the
   user not having passed LoRAs themselves — dramatically
   improves out-of-box output quality without locking users into
   curated picks.

5. **Defers AnimateDiff cleanly.** AnimateDiff remains the v0.26+
   carry; v0.25 is a feature-density cycle (presets across all
   surfaces) rather than a model-architecture cycle.

## 3. Architectural constraints we keep

Same five from prior RFCs, plus one new:

1. Built-our-own VM; restricted stdlib.
2. Singleton context; one script per process.
3. Async bridge via `block_in_place`.
4. v0.22 relaxed-compat carries forward (no backwards-compat
   hacks).
5. SD-family two-slot cache (SdT2i + SdPortrait, sharing
   `Arc<SdCore>`) — extend only when load-time concerns require
   it.
6. **NEW: Presets are layering, not replacement.** `--look` and
   `--genre` never alter pipeline behavior on their own —
   they only mutate the generation request fields the user didn't
   already populate. Composes cleanly with everything; tests
   verify that a fully-flagged command ignores look/genre fields.

## 4. The deliverables

### 4.1 Looks (the medium axis)

| Item | What ships |
|---|---|
| `LookSpec` type | `{ name, description, prompt_prefix, prompt_suffix, negative_extras, scheduler_hint, steps, guidance, lora_query, base_compat }`. All scalar fields `Option<T>`; preset only modifies fields the user didn't pass. |
| `LookCatalog` | Bundled `assets/looks.json` with 8 entries: `ink-wash`, `watercolor`, `oil-painting`, `charcoal`, `pencil`, `chalk-pastel`, `linocut`, `gouache`. |
| User catalog | `$CONFIG_DIR/looks/*.json` glob-loaded at startup; user entries shadow bundled by name. |
| CLI flag | `--look NAME` on `generate` / `portrait` / `img2img` / `inpaint` / `outpaint` / `upscale`. |
| Scenario field | Global `look:` + per-task `look:` override. |
| Bund | `plakat.look.{apply, clear, list}` host words. |

### 4.2 Genres (the subject-domain axis)

| Item | What ships |
|---|---|
| `GenreSpec` type | Same shape as `LookSpec`. Distinct axis so a script can `--look watercolor --genre anime` independently. |
| `GenreCatalog` | Bundled `assets/genres.json` with `anime` only. |
| User catalog | `$CONFIG_DIR/genres/*.json` — directory wired but ships empty. Users add genres without code changes. |
| CLI flag | `--genre NAME` on the same six subcommands. |
| Scenario field | Global `genre:` + per-task `genre:` override. |
| Bund | `plakat.genre.{apply, clear, list}` host words. |

### 4.3 Auto-LoRA discovery (the magic)

| Item | What ships |
|---|---|
| Trigger | Only when `ctx.loras` is empty AND the look/genre carries a `lora_query`. User-passed LoRAs always win. |
| Source order | Civitai (primary) → HF Hub (fallback) → local-cache scan (offline). First source with a match wins. |
| Base-model filter | Match current pipeline base (sd15 / sdxl / flux / sd3). Incompatible LoRAs skipped. |
| Cache | `$PLAKAT_CACHE_DIR/look-cache/` keyed by `(look_or_genre, base_model, source)`. Cached entries skip the network. |
| Offline mode | `--offline` flag short-circuits remote sources; local-cache only. |
| Trigger phrases | Discovered LoRAs that expose trigger phrases get them auto-injected into the prompt prefix (same machinery as `--style`'s trigger injection). |

### 4.4 Docs + tests (1 phase)

Mirror of v0.22 phase 12 / v0.23 phase 8 / v0.24 phase 10:
- New `Documentation/LOOKS.md` (medium catalog + extension format).
- New `Documentation/GENRES.md` (genre catalog + extension format).
- SCRIPTING.md update (new namespaces + config keys).
- SCRIPTING_TUTORIAL.md §11 "What's new in v0.25".
- Composition tests + integration tests covering online + offline paths.

## 5. Cycle scope

~12 phases including hygiene + docs:

| Phase | Deliverable | Est. |
|---|---|---|
| 0 | Sweep stale v0.25 markers + RFC commit + version bump | ~0.25 session |
| 1 | `LookCatalog` + `GenreCatalog` JSON schemas + bundled assets | ~0.5 session |
| 2 | `LookSpec` / `GenreSpec` types + override-merge logic | ~0.75 session |
| 3 | CLI flags on `generate` (proves the pattern end-to-end) | ~0.75 session |
| 4 | Civitai discovery client + on-disk cache + `--offline` | ~1.5 sessions |
| 5 | HF Hub fallback + local-cache scan | ~1 session |
| 6 | Wire across `portrait` / `img2img` / `inpaint` / `outpaint` / `upscale` | ~0.75 session |
| 7 | Scenario fields (global + per-task) | ~0.75 session |
| 8 | Bund host words (`plakat.look.*`, `plakat.genre.*`) | ~1 session |
| 9 | User-extension directories + validation | ~0.5 session |
| 10 | Docs (LOOKS.md, GENRES.md, scripting + tutorial) | ~0.75 session |
| 11 | Composition tests + integration (full surface, online + offline) | ~0.75 session |

**Total estimate:** 9.25–10.25 sessions. Larger than v0.24 (~8)
because the auto-discovery client is a substantive new subsystem.

## 6. Decisions (locked 2026-05-27)

### Q1: CLI naming

Options considered:
- **A. `--look` + `--genre`** — two independent axes. Concise. No overload with `--style`.
- **B. `--medium`** — semantically precise but verbose; pairs awkwardly with `--genre`.
- **C. `--art-preset`** — unambiguous but a mouthful, especially in scenarios.
- **D. Extend `--style`** — overloads detection (CLIP-H embeddings) with prescription (named bundles).

**Locked: A.** `--look` is the medium axis; `--genre` is the
subject-domain axis. Independent. Both compose with the existing
`--style` (detection), `--fast` (distillation), and
`--negative-preset` (negative shorthand).

### Q2: LoRA discovery source

Options considered:
- **A. Civitai only** — rich metadata, public API, network required.
- **B. HF Hub only** — already a hard dep, smaller art-LoRA selection, weaker metadata.
- **C. Local-cache scan only** — fully offline; limited to what's downloaded.
- **D. Civitai + HF + local-cache layered** — best discovery quality, most surface.

**Locked: D.** Civitai primary → HF Hub fallback → local-cache
scan. Cache discoveries to `$PLAKAT_CACHE_DIR/look-cache/`
keyed by `(look_or_genre, base_model)`.

### Q3: Discovery trigger semantics

Options considered:
- **A. Always discover** — every `--look` call hits the discovery client.
- **B. Discover only when user's LoRA stack is empty** — respect user-passed LoRAs.
- **C. Discover and append** — discovery results stack on top of user LoRAs.

**Locked: B.** User-passed LoRAs always win. Discovery fires
only when `ctx.loras.is_empty()` at the moment of generate. This
matches the override-only-if-user-didn't-pass rule from v0.14
fast presets.

### Q4: Genre catalog scope

Options considered:
- **A. Anime-only first cut** — ships just anime; other genres v0.26+.
- **B. Anime + 2-3 more bundled** — anime, photoreal, fantasy, cyberpunk.
- **C. Anime built-in + open extension directory** — anime bundled, `$CONFIG_DIR/genres/*.json` wired for user-added genres.

**Locked: C.** Anime is the only bundled genre for v0.25.
User-extension directory wired but ships empty. This validates
the axis pattern with one curated genre while letting users add
more without waiting on plakat releases.

### Q5: Look catalog scope

Options considered:
- **A. 4-5 first-class looks** — minimal curated set, expand later.
- **B. 8-10 first-class looks** — covers the common art-medium space.
- **C. Open-only** — no bundled catalog, all user-supplied.

**Locked: B.** Eight bundled looks: `ink-wash`, `watercolor`,
`oil-painting`, `charcoal`, `pencil`, `chalk-pastel`, `linocut`,
`gouache`. User catalog at `$CONFIG_DIR/looks/*.json` for
extension.

### Q6: Network policy

Options considered:
- **A. Online by default, `--offline` flag** — matches existing model-download UX.
- **B. Offline by default, `--online` flag** — conservative.
- **C. Online required** — simpler code, breaks CI/repro.

**Locked: A.** Online by default. Cached discoveries make
subsequent runs offline-fast. `--offline` short-circuits to
cache + local-scan.

### Q7: Override semantics

Options considered:
- **A. Override-only-if-user-didn't-pass** — preset is suggestion; user wins.
- **B. Preset wins** — preset overrides user-passed flags.
- **C. Preset wins for "preset-owned" fields, user wins for everything else** — split rules per field.

**Locked: A.** Same rule as v0.14 fast presets. Preset values
fill `Option<T>` fields the user left unset (`None`); never
overwrite `Some(_)`. Tests assert that a fully-flagged command
is byte-identical with or without `--look`.

### Q8: Surfaces

Options considered:
- **A. CLI only** — looks/genres available as `--look` / `--genre` flags; no scripting.
- **B. CLI + scenarios** — also `look:` / `genre:` fields in HJSON scenarios.
- **C. CLI + scenarios + bund** — full triple-surface.

**Locked: C.** All three surfaces:

- CLI: `generate`, `portrait`, `img2img`, `inpaint`, `outpaint`, `upscale`.
- Scenarios: global `look:` / `genre:` + per-task overrides.
- Bund: `plakat.look.{apply, clear, list}` + `plakat.genre.{apply, clear, list}`.

## 7. Phase plan (locked 2026-05-27)

See §5.

## 8. What's NOT in v0.25 (explicitly deferred to v0.26+)

- **AnimateDiff** — still the long carry; v0.26 candidate.
- **SD3 / SD3.5 animate** — 3-encoder lerp + MMDiT integrator.
- **Real-ESRGAN ML upscaling** in `plakat.upscale` (the
  standalone word).
- **`plakat.metadata.write`** — gated on `plakat.save` writing
  JSON sidecars.
- **Multiple looks stacked** — one `--look` at a time. LoRA
  stacking already handles per-LoRA blending if users want it.
- **Look composition/inheritance** — a JSON `extends:` field on
  look entries. Possible future enhancement; not needed for v0.25
  scope.
- **Civitai login / paid-API tier** — anonymous-API only for v0.25.
- **Discovery for `--style`** — detection-flavored style stays
  on its curated CLIP-H catalog; only `--look` and `--genre`
  trigger auto-discovery.

## 9. Appendix: starting state survey

Source-of-truth from the 2026-05-27 codebase (post-v0.24):

- **42 host words** across 11 namespaces (v0.24).
- **817 lib tests** green.
- **`src/style/`** — v0.23 style catalog (5 entries:
  watercolor / photorealistic / oil_painting / ukiyo_e /
  art_nouveau). Detection-flavored: CLIP-H exemplars, confidence
  margins, top3-mean aggregation. Per-style LoRA refs + trigger
  phrases + negative_extras per base model. To compose with —
  not to repurpose.
- **`src/pipelines/flux_fast.rs`** — `FastPreset` template
  (`name`, `target`, `lora_repo`, `lora_scale`, `steps`,
  `guidance`, `scheduler_hint`). The shape `LookSpec` /
  `GenreSpec` mirror.
- **`src/prompt/negative_presets.rs`** — built-in 4-entry
  negative-preset catalog + `$CONFIG_DIR/negative-presets/*.txt`
  user-extension directory. Pattern for the new looks/genres
  user directories.
- **`src/cli/scenario.rs`** — HJSON scenario loader with global
  + per-task overrides; already wires `fast:`, `style:`,
  `negative:`, etc. Same site for `look:` / `genre:`.
- **`src/scripting/config.rs`** — `GenerationConfig` +
  `plakat.config.set` validated key/value setter. Not used for
  look/genre proper (they get dedicated host words), but the
  pattern for the offline flag (`offline_discovery` config key?
  decide in phase 4).
- **No existing Civitai client** — new code. Civitai REST API is
  documented at `civitai.com/api` (anonymous endpoints exist).
- **No existing "look book" or named-style file format** — green
  field for the JSON schema.

## 10. Backwards-compatibility considerations

v0.25 is **additive**. No existing CLI flag, scenario field,
host word, or config key changes shape. The preset axes are
optional opt-ins.

One soft constraint: when `--look` is passed and the user's
LoRA stack is empty, the user might be surprised by automatic
LoRA downloads on first run. Mitigations:

- First-time `--look` with discovery emits an INFO-level log
  describing which LoRA is being fetched + from where.
- `--offline` flag is documented prominently for CI/repro use.
- Cached results make subsequent runs network-free.
- `--look NAME --no-discover` (TBD: phase 4 decides if this is a
  separate flag or just `--offline`) lets users skip discovery
  per call.

## 11. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Civitai metadata quality varies wildly | High | Cache by `(look, base_model)` so once-curated query persists; validate downloads expose a trigger phrase or fall to HF/local. |
| Civitai API rate-limits anonymous users | Medium | Cache aggressively; respect 429 + retry with backoff; fall through to HF on persistent failure. |
| Civitai TOS or content concerns | Medium | Document that LoRA downloads carry their source's license; surface license info in the discovery log line; don't auto-download NSFW-flagged LoRAs (skip + log). |
| User's LoRA universe doesn't match our looks | Medium | The 8 bundled looks are mainstream art mediums with broad LoRA availability. Niche mediums become user-catalog entries. |
| Discovery network calls slow first-run UX | Low | Cache hits make second+ calls instant; first call shows a progress log. |
| Override-semantics get subtly wrong | Low (but high impact) | Phase 11 has a dedicated "fully-flagged command is byte-identical with/without `--look`" test. |
