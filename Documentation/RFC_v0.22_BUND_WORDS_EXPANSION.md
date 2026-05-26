# RFC: Extend `plakat.*` words for low-level generation coverage (v0.22)

**Status:** Decisions locked 2026-05-26. Ready to convert to phases.
**Author:** v0.22 cycle research.
**Date:** 2026-05-26.
**Precedent:** [`RFC_v0.21_BUND_SCRIPTING.md`](RFC_v0.21_BUND_SCRIPTING.md)
shipped the 7-word MVP; this RFC plans the expansion.

---

## 1. TL;DR

v0.21 shipped a deliberately minimal 7-word MVP. v0.22 expands it
to give scripts roughly the same surface the CLI exposes (LoRA
stacking, ControlNet, refiner, style transfer, ADetailer, hires-
fix, artefacts, ...) **plus** the family expansion deferred from
v0.21 (Flux + SD3 / SD3.5) **plus** a pipeline cache so chained
calls don't reload the model.

**The gap is 51 new host words across 8-13 namespaces + 27 new
config.set keys + 13 family-specific knobs.** All of those at
once is multi-cycle; this RFC's job is to pick a coherent v0.22
subset and defer the rest to v0.23.

**Recommended v0.22 scope** (proposed in §6, locked in §8):
- Family expansion (Flux + SD3 + SD3.5 — the v0.21 "phase 2b")
- Pipeline cache (the load-bearing UX fix)
- Top 4 word namespaces: `lora` / `controlnet` / `refiner` / `style`
- All 27 easy config.set keys (cheap surface wins)

That's roughly **8 phases, ~5-6 sessions** — comparable to v0.21's
big-swing cycle. Bigger swing than v0.21 because Flux + SD3 +
the pipeline cache all touch foundational scripting infrastructure
that the new words then build on.

---

## 2. Why this is a big swing

The research output (file: this RFC's "Appendix A" inventory)
quantifies what the v0.21 surface doesn't cover:

| Category | Count | What it is |
|---|---|---|
| A — already covered | 35 | Core v0.21 surface (load, generate, the 9 config keys) |
| B — easy config.set additions | 27 | Single-scalar knobs (aspect, lora_scale, clip_skip, ...) |
| C — needs new host word | 51 | Collection-state things: LoRA stacks, ControlNet adapters, refiner toggle, ADetailer pipeline, hires-fix, artefacts, style transfer, embeddings, enhance, ... |
| D — family-specific | 13 | Flux quant levels, T5 quant, SD3 tiling, Kontext bucket, fast presets, Redux images |
| E — intentional gap | 53 | UI/CLI flow: grid, count, recipe, format, metadata, asset-library paths — not scripting concerns |

180 flags total across the 6 image-producing subcommands.
v0.21 covers ~20%. Achieving "low-level generation coverage" as
the user phrased it means closing the **A+B+C+D = 126 flag**
gap.

That's beyond one cycle. The decisions in §8 pick the subset
v0.22 can deliver well rather than a thinner pass over everything.

---

## 3. Architectural constraints we keep

The v0.21 RFC locked seven decisions (embed crate, build-our-own
VM, `plakat run` subcommand, default-on, `.bund` extension,
REPL, MVP word set). All seven remain in force for v0.22. The
new architectural questions are subordinate:

- New collection words (`plakat.lora.add`, `plakat.controlnet.add`)
  add **stack-managed state** on top of the per-call config —
  the LoRA stack is a Vec<Spec>, not a scalar. We need a
  retain-across-calls strategy + a clear-it word.
- Pipeline cache forces an architectural change to `ScriptCtx`:
  the `loaded_model: Option<String>` field becomes a
  `HashMap<String, LoadedPipeline>` so subsequent calls reuse
  the model. The actual pipeline types (t2i / flux / sd3) are
  unioned into a `LoadedPipeline` enum.
- Family expansion lifts the v0.21 phase-2 gate in
  `script_entry::validate_supported_for_phase_2`. Each family's
  `Request` has different fields → we'll need a family-
  dispatched entry function rather than one t2i wrapper.

None of these break v0.21 scripts. The 7 existing host words +
9 config keys keep their signatures.

---

## 4. The new word namespaces

Cribbed from the research output (§Appendix A) with light edits
for naming consistency.

### 4.1 `plakat.lora.*`

```
plakat.lora.add    ( path weight -- )    Stack-extend the LoRA list
plakat.lora.clear  ( -- )                Drop the entire stack
plakat.lora.list   ( -- list )           Push current stack as a list (introspection)
```

Plus the existing scalar `plakat.config.set "lora_scale" F`
(global multiplier). The collection mutator pair (add / clear)
matches how `--lora PATH:weight` accepts repeated values on
the CLI; `list` is the introspection affordance that scripts
need to confirm what's loaded.

### 4.2 `plakat.controlnet.*`

```
plakat.controlnet.add        ( kind image-path -- )    Pre-rendered conditioning
plakat.controlnet.annotate   ( kind from-path -- )     Auto-annotate from a photo
plakat.controlnet.spec       ( spec-string -- )        Full --control-spec grammar
plakat.controlnet.clear      ( -- )
```

Add / annotate cover the two common cases (pre-rendered map,
auto-annotate). `spec` accepts the rich CLI `kind:image=PATH:
strength=F:start=F:end=F` grammar for power users. Strength /
start / end can be set per-CN via the spec string; no separate
`set-strength` word needed.

### 4.3 `plakat.refiner.*`

```
plakat.refiner.enable   ( -- )           Enable SDXL refiner pass
plakat.refiner.disable  ( -- )
```

Plus `plakat.config.set "refiner_frac" F` (where the refiner
takes over, default 0.8) and the existing `refine` / `refine_strength`
config keys for the same-model polish pass.

### 4.4 `plakat.style.*`

```
plakat.style.apply   ( id -- )                Apply a named style from catalog
plakat.style.detect  ( ref-image-path -- id )  Detect + apply; pushes the style id
plakat.style.clear   ( -- )
```

Plus `plakat.config.set "style_strength" F`. Style is per-call
(not collection); only one style applies at a time.

### Deferred to v0.23 (in §6's scope decision)

- `plakat.adetailer.*` (6 keys)
- `plakat.hires.*` (5 keys)
- `plakat.artefact.*` (collection + 4 keys + smart-zones)
- `plakat.enhance.*` (3+ keys; can also fold into config.set with `enhance_provider`)
- `plakat.embedding.*` (collection)
- `plakat.stylize` (separate workflow word, different from style-catalog)
- `plakat.outpaint` (separate workflow word)
- `plakat.portrait.set-bbox` / `set-landmarks` (lift the single-photo limitation)

---

## 5. The new config.set keys (Category B)

27 keys that fold into the existing `plakat.config.set ( value
key -- )` surface without needing new words. Grouped for clarity:

**Size:** `aspect` (string like "16:9"), `base` (int)

**Mask:** `mask_feather` (px), `mask_invert` (bool)

**LoRA:** `lora_scale` (f32, global multiplier)

**Style:** `style_strength` (unit f32)

**Refine pass:** `refine_steps` (int), `refine_strength` (unit f32), `refiner_frac` (unit f32)

**ControlNet defaults:** (set-once defaults that per-CN spec can override)

**Enhance:** `enhance_provider` (string), `enhance_temp` (f32), `enhance_max_tokens` (int), `enhance_keep_original` (bool), `enhance_system` (path)

**Negative preset:** `negative_preset` (string)

**Wildcards:** `wildcard_dir` (path)

**CLIP:** `clip_skip` (int)

**ADetailer:** `adetailer` (bool), `adetailer_strength` (unit f32), `adetailer_padding` (unit f32), `adetailer_feather` (unit f32), `adetailer_confidence` (unit f32), `adetailer_size` (int), `adetailer_prompt` (string)

**Hires-fix:** `hires_fix` (bool), `hires_scale` (f32), `hires_strength` (unit f32), `hires_upscaler` (string), `hires_steps` (int)

Total: 27 keys. Each adds ~5 lines to `config.rs::set_str` + a
validation rule. Cheap surface wins — all 27 land in one phase.

---

## 6. Family expansion (the v0.21 "phase 2b")

The v0.21 gate `script_entry::validate_supported_for_phase_2`
rejects Flux + SD3 + SD3.5 with a "Phase 2b" pointer. v0.22
lifts the gate. This is the biggest single piece of
infrastructure work in the cycle.

**Three sub-tasks:**

1. **Flux family.** `flux-dev` / `flux-schnell` / `flux-fill-dev`
   / `flux-canny-dev` / `flux-depth-dev` / `flux-kontext-dev` +
   GGUF / NF4 quantized variants. Different Request type
   (`pipelines::flux::Request`), different default sizes, T5 in
   the encoder stack. Existing CLI plumbing in `cli::generate`
   does the routing — `script_entry` learns the same dispatch.
2. **SD3 / SD3.5 family.** Similar shape — `pipelines::sd3::Request`,
   different default sizes, three text encoders.
3. **Family-specific config keys (Category D):** `quantize_t5`,
   `quant_level`, `t5_quant_level`, `tiled`, `tile_size`,
   `tile_stride`, `kontext_bucket`, `fast` (preset name), Redux
   images (a Vec; collection).

Once the gate lifts, the new word namespaces (LoRA, ControlNet)
have to know which family's `Request` to populate. That's a
union enum on `ScriptCtx` — see §7 below.

---

## 7. The `LoadedPipeline` enum + pipeline cache

v0.21's `ScriptCtx.loaded_model: Option<String>` was a sentinel
— each `plakat.generate` reloaded the model. v0.22 promotes
it to a cached pipeline:

```rust
enum LoadedPipeline {
    SdFamily(pipelines::t2i::Pipeline),
    Flux(pipelines::flux::Pipeline),
    Sd3(pipelines::sd3::Pipeline),
}

struct ScriptCtx {
    // ...existing fields...
    loaded: Option<(String, LoadedPipeline)>,
    // The collections that drive new words:
    loras: Vec<LoraSpec>,
    controlnets: Vec<ControlSpec>,
    // ...etc per word namespace
}
```

On `plakat.generate`:
1. If `ctx.loaded` is `Some(alias)` matching the current call,
   reuse it.
2. Else, drop the old pipeline (RAII-frees GPU memory) and load
   the new one.

The cache is per-alias — switching models reloads, but
multi-call scripts on one model stay fast. For SDXL → x100 image
batch, that's the difference between ~30 min and ~3 min on a
4090.

**Trickier:** LoRA + ControlNet mutations invalidate the cache.
`plakat.lora.add` after a generate has to either: (a) reload
the pipeline with the new LoRA merged in, or (b) defer the
mutation to the next generate call. Option (b) is simpler and
matches Flux's runtime LoRA story. The first generate after a
mutation pays a re-merge cost; subsequent same-call-config
generates are free.

---

## 8. Decisions (locked 2026-05-26)

| # | Question | Decision | Notes |
|---|---|---|---|
| 1 | Namespace scope | **Full sweep — all 8 namespaces** | lora + controlnet + refiner + style + adetailer + hires + artefact + enhance. Bigger than the recommendation; cycle grows accordingly. |
| 2 | Family expansion | **Flux + SD3 + SD3.5 all in v0.22** | One round of `LoadedPipeline` refactoring; all three families ship together. |
| 3 | Pipeline cache | **Yes — phase 1** | Foundational; without it, LoRA + ControlNet workflows are too slow to be useful. |
| 4 | Category-B config keys | **Split per-namespace** | The 27 keys land with their related word phases (e.g. adetailer_* keys with `plakat.adetailer.*`). NOT batched into one phase. Cleaner per-phase scope at the cost of more boundary edits in `config.rs`. |
| 5 | Family-specific D keys | **Bundle with family phases** | `quantize_t5` ships with Flux; `tiled` semantics differ per family, so per-family validation; `kontext_bucket` only when Kontext loads. |
| 6 | Introspection words | **Ship `.list` for collections** | `plakat.lora.list`, `plakat.controlnet.list`, etc. Push current state as a list onto the workbench. REPL debugging affordance. |
| 7 | v0.21 backwards compat | **Relax — may break if improvement warrants** | NOT strict. v0.22 has freedom to rename / reshape v0.21 words if a better pattern emerges. v0.21 scripts may need a migration pass. Practical consequence: `plakat.upscale ( handle scale -- handle )` may grow new args, `config.set` keys may rename, etc. |

The full-sweep + family-expansion choice means v0.22 is the
**largest single cycle since the project started**. Realistic
estimate: 8-10 sessions, 13 phases (§9 below). Comparable to a
v0.20-style "three big swings" cycle compressed into one focus
area.

---

## 9. Phase plan (locked post-decision)

| Phase | Deliverable | Estimate |
|---|---|---|
| **1** | **Foundation: pipeline cache + `LoadedPipeline` enum.** `ScriptCtx.loaded` becomes `Option<(String, LoadedPipeline)>`. v0.21's `plakat.generate` keeps working but now reuses the cached pipeline across calls. ~3× speedup on multi-image scripts. | ~1 session |
| **2** | **Flux family expansion.** Lift the phase-2 gate for Flux variants (`flux-dev`, `flux-schnell`, `flux-fill-dev`, `flux-canny-dev`, `flux-depth-dev`, `flux-kontext-dev` + GGUF/NF4). Routes through `pipelines::flux::Pipeline`. Bundled D-keys: `quantize_t5`, `quant_level`, `t5_quant_level`, `fast` (preset), `kontext_bucket`, `redux_image` collection. | ~1 session |
| **3** | **SD3 / SD3.5 family expansion.** Route `sd3-medium`, `sd35-medium`, `sd35-large`, `sd35-large-turbo` through `pipelines::sd3::Pipeline`. Bundled D-keys: `tiled` (SD3 semantics), `tile_size`, `tile_stride`. | ~0.5 session |
| **4** | **`plakat.lora.*` namespace.** `add` / `clear` / `list` + the `lora_scale` Category-B key. Cache-aware: LoRA mutations defer to next generate call. | ~1 session |
| **5** | **`plakat.controlnet.*` namespace.** `add` / `annotate` / `spec` / `clear` / `list`. ControlSpec grammar matches the CLI's `--control-spec`. Per-CN strength/start/end via the spec string. | ~1 session |
| **6** | **`plakat.refiner.*` + `plakat.style.*` (smaller pair).** Refiner: `enable` / `disable` + `refiner_frac`, `refine_steps`, `refine_strength` config keys. Style: `apply` / `detect` / `clear` + `style_strength` key. | ~0.5 session |
| **7** | **`plakat.adetailer.*` namespace.** `enable` / `disable` + 6 bundled Category-B keys (`adetailer_strength`, `_padding`, `_feather`, `_confidence`, `_size`, `_prompt`). | ~0.5 session |
| **8** | **`plakat.hires.*` namespace.** `enable` / `disable` + 5 bundled Category-B keys (`hires_scale`, `_strength`, `_upscaler`, `_steps`). | ~0.5 session |
| **9** | **`plakat.artefact.*` namespace.** `add` (NAME ZONE SCALE) / `clear` / `list` + 4 bundled keys (`artefact_blend`, `_strength`, `smart_zones`, `artefact_library` path). | ~1 session |
| **10** | **`plakat.enhance.*` namespace.** Provider selection (`local` / `api`) + 5 bundled keys (`enhance_provider`, `_temp`, `_max_tokens`, `_keep_original`, `_system` path). | ~0.5 session |
| **11** | **Remaining Category-B keys** that don't fit a namespace: `aspect`, `base`, `mask_feather`, `mask_invert`, `clip_skip`, `wildcard_dir`, `negative_preset`. | ~0.5 session |
| **12** | **Docs + composition tests.** SCRIPTING.md + SCRIPTING_TUTORIAL.md grow coverage sections for every new namespace. Composition tests exercise the full surface end-to-end. | ~1 session |

**Total: ~9-10 sessions** (13 phases). The largest single cycle
since v0.13's Flux modernization.

### Phase ordering rationale

- **Foundation first.** Phase 1's pipeline cache is the
  load-bearing prerequisite for every subsequent phase. Without
  it, every test that touches a host word pays the SD model
  load cost — the test cycle becomes intolerably slow.
- **Family expansion second.** Phases 2-3 lift the SD-family
  gate before LoRA / ControlNet land. LoRA + CN behave
  differently across families (Flux runtime LoRA vs SD merge-
  into-UNet); shipping families first means new words only
  write against one mature `LoadedPipeline` enum.
- **Word namespaces in order of utility.** LoRA (phase 4) +
  ControlNet (phase 5) are the highest-demand additions; the
  refiner + style pair (phase 6) is smaller and shares
  patterns; ADetailer (7) + Hires (8) are post-processing;
  Artefact (9) + Enhance (10) round out the surface.
- **Misc keys (11) before docs.** Sweep up the remaining
  Category-B keys that don't fit a namespace just before the
  docs phase, so the tutorial gets a complete word + key
  catalog.
- **Docs (12) last** so the tutorial reflects shipped reality
  not RFC speculation. Same pattern as v0.21 phase 8.

---

## 10. What's NOT in v0.22 (explicitly deferred to v0.23)

After the full-sweep decision (decision #1), v0.22 covers
**eight** of the original 8 namespaces from the research. The
following are still deferred to v0.23:

- `plakat.embedding.*` (collection — Textual Inversion specs)
- `plakat.stylize` (separate IP-Adapter-based workflow word,
  distinct from `plakat.style`'s catalog-based path)
- `plakat.outpaint` (separate workflow word — `plakat.img2img`
  with a mask wraps it today, but a dedicated word would expose
  per-side expansion)
- Multi-photo portrait + FaceID + manual `face_bbox` /
  `face_landmarks` (lifts the v0.21 phase-5 limitation)
- Real-ESRGAN ML upscaling (the v0.21 phase-6 deferred item)
- `plakat.metadata.*` (JSON sidecar read/write from scripts)
- SD3 animate (carried over from v0.20)
- AnimateDiff (carried over from v0.20)

v0.23 picks up where v0.22 leaves off. The RFC at that point
will be a continuation of this one's structure.

---

## Appendix A: Full flag inventory

180 flags across 6 subcommands; classified into 5 categories
(A/B/C/D/E). See the v0.22 research output for the per-flag
table. Summary numbers:

| Subcommand | Flags | A | B | C | D | E |
|---|---|---|---|---|---|---|
| generate | 65 | 7 | 8 | 23 | 9 | 18 |
| img2img | 39 | 8 | 8 | 9 | 3 | 11 |
| portrait | 44 | 9 | 5 | 16 | 0 | 14 |
| outpaint | 20 | 8 | 6 | 1 | 0 | 5 |
| upscale | 4 | 2 | 0 | 0 | 0 | 2 |
| stylize | 8 | 3 | 0 | 2 | 0 | 3 |
| **Total** | **180** | **37** | **27** | **51** | **12** | **53** |

The "C" column drives the new-word categories; the "B" column
drives the new-config-set keys; "D" drives the family-expansion
phases.
