# ROADMAP — plakat 6.1.0

The follow-on to the 6.0 `plakat bookart` flagship (RFC BOOKART-1). 6.0.x shipped the whole `bookart`
vertical as a **standalone CLI** (`new · lint · show · render · illustrate · verify · kit · manuscript ·
proof · diff · edit · blend`) and deferred *integration parity* + a set of documented fast-follows. 6.1.0
closes those — wiring `bookart` into the rest of plakat and rounding out the feature — the same way the
persona 5.1 arc followed the 5.0 cut.

**Theme:** *bookart everywhere, and finished.* No new RFC — this is the deferred tail of BOOKART-1.
Fully additive; the CLI surface and outputs from 6.0.x are unchanged.

---

## Track A — ecosystem integration parity (the headline)

The rule from every prior cycle: a feature stranded on one surface isn't finished. `bookart` currently
only exists as its own subcommand. Wire it into the automation surfaces, mirroring how `persona` /
`fractals` were integrated.

- **A1 — library API `BookArt` builder. DONE.** Extracted the render core out of
  `cli/bookart.rs::do_render` into `src/bookart/render.rs`: `render_spec(spec, &RenderOpts) → Rendered`
  (in-memory transparent RGBA page + optional SVG + resolved plan + scorecard + piece count), with the
  moved `gen_size`/`diffuse`/`matte_silhouette`. The CLI `do_render` is now a thin delegate + file I/O.
  Added `plakat::api::BookArt` (`load`/`from_spec` · `model`/`seed`/`steps`/`svg`/`attempts` · `run`),
  mirroring `Generate`/`Portrait`. CI test exercises the procedural core (no GPU); the CLI is unchanged
  (smoke-tested). *Unblocks A2–A5.*
- **A2 — scenario `type: bookart`. DONE.** `src/bookart/scenario_task.rs` (`BookartTaskCfg` = inline
  `spec` or `spec_file` + model/seed/steps/svg/attempts; `validate` + `run_bookart_task` → the shared
  `render_spec` → `<out>/<name>/ornament.{png,svg}`). Wired the 7 `scenario.rs` touchpoints (TaskKind
  variant, `from_strs`, TaskDef `bookart:` field, cache-eviction, preflight validate, scene/weather
  skip, execution dispatch). Verified live: a 2-task scenario (procedural border + SVG, diffusion
  vignette) renders end-to-end + `--dry-run` validates.
- **A3 — compile `bookart:` directive. DONE.** A `type: bookart` block (+ `bookart-origin` /
  `-technique` / `-type` / `-page` / `-svg` directives; the prose is the ornament prompt) in a `compile`
  prompts file compiles to a scenario `bookart` task. Threaded parser (prose-optional detector),
  resolver (5 `bookart_*` fields), emitter (nested `bookart: { spec: {…} }`; suppresses the dead
  top-level prompt/negative). Verified: a 2-block prompts file → a scenario the runner dry-run-validates
  (diffusion vignette + procedural border).
- **A4 — Bund `plakat.bookart.*` words. DONE.** `src/scripting/words/bookart.rs`: `plakat.bookart.origin`
  / `.technique` (illustrate overrides, `none`/`auto` clears) · `.render ( spec-path -- handle )` (any
  tier; procedural = no GPU) · `.illustrate ( prompt -- handle )` (prose → a diffusion B/W plate, GPU +
  origin LoRA). The pushed handle is the transparent, exactly-page-sized RGBA ornament — it flows into
  the existing `plakat.save` / `plakat.metadata.write` / `plakat.upscale` words (alpha preserved).
  model/seed/steps come from the shared `plakat.config.*` state. 2 new `ScriptCtx` fields
  (`bookart_origin`/`bookart_technique`). Verified live: procedural border via `.render` → clean A5
  guilloché frame; `.origin russian` + `.illustrate "…firebird…"` → Bilibin foliate line plate (LoRA
  128/128 merged). Corpus `corpus/bookart_script.bund`. (SVG is a file-only artefact, not a handle — use
  the CLI `bookart render --svg`.)
- **A5 — sidecar + `--import`. DONE.** Every `bookart render` / `illustrate` (and each kit/manuscript
  ornament) now writes its reproducibility recipe: `render::recipe_metadata(plan, model, seed, steps)`
  → a `GenerationMetadata` carrying origin/technique/tier/ornament/symmetry/page + a stable **spec-hash**
  (FNV-1a of the resolved plan identity) in `extras`, the hosted origin LoRA in `loras`, embedded as an
  Auto1111 `parameters` PNG `tEXt` chunk (ASCII, alongside the pHYs DPI) **and** a `<png>.json` sidecar
  (`canvas::save_png_dpi_with_metadata`). `bookart render|illustrate --import <album>` then lands the
  ornament + sidecar in a `plakat photos` album via `photos::import::import_outputs` (auto-tagged from
  the recipe: `ai`/`bookart`/origin/ornament); `--import` is `#[cfg(feature="photos")]` (a clear
  "needs the photos feature" note when compiled out). Verified live: tEXt + sidecar + album.hjson all
  carry the recipe; pHYs 300 DPI preserved. This completes **Track A** — bookart is now on every
  automation surface (CLI, API, scenario, compile, Bund) and curates into photos.

## Track B — round out the feature (fast-follows deferred from B6–B10)

- **B1 — raster→SVG trace.** `--format svg` on the diffusion/composite tiers + a standalone `bookart
  vectorize <raster> --out svg`, via a **permissively-licensed** tracer (`vtracer`/`visioncortex`,
  MIT — resolved in G0.1). Isolate it behind a feature/opt-in so the base build stays lean (vtracer
  pulls an old `image` stack). Procedural born-vector SVG already ships.
- **B2 — glyph-driven initials (§6.5).** `initial` renders the actual letter (any script, incl.
  Cyrillic) as a mask/ControlNet so a historiated initial is built *around* a legible letterform —
  the one intentional-text path. Needs a font rasteriser (reuse the `shaped-labels` ab_glyph path).
- **B3 — `bookart origins`.** List the origin × technique presets + which origin LoRAs are present/
  hosted (mirrors `plakat style` / the `doctor` section). Load `assets/bookart/lexicon.hjson` as an
  override of the built-in lexicon (currently built-in only).
- **B4 — `bookart font`.** Export a set of small ornaments (fleurons/dinkus) as an `.otf` dingbat font
  for inline use in InDesign/LaTeX.
- **B5 — more origin LoRAs.** american / european / chinese, trained on PD corpora (extend the G0.3
  `datasets/bookart_training` + `train_origins.sh` machinery) and hosted at `vulogov98/plakat-bookart`.
  The generic path already covers them; these raise the ceiling.
- **B6 — EPUB manuscript input.** `bookart manuscript book.epub` (parse the spine) alongside the
  Markdown / plain-list inputs.

## Track C — quality / polish

- **C1 — procedural richness.** More band motifs (foliate scroll via a small L-system, Greek-key,
  guilloché-rosette variety); per-fold tailpiece variation; a knotwork/interlace generator (the one
  net-new from G0.4 not yet built).
- **C2 — ink-weight as a real dial.** Cache the raw grayscale render so `ink.weight` / `transparency`
  become `post`-class edits (today they force a re-gen — see `bookart/edit.rs`); wire `bookart edit`
  to re-run the finisher without re-sampling.
- **C3 — composite framing polish.** Band-shaped composite frames (a headpiece cartouche = a wide frame
  with a central window) so `headpiece`/`tailpiece` composites aren't squished into a square window.

---

## Sequencing

**A1 first** (the render-core extraction) unblocks A2–A5 and the API. Then Track A in order (each surface
is small once the core is a library fn). Track B/C items are independent and can land opportunistically.
Cut 6.1.0 when Track A is complete + a reasonable slice of B/C; further B/C items roll to 6.2.

## Release-flow reminders (from auto-memory)

Bump `Cargo.toml` **+ `Cargo.lock`** in sync (`--locked` CI); gate = `cargo test --no-default-features
--lib`; new capability surfaces in `doctor`; no Claude/Anthropic co-authoring; FF `main` + tag → 6-asset
CI release + `cargo publish --locked --allow-dirty` (untracked `assets/filipok.md` is excluded);
`gh release edit` for notes (owner `vulogov` GH token).
