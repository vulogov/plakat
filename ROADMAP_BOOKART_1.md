# ROADMAP — BOOKART-1 implementation plan

Implementation plan for **RFC BOOKART-1** (`Documentation/RFC_BOOKART_1.md`). Same
build discipline as the persona cycle (`ROADMAP_5.0.0.md`): front-load the pure,
GPU-free, golden-tested half; prove the output contract before generation; keep it
**fully additive**. Tentative version target **6.0.0** (a major, like persona was
5.0) — confirmed at cut time, not now.

**The output contract that every phase must preserve:** an ornament is
*generated → transparented (luminance-native) → sized to the exact page/DPI →
emitted as PNG*. SVG is a secondary, `--format svg` opt-in.

---

## Module layout (new, all additive)

```
src/bookart/
  spec.rs          # BookArtSpec (permissive serde), OrnamentSpec, Ink, PageSpec, Output
  lexicon.rs       # origin × technique × motif presets + per-type defaults
  compile.rs       # resolver: (spec, lexicon) → RenderPlan   [pure, byte-stable]
  lint.rs          # schema / contradiction / page-validity / anti-text safety
  geometry/
    mod.rs
    page.rs        # PageSpec, named sizes → px, text-block, DPI, bleed
    layout.rs      # per-ornament-type canonical layout vs the text block
    symmetry.rs    # fundamental domain + replicate (bilateral/radial:N/frieze)
  finish/
    mod.rs
    binarize.rs    # technique binarisers: XDoG, adaptive-threshold, dither, engraving
    alpha.rs       # transparency: luminance / threshold / matte / fade + tint
    canvas.rs      # place finished ornament onto exact page canvas + pHYs DPI
    vectorize.rs   # opt-in: born-vector emit (procedural) + trace (diffusion)
  procedural/
    mod.rs  border.rs  corner.rs  rosette.rs  knot.rs  guilloche.rs
  render.rs        # the hybrid router: procedural | diffusion | composite
  scorecard.rs     # chroma / alpha / symmetry / ink / DPI / safe-area / glyph probes
  kit.rs           # coherent set: motif DNA + seed lineage + kit-coherence
  manuscript.rs    # chapter parse → per-chapter matched set + manifest
  mod.rs
src/cli/bookart.rs
assets/bookart/
  lexicon.hjson
  origins/{russian,english,japanese}.hjson         # prompt scaffold + defaults
  origins/{russian,english,japanese}.safetensors   # trained LoRAs (from G0)
```

---

## Substrate reuse (do not rebuild)

| Need | Reuse | Path |
|---|---|---|
| Procedural ornament (rosette / knot / guilloché / border) | fractal engine (compose, coloring, flame, control-source) | `src/fractals/{compose,coloring,flame,control_source}.rs` |
| Diffusion render + LoRA stacking | the generation facade | `src/api` (`Generate`), `src/pipelines/*_unet.rs` |
| Line structure enforcement | lineart ControlNet + annotator | `src/pipelines/{lineart,controlnet,controlnet_annotator}.rs` |
| Origin-LoRA training | style LoRA trainer | `src/cli/style.rs` (`style train`, SD3.5 LoRA) |
| Style-preset catalog pattern | style catalog loader | `src/style/{catalog,mod}.rs` |
| `matte` transparency mode | U2Net matting | `src/pipelines/matting.rs` |
| Candidate ranking | LAION aesthetic scorer | (persona reuses `aesthetic::AestheticScorer`) |
| Stray-glyph probe | OWL-ViT / detector | `src/pipelines/owlvit.rs` |
| Whole spec→resolve→verify spine | persona as the structural template | `src/persona/*` (mirror, don't couple) |
| scenario / compile / Bund / API wiring | existing integration surfaces | as persona/fractals did |

**Genuinely net-new:** the `finish/` chain (binarize/alpha/canvas/vectorize), the
`geometry/` engine (layout/symmetry/page), the procedural *ornament* primitives on
top of the fractal core, the print/ink `scorecard`, `kit`/`manuscript`, and a
**permissive raster→SVG tracer** dependency (§G0.1).

---

## G0 — Gating research (do FIRST; architecture-shaping)

Resolve RFC §13 before B0. Each is cheap and de-risks a load-bearing choice.

- [x] **G0.1 Vectorizer license.** **RESOLVED.** `vtracer` 1.0.0-alpha.3 and
      `visioncortex` 0.9.1 are both **MIT OR Apache-2.0** — permissive, compatible
      with the Unlicense repo; no GPL potrace crate exists or is needed. **Pick:
      `vtracer`** (binary/B&W mode) on `visioncortex`. Gates only the opt-in SVG path;
      trace-*quality* prototype deferred to **B6** (off the critical path).
- [x] **G0.2 Baseline harness.** **DONE** — `examples/bookart_baseline.rs` (runnable
      metrics: chroma / page-haze / alpha-halo / bilateral-symmetry-RMS + Otsu binarise
      + the G0.5 curve). Measured on two Metal-generated controls (sd15, 512²):
      | metric | naive woodcut | naive line-art | ours |
      |---|---|---|---|
      | chroma coloured-frac | 0.380 | 0.162 | **0.000** |
      | alpha-halo (partial-α) | 0.418 | 0.355 | **0.000** |
      | page-haze | 3.4 | 2.1 | **0.0** |
      | bilateral symmetry RMS | 0.477 | 0.286 | *unchanged (finisher can't fix)* |
      Findings: diffusion "B/W" is 16–38% **tinted** and naive keying leaves a 35–42%
      **partial-alpha halo** → bookart justified; binarise + luminance-alpha zeroes
      chroma/halo/haze; **symmetry is untouched by the finisher → the symmetry engine
      is essential + independent.** The metric fns become the B1 scorecard.
- [x] **G0.3 Origin-LoRA feasibility — decision resolved.** The **generic line-art
      path** (sd15 + a line prompt + the finisher, *no LoRA*) already yields usable,
      clean, transparent ornament (G0.2 woodcut control); the stronger line prompt
      **halved** chroma-leak + asymmetry. → **Origin-LoRAs are a ceiling-raiser, NOT a
      v1 blocker** (confirms D4/R3): v1 ships on the generic path. **UPDATE — corpus
      + train DONE, and the LoRA BEATS generic.** 120 PD images (40 each Bilibin/
      Beardsley/Hokusai, Wikimedia, grayscale-prepped) → 3 sd15 LoRAs (rank 16, 256²,
      180 steps) trained + **hosted at `vulogov98/plakat-bookart`**
      (`<origin>-sd15.safetensors`). A/B on the G0.2 metrics (russian): LoRA render
      **chroma 0.053 / symmetry-RMS 0.134** vs generic **0.16–0.38 / 0.29–0.48** — the
      LoRA is measurably cleaner AND more symmetric. Scripts:
      `tools/bookart/{fetch_training_corpus.py, prep_grayscale.py, train_origins.sh}`.
- [x] **G0.4 Procedural coverage audit.** **DONE** (findings below). Fractal engine is
      **raster-only, no vector, no binarisation, no seamless/tiling, no heterogeneous
      compositor**; reusable pieces + net-new gaps enumerated.
- [x] **G0.5 Transparency curve.** **DONE** — `examples/bookart_alpha_probe.rs`
      (runnable). Chosen `luminance` default: `alpha = clamp((1−L − white_cut)/
      (1−white_cut))^γ` with **`white_cut ≈ 0.07, γ ≈ 0.70`** → measured page-haze on
      near-white = **0.0** (linear = 9.5; γ0.6 = 33.2 = too hazy) while lifting
      mid-grey line α to 147 (vs linear 127). Kills anti-alias halos, keeps thin/grey
      lines opaque. Params exposed via `ink.transparency`.

### G0 findings — refinements to fold into B1–B6

- **`procedural = born-vector` is only true for NEW parametric generators**
  (guilloché, knotwork, border edge/corner units — those emit SVG paths directly).
  The **fractal-engine reuse is RASTER** (flame-rosette, L-system scrollwork) → it
  goes through the *same* binarise → trace path as diffusion for SVG. Update RFC §7.5
  wording accordingly (done).
- **Reuse directly:** `fractals::plot::Fit` (model→pixel framing), `palette.rs`
  (2-stop B/W / grayscale ramp + `parse_hex`), `lsystem.rs` `turtle()` + `draw_thick`
  (the ONLY real stroke infra — foliate scrollwork starts here), the deterministic
  `render_spec` seed contract.
- **Wrap:** `flame.rs` symmetry (N-fold rotational → rosette/mon), Newton basins +
  `Angle` colouring (radial colophon).
- **Net-new (fractal engine offers no generator):** the seamless **border
  edge/corner + L-corner** system **and a real ornament compositor** (`compose.rs`
  only grid-blits — cannot place/mirror/rotate heterogeneous pieces), **guilloché**,
  **Celtic knotwork**, and the **binarise pass** (none exists today — grayscale ≠
  1-bit). All already in the module layout (`geometry/layout`, `finish/binarize`,
  `procedural/*`); the audit confirms scope, no surprises.
- **Watch-out:** the fractal engine's advertised top-level `spec.symmetry` is
  **unimplemented/dead** (only `flame.symmetry` is wired) — do not rely on it; the
  bookart symmetry engine (`geometry/symmetry.rs`) is fully net-new.

**Thin de-risking slice (recommended alongside remaining G0):** one *procedural*
ornament (`divider`) all the way through spec → procedural → alpha → canvas → PNG,
and one *diffusion* ornament (`vignette`, generic path) end to end. Proves both
spine ends before breadth.

---

## Phases

Checkbox granularity mirrors `ROADMAP_5.0.0.md`. **CI** = testable under
`cargo test --no-default-features --lib` (no GPU); **GPU** = weights phase.

### B0 — spec + resolver + lexicon + lint/show  · CI  · **DONE** (18 tests, full suite 1689 green)
- [x] `spec.rs` — `BookArtSpec` + sub-structs, permissive serde (all `Option`, string
      enums, `type`→`kind` rename, unknown-key tolerant), `from_hjson`/`load`. **HJSON
      gotcha: quote string values in inline objects/arrays (deser_hjson quoteless
      strings run to EOL) — specs/tests use JSON-style quoting.**
- [x] `lexicon.rs` — built-in origin/technique/ornament vocab + defaults (origin
      scaffolds, technique binarisers, per-type default tier/symmetry, nearest-match).
      (`assets/bookart/lexicon.hjson` override loading deferred — built-in is the
      always-present fallback.)
- [x] `geometry/page.rs` — named sizes → exact px @ DPI (A4@300 = 2480×3508), custom,
      orientation. (Text-block/margins → B2.)
- [x] `compile.rs::resolve(spec) → RenderPlan` — fills lexicon defaults, resolves tier
      (auto→concrete), symmetry, canvas, finisher (transparency mode + binariser), and
      the diffusion prompt/negative with **anti-text + anti-colour baked in**. **Pure,
      byte-stable.**
- [x] `lint.rs` — schema/vocab(nearest-match)/ranges/page/ornament-xor-kit. `cli/
      bookart.rs` `new`/`lint`/`show` wired into the CLI (`Command::Bookart`).
- [x] Goldens: `resolve` byte-stable + a golden russian-woodcut-headpiece prompt;
      determinism test; page px goldens.

### B1 — finisher (transparency + binarizers) + scorecard  · CI  · **DONE** (11 tests, full suite 1700 green)
- [x] `finish/alpha.rs` — `luminance` (G0.5 defaults `white_cut 0.07 / γ 0.70`),
      `threshold` (soft ramp), `fade` (edge falloff); `parse_tint` (black/white/sepia/
      #hex) recolour. **Pure**, tested. (`matte` falls back to `luminance` — real
      U2Net matte is wired at render time, B5.)
- [x] `finish/binarize.rs` — XDoG (self-contained separable Gaussian, no
      feature-gated deps), threshold-bold (+ink dilate), engrave-invert, Floyd–
      Steinberg dither, matte-solid, threshold-invert, Bayer halftone; Otsu; dispatch
      by lexicon binariser name + `ink_weight` bias. **Pure, deterministic.**
- [x] `finish/mod.rs` — `finish_ornament(raw, plan) → RgbaImage` (to_luma → binarise →
      transparency). `scorecard.rs` — chroma-purity / alpha-clean / symmetry-RMS /
      ink-coverage / resolution probes + pass/fail (**pure**). (stray-glyph OWL-ViT +
      aesthetic LAION deferred to render wiring; safe-area needs the B2 text-block.)
- [x] `bookart verify <spec> --image [--out] [--finished]` — finish a render + score.
- [x] Goldens: transparency math (ink opaque / paper transparent / tint), every
      binariser deterministic, scorecard pass/asymmetry/blank cases.
- **Milestone MET:** the always-on transparent-finish is proven — verified live on
      the real russian-LoRA render: chroma **0.000**, alpha-halo **0.000**, ink 0.260;
      the scorecard correctly isolates **symmetry RMS 0.527 (FAIL)** as the only defect
      → exactly what the B2 symmetry engine fixes (the finisher provably can't).

### B2 — geometry: page + layout + symmetry  · CI  · **DONE** (8 tests, full suite 1708 green)
- [x] `geometry/page.rs` — named sizes → px @ DPI (landed in B0).
- [x] `geometry/layout.rs` — `text_block` (margins+gutter, book defaults) + `layout_for`
      per type: headpiece top-band, tapering tailpiece, thin divider, centred
      fleuron/dinkus, border/endpaper full-block, **4 inward-flipped corners**, initial
      cell, page-fill frontispiece, centred vignette/colophon. Golden.
- [x] `geometry/symmetry.rs` — `symmetrize`: **bilateral** (mirror-union, exact) +
      **radial:N** (N-fold rotational union, bilinear) + frieze/none passthrough. Golden:
      bilateral RMS→0, radial replicates the domain.
- [x] `finish/canvas.rs` — `place_on_canvas` (resize + flip + alpha-over onto the exact
      page canvas) + `save_png_dpi` (**pHYs DPI** via the `png` crate). Golden.
- [x] `bookart verify --symmetrize --page` wires both into the CLI.
- **Milestone MET — the primary output contract is complete.** Live on the russian-
      LoRA render: `--symmetrize` drove **symmetry RMS 0.527 → 0.000 (PASS)** (the
      finisher's one blind spot, fixed); `--page` produced a **1748×2480 A5 @300 DPI**
      transparent headpiece band, "resolution matches page". Transparent **+** exactly
      page-sized **+** symmetric.

### B3 — procedural ornament generators  · CI  · **DONE** (3 tests, full suite 1711 green)
- [x] `procedural/mod.rs` — **self-contained, vector-native** parametric generators
      (NOT fractal-engine-coupled — `fractals` is off under `--no-default-features`, so
      coupling would break the CI gate; the audit's fractal reuse becomes an optional
      feature-gated enhancement later). Curve primitives: circle, rhodonea **rose**,
      **hypotrochoid guilloché**, line. Ornaments: **rosette** (ring + rose + counter-
      rose), **divider** (central rosette + mirrored tapering lines + end caps),
      **border** (nested rects + bead-and-reel run + 4 corner rosettes), **corner**
      (L + guilloché scroll). Emit **born-vector `Polyline`s** (ready for B6 SVG) →
      `rasterise` strokes to clean antialiased line art. Deterministic, golden.
- [x] `finish::finish_procedural` (skip binarise — procedural is born-clean → straight
      to transparency). `bookart render <spec> --out` — **procedural tier live e2e**
      (spec → generate → finish → symmetrise → place on page canvas → DPI PNG); bails
      with a clear message on diffusion/composite (B4/B5).
- **Verified live**: `border` → a publication-quality A5 frame (nested rects, even
      bead run, 4 six-fold guilloché corner rosettes, crisp + exactly symmetric, zero
      weights); `fleuron` → a clean 12-fold rosette; `divider`, `corner` (4 pieces) OK.

### B4 — diffusion path: generic line-art + 2–3 origin LoRAs  · GPU  · **DONE** (live-verified)
- [x] `bookart render` diffusion tier — compile prompt/negative → `api::Generate`
      (sd15) → technique finish (binarise + transparency) → symmetrise → place on page.
      `gen_size` derives a /8-snapped working res from the layout rect aspect.
- [x] **Origin-LoRA loader** — non-`generic` origins attach the hosted LoRA via the
      `repo#file` source form `vulogov98/plakat-bookart#<origin>-sd15.safetensors`
      (hf-hub resolves + downloads) + weave the `bookart_<origin> style` trigger.
      **`generic` origin = the LoRA-free fallback** (R3). Hosting convention: repo
      `vulogov98/plakat-bookart` (public), `<origin>-sd15.safetensors`.
- [x] Train + host russian/english/japanese LoRAs — done in G0.3.
- **Verified live**: russian vignette → hosted LoRA resolved from HF (128/128 merged),
      512² diffusion → a bold Bilibin-woodcut firebird-on-oak vignette, finished
      transparent + placed on A5 @300 DPI. **Deferred:** lineart-ControlNet (needs a
      structure source → composite/img2img, B5); `bookart origins` listing;
      `assets/bookart/lexicon.hjson` override.

### B5 — composite tier + render router  · GPU  · **DONE** (live-verified)
- [x] **Composite tier** — `procedural::frame(symmetry, w, h)` (nested rects + 4 corner
      rosettes + an inner **window** rect) + a diffusion picture generated at the window
      size, finished as clean **line art** (forced `xdog` + luminance so paper stays
      transparent — never a solid slab), inlaid into the window, frame overlaid on top.
      The persona geometry+detail-composite analog. Symmetry skipped for composite (frame
      already symmetric; the picture is a scene, not mirror-doubled).
- [x] **Render router** — `run_render` dispatches all three tiers (procedural/diffusion/
      composite) off `plan.tier`; `diffuse()` helper shared by diffusion + composite.
- [x] **U2Net matte mode** — `transparency: matte` → `matte_silhouette` (`pipelines::
      matting::matte` → subject mask → solid tint silhouette). Wired; **U2Net is trained
      on natural images so B/W-ornament matte quality is unverified** — a silhouette-
      technique convenience, not the primary path.
- **Verified live**: composite russian vignette → a crisp procedural frame (nested rects
      + 4 guilloché corner rosettes) with a Bilibin firebird line-art inlay in the window,
      transparent + A5 @300 DPI. Publication-quality cartouche.

### B6 — single-ornament end-to-end + opt-in vectorize  · GPU  · **DONE — v1 MILESTONE** (2 tests, suite 1713 green)
- [x] `cli/bookart.rs` — shared `do_render` core (used by `render` + `illustrate`);
      **`bookart illustrate "<prompt>"`** synthesises a diffusion-tier spec (frontispiece/
      vignette) for a quick standalone B/W plate.
- [x] `finish/vector.rs` — **born-vector SVG** for the procedural tier: serialise
      `generate_paths` polylines (transformed to page rects, corner flips) → print-sized
      SVG (physical `mm` + px `viewBox`, stroked). `render --svg` / spec `output.formats`.
      **Raster→SVG *trace* (diffusion/composite) deferred** — `vtracer` pulls an old
      `image 0.23` stack for a secondary feature; the born-vector procedural SVG is the
      high-value path, the PNG is always the deliverable (§7.5).
- [x] Rejection sampling — `render/illustrate --attempts N` (diffusion tier): try up to N
      seeds, keep the first that clears the scorecard, else the fewest-issues one.
- **Verified live**: border `--svg` → 120-path A5 born-vector SVG (148×210 mm, viewBox
      1748×2480); `illustrate "wolf in a snowy pine forest" --origin japanese` → a clean
      Hokusai-style ukiyo-e line-art landscape plate. **v1 = single-ornament render across
      all 3 tiers + illustrate + born-vector SVG.**

### B7 — kit: motif DNA + coherence  · GPU  · **flagship pt 1**
- [ ] `kit.rs` — `KitSpec`, seed lineage, motif threading into every ornament,
      generate the set, **kit-coherence** score (CLIP-embedding pairwise + ink-weight
      consistency), re-derive below threshold. Output dir + contact-sheet PDF.

### B8 — manuscript set + proof + font  · GPU  · **flagship pt 2**
- [ ] `manuscript.rs` — parse Markdown / plain chapter-list (EPUB fast-follow) →
      per-chapter matched set (seed-varied headpiece variation, tailpiece, first-letter
      initial, frontispiece) → `manifest.json` + optional LaTeX/InDesign includes.
- [ ] `bookart proof` (contact sheet / page proof); `bookart font` (fleurons → OTF,
      fast-follow candidate).

### B9 — repair/edit loop + lineage  · mixed
- [ ] `edit.rs` — class-aware repair (ink-weight → post-only; size/DPI → re-raster;
      motif/origin → re-gen). Lineage: variant kit, origin blend.

### B10 — integration + docs + corpus  · **cut**
- [ ] scenario `type: bookart`; compile `bookart:` directive; Bund `plakat.bookart.*`;
      library API `BookArt` builder; `doctor` capability section (origins/techniques/
      tracer/LoRA presence — [[feedback_capability_doctor]] rule).
- [ ] Docs: `BOOKART.md`, `Tutorials/BOOKART_TUTORIAL.md`, `BOOKART_STYLES.md`,
      `BOOKART_TRANSPARENCY.md`; docs-index + README announcement.
- [ ] `./corpus`: 2–3 authored specs + a kit + a manuscript demo + a driver.

---

## Sequencing rationale

- **G0 → B0 → B1 → B2 first** gets the full *non-generative* spine — spec resolves,
  and any raster (even a placeholder) is transparented + page-sized + scored — under
  CI without a GPU, exactly as persona P0–P3 were. Quality is falsifiable from render
  one.
- **B3 before B4** ships value (all geometric ornament) with zero weights and zero
  model download; the diffusion half is additive on top.
- **B6 is the v1 line.** Everything after is the flagship (kit/manuscript) and polish.
- **The generic line-art path (B4) is the safety net:** if the origin LoRAs
  underperform (G0.3), v1 still ships every origin×technique via prompt + lineart CN +
  XDoG. LoRAs raise the ceiling, they are not on the critical path.

## Release-flow reminders (from auto-memory)

Bump `Cargo.toml` **+ `Cargo.lock`** in sync (`--locked` CI); gate =
`cargo test --no-default-features --lib`; new capability must surface in `doctor
--capability`; no Claude/Anthropic co-authoring; FF `main` + tag → 6-asset CI
release + crates.io; `gh release edit` for notes (owner token — the keyring alt has
read-only).

## Open questions carried from the RFC (§20)

Font-export timing · manuscript input formats (md+list v1, EPUB later) ·
`illustrate` kit-fit vs standalone · tone/halftone in v1 · per-type radial defaults.
