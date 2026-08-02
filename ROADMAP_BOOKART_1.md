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
- [ ] **G0.2 Baseline harness.** `tools/reference/bookart_baseline.*`: naive "grey +
      `replace-bg`" control over origin×technique × N seeds → chroma-leak, alpha-halo,
      symmetry-RMS, stray-glyph, vectorability. **Metric fns are pure (write in B1);
      the naive-control generation is a scheduled GPU job.** Commit a corpus entry.
- [ ] **G0.3 Origin-LoRA feasibility.** Source PD illustration corpora (pre-1929
      scans: Bilibin, Beardsley/Rackham, Hokusai *manga*); smoke-train ONE LoRA via
      `style train` and confirm it beats the generic line-art path on B/W-line
      cleanliness. **GPU + corpora job — schedule.** If it does not beat the generic
      path → generic path is the v1 default, LoRAs slip to fast-follow.
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

### B0 — spec + resolver + lexicon + lint/show  · CI
- [ ] `spec.rs` — `BookArtSpec` + sub-structs, permissive serde (all `Option`, string
      enums, untagged colour, unknown-key tolerant), byte-stable.
- [ ] `lexicon.rs` + `assets/bookart/lexicon.hjson` — origin/technique/motif presets,
      per-ornament defaults, aspect/symmetry defaults.
- [ ] `compile.rs::resolve(spec, lexicon) → RenderPlan` — layout params + tier
      decision + finisher chain + canvas spec + (for diffusion) prompt/negative with
      anti-text baked in. **Pure.**
- [ ] `lint.rs` + `cli/bookart.rs` `new`/`lint`/`show`.
- [ ] Goldens: `resolve` byte-stable; `show` output stable.

### B1 — finisher (transparency + binarizers) + scorecard  · CI (matte/glyph lazy-GPU)
- [ ] `finish/alpha.rs` — `luminance` (γ from G0.5), `threshold` (soft ramp), `fade`;
      `tint` recolour; premultiplied option. **Pure**, golden on synthetic ramps.
- [ ] `finish/binarize.rs` — XDoG, adaptive-threshold, error-diffusion dither,
      engraving invert. **Pure.**
- [ ] `matte` mode → delegate to `pipelines/matting.rs` (lazy U2Net).
- [ ] `scorecard.rs` — chroma-purity / alpha-clean / symmetry-error / ink-weight /
      resolution-DPI / safe-area probes (**pure**); stray-glyph (lazy OWL-ViT),
      aesthetic (lazy LAION).
- [ ] Goldens: transparency math; scorecard on synthetic images.
- **Milestone:** the always-on transparent-finish is proven before any generation.

### B2 — geometry: page + layout + symmetry  · CI
- [ ] `geometry/page.rs` — named sizes (A4/A5/A6/B5/Letter/Legal/Trade/custom) → px at
      DPI, text-block from margins+gutter, bleed. Golden.
- [ ] `geometry/layout.rs` — per-type canonical layout vs text-block (headpiece band,
      tapering tailpiece, corner-L, border assembly, initial cell, divider/fleuron).
- [ ] `geometry/symmetry.rs` — fundamental-domain compute + replicate
      (bilateral/radial:N/frieze) + mirror-blend. Golden on a synthetic domain.
- [ ] `finish/canvas.rs` — composite finished ornament onto exact page canvas; write
      pHYs DPI + physical-mm.
- **Milestone:** primary output contract complete — transparent **and** exactly
  page-sized PNG.

### B3 — procedural ornament generators  · CI
- [ ] `procedural/{border,corner,rosette,knot,guilloche}.rs` — reuse
      `fractals::{compose,coloring,control_source}`; emit raster **and** born-vector
      paths. Deterministic, golden.
- [ ] Router `procedural` tier live end-to-end (spec→procedural→finish→PNG[/SVG]).

### B4 — diffusion path: generic line-art + 2–3 origin LoRAs  · GPU (first weights)
- [ ] `render.rs` diffusion tier — compile prompt/negative → `api::Generate` +
      **lineart ControlNet** → XDoG finish → alpha. Generic path first (no LoRA).
- [ ] Load origin LoRAs (`assets/bookart/origins/*.safetensors`) via existing LoRA
      stacking; `bookart origins` lists origin×technique (reuse `style` catalog shape).
- [ ] Train russian/english/japanese LoRAs (`style train`, G0.3 corpora).

### B5 — composite tier + render router  · GPU
- [ ] `render.rs` — router picks tier from ornament type + motif kind; **composite** =
      procedural frame (B3) + diffusion inlay (B4) alpha-composited into the frame
      window (the persona geometry+detail-composite analog). `--tier`/`tier:` override.

### B6 — single-ornament end-to-end + opt-in vectorize  · GPU  · **v1 milestone**
- [ ] `cli/bookart.rs` — `render` / `illustrate` / `vectorize` wired end to end.
- [ ] `finish/vectorize.rs` — opt-in SVG: born-vector (procedural) + trace (diffusion,
      G0.1 tracer); EPS/PDF.
- [ ] Rejection sampling via scorecard (`--min-score`/`--attempts`).

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
