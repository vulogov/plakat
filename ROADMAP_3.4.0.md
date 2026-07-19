# plakat 3.4.0 — roadmap (shipped)

**Shipped — a huge non-AI "full studio" cut.** Creative looks (invert/sepia/duotone/posterize/
solarize/threshold); a deep filter library (oil-paint ×10, watercolour ×10, ink Euro/JP/CN/RU,
pencil/charcoal/cartoon/emboss/halftone/pixelate/blur/bloom, false-colour thermal/IR/night-vision)
all **adjustable 0–100 %**; look presets (vintage/lomo/cross-process/noir/pop-art/golden-hour/
old-photo/daguerreotype + Apple vivid/dramatic/mono/silvertone). Framing: keystone/perspective,
border/letterbox, circle crop, watermark/caption (font selection), `.cube` LUT. Workflow:
take/duplicate/put-back working sub-albums. Composites: panorama + collage. Perf: cached
working-resolution preview (base caching). See README for the cut.

Opening the cycle after 3.3.0 (the photos "pro darkroom" — non-AI editing/analysis + a full
tree-pane album manager). `plakat photos` is now a deep, non-destructive editor; 3.4 has room to
round out the remaining non-AI editing gaps, tackle the deferred quality/perf debt, and finish the
distribution loop. Candidate tracks below — narrowed with the owner before build.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Track A — remaining non-AI editing

- [ ] **Perspective / keystone correction** — straighten converging verticals (architecture).
- [ ] **Border / letterbox to aspect**, **circle / rounded-corner crop** → transparent PNG.
- [ ] **Watermark / caption / timestamp burn-in** — expose the portfolio watermarker as an edit.
- [ ] **LUT (`.cube`) apply** — load a standard film-look LUT (trilinear interp), portable, non-AI.
- [ ] **HSL per-band** — hue/sat/lum of 8 colour ranges (the full selective-colour).
- [ ] **Bilateral / NLM denoise** — edge-preserving, better than the median despeckle.
- [ ] Quick creative ops — duotone / gradient-map, posterize, threshold, solarize, invert, sepia,
      Kelvin white balance; **gray-point white balance** + an **eyedropper** colour sampler.
- [ ] **Multi-shot (advanced)** — HDR exposure blend / focus stacking across a burst (align + blend).

## Track B — performance & big images

- [ ] **Working-resolution preview** — the interactive modes re-derive from the pristine original on
      every keypress; bind full-res only on apply/flatten so 24 MP+ stays snappy (also bounds the
      top-bar histogram + analysis recompute).
- [ ] Carry-throughs from the 2.4 performance pass (step-caching, attention, VAE, weight-load).

## Track C — Phase 7 (vision, the manager's own) & verification

- [ ] **Face-scan** — detect/group faces across the library (SCRFD/ArcFace already in-tree).
- [⏸] **CLIP visual-search live-verify** — blocked on the external HF cache disk; when reconnected:
      `cargo test --features photos -- --ignored clip_loads_and_embeds`.

## Track D — distribution

- [ ] **Publish 3.3.0** — `cargo publish` (crates.io) + GitHub release assets (deferred at the cut).
- [ ] Merge 3.3.0 → `main`.

## Ground rules (unchanged)

- Non-destructive by default; per-album `album.hjson` authoritative + additive.
- The `:` planner stays inside the closed, album-scoped vocabulary — `export`/`convert` the only
  create-only outward writes; no external read, no exec.
- Verify-safe: default CLI image output byte-identical; new work lands with unit tests.
