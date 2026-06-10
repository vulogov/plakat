# Integral artefact embedding — implementation plan

**Goal:** composited artefacts become an **integral** part of the generated scene
(matched scale, light, colour, grounding) — not cut & paste. Decided after the
first multi-artefact render came out as pasted stickers.

## Diagnosis (from the first render)

- **Scaling (bug).** `compositing.rs:127` — `target_h = zone_h × scale_fraction`,
  and `zones.rs` bands are each `h/4` tall (sky 0–h/4, middle h/2–3h/4, close
  3h/4–h). So nothing can exceed ~0.25·canvas → a foreground cottage is
  structurally capped tiny while a sky balloon looks fine.
- **Blend (structural).** `--artefact-blend` is a **low-strength, unguided**
  masked img2img: at 0.3 it barely touches the artefact (stays a sticker); raise
  it and nothing holds the shape (it dissolves); and it does nothing for colour
  or light. Composite + a wash can't be integral.

## Phases

### Phase 1 — Scaling: canvas-relative  *(this commit; GPU-free)*
- `target_h = canvas_h × scale_fraction`; zones drive **position** only.
- `natural_size_pct` now means *fraction of canvas height*. Re-tune both
  libraries (bundled `assets/…` + corpus `corpus/assets/…`) and the default.

### Phase 2 — Contact shadow (grounding)  *(GPU-free — DONE)*
- `draw_contact_shadow`: soft elliptical darkening at a ground-anchored artefact's
  base (gated on `anchor.y >= 0.75`), composited before the artefact. Photoreal,
  no generation — the biggest "sitting in the scene" cue.

### PHOTOREAL PIVOT (after the first re-paint render)
The generative re-paint (Phase 4) **stylizes toward illustration** ("anime, not
photo") — SDXL img2img always risks this. So for a *photo* result the strategy
shifts: lean on **classical, generation-free** cues (shadow ✓, colour transfer
↓) that can't stylize, and dial Phase 4 down to a *light seam-blend*
(strength 0.55 → 0.4) with a photoreal negative — not a relight.

### Phase 3 — Colour harmonisation  *(GPU-free)*
- Transfer the scene's per-channel colour statistics (mean/std, or a soft
  histogram match over the target region) onto the artefact before compositing →
  its palette sits in the scene instead of clashing.

### Phase 4 — ControlNet-guided re-paint (the core)  *(GPU — IMPLEMENTED, verifying)*
- **DONE in code:** the blend's existing `&[ControlRequest]` slot (was `&[]`) now
  carries a **canny of the composited canvas** (`artefact_blend.rs`): `ControlSpec
  { kind: Canny, from: composited_path, strength: 0.9 }` → `load_control_stack` →
  fed to `blend_latents_one`. Corpus strength raised 0.3 → 0.55. Pending a GPU
  render to judge integration (downloads the SDXL canny ControlNet on first run).
- Replace `--artefact-blend`: take **canny/depth of the composited canvas** → run
  ControlNet img2img at a *meaningful* strength (~0.4–0.6) over the artefact
  regions. ControlNet **holds the shape**; the denoise repaints the surface to
  match the scene's light/colour/texture. Reuses plakat's existing canny
  ControlNet (SDXL) + img2img. This is what makes it *integral*.

### Phase 5 — CLI + verify  *(GPU)*
- Default the new pipeline (shadow + harmonise + CN re-paint); flags for each +
  `--artefact-blend-strength`. Render the corpus proof; judge harshly that the
  artefacts read as part of the scene. Update COVERAGE/docs.

## Risks / decisions
- **Phase 4 is the core + heavy piece.** The shadow/colour (2,3) may be partly
  subsumed by a good re-paint — sequence is 1 → 4 → (2,3 if still needed).
- **Strength tuning** is the knife-edge: high enough to integrate light/colour,
  low enough + CN-constrained to keep identity. Per-artefact, not global.
- Canvas-relative scaling changes every `natural_size_pct`'s meaning — re-tune the
  bundled library + the compositor tests in the same pass.
- No single reference to diff against (unlike the matte) — judged visually.
