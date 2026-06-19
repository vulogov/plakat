# plakat 1.1.0 — roadmap

1.0.0 shipped "compose & edit scenes" + "train your own everything" + the Metal
OOM guard. 1.1.0 finishes the threads left open, all SemVer-additive on the frozen
1.0 contracts.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## Carryovers from the 1.0.0 cycle

- [x] **Textual-Inversion training — SHIPPED & VERIFIED (SD 1.5 / 2.1 / SDXL).**
      `plakat embedding train` learns one token embedding from a few images, model
      frozen (differentiable splice into the vendored CLIP via `embed_tokens` +
      `forward_from_input_embeds`). ~0.1 s/step (single vector). Verified: stained-
      glass style set → `a sgwin cat` takes the look (`corpus/embedding_train.sh`,
      proofs in `corpus/images/embedding-train/{sd15,sd21,sdxl}/`).
      **SDXL (dual-encoder) done**: learns a CLIP-L 768d + CLIP-G 1280d vector pair
      (splice into both encoders, penultimate-L ⊕ penultimate-G + CLIP-G pooled —
      bit-identical to inference `embed_xl`); saved as a dual `clip_l`+`clip_g` TI,
      applied via the existing v0.31 dual-encoder load path. `--base sdxl`.
- [x] **Compose `generate:` / inline `matte:` layers — DONE & VERIFIED.** A
      `plakat compose` layer's pixels now come from one of `load:` (existing image),
      `matte:` (U2Net cutout on the fly), or `generate:` (t2i render inline, with
      optional `model`/`seed`/`steps`/`gen_size`). `matte`/`generate` render to a
      tempfile and read back, reusing the file-based pipelines; added `Request::simple`
      so callers don't hand-build the 45-field t2i Request. Verified end-to-end:
      generated beach backdrop + matted astronaut composited with no pre-made assets
      (`corpus/compose_generate_scene.hjson`, proof `images/compose/
      beach-generate-matte.png`). Light (sd15 512² + U2Net) — runs on CPU.
- [ ] **Flux regional prompting** — *(M)* `--region` for Flux (today it bails).
      Flux's flow-matching transformer needs its own per-region velocity blend.
- [x] **sd35 DreamBooth — DONE (render verification pending).** Prior preservation
      ported to the SD3.5 MMDiT trainer: the class loss uses the rectified-flow
      objective (`v=ε−x₀`, independent class σ/noise, λ-weighted), mirroring the
      SD/SDXL DreamBooth. `--base sd35 --class-dir/--class-prompt --prior-weight`.
      Driver `corpus/dreambooth_sd35.sh` (subject/class sets generated on a light
      base; train 256²). **Verified live:** training ran end-to-end (120 steps, class
      forward active) + the LoRA merges into the MMDiT correctly. **Render is
      CANNOT-VERIFY on this 24 GB box** — it OOMs at the LoRA-merge step even at 512²
      and even `--device cpu` (the full MMDiT+T5+merge won't fit alongside apps; the
      guard keys on system-wide pressure). Mechanism is identical to the proven
      sd15/sdxl DreamBooth, so it's code-complete; render proof awaits more RAM.
- [ ] **IC-Light relighting** — *(L, stretch)* relight composited artefacts so they
      sit in the scene's light, not just on it. SD 1.5-based; porting the model is
      the work.
- [x] **SAM depth-band selection — DONE & VERIFIED.** `plakat segment --depth-band
      LO,HI` (normalized depth, 1.0 = nearest) is a click-free extra mask source
      via Depth-Anything-V2; combinable with `--point` (intersect). Refactored SAM
      into `build_selection_mask` / `finish_mask` + pure `depth_band_to_mask` /
      `intersect_masks` (unit-tested). Verified end-to-end on the astronaut + a
      portrait (foreground vs far-background bands are depth-distinct) — light
      enough to run on CPU, so no memory wall. Proof: `corpus/images/segment/
      depth-foreground.png` (driver `segment.sh` stage 3).

## Verification debt (GPU)

- [ ] `regional.sh sdxl` / `sd35` (sd15 verified; sd35 likely OOMs 24 GB).
- [ ] `resume_train.sh` final render (resume verified; the render OOM'd — re-run with
      freed RAM to commit the proof image).

## Explicitly out of scope (still)

- Flux-on-Metal (candle GGUF kernel broken upstream); `plakat serve` HTTP daemon
  (its own cycle); additional model families.
