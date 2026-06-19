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
      proofs in `corpus/images/embedding-train/{sd15,sd21}/`).
      **SDXL (dual-encoder) done**: learns a CLIP-L 768d + CLIP-G 1280d vector pair
      (splice into both encoders, penultimate-L ⊕ penultimate-G + CLIP-G pooled —
      bit-identical to inference `embed_xl`); saved as a dual `clip_l`+`clip_g` TI,
      applied via the existing v0.31 dual-encoder load path. `--base sdxl`.
- [ ] **Compose `generate:` / inline `matte:` layers** — *(M)* render a layer
      inline, or U2Net-matte a layer on the fly, inside `plakat compose` (today's
      layers are `load`-only; pre-render / pre-matte for now). GPU.
- [ ] **Flux regional prompting** — *(M)* `--region` for Flux (today it bails).
      Flux's flow-matching transformer needs its own per-region velocity blend.
- [x] **sd35 DreamBooth — DONE (render verification pending).** Prior preservation
      ported to the SD3.5 MMDiT trainer: the class loss uses the rectified-flow
      objective (`v=ε−x₀`, independent class σ/noise, λ-weighted), mirroring the
      SD/SDXL DreamBooth. `--base sd35 --class-dir/--class-prompt --prior-weight`.
      Driver `corpus/dreambooth_sd35.sh` (subject/class sets generated on a light
      base; train 256², render 1024²/768²). SD3.5 is memory-heavy — render proof is
      memory-bound on 24 GB (carries the same debt as sd21/sdxl renders).
- [ ] **IC-Light relighting** — *(L, stretch)* relight composited artefacts so they
      sit in the scene's light, not just on it. SD 1.5-based; porting the model is
      the work.
- [ ] **SAM depth-band selection** — *(S)* `depth.rs` exists; "select by depth band"
      is a nearly-free extra mask source for `segment`.

## Verification debt (GPU)

- [ ] `regional.sh sdxl` / `sd35` (sd15 verified; sd35 likely OOMs 24 GB).
- [ ] `resume_train.sh` final render (resume verified; the render OOM'd — re-run with
      freed RAM to commit the proof image).

## Explicitly out of scope (still)

- Flux-on-Metal (candle GGUF kernel broken upstream); `plakat serve` HTTP daemon
  (its own cycle); additional model families.
