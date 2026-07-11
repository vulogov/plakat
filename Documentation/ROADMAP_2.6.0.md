# plakat 2.6.0 — roadmap (planning)

On the [road to 3.0](ROADMAP_TO_3.0.md). Theme: **finish the SD UNet SDPA thread (the workhorses)
+ grow the image-quality track**, with high-res/control quality as the 2.6 flavour. Scope with the user.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Carried from 2.5 (finish these first)

- [~] **SD UNet SDPA — step 2 (the workhorses)** *(in progress next)*. Route main SD1.5/2.1/SDXL
      generation through plakat's own `sd_train::unet` (which has the 2.5 SDPA attention) instead of
      candle's `SdxlUNet2DConditionModel`. Signatures already match (`forward`/`forward_sdxl`);
      touches the `SdUNet` enum + LoRA + ControlNet + motion + weight-load keys. **Verify-gated**:
      switch → `plakat verify` → corr 1.0 confirms UNet equivalence (corr < 1.0 = mismatch to debug;
      bail cleanly if so). Also decouples plakat from candle's churny surface.
- [ ] **SD3 MMDiT PAG** + a **`--pag-scale` CLI flag** (promote PAG off the env knob).
- [ ] **FreeU** + **CFG-rescale** / dynamic thresholding (finish free-quality guidance).

## 2.6 flavour — high-res & control quality

- [ ] **ControlNet-Tile + diffusion tiled-upscale (SUPIR-lite)** — coherent 512→2K/4K with
      hallucinated detail; the missing control kind.
- [ ] **Face/detail restoration** (GFPGAN / CodeFormer) + better upscalers.
- [ ] **Aesthetic scoring** (CLIP predictor → rank generations) — feeds the 3.0 manager's curation.

## House-keeping

- [x] **Open 2.6.0** — branch off `main` (2.5.0 release), version bump `2.5.0 → 2.6.0`.
