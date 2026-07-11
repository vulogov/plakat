# plakat 2.6.0 — roadmap (planning)

On the [road to 3.0](ROADMAP_TO_3.0.md). Theme: **finish the SD UNet SDPA thread (the workhorses)
+ grow the image-quality track**, with high-res/control quality as the 2.6 flavour. Scope with the user.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Carried from 2.5 (finish these first)

- [x] **SD UNet SDPA — step 2. DONE (c54ccc1).** Wired SDXL through `sd_train::unet::forward_sdxl`
      (+ `forward_sdxl_with_residuals` for ControlNet) and **flipped the default**: SD 1.5 + SDXL now
      run plakat's OWN SDPA UNet by default; `PLAKAT_CANDLE_UNET=1` reverts; SD 2.1 stays on candle.
      **Proven safe (own ≡ candle, 3 ways):** `sd15.unet.out` corr 1.00000 (max_abs 3e-4);
      `sdxl.unet.out` corr 1.00000 (max_abs 2e-4); and end-to-end on Metal the own UNet reproduces
      candle's SDXL trajectory *bit-for-bit* — own 0.9347 ≡ candle 0.9343 SSIM (see golden note below).
      SD 1.5 Tier-2 passes 0.9976. **Speed (per-step): SD 1.5 ~2.5% (708→690), SDXL ~8% (4297→3945).**
      Modest vs the DiTs' 1.2–1.9× — the SD UNet is conv-dominated (attention is a small slice even
      with SDXL's full head_dim-64 SDPA coverage). The durable win is **decoupling from candle's
      registry UNet/attention** (aligns with the candle-stability wariness).
      - [ ] **Pre-existing (NOT this change): the SDXL Tier-2 golden is stale.** Candle's SDXL UNet —
            the *old* default — also scores SSIM 0.9343 < 0.97 against the HF golden, so the gate was
            already red. Almost certainly predates the SDXL CLIP-L pad-token fix (3a28f67), which
            legitimately shifted SDXL conditioning (the `unet.out` golden uses fixed conditioning →
            still corr 1.0; end-to-end uses real conditioning → drifted). **Fix = re-author
            `sdxl/tier2/golden.png` on the HF dataset** (needs a dataset write — deferred to the user;
            I won't push to HF unprompted). Verify-harness truth, not a pipeline bug.
      - [ ] **Follow-up: exercise ControlNet + inpaint through the own UNet default** end-to-end
            (`forward_sdxl_with_residuals` / 9-ch inpaint conv_in) — wired but not yet run.
- [ ] **SD3 MMDiT PAG** + a **`--pag-scale` CLI flag** (promote PAG off the env knob).
- [ ] **FreeU** + **CFG-rescale** / dynamic thresholding (finish free-quality guidance).

## 2.6 flavour — high-res & control quality

- [ ] **ControlNet-Tile + diffusion tiled-upscale (SUPIR-lite)** — coherent 512→2K/4K with
      hallucinated detail; the missing control kind.
- [ ] **Face/detail restoration** (GFPGAN / CodeFormer) + better upscalers.
- [ ] **Aesthetic scoring** (CLIP predictor → rank generations) — feeds the 3.0 manager's curation.

## House-keeping

- [x] **Open 2.6.0** — branch off `main` (2.5.0 release), version bump `2.5.0 → 2.6.0`.
