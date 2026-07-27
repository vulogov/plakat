# plakat 4.7.0 — roadmap: finish the Sana family

4.5/4.6 built + deepened Sana (DC-AE + Gemma-2 + Linear-DiT, DPM++ scheduler, img2img, 0.6B/512/2K
variants). 4.7.0 closes the two items 4.6 deferred: the **Sana-1.5** checkpoint (needs `qk_norm`) and
**inpaint**. Both build on the verified components; frozen paths stay byte-identical.

Ground rules: additive; each phase lands with a reference-corr or coherence check; `Cargo.lock` in sync.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## Phase 1 — Sana-1.5 (`qk_norm = rms_norm_across_heads`) — DONE

- [x] Implement `qk_norm`: an RMSNorm over the **full inner dim** (`heads·head_dim` = hidden) applied
      to q and k **before** the head reshape, in both `LinearSelfAttn` and `CrossAttn` (adds
      `norm_q`/`norm_k` weights `[hidden]` per block). Gate on the config value.
- [x] `sana_dit::Config`: replace the `qk_norm` bail with a flag; support `"rms_norm_across_heads"`,
      still bail on other qk_norm kinds. Determine the exact RMSNorm eps from the checkpoint.
- [x] Alias `sana-1.5` → `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` (registry + capability).
- [x] Verify: DONE — SANA1.5 DiT single-forward **corr 0.999998** vs a diffusers dump
      (`sana_dit_dump.py --repo …SANA1.5…`). RMSNorm eps 1e-5 (Attention default); 1.6B path unchanged
      (qk closure is identity when norm=None).

## Phase 2 — Sana inpaint (`plakat img2img --model sana --mask …`)

- [ ] Mask-aware img2img: load the mask → latent-space mask (32× downsample); RePaint-style — after
      each denoise step, blend the **known** region back from the (flow-noised) init latent so only the
      masked region is regenerated. Reuse the Phase-2 (4.6) img2img start + DC-AE encode.
- [ ] Drop the `--mask` bail in the Sana img2img arm; wire mask feather/invert like SD3.
- [ ] Verify: masked region preserved outside the mask; coherent fill inside; strength honored.

## Phase 3 — docs + release

- [ ] GENERATE tutorial: Sana-1.5 + inpaint notes; capability hint; README what's-new. Update
      [[reference_sana]]. Cut the 4.7.0 release.

## Notes / risks

- Sana-1.5 shares the DC-AE + Gemma-2 + base DiT arch (20 layers / 2240 hidden); only `qk_norm` (+
  `guidance_embeds:false`, already handled) differs. Verifying needs the ~3.3 GB SANA1.5 transformer.
- Inpaint mask is at the 32× DC-AE latent resolution (coarse); document the granularity.
- Still deferred: Sana LoRA (no real LoRAs), outpaint, ControlNet.
