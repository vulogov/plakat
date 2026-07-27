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

## Phase 2 — Sana inpaint (`plakat img2img --model sana --mask …`) — DONE

- [x] Mask-aware img2img: load the mask → latent-space mask (32× downsample, `Mask::to_latent_tensor_factor`);
      RePaint-style — after each denoise step, blend the **known** region back from the (flow-noised) init
      latent (`known = (1-σ)·z0 + σ·noise`) so only the masked region is regenerated. Reuses the 4.6 img2img
      start + DC-AE encode.
- [x] Drop the `--mask` bail in the Sana img2img arm; wire mask feather/invert; strength defaults to 1.0
      for inpaint (vs 0.6 for plain img2img), picked in one place (`sana::generate_all`).
- [x] **Metal DC-AE encode bug found + fixed** (blocked inpaint AND 4.6 img2img preservation on Metal):
      candle 0.10.2 Metal returns garbage for `mean` over a non-trailing axis of a **rank-5** tensor
      (diverges ~1e26; rank-≤4 fine — `examples/dcae_metal_probe.rs`). The encoder's two group-average
      shortcuts used exactly that → encode returned black on Metal. Replaced with a Metal-safe rank-3
      `group_mean`. CPU DC-AE corr verify still passes; the naive/fixed forms match at max|Δ|=0.
- [x] Verify (Metal): round-trip mean|Δ| 124→**10.7** (proper 32× AE floor); inpaint preserves OUTSIDE the
      mask (7.7, ≤ AE floor) and repaints INSIDE (22.3) — a glowing full moon filled the masked ellipse,
      the rest of the townsquare byte-preserved.

## Phase 3 — docs + release

- [x] GENERATE tutorial: Sana-1.5 model row + inpaint section (with the 32× coarse-boundary note);
      README banner + "what's new in 4.7.0" (Sana-1.5, inpaint, Metal encode fix). `reference_sana`
      memory updated (4.7 section + the rank-5-mean Metal gotcha).
- [x] Cut the 4.7.0 release — v4.7.0 @ 16bc1d4: tag pushed → Release CI green (6 assets + SHA256SUMS),
      `cargo publish --locked` (on crates.io), main fast-forwarded, notes set via `gh release edit`.

## Notes / risks

- Sana-1.5 shares the DC-AE + Gemma-2 + base DiT arch (20 layers / 2240 hidden); only `qk_norm` (+
  `guidance_embeds:false`, already handled) differs. Verifying needs the ~3.3 GB SANA1.5 transformer.
- Inpaint mask is at the 32× DC-AE latent resolution (coarse); document the granularity.
- Still deferred: Sana LoRA (no real LoRAs), outpaint, ControlNet.
