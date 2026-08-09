# Research spike — ETCH-1 L2 DDIM-inversion detection

**Question:** can `doctor --if-plakat --verify` *read back* the L2 Fourier-ring mark that 6.7.0 already
*writes* into `z_T` — i.e. is DDIM-inversion detection worth building?

**Status:** spike (feasibility mapped, not built). **Recommendation: defer** — feasible, but a real
feature with uncertain payoff; L3 already covers the case it targets.

---

## What the spike found

The detection pipeline is: `image → VAE-encode → z_0 → DDIM-invert (T UNet steps) → z_T' →
correlate_rings(z_T', key) → presence + id prefix`. Every piece was located; none is a blocker, but
together they're a careful ~150–250-line **in-crate** build:

| piece | status | note |
|---|---|---|
| VAE encode | ✅ available | `SdCore.vae: Arc<AutoEncoderKL>` is public; `AutoEncoderKL::encode(x) → DiagonalGaussianDistribution`, `.sample()` / `.mean`, × `SdCore::vae_scale()`. |
| UNet forward | ✅ available | `SdCore.unet: SdUNet`, `SdUNet::forward(xs, t, hidden, add_text_embeds, add_time_ids)` is public. |
| Empty-prompt embeds | ⚠️ in-crate only | `Pipeline::encode_prompt` is `pub(crate)` — so the detector must live **inside the crate** (a `Pipeline` method / a `doctor` arm), not an `examples/` probe. SDXL also needs `build_add_time_ids_base`. |
| Alpha schedule | ⚠️ must reconstruct | the scheduler exposes `timesteps()`/`init_noise_sigma()` but **not** `alphas_cumprod`. Rebuild from SD's `scaled_linear` betas (`beta_start 0.00085`, `beta_end 0.012`, 1000 steps) → cumulative products. ~5 lines, but a place to get subtly wrong. |
| DDIM inversion loop | ➖ to write | standard deterministic inversion (unconditional, guidance = 1); ~30 lines + the ring correlation (the codec exists: `etch::latent::correlate_rings`). |
| The codec | ✅ done | `embed_rings`/`correlate_rings` shipped + tested in 6.7.0 (round-trip recovers presence ~1 + the id prefix). |

## Why defer (the honest cost/benefit)

1. **Correctness-sensitive.** A subtly wrong DDIM inversion or alpha schedule yields a *meaningless*
   presence number — the measurement is only useful if the inversion is right, so this can't be a
   throwaway. It's a feature to build carefully, not a quick probe.
2. **Model-coupled + RFC-weak.** The RFC itself rates L2 detection "strong for plakat's own img2img,
   moderate for SD 1.5, weak across families," with ~16-bit capacity. SDXL's inversion also needs the
   pooled + time-id conditioning; SD3 / Flux / Cascade have different latent geometries (unsupported).
3. **L3 already covers the target case.** The scenario L2 is *for* — recovering origin through a
   generative edit — is handled by L3 today, **live-proven**: a rescaled + metadata-stripped copy (L0 +
   L1 gone) still matched semantically → the exact origin id → `probable-derivative`. L2 would add a
   second, weaker, heavier signal for the same case.

## If/when it's built (the groundwork, so it's a clean start)

A `Pipeline::l2_invert_and_correlate(image_path, key) -> (presence, EtchId)`:
1. Load the pipeline for the image's model (from the L0 manifest's `model`, or a `--model` flag).
2. `z0 = vae.encode(preprocess(image)).sample()? * vae_scale`.
3. Build `alphas_cumprod` from the `scaled_linear` betas; pick the inversion timestep subset.
4. `(hidden, pooled) = encode_prompt("", ...)`; SDXL `add_time_ids = build_add_time_ids_base(...)`.
5. DDIM inversion loop: `for t asc: eps = unet.forward(z, t, hidden, pooled, ids); z = ddim_invert_step(z, eps, ᾱ_t, ᾱ_{t+1})`.
6. `correlate_rings(z, key)` → presence; L2 present iff presence > τ **and** the recovered prefix
   matches `latent::key_tag(key)`'s prefix.
Wire behind `doctor --if-plakat --verify` (the model-load escape hatch); report `unsupported` for
non-SD1.5/SDXL. Validate by generating a `--etch` image and checking etched-vs-control presence
separation before trusting the threshold.

**Bottom line:** the codec + write shipped in 6.7.0; the reader is a well-scoped but real follow-on whose
value is marginal over L3. Build it only if native L2 verification is specifically wanted — otherwise the
graded verdict already degrades correctly without it.
