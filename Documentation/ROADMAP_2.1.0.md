# plakat 2.1.0 — roadmap (planning)

2.0.0 shipped **`plakat verify`** — a self-contained, pure-Rust, CI-gated model-correctness
harness (Tiers 0/1/2; all 7 families' conditioning + denoiser/transformer cores verified vs a
diffusers reference at corr 1.0; goldens hosted on `vulogov98/plakat-verify`). It already
caught + fixed a real SDXL bug (CLIP-L pad token). See `RFC_VERIFY.md` / `VERIFY.md`.

2.1.0 opens as a **planning landing pad** — scope is decided with the user, not pre-committed.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Now-unblocked by the verify harness

The harness exists specifically so these can be done *safely* — each has a reference-comparison
gate to prove it doesn't regress a currently-correct model.

- [x] **T5 pad attention mask** (BUGFIX 1.2, carried from 2.0.0) — PixArt/SD3 encoded captions
      without masking pad tokens; measured at corr **0.70** vs the correct masked output on the
      real-token rows (a genuine image-affecting bug). Fixed: `vendored_t5::forward_with_mask`
      threads a `(B,1,1,L)` pad bias into the encoder self-attention; PixArt now routes T5
      through the vendored copy, SD3 derives the mask from `(ids != 0)`. New `t5.hidden` capture
      point on both → **corr 0.70 → 1.00000** vs diffusers. *(T5 SELF-attention done; DiT/MMDiT
      CROSS-attention masking — image tokens not attending to pad caption positions — is the
      remaining half, below.)*
- [x] **DiT cross-attention pad mask** — the second half of the T5-mask fix. Threaded the caption
      mask into PixArt's DiT cross-attention (`MultiHeadCrossAttention::forward_masked` +
      `caption_mask_to_bias`; `encode_prompt` returns `(hidden, mask)`; `generate` CFG-batches it).
      `dit.block0` now carries a deterministic mask → corr 1.0 (max_abs 0.0563→0.0278, the mask
      changes the output and still matches diffusers `encoder_attention_mask`). **SD3 needs
      nothing** — diffusers applies no mask to the MMDiT joint attention (verified from source);
      adding one would break correspondence. **The full T5 pad-mask fix (self + cross) is done.**

## Verify follow-ups (breadth / depth)

- [x] **Tier-1 breadth** — added a second fixture `still_life_v1` (a much shorter prompt → heavier
      padding) + multi-fixture Tier-1 iteration (`fixtures::all()`, `run_model` per fixture, missing
      pairs skip). Every prompt-dependent tap now verified at two prompt lengths across all 6
      non-trivial models (animatediff's `motion.block0` is prompt-independent). All corr 1.0 — no
      new bug, broader coverage. A third fixture (attention syntax / BREAK / special chars) is a
      cheap future add.
- [x] **Tier-2 breadth** — extended the end-to-end perceptual gate to **SDXL + PixArt** (was
      sd15 only). Generalized `tier2::run_model` with a per-model `GenSpec` + a `Family` dispatch
      (t2i pipeline vs PixArt's own `generate`); wired `PLAKAT_VERIFY_DET_INIT` into PixArt's
      noise init. Frozen golden PNGs authored for all three; SSIM 1.0 / mean_abs 0.0 each. Now
      three models across two pipeline families.
- [~] **Deeper taps** — `t5.hidden` ✅ (the pad-mask fix), `adaln.embedded_timestep` ✅ (PixArt
      final-adaLN input, corr 1.0). **Finding: a full-DiT `dit.out` tap is NOT viable** — PixArt
      is a pure transformer, so a forward on a white-noise deterministic latent is severely OOD
      (corr 0.64 vs diffusers; block0 is 1.0). Unlike SD's conv-heavy `unet.out` (1.0), the DiT
      full-forward can't be verified with synthetic input — dropped, documented. **SD 1.5
      `unet.mid`: decided AGAINST** — candle's stock SD UNet has private blocks, so it needs
      vendoring the whole UNet (a load-bearing swap of a working backbone) for
      localization-only value, since `unet.out` already verifies SD 1.5's full UNet at corr 1.0.
      Not worth it. Cascade attention (OOD) remains a theoretical open item.
- [ ] **Phase 4 hardening** — the opt-in `verify-models` CI job downloads multi-GB weights;
      a smaller cached/quantized fixture model would make a full-correctness gate cheaper.

## Other candidate themes (deferred / secondary, carried from 2.0.0)

- [ ] **Library / API stabilization** — a documented, semver-stable Rust crate API.
- [ ] **Performance pass** — profile hot paths (VAE decode, attention, weight load).
- [ ] **Flux in the UI** — unblock when capable hardware is available.

## House-keeping

- [x] **Open 2.1.0** — branch off `main` (2.0.0 release), version bump `2.0.0 → 2.1.0`.
