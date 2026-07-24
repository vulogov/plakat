# plakat 4.5.0 — roadmap: Sana (1.6B @ 1024px)

**Add the Sana text-to-image family** — NVIDIA/MIT's linear-attention DiT — to plakat. This is the
**largest single-model port in the project's history**: Sana has *zero* candle support, so it means
implementing **three net-new neural components** from scratch, each verified against a diffusers
reference before the pipeline produces anything coherent:

1. **DC-AE** — a deep-compression autoencoder (32× spatial, 32 latent channels), *not* the VAE-KL every
   existing pipeline uses. The biggest build.
2. **Gemma-2-2B** — a decoder-only LLM used as the text encoder (not T5/CLIP). The smallest lift.
3. **Sana Linear-DiT** — 20-layer DiT with ReLU **linear** self-attention + GLUMBConv Mix-FFN +
   AdaLN-single. PixArt-Σ-analogous; linear attn is *simpler* than softmax to implement.

Plus a flow-matching scheduler (reuse SD3's) and the dispatch plumbing for a new family.

**Weights:** `Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers` (~13 GB BF16): `transformer/`
(1.6B DiT), `vae/` (DC-AE), `text_encoder/` (Gemma-2-2B), `tokenizer/`, `scheduler/`.

Ground rules: existing pipelines stay **byte-identical** (verify green) — Sana is additive, behind a new
`Variant::Sana` and a new `ImageVae` trait used only by Sana. Each phase lands with a reference-corr
check where possible. `Cargo.lock` in sync.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## Phase 0 — Plumbing skeleton (no inference) — DONE

- [x] `Variant::Sana` (`t2i.rs:351`) + `Variant::detect()` arm (`m.contains("sana")`, before the pixart
      check) + `is_sana()` predicate + the SD-backbone-loader reject guard (mirror PixArt).
- [x] Dispatch fan-out: `if variant.is_sana() { sana::run(...).await }` beside the PixArt block
      (`t2i.rs:~2687`).
- [x] `ALIAS_TABLE` entry (`hf/mod.rs`) — `sana` / `sana-1600m` / `sana-1024` →
      `Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers`.
- [x] `capability.rs` `ModelMeta` row (native_res 1024, BF16) + `gen_base_gb` / `rough_weight_gb` arms +
      the `doctor --capability` listing.
- [x] Stub `src/pipelines/sana.rs` `run()` that errors "not implemented yet". Unit test: `detects_sana`.

## Phase 1 — DC-AE autoencoder (`src/pipelines/dc_ae.rs`) + `ImageVae` trait — DONE

- [x] `ImageVae` trait (encode→Tensor deterministic, decode, scaling_factor 0.41407, latent_channels 32,
      spatial_compression 32). Retrofit **nothing** — existing pipelines keep the concrete `AutoEncoderKL`.
- [x] `AutoencoderDC`: 6 stages `[128,256,512,512,1024,1024]`, ResBlock ×3 then EfficientViTBlock ×3
      (encoder 2/2/2/3/3/3, decoder 3×6). Build the novel pieces from candle primitives:
  - **EfficientViT ReLU-linear multiscale attention** (1×1 qkv, 5×5 depthwise multiscale, `relu(Q)·(relu(K)ᵀV)` with the ones-row denominator) — **run in F32** (not self-normalizing → F16 NaN on Metal).
  - channel-dim **RMSNorm2d**, DCDownBlock (stride-2 conv + pixel-unshuffle-avg shortcut), DCUpBlock
    (interpolate + conv + channel-duplicate shortcut), GLU-MBConv FFN.
- [x] **Verify:** DONE — env-gated test (`PLAKAT_DCAE_VERIFY=1`) vs a diffusers dump
      (`tools/reference/sana_dcae_dump.py`): decode/encode/round-trip all **corr 1.000000** (max_abs ~1e-5).
      F32 island held (no NaN). Downsample=Conv (stride-2), upsample=interpolate per the model config.

## Phase 2 — Gemma-2-2B text encoder (`src/pipelines/vendored_gemma2.rs`) — DONE

- [x] Vendor candle's `gemma2.rs` + add `forward_hidden(ids) -> Tensor` (embed → layers → final norm over
      **all** positions, **no** lm_head; candle's `forward` returns last-token logits and its fields are
      private). Load `text_encoder/` safetensors + Gemma `tokenizer.json` (generic `tokenizers` crate).
- [x] `encode_prompt`: prepend the **CHI** ("complex human instruction") string, tokenize
      `padding_side="right"` to `chi_len + 300 − 2`, take last_hidden_state, **re-slice `[0] + last 299`**
      → `(B, 300, 2304)` + the matching attention mask (for DiT cross-attn). No norm/scaling. BF16 encoder.
- [x] **Verify:** DONE — env-gated (`PLAKAT_GEMMA_VERIFY=1`) vs a diffusers dump
      (`tools/reference/sana_gemma_dump.py`, Sana's own ungated text_encoder, F32/CPU): `forward_hidden`
      (all 506 positions incl. pad) **corr 0.999804**; after the `[0]+last-299` reslice **corr 0.999721**.
      Small drift is f32 compounding over 26 tanh-softcapped layers (benign — pad is masked in DiT cross-attn).
      Gemma-2-2B: hidden 2304, head_dim 256, 26 layers; text_encoder weights are root-keyed (no `model.` prefix).

## Phase 3 — Sana Linear-DiT (`src/pipelines/sana_dit.rs`)

- [ ] `SanaTransformer2DModel`: 20 layers, hidden 2240 (70×32), cross-attn 20×112, caption_channels 2304,
      in/out 32, patch_size 1 (→ trivial patchify of the 32×32 latent = 1024 tokens), mlp_ratio 2.5.
  - **ReLU linear self-attention** (F32 reduction island, `+1e-15` denom); **vanilla softmax cross-attention** to the caption.
  - **GLUMBConv Mix-FFN**: pointwise expand → 3×3 **depthwise** conv → GLU gate (SiLU) → pointwise project.
  - **AdaLN-single** (6-chunk scale_shift_table, shared timestep embed) — reuse PixArt's code shape.
- [ ] **Verify:** single forward with frozen reference caption embeds + fixed latent + timestep; corr the
      velocity output vs diffusers → 1.0.

## Phase 4 — End-to-end t2i + flow-matching

- [ ] Extract SD3's flow-match schedule (`build_img2img_timesteps` + the `t_prev−t_curr` velocity update)
      into a shared `pipelines/flow_match.rs` (or copy into `sana.rs`). Sana: FlowMatchEuler, 20 steps,
      guidance 4.5, flow_shift 3.0.
- [ ] `sana.rs` pipeline mirroring `pixart.rs` (RunRequest/run/run_hooked/load, alias resolve, LoRA
      resolve-then-load, StepHook). CFG batch (pos/neg caption). Initial noise `(B, 32, 32, 32)`.
- [ ] **Verify:** full seeded `plakat generate --model sana "…"` → coherent 1024² image; add a verify
      fixture (Tier-2 CPU-canonical).

## Phase 5 — Memory staging, verify fixtures, docs

- [ ] Non-co-resident staging on 24 GB Metal (Gemma-2-2B ~5 GB + DiT ~3.3 GB + DC-AE): mirror the SD3.5
      `PLAKAT_SD3_LOWMEM` pattern — encode with Gemma, free it, then load the DiT.
- [ ] Verify harness fixtures (Tier-1 golden corr for DC-AE / Gemma / DiT; Tier-2 end-to-end SSIM).
- [ ] Docs: GENERATE tutorial model table + a Sana note; capability tuning hint; README what's-new (at
      release). Memory: [[reference_feature_directions]] #7 done.

## Risks (ranked)

1. **F32 numerical islands for the two linear-attention sites** (DC-AE EfficientViT + DiT self-attn). ReLU
   linear attention isn't self-normalizing → unbounded `Σφ(k)v` accumulation overflows/loses precision in
   F16/BF16 on Metal → NaN/garbage. diffusers casts to fp32 for the reduction; **replicate that exactly.**
   Contained by the Phase-1/3 reference checks before any end-to-end wiring.
2. **candle 0.10.2 Metal grouped-conv** (GLUMBConv depthwise, ~5600 groups): historically flaky on Metal
   (cf. the GGUF quantized-matmul Metal bug). Treat as suspect until the Phase-3 corr check clears it;
   CPU-fallback that block if needed.
3. **Memory on 24 GB Metal** — three big models; needs the low-mem staging (Phase 5), but Phases 1–3 verify
   on CPU-canonical anyway.
4. **The Gemma CHI conditioning recipe** — miss the prepend / the `[0]+last-299` re-slice and all 300
   caption tokens are wrong. Pinned by the Phase-2 corr check.

## Reference dumps needed (from a diffusers Sana install, one-time)

For the per-phase corr checks: (a) a DC-AE latent + its decoded image, (b) Gemma hidden states for a known
prompt (with CHI), (c) a single DiT forward's input latent/timestep/caption + output velocity, (d) a full
seeded end-to-end image. Captured via `tools/reference/` (the existing verify-fixture tooling).
