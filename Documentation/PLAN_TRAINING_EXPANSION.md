# plakat — training-expansion plan (LoRA / TI across the model families)

Where training stands today, and a concrete plan to close the gaps. Ordered by
effort × verifiability-on-Apple-Silicon. **Flux is back-burnered** (broken on Metal
for inference → a trained Flux LoRA can't be verified on-box; CPU/CUDA only).

## Current coverage

| Family | Style LoRA | DreamBooth | Textual Inversion |
|---|---|---|---|
| SD 1.5 | ✅ | ✅ | ✅ |
| SD 2.1 | ❌ | ❌ | ✅ |
| SDXL | ✅ | ✅ | ✅ |
| SD 3.5 | ✅ | ✅ | **❌ (this plan)** |
| Flux | ❌ | ❌ | ❌ (back-burner) |
| PixArt-Σ | **❌ (this plan)** | ❌ | ❌ |
| Stable Cascade | **❌ (this plan)** | ❌ | ❌ |

## The reusable training spine (what every plan builds on)

`src/pipelines/sd_train/trainer.rs` is the template:

- **Phase A** — load the model, VAE-encode the training images → latents, text-encode
  the trigger → embeddings, capture cfg, drop the load.
- **Phase B** — load the (vendored) denoiser in BF16, `install_train_adapters(rank,
  scale, device)` to attach **trainable LoRA adapters** to the attention linears
  (`LoraLinear::set_train_adapter(a, b, scale)` in `lora_linear.rs`), collect the
  adapter `Var`s, run the loss loop (DDPM-ε for SD; rectified-flow for SD3/Flux),
  periodic checkpoints (kohya `…-step<N>.safetensors`), `--resume` continues.
- DreamBooth = the same loop plus a **prior-preservation** term over a class set
  (`prior_weight` · class loss).

**The cross-cutting work item** each new architecture needs: its attention `Linear`s
must be `LoraLinear` so `install_train_adapters` has somewhere to attach. SD UNet
already is; the DiT / MMDiT / Stage-C modules need that retrofit (mechanical — wrap
the qkv/out/cross-attn projections), which is the bulk of each task below.

---

## 1. SD 2.1 — style LoRA + DreamBooth  ·  effort: **S**  ·  verify: **on-box**

**Goal.** Bring SD 2.1 to parity with SD 1.5 (it already has TI).

**Why easy.** SD 2.1 *is* the SD 1.5 UNet architecture — the only deltas are the
text encoder (OpenCLIP ViT-H, 1024-dim, vs ViT-L 768) and the **v-prediction**
objective (vs ε). The `train_style_lora_sd` sd15 path already handles the UNet +
adapters; sd21 reuses it.

**Steps.**
1. Route `sd21` into `train_style_lora_sd` (today it falls through to the `not
   supported` bail) using the sd21 alias → its 1024-dim CLIP encoder.
2. Switch the loss target to **v-prediction** when the variant is v-pred (the
   scheduler/variant already knows; the loss computes `v = α·ε − σ·x0` target).
3. CLI: extend `style train --base` + `dreambooth` to accept `sd21`.
4. **Verify on-box:** `corpus/style_train.sh sd21` (mirror the sd15 driver) → a
   trained LoRA that visibly transfers; commit a showcase.

**Risk.** Low. The v-pred target is the only real subtlety; sd21 inference already
uses the v_2_1 scheduler, so the convention is known.

---

## 2. PixArt-Σ — style/subject LoRA  ·  effort: **M**  ·  verify: **on-box**

**Goal.** LoRA on the PixArt-Σ **DiT** (a transformer, not a UNet).

**Why feasible now.** The inference DiT is correct (matches diffusers, corr 0.987),
so the forward + T5 + VAE all exist. LoRA attaches to the DiT block linears
(`attn.to_{q,k,v}`, `attn.proj`, the cross-attention to the T5 context, and
optionally `mlp.fc{1,2}`) — the same idea the SD3 MMDiT LoRA *merge* already does.

**Steps.**
1. Retrofit the PixArt DiT attention/MLP projections to `LoraLinear` (wrap the
   `nn::Linear`s) + a PixArt `install_train_adapters` that selects the DiT targets.
2. Training forward: VAE-encode images, **T5-encode** the trigger (BF16 — F16 T5
   overflows, the known PixArt gotcha), sample timesteps, **IDDPM/DDPM-ε** loss
   through the frozen DiT with the trainable adapters, embedded-timestep adaLN as in
   inference.
3. Save as a kohya-style LoRA keyed for the PixArt DiT; wire `--lora` load (the merge
   path needs PixArt DiT key support if not already present).
4. CLI: `style train --base pixart`. **Verify on-box** at 512-MS.

**Risk.** Medium — adaLN-single + the embedded-timestep path must match inference;
512² keeps memory in budget.

---

## 3. Stable Cascade — Stage-C LoRA  ·  effort: **M–L**  ·  verify: **on-box**

**Goal.** A style LoRA on **Stage C** (the semantic generator; Stage A VAE + Stage B
decoder stay frozen).

**Why distinct.** Cascade trains in the **Würstchen semantic latent space**: an
image → EfficientNet "effnet" conditioning + the small (≈24²) Stage-C latent. Stage C
is the diffusion model; its attention is where style lives. The Cascade LoRA *merge*
(kohya prefix + DoRA) already exists, so it's the **training loop + adapters** that
are new.

**Steps.**
1. Retrofit Stage-C attention linears to `LoraLinear` + a Cascade
   `install_train_adapters` (Stage C only).
2. Training forward: encode images to the Stage-C latent (Stage A/effnet), CLIP-G
   text-encode the trigger, the **Wuerstchen/rectified scheduler** loss target (not
   plain DDPM-ε — Cascade uses its own noising), train the adapters.
3. Save Stage-C LoRA (the merge path already loads it); CLI `style train --base
   cascade`. **Verify on-box** (Cascade fits via the staged loads).

**Risk.** Medium-high — the Stage-C latent encoding + the Würstchen loss objective are
the unknowns; lean on the existing inference scheduler to get the noising right.

---

## 4. SD 3.5 — Textual Inversion  ·  effort: **M**  ·  verify: **on-box (memory-bound)**

**Goal.** TI for SD 3.5 (it has LoRA + DreamBooth; TI bails today).

**Why feasible.** TI freezes the whole model and optimizes only a **new token
embedding** — `ti_train.rs` already does this for sd15/sd21 (single CLIP) and SDXL
(dual `clip_l`+`clip_g`). SD 3.5 adds **T5** alongside CLIP-L + CLIP-G, so the
placeholder token gets a learned vector in **each** encoder; the gradient flows from
the frozen MMDiT denoise loss back to those embedding rows.

**Steps.**
1. Add an `sd35` arm to `train_textual_inversion`: place a placeholder token in
   CLIP-L, CLIP-G **and** T5; mask-combine from an init word per encoder (the
   differentiable slot trick already used for SDXL's dual encoders, extended to 3).
2. Loss: rectified-flow velocity loss through the frozen MMDiT (reuse the SD3 forward;
   pooled-y `[CLIP-L, CLIP-G]` order — the known SD3 gotcha).
3. Save a triple embedding file (`clip_l` + `clip_g` + `t5`); `--embedding` loader
   reads all three for SD3.5. **Verify on-box** at a modest size (SD3.5 render is
   memory-bound — keep the TI render small).

**Risk.** Medium — three-encoder bookkeeping + the memory wall on the verify render;
training itself (embeddings only) is light.

---

## 5. Flux — LoRA / DreamBooth / TI  ·  **BACK-BURNER**

Implementable (Flux LoRA *merge* works; rectified-flow training mirrors SD3), **but
unverifiable on Apple Silicon** — Flux is broken on Metal for inference (candle GGUF
mat×mat kernel bug). A trained Flux LoRA could only be proven on CPU (too slow) or
CUDA (not this box). Park it until a CUDA/CI verify path exists or the Metal kernel is
fixed upstream.

---

## Suggested sequence

1. **SD 2.1 LoRA + DreamBooth** — cheap, on-box, closes an SD-family gap. *(start here)*
2. **PixArt-Σ LoRA** — the DiT-adapter retrofit is reusable groundwork for any
   transformer-denoiser training (and de-risks a future Flux/SD3 LoRA-*train* path).
3. **SD 3.5 TI** — completes SD3.5's training trio; light training, memory-bound verify.
4. **Stable Cascade LoRA** — the most exotic objective; do it once the adapter-retrofit
   pattern is proven on PixArt.
5. **Flux** — when verifiable.

Each lands as its own release increment with a `corpus/*_train.sh` driver + a committed
showcase (training output is non-deterministic → a showcase, not a byte-check, per the
corpus convention).
