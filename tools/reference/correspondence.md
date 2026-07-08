# Capture-point correspondence — diffusers ↔ plakat

The contract that makes a golden meaningful: each capture-point **name** must denote the
*same* intermediate on both sides. Authoring taps it in diffusers (here); `plakat verify`
taps it via `TensorTap` in the Rust pipeline. If the two disagree about *what* the name
means, the golden is wrong even when the code is right — chase any mismatch here first.

A `–` in the plakat column means the capture point is **not yet wired** (Phase 1b work).

## SD 1.5 (`sd_core@1`)

| name | diffusers module / value | plakat module | notes |
|---|---|---|---|
| `clip_l.penultimate` | `text_encoder(..., output_hidden_states=True).hidden_states[-2]` | `pipelines/sd_core.rs` text encode (clip-skip layer) | **The SD clip-skip noise bug lived here** — plakat must return the penultimate hidden state (pre-final-layernorm), matching `[-2]`. |
| `unet.out` | `unet(latent, 500, encoder_hidden_states=prompt_embeds).sample` — full ε | `t2i::capture_intermediates` → `core.unet.forward(deterministic_latent, 500.0, hidden, …)` | ✅ wired, corr 1.0. **First tap exercising the UNet CORE (down+mid+up), not just conditioning.** Shared LCG latent + fixed t=500 + golden-verified `clip.encoded`. |
| `unet.mid` | forward hook on `unet.mid_block` (same forward) | SDXL-only: `sdxl_unet::capture_mid` (candle's stock SD UNet has no exposed mid) | ✅ wired (SDXL). Localizes a UNet-core bug the full ε can't. |
| `vae.decoded` | `vae.decode(deterministic_latent).sample` (the **shared LCG** latent, NOT a seeded RNG) | `t2i::capture_intermediates` → `core.vae.decode(deterministic_latent())` | ✅ wired. Both decode the SAME LCG latent (`fixtures.deterministic_latent` ↔ `verify::deterministic_latent`), so no RNG-matching problem. F16-VAE class. |

## SD 2.1 (`sd_core@1`)

| name | diffusers | plakat | notes |
|---|---|---|---|
| `clip.encoded` | `text_encoder(ids)[0]` — OpenCLIP ViT-H last_hidden_state (post-LN), `(1,77,1024)` | `t2i::capture_intermediates` → `encode_prompt().0` (clip-skip 1) | ✅ wired, corr 0.99999. Validates the **"!"-pad branch** of the rule below. |
| `vae.decoded` | `vae.decode(deterministic_latent).sample` | `t2i::capture_intermediates` | ✅ wired, corr 1.0. |

### CLIP padding-token rule (why the SDXL finding happened)

The pad token depends on the **tokenizer family**, and the pre-final-LN penultimate is
numerically ill-conditioned (attention-sink magnitudes ~100–850), so a wrong pad token
shows as a *padding-only* divergence while content tokens still match:

| tokenizer | used by | `pad_token` | plakat `pad_with` |
|---|---|---|---|
| openai CLIP-L | SD1.5, SDXL `text_encoder`, SD3 CLIP-L | `<|endoftext|>` (49407) | `None` (→ EOS) |
| OpenCLIP / laion bigG | SD2.1 ViT-H, SDXL CLIP-G, SD3 CLIP-G | `"!"` (id 0) | `Some("!")` |

## SDXL (`sdxl_unet@1`)

| name | diffusers | plakat | notes |
|---|---|---|---|
| `clip.encoded` | `encode_prompt(...)` → `prompt_embeds` (concatenated dual-encoder penultimate, `(1,77,2048)`) | `t2i::capture_intermediates` → `encode_prompt().0` | ✅ wired. The text conditioning fed to cross-attn. **FINDING (this harness): CLIP-L was padded with `"!"`/id 0 instead of `<|endoftext|>`/49407** — content+EOS matched but padding rows diverged → corr 0.991. Fixed in `sd_core.rs::config` (`clip.pad_with = None`); now corr 1.0. See the pad-token rule under SD 2.1. |
| `clip_g.pooled` | `encode_prompt(...)` → `pooled_prompt_embeds` (CLIP-G pooled, `(1,1280)`) | `t2i::capture_intermediates` → `encode_prompt().1` (via `sdxl_clip::forward_for_sdxl`) | ✅ wired. **Pooled at the EOS id, not argmax** (TI-vocab bug, BUGFIX 1.5). |
| `add_time_ids` | `(orig_h, orig_w, crop_top, crop_left, target_h, target_w)` | `sdxl_unet::build_add_time_ids_base(h, w, …)` | order is **(h, w)** (regional-swap bug). Fed to the `unet.out`/`unet.mid` taps below. |
| `unet.out` | `unet(latent, 500, …, added_cond_kwargs={text_embeds, time_ids}).sample` | `core.unet.forward(latent, 500.0, hidden, Some(pooled), Some(add_time_ids))` | ✅ wired. Full SDXL ε incl. `add_embedding`. |
| `unet.mid` | forward hook on `unet.mid_block` | `sdxl_unet::capture_mid` | ✅ wired. SDXL mid-block activation. |

## SD 3.5-medium — MMDiT (`mmdit_inner@1`)

| name | diffusers | plakat | notes |
|---|---|---|---|
| `pooled_y` | `encode_prompt(...)` → `pooled_prompt_embeds` (concat of the two CLIP pooled) | `sd3::capture_intermediates` → `encode_prompt().0` | ✅ wired. The concat **ORDER** was the killer bug — the golden is diffusers' authoritative order; the comparison decides. |
| `t5.hidden` | `text_encoder_3(ids, attention_mask=mask)[0]` — masked T5 caption | `sd3::encode_prompt` → `vendored_t5::forward_with_mask` | ✅ wired, corr 1.0. Same v2.1 pad-mask fix as PixArt: SD3 forwarded T5 with no mask (real tokens attend to pad). Mask = `(ids != 0)`. **BF16** (F16 overflowed → inf captions). |
| `mmdit.block0` | forward hook on `transformer.transformer_blocks[0]` (x-stream) on DETERMINISTIC latent/`y`/`context` | `mmdit_inner.rs::MMDiT::capture_block0` | ✅ wired, corr 1.0. Embed prologue (patch+pos, t+y, context-embed) + joint block 0. `y`/`context` are shared-LCG synthetic → isolates the joint-block math (QK-norm, timestep, (scale,shift)) from CLIP/T5. |
| `vae.decoded` | | – | |

## PixArt-Σ — DiT (`pixart_dit@1`)

| name | diffusers | plakat | notes |
|---|---|---|---|
| `dit.pos_embed` | `get_2d_sincos_pos_embed(embed, grid, base_size, interpolation_scale)` | `pixart::capture_intermediates` → `build_2d_sincos_pos_embed(...)` | ✅ wired. H/W half-swap + base_size/interp scaling (past bug). Prompt-independent. |
| `t5.hidden` | `text_encoder(ids, attention_mask=mask)[0]` — masked T5 caption | `pixart::encode_prompt` → `vendored_t5::forward_with_mask` | ✅ wired, corr 1.0. **FINDING (v2.1): captions were encoded WITHOUT the pad attention mask** — real tokens attended to pad, drifting the caption to corr ~0.70 vs correct. Fixed: PixArt now routes T5 through the vendored copy + passes the mask. BF16 on GPU / F32 on CPU. |
| `adaln.embedded_timestep` | `transformer.adaln_single(timestep, added_cond, ...)[1]` — the embedded timestep | `pixart::capture_intermediates` → `dit.adaln_single.forward(...).1` | ✅ wired, corr 1.0. The `(1, hidden)` vector the FINAL adaLN consumes (NOT the 6-way `t_block` the blocks use — a real bug surface `dit.block0` doesn't cover). Prompt-independent (timestep+res+aspect). |
| `dit.block0` | forward hook on `transformer.transformer_blocks[0]` on DETERMINISTIC latent/caption + `encoder_attention_mask` (first half real, second half pad) | `pixart_dit.rs::PixArtSigmaXL::capture_block0` (+ deterministic `caption_mask`) | ✅ wired, corr 1.0. Patch+pos + adaLN + caption-projection + block 0, **now including the v2.1 cross-attention pad mask** (image queries don't attend to masked caption keys — matches diffusers `encoder_attention_mask`). Deterministic caption+mask isolates from T5. **2K KV-compression auto-detected from `attn1.kv_proj_conv2d.weight`.** |
| `vae.decoded` | | | |

## Stable Cascade (`cascade_prior@1`)

| name | diffusers | plakat | notes |
|---|---|---|---|
| `clip_g.pooled` | `StableCascadePriorPipeline.encode_prompt(...)` → pooled | `cascade::capture_intermediates` → `encode_prompt().1` | ✅ wired, corr 0.99997. Stage C's `clip_txt_pooled_mapper` + Stage B's only conditioning. |
| `stage_c.block0` | forward hook on `down_blocks[0][1]` (first `SDCascadeTimestepBlock`) — embedding→Res→Time — during a full `StableCascadeUNet.forward(sample=det, timestep_ratio=0.5, clip_text_pooled, clip_text, sca=None, crp=None)` | `cascade::capture_intermediates` → `stage_c.capture_block0(det_latent, sinusoidal(0.5), sinusoidal(0)×2, build_clip_conditioning(real))` | ✅ wired, corr **1.0**. The conditioned-conv core (embedding + first Res + Time), tapped BEFORE the first Attn — self-attention over the 576 white-noise tokens is OOD-ill-conditioned (that's why the earlier deep full-forward `stage_c.out` was only a coarse 0.989). `sca=None`→zeros matches plakat's `sinusoidal(0)`. |
| `effnet` | EfficientNetV2-S image embedding | `cascade_cn.rs` effnet | – Stage-C conditioning. |
| `stage_c.block0` | first Stage-C prior block | `cascade_prior.rs` | – FiLM time injection, sca/crp, Wuerstchen scheduler. |

## AnimateDiff (`animatediff@1`)

| name | diffusers | plakat | notes |
|---|---|---|---|
| `motion.block0` | `MotionAdapter.down_blocks[0].motion_modules[0]` (`AnimateDiffTransformer3D`) on a DETERMINISTIC per-frame input (16, 320, 8, 8) | `tier1::run_model` AnimateDiff branch → `modules.modules[0].forward(det, F=16)` | ✅ wired, corr 1.0. Loaded via the flag path (`load_v3`), not the alias dispatch. Motion weights are base-independent → SD 1.5 base only builds the pipeline. Both add the residual internally. pos-embed placement was the dominant v0.43 bug. |
| `cfg_batch.layout` | a synthetic layout probe | `animatediff.rs` | **BLOCKED `[uncond×F, cond×F]`** on the SDXL path (the frame ≥ 2 scramble, BUGFIX 1.1). Structurally guarded in verify Tier 0; the golden confirms the real motion forward respects it. |
| `unet.mid`, `vae.decoded` | | | |

---

**Discipline:** when plakat wires a capture point, fill its `plakat module` cell with the
exact `TensorTap` name + file:fn, and confirm the tapped tensor is the same object the
diffusers row describes (shape, dtype-before-cast, and — critically — *which layer*). Record
the diffusers version + resolved model revision in the manifest's `provenance` / `model_revision`.
