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
| `unet.mid` | `unet.mid_block` forward output at t=500 | `pipelines/sd_core.rs` / `sdxl_unet.rs` mid block | timestep must match plakat's tap. |
| `vae.decoded` | `vae.decode(latents / scaling_factor).sample` | `pipelines` VAE decode | the F16-VAE class; author in F32, compare plakat's (possibly F16) cast to F32. |

## SDXL (`sdxl_unet@1`)

| name | diffusers | plakat | notes |
|---|---|---|---|
| `clip.encoded` | `encode_prompt(...)` → `prompt_embeds` (concatenated dual-encoder penultimate, `(1,77,2048)`) | `t2i::capture_intermediates` → `encode_prompt().0` | ✅ wired. The text conditioning fed to cross-attn. |
| `clip_g.pooled` | `encode_prompt(...)` → `pooled_prompt_embeds` (CLIP-G pooled, `(1,1280)`) | `t2i::capture_intermediates` → `encode_prompt().1` (via `sdxl_clip::forward_for_sdxl`) | ✅ wired. **Pooled at the EOS id, not argmax** (TI-vocab bug, BUGFIX 1.5). |
| `add_time_ids` | `(orig_h, orig_w, crop_top, crop_left, target_h, target_w)` | `sdxl_unet::build_add_time_ids_base(h, w, …)` | – order is **(h, w)** (regional-swap bug). |
| `unet.mid`, `vae.decoded` | as SD1.5 | – | |

## SD 3.5-medium — MMDiT (`mmdit_inner@1`)

| name | diffusers | plakat | notes |
|---|---|---|---|
| `pooled_y` | `encode_prompt(...)` → `pooled_prompt_embeds` (concat of the two CLIP pooled) | `sd3::capture_intermediates` → `encode_prompt().0` | ✅ wired. The concat **ORDER** was the killer bug — the golden is diffusers' authoritative order; the comparison decides. |
| `t5.hidden` | T5 caption embedding | `vendored_t5.rs` | – **BF16** (F16 overflowed → inf captions). |
| `mmdit.block0` | first joint transformer block output | `mmdit_inner.rs` | – AdaLayerNormContinuous (scale, shift) order; QK-norm; timestep ×1000. |
| `vae.decoded` | | – | |

## PixArt-Σ — DiT (`pixart_dit@1`)

| name | diffusers | plakat | notes |
|---|---|---|---|
| `dit.pos_embed` | `get_2d_sincos_pos_embed(embed, grid, base_size, interpolation_scale)` | `pixart::capture_intermediates` → `build_2d_sincos_pos_embed(...)` | ✅ wired. H/W half-swap + base_size/interp scaling (past bug). Prompt-independent. |
| `t5.hidden` | T5 caption | `vendored_t5.rs` | – BF16 (overflow bug). |
| `adaln.embedded_timestep` | `adaln_single` embedded timestep | `pixart_dit.rs::AdaLnSingle` | – final-adaLN uses the *embedded* timestep. |
| `dit.block0` | first block; **detect 2K KV-compression from `attn1.kv_proj_conv2d.weight`**, not the repo name | `pixart_dit.rs::PixArtBlock` | plakat now auto-detects this (BUGFIX 3.6). |
| `vae.decoded` | | | |

## Stable Cascade (`cascade_prior@1`)

| name | diffusers | plakat | notes |
|---|---|---|---|
| `clip_g.pooled` | CLIP-G pooled | `cascade.rs` | |
| `effnet` | EfficientNetV2-S image embedding | `cascade_cn.rs` effnet | Stage-C conditioning. |
| `stage_c.block0` | first Stage-C prior block | `cascade_prior.rs` | FiLM time injection, sca/crp, Wuerstchen scheduler. |

## AnimateDiff (`animatediff@1`)

| name | diffusers | plakat | notes |
|---|---|---|---|
| `motion.block0` | first motion module output | `animatediff.rs` / `motion_module.rs` | pos-embed placement was the dominant bug. |
| `cfg_batch.layout` | a synthetic layout probe | `animatediff.rs` | **BLOCKED `[uncond×F, cond×F]`** on the SDXL path (the frame ≥ 2 scramble, BUGFIX 1.1). Structurally guarded in verify Tier 0; the golden confirms the real motion forward respects it. |
| `unet.mid`, `vae.decoded` | | | |

---

**Discipline:** when plakat wires a capture point, fill its `plakat module` cell with the
exact `TensorTap` name + file:fn, and confirm the tapped tensor is the same object the
diffusers row describes (shape, dtype-before-cast, and — critically — *which layer*). Record
the diffusers version + resolved model revision in the manifest's `provenance` / `model_revision`.
