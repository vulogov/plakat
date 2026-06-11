//! InstantStyle — block-specific IP-Adapter injection for true painterly style
//! transfer (vs `stylize`'s content/palette-only IP-Adapter). Plan:
//! `Documentation/INSTANTSTYLE_PLAN.md`.
//!
//! ## Phase 1 (this module) — vendored UNet for inference
//! The vendored SD UNet (`sd_train::unet::UNet2DConditionModel`) carries the
//! LoRA-hooked cross-attention we will extend with a decoupled IP term — but it
//! was train-only. This loads it for **inference**.
//!
//! Key finding: the denoise loop itself is **identical** to `stylize`'s
//! (`stylize.rs` — `build_scheduler` → `add_noise` → per-step forward →
//! `scheduler.step` → `vae.decode`), all via `SdCore`. The *only* thing that
//! changes for InstantStyle is the per-step UNet forward: `core.unet.forward`
//! (candle UNet) → this vendored UNet's `forward` / `forward_sdxl`, which the
//! trainer already exercises every step (`trainer.rs:195`). So Phase 1 is "load
//! it + confirm forward-parity," not a from-scratch engine.
//!
//! ## Next
//! - Verify parity: vendored `forward` ≈ `core.unet.forward` on the same
//!   `(latent, t, ehs)` (GPU).
//! - Phase 2: add per-block decoupled IP cross-attention (`to_k_ip`/`to_v_ip`)
//!   to the vendored `CrossAttention`, injected only into the style block.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::sync::{Arc, RwLock};

use crate::pipelines::sd_train::attention::IpInjection;
use crate::pipelines::sd_train::trainer::{sd15_unet_config, sdxl_unet_config};
use crate::pipelines::sd_train::unet::UNet2DConditionModel;

// --- SDXL style block (InstantStyle): up_blocks.0.attentions.1 ---
// diffusers enumerates attn_processors down → up → mid (mid LAST). With that
// order up_blocks.0 starts at cross-ordinal 24, so attentions.1 = ordinals
// 34..44 → ip_adapter raw keys 2·ord+1 = 69,71,…,87 (attn2 at odd raw indices).
// inner_dim 1280, ctx 2048. (89..107 would be attentions.2 — the off-by-one from
// the earlier down→mid→up assumption.)
const SDXL_STYLE_UP_IDX: usize = 0;
const SDXL_STYLE_ATTN_IDX: usize = 1;
const SDXL_STYLE_IP_KEYS: [usize; 10] = [69, 71, 73, 75, 77, 79, 81, 83, 85, 87];
const SDXL_STYLE_INNER_DIM: usize = 1280;
const SDXL_CTX_DIM: usize = 2048;

// --- SD 1.5 style block: up_blocks.1.attentions.1 (analogous to SDXL's
// up_blocks.0.attentions.1 — SD15's up_blocks.0 is a plain UpBlock2D with no
// attention; up_blocks.1 is the first CrossAttnUpBlock2D at 1280). 1 attn2 layer
// (transformer_layers_per_block=1). Verified against the real SD1.5 UNet
// attn_processors (down → up → mid): cross-ordinal 7 → ip_adapter raw key 15.
const SD15_STYLE_UP_IDX: usize = 1;
const SD15_STYLE_ATTN_IDX: usize = 1;
const SD15_STYLE_IP_KEYS: [usize; 1] = [15];
const SD15_STYLE_INNER_DIM: usize = 1280;
const SD15_CTX_DIM: usize = 768;
/// IP-Adapter `image_proj` → 4 style tokens (8192 / 2048).
pub const IP_NUM_TOKENS: usize = 4;

/// Load the vendored SD UNet for **inference** (no train adapters), SD 1.5 or
/// SDXL. Mirrors the trainer's load (`trainer.rs:161-168`); the resulting UNet's
/// `forward` / `forward_sdxl` drives a standard denoise loop. The `.fp16`
/// weights are preferred, with the full-precision file as a fallback.
#[allow(dead_code)] // Phase 1 foundation — wired in Phase 2/3.
pub async fn load_vendored_unet(
    base_repo: &str,
    is_xl: bool,
    device: &Device,
    dtype: DType,
) -> Result<UNet2DConditionModel> {
    let unet_path = crate::hf::download::get_first_of(&[
        (base_repo, "unet/diffusion_pytorch_model.fp16.safetensors"),
        (base_repo, "unet/diffusion_pytorch_model.safetensors"),
    ])
    .await?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[unet_path], dtype, device)? };
    let cfg = if is_xl {
        sdxl_unet_config()
    } else {
        sd15_unet_config()
    };
    Ok(UNet2DConditionModel::new(vb, 4, 4, false, cfg, None)?)
}

/// Load the SDXL style block's IP-Adapter K/V and install them on the vendored
/// UNet's style block (`up_blocks.0.attentions.1`), sharing `tokens` — the
/// projected style embedding, filled before the denoise loop. `ip_vb` is a
/// VarBuilder over the IP-Adapter safetensors at the UNet's dtype. SDXL only.
#[allow(dead_code)] // wired by the stylize InstantStyle path.
pub fn install_instantstyle(
    unet: &mut UNet2DConditionModel,
    ip_vb: &VarBuilder,
    scale: f64,
    tokens: Arc<RwLock<Option<Tensor>>>,
    is_xl: bool,
) -> Result<()> {
    let (up_idx, attn_idx, keys, inner, ctx): (usize, usize, &[usize], usize, usize) = if is_xl {
        (
            SDXL_STYLE_UP_IDX,
            SDXL_STYLE_ATTN_IDX,
            &SDXL_STYLE_IP_KEYS,
            SDXL_STYLE_INNER_DIM,
            SDXL_CTX_DIM,
        )
    } else {
        (
            SD15_STYLE_UP_IDX,
            SD15_STYLE_ATTN_IDX,
            &SD15_STYLE_IP_KEYS,
            SD15_STYLE_INNER_DIM,
            SD15_CTX_DIM,
        )
    };
    let ip = ip_vb.pp("ip_adapter");
    let mut ips = Vec::with_capacity(keys.len());
    for &idx in keys.iter() {
        let lvb = ip.pp(idx.to_string());
        // to_k_ip/to_v_ip: weight (inner_dim, ctx) = (out, in), no bias.
        let to_k_ip = candle_nn::linear_no_bias(ctx, inner, lvb.pp("to_k_ip"))?;
        let to_v_ip = candle_nn::linear_no_bias(ctx, inner, lvb.pp("to_v_ip"))?;
        ips.push(IpInjection::new(to_k_ip, to_v_ip, scale, tokens.clone()));
    }
    unet.install_style_ip(up_idx, attn_idx, ips)?;
    Ok(())
}
