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
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::pipelines::sd_train::trainer::{sd15_unet_config, sdxl_unet_config};
use crate::pipelines::sd_train::unet::UNet2DConditionModel;

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
