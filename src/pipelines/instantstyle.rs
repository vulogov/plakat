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

// --- SD 1.5 style-block candidates (EXPLORATION) ---
// The SDXL analogy (`up_blocks.1.attentions.1`) corrupts STRUCTURE on SD 1.5
// (melts faces, no texture) — SD 1.5's deep up-block is a single structural
// layer, not a style carrier like SDXL's 10-layer one. Until the real style
// block is found empirically, every cross-attn up-block is a candidate,
// selectable at runtime via `PLAKAT_SD15_STYLE_BLOCK` (e.g. "up2.1"). The
// attn_processors order is down→up→mid; raw = 2·cross_ordinal+1;
// transformer_layers_per_block=1 → one attn2 per attention.
// (name, up_idx, attn_idx, raw_key, inner_dim).
const SD15_CANDIDATES: &[(&str, usize, usize, usize, usize)] = &[
    ("up1.0", 1, 0, 13, 1280),
    ("up1.1", 1, 1, 15, 1280), // SDXL-analogy default — corrupts structure
    ("up1.2", 1, 2, 17, 1280),
    ("up2.0", 2, 0, 19, 640),
    ("up2.1", 2, 1, 21, 640),
    ("up2.2", 2, 2, 23, 640),
    ("up3.0", 3, 0, 25, 320),
    ("up3.1", 3, 1, 27, 320),
    ("up3.2", 3, 2, 29, 320),
];
// Default = InstantStyle's official SD 1.5 target `["up_blocks.1"]`, which is a
// SUBSTRING match → ALL 3 attentions of up_blocks.1 (keys 13/15/17), NOT a single
// attention. (`infer_style_sd15.py`; SDXL's single-`.attentions.1` is XL-specific.)
const SD15_DEFAULT_BLOCK: &str = "up1.all";
const SD15_CTX_DIM: usize = 768;

/// The 3 cross-attn layers of an SD 1.5 up-block: (up_idx, attn_idx, [raw_key]) ×3
/// + inner dim. Raw keys are contiguous odds from the block base (up1→13,15,17
/// @1280; up2→19,21,23 @640; up3→25,27,29 @320).
fn sd15_full_block(up_idx: usize) -> (Vec<(usize, usize, Vec<usize>)>, usize) {
    let (base, inner) = match up_idx {
        2 => (19, 640),
        3 => (25, 320),
        _ => (13, 1280), // up_blocks.1
    };
    let up = if (1..=3).contains(&up_idx) { up_idx } else { 1 };
    let groups = (0..3).map(|a| (up, a, vec![base + 2 * a])).collect();
    (groups, inner)
}

/// Resolve the SD 1.5 style target → injection groups + inner dim.
/// `PLAKAT_SD15_STYLE_BLOCK` overrides: `upN.all` (full block, the real target) or
/// `upN.M` (one attention, for diagnosis).
fn sd15_groups() -> (Vec<(usize, usize, Vec<usize>)>, usize, usize) {
    let sel = std::env::var("PLAKAT_SD15_STYLE_BLOCK")
        .unwrap_or_else(|_| SD15_DEFAULT_BLOCK.to_string());
    let (groups, inner) = if let Some(n) = sel.strip_suffix(".all") {
        sd15_full_block(n.trim_start_matches("up").parse().unwrap_or(1))
    } else if let Some(c) = SD15_CANDIDATES.iter().find(|c| c.0 == sel) {
        (vec![(c.1, c.2, vec![c.3])], c.4)
    } else {
        sd15_full_block(1)
    };
    crate::ui::progress::println(&format!(
        "InstantStyle: SD1.5 style target = {sel} ({} attn layer(s), inner {inner}) \
         — set PLAKAT_SD15_STYLE_BLOCK to explore",
        groups.len()
    ));
    (groups, inner, SD15_CTX_DIM)
}
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
    // SDXL needs the add_embedding (text_time conditioning) config, or
    // forward_sdxl has no aug-emb to build. SD 1.5 has none (uses `forward`).
    let (cfg, add_cfg) = if is_xl {
        (
            sdxl_unet_config(),
            Some(crate::pipelines::sdxl_unet::SdxlAddEmbedConfig::base()),
        )
    } else {
        (sd15_unet_config(), None)
    };
    Ok(UNet2DConditionModel::new(vb, 4, 4, false, cfg, add_cfg)?)
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
    let (groups, inner, ctx): (Vec<(usize, usize, Vec<usize>)>, usize, usize) = if is_xl {
        (
            vec![(
                SDXL_STYLE_UP_IDX,
                SDXL_STYLE_ATTN_IDX,
                SDXL_STYLE_IP_KEYS.to_vec(),
            )],
            SDXL_STYLE_INNER_DIM,
            SDXL_CTX_DIM,
        )
    } else {
        sd15_groups()
    };
    let ip = ip_vb.pp("ip_adapter");
    for (up_idx, attn_idx, keys) in groups {
        let mut ips = Vec::with_capacity(keys.len());
        for &idx in keys.iter() {
            let lvb = ip.pp(idx.to_string());
            // to_k_ip/to_v_ip: weight (inner_dim, ctx) = (out, in), no bias.
            let to_k_ip = candle_nn::linear_no_bias(ctx, inner, lvb.pp("to_k_ip"))?;
            let to_v_ip = candle_nn::linear_no_bias(ctx, inner, lvb.pp("to_v_ip"))?;
            ips.push(IpInjection::new(to_k_ip, to_v_ip, scale, tokens.clone()));
        }
        unet.install_style_ip(up_idx, attn_idx, ips)?;
    }
    Ok(())
}
