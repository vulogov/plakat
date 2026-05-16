//! IP-Adapter: CLIP-H image encoder + image projection module.
//!
//! This is the "shared cross-attention" variant of IP-Adapter:
//!   * The `image_proj.*` weights from `models/ip-adapter_sd15.safetensors`
//!     project CLIP-H image_embeds into the text-token space (4 tokens × 768).
//!   * Those tokens are CONCATENATED onto the text token sequence (in
//!     `stylize.rs`) so the UNet's existing cross-attention attends to both.
//!
//! The reference IP-Adapter uses *decoupled* cross-attention with separate
//! `to_k_ip` / `to_v_ip` projections in every UNet cross-attention layer.
//! candle 0.8's UNet doesn't expose attention hooks, so those weights are
//! unused here. Quality is lower than reference IP-Adapter; visible style
//! transfer still occurs.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::text_model::Activation;
use candle_transformers::models::clip::vision_model::{
    ClipVisionConfig, ClipVisionTransformer,
};
use std::path::Path;

/// Config for the CLIP-H/14 image encoder shipped with IP-Adapter.
/// Mirrors `models/image_encoder/config.json` in `h94/IP-Adapter`.
pub fn clip_h_vision_config() -> ClipVisionConfig {
    ClipVisionConfig {
        embed_dim: 1280,
        intermediate_size: 5120,
        num_hidden_layers: 32,
        num_attention_heads: 16,
        projection_dim: 1024,
        num_channels: 3,
        image_size: 224,
        patch_size: 14,
        // IP-Adapter's image_encoder/config.json says `hidden_act: "gelu"`
        // (exact erf-based GELU). candle 0.8's CLIP `Activation` enum only
        // exposes `QuickGelu` — using it here is a small documented
        // approximation; max per-element error ≈ 0.02 in the activation,
        // which compounds modestly across 32 layers.
        activation: Activation::QuickGelu,
    }
}

pub struct ImageEncoder {
    vision: ClipVisionTransformer,
    visual_projection: candle_nn::Linear,
}

impl ImageEncoder {
    /// Load `vision_model.*` + `visual_projection.*` from a single safetensors file.
    pub fn load(weights: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        let cfg = clip_h_vision_config();
        let vision = ClipVisionTransformer::new(vb.pp("vision_model"), &cfg)?;
        // CLIPVisionModelWithProjection has bias-less visual_projection.
        let visual_projection = candle_nn::linear_no_bias(
            cfg.embed_dim,
            cfg.projection_dim,
            vb.pp("visual_projection"),
        )?;
        Ok(Self {
            vision,
            visual_projection,
        })
    }

    /// (B, 3, 224, 224) → (B, projection_dim=1024)
    pub fn encode(&self, pixels: &Tensor) -> Result<Tensor> {
        let pooled = self.vision.forward(pixels)?;
        Ok(self.visual_projection.forward(&pooled)?)
    }
}

/// IP-Adapter image projection: Linear(clip_embed_dim → tokens·cross_attn_dim) + LayerNorm.
pub struct ImageProj {
    proj: candle_nn::Linear,
    norm: candle_nn::LayerNorm,
    num_tokens: usize,
    cross_attn_dim: usize,
}

impl ImageProj {
    /// Load just the `image_proj.*` subtree from an IP-Adapter safetensors file
    /// (e.g. `models/ip-adapter_sd15.safetensors`).
    pub fn load(
        weights: &Path,
        clip_embed_dim: usize,
        cross_attn_dim: usize,
        num_tokens: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        let vb = vb.pp("image_proj");
        let proj = candle_nn::linear(
            clip_embed_dim,
            num_tokens * cross_attn_dim,
            vb.pp("proj"),
        )?;
        let norm = candle_nn::layer_norm(cross_attn_dim, 1e-5, vb.pp("norm"))?;
        Ok(Self {
            proj,
            norm,
            num_tokens,
            cross_attn_dim,
        })
    }

    /// (B, clip_embed_dim) → (B, num_tokens, cross_attn_dim)
    pub fn forward(&self, image_embeds: &Tensor) -> Result<Tensor> {
        let b = image_embeds.dim(0)?;
        let x = self.proj.forward(image_embeds)?;
        let x = x.reshape((b, self.num_tokens, self.cross_attn_dim))?;
        Ok(self.norm.forward(&x)?)
    }
}
