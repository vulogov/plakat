//! SDXL UNet with `text_time` micro-conditioning (phase 8a).
//!
//! Vendored + extended port of candle-transformers 0.8.4's
//! `stable_diffusion::unet_2d::UNet2DConditionModel`. Upstream loads the
//! SDXL safetensors fine but silently skips the `add_embedding`
//! projection — SDXL's UNet config has `addition_embed_type: "text_time"`
//! which composes a side embedding from the pooled CLIP-G text output
//! and the size/crop micro-conditioning floats, then *adds* it to the
//! time embedding that propagates through every block. Without it the
//! denoiser runs but output quality lands at ~50–70 % of the diffusers
//! reference.
//!
//! This module re-implements the outer UNet using upstream block types
//! directly (so we vendor ~400 lines rather than the full ~2000-line
//! UNet + blocks stack) and adds:
//!
//!   * `add_time_proj`  — sinusoidal embedding for the time_ids floats
//!     (same shape as `time_proj`, with `addition_time_embed_dim = 256`).
//!   * `add_embedding`  — `Linear → SiLU → Linear` from
//!     `concat(time_ids_embed, pooled_text_embed) → time_embed_dim`.
//!   * Forward parameters for `add_text_embeds` and `add_time_ids` that
//!     drive the projection above.
//!
//! Only used for SDXL (base + refiner). SD 1.5 and SD 2.1 continue to
//! use candle's upstream UNet — they have `addition_embed_type: null`
//! and would only carry dead weight.
//!
//! When candle gains a `text_time` UNet upstream this file can be
//! deleted and `SdxlUNet2DConditionModel` aliased back to the upstream
//! type — the public surface here matches upstream's `forward` /
//! `forward_with_additional_residuals` signatures one-for-one apart
//! from the two new tensor parameters.

use candle_core::{Result, Tensor};
use candle_nn as nn;
use candle_nn::{conv2d, Conv2d, Module};
use candle_transformers::models::stable_diffusion::{
    embeddings::{TimestepEmbedding, Timesteps},
    unet_2d::{BlockConfig, UNet2DConditionModelConfig},
    unet_2d_blocks::{
        CrossAttnDownBlock2D, CrossAttnDownBlock2DConfig, CrossAttnUpBlock2D,
        CrossAttnUpBlock2DConfig, DownBlock2D, DownBlock2DConfig, UNetMidBlock2DCrossAttn,
        UNetMidBlock2DCrossAttnConfig, UpBlock2D, UpBlock2DConfig,
    },
};

/// Diffusers-matching defaults for the refiner's micro-conditioning.
/// `aesthetic_score = 6.0` is the standard "good aesthetics" positive
/// signal that pre-trained-refiner pipelines use; `2.5` is the matching
/// negative-CFG anchor that pulls outputs toward higher aesthetics.
pub const REFINER_AESTHETIC_SCORE_POS: f32 = 6.0;
pub const REFINER_AESTHETIC_SCORE_NEG: f32 = 2.5;

/// Build base SDXL's `add_time_ids` tensor for **one** branch (cond
/// **or** uncond). Shape: `(1, 6)`. The order matches diffusers'
/// `_get_add_time_ids`: `[orig_h, orig_w, crop_top, crop_left,
/// target_h, target_w]`. Caller duplicates / concatenates across the
/// batch dim as needed for CFG.
///
/// Defaults align with diffusers' high-quality inference:
///   * `orig_size = target_size` — pretend the training image was the
///     same size as the target. Lying about original_size as larger
///     than target can be used to bias toward zoomed-out compositions
///     (advanced — exposed as a CLI flag in a future phase if asked).
///   * `crops_coords_top_left = (0, 0)` — pretend no crop was applied
///     during training. Non-zero values pull the model toward off-
///     centre compositions.
pub fn build_add_time_ids_base(
    target_h: u32,
    target_w: u32,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> Result<Tensor> {
    let vals: [f32; 6] = [
        target_h as f32,
        target_w as f32,
        0.0,
        0.0,
        target_h as f32,
        target_w as f32,
    ];
    Tensor::from_slice(&vals, (1, 6), device)?.to_dtype(dtype)
}

/// Build the refiner's `add_time_ids` for one branch. Shape: `(1, 5)`.
/// Order: `[orig_h, orig_w, crop_top, crop_left, aesthetic_score]`.
/// `aesthetic_score` should be [`REFINER_AESTHETIC_SCORE_POS`] for the
/// conditional branch and [`REFINER_AESTHETIC_SCORE_NEG`] for the
/// unconditional branch so the CFG step pulls toward higher quality.
pub fn build_add_time_ids_refiner(
    target_h: u32,
    target_w: u32,
    aesthetic_score: f32,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> Result<Tensor> {
    let vals: [f32; 5] = [
        target_h as f32,
        target_w as f32,
        0.0,
        0.0,
        aesthetic_score,
    ];
    Tensor::from_slice(&vals, (1, 5), device)?.to_dtype(dtype)
}

/// Shape of the SDXL `text_time` add_embedding. Same struct used for
/// base SDXL (`num_time_ids = 6`) and the refiner (`num_time_ids = 5`
/// — the last slot is `aesthetic_score` instead of `target_size`).
#[derive(Debug, Clone, Copy)]
pub struct SdxlAddEmbedConfig {
    /// Per-float sinusoidal embedding width. Always 256 for SDXL.
    pub addition_time_embed_dim: usize,
    /// Number of time-id floats fed to the side embedding. 6 for base,
    /// 5 for the refiner.
    pub num_time_ids: usize,
    /// Width of the pooled text embedding fed alongside the time ids.
    /// 1280 for both SDXL base and the refiner (the CLIP-G text-encoder
    /// projection dim).
    pub pooled_text_dim: usize,
}

impl SdxlAddEmbedConfig {
    /// Default config for the base SDXL UNet (6 time_ids = orig_size,
    /// crops_coords_top_left, target_size).
    pub fn base() -> Self {
        Self {
            addition_time_embed_dim: 256,
            num_time_ids: 6,
            pooled_text_dim: 1280,
        }
    }

    /// Default config for the SDXL refiner UNet (5 time_ids = orig_size,
    /// crops_coords_top_left, aesthetic_score).
    pub fn refiner() -> Self {
        Self {
            addition_time_embed_dim: 256,
            num_time_ids: 5,
            pooled_text_dim: 1280,
        }
    }

    /// Input width of the `add_embedding`'s first Linear:
    /// `num_time_ids * addition_time_embed_dim + pooled_text_dim`.
    pub fn in_dim(&self) -> usize {
        self.num_time_ids * self.addition_time_embed_dim + self.pooled_text_dim
    }
}

// Re-defined locally because candle's `UNetDownBlock` / `UNetUpBlock`
// are `pub(crate)` — same dispatch, same variants.
#[derive(Debug)]
enum UNetDownBlock {
    Basic(DownBlock2D),
    CrossAttn(CrossAttnDownBlock2D),
}

#[derive(Debug)]
enum UNetUpBlock {
    Basic(UpBlock2D),
    CrossAttn(CrossAttnUpBlock2D),
}

#[derive(Debug)]
pub struct SdxlUNet2DConditionModel {
    conv_in: Conv2d,
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
    // -------- SDXL-only additions --------
    add_time_proj: Timesteps,
    add_embedding: TimestepEmbedding,
    add_cfg: SdxlAddEmbedConfig,
    // -------- standard UNet remainder --------
    down_blocks: Vec<UNetDownBlock>,
    mid_block: UNetMidBlock2DCrossAttn,
    up_blocks: Vec<UNetUpBlock>,
    conv_norm_out: nn::GroupNorm,
    conv_out: Conv2d,
    config: UNet2DConditionModelConfig,
}

impl SdxlUNet2DConditionModel {
    /// Construct the UNet. Mirrors candle's upstream constructor and
    /// additionally builds `add_time_proj` + `add_embedding` from
    /// `add_cfg`. Weights for the new modules are read from
    /// `add_time_proj` (no weights — it's a constant embedding) and
    /// `add_embedding.linear_{1,2}.{weight,bias}` in the safetensors —
    /// the same paths diffusers writes to.
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        use_flash_attn: bool,
        config: UNet2DConditionModelConfig,
        add_cfg: SdxlAddEmbedConfig,
    ) -> Result<Self> {
        let n_blocks = config.blocks.len();
        let b_channels = config.blocks[0].out_channels;
        let bl_channels = config.blocks.last().unwrap().out_channels;
        let bl_attention_head_dim = config.blocks.last().unwrap().attention_head_dim;
        let time_embed_dim = b_channels * 4;
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = conv2d(in_channels, b_channels, 3, conv_cfg, vs.pp("conv_in"))?;

        let time_proj = Timesteps::new(b_channels, config.flip_sin_to_cos, config.freq_shift);
        let time_embedding =
            TimestepEmbedding::new(vs.pp("time_embedding"), b_channels, time_embed_dim)?;

        // -------- SDXL add_embedding (phase 8a) --------
        // `add_time_proj` shares the same sinusoidal formula as `time_proj`
        // (flip_sin_to_cos = true, no freq_shift) but with width
        // `addition_time_embed_dim` (256 in stock SDXL) so each of the
        // `num_time_ids` floats expands to a 256-wide embedding. The
        // concatenation with the pooled text embedding then flows
        // through `add_embedding` (Linear+SiLU+Linear) to land back in
        // `time_embed_dim` so it can be added onto the time emb.
        let add_time_proj = Timesteps::new(add_cfg.addition_time_embed_dim, true, 0.0);
        let add_embedding =
            TimestepEmbedding::new(vs.pp("add_embedding"), add_cfg.in_dim(), time_embed_dim)?;

        let vs_db = vs.pp("down_blocks");
        let down_blocks = (0..n_blocks)
            .map(|i| {
                let BlockConfig {
                    out_channels,
                    use_cross_attn,
                    attention_head_dim,
                } = config.blocks[i];

                let sliced_attention_size = match config.sliced_attention_size {
                    Some(0) => Some(attention_head_dim / 2),
                    _ => config.sliced_attention_size,
                };

                let in_channels = if i > 0 {
                    config.blocks[i - 1].out_channels
                } else {
                    b_channels
                };
                let db_cfg = DownBlock2DConfig {
                    num_layers: config.layers_per_block,
                    resnet_eps: config.norm_eps,
                    resnet_groups: config.norm_num_groups,
                    add_downsample: i < n_blocks - 1,
                    downsample_padding: config.downsample_padding,
                    ..Default::default()
                };
                if let Some(transformer_layers_per_block) = use_cross_attn {
                    let xa_cfg = CrossAttnDownBlock2DConfig {
                        downblock: db_cfg,
                        attn_num_head_channels: attention_head_dim,
                        cross_attention_dim: config.cross_attention_dim,
                        sliced_attention_size,
                        use_linear_projection: config.use_linear_projection,
                        transformer_layers_per_block,
                    };
                    let block = CrossAttnDownBlock2D::new(
                        vs_db.pp(i.to_string()),
                        in_channels,
                        out_channels,
                        Some(time_embed_dim),
                        use_flash_attn,
                        xa_cfg,
                    )?;
                    Ok(UNetDownBlock::CrossAttn(block))
                } else {
                    let block = DownBlock2D::new(
                        vs_db.pp(i.to_string()),
                        in_channels,
                        out_channels,
                        Some(time_embed_dim),
                        db_cfg,
                    )?;
                    Ok(UNetDownBlock::Basic(block))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let mid_transformer_layers_per_block = match config.blocks.last() {
            None => 1,
            Some(block) => block.use_cross_attn.unwrap_or(1),
        };
        let mid_cfg = UNetMidBlock2DCrossAttnConfig {
            resnet_eps: config.norm_eps,
            output_scale_factor: config.mid_block_scale_factor,
            cross_attn_dim: config.cross_attention_dim,
            attn_num_head_channels: bl_attention_head_dim,
            resnet_groups: Some(config.norm_num_groups),
            use_linear_projection: config.use_linear_projection,
            transformer_layers_per_block: mid_transformer_layers_per_block,
            ..Default::default()
        };
        let mid_block = UNetMidBlock2DCrossAttn::new(
            vs.pp("mid_block"),
            bl_channels,
            Some(time_embed_dim),
            use_flash_attn,
            mid_cfg,
        )?;

        let vs_ub = vs.pp("up_blocks");
        let up_blocks = (0..n_blocks)
            .map(|i| {
                let BlockConfig {
                    out_channels,
                    use_cross_attn,
                    attention_head_dim,
                } = config.blocks[n_blocks - 1 - i];

                let sliced_attention_size = match config.sliced_attention_size {
                    Some(0) => Some(attention_head_dim / 2),
                    _ => config.sliced_attention_size,
                };

                let prev_out_channels = if i > 0 {
                    config.blocks[n_blocks - i].out_channels
                } else {
                    bl_channels
                };
                let in_channels = {
                    let index = if i == n_blocks - 1 {
                        0
                    } else {
                        n_blocks - i - 2
                    };
                    config.blocks[index].out_channels
                };
                let ub_cfg = UpBlock2DConfig {
                    num_layers: config.layers_per_block + 1,
                    resnet_eps: config.norm_eps,
                    resnet_groups: config.norm_num_groups,
                    add_upsample: i < n_blocks - 1,
                    ..Default::default()
                };
                if let Some(transformer_layers_per_block) = use_cross_attn {
                    let xa_cfg = CrossAttnUpBlock2DConfig {
                        upblock: ub_cfg,
                        attn_num_head_channels: attention_head_dim,
                        cross_attention_dim: config.cross_attention_dim,
                        sliced_attention_size,
                        use_linear_projection: config.use_linear_projection,
                        transformer_layers_per_block,
                    };
                    let block = CrossAttnUpBlock2D::new(
                        vs_ub.pp(i.to_string()),
                        in_channels,
                        prev_out_channels,
                        out_channels,
                        Some(time_embed_dim),
                        use_flash_attn,
                        xa_cfg,
                    )?;
                    Ok(UNetUpBlock::CrossAttn(block))
                } else {
                    let block = UpBlock2D::new(
                        vs_ub.pp(i.to_string()),
                        in_channels,
                        prev_out_channels,
                        out_channels,
                        Some(time_embed_dim),
                        ub_cfg,
                    )?;
                    Ok(UNetUpBlock::Basic(block))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let conv_norm_out = nn::group_norm(
            config.norm_num_groups,
            b_channels,
            config.norm_eps,
            vs.pp("conv_norm_out"),
        )?;
        let conv_out = conv2d(b_channels, out_channels, 3, conv_cfg, vs.pp("conv_out"))?;
        Ok(Self {
            conv_in,
            time_proj,
            time_embedding,
            add_time_proj,
            add_embedding,
            add_cfg,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            config,
        })
    }

    /// Plain forward — no ControlNet residuals.
    pub fn forward(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        add_text_embeds: &Tensor,
        add_time_ids: &Tensor,
    ) -> Result<Tensor> {
        self.forward_with_additional_residuals(
            xs,
            timestep,
            encoder_hidden_states,
            add_text_embeds,
            add_time_ids,
            None,
            None,
        )
    }

    /// ControlNet-aware forward. `down_block_additional_residuals` and
    /// `mid_block_additional_residual` follow the same semantics as
    /// candle's upstream UNet — see [`crate::pipelines::controlnet`].
    pub fn forward_with_additional_residuals(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        add_text_embeds: &Tensor,
        add_time_ids: &Tensor,
        down_block_additional_residuals: Option<&[Tensor]>,
        mid_block_additional_residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (bsize, _channels, height, width) = xs.dims4()?;
        let device = xs.device();
        let n_blocks = self.config.blocks.len();
        let num_upsamplers = n_blocks - 1;
        let default_overall_up_factor = 2usize.pow(num_upsamplers as u32);
        let forward_upsample_size =
            height % default_overall_up_factor != 0 || width % default_overall_up_factor != 0;

        // 0. center input if necessary
        let xs = if self.config.center_input_sample {
            ((xs * 2.0)? - 1.0)?
        } else {
            xs.clone()
        };

        // 1. time embedding (b, time_embed_dim)
        let t_emb = (Tensor::ones(bsize, xs.dtype(), device)? * timestep)?;
        let t_emb = self.time_proj.forward(&t_emb)?;
        let emb = self.time_embedding.forward(&t_emb)?;

        // 1b. SDXL add_embedding — `add_time_ids` is shaped (b,
        //     num_time_ids); flatten through `add_time_proj` to
        //     (b, num_time_ids * addition_time_embed_dim), concat with
        //     `add_text_embeds` (b, pooled_text_dim), project to
        //     time_embed_dim, add onto `emb`.
        let (b_a, n_ids) = add_time_ids.dims2()?;
        if b_a != bsize {
            candle_core::bail!(
                "add_time_ids batch {b_a} mismatches input batch {bsize}"
            );
        }
        if n_ids != self.add_cfg.num_time_ids {
            candle_core::bail!(
                "add_time_ids has {n_ids} columns but SdxlAddEmbedConfig.num_time_ids = {}",
                self.add_cfg.num_time_ids
            );
        }
        let flat_ids = add_time_ids.reshape((b_a * n_ids,))?;
        let time_ids_emb = self.add_time_proj.forward(&flat_ids)?;
        let time_ids_emb = time_ids_emb.reshape((
            b_a,
            n_ids * self.add_cfg.addition_time_embed_dim,
        ))?;
        let add_in = Tensor::cat(&[&add_text_embeds.to_dtype(time_ids_emb.dtype())?, &time_ids_emb], 1)?;
        let aug_emb = self.add_embedding.forward(&add_in)?;
        let emb = emb.broadcast_add(&aug_emb.to_dtype(emb.dtype())?)?;

        // 2. pre-process
        let xs = self.conv_in.forward(&xs)?;

        // 3. down
        let mut down_block_res_xs = vec![xs.clone()];
        let mut xs = xs;
        for down_block in self.down_blocks.iter() {
            let (next_xs, res_xs) = match down_block {
                UNetDownBlock::Basic(b) => b.forward(&xs, Some(&emb))?,
                UNetDownBlock::CrossAttn(b) => {
                    b.forward(&xs, Some(&emb), Some(encoder_hidden_states))?
                }
            };
            down_block_res_xs.extend(res_xs);
            xs = next_xs;
        }

        let new_down_block_res_xs =
            if let Some(additional) = down_block_additional_residuals {
                let mut v = Vec::with_capacity(down_block_res_xs.len());
                for (i, residuals) in additional.iter().enumerate() {
                    v.push((&down_block_res_xs[i] + residuals)?)
                }
                v
            } else {
                down_block_res_xs
            };
        let mut down_block_res_xs = new_down_block_res_xs;

        // 4. mid
        let xs = self
            .mid_block
            .forward(&xs, Some(&emb), Some(encoder_hidden_states))?;
        let xs = match mid_block_additional_residual {
            None => xs,
            Some(m) => (m + xs)?,
        };

        // 5. up
        let mut xs = xs;
        let mut upsample_size = None;
        for (i, up_block) in self.up_blocks.iter().enumerate() {
            let n_resnets = match up_block {
                UNetUpBlock::Basic(b) => b.resnets.len(),
                UNetUpBlock::CrossAttn(b) => b.upblock.resnets.len(),
            };
            let res_xs = down_block_res_xs.split_off(down_block_res_xs.len() - n_resnets);
            if i < n_blocks - 1 && forward_upsample_size {
                let (_, _, h, w) = down_block_res_xs.last().unwrap().dims4()?;
                upsample_size = Some((h, w));
            }
            xs = match up_block {
                UNetUpBlock::Basic(b) => b.forward(&xs, &res_xs, Some(&emb), upsample_size)?,
                UNetUpBlock::CrossAttn(b) => b.forward(
                    &xs,
                    &res_xs,
                    Some(&emb),
                    upsample_size,
                    Some(encoder_hidden_states),
                )?,
            };
        }

        // 6. post-process
        let xs = self.conv_norm_out.forward(&xs)?;
        let xs = nn::ops::silu(&xs)?;
        self.conv_out.forward(&xs)
    }
}
