//! 2D UNet Denoising Models
//!
//! The 2D Unet models take as input a noisy sample and the current diffusion
//! timestep and return a denoised version of the input.
use candle_transformers::models::stable_diffusion::embeddings::{TimestepEmbedding, Timesteps};
use super::blocks::*;
use candle_nn::{conv2d, Conv2d};
use candle_core::{Result, Tensor};
use candle_nn as nn;
use candle_nn::Module;
use std::sync::{Arc, RwLock};
use crate::pipelines::lora_linear::LoraRegistry;

// Config is just data (no LoRA hooks) — reuse candle's public types so a
// `StableDiffusionConfig`'s `.unet` flows straight into our trainable UNet.
pub use candle_transformers::models::stable_diffusion::unet_2d::{
    BlockConfig, UNet2DConditionModelConfig,
};

#[derive(Debug)]
pub(crate) enum UNetDownBlock {
    Basic(DownBlock2D),
    CrossAttn(CrossAttnDownBlock2D),
}

#[derive(Debug)]
enum UNetUpBlock {
    Basic(UpBlock2D),
    CrossAttn(CrossAttnUpBlock2D),
}

#[derive(Debug)]
pub struct UNet2DConditionModel {
    conv_in: Conv2d,
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
    down_blocks: Vec<UNetDownBlock>,
    mid_block: UNetMidBlock2DCrossAttn,
    up_blocks: Vec<UNetUpBlock>,
    conv_norm_out: nn::GroupNorm,
    conv_out: Conv2d,
    span: tracing::Span,
    config: UNet2DConditionModelConfig,
    /// LoRA registry — path → entry for every CrossAttention q/k/v/out
    /// LoraLinear. `plakat style train` installs trainable adapters here.
    pub(crate) lora_registry: LoraRegistry,
}

impl UNet2DConditionModel {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        use_flash_attn: bool,
        config: UNet2DConditionModelConfig,
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

        // LoRA registry — populated by the CrossAttention `wrap_lin` calls
        // as the blocks below are built; unwrapped into the field at the end.
        let registry: Arc<RwLock<LoraRegistry>> = Arc::new(RwLock::new(LoraRegistry::new()));

        let vs_db = vs.pp("down_blocks");
        let down_blocks = (0..n_blocks)
            .map(|i| {
                let BlockConfig {
                    out_channels,
                    use_cross_attn,
                    attention_head_dim,
                } = config.blocks[i];

                // Enable automatic attention slicing if the config sliced_attention_size is set to 0.
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
                    let config = CrossAttnDownBlock2DConfig {
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
                        config,
                        &registry,
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

        // https://github.com/huggingface/diffusers/blob/a76f2ad538e73b34d5fe7be08c8eb8ab38c7e90c/src/diffusers/models/unet_2d_condition.py#L462
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
            &registry,
        )?;

        let vs_ub = vs.pp("up_blocks");
        let up_blocks = (0..n_blocks)
            .map(|i| {
                let BlockConfig {
                    out_channels,
                    use_cross_attn,
                    attention_head_dim,
                } = config.blocks[n_blocks - 1 - i];

                // Enable automatic attention slicing if the config sliced_attention_size is set to 0.
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
                    let config = CrossAttnUpBlock2DConfig {
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
                        config,
                        &registry,
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
        let span = tracing::span!(tracing::Level::TRACE, "unet2d");
        // All blocks built — reclaim the registry (refcount 1; blocks held
        // only `&registry`, never cloned the Arc).
        let lora_registry = Arc::try_unwrap(registry)
            .map_err(|_| candle_core::Error::Msg("sd_train registry still shared".into()))?
            .into_inner()
            .map_err(|_| candle_core::Error::Msg("sd_train registry poisoned".into()))?;
        Ok(Self {
            conv_in,
            time_proj,
            time_embedding,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            span,
            config,
            lora_registry,
        })
    }

    /// `plakat style train`: install a fresh trainable LoRA adapter on
    /// every wrapped CrossAttention projection (the whole registry — only
    /// q/k/v/out are wrapped). Returns `(key, A, B)` per target for the
    /// optimizer + kohya save. Init `A ~ N(0, 0.02)`, `B = 0` (no-op start),
    /// Vars F32 (mixed precision — the LoraLinear forward casts).
    pub fn install_train_adapters(
        &self,
        rank: usize,
        scale: f64,
        device: &candle_core::Device,
    ) -> Result<Vec<(String, candle_core::Var, candle_core::Var)>> {
        use candle_core::{DType, Tensor, Var};
        let mut keys: Vec<&String> = self.lora_registry.keys().collect();
        keys.sort();
        let mut out = Vec::with_capacity(keys.len());
        for key in keys {
            let entry = &self.lora_registry[key];
            let a = Var::from_tensor(&Tensor::randn(0f32, 0.02f32, (rank, entry.in_dim), device)?)?;
            let b = Var::from_tensor(&Tensor::zeros((entry.out_dim, rank), DType::F32, device)?)?;
            *entry
                .train
                .write()
                .map_err(|_| candle_core::Error::Msg("sd_train train slot poisoned".into()))? =
                Some((a.clone(), b.clone(), scale));
            out.push((key.clone(), a, b));
        }
        Ok(out)
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.forward_with_additional_residuals(xs, timestep, encoder_hidden_states, None, None)
    }

    pub fn forward_with_additional_residuals(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
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
        // 1. time
        let emb = (Tensor::ones(bsize, xs.dtype(), device)? * timestep)?;
        let emb = self.time_proj.forward(&emb)?;
        let emb = self.time_embedding.forward(&emb)?;
        // 2. pre-process
        let xs = self.conv_in.forward(&xs)?;
        // 3. down
        let mut down_block_res_xs = vec![xs.clone()];
        let mut xs = xs;
        for down_block in self.down_blocks.iter() {
            let (_xs, res_xs) = match down_block {
                UNetDownBlock::Basic(b) => b.forward(&xs, Some(&emb))?,
                UNetDownBlock::CrossAttn(b) => {
                    b.forward(&xs, Some(&emb), Some(encoder_hidden_states))?
                }
            };
            down_block_res_xs.extend(res_xs);
            xs = _xs;
        }

        let new_down_block_res_xs =
            if let Some(down_block_additional_residuals) = down_block_additional_residuals {
                let mut v = vec![];
                // A previous version of this code had a bug because of the addition being made
                // in place via += hence modifying the input of the mid block.
                for (i, residuals) in down_block_additional_residuals.iter().enumerate() {
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
                upsample_size = Some((h, w))
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
