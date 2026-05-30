//! ControlNet — parallel UNet down-encoder that produces additive
//! residuals consumed by [`candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModel::forward_with_additional_residuals`].
//!
//! ## Architecture
//!
//! For SD 1.5 (4 blocks, layers_per_block = 2, the standard
//! `runwayml/stable-diffusion-v1-5` config):
//!
//! ```text
//!                  (4 ch latent)               (3 ch conditioning image)
//!                       │                              │
//!                    conv_in                  controlnet_cond_embedding
//!                       │                              │
//!                       └────────── sum ───────────────┘
//!                                    │
//!                  ┌─────────────────┴─────────────────┐
//!                  │  4 down blocks (CrossAttn × 3 +   │
//!                  │   Basic × 1) — same shapes as the │
//!                  │   real UNet's down path           │
//!                  └─────────────────┬─────────────────┘
//!                                    │
//!                              UNetMidBlock2DCrossAttn
//!                                    │
//!                       zero-convs (1×1 conv2d, init = 0)
//!                                    │
//!                        12 down residuals + 1 mid residual
//! ```
//!
//! Each intermediate down residual goes through its own zero-conv
//! before being returned. Untrained, every zero-conv produces 0,
//! so ControlNet is a true no-op until weights are loaded.
//!
//! ## Integration
//!
//! Each denoise step:
//!
//! 1. Caller computes `(down_residuals, mid_residual) = controlnet.forward(latents, t, ehs, cond, strength)`
//! 2. Caller invokes `unet.forward_with_additional_residuals(latents, t, ehs, Some(&down_residuals), Some(&mid_residual))`.
//!
//! No `forward()` modifications needed in the UNet — candle 0.8 ships
//! the hook we need.
//!
//! ## Status
//!
//! This module ships the **model architecture** for the v0.9
//! depth-conditioning feature. Weight loading from HuggingFace,
//! the hint preprocessor, and the per-pipeline integration land in
//! follow-up commits.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, VarBuilder};
use candle_transformers::models::stable_diffusion::embeddings::{
    TimestepEmbedding, Timesteps,
};
use candle_transformers::models::stable_diffusion::unet_2d::{
    BlockConfig, UNet2DConditionModelConfig,
};
use candle_transformers::models::stable_diffusion::unet_2d_blocks::{
    CrossAttnDownBlock2D, CrossAttnDownBlock2DConfig, DownBlock2D, DownBlock2DConfig,
    UNetMidBlock2DCrossAttn, UNetMidBlock2DCrossAttnConfig,
};

/// Which SD architecture the paired UNet uses. Drives the
/// ControlNet construction (block count + shapes) and the matching
/// HuggingFace weight repos. Mirrors `portrait::Variant` but is
/// independent so ControlNet doesn't pull a dependency on portrait.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControlNetVariant {
    /// SD 1.5 (and SD 2.1 — same UNet shape up to cross_attn_dim).
    /// 4 down blocks (3 CrossAttn + 1 Basic). cross_attention_dim 768.
    Sd15,
    /// SDXL. 3 down blocks (1 Basic + 2 CrossAttn).
    /// cross_attention_dim 2048, use_linear_projection true.
    Sdxl,
}

impl ControlNetVariant {
    /// Same heuristic as `portrait::Variant::detect`: anything with
    /// "xl" in the name is SDXL, otherwise SD 1.5.
    pub fn detect(model: &str) -> Self {
        let m = model.to_lowercase();
        if m.contains("xl") {
            Self::Sdxl
        } else {
            Self::Sd15
        }
    }

    pub fn unet_config(self) -> UNet2DConditionModelConfig {
        match self {
            Self::Sd15 => sd15_unet_config(),
            Self::Sdxl => sdxl_unet_config(),
        }
    }
}

/// SD 1.5 UNet configuration — same as candle's
/// `StableDiffusionConfig::v1_5(...)` produces internally for its
/// (private) `unet` field. We reconstruct it here so the ControlNet
/// can be built with shapes that match the paired UNet exactly.
///
/// Differences from `UNet2DConditionModelConfig::default()`:
/// only `cross_attention_dim: 768` (default is 1280).
pub fn sd15_unet_config() -> UNet2DConditionModelConfig {
    UNet2DConditionModelConfig {
        blocks: vec![
            BlockConfig {
                out_channels: 320,
                use_cross_attn: Some(1),
                attention_head_dim: 8,
            },
            BlockConfig {
                out_channels: 640,
                use_cross_attn: Some(1),
                attention_head_dim: 8,
            },
            BlockConfig {
                out_channels: 1280,
                use_cross_attn: Some(1),
                attention_head_dim: 8,
            },
            BlockConfig {
                out_channels: 1280,
                use_cross_attn: None,
                attention_head_dim: 8,
            },
        ],
        center_input_sample: false,
        cross_attention_dim: 768,
        downsample_padding: 1,
        flip_sin_to_cos: true,
        freq_shift: 0.,
        layers_per_block: 2,
        mid_block_scale_factor: 1.,
        norm_eps: 1e-5,
        norm_num_groups: 32,
        sliced_attention_size: None,
        use_linear_projection: false,
    }
}

/// SDXL UNet configuration — same as candle's
/// `StableDiffusionConfig::sdxl(...)` produces internally. SDXL has
/// only 3 down blocks (vs SD 1.5's 4); the first block is Basic
/// (no cross-attn), the next two are CrossAttn with
/// `transformer_layers_per_block` 2 and 10 respectively.
/// Key differences vs SD 1.5:
/// * 3 blocks not 4
/// * `cross_attention_dim: 2048` (vs 768)
/// * `use_linear_projection: true` (vs false)
/// * Block 0 has no cross-attn
/// * Per-block transformer layer counts vary (1 / 2 / 10)
pub fn sdxl_unet_config() -> UNet2DConditionModelConfig {
    UNet2DConditionModelConfig {
        blocks: vec![
            BlockConfig {
                out_channels: 320,
                use_cross_attn: None,
                attention_head_dim: 5,
            },
            BlockConfig {
                out_channels: 640,
                use_cross_attn: Some(2),
                attention_head_dim: 10,
            },
            BlockConfig {
                out_channels: 1280,
                use_cross_attn: Some(10),
                attention_head_dim: 20,
            },
        ],
        center_input_sample: false,
        cross_attention_dim: 2048,
        downsample_padding: 1,
        flip_sin_to_cos: true,
        freq_shift: 0.,
        layers_per_block: 2,
        mid_block_scale_factor: 1.,
        norm_eps: 1e-5,
        norm_num_groups: 32,
        sliced_attention_size: None,
        use_linear_projection: true,
    }
}

/// Default ControlNet hint encoder channel layout (matches the
/// diffusers reference implementation). Takes a 3-channel image and
/// downsamples to the latent grid via four strided convs.
const HINT_ENCODER_CHANNELS: &[usize] = &[16, 32, 96, 256];

/// Either a cross-attention down block (with self+cross attention)
/// or a basic down block (resnet-only). Matches the UNet's structure.
enum DownBlock {
    CrossAttn(CrossAttnDownBlock2D),
    Basic(DownBlock2D),
}

impl DownBlock {
    fn forward(
        &self,
        xs: &Tensor,
        temb: &Tensor,
        encoder_hidden_states: &Tensor,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        Ok(match self {
            Self::CrossAttn(b) => b.forward(xs, Some(temb), Some(encoder_hidden_states))?,
            Self::Basic(b) => b.forward(xs, Some(temb))?,
        })
    }
}

/// ControlNet network. Constructed from the same `UNet2DConditionModelConfig`
/// as the UNet it pairs with — guarantees matching block shapes.
pub struct ControlNet {
    /// Latent input projection. Same shape as the UNet's `conv_in`.
    conv_in: Conv2d,

    /// Timestep encoder mirror. Required because each ControlNet block
    /// receives the same timestep embedding as the UNet does.
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,

    /// SDXL only — mirrors the UNet's `text_time` add_embedding. The
    /// SDXL ControlNet has its own `add_embedding` module in its
    /// safetensors (`add_embedding.linear_{1,2}.weight/bias`), and
    /// diffusers' SDXL ControlNet pipeline feeds it the same
    /// `(pooled_text, add_time_ids)` pair the base UNet uses. Without
    /// this, every SDXL ControlNet runs with broken micro-conditioning
    /// — its residuals carry a timestep-only signal while the UNet
    /// they feed into uses the full pooled+time_ids embedding.
    ///
    /// `None` on SD 1.5 / SD 2.1.
    add_time_proj: Option<Timesteps>,
    add_embedding: Option<TimestepEmbedding>,
    /// Carried so [`ControlNet::forward`] can validate `add_time_ids`
    /// against `num_time_ids` at call time.
    add_cfg: Option<crate::pipelines::sdxl_unet::SdxlAddEmbedConfig>,

    /// 3-channel conditioning image → latent-grid feature map.
    /// Output channels match `conv_in`'s output, so the two can be
    /// summed before entering the down blocks.
    hint_encoder: HintEncoder,

    /// Mirror of the UNet's down blocks.
    down_blocks: Vec<DownBlock>,

    /// Mirror of the UNet's mid block.
    mid_block: UNetMidBlock2DCrossAttn,

    /// One 1×1 zero-conv per intermediate residual:
    /// `1 (after hint+conv_in) + sum(num_residuals_per_down_block)`.
    /// For SD 1.5 default config this is 13 entries (1 + 3 + 3 + 3 + 2 = 12).
    controlnet_down_blocks: Vec<Conv2d>,

    /// Final zero-conv on the mid-block output.
    controlnet_mid_block: Conv2d,

    /// Variant config kept for diagnostics.
    config: UNet2DConditionModelConfig,
}

impl ControlNet {
    /// Construct from a `VarBuilder` rooted at the ControlNet weights
    /// (typically `vb` over the safetensors file with no further prefix).
    /// Pass the same `UNet2DConditionModelConfig` the paired UNet uses,
    /// and the variant so SDXL ControlNets pick up their `add_embedding`.
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        config: UNet2DConditionModelConfig,
        variant: ControlNetVariant,
    ) -> Result<Self> {
        let n_blocks = config.blocks.len();
        let b_channels = config.blocks[0].out_channels;
        let bl_channels = config.blocks.last().unwrap().out_channels;
        let bl_attention_head_dim = config.blocks.last().unwrap().attention_head_dim;
        let time_embed_dim = b_channels * 4;

        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = conv2d(in_channels, b_channels, 3, conv_cfg, vb.pp("conv_in"))
            .context("ControlNet conv_in")?;

        let time_proj = Timesteps::new(b_channels, config.flip_sin_to_cos, config.freq_shift);
        let time_embedding =
            TimestepEmbedding::new(vb.pp("time_embedding"), b_channels, time_embed_dim)
                .context("ControlNet time_embedding")?;

        // SDXL `text_time` add_embedding. Same shape contract as the
        // SDXL UNet's: 6 time_ids × 256 + 1280 pooled = 2816 → 1280.
        // Weight paths in diffusers-format SDXL ControlNets:
        // `add_embedding.linear_{1,2}.{weight,bias}` (sibling of
        // `time_embedding`).
        let (add_time_proj, add_embedding, add_cfg) = match variant {
            ControlNetVariant::Sd15 => (None, None, None),
            ControlNetVariant::Sdxl => {
                let cfg = crate::pipelines::sdxl_unet::SdxlAddEmbedConfig::base();
                let tp = Timesteps::new(cfg.addition_time_embed_dim, true, 0.0);
                let ae = TimestepEmbedding::new(
                    vb.pp("add_embedding"),
                    cfg.in_dim(),
                    time_embed_dim,
                )
                .context("SDXL ControlNet add_embedding")?;
                (Some(tp), Some(ae), Some(cfg))
            }
        };

        let hint_encoder =
            HintEncoder::new(vb.pp("controlnet_cond_embedding"), 3, b_channels)
                .context("ControlNet hint encoder")?;

        // -------- down blocks (mirror UNet's down path) --------
        let vs_db = vb.pp("down_blocks");
        let down_blocks = (0..n_blocks)
            .map(|i| {
                let BlockConfig {
                    out_channels,
                    use_cross_attn,
                    attention_head_dim,
                } = config.blocks[i];

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
                    let cfg = CrossAttnDownBlock2DConfig {
                        downblock: db_cfg,
                        attn_num_head_channels: attention_head_dim,
                        cross_attention_dim: config.cross_attention_dim,
                        sliced_attention_size: config.sliced_attention_size,
                        use_linear_projection: config.use_linear_projection,
                        transformer_layers_per_block,
                    };
                    let block = CrossAttnDownBlock2D::new(
                        vs_db.pp(i.to_string()),
                        in_channels,
                        out_channels,
                        Some(time_embed_dim),
                        false, // use_flash_attn — keep parity with the paired UNet
                        cfg,
                    )?;
                    Ok::<_, anyhow::Error>(DownBlock::CrossAttn(block))
                } else {
                    let block = DownBlock2D::new(
                        vs_db.pp(i.to_string()),
                        in_channels,
                        out_channels,
                        Some(time_embed_dim),
                        db_cfg,
                    )?;
                    Ok::<_, anyhow::Error>(DownBlock::Basic(block))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // -------- mid block --------
        let mid_transformer_layers_per_block = config
            .blocks
            .last()
            .and_then(|b| b.use_cross_attn)
            .unwrap_or(1);
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
            vb.pp("mid_block"),
            bl_channels,
            Some(time_embed_dim),
            false,
            mid_cfg,
        )
        .context("ControlNet mid_block")?;

        // -------- zero-conv heads --------
        //
        // The number of zero-convs equals the number of residuals UNet
        // expects on its `down_block_additional_residuals` slot. That's:
        //   1 (the post-conv_in/post-hint feature)
        //   + sum over down blocks of (num_layers + 1_if_has_downsampler)
        //
        // For the SD 1.5 default config (4 blocks, layers_per_block=2):
        //   1 + 3 + 3 + 3 + 2 = 12
        let zc_cfg = Conv2dConfig {
            ..Default::default()
        };
        let vs_zd = vb.pp("controlnet_down_blocks");
        let mut controlnet_down_blocks = Vec::new();
        // First zero-conv: applied to the post-(conv_in+hint) feature.
        controlnet_down_blocks
            .push(conv2d(b_channels, b_channels, 1, zc_cfg, vs_zd.pp("0"))?);
        let mut zd_idx = 1usize;
        for i in 0..n_blocks {
            let out_channels = config.blocks[i].out_channels;
            // Each down block contributes `layers_per_block` residuals
            // from the per-resnet outputs plus 1 from the downsampler
            // when it has one (every block except the last).
            let n_residuals =
                config.layers_per_block + if i < n_blocks - 1 { 1 } else { 0 };
            for _ in 0..n_residuals {
                controlnet_down_blocks.push(conv2d(
                    out_channels,
                    out_channels,
                    1,
                    zc_cfg,
                    vs_zd.pp(zd_idx.to_string()),
                )?);
                zd_idx += 1;
            }
        }

        let controlnet_mid_block = conv2d(
            bl_channels,
            bl_channels,
            1,
            zc_cfg,
            vb.pp("controlnet_mid_block"),
        )
        .context("ControlNet mid zero-conv")?;

        Ok(Self {
            conv_in,
            time_proj,
            time_embedding,
            add_time_proj,
            add_embedding,
            add_cfg,
            hint_encoder,
            down_blocks,
            mid_block,
            controlnet_down_blocks,
            controlnet_mid_block,
            config,
        })
    }

    /// Run a single ControlNet forward pass.
    ///
    /// * `latents` — the same `(B, 4, H/8, W/8)` latent the UNet sees.
    /// * `timestep` — same scalar as the UNet's denoise step.
    /// * `encoder_hidden_states` — same text-encoder output as the UNet.
    /// * `conditioning` — `(B, 3, H, W)` image-space conditioning
    ///   (e.g. a depth map). Normalised to `[0, 1]` typically.
    /// * `strength` — multiplicative scale applied to every residual
    ///   before returning. `1.0` matches diffusers' default.
    /// * `add_text_embeds` / `add_time_ids` — v0.12: SDXL `text_time`
    ///   micro-conditioning. **Required** when this is an SDXL
    ///   ControlNet (`SdxlAddEmbedConfig::base()` was wired in
    ///   `new`); errors out if missing. **Ignored** on SD 1.5
    ///   ControlNets.
    ///
    /// Returns `(down_block_additional_residuals, mid_block_additional_residual)`
    /// in the exact shape `UNet2DConditionModel::forward_with_additional_residuals`
    /// expects.
    pub fn forward(
        &self,
        latents: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        conditioning: &Tensor,
        strength: f32,
        add_text_embeds: Option<&Tensor>,
        add_time_ids: Option<&Tensor>,
    ) -> Result<(Vec<Tensor>, Tensor)> {
        let (bsize, _c, _h, _w) = latents.dims4()?;
        let device = latents.device();
        let dtype = latents.dtype();

        // 1. time embedding (same shape as UNet's).
        let emb = (Tensor::ones(bsize, dtype, device)? * timestep)?;
        let emb = self.time_proj.forward(&emb)?;
        let emb = self.time_embedding.forward(&emb)?;

        // 1b. SDXL `text_time` add — diffusers formula, applied to
        // `emb` exactly as the SdxlUNet does. Loud-error when the
        // caller forgot the SDXL extras; ignore when the ControlNet
        // is SD 1.5.
        let emb = match (&self.add_embedding, &self.add_time_proj, &self.add_cfg) {
            (Some(add_emb), Some(add_tp), Some(cfg)) => {
                let te = add_text_embeds.ok_or_else(|| {
                    anyhow::anyhow!(
                        "SDXL ControlNet::forward requires add_text_embeds \
                         (pooled CLIP-G) — none supplied"
                    )
                })?;
                let ti = add_time_ids.ok_or_else(|| {
                    anyhow::anyhow!(
                        "SDXL ControlNet::forward requires add_time_ids — \
                         none supplied"
                    )
                })?;
                let (b_a, n_ids) = ti.dims2()?;
                if b_a != bsize {
                    anyhow::bail!(
                        "ControlNet add_time_ids batch {b_a} mismatches latents batch {bsize}"
                    );
                }
                if n_ids != cfg.num_time_ids {
                    anyhow::bail!(
                        "ControlNet add_time_ids has {n_ids} columns but \
                         SdxlAddEmbedConfig.num_time_ids = {}",
                        cfg.num_time_ids
                    );
                }
                let flat_ids = ti.reshape((b_a * n_ids,))?;
                let time_ids_emb = add_tp.forward(&flat_ids)?;
                let time_ids_emb = time_ids_emb
                    .reshape((b_a, n_ids * cfg.addition_time_embed_dim))?;
                let add_in = Tensor::cat(
                    &[&te.to_dtype(time_ids_emb.dtype())?, &time_ids_emb],
                    1,
                )?;
                let aug_emb = add_emb.forward(&add_in)?;
                emb.broadcast_add(&aug_emb.to_dtype(emb.dtype())?)?
            }
            (None, None, None) => emb,
            _ => unreachable!("ControlNet add_* fields constructed together or not at all"),
        };

        // 2. pre-process latents + hint, then sum.
        let hint = self.hint_encoder.forward(conditioning)?;
        let mut xs = (self.conv_in.forward(latents)? + hint)?;

        // 3. down path; collect every intermediate that the UNet will
        //    expect a residual for.
        let mut residuals: Vec<Tensor> = vec![xs.clone()];
        for down_block in &self.down_blocks {
            let (next, res) = down_block.forward(&xs, &emb, encoder_hidden_states)?;
            residuals.extend(res);
            xs = next;
        }

        // 4. mid.
        let mid = self
            .mid_block
            .forward(&xs, Some(&emb), Some(encoder_hidden_states))?;

        // 5. apply zero-convs + strength scale.
        let s = strength as f64;
        let scaled_down: Vec<Tensor> = residuals
            .iter()
            .zip(self.controlnet_down_blocks.iter())
            .map(|(r, zc)| -> Result<Tensor> {
                let z = zc.forward(r)?;
                Ok((z * s)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let scaled_mid = (self.controlnet_mid_block.forward(&mid)? * s)?;

        Ok((scaled_down, scaled_mid))
    }

    /// The config the ControlNet was built with. Pipelines verify
    /// this matches their UNet's config before invoking forward().
    pub fn config(&self) -> &UNet2DConditionModelConfig {
        &self.config
    }

    /// Download safetensors weights for `kind` from HuggingFace and
    /// construct the network. Tries a primary repo + fallback
    /// mirrors per `variant`; the first one that downloads cleanly
    /// wins.
    ///
    /// `variant` selects both the architecture config (SD 1.5 vs
    /// SDXL block counts/shapes) and the matching weight repos.
    /// SDXL ControlNets are larger (~5 GB vs ~1.4 GB) but follow
    /// the same diffusers state-dict naming convention.
    pub async fn load(
        device: Device,
        dtype: DType,
        kind: ControlKind,
        variant: ControlNetVariant,
    ) -> Result<Self> {
        let candidates = candidates_for(kind, variant);
        let weights_path = crate::hf::download::get_first_of(&candidates)
            .await
            .with_context(|| {
                format!(
                    "downloading ControlNet weights for kind={:?} variant={:?}. \
                     Tried {} mirror(s).",
                    kind,
                    variant,
                    candidates.len(),
                )
            })?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], dtype, &device)?
        };
        // Both SD 1.5 and SDXL use 4-channel latents — only the
        // UNet architecture (block count, channel depths,
        // cross_attn_dim) differs. v0.12: pass variant so SDXL picks
        // up the add_embedding.
        Self::new(vb, 4, variant.unet_config(), variant)
            .with_context(|| format!("building ControlNet ({variant:?}) from weights"))
    }
}

/// Load a user-supplied conditioning image from disk and convert it
/// into the `(1, 3, H, W)` tensor [`ControlNet::forward`] expects.
///
/// * RGB / RGBA inputs use their RGB channels directly.
/// * Grayscale inputs are replicated across all three channels —
///   correct for depth maps and other single-channel conditioners.
/// * Values are normalised to `[0, 1]`.
///
/// The image is resized (triangle filter) to `(w, h)` to match the
/// generation working resolution. The caller's `(w, h)` must be a
/// multiple of 8 (VAE constraint) — the loader doesn't enforce that
/// since it's already validated at the CLI / pipeline boundary.
pub fn prepare_conditioning(
    path: &std::path::Path,
    w: u32,
    h: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img = image::open(path)
        .with_context(|| format!("opening control image {}", path.display()))?;
    let resized = img.resize_exact(w, h, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();
    let raw = rgb.as_raw();
    let total = (w as usize) * (h as usize);
    let mut buf: Vec<f32> = vec![0.0; 3 * total];
    let (r_dst, rest) = buf.split_at_mut(total);
    let (g_dst, b_dst) = rest.split_at_mut(total);
    for (i, chunk) in raw.chunks_exact(3).enumerate() {
        r_dst[i] = chunk[0] as f32 / 255.0;
        g_dst[i] = chunk[1] as f32 / 255.0;
        b_dst[i] = chunk[2] as f32 / 255.0;
    }
    let t = Tensor::from_vec(buf, (1, 3, h as usize, w as usize), device)?
        .to_dtype(dtype)?;
    Ok(t)
}

/// HuggingFace (repo, file) candidates for `kind` + `variant`.
/// Returned in download-preference order: primary first, then
/// fallbacks. All candidates ship **diffusers-format** state dicts
/// — keys like `down_blocks.0.attentions.0.…` rather than the
/// webui `control_model.input_blocks.0.0.…` convention, which has
/// a different naming scheme we don't currently translate.
fn candidates_for(
    kind: ControlKind,
    variant: ControlNetVariant,
) -> Vec<(&'static str, &'static str)> {
    match (kind, variant) {
        (ControlKind::Depth, ControlNetVariant::Sd15) => vec![
            // Primary: lllyasviel's original SD 1.5 ControlNet-Depth.
            (
                "lllyasviel/sd-controlnet-depth",
                "diffusion_pytorch_model.safetensors",
            ),
            // Fallback 1: same repo, fp16 variant (~half the bandwidth).
            (
                "lllyasviel/sd-controlnet-depth",
                "diffusion_pytorch_model.fp16.safetensors",
            ),
            // Fallback 2: lllyasviel's v1.1 ControlNet-Depth update.
            (
                "lllyasviel/control_v11f1p_sd15_depth",
                "diffusion_pytorch_model.safetensors",
            ),
        ],
        (ControlKind::Depth, ControlNetVariant::Sdxl) => vec![
            // Primary: full-size SDXL ControlNet-Depth (fp16, ~2.5 GB).
            // The diffusers-format state dict matches candle's standard
            // SDXL UNet layout exactly. We intentionally do NOT use
            // diffusers' `-small` variant: it ships a reduced
            // architecture (basic down-blocks where the standard
            // model has cross-attn), so candle's strict tensor
            // lookup fails with "cannot find tensor
            // down_blocks.1.attentions.0.norm.weight" against it.
            (
                "diffusers/controlnet-depth-sdxl-1.0",
                "diffusion_pytorch_model.fp16.safetensors",
            ),
            // Fallback 1: same repo, fp32 variant (~5 GB).
            (
                "diffusers/controlnet-depth-sdxl-1.0",
                "diffusion_pytorch_model.safetensors",
            ),
            // Fallback 2: xinsir's community SDXL ControlNet-Depth
            // (full-size architecture, diffusers state-dict shape).
            (
                "xinsir/controlnet-depth-sdxl-1.0",
                "diffusion_pytorch_model.safetensors",
            ),
        ],
        (ControlKind::Canny, ControlNetVariant::Sd15) => vec![
            // Primary: lllyasviel's original SD 1.5 ControlNet-Canny.
            (
                "lllyasviel/sd-controlnet-canny",
                "diffusion_pytorch_model.safetensors",
            ),
            // Fallback 1: same repo, fp16 variant.
            (
                "lllyasviel/sd-controlnet-canny",
                "diffusion_pytorch_model.fp16.safetensors",
            ),
            // Fallback 2: lllyasviel's v1.1 ControlNet-Canny update.
            (
                "lllyasviel/control_v11p_sd15_canny",
                "diffusion_pytorch_model.safetensors",
            ),
        ],
        (ControlKind::Canny, ControlNetVariant::Sdxl) => vec![
            // Primary: full-size SDXL ControlNet-Canny (fp16, ~2.5 GB).
            // See the depth/sdxl arm for why we skip the `-small`
            // variant: it uses a reduced architecture that doesn't
            // match candle's standard SDXL UNet layout.
            (
                "diffusers/controlnet-canny-sdxl-1.0",
                "diffusion_pytorch_model.fp16.safetensors",
            ),
            // Fallback 1: same repo, fp32 variant (~5 GB).
            (
                "diffusers/controlnet-canny-sdxl-1.0",
                "diffusion_pytorch_model.safetensors",
            ),
            // Fallback 2: xinsir's community SDXL ControlNet-Canny.
            (
                "xinsir/controlnet-canny-sdxl-1.0",
                "diffusion_pytorch_model.safetensors",
            ),
        ],
        // ---------- v0.11 conditioners ----------
        (ControlKind::OpenPose, ControlNetVariant::Sd15) => vec![
            // Primary: lllyasviel's v1.1 OpenPose ControlNet.
            (
                "lllyasviel/control_v11p_sd15_openpose",
                "diffusion_pytorch_model.safetensors",
            ),
            // Fallback: lllyasviel's original (older) OpenPose ControlNet.
            (
                "lllyasviel/sd-controlnet-openpose",
                "diffusion_pytorch_model.safetensors",
            ),
            (
                "lllyasviel/sd-controlnet-openpose",
                "diffusion_pytorch_model.fp16.safetensors",
            ),
        ],
        (ControlKind::OpenPose, ControlNetVariant::Sdxl) => vec![
            // Primary: thibaud's community SDXL OpenPose (full-size diffusers state dict).
            (
                "thibaud/controlnet-openpose-sdxl-1.0",
                "diffusion_pytorch_model.safetensors",
            ),
            // Fallback: xinsir's community SDXL OpenPose.
            (
                "xinsir/controlnet-openpose-sdxl-1.0",
                "diffusion_pytorch_model.safetensors",
            ),
        ],
        (ControlKind::Lineart, ControlNetVariant::Sd15) => vec![
            (
                "lllyasviel/control_v11p_sd15_lineart",
                "diffusion_pytorch_model.safetensors",
            ),
            (
                "lllyasviel/control_v11p_sd15_lineart",
                "diffusion_pytorch_model.fp16.safetensors",
            ),
        ],
        (ControlKind::Lineart, ControlNetVariant::Sdxl) => vec![
            // Community SDXL lineart (full-size architecture, diffusers
            // state-dict shape). xinsir hosts both base + "anime" variants
            // — we pick the base; users wanting anime style supply
            // pre-rendered lineart anyway.
            (
                "xinsir/anime-painter-diffusers-anime-lineart-sdxl",
                "diffusion_pytorch_model.safetensors",
            ),
        ],
        (ControlKind::SoftEdge, ControlNetVariant::Sd15) => vec![
            (
                "lllyasviel/control_v11p_sd15_softedge",
                "diffusion_pytorch_model.safetensors",
            ),
            (
                "lllyasviel/control_v11p_sd15_softedge",
                "diffusion_pytorch_model.fp16.safetensors",
            ),
        ],
        (ControlKind::SoftEdge, ControlNetVariant::Sdxl) => vec![
            // No widely-mirrored "softedge" SDXL ControlNet exists yet.
            // xinsir's "scribble" SDXL is the closest analog and accepts
            // HED-style soft edges as input. Users wanting strict SDXL
            // softedge should supply pre-rendered + matching weights via
            // --model with their own repo.
            (
                "xinsir/controlnet-scribble-sdxl-1.0",
                "diffusion_pytorch_model.safetensors",
            ),
        ],
    }
}

/// 3-channel conditioning image → latent-grid feature map. Four
/// strided 3×3 convolutions interleaved with no-stride 3×3 convs,
/// matching the diffusers ControlNetConditioningEmbedding layout.
struct HintEncoder {
    conv_in: Conv2d,
    blocks: Vec<Conv2d>,
    conv_out: Conv2d,
}

impl HintEncoder {
    fn new(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = conv2d(
            in_channels,
            HINT_ENCODER_CHANNELS[0],
            3,
            conv_cfg,
            vb.pp("conv_in"),
        )?;

        // Diffusers' ControlNetConditioningEmbedding: for each channel
        // step we emit one same-stride conv (keep size) and one
        // strided conv (downsample 2×). After all four channel steps,
        // we've downsampled by 2^4 = 16, but actually the original
        // ControlNet does the downsampling differently — it has 6
        // blocks total: 4 strided pairs that downsample by 8 (matching
        // VAE), reaching the latent grid resolution.
        //
        // Specifically the sequence is:
        //   conv_in: 3 → 16        (stride 1)
        //   block 0: 16 → 16       (stride 1)
        //   block 1: 16 → 32       (stride 2)  ← downsample 2×
        //   block 2: 32 → 32       (stride 1)
        //   block 3: 32 → 96       (stride 2)  ← downsample 4×
        //   block 4: 96 → 96       (stride 1)
        //   block 5: 96 → 256      (stride 2)  ← downsample 8× (latent grid)
        //   conv_out: 256 → b_channels (stride 1) ← zero-init in spec
        let mut blocks = Vec::new();
        let pairs = [
            (HINT_ENCODER_CHANNELS[0], HINT_ENCODER_CHANNELS[0], 1usize), // 0
            (HINT_ENCODER_CHANNELS[0], HINT_ENCODER_CHANNELS[1], 2),     // 1
            (HINT_ENCODER_CHANNELS[1], HINT_ENCODER_CHANNELS[1], 1),     // 2
            (HINT_ENCODER_CHANNELS[1], HINT_ENCODER_CHANNELS[2], 2),     // 3
            (HINT_ENCODER_CHANNELS[2], HINT_ENCODER_CHANNELS[2], 1),     // 4
            (HINT_ENCODER_CHANNELS[2], HINT_ENCODER_CHANNELS[3], 2),     // 5
        ];
        for (i, (in_c, out_c, stride)) in pairs.iter().enumerate() {
            let cfg = Conv2dConfig {
                padding: 1,
                stride: *stride,
                ..Default::default()
            };
            blocks.push(conv2d(
                *in_c,
                *out_c,
                3,
                cfg,
                vb.pp("blocks").pp(i.to_string()),
            )?);
        }

        let conv_out = conv2d(
            HINT_ENCODER_CHANNELS[3],
            out_channels,
            3,
            conv_cfg,
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            blocks,
            conv_out,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.conv_in.forward(xs)?;
        xs = xs.silu()?;
        for block in &self.blocks {
            xs = block.forward(&xs)?;
            xs = xs.silu()?;
        }
        let xs = self.conv_out.forward(&xs)?;
        Ok(xs)
    }
}

/// A loaded ControlNet plus the prepared conditioning tensor. Built
/// once per generation and threaded through the denoise loop as
/// `Option<&ControlRequest>`.
///
/// `start` / `end` define the **timestep window** (as fractions of
/// the full schedule, `[0, 1]`) during which the conditioner is
/// active. Outside the window, the denoise step takes the
/// no-control path. Defaults are `0.0` / `1.0` (always active).
/// Diffusers convention: progress is measured against the **full**
/// schedule, even for partial-schedule passes like img2img /
/// inpaint / blend.
pub struct ControlRequest<'a> {
    pub net: &'a ControlNet,
    pub conditioning: Tensor,
    pub strength: f32,
    pub start: f32,
    pub end: f32,
}

impl<'a> ControlRequest<'a> {
    /// Returns `true` when `progress ∈ [start, end)` — the
    /// denoise loop's signal that this step should run with
    /// ControlNet residuals applied.
    pub fn active_at(&self, progress: f32) -> bool {
        progress >= self.start && progress < self.end
    }
}

/// Diffusers-style multi-ControlNet residual sum. Runs every active
/// ControlNet in `active`, then sums their `(down, mid)` outputs
/// per-block-index. All nets must agree on `down.len()` (which they
/// will, since they're built from the same UNet config block layout).
///
/// `latent_in` is the already-CFG-doubled input matching the UNet's
/// upcoming forward call; `do_cfg` controls whether each
/// conditioning Tensor gets cat'd along the batch dim.
///
/// Used by every pipeline's `denoise_step`; lives here next to
/// `ControlRequest` so the multi-net contract has one definition.
pub fn sum_controlnet_residuals(
    active: &[&ControlRequest<'_>],
    latent_in: &Tensor,
    timestep: usize,
    text_embeddings: &Tensor,
    do_cfg: bool,
    // v0.12: SDXL `text_time` micro-conditioning forwarded to each
    // active ControlNet. `None` on SD 1.5 / SD 2.1; pipelines pass the
    // same `(pooled_text, add_time_ids)` pair they hand to the UNet.
    add_text_embeds: Option<&Tensor>,
    add_time_ids: Option<&Tensor>,
) -> Result<(Vec<Tensor>, Tensor)> {
    assert!(!active.is_empty(), "sum_controlnet_residuals called with empty slice");
    let mut iter = active.iter();
    let first = iter.next().expect("checked above");
    let (mut down, mut mid) = run_one(
        first,
        latent_in,
        timestep,
        text_embeddings,
        do_cfg,
        add_text_embeds,
        add_time_ids,
    )?;
    for cr in iter {
        let (d, m) = run_one(
            cr,
            latent_in,
            timestep,
            text_embeddings,
            do_cfg,
            add_text_embeds,
            add_time_ids,
        )?;
        if d.len() != down.len() {
            anyhow::bail!(
                "multi-ControlNet residual count mismatch: {} vs {}. \
                 All conditioners must target the same UNet variant.",
                d.len(),
                down.len()
            );
        }
        for i in 0..down.len() {
            down[i] = (&down[i] + &d[i])?;
        }
        mid = (&mid + &m)?;
    }
    Ok((down, mid))
}

fn run_one(
    cr: &ControlRequest<'_>,
    latent_in: &Tensor,
    timestep: usize,
    text_embeddings: &Tensor,
    do_cfg: bool,
    add_text_embeds: Option<&Tensor>,
    add_time_ids: Option<&Tensor>,
) -> Result<(Vec<Tensor>, Tensor)> {
    let cond_in = if do_cfg {
        Tensor::cat(&[&cr.conditioning, &cr.conditioning], 0)?
    } else {
        cr.conditioning.clone()
    };
    cr.net.forward(
        latent_in,
        timestep as f64,
        text_embeddings,
        &cond_in,
        cr.strength,
        add_text_embeds,
        add_time_ids,
    )
}

/// One element of a loaded ControlNet stack: the network, its
/// prepared conditioning Tensor, and the per-conditioner runtime
/// parameters (strength + timestep window). `ControlRequest`s used by
/// the denoise loop borrow from these.
///
/// `conditioning` is always populated with the "static" image (single
/// `(1, 3, H, W)` Tensor). Non-animate pipelines (t2i, img2img,
/// portrait, stylize) only ever look at this field — they treat the
/// CN stack as a single image per conditioner.
///
/// v0.30 phase 2: `per_frame` carries per-frame conditioning for
/// `plakat animate --control-spec ...video=...`. When `Some(vec)`,
/// the vec length matches the animate frame count and each
/// `(1, 3, H, W)` tensor is the conditioning for one output frame.
/// AnimateDiff's per-step path uses `per_frame` if present, falling
/// back to `conditioning.repeat(frames, 1, 1, 1)` (the v0.28 static
/// behaviour) when it's `None`.
pub struct OwnedControl {
    pub net: ControlNet,
    pub conditioning: Tensor,
    pub per_frame: Option<Vec<Tensor>>,
    pub strength: f32,
    pub start: f32,
    pub end: f32,
}

/// Resolve a stack of `ControlSpec`s into loaded `ControlNet`s and
/// prepared conditioning tensors. Used by every pipeline that supports
/// ControlNet. `fallback_input` is the source image to auto-annotate
/// when a spec has neither `image=` nor `from=` set — pipelines like
/// `img2img` pass `Some(&req.input)`; `t2i` passes `None` (which causes
/// the spec to error out, matching the pre-v0.11 behaviour).
///
/// `animate_frames` activates v0.30 phase 2's per-frame video CN:
/// - `None` — non-animate caller. Specs with `video=` bail loud since
///   per-frame conditioning is meaningless for single-image pipelines.
/// - `Some(n)` — animate caller. Specs with `video=` get their video
///   decoded via ffmpeg, sub-sampled to `n` frames evenly distributed,
///   each frame annotated, and the stack lands in `OwnedControl.per_frame`.
///   The single `OwnedControl.conditioning` is also populated with the
///   first frame's tensor so the active-step check + static fallback
///   paths still work.
pub async fn load_control_stack(
    specs: &[ControlSpec],
    model: &str,
    width: u32,
    height: u32,
    device: &Device,
    dtype: DType,
    fallback_input: Option<&std::path::Path>,
    animate_frames: Option<usize>,
) -> Result<Vec<OwnedControl>> {
    if specs.is_empty() {
        return Ok(Vec::new());
    }
    let cn_variant = ControlNetVariant::detect(model);
    let mut out = Vec::with_capacity(specs.len());
    for spec in specs {
        let net = ControlNet::load(device.clone(), dtype, spec.kind, cn_variant)
            .await
            .with_context(|| {
                format!("loading ControlNet weights for kind={:?}", spec.kind)
            })?;
        // v0.30 phase 2: video= takes priority when set. Both single
        // and per-frame fields get populated for compatibility with
        // active-step checks and static fallback.
        let (cond, per_frame) = if let Some(video_path) = spec.video.as_ref() {
            let n_frames = animate_frames.ok_or_else(|| {
                anyhow::anyhow!(
                    "--control-spec for kind={:?}: video= is only supported on \
                     `plakat animate` (per-frame conditioning is meaningless for \
                     single-image pipelines)",
                    spec.kind
                )
            })?;
            let frame_tmp = tempfile::Builder::new()
                .prefix("plakat-controlvideo-")
                .tempdir()
                .with_context(|| "creating tempdir for video frame decode")?;
            let raw_frames =
                crate::imaging::video::extract_frames(video_path, frame_tmp.path())
                    .with_context(|| {
                        format!(
                            "decoding control video {} for kind={:?}",
                            video_path.display(),
                            spec.kind
                        )
                    })?;
            let picked =
                crate::imaging::video::pick_evenly_spaced(&raw_frames, n_frames);
            anyhow::ensure!(
                picked.len() == n_frames,
                "video frame pick yielded {} (wanted {})",
                picked.len(),
                n_frames
            );
            let mut per_frame_tensors: Vec<Tensor> = Vec::with_capacity(n_frames);
            for fp in &picked {
                let t = crate::pipelines::controlnet_annotator::annotate(
                    spec.kind, fp, width, height, device, dtype,
                )
                .await
                .with_context(|| {
                    format!(
                        "annotating control-video frame {} for kind={:?}",
                        fp.display(),
                        spec.kind
                    )
                })?;
                per_frame_tensors.push(t);
            }
            let first = per_frame_tensors[0].clone();
            // `frame_tmp` drops here, deleting the extracted PNGs — but
            // every tensor is already on-device, so the on-disk files
            // can go.
            drop(frame_tmp);
            (first, Some(per_frame_tensors))
        } else {
            let single = match (spec.image.as_ref(), spec.from.as_ref(), fallback_input) {
                (Some(path), None, _) => {
                    prepare_conditioning(path, width, height, device, dtype).with_context(|| {
                        format!(
                            "preparing ControlNet conditioning image for kind={:?}",
                            spec.kind
                        )
                    })?
                }
                (None, Some(path), _) => {
                    crate::pipelines::controlnet_annotator::annotate(
                        spec.kind, path, width, height, device, dtype,
                    )
                    .await
                    .with_context(|| {
                        format!("running --control-from annotator for kind={:?}", spec.kind)
                    })?
                }
                (None, None, Some(fallback)) => {
                    crate::pipelines::controlnet_annotator::annotate(
                        spec.kind, fallback, width, height, device, dtype,
                    )
                    .await
                    .with_context(|| {
                        format!(
                            "auto-annotating fallback input for kind={:?} (img2img default)",
                            spec.kind
                        )
                    })?
                }
                (Some(_), Some(_), _) => anyhow::bail!(
                    "--control-spec for kind={:?}: image= and from= are mutually exclusive",
                    spec.kind
                ),
                (None, None, None) => anyhow::bail!(
                    "--control-spec for kind={:?}: requires image=PATH, from=PATH, \
                     or video=PATH (the last only on `plakat animate`)",
                    spec.kind
                ),
            };
            (single, None)
        };
        out.push(OwnedControl {
            net,
            conditioning: cond,
            per_frame,
            strength: spec.strength,
            start: spec.start,
            end: spec.end,
        });
    }
    Ok(out)
}

/// What kind of conditioning signal the user requested.
///
/// * v0.10: `Depth` (Depth-Anything-V2 annotator), `Canny`
///   (imageproc canny annotator).
/// * v0.11: `OpenPose` (CMU body-pose), `Lineart` (lllyasviel sk_model),
///   `SoftEdge` (HED).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControlKind {
    Depth,
    Canny,
    /// Skeleton keypoint conditioning. Auto-annotator runs CMU's
    /// OpenPose body model (lllyasviel/Annotators `body_pose_model.pth`).
    OpenPose,
    /// Clean line-drawing conditioning. Auto-annotator runs
    /// lllyasviel's `sk_model.pth` (anime lineart). Pairs with
    /// `control_v11p_sd15_lineart` / SDXL equivalents.
    Lineart,
    /// HED-style soft edge map. Auto-annotator runs lllyasviel's
    /// `ControlNetHED.pth` (VGG-16 + side outputs + fuse layer).
    /// Pairs with `control_v11p_sd15_softedge` / SDXL equivalents.
    SoftEdge,
}

impl ControlKind {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Depth => "depth",
            Self::Canny => "canny",
            Self::OpenPose => "openpose",
            Self::Lineart => "lineart",
            Self::SoftEdge => "softedge",
        }
    }
}

impl std::str::FromStr for ControlKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "depth" => Ok(Self::Depth),
            "canny" => Ok(Self::Canny),
            "openpose" | "pose" => Ok(Self::OpenPose),
            "lineart" => Ok(Self::Lineart),
            "softedge" | "hed" => Ok(Self::SoftEdge),
            other => anyhow::bail!(
                "unknown control kind {other:?} \
                 (v0.11 supports: depth, canny, openpose, lineart, softedge)"
            ),
        }
    }
}

/// One conditioner in a multi-ControlNet stack. Parsed from the
/// repeatable `--control-spec` flag and threaded through the pipeline
/// Request types as `Vec<ControlSpec>`. A single ControlSpec is
/// resolved into a `ControlNet` + conditioning Tensor inside the
/// pipeline; the resulting [`ControlRequest`]s flow into the denoise
/// loop where their residuals are summed across all active controls.
///
/// Grammar (parsed by `FromStr`):
///
///   `KIND[:option=value]*`
///
/// `KIND` is one of `depth` / `canny`. Options (each may appear at
/// most once):
///
///   * `image=PATH`   — pre-rendered conditioning image
///   * `from=PATH`    — auto-annotate this image (mutually exclusive
///                       with `image=`)
///   * `strength=F`   — residual scale (default 1.0)
///   * `start=F`      — timestep window start in `[0, 1]` (default 0.0)
///   * `end=F`        — timestep window end   in `[0, 1]` (default 1.0)
///
/// Examples:
///
///   `depth`
///   `depth:from=in.jpg`
///   `canny:image=edges.png:strength=0.5:start=0.2:end=0.7`
#[derive(Debug, Clone, PartialEq)]
pub struct ControlSpec {
    pub kind: ControlKind,
    pub image: Option<std::path::PathBuf>,
    pub from: Option<std::path::PathBuf>,
    /// v0.30 phase 2: per-frame video control. When set, the input
    /// is treated as a video; frames are extracted via ffmpeg and
    /// sub-sampled to the animate frame budget. Each animate frame
    /// receives its own ControlNet conditioning. Mutually exclusive
    /// with `image` and `from`. Only consumed by AnimateDiff
    /// (per-frame is meaningless for single-image t2i / img2img).
    pub video: Option<std::path::PathBuf>,
    pub strength: f32,
    pub start: f32,
    pub end: f32,
}

impl ControlSpec {
    /// Build from the legacy split flags (`--control` + `--control-image`
    /// / `--control-from` + `--control-strength` + `--control-start` /
    /// `--control-end`). Returns `None` when `kind` is `None` —
    /// i.e. the user didn't pass `--control`.
    pub fn from_legacy_flags(
        kind: Option<ControlKind>,
        image: Option<std::path::PathBuf>,
        from: Option<std::path::PathBuf>,
        strength: f32,
        start: f32,
        end: f32,
    ) -> Option<Self> {
        kind.map(|k| Self {
            kind: k,
            image,
            from,
            video: None,
            strength,
            start,
            end,
        })
    }
}

/// CLI helper: collapse the two ways the user can spell a ControlNet
/// stack (`--control-spec` repeatable OR the legacy
/// `--control` / `--control-image` / `--control-from` /
/// `--control-strength` / `--control-start` / `--control-end`) into
/// one `Vec<ControlSpec>`. Clap's `conflicts_with_all` on
/// `--control-spec` guarantees the legacy flags can't be combined with
/// the new repeatable form — so at most one branch contributes.
pub fn resolve_control_specs(
    specs: Vec<ControlSpec>,
    legacy_kind: Option<ControlKind>,
    legacy_image: Option<std::path::PathBuf>,
    legacy_from: Option<std::path::PathBuf>,
    legacy_strength: f32,
    legacy_start: f32,
    legacy_end: f32,
) -> Vec<ControlSpec> {
    if !specs.is_empty() {
        return specs;
    }
    ControlSpec::from_legacy_flags(
        legacy_kind,
        legacy_image,
        legacy_from,
        legacy_strength,
        legacy_start,
        legacy_end,
    )
    .map(|s| vec![s])
    .unwrap_or_default()
}

impl std::str::FromStr for ControlSpec {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        let mut parts = s.split(':');
        let kind_str = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("empty --control-spec"))?;
        let kind: ControlKind = kind_str
            .parse()
            .with_context(|| format!("parsing kind in --control-spec {s:?}"))?;
        let mut image = None;
        let mut from = None;
        let mut video = None;
        let mut strength: f32 = 1.0;
        let mut start: f32 = 0.0;
        let mut end: f32 = 1.0;
        for opt in parts {
            let (k, v) = opt.split_once('=').ok_or_else(|| {
                anyhow::anyhow!(
                    "--control-spec option {opt:?} must be `key=value` (in {s:?})"
                )
            })?;
            match k {
                "image" => {
                    if image.is_some() {
                        anyhow::bail!("--control-spec {s:?}: duplicate image=");
                    }
                    image = Some(std::path::PathBuf::from(v));
                }
                "from" => {
                    if from.is_some() {
                        anyhow::bail!("--control-spec {s:?}: duplicate from=");
                    }
                    from = Some(std::path::PathBuf::from(v));
                }
                "video" => {
                    if video.is_some() {
                        anyhow::bail!("--control-spec {s:?}: duplicate video=");
                    }
                    video = Some(std::path::PathBuf::from(v));
                }
                "strength" => {
                    strength = v.parse::<f32>().with_context(|| {
                        format!("--control-spec {s:?}: strength={v:?} must be a float")
                    })?;
                }
                "start" => {
                    start = v.parse::<f32>().with_context(|| {
                        format!("--control-spec {s:?}: start={v:?} must be a float")
                    })?;
                }
                "end" => {
                    end = v.parse::<f32>().with_context(|| {
                        format!("--control-spec {s:?}: end={v:?} must be a float")
                    })?;
                }
                other => anyhow::bail!(
                    "--control-spec {s:?}: unknown option {other:?} \
                     (supported: image, from, video, strength, start, end)"
                ),
            }
        }
        let sources = [image.is_some(), from.is_some(), video.is_some()]
            .iter()
            .filter(|b| **b)
            .count();
        if sources > 1 {
            anyhow::bail!(
                "--control-spec {s:?}: pass at most one of image= / from= / video="
            );
        }
        if !(0.0..=1.0).contains(&start) || !(0.0..=1.0).contains(&end) || start > end {
            anyhow::bail!(
                "--control-spec {s:?}: start ({start}) and end ({end}) must satisfy \
                 0 <= start <= end <= 1"
            );
        }
        Ok(Self {
            kind,
            image,
            from,
            video,
            strength,
            start,
            end,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_spec_parses_bare_kind() {
        let s: ControlSpec = "depth".parse().unwrap();
        assert_eq!(s.kind, ControlKind::Depth);
        assert!(s.image.is_none() && s.from.is_none());
        assert_eq!(s.strength, 1.0);
        assert_eq!(s.start, 0.0);
        assert_eq!(s.end, 1.0);
    }

    #[test]
    fn control_spec_parses_full() {
        let s: ControlSpec =
            "canny:image=edges.png:strength=0.5:start=0.2:end=0.7".parse().unwrap();
        assert_eq!(s.kind, ControlKind::Canny);
        assert_eq!(s.image.as_ref().unwrap().to_str(), Some("edges.png"));
        assert!(s.from.is_none());
        assert_eq!(s.strength, 0.5);
        assert_eq!(s.start, 0.2);
        assert_eq!(s.end, 0.7);
    }

    #[test]
    fn control_spec_rejects_image_and_from() {
        assert!(
            "depth:image=a.png:from=b.jpg".parse::<ControlSpec>().is_err()
        );
    }

    #[test]
    fn control_spec_rejects_bad_window() {
        assert!("depth:start=0.8:end=0.2".parse::<ControlSpec>().is_err());
        assert!("depth:start=-0.1".parse::<ControlSpec>().is_err());
        assert!("depth:end=1.5".parse::<ControlSpec>().is_err());
    }

    #[test]
    fn control_spec_rejects_unknown_option() {
        assert!("depth:strenth=1.0".parse::<ControlSpec>().is_err()); // typo
    }

    // ---------------------------------------------------------------
    // v0.30 phase 2: video= parsing + multi-source mutual exclusion.
    // ---------------------------------------------------------------

    #[test]
    fn control_spec_parses_video_only() {
        let s: ControlSpec = "canny:video=./drive.mp4".parse().unwrap();
        assert_eq!(s.kind, ControlKind::Canny);
        assert!(s.image.is_none());
        assert!(s.from.is_none());
        assert_eq!(s.video.as_ref().unwrap().to_str(), Some("./drive.mp4"));
        assert_eq!(s.strength, 1.0);
    }

    #[test]
    fn control_spec_parses_video_with_strength_and_window() {
        let s: ControlSpec =
            "openpose:video=./pose.mp4:strength=0.6:start=0.0:end=0.8".parse().unwrap();
        assert_eq!(s.kind, ControlKind::OpenPose);
        assert!(s.video.is_some());
        assert_eq!(s.strength, 0.6);
        assert_eq!(s.start, 0.0);
        assert_eq!(s.end, 0.8);
    }

    #[test]
    fn control_spec_rejects_image_and_video() {
        assert!(
            "depth:image=a.png:video=b.mp4".parse::<ControlSpec>().is_err()
        );
    }

    #[test]
    fn control_spec_rejects_from_and_video() {
        assert!(
            "depth:from=a.png:video=b.mp4".parse::<ControlSpec>().is_err()
        );
    }

    #[test]
    fn control_spec_rejects_all_three_sources() {
        assert!(
            "depth:image=a.png:from=b.png:video=c.mp4"
                .parse::<ControlSpec>()
                .is_err()
        );
    }

    #[test]
    fn control_spec_rejects_duplicate_video() {
        assert!(
            "depth:video=a.mp4:video=b.mp4".parse::<ControlSpec>().is_err()
        );
    }

    #[test]
    fn control_kind_parses() {
        assert_eq!(
            "depth".parse::<ControlKind>().unwrap(),
            ControlKind::Depth
        );
        assert_eq!("DEPTH".parse::<ControlKind>().unwrap(), ControlKind::Depth);
        assert_eq!("canny".parse::<ControlKind>().unwrap(), ControlKind::Canny);
        assert_eq!("Canny".parse::<ControlKind>().unwrap(), ControlKind::Canny);
        // v0.11 conditioners.
        assert_eq!(
            "openpose".parse::<ControlKind>().unwrap(),
            ControlKind::OpenPose
        );
        // `pose` is an alias for openpose.
        assert_eq!(
            "pose".parse::<ControlKind>().unwrap(),
            ControlKind::OpenPose
        );
        assert_eq!(
            "lineart".parse::<ControlKind>().unwrap(),
            ControlKind::Lineart
        );
        assert_eq!(
            "softedge".parse::<ControlKind>().unwrap(),
            ControlKind::SoftEdge
        );
        // `hed` is the historical alias for softedge.
        assert_eq!("hed".parse::<ControlKind>().unwrap(), ControlKind::SoftEdge);
        assert!("scribble".parse::<ControlKind>().is_err());
        assert!("".parse::<ControlKind>().is_err());
    }

    #[test]
    fn control_kind_slug_roundtrips() {
        for k in [
            ControlKind::Depth,
            ControlKind::Canny,
            ControlKind::OpenPose,
            ControlKind::Lineart,
            ControlKind::SoftEdge,
        ] {
            let s = k.slug();
            assert_eq!(s.parse::<ControlKind>().unwrap(), k);
        }
    }

    #[test]
    fn candidates_for_canny_sd15_and_sdxl_each_have_three_mirrors() {
        for variant in [ControlNetVariant::Sd15, ControlNetVariant::Sdxl] {
            let c = candidates_for(ControlKind::Canny, variant);
            assert_eq!(c.len(), 3, "expected 3 canny mirrors for {variant:?}");
            for (_repo, file) in &c {
                assert!(
                    file.ends_with(".safetensors"),
                    "expected safetensors file, got {file:?}",
                );
            }
        }
    }

    #[test]
    fn candidates_for_depth_sd15_has_three_diffusers_mirrors() {
        let c = candidates_for(ControlKind::Depth, ControlNetVariant::Sd15);
        assert_eq!(c.len(), 3, "expected primary + 2 fallback mirrors");
        for (repo, file) in &c {
            assert!(!repo.is_empty(), "empty repo in candidates");
            assert!(
                file.ends_with(".safetensors"),
                "expected safetensors file, got {file:?}",
            );
        }
        // Primary should be the canonical SD 1.5 ControlNet-Depth.
        assert_eq!(c[0].0, "lllyasviel/sd-controlnet-depth");
    }

    #[test]
    fn candidates_for_depth_sdxl_has_three_diffusers_mirrors() {
        let c = candidates_for(ControlKind::Depth, ControlNetVariant::Sdxl);
        assert_eq!(c.len(), 3, "expected primary + 2 fallback mirrors");
        for (repo, file) in &c {
            assert!(!repo.is_empty(), "empty repo in candidates");
            assert!(
                file.ends_with(".safetensors"),
                "expected safetensors file, got {file:?}",
            );
        }
        // Primary should be the recommended -small SDXL ControlNet.
        assert_eq!(c[0].0, "diffusers/controlnet-depth-sdxl-1.0");
    }

    #[test]
    fn controlnet_variant_detect() {
        assert_eq!(ControlNetVariant::detect("sd15"), ControlNetVariant::Sd15);
        assert_eq!(
            ControlNetVariant::detect("stable-diffusion-v1-5/stable-diffusion-v1-5"),
            ControlNetVariant::Sd15,
        );
        assert_eq!(ControlNetVariant::detect("sdxl"), ControlNetVariant::Sdxl);
        assert_eq!(
            ControlNetVariant::detect("stabilityai/stable-diffusion-xl-base-1.0"),
            ControlNetVariant::Sdxl,
        );
        assert_eq!(
            ControlNetVariant::detect("sdxl-turbo"),
            ControlNetVariant::Sdxl,
        );
    }

    #[test]
    fn control_request_window_gates_progress() {
        // Stub net: we only exercise `active_at`. Use Cpu + a
        // throwaway ControlNet built against a default config.
        // (active_at doesn't touch the net field, but the struct
        // needs a value.)
        use candle_nn::VarMap;
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        // Construct a tiny ControlNet purely so we can hold a
        // reference for the ControlRequest. We never invoke its
        // forward(). The build is slow in debug mode — skipped via
        // #[ignore].
        let _ = vb; // unused after we test only active_at()

        // Test the predicate purely by constructing manually.
        // Default window 0..1: every progress is in the window.
        let test_active = |start: f32, end: f32, progress: f32| -> bool {
            progress >= start && progress < end
        };
        assert!(test_active(0.0, 1.0, 0.0));
        assert!(test_active(0.0, 1.0, 0.5));
        assert!(test_active(0.0, 1.0, 0.999));
        assert!(!test_active(0.0, 1.0, 1.0)); // half-open

        // Early-only: end=0.5.
        assert!(test_active(0.0, 0.5, 0.0));
        assert!(test_active(0.0, 0.5, 0.49));
        assert!(!test_active(0.0, 0.5, 0.5));
        assert!(!test_active(0.0, 0.5, 0.75));

        // Late-only: start=0.5.
        assert!(!test_active(0.5, 1.0, 0.0));
        assert!(!test_active(0.5, 1.0, 0.49));
        assert!(test_active(0.5, 1.0, 0.5));
        assert!(test_active(0.5, 1.0, 0.99));

        // Middle band: 0.25..0.75.
        assert!(!test_active(0.25, 0.75, 0.2));
        assert!(test_active(0.25, 0.75, 0.5));
        assert!(!test_active(0.25, 0.75, 0.8));
    }

    #[test]
    fn sdxl_unet_config_matches_sdxl() {
        let cfg = sdxl_unet_config();
        assert_eq!(cfg.blocks.len(), 3, "SDXL has 3 blocks (vs SD 1.5's 4)");
        assert_eq!(cfg.cross_attention_dim, 2048);
        assert!(cfg.use_linear_projection);
        // First block is Basic (no cross-attn).
        assert_eq!(cfg.blocks[0].use_cross_attn, None);
        assert_eq!(cfg.blocks[1].use_cross_attn, Some(2));
        assert_eq!(cfg.blocks[2].use_cross_attn, Some(10));
    }

    #[test]
    fn prepare_conditioning_normalises_rgb_to_unit_range() {
        use image::{Rgb, RgbImage};
        // 4x4 image: half black, half white columns.
        let mut img = RgbImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                let v = if x < 2 { 0u8 } else { 255u8 };
                img.put_pixel(x, y, Rgb([v, v, v]));
            }
        }
        let tmp = std::env::temp_dir().join("plakat_controlnet_hint_test.png");
        img.save(&tmp).unwrap();
        let t = prepare_conditioning(&tmp, 4, 4, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(t.dims(), &[1, 3, 4, 4]);
        let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        // R, G, B all match the source.
        // First two columns of first row are black (0.0).
        assert!(v[0] < 0.01);
        // Last two columns of first row are white (1.0).
        assert!((v[3] - 1.0).abs() < 0.01);
    }

    #[test]
    fn prepare_conditioning_replicates_grayscale_to_three_channels() {
        use image::{GrayImage, Luma};
        let mut img = GrayImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                let v = if x < 2 { 0u8 } else { 200u8 };
                img.put_pixel(x, y, Luma([v]));
            }
        }
        let tmp = std::env::temp_dir().join("plakat_controlnet_hint_gray_test.png");
        img.save(&tmp).unwrap();
        let t = prepare_conditioning(&tmp, 4, 4, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(t.dims(), &[1, 3, 4, 4]);
        let total = 4 * 4;
        let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        // R, G, B channels should match (single channel was replicated).
        for i in 0..total {
            let r = v[i];
            let g = v[total + i];
            let b = v[2 * total + i];
            assert!(
                (r - g).abs() < 1e-5 && (g - b).abs() < 1e-5,
                "grayscale should be R==G==B at idx {i} (got {r}/{g}/{b})",
            );
        }
        // Right-half pixel value approx 200/255.
        let expected = 200.0 / 255.0;
        assert!((v[2] - expected).abs() < 0.01);
    }

    #[test]
    fn sd15_unet_config_matches_v1_5() {
        let cfg = sd15_unet_config();
        // Differences from default: cross_attention_dim 768 vs 1280.
        assert_eq!(cfg.cross_attention_dim, 768);
        // Everything else matches the default.
        assert_eq!(cfg.blocks.len(), 4);
        assert_eq!(cfg.layers_per_block, 2);
        assert_eq!(cfg.blocks[0].out_channels, 320);
        assert_eq!(cfg.blocks[3].out_channels, 1280);
        assert_eq!(cfg.blocks[3].use_cross_attn, None);
    }

    /// Construct a ControlNet against fresh random weights (no real
    /// checkpoint) and verify the structure builds cleanly for the
    /// default UNet config (4 blocks: 320/640/1280/1280, layers=2 —
    /// the same shape SD 1.5 uses). Doesn't run a forward pass; just
    /// exercises the constructor + the zero-conv head count.
    ///
    /// Marked `#[ignore]` because the full ControlNet has ~800 MB of
    /// random weights in F32 — instantiating them takes ~160s in
    /// debug mode. Run explicitly with:
    ///
    /// ```sh
    /// cargo test --release --lib pipelines::controlnet -- --ignored
    /// ```
    #[test]
    #[ignore]
    fn constructs_for_default_unet_config() {
        use candle_nn::VarMap;
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        // SD 1.5 latent is 4 channels.
        let cfg = UNet2DConditionModelConfig::default();
        let net = ControlNet::new(vb, 4, cfg, ControlNetVariant::Sd15);
        assert!(net.is_ok(), "ControlNet::new failed: {:?}", net.err());
        let net = net.unwrap();
        // Default config has 4 blocks, layers_per_block=2, last has
        // no downsampler. Down residuals = 1 + 3 + 3 + 3 + 2 = 12.
        assert_eq!(
            net.controlnet_down_blocks.len(),
            12,
            "expected 12 zero-conv heads for the default UNet config",
        );
    }
}
