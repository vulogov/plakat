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
    /// Wait — actually it's 1 + 3 + 3 + 3 + 2 = 12 total down residuals.
    controlnet_down_blocks: Vec<Conv2d>,

    /// Final zero-conv on the mid-block output.
    controlnet_mid_block: Conv2d,

    /// Variant config kept for diagnostics.
    config: UNet2DConditionModelConfig,
}

impl ControlNet {
    /// Construct from a `VarBuilder` rooted at the ControlNet weights
    /// (typically `vb` over the safetensors file with no further prefix).
    /// Pass the same `UNet2DConditionModelConfig` the paired UNet uses.
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        config: UNet2DConditionModelConfig,
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
    ) -> Result<(Vec<Tensor>, Tensor)> {
        let (bsize, _c, _h, _w) = latents.dims4()?;
        let device = latents.device();
        let dtype = latents.dtype();

        // 1. time embedding (same shape as UNet's).
        let emb = (Tensor::ones(bsize, dtype, device)? * timestep)?;
        let emb = self.time_proj.forward(&emb)?;
        let emb = self.time_embedding.forward(&emb)?;

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
        // cross_attn_dim) differs.
        Self::new(vb, 4, variant.unet_config())
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
            // Primary: diffusers' official SDXL ControlNet-Depth (small,
            // 600M parameters — the recommended SDXL ControlNet).
            (
                "diffusers/controlnet-depth-sdxl-1.0-small",
                "diffusion_pytorch_model.safetensors",
            ),
            // Fallback 1: same repo, fp16 variant.
            (
                "diffusers/controlnet-depth-sdxl-1.0-small",
                "diffusion_pytorch_model.fp16.safetensors",
            ),
            // Fallback 2: the full-size SDXL ControlNet-Depth (~5 GB).
            // Higher quality, much heavier; only used if -small mirrors
            // are unreachable.
            (
                "diffusers/controlnet-depth-sdxl-1.0",
                "diffusion_pytorch_model.fp16.safetensors",
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
            // Primary: diffusers' SDXL ControlNet-Canny (small).
            (
                "diffusers/controlnet-canny-sdxl-1.0-small",
                "diffusion_pytorch_model.safetensors",
            ),
            // Fallback 1: same repo, fp16 variant.
            (
                "diffusers/controlnet-canny-sdxl-1.0-small",
                "diffusion_pytorch_model.fp16.safetensors",
            ),
            // Fallback 2: full-size SDXL ControlNet-Canny.
            (
                "diffusers/controlnet-canny-sdxl-1.0",
                "diffusion_pytorch_model.fp16.safetensors",
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
pub struct ControlRequest<'a> {
    pub net: &'a ControlNet,
    pub conditioning: Tensor,
    pub strength: f32,
}

/// What kind of conditioning signal the user requested. v0.10 ships
/// `Depth` (Depth-Anything-V2 annotator) and `Canny` (imageproc
/// canny annotator). Other conditioners (scribble, MLSD, pose, ...)
/// land in later releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControlKind {
    Depth,
    Canny,
}

impl ControlKind {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Depth => "depth",
            Self::Canny => "canny",
        }
    }
}

impl std::str::FromStr for ControlKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "depth" => Ok(Self::Depth),
            "canny" => Ok(Self::Canny),
            other => anyhow::bail!(
                "unknown control kind {other:?} (v0.10 supports: depth, canny)"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_kind_parses() {
        assert_eq!(
            "depth".parse::<ControlKind>().unwrap(),
            ControlKind::Depth
        );
        assert_eq!("DEPTH".parse::<ControlKind>().unwrap(), ControlKind::Depth);
        assert_eq!("canny".parse::<ControlKind>().unwrap(), ControlKind::Canny);
        assert_eq!("Canny".parse::<ControlKind>().unwrap(), ControlKind::Canny);
        assert!("scribble".parse::<ControlKind>().is_err());
        assert!("".parse::<ControlKind>().is_err());
    }

    #[test]
    fn control_kind_slug_roundtrips() {
        for k in [ControlKind::Depth, ControlKind::Canny] {
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
        assert_eq!(c[0].0, "diffusers/controlnet-depth-sdxl-1.0-small");
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
        let net = ControlNet::new(vb, 4, cfg);
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
