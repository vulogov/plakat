//! PixArt-Σ DiT-XL/2 transformer.
//!
//! v0.35 phase 1: full architectural implementation with weights
//! loadable from the canonical diffusers
//! `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` checkpoint
//! (`transformer/diffusion_pytorch_model.safetensors`).
//!
//! Tensor-name layout (verified against `diffusers >= 0.27`'s
//! `PixArtTransformer2DModel`):
//!
//! ```text
//! pos_embed.proj.{weight,bias}                              # Conv2d patch embed
//! adaln_single.linear.{weight,bias}                         # global t_block
//! adaln_single.emb.timestep_embedder.linear_1.{weight,bias} # timestep MLP layer 1
//! adaln_single.emb.timestep_embedder.linear_2.{weight,bias}
//! adaln_single.emb.resolution_embedder.linear_1.{weight,bias}    # Σ resolution conditioning
//! adaln_single.emb.resolution_embedder.linear_2.{weight,bias}
//! adaln_single.emb.aspect_ratio_embedder.linear_1.{weight,bias}  # Σ aspect ratio conditioning
//! adaln_single.emb.aspect_ratio_embedder.linear_2.{weight,bias}
//! caption_projection.linear_1.{weight,bias}                 # T5 → hidden
//! caption_projection.linear_2.{weight,bias}
//! transformer_blocks.{i}.scale_shift_table                  # per-block adaLN bias (6, hidden)
//! transformer_blocks.{i}.attn1.to_q.{weight,bias}           # self-attention QKV
//! transformer_blocks.{i}.attn1.to_k.{weight,bias}
//! transformer_blocks.{i}.attn1.to_v.{weight,bias}
//! transformer_blocks.{i}.attn1.to_out.0.{weight,bias}
//! transformer_blocks.{i}.attn2.to_q.{weight,bias}           # cross-attention QKV
//! transformer_blocks.{i}.attn2.to_k.{weight,bias}
//! transformer_blocks.{i}.attn2.to_v.{weight,bias}
//! transformer_blocks.{i}.attn2.to_out.0.{weight,bias}
//! transformer_blocks.{i}.ff.net.0.proj.{weight,bias}        # MLP fc1 (GELU-tanh wrapped)
//! transformer_blocks.{i}.ff.net.2.{weight,bias}             # MLP fc2
//! scale_shift_table                                         # final (2, hidden) on top-level
//! proj_out.{weight,bias}                                    # final output projection
//! ```
//!
//! v0.35 phase 1 scope: tensor structure + per-block + full forward-
//! pass shape verification with random weights. Numerical
//! verification against a real checkpoint defers to phase 2 smoke
//! when inference end-to-end runs.

use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Module, Tensor, Var};
use candle_nn::{self as nn, VarBuilder};
// v1.10.0: every attention/FF Linear becomes a `LoraLinear` so the
// DiT can host a trainable LoRA adapter on its attention projections
// (`plakat style train` / DreamBooth). Mirrors `mmdit_inner.rs`
// (SD3) exactly. Inference path stays byte-identical to `nn::Linear`
// (empty runtime stack + no train adapter).
use crate::pipelines::lora_linear::{LoraLinear, LoraRegistry, LoraRegistryEntry};
use std::sync::{Arc, RwLock};

/// Wrap a candle Linear, register the slots/train handles in
/// `<vb.prefix()>.weight` of the shared LoRA registry, return the
/// `LoraLinear` ready to plug into a struct field. Same pattern as the
/// helper in `mmdit_inner.rs`.
fn wrap_linear(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    registry: &Arc<RwLock<LoraRegistry>>,
) -> Result<LoraLinear> {
    let base = nn::linear(in_dim, out_dim, vb.clone())
        .map_err(|e| anyhow!("PixArt wrap_linear at {}: {e}", vb.prefix()))?;
    let ll = LoraLinear::from_linear(base)
        .map_err(|e| anyhow!("PixArt wrap_linear from_linear at {}: {e}", vb.prefix()))?;
    let key = format!("{}.weight", vb.prefix());
    registry
        .write()
        .map_err(|_| anyhow!("PixArt LoRA registry poisoned during construction"))?
        .insert(
            key,
            LoraRegistryEntry {
                handle: ll.slots_handle(),
                out_dim,
                in_dim,
                train: ll.train_handle(),
            },
        );
    Ok(ll)
}

/// v0.36 phase 3: KV-compression in self-attention. PixArt-Σ's
/// Σ-specific addition — downsamples the image-token K/V sequence
/// via per-block depthwise Conv2d before computing K/V. Lets
/// self-attention scale to long token sequences (2K² output has a
/// 128×128 = 16384-token grid; uncompressed self-attention is
/// O(T²) and prohibitive at that scale). The trick: Q stays full;
/// K/V are computed from a `scale_factor`× downsampled spatial
/// representation (depthwise Conv2d, kernel=stride=scale_factor),
/// shrinking the attention matrix to (T × T/scale²).
///
/// Applies only to self-attention (`attn1`). Cross-attention to T5
/// (`attn2`) stays uncompressed — T5's 300-token sequence is
/// already short.
#[derive(Debug, Clone, Copy)]
pub struct KvCompressionConfig {
    /// Spatial downsample factor in each axis. 2 produces a 4×
    /// sequence reduction (most common — used by Σ-2K-MS).
    pub scale_factor: usize,
}

/// Architecture config for the PixArt-Σ DiT.
#[derive(Debug, Clone)]
pub struct Config {
    pub in_channels: usize,
    pub out_channels: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub mlp_ratio: usize,
    pub patch_size: usize,
    pub sample_size: usize,
    pub caption_channels: usize,
    pub max_caption_tokens: usize,
    /// v0.36 phase 3: KV-compression in self-attention. `Some` →
    /// every transformer block's `attn1` uses a depthwise Conv2d
    /// to downsample K/V before attention. `None` → 1024-MS /
    /// 512-MS behaviour (full self-attention).
    pub kv_compression: Option<KvCompressionConfig>,
}

impl Config {
    /// PixArt-Σ-XL-2-1024-MS — v0.35 phase 2's first ship target.
    /// `sample_size: 64` covers a 64×64 token grid at patch 2
    /// (latent 128×128, output 1024²). No KV-compression.
    pub fn sigma_xl_1024() -> Self {
        Self {
            in_channels: 4,
            out_channels: 8,
            hidden_size: 1152,
            num_layers: 28,
            num_heads: 16,
            mlp_ratio: 4,
            patch_size: 2,
            sample_size: 64,
            caption_channels: 4096,
            max_caption_tokens: 300,
            kv_compression: None,
        }
    }

    /// v0.36 phase 2: PixArt-Σ-XL-2-512-MS. Same DiT-XL/2
    /// architecture as 1024-MS (identical num_layers / hidden_size
    /// / num_heads / patch_size / out_channels) — only the upstream
    /// training distribution + `sample_size` differ. `sample_size:
    /// 32` covers a 32×32 token grid (latent 64×64, output 512²).
    ///
    /// In practice `sample_size` is informational only — plakat
    /// computes the 2D sincos positional embedding from the actual
    /// (grid_h, grid_w) at forward time, so it's robust to any
    /// width/height the user requests. The variant constructor
    /// exists to document the upstream config and to make the
    /// `Pipeline::load` repo-detection branch self-explanatory.
    pub fn sigma_xl_512() -> Self {
        Self {
            sample_size: 32,
            ..Self::sigma_xl_1024()
        }
    }

    /// v0.36 phase 3: PixArt-Σ-XL-2-2K-MS. Same DiT-XL/2 backbone
    /// + 2× KV-compression in self-attention so the 128×128 = 16384
    /// token grid at 2048² is computationally tractable. The
    /// compression applies to ALL 28 transformer blocks' self-attn
    /// (PixArt-Σ paper §3.2 — "We apply KV compression on all 28
    /// transformer blocks"). `sample_size: 128`.
    pub fn sigma_xl_2k() -> Self {
        Self {
            sample_size: 128,
            kv_compression: Some(KvCompressionConfig { scale_factor: 2 }),
            ..Self::sigma_xl_1024()
        }
    }

    /// v0.36 phase 2 / 3: pick the right config from a resolved
    /// repo path. Falls back to the 1024-MS config when the repo
    /// isn't recognised.
    pub fn for_pixart_repo(repo: &str) -> Self {
        let r = repo.to_lowercase();
        if r.contains("sigma-xl-2-2k-ms") {
            Self::sigma_xl_2k()
        } else if r.contains("sigma-xl-2-512-ms") {
            Self::sigma_xl_512()
        } else {
            Self::sigma_xl_1024()
        }
    }
}

// ---------------------------------------------------------------------
// Embedding helpers.
// ---------------------------------------------------------------------

/// Conv2d patch embedding. Latent (B, C, H, W) → token (B, T, hidden)
/// where T = (H / patch) * (W / patch).
pub struct PatchEmbed {
    proj: nn::Conv2d,
    patch_size: usize,
}

impl PatchEmbed {
    pub fn new(
        in_channels: usize,
        hidden_size: usize,
        patch_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = nn::Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let proj = nn::conv2d(in_channels, hidden_size, patch_size, cfg, vb.pp("proj"))
            .map_err(|e| anyhow!("PatchEmbed proj: {e}"))?;
        Ok(Self { proj, patch_size })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.proj.forward(x)?;
        let (b, c, h, w) = x.dims4()?;
        Ok(x.reshape((b, c, h * w))?.transpose(1, 2)?)
    }

    pub fn grid_dims(&self, latent_h: usize, latent_w: usize) -> (usize, usize) {
        (latent_h / self.patch_size, latent_w / self.patch_size)
    }
}

/// 2D sinusoidal positional embedding (port of the upstream
/// `get_2d_sincos_pos_embed_from_grid`).
pub fn build_2d_sincos_pos_embed(
    hidden_size: usize,
    grid_h: usize,
    grid_w: usize,
    // PixArt scales the grid coordinates by `base_size / (grid * interp)`
    // (diffusers `get_2d_sincos_pos_embed`): at native resolution
    // grid==base_size so the factor is `1/interpolation_scale` (= 1/2 for
    // 1024-MS). Omitting it makes every frequency `interp`× too high →
    // wrong positional embedding → the DiT can't denoise. For off-native
    // (multi-scale) grids the base/grid ratio interpolates.
    base_size: usize,
    interpolation_scale: f32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let scale_h = base_size as f32 / (grid_h as f32 * interpolation_scale);
    let scale_w = base_size as f32 / (grid_w as f32 * interpolation_scale);
    let grid_h_idx: Vec<f32> = (0..grid_h).map(|i| i as f32 * scale_h).collect();
    let grid_w_idx: Vec<f32> = (0..grid_w).map(|i| i as f32 * scale_w).collect();
    let half = hidden_size / 2;
    let quarter = half / 2;
    let omega: Vec<f32> = (0..quarter)
        .map(|i| 1.0_f32 / 10000_f32.powf((i as f32) / (quarter as f32)))
        .collect();
    let make_axis = |g: &[f32]| -> Result<Tensor> {
        let mut data: Vec<f32> = Vec::with_capacity(g.len() * 2 * quarter);
        for &p in g {
            for &w in &omega {
                data.push((p * w).sin());
            }
            for &w in &omega {
                data.push((p * w).cos());
            }
        }
        Ok(Tensor::from_vec(data, (g.len(), 2 * quarter), device)?)
    };
    let emb_h = make_axis(&grid_h_idx)?;
    let emb_w = make_axis(&grid_w_idx)?;
    let h_repeat = emb_h.unsqueeze(1)?.expand((grid_h, grid_w, half))?;
    let w_repeat = emb_w.unsqueeze(0)?.expand((grid_h, grid_w, half))?;
    // diffusers concatenates [emb(W), emb(H)] per position (from
    // meshgrid(w, h)), NOT [emb(H), emb(W)] — getting this backwards
    // transposes the positional grid.
    let pe = Tensor::cat(&[w_repeat, h_repeat], 2)?
        .reshape((grid_h * grid_w, hidden_size))?
        .to_dtype(dtype)?
        .unsqueeze(0)?;
    Ok(pe)
}

/// Timestep → sinusoidal embedding → 2-layer MLP.
pub struct TimestepEmbedder {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let frequency_embedding_size = 256;
        let linear_1 = nn::linear(frequency_embedding_size, hidden_size, vb.pp("linear_1"))
            .map_err(|e| anyhow!("TimestepEmbedder linear_1: {e}"))?;
        let linear_2 = nn::linear(hidden_size, hidden_size, vb.pp("linear_2"))
            .map_err(|e| anyhow!("TimestepEmbedder linear_2: {e}"))?;
        Ok(Self {
            linear_1,
            linear_2,
            frequency_embedding_size,
        })
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let device = t.device();
        let half = self.frequency_embedding_size / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-(10000_f32.ln()) * (i as f32) / (half as f32)).exp())
            .collect();
        let freqs = Tensor::from_vec(freqs, half, device)?;
        let t_f32 = t.to_dtype(DType::F32)?;
        let args = t_f32.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?;
        let emb = emb.to_dtype(self.linear_1.weight().dtype())?;
        let h = self.linear_1.forward(&emb)?;
        let h = h.silu()?;
        Ok(self.linear_2.forward(&h)?)
    }
}

/// Same shape as `TimestepEmbedder` — used for Σ-specific resolution
/// + aspect-ratio conditioning.
pub struct SizeEmbedder {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
    frequency_embedding_size: usize,
}

impl SizeEmbedder {
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let frequency_embedding_size = 256;
        let linear_1 = nn::linear(frequency_embedding_size, hidden_size, vb.pp("linear_1"))
            .map_err(|e| anyhow!("SizeEmbedder linear_1: {e}"))?;
        let linear_2 = nn::linear(hidden_size, hidden_size, vb.pp("linear_2"))
            .map_err(|e| anyhow!("SizeEmbedder linear_2: {e}"))?;
        Ok(Self {
            linear_1,
            linear_2,
            frequency_embedding_size,
        })
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let device = t.device();
        let half = self.frequency_embedding_size / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-(10000_f32.ln()) * (i as f32) / (half as f32)).exp())
            .collect();
        let freqs = Tensor::from_vec(freqs, half, device)?;
        let t_f32 = t.to_dtype(DType::F32)?;
        let args = t_f32.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?;
        let emb = emb.to_dtype(self.linear_1.weight().dtype())?;
        let h = self.linear_1.forward(&emb)?;
        let h = h.silu()?;
        Ok(self.linear_2.forward(&h)?)
    }
}

/// The Σ-additional embedding head: timestep + resolution + aspect.
pub struct AdaLnSingleEmb {
    timestep_embedder: TimestepEmbedder,
    // PixArt-Σ checkpoints carry NO resolution/aspect-ratio micro-
    // conditioning (`use_additional_conditions=False`); only PixArt-α
    // 1024-MS ships these embedders. Auto-detect from the checkpoint so a
    // single code path loads both families.
    resolution_embedder: Option<SizeEmbedder>,
    aspect_ratio_embedder: Option<SizeEmbedder>,
}

impl AdaLnSingleEmb {
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let has_size_cond = vb.contains_tensor("resolution_embedder.linear_1.weight");
        let resolution_embedder = if has_size_cond {
            Some(SizeEmbedder::new(hidden_size, vb.pp("resolution_embedder"))?)
        } else {
            None
        };
        let aspect_ratio_embedder = if has_size_cond {
            Some(SizeEmbedder::new(hidden_size, vb.pp("aspect_ratio_embedder"))?)
        } else {
            None
        };
        Ok(Self {
            timestep_embedder: TimestepEmbedder::new(hidden_size, vb.pp("timestep_embedder"))?,
            resolution_embedder,
            aspect_ratio_embedder,
        })
    }

    pub fn forward(
        &self,
        timestep: &Tensor,
        resolution: &Tensor,
        aspect_ratio: &Tensor,
    ) -> Result<Tensor> {
        let t_emb = self.timestep_embedder.forward(timestep)?;
        // Σ: timestep only.
        let (Some(res_e), Some(asp_e)) =
            (&self.resolution_embedder, &self.aspect_ratio_embedder)
        else {
            return Ok(t_emb);
        };
        let res_flat = resolution.reshape(((),))?;
        let asp_flat = aspect_ratio.reshape(((),))?;
        let res_emb = res_e.forward(&res_flat)?;
        let asp_emb = asp_e.forward(&asp_flat)?;
        let b = timestep.dim(0)?;
        let hidden = t_emb.dim(1)?;
        // (B*2, hidden) → (B, 2, hidden) → sum over the pair → (B, hidden).
        let res_emb = res_emb.reshape((b, 2, hidden))?.sum(1)?;
        let asp_emb = asp_emb.reshape((b, 2, hidden))?.sum(1)?;
        Ok(t_emb.add(&res_emb)?.add(&asp_emb)?)
    }
}

/// Top-level adaLN-single: `Sigma_emb → SiLU → Linear → (6 * hidden)`
/// global modulation that every block adds its `scale_shift_table`
/// to.
pub struct AdaLnSingle {
    emb: AdaLnSingleEmb,
    linear: nn::Linear,
}

impl AdaLnSingle {
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let emb = AdaLnSingleEmb::new(hidden_size, vb.pp("emb"))?;
        let linear = nn::linear(hidden_size, 6 * hidden_size, vb.pp("linear"))
            .map_err(|e| anyhow!("AdaLnSingle linear: {e}"))?;
        Ok(Self { emb, linear })
    }

    /// Returns `(t_block, embedded)`.
    /// - `t_block`: `(B, 6 * hidden)`.
    /// - `embedded`: `(B, hidden)` — Σ-conditioning carrier; phase 2
    ///    will feed this into cross-attention if needed (PixArt-Σ
    ///    impls use it for the additional T5 caption gating).
    pub fn forward(
        &self,
        timestep: &Tensor,
        resolution: &Tensor,
        aspect_ratio: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let embedded = self.emb.forward(timestep, resolution, aspect_ratio)?;
        let t_block = self.linear.forward(&embedded.silu()?)?;
        Ok((t_block, embedded))
    }
}

/// Caption projection: T5 hidden_dim (4096) → DiT hidden_size (1152).
pub struct CaptionProjection {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl CaptionProjection {
    pub fn new(t5_dim: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear_1 = nn::linear(t5_dim, hidden_size, vb.pp("linear_1"))
            .map_err(|e| anyhow!("CaptionProjection linear_1: {e}"))?;
        let linear_2 = nn::linear(hidden_size, hidden_size, vb.pp("linear_2"))
            .map_err(|e| anyhow!("CaptionProjection linear_2: {e}"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear_1.forward(x)?;
        let h = h.gelu_erf()?;
        Ok(self.linear_2.forward(&h)?)
    }
}

// ---------------------------------------------------------------------
// Attention + FFN.
// ---------------------------------------------------------------------

/// Multi-head attention with separate `to_q`, `to_k`, `to_v`
/// projections (matches diffusers `Attention` for PixArt).
///
/// v0.36 phase 3: optional depthwise Conv2d KV-compression. When
/// enabled (Σ-2K-MS), the KV-input tensor is reshaped to a 2D
/// spatial layout, downsampled via the Conv2d (kernel = stride =
/// `scale_factor`, `groups = hidden_size` → depthwise), and the K
/// + V projections operate on the downsampled sequence. Q stays
/// computed from the full input. This is the Σ paper's mechanism
/// for scaling self-attention to long token sequences.
pub struct Attention {
    to_q: LoraLinear,
    to_k: LoraLinear,
    to_v: LoraLinear,
    to_out: LoraLinear,
    num_heads: usize,
    head_dim: usize,
    /// v0.36 phase 3: depthwise Conv2d that downsamples the K/V
    /// image-token sequence in self-attention. Stored alongside
    /// the scale factor so the forward can reshape back to a
    /// flat token sequence after compression.
    kv_compress: Option<KvCompress>,
}

/// Σ-2K-MS KV-compression layer. Depthwise Conv2d (groups =
/// hidden_size, kernel = stride = scale_factor). Tensor keys:
/// `<attn-prefix>.kv_proj_conv2d.{weight,bias}` (diffusers
/// convention for the PixArt-Σ kv_proj Conv2d).
pub struct KvCompress {
    conv: nn::Conv2d,
    scale_factor: usize,
}

impl Attention {
    pub fn new(
        query_dim: usize,
        kv_dim: usize,
        num_heads: usize,
        vb: VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        Self::new_with_compression(query_dim, kv_dim, num_heads, vb, None, registry)
    }

    /// v0.36 phase 3: variant constructor that registers an optional
    /// KV-compression Conv2d (registered only when
    /// `kv_compression.is_some()`). `query_dim` is reused for the
    /// Conv2d channel count because Σ's compression preserves the
    /// channel dim (depthwise) — input shape `(B, query_dim,
    /// grid_h, grid_w)` → output `(B, query_dim, grid_h/scale,
    /// grid_w/scale)`.
    pub fn new_with_compression(
        query_dim: usize,
        kv_dim: usize,
        num_heads: usize,
        vb: VarBuilder,
        kv_compression: Option<KvCompressionConfig>,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let head_dim = query_dim / num_heads;
        let to_q = wrap_linear(query_dim, query_dim, vb.pp("to_q"), registry)?;
        let to_k = wrap_linear(kv_dim, query_dim, vb.pp("to_k"), registry)?;
        let to_v = wrap_linear(kv_dim, query_dim, vb.pp("to_v"), registry)?;
        let to_out = wrap_linear(query_dim, query_dim, vb.pp("to_out").pp("0"), registry)?;
        // Σ-only: depthwise Conv2d for KV downsampling. Skipped
        // when kv_compression is None (1024-MS / 512-MS path).
        let kv_compress = match kv_compression {
            None => None,
            Some(cfg) => {
                let conv_cfg = nn::Conv2dConfig {
                    stride: cfg.scale_factor,
                    groups: query_dim, // depthwise
                    ..Default::default()
                };
                // Depthwise Conv2d: same in/out channels, kernel ==
                // stride == scale_factor. The diffusers PixArt-Σ
                // convention names this `kv_proj_conv2d`.
                let conv = nn::conv2d(
                    query_dim,
                    query_dim,
                    cfg.scale_factor,
                    conv_cfg,
                    vb.pp("kv_proj_conv2d"),
                )
                .map_err(|e| anyhow!("Attention kv_proj_conv2d: {e}"))?;
                Some(KvCompress {
                    conv,
                    scale_factor: cfg.scale_factor,
                })
            }
        };
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            num_heads,
            head_dim,
            kv_compress,
        })
    }

    /// Cross-attention path: `kv` is taken as-is (the T5 sequence
    /// in PixArt's case). No grid dims required.
    pub fn forward(&self, x: &Tensor, kv: &Tensor) -> Result<Tensor> {
        self.forward_inner(x, kv, None)
    }

    /// v0.36 phase 3: self-attention path with optional KV
    /// compression. `grid_dims` is required when `self.kv_compress`
    /// is `Some` — the Conv2d reshapes the flat token sequence
    /// back to 2D before downsampling. Pass the source `(grid_h,
    /// grid_w)` of the image-token sequence.
    pub fn forward_self_attn(
        &self,
        x: &Tensor,
        grid_dims: Option<(usize, usize)>,
    ) -> Result<Tensor> {
        self.forward_inner(x, x, grid_dims)
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        kv_in: &Tensor,
        grid_dims: Option<(usize, usize)>,
    ) -> Result<Tensor> {
        let (b, lq, hidden) = x.dims3()?;

        // Optional KV downsample. When kv_compress is Some, we
        // reshape kv_in from (B, lkv, hidden) to (B, hidden,
        // grid_h, grid_w), apply the depthwise Conv2d, and reshape
        // the result back to a flat (B, lkv', hidden) sequence.
        let kv_for_proj: Tensor = if let Some(kvc) = &self.kv_compress {
            let (gh, gw) = grid_dims.ok_or_else(|| {
                anyhow!(
                    "Attention with kv_compress requires grid_dims; \
                     caller must pass forward_self_attn"
                )
            })?;
            let (_b, lkv, _h) = kv_in.dims3()?;
            anyhow::ensure!(
                lkv == gh * gw,
                "kv sequence length {lkv} must equal grid_h*grid_w ({gh}*{gw}={})",
                gh * gw
            );
            // (B, T, H) → (B, H, gh, gw)
            let spatial = kv_in
                .reshape((b, gh, gw, hidden))?
                .permute((0, 3, 1, 2))?;
            let down = kvc.conv.forward(&spatial)?;
            let (_b2, _h2, gh2, gw2) = down.dims4()?;
            // (B, H, gh', gw') → (B, gh'*gw', H)
            let flat = down
                .permute((0, 2, 3, 1))?
                .reshape((b, gh2 * gw2, hidden))?;
            let _ = kvc.scale_factor; // documented; not needed beyond conv stride
            flat
        } else {
            kv_in.clone()
        };

        let lkv = kv_for_proj.dim(1)?;
        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(&kv_for_proj)?;
        let v = self.to_v.forward(&kv_for_proj)?;
        let q = q.reshape((b, lq, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, lkv, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, lkv, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q
            .contiguous()?
            .matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
            .affine(scale, 0.)?;
        let probs = nn::ops::softmax(&scores, D::Minus1)?;
        let out = probs.matmul(&v.contiguous()?)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, lq, self.num_heads * self.head_dim))?;
        Ok(self.to_out.forward(&out)?)
    }
}

/// FeedForward — `ff.net.0.proj` + `ff.net.2`. GELU-tanh approx.
pub struct FeedForward {
    fc1: LoraLinear,
    fc2: LoraLinear,
}

impl FeedForward {
    pub fn new(
        hidden_size: usize,
        mlp_ratio: usize,
        vb: VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let inner = hidden_size * mlp_ratio;
        let fc1 = wrap_linear(hidden_size, inner, vb.pp("net").pp("0").pp("proj"), registry)?;
        let fc2 = wrap_linear(inner, hidden_size, vb.pp("net").pp("2"), registry)?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?;
        let h = h.gelu()?;
        Ok(self.fc2.forward(&h)?)
    }
}

// ---------------------------------------------------------------------
// Block.
// ---------------------------------------------------------------------

pub struct PixArtBlock {
    scale_shift_table: Tensor,
    attn1: Attention,
    attn2: Attention,
    ff: FeedForward,
    eps_ln: f64,
}

impl PixArtBlock {
    pub fn new(
        cfg: &Config,
        t5_hidden_after_proj: usize,
        vb: VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let scale_shift_table = vb
            .get((6, cfg.hidden_size), "scale_shift_table")
            .map_err(|e| anyhow!("PixArtBlock scale_shift_table: {e}"))?;
        // v0.36 phase 3: PixArt-Σ-2K-MS applies KV-compression to
        // EVERY block's self-attention (paper §3.2). attn1 picks up
        // the config; attn2 (cross-attn to T5) never compresses.
        let attn1 = Attention::new_with_compression(
            cfg.hidden_size,
            cfg.hidden_size,
            cfg.num_heads,
            vb.pp("attn1"),
            cfg.kv_compression,
            registry,
        )?;
        let attn2 = Attention::new(
            cfg.hidden_size,
            t5_hidden_after_proj,
            cfg.num_heads,
            vb.pp("attn2"),
            registry,
        )?;
        let ff = FeedForward::new(cfg.hidden_size, cfg.mlp_ratio, vb.pp("ff"), registry)?;
        Ok(Self {
            scale_shift_table,
            attn1,
            attn2,
            ff,
            eps_ln: 1e-6,
        })
    }

    /// v0.36 phase 3: `grid_dims` is `Some((grid_h, grid_w))` for
    /// the image-token grid. Required when the block was built with
    /// KV-compression enabled; ignored when not.
    pub fn forward(
        &self,
        x: &Tensor,
        t_block: &Tensor,
        kv: &Tensor,
        grid_dims: Option<(usize, usize)>,
    ) -> Result<Tensor> {
        let (b, _t, hidden) = x.dims3()?;
        // (6, hidden) + (B, 6*hidden) → broadcast → (B, 6, hidden).
        let t_block = t_block.reshape((b, 6, hidden))?;
        let mod_vec = self.scale_shift_table.unsqueeze(0)?.broadcast_add(&t_block)?;
        let chunks = mod_vec.chunk(6, 1)?;
        let shift_msa = chunks[0].squeeze(1)?;
        let scale_msa = chunks[1].squeeze(1)?;
        let gate_msa = chunks[2].squeeze(1)?;
        let shift_mlp = chunks[3].squeeze(1)?;
        let scale_mlp = chunks[4].squeeze(1)?;
        let gate_mlp = chunks[5].squeeze(1)?;

        let norm_x = layernorm_no_affine(x, self.eps_ln)?;
        let one = Tensor::ones((1, 1, hidden), x.dtype(), x.device())?;
        let scale_msa_3d = scale_msa.unsqueeze(1)?;
        let shift_msa_3d = shift_msa.unsqueeze(1)?;
        let modulated_x = norm_x
            .broadcast_mul(&one.broadcast_add(&scale_msa_3d)?)?
            .broadcast_add(&shift_msa_3d)?;
        // v0.36 phase 3: self-attn forward picks up grid_dims so
        // the Conv2d (if any) can reshape the KV side spatially.
        let attn1_out = self.attn1.forward_self_attn(&modulated_x, grid_dims)?;
        let x = x.add(&attn1_out.broadcast_mul(&gate_msa.unsqueeze(1)?)?)?;

        let attn2_out = self.attn2.forward(&x, kv)?;
        let x = x.add(&attn2_out)?;

        let norm_x2 = layernorm_no_affine(&x, self.eps_ln)?;
        let scale_mlp_3d = scale_mlp.unsqueeze(1)?;
        let shift_mlp_3d = shift_mlp.unsqueeze(1)?;
        let modulated_x2 = norm_x2
            .broadcast_mul(&one.broadcast_add(&scale_mlp_3d)?)?
            .broadcast_add(&shift_mlp_3d)?;
        let ff_out = self.ff.forward(&modulated_x2)?;
        Ok(x.add(&ff_out.broadcast_mul(&gate_mlp.unsqueeze(1)?)?)?)
    }
}

/// LayerNorm with no affine params (matches `nn.LayerNorm(hidden,
/// elementwise_affine=False)` in PyTorch).
pub fn layernorm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let h = x.dim(D::Minus1)?;
    let weight = Tensor::ones(h, x.dtype(), x.device())?;
    Ok(nn::LayerNorm::new_no_bias(weight, eps).forward(x)?)
}

// ---------------------------------------------------------------------
// Top-level model.
// ---------------------------------------------------------------------

pub struct PixArtSigmaXL {
    pub cfg: Config,
    pub patch_embed: PatchEmbed,
    pub adaln_single: AdaLnSingle,
    pub caption_projection: CaptionProjection,
    pub blocks: Vec<PixArtBlock>,
    pub proj_out: nn::Linear,
    pub final_scale_shift: Tensor,
    pub dtype: DType,
    pub device: Device,
    /// v1.10.0: path → LoRA-registry-entry for every attention/FF
    /// projection, keyed by full safetensors key. Populated during
    /// construction; consumed by `install_train_adapters`. Stored as
    /// the unwrapped `HashMap` (the `Arc<RwLock<…>>` used during build
    /// is dropped at the end of `new`), mirroring `mmdit_inner.rs`.
    lora_registry: LoraRegistry,
}

impl PixArtSigmaXL {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let patch_embed = PatchEmbed::new(
            cfg.in_channels,
            cfg.hidden_size,
            cfg.patch_size,
            vb.pp("pos_embed"),
        )?;
        let adaln_single = AdaLnSingle::new(cfg.hidden_size, vb.pp("adaln_single"))?;
        let caption_projection =
            CaptionProjection::new(cfg.caption_channels, cfg.hidden_size, vb.pp("caption_projection"))?;
        // v1.10.0: shared LoRA registry — every constructed LoraLinear
        // writes its slot/train handles into this map. After all blocks
        // are built (and the sub-loaders go out of scope), we unwrap the
        // Arc and move the inner HashMap into the struct field.
        let registry_arc = Arc::new(RwLock::new(LoraRegistry::new()));
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(PixArtBlock::new(
                &cfg,
                cfg.hidden_size,
                vb.pp("transformer_blocks").pp(&i.to_string()),
                &registry_arc,
            )?);
        }
        let proj_out = nn::linear(
            cfg.hidden_size,
            cfg.patch_size * cfg.patch_size * cfg.out_channels,
            vb.pp("proj_out"),
        )
        .map_err(|e| anyhow!("proj_out: {e}"))?;
        let final_scale_shift = vb
            .get((2, cfg.hidden_size), "scale_shift_table")
            .map_err(|e| anyhow!("final scale_shift_table: {e}"))?;
        // Move the registry out of the Arc — every block + sub-loader is
        // dropped by now, so the ref count is 1. (Mirrors MMDiT.)
        let lora_registry = Arc::try_unwrap(registry_arc)
            .map_err(|_| anyhow!("PixArt LoRA registry still has outstanding refs after construction"))?
            .into_inner()
            .map_err(|_| anyhow!("PixArt LoRA registry RwLock poisoned at construction"))?;
        Ok(Self {
            cfg,
            patch_embed,
            adaln_single,
            caption_projection,
            blocks,
            proj_out,
            final_scale_shift,
            dtype,
            device,
            lora_registry,
        })
    }

    /// `plakat style train` / DreamBooth: install a fresh **trainable**
    /// LoRA adapter on every attention projection (registry keys
    /// containing `.attn1.` or `.attn2.` — self- and cross-attention
    /// q/k/v/out; feed-forward is excluded). Returns `(registry_key, A,
    /// B)` for each, so the caller drives AdamW and writes the save.
    /// Standard init: `A ~ N(0, 0.02)`, `B = 0`, so the adapter starts as
    /// a no-op on the frozen base and learns the style delta. Vars are
    /// F32 (training dtype). Mirrors `MMDiT::install_train_adapters`.
    pub fn install_train_adapters(
        &self,
        rank: usize,
        scale: f64,
        device: &Device,
    ) -> Result<Vec<(String, Var, Var)>> {
        let mut keys: Vec<&String> = self
            .lora_registry
            .keys()
            .filter(|k| k.contains(".attn1.") || k.contains(".attn2."))
            .collect();
        keys.sort();
        let mut out = Vec::with_capacity(keys.len());
        for key in keys {
            let entry = &self.lora_registry[key];
            let a = Var::from_tensor(&Tensor::randn(
                0f32,
                0.02f32,
                (rank, entry.in_dim),
                device,
            )?)?;
            let b = Var::from_tensor(&Tensor::zeros((entry.out_dim, rank), DType::F32, device)?)?;
            *entry
                .train
                .write()
                .map_err(|_| anyhow!("PixArt train slot poisoned"))? =
                Some((a.clone(), b.clone(), scale));
            out.push((key.clone(), a, b));
        }
        Ok(out)
    }

    pub fn forward(
        &self,
        latent: &Tensor,
        timestep: &Tensor,
        caption: &Tensor,
        resolution: &Tensor,
        aspect_ratio: &Tensor,
    ) -> Result<Tensor> {
        let (b, _c, lh, lw) = latent.dims4()?;
        let x = self.patch_embed.forward(latent)?;
        let (grid_h, grid_w) = self.patch_embed.grid_dims(lh, lw);
        // interpolation_scale = max(latent_sample // 64, 1), where the
        // latent sample size is grid·patch (diffusers default). base_size
        // is the model's native token grid (`sample_size`).
        let interp = (((self.cfg.sample_size * self.cfg.patch_size) as f32) / 64.0).floor().max(1.0);
        let pe = build_2d_sincos_pos_embed(
            self.cfg.hidden_size,
            grid_h,
            grid_w,
            self.cfg.sample_size,
            interp,
            x.device(),
            x.dtype(),
        )?;
        let x = x.broadcast_add(&pe)?;

        let (t_block, embedded) =
            self.adaln_single.forward(timestep, resolution, aspect_ratio)?;
        let kv = self.caption_projection.forward(caption)?;

        // v0.36 phase 3: pass the image-token grid through to each
        // block so its self-attn KV-compression Conv2d (if any) can
        // reshape spatially. Same `grid_dims` shared by every layer.
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x, &t_block, &kv, Some((grid_h, grid_w)))?;
        }

        // Final adaLN + proj_out. diffusers uses the raw `embedded_timestep`
        // (B, hidden) here — NOT the first two chunks of the 6-way block
        // `t_block` (which is linear(silu(embedded)), a different tensor).
        let emb_unsq = embedded.unsqueeze(1)?; // (B, 1, hidden)
        let mod_final = self.final_scale_shift.unsqueeze(0)?.broadcast_add(&emb_unsq)?;
        let shift = mod_final.i((.., 0, ..))?.unsqueeze(1)?;
        let scale = mod_final.i((.., 1, ..))?.unsqueeze(1)?;
        let norm_x = layernorm_no_affine(&x, 1e-6)?;
        let one = Tensor::ones(
            (1, 1, self.cfg.hidden_size),
            x.dtype(),
            x.device(),
        )?;
        let x = norm_x
            .broadcast_mul(&one.broadcast_add(&scale)?)?
            .broadcast_add(&shift)?;
        let x = self.proj_out.forward(&x)?;

        let out_h = grid_h * self.cfg.patch_size;
        let out_w = grid_w * self.cfg.patch_size;
        let x = x.reshape((
            b,
            grid_h,
            grid_w,
            self.cfg.patch_size,
            self.cfg.patch_size,
            self.cfg.out_channels,
        ))?;
        let x = x.permute((0, 5, 1, 3, 2, 4))?;
        Ok(x.reshape((b, self.cfg.out_channels, out_h, out_w))?)
    }
}

// =====================================================================
// Tests — shape verification with random weights.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    /// REFERENCE-COMPARISON DUMP (diagnostic; `#[ignore]`d). Builds the
    /// real PixArt-Σ DiT from the HF cache, runs it on a deterministic
    /// input, and writes input + output as raw f32 to /tmp for an
    /// element-wise diff against diffusers' PixArtTransformer2DModel.
    /// Run: `cargo test --release dump_pixart_dit_ref -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dump_pixart_dit_ref() {
        use std::io::Write;
        let home = std::env::var("HOME").unwrap();
        let base = format!(
            "{home}/.cache/huggingface/hub/models--PixArt-alpha--PixArt-Sigma-XL-2-1024-MS/snapshots"
        );
        let snap = std::fs::read_dir(&base)
            .expect("checkpoint cached")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.join("transformer/diffusion_pytorch_model.safetensors").exists())
            .expect("snapshot with transformer weights");
        let weights = snap.join("transformer/diffusion_pytorch_model.safetensors");

        let device = Device::Cpu;
        let dtype = DType::F32;
        let cfg = Config::for_pixart_repo("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], dtype, &device).unwrap() };
        let dit = PixArtSigmaXL::new(cfg.clone(), vb).expect("build DiT from real weights");

        let det = |n: usize, off: usize| -> Vec<f32> {
            (0..n).map(|i| (((i + off) as f32 * 37.0) % 1000.0) / 1000.0 - 0.5).collect()
        };
        let (lh, lw, seq) = (16usize, 16usize, 16usize);
        let latent_v = det(4 * lh * lw, 0);
        let caption_v = det(seq * cfg.caption_channels, 100);
        let latent = Tensor::from_vec(latent_v.clone(), (1, 4, lh, lw), &device).unwrap();
        let caption =
            Tensor::from_vec(caption_v.clone(), (1, seq, cfg.caption_channels), &device).unwrap();
        let t = Tensor::from_vec(vec![500.0f32], (1,), &device).unwrap();
        let res = Tensor::from_vec(vec![1024.0f32, 1024.0], (1, 2), &device).unwrap();
        let asp = Tensor::from_vec(vec![1.0f32, 1.0], (1, 2), &device).unwrap();

        let write = |p: &str, t: &Tensor| {
            let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let mut f = std::fs::File::create(p).unwrap();
            for x in &v {
                f.write_all(&x.to_le_bytes()).unwrap();
            }
        };
        let write_raw = |p: &str, v: &[f32]| {
            let mut f = std::fs::File::create(p).unwrap();
            for x in v {
                f.write_all(&x.to_le_bytes()).unwrap();
            }
        };
        write_raw("/tmp/dit_latent.f32", &latent_v);
        write_raw("/tmp/dit_caption.f32", &caption_v);

        // --- Intermediate stages, to localize the divergence ---
        let x_patch = dit.patch_embed.forward(&latent).unwrap();
        let (gh, gw) = dit.patch_embed.grid_dims(lh, lw);
        let interp = (((cfg.sample_size * cfg.patch_size) as f32) / 64.0).floor().max(1.0);
        let pe =
            build_2d_sincos_pos_embed(cfg.hidden_size, gh, gw, cfg.sample_size, interp, &device, dtype)
                .unwrap();
        let x_input = x_patch.broadcast_add(&pe).unwrap();
        write("/tmp/dit_xinput.f32", &x_input); // patch_embed + pos_embed
        write("/tmp/dit_pe.f32", &pe); // positional embedding alone
        let cap_proj = dit.caption_projection.forward(&caption).unwrap();
        write("/tmp/dit_capproj.f32", &cap_proj);
        let (t_block, _) = dit.adaln_single.forward(&t, &res, &asp).unwrap();
        write("/tmp/dit_tblock.f32", &t_block);
        let blk0 = dit.blocks[0]
            .forward(&x_input, &t_block, &cap_proj, Some((gh, gw)))
            .unwrap();
        write("/tmp/dit_blk0.f32", &blk0);

        let out = dit.forward(&latent, &t, &caption, &res, &asp).expect("DiT forward");
        let out_v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        write("/tmp/dit_out.f32", &out);
        let (mn, mx) = out_v
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        eprintln!("DIT DUMP out_dims={:?} n={} min={mn:.4} max={mx:.4}", out.dims(), out_v.len());
    }

    fn small_cfg() -> Config {
        Config {
            in_channels: 4,
            out_channels: 8,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            mlp_ratio: 2,
            patch_size: 2,
            sample_size: 8,
            caption_channels: 96,
            max_caption_tokens: 12,
            kv_compression: None,
        }
    }

    fn random_model(cfg: Config) -> (PixArtSigmaXL, VarMap) {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let model = PixArtSigmaXL::new(cfg, vb).expect("PixArtSigmaXL::new");
        (model, varmap)
    }

    #[test]
    fn patch_embed_produces_correct_token_count() {
        let (model, _vm) = random_model(small_cfg());
        let device = &model.device;
        let latent = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let tokens = model.patch_embed.forward(&latent).unwrap();
        let (b, t, h) = tokens.dims3().unwrap();
        assert_eq!(b, 1);
        assert_eq!(t, 16);
        assert_eq!(h, 64);
    }

    #[test]
    fn timestep_embedder_shape_round_trips() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let te = TimestepEmbedder::new(64, vb).unwrap();
        let t = Tensor::new(&[100f32, 250.0], &device).unwrap();
        let emb = te.forward(&t).unwrap();
        assert_eq!(emb.dims(), &[2, 64]);
    }

    #[test]
    fn adaln_single_forward_returns_six_hidden_modulation() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let block = AdaLnSingle::new(64, vb).unwrap();
        let t = Tensor::new(&[100f32], &device).unwrap();
        let res = Tensor::new(&[1024f32, 1024.0], &device).unwrap().reshape((1, 2)).unwrap();
        let asp = Tensor::new(&[1f32, 1.0], &device).unwrap().reshape((1, 2)).unwrap();
        let (t_block, emb) = block.forward(&t, &res, &asp).unwrap();
        assert_eq!(t_block.dims(), &[1, 6 * 64]);
        assert_eq!(emb.dims(), &[1, 64]);
    }

    #[test]
    fn caption_projection_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cp = CaptionProjection::new(96, 64, vb).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (1, 12, 96), &device).unwrap();
        let out = cp.forward(&cap).unwrap();
        assert_eq!(out.dims(), &[1, 12, 64]);
    }

    #[test]
    fn attention_self_attn_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = Attention::new(64, 64, 4, vb, &Arc::new(RwLock::new(LoraRegistry::new()))).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 16, 64), &device).unwrap();
        let out = attn.forward(&x, &x).unwrap();
        assert_eq!(out.dims(), &[1, 16, 64]);
    }

    #[test]
    fn attention_cross_attn_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = Attention::new(64, 64, 4, vb, &Arc::new(RwLock::new(LoraRegistry::new()))).unwrap();
        let q = Tensor::randn(0f32, 1f32, (1, 16, 64), &device).unwrap();
        let kv = Tensor::randn(0f32, 1f32, (1, 12, 64), &device).unwrap();
        let out = attn.forward(&q, &kv).unwrap();
        assert_eq!(out.dims(), &[1, 16, 64]);
    }

    #[test]
    fn feedforward_inner_dim_round_trip() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let ff = FeedForward::new(64, 2, vb, &Arc::new(RwLock::new(LoraRegistry::new()))).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 16, 64), &device).unwrap();
        let out = ff.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 16, 64]);
    }

    #[test]
    fn full_forward_round_trips_latent_shape() {
        let (model, _vm) = random_model(small_cfg());
        let device = &model.device;
        let latent = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let t = Tensor::new(&[100f32], device).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (1, 12, 96), device).unwrap();
        let res = Tensor::new(&[1024f32, 1024.0], device).unwrap().reshape((1, 2)).unwrap();
        let asp = Tensor::new(&[1f32, 1.0], device).unwrap().reshape((1, 2)).unwrap();
        let out = model.forward(&latent, &t, &cap, &res, &asp).unwrap();
        // learn_sigma=True doubles channels: 4 → 8.
        assert_eq!(out.dims(), &[1, 8, 8, 8]);
    }

    #[test]
    fn sincos_pos_embed_has_correct_token_count() {
        let device = Device::Cpu;
        let pe = build_2d_sincos_pos_embed(64, 4, 4, 4, 1.0, &device, DType::F32).unwrap();
        assert_eq!(pe.dims(), &[1, 16, 64]);
    }

    /// v0.36 phase 2: 512-MS config differs from 1024-MS ONLY in
    /// `sample_size` — every other field (which affects parameter
    /// shapes + safetensors loading) must match exactly.
    #[test]
    fn sigma_xl_512_differs_only_in_sample_size() {
        let c1024 = Config::sigma_xl_1024();
        let c512 = Config::sigma_xl_512();
        assert_eq!(c1024.in_channels, c512.in_channels);
        assert_eq!(c1024.out_channels, c512.out_channels);
        assert_eq!(c1024.hidden_size, c512.hidden_size);
        assert_eq!(c1024.num_layers, c512.num_layers);
        assert_eq!(c1024.num_heads, c512.num_heads);
        assert_eq!(c1024.mlp_ratio, c512.mlp_ratio);
        assert_eq!(c1024.patch_size, c512.patch_size);
        assert_eq!(c1024.caption_channels, c512.caption_channels);
        assert_eq!(c1024.max_caption_tokens, c512.max_caption_tokens);
        // The one intentional difference.
        assert_eq!(c1024.sample_size, 64);
        assert_eq!(c512.sample_size, 32);
    }

    /// v0.36 phase 2 / 3: `Config::for_pixart_repo` routes to the
    /// right constructor based on the canonical repo path. 2K-MS
    /// precedes 512-MS in detection priority (its substring
    /// `Sigma-XL-2-2K-MS` doesn't conflict with 512-MS's
    /// `Sigma-XL-2-512-MS`, but the early-return order is documented
    /// here).
    #[test]
    fn for_pixart_repo_picks_correct_variant() {
        let c1024 = Config::for_pixart_repo("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS");
        assert_eq!(c1024.sample_size, 64);
        assert!(c1024.kv_compression.is_none());

        let c512 = Config::for_pixart_repo("PixArt-alpha/PixArt-Sigma-XL-2-512-MS");
        assert_eq!(c512.sample_size, 32);
        assert!(c512.kv_compression.is_none());

        let c2k = Config::for_pixart_repo("PixArt-alpha/PixArt-Sigma-XL-2-2K-MS");
        assert_eq!(c2k.sample_size, 128);
        let kvc = c2k.kv_compression.expect("2K-MS must enable KV-compression");
        assert_eq!(kvc.scale_factor, 2);

        // Mixed case still matches.
        let c2k_mixed = Config::for_pixart_repo("PIXART-ALPHA/PIXART-SIGMA-XL-2-2K-MS");
        assert!(c2k_mixed.kv_compression.is_some());

        // Unknown repo string falls back to 1024 (safe — KV-
        // compression off; the inference path skips the Conv2d).
        let c_fallback = Config::for_pixart_repo("user/some-fork");
        assert_eq!(c_fallback.sample_size, 64);
        assert!(c_fallback.kv_compression.is_none());
    }

    /// v0.36 phase 3: 2K-MS config differs from 1024-MS in
    /// `sample_size` AND `kv_compression`. Every other architectural
    /// param matches exactly (KV-compression is a runtime addition,
    /// not a different transformer).
    #[test]
    fn sigma_xl_2k_differs_only_in_sample_size_and_kv_compress() {
        let c1024 = Config::sigma_xl_1024();
        let c2k = Config::sigma_xl_2k();
        assert_eq!(c1024.in_channels, c2k.in_channels);
        assert_eq!(c1024.out_channels, c2k.out_channels);
        assert_eq!(c1024.hidden_size, c2k.hidden_size);
        assert_eq!(c1024.num_layers, c2k.num_layers);
        assert_eq!(c1024.num_heads, c2k.num_heads);
        assert_eq!(c1024.mlp_ratio, c2k.mlp_ratio);
        assert_eq!(c1024.patch_size, c2k.patch_size);
        assert_eq!(c1024.caption_channels, c2k.caption_channels);
        assert_eq!(c1024.max_caption_tokens, c2k.max_caption_tokens);
        // The two intentional differences.
        assert_eq!(c1024.sample_size, 64);
        assert_eq!(c2k.sample_size, 128);
        assert!(c1024.kv_compression.is_none());
        assert_eq!(c2k.kv_compression.unwrap().scale_factor, 2);
    }

    /// v0.36 phase 3: KV-compression downsamples the K/V sequence
    /// by `scale²×` (factor in each axis). With scale=2 on a 16-
    /// token grid (4×4), KV becomes 4 tokens (2×2). The Q dim
    /// stays at 16. Output rejoins the Q-side sequence length.
    #[test]
    fn self_attn_with_kv_compression_downsamples_kv_seq() {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = Attention::new_with_compression(
            64, // query_dim
            64, // kv_dim (same as query for self-attn)
            4,  // num_heads
            vb,
            Some(KvCompressionConfig { scale_factor: 2 }),
            &Arc::new(RwLock::new(LoraRegistry::new())),
        )
        .unwrap();
        // Image tokens: 4×4 grid = 16 tokens, 64 hidden each.
        let x = Tensor::randn(0f32, 1f32, (1, 16, 64), &device).unwrap();
        let out = attn.forward_self_attn(&x, Some((4, 4))).unwrap();
        // Q-side sequence length preserved.
        assert_eq!(out.dims(), &[1, 16, 64]);
    }

    /// Without compression, `forward_self_attn` produces the same
    /// shape as the legacy `forward(x, x)` path.
    #[test]
    fn self_attn_without_compression_matches_legacy_shape() {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = Attention::new_with_compression(64, 64, 4, vb, None, &Arc::new(RwLock::new(LoraRegistry::new()))).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 16, 64), &device).unwrap();
        let out = attn.forward_self_attn(&x, Some((4, 4))).unwrap();
        assert_eq!(out.dims(), &[1, 16, 64]);
        // Sanity: grid_dims is ignored when no compression.
        let out2 = attn.forward_self_attn(&x, None).unwrap();
        assert_eq!(out2.dims(), &[1, 16, 64]);
    }

    /// Compression with mismatched grid_dims (lkv != grid_h*grid_w)
    /// bails loudly. Guards against feeding cross-attn streams into
    /// the self-attn-compression path.
    #[test]
    fn self_attn_with_compression_bails_on_grid_mismatch() {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = Attention::new_with_compression(
            64,
            64,
            4,
            vb,
            Some(KvCompressionConfig { scale_factor: 2 }),
            &Arc::new(RwLock::new(LoraRegistry::new())),
        )
        .unwrap();
        // 12 tokens claimed as 4×4 grid → mismatch (16 expected).
        let x = Tensor::randn(0f32, 1f32, (1, 12, 64), &device).unwrap();
        let err = attn.forward_self_attn(&x, Some((4, 4))).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("grid_h*grid_w"), "got {msg}");
    }

    /// Compression without grid_dims bails loudly (caller forgot
    /// to thread the dims through).
    #[test]
    fn self_attn_with_compression_bails_without_grid_dims() {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = Attention::new_with_compression(
            64,
            64,
            4,
            vb,
            Some(KvCompressionConfig { scale_factor: 2 }),
            &Arc::new(RwLock::new(LoraRegistry::new())),
        )
        .unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 16, 64), &device).unwrap();
        let err = attn.forward_self_attn(&x, None).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("grid_dims"), "got {msg}");
    }

    /// v0.36 phase 3: end-to-end forward with KV-compression
    /// enabled produces the same output shape as without. Uses a
    /// small config (2 layers, 64 hidden) for fast tests.
    #[test]
    fn full_forward_with_kv_compression_preserves_output_shape() {
        let cfg = Config {
            in_channels: 4,
            out_channels: 8,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            mlp_ratio: 2,
            patch_size: 2,
            sample_size: 8,
            caption_channels: 96,
            max_caption_tokens: 12,
            kv_compression: Some(KvCompressionConfig { scale_factor: 2 }),
        };
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = PixArtSigmaXL::new(cfg, vb).expect("PixArtSigmaXL::new with kv_compress");

        // (1, 4, 8, 8) latent → 4×4 grid → KV downsamples to 2×2 = 4
        // tokens internally, but the Q-side stays at 16. Output
        // shape matches the no-compression path: (1, 8, 8, 8).
        let latent = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &device).unwrap();
        let t = Tensor::new(&[100f32], &device).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (1, 12, 96), &device).unwrap();
        let res = Tensor::new(&[1024f32, 1024.0], &device).unwrap().reshape((1, 2)).unwrap();
        let asp = Tensor::new(&[1f32, 1.0], &device).unwrap().reshape((1, 2)).unwrap();
        let out = model.forward(&latent, &t, &cap, &res, &asp).unwrap();
        assert_eq!(out.dims(), &[1, 8, 8, 8]);
    }
}
