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

use anyhow::{Result, anyhow, bail};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};
use candle_transformers::models::clip::text_model::Activation;
use candle_transformers::models::clip::vision_model::{
    ClipVisionConfig, ClipVisionTransformer,
};
use std::path::Path;

/// HF repo that hosts every IP-Adapter weight file plakat consumes.
pub const IPA_REPO: &str = "h94/IP-Adapter";

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

    /// Per-layer hidden states, indexed from the end of the encoder.
    /// `n_from_end == 2` returns the penultimate transformer block's output
    /// — the (B, 257, 1280) tensor IP-Adapter-Plus consumes.
    ///
    /// candle's `output_hidden_states` returns `num_layers + 1` entries:
    ///   * indices `0 .. num_layers`: per-transformer-block outputs
    ///   * index `num_layers`:        pooled + post-layernormed CLS token
    /// Diffusers' `hidden_states[-2]` (used by IP-Adapter-Plus) is the
    /// second-to-last transformer output, which lives at `len - 3` here
    /// (one extra "pooled" entry past the last layer plus a step back).
    pub fn hidden_state_from_end(&self, pixels: &Tensor, n_from_end: usize) -> Result<Tensor> {
        let states = self.vision.output_hidden_states(pixels)?;
        if n_from_end < 1 || n_from_end + 1 > states.len() {
            return Err(anyhow!(
                "hidden_state_from_end({n_from_end}) out of range; have {} states",
                states.len()
            ));
        }
        // states.len() = num_layers + 1.
        // Diffusers index [-k] over a length-(num_layers+1) list = our (len - 1 - k).
        // For k=2 (penultimate): index = len - 3.
        Ok(states[states.len() - 1 - n_from_end].clone())
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

// =====================================================================
// IP-Adapter Plus: Resampler-based projection.
//
// "Plus" variants (incl. Plus-Face) replace the simple Linear+LayerNorm
// projection with a Perceiver-style resampler that produces 16 image
// tokens instead of 4, attending over per-layer CLIP-H hidden states
// (penultimate) rather than the pooled CLIP-H output. Architecturally:
//
//   latents (16, dim) ←attn← CLIP-H hidden states (257, embed_dim)
//                     ↓ FF
//   ... × 4 layers ...
//   proj_out → (16, output_dim)  ← concatenated onto text token sequence
//
// Used by `portrait` (Plus-Face) and a future SDXL Plus path.
// =====================================================================

/// Hyperparameters for the Plus resampler. The defaults at
/// `PlusConfig::sd15_face` match `ip-adapter-plus-face_sd15.safetensors`.
#[derive(Clone, Copy, Debug)]
pub struct PlusConfig {
    /// CLIP-H hidden size feeding the resampler (1280 for CLIP-H/14).
    pub embedding_dim: usize,
    /// Resampler internal hidden size.
    pub dim: usize,
    /// SD cross-attention dim (768 for SD 1.5, 2048 for SDXL).
    pub output_dim: usize,
    /// Number of Perceiver layers (attn + FF blocks).
    pub depth: usize,
    /// Number of learned query latents (= image tokens emitted).
    pub num_queries: usize,
    /// Multi-head attention heads.
    pub heads: usize,
    /// Per-head channel dimension.
    pub dim_head: usize,
    /// Feed-forward expansion multiplier.
    pub ff_mult: usize,
}

impl PlusConfig {
    /// Matches `h94/IP-Adapter/models/ip-adapter-plus-face_sd15.safetensors`.
    pub fn sd15_face() -> Self {
        Self {
            embedding_dim: 1280,
            dim: 1280,
            output_dim: 768,
            depth: 4,
            num_queries: 16,
            heads: 20,
            dim_head: 64,
            ff_mult: 4,
        }
    }
}

/// One Perceiver attention block: latents query against `cat(image, latents)`.
struct PerceiverAttention {
    norm1: LayerNorm,
    norm2: LayerNorm,
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl PerceiverAttention {
    fn new(vs: VarBuilder, dim: usize, heads: usize, head_dim: usize) -> Result<Self> {
        let inner = heads * head_dim;
        Ok(Self {
            norm1: candle_nn::layer_norm(dim, 1e-5, vs.pp("norm1"))?,
            norm2: candle_nn::layer_norm(dim, 1e-5, vs.pp("norm2"))?,
            to_q: candle_nn::linear_no_bias(dim, inner, vs.pp("to_q"))?,
            to_kv: candle_nn::linear_no_bias(dim, inner * 2, vs.pp("to_kv"))?,
            to_out: candle_nn::linear_no_bias(inner, dim, vs.pp("to_out"))?,
            heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, x: &Tensor, latents: &Tensor) -> Result<Tensor> {
        // x:       (B, n_img,  dim)
        // latents: (B, n_lat,  dim)
        let x_n = self.norm1.forward(x)?;
        let lat_n = self.norm2.forward(latents)?;

        let q = self.to_q.forward(&lat_n)?; // (B, n_lat, inner)
        let kv_in = Tensor::cat(&[&x_n, &lat_n], 1)?; // (B, n_img + n_lat, dim)
        let kv = self.to_kv.forward(&kv_in)?; // (B, n_kv, 2*inner)
        let kv_parts = kv.chunk(2, D::Minus1)?;
        let k = &kv_parts[0];
        let v = &kv_parts[1];

        let (b, n_q, _) = q.dims3()?;
        let n_kv = k.dim(1)?;
        let q = q
            .reshape((b, n_q, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, n_kv, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, n_kv, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?; // (B, heads, n_q, head_dim)

        let inner = self.heads * self.head_dim;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, n_q, inner))?;
        Ok(self.to_out.forward(&out)?)
    }
}

/// FeedForward block. Stored as `Sequential[LayerNorm, Linear, GELU, Linear]`
/// in the reference (keys: layers.<i>.1.0 / .1.1 / .1.3 — index 2 is the GELU,
/// which carries no parameters).
struct PerceiverFeedForward {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl PerceiverFeedForward {
    fn new(vs: VarBuilder, dim: usize, mult: usize) -> Result<Self> {
        let inner = dim * mult;
        Ok(Self {
            norm: candle_nn::layer_norm(dim, 1e-5, vs.pp("0"))?,
            fc1: candle_nn::linear_no_bias(dim, inner, vs.pp("1"))?,
            fc2: candle_nn::linear_no_bias(inner, dim, vs.pp("3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm.forward(x)?;
        let h = self.fc1.forward(&h)?;
        let h = h.gelu()?;
        Ok(self.fc2.forward(&h)?)
    }
}

struct PerceiverLayer {
    attn: PerceiverAttention,
    ff: PerceiverFeedForward,
}

impl PerceiverLayer {
    fn new(
        vs: VarBuilder,
        dim: usize,
        heads: usize,
        head_dim: usize,
        ff_mult: usize,
    ) -> Result<Self> {
        Ok(Self {
            attn: PerceiverAttention::new(vs.pp("0"), dim, heads, head_dim)?,
            ff: PerceiverFeedForward::new(vs.pp("1"), dim, ff_mult)?,
        })
    }

    fn forward(&self, x: &Tensor, latents: &Tensor) -> Result<Tensor> {
        let latents = (latents + self.attn.forward(x, latents)?)?;
        let latents = (&latents + self.ff.forward(&latents)?)?;
        Ok(latents)
    }
}

/// Plus-variant IP-Adapter projection (Perceiver resampler). Loads the
/// `image_proj.*` subtree from `ip-adapter-plus-face_sd15.safetensors` etc.
pub struct ImageProjPlus {
    proj_in: Linear,
    proj_out: Linear,
    /// Learned query latents, shape `(1, num_queries, dim)`.
    latents: Tensor,
    layers: Vec<PerceiverLayer>,
    norm_out: LayerNorm,
}

impl ImageProjPlus {
    pub fn load(
        weights: &Path,
        cfg: PlusConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        let vb = vb.pp("image_proj");
        let proj_in = candle_nn::linear(cfg.embedding_dim, cfg.dim, vb.pp("proj_in"))?;
        let proj_out = candle_nn::linear(cfg.dim, cfg.output_dim, vb.pp("proj_out"))?;
        let latents = vb.get((1, cfg.num_queries, cfg.dim), "latents")?;
        let norm_out = candle_nn::layer_norm(cfg.output_dim, 1e-5, vb.pp("norm_out"))?;
        let mut layers = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            layers.push(PerceiverLayer::new(
                vb.pp("layers").pp(i.to_string()),
                cfg.dim,
                cfg.heads,
                cfg.dim_head,
                cfg.ff_mult,
            )?);
        }
        Ok(Self {
            proj_in,
            proj_out,
            latents,
            layers,
            norm_out,
        })
    }

    /// `(B, n_features, embedding_dim)` → `(B, num_queries, output_dim)`.
    /// `n_features` is typically 257 for CLIP-H/14 at 224×224
    /// (1 CLS + 256 patches).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let b = x.dim(0)?;
        let (_, n_lat, d) = self.latents.dims3()?;
        let mut latents = self.latents.broadcast_as((b, n_lat, d))?.contiguous()?;
        let x = self.proj_in.forward(x)?;
        for layer in &self.layers {
            latents = layer.forward(&x, &latents)?;
        }
        let out = self.proj_out.forward(&latents)?;
        Ok(self.norm_out.forward(&out)?)
    }
}

// =====================================================================
// IdentityEncoder trait — Phase-2 plug-in point.
//
// Whatever identity-preservation strategy a `portrait` call uses
// (Plus-Face today; FaceID/InstantID later), it boils down to producing
// a `(1, num_tokens, cross_attn_dim)` tensor that gets concatenated onto
// the text embeddings. This trait isolates that contract so adding a new
// strategy is a drop-in.
// =====================================================================

pub trait IdentityEncoder: Send + Sync {
    /// Produce identity tokens for `photo`. Shape: `(1, num_tokens, cross_attn_dim)`.
    fn encode(&self, photo_path: &Path) -> Result<Tensor>;
    /// Number of image tokens this encoder emits per call.
    fn num_tokens(&self) -> usize;
}

/// IP-Adapter-Plus-Face encoder: CLIP-H penultimate hidden states →
/// Perceiver resampler → 16 image tokens.
pub struct PlusFaceEncoder {
    clip_vision: ImageEncoder,
    image_proj: ImageProjPlus,
    cfg: PlusConfig,
    device: Device,
    dtype: DType,
}

impl PlusFaceEncoder {
    pub fn load(
        clip_vision_weights: &Path,
        plus_face_weights: &Path,
        cfg: PlusConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let clip_vision = ImageEncoder::load(clip_vision_weights, device, dtype)?;
        let image_proj = ImageProjPlus::load(plus_face_weights, cfg, device, dtype)?;
        Ok(Self {
            clip_vision,
            image_proj,
            cfg,
            device: device.clone(),
            dtype,
        })
    }
}

impl IdentityEncoder for PlusFaceEncoder {
    fn num_tokens(&self) -> usize {
        self.cfg.num_queries
    }

    fn encode(&self, photo_path: &Path) -> Result<Tensor> {
        let pixels = crate::imaging::preprocess::clip_image_tensor(
            photo_path,
            224,
            &self.device,
            self.dtype,
        )?;
        // Plus uses CLIP-H's penultimate transformer hidden state, not the
        // pooled projection output. (1, 257, 1280) for CLIP-H/14 @ 224.
        let hidden = self.clip_vision.hidden_state_from_end(&pixels, 2)?;
        self.image_proj.forward(&hidden)
    }
}

// =====================================================================
// IdentityKind — Phase-2 drop-in surface.
//
// Adding a new identity-preservation strategy (FaceID, InstantID, …) is
// fully contained in this file:
//   1. add a variant to `IdentityKind`
//   2. add a parse arm to its `FromStr`
//   3. add a load arm to `IdentityKind::load_encoder`
//   4. write the `IdentityEncoder` impl
// `portrait::Pipeline` doesn't change.
// =====================================================================

/// Which identity-preservation strategy `portrait` should wire up.
#[derive(Clone, Copy, Debug)]
pub enum IdentityKind {
    /// IP-Adapter-Plus-Face on SD 1.5 (Phase 1).
    /// Weights: `models/ip-adapter-plus-face_sd15.safetensors` (Plus
    /// resampler) + `models/image_encoder/model.safetensors` (CLIP-H).
    PlusFace,
    // Phase 2 placeholders, NOT YET IMPLEMENTED:
    //   FaceId    — InsightFace ArcFace ID embedding + ip-adapter-faceid_*
    //   InstantId — ID + landmarks via a ControlNet-style branch
    // When landing them, add a variant here and a `Self::FaceId => ...`
    // arm to `load_encoder`. No portrait::Pipeline edits required.
}

impl IdentityKind {
    /// Cross-attention dim this strategy targets. Used by `portrait` for
    /// shape sanity (e.g. refusing an SD 1.5 strategy on an SDXL UNet).
    pub fn cross_attn_dim(self) -> usize {
        match self {
            Self::PlusFace => 768,
        }
    }

    /// Human-readable label for progress UI.
    pub fn label(self) -> &'static str {
        match self {
            Self::PlusFace => "IP-Adapter-Plus-Face (SD 1.5)",
        }
    }

    /// Download + build the encoder. This is the only hook `portrait`
    /// calls — adding a new strategy means adding a match arm here.
    pub async fn load_encoder(
        self,
        device: &Device,
        dtype: DType,
    ) -> Result<Box<dyn IdentityEncoder>> {
        let dl = crate::ui::progress::spinner(&format!(
            "Resolving identity weights — {}",
            self.label()
        ));
        let encoder: Box<dyn IdentityEncoder> = match self {
            Self::PlusFace => {
                let face_weights = crate::hf::download::get_file(
                    IPA_REPO,
                    "models/ip-adapter-plus-face_sd15.safetensors",
                )
                .await?;
                let clip_weights = crate::hf::download::get_file(
                    IPA_REPO,
                    "models/image_encoder/model.safetensors",
                )
                .await?;
                let enc = PlusFaceEncoder::load(
                    &clip_weights,
                    &face_weights,
                    PlusConfig::sd15_face(),
                    device,
                    dtype,
                )?;
                Box::new(enc)
            }
        };
        dl.finish_with_message(format!("✓ identity ready — {}", self.label()));
        Ok(encoder)
    }
}

impl std::str::FromStr for IdentityKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_lowercase().as_str() {
            "plus-face" | "plusface" | "plus_face" => Self::PlusFace,
            other => bail!(
                "unknown identity kind {other:?} (try: plus-face). \
                 FaceID / InstantID are Phase 2 — not yet implemented."
            ),
        })
    }
}
