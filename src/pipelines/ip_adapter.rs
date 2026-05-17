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

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};
use candle_transformers::models::clip::text_model::Activation;
use candle_transformers::models::clip::vision_model::{
    ClipVisionConfig, ClipVisionTransformer,
};
use std::path::Path;

/// HF repo that hosts every IP-Adapter weight file plakat consumes.
pub const IPA_REPO: &str = "h94/IP-Adapter";

/// FaceID weights live in a separate h94 repo at the root path (not
/// nested under `models/`). Splitting them out keeps the IP-Adapter
/// repo from ballooning past LFS limits.
pub const IPA_FACEID_REPO: &str = "h94/IP-Adapter-FaceID";

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

    /// Load from a PyTorch `.bin` state dict, rooted at a sub-key.
    /// Originally written for FaceID, but FaceID turned out to need
    /// the 2-layer MLP variant (`face_models::FaceIdImageProj`) — so
    /// this single-Linear loader is currently unused. Kept around in
    /// case some future IP-Adapter variant lives in a `.bin` AND uses
    /// the basic single-Linear projection.
    ///
    /// `from_pth_with_state` roots the `VarBuilder` at
    /// `state_dict[state_key]`, so the inner keys read as `proj.weight`,
    /// `norm.weight` etc. (no `image_proj.` prefix).
    #[allow(dead_code)]
    pub fn load_from_pth_subtree(
        weights: &Path,
        state_key: &str,
        clip_embed_dim: usize,
        cross_attn_dim: usize,
        num_tokens: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb =
            VarBuilder::from_pth_with_state(weights, dtype, state_key, device)?;
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
    ///
    /// The published weights use `dim = 768` (matching SD 1.5's cross-attn
    /// output, not CLIP-H's 1280-d input). `proj_in` projects 1280→768
    /// up front; everything inside the resampler operates at 768. The
    /// diffusers reference Plus (non-Face) uses `dim = 1280` — Plus-Face
    /// was trained with the smaller resampler. Verified from
    /// `proj_in.weight.shape == [768, 1280]` in the safetensors file.
    pub fn sd15_face() -> Self {
        Self {
            embedding_dim: 1280,
            dim: 768,
            output_dim: 768,
            depth: 4,
            num_queries: 16,
            heads: 12,
            dim_head: 64,
            ff_mult: 4,
        }
    }

    /// Matches `h94/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors`.
    /// The `vit-h` suffix is significant: this SDXL variant reuses the SD 1.5
    /// CLIP-H image encoder rather than the SDXL CLIP-G encoder, so we
    /// don't need a separate image-encoder download.
    pub fn sdxl_face() -> Self {
        Self {
            embedding_dim: 1280,
            dim: 1280,
            output_dim: 2048,
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

/// Per-call options threaded through `IdentityEncoder::encode`. Kept as a
/// struct so future strategy-specific knobs (face-bbox in 4c.1, landmarks
/// in 4c.3, identity-strength multipliers, …) can extend without
/// re-breaking the trait.
#[derive(Clone, Copy, Debug, Default)]
pub struct EncodeOptions {
    /// Normalised `[x0, y0, x1, y1]` bbox locating the subject's face in
    /// the photo. When `Some`, the photo is cropped to this region before
    /// the strategy's normal preprocessing. Currently used only by
    /// `IdentityKind::FaceId` / `FaceIdSdxl`; CLIP-H-based strategies
    /// (`PlusFace` / `PlusFaceSdxl`) ignore it (CLIP-H sees the whole
    /// image anyway, so a bbox would just throw away context).
    pub face_bbox: Option<[f32; 4]>,
    /// Normalised 5-point landmarks `[[x, y]; 5]` for the subject's face,
    /// ordered: `left_eye, right_eye, nose, left_mouth, right_mouth`.
    /// **Takes precedence over `face_bbox`** — when supplied, FaceID
    /// strategies do a similarity-transform alignment to ArcFace's
    /// canonical 112×112 template. The right way to align; recovers
    /// ~15–25% of identity-discrimination over crop-based alignment.
    /// Currently used only by FaceID strategies.
    pub face_landmarks: Option<[[f32; 2]; 5]>,
}

pub trait IdentityEncoder: Send + Sync {
    /// Produce identity tokens for `photo`. Shape: `(1, num_tokens, cross_attn_dim)`.
    /// `opts` carries per-call adjustments (see [`EncodeOptions`]).
    fn encode(&self, photo_path: &Path, opts: EncodeOptions) -> Result<Tensor>;
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

    fn encode(&self, photo_path: &Path, _opts: EncodeOptions) -> Result<Tensor> {
        // CLIP-H Plus-Face ignores `face_bbox` — CLIP processes the whole
        // image regardless, and pre-cropping to a bbox would just throw
        // away surrounding context that helps identity recognition.
        if !photo_path.exists() {
            return Err(anyhow!(
                "persona photo not found: {} (resolved from current working \
                 directory). Check the path and re-run.",
                photo_path.display()
            ));
        }
        let pixels = crate::imaging::preprocess::clip_image_tensor(
            photo_path,
            224,
            &self.device,
            self.dtype,
        )
        .with_context(|| format!("reading persona photo {}", photo_path.display()))?;
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IdentityKind {
    /// IP-Adapter-Plus-Face on SD 1.5.
    /// Weights: `models/ip-adapter-plus-face_sd15.safetensors` (Plus
    /// resampler) + `models/image_encoder/model.safetensors` (CLIP-H).
    PlusFace,
    /// IP-Adapter-Plus-Face on SDXL via the `vit-h` variant. Reuses
    /// the SD 1.5 CLIP-H image encoder. Resampler outputs at SDXL's
    /// 2048-d cross-attention dim instead of 768.
    /// Weights: `sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors`
    /// + the same `models/image_encoder/model.safetensors` (CLIP-H).
    PlusFaceSdxl,
    /// IP-Adapter-FaceID on SD 1.5.
    ///
    /// Identity is encoded by InsightFace's ArcFace (IR-ResNet50) — a
    /// face-recognition embedding trained specifically to be identity-
    /// discriminative. Markedly better identity preservation than the
    /// general-purpose CLIP-H features Plus-Face uses, when the input
    /// photo is well-aligned.
    ///
    /// Weights:
    ///   * ArcFace IR-ResNet50 safetensors — user-supplied via
    ///     `PLAKAT_ARCFACE_WEIGHTS` (local path) or `PLAKAT_ARCFACE_HF`
    ///     (HuggingFace `repo#file`). See PERSONA.md "FaceID setup".
    ///   * `ip-adapter-faceid_sd15.bin` from h94/IP-Adapter-FaceID —
    ///     auto-downloaded; the `image_proj.*` subtree is consumed by
    ///     the encoder, the separate `*_lora.safetensors` is merged
    ///     into the UNet automatically.
    ///
    /// Alignment fallbacks (richest first): user-supplied landmarks,
    /// SCRFD-detected landmarks (when `PLAKAT_SCRFD_*` set), user-
    /// supplied bbox, centre-crop. See `face_models::prepare_face_tensor`.
    FaceId,
    /// IP-Adapter-FaceID on SDXL. Same ArcFace IR-ResNet50 backbone as
    /// `FaceId` — re-uses the same ArcFace env vars. The difference is
    /// the image-proj output dim (2048 vs 768) and a separate FaceID
    /// weight file from h94/IP-Adapter-FaceID:
    /// `ip-adapter-faceid_sdxl.bin`.
    ///
    /// Same alignment options as `FaceId`. UNet LoRA component is
    /// applied automatically.
    FaceIdSdxl,
    // Future identity strategies, NOT YET IMPLEMENTED:
    //   InstantId — ID + landmarks via a ControlNet-style branch
    // When landing it, add a variant here and a `Self::InstantId => ...`
    // arm to `load_encoder`. No portrait::Pipeline edits required.
}

impl IdentityKind {
    /// Cross-attention dim this strategy targets. Used by `portrait` for
    /// shape sanity (refusing an SD 1.5 strategy on an SDXL UNet etc.).
    pub fn cross_attn_dim(self) -> usize {
        match self {
            Self::PlusFace => 768,
            Self::PlusFaceSdxl => 2048,
            Self::FaceId => 768,
            Self::FaceIdSdxl => 2048,
        }
    }

    /// Human-readable label for progress UI.
    pub fn label(self) -> &'static str {
        match self {
            Self::PlusFace => "IP-Adapter-Plus-Face (SD 1.5)",
            Self::PlusFaceSdxl => "IP-Adapter-Plus-Face (SDXL, vit-h)",
            Self::FaceId => "IP-Adapter-FaceID (SD 1.5, ArcFace)",
            Self::FaceIdSdxl => "IP-Adapter-FaceID (SDXL, ArcFace)",
        }
    }

    /// Which SD variant this strategy expects. Scenario / CLI use this
    /// to validate or auto-pick the portrait pipeline's base model.
    pub fn target_variant(self) -> &'static str {
        match self {
            Self::PlusFace => "sd15",
            Self::PlusFaceSdxl => "sdxl",
            Self::FaceId => "sd15",
            Self::FaceIdSdxl => "sdxl",
        }
    }

    /// Verify strategy-specific local weight requirements *before* the
    /// portrait pipeline starts downloading the (potentially multi-GB)
    /// base model. Currently this only matters for FaceID strategies,
    /// which require `PLAKAT_ARCFACE_WEIGHTS` to point at an existing
    /// safetensors file; other strategies have no local requirements
    /// (everything else flows through `hf::download`).
    pub fn preflight_weights(self) -> Result<()> {
        match self {
            Self::FaceId | Self::FaceIdSdxl => preflight_arcface_local(),
            Self::PlusFace | Self::PlusFaceSdxl => Ok(()),
        }
    }

    /// Resolve this strategy's auxiliary UNet LoRA, if any. FaceID
    /// strategies (SD 1.5 + SDXL) ship a UNet cross-attention LoRA as a
    /// separate kohya-format safetensors next to the `image_proj.*`
    /// `.bin` in `h94/IP-Adapter-FaceID`. We download the safetensors
    /// directly — it's already in the format the existing
    /// `merge_loras_into_weights` consumes, so no conversion needed.
    ///
    /// Returns a path into the HF cache (persistent) or `None` for
    /// strategies without an aux LoRA (`PlusFace*`).
    ///
    /// Opt out via the env var `PLAKAT_FACEID_LORA=off` — useful for
    /// A/B testing if the shared-cross-attention application of this
    /// LoRA degrades text-prompt fidelity for a specific use case.
    ///
    /// h94 ships the UNet LoRA as a separate kohya-format
    /// `_lora.safetensors` next to the FaceID `.bin`; we download
    /// it directly. A converter in `faceid_lora.rs` is retained for
    /// FaceID variants that bundle the LoRA inline in the `.bin`.
    pub async fn aux_unet_lora(
        self,
        _device: &Device,
    ) -> Result<Option<std::path::PathBuf>> {
        if std::env::var("PLAKAT_FACEID_LORA").as_deref() == Ok("off") {
            return Ok(None);
        }
        match self {
            Self::FaceId => {
                // h94/IP-Adapter-FaceID ships the UNet LoRA in a *separate*
                // kohya-format safetensors — the `.bin` next to it only
                // contains the `image_proj.*` MLP. Download the LoRA file
                // directly; no conversion needed (the existing merger
                // consumes it as-is).
                let lora_path = crate::hf::download::get_file(
                    IPA_FACEID_REPO,
                    "ip-adapter-faceid_sd15_lora.safetensors",
                )
                .await?;
                crate::ui::progress::println(&format!(
                    "  ✓ FaceID UNet LoRA: {}",
                    lora_path.display()
                ));
                Ok(Some(lora_path))
            }
            Self::FaceIdSdxl => {
                let lora_path = crate::hf::download::get_file(
                    IPA_FACEID_REPO,
                    "ip-adapter-faceid_sdxl_lora.safetensors",
                )
                .await?;
                crate::ui::progress::println(&format!(
                    "  ✓ FaceID UNet LoRA (SDXL): {}",
                    lora_path.display()
                ));
                Ok(Some(lora_path))
            }
            Self::PlusFace | Self::PlusFaceSdxl => Ok(None),
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
            Self::PlusFaceSdxl => {
                let face_weights = crate::hf::download::get_file(
                    IPA_REPO,
                    "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
                )
                .await?;
                // Same CLIP-H file as SD 1.5 — the `vit-h` SDXL variant
                // reuses it, so the cache hit is free for users who've
                // already run stylize / PlusFace portraits.
                let clip_weights = crate::hf::download::get_file(
                    IPA_REPO,
                    "models/image_encoder/model.safetensors",
                )
                .await?;
                let enc = PlusFaceEncoder::load(
                    &clip_weights,
                    &face_weights,
                    PlusConfig::sdxl_face(),
                    device,
                    dtype,
                )?;
                Box::new(enc)
            }
            Self::FaceId => {
                let arcface_path = resolve_arcface_weights().await?;
                let faceid_weights = crate::hf::download::get_file(
                    IPA_FACEID_REPO,
                    "ip-adapter-faceid_sd15.bin",
                )
                .await?;
                let scrfd_path =
                    crate::pipelines::scrfd::resolve_scrfd_weights().await?;
                let enc = crate::pipelines::face_models::FaceIdEncoder::load_faceid_sd15(
                    &arcface_path,
                    &faceid_weights,
                    scrfd_path.as_deref(),
                    device,
                    dtype,
                )?;
                Box::new(enc)
            }
            Self::FaceIdSdxl => {
                let arcface_path = resolve_arcface_weights().await?;
                let faceid_weights = crate::hf::download::get_file(
                    IPA_FACEID_REPO,
                    "ip-adapter-faceid_sdxl.bin",
                )
                .await?;
                let scrfd_path =
                    crate::pipelines::scrfd::resolve_scrfd_weights().await?;
                let enc = crate::pipelines::face_models::FaceIdEncoder::load_faceid_sdxl(
                    &arcface_path,
                    &faceid_weights,
                    scrfd_path.as_deref(),
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
            "plus-face-sdxl"
            | "plusface-sdxl"
            | "plus_face_sdxl"
            | "plus-face-xl"
            | "plusface-xl"
            | "sdxl-plus-face" => Self::PlusFaceSdxl,
            "faceid" | "face-id" | "face_id" => Self::FaceId,
            "faceid-sdxl"
            | "face-id-sdxl"
            | "face_id_sdxl"
            | "faceid-xl"
            | "sdxl-faceid" => Self::FaceIdSdxl,
            other => bail!(
                "unknown identity kind {other:?} \
                 (try: plus-face, plus-face-sdxl, faceid, faceid-sdxl). \
                 InstantID is roadmap — not yet implemented."
            ),
        })
    }
}

/// Setup-instructions text used by both the sync preflight and the async
/// resolver. Kept in one place so updates to the setup story don't drift.
fn arcface_setup_message() -> &'static str {
    "FaceID requires ArcFace IR-ResNet50 weights. Two ways to provide them:\n\
     \n\
     A. Local file (`PLAKAT_ARCFACE_WEIGHTS`):\n\
     \n     1. Download antelopev2.zip from InsightFace:\n\
     \n        https://github.com/deepinsight/insightface/releases/tag/v0.7\n\
     \n     2. Extract w600k_r50.onnx from the bundle.\n\
     \n     3. Convert to safetensors (one-time):\n\
     \n        python -c \"import onnx, torch; \\\n\
            from onnx2torch import convert; \\\n\
            from safetensors.torch import save_file; \\\n\
            m = convert(onnx.load('w600k_r50.onnx')); \\\n\
            save_file(m.state_dict(), 'arcface_r50.safetensors')\"\n\
     \n     4. export PLAKAT_ARCFACE_WEIGHTS=/path/to/arcface_r50.safetensors\n\
     \n\
     B. HuggingFace-hosted (`PLAKAT_ARCFACE_HF`):\n\
     \n     Point at any HF safetensors of the IR-ResNet50 ArcFace weights:\n\
     \n        export PLAKAT_ARCFACE_HF=<user>/<repo>#<path/in/repo.safetensors>\n\
     \n     plakat downloads + caches automatically. No canonical default\n\
     \n     repo yet — discover candidates at:\n\
     \n        https://huggingface.co/models?search=arcface+iresnet50\n\
     \n     or:\n\
     \n        https://huggingface.co/models?search=insightface+r50\n\
     \n     (plakat doesn't endorse any specific community upload — check\n\
     \n     the repo's README + license before depending on it.)\n\
     \n\
     Both routes also work for `IdentityKind::FaceIdSdxl`.\n\
     Run `plakat doctor` to check your current setup; add `--verify`\n\
     to actively test the configured HF spec downloads."
}

/// Sync preflight — confirms ArcFace can plausibly resolve later. Doesn't
/// hit the network. Used by `portrait::Pipeline::load` to fail fast before
/// kicking off the base-model download.
pub(crate) fn preflight_arcface_local() -> Result<()> {
    let has_local = std::env::var("PLAKAT_ARCFACE_WEIGHTS").is_ok();
    let has_hf = std::env::var("PLAKAT_ARCFACE_HF").is_ok();
    if !has_local && !has_hf {
        bail!("{}", arcface_setup_message());
    }
    if let Ok(env) = std::env::var("PLAKAT_ARCFACE_WEIGHTS") {
        let path = std::path::PathBuf::from(&env);
        if !path.exists() {
            bail!(
                "PLAKAT_ARCFACE_WEIGHTS points to {} which doesn't exist. \
                 (You can also set PLAKAT_ARCFACE_HF=repo#file to download \
                  from HuggingFace instead — see `plakat doctor`.)",
                path.display()
            );
        }
    }
    // PLAKAT_ARCFACE_HF can't be validated without a network call;
    // the async resolver will surface 404s clearly.
    Ok(())
}

/// Async resolver — turns env-var configuration into an on-disk
/// safetensors path. `PLAKAT_ARCFACE_WEIGHTS` (local) wins over
/// `PLAKAT_ARCFACE_HF` (HuggingFace).
async fn resolve_arcface_weights() -> Result<std::path::PathBuf> {
    if let Ok(env) = std::env::var("PLAKAT_ARCFACE_WEIGHTS") {
        let path = std::path::PathBuf::from(&env);
        if !path.exists() {
            bail!(
                "PLAKAT_ARCFACE_WEIGHTS points to {} which doesn't exist.",
                path.display()
            );
        }
        return Ok(path);
    }
    if let Ok(spec) = std::env::var("PLAKAT_ARCFACE_HF") {
        let (repo, file) = parse_hf_spec(&spec, "PLAKAT_ARCFACE_HF")?;
        let s = crate::ui::progress::spinner(&format!(
            "Downloading ArcFace from {repo}/{file}"
        ));
        let path = crate::hf::download::get_file(&repo, &file)
            .await
            .with_context(|| {
                format!("downloading ArcFace from {repo}/{file} via PLAKAT_ARCFACE_HF")
            })?;
        s.finish_with_message(format!("✓ ArcFace cached at {}", path.display()));
        return Ok(path);
    }
    bail!("{}", arcface_setup_message())
}

/// Parse a `repo#file` HF spec used by `PLAKAT_*_HF` env vars.
pub(crate) fn parse_hf_spec(s: &str, var_name: &str) -> Result<(String, String)> {
    let Some((repo, file)) = s.split_once('#') else {
        bail!(
            "{var_name} must be `repo#file` (got {s:?}, no `#` separator). \
             Example: huggingface_user/insightface_models#arcface_r50.safetensors"
        );
    };
    if repo.is_empty() || file.is_empty() {
        bail!(
            "{var_name} must be `repo#file` with both sides non-empty (got {s:?})"
        );
    }
    Ok((repo.to_string(), file.to_string()))
}
