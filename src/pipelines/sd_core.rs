//! Shared SD-family backbone — the UNet, VAE, text encoder(s), and
//! associated config that every SD-based plakat pipeline (`t2i`,
//! `portrait`, `stylize`, `img2img`, `artefact_blend`) needs.
//!
//! Status: phase 7a of the v0.10 shared-pipeline refactor. Defines
//! the [`SdCore`] type and an async [`load`](SdCore::load)
//! constructor that mirrors the SD-loading half of
//! `portrait::Pipeline::load`. **portrait.rs and t2i.rs do not use
//! this module yet** — that's phase 7b/7c.
//!
//! The duplication between this file and `portrait::Pipeline::load`
//! is intentional and temporary. Phase 7b will rewrite
//! `portrait::Pipeline::load` to delegate to `SdCore::load`, then
//! the duplication collapses.
//!
//! # What lives in `SdCore`
//!
//! Everything that's identical across SD-based pipelines:
//!
//! * `variant` — SD 1.5 vs SDXL (detected from the model id).
//! * `cfg` — candle's `StableDiffusionConfig` for the variant.
//! * `tokenizer_l` + `text_encoder_l` — CLIP-L (used by both SD 1.5 and SDXL).
//! * `tokenizer_g` + `text_encoder_g` — CLIP-G (SDXL only; `None` for SD 1.5).
//! * `vae` — AutoEncoder-KL for VAE encode/decode.
//! * `unet` — the noise-prediction UNet (with any user LoRAs merged in).
//! * `device`, `dtype` — F16 on GPU, F32 on CPU.
//! * `_lora_tmp` — temp-file handles keeping merged LoRA mmaps alive.
//!
//! # What does **not** live here
//!
//! Task-specific add-ons stay with the pipeline that owns them:
//!
//! * `portrait::Pipeline.identity_encoder` (CLIP-H / FaceID for IP-Adapter)
//! * `t2i::Pipeline.refiner_unet` (SDXL refiner)
//! * `stylize::Pipeline.image_encoder` (CLIP-H for style transfer)
//!
//! Those modules will hold an `Arc<SdCore>` plus their own
//! task-specific fields after phases 7b/7c.

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::{
    StableDiffusionConfig,
    vae::AutoEncoderKL,
};
use tokenizers::Tokenizer;

use crate::pipelines::lora::ResolvedLora;
use crate::pipelines::sdxl_clip::SdxlClipGTextTransformer;
use crate::pipelines::sdxl_unet::{SdUNet, SdxlAddEmbedConfig, SdxlUNet2DConditionModel};
use crate::ui::progress;

/// SD variant the backbone routes through. Detected from the model
/// alias / repo at load time. Covers every SD-family architecture
/// plakat supports (SD 1.5, SD 2.1, SDXL — SDXL-Turbo is
/// architecturally identical to SDXL and uses the `Sdxl` variant;
/// only the caller's scheduler defaults differ).
///
/// Flux is **not** supported by `SdCore`. Flux's pipeline has a
/// different architecture (transformer + T5, not UNet + CLIP) and
/// stays in `pipelines::flux`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdVariant {
    /// SD 1.5. CLIP-L only, `cross_attention_dim = 768`.
    Sd15,
    /// SD 2.1. OpenCLIP-H, `cross_attention_dim = 1024`,
    /// `use_linear_projection = true`. Architecturally distinct
    /// from SD 1.5.
    Sd21,
    /// SDXL (and SDXL-Turbo). Dual CLIP-L + CLIP-G,
    /// `cross_attention_dim = 2048`, `use_linear_projection = true`.
    Sdxl,
}

impl SdVariant {
    /// Detect the variant from a model name / HF repo id.
    /// Priority: Flux markers raise an error to the caller (we
    /// can't return None from a `-> Self` function — t2i checks
    /// for Flux separately before calling SdCore::load).
    pub fn detect(model: &str) -> Self {
        let m = model.to_lowercase();
        // SDXL Turbo / SDXL / SDXL-Inpaint → Sdxl (same architecture
        // apart from the inpaint UNet's conv_in channel count, which
        // SdCore handles via the `is_inpaint` flag).
        if m.contains("xl") {
            return Self::Sdxl;
        }
        // SD 2.1: explicit "2-1" / "2.1" / "v2" markers.
        if m.contains("2-1") || m.contains("2.1") || m.contains("v2") {
            return Self::Sd21;
        }
        Self::Sd15
    }

    pub fn cross_attn_dim(self) -> usize {
        match self {
            Self::Sd15 => 768,
            Self::Sd21 => 1024,
            Self::Sdxl => 2048,
        }
    }

    /// v0.14 phase 4: `true` for the SDXL family (which carries the
    /// `add_embedding` micro-conditioning input). SD 1.5 / SD 2.1
    /// don't, so callers gate `add_text_embeds` / `add_time_ids` on
    /// this predicate.
    pub fn is_xl(self) -> bool {
        matches!(self, Self::Sdxl)
    }

    pub fn vae_scale(self) -> f64 {
        match self {
            // SD 1.5 and SD 2.1 share the same VAE scaling factor.
            Self::Sd15 | Self::Sd21 => 0.18215,
            Self::Sdxl => 0.13025,
        }
    }

    pub fn config(self, w: usize, h: usize) -> StableDiffusionConfig {
        match self {
            Self::Sd15 => StableDiffusionConfig::v1_5(None, Some(h), Some(w)),
            Self::Sd21 => StableDiffusionConfig::v2_1(None, Some(h), Some(w)),
            Self::Sdxl => {
                let mut c = StableDiffusionConfig::sdxl(None, Some(h), Some(w));
                // Candle defaults SDXL CLIP-L (`text_encoder`) padding to "!"
                // (id 0), but diffusers pads `text_encoder` with `<|endoftext|>`
                // (id 49407) — only `tokenizer_2` (CLIP-G, `clip2`) pads with
                // "!". `tokenize_padded` reads THIS config's `pad_with`, so the
                // mispadding lived here: SDXL `clip.encoded` padding rows carried
                // id-0 embeddings, dropping the golden correspondence to corr
                // ~0.991. Caught by `plakat verify --tier 1`; leave clip2 as-is.
                c.clip.pad_with = None;
                c
            }
        }
    }
}

/// Load-time inputs for the SD core. Identity / refiner / image
/// encoder / etc. live on the task-specific wrappers, not here.
///
/// `loras` is already resolved (paths + scales + display names);
/// the caller decides where they came from. portrait::Pipeline
/// uses this to inject FaceID's auto-LoRA before resolution so
/// the SD core merges it transparently.
pub struct SdLoadRequest {
    pub model: String,
    pub device: Device,
    pub loras: Vec<ResolvedLora>,
    pub lora_scale: f32,
    /// v0.16 phase 9: zero or more Textual Inversion embeddings to
    /// register at load time. Each is parsed via
    /// [`crate::pipelines::embedding::parse_safetensors`] and merged
    /// into the CLIP-L text encoder weights via a tempfile (same
    /// pattern LoRA uses). The tokenizer is mutated to add the new
    /// trigger token IDs. SD 1.5 / SD 2.1 only — SDXL dual-encoder
    /// TIs bail loud in the parser.
    pub embeddings: Vec<crate::pipelines::embedding::ResolvedEmbedding>,
    /// v0.32 phase 2: optional pre-built VAE. When `Some`, the load
    /// path uses it directly instead of materializing a fresh
    /// `AutoEncoderKL` from the safetensors. Used by the scenario
    /// runner's VAE cache to skip the ~330 MB SDXL VAE rebuild on
    /// mixed-kind pipeline reloads.
    pub vae_cache: Option<std::sync::Arc<AutoEncoderKL>>,
}

/// The shared SD backbone. Held behind `Arc` by every task-specific
/// pipeline that consumes it — letting `plakat generate
/// --artefact-blend` load weights once and reuse them across the
/// base generation pass + the blend pass.
///
/// Fields are public-in-crate so the wrapping pipelines (in
/// phases 7b/7c) can call methods directly on the underlying
/// candle objects rather than going through accessor methods.
pub struct SdCore {
    pub variant: SdVariant,
    pub cfg: StableDiffusionConfig,
    pub tokenizer_l: Tokenizer,
    /// SDXL only — the CLIP-G tokenizer. `None` for SD 1.5.
    pub tokenizer_g: Option<Tokenizer>,
    pub text_encoder_l: crate::pipelines::vendored_clip::ClipTextTransformer,
    /// SDXL only — CLIP-G (text_encoder_2) wrapped with the v0.11
    /// `text_projection` pooling head needed by the UNet's
    /// `add_embedding`. `None` for SD 1.5 / SD 2.1.
    pub text_encoder_g: Option<SdxlClipGTextTransformer>,
    /// v0.32 phase 2: wrapped in `Arc` so the scenario runner can
    /// share one VAE across mixed-kind pipeline reloads (t2i ↔
    /// animate). Auto-deref keeps every `.vae.encode(...)` /
    /// `.vae.decode(...)` call site unchanged.
    pub vae: std::sync::Arc<AutoEncoderKL>,
    /// Backbone UNet. `SdUNet::Sd` for SD 1.5 / SD 2.1 (candle's
    /// upstream type); `SdUNet::Sdxl` for SDXL (v0.11 phase 8 — adds
    /// `text_time` micro-conditioning that diffusers' SDXL relies on
    /// for full-quality outputs).
    pub unet: SdUNet,
    /// v0.12: 9-channel inpainting UNet (
    /// `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` for SDXL,
    /// `stable-diffusion-v1-5/stable-diffusion-inpainting` for SD 1.5,
    /// or any SD 2.x inpainting mirror that follows the same naming).
    /// When set, the loaded UNet expects a 9-channel input
    /// `[noisy_latents(4), mask(1), masked_image_latents(4)]`. The
    /// img2img/portrait masked paths skip RePaint-style mask blending
    /// and instead concat the mask + masked-image latents along the
    /// channel dim at every denoise step. `false` for everything else
    /// (regular 4-channel UNets — RePaint blending stays in play).
    pub is_inpaint: bool,
    pub device: Device,
    pub dtype: DType,
    /// Kept alive so merged-LoRA safetensors mmaps stay valid for
    /// the core's lifetime. Don't drop unless you also drop every
    /// pipeline holding an `Arc<SdCore>`.
    pub _lora_tmp: Vec<tempfile::NamedTempFile>,
}

/// v0.12: does this resolved repo id name a 9-channel inpainting
/// UNet? Determines `SdCore.is_inpaint`. The check is intentionally
/// loose — any SD-architecture repo whose id contains "inpaint" /
/// "inpainting" gets the 9-channel UNet build path. Covers stock
/// SDXL-Inpaint, SD 1.5 inpaint mirrors, and community SD 2.x
/// inpainting checkpoints that follow the same naming.
pub fn detect_inpaint(base_repo: &str) -> bool {
    let m = base_repo.to_lowercase();
    m.contains("inpaint") || m.contains("inpainting")
}

impl SdCore {
    /// Resolve the model id, download weights, build the UNet / VAE
    /// / text encoder(s), and merge any user-supplied LoRAs.
    ///
    /// Flux models are rejected here (SdCore is SD-architecture only).
    /// Task-specific load (identity encoder, refiner, etc.) happens
    /// on the wrapping pipeline that owns the `Arc<SdCore>`.
    pub async fn load(mut req: SdLoadRequest) -> Result<Self> {
        // Load-path profiler (Phase 1 of the 2.4 perf pass). Env-gated (dormant otherwise):
        // `PLAKAT_PROFILE_LOAD=1` prints per-phase deltas to stderr, so we can see whether the
        // cold-load cost is download-resolve, weight read+build (which module), or construct.
        struct LoadProf {
            on: bool,
            start: std::time::Instant,
            last: std::cell::Cell<std::time::Instant>,
        }
        impl LoadProf {
            fn new() -> Self {
                let now = std::time::Instant::now();
                LoadProf {
                    on: std::env::var("PLAKAT_PROFILE_LOAD").is_ok(),
                    start: now,
                    last: std::cell::Cell::new(now),
                }
            }
            fn mark(&self, name: &str) {
                if !self.on {
                    return;
                }
                let now = std::time::Instant::now();
                let d = now.duration_since(self.last.get()).as_secs_f64() * 1e3;
                let t = now.duration_since(self.start).as_secs_f64() * 1e3;
                eprintln!("[load-prof] {name:<16} +{d:>8.1} ms   (total {t:>8.1} ms)");
                self.last.set(now);
            }
        }
        let prof = LoadProf::new();
        let base_repo = if req.model.contains('/') {
            req.model.clone()
        } else {
            crate::hf::resolve_alias(&req.model).to_string()
        };
        let lc = base_repo.to_lowercase();
        if lc.contains("flux") {
            bail!(
                "SdCore does not support Flux (different architecture). \
                 Use --model sd15 (default), sd21, sdxl, sdxl-turbo, or any \
                 SD-family HF repo. Flux routes through pipelines::flux."
            );
        }
        // SdCore is the SD-UNet backbone (SD 1.5 / 2.1 / SDXL). The DiT/MMDiT and
        // Cascade families are entirely different architectures with their own
        // pipelines — without this guard their repos fall through `SdVariant::detect`
        // to the SD 1.5 default and 404 on the non-existent `unet/` path (e.g. SD3.5
        // ships a `transformer/`, not a `unet/`). Portrait / FaceID and the other
        // SdCore-based features are therefore SD-UNet-only.
        let model_lc = req.model.to_lowercase();
        for (needle, fam) in [
            ("sd3", "SD3 / SD3.5"),
            ("stable-diffusion-3", "SD3 / SD3.5"),
            ("pixart", "PixArt-Σ"),
            ("cascade", "Stable Cascade"),
            ("würstchen", "Stable Cascade"),
            ("wuerstchen", "Stable Cascade"),
        ] {
            if lc.contains(needle) || model_lc.contains(needle) {
                bail!(
                    "SdCore (the SD 1.5 / 2.1 / SDXL UNet backbone) does not support \
                     {fam} — it's a different architecture with its own pipeline, and \
                     SdCore-based features (portrait / FaceID, etc.) are SD-UNet-only. \
                     For plain generation, `plakat generate --model {}` routes {fam} to \
                     the correct pipeline.",
                    req.model
                );
            }
        }
        let variant = SdVariant::detect(&base_repo);
        let is_inpaint = detect_inpaint(&base_repo);
        let cfg = variant.config(512, 512);
        // v0.30 phase 0: Textual Inversion runtime injection.
        // The parser + merger from v0.16 phase 9 produce an extended
        // token_embedding matrix (tempfile) + a MergeReport with the
        // new vocab size and per-embedding token registration. The
        // load path below builds CLIP-L via the vendored CLIP module
        // with a `Config::with_vocab(new_vocab_size)` override, then
        // adds the trigger tokens to the tokenizer so user prompts
        // resolve them to the new IDs.
        //
        // SDXL CLIP-G is not yet TI-extended — the parser still
        // rejects SDXL dual-encoder TIs. Full SDXL dual TI lands
        // alongside a parser extension (stretch goal this phase,
        // otherwise phase 1 of v0.31).
        let dtype = if matches!(req.device, Device::Cpu) {
            DType::F32
        } else {
            DType::F16
        };

        // -------- download base weights (variant-aware) --------
        let dl = progress::spinner(&format!(
            "Resolving {} weights",
            match variant {
                SdVariant::Sd15 => "SD 1.5",
                SdVariant::Sd21 => "SD 2.1",
                SdVariant::Sdxl => "SDXL",
            }
        ));
        let tokenizer_l_path = crate::hf::download::get_first_of(&[
            (&base_repo, "tokenizer/tokenizer.json"),
            ("openai/clip-vit-large-patch14", "tokenizer.json"),
        ])
        .await
        .with_context(|| format!("tokenizer (CLIP-L) for {base_repo}"))?;
        let text_enc_l_path = crate::hf::download::get_first_of(&[
            (&base_repo, "text_encoder/model.fp16.safetensors"),
            (&base_repo, "text_encoder/model.safetensors"),
        ])
        .await?;
        let (tokenizer_g_path, text_enc_g_path) = match variant {
            // SD 1.5 + SD 2.1 each have a single text encoder
            // (no CLIP-G dual encoder).
            SdVariant::Sd15 | SdVariant::Sd21 => (None, None),
            SdVariant::Sdxl => {
                let t = crate::hf::download::get_first_of(&[
                    (&base_repo, "tokenizer_2/tokenizer.json"),
                    ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json"),
                    ("openai/clip-vit-large-patch14", "tokenizer.json"),
                ])
                .await
                .with_context(|| format!("tokenizer (CLIP-G) for {base_repo}"))?;
                let e = crate::hf::download::get_first_of(&[
                    (&base_repo, "text_encoder_2/model.fp16.safetensors"),
                    (&base_repo, "text_encoder_2/model.safetensors"),
                ])
                .await
                .with_context(|| format!("text_encoder_2 in {base_repo}"))?;
                (Some(t), Some(e))
            }
        };
        let unet_path = crate::hf::download::get_first_of(&[
            (&base_repo, "unet/diffusion_pytorch_model.fp16.safetensors"),
            (&base_repo, "unet/diffusion_pytorch_model.safetensors"),
        ])
        .await?;
        // v0.43: SDXL's stock VAE overflows F16's ~65k ceiling in its
        // decoder → NaN → all-black on half-precision backends (Metal).
        // On a non-CPU (F16) SDXL run, swap in madebyollin's
        // `sdxl-vae-fp16-fix` — a retrained drop-in VAE (identical
        // architecture, same config) that is numerically stable in F16.
        // This keeps the VAE in F16 (no OOM, no extra memory, no tiling)
        // while producing correct output. CPU runs F32, where the stock
        // VAE is fine, so it's left untouched there. SD 1.5 / 2.1 VAEs
        // tolerate F16, so only SDXL is redirected.
        let use_fp16_vae_fix = variant.is_xl() && !matches!(req.device, Device::Cpu);
        let vae_path = if use_fp16_vae_fix {
            const VAE_FIX_REPO: &str = "madebyollin/sdxl-vae-fp16-fix";
            crate::hf::download::get_first_of(&[
                (VAE_FIX_REPO, "diffusion_pytorch_model.safetensors"),
                (VAE_FIX_REPO, "sdxl_vae.safetensors"),
                (VAE_FIX_REPO, "sdxl.vae.safetensors"),
            ])
            .await
            .context(
                "downloading the SDXL fp16-fix VAE (madebyollin/sdxl-vae-fp16-fix); \
                 SDXL's stock VAE produces black images in F16",
            )?
        } else {
            crate::hf::download::get_first_of(&[
                (&base_repo, "vae/diffusion_pytorch_model.fp16.safetensors"),
                (&base_repo, "vae/diffusion_pytorch_model.safetensors"),
            ])
            .await?
        };
        dl.finish_with_message("✓ base weights ready");
        prof.mark("download-resolve");

        // LoRAs arrive pre-resolved. Temp-file handles for merged
        // weights are accumulated below.
        let mut lora_tmps: Vec<tempfile::NamedTempFile> = Vec::new();
        let resolved_loras = &req.loras;

        // -------- build models --------
        let build = progress::spinner("Loading SD core");
        let mut tokenizer_l = Tokenizer::from_file(&tokenizer_l_path)
            .map_err(|e| anyhow!("tokenizer (CLIP-L): {e}"))?;
        let mut tokenizer_g = match tokenizer_g_path.as_ref() {
            Some(p) => Some(
                Tokenizer::from_file(p).map_err(|e| anyhow!("tokenizer (CLIP-G): {e}"))?,
            ),
            None => None,
        };
        // v0.32 phase 2: VAE cache. When the scenario passes a pre-
        // built VAE in `req.vae_cache`, skip the ~330 MB load and
        // reuse the cached Arc. Otherwise build fresh and wrap.
        let vae = match req.vae_cache.take() {
            Some(arc) => {
                tracing::info!(
                    target: "plakat",
                    "SdCore: reusing cached VAE (skipping {} build)",
                    vae_path.display()
                );
                arc
            }
            None => std::sync::Arc::new(cfg.build_vae(&vae_path, &req.device, dtype)?),
        };

        // UNet (with optional LoRA merge).
        let effective_unet_path = if resolved_loras.is_empty() {
            unet_path.clone()
        } else {
            let spin = progress::spinner("Merging LoRA into UNet");
            let tmp = tempfile::Builder::new()
                .prefix("plakat-sd-unet-")
                .suffix(".safetensors")
                .tempfile()?;
            let (modified, targets) = crate::pipelines::lora::merge_loras_into_weights(
                &unet_path,
                tmp.path(),
                resolved_loras,
                req.lora_scale,
                &req.device,
                crate::pipelines::lora::MergeTarget::UNET,
            )?;
            spin.finish_with_message(format!(
                "✓ merged {modified}/{targets} UNet LoRA target(s)"
            ));
            let p = tmp.path().to_path_buf();
            lora_tmps.push(tmp);
            p
        };
        // v0.12: inpainting checkpoints (SD 1.5 / SD 2.1 / SDXL) carry
        // 9 input channels instead of 4 — same UNet architecture
        // otherwise, only `conv_in` changes shape.
        prof.mark("tokenizers+vae");
        let unet_in_channels = if is_inpaint { 9 } else { 4 };
        let unet = match variant {
            // SD 1.5 / SD 2.1 — candle's stock UNet (no add_embedding).
            SdVariant::Sd15 | SdVariant::Sd21 => SdUNet::Sd(
                cfg.build_unet(
                    &effective_unet_path,
                    &req.device,
                    unet_in_channels,
                    false,
                    dtype,
                )?,
            ),
            // SDXL — vendored UNet with `text_time` add_embedding.
            // Reuses controlnet::sdxl_unet_config() so the SDXL UNet
            // shape stays defined in one place. AddEmbedConfig::base()
            // = 6 time_ids; the refiner gets its own variant in 8e.
            SdVariant::Sdxl => {
                let vs_unet = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &[effective_unet_path.as_path()],
                        dtype,
                        &req.device,
                    )?
                };
                let sdxl_unet = SdxlUNet2DConditionModel::new(
                    vs_unet,
                    unet_in_channels,
                    4,
                    false,
                    crate::pipelines::controlnet::sdxl_unet_config(),
                    SdxlAddEmbedConfig::base(),
                )?;
                SdUNet::Sdxl(sdxl_unet)
            }
        };

        // CLIP-L text encoder (with optional LoRA merge).
        // SD 2.1 uses the same key naming as SD 1.5 for LoRA merge
        // targets (both have a single `text_encoder` module on disk).
        let te_l_target = match variant {
            SdVariant::Sd15 | SdVariant::Sd21 => crate::pipelines::lora::MergeTarget::TE_SD15,
            SdVariant::Sdxl => crate::pipelines::lora::MergeTarget::TE1_SDXL,
        };
        let effective_te_l_path = if resolved_loras.is_empty() {
            text_enc_l_path.clone()
        } else {
            let spin = progress::spinner(&format!("Merging LoRA into {}", te_l_target.name));
            let tmp = tempfile::Builder::new()
                .prefix("plakat-sd-te-l-")
                .suffix(".safetensors")
                .tempfile()?;
            let (modified, targets) = crate::pipelines::lora::merge_loras_into_weights(
                &text_enc_l_path,
                tmp.path(),
                resolved_loras,
                req.lora_scale,
                &req.device,
                te_l_target,
            )?;
            spin.finish_with_message(format!(
                "✓ merged {modified}/{targets} {} LoRA target(s)",
                te_l_target.name
            ));
            let p = tmp.path().to_path_buf();
            lora_tmps.push(tmp);
            p
        };
        // v0.30 phase 0: pick the matching vendored CLIP-L config.
        // Numerically identical to candle's `cfg.clip` for the variant,
        // but with a public `vocab_size` we can override for TI.
        prof.mark("unet");
        let base_clip_l_cfg = match variant {
            SdVariant::Sd15 => crate::pipelines::vendored_clip::Config::v1_5(),
            SdVariant::Sd21 => crate::pipelines::vendored_clip::Config::v2_1(),
            SdVariant::Sdxl => crate::pipelines::vendored_clip::Config::sdxl(),
        };

        // If the caller passed embeddings, merge their token vectors
        // into the (already LoRA-merged) CLIP-L safetensors via a
        // tempfile, then build the encoder with an extended-vocab
        // Config. Otherwise the CLIP-L path is exactly the v0.29
        // build (numerically identical via the vendored module).
        let (text_encoder_l, ti_registrations) = if req.embeddings.is_empty() {
            let enc = crate::pipelines::vendored_clip::build_clip_transformer(
                &base_clip_l_cfg,
                &effective_te_l_path,
                &req.device,
                dtype,
            )?;
            (enc, Vec::new())
        } else {
            let spin = progress::spinner(&format!(
                "Merging {} Textual Inversion embedding(s) into CLIP-L",
                req.embeddings.len()
            ));
            let tmp = tempfile::Builder::new()
                .prefix("plakat-sd-te-l-ti-")
                .suffix(".safetensors")
                .tempfile()?;
            let report = crate::pipelines::embedding::merge_embeddings_into_te_weights(
                &effective_te_l_path,
                tmp.path(),
                &req.embeddings,
                base_clip_l_cfg.embed_dim,
                &req.device,
                crate::pipelines::embedding::EmbeddingHalf::ClipL,
            )?;
            spin.finish_with_message(format!(
                "✓ TI extended CLIP-L vocab to {} (added {} token(s))",
                report.new_vocab_size,
                report
                    .registered
                    .iter()
                    .map(|r| r.num_tokens)
                    .sum::<usize>()
            ));
            let extended_cfg = base_clip_l_cfg.with_vocab(report.new_vocab_size);
            let extended_path = tmp.path().to_path_buf();
            // Keep the tempfile alive past the load (mmap stays valid
            // for the lifetime of the SdCore).
            lora_tmps.push(tmp);
            let enc = crate::pipelines::vendored_clip::build_clip_transformer(
                &extended_cfg,
                &extended_path,
                &req.device,
                dtype,
            )?;
            (enc, report.registered)
        };

        // v0.30 phase 0: mutate the CLIP-L tokenizer so user prompts
        // referencing a TI trigger word resolve to the new vocab IDs
        // we just appended. Each registered embedding contributes
        // `num_tokens` IDs (the trigger string itself + `<trigger>_1`,
        // `<trigger>_2`, ... for multi-vector TIs).
        if !ti_registrations.is_empty() {
            let mut added: Vec<tokenizers::AddedToken> = Vec::new();
            for reg in &ti_registrations {
                for tok_str in reg.token_strings() {
                    added.push(tokenizers::AddedToken::from(tok_str, false));
                }
            }
            let n = tokenizer_l.add_tokens(&added);
            tracing::info!(
                target: "plakat",
                "TI registered {} new tokenizer entries ({} embedding(s))",
                n,
                ti_registrations.len()
            );
        }

        // SDXL only: CLIP-G text encoder (with optional LoRA merge).
        let text_encoder_g = match variant {
            SdVariant::Sd15 | SdVariant::Sd21 => {
                // v0.31 phase 0: dual-encoder TI on a non-SDXL model
                // doesn't make sense — bail loud so the user gets a
                // pointer toward the CLIP-L-only variant.
                if req.embeddings.iter().any(|e| e.has_clip_g()) {
                    bail!(
                        "Textual Inversion: dual-encoder TI (clip_l + clip_g \
                         in the same file) requires SDXL — got {variant:?}. \
                         Either pick an SDXL model (`--model sdxl`) or use a \
                         CLIP-L-only TI."
                    );
                }
                None
            }
            SdVariant::Sdxl => {
                // v0.30 phase 0: vendored CLIP Config for SDXL CLIP-G.
                // Numerically identical to candle's `clip2`.
                let _ = cfg.clip2.as_ref(); // keep the StableDiffusionConfig
                let base_cfg_g = crate::pipelines::vendored_clip::Config::sdxl2();
                let p = text_enc_g_path
                    .as_ref()
                    .ok_or_else(|| anyhow!("missing text_encoder_2 path"))?;
                let effective_te_g_path = if resolved_loras.is_empty() {
                    p.clone()
                } else {
                    let target = crate::pipelines::lora::MergeTarget::TE2_SDXL;
                    let spin = progress::spinner(&format!("Merging LoRA into {}", target.name));
                    let tmp = tempfile::Builder::new()
                        .prefix("plakat-sd-te-g-")
                        .suffix(".safetensors")
                        .tempfile()?;
                    let (modified, targets) = crate::pipelines::lora::merge_loras_into_weights(
                        p,
                        tmp.path(),
                        resolved_loras,
                        req.lora_scale,
                        &req.device,
                        target,
                    )?;
                    spin.finish_with_message(format!(
                        "✓ merged {modified}/{targets} {} LoRA target(s)",
                        target.name
                    ));
                    let path = tmp.path().to_path_buf();
                    lora_tmps.push(tmp);
                    path
                };

                // v0.31 phase 0: dual-encoder TI extension for CLIP-G.
                // Mirrors the CLIP-L pattern above. Only fires when at
                // least one TI in the stack carries a clip_g half.
                let dual_count = req.embeddings.iter().filter(|e| e.has_clip_g()).count();
                let (effective_te_g_path, cfg_g, dual_registrations) = if dual_count == 0 {
                    (effective_te_g_path, base_cfg_g, Vec::new())
                } else {
                    let spin = progress::spinner(&format!(
                        "Merging {dual_count} dual-encoder TI clip_g half(s) into SDXL CLIP-G"
                    ));
                    let tmp = tempfile::Builder::new()
                        .prefix("plakat-sd-te-g-ti-")
                        .suffix(".safetensors")
                        .tempfile()?;
                    let report = crate::pipelines::embedding::merge_embeddings_into_te_weights(
                        &effective_te_g_path,
                        tmp.path(),
                        &req.embeddings,
                        base_cfg_g.embed_dim,
                        &req.device,
                        crate::pipelines::embedding::EmbeddingHalf::ClipG,
                    )?;
                    spin.finish_with_message(format!(
                        "✓ TI extended CLIP-G vocab to {} (added {} token(s))",
                        report.new_vocab_size,
                        report
                            .registered
                            .iter()
                            .map(|r| r.num_tokens)
                            .sum::<usize>()
                    ));
                    let extended_cfg = base_cfg_g.with_vocab(report.new_vocab_size);
                    let extended_path = tmp.path().to_path_buf();
                    lora_tmps.push(tmp);
                    (extended_path, extended_cfg, report.registered)
                };

                // v0.11 phase 8b: load via the SdxlClipGTextTransformer
                // wrapper so the `text_projection` Linear is also
                // pulled out of the safetensors. embed_dim = 1280 is
                // the stock SDXL CLIP-G width.
                let vs_g = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &[effective_te_g_path.as_path()],
                        dtype,
                        &req.device,
                    )?
                };
                let wrapper = SdxlClipGTextTransformer::new(vs_g, &cfg_g, 1280)?;

                // v0.31 phase 0: register the dual TI triggers in
                // tokenizer_g so prompts referencing the trigger
                // resolve to the new CLIP-G vocab IDs. Mirror of the
                // tokenizer_l mutation above.
                if !dual_registrations.is_empty() {
                    let tok_g = tokenizer_g
                        .as_mut()
                        .ok_or_else(|| anyhow!("SDXL missing tokenizer_g for TI registration"))?;
                    let mut added: Vec<tokenizers::AddedToken> = Vec::new();
                    for reg in &dual_registrations {
                        for tok_str in reg.token_strings() {
                            added.push(tokenizers::AddedToken::from(tok_str, false));
                        }
                    }
                    let n = tok_g.add_tokens(&added);
                    tracing::info!(
                        target: "plakat",
                        "TI registered {} new CLIP-G tokenizer entries ({} dual embedding(s))",
                        n,
                        dual_registrations.len()
                    );
                }
                Some(wrapper)
            }
        };

        build.finish_with_message("✓ SD core loaded");

        prof.mark("text-enc+rest");
        Ok(Self {
            variant,
            cfg,
            tokenizer_l,
            tokenizer_g,
            text_encoder_l,
            text_encoder_g,
            vae,
            unet,
            is_inpaint,
            device: req.device,
            dtype,
            _lora_tmp: lora_tmps,
        })
    }

    /// Device the SD core lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// dtype the SD core's tensors live at (F16 on accelerator,
    /// F32 on CPU).
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn variant(&self) -> SdVariant {
        self.variant
    }

    pub fn cfg(&self) -> &StableDiffusionConfig {
        &self.cfg
    }

    /// v0.15 phase 7b-6: scenario per-task LoRA dispatch surface for
    /// the SD-family backbone. Delegates to `SdUNet::apply_loras`,
    /// which bails loud for SD-family (the UNet's Linears aren't yet
    /// wrapped as `LoraLinear` — full vendor deferred). Provided for
    /// API uniformity with the Flux / SD3 backbones; the scenario
    /// dispatcher (7b-7) can call this for any variant.
    pub fn apply_loras(
        &self,
        specs: std::collections::HashMap<
            String,
            Vec<crate::pipelines::lora_linear::LoraSpec>,
        >,
    ) -> Result<usize> {
        // sd_core uses anyhow::Result; SdUNet returns candle_core::Result.
        self.unet.apply_loras(specs).map_err(anyhow::Error::from)
    }

    /// v0.15 phase 7b-6: no-op for SD-family (no runtime stack
    /// exists). Mirrors the Flux / SD3 API.
    pub fn clear_all_loras(&self) -> Result<()> {
        self.unet.clear_all_loras().map_err(anyhow::Error::from)?;
        Ok(())
    }

    /// v0.15 phase 7b-6: zero for SD-family. Mirrors the Flux / SD3
    /// API so the dispatcher can query without backbone-specific
    /// branches.
    pub fn n_registered_linears(&self) -> usize {
        self.unet.n_registered_linears()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sd_variant_detect() {
        assert_eq!(SdVariant::detect("sd15"), SdVariant::Sd15);
        assert_eq!(
            SdVariant::detect("stable-diffusion-v1-5/stable-diffusion-v1-5"),
            SdVariant::Sd15,
        );
        assert_eq!(SdVariant::detect("sdxl"), SdVariant::Sdxl);
        assert_eq!(SdVariant::detect("SDXL-turbo"), SdVariant::Sdxl);
    }

    #[test]
    fn sd_variant_cross_attn_dim() {
        assert_eq!(SdVariant::Sd15.cross_attn_dim(), 768);
        assert_eq!(SdVariant::Sdxl.cross_attn_dim(), 2048);
    }

    #[test]
    fn sd_variant_vae_scale_matches_sd_constants() {
        // diffusers' SD 1.5 and SDXL VAE scaling factors (verified
        // against the upstream config.json's `scaling_factor`).
        assert!((SdVariant::Sd15.vae_scale() - 0.18215).abs() < 1e-6);
        assert!((SdVariant::Sdxl.vae_scale() - 0.13025).abs() < 1e-6);
    }
}
