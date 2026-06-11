//! Style-reference pipeline: IN + REF → OUT.
//!
//! Architecture:
//!   1. VAE-encode IN → latents.
//!   2. CLIP-H image-encode REF → image_embeds (1024-d). Project via
//!      IP-Adapter `image_proj` → 4 image tokens (768-d each).
//!   3. Concat empty-text tokens (77) with image tokens (4) → (1, 81, 768)
//!      encoder_hidden_states.
//!   4. Img2img denoise: add noise to IN-latents at strength·T, run the
//!      denoising loop from that timestep with the conditioning above.
//!   5. VAE-decode → OUT.
//!
//! Currently SD 1.5 only. SDXL IP-Adapter (different image encoder dims,
//! different projection target) is a follow-up.
//!
//! Two ways to use this module:
//!   * `stylize::run(Request)` — single-shot. `plakat stylize` uses this.
//!   * `Pipeline::load(...)` + repeated `Pipeline::stylize_one(...)` —
//!     share loaded weights (notably the 2.5 GB CLIP-H image encoder)
//!     across many calls. `plakat scenario` uses this when tasks declare
//!     a `style` reference.

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_transformers::models::stable_diffusion::clip as sdclip;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::pipelines::ip_adapter::{ImageEncoder, ImageProj};
use crate::ui::progress;

// =====================================================================
// Single-shot request type — back-compat with the CLI subcommand.
// =====================================================================

pub struct Request {
    pub input: PathBuf,
    pub reference: PathBuf,
    pub out: PathBuf,
    pub strength: f32,
    pub model: String,
    pub steps: usize,
    pub seed: Option<u64>,
    pub ref_blur: f32,
    pub ref_weight: f32,
    /// InstantStyle: true painterly style transfer via decoupled IP injection on
    /// the SDXL style block (vs the content/palette concat path). SDXL-only.
    pub instantstyle: bool,
    /// InstantStyle injection strength (the style-block IP scale).
    pub style_scale: f32,
    pub device: Device,
}

const IPA_REPO: &str = "h94/IP-Adapter";
const SD15_CROSS_ATTN_DIM: usize = 768;
const SDXL_CROSS_ATTN_DIM: usize = 2048;
const IPA_TOKENS: usize = 4;
const CLIP_H_PROJ_DIM: usize = 1024;
const CLIP_H_INPUT: u32 = 224;

// =====================================================================
// Pipeline: load once, stylize many.
// =====================================================================

pub struct LoadRequest {
    pub model: String,
    pub device: Device,
    /// Phase 7f. Optional pre-loaded CLIP-H image encoder to share
    /// with `portrait::Pipeline`'s identity encoder. `None` causes
    /// stylize to download + load CLIP-H itself (pre-7f behaviour).
    pub shared_clip_h: Option<std::sync::Arc<ImageEncoder>>,
    /// InstantStyle (SDXL only): load the vendored UNet + install the style-block
    /// IP injection at `style_scale`. `false` keeps the concat ref-variation path.
    pub instantstyle: bool,
    pub style_scale: f32,
}

pub struct GenRequest {
    pub input: PathBuf,
    pub reference: PathBuf,
    pub out: PathBuf,
    pub strength: f32,
    pub steps: usize,
    pub seed: Option<u64>,
    /// Gaussian-blur the reference before CLIP-encoding it (sigma; 0 = off).
    /// Blurring wipes the ref's fine content (the subject/face) while keeping
    /// its broad style — palette, texture, composition — so the transfer is
    /// *style*, not subject. The cheap "style not content" knob.
    pub ref_blur: f32,
    /// Scale the reference's image-token contribution (1.0 = full). Lower lets
    /// the prompt own the subject while the ref owns the look.
    pub ref_weight: f32,
}

/// Blur the reference to a temp PNG when `sigma > 0` (the style-not-content
/// heuristic), else return the original path. Normalises the short side to
/// 512 first so `sigma` means the same thing at any reference resolution.
fn maybe_blur_ref(path: &std::path::Path, sigma: f32) -> Result<std::path::PathBuf> {
    if sigma <= 0.0 {
        return Ok(path.to_path_buf());
    }
    let img = image::open(path)
        .with_context(|| format!("opening reference {} for blur", path.display()))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    let (rw, rh) = if w < h {
        (512, ((h as f32) * 512.0 / (w as f32)).round() as u32)
    } else {
        (((w as f32) * 512.0 / (h as f32)).round() as u32, 512)
    };
    let resized = image::imageops::resize(&img, rw, rh, image::imageops::FilterType::Triangle);
    let blurred = image::imageops::blur(&resized, sigma);
    let tmp = std::env::temp_dir().join(format!("plakat-stylize-ref-{}.png", std::process::id()));
    blurred
        .save(&tmp)
        .with_context(|| format!("writing blurred reference {}", tmp.display()))?;
    Ok(tmp)
}

/// Encode the empty prompt → (hidden, pooled). SD 1.5: (1,77,768) + None.
/// SDXL: (1,77,2048) (CLIP-L ⊕ CLIP-G penultimate hidden) + Some((1,1280))
/// pooled CLIP-G for the UNet's add_embedding. Mirrors portrait's encode_text.
fn encode_empty_text(
    core: &crate::pipelines::sd_core::SdCore,
) -> Result<(Tensor, Option<Tensor>)> {
    if core.variant.is_xl() {
        let cfg_g = core
            .cfg
            .clip2
            .as_ref()
            .ok_or_else(|| anyhow!("SDXL stylize missing clip2 config"))?;
        let tok_g = core
            .tokenizer_g
            .as_ref()
            .ok_or_else(|| anyhow!("SDXL stylize missing tokenizer_g"))?;
        let enc_g = core
            .text_encoder_g
            .as_ref()
            .ok_or_else(|| anyhow!("SDXL stylize missing text_encoder_g"))?;
        let ids_l = tokenize_padded(&core.tokenizer_l, &core.cfg.clip, "", &core.device)?;
        let ids_g = tokenize_padded(tok_g, cfg_g, "", &core.device)?;
        let (_final_l, hidden_l) = core
            .text_encoder_l
            .forward_until_encoder_layer(&ids_l, usize::MAX, -2)?;
        let (hidden_g, pooled_g) = enc_g.forward_for_sdxl(&ids_g)?;
        let hidden = Tensor::cat(&[&hidden_l, &hidden_g], 2)?.to_dtype(core.dtype)?;
        Ok((hidden, Some(pooled_g.to_dtype(core.dtype)?)))
    } else {
        let ids = tokenize_padded(&core.tokenizer_l, &core.cfg.clip, "", &core.device)?;
        let hidden = core.text_encoder_l.forward(&ids)?.to_dtype(core.dtype)?;
        Ok((hidden, None))
    }
}

/// Tokenize `text` padded to the config's max length (mirrors portrait's
/// helper — trivial duplication preferred over a shared module).
fn tokenize_padded(
    tokenizer: &Tokenizer,
    cfg: &sdclip::Config,
    text: &str,
    device: &Device,
) -> Result<Tensor> {
    let pad_id: u32 = match &cfg.pad_with {
        Some(s) => tokenizer
            .token_to_id(s)
            .ok_or_else(|| anyhow!("tokenizer missing pad token {s:?}"))?,
        None => tokenizer
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| anyhow!("tokenizer missing <|endoftext|>"))?,
    };
    let mut ids = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    ids.resize(cfg.max_position_embeddings, pad_id);
    Ok(Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?)
}

pub struct Pipeline {
    /// The SD backbone (UNet / VAE / CLIP-L [+ CLIP-G for SDXL] /
    /// scheduler config), variant-dispatched. Delegating to `SdCore`
    /// gives stylize SDXL for free — the dual encoders, the SDXL UNet
    /// `add_embedding`, and the F16 SDXL-VAE black-image fix — exactly
    /// as portrait does. `Arc` so it can be shared.
    core: std::sync::Arc<crate::pipelines::sd_core::SdCore>,
    /// Phase 7f: `Arc` so the same CLIP-H weights can back both this
    /// pipeline and portrait's identity encoder when both run in one
    /// process.
    image_encoder: std::sync::Arc<ImageEncoder>,
    /// IP-Adapter projection — `cross_attn_dim` 768 (SD 1.5) or 2048
    /// (SDXL), loaded from the matching adapter file.
    image_proj: ImageProj,
    /// Pre-computed empty-prompt text embeddings — (1, 77, 768) for
    /// SD 1.5, (1, 77, 2048) for SDXL. Constant per pipeline.
    empty_text_embeds: Tensor,
    /// SDXL only — the pooled CLIP-G empty-prompt embedding (1, 1280)
    /// for the UNet's `add_embedding`. `None` for SD 1.5.
    empty_pooled: Option<Tensor>,
    /// InstantStyle: the vendored UNet (style-block IP injection installed) +
    /// the shared style-token cell it reads. `None` unless `--instantstyle`.
    instant: Option<InstantCtx>,
}

/// InstantStyle context: the vendored SD UNet with a decoupled IP cross-attention
/// installed on the style block, plus the shared style-token cell it reads.
struct InstantCtx {
    unet: crate::pipelines::sd_train::unet::UNet2DConditionModel,
    tokens: std::sync::Arc<std::sync::RwLock<Option<Tensor>>>,
}

impl Pipeline {
    /// Download + load SD 1.5 base + IP-Adapter (image encoder + projection).
    /// First run downloads ~2.5 GB of CLIP-H weights plus SD 1.5 base.
    pub async fn load(req: LoadRequest) -> Result<Self> {
        let base_repo = if req.model.contains('/') {
            req.model.clone()
        } else {
            crate::hf::resolve_alias(&req.model).to_string()
        };
        if base_repo.to_lowercase().contains("flux") {
            bail!(
                "stylize supports SD 1.5 and SDXL only (not Flux). \
                 Use --model sd15 or --model sdxl."
            );
        }
        let variant = crate::pipelines::sd_core::SdVariant::detect(&base_repo);
        let (ipa_file, cross_attn_dim) = match variant {
            crate::pipelines::sd_core::SdVariant::Sd15 => {
                ("models/ip-adapter_sd15.safetensors", SD15_CROSS_ATTN_DIM)
            }
            crate::pipelines::sd_core::SdVariant::Sdxl => (
                "sdxl_models/ip-adapter_sdxl_vit-h.safetensors",
                SDXL_CROSS_ATTN_DIM,
            ),
            crate::pipelines::sd_core::SdVariant::Sd21 => bail!(
                "stylize has no IP-Adapter wired for SD 2.1; use --model sd15 or --model sdxl."
            ),
        };

        // -------- download the IP-Adapter weights --------
        let dl = progress::spinner("Downloading IP-Adapter weights");
        let ipa_weights = crate::hf::download::get_file(IPA_REPO, ipa_file).await?;
        // Skip the CLIP-H download when the caller supplied a shared encoder.
        let img_enc_weights = if req.shared_clip_h.is_none() {
            Some(
                crate::hf::download::get_file(IPA_REPO, "models/image_encoder/model.safetensors")
                    .await?,
            )
        } else {
            None
        };
        dl.finish_with_message("✓ IP-Adapter weights ready");

        // -------- load the SD backbone via SdCore (SDXL dual-CLIP + F16-VAE-fix) --------
        let build = progress::spinner("Loading stylize backbone");
        let core = std::sync::Arc::new(
            crate::pipelines::sd_core::SdCore::load(crate::pipelines::sd_core::SdLoadRequest {
                model: req.model.clone(),
                device: req.device.clone(),
                loras: vec![],
                lora_scale: 1.0,
                embeddings: vec![],
                vae_cache: None,
            })
            .await?,
        );
        let dtype = core.dtype;

        let image_encoder = match req.shared_clip_h {
            Some(shared) => shared,
            None => std::sync::Arc::new(ImageEncoder::load(
                img_enc_weights
                    .as_ref()
                    .expect("img_enc_weights set when shared_clip_h is None"),
                &req.device,
                dtype,
            )?),
        };
        let image_proj = ImageProj::load(
            &ipa_weights,
            CLIP_H_PROJ_DIM,
            cross_attn_dim,
            IPA_TOKENS,
            &req.device,
            dtype,
        )?;
        // Pre-compute empty-text embeddings (variant-aware) — constant per pipeline.
        let (empty_text_embeds, empty_pooled) = encode_empty_text(&core)?;

        // InstantStyle: load the vendored UNet and install the decoupled IP
        // cross-attention on the style block (SDXL `up_blocks.0.attentions.1`,
        // SD 1.5 `up_blocks.1.attentions.1`), so the style ref drives that block
        // only — true style transfer, not the content/palette of the concat path.
        let instant = if req.instantstyle {
            let is_xl = core.variant.is_xl();
            let tokens = std::sync::Arc::new(std::sync::RwLock::new(None));
            let mut unet = crate::pipelines::instantstyle::load_vendored_unet(
                &base_repo, is_xl, &req.device, dtype,
            )
            .await?;
            let ip_vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[ipa_weights.clone()],
                    dtype,
                    &req.device,
                )?
            };
            crate::pipelines::instantstyle::install_instantstyle(
                &mut unet,
                &ip_vb,
                req.style_scale as f64,
                tokens.clone(),
                is_xl,
            )?;
            Some(InstantCtx { unet, tokens })
        } else {
            None
        };
        build.finish_with_message("✓ stylize models loaded");

        Ok(Self {
            core,
            image_encoder,
            image_proj,
            empty_text_embeds,
            empty_pooled,
            instant,
        })
    }

    /// Apply one IN + REF → OUT stylization using the loaded models.
    pub fn stylize_one(&self, req: &GenRequest) -> Result<()> {
        let dtype = self.core.dtype;
        let device = &self.core.device;
        let vae_scale = self.core.variant.vae_scale();
        // Resolve output dims from IN (multiples of 8; SDXL native = 1024).
        let (in_w, in_h) = read_image_size(&req.input)?;
        let (w, h) = sd_dims(in_w, in_h, self.core.variant.is_xl());
        let strength = req.strength.clamp(0.0, 1.0);

        // -------- encode REF → image tokens --------
        let s = progress::spinner("Encoding reference image");
        // Cheap "style not content" heuristic: blur the reference first so
        // CLIP sees its broad style (palette/texture/composition), not the
        // fine content that otherwise hijacks the subject.
        let ref_for_clip = maybe_blur_ref(&req.reference, req.ref_blur)?;
        let ref_pixels = crate::imaging::preprocess::clip_image_tensor(
            &ref_for_clip,
            CLIP_H_INPUT,
            device,
            dtype,
        )?;
        let img_embeds = self.image_encoder.encode(&ref_pixels)?;
        let mut image_tokens = self.image_proj.forward(&img_embeds)?; // (1, 4, 768)
        if (req.ref_weight - 1.0).abs() > f32::EPSILON {
            image_tokens = (image_tokens * req.ref_weight as f64)?;
        }
        s.finish_with_message("✓ reference encoded");

        // Concat path: SD15 (1,77,768)⊕(1,4,768)→(1,81,768); SDXL ⊕(1,4,2048).
        // InstantStyle path: style rides the IP injection (style block), NOT the
        // cross-attn context — the UNet sees just the (empty) text and the style
        // tokens go into the shared cell the injection reads each step.
        let encoder_hidden_states = if let Some(ic) = &self.instant {
            *ic.tokens.write().unwrap() = Some(image_tokens.clone());
            self.empty_text_embeds.clone()
        } else {
            Tensor::cat(&[&self.empty_text_embeds, &image_tokens], 1)?
        };

        // SDXL micro-conditioning (target size). stylize runs no CFG → batch 1
        // (do NOT tile to 2 like the CFG portrait path).
        let add_time_ids = if self.core.variant.is_xl() {
            Some(crate::pipelines::sdxl_unet::build_add_time_ids_base(
                h, w, device, dtype,
            )?)
        } else {
            None
        };

        // v0.34 phase 1 fix: seed the device RNG BEFORE VAE encode.
        // `init_dist.sample()` below is RNG-touching — pre-v0.34
        // it used leftover state from prior ops, ignoring --seed.
        // Also: device-aware seed prep replaces the old `& u32::MAX`
        // mask. CPU/CUDA now get full u64 entropy; Metal high seeds
        // hash through SplitMix64 instead of colliding to low bits.
        let seed = req.seed.unwrap_or_else(rand::random);
        let prepared = crate::pipelines::seeds::prepare_seed(seed, device);
        if let Err(e) = device.set_seed(prepared) {
            tracing::debug!(target: "plakat", "set_seed not supported ({e}); using global RNG");
        }

        // -------- encode IN → latents --------
        let s = progress::spinner("Encoding input image");
        let in_pixels = crate::imaging::preprocess::sd_image_tensor(
            &req.input,
            w,
            h,
            device,
            dtype,
        )?;
        let init_dist = self.core.vae.encode(&in_pixels)?;
        let init_latents = (init_dist.sample()? * vae_scale)?;
        s.finish_with_message("✓ input encoded");

        // -------- img2img denoise --------

        let mut scheduler = self.core.cfg.build_scheduler(req.steps)?;
        let timesteps = scheduler.timesteps().to_vec();
        let init_skip = ((req.steps as f32) * (1.0 - strength)).round().max(0.0) as usize;
        let init_skip = init_skip.min(req.steps.saturating_sub(1));
        let active = &timesteps[init_skip..];
        let start_t = *active.first().ok_or_else(|| anyhow!("empty timestep list"))?;

        let noise = Tensor::randn(0f32, 1f32, init_latents.shape(), device)?
            .to_dtype(dtype)?;
        let mut latents = scheduler.add_noise(&init_latents, noise, start_t)?;

        let bar = progress::step_bar(active.len() as u64, "stylize");
        for &timestep in active {
            let latent_in = scheduler.scale_model_input(latents.clone(), timestep)?;
            let noise_pred = if let Some(ic) = &self.instant {
                // InstantStyle: the vendored UNet, with the style block injecting
                // the style ref via its decoupled IP cross-attention.
                if self.core.variant.is_xl() {
                    ic.unet.forward_sdxl(
                        &latent_in,
                        timestep as f64,
                        &encoder_hidden_states,
                        self.empty_pooled.as_ref().expect("SDXL pooled for instantstyle"),
                        add_time_ids.as_ref().expect("SDXL add_time_ids for instantstyle"),
                    )?
                } else {
                    ic.unet
                        .forward(&latent_in, timestep as f64, &encoder_hidden_states)?
                }
            } else {
                self.core.unet.forward(
                    &latent_in,
                    timestep as f64,
                    &encoder_hidden_states,
                    self.empty_pooled.as_ref(),
                    add_time_ids.as_ref(),
                )?
            };
            latents = scheduler.step(&noise_pred, timestep, &latents)?;
            bar.inc(1);
            bar.set_message(format!("t={timestep} strength={strength:.2}"));
        }
        bar.finish_and_clear();

        // -------- decode + save --------
        let image = self.core.vae.decode(&(&latents / vae_scale)?)?;
        let image = ((image / 2.0)? + 0.5)?.clamp(0f32, 1f32)?;
        let image = (image * 255.0)?
            .to_dtype(DType::U8)?
            .i(0)?
            .permute((1, 2, 0))?;
        let (oh, ow, _) = image.dims3()?;
        let buf = image.flatten_all()?.to_vec1::<u8>()?;
        if let Some(parent) = req.out.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, &req.out)?;
        crate::ui::progress::println(&format!("→ {}", req.out.display()));
        Ok(())
    }
}

// =====================================================================
// Single-shot entry — preserves the existing `plakat stylize` API.
// =====================================================================

pub async fn run(req: Request) -> Result<()> {
    let p = Pipeline::load(LoadRequest {
        model: req.model,
        device: req.device,
        shared_clip_h: None,
        instantstyle: req.instantstyle,
        style_scale: req.style_scale,
    })
    .await?;
    p.stylize_one(&GenRequest {
        input: req.input,
        reference: req.reference,
        out: req.out,
        strength: req.strength,
        steps: req.steps,
        seed: req.seed,
        ref_blur: req.ref_blur,
        ref_weight: req.ref_weight,
    })
}

fn read_image_size(path: &std::path::Path) -> Result<(u32, u32)> {
    let img = image::open(path)?;
    Ok(image::GenericImageView::dimensions(&img))
}

/// Round IN dims to multiples of 8, targeting the base's native size
/// (768 SD 1.5, 1024 SDXL). SDXL is trained at ~1024² and degrades into
/// glitch below it, so small inputs are scaled UP to the long side = 1024;
/// SD 1.5 only ever scales down.
fn sd_dims(in_w: u32, in_h: u32, is_xl: bool) -> (u32, u32) {
    let cap = if is_xl { 1024u32 } else { 768u32 };
    let raw = cap as f32 / in_w.max(in_h) as f32;
    let scale = if is_xl { raw } else { raw.min(1.0) };
    let w = ((in_w as f32) * scale).round() as u32;
    let h = ((in_h as f32) * scale).round() as u32;
    ((w / 8).max(1) * 8, (h / 8).max(1) * 8)
}
