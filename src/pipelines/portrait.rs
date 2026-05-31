//! Portrait generation pipeline.
//!
//! Supports two SD variants:
//!   * **SD 1.5** with `IdentityKind::PlusFace` — CLIP-H penultimate hidden
//!     state → Perceiver resampler (16 tokens × 768-d) → concat onto
//!     `(1, 77, 768)` text tokens → denoise from noise.
//!   * **SDXL** with `IdentityKind::PlusFaceSdxl` — same CLIP-H encoder
//!     (the `vit-h` SDXL Plus-Face variant), but the Resampler emits at
//!     SDXL's 2048-d cross-attn dim; concat onto SDXL's dual-encoder
//!     `(1, 77, 2048)` text tokens.
//!
//! Without a photo, behaves like a text-only portrait-tuned generate
//! (3:4 aspect default, face/anatomy negatives baked in at the CLI layer).
//!
//! Limitations carried over from `stylize`'s IP-Adapter integration:
//!   * candle 0.8 has no UNet attention hooks, so the *decoupled* cross-
//!     attention path (separate `to_k_ip` / `to_v_ip` per block) is not
//!     wired up. Identity tokens travel via the same cross-attention as
//!     text. Quality is recognisable but not pixel-perfect — typically
//!     ~50–70% of diffusers' reference. FaceID / InstantID (Phase-3+) are
//!     the path to better identity preservation.
//!   * SDXL micro-conditioning (`text_time` add-embedding from pooled
//!     CLIP-G + size/crop time-ids) is not wired up — candle's UNet has
//!     no `add_embedding` projection. Same gap as our base SDXL t2i path.
//!   * No automatic face crop. Pass a reasonably tight head-and-shoulders
//!     photo for best results.

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_transformers::models::stable_diffusion::{self, clip as sdclip};
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::pipelines::ip_adapter::IdentityEncoder;
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::scheduler::SchedulerKind;
use crate::ui::progress;

// Re-export so callers (CLI, scenario, future tools) can keep using
// `portrait::IdentityKind` even though the enum lives next to its loaders
// in `ip_adapter`. New strategies are added there, not here.
pub use crate::pipelines::ip_adapter::IdentityKind;

/// SD variant the portrait pipeline routes through. Phase 7b
/// re-exports [`sd_core::SdVariant`](crate::pipelines::sd_core::SdVariant)
/// — same Sd15 / Sdxl values, same `cross_attn_dim` / `vae_scale` /
/// `config` / `detect` helpers. Keeping the `portrait::Variant`
/// name preserves the existing internal call sites
/// (`Variant::Sd15`, `Variant::detect(...)`, etc.) untouched.
pub use crate::pipelines::sd_core::SdVariant as Variant;

// =====================================================================
// Request types.
// =====================================================================

/// Single-shot back-compat request (mirrors the t2i::Request shape).
pub struct Request {
    pub prompt: String,
    pub negative: String,
    /// Reference photos with merge weights. Empty = no identity
    /// encoding (text-only portrait). One = single-photo identity.
    /// Multiple = weighted merge in the encoder's embedding space.
    /// Weights normalized to sum to 1.0 by [`Pipeline::generate`].
    pub photos: Vec<crate::pipelines::ip_adapter::WeightedPhoto>,
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub count: u32,
    pub steps: usize,
    pub guidance: f64,
    pub seed: Option<u64>,
    pub out_dir: PathBuf,
    pub device: Device,
    pub loras: Vec<LoraSpec>,
    pub lora_scale: f32,
    pub scheduler: SchedulerKind,
    pub refine: Option<usize>,
    pub refine_strength: f32,
    pub face_strength: f32,
    pub face_bbox: Option<[f32; 4]>,
    pub face_landmarks: Option<[[f32; 2]; 5]>,
    /// Which identity strategy to wire up. `None` collapses portrait into a
    /// portrait-tuned text-only generate.
    pub identity: Option<IdentityKind>,
    /// Phase 7f optional shared CLIP-H. Forwarded into the pipeline's
    /// `LoadRequest` and consumed only by `PlusFace` / `PlusFaceSdxl`
    /// identity strategies.
    pub shared_clip_h: Option<std::sync::Arc<crate::pipelines::ip_adapter::ImageEncoder>>,

    // ---------- v0.9 ControlNet (v0.11: multi) ----------
    /// Stack of ControlNet conditioners. See `t2i::Request::controls`.
    pub controls: Vec<crate::pipelines::controlnet::ControlSpec>,
}

pub struct LoadRequest {
    pub model: String,
    pub device: Device,
    pub loras: Vec<LoraSpec>,
    pub lora_scale: f32,
    /// `Some(kind)` to pre-load the identity encoder. If `None`, the loaded
    /// pipeline can only do text-only portrait generation even if the
    /// caller later passes a `photo`.
    pub identity: Option<IdentityKind>,
    /// Phase 7f. Optional pre-loaded CLIP-H image encoder to share
    /// with `stylize::Pipeline` / `style::runtime::StyleSession`.
    /// Used only when `identity` is `PlusFace` / `PlusFaceSdxl`
    /// (FaceID strategies don't touch CLIP-H). `None` causes the
    /// identity encoder to download + load CLIP-H itself — the
    /// pre-7f behaviour.
    pub shared_clip_h: Option<std::sync::Arc<crate::pipelines::ip_adapter::ImageEncoder>>,
}

pub struct GenRequest {
    pub prompt: String,
    pub negative: String,
    /// Reference photos with merge weights. Same shape as
    /// [`Request::photos`]; see that field for semantics.
    pub photos: Vec<crate::pipelines::ip_adapter::WeightedPhoto>,
    pub width: u32,
    pub height: u32,
    pub count: u32,
    pub steps: usize,
    pub guidance: f64,
    pub seed: Option<u64>,
    pub out_dir: PathBuf,
    pub scheduler: SchedulerKind,
    pub refine: Option<usize>,
    pub refine_strength: f32,
    /// 0..1 multiplier applied to image-token contribution. Diffusers'
    /// IP-Adapter `set_scale` equivalent — at 1.0 image tokens carry full
    /// weight, at 0.0 they vanish (= text-only).
    pub face_strength: f32,
    /// Optional `[x0, y0, x1, y1]` (normalised in the photo's coordinate
    /// system) marking where the subject's face is. Used by FaceID
    /// strategies to crop the photo before ArcFace embedding. CLIP-H
    /// strategies (PlusFace*) ignore it.
    pub face_bbox: Option<[f32; 4]>,
    /// Optional 5-point landmarks. Takes precedence over
    /// `face_bbox`. FaceID strategies do similarity-transform alignment
    /// to ArcFace's canonical 112×112 template. Order: left_eye,
    /// right_eye, nose, left_mouth, right_mouth. Normalised to `[0, 1]`.
    pub face_landmarks: Option<[[f32; 2]; 5]>,
}

// =====================================================================
// Pipeline.
// =====================================================================

/// Portrait wrapping pipeline. Phase 7b: holds an `Arc<SdCore>`
/// for the shared SD backbone (UNet / VAE / CLIP / tokenizers /
/// device / dtype / merged-LoRA tempfiles), plus the portrait-
/// specific identity encoder. Multiple pipelines can share the
/// same SdCore in v0.10 phase 7d+ to eliminate redundant model
/// loads on `--artefact-blend` paths.
pub struct Pipeline {
    core: std::sync::Arc<crate::pipelines::sd_core::SdCore>,
    identity_encoder: Option<Box<dyn IdentityEncoder>>,
    /// Number of image tokens emitted by `identity_encoder`, when present.
    /// Cached so a zero-tokens tensor for the CFG uncond branch is the
    /// right shape without re-querying the trait.
    identity_num_tokens: usize,
}

impl Pipeline {
    /// Load weights for SD 1.5 or SDXL based on the model alias / repo.
    /// Flux models are rejected (portrait is a SD-architecture feature).
    ///
    /// Phase 7b: the SD backbone load delegates to
    /// [`SdCore::load`](crate::pipelines::sd_core::SdCore::load).
    /// Portrait-specific concerns (identity sanity check, FaceID
    /// auto-LoRA injection, identity encoder construction) stay here.
    pub async fn load(req: LoadRequest) -> Result<Self> {
        let base_repo = if req.model.contains('/') {
            req.model.clone()
        } else {
            crate::hf::resolve_alias(&req.model).to_string()
        };
        let lc = base_repo.to_lowercase();
        if lc.contains("flux") {
            bail!(
                "portrait does not support Flux. Use --model sd15 (default) \
                 or --model sdxl. Flux portraits would need a separate \
                 identity-adapter family — not yet ported."
            );
        }
        let variant = Variant::detect(&base_repo);

        // Sanity-check the identity strategy against the model variant.
        // Catches `--model sdxl --identity plus-face` (or vice versa)
        // before the model load eats seconds of download time.
        if let Some(kind) = req.identity {
            if kind.cross_attn_dim() != variant.cross_attn_dim() {
                bail!(
                    "identity strategy {:?} targets cross_attn_dim {} but \
                     model {:?} ({:?}) expects {}. Pick an identity that \
                     matches the model: SD 1.5 → `plus-face` or `faceid`, \
                     SDXL → `plus-face-sdxl` or `faceid-sdxl`.",
                    kind,
                    kind.cross_attn_dim(),
                    base_repo,
                    variant,
                    variant.cross_attn_dim(),
                );
            }
            // Pre-flight FaceID strategies' weight requirements before the
            // (potentially multi-GB) base model download.
            kind.preflight_weights()?;
        }

        // -------- resolve LoRAs (FaceID auto-LoRA + user LoRAs) --------
        // The auto-LoRA is portrait-specific (FaceID UNet adapter
        // bundled in the identity .bin); user LoRAs are general.
        // Both are resolved here and passed pre-resolved to
        // SdCore::load which merges them into the UNet + text
        // encoder weights.
        let mut auto_loras: Vec<crate::pipelines::lora::ResolvedLora> = Vec::new();
        if let Some(kind) = req.identity {
            if let Some(path) = kind.aux_unet_lora(&req.device).await? {
                auto_loras.push(crate::pipelines::lora::ResolvedLora {
                    path,
                    scale: 1.0,
                    display: format!("{} (auto)", kind.label()),
                });
            }
        }
        let resolved_loras: Vec<crate::pipelines::lora::ResolvedLora> = if req.loras.is_empty()
            && auto_loras.is_empty()
        {
            Vec::new()
        } else {
            let s = progress::spinner("Resolving LoRA file(s)");
            let mut v = Vec::with_capacity(req.loras.len() + auto_loras.len());
            v.extend(auto_loras);
            for spec in &req.loras {
                v.push(spec.resolve().await?);
            }
            s.finish_with_message(format!("✓ resolved {} LoRA file(s)", v.len()));
            v
        };

        // -------- delegate the SD backbone load --------
        let core = crate::pipelines::sd_core::SdCore::load(
            crate::pipelines::sd_core::SdLoadRequest {
                model: req.model.clone(),
                device: req.device.clone(),
                loras: resolved_loras,
                lora_scale: req.lora_scale,
                // v0.16 phase 9: portrait pipeline doesn't yet take
                // --embedding (sd_core bails loud when set). Pass
                // empty to stay byte-compatible with pre-phase-9.
                embeddings: Vec::new(),
                // v0.32 phase 2: portrait doesn't yet plumb VAE cache.
                vae_cache: None,
            },
        )
        .await
        .context("loading SD backbone for portrait pipeline")?;
        let dtype = core.dtype;

        // -------- identity encoder (portrait-specific) --------
        let (identity_encoder, identity_num_tokens) = if let Some(kind) = req.identity {
            let enc = kind
                .load_encoder_with_shared_clip(&req.device, dtype, req.shared_clip_h.clone())
                .await?;
            let n = enc.num_tokens();
            (Some(enc), n)
        } else {
            (None, 0)
        };

        Ok(Self {
            core: std::sync::Arc::new(core),
            identity_encoder,
            identity_num_tokens,
        })
    }

    /// Hand out a cheap `Arc` clone of the loaded SD backbone so a
    /// follow-on step (e.g. `--artefact-blend`) can build its own
    /// pipeline (`Pipeline::from_core`) without paying for a second
    /// model load. Phase 7e — mirrors `t2i::Pipeline::core`.
    pub fn core(&self) -> std::sync::Arc<crate::pipelines::sd_core::SdCore> {
        std::sync::Arc::clone(&self.core)
    }

    /// Construct a no-identity portrait pipeline from an already-loaded
    /// SD backbone. Phase 7d — lets follow-on steps such as
    /// `--artefact-blend` reuse the core loaded by `t2i::run` without
    /// downloading + re-merging weights a second time.
    ///
    /// The caller is responsible for making sure `core` was loaded
    /// with the model / device / LoRA set the blend pass expects;
    /// portrait does not re-validate those here. Identity adapters
    /// (FaceID / IP-Adapter) are unavailable on a `from_core` pipeline
    /// — blend passes don't use them anyway.
    pub fn from_core(core: std::sync::Arc<crate::pipelines::sd_core::SdCore>) -> Self {
        Self {
            core,
            identity_encoder: None,
            identity_num_tokens: 0,
        }
    }

    /// Encode text into the form the UNet expects:
    ///   * SD 1.5 — `(1, 77, 768)` from CLIP-L's final hidden state.
    ///   * SDXL   — `(1, 77, 2048)` from `concat(CLIP-L penultimate,
    ///              CLIP-G penultimate)` along the channel dim.
    /// Encode `text` into one batch row's worth of inputs the
    /// downstream UNet needs.
    ///
    /// Returns `(hidden_states, pooled_text)`:
    ///   * SD 1.5 / SD 2.1 — `hidden_states` only (one branch row);
    ///     `pooled_text` is `None` (no add_embedding in those UNets).
    ///   * SDXL — `hidden_states` is `(1, 77, 2048)` concat of
    ///     CLIP-L + CLIP-G penultimate; `pooled_text` is `Some((1, 1280))`
    ///     from the projected EOT row of CLIP-G's final hidden state,
    ///     consumed by [`SdxlUNet`]'s `add_embedding` after
    ///     `build_encoder_hidden_states` stitches the cond + uncond
    ///     branches together.
    fn encode_text(&self, text: &str) -> Result<(Tensor, Option<Tensor>)> {
        match self.core.variant {
            Variant::Sd15 | Variant::Sd21 => Ok((self.encode_text_sd15(text)?, None)),
            Variant::Sdxl => {
                let (h, p) = self.encode_text_sdxl(text)?;
                Ok((h, Some(p)))
            }
        }
    }

    fn encode_text_sd15(&self, text: &str) -> Result<Tensor> {
        let ids = tokenize_padded(&self.core.tokenizer_l, &self.core.cfg.clip, text, &self.core.device)?;
        Ok(self.core.text_encoder_l.forward(&ids)?.to_dtype(self.core.dtype)?)
    }

    fn encode_text_sdxl(&self, text: &str) -> Result<(Tensor, Tensor)> {
        let cfg_g = self
            .core
            .cfg
            .clip2
            .as_ref()
            .ok_or_else(|| anyhow!("SDXL Pipeline missing clip2 config"))?;
        let tok_g = self
            .core
            .tokenizer_g
            .as_ref()
            .ok_or_else(|| anyhow!("SDXL Pipeline missing tokenizer_g"))?;
        let enc_g = self
            .core
            .text_encoder_g
            .as_ref()
            .ok_or_else(|| anyhow!("SDXL Pipeline missing text_encoder_g"))?;
        let ids_l = tokenize_padded(&self.core.tokenizer_l, &self.core.cfg.clip, text, &self.core.device)?;
        let ids_g = tokenize_padded(tok_g, cfg_g, text, &self.core.device)?;
        let (_final_l, hidden_l) = self
            .core
            .text_encoder_l
            .forward_until_encoder_layer(&ids_l, usize::MAX, -2)?;
        // v0.11 phase 8d: also fetch the pooled CLIP-G output for the
        // UNet's add_embedding. forward_for_sdxl runs the encoder once
        // and hands back both penultimate hidden and projected pooled.
        let (hidden_g, pooled_g) = enc_g.forward_for_sdxl(&ids_g)?;
        let hidden = Tensor::cat(&[&hidden_l, &hidden_g], 2)?.to_dtype(self.core.dtype)?;
        let pooled = pooled_g.to_dtype(self.core.dtype)?;
        Ok((hidden, pooled))
    }

    /// Run `req.count` portraits. Reuses loaded weights across calls.
    ///
    /// `control` is the v0.9 ControlNet hook — when `Some`, the
    /// supplied conditioning is applied at every denoise step via
    /// [`UNet2DConditionModel::forward_with_additional_residuals`].
    /// `None` preserves byte-identical pre-v0.9 behaviour.
    pub fn generate(
        &self,
        req: &GenRequest,
        controls: &[crate::pipelines::controlnet::ControlRequest],
    ) -> Result<()> {
        crate::pipelines::scheduler::check_device_support(req.scheduler, &self.core.device)?;
        std::fs::create_dir_all(&req.out_dir)
            .with_context(|| format!("creating output dir {}", req.out_dir.display()))?;

        let (w, h) = (req.width as usize, req.height as usize);
        let do_cfg = req.guidance > 1.0;

        let (encoder_hidden_states, has_face, pooled_text_sdxl) =
            self.build_encoder_hidden_states(
                &req.prompt,
                &req.negative,
                &req.photos,
                req.face_strength,
                req.face_bbox,
                req.face_landmarks,
                do_cfg,
            )?;

        let bsz: usize = 1;
        let latent_h = h / 8;
        let latent_w = w / 8;
        let vae_scale: f64 = self.core.variant.vae_scale();

        // v0.11 phase 8d: build SDXL add_time_ids once per call. Same
        // pattern as t2i — function of target size only, replicated
        // for CFG. None on SD 1.5 / SD 2.1.
        let add_time_ids = build_sdxl_add_time_ids_base(
            self.core.variant,
            req.width,
            req.height,
            &self.core.device,
            self.core.dtype,
            do_cfg,
            pooled_text_sdxl.is_some(),
        )?;

        for idx in 0..req.count {
            let seed = req
                .seed
                .map(|s| s + idx as u64)
                .unwrap_or_else(rand::random);
            // v0.34 phase 1: device-aware seed prep.
            let prepared = crate::pipelines::seeds::prepare_seed(seed, &self.core.device);
            if let Err(e) = self.core.device.set_seed(prepared) {
                tracing::debug!(target: "plakat", "set_seed not supported ({e}); using global RNG");
            }

            let mut scheduler =
                crate::pipelines::scheduler::build(req.scheduler, &self.core.cfg, req.steps)?;
            let timesteps = scheduler.timesteps().to_vec();

            let mut latents =
                Tensor::randn(0f32, 1f32, (bsz, 4, latent_h, latent_w), &self.core.device)?
                    .to_dtype(self.core.dtype)?;
            latents = (latents * scheduler.init_noise_sigma())?;

            let face_tag = if has_face { "+face" } else { "txt" };
            let bar = progress::step_bar(
                timesteps.len() as u64,
                &format!("portrait {}/{} {}", idx + 1, req.count, face_tag),
            );
            let total_steps = timesteps.len();
            for (step_idx, &timestep) in timesteps.iter().enumerate() {
                let progress = step_idx as f32 / total_steps as f32;
                let active_controls: Vec<&crate::pipelines::controlnet::ControlRequest> =
                controls.iter().filter(|c| c.active_at(progress)).collect();
                latents = self.denoise_step(
                    &latents,
                    timestep,
                    &encoder_hidden_states,
                    pooled_text_sdxl.as_ref(),
                    add_time_ids.as_ref(),
                    &mut scheduler,
                    req.guidance,
                    do_cfg,
                    &active_controls,
                    // generate() is text-to-image — no inpaint mask.
                    None,
                    None,
                )?;
                bar.inc(1);
                bar.set_message(format!("t={timestep} seed={seed}"));
            }
            bar.finish_and_clear();

            // Optional same-model polish pass.
            if let Some(rsteps) = req.refine {
                if rsteps > 0 {
                    let strength = req.refine_strength.clamp(0.0, 1.0);
                    let mut polish =
                        crate::pipelines::scheduler::build(req.scheduler, &self.core.cfg, rsteps)?;
                    let pts = polish.timesteps().to_vec();
                    let init_skip = ((rsteps as f32) * (1.0 - strength)).round() as usize;
                    let init_skip = init_skip.min(rsteps.saturating_sub(1));
                    let active = &pts[init_skip..];
                    if let Some(&start_t) = active.first() {
                        let noise = Tensor::randn(0f32, 1f32, latents.shape(), &self.core.device)?
                            .to_dtype(self.core.dtype)?;
                        latents = polish.add_noise(&latents, noise, start_t)?;
                        let rbar = progress::step_bar(
                            active.len() as u64,
                            &format!("polish {}/{}", idx + 1, req.count),
                        );
                        let total_polish = active.len();
                        for (step_idx, &timestep) in active.iter().enumerate() {
                            let progress = step_idx as f32 / total_polish as f32;
                            let active_controls: Vec<&crate::pipelines::controlnet::ControlRequest> =
                controls.iter().filter(|c| c.active_at(progress)).collect();
                            latents = self.denoise_step(
                                &latents,
                                timestep,
                                &encoder_hidden_states,
                                pooled_text_sdxl.as_ref(),
                                add_time_ids.as_ref(),
                                &mut polish,
                                req.guidance,
                                do_cfg,
                                &active_controls,
                                // Polish pass is text-to-image — no inpaint mask.
                                None,
                                None,
                            )?;
                            rbar.inc(1);
                            rbar.set_message(format!("polish t={timestep}"));
                        }
                        rbar.finish_and_clear();
                    }
                }
            }

            // Decode + save.
            let image = self.core.vae.decode(&(&latents / vae_scale)?)?;
            let image = ((image / 2.0)? + 0.5)?.clamp(0f32, 1f32)?;
            let image = (image * 255.0)?
                .to_dtype(DType::U8)?
                .i(0)?
                .permute((1, 2, 0))?;
            let (oh, ow, _) = image.dims3()?;
            let buf = image.flatten_all()?.to_vec1::<u8>()?;
            let out_path = req.out_dir.join(format!("plakat-portrait-{seed}.png"));
            crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, &out_path)?;
            crate::ui::progress::println(&format!("→ {}", out_path.display()));
        }
        Ok(())
    }

    // =================================================================
    // Phase-2 multi-persona compositing primitives.
    //
    // The scenario runner orchestrates by calling these in sequence:
    //   let mut latents = pipeline.generate_latents_one(&base_req, seed)?;
    //   for (persona_req, mask) in passes {
    //       latents = pipeline.inpaint_latents_one(&latents, &mask, &persona_req, seed)?;
    //   }
    //   pipeline.save_image(&latents, &out_path)?;
    // =================================================================

    /// Build the encoder-hidden-states tensor for one call. With CFG this
    /// is `(2, 77 + K, 768)` where K = 0 (no face), 4 (plain), or 16
    /// (Plus). Returns the tensor plus a flag indicating whether image
    /// tokens were included (used for progress-bar labels).
    /// Returns `(ehs, has_face, pooled_text_sdxl)`:
    ///   * `ehs` — text + identity tokens concat'd along the seq dim
    ///     and (when `do_cfg`) along the batch dim (uncond, then cond).
    ///   * `has_face` — true iff identity tokens were attached.
    ///   * `pooled_text_sdxl` — `Some((B, 1280))` for SDXL with the
    ///     same uncond-then-cond batch layout as `ehs`. `None` for SD
    ///     1.5 / SD 2.1. Caller pairs this with `add_time_ids` and
    ///     feeds both to `denoise_step` so the SDXL UNet's
    ///     `add_embedding` receives the micro-conditioning signal.
    fn build_encoder_hidden_states(
        &self,
        prompt: &str,
        negative: &str,
        photos: &[crate::pipelines::ip_adapter::WeightedPhoto],
        face_strength: f32,
        face_bbox: Option<[f32; 4]>,
        face_landmarks: Option<[[f32; 2]; 5]>,
        do_cfg: bool,
    ) -> Result<(Tensor, bool, Option<Tensor>)> {
        let (text_cond, cond_pooled) = self.encode_text(prompt)?;
        let text_uncond_pair = if do_cfg {
            Some(self.encode_text(negative)?)
        } else {
            None
        };

        let face_strength = face_strength.clamp(0.0, 2.0);
        let identity_tokens = match (&self.identity_encoder, photos.is_empty()) {
            (Some(enc), false) => {
                let s = if photos.len() == 1 {
                    progress::spinner("Encoding reference photo")
                } else {
                    progress::spinner(&format!(
                        "Encoding {} reference photos (weighted merge)",
                        photos.len()
                    ))
                };
                let opts = crate::pipelines::ip_adapter::EncodeOptions {
                    face_bbox,
                    face_landmarks,
                };
                let tok = enc.encode(photos, opts)?;
                let tok = (tok * (face_strength as f64))?.to_dtype(self.core.dtype)?;
                s.finish_with_message("✓ identity encoded");
                Some(tok)
            }
            (None, false) => {
                bail!(
                    "this Pipeline was loaded without an identity encoder \
                     but a photo was provided. Reload with `identity: Some(IdentityKind::PlusFace)`."
                );
            }
            (Some(_), true) => None,
            (None, true) => None,
        };

        let cond_full = match &identity_tokens {
            Some(img) => Tensor::cat(&[&text_cond, img], 1)?,
            None => text_cond.clone(),
        };
        let ehs = if do_cfg {
            let (uncond_text, _) = text_uncond_pair.as_ref().unwrap();
            let uncond_full = match &identity_tokens {
                Some(img) => {
                    let zero = img.zeros_like()?;
                    Tensor::cat(&[uncond_text, &zero], 1)?
                }
                None => uncond_text.clone(),
            };
            Tensor::cat(&[&uncond_full, &cond_full], 0)?
        } else {
            cond_full
        };

        // Pooled text for SDXL's add_embedding. Same uncond-then-cond
        // batch layout as `ehs` so a single index aligns both.
        let pooled = match (&cond_pooled, text_uncond_pair.as_ref().and_then(|(_, p)| p.as_ref())) {
            (Some(c), Some(u)) if do_cfg => Some(Tensor::cat(&[u, c], 0)?),
            (Some(c), _) if !do_cfg => Some(c.clone()),
            _ => None,
        };

        Ok((ehs, identity_tokens.is_some(), pooled))
    }

    /// Generate one sample of latents from text alone (no inpainting).
    /// Used as the base for multi-persona compositing. Skips the polish
    /// pass — orchestrator may run polish on the final composite.
    ///
    /// `control` — same v0.9 ControlNet hook as [`Self::generate`].
    pub fn generate_latents_one(
        &self,
        req: &GenRequest,
        seed: u64,
        controls: &[crate::pipelines::controlnet::ControlRequest],
    ) -> Result<Tensor> {
        crate::pipelines::scheduler::check_device_support(req.scheduler, &self.core.device)?;
        let (w, h) = (req.width as usize, req.height as usize);
        let do_cfg = req.guidance > 1.0;
        let (ehs, has_face, pooled_text_sdxl) = self.build_encoder_hidden_states(
            &req.prompt,
            &req.negative,
            &req.photos,
            req.face_strength,
            req.face_bbox,
            req.face_landmarks,
            do_cfg,
        )?;
        let add_time_ids = build_sdxl_add_time_ids_base(
            self.core.variant,
            req.width,
            req.height,
            &self.core.device,
            self.core.dtype,
            do_cfg,
            pooled_text_sdxl.is_some(),
        )?;

        // v0.34 phase 1: device-aware seed prep.
        let prepared = crate::pipelines::seeds::prepare_seed(seed, &self.core.device);
        if let Err(e) = self.core.device.set_seed(prepared) {
            tracing::debug!(target: "plakat", "set_seed not supported ({e}); using global RNG");
        }
        let mut scheduler =
            crate::pipelines::scheduler::build(req.scheduler, &self.core.cfg, req.steps)?;
        let timesteps = scheduler.timesteps().to_vec();
        let latent_h = h / 8;
        let latent_w = w / 8;
        let mut latents = Tensor::randn(0f32, 1f32, (1, 4, latent_h, latent_w), &self.core.device)?
            .to_dtype(self.core.dtype)?;
        latents = (latents * scheduler.init_noise_sigma())?;

        let face_tag = if has_face { "+face" } else { "txt" };
        let bar = progress::step_bar(
            timesteps.len() as u64,
            &format!("composite-base {face_tag}"),
        );
        let total_steps = timesteps.len();
        for (step_idx, &t) in timesteps.iter().enumerate() {
            let progress = step_idx as f32 / total_steps as f32;
            let active_controls: Vec<&crate::pipelines::controlnet::ControlRequest> =
                controls.iter().filter(|c| c.active_at(progress)).collect();
            latents = self.denoise_step(
                &latents,
                t,
                &ehs,
                pooled_text_sdxl.as_ref(),
                add_time_ids.as_ref(),
                &mut scheduler,
                req.guidance,
                do_cfg,
                &active_controls,
                // generate_latents_one is text-to-image — no inpaint mask.
                None,
                None,
            )?;
            bar.inc(1);
            bar.set_message(format!("t={t} seed={seed}"));
        }
        bar.finish_and_clear();
        Ok(latents)
    }

    /// Inpaint one persona into `base_latents` inside `mask`. Uses
    /// RePaint-style latent blending: at each timestep, the unmasked
    /// region is replaced with a re-noised copy of `base_latents`, so
    /// the denoiser only meaningfully drives the masked region.
    ///
    /// `mask` is `(1, 1, latent_h, latent_w)` at the pipeline's dtype,
    /// values in `[0, 1]` (1 = inpaint here, 0 = preserve base).
    pub fn inpaint_latents_one(
        &self,
        base_latents: &Tensor,
        mask: &Tensor,
        req: &GenRequest,
        seed: u64,
        controls: &[crate::pipelines::controlnet::ControlRequest],
        // v0.12 Inpaint UNet: VAE-encoded `input × (1 - mask)` at
        // latent resolution. When `Some` AND `self.core.is_inpaint`
        // the function takes the 9-channel UNet path (no RePaint
        // mask-blending; mask + masked latents concat'd into UNet input
        // every step). `None` keeps the RePaint path for regular SDXL
        // and SD 1.5 / SD 2.1 UNets.
        inpaint_masked_latents: Option<&Tensor>,
    ) -> Result<Tensor> {
        crate::pipelines::scheduler::check_device_support(req.scheduler, &self.core.device)?;
        let do_cfg = req.guidance > 1.0;
        let (ehs, has_face, pooled_text_sdxl) = self.build_encoder_hidden_states(
            &req.prompt,
            &req.negative,
            &req.photos,
            req.face_strength,
            req.face_bbox,
            req.face_landmarks,
            do_cfg,
        )?;
        let add_time_ids = build_sdxl_add_time_ids_base(
            self.core.variant,
            req.width,
            req.height,
            &self.core.device,
            self.core.dtype,
            do_cfg,
            pooled_text_sdxl.is_some(),
        )?;
        let use_inpaint_unet = self.core.is_inpaint
            && inpaint_masked_latents.is_some();
        if self.core.is_inpaint && inpaint_masked_latents.is_none() {
            anyhow::bail!(
                "Inpaint UNet UNet loaded but no masked-image latents supplied. \
                 The 9-channel UNet requires VAE(input × (1 - mask)) at every step. \
                 Use a regular SDXL model with --mask for RePaint-style inpaint."
            );
        }

        // v0.34 phase 1: device-aware seed prep.
        let prepared = crate::pipelines::seeds::prepare_seed(seed, &self.core.device);
        if let Err(e) = self.core.device.set_seed(prepared) {
            tracing::debug!(target: "plakat", "set_seed not supported ({e}); using global RNG");
        }
        let mut scheduler =
            crate::pipelines::scheduler::build(req.scheduler, &self.core.cfg, req.steps)?;
        let timesteps = scheduler.timesteps().to_vec();
        let first_t = *timesteps
            .first()
            .ok_or_else(|| anyhow!("inpaint scheduler produced 0 timesteps"))?;

        // Start: re-noise the base at the first timestep. The masked region
        // gets driven by the denoiser; the unmasked region gets re-noised
        // again at each step so the masked region sees a coherent neighbour.
        let initial_noise = Tensor::randn(0f32, 1f32, base_latents.shape(), &self.core.device)?
            .to_dtype(self.core.dtype)?;
        let mut latents = scheduler.add_noise(base_latents, initial_noise, first_t)?;

        let inv_mask = (mask.ones_like()? - mask)?;
        let face_tag = if has_face { "+face" } else { "txt" };
        let mode_tag = if use_inpaint_unet {
            "inpaint-unet"
        } else {
            "inpaint"
        };
        let bar = progress::step_bar(
            timesteps.len() as u64,
            &format!("{mode_tag} {face_tag}"),
        );
        let total_steps = timesteps.len();
        for (step_idx, &t) in timesteps.iter().enumerate() {
            let progress = step_idx as f32 / total_steps as f32;
            let active_controls: Vec<&crate::pipelines::controlnet::ControlRequest> =
                controls.iter().filter(|c| c.active_at(progress)).collect();

            // RePaint-style mask blending is for regular UNets only.
            // Inpaint UNet feeds the mask + masked-image latents through
            // the 9-channel UNet input every step, so the network itself
            // handles "preserve outside the mask"; doing the per-step
            // re-noise on top would double-up the conditioning.
            if !use_inpaint_unet {
                let fresh_noise = Tensor::randn(0f32, 1f32, base_latents.shape(), &self.core.device)?
                    .to_dtype(self.core.dtype)?;
                let base_noised = scheduler.add_noise(base_latents, fresh_noise, t)?;
                latents = (latents.broadcast_mul(mask)?
                    + base_noised.broadcast_mul(&inv_mask)?)?;
            }

            latents = self.denoise_step(
                &latents,
                t,
                &ehs,
                pooled_text_sdxl.as_ref(),
                add_time_ids.as_ref(),
                &mut scheduler,
                req.guidance,
                do_cfg,
                &active_controls,
                if use_inpaint_unet { Some(mask) } else { None },
                if use_inpaint_unet {
                    inpaint_masked_latents
                } else {
                    None
                },
            )?;
            bar.inc(1);
            bar.set_message(format!("t={t} seed={seed}"));
        }
        bar.finish_and_clear();

        // Final blend: pin unmasked region to the *clean* base latents (no
        // residual noise). The masked region keeps the denoiser's output.
        // For Inpaint UNet the 9-channel UNet already preserves the
        // unmasked region internally, but we still composite to clamp
        // any edge bleed and to keep the contract uniform.
        let composited = (latents.broadcast_mul(mask)?
            + base_latents.broadcast_mul(&inv_mask)?)?;
        Ok(composited)
    }

    /// VAE-encode an existing image file into the pipeline's latent
    /// space. The image is rescaled to `(w, h)` (the generation
    /// dimensions); the result is `(1, 4, h/8, w/8)` in the pipeline's
    /// dtype.
    ///
    /// Used by [`crate::pipelines::artefact_blend`] to seed a masked
    /// denoise pass from an already-composited PNG.
    pub fn vae_encode_image_file(
        &self,
        path: &std::path::Path,
        w: u32,
        h: u32,
    ) -> Result<Tensor> {
        let pixels = crate::imaging::preprocess::sd_image_tensor(path, w, h, &self.core.device, self.core.dtype)
            .with_context(|| format!("VAE-encoding {}", path.display()))?;
        self.vae_encode_pixels(&pixels)
    }

    /// VAE-encode an already-prepared pixel tensor. The caller is
    /// responsible for shape `(1, 3, H, W)` and the same `[-1, 1]`
    /// normalisation that `sd_image_tensor` produces. Used by
    /// Inpaint UNet's masked-image-latents preparation.
    pub fn vae_encode_pixels(&self, pixels: &Tensor) -> Result<Tensor> {
        let vae_scale: f64 = self.core.variant.vae_scale();
        let dist = self.core.vae.encode(pixels)?;
        let latents = (dist.sample()? * vae_scale)?;
        Ok(latents)
    }

    /// Masked partial-strength denoise — the v2 artefact-blend primitive.
    ///
    /// Same RePaint-style mask blending as [`Self::inpaint_latents_one`],
    /// but starts denoising at `(1 − strength) * len(timesteps)` instead
    /// of from full noise. That gives standard img2img "strength"
    /// semantics:
    ///
    /// * `strength = 0.0` → no denoising, returns base_latents unchanged
    ///   (apart from a final clean blend).
    /// * `strength = 0.25` → light touch; smooths the masked region
    ///   without redrawing it.
    /// * `strength = 0.5` → mid-strength repaint.
    /// * `strength = 1.0` → equivalent to `inpaint_latents_one`
    ///   (re-noise to max, full denoise).
    ///
    /// `mask` is `(1, 1, latent_h, latent_w)`: `1.0` = inpaint here,
    /// `0.0` = preserve.
    pub fn blend_latents_one(
        &self,
        base_latents: &Tensor,
        mask: &Tensor,
        req: &GenRequest,
        strength: f32,
        seed: u64,
        controls: &[crate::pipelines::controlnet::ControlRequest],
        // v0.12 Inpaint UNet: VAE-encoded masked-image latents.
        // Same contract as [`Self::inpaint_latents_one`].
        inpaint_masked_latents: Option<&Tensor>,
    ) -> Result<Tensor> {
        crate::pipelines::scheduler::check_device_support(req.scheduler, &self.core.device)?;
        let strength = strength.clamp(0.0, 1.0);
        let do_cfg = req.guidance > 1.0;
        let (ehs, _has_face, pooled_text_sdxl) = self.build_encoder_hidden_states(
            &req.prompt,
            &req.negative,
            &req.photos,
            req.face_strength,
            req.face_bbox,
            req.face_landmarks,
            do_cfg,
        )?;
        let add_time_ids = build_sdxl_add_time_ids_base(
            self.core.variant,
            req.width,
            req.height,
            &self.core.device,
            self.core.dtype,
            do_cfg,
            pooled_text_sdxl.is_some(),
        )?;
        let use_inpaint_unet = self.core.is_inpaint
            && inpaint_masked_latents.is_some();
        if self.core.is_inpaint && inpaint_masked_latents.is_none() {
            anyhow::bail!(
                "Inpaint UNet UNet loaded but no masked-image latents supplied. \
                 The 9-channel UNet requires VAE(input × (1 - mask)) at every step."
            );
        }

        // v0.34 phase 1: device-aware seed prep.
        let prepared = crate::pipelines::seeds::prepare_seed(seed, &self.core.device);
        if let Err(e) = self.core.device.set_seed(prepared) {
            tracing::debug!(target: "plakat", "set_seed not supported ({e}); using global RNG");
        }
        let mut scheduler =
            crate::pipelines::scheduler::build(req.scheduler, &self.core.cfg, req.steps)?;
        let timesteps = scheduler.timesteps().to_vec();

        // start_idx selects where on the noise schedule we begin. At
        // strength=0 we'd start at the very end (no denoising); at
        // strength=1 we'd start at index 0 (max noise). Clamp to at
        // least 1 step so we still run one denoise iteration.
        let total = timesteps.len();
        let start_idx = (((1.0 - strength) * total as f32).round() as usize).min(total.saturating_sub(1));
        let active = &timesteps[start_idx..];
        if active.is_empty() {
            // strength == 0 with degenerate scheduler — return base
            // unchanged (apart from the clean-blend semantics).
            return Ok(base_latents.clone());
        }

        let first_t = active[0];

        // Re-noise the base latents at the partial-noise level.
        let initial_noise = Tensor::randn(0f32, 1f32, base_latents.shape(), &self.core.device)?
            .to_dtype(self.core.dtype)?;
        let mut latents = scheduler.add_noise(base_latents, initial_noise, first_t)?;

        let inv_mask = (mask.ones_like()? - mask)?;
        let bar_tag = if use_inpaint_unet { "inpaint-blend" } else { "blend" };
        let bar = progress::step_bar(active.len() as u64, bar_tag);
        // Diffusers convention: control_start/end is measured
        // against the FULL schedule, not the active subset. So
        // step_idx counts from `start_idx`, not from 0.
        for (i, &t) in active.iter().enumerate() {
            let progress = (start_idx + i) as f32 / total as f32;
            let active_controls: Vec<&crate::pipelines::controlnet::ControlRequest> =
                controls.iter().filter(|c| c.active_at(progress)).collect();

            // RePaint-style per-step blend only for the regular
            // (4-channel) UNet path — Inpaint UNet handles the
            // unmasked region inside the network.
            if !use_inpaint_unet {
                let fresh_noise = Tensor::randn(0f32, 1f32, base_latents.shape(), &self.core.device)?
                    .to_dtype(self.core.dtype)?;
                let base_noised = scheduler.add_noise(base_latents, fresh_noise, t)?;
                latents = (latents.broadcast_mul(mask)?
                    + base_noised.broadcast_mul(&inv_mask)?)?;
            }

            latents = self.denoise_step(
                &latents,
                t,
                &ehs,
                pooled_text_sdxl.as_ref(),
                add_time_ids.as_ref(),
                &mut scheduler,
                req.guidance,
                do_cfg,
                &active_controls,
                if use_inpaint_unet { Some(mask) } else { None },
                if use_inpaint_unet {
                    inpaint_masked_latents
                } else {
                    None
                },
            )?;
            bar.inc(1);
            bar.set_message(format!("t={t}"));
        }
        bar.finish_and_clear();

        // Final clean blend: pin unmasked region to base, keep
        // denoised in masked region.
        let composited = (latents.broadcast_mul(mask)?
            + base_latents.broadcast_mul(&inv_mask)?)?;
        Ok(composited)
    }

    /// VAE-decode `latents` and save as PNG at `out_path`.
    pub fn save_image(
        &self,
        latents: &Tensor,
        out_path: &std::path::Path,
    ) -> Result<()> {
        let vae_scale: f64 = self.core.variant.vae_scale();
        let image = self.core.vae.decode(&(latents / vae_scale)?)?;
        let image = ((image / 2.0)? + 0.5)?.clamp(0f32, 1f32)?;
        let image = (image * 255.0)?
            .to_dtype(DType::U8)?
            .i(0)?
            .permute((1, 2, 0))?;
        let (oh, ow, _) = image.dims3()?;
        let buf = image.flatten_all()?.to_vec1::<u8>()?;
        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, out_path)?;
        crate::ui::progress::println(&format!("→ {}", out_path.display()));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn denoise_step(
        &self,
        latents: &Tensor,
        timestep: usize,
        encoder_hidden_states: &Tensor,
        // v0.11 phase 8d: SDXL `text_time` micro-conditioning.
        // Required for SdUNet::Sdxl (set on SDXL portraits), ignored
        // for SdUNet::Sd (SD 1.5 / SD 2.1).
        add_text_embeds: Option<&Tensor>,
        add_time_ids: Option<&Tensor>,
        scheduler: &mut Box<dyn stable_diffusion::schedulers::Scheduler>,
        guidance: f64,
        do_cfg: bool,
        // v0.11 multi-ControlNet: caller pre-filters to active controls.
        active_controls: &[&crate::pipelines::controlnet::ControlRequest],
        // v0.12 SDXL Inpainting extras. Both `Some` for Inpaint UNet
        // (the 9-channel UNet path); both `None` for every regular
        // (4-channel) UNet — including RePaint-style SD 1.5/2.1/SDXL
        // inpaint via mask blending. Tensors:
        //   * `inpaint_mask`           — `(1, 1, h/8, w/8)`, same one
        //                                the mask-blend uses.
        //   * `inpaint_masked_latents` — `(1, 4, h/8, w/8)`, VAE-encode
        //                                of the pixel-space `input ×
        //                                (1 - mask)`.
        // Tiled to 2× along the batch dim under CFG, then concat'd
        // along the channel dim onto the scaled latent input before
        // the UNet forward.
        inpaint_mask: Option<&Tensor>,
        inpaint_masked_latents: Option<&Tensor>,
    ) -> Result<Tensor> {
        let latent_in = if do_cfg {
            Tensor::cat(&[latents, latents], 0)?
        } else {
            latents.clone()
        };
        let latent_in = scheduler.scale_model_input(latent_in, timestep)?;
        // Inpaint UNet: concat the 9-channel input. Caller has already
        // built the mask + masked-image latents at latent resolution.
        let latent_in = match (inpaint_mask, inpaint_masked_latents) {
            (Some(m), Some(ml)) => {
                let m_tiled = if do_cfg {
                    Tensor::cat(&[m, m], 0)?
                } else {
                    m.clone()
                };
                let ml_tiled = if do_cfg {
                    Tensor::cat(&[ml, ml], 0)?
                } else {
                    ml.clone()
                };
                Tensor::cat(&[&latent_in, &m_tiled, &ml_tiled], 1)?
            }
            (None, None) => latent_in,
            _ => anyhow::bail!(
                "Inpaint UNet denoise: inpaint_mask and inpaint_masked_latents \
                 must be supplied together"
            ),
        };
        let noise_pred = if active_controls.is_empty() {
            self.core.unet.forward(
                &latent_in,
                timestep as f64,
                encoder_hidden_states,
                add_text_embeds,
                add_time_ids,
            )?
        } else {
            let (down, mid) = crate::pipelines::controlnet::sum_controlnet_residuals(
                active_controls,
                &latent_in,
                timestep,
                encoder_hidden_states,
                do_cfg,
                // v0.12: SDXL ControlNet now consumes the same
                // text_time micro-conditioning as the UNet.
                add_text_embeds,
                add_time_ids,
            )?;
            self.core.unet.forward_with_additional_residuals(
                &latent_in,
                timestep as f64,
                encoder_hidden_states,
                add_text_embeds,
                add_time_ids,
                Some(&down),
                Some(&mid),
            )?
        };
        let noise_pred = if do_cfg {
            let chunks = noise_pred.chunk(2, 0)?;
            let uncond = &chunks[0];
            let text = &chunks[1];
            (uncond + ((text - uncond)? * guidance)?)?
        } else {
            noise_pred
        };
        Ok(scheduler.step(&noise_pred, timestep, latents)?)
    }
}

/// Suppress dead-code warning on the cached token count until something
/// else queries it (debug logging, etc.). Kept on the struct so future
/// callers don't have to thread a separate value.
impl Pipeline {
    #[allow(dead_code)]
    pub fn identity_num_tokens(&self) -> usize {
        self.identity_num_tokens
    }

    /// The dtype the pipeline's tensors live at (F16 on accelerators,
    /// F32 on CPU). Callers building masks for `inpaint_latents_one`
    /// need this so the mask matches.
    pub fn latent_dtype(&self) -> DType {
        self.core.dtype
    }

    /// The device backing this pipeline's tensors. Needed by callers
    /// that build mask tensors outside the pipeline (e.g. v2 artefact
    /// blending).
    pub fn device(&self) -> &Device {
        &self.core.device
    }
}

// =====================================================================
// Single-shot entry — what `plakat portrait` calls.
// =====================================================================

/// Run a portrait task. Returns the loaded `SdCore` so a follow-on
/// step (e.g. `--artefact-blend`) can reuse the same weights via
/// [`Pipeline::from_core`] instead of paying for a second load.
/// Portrait does not route through Flux (Flux portraits are rejected
/// inside `Pipeline::load`), so the return is unconditional. Phase 7e.
pub async fn run(req: Request) -> Result<std::sync::Arc<crate::pipelines::sd_core::SdCore>> {
    // Preload ControlNet stack + conditioning(s) before the pipeline
    // (same ordering as `t2i::run`). Owned data lives on this stack
    // frame; `ControlRequest`s borrow from it just before generate().
    let dtype = if matches!(req.device, Device::Cpu) {
        DType::F32
    } else {
        DType::F16
    };
    let control_owned = crate::pipelines::controlnet::load_control_stack(
        &req.controls,
        &req.model,
        req.width,
        req.height,
        &req.device,
        dtype,
        None,
        None, // portrait is single-image — no per-frame video CN
    )
    .await?;

    let pipeline = Pipeline::load(LoadRequest {
        model: req.model,
        device: req.device,
        loras: req.loras,
        lora_scale: req.lora_scale,
        identity: req.identity,
        shared_clip_h: req.shared_clip_h,
    })
    .await?;

    // Normalize weights before handing off to the encoder. Done at the
    // top-level boundary so internal GenRequest invariant (every photo's
    // weight is Some) holds end-to-end.
    let mut photos = req.photos;
    if !photos.is_empty() {
        crate::pipelines::ip_adapter::normalize_photo_weights(&mut photos)?;
    }

    let control_reqs: Vec<crate::pipelines::controlnet::ControlRequest> = control_owned
        .iter()
        .map(|owned| crate::pipelines::controlnet::ControlRequest {
            net: &owned.net,
            conditioning: owned.conditioning.clone(),
            strength: owned.strength,
            start: owned.start,
            end: owned.end,
        })
        .collect();

    pipeline.generate(
        &GenRequest {
            prompt: req.prompt,
            negative: req.negative,
            photos,
            width: req.width,
            height: req.height,
            count: req.count,
            steps: req.steps,
            guidance: req.guidance,
            seed: req.seed,
            out_dir: req.out_dir,
            scheduler: req.scheduler,
            refine: req.refine,
            refine_strength: req.refine_strength,
            face_strength: req.face_strength,
            face_bbox: req.face_bbox,
            face_landmarks: req.face_landmarks,
        },
        &control_reqs,
    )?;
    Ok(pipeline.core())
}

// =====================================================================
// Tokenisation helper — shared by SD 1.5 + SDXL encode paths.
// Mirrors `t2i::tokenize_padded`. Lives here to avoid making t2i's helper
// pub; both modules now have their own copy. Trivial duplication is
// preferable to a third "shared" module just for this.
// =====================================================================
/// v0.11 phase 8d helper. Returns `Some((B, 6))` add_time_ids for
/// SDXL portraits, replicated across the CFG branch dim when present.
/// Returns `None` for SD 1.5 / SD 2.1 or when the caller signals
/// `pooled_text_present = false` (e.g. variant mismatch — should not
/// normally happen). Mirrors the t2i helper.
fn build_sdxl_add_time_ids_base(
    variant: Variant,
    width: u32,
    height: u32,
    device: &Device,
    dtype: DType,
    do_cfg: bool,
    pooled_text_present: bool,
) -> Result<Option<Tensor>> {
    if variant != Variant::Sdxl || !pooled_text_present {
        return Ok(None);
    }
    let row = crate::pipelines::sdxl_unet::build_add_time_ids_base(
        height, width, device, dtype,
    )?;
    let stacked = if do_cfg {
        Tensor::cat(&[&row, &row], 0)?
    } else {
        row
    };
    Ok(Some(stacked))
}

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
