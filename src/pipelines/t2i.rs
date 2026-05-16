//! Text-to-image inference pipeline.
//!
//! Supported in candle 0.8:
//!   * SD 1.5 / 2.1            — single CLIP-L text encoder, VAE scale 0.18215
//!   * SDXL / SDXL-Turbo       — dual encoder (CLIP-L + CLIP-G), penultimate
//!                               hidden states concatenated to 2048 channels;
//!                               VAE scale 0.13025
//!
//! Flux is detected but routes out to `pipelines::flux`.
//!
//! Two ways to use this module:
//!   * `t2i::run(Request)` — single-shot. Loads everything then generates.
//!     This is what `plakat generate` uses.
//!   * `Pipeline::load(...)` then `pipeline.generate(...)` per task. Reuses
//!     loaded weights across calls so multi-task scenarios don't pay the
//!     ~10s model-load overhead N times. This is what `plakat scenario` uses.

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_transformers::models::stable_diffusion::{
    self, StableDiffusionConfig, clip as sdclip,
    unet_2d::UNet2DConditionModel,
    vae::AutoEncoderKL,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::pipelines::lora::LoraSpec;
use crate::pipelines::scheduler::SchedulerKind;
use crate::ui::progress;

// =====================================================================
// Public types: single-shot Request/run (back-compat) + Pipeline API.
// =====================================================================

pub struct Request {
    pub prompt: String,
    pub negative: String,
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
}

/// Stuff that's fixed for the lifetime of a Pipeline.
pub struct LoadRequest {
    pub model: String,
    pub device: Device,
    pub loras: Vec<LoraSpec>,
    pub lora_scale: f32,
}

/// Stuff that can vary per `Pipeline::generate` call.
pub struct GenRequest {
    pub prompt: String,
    pub negative: String,
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
}

// =====================================================================
// Variant detection.
// =====================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Sd15,
    Sd21,
    Sdxl,
    SdxlTurbo,
    FluxSchnell,
    FluxDev,
}

impl Variant {
    pub fn detect(model: &str) -> Self {
        let m = model.to_lowercase();
        if m.contains("flux") {
            if m.contains("dev") {
                Self::FluxDev
            } else {
                Self::FluxSchnell
            }
        } else if m.contains("turbo") {
            Self::SdxlTurbo
        } else if m.contains("xl") {
            Self::Sdxl
        } else if m.contains("2-1") || m.contains("2.1") || m.contains("v2") {
            Self::Sd21
        } else {
            Self::Sd15
        }
    }

    fn config(self, w: usize, h: usize) -> Result<StableDiffusionConfig> {
        Ok(match self {
            Self::Sd15 => StableDiffusionConfig::v1_5(None, Some(h), Some(w)),
            Self::Sd21 => StableDiffusionConfig::v2_1(None, Some(h), Some(w)),
            Self::Sdxl => StableDiffusionConfig::sdxl(None, Some(h), Some(w)),
            Self::SdxlTurbo => StableDiffusionConfig::sdxl_turbo(None, Some(h), Some(w)),
            Self::FluxSchnell | Self::FluxDev => unreachable!(
                "Flux variants route through pipelines::flux::run, not Pipeline::load"
            ),
        })
    }

    fn dtype(self, dev: &Device) -> DType {
        if matches!(dev, Device::Cpu) {
            DType::F32
        } else {
            DType::F16
        }
    }

    fn vae_scale(self) -> f64 {
        match self {
            Self::Sdxl | Self::SdxlTurbo => 0.13025,
            _ => 0.18215,
        }
    }

    pub fn is_xl(self) -> bool {
        matches!(self, Self::Sdxl | Self::SdxlTurbo)
    }
    pub fn is_flux(self) -> bool {
        matches!(self, Self::FluxSchnell | Self::FluxDev)
    }
}

fn resolve_repo(model: &str) -> String {
    if model.contains('/') {
        model.to_string()
    } else {
        crate::hf::resolve_alias(model).to_string()
    }
}

async fn fetch_first(repo: &str, candidates: &[&str]) -> Result<PathBuf> {
    let mut last_err = None;
    for f in candidates {
        match crate::hf::download::get_file(repo, f).await {
            Ok(p) => return Ok(p),
            Err(e) => {
                tracing::debug!(target: "plakat", "miss {repo}/{f}: {e}");
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("no candidates given")))
}

// =====================================================================
// Pipeline: load once, generate many.
// =====================================================================

pub struct Pipeline {
    pub variant: Variant,
    /// Resolved HF repo id this pipeline was loaded from (after alias resolution).
    #[allow(dead_code)]
    pub repo: String,
    cfg: StableDiffusionConfig,
    tokenizer_l: Tokenizer,
    tokenizer_g: Option<Tokenizer>,
    text_encoder_l: sdclip::ClipTextTransformer,
    text_encoder_g: Option<sdclip::ClipTextTransformer>,
    vae: AutoEncoderKL,
    unet: UNet2DConditionModel,
    device: Device,
    dtype: DType,
    // Held to keep the merged-UNet tempfile alive for the Pipeline's lifetime.
    // The UNet's mmap actually survives the temp file's unlink on Unix, but
    // holding the guard avoids relying on that.
    _lora_tmp: Option<tempfile::NamedTempFile>,
}

impl Pipeline {
    /// Download + load + merge LoRAs once. SD/SDXL only.
    pub async fn load(req: LoadRequest) -> Result<Self> {
        let variant = Variant::detect(&req.model);
        if variant.is_flux() {
            anyhow::bail!(
                "Pipeline::load is SD-only; Flux models use pipelines::flux::run"
            );
        }
        let repo = resolve_repo(&req.model);
        // Placeholder dims — not baked into model weights, only stored in cfg.
        let cfg = variant.config(512, 512)?;
        let dtype = variant.dtype(&req.device);

        // ---- download weights ----
        let dl = progress::spinner(&format!("Resolving weights for {repo}"));

        let tokenizer_l_path = crate::hf::download::get_first_of(&[
            (&repo, "tokenizer/tokenizer.json"),
            ("openai/clip-vit-large-patch14", "tokenizer.json"),
        ])
        .await
        .with_context(|| format!("tokenizer (CLIP-L) for {repo}"))?;
        let text_enc_l_path = fetch_first(
            &repo,
            &[
                "text_encoder/model.fp16.safetensors",
                "text_encoder/model.safetensors",
            ],
        )
        .await
        .with_context(|| format!("text_encoder weights in {repo}"))?;

        let (tokenizer_g_path, text_enc_g_path) = if variant.is_xl() {
            let t = crate::hf::download::get_first_of(&[
                (&repo, "tokenizer_2/tokenizer.json"),
                ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json"),
                ("openai/clip-vit-large-patch14", "tokenizer.json"),
            ])
            .await
            .with_context(|| format!("tokenizer (CLIP-G) for {repo}"))?;
            let e = fetch_first(
                &repo,
                &[
                    "text_encoder_2/model.fp16.safetensors",
                    "text_encoder_2/model.safetensors",
                ],
            )
            .await
            .with_context(|| format!("text_encoder_2 in {repo}"))?;
            (Some(t), Some(e))
        } else {
            (None, None)
        };

        let unet_path = fetch_first(
            &repo,
            &[
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "unet/diffusion_pytorch_model.safetensors",
            ],
        )
        .await
        .with_context(|| format!("unet weights in {repo}"))?;
        let vae_path = fetch_first(
            &repo,
            &[
                "vae/diffusion_pytorch_model.fp16.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
            ],
        )
        .await
        .with_context(|| format!("vae weights in {repo}"))?;
        dl.finish_with_message(format!("✓ weights ready for {repo}"));

        // ---- build models ----
        let build = progress::spinner("Loading models");

        let tokenizer_l = Tokenizer::from_file(&tokenizer_l_path)
            .map_err(|e| anyhow!("tokenizer (CLIP-L): {e}"))?;
        let tokenizer_g = match tokenizer_g_path.as_ref() {
            Some(p) => Some(
                Tokenizer::from_file(p).map_err(|e| anyhow!("tokenizer (CLIP-G): {e}"))?,
            ),
            None => None,
        };

        let text_encoder_l = stable_diffusion::build_clip_transformer(
            &cfg.clip,
            &text_enc_l_path,
            &req.device,
            dtype,
        )?;
        let text_encoder_g = if variant.is_xl() {
            let cfg_g = cfg
                .clip2
                .as_ref()
                .ok_or_else(|| anyhow!("SDXL config is missing clip2"))?;
            let p = text_enc_g_path
                .as_ref()
                .ok_or_else(|| anyhow!("missing text_encoder_2 path"))?;
            Some(stable_diffusion::build_clip_transformer(
                cfg_g,
                p,
                &req.device,
                dtype,
            )?)
        } else {
            None
        };

        let vae = cfg.build_vae(&vae_path, &req.device, dtype)?;

        // LoRAs: merge into a temp UNet safetensors, then build_unet from that path.
        let (effective_unet_path, lora_tmp) = if req.loras.is_empty() {
            (unet_path.clone(), None)
        } else {
            let resolve_spinner = progress::spinner("Resolving LoRA file(s)");
            let mut resolved = Vec::with_capacity(req.loras.len());
            for spec in &req.loras {
                resolved.push(spec.resolve().await?);
            }
            resolve_spinner
                .finish_with_message(format!("✓ resolved {} LoRA file(s)", resolved.len()));

            let lora_spinner = progress::spinner("Merging LoRA into UNet");
            let tmp = tempfile::Builder::new()
                .prefix("plakat-merged-unet-")
                .suffix(".safetensors")
                .tempfile()?;
            let (modified, targets) = crate::pipelines::lora::merge_loras_into_unet(
                &unet_path,
                tmp.path(),
                &resolved,
                req.lora_scale,
                &req.device,
            )?;
            lora_spinner.finish_with_message(format!(
                "✓ merged {modified}/{targets} LoRA target(s) into UNet"
            ));
            (tmp.path().to_path_buf(), Some(tmp))
        };
        let unet = cfg.build_unet(&effective_unet_path, &req.device, 4, false, dtype)?;
        build.finish_with_message("✓ models loaded");

        Ok(Self {
            variant,
            repo,
            cfg,
            tokenizer_l,
            tokenizer_g,
            text_encoder_l,
            text_encoder_g,
            vae,
            unet,
            device: req.device,
            dtype,
            _lora_tmp: lora_tmp,
        })
    }

    /// Encode `prompt` (and optionally `negative` for CFG) into the
    /// `encoder_hidden_states` tensor the UNet expects.
    fn encode_prompt(&self, prompt: &str, negative: &str, do_cfg: bool) -> Result<Tensor> {
        if self.variant.is_xl() {
            self.encode_xl(prompt, negative, do_cfg)
        } else {
            self.encode_single(prompt, negative, do_cfg)
        }
    }

    fn encode_single(&self, prompt: &str, negative: &str, do_cfg: bool) -> Result<Tensor> {
        let cond_ids = tokenize_padded(&self.tokenizer_l, &self.cfg.clip, prompt, &self.device)?;
        let cond = self.text_encoder_l.forward(&cond_ids)?;
        if !do_cfg {
            return Ok(cond.to_dtype(self.dtype)?);
        }
        let uncond_ids =
            tokenize_padded(&self.tokenizer_l, &self.cfg.clip, negative, &self.device)?;
        let uncond = self.text_encoder_l.forward(&uncond_ids)?;
        Ok(Tensor::cat(&[&uncond, &cond], 0)?.to_dtype(self.dtype)?)
    }

    fn encode_xl(&self, prompt: &str, negative: &str, do_cfg: bool) -> Result<Tensor> {
        let cfg_g = self
            .cfg
            .clip2
            .as_ref()
            .ok_or_else(|| anyhow!("SDXL config is missing clip2"))?;
        let tok_g = self
            .tokenizer_g
            .as_ref()
            .ok_or_else(|| anyhow!("SDXL missing tokenizer_g"))?;
        let enc_g = self
            .text_encoder_g
            .as_ref()
            .ok_or_else(|| anyhow!("SDXL missing text_encoder_g"))?;

        let cond = embed_xl(
            prompt,
            &self.tokenizer_l,
            tok_g,
            &self.cfg.clip,
            cfg_g,
            &self.text_encoder_l,
            enc_g,
            &self.device,
        )?;
        if !do_cfg {
            return Ok(cond.to_dtype(self.dtype)?);
        }
        let uncond = embed_xl(
            negative,
            &self.tokenizer_l,
            tok_g,
            &self.cfg.clip,
            cfg_g,
            &self.text_encoder_l,
            enc_g,
            &self.device,
        )?;
        Ok(Tensor::cat(&[&uncond, &cond], 0)?.to_dtype(self.dtype)?)
    }

    /// Generate `req.count` images for one prompt. Reuses the loaded
    /// UNet/VAE/text encoder.
    pub fn generate(&self, req: &GenRequest) -> Result<()> {
        crate::pipelines::scheduler::check_device_support(req.scheduler, &self.device)?;
        std::fs::create_dir_all(&req.out_dir)
            .with_context(|| format!("creating output dir {}", req.out_dir.display()))?;

        let (w, h) = (req.width as usize, req.height as usize);
        let do_cfg = req.guidance > 1.0;
        let text_embeddings = self.encode_prompt(&req.prompt, &req.negative, do_cfg)?;

        let bsz: usize = 1;
        let latent_h = h / 8;
        let latent_w = w / 8;
        let vae_scale = self.variant.vae_scale();

        for idx in 0..req.count {
            let seed = req
                .seed
                .map(|s| s + idx as u64)
                .unwrap_or_else(rand::random)
                & (u32::MAX as u64);
            if let Err(e) = self.device.set_seed(seed) {
                tracing::debug!(
                    target: "plakat",
                    "set_seed not supported ({e}); using global RNG"
                );
            }

            let mut scheduler =
                crate::pipelines::scheduler::build(req.scheduler, &self.cfg, req.steps)?;
            let timesteps = scheduler.timesteps().to_vec();

            let mut latents =
                Tensor::randn(0f32, 1f32, (bsz, 4, latent_h, latent_w), &self.device)?
                    .to_dtype(self.dtype)?;
            latents = (latents * scheduler.init_noise_sigma())?;

            let bar = progress::step_bar(
                timesteps.len() as u64,
                &format!("img {}/{}", idx + 1, req.count),
            );
            for &timestep in &timesteps {
                latents = self.denoise_step(
                    &latents,
                    timestep,
                    &text_embeddings,
                    &mut scheduler,
                    req.guidance,
                    do_cfg,
                )?;
                bar.inc(1);
                bar.set_message(format!("t={timestep} seed={seed}"));
            }
            bar.finish_and_clear();

            // Optional polish pass: img2img with same model at low strength.
            if let Some(rsteps) = req.refine {
                if rsteps > 0 {
                    let strength = req.refine_strength.clamp(0.0, 1.0);
                    let mut polish = crate::pipelines::scheduler::build(
                        req.scheduler,
                        &self.cfg,
                        rsteps,
                    )?;
                    let pts = polish.timesteps().to_vec();
                    let init_skip = ((rsteps as f32) * (1.0 - strength)).round() as usize;
                    let init_skip = init_skip.min(rsteps.saturating_sub(1));
                    let active = &pts[init_skip..];
                    if let Some(&start_t) = active.first() {
                        let noise = Tensor::randn(0f32, 1f32, latents.shape(), &self.device)?
                            .to_dtype(self.dtype)?;
                        latents = polish.add_noise(&latents, noise, start_t)?;
                        let rbar = progress::step_bar(
                            active.len() as u64,
                            &format!("polish {}/{}", idx + 1, req.count),
                        );
                        for &timestep in active {
                            latents = self.denoise_step(
                                &latents,
                                timestep,
                                &text_embeddings,
                                &mut polish,
                                req.guidance,
                                do_cfg,
                            )?;
                            rbar.inc(1);
                            rbar.set_message(format!("polish t={timestep}"));
                        }
                        rbar.finish_and_clear();
                    }
                }
            }

            // VAE decode + save
            let image = self.vae.decode(&(&latents / vae_scale)?)?;
            let image = ((image / 2.0)? + 0.5)?.clamp(0f32, 1f32)?;
            let image = (image * 255.0)?
                .to_dtype(DType::U8)?
                .i(0)?
                .permute((1, 2, 0))?;
            let (oh, ow, _) = image.dims3()?;
            let buf = image.flatten_all()?.to_vec1::<u8>()?;
            let out_path = req.out_dir.join(format!("plakat-{seed}.png"));
            crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, &out_path)?;
            crate::ui::progress::println(&format!("→ {}", out_path.display()));
        }
        Ok(())
    }

    fn denoise_step(
        &self,
        latents: &Tensor,
        timestep: usize,
        text_embeddings: &Tensor,
        scheduler: &mut Box<dyn stable_diffusion::schedulers::Scheduler>,
        guidance: f64,
        do_cfg: bool,
    ) -> Result<Tensor> {
        let latent_in = if do_cfg {
            Tensor::cat(&[latents, latents], 0)?
        } else {
            latents.clone()
        };
        let latent_in = scheduler.scale_model_input(latent_in, timestep)?;
        let noise_pred = self
            .unet
            .forward(&latent_in, timestep as f64, text_embeddings)?;
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

// =====================================================================
// Tokenization + SDXL embedding helpers (used by Pipeline methods).
// =====================================================================

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

#[allow(clippy::too_many_arguments)]
fn embed_xl(
    text: &str,
    tok_l: &Tokenizer,
    tok_g: &Tokenizer,
    cfg_l: &sdclip::Config,
    cfg_g: &sdclip::Config,
    enc_l: &sdclip::ClipTextTransformer,
    enc_g: &sdclip::ClipTextTransformer,
    device: &Device,
) -> Result<Tensor> {
    let ids_l = tokenize_padded(tok_l, cfg_l, text, device)?;
    let ids_g = tokenize_padded(tok_g, cfg_g, text, device)?;
    let (_final_l, hidden_l) = enc_l.forward_until_encoder_layer(&ids_l, usize::MAX, -2)?;
    let (_final_g, hidden_g) = enc_g.forward_until_encoder_layer(&ids_g, usize::MAX, -2)?;
    Tensor::cat(&[&hidden_l, &hidden_g], 2).map_err(Into::into)
}

// =====================================================================
// Public entry point — single-shot wrapper for back-compat with
// `plakat generate` and any direct caller.
// =====================================================================

pub async fn run(req: Request) -> Result<()> {
    let variant = Variant::detect(&req.model);

    // Flux routes to its own pipeline; LoRAs are not supported there yet.
    if variant.is_flux() {
        if !req.loras.is_empty() {
            tracing::warn!(target: "plakat",
                "ignoring {} LoRA file(s): kohya SD LoRAs don't apply to Flux's transformer",
                req.loras.len()
            );
        }
        use crate::pipelines::flux;
        let fvar = if matches!(variant, Variant::FluxDev) {
            flux::Variant::Dev
        } else {
            flux::Variant::Schnell
        };
        return flux::run(flux::Request {
            prompt: req.prompt,
            variant: fvar,
            repo: resolve_repo(&req.model),
            width: req.width,
            height: req.height,
            count: req.count,
            steps: if req.steps == 28 { None } else { Some(req.steps) },
            guidance: if (req.guidance - 7.5).abs() < f64::EPSILON {
                None
            } else {
                Some(req.guidance)
            },
            seed: req.seed,
            out_dir: req.out_dir,
            device: req.device,
        })
        .await;
    }

    let pipeline = Pipeline::load(LoadRequest {
        model: req.model,
        device: req.device,
        loras: req.loras,
        lora_scale: req.lora_scale,
    })
    .await?;

    pipeline.generate(&GenRequest {
        prompt: req.prompt,
        negative: req.negative,
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
    })
}
