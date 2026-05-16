//! Text-to-image inference pipeline.
//!
//! Supported in candle 0.8:
//!   * SD 1.5 / 2.1            — single CLIP-L text encoder, VAE scale 0.18215
//!   * SDXL / SDXL-Turbo       — dual encoder (CLIP-L + CLIP-G), penultimate
//!                               hidden states concatenated to 2048 channels;
//!                               VAE scale 0.13025
//!
//! Flux is detected but errors out — it's a different architecture and lives
//! in a separate candle module.

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_transformers::models::stable_diffusion::{
    self, StableDiffusionConfig, clip as sdclip,
};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use crate::ui::progress;

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
    pub loras: Vec<crate::pipelines::lora::LoraSpec>,
    pub lora_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    Sd15,
    Sd21,
    Sdxl,
    SdxlTurbo,
    FluxSchnell,
    FluxDev,
}

impl Variant {
    fn detect(model: &str) -> Self {
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
                "Flux variants route through pipelines::flux::run before reaching here"
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

    /// VAE latent scaling. SDXL/Turbo retrained their VAE with a different factor.
    fn vae_scale(self) -> f64 {
        match self {
            Self::Sdxl | Self::SdxlTurbo => 0.13025,
            _ => 0.18215,
        }
    }

    fn is_xl(self) -> bool {
        matches!(self, Self::Sdxl | Self::SdxlTurbo)
    }
    #[allow(dead_code)]
    fn is_flux(self) -> bool {
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

/// Try several candidate file names within a repo; return the first that downloads.
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

pub async fn run(req: Request) -> Result<()> {
    let variant = Variant::detect(&req.model);
    let repo = resolve_repo(&req.model);

    if matches!(variant, Variant::FluxSchnell | Variant::FluxDev) {
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
            repo,
            width: req.width,
            height: req.height,
            count: req.count,
            // User-provided values override variant defaults only if they
            // diverge from t2i's own defaults (28 steps / 7.5 guidance).
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

    let (w, h) = (req.width as usize, req.height as usize);
    let cfg = variant.config(w, h)?;
    let dtype = variant.dtype(&req.device);
    let do_cfg = req.guidance > 1.0;

    let mp = progress::multi();

    // ---- download weights ----
    let dl = progress::spinner(&mp, &format!("Resolving weights for {repo}"));

    // Legacy SD repos ship vocab.json + merges.txt instead of the consolidated
    // tokenizer.json. The OpenAI CLIP tokenizer is bit-for-bit identical to
    // what every SD/SDXL encoder uses, so we use it as a universal fallback.
    let tokenizer_l = crate::hf::download::get_first_of(&[
        (&repo, "tokenizer/tokenizer.json"),
        ("openai/clip-vit-large-patch14", "tokenizer.json"),
    ])
    .await
    .with_context(|| format!("tokenizer (CLIP-L) for {repo}"))?;
    let text_enc_l = fetch_first(
        &repo,
        &[
            "text_encoder/model.fp16.safetensors",
            "text_encoder/model.safetensors",
        ],
    )
    .await
    .with_context(|| format!("text_encoder weights in {repo}"))?;

    let (tokenizer_g, text_enc_g) = if variant.is_xl() {
        let t = crate::hf::download::get_first_of(&[
            (&repo, "tokenizer_2/tokenizer.json"),
            (
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                "tokenizer.json",
            ),
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

    // ---- load models + text embeddings ----
    let build = progress::spinner(&mp, "Loading models");

    let tok_l =
        Tokenizer::from_file(&tokenizer_l).map_err(|e| anyhow!("tokenizer (CLIP-L): {e}"))?;
    let text_embeddings = if variant.is_xl() {
        let tok_g = Tokenizer::from_file(tokenizer_g.as_ref().unwrap())
            .map_err(|e| anyhow!("tokenizer (CLIP-G): {e}"))?;
        encode_prompt_xl(
            &tok_l,
            &tok_g,
            &req.prompt,
            &req.negative,
            &cfg,
            &text_enc_l,
            text_enc_g.as_ref().unwrap(),
            &req.device,
            dtype,
            do_cfg,
        )?
    } else {
        encode_prompt_single(
            &tok_l,
            &req.prompt,
            &req.negative,
            &cfg,
            &text_enc_l,
            &req.device,
            dtype,
            do_cfg,
        )?
    };

    let vae = cfg.build_vae(&vae_path, &req.device, dtype)?;

    // If LoRA(s) requested: resolve (download from HF if needed), merge into a
    // temp UNet safetensors, feed that path into build_unet. The temp file is
    // dropped when `_lora_tmp` goes out of scope at the end of `run`.
    let (effective_unet_path, _lora_tmp) = if req.loras.is_empty() {
        (unet_path.clone(), None)
    } else {
        let resolve_spinner = progress::spinner(&mp, "Resolving LoRA file(s)");
        let mut resolved = Vec::with_capacity(req.loras.len());
        for spec in &req.loras {
            resolved.push(spec.resolve().await?);
        }
        resolve_spinner.finish_with_message(format!(
            "✓ resolved {} LoRA file(s)",
            resolved.len()
        ));

        let lora_spinner = progress::spinner(&mp, "Merging LoRA into UNet");
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

    // ---- generation loop ----
    let bsz: usize = 1;
    let latent_h = h / 8;
    let latent_w = w / 8;
    let vae_scale = variant.vae_scale();

    for idx in 0..req.count {
        let seed = req
            .seed
            .map(|s| s + idx as u64)
            .unwrap_or_else(rand::random)
            & (u32::MAX as u64);
        req.device
            .set_seed(seed)
            .map_err(|e| anyhow!("set_seed: {e}"))?;

        let mut scheduler = cfg.build_scheduler(req.steps)?;
        let timesteps = scheduler.timesteps().to_vec();

        let mut latents = Tensor::randn(0f32, 1f32, (bsz, 4, latent_h, latent_w), &req.device)?
            .to_dtype(dtype)?;
        latents = (latents * scheduler.init_noise_sigma())?;

        let bar = progress::step_bar(
            &mp,
            timesteps.len() as u64,
            &format!("img {}/{}", idx + 1, req.count),
        );

        for &timestep in &timesteps {
            let latent_in = if do_cfg {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };
            let latent_in = scheduler.scale_model_input(latent_in, timestep)?;
            let noise_pred = unet.forward(&latent_in, timestep as f64, &text_embeddings)?;
            let noise_pred = if do_cfg {
                let chunks = noise_pred.chunk(2, 0)?;
                let uncond = &chunks[0];
                let text = &chunks[1];
                (uncond + ((text - uncond)? * req.guidance)?)?
            } else {
                noise_pred
            };
            latents = scheduler.step(&noise_pred, timestep, &latents)?;
            bar.inc(1);
            bar.set_message(format!("t={timestep} seed={seed}"));
        }
        bar.finish_and_clear();

        let image = vae.decode(&(&latents / vae_scale)?)?;
        let image = ((image / 2.0)? + 0.5)?.clamp(0f32, 1f32)?;
        let image = (image * 255.0)?.to_dtype(DType::U8)?.i(0)?.permute((1, 2, 0))?;
        let (oh, ow, _) = image.dims3()?;
        let buf = image.flatten_all()?.to_vec1::<u8>()?;

        let out_path = req.out_dir.join(format!("plakat-{seed}.png"));
        crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, &out_path)?;
        tracing::info!(target: "plakat", "→ {}", out_path.display());
    }

    Ok(())
}

/// Build a (max_position_embeddings,) token-id tensor padded with `pad_id`.
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

/// SD 1.5 / 2.1 — single CLIP-L encoder, final-layer hidden states.
fn encode_prompt_single(
    tokenizer: &Tokenizer,
    prompt: &str,
    negative: &str,
    cfg: &StableDiffusionConfig,
    weights: &Path,
    device: &Device,
    dtype: DType,
    do_cfg: bool,
) -> Result<Tensor> {
    let cond_ids = tokenize_padded(tokenizer, &cfg.clip, prompt, device)?;
    let text_encoder =
        stable_diffusion::build_clip_transformer(&cfg.clip, weights, device, dtype)?;
    let cond = text_encoder.forward(&cond_ids)?;
    if !do_cfg {
        return Ok(cond.to_dtype(dtype)?);
    }
    let uncond_ids = tokenize_padded(tokenizer, &cfg.clip, negative, device)?;
    let uncond = text_encoder.forward(&uncond_ids)?;
    Ok(Tensor::cat(&[&uncond, &cond], 0)?.to_dtype(dtype)?)
}

/// SDXL / SDXL-Turbo — dual encoder, concat penultimate hidden states on channel dim.
///
/// Returns a tensor of shape:
///   (B, max_pos, 768 + 1280 = 2048) with B = 2 if CFG, else 1.
///
/// Note: candle 0.8's UNet does not consume the SDXL `add_embedding`
/// (pooled CLIP-G output + time_ids micro-conditioning). The model still
/// produces reasonable output from token-level features alone; quality is
/// slightly below the diffusers reference.
fn encode_prompt_xl(
    tok_l: &Tokenizer,
    tok_g: &Tokenizer,
    prompt: &str,
    negative: &str,
    cfg: &StableDiffusionConfig,
    weights_l: &Path,
    weights_g: &Path,
    device: &Device,
    dtype: DType,
    do_cfg: bool,
) -> Result<Tensor> {
    let cfg_l = &cfg.clip;
    let cfg_g = cfg
        .clip2
        .as_ref()
        .ok_or_else(|| anyhow!("SDXL config is missing clip2"))?;

    let enc_l = stable_diffusion::build_clip_transformer(cfg_l, weights_l, device, dtype)?;
    let enc_g = stable_diffusion::build_clip_transformer(cfg_g, weights_g, device, dtype)?;

    // (B, 77, 768+1280)
    let cond = embed_xl(prompt, tok_l, tok_g, cfg_l, cfg_g, &enc_l, &enc_g, device)?;
    if !do_cfg {
        return Ok(cond.to_dtype(dtype)?);
    }
    let uncond = embed_xl(negative, tok_l, tok_g, cfg_l, cfg_g, &enc_l, &enc_g, device)?;
    Ok(Tensor::cat(&[&uncond, &cond], 0)?.to_dtype(dtype)?)
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

    // SDXL uses the penultimate (-2) hidden states of both encoders.
    let (_final_l, hidden_l) = enc_l.forward_until_encoder_layer(&ids_l, usize::MAX, -2)?;
    let (_final_g, hidden_g) = enc_g.forward_until_encoder_layer(&ids_g, usize::MAX, -2)?;

    // (B, 77, 768) ⊕ (B, 77, 1280) → (B, 77, 2048)
    Tensor::cat(&[&hidden_l, &hidden_g], 2).map_err(Into::into)
}
