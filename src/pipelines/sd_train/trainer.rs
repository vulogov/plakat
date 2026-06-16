//! SD 1.5 / SDXL style-LoRA trainer (Phase 1: SD 1.5).
//!
//! Mirrors `sd3::train_style_lora` but for the UNet: a **DDPM-epsilon**
//! objective with CLIP-L conditioning, training the vendored LoRA-wired
//! UNet. Mixed precision (BF16 base + F32 LoRA — the LoraLinear forward
//! casts). Output is a **kohya**-format `.safetensors` (`lora_unet_…`).
use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::unet::UNet2DConditionModel;
use crate::pipelines::sd_core::{SdCore, SdLoadRequest};
use candle_transformers::models::stable_diffusion::unet_2d::{
    BlockConfig, UNet2DConditionModelConfig,
};

/// SD 1.5 UNet config (candle's `v1_5` values; `cfg.unet` is private).
pub(crate) fn sd15_unet_config() -> UNet2DConditionModelConfig {
    let bc = |out_channels, use_cross_attn, attention_head_dim| BlockConfig {
        out_channels,
        use_cross_attn,
        attention_head_dim,
    };
    UNet2DConditionModelConfig {
        blocks: vec![
            bc(320, Some(1), 8),
            bc(640, Some(1), 8),
            bc(1280, Some(1), 8),
            bc(1280, None, 8),
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

/// Inputs for [`train_style_lora_sd`].
pub struct SdStyleTrainRequest {
    pub model: String,
    pub device: Device,
    pub images: Vec<PathBuf>,
    pub trigger: String,
    pub rank: usize,
    pub steps: usize,
    pub lr: f64,
    pub size: u32,
    pub out: PathBuf,
    /// Explicit checkpoint interval in steps. `None` → ~10 evenly-spaced
    /// (see [`checkpoint_interval`]). `0` is treated as `None`.
    pub checkpoint_every: Option<usize>,
    /// Log a progress line every N steps (min 1).
    pub log_every: usize,
    /// Resume from a checkpoint (a kohya LoRA written by an earlier run). The
    /// adapters are initialized from it and the step counter continues from the
    /// checkpoint's step (parsed from `…-step<N>.safetensors`), so training runs
    /// up to `steps`. `None` = train from scratch (the default). Additive: when
    /// unset the loop is identical to before.
    pub resume_from: Option<PathBuf>,
}

/// SD 1.5 scaled-linear beta schedule → cumulative alphas (length 1000).
fn alphas_cumprod() -> Vec<f64> {
    let (n, bs, be) = (1000usize, 0.00085f64, 0.012f64);
    let mut acc = 1.0;
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let beta = (bs.sqrt() * (1.0 - t) + be.sqrt() * t).powi(2);
            acc *= 1.0 - beta;
            acc
        })
        .collect()
}

/// SDXL UNet config (candle's `sdxl` values; cfg.unet is private).
pub(crate) fn sdxl_unet_config() -> UNet2DConditionModelConfig {
    let bc = |out_channels, use_cross_attn, attention_head_dim| BlockConfig {
        out_channels,
        use_cross_attn,
        attention_head_dim,
    };
    UNet2DConditionModelConfig {
        blocks: vec![bc(320, None, 5), bc(640, Some(2), 10), bc(1280, Some(10), 20)],
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

/// Train a style LoRA on the SD UNet attention; write a kohya safetensors.
pub async fn train_style_lora_sd(req: SdStyleTrainRequest) -> Result<()> {
    if req.model.contains("sdxl") {
        return train_sdxl(req).await;
    }
    let device = req.device.clone();
    let dtype = DType::BF16; // training base dtype; LoRA Vars stay F32

    // --- Phase A: load SD, encode images + trigger, capture cfg + repo, drop.
    tracing::info!(
        "sd-style-train: encoding {} image(s) + caption \"{}\"",
        req.images.len(),
        req.trigger
    );
    let (latents, text_emb, base_repo) = {
        let core = SdCore::load(SdLoadRequest {
            model: req.model.clone(),
            device: device.clone(),
            loras: Vec::new(),
            lora_scale: 1.0,
            embeddings: Vec::new(),
            vae_cache: None,
        })
        .await?;
        let scale = core.variant.vae_scale();
        let text_emb = crate::pipelines::t2i::encode_with_attention(
            &core.tokenizer_l,
            &core.cfg.clip,
            &core.text_encoder_l,
            &req.trigger,
            1,
            &device,
            dtype,
        )?
        .to_dtype(dtype)?; // encode_with_attention's simple path keeps the encoder dtype (F16)
        let mut latents = Vec::with_capacity(req.images.len());
        for img in &req.images {
            let px = crate::imaging::preprocess::sd_image_tensor(
                img.as_path(),
                req.size,
                req.size,
                &device,
                core.dtype,
            )?;
            let z = core.vae.encode(&px)?.sample()?;
            let lat = (z * scale)?.to_dtype(dtype)?;
            latents.push(lat);
        }
        let base_repo = if req.model.contains('/') {
            req.model.clone()
        } else {
            crate::hf::resolve_alias(&req.model).to_string()
        };
        (latents, text_emb, base_repo)
    };

    // --- Phase B: load the vendored UNet (BF16) + install adapters.
    tracing::info!("sd-style-train: loading UNet for training");
    let unet_path = crate::hf::download::get_first_of(&[
        (&base_repo, "unet/diffusion_pytorch_model.fp16.safetensors"),
        (&base_repo, "unet/diffusion_pytorch_model.safetensors"),
    ])
    .await?;
    let paths = [unet_path];
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, &device)? };
    let unet = UNet2DConditionModel::new(vb, 4, 4, false, sd15_unet_config(), None)?;
    let adapters = unet.install_train_adapters(req.rank, 1.0, &device)?;
    tracing::info!(
        "sd-style-train: {} trainable attention adapters (rank {})",
        adapters.len(),
        req.rank
    );
    let vars: Vec<Var> = adapters
        .iter()
        .flat_map(|(_, a, b)| [a.clone(), b.clone()])
        .collect();
    let mut opt = AdamW::new(vars.clone(), ParamsAdamW { lr: req.lr, ..Default::default() })?;

    // --- Phase C: DDPM-epsilon loop. x_t = √ᾱ·x0 + √(1-ᾱ)·ε; predict ε.
    let abar = alphas_cumprod();
    let n = latents.len().max(1);
    let start_step =
        resume_start_step(&req.resume_from, &adapters, &device, req.steps, "sd-style-train")?;
    let mut progress = crate::pipelines::train_progress::TrainProgress::new(
        req.steps,
        req.lr,
        checkpoint_interval(req.checkpoint_every, req.steps),
    );
    for step in start_step..req.steps {
        let x0 = &latents[step % n];
        let noise = Tensor::randn(0f32, 1f32, x0.dims(), &device)?.to_dtype(dtype)?;
        let t = (Tensor::rand(0f32, 1f32, (1usize,), &device)?.to_vec1::<f32>()?[0] * 999.0) as usize;
        let a = abar[t];
        let x_t = ((x0 * a.sqrt())? + (&noise * (1.0 - a).sqrt())?)?;
        let pred = unet.forward(&x_t, t as f64, &text_emb)?;
        let loss = (&pred - &noise)?.sqr()?.mean_all()?.to_dtype(DType::F32)?;
        let mut grads = loss.backward()?;
        crate::pipelines::lora_linear::clip_grad_norm(&mut grads, &vars, 1.0)?;
        opt.step(&grads)?;
        if step % req.log_every.max(1) == 0 || step + 1 == req.steps {
            tracing::info!(
                "{}",
                progress.line("sd-style-train", step + 1, loss.to_scalar::<f32>()?)
            );
        }
        if (step + 1) % checkpoint_interval(req.checkpoint_every, req.steps) == 0
            && step + 1 != req.steps
        {
            let ckpt = checkpoint_path(&req.out, step + 1);
            save_kohya_lora(&adapters, req.rank, &ckpt)?;
            tracing::info!("sd-style-train: checkpoint @ step {} → {}", step + 1, ckpt.display());
        }
    }
    save_kohya_lora(&adapters, req.rank, &req.out)?;
    tracing::info!("sd-style-train: wrote {}", req.out.display());
    tracing::info!("{}", progress.finish("sd-style-train", &req.out));
    Ok(())
}

/// SDXL branch — dual-CLIP conditioning (hidden 2048 + pooled 1280) +
/// add_time_ids, trains the SDXL UNet via forward_sdxl. Same DDPM-epsilon
/// loop + kohya save as SD 1.5.
async fn train_sdxl(req: SdStyleTrainRequest) -> Result<()> {
    use crate::pipelines::sdxl_unet::SdxlAddEmbedConfig;
    let device = req.device.clone();
    let dtype = DType::BF16;

    tracing::info!(
        "sdxl-style-train: encoding {} image(s) + caption \"{}\"",
        req.images.len(),
        req.trigger
    );
    let (latents, hidden, pooled, add_time_ids, base_repo) = {
        let pipe = crate::pipelines::t2i::Pipeline::load(crate::pipelines::t2i::LoadRequest {
            model: req.model.clone(),
            device: device.clone(),
            loras: Vec::new(),
            lora_scale: 1.0,
            use_refiner: false,
            embeddings: Vec::new(),
            vae_cache: None,
        })
        .await?;
        let (hidden, pooled_opt) = pipe.encode_prompt(&req.trigger, "", false, 1)?;
        let hidden = hidden.to_dtype(dtype)?;
        let pooled = pooled_opt
            .ok_or_else(|| anyhow::anyhow!("SDXL encode returned no pooled embedding"))?
            .to_dtype(dtype)?;
        let core = pipe.core();
        let scale = core.variant.vae_scale();
        let mut latents = Vec::with_capacity(req.images.len());
        for img in &req.images {
            let px = crate::imaging::preprocess::sd_image_tensor(
                img.as_path(),
                req.size,
                req.size,
                &device,
                core.dtype,
            )?;
            let z = core.vae.encode(&px)?.sample()?;
            latents.push((z * scale)?.to_dtype(dtype)?);
        }
        let add_time_ids =
            crate::pipelines::sdxl_unet::build_add_time_ids_base(req.size, req.size, &device, dtype)?;
        let base_repo = if req.model.contains('/') {
            req.model.clone()
        } else {
            crate::hf::resolve_alias(&req.model).to_string()
        };
        (latents, hidden, pooled, add_time_ids, base_repo)
    };

    tracing::info!("sdxl-style-train: loading UNet for training");
    let unet_path = crate::hf::download::get_first_of(&[
        (&base_repo, "unet/diffusion_pytorch_model.fp16.safetensors"),
        (&base_repo, "unet/diffusion_pytorch_model.safetensors"),
    ])
    .await?;
    let paths = [unet_path];
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, &device)? };
    let unet = UNet2DConditionModel::new(
        vb,
        4,
        4,
        false,
        sdxl_unet_config(),
        Some(SdxlAddEmbedConfig::base()),
    )?;
    let adapters = unet.install_train_adapters(req.rank, 1.0, &device)?;
    tracing::info!(
        "sdxl-style-train: {} trainable attention adapters (rank {})",
        adapters.len(),
        req.rank
    );
    let vars: Vec<Var> = adapters
        .iter()
        .flat_map(|(_, a, b)| [a.clone(), b.clone()])
        .collect();
    let mut opt = AdamW::new(vars.clone(), ParamsAdamW { lr: req.lr, ..Default::default() })?;

    let abar = alphas_cumprod();
    let n = latents.len().max(1);
    let start_step =
        resume_start_step(&req.resume_from, &adapters, &device, req.steps, "sdxl-style-train")?;
    let mut progress = crate::pipelines::train_progress::TrainProgress::new(
        req.steps,
        req.lr,
        checkpoint_interval(req.checkpoint_every, req.steps),
    );
    for step in start_step..req.steps {
        let x0 = &latents[step % n];
        let noise = Tensor::randn(0f32, 1f32, x0.dims(), &device)?.to_dtype(dtype)?;
        let t = (Tensor::rand(0f32, 1f32, (1usize,), &device)?.to_vec1::<f32>()?[0] * 999.0) as usize;
        let a = abar[t];
        let x_t = ((x0 * a.sqrt())? + (&noise * (1.0 - a).sqrt())?)?;
        let pred = unet.forward_sdxl(&x_t, t as f64, &hidden, &pooled, &add_time_ids)?;
        let loss = (&pred - &noise)?.sqr()?.mean_all()?.to_dtype(DType::F32)?;
        let mut grads = loss.backward()?;
        crate::pipelines::lora_linear::clip_grad_norm(&mut grads, &vars, 1.0)?;
        opt.step(&grads)?;
        if step % req.log_every.max(1) == 0 || step + 1 == req.steps {
            tracing::info!(
                "{}",
                progress.line("sdxl-style-train", step + 1, loss.to_scalar::<f32>()?)
            );
        }
        if (step + 1) % checkpoint_interval(req.checkpoint_every, req.steps) == 0
            && step + 1 != req.steps
        {
            let ckpt = checkpoint_path(&req.out, step + 1);
            save_kohya_lora(&adapters, req.rank, &ckpt)?;
            tracing::info!("sdxl-style-train: checkpoint @ step {} → {}", step + 1, ckpt.display());
        }
    }
    save_kohya_lora(&adapters, req.rank, &req.out)?;
    tracing::info!("sdxl-style-train: wrote {}", req.out.display());
    tracing::info!("{}", progress.finish("sdxl-style-train", &req.out));
    Ok(())
}

/// Path for a periodic checkpoint at `step`: **numbered by default**
/// (`<stem>-step<N>.<ext>`), so a run keeps every checkpoint and you can sweep
/// for the best step after the fact — the best LoRA is rarely the last step
/// (style training over-cooks). Set `PLAKAT_TRAIN_SINGLE_FILE=1` to instead
/// overwrite the plain `--out` each interval (one file, no sweep). The final
/// save always writes the plain `--out`.
fn checkpoint_path(out: &Path, step: usize) -> PathBuf {
    if std::env::var_os("PLAKAT_TRAIN_SINGLE_FILE").is_some() {
        return out.to_path_buf();
    }
    let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("lora");
    let ext = out.extension().and_then(|s| s.to_str()).unwrap_or("safetensors");
    out.with_file_name(format!("{stem}-step{step}.{ext}"))
}

/// Resolve the checkpoint interval in steps: an explicit `--checkpoint-every`
/// (`every`, a positive value) wins; otherwise ~10 evenly-spaced (min every
/// 30). 900 steps → every 90; 90 steps → every 30.
fn checkpoint_interval(every: Option<usize>, total_steps: usize) -> usize {
    every
        .filter(|&n| n > 0)
        .unwrap_or_else(|| (total_steps / 10).max(30))
}

/// Write trained adapters as a kohya SD LoRA: `lora_unet_<slug>.lora_down
/// .weight` / `.lora_up.weight` / `.alpha`, slug = registry key minus
/// `.weight`, dots→underscores. lora_down = A, lora_up = B, alpha = rank
/// (so the loader's alpha/rank = the training scale of 1.0).
fn save_kohya_lora(adapters: &[(String, Var, Var)], rank: usize, out: &Path) -> Result<()> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let alpha = Tensor::new(rank as f32, &Device::Cpu)?;
    for (key, a, b) in adapters {
        let logical = key.strip_suffix(".weight").unwrap_or(key);
        let slug = format!("lora_unet_{}", logical.replace('.', "_"));
        let a_t = a.as_tensor().to_dtype(DType::F16)?.to_device(&Device::Cpu)?;
        let b_t = b.as_tensor().to_dtype(DType::F16)?.to_device(&Device::Cpu)?;
        tensors.insert(format!("{slug}.lora_down.weight"), a_t);
        tensors.insert(format!("{slug}.lora_up.weight"), b_t);
        tensors.insert(format!("{slug}.alpha"), alpha.clone());
    }
    candle_core::safetensors::save(&tensors, out)?;
    Ok(())
}

/// Parse the step number from a checkpoint filename written by
/// [`checkpoint_path`] (`<stem>-step<N>.<ext>`). `None` if the name carries no
/// step (e.g. resuming from the final, no-suffix output) — caller defaults to 0.
pub(crate) fn parse_resume_step(path: &Path) -> Option<usize> {
    let stem = path.file_stem().and_then(|s| s.to_str())?;
    let idx = stem.rfind("-step")?;
    stem[idx + "-step".len()..].parse::<usize>().ok()
}

/// Load a kohya LoRA checkpoint (written by [`save_kohya_lora`]) back into the
/// live training adapters — the inverse of the save. Used by `--resume`. Errors
/// if a tensor is missing (a rank / base-model mismatch with the current run).
fn load_kohya_into_adapters(
    adapters: &[(String, Var, Var)],
    path: &Path,
    device: &Device,
) -> Result<()> {
    let loaded = candle_core::safetensors::load(path, device)
        .with_context(|| format!("loading resume checkpoint {}", path.display()))?;
    for (key, a, b) in adapters {
        let logical = key.strip_suffix(".weight").unwrap_or(key);
        let slug = format!("lora_unet_{}", logical.replace('.', "_"));
        let down = loaded.get(&format!("{slug}.lora_down.weight")).ok_or_else(|| {
            anyhow!("resume: checkpoint missing {slug}.lora_down.weight (rank/base mismatch?)")
        })?;
        let up = loaded
            .get(&format!("{slug}.lora_up.weight"))
            .ok_or_else(|| anyhow!("resume: checkpoint missing {slug}.lora_up.weight"))?;
        a.set(&down.to_dtype(a.as_tensor().dtype())?)?;
        b.set(&up.to_dtype(b.as_tensor().dtype())?)?;
    }
    Ok(())
}

/// Shared `--resume` handling: load the checkpoint into the adapters and return
/// the step to continue from (clamped below `steps`). `None` request → 0.
fn resume_start_step(
    resume_from: &Option<PathBuf>,
    adapters: &[(String, Var, Var)],
    device: &Device,
    steps: usize,
    tag: &str,
) -> Result<usize> {
    let Some(ckpt) = resume_from else {
        return Ok(0);
    };
    load_kohya_into_adapters(adapters, ckpt, device)?;
    let start = parse_resume_step(ckpt).unwrap_or(0).min(steps);
    if start >= steps {
        anyhow::bail!(
            "{tag}: --resume checkpoint is already at step {start} ≥ --steps {steps}; \
             raise --steps to continue training"
        );
    }
    tracing::info!("{tag}: resuming from {} at step {start}/{steps}", ckpt.display());
    Ok(start)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_checkpoint_step_from_filename() {
        assert_eq!(
            parse_resume_step(Path::new("watercolour-step288.safetensors")),
            Some(288)
        );
        assert_eq!(
            parse_resume_step(Path::new("/a/b/my-lora-step1440.safetensors")),
            Some(1440)
        );
        assert_eq!(parse_resume_step(Path::new("style-step0.safetensors")), Some(0));
        // No "-step<N>" → None (caller defaults to 0).
        assert_eq!(parse_resume_step(Path::new("final.safetensors")), None);
        assert_eq!(parse_resume_step(Path::new("lora-step.safetensors")), None);
    }
}
