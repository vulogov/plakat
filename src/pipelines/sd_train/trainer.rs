//! SD 1.5 / SDXL style-LoRA trainer (Phase 1: SD 1.5).
//!
//! Mirrors `sd3::train_style_lora` but for the UNet: a **DDPM-epsilon**
//! objective with CLIP-L conditioning, training the vendored LoRA-wired
//! UNet. Mixed precision (BF16 base + F32 LoRA — the LoraLinear forward
//! casts). Output is a **kohya**-format `.safetensors` (`lora_unet_…`).
use anyhow::Result;
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
fn sd15_unet_config() -> UNet2DConditionModelConfig {
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

/// Train a style LoRA on the SD UNet attention; write a kohya safetensors.
pub async fn train_style_lora_sd(req: SdStyleTrainRequest) -> Result<()> {
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
    let unet = UNet2DConditionModel::new(vb, 4, 4, false, sd15_unet_config())?;
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
    let mut opt = AdamW::new(vars, ParamsAdamW { lr: req.lr, ..Default::default() })?;

    // --- Phase C: DDPM-epsilon loop. x_t = √ᾱ·x0 + √(1-ᾱ)·ε; predict ε.
    let abar = alphas_cumprod();
    let n = latents.len().max(1);
    for step in 0..req.steps {
        let x0 = &latents[step % n];
        let noise = Tensor::randn(0f32, 1f32, x0.dims(), &device)?.to_dtype(dtype)?;
        let t = (Tensor::rand(0f32, 1f32, (1usize,), &device)?.to_vec1::<f32>()?[0] * 999.0) as usize;
        let a = abar[t];
        let x_t = ((x0 * a.sqrt())? + (&noise * (1.0 - a).sqrt())?)?;
        let pred = unet.forward(&x_t, t as f64, &text_emb)?;
        let loss = (&pred - &noise)?.sqr()?.mean_all()?.to_dtype(DType::F32)?;
        opt.step(&loss.backward()?)?;
        if step % 10 == 0 || step + 1 == req.steps {
            tracing::info!(
                "sd-style-train: step {}/{} loss {:.5}",
                step + 1,
                req.steps,
                loss.to_scalar::<f32>()?
            );
        }
        if (step + 1) % 30 == 0 && step + 1 != req.steps {
            save_kohya_lora(&adapters, req.rank, &req.out)?;
            tracing::info!("sd-style-train: checkpoint @ step {} → {}", step + 1, req.out.display());
        }
    }
    save_kohya_lora(&adapters, req.rank, &req.out)?;
    tracing::info!("sd-style-train: wrote {}", req.out.display());
    Ok(())
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
