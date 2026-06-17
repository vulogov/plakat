//! Textual Inversion training — learn ONE new token embedding (a "word") for the
//! SD CLIP-L text encoder with the whole model **frozen**; only the placeholder
//! vector is optimized. SD 1.5 / 2.1 (single CLIP-L). The output is an A1111
//! `emb_params` safetensors loadable via `--embedding PATH:trigger` — the inverse
//! of what plakat already does at load time (it can use TIs; this *makes* them).
//!
//! Mechanism: forward through the frozen UNet + CLIP with the template
//! "a photo of <init-word>", but **splice the trainable vector** into the
//! init-word's token slot (a differentiable masked combine, so the gradient
//! reaches only that vector), and backprop the DDPM-ε loss into it.

use anyhow::{Result, anyhow, bail};
use candle_core::{DType, Device, IndexOp, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::pipelines::sd_core::{SdCore, SdLoadRequest};

/// Inputs for [`train_textual_inversion`].
pub struct TiTrainRequest {
    pub model: String,
    pub device: Device,
    pub images: Vec<PathBuf>,
    /// The trigger the embedding will be used under (for messaging; the loader
    /// takes the trigger from `--embedding PATH:trigger`).
    pub token: String,
    /// A coarse class word to initialize from (a single simple word, e.g. "toy",
    /// "art") — TI converges far faster from a sensible starting point.
    pub init_word: String,
    pub steps: usize,
    pub lr: f64,
    pub size: u32,
    pub out: PathBuf,
    pub log_every: usize,
}

pub async fn train_textual_inversion(req: TiTrainRequest) -> Result<()> {
    let m = req.model.to_lowercase();
    if m.contains("sdxl")
        || m.contains("sd3")
        || m.contains("flux")
        || m.contains("cascade")
        || m.contains("pixart")
    {
        bail!(
            "textual-inversion training is SD 1.5 / 2.1 only for now (single CLIP-L); \
             SDXL (dual encoder) is a planned follow-up"
        );
    }
    let device = req.device.clone();
    tracing::info!(
        "ti-train: loading {} (frozen) + encoding {} image(s)",
        req.model,
        req.images.len()
    );
    let core = SdCore::load(SdLoadRequest {
        model: req.model.clone(),
        device: device.clone(),
        loras: Vec::new(),
        lora_scale: 1.0,
        embeddings: Vec::new(),
        vae_cache: None,
    })
    .await?;
    let dtype = core.dtype;
    let scale = core.variant.vae_scale();

    // --- encode images → latents (frozen VAE) ---
    let mut latents = Vec::with_capacity(req.images.len());
    for img in &req.images {
        let px = crate::imaging::preprocess::sd_image_tensor(
            img.as_path(),
            req.size,
            req.size,
            &device,
            dtype,
        )?;
        let z = core.vae.encode(&px)?.sample()?;
        latents.push((z * scale)?.to_dtype(dtype)?);
    }
    let n = latents.len().max(1);

    // --- template "a photo of <init-word>"; the init-word token is the slot ---
    let prompt = format!("a photo of {}", req.init_word.trim());
    let token_ids =
        crate::pipelines::t2i::tokenize_padded(&core.tokenizer_l, &core.cfg.clip, &prompt, &device)?;
    let max_pos = core.cfg.clip.max_position_embeddings;
    let eot = core
        .tokenizer_l
        .token_to_id("<|endoftext|>")
        .ok_or_else(|| anyhow!("tokenizer missing <|endoftext|>"))?;
    let ids_vec: Vec<u32> = token_ids.i(0)?.to_vec1()?;
    let eos_pos = ids_vec
        .iter()
        .position(|&t| t == eot)
        .ok_or_else(|| anyhow!("template produced no EOS token"))?;
    if eos_pos < 2 {
        bail!(
            "init word {:?} tokenized oddly — pick a simple single class word (e.g. 'toy', 'art')",
            req.init_word
        );
    }
    let slot = eos_pos - 1; // last content token = the init word

    // --- init the placeholder vector from the init word's frozen embedding ---
    let base0 = core.text_encoder_l.embed_tokens(&token_ids)?; // (1, max_pos, dim), frozen
    let embed_dim = base0.dim(2)?;
    let init = base0.i((0, slot))?.to_dtype(DType::F32)?.unsqueeze(0)?; // (1, dim) F32
    let placeholder = Var::from_tensor(&init)?; // the ONLY trainable parameter
    let mut opt = AdamW::new(
        vec![placeholder.clone()],
        ParamsAdamW {
            lr: req.lr,
            ..Default::default()
        },
    )?;

    // One-hot slot mask (1, max_pos, 1) for a differentiable splice.
    let mut mask_data = vec![0f32; max_pos];
    mask_data[slot] = 1.0;
    let slot_mask = Tensor::from_vec(mask_data, (1, max_pos, 1), &device)?.to_dtype(dtype)?;
    let inv_mask = (slot_mask.ones_like()? - &slot_mask)?;

    // --- DDPM-ε loop; ONLY the placeholder embedding is trained ---
    let abar = crate::pipelines::sd_train::trainer::alphas_cumprod();
    let mut progress = crate::pipelines::train_progress::TrainProgress::new(
        req.steps,
        req.lr,
        req.steps + 1, // no periodic checkpoints
    );
    tracing::info!(
        "ti-train: token {:?} init from {:?} (slot {slot}), {} steps",
        req.token,
        req.init_word,
        req.steps
    );
    for step in 0..req.steps {
        // Splice the trainable vector into the slot (masked combine = differentiable):
        // spliced = base·(1−mask) + placeholder·mask.
        let base = core.text_encoder_l.embed_tokens(&token_ids)?;
        let ph_field = placeholder
            .as_tensor()
            .to_dtype(dtype)?
            .unsqueeze(1)? // (1, 1, dim)
            .broadcast_mul(&slot_mask)?; // → (1, max_pos, dim), placeholder at the slot, 0 elsewhere
        let spliced = (base.broadcast_mul(&inv_mask)? + ph_field)?;
        let hidden = core
            .text_encoder_l
            .forward_from_input_embeds(&spliced, max_pos)?;

        let x0 = &latents[step % n];
        let noise = Tensor::randn(0f32, 1f32, x0.dims(), &device)?.to_dtype(dtype)?;
        let t =
            (Tensor::rand(0f32, 1f32, (1usize,), &device)?.to_vec1::<f32>()?[0] * 999.0) as usize;
        let a = abar[t];
        let x_t = ((x0 * a.sqrt())? + (&noise * (1.0 - a).sqrt())?)?;
        let pred = core.unet.forward(&x_t, t as f64, &hidden, None, None)?;
        let loss = (&pred - &noise)?.sqr()?.mean_all()?.to_dtype(DType::F32)?;
        let grads = loss.backward()?;
        opt.step(&grads)?;
        if step % req.log_every.max(1) == 0 || step + 1 == req.steps {
            tracing::info!("{}", progress.line("ti-train", step + 1, loss.to_scalar::<f32>()?));
        }
    }

    // --- save A1111 `emb_params` (N=1, dim) F16; load via --embedding PATH:trigger ---
    if let Some(parent) = req.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert(
        "emb_params".to_string(),
        placeholder
            .as_tensor()
            .to_dtype(DType::F16)?
            .to_device(&Device::Cpu)?,
    );
    candle_core::safetensors::save(&tensors, &req.out)?;
    tracing::info!(
        "ti-train: wrote {} — use it with  --embedding {}:{}",
        req.out.display(),
        req.out.display(),
        req.token
    );
    tracing::info!("{}", progress.finish("ti-train", &req.out));
    let _ = embed_dim;
    Ok(())
}
