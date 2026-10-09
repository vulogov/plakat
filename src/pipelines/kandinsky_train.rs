//! LoRA training for the Kandinsky 5 DiT (`plakat style train --base kandinsky5`).
//!
//! The recipe is the reference trainer's (`kandinskylab/kandinsky-5-lora-train`, `train/lora_train.py`
//! and `configs/trainer/lora_image.yaml`): adapters on the attention and feed-forward layers of every
//! block, `t = sigmoid(N(0, 1))` shifted by 3, `x_t = (1 − t)·z + t·noise`, the target `noise − z`, MSE,
//! AdamW (β 0.9 / 0.95, no weight decay) with a linear warm-up, the gradient clipped at norm 1, and the
//! caption dropped for the empty one half the time.
//!
//! **The backward goes block by block.** candle's autograd keeps every activation of a forward alive
//! until `backward`, and it has no gradient checkpointing (`Documentation/GRADIENT_CHECKPOINTING.md`);
//! fifty blocks of a megapixel image do not fit. But the DiT is a chain of blocks, and a chain can be
//! differentiated one link at a time with nothing but `backward`:
//!
//! 1. Forward once, detaching each block's output and keeping it. No graph outlives its block.
//! 2. The loss is taken on the output layer from the last block's kept output, as a variable: that
//!    gives the gradient `g` with respect to the visual stream.
//! 3. Walk the blocks backwards. Block `i` is run again from its kept input, as a variable `x`; then
//!    `Σ(block(x) · g)` is a scalar whose gradient with respect to `x` is the next `g`, and whose
//!    gradients with respect to the block's adapters are their gradients of the loss.
//!
//! The text stream is an input of every visual block, so its gradient is summed over them and then
//! walked back through the text blocks the same way. Each block is run twice and differentiated once;
//! the memory is one block's graph, whatever the depth. A test checks the result against one backward
//! through the whole model.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};

use super::kandinsky_dit::{patchify_out, Adapter, Dit};

/// The reference trainer's `scheduler_scale`: the shift of the sampled timestep. (Sampling uses 5.0.)
pub const TRAIN_SHIFT: f64 = 3.0;
/// The share of steps trained on the empty caption (`uncond_prob`), which keeps CFG working.
pub const UNCOND_PROB: f64 = 0.5;
/// The reference's warm-up, in steps; a short run warms up over a tenth of itself.
pub const WARMUP_STEPS: usize = 100;
/// Global gradient-norm clip (`max_norm`).
pub const MAX_GRAD_NORM: f64 = 1.0;

/// The gradient of `Σ(out · g)` with respect to `var`, taken out of `grads`.
fn take(grads: &mut GradStore, var: &Var, what: &str) -> Result<Tensor> {
    grads.remove(var.as_tensor()).with_context(|| format!("no gradient reached {what}"))
}

/// A tensor that outlives its block, moved off the device. On Metal candle hands an op's output the
/// smallest free buffer that is large enough, and never shrinks that pool: a 0.3 MB adapter gradient
/// or a 10 MB block output that is kept can sit in — and pin — a 105 MB buffer that a weight just
/// vacated. Kept on the device, the block outputs and the adapter gradients of one step pinned 25 GB
/// at 512² (measured: 0.41 GB a block, the size of a block's F32 weights). On the CPU they are their
/// own size; coming back, a tensor gets a buffer of exactly its size.
fn park(t: &Tensor) -> Result<Tensor> {
    Ok(t.detach().to_device(&Device::Cpu)?)
}

/// Move the gradients of this block's adapters from `from` into `kept`, parked.
fn collect(from: &mut GradStore, kept: &mut Vec<(Var, Tensor)>, adapters: &[Adapter], prefix: &str) -> Result<()> {
    for ad in adapters.iter().filter(|ad| ad.module.starts_with(prefix)) {
        for var in [&ad.a, &ad.b] {
            if let Some(g) = from.remove(var.as_tensor()) {
                kept.push((var.clone(), park(&g)?));
            }
        }
    }
    Ok(())
}

/// `PLAKAT_K5_TRAIN_TRACE=1`: print the process footprint at each stage of a step.
fn trace(what: &str) {
    if std::env::var_os("PLAKAT_K5_TRAIN_TRACE").is_some() {
        crate::ui::progress::println(&format!("    {:>6.2} GB  {what}", crate::memwatch::footprint_gb().unwrap_or(0.0)));
    }
}

/// The flow-matching loss of one sample at noise level `sigma`, and the gradient of every adapter —
/// block by block, as the module header describes. `z0` and `noise` are NCHW `(1, C, H/8, W/8)`.
pub fn loss_and_grads(dit: &Dit, adapters: &[Adapter], z0: &Tensor, text: &Tensor, pooled: &Tensor, sigma: f64, noise: &Tensor) -> Result<(f32, GradStore)> {
    anyhow::ensure!(dit.compute() == DType::F32, "LoRA training runs the DiT in F32 (unset PLAKAT_K5_DIT_COMPUTE)");
    let device = dit.device().clone();
    let (z0, noise) = (z0.to_device(&Device::Cpu)?.to_dtype(DType::F32)?, noise.to_device(&Device::Cpu)?.to_dtype(DType::F32)?);
    let x_t = ((&z0 * (1.0 - sigma))? + (&noise * sigma)?)?;
    let target = patchify_out(&(&noise - &z0)?)?.to_device(&device)?;
    let s = dit.stage(&x_t, text, pooled, sigma * 1000.0)?;
    let (nt, nv) = dit.block_counts();
    trace("staged");

    // 1. Forward, keeping each block's input (parked) and no graph.
    let mut texts = vec![park(&s.text)?];
    let mut text = s.text.detach();
    for i in 0..nt {
        text = dit.text_block(i, &text, &s)?.detach();
        texts.push(park(&text)?);
    }
    let text_out = texts[nt].clone();
    let text = text_out.to_device(&device)?;
    let mut visuals = vec![park(&s.visual)?];
    let mut visual = s.visual.detach();
    for i in 0..nv {
        visual = dit.visual_block(i, &visual, &text, &s)?.detach();
        visuals.push(park(&visual)?);
        if i % 10 == 9 {
            trace(&format!("forward through visual block {i}"));
        }
    }

    // 2. The loss, and its gradient with respect to the visual stream.
    drop((visual, text));
    let mut kept: Vec<(Var, Tensor)> = Vec::new();
    let last = Var::from_tensor(&visuals.pop().expect("the visual stream").to_device(&device)?)?;
    let loss = (dit.head(last.as_tensor(), &s)? - &target)?.sqr()?.mean_all()?;
    let (loss, mut g) = {
        let mut grads = loss.backward()?;
        (loss.to_scalar::<f32>()?, park(&take(&mut grads, &last, "the visual stream")?)?)
    };
    drop(last);
    trace("the head differentiated");

    // 3. Back through the visual blocks; the text stream's gradient adds up over them.
    let mut g_text: Option<Tensor> = None;
    for i in (0..nv).rev() {
        let x = Var::from_tensor(&visuals.pop().expect("a block's input").to_device(&device)?)?;
        let t = Var::from_tensor(&text_out.to_device(&device)?)?;
        let out = dit.visual_block(i, x.as_tensor(), t.as_tensor(), &s)?;
        let mut grads = (out.to_dtype(DType::F32)? * g.to_device(&device)?)?.sum_all()?.backward()?;
        g = park(&take(&mut grads, &x, "a visual block's input")?)?;
        let gt = park(&take(&mut grads, &t, "the text stream")?)?;
        g_text = Some(match g_text {
            Some(sum) => (sum + gt)?,
            None => gt,
        });
        collect(&mut grads, &mut kept, adapters, &format!("visual_transformer_blocks.{i}."))?;
        if i % 10 == 0 {
            trace(&format!("back through visual block {i}"));
        }
    }

    // 4. Back through the text blocks.
    if let Some(mut g) = g_text {
        texts.pop();
        for i in (0..nt).rev() {
            let x = Var::from_tensor(&texts.pop().expect("a text block's input").to_device(&device)?)?;
            let out = dit.text_block(i, x.as_tensor(), &s)?;
            let mut grads = (out.to_dtype(DType::F32)? * g.to_device(&device)?)?.sum_all()?.backward()?;
            g = park(&take(&mut grads, &x, "a text block's input")?)?;
            collect(&mut grads, &mut kept, adapters, &format!("text_transformer_blocks.{i}."))?;
        }
    }
    // The gradients go back to the device, each into a buffer of its own size, in a store the
    // optimizer can read. (A store cannot be made empty; a trivial backward gives one.)
    let mut store = Tensor::zeros((), DType::F32, &Device::Cpu)?.backward()?;
    for (var, g) in kept {
        store.insert(var.as_tensor(), g.to_device(&device)?);
    }
    trace("back through the text blocks");
    Ok((loss, store))
}

/// A small deterministic generator for the trainer's own draws (timestep, noise, order, caption
/// drop): the same run on every device, which the devices' own generators do not give.
pub struct Draws(u64);

impl Draws {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x4B35_4C4F_5241_5F31)
    }

    /// Uniform in `(0, 1]` (SplitMix64).
    pub fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (((z ^ (z >> 31)) >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }

    pub fn below(&mut self, n: usize) -> usize {
        ((self.uniform() * n as f64) as usize).min(n.saturating_sub(1))
    }

    /// The reference's timestep: `sigmoid(N(0, 1))`, shifted by [`TRAIN_SHIFT`].
    pub fn sigma(&mut self) -> f64 {
        let t = 1.0 / (1.0 + (-self.normal()).exp());
        TRAIN_SHIFT * t / (1.0 + (TRAIN_SHIFT - 1.0) * t)
    }

    pub fn noise_like(&mut self, like: &Tensor) -> Result<Tensor> {
        let n = like.elem_count();
        let v: Vec<f32> = (0..n).map(|_| self.normal() as f32).collect();
        Ok(Tensor::from_vec(v, like.shape(), &Device::Cpu)?)
    }
}

/// The learning rate at `step` (from 0): a linear warm-up to `lr`, then flat.
pub fn warmed(lr: f64, step: usize, total: usize) -> f64 {
    let warm = WARMUP_STEPS.min((total / 10).max(1));
    lr * ((step + 1) as f64 / warm as f64).min(1.0)
}

/// What `plakat style train --base kandinsky5` asks for.
pub struct TrainRequest {
    pub repo: String,
    pub device: Device,
    pub images: Vec<PathBuf>,
    /// The caption of an image that has no `<name>.txt` beside it, and prefixed to one that has but
    /// does not contain it.
    pub trigger: String,
    pub rank: usize,
    pub steps: usize,
    pub lr: f64,
    /// Training resolution, square, a multiple of 16.
    pub size: u32,
    pub out: PathBuf,
    pub checkpoint_every: Option<usize>,
    pub log_every: usize,
    pub resume_from: Option<PathBuf>,
    pub seed: u64,
    /// The frozen DiT in NF4 (`--dit-nf4`): 3.7 GB resident instead of 12.
    pub dit_nf4: bool,
    pub quantize_qwen: bool,
}

/// One training image, ready: its latent and its caption's embeddings, on the CPU.
struct Example {
    z0: Tensor,
    caption: String,
}

fn caption_for(image: &Path, trigger: &str) -> String {
    match std::fs::read_to_string(image.with_extension("txt")) {
        Ok(text) if !text.trim().is_empty() => {
            let text = text.trim();
            if trigger.is_empty() || text.contains(trigger) { text.to_string() } else { format!("{trigger}, {text}") }
        }
        _ => trigger.to_string(),
    }
}

fn checkpoint_path(out: &Path, step: usize) -> PathBuf {
    let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("lora");
    let ext = out.extension().and_then(|s| s.to_str()).unwrap_or("safetensors");
    out.with_file_name(format!("{stem}-step{step}.{ext}"))
}

fn save(adapters: &[Adapter], out: &Path) -> Result<()> {
    super::kandinsky_lora::save(adapters.iter().map(|ad| (ad.module.as_str(), ad.a.as_tensor(), ad.b.as_tensor())), out)
}

/// Put a saved LoRA back into the live adapters (`--resume`).
fn restore(adapters: &[Adapter], path: &Path, device: &Device) -> Result<()> {
    let file = super::kandinsky_lora::LoraFile::load(path)?;
    for ad in adapters {
        let p = file.layers.get(&ad.module).with_context(|| format!("{} has no adapter for {}", path.display(), ad.module))?;
        anyhow::ensure!(p.a.dims() == ad.a.dims() && p.b.dims() == ad.b.dims(), "{}: {} is rank {}, this run trains rank {}", path.display(), ad.module, p.a.dim(0)?, ad.a.dim(0)?);
        ad.a.set(&p.a.to_device(device)?)?;
        ad.b.set(&p.b.to_device(device)?)?;
    }
    Ok(())
}

/// Train a LoRA on `req.images`. Staged like generation: the text encoders, then the VAE, then the DiT —
/// never together.
pub async fn train_lora(req: TrainRequest) -> Result<()> {
    use super::kandinsky as k5;
    use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
    let println = |m: String| crate::ui::progress::println(&m);
    anyhow::ensure!(!req.images.is_empty(), "no training images");
    anyhow::ensure!(req.size >= 256 && req.size % 16 == 0, "--size must be a multiple of 16, at least 256 (got {})", req.size);
    anyhow::ensure!(req.steps > 0 && req.rank > 0, "--steps and --rank must be positive");

    // Phase A1 — captions. Each distinct one is encoded once, and the empty one beside them.
    let captions: Vec<String> = req.images.iter().map(|p| caption_for(p, &req.trigger)).collect();
    let t0 = std::time::Instant::now();
    let text_device = if req.quantize_qwen && req.device.is_metal() { Device::Cpu } else { k5::stage_device(&req.device)? };
    // The checkpoints of the family differ in the transformer alone: the text encoders and the VAE of
    // the pretrain repo are the generation repo's files, byte for byte. They are read from the
    // generation repo, so training on another checkpoint downloads its 12 GB and not 18 GB more.
    let shared = crate::hf::resolve_alias("kandinsky5");
    let mut encoders = super::kandinsky_text::TextEncoders::load_with(shared, &text_device, req.quantize_qwen).await.context("loading the Kandinsky 5 text encoders")?;
    let mut embeds: std::collections::HashMap<String, (Tensor, Tensor)> = std::collections::HashMap::new();
    for text in captions.iter().map(String::as_str).chain([""]) {
        if !embeds.contains_key(text) {
            let e = encoders.encode(text, k5::DEFAULT_MAX_SEQ)?;
            embeds.insert(text.to_string(), (e.qwen.to_device(&Device::Cpu)?, e.pooled.to_device(&Device::Cpu)?));
        }
    }
    drop((encoders, text_device));
    println(format!("kandinsky5 train: encoded {} caption(s) in {:.1}s", embeds.len(), t0.elapsed().as_secs_f64()));

    // Phase A2 — latents.
    let t1 = std::time::Instant::now();
    let examples: Vec<Example> = {
        let vae_device = k5::decode_device(&req.device)?;
        let vae = super::kandinsky_text::Vae::load(shared, &vae_device).await?;
        req.images
            .iter()
            .zip(&captions)
            .map(|(path, caption)| {
                let pixels = crate::imaging::preprocess::sd_image_tensor(path, req.size, req.size, &vae_device, DType::F32).with_context(|| format!("reading {}", path.display()))?;
                Ok(Example { z0: vae.encode(&pixels)?.to_device(&Device::Cpu)?, caption: caption.clone() })
            })
            .collect::<Result<_>>()?
    };
    println(format!("kandinsky5 train: encoded {} image(s) at {}x{} in {:.1}s", examples.len(), req.size, req.size, t1.elapsed().as_secs_f64()));

    // Phase B — the DiT with adapters.
    let t2 = std::time::Instant::now();
    let device = k5::stage_device(&req.device)?;
    let mut dit = Dit::load_full(&req.repo, &device, req.dit_nf4, None).await.context("loading the Kandinsky 5 DiT")?;
    let adapters = dit.install_adapters(req.rank, 1.0, req.seed)?;
    let vars: Vec<Var> = adapters.iter().flat_map(|ad| [ad.a.clone(), ad.b.clone()]).collect();
    let params: usize = vars.iter().map(|v| v.elem_count()).sum();
    println(format!("kandinsky5 train: DiT loaded in {:.1}s; {} adapters of rank {}, {:.1}M parameters", t2.elapsed().as_secs_f64(), adapters.len(), req.rank, params as f64 / 1e6));

    let mut start = 0;
    if let Some(ckpt) = &req.resume_from {
        restore(&adapters, ckpt, &device)?;
        start = super::sd_train::trainer::parse_resume_step(ckpt).unwrap_or(0);
        println(format!("kandinsky5 train: resumed from {} at step {start} (the optimizer's moments start cold)", ckpt.display()));
        anyhow::ensure!(start < req.steps, "the checkpoint is at step {start}; raise --steps above it to train further");
    }
    let mut opt = AdamW::new(vars.clone(), ParamsAdamW { lr: req.lr, beta1: 0.9, beta2: 0.95, eps: 1e-8, weight_decay: 0.0 })?;
    let every = req.checkpoint_every.filter(|&n| n > 0).unwrap_or_else(|| (req.steps / 10).max(30));
    let mut draws = Draws::new(req.seed);
    // The draws of the steps already taken, so a resumed run continues the same sequence.
    let mut order: Vec<usize> = Vec::new();
    let (mut running, mut counted) = (0f64, 0usize);
    let loop_start = std::time::Instant::now();
    for step in 0..req.steps {
        if order.is_empty() {
            order = (0..examples.len()).collect();
            for i in (1..order.len()).rev() {
                order.swap(i, draws.below(i + 1));
            }
        }
        let ex = &examples[order.pop().expect("an example")];
        let (sigma, uncond) = (draws.sigma(), draws.uniform() < UNCOND_PROB);
        let noise = draws.noise_like(&ex.z0)?;
        if step < start {
            continue;
        }
        let (text, pooled) = &embeds[if uncond { "" } else { ex.caption.as_str() }];
        let (loss, mut grads) = loss_and_grads(&dit, &adapters, &ex.z0, text, pooled, sigma, &noise)?;
        anyhow::ensure!(loss.is_finite(), "the loss became {loss} at step {} — lower --lr", step + 1);
        crate::pipelines::lora_linear::clip_grad_norm(&mut grads, &vars, MAX_GRAD_NORM)?;
        opt.set_learning_rate(warmed(req.lr, step, req.steps));
        trace("gradients clipped");
        opt.step(&grads)?;
        drop(grads);
        trace("optimizer stepped");
        running += loss as f64;
        counted += 1;
        let done = step + 1;
        if done % req.log_every.max(1) == 0 || done == req.steps {
            let per = loop_start.elapsed().as_secs_f64() / (done - start) as f64;
            println(format!("  step {done}/{}  loss {:.4}  {:.1}s a step, {:.0} min left", req.steps, running / counted as f64, per, per * (req.steps - done) as f64 / 60.0));
            (running, counted) = (0.0, 0);
        }
        if done % every == 0 && done < req.steps {
            let ckpt = checkpoint_path(&req.out, done);
            save(&adapters, &ckpt)?;
            println(format!("  checkpoint → {}", ckpt.display()));
        }
    }
    save(&adapters, &req.out)?;
    println(format!("kandinsky5 train: LoRA → {} (use it with --model kandinsky5 --lora {})", req.out.display(), req.out.display()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::kandinsky_dit::tests::{tiny, tiny_inputs, tiny_weights};
    use candle_nn::VarBuilder;

    /// `tiny_weights` draws a model of its own on every call, so a test that compares two builds makes
    /// the weights once and builds from copies.
    fn tiny_from(weights: &std::collections::HashMap<String, Tensor>, device: &Device) -> Dit {
        Dit::new(tiny(), VarBuilder::from_tensors(weights.clone(), DType::F32, device), DType::F32).unwrap()
    }

    fn tiny_dit(device: &Device) -> Dit {
        tiny_from(&tiny_weights(&tiny()), device)
    }

    /// `B` away from zero, so that `A` has a gradient too.
    fn wake(adapters: &[Adapter]) {
        for (n, ad) in adapters.iter().enumerate() {
            let (out, r) = ad.b.dims2().unwrap();
            let v: Vec<f32> = (0..out * r).map(|i| ((i + 7 * n) as f32 * 0.37).sin() * 0.05).collect();
            ad.b.set(&Tensor::from_vec(v, (out, r), ad.b.device()).unwrap()).unwrap();
        }
    }

    /// The same loss with one backward through the whole model.
    fn whole(dit: &Dit, z0: &Tensor, text: &Tensor, pooled: &Tensor, sigma: f64, noise: &Tensor) -> (f32, GradStore) {
        let x_t = ((z0 * (1.0 - sigma)).unwrap() + (noise * sigma).unwrap()).unwrap();
        let target = patchify_out(&(noise - z0).unwrap()).unwrap().to_device(dit.device()).unwrap();
        let s = dit.stage(&x_t, text, pooled, sigma * 1000.0).unwrap();
        let (nt, nv) = dit.block_counts();
        let mut t = s.text.clone();
        for i in 0..nt {
            t = dit.text_block(i, &t, &s).unwrap();
        }
        let mut v = s.visual.clone();
        for i in 0..nv {
            v = dit.visual_block(i, &v, &t, &s).unwrap();
        }
        let loss = (dit.head(&v, &s).unwrap() - target).unwrap().sqr().unwrap().mean_all().unwrap();
        (loss.to_scalar::<f32>().unwrap(), loss.backward().unwrap())
    }

    fn compare_on(device: &Device, tolerance: f32) {
        let cfg = tiny();
        let mut dit = tiny_dit(device);
        let adapters = dit.install_adapters(4, 1.0, 11).unwrap();
        // 2 text blocks × 6 layers + 3 visual blocks × 10.
        assert_eq!(adapters.len(), 2 * 6 + 3 * 10);
        wake(&adapters);
        let (z0, text, pooled) = tiny_inputs(&cfg);
        let noise = Draws::new(3).noise_like(&z0).unwrap();
        let (loss, grads) = loss_and_grads(&dit, &adapters, &z0, &text, &pooled, 0.6, &noise).unwrap();
        let (loss_whole, grads_whole) = whole(&dit, &z0, &text, &pooled, 0.6, &noise);
        assert!((loss - loss_whole).abs() <= 1e-5 * loss_whole.abs().max(1.0), "{loss} against {loss_whole}");
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for ad in &adapters {
            for (half, var) in [("A", &ad.a), ("B", &ad.b)] {
                let ours = flat(grads.get(var.as_tensor()).unwrap_or_else(|| panic!("no gradient for {} {half}", ad.module)));
                let want = flat(grads_whole.get(var.as_tensor()).unwrap());
                let scale = want.iter().fold(0f32, |m, x| m.max(x.abs()));
                assert!(scale > 0.0, "{} {half}: the whole-model gradient is zero", ad.module);
                let worst = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                assert!(worst <= tolerance * scale, "{} {half}: off by {worst} of {scale}", ad.module);
            }
        }
    }

    #[test]
    fn the_block_by_block_gradient_is_the_whole_model_gradient() {
        compare_on(&Device::Cpu, 2e-3);
    }

    /// The same on the GPU: every op of a block has a backward there.
    #[cfg(feature = "metal")]
    #[test]
    fn the_block_by_block_gradient_holds_on_metal() {
        let Ok(device) = Device::new_metal(0) else { return };
        compare_on(&device, 5e-3);
    }

    #[test]
    fn fresh_adapters_change_nothing_and_training_lowers_the_loss() {
        use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
        let cfg = tiny();
        let (z0, text, pooled) = tiny_inputs(&cfg);
        let weights = tiny_weights(&cfg);
        let plain = tiny_from(&weights, &Device::Cpu).forward(&z0, &text, &pooled, 500.0).unwrap();
        let mut dit = tiny_from(&weights, &Device::Cpu);
        let adapters = dit.install_adapters(4, 1.0, 5).unwrap();
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // B = 0: the adapted model is the model.
        let fresh = flat(&dit.forward(&z0, &text, &pooled, 500.0).unwrap());
        // (Up to the training softmax, which is the same function computed another way.)
        let plain_scale = flat(&plain).iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!(fresh.iter().zip(flat(&plain)).all(|(a, b)| (a - b).abs() < 1e-4 * plain_scale));

        let vars: Vec<Var> = adapters.iter().flat_map(|ad| [ad.a.clone(), ad.b.clone()]).collect();
        let mut opt = AdamW::new(vars.clone(), ParamsAdamW { lr: 2e-3, beta1: 0.9, beta2: 0.95, eps: 1e-8, weight_decay: 0.0 }).unwrap();
        let noise = Draws::new(9).noise_like(&z0).unwrap();
        let mut losses = Vec::new();
        for _ in 0..40 {
            let (loss, mut grads) = loss_and_grads(&dit, &adapters, &z0, &text, &pooled, 0.7, &noise).unwrap();
            crate::pipelines::lora_linear::clip_grad_norm(&mut grads, &vars, MAX_GRAD_NORM).unwrap();
            opt.step(&grads).unwrap();
            losses.push(loss);
        }
        assert!(losses[39] < losses[0] * 0.8, "{} → {}", losses[0], losses[39]);

        // What was trained, saved and merged into a fresh model, is the trained model.
        let dir = std::env::temp_dir().join(format!("plakat-k5-train-{}", std::process::id()));
        let path = dir.join("tiny.safetensors");
        save(&adapters, &path).unwrap();
        let mut set = crate::pipelines::kandinsky_lora::LoraSet::new();
        set.push(crate::pipelines::kandinsky_lora::LoraFile::load(&path).unwrap(), 1.0, "tiny");
        let merged = Dit::new_with_lora(cfg.clone(), VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu), None, &set, DType::F32).unwrap();
        let (used, unmatched) = set.report();
        assert_eq!((used, unmatched.len()), (adapters.len(), 0));
        let (trained, loaded) = (flat(&dit.forward(&z0, &text, &pooled, 500.0).unwrap()), flat(&merged.forward(&z0, &text, &pooled, 500.0).unwrap()));
        let scale = trained.iter().fold(0f32, |m, x| m.max(x.abs()));
        let worst = trained.iter().zip(&loaded).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        // The file is BF16, so the merged weights are the trained ones rounded.
        assert!(worst < 0.02 * scale, "off by {worst} of {scale}");
        // And it is not the untrained model.
        assert!(trained.iter().zip(flat(&plain)).any(|(a, b)| (a - b).abs() > 1e-3 * scale));

        // With NF4 the LoRA goes in before the quantization: the same model as quantizing merged weights.
        let mut premerged = weights.clone();
        for (module, pair) in &crate::pipelines::kandinsky_lora::LoraFile::load(&path).unwrap().layers {
            let w = premerged.get_mut(&format!("{module}.weight")).unwrap();
            *w = (&*w + pair.b.matmul(&pair.a).unwrap()).unwrap();
        }
        let vb = |w: &std::collections::HashMap<String, Tensor>| VarBuilder::from_tensors(w.clone(), DType::F32, &Device::Cpu);
        let quantized_after = Dit::new_nf4(cfg.clone(), vb(&premerged), vb(&premerged), DType::F32).unwrap();
        let quantized_with = Dit::new_with_lora(cfg.clone(), vb(&weights), Some(vb(&weights)), &set, DType::F32).unwrap();
        let (qa, qw) = (flat(&quantized_after.forward(&z0, &text, &pooled, 500.0).unwrap()), flat(&quantized_with.forward(&z0, &text, &pooled, 500.0).unwrap()));
        assert!(qa.iter().zip(&qw).all(|(a, b)| (a - b).abs() < 1e-5 * scale.max(1.0)));

        // A checkpoint goes back into live adapters.
        let mut again = tiny_from(&weights, &Device::Cpu);
        let fresh = again.install_adapters(4, 1.0, 99).unwrap();
        restore(&fresh, &path, &Device::Cpu).unwrap();
        let resumed = flat(&again.forward(&z0, &text, &pooled, 500.0).unwrap());
        assert!(resumed.iter().zip(&loaded).all(|(a, b)| (a - b).abs() < 1e-4 * scale.max(1.0)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_draws_repeat_and_the_schedule_warms_up() {
        let (mut a, mut b) = (Draws::new(1), Draws::new(1));
        let seq: Vec<f64> = (0..2000).map(|_| a.sigma()).collect();
        assert!(seq.iter().zip((0..2000).map(|_| b.sigma())).all(|(x, y)| *x == y));
        assert!(seq.iter().all(|s| *s > 0.0 && *s < 1.0));
        // sigmoid(N(0,1)) has median 0.5, which the shift of 3 moves to 0.75.
        let above = seq.iter().filter(|s| **s > 0.75).count() as f64 / 2000.0;
        assert!((above - 0.5).abs() < 0.05, "{above}");
        assert!((0..1000).map(|_| a.below(7)).all(|i| i < 7));
        assert!((warmed(1e-4, 0, 1000) - 1e-6).abs() < 1e-12 && warmed(1e-4, 99, 1000) == 1e-4 && warmed(1e-4, 500, 1000) == 1e-4);
        // A 50-step run warms up over 5 steps.
        assert!(warmed(1e-4, 4, 50) == 1e-4 && warmed(1e-4, 0, 50) < 1e-4);
    }

    #[test]
    fn a_caption_file_beside_the_image_is_its_caption() {
        let dir = std::env::temp_dir().join(format!("plakat-k5-cap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), "a fox in snow\n").unwrap();
        std::fs::write(dir.join("b.txt"), "sks style, a pier").unwrap();
        assert_eq!(caption_for(&dir.join("a.png"), "sks style"), "sks style, a fox in snow");
        assert_eq!(caption_for(&dir.join("b.png"), "sks style"), "sks style, a pier");
        assert_eq!(caption_for(&dir.join("c.png"), "sks style"), "sks style");
        assert_eq!(checkpoint_path(Path::new("/x/my.safetensors"), 200), Path::new("/x/my-step200.safetensors"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
