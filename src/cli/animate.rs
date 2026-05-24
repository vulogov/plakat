//! `plakat animate` — frame-by-frame embedding interpolation.
//!
//! Two prompts (`--from` / `--to`) get encoded once each; the
//! denoise loop runs N times with linearly-lerped CLIP hidden
//! states. Frames land in `<out>/frame-NNNN.png` plus an optional
//! `<out>/animation.gif` when `--gif` is passed.
//!
//! Scope notes:
//!
//! * SD 1.5 / SD 2.1 only in this release. SDXL has dual encoders
//!   + pooled add_embedding that complicate the lerp; Flux + SD3
//!   use T5 + rectified flow that need their own machinery.
//!   The CLI dispatch bails loud if the model isn't SD-family.
//! * No `--lora` / `--control` / `--refiner` plumbing — animate
//!   keeps the pipeline narrow on purpose. Bake LoRAs into the
//!   prompts via wildcards or use the standard `plakat generate`
//!   if you need a single frame with adapters.
//! * The seed stays fixed across frames so the initial noise is
//!   constant — only the prompt-driven trajectory varies. This
//!   produces a smooth morph; randomising the seed per frame
//!   produces a sweep + morph that flickers.

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use clap::Args;
use std::path::PathBuf;

use crate::pipelines::scheduler::SchedulerKind;
use crate::pipelines::sd_core::{SdCore, SdLoadRequest, SdVariant};

#[derive(Args, Debug)]
pub struct AnimateArgs {
    /// First prompt — frame 0 renders this.
    #[arg(long)]
    pub from: String,

    /// Second prompt — the last frame renders this.
    #[arg(long)]
    pub to: String,

    /// Frame count (≥ 2). Frame N maps to lerp factor
    /// `i / (N - 1)` so frame 0 = `--from`, frame N-1 = `--to`,
    /// midpoint is 50/50.
    #[arg(long, default_value_t = 16)]
    pub frames: u32,

    /// Shared seed for every frame. Locking the seed keeps the
    /// initial noise constant so the prompt morph is the only
    /// changing variable — producing a smooth animation rather
    /// than a flickery seed sweep.
    #[arg(long)]
    pub seed: Option<u64>,

    /// SD-family model. Defaults to `sd15`. SDXL / Flux / SD3
    /// bail loud — see module docs.
    #[arg(long, default_value = "sd15")]
    pub model: String,

    /// Output dimensions. Multiple of 8 required (VAE constraint).
    #[arg(long, default_value = "512x512")]
    pub size: String,

    /// Denoise steps per frame. Lower (15-20) is fine for
    /// animations since per-frame quality matters less than
    /// smoothness across frames.
    #[arg(long, default_value_t = 20)]
    pub steps: usize,

    /// CFG guidance. Standard SD 1.5 / 2.1 default applies.
    #[arg(long, default_value_t = 7.5)]
    pub guidance: f64,

    /// Negative prompt (shared across all frames).
    #[arg(long, default_value = "")]
    pub negative: String,

    /// Scheduler. Default = the model's built-in (DDIM for SD 1.5).
    #[arg(long, default_value = "default")]
    pub scheduler: SchedulerKind,

    /// Output directory. Frames land as `frame-NNNN.png`
    /// (zero-padded to 4 digits — 9999 frames max).
    #[arg(long, default_value = "./out")]
    pub out: PathBuf,

    /// Also bundle the frames into `<out>/animation.gif`. Uses
    /// the `image` crate's GIF encoder. Frame delay is 100 ms by
    /// default (10 fps); override with `--gif-delay-ms`.
    #[arg(long, default_value_t = false)]
    pub gif: bool,

    /// GIF frame delay in milliseconds. 100 ms = 10 fps;
    /// 41 ms ≈ 24 fps (cinematic); 33 ms ≈ 30 fps.
    #[arg(long, default_value_t = 100)]
    pub gif_delay_ms: u16,
}

pub async fn run(args: AnimateArgs, device: Device) -> Result<()> {
    if args.frames < 2 {
        anyhow::bail!("--frames must be ≥ 2 (got {})", args.frames);
    }
    let (width, height) = parse_size(&args.size)?;
    if width % 8 != 0 || height % 8 != 0 {
        anyhow::bail!(
            "--size {} not divisible by 8 (VAE constraint). \
             Try 512x512 / 768x768.",
            args.size
        );
    }
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating output dir {}", args.out.display()))?;

    // Model gate: SD 1.5 / SD 2.1 only. Reuse `SdVariant::detect`
    // so the alias + repo-id resolution is the same as the t2i
    // dispatch.
    let variant = {
        let repo = if args.model.contains('/') {
            args.model.clone()
        } else {
            crate::hf::resolve_alias(&args.model).to_string()
        };
        SdVariant::detect(&repo)
    };
    if !matches!(variant, SdVariant::Sd15 | SdVariant::Sd21) {
        anyhow::bail!(
            "`plakat animate` is SD 1.5 / SD 2.1 only in this release \
             (got --model {} = {:?}). SDXL / Flux / SD3 animation lands \
             in a follow-up — the per-frame embedding lerp needs \
             different machinery for those families.",
            args.model,
            variant
        );
    }

    // Load the SD backbone once; share across all frames.
    let load_spin = crate::ui::progress::spinner(&format!(
        "Loading {} for animation",
        match variant {
            SdVariant::Sd15 => "SD 1.5",
            SdVariant::Sd21 => "SD 2.1",
            _ => unreachable!(),
        }
    ));
    let core = SdCore::load(SdLoadRequest {
        model: args.model.clone(),
        device: device.clone(),
        loras: Vec::new(),
        lora_scale: 1.0,
        embeddings: Vec::new(),
    })
    .await?;
    load_spin.finish_with_message("✓ SD backbone ready");

    let do_cfg = args.guidance > 1.0;
    let dtype = core.dtype;

    // Encode the two endpoint prompts + (optionally) the negative
    // once each. Frame-time work is just (a) lerp two existing
    // tensors and (b) run the denoise loop.
    let encode_spin = crate::ui::progress::spinner("Encoding endpoint prompts");
    let cond_a = encode_branch(&core, &args.from, dtype)?;
    let cond_b = encode_branch(&core, &args.to, dtype)?;
    let uncond = if do_cfg {
        Some(encode_branch(&core, &args.negative, dtype)?)
    } else {
        None
    };
    encode_spin.finish_with_message("✓ endpoint embeddings ready");

    // Seed: explicit when given, otherwise generate one + log so
    // the run is reproducible if the user wants to re-render.
    let seed = args.seed.unwrap_or_else(rand::random) & (u32::MAX as u64);
    crate::ui::progress::println(&format!(
        "  animation: {} frames, seed {seed}, model {}, {}x{}",
        args.frames, args.model, width, height,
    ));

    let mut frame_paths: Vec<PathBuf> = Vec::with_capacity(args.frames as usize);
    for frame_i in 0..args.frames {
        let t = if args.frames == 1 {
            0.0
        } else {
            frame_i as f64 / (args.frames - 1) as f64
        };
        let lerped_cond = lerp_tensors(&cond_a, &cond_b, t)?;
        let text_embeddings = match uncond.as_ref() {
            Some(u) => Tensor::cat(&[u, &lerped_cond], 0)?,
            None => lerped_cond,
        };

        let frame_path = args.out.join(format!("frame-{frame_i:04}.png"));
        denoise_one_frame(
            &core,
            &text_embeddings,
            width,
            height,
            args.steps,
            args.guidance,
            args.scheduler,
            seed,
            &frame_path,
        )?;
        frame_paths.push(frame_path);
        crate::ui::progress::println(&format!(
            "  frame {}/{} → {} (t={:.3})",
            frame_i + 1,
            args.frames,
            frame_paths.last().unwrap().display(),
            t,
        ));
    }

    if args.gif {
        let gif_path = args.out.join("animation.gif");
        let spin = crate::ui::progress::spinner(&format!(
            "Bundling {} frames → {}",
            frame_paths.len(),
            gif_path.display()
        ));
        write_gif(&frame_paths, &gif_path, args.gif_delay_ms)?;
        spin.finish_with_message(format!("✓ {}", gif_path.display()));
    }

    Ok(())
}

/// Encode one prompt branch (cond or uncond) into the SD 1.5 /
/// SD 2.1 hidden-state shape `(1, 77, embed_dim)`. Mirrors what
/// `t2i::Pipeline::encode_single` does for a single branch but
/// without the CFG concat.
fn encode_branch(core: &SdCore, text: &str, dtype: DType) -> Result<Tensor> {
    let pad_id: u32 = match &core.cfg.clip.pad_with {
        Some(s) => core
            .tokenizer_l
            .token_to_id(s)
            .ok_or_else(|| anyhow!("tokenizer missing pad token {s:?}"))?,
        None => core
            .tokenizer_l
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| anyhow!("tokenizer missing <|endoftext|>"))?,
    };
    let mut ids = core
        .tokenizer_l
        .encode(text, true)
        .map_err(|e| anyhow!("CLIP encode of {text:?}: {e}"))?
        .get_ids()
        .to_vec();
    ids.resize(core.cfg.clip.max_position_embeddings, pad_id);
    let ids_t = Tensor::new(ids.as_slice(), &core.device)?.unsqueeze(0)?;
    let hidden = core.text_encoder_l.forward(&ids_t)?;
    Ok(hidden.to_dtype(dtype)?)
}

/// Linear interpolation between two same-shape tensors at scalar
/// `t` ∈ [0, 1]. `t = 0` → all `a`; `t = 1` → all `b`.
fn lerp_tensors(a: &Tensor, b: &Tensor, t: f64) -> Result<Tensor> {
    let t = t.clamp(0.0, 1.0);
    let inv = 1.0 - t;
    let lerped = ((a * inv)? + (b * t)?)?;
    Ok(lerped)
}

/// Run a minimal denoise loop using `core`'s UNet + scheduler +
/// VAE. SD 1.5 / SD 2.1 only (no SDXL pooled / add_time_ids
/// handling). Saves the result to `out_path`. Caller has already
/// CFG-cat'd the embeddings if needed.
#[allow(clippy::too_many_arguments)]
fn denoise_one_frame(
    core: &SdCore,
    text_embeddings: &Tensor,
    width: u32,
    height: u32,
    steps: usize,
    guidance: f64,
    scheduler_kind: SchedulerKind,
    seed: u64,
    out_path: &std::path::Path,
) -> Result<()> {
    use crate::pipelines::sdxl_unet::SdUNet;

    let do_cfg = guidance > 1.0;
    let w = width as usize;
    let h = height as usize;

    // Same seeding path the t2i pipeline uses. Metal accepts only
    // u32 seeds; mask before calling.
    if let Err(e) = core.device.set_seed(seed) {
        tracing::debug!(target: "plakat", "set_seed ignored: {e}");
    }

    let mut scheduler = crate::pipelines::scheduler::build(
        scheduler_kind,
        &core.cfg,
        steps,
    )?;

    let mut latents = Tensor::randn(
        0f32,
        1f32,
        (1, 4, h / 8, w / 8),
        &core.device,
    )?
    .to_dtype(core.dtype)?;
    latents = (latents * scheduler.init_noise_sigma())?;

    let timesteps = scheduler.timesteps().to_vec();
    for &timestep in timesteps.iter() {
        let model_input = if do_cfg {
            Tensor::cat(&[&latents, &latents], 0)?
        } else {
            latents.clone()
        };
        let model_input = scheduler.scale_model_input(model_input, timestep)?;
        let noise_pred = match &core.unet {
            SdUNet::Sd(unet) => unet.forward(&model_input, timestep as f64, text_embeddings)?,
            SdUNet::Sdxl(_) => {
                // animate is SD 1.5 / SD 2.1 only — gated at the
                // CLI boundary above. This arm is unreachable in
                // practice; bail loud so a future refactor can't
                // silently hit it.
                anyhow::bail!(
                    "plakat animate doesn't support SDXL backbones — \
                     guard at the CLI entry should have caught this."
                );
            }
        };
        let noise_pred = if do_cfg {
            let pieces = noise_pred.chunk(2, 0)?;
            let uncond = &pieces[0];
            let cond = &pieces[1];
            (uncond + ((cond - uncond)? * guidance)?)?
        } else {
            noise_pred
        };
        latents = scheduler.step(&noise_pred, timestep, &latents)?;
    }

    // VAE decode + save (same recipe t2i::Pipeline::generate uses).
    let image = core.vae.decode(&(&latents / 0.18215)?)?;
    let image = ((image / 2.0)? + 0.5)?.clamp(0f32, 1f32)?;
    let image = (image * 255.0)?
        .to_dtype(DType::U8)?
        .i(0)?
        .permute((1, 2, 0))?;
    let (oh, ow, _) = image.dims3()?;
    let buf = image.flatten_all()?.to_vec1::<u8>()?;
    crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, out_path)?;
    Ok(())
}

fn parse_size(s: &str) -> Result<(u32, u32)> {
    let parts: Vec<&str> = s.split(['x', 'X']).collect();
    if parts.len() != 2 {
        anyhow::bail!("--size must be WxH (e.g. 512x512), got {s:?}");
    }
    let w: u32 = parts[0]
        .parse()
        .with_context(|| format!("parsing width from {s:?}"))?;
    let h: u32 = parts[1]
        .parse()
        .with_context(|| format!("parsing height from {s:?}"))?;
    Ok((w, h))
}

fn write_gif(
    frame_paths: &[PathBuf],
    out_path: &std::path::Path,
    delay_ms: u16,
) -> Result<()> {
    use image::codecs::gif::{GifEncoder, Repeat};
    use image::Frame;
    let file = std::fs::File::create(out_path)
        .with_context(|| format!("creating {}", out_path.display()))?;
    let writer = std::io::BufWriter::new(file);
    let mut encoder = GifEncoder::new(writer);
    encoder
        .set_repeat(Repeat::Infinite)
        .with_context(|| "set GIF infinite-loop flag")?;
    for path in frame_paths {
        let img = image::open(path)
            .with_context(|| format!("opening frame {}", path.display()))?
            .to_rgba8();
        let delay = image::Delay::from_numer_denom_ms(delay_ms as u32, 1);
        let frame = Frame::from_parts(img, 0, 0, delay);
        encoder
            .encode_frame(frame)
            .with_context(|| format!("encoding frame {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn lerp_at_zero_returns_a() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &Device::Cpu).unwrap();
        let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], (3,), &Device::Cpu).unwrap();
        let out = lerp_tensors(&a, &b, 0.0).unwrap();
        let v: Vec<f32> = out.to_vec1().unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn lerp_at_one_returns_b() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &Device::Cpu).unwrap();
        let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], (3,), &Device::Cpu).unwrap();
        let out = lerp_tensors(&a, &b, 1.0).unwrap();
        let v: Vec<f32> = out.to_vec1().unwrap();
        assert_eq!(v, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn lerp_at_midpoint_averages() {
        let a = Tensor::from_vec(vec![0.0f32, 10.0], (2,), &Device::Cpu).unwrap();
        let b = Tensor::from_vec(vec![10.0f32, 0.0], (2,), &Device::Cpu).unwrap();
        let out = lerp_tensors(&a, &b, 0.5).unwrap();
        let v: Vec<f32> = out.to_vec1().unwrap();
        assert_eq!(v, vec![5.0, 5.0]);
    }

    #[test]
    fn lerp_clamps_t_below_zero() {
        // Negative t should pin to 0 (returns `a`), not extrapolate.
        let a = Tensor::from_vec(vec![5.0f32], (1,), &Device::Cpu).unwrap();
        let b = Tensor::from_vec(vec![10.0f32], (1,), &Device::Cpu).unwrap();
        let out = lerp_tensors(&a, &b, -0.5).unwrap();
        let v: Vec<f32> = out.to_vec1().unwrap();
        assert_eq!(v, vec![5.0]);
    }

    #[test]
    fn lerp_clamps_t_above_one() {
        // t > 1 should pin to 1 (returns `b`), not extrapolate.
        let a = Tensor::from_vec(vec![5.0f32], (1,), &Device::Cpu).unwrap();
        let b = Tensor::from_vec(vec![10.0f32], (1,), &Device::Cpu).unwrap();
        let out = lerp_tensors(&a, &b, 1.5).unwrap();
        let v: Vec<f32> = out.to_vec1().unwrap();
        assert_eq!(v, vec![10.0]);
    }

    #[test]
    fn parse_size_accepts_lowercase_x() {
        assert_eq!(parse_size("512x512").unwrap(), (512, 512));
        assert_eq!(parse_size("768x1024").unwrap(), (768, 1024));
    }

    #[test]
    fn parse_size_accepts_uppercase_x() {
        assert_eq!(parse_size("512X768").unwrap(), (512, 768));
    }

    #[test]
    fn parse_size_rejects_garbage() {
        assert!(parse_size("nonsense").is_err());
        assert!(parse_size("512").is_err());
        assert!(parse_size("axb").is_err());
    }
}
