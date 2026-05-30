//! `plakat animate` — frame-by-frame embedding interpolation.
//!
//! Two prompts (`--from` / `--to`) get encoded once each; the
//! denoise loop runs N times with linearly-lerped CLIP hidden
//! states. Frames land in `<out>/frame-NNNN.png` plus an optional
//! `<out>/animation.gif` when `--gif` is passed.
//!
//! Scope notes:
//!
//! * SD 1.5 / SD 2.1 / SDXL via the shared CLIP encoder lerp.
//!   SDXL also lerps the pooled `add_text_embeds` + builds a
//!   `add_time_ids` micro-conditioning vector.
//! * **v0.20 #9**: Flux (Dev / Schnell) via T5 + CLIP-L-pooled
//!   lerp + flow-match per-frame. Flux is guidance-distilled so
//!   there's no CFG batching — `--negative` is a no-op on Flux
//!   variants (we warn at the CLI layer if the user passes one).
//!   The Kontext / Fill / Canny / Depth Flux variants are
//!   refused: they need a reference / conditioning image which
//!   doesn't fit the per-frame morph contract.
//! * SD3 / SD3.5 animate is a follow-up (needs three-encoder
//!   lerp + the rectified-flow MMDiT integrator wiring).
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

    /// Second prompt — the last frame renders this. Required for
    /// the v0.20 lerp morph (default mode); **optional and ignored**
    /// in `--animatediff` mode (single-prompt motion-coherent
    /// generation reuses `--from` for every frame).
    #[arg(long, default_value = "")]
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

    /// SD 1.5 / SD 2.1 / SDXL model. Defaults to `sd15`. Flux / SD3
    /// bail loud — they use T5 + rectified-flow and need separate
    /// animate machinery (deferred).
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

    /// **v0.26**: enable AnimateDiff mode (SD 1.5 only). Switches
    /// from the v0.20 prompt-lerp morph to motion-coherent N-frame
    /// generation using a downloaded AnimateDiff V3 motion adapter
    /// (`guoyww/animatediff-motion-adapter-v1-5-3`, ~1.4 GB).
    /// Single prompt is the same across all frames (`--from`);
    /// `--to` is ignored. Frame count is the motion adapter's
    /// trained window (default 16); the adapter's
    /// `motion_max_seq_length = 32` is a hard cap. Output formats
    /// expand in phase 5 (`--format mp4` / `webm`).
    #[arg(long, default_value_t = false)]
    pub animatediff: bool,

    /// **v0.26**: motion LoRAs to stack onto the AnimateDiff
    /// motion adapter. Same `LoraSpec` grammar as `--lora`:
    /// `hf:user/repo:0.7`, `civitai:NNNNNN:0.5`, or a local path.
    /// Lets you pick community motion LoRAs (zoom-in / pan-left /
    /// panic / etc.) from `guoyww/animatediff-motion-lora-*`.
    /// Repeatable. No-op outside `--animatediff` mode.
    #[arg(long = "motion-lora", value_name = "SPEC")]
    pub motion_loras: Vec<crate::pipelines::lora::LoraSpec>,

    /// **v0.26**: per-LoRA scale multiplier for `--motion-lora`,
    /// stacked on top of each spec's own `:scale` suffix. Mirrors
    /// `--lora-scale` from `plakat generate`. Default `1.0`.
    #[arg(long = "motion-lora-scale", default_value_t = 1.0)]
    pub motion_lora_scale: f32,

    /// **v0.26**: output format(s). `frames` is the default
    /// (always writes per-frame PNGs). `gif` adds an animated
    /// GIF (equivalent to the v0.20 `--gif` flag). `mp4` / `webm`
    /// invoke ffmpeg (must be on `$PATH`) to encode a video.
    /// `all` writes every format.
    ///
    /// Composes with `--gif`: passing `--gif` is equivalent to
    /// `--format gif`. When both are set, `--format` wins.
    #[arg(long, value_name = "FMT", default_value = "frames")]
    pub format: crate::imaging::video::Format,

    /// v0.18 phase 6: skip the A1111 `parameters` PNG tEXt chunk
    /// and the `.json` sidecar that animate writes alongside each
    /// `frame-NNNN.png`. Default off — metadata helps you re-render
    /// any frame from its sidecar's `Lerp t` / `Animate from` /
    /// `Animate to` entries.
    #[arg(long = "no-metadata", default_value_t = false)]
    pub no_metadata: bool,

    /// v0.19: skip frames already on disk. Scans `<out>/frame-NNNN.png`
    /// over `0..frames` and re-runs only the missing ones. Crucial
    /// when a long animate run crashes on frame 23 of 24 — without
    /// this flag the only recovery is rerunning all 24 from frame 0.
    /// Mirrors the scenario `--resume` semantics added in v0.17.
    #[arg(long, default_value_t = false)]
    pub resume: bool,

    /// **v0.27 phase 3 / v0.28 phase 0**: ControlNet conditioning kind
    /// for the AnimateDiff path. `depth | canny | openpose | lineart
    /// | softedge`. Required to enable CN; without it the other
    /// `--control-*` flags are ignored. The single conditioning
    /// image is applied to every frame (the same hint at every
    /// frame — per-frame video control deferred to v0.29+).
    ///
    /// For multi-ControlNet (depth + canny stacked etc.) use
    /// `--control-spec` instead — see below.
    ///
    /// SD 1.5 + SDXL (v0.27 phase 4). No-op outside `--animatediff` mode.
    #[arg(long = "control", value_name = "KIND")]
    pub control: Option<String>,

    /// Pre-rendered conditioning image (depth map, canny edge map,
    /// etc.). Mutually exclusive with `--control-from`.
    #[arg(long = "control-image", value_name = "PATH")]
    pub control_image: Option<PathBuf>,

    /// Source image that the annotator auto-converts into the
    /// conditioning. Mutually exclusive with `--control-image`.
    #[arg(long = "control-from", value_name = "PATH")]
    pub control_from: Option<PathBuf>,

    /// ControlNet strength multiplier. 1.0 is the diffusers default;
    /// 0.5 = soft guidance, 1.5+ = heavy.
    #[arg(long = "control-strength", default_value_t = 1.0)]
    pub control_strength: f32,

    /// **v0.28 phase 0**: full ControlNet spec, repeatable for
    /// multi-ControlNet AnimateDiff (depth + canny stacked etc.).
    /// Each occurrence stacks one conditioner; residuals from every
    /// conditioner are summed per denoise step inside the motion
    /// UNet.
    ///
    /// Grammar: `KIND[:option=value]*` where KIND ∈ {depth, canny,
    /// openpose, lineart, softedge} and options are `image=PATH`,
    /// `from=PATH`, `strength=F`, `start=F`, `end=F`. Examples:
    ///
    ///   --control-spec 'depth:from=source.jpg'
    ///   --control-spec 'canny:image=edges.png:strength=0.5'
    ///
    /// Mutually exclusive with the legacy single-conditioner flags
    /// (`--control`, `--control-image`, `--control-from`,
    /// `--control-strength`). All conditioners in the stack share
    /// the model variant — mixing SD 1.5 / SDXL is not supported.
    #[arg(
        long = "control-spec",
        value_name = "SPEC",
        conflicts_with_all = [
            "control", "control_image", "control_from", "control_strength",
        ],
    )]
    pub control_specs: Vec<crate::pipelines::controlnet::ControlSpec>,

    /// **v0.28 phase 1**: AnimateLCM mode for 4-step generation
    /// (~5× speedup vs V3 + DDIM at 20 steps). Switches the motion
    /// adapter from V3 to `wangfuyun/AnimateLCM`, the scheduler to
    /// `lcm`, and the defaults to `--steps 4 --guidance 1.5`.
    /// User-supplied `--steps` / `--guidance` still take precedence
    /// — pass `--lcm --steps 8` for higher quality at 2× cost.
    /// SD 1.5 only in v0.28; SDXL AnimateLCM repo isn't publicly
    /// available.
    #[arg(long = "lcm", default_value_t = false)]
    pub lcm: bool,

    /// **v0.27 phase 5**: per-window frame count for long-form
    /// sliding-window AnimateDiff. The motion adapter is trained at
    /// 16 frames; values > 32 exceed the trained positional
    /// embedding's `motion_max_seq_length`. No-op when `--frames` is
    /// ≤ `--window-size` (single-window path).
    #[arg(long = "window-size", default_value_t = 16)]
    pub window_size: u32,

    /// **v0.27 phase 5**: overlap between consecutive sliding-
    /// windows in frames. Each window covers
    /// `[i*stride, i*stride + window-size)` where
    /// `stride = window-size - window-overlap`. Higher overlap
    /// reduces seam artefacts at the cost of more redundant compute.
    /// Default 4 (25 % of the 16-frame window) is the community
    /// sweet spot.
    #[arg(long = "window-overlap", default_value_t = 4)]
    pub window_overlap: u32,

    /// **v0.32 phase 0**: FreeNoise — pre-generate a full-length
    /// noise tensor at the user's seed, then slice per sliding-
    /// window so adjacent windows share noise in the overlap region.
    /// Eliminates the cross-fade seam artefact v0.27 phase 5's
    /// random-per-window approach exhibits on >32-frame animations.
    /// Cao et al., "FreeNoise: Tuning-Free Longer Video Diffusion".
    ///
    /// No-op when `--frames ≤ --window-size` (single-window path —
    /// no adjacent windows to share noise across). Opt-in to keep
    /// existing `--seed` reproducibility unchanged when the flag is
    /// off — output is byte-identical to v0.31 in that case. SD 1.5
    /// + SDXL.
    #[arg(long = "free-noise", default_value_t = false)]
    pub free_noise: bool,
}

pub async fn run(args: AnimateArgs, device: Device) -> Result<()> {
    // v0.27 phase 0: AnimateDiff dispatch. When --animatediff is
    // set, route through the AnimateDiff stack (motion adapter +
    // vendored motion UNet) instead of the v0.20 prompt-lerp morph.
    if args.animatediff {
        return run_animatediff(args, device).await;
    }

    if args.to.is_empty() {
        anyhow::bail!(
            "--to is required for prompt-lerp animate mode (SD-family, Flux, \
             SD3). Pass `--to \"<second prompt>\"` for the lerp endpoint, or \
             switch to AnimateDiff with `--animatediff` (single-prompt \
             motion-coherent mode, SD 1.5 only)."
        );
    }
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

    // Model gate. v0.20 #9: split into SD-family + Flux dispatch.
    // SD3 / SD3.5 stays gated — it needs three-encoder lerp + the
    // rectified-flow MMDiT integrator wiring, which lands in a
    // follow-up.
    let repo = if args.model.contains('/') {
        args.model.clone()
    } else {
        crate::hf::resolve_alias(&args.model).to_string()
    };
    let t2i_variant = crate::pipelines::t2i::Variant::detect(&repo);
    if t2i_variant.is_sd3() {
        // v0.26 phase 6: SD3 / SD3.5 animate via 3-encoder lerp +
        // rectified-flow MMDiT per frame. Mirrors the v0.20 Flux
        // animate pattern (Flux has T5 + CLIP-L; SD3 adds CLIP-G).
        return run_sd3(args, device, repo, t2i_variant).await;
    }
    if t2i_variant.is_flux() {
        // Flux Kontext / Fill / Canny / Depth need a reference or
        // conditioning image per call — there's no place to plug
        // one in `--from` / `--to`. Refuse up front rather than
        // OOM the user 30s into a load.
        let fvar = map_to_flux_variant(t2i_variant);
        if !matches!(fvar, crate::pipelines::flux::Variant::Dev | crate::pipelines::flux::Variant::Schnell) {
            anyhow::bail!(
                "`plakat animate` on Flux supports `--model flux-dev` and \
                 `--model flux-schnell` (got --model {} = {:?}). Kontext / \
                 Fill / Canny / Depth need a reference image per call which \
                 doesn't fit the per-frame morph contract.",
                args.model,
                fvar,
            );
        }
        return run_flux(args, device, fvar).await;
    }
    let variant = SdVariant::detect(&repo);
    if !matches!(variant, SdVariant::Sd15 | SdVariant::Sd21 | SdVariant::Sdxl) {
        anyhow::bail!(
            "`plakat animate` is SD 1.5 / SD 2.1 / SDXL or Flux Dev / Schnell \
             in this release (got --model {} = {:?}).",
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
            SdVariant::Sdxl => "SDXL",
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
    // once each. Frame-time work is just (a) lerp the cached tensors
    // and (b) run the denoise loop.
    let encode_spin = crate::ui::progress::spinner("Encoding endpoint prompts");
    let endpoints = match variant {
        SdVariant::Sdxl => Endpoints::Sdxl(SdxlEndpoints::encode(
            &core,
            &args.from,
            &args.to,
            &args.negative,
            do_cfg,
            width,
            height,
            dtype,
        )?),
        _ => Endpoints::Sd(SdEndpoints::encode(
            &core,
            &args.from,
            &args.to,
            &args.negative,
            do_cfg,
            dtype,
        )?),
    };
    encode_spin.finish_with_message("✓ endpoint embeddings ready");

    // Seed: explicit when given, otherwise generate one + log so
    // the run is reproducible if the user wants to re-render.
    let seed = args.seed.unwrap_or_else(rand::random) & (u32::MAX as u64);
    crate::ui::progress::println(&format!(
        "  animation: {} frames, seed {seed}, model {}, {}x{}",
        args.frames, args.model, width, height,
    ));

    // v0.18 phase 6: build per-frame metadata when `--no-metadata`
    // wasn't passed. The model / seed / steps / guidance / scheduler
    // fields stay constant across the run; only the prompt (synthetic
    // "lerp(t): from | to" description) and the extras (structured
    // t / from / to) change per frame.
    let scheduler_name = format!("{:?}", args.scheduler).to_lowercase();

    // v0.19: --resume skips frame-NNNN.png files that already exist
    // on disk. The lerp parameter `t` is recomputed identically per
    // frame index, so a partial run can be completed without
    // re-rendering the frames already on disk.
    let mut skipped = 0u32;
    let mut frame_paths: Vec<PathBuf> = Vec::with_capacity(args.frames as usize);
    for frame_i in 0..args.frames {
        let t = if args.frames == 1 {
            0.0
        } else {
            frame_i as f64 / (args.frames - 1) as f64
        };
        let frame_path = args.out.join(format!("frame-{frame_i:04}.png"));
        if args.resume && frame_path.exists() {
            skipped += 1;
            frame_paths.push(frame_path.clone());
            crate::ui::progress::println(&format!(
                "  frame {}/{} → {} (t={:.3}) {}",
                frame_i + 1,
                args.frames,
                frame_path.display(),
                t,
                console::style("(resume — already on disk)").dim(),
            ));
            continue;
        }
        let frame = endpoints.lerp_at(t)?;
        let meta = if !args.no_metadata {
            let prompt_desc =
                format!("lerp({t:.4}): {:?} | {:?}", args.from, args.to);
            let mut m = crate::imaging::metadata::GenerationMetadata::new(
                prompt_desc,
                args.model.clone(),
                seed,
                args.steps,
                args.guidance,
                scheduler_name.clone(),
                width,
                height,
            );
            m.negative = args.negative.clone();
            m.mode = Some("animate".to_string());
            m.with_animate_lerp(t, &args.from, &args.to);
            Some(m)
        } else {
            None
        };
        denoise_one_frame(
            &core,
            &frame,
            width,
            height,
            args.steps,
            args.guidance,
            args.scheduler,
            seed,
            &frame_path,
            meta.as_ref(),
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

    if args.resume && skipped > 0 {
        crate::ui::progress::println(&format!(
            "  {} skipped {skipped}/{} frame(s) that were already on disk",
            console::style("(resume)").dim(),
            args.frames,
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

/// v0.18 phase 5: SDXL single-branch encode. Returns
/// `(hidden_concat_2048d, pooled_g_1280d)` — the dual-encoder
/// penultimate stack ready for cross-attention, and the pooled
/// CLIP-G output that drives SDXL's `add_text_embeds`.
///
/// Mirrors `t2i::embed_xl` for a single branch (no CFG concat).
/// Animate-XL uses the unweighted CLIP tokenize path; per-prompt
/// attention syntax in animate prompts is uncommon enough that the
/// extra branch isn't worth the complexity here. The plain `--from`
/// / `--to` prompts still benefit from the full SDXL micro-
/// conditioning chain at frame time.
fn encode_branch_xl(core: &SdCore, text: &str, dtype: DType) -> Result<(Tensor, Tensor)> {
    use crate::pipelines::vendored_clip::ClipTextTransformer;

    let tok_g = core
        .tokenizer_g
        .as_ref()
        .ok_or_else(|| anyhow!("SDXL animate needs tokenizer_g"))?;
    let enc_g = core
        .text_encoder_g
        .as_ref()
        .ok_or_else(|| anyhow!("SDXL animate needs text_encoder_g"))?;
    let cfg_g = core
        .cfg
        .clip2
        .as_ref()
        .ok_or_else(|| anyhow!("SDXL config missing clip2"))?;

    // CLIP-L tokenize + penultimate hidden state.
    let pad_l: u32 = match &core.cfg.clip.pad_with {
        Some(s) => core
            .tokenizer_l
            .token_to_id(s)
            .ok_or_else(|| anyhow!("CLIP-L tokenizer missing pad token {s:?}"))?,
        None => core
            .tokenizer_l
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| anyhow!("CLIP-L tokenizer missing <|endoftext|>"))?,
    };
    let mut ids_l = core
        .tokenizer_l
        .encode(text, true)
        .map_err(|e| anyhow!("CLIP-L encode of {text:?}: {e}"))?
        .get_ids()
        .to_vec();
    ids_l.resize(core.cfg.clip.max_position_embeddings, pad_l);
    let ids_l_t = Tensor::new(ids_l.as_slice(), &core.device)?.unsqueeze(0)?;
    let (_final_l, hidden_l) = ClipTextTransformer::forward_until_encoder_layer(
        &core.text_encoder_l,
        &ids_l_t,
        usize::MAX,
        -2,
    )?;
    let hidden_l = hidden_l.to_dtype(dtype)?;

    // CLIP-G tokenize + (penult, pooled).
    let pad_g: u32 = match &cfg_g.pad_with {
        Some(s) => tok_g
            .token_to_id(s)
            .ok_or_else(|| anyhow!("CLIP-G tokenizer missing pad token {s:?}"))?,
        None => tok_g
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| anyhow!("CLIP-G tokenizer missing <|endoftext|>"))?,
    };
    let mut ids_g = tok_g
        .encode(text, true)
        .map_err(|e| anyhow!("CLIP-G encode of {text:?}: {e}"))?
        .get_ids()
        .to_vec();
    ids_g.resize(cfg_g.max_position_embeddings, pad_g);
    let ids_g_t = Tensor::new(ids_g.as_slice(), &core.device)?.unsqueeze(0)?;
    let (hidden_g, pooled_g) = enc_g.forward_for_sdxl(&ids_g_t)?;
    let hidden_g = hidden_g.to_dtype(dtype)?;
    let pooled_g = pooled_g.to_dtype(dtype)?;

    // Concat CLIP-L penult (768) + CLIP-G penult (1280) along channel
    // dim → (1, 77, 2048). Matches the SDXL UNet's cross-attention
    // expectation.
    let hidden = Tensor::cat(&[&hidden_l, &hidden_g], candle_core::D::Minus1)?;
    Ok((hidden, pooled_g))
}

/// Per-frame inputs to the denoise loop. SD-family carries only the
/// cross-attention hidden state. SDXL additionally needs the pooled
/// `add_text_embeds` and the `add_time_ids` micro-conditioning
/// vector (CFG-stacked if guidance > 1).
struct Frame {
    text_embeddings: Tensor,
    add_text_embeds: Option<Tensor>,
    add_time_ids: Option<Tensor>,
}

enum Endpoints {
    Sd(SdEndpoints),
    Sdxl(SdxlEndpoints),
}

impl Endpoints {
    fn lerp_at(&self, t: f64) -> Result<Frame> {
        match self {
            Endpoints::Sd(e) => e.lerp_at(t),
            Endpoints::Sdxl(e) => e.lerp_at(t),
        }
    }
}

struct SdEndpoints {
    cond_a: Tensor,
    cond_b: Tensor,
    uncond: Option<Tensor>,
}

impl SdEndpoints {
    fn encode(
        core: &SdCore,
        from: &str,
        to: &str,
        negative: &str,
        do_cfg: bool,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            cond_a: encode_branch(core, from, dtype)?,
            cond_b: encode_branch(core, to, dtype)?,
            uncond: if do_cfg {
                Some(encode_branch(core, negative, dtype)?)
            } else {
                None
            },
        })
    }

    fn lerp_at(&self, t: f64) -> Result<Frame> {
        let lerped = lerp_tensors(&self.cond_a, &self.cond_b, t)?;
        let text_embeddings = match self.uncond.as_ref() {
            Some(u) => Tensor::cat(&[u, &lerped], 0)?,
            None => lerped,
        };
        Ok(Frame {
            text_embeddings,
            add_text_embeds: None,
            add_time_ids: None,
        })
    }
}

struct SdxlEndpoints {
    cond_a_hidden: Tensor,
    cond_a_pooled: Tensor,
    cond_b_hidden: Tensor,
    cond_b_pooled: Tensor,
    uncond_hidden: Option<Tensor>,
    uncond_pooled: Option<Tensor>,
    /// Pre-built (CFG-stacked when do_cfg) add_time_ids vector.
    /// Constant across frames so building it once amortises the cost
    /// across the loop.
    add_time_ids: Tensor,
    do_cfg: bool,
}

impl SdxlEndpoints {
    #[allow(clippy::too_many_arguments)]
    fn encode(
        core: &SdCore,
        from: &str,
        to: &str,
        negative: &str,
        do_cfg: bool,
        width: u32,
        height: u32,
        dtype: DType,
    ) -> Result<Self> {
        let (a_h, a_p) = encode_branch_xl(core, from, dtype)?;
        let (b_h, b_p) = encode_branch_xl(core, to, dtype)?;
        let (u_h, u_p) = if do_cfg {
            let (h, p) = encode_branch_xl(core, negative, dtype)?;
            (Some(h), Some(p))
        } else {
            (None, None)
        };
        // add_time_ids: 6 floats per row for SDXL base —
        // [orig_h, orig_w, crop_top, crop_left, target_h, target_w].
        // build_add_time_ids_base builds (1, 6); CFG stacks
        // uncond + cond identically (diffusers does the same).
        let one = crate::pipelines::sdxl_unet::build_add_time_ids_base(
            height,
            width,
            &core.device,
            dtype,
        )?;
        let add_time_ids = if do_cfg {
            Tensor::cat(&[&one, &one], 0)?
        } else {
            one
        };
        Ok(Self {
            cond_a_hidden: a_h,
            cond_a_pooled: a_p,
            cond_b_hidden: b_h,
            cond_b_pooled: b_p,
            uncond_hidden: u_h,
            uncond_pooled: u_p,
            add_time_ids,
            do_cfg,
        })
    }

    fn lerp_at(&self, t: f64) -> Result<Frame> {
        let hidden = lerp_tensors(&self.cond_a_hidden, &self.cond_b_hidden, t)?;
        let pooled = lerp_tensors(&self.cond_a_pooled, &self.cond_b_pooled, t)?;
        let (text_embeddings, add_text_embeds) = if self.do_cfg {
            let uh = self.uncond_hidden.as_ref().unwrap();
            let up = self.uncond_pooled.as_ref().unwrap();
            (
                Tensor::cat(&[uh, &hidden], 0)?,
                Tensor::cat(&[up, &pooled], 0)?,
            )
        } else {
            (hidden, pooled)
        };
        Ok(Frame {
            text_embeddings,
            add_text_embeds: Some(add_text_embeds),
            add_time_ids: Some(self.add_time_ids.clone()),
        })
    }
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
/// VAE. SD 1.5 / SD 2.1 / SDXL — the unified `SdUNet::forward`
/// wrapper handles either variant; the caller passes the SDXL
/// extras (`add_text_embeds`, `add_time_ids`) via `Frame` when
/// they apply. Saves the result to `out_path`. When `metadata` is
/// `Some`, the saved PNG carries the Auto1111 `parameters` tEXt
/// chunk and a sibling JSON sidecar.
#[allow(clippy::too_many_arguments)]
fn denoise_one_frame(
    core: &SdCore,
    frame: &Frame,
    width: u32,
    height: u32,
    steps: usize,
    guidance: f64,
    scheduler_kind: SchedulerKind,
    seed: u64,
    out_path: &std::path::Path,
    metadata: Option<&crate::imaging::metadata::GenerationMetadata>,
) -> Result<()> {
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
        let noise_pred = core.unet.forward(
            &model_input,
            timestep as f64,
            &frame.text_embeddings,
            frame.add_text_embeds.as_ref(),
            frame.add_time_ids.as_ref(),
        )?;
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
    // SDXL's VAE was trained with scaling_factor 0.13025 vs SD 1.5/2.1
    // at 0.18215 — using the wrong constant produces washed-out output.
    let vae_scale = core.variant.vae_scale();
    let image = core.vae.decode(&(&latents / vae_scale)?)?;
    let image = ((image / 2.0)? + 0.5)?.clamp(0f32, 1f32)?;
    let image = (image * 255.0)?
        .to_dtype(DType::U8)?
        .i(0)?
        .permute((1, 2, 0))?;
    let (oh, ow, _) = image.dims3()?;
    let buf = image.flatten_all()?.to_vec1::<u8>()?;
    match metadata {
        Some(meta) => crate::imaging::io::save_rgb_u8_with_metadata(
            &buf,
            ow as u32,
            oh as u32,
            out_path,
            meta,
        )?,
        None => crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, out_path)?,
    }
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

/// v0.20 #9: map the unified `t2i::Variant` to `flux::Variant`.
/// Mirrors the same match inside `t2i::run` so the alias lookup
/// stays in one place. Defaults to `Schnell` for anything not
/// explicitly listed — the caller has already vetted the variant
/// via `is_flux()` so the fallback only fires on future Flux
/// additions.
fn map_to_flux_variant(
    v: crate::pipelines::t2i::Variant,
) -> crate::pipelines::flux::Variant {
    use crate::pipelines::flux::Variant as FV;
    use crate::pipelines::t2i::Variant;
    match v {
        Variant::FluxDev => FV::Dev,
        Variant::FluxFillDev => FV::FillDev,
        Variant::FluxCannyDev => FV::CannyDev,
        Variant::FluxDepthDev => FV::DepthDev,
        Variant::FluxKontextDev => FV::KontextDev,
        _ => FV::Schnell,
    }
}

/// v0.26 phase 6: SD3 / SD3.5 animate. Mirrors `run_flux` but
/// uses SD3's three-encoder embeddings (pooled `y` + joint
/// `context`). Loads sd3::Pipeline once, pre-encodes both
/// endpoint prompts + the negative once, then per frame:
/// lerp `(pos_y, pos_ctx)` → call `Pipeline::animate_frame` →
/// save with metadata.
async fn run_sd3(
    args: AnimateArgs,
    device: Device,
    repo: String,
    t2i_variant: crate::pipelines::t2i::Variant,
) -> Result<()> {
    use crate::pipelines::sd3;

    if args.to.is_empty() {
        anyhow::bail!(
            "--to is required for `plakat animate --model sd3*` \
             (it's the second-prompt endpoint of the lerp)."
        );
    }
    let (width, height) = parse_size(&args.size)?;
    if width % 16 != 0 || height % 16 != 0 {
        anyhow::bail!(
            "SD3 animate requires --size dims divisible by 16 (got {}x{}). \
             Try 512x512 / 768x768 / 1024x1024.",
            width,
            height
        );
    }
    if args.frames < 2 {
        anyhow::bail!("--frames must be ≥ 2 (got {})", args.frames);
    }

    // Map t2i::Variant → sd3::Variant.
    use crate::pipelines::t2i::Variant as TV;
    let sd3_variant = match t2i_variant {
        TV::Sd3Medium => sd3::Variant::Sd3Medium,
        TV::Sd35Medium => sd3::Variant::Sd35Medium,
        TV::Sd35Large => sd3::Variant::Sd35Large,
        TV::Sd35LargeTurbo => sd3::Variant::Sd35LargeTurbo,
        other => anyhow::bail!(
            "internal: SD3 dispatch reached non-SD3 variant {other:?}"
        ),
    };

    let load_spin = crate::ui::progress::spinner(&format!(
        "Loading SD3 {:?} for animation",
        sd3_variant
    ));
    let mut pipeline = sd3::Pipeline::load(sd3::LoadRequest {
        variant: sd3_variant,
        repo,
        device: device.clone(),
        loras: Vec::new(),
        lora_scale: 1.0,
        controlnets: Vec::new(),
    })
    .await?;
    load_spin.finish_with_message("✓ SD3 backbone ready");

    let encode_spin =
        crate::ui::progress::spinner("Encoding SD3 endpoint prompts");
    let (pos_y_a, pos_ctx_a) = pipeline.encode_for_animate(&args.from)?;
    let (pos_y_b, pos_ctx_b) = pipeline.encode_for_animate(&args.to)?;
    let (neg_y, neg_ctx) = pipeline.encode_for_animate(&args.negative)?;
    encode_spin.finish_with_message("✓ endpoint embeddings ready");

    let seed = args.seed.unwrap_or_else(rand::random) & (u32::MAX as u64);
    crate::ui::progress::println(&format!(
        "  animation: {} frames, seed {seed}, model {}, {}x{}",
        args.frames, args.model, width, height,
    ));

    let mut skipped = 0u32;
    let mut frame_paths: Vec<PathBuf> = Vec::with_capacity(args.frames as usize);
    for frame_i in 0..args.frames {
        let t = if args.frames == 1 {
            0.0
        } else {
            frame_i as f64 / (args.frames - 1) as f64
        };
        let frame_path = args.out.join(format!("frame-{frame_i:04}.png"));
        if args.resume && frame_path.exists() {
            skipped += 1;
            frame_paths.push(frame_path.clone());
            crate::ui::progress::println(&format!(
                "  frame {}/{} → {} (t={:.3}) {}",
                frame_i + 1,
                args.frames,
                frame_path.display(),
                t,
                console::style("(resume — already on disk)").dim(),
            ));
            continue;
        }
        let pos_y_lerp = lerp_tensors(&pos_y_a, &pos_y_b, t)?;
        let pos_ctx_lerp = lerp_tensors(&pos_ctx_a, &pos_ctx_b, t)?;
        let (buf, ow, oh) = pipeline.animate_frame(
            &pos_y_lerp,
            &pos_ctx_lerp,
            &neg_y,
            &neg_ctx,
            width,
            height,
            args.steps,
            args.guidance,
            seed,
        )?;
        if !args.no_metadata {
            let prompt_desc =
                format!("lerp({t:.4}): {:?} | {:?}", args.from, args.to);
            let mut m = crate::imaging::metadata::GenerationMetadata::new(
                prompt_desc,
                args.model.clone(),
                seed,
                args.steps,
                args.guidance,
                format!("{:?}", args.scheduler).to_lowercase(),
                ow,
                oh,
            );
            m.with_animate_lerp(t, &args.from, &args.to);
            crate::imaging::io::save_rgb_u8_with_metadata(
                &buf, ow, oh, &frame_path, &m,
            )?;
        } else {
            crate::imaging::io::save_rgb_u8(&buf, ow, oh, &frame_path)?;
        }
        crate::ui::progress::println(&format!(
            "  frame {}/{} → {} (t={:.3})",
            frame_i + 1,
            args.frames,
            frame_path.display(),
            t,
        ));
        frame_paths.push(frame_path);
    }

    if args.format.needs_gif() || args.gif {
        let gif_path = args.out.join("animation.gif");
        write_gif(&frame_paths, &gif_path, args.gif_delay_ms)?;
        crate::ui::progress::println(&format!("→ {}", gif_path.display()));
    }
    if args.format.needs_mp4() {
        let pattern = args
            .out
            .join("frame-%04d.png")
            .to_string_lossy()
            .into_owned();
        crate::imaging::video::frames_to_mp4(
            &pattern,
            &args.out.join("animation.mp4"),
            (1000 / args.gif_delay_ms.max(1) as u32).max(1),
        )?;
    }
    if args.format.needs_webm() {
        let pattern = args
            .out
            .join("frame-%04d.png")
            .to_string_lossy()
            .into_owned();
        crate::imaging::video::frames_to_webm(
            &pattern,
            &args.out.join("animation.webm"),
            (1000 / args.gif_delay_ms.max(1) as u32).max(1),
        )?;
    }

    crate::ui::progress::println(&format!(
        "✓ {} frames written ({skipped} skipped)",
        args.frames as usize - skipped as usize,
    ));
    Ok(())
}

/// v0.20 #9: Flux animate dispatch. Loads `flux::Pipeline` once,
/// pre-encodes both endpoint prompts via `Pipeline::encode_prompt`,
/// then per frame: lerp `(clip_pooled, t5_emb)` → call
/// `Pipeline::animate_frame` → save with metadata.
///
/// Mirrors the SD path's frame/resume/gif structure as closely as
/// possible so behavioural differences across families stay
/// contained to the encode + denoise seams.
async fn run_flux(
    args: AnimateArgs,
    device: Device,
    fvar: crate::pipelines::flux::Variant,
) -> Result<()> {
    use crate::pipelines::flux;

    let (width, height) = parse_size(&args.size)?;

    if !args.negative.is_empty() {
        tracing::warn!(
            target: "plakat",
            "--negative is ignored on Flux animate: Flux is guidance-distilled \
             (no CFG batching), so the unconditional branch can't be steered. \
             Drop --negative or move the suppressors into the positive prompts."
        );
    }

    let load_spin = crate::ui::progress::spinner(&format!(
        "Loading Flux {} for animation",
        match fvar {
            flux::Variant::Dev => "Dev",
            flux::Variant::Schnell => "Schnell",
            _ => "unknown",
        }
    ));
    let repo = if args.model.contains('/') {
        args.model.clone()
    } else {
        crate::hf::resolve_alias(&args.model).to_string()
    };
    let mut pipeline = flux::Pipeline::load(flux::LoadRequest {
        variant: fvar,
        repo,
        device,
        loras: Vec::new(),
        lora_scale: 1.0,
        controlnets: Vec::new(),
        quantize_t5: false,
        flux_quant_level: None,
        t5_quant_level: None,
        redux: false,
    })
    .await?;
    load_spin.finish_with_message("✓ Flux backbone ready");

    let encode_spin =
        crate::ui::progress::spinner("Encoding Flux endpoint prompts");
    let (clip_a, t5_a) = pipeline.encode_prompt(&args.from)?;
    let (clip_b, t5_b) = pipeline.encode_prompt(&args.to)?;
    encode_spin.finish_with_message("✓ endpoint embeddings ready");

    let seed = args.seed.unwrap_or_else(rand::random) & (u32::MAX as u64);
    crate::ui::progress::println(&format!(
        "  animation: {} frames, seed {seed}, model {}, {}x{}",
        args.frames, args.model, width, height,
    ));

    let scheduler_name = format!("{:?}", args.scheduler).to_lowercase();
    let mut skipped = 0u32;
    let mut frame_paths: Vec<PathBuf> = Vec::with_capacity(args.frames as usize);
    for frame_i in 0..args.frames {
        let t = if args.frames == 1 {
            0.0
        } else {
            frame_i as f64 / (args.frames - 1) as f64
        };
        let frame_path = args.out.join(format!("frame-{frame_i:04}.png"));
        if args.resume && frame_path.exists() {
            skipped += 1;
            frame_paths.push(frame_path.clone());
            crate::ui::progress::println(&format!(
                "  frame {}/{} → {} (t={:.3}) {}",
                frame_i + 1,
                args.frames,
                frame_path.display(),
                t,
                console::style("(resume — already on disk)").dim(),
            ));
            continue;
        }
        let clip_lerp = lerp_tensors(&clip_a, &clip_b, t)?;
        let t5_lerp = lerp_tensors(&t5_a, &t5_b, t)?;
        let (buf, ow, oh) = pipeline.animate_frame(
            &clip_lerp,
            &t5_lerp,
            width,
            height,
            args.steps,
            args.guidance,
            seed,
        )?;
        if !args.no_metadata {
            let prompt_desc =
                format!("lerp({t:.4}): {:?} | {:?}", args.from, args.to);
            let mut m = crate::imaging::metadata::GenerationMetadata::new(
                prompt_desc,
                args.model.clone(),
                seed,
                args.steps,
                args.guidance,
                scheduler_name.clone(),
                width,
                height,
            );
            m.negative = args.negative.clone();
            m.mode = Some("animate".to_string());
            m.with_animate_lerp(t, &args.from, &args.to);
            crate::imaging::io::save_rgb_u8_with_metadata(
                &buf, ow, oh, &frame_path, &m,
            )?;
        } else {
            crate::imaging::io::save_rgb_u8(&buf, ow, oh, &frame_path)?;
        }
        frame_paths.push(frame_path);
        crate::ui::progress::println(&format!(
            "  frame {}/{} → {} (t={:.3})",
            frame_i + 1,
            args.frames,
            frame_paths.last().unwrap().display(),
            t,
        ));
    }

    if args.resume && skipped > 0 {
        crate::ui::progress::println(&format!(
            "  {} skipped {skipped}/{} frame(s) that were already on disk",
            console::style("(resume)").dim(),
            args.frames,
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

/// v0.27 phase 0: AnimateDiff V3 dispatch. Loads the V3 motion
/// stack (with optional motion LoRAs) on top of an SD 1.5 backbone,
/// runs N-frame inference, and writes per-frame PNGs plus optional
/// GIF / MP4 / WebM via the [`crate::imaging::video::Format`] enum.
///
/// Single-prompt mode: `--from` is the prompt, `--to` is ignored
/// (logged at info level if non-empty).
async fn run_animatediff(args: AnimateArgs, device: Device) -> Result<()> {
    use crate::pipelines::animatediff::AnimateDiffPipeline;

    // Variant detection. SD 1.5 + SDXL both supported in v0.27;
    // SD 2.1 / Flux / SD3 / etc. bail loud since no motion adapter
    // exists upstream. Detect on the resolved repo path because
    // `SdVariant::detect` keys off substrings like "xl" / "2-1"
    // which are present in repo ids but not in plakat's short
    // aliases ("sd21" doesn't match the "2-1" / "2.1" / "v2" gate).
    let resolved_for_detect = if args.model.contains('/') {
        args.model.clone()
    } else {
        crate::hf::resolve_alias(&args.model).to_string()
    };
    let variant = SdVariant::detect(&resolved_for_detect);
    let is_sd15_or_sdxl =
        matches!(variant, SdVariant::Sd15 | SdVariant::Sdxl);
    if !is_sd15_or_sdxl {
        anyhow::bail!(
            "`--animatediff` requires --model sd15 or an SDXL alias \
             (got --model {} = {:?}). SD 2.1 / Flux / SD3 have no \
             upstream motion adapter.",
            args.model,
            variant,
        );
    }
    // v0.27 phase 5: with --window-size ≤ 32, --frames > 32 is fine
    // — the sliding-window stitcher handles it. The per-window
    // bound is what actually matters.
    let max_seq = 32usize;
    let window_size = args.window_size as usize;
    let window_overlap = args.window_overlap as usize;
    if window_size > max_seq {
        anyhow::bail!(
            "--window-size {} exceeds AnimateDiff motion_max_seq_length ({}). \
             Use --window-size ≤ {}.",
            args.window_size,
            max_seq,
            max_seq,
        );
    }
    if window_overlap >= window_size {
        anyhow::bail!(
            "--window-overlap {} must be < --window-size {}",
            args.window_overlap,
            args.window_size,
        );
    }
    if args.frames < 1 {
        anyhow::bail!("--frames must be ≥ 1 (got {})", args.frames);
    }
    let (width, height) = parse_size(&args.size)?;
    if width % 8 != 0 || height % 8 != 0 {
        anyhow::bail!(
            "--size {} not divisible by 8 (VAE constraint).",
            args.size
        );
    }

    if !args.to.is_empty() {
        tracing::info!(
            target: "plakat",
            "--animatediff ignores --to (single-prompt mode). \
             Using --from for every frame."
        );
    }
    if args.format.needs_ffmpeg() {
        let v = crate::imaging::video::ffmpeg_version()?;
        tracing::info!(target: "plakat", "ffmpeg detected: {v}");
    }

    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating output dir {}", args.out.display()))?;

    let dtype = if matches!(device, Device::Cpu) {
        candle_core::DType::F32
    } else {
        candle_core::DType::BF16
    };

    let seed = args.seed.unwrap_or_else(rand::random) & (u32::MAX as u64);

    // v0.27 phase 3 (SD 1.5) + phase 4 (SDXL): optional ControlNet
    // stack. Single conditioner for v0.27 — the same hint tiles to
    // every frame inside the pipeline. The model picker routes the
    // CN load to the right variant (SD 1.5 vs SDXL) via the standard
    // `ControlNetVariant::detect`.
    // v0.28 phase 0: merge legacy --control/--control-image/--control-from
    // /--control-strength with the repeatable --control-spec. clap's
    // conflicts_with_all already guarantees the two forms aren't mixed;
    // resolve_control_specs prefers the spec list when non-empty,
    // otherwise builds a single-CN list from the legacy flags.
    use crate::pipelines::controlnet::{
        ControlKind, load_control_stack, resolve_control_specs,
    };
    use std::str::FromStr;
    if args.control_image.is_some() && args.control_from.is_some() {
        anyhow::bail!(
            "--control-image and --control-from are mutually exclusive"
        );
    }
    let legacy_kind = match args.control.as_deref() {
        Some(s) => Some(ControlKind::from_str(s).with_context(|| {
            format!(
                "parsing --control {s:?} — expected depth | canny | \
                 openpose | lineart | softedge"
            )
        })?),
        None => None,
    };
    if legacy_kind.is_some()
        && args.control_image.is_none()
        && args.control_from.is_none()
    {
        anyhow::bail!(
            "--control requires --control-image PATH or --control-from PATH"
        );
    }
    let resolved_specs = resolve_control_specs(
        args.control_specs.clone(),
        legacy_kind,
        args.control_image.clone(),
        args.control_from.clone(),
        args.control_strength,
        0.0,
        1.0,
    );
    let controls = if resolved_specs.is_empty() {
        Vec::new()
    } else {
        load_control_stack(
            &resolved_specs,
            &args.model,
            width,
            height,
            &device,
            dtype,
            None,
            Some(args.frames as usize), // v0.30 phase 2: per-frame video CN
        )
        .await
        .context("loading ControlNet stack for --animatediff")?
    };
    if controls.len() > 1 {
        tracing::info!(
            target: "plakat",
            "AnimateDiff multi-CN: {} conditioners stacked",
            controls.len(),
        );
    }

    // v0.28 phase 1: AnimateLCM mode. When --lcm is set, switch the
    // motion adapter to wangfuyun/AnimateLCM, force the scheduler to
    // LCM, and apply the diffusers-recommended LCM defaults for
    // steps + guidance. User-supplied values for `--steps` /
    // `--guidance` take precedence over the LCM defaults (detected
    // by comparing against the args' built-in defaults of 20 / 7.5).
    if args.lcm && matches!(variant, SdVariant::Sdxl) {
        anyhow::bail!(
            "--lcm + --model {} not supported: AnimateLCM-SDXL isn't \
             publicly available. v0.28 ships AnimateLCM on SD 1.5 only.",
            args.model
        );
    }
    let (eff_scheduler, eff_steps, eff_guidance) = if args.lcm {
        let s = if args.steps == 20 { 4 } else { args.steps };
        let g = if (args.guidance - 7.5).abs() < f64::EPSILON {
            1.5
        } else {
            args.guidance
        };
        (SchedulerKind::Lcm, s, g)
    } else {
        (args.scheduler, args.steps, args.guidance)
    };

    // Variant-specific load + inference. Both branches return
    // `Vec<DynamicImage>` so the output dispatch below is shared.
    let frames = match variant {
        SdVariant::Sd15 => {
            let (pipeline, label) = if args.lcm {
                let p = AnimateDiffPipeline::load_animatelcm(
                    &device,
                    dtype,
                    &args.motion_loras,
                    args.motion_lora_scale,
                )
                .await
                .context("loading AnimateLCM stack")?;
                (p, "AnimateLCM")
            } else {
                let p = AnimateDiffPipeline::load_v3(
                    &device,
                    dtype,
                    &args.motion_loras,
                    args.motion_lora_scale,
                )
                .await
                .context("loading AnimateDiff V3 stack")?;
                (p, "AnimateDiff V3")
            };
            tracing::info!(
                target: "plakat",
                "{label} stack loaded: {} motion modules, max_seq_length={}",
                pipeline.modules.modules.len(),
                pipeline.max_frames,
            );
            crate::ui::progress::println(&format!(
                "  animatediff (SD 1.5{}): {} frames, seed {seed}, \
                 model {}, {}x{}, steps={}, guidance={}, scheduler={:?}",
                if args.lcm { ", LCM" } else { "" },
                args.frames, args.model, width, height, eff_steps, eff_guidance,
                eff_scheduler,
            ));
            // v0.27 phase 5: route through generate_long so frames >
            // window_size triggers the sliding-window stitcher.
            // When frames ≤ window_size, generate_long is a thin
            // pass-through to generate.
            pipeline.generate_long(
                &args.from,
                &args.negative,
                args.frames as usize,
                window_size,
                window_overlap,
                seed,
                width,
                height,
                eff_steps,
                eff_guidance,
                eff_scheduler,
                &controls,
                args.free_noise,
            )?
        }
        SdVariant::Sdxl => {
            use crate::pipelines::animatediff::AnimateDiffSdxlPipeline;
            let pipeline = AnimateDiffSdxlPipeline::load_sdxl_beta(
                &device,
                dtype,
                &args.model,
                &args.motion_loras,
                args.motion_lora_scale,
            )
            .await
            .context("loading AnimateDiff SDXL beta stack")?;
            tracing::info!(
                target: "plakat",
                "AnimateDiff SDXL beta stack loaded: {} motion modules, max_seq_length={}",
                pipeline.modules.modules.len(),
                pipeline.max_frames,
            );
            crate::ui::progress::println(&format!(
                "  animatediff (SDXL beta): {} frames, seed {seed}, \
                 model {}, {}x{}, steps={}, guidance={}",
                args.frames, args.model, width, height, args.steps, args.guidance,
            ));
            // v0.27 phase 6: same long-form routing as SD 1.5 — when
            // frames > window_size, the SDXL pipeline's generate_long
            // engages the sliding-window stitcher.
            pipeline.generate_long(
                &args.from,
                &args.negative,
                args.frames as usize,
                window_size,
                window_overlap,
                seed,
                width,
                height,
                args.steps,
                args.guidance,
                args.scheduler,
                &controls,
                args.free_noise,
            )?
        }
        // SdVariant::Sd21 already rejected above.
        _ => unreachable!("variant gate filtered to SD 1.5 / SDXL"),
    };

    // -------- write per-frame PNGs (always) + optional GIF + MP4 + WebM.
    let scheduler_name = format!("{:?}", args.scheduler).to_lowercase();
    let mut frame_paths: Vec<PathBuf> = Vec::with_capacity(frames.len());
    for (i, img) in frames.iter().enumerate() {
        let frame_path = args.out.join(format!("frame-{i:04}.png"));
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        if args.no_metadata {
            crate::imaging::io::save_rgb_u8(rgb.as_raw(), w, h, &frame_path)?;
        } else {
            let mut meta = crate::imaging::metadata::GenerationMetadata::new(
                args.from.clone(),
                args.model.clone(),
                seed,
                args.steps,
                args.guidance,
                scheduler_name.clone(),
                width,
                height,
            );
            meta.negative = args.negative.clone();
            meta.mode = Some("animatediff".to_string());
            meta.extras.push((
                "AnimateDiff frame".to_string(),
                format!("{i}/{}", frames.len()),
            ));
            crate::imaging::io::save_rgb_u8_with_metadata(
                rgb.as_raw(),
                w,
                h,
                &frame_path,
                &meta,
            )?;
        }
        frame_paths.push(frame_path);
    }
    crate::ui::progress::println(&format!(
        "  wrote {} PNG frame(s) → {}",
        frame_paths.len(),
        args.out.display()
    ));

    // GIF: triggered by --format gif|all OR the legacy --gif flag.
    let want_gif = args.format.needs_gif() || args.gif;
    if want_gif {
        let gif_path = args.out.join("animation.gif");
        let spin = crate::ui::progress::spinner(&format!(
            "Bundling {} frame(s) → {}",
            frame_paths.len(),
            gif_path.display()
        ));
        write_gif(&frame_paths, &gif_path, args.gif_delay_ms)?;
        spin.finish_with_message(format!("✓ {}", gif_path.display()));
    }
    // MP4 / WebM via ffmpeg pattern `frame-%04d.png` in args.out.
    if args.format.needs_mp4() || args.format.needs_webm() {
        // ffmpeg input pattern: literal path with the %04d format
        // specifier. ffmpeg reads `frame-0000.png .. frame-NNNN.png`
        // contiguous from disk.
        let pattern = args
            .out
            .join("frame-%04d.png")
            .to_string_lossy()
            .to_string();
        // 8 fps is the AnimateDiff default in the upstream community.
        let fps = 8u32;
        if args.format.needs_mp4() {
            let mp4_path = args.out.join("animation.mp4");
            let spin = crate::ui::progress::spinner(&format!(
                "Encoding MP4 ({fps} fps) → {}",
                mp4_path.display()
            ));
            crate::imaging::video::frames_to_mp4(&pattern, &mp4_path, fps)?;
            spin.finish_with_message(format!("✓ {}", mp4_path.display()));
        }
        if args.format.needs_webm() {
            let webm_path = args.out.join("animation.webm");
            let spin = crate::ui::progress::spinner(&format!(
                "Encoding WebM ({fps} fps) → {}",
                webm_path.display()
            ));
            crate::imaging::video::frames_to_webm(&pattern, &webm_path, fps)?;
            spin.finish_with_message(format!("✓ {}", webm_path.display()));
        }
    }
    Ok(())
}

pub(crate) fn write_gif(
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

    // v0.18 phase 5 — SDXL animate exercises lerp on tensors of two
    // distinct shapes per frame: (1, 77, 2048) hidden and (1, 1280)
    // pooled add_text_embeds. Verify the pooled shape doesn't trip
    // the lerp (different rank, same broadcast semantics).

    #[test]
    fn lerp_on_pooled_shape_1_1280() {
        let a = Tensor::zeros((1, 1280), DType::F32, &Device::Cpu).unwrap();
        let b = (Tensor::ones((1, 1280), DType::F32, &Device::Cpu).unwrap() * 4.0).unwrap();
        let out = lerp_tensors(&a, &b, 0.25).unwrap();
        // 0 * 0.75 + 4 * 0.25 = 1.0 everywhere.
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(flat.len(), 1280);
        for v in &flat {
            assert!((v - 1.0).abs() < 1e-5, "expected ~1.0, got {v}");
        }
    }

    // v0.20 #9 — Flux animate routing + lerp on Flux's tensor shapes.

    #[test]
    fn map_to_flux_variant_covers_flux_family() {
        use crate::pipelines::flux::Variant as FV;
        use crate::pipelines::t2i::Variant;
        assert!(matches!(map_to_flux_variant(Variant::FluxDev), FV::Dev));
        assert!(matches!(
            map_to_flux_variant(Variant::FluxFillDev),
            FV::FillDev
        ));
        assert!(matches!(
            map_to_flux_variant(Variant::FluxCannyDev),
            FV::CannyDev
        ));
        assert!(matches!(
            map_to_flux_variant(Variant::FluxDepthDev),
            FV::DepthDev
        ));
        assert!(matches!(
            map_to_flux_variant(Variant::FluxKontextDev),
            FV::KontextDev
        ));
        assert!(matches!(
            map_to_flux_variant(Variant::FluxSchnell),
            FV::Schnell
        ));
    }

    #[test]
    fn lerp_on_flux_t5_shape_1_512_4096() {
        // Flux Dev's T5 hidden state: (1, 512, 4096). Make sure
        // lerp handles the volume + that midpoint averages
        // correctly for the larger tensor.
        let a = Tensor::zeros((1, 512, 4096), DType::F32, &Device::Cpu).unwrap();
        let b = (Tensor::ones((1, 512, 4096), DType::F32, &Device::Cpu).unwrap() * 6.0).unwrap();
        let out = lerp_tensors(&a, &b, 0.5).unwrap();
        assert_eq!(out.dims(), &[1, 512, 4096]);
        // Sample one position; with floats the broadcast is exact.
        let mid = out.i(0).unwrap().i(256).unwrap().i(2048).unwrap();
        let v: f32 = mid.to_vec0().unwrap();
        assert!((v - 3.0).abs() < 1e-5);
    }

    #[test]
    fn lerp_on_flux_clip_pooled_shape_1_768() {
        // Flux CLIP-L pooled output is (1, 768) — single row of
        // pooled features. Same broadcast contract as the SDXL
        // pooled tensor; verify the shape passes through.
        let a = Tensor::zeros((1, 768), DType::F32, &Device::Cpu).unwrap();
        let b = (Tensor::ones((1, 768), DType::F32, &Device::Cpu).unwrap() * 2.0).unwrap();
        let out = lerp_tensors(&a, &b, 0.5).unwrap();
        assert_eq!(out.dims(), &[1, 768]);
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for x in &v {
            assert!((x - 1.0).abs() < 1e-5, "expected 1.0, got {x}");
        }
    }

    #[test]
    fn lerp_on_flux_schnell_t5_shape_1_256_4096() {
        // Schnell uses a 256-token T5 budget (vs Dev's 512).
        // Same lerp contract, smaller tensor.
        let a = Tensor::zeros((1, 256, 4096), DType::F32, &Device::Cpu).unwrap();
        let b = (Tensor::ones((1, 256, 4096), DType::F32, &Device::Cpu).unwrap() * 4.0).unwrap();
        let out = lerp_tensors(&a, &b, 0.25).unwrap();
        assert_eq!(out.dims(), &[1, 256, 4096]);
        let mid = out.i(0).unwrap().i(128).unwrap().i(2000).unwrap();
        let v: f32 = mid.to_vec0().unwrap();
        // 0 * 0.75 + 4 * 0.25 = 1.0
        assert!((v - 1.0).abs() < 1e-5);
    }

    #[test]
    fn lerp_on_hidden_shape_1_77_2048() {
        // Same broadcast contract for the cross-attention shape.
        let a = Tensor::zeros((1, 77, 2048), DType::F32, &Device::Cpu).unwrap();
        let b = (Tensor::ones((1, 77, 2048), DType::F32, &Device::Cpu).unwrap() * 8.0).unwrap();
        let out = lerp_tensors(&a, &b, 0.5).unwrap();
        let mid: Vec<f32> = out
            .i(0)
            .unwrap()
            .i(38)
            .unwrap()
            .i(1024)
            .unwrap()
            .to_vec0::<f32>()
            .map(|v| vec![v])
            .unwrap_or_default();
        assert!((mid[0] - 4.0).abs() < 1e-5);
        assert_eq!(out.dims(), &[1, 77, 2048]);
    }
}
