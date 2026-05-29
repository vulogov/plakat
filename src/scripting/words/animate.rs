//! v0.28 phase 2: `plakat.animate ( prompt out_dir -- )`.
//!
//! Renders a motion-coherent N-frame sequence using AnimateDiff and
//! writes the frames to `<out_dir>/frame-NNNN.png` — mirroring the
//! `plakat animate --animatediff` CLI output layout exactly. Reads
//! frames + window + LCM flag + size + steps + guidance + scheduler
//! + ControlNet stack from `ctx.config`.
//!
//! ## Stack effect
//!
//! `( prompt out_dir -- )`. Both args are strings:
//! - `prompt` — the AnimateDiff prompt (single prompt, no lerp; the
//!   v0.20-era two-prompt morph mode isn't exposed in Bund).
//! - `out_dir` — directory that receives `frame-NNNN.png` plus
//!   per-frame `frame-NNNN.json` metadata sidecars when metadata is
//!   enabled. Relative paths resolve against `ctx.out_dir`.
//!
//! ## Configurable knobs (via `plakat.config.set`)
//!
//! - `animate_frames` (default 16): total output frames.
//! - `animate_window_size` (default 16): per-window frame count for
//!   long-form sliding-window stitching when `frames > window_size`.
//! - `animate_window_overlap` (default 4): cross-fade region.
//! - `animate_lcm` (default false): use AnimateLCM motion adapter +
//!   LCM scheduler + 4-step / guidance=1.5 defaults.
//!
//! Standard config knobs honoured: `steps`, `guidance`, `seed`,
//! `width`, `height`, `negative`, `scheduler`. ControlNet stack
//! from `ctx.controlnets` flows through too — same multi-CN sum as
//! the CLI surface.
//!
//! ## Scope
//!
//! SD 1.5 only (matching the V3 / AnimateLCM motion adapters).
//! SDXL motion adapter via `plakat.animate` deferred to v0.29
//! (see RFC §10).
//!
//! ```bund
//! "sd15" plakat.load
//! "true" "animate_lcm" plakat.config.set
//! "32" "animate_frames" plakat.config.set
//! "a watercolor cottage at dawn" "./out" plakat.animate
//! // → ./out/frame-0000.png ... ./out/frame-0031.png + sidecars
//! ```

use rust_multistackvm::multistackvm::VM;
use std::path::PathBuf;

use crate::scripting::ctx::{with_ctx, with_ctx_mut};
use crate::scripting::helpers::{
    BundResult, pull, require_depth, to_bund_err, value_to_string,
};

const TAG: &str = "plakat.animate";

pub fn plakat_animate(vm: &mut VM) -> BundResult<'_> {
    do_plakat_animate(vm).map_err(to_bund_err)
}

fn do_plakat_animate(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 2, TAG)?;
    // Top pops first: out_dir, then prompt.
    let out_dir_v = pull(vm, TAG)?;
    let prompt_v = pull(vm, TAG)?;
    let prompt = value_to_string(prompt_v, "prompt", TAG)?;
    let out_dir_str = value_to_string(out_dir_v, "out_dir", TAG)?;
    if prompt.is_empty() {
        anyhow::bail!("{TAG}: prompt can't be empty");
    }
    if out_dir_str.is_empty() {
        anyhow::bail!("{TAG}: out_dir can't be empty");
    }

    // Snapshot everything from ctx that the inference needs. Pull
    // ControlNet specs out as owned ControlSpec (not borrowed); the
    // CN load needs them owned anyway.
    let (
        alias,
        cn_specs,
        out_dir_root,
        width,
        height,
        steps_cfg,
        guidance_cfg,
        seed_cfg,
        negative,
        scheduler_cfg,
        animate_frames,
        animate_window_size,
        animate_window_overlap,
        animate_lcm,
        animate_format,
        device,
    ) = with_ctx(|ctx| {
        (
            ctx.loaded_model().map(|s| s.to_string()),
            ctx.controlnets.clone(),
            ctx.out_dir.clone(),
            ctx.config.width,
            ctx.config.height,
            ctx.config.steps,
            ctx.config.guidance,
            ctx.config.seed,
            ctx.config.negative.clone(),
            ctx.config.scheduler,
            ctx.config.animate_frames,
            ctx.config.animate_window_size,
            ctx.config.animate_window_overlap,
            ctx.config.animate_lcm,
            ctx.config.animate_format,
            ctx.device.clone(),
        )
    })?;

    let alias = alias.ok_or_else(|| {
        anyhow::anyhow!(
            "{TAG}: no model loaded. Call \"sd15\" plakat.load before {TAG}."
        )
    })?;
    // Hard gate: SD 1.5 only. SDXL motion adapters work via CLI but
    // we don't have a scripting cache slot for them yet.
    let alias_l = alias.to_lowercase();
    if alias_l.contains("xl") || alias_l.contains("flux") || alias_l.contains("sd3") {
        anyhow::bail!(
            "{TAG}: AnimateDiff in scripting is SD 1.5 only (got {alias:?}). \
             Call \"sd15\" plakat.load before {TAG}. SDXL animate in scripting \
             is deferred to v0.29."
        );
    }

    // Resolve out_dir against ctx.out_dir when relative.
    let out_dir: PathBuf = {
        let p = PathBuf::from(&out_dir_str);
        if p.is_absolute() { p } else { out_dir_root.join(p) }
    };
    std::fs::create_dir_all(&out_dir).map_err(|e| {
        anyhow::anyhow!(
            "{TAG}: creating out_dir {}: {e}",
            out_dir.display()
        )
    })?;

    // Validate frames/window numbers.
    let frames = animate_frames as usize;
    let window_size = animate_window_size as usize;
    let window_overlap = animate_window_overlap as usize;
    anyhow::ensure!(frames >= 1, "{TAG}: animate_frames must be ≥ 1");
    anyhow::ensure!(
        window_size <= 32,
        "{TAG}: animate_window_size {window_size} exceeds motion_max_seq_length (32)"
    );
    anyhow::ensure!(
        window_overlap < window_size,
        "{TAG}: animate_window_overlap {window_overlap} must be < animate_window_size {window_size}"
    );
    let (width, height) = (width.max(8), height.max(8));
    anyhow::ensure!(
        width.is_multiple_of(8) && height.is_multiple_of(8),
        "{TAG}: width/height must be divisible by 8 (got {width}x{height})"
    );

    // Apply LCM defaults when animate_lcm is set and the user
    // hasn't overridden steps / guidance / scheduler from their
    // built-in defaults. Mirrors the CLI's --lcm logic.
    let (eff_steps, eff_guidance, eff_scheduler) = if animate_lcm {
        use crate::pipelines::scheduler::SchedulerKind;
        let s = if steps_cfg == 28 || steps_cfg == 20 {
            4
        } else {
            steps_cfg
        };
        let g = if (guidance_cfg - 7.5).abs() < f64::EPSILON {
            1.5
        } else {
            guidance_cfg
        };
        (s, g, SchedulerKind::Lcm)
    } else {
        (steps_cfg, guidance_cfg, scheduler_cfg)
    };

    let dtype = if matches!(device, candle_core::Device::Cpu) {
        candle_core::DType::F32
    } else {
        candle_core::DType::BF16
    };

    // Seed: explicit when given, otherwise random per call.
    let seed = seed_cfg.unwrap_or_else(rand::random) & (u32::MAX as u64);

    // Block on the async load (Bund VM is sync; we bridge via the
    // ambient tokio runtime — `plakat run` installs one).
    let rt = tokio::runtime::Handle::try_current().map_err(|e| {
        anyhow::anyhow!(
            "{TAG}: AnimateDiff load requires a tokio runtime in scope ({e})"
        )
    })?;
    let pipeline = tokio::task::block_in_place(|| {
        rt.block_on(async {
            if animate_lcm {
                crate::pipelines::animatediff::AnimateDiffPipeline::load_animatelcm(
                    &device,
                    dtype,
                    &[],
                    1.0,
                )
                .await
            } else {
                crate::pipelines::animatediff::AnimateDiffPipeline::load_v3(
                    &device,
                    dtype,
                    &[],
                    1.0,
                )
                .await
            }
        })
    })?;

    // Load ControlNet stack (if any) at the resolved size.
    let controls = if cn_specs.is_empty() {
        Vec::new()
    } else {
        let rt = rt.clone();
        tokio::task::block_in_place(|| {
            rt.block_on(crate::pipelines::controlnet::load_control_stack(
                &cn_specs, &alias, width, height, &device, dtype, None,
            ))
        })?
    };

    tracing::info!(
        target: "plakat",
        "{TAG}: loaded {} motion modules; {frames} frames at {width}x{height}, \
         steps={eff_steps}, guidance={eff_guidance:.2}, scheduler={eff_scheduler:?}, \
         CN={}",
        pipeline.modules.modules.len(),
        controls.len(),
    );

    // Generate. Uses generate_long so frames > window_size triggers
    // sliding-window stitching automatically.
    let images = pipeline.generate_long(
        &prompt,
        &negative,
        frames,
        window_size,
        window_overlap,
        seed,
        width,
        height,
        eff_steps,
        eff_guidance,
        eff_scheduler,
        &controls,
    )?;

    // v0.29 phase 0: ffmpeg availability check fires once when the
    // format wants it. Fails fast rather than after the (expensive)
    // inference loop.
    if animate_format.needs_ffmpeg() {
        let v = crate::imaging::video::ffmpeg_version()?;
        tracing::info!(target: "plakat", "{TAG}: ffmpeg detected ({v})");
    }

    // Build per-frame metadata + write each PNG. Collect paths so the
    // format dispatch below can feed GIF / MP4 / WebM encoders.
    let scheduler_name = format!("{eff_scheduler:?}").to_lowercase();
    let mode_label = if animate_lcm {
        "animatediff-lcm"
    } else {
        "animatediff"
    };
    let mut frame_paths: Vec<std::path::PathBuf> = Vec::with_capacity(images.len());
    for (i, img) in images.iter().enumerate() {
        let frame_path = out_dir.join(format!("frame-{i:04}.png"));
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let mut meta = crate::imaging::metadata::GenerationMetadata::new(
            prompt.clone(),
            alias.clone(),
            seed,
            eff_steps,
            eff_guidance,
            scheduler_name.clone(),
            width,
            height,
        );
        meta.negative = negative.clone();
        meta.mode = Some(mode_label.to_string());
        meta.extras.push((
            "AnimateDiff frame".to_string(),
            format!("{i}/{}", images.len()),
        ));
        crate::imaging::io::save_rgb_u8_with_metadata(
            rgb.as_raw(),
            w,
            h,
            &frame_path,
            &meta,
        )?;
        frame_paths.push(frame_path);
    }

    // v0.29 phase 0: format dispatch — GIF / MP4 / WebM / All. The
    // `Frames` default skips this block entirely (only PNGs land).
    if animate_format.needs_gif() {
        let gif_path = out_dir.join("animation.gif");
        // 100 ms = 10 fps; matches the CLI animate default.
        crate::cli::animate::write_gif(&frame_paths, &gif_path, 100)?;
        tracing::info!(
            target: "plakat",
            "{TAG}: wrote GIF → {}",
            gif_path.display()
        );
    }
    if animate_format.needs_mp4() || animate_format.needs_webm() {
        let pattern = out_dir
            .join("frame-%04d.png")
            .to_string_lossy()
            .to_string();
        let fps = 8u32; // matches CLI animate default
        if animate_format.needs_mp4() {
            let mp4_path = out_dir.join("animation.mp4");
            crate::imaging::video::frames_to_mp4(&pattern, &mp4_path, fps)?;
            tracing::info!(
                target: "plakat",
                "{TAG}: wrote MP4 → {}",
                mp4_path.display()
            );
        }
        if animate_format.needs_webm() {
            let webm_path = out_dir.join("animation.webm");
            crate::imaging::video::frames_to_webm(&pattern, &webm_path, fps)?;
            tracing::info!(
                target: "plakat",
                "{TAG}: wrote WebM → {}",
                webm_path.display()
            );
        }
    }

    // Optional: register the first frame as a handle so the script
    // can compose with plakat.upscale / plakat.save. Skip if frames
    // is zero (shouldn't happen given the ensure! above).
    if let Some(first) = images.into_iter().next() {
        let _handle = with_ctx_mut(|ctx| ctx.push_image(first))?;
    }
    tracing::info!(
        target: "plakat",
        "{TAG}: wrote {frames} frame(s) → {} (format={animate_format})",
        out_dir.display()
    );
    Ok(vm)
}
