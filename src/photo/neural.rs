//! The model-backed stages, each behind `--neural <name>` and each disclosed in the report:
//!
//! - **fill** — the ground a camera move opened, inpainted by a diffusion model inside the holes' mask (the
//!   rest of the frame kept to the pixel); the weight-free `propagate` is the default and this is the option
//!   for a move that opens more than the background can be grown into;
//! - **refine** — a whole-frame img2img at the plan's `refine` strength with the medium's `prompt`, the faces
//!   composited back from before (a small face does not survive a denoise); the material of the carrier the
//!   physics cannot give — the albumen's gloss, the collodion's dirt;
//! - **relight** — IC-Light's restaging of the subject's light by a preset (the plan's `relight`), the subject
//!   matted out and relit on a directional backdrop; the result is the model's whole frame, so the background
//!   is the model's too, and the report says so.
//!
//! Each needs weights and a device; a failure is reported and the chain goes on without it.

use super::{Field, Frame};
use anyhow::{Context, Result};
use std::path::Path;

fn produced_png(dir: &Path) -> Result<std::path::PathBuf> {
    std::fs::read_dir(dir).context("reading the render's output dir")?.filter_map(|e| e.ok().map(|e| e.path())).find(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("png"))).context("the model produced no image")
}

/// Inpaint the holes (1 = invented) with a model; the frame outside them kept to the pixel.
pub async fn fill(frame: &Frame, holes: &Field, prompt: &str, model: &str, seed: u64) -> Result<Frame> {
    let dir = tempfile::tempdir().context("a temp dir for the fill")?;
    let input = dir.path().join("frame.png");
    frame.to_srgb().save(&input)?;
    let mask_path = dir.path().join("mask.png");
    let mut m = image::GrayImage::new(holes.w, holes.h);
    for (i, p) in m.pixels_mut().enumerate() {
        p.0 = [if holes.v[i] > 0.5 { 255 } else { 0 }];
    }
    m.save(&mask_path)?;
    let (pipeline, device) = crate::naturalize::refine::load_repaint_pipeline(model, None, None).await?;
    let out_dir = dir.path().join("out");
    std::fs::create_dir_all(&out_dir)?;
    let req = crate::pipelines::img2img::Request {
        prompt: if prompt.trim().is_empty() { "a photograph".to_string() } else { prompt.to_string() },
        negative: "text, watermark, frame, border, cartoon, painting, blurry".to_string(),
        model: model.to_string(),
        device,
        loras: Vec::new(),
        lora_scale: 1.0,
        input: input.clone(),
        mask: Some(mask_path),
        mask_feather: 8,
        mask_invert: false,
        width: (frame.w / 8 * 8).max(64),
        height: (frame.h / 8 * 8).max(64),
        count: 1,
        steps: 24,
        guidance: 6.0,
        scheduler: crate::pipelines::scheduler::SchedulerKind::default(),
        strength: 0.95,
        seed: Some(seed),
        out_dir: out_dir.clone(),
        controls: Vec::new(),
    };
    let _quiet = crate::ui::progress::quiet_scope();
    crate::pipelines::img2img::run_with_pipeline(&pipeline, &req).await.context("the neural fill (img2img)")?;
    let got = image::open(produced_png(&out_dir)?)?.to_rgb8();
    let filled = Frame::from_srgb(&got).resized(frame.w, frame.h);
    let soft = super::gaussian_field(holes, 2.0);
    let mut out = frame.clone();
    for i in 0..out.px.len() {
        let t = soft.v[i].clamp(0.0, 1.0);
        if t > 0.0 {
            for c in 0..3 {
                out.px[i][c] = out.px[i][c] * (1.0 - t) + filled.px[i][c] * t;
            }
        }
    }
    Ok(out)
}

/// A whole-frame img2img at `strength` with the medium's prompt. `controls` are ControlNet specs
/// (`depth:strength=0.5`) that hold the picture's structure while the rendering changes. The faces are
/// composited back from before the pass only at a light strength (≤ 0.4, a photograph refined); a heavy pass is
/// a conversion — a drawing's painted faces are what it is for. The pass runs at the model's working size
/// (`px` on the long side; SDXL draws at 1024 and a 2048 frame does not fit a GPU's buffer) and the result is
/// brought back to the frame's size.
pub async fn refine(frame: &Frame, strength: f32, prompt: &str, model: &str, seed: u64, controls: &[String], px: u32) -> Result<Frame> {
    let dir = tempfile::tempdir().context("a temp dir for the refine")?;
    let input = dir.path().join("frame.png");
    let long = frame.w.max(frame.h) as f32;
    let s = (px.max(256) as f32 / long).min(1.0);
    let (ww, hh) = (((frame.w as f32 * s) as u32 / 8 * 8).max(64), ((frame.h as f32 * s) as u32 / 8 * 8).max(64));
    let work = frame.resized(ww, hh);
    work.to_srgb().save(&input)?;
    let (pipeline, device) = crate::naturalize::refine::load_repaint_pipeline(model, None, None).await?;
    let out_dir = dir.path().join("out");
    std::fs::create_dir_all(&out_dir)?;
    let req = crate::pipelines::img2img::Request {
        prompt: if prompt.trim().is_empty() { "a photograph".to_string() } else { prompt.to_string() },
        negative: "text, watermark, cartoon, painting, illustration, 3d render, deformed".to_string(),
        model: model.to_string(),
        device,
        loras: Vec::new(),
        lora_scale: 1.0,
        input: input.clone(),
        mask: None,
        mask_feather: 0,
        mask_invert: false,
        width: ww,
        height: hh,
        count: 1,
        steps: 24,
        guidance: 5.5,
        scheduler: crate::pipelines::scheduler::SchedulerKind::default(),
        strength: strength.clamp(0.05, 0.95),
        seed: Some(seed),
        out_dir: out_dir.clone(),
        controls: controls.iter().map(|c| c.parse::<crate::pipelines::controlnet::ControlSpec>().with_context(|| format!("--neural-control {c:?}"))).collect::<Result<Vec<_>>>()?,
    };
    {
        let _quiet = crate::ui::progress::quiet_scope();
        crate::pipelines::img2img::run_with_pipeline(&pipeline, &req).await.context("the neural refine (img2img)")?;
    }
    let produced = produced_png(&out_dir)?;
    let refined = dir.path().join("refined.png");
    std::fs::copy(&produced, &refined)?;
    if strength <= 0.4 {
        let _ = crate::naturalize::refine::protect_repaint_faces(&input, &refined, None, 0.85).await;
    }
    let got = image::open(&refined)?.to_rgb8();
    Ok(Frame::from_srgb(&got).resized(frame.w, frame.h))
}

/// Restage the subject's light with IC-Light by a preset; the result is the model's frame at up to 768 px on
/// the long side, resampled to the frame's size.
pub async fn relight(frame: &Frame, preset: &str, seed: u64) -> Result<Frame> {
    let p = crate::pipelines::ic_light::light_preset(preset).with_context(|| format!("no relight preset {preset:?} — one of {}", crate::pipelines::ic_light::light_presets().iter().map(|p| p.name).collect::<Vec<_>>().join(", ")))?;
    let device = crate::device::select("auto")?;
    let dir = tempfile::tempdir().context("a temp dir for the relight")?;
    let long = frame.w.max(frame.h) as f32;
    let s = (768.0 / long).min(1.0);
    let (w, h) = (((frame.w as f32 * s) as u32 / 8 * 8).max(64), ((frame.h as f32 * s) as u32 / 8 * 8).max(64));
    let input = dir.path().join("frame.png");
    frame.resized(w, h).to_srgb().save(&input)?;
    let subject = dir.path().join("subject.png");
    crate::pipelines::matting::cutout(&input, &subject, false, &device).await.context("matting the subject for the relight")?;
    let pipe = crate::pipelines::ic_light::Pipeline::load(device.clone()).await.context("loading IC-Light")?;
    let (rgb, ow, oh) = pipe.relight(&subject, p.prompt, p.negative, w, h, 20, 2.0, seed, p.backdrop).context("the relight")?;
    let img = image::RgbImage::from_raw(ow, oh, rgb).context("the relight's frame")?;
    Ok(Frame::from_srgb(&img).resized(frame.w, frame.h))
}
