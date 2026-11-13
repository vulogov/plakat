//! The model-backed stages, each behind `--neural <name>` and each disclosed in the report:
//!
//! - **fill** — the ground a camera move opened, inpainted by a diffusion model inside the holes' mask (the
//!   rest of the frame kept to the pixel); the weight-free `propagate` is the default and this is the option
//!   for a move that opens more than the background can be grown into;
//! - **convert** — a drawing or a painting re-rendered as a NEUTRAL photograph of the same scene (the plan's
//!   `convert`, `convert_ground`, `convert_control`, `scene`) before the medium's pass: the heavy denoise the
//!   conversion needs with the structure held by the depth and the soft edges, the figure and the ground at
//!   their own strengths; the medium's refine then comes light — one pass cannot both understand a watercolour
//!   and lay a tintype's milk over it;
//! - **refine** — a whole-frame img2img at the plan's `refine` strength with the medium's `prompt`, the faces
//!   composited back from before (a small face does not survive a denoise); the material of the carrier the
//!   physics cannot give — the albumen's gloss, the collodion's dirt;
//! - **relight** — IC-Light's restaging of the subject's light by a preset (the plan's `relight`), the subject
//!   matted out and relit on a directional backdrop; the result is the model's whole frame, so the background
//!   is the model's too, and the report says so.
//!
//! Each needs weights and a device; a failure is reported and the chain goes on without it. The model is the
//! plan's `neural_model` (`--neural-model`), run through its family's pipeline ([`Family`]): SD3.5 medium by
//! default — on the verification set it converts a watercolour into a photograph that keeps the house, the path,
//! the boat and the cloak with no ControlNet, and a light pass over its own output stays clean, where SDXL's
//! stipples every smooth field; Kandinsky 5 does as well and takes seven times as long.

use super::{Field, Frame};
use anyhow::{Context, Result};
use std::path::Path;

fn produced_png(dir: &Path) -> Result<std::path::PathBuf> {
    std::fs::read_dir(dir).context("reading the render's output dir")?.filter_map(|e| e.ok().map(|e| e.path())).find(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("png"))).context("the model produced no image")
}

/// The model's family, by its alias: the passes run through the family's own pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// SD 1.5 / SDXL: the portrait pipeline, with ControlNets.
    Sd,
    /// SD3 / SD3.5: the rectified-flow img2img (the init lerped with the noise at the strength) — it holds the
    /// composition without a ControlNet and a light pass over its own output stays clean, which SDXL's does not.
    Sd3,
    /// Kandinsky 5: the same flow img2img (SDEdit), the best photograph on the verification set and the slowest
    /// (14 minutes a pass on Metal); no ControlNet.
    Kandinsky,
}

pub fn family_of(model: &str) -> Family {
    if crate::pipelines::kandinsky::is_kandinsky(model) {
        Family::Kandinsky
    } else if crate::pipelines::t2i::Variant::detect(model).is_sd3() {
        Family::Sd3
    } else {
        Family::Sd
    }
}

impl Family {
    pub fn name(self) -> &'static str {
        match self {
            Family::Sd => "SD",
            Family::Sd3 => "SD3",
            Family::Kandinsky => "Kandinsky 5",
        }
    }
    /// Whether the family's img2img takes ControlNets.
    pub fn takes_controls(self) -> bool {
        matches!(self, Family::Sd)
    }
}

/// One img2img or inpaint pass at the model's working size (`px` on the long side; SDXL draws at 1024 and a 2048
/// frame does not fit a GPU's buffer), the result brought back to the frame's size. With a `mask` only the
/// masked pixels (1) are repainted. With `protect_faces` the faces are composited back from before the pass.
struct Pass<'a> {
    strength: f32,
    prompt: &'a str,
    negative: &'a str,
    model: &'a str,
    seed: u64,
    controls: &'a [String],
    px: u32,
    steps: f32,
    guidance: f32,
    mask: Option<&'a Field>,
    protect_faces: bool,
}

async fn img2img(frame: &Frame, p: &Pass<'_>) -> Result<Frame> {
    let family = family_of(p.model);
    let dir = tempfile::tempdir().context("a temp dir for the pass")?;
    let input = dir.path().join("frame.png");
    let long = frame.w.max(frame.h) as f32;
    let s = (p.px.max(256) as f32 / long).min(1.0);
    // SD3's patch is 2 on a VAE of 8: a side a multiple of 16; the SD family needs 8.
    let unit = if family == Family::Sd { 8 } else { 16 };
    let (ww, hh) = (((frame.w as f32 * s) as u32 / unit * unit).max(64), ((frame.h as f32 * s) as u32 / unit * unit).max(64));
    let (ww, hh) = if family == Family::Kandinsky { crate::pipelines::kandinsky::snap_bucket(ww, hh) } else { (ww, hh) };
    frame.resized(ww, hh).to_srgb().save(&input)?;
    let mask_path = p.mask.map(|m| -> Result<std::path::PathBuf> {
        let path = dir.path().join("mask.png");
        let small = m.resized(ww, hh);
        let mut g = image::GrayImage::new(ww, hh);
        for (i, px) in g.pixels_mut().enumerate() {
            px.0 = [if small.v[i] > 0.5 { 255 } else { 0 }];
        }
        g.save(&path)?;
        Ok(path)
    }).transpose()?;
    let out_dir = dir.path().join("out");
    std::fs::create_dir_all(&out_dir)?;
    let steps = schedule_steps(p.steps, p.strength);
    let strength = p.strength.clamp(0.05, 0.95);
    let _quiet = crate::ui::progress::quiet_scope();
    match family {
        Family::Sd => {
            let (pipeline, device) = crate::naturalize::refine::load_repaint_pipeline(p.model, None, None).await?;
            let req = crate::pipelines::img2img::Request {
                prompt: p.prompt.to_string(),
                negative: p.negative.to_string(),
                model: p.model.to_string(),
                device,
                loras: Vec::new(),
                lora_scale: 1.0,
                input: input.clone(),
                mask: mask_path,
                mask_feather: 8,
                mask_invert: false,
                width: ww,
                height: hh,
                count: 1,
                steps,
                guidance: p.guidance.clamp(1.0, 15.0) as f64,
                scheduler: crate::pipelines::scheduler::SchedulerKind::default(),
                strength,
                seed: Some(p.seed),
                out_dir: out_dir.clone(),
                controls: p.controls.iter().map(|c| c.parse::<crate::pipelines::controlnet::ControlSpec>().with_context(|| format!("the control spec {c:?}"))).collect::<Result<Vec<_>>>()?,
            };
            crate::pipelines::img2img::run_with_pipeline(&pipeline, &req).await.context("the img2img pass")?;
        }
        Family::Sd3 => {
            use crate::pipelines::{sd3, t2i};
            let variant = match t2i::Variant::detect(p.model) {
                t2i::Variant::Sd3Medium => sd3::Variant::Sd3Medium,
                t2i::Variant::Sd35Medium => sd3::Variant::Sd35Medium,
                t2i::Variant::Sd35Large => sd3::Variant::Sd35Large,
                t2i::Variant::Sd35LargeTurbo => sd3::Variant::Sd35LargeTurbo,
                other => anyhow::bail!("{other:?} is not an SD3 variant"),
            };
            let repo = if p.model.contains('/') { p.model.to_string() } else { crate::hf::resolve_alias(p.model).to_string() };
            let device = crate::device::select("auto")?;
            sd3::run(sd3::Request {
                prompt: p.prompt.to_string(),
                negative: p.negative.to_string(),
                variant,
                repo,
                width: ww,
                height: hh,
                count: 1,
                steps: Some(steps),
                guidance: Some(p.guidance.clamp(0.0, 15.0) as f64),
                seed: Some(p.seed),
                out_dir: out_dir.clone(),
                device,
                init_image: Some(input.clone()),
                mask: mask_path,
                mask_feather: 8,
                mask_invert: false,
                strength: Some(strength),
                loras: Vec::new(),
                lora_scale: 1.0,
                tiled: None,
                regions: Vec::new(),
                controlnets: Vec::new(),
                embeddings: Vec::new(),
                output_format: crate::imaging::io::OutputFormat::Png,
            })
            .await
            .context("the SD3 img2img pass")?;
        }
        Family::Kandinsky => {
            use crate::pipelines::kandinsky as k5;
            let device = crate::device::select("auto")?;
            k5::run(k5::RunRequest {
                model: p.model.to_string(),
                device,
                prompt: p.prompt.to_string(),
                negative: p.negative.to_string(),
                width: ww,
                height: hh,
                steps,
                guidance: p.guidance.clamp(0.0, 15.0) as f64,
                seed: Some(p.seed),
                out_dir: out_dir.clone(),
                count: 1,
                max_seq: k5::DEFAULT_MAX_SEQ,
                keep_encoders: false,
                quantize_qwen: false,
                dit_nf4: false,
                init: Some(k5::Init { image: input.clone(), strength, mask: mask_path, mask_feather: k5::DEFAULT_MASK_FEATHER, mask_invert: false }),
                loras: Vec::new(),
            })
            .await
            .context("the Kandinsky 5 img2img pass")?;
        }
    }
    let produced = produced_png(&out_dir)?;
    let refined = dir.path().join("refined.png");
    std::fs::copy(&produced, &refined)?;
    if p.protect_faces {
        let _ = crate::naturalize::refine::protect_repaint_faces(&input, &refined, None, 0.85).await;
    }
    let got = image::open(&refined)?.to_rgb8();
    Ok(Frame::from_srgb(&got).resized(frame.w, frame.h))
}

/// Inpaint the holes (1 = invented) with a model; the frame outside them kept to the pixel.
pub async fn fill(frame: &Frame, holes: &Field, prompt: &str, model: &str, seed: u64, px: u32) -> Result<Frame> {
    let prompt = if prompt.trim().is_empty() { "a photograph".to_string() } else { prompt.to_string() };
    let filled = img2img(frame, &Pass { strength: 0.95, prompt: &prompt, negative: "text, watermark, frame, border, cartoon, painting, blurry", model, seed, controls: &[], px, steps: 24.0, guidance: 6.0, mask: Some(holes), protect_faces: false }).await.context("the neural fill")?;
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

/// The ControlNet specs of a plan's string: separated by spaces, commas or semicolons; "none" = none.
pub fn split_controls(s: &str) -> Vec<String> {
    s.split(|c: char| c.is_whitespace() || c == ',' || c == ';').map(str::trim).filter(|c| !c.is_empty() && !c.eq_ignore_ascii_case("none")).map(str::to_string).collect()
}

/// The conversion's prompt: a neutral photograph of what the picture shows — no period, no process, the medium's
/// pass comes after.
pub fn photo_prompt(scene: &str) -> String {
    let what = if scene.trim().is_empty() { "the same scene".to_string() } else { scene.trim().to_string() };
    format!("a photograph of {what}, realistic, natural daylight, true colours, sharp detail, 50 mm lens")
}

/// The medium's prompt with the scene's words after it, when the plan has them.
pub fn medium_prompt(prompt: &str, scene: &str) -> String {
    match (prompt.trim().is_empty(), scene.trim().is_empty()) {
        (true, true) => "a photograph".to_string(),
        (true, false) => format!("a photograph of {}", scene.trim()),
        (false, true) => prompt.trim().to_string(),
        (false, false) => format!("{}, {}", prompt.trim(), scene.trim()),
    }
}

const CONVERT_NEGATIVE: &str = "painting, watercolor, drawing, illustration, sketch, brush strokes, paper texture, cartoon, anime, 3d render, deformed, text, watermark";
const REFINE_NEGATIVE: &str = "text, watermark, cartoon, painting, illustration, 3d render, deformed";

/// The schedule's total for `steps` real steps at `strength`: an img2img pass at strength s runs only the last
/// s of the schedule, so a light pass on 24 steps runs six.
pub fn schedule_steps(steps: f32, strength: f32) -> usize {
    ((steps.max(1.0) / strength.clamp(0.05, 0.95)).round() as usize).clamp(24, 100)
}

/// The pass's negative with the plan's words after it.
pub fn join_negative(base: &str, extra: &str) -> String {
    if extra.trim().is_empty() { base.to_string() } else { format!("{base}, {}", extra.trim()) }
}

/// THE CONVERSION: a drawing, a watercolour, a painting re-rendered as a neutral photograph of what it shows
/// (`photo_prompt(scene)`), the structure held by `controls` on the SD family (the soft edges for the contours —
/// the hat's shape, the branches) and by the flow's own init on SD3 and Kandinsky, at a heavy `strength`. With a
/// `ground` strength that differs and a matte, the pass runs twice and the figure (the matte) takes the first,
/// the ground the second — the background held closer to the picture than the figure, or the other way. No
/// face is protected: a drawing's faces are what the conversion is for. The medium's pass (`refine`) comes after.
pub async fn convert(frame: &Frame, matte: Option<&Field>, strength: f32, ground: f32, scene: &str, negative: &str, model: &str, seed: u64, controls: &[String], px: u32, steps: f32, guidance: f32) -> Result<(Frame, bool)> {
    let prompt = photo_prompt(scene);
    let neg = join_negative(CONVERT_NEGATIVE, negative);
    let pass = |strength: f32| Pass { strength, prompt: &prompt, negative: &neg, model, seed, controls, px, steps, guidance, mask: None, protect_faces: false };
    let figure = img2img(frame, &pass(strength)).await.context("the conversion")?;
    let split = ground > 0.0 && (ground - strength).abs() > 0.02 && matte.map_or(false, |m| m.w == frame.w && m.h == frame.h && m.v.iter().any(|v| *v > 0.5));
    if !split {
        return Ok((figure, false));
    }
    let m = matte.unwrap();
    let back = img2img(frame, &pass(ground)).await.context("the conversion of the ground")?;
    let soft = super::gaussian_field(m, (frame.w.max(frame.h) as f32 / 400.0).max(1.0));
    let mut out = figure.clone();
    for i in 0..out.px.len() {
        let t = soft.v[i].clamp(0.0, 1.0);
        for c in 0..3 {
            out.px[i][c] = figure.px[i][c] * t + back.px[i][c] * (1.0 - t);
        }
    }
    Ok((out, true))
}

/// THE REFINE: a whole-frame img2img at `strength` with the medium's prompt (and the scene's words), the
/// structure held by `controls` on the SD family. The faces are composited back from before the pass only at a
/// light strength (≤ 0.4, a photograph refined); a heavy pass re-renders them too.
pub async fn refine(frame: &Frame, strength: f32, prompt: &str, negative: &str, model: &str, seed: u64, controls: &[String], px: u32, steps: f32, guidance: f32) -> Result<Frame> {
    let neg = join_negative(REFINE_NEGATIVE, negative);
    img2img(frame, &Pass { strength, prompt, negative: &neg, model, seed, controls, px, steps, guidance, mask: None, protect_faces: strength <= 0.4 }).await.context("the neural refine")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompts_and_the_control_specs_compose() {
        assert_eq!(split_controls("depth:strength=0.7 softedge:strength=0.5"), vec!["depth:strength=0.7", "softedge:strength=0.5"]);
        assert_eq!(split_controls("depth:strength=0.5, none"), vec!["depth:strength=0.5"]);
        assert!(split_controls("none").is_empty() && split_controls("").is_empty());
        assert!(photo_prompt("").starts_with("a photograph of the same scene"));
        assert!(photo_prompt(" a man on a path ").starts_with("a photograph of a man on a path, realistic"));
        assert_eq!(medium_prompt("", ""), "a photograph");
        assert_eq!(medium_prompt("an albumen print", ""), "an albumen print");
        assert_eq!(medium_prompt("an albumen print", "two boys"), "an albumen print, two boys");
        assert_eq!(medium_prompt("", "two boys"), "a photograph of two boys");
        assert_eq!(join_negative("a, b", ""), "a, b");
        assert_eq!(join_negative("a, b", " c "), "a, b, c");
        assert_eq!(family_of("sdxl"), Family::Sd);
        assert_eq!(family_of("sd15"), Family::Sd);
        assert_eq!(family_of("sd35-medium"), Family::Sd3);
        assert_eq!(family_of("sd35-large"), Family::Sd3);
        assert_eq!(family_of("kandinsky5"), Family::Kandinsky);
        assert!(Family::Sd.takes_controls() && !Family::Sd3.takes_controls());
        assert_eq!(schedule_steps(20.0, 0.25), 80);
        assert_eq!(schedule_steps(20.0, 0.7), 29);
        assert_eq!(schedule_steps(20.0, 0.95), 24);
        assert_eq!(schedule_steps(60.0, 0.1), 100);
    }
}
