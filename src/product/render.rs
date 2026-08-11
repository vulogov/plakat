//! The render orchestration (RFC PRODUCT-1). P1 shipped the weight-free path (a supplied cutout → sweep +
//! grounding → shot). **P2 adds the model half**: a subject from a **photo** (`matting::matte`) or a
//! **prompt** (`api::Generate` → matte), and **relight** to a `lighting` rig via IC-Light. The grounding +
//! composite stay weight-free; a supplied cutout with relight off is still no-GPU.

use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, RgbImage, RgbaImage};
use std::path::{Path, PathBuf};

use super::compose::{self, Plan};
use super::spec::ProductSpec;

/// Options for [`render_spec`].
#[derive(Debug, Clone, Default)]
pub struct RenderOpts {
    /// Override the spec's `subject.image` with this cutout path.
    pub subject: Option<PathBuf>,
    /// Relight the subject to the `lighting` rig (IC-Light). A `lighting:` block or `--relight` opts in;
    /// `--no-relight` forces off.
    pub relight: bool,
    /// Device selector for the model steps (matte / generate / relight).
    pub device: Option<String>,
}

/// What a render produced.
#[derive(Debug, Clone)]
pub struct Report {
    pub shot: PathBuf,
    pub sidecar: PathBuf,
    pub w: u32,
    pub h: u32,
    /// True when no model ran (a supplied cutout, no relight).
    pub weight_free: bool,
    /// How the subject was obtained: `cutout` | `photo` | `prompt`, and whether it was relit.
    pub subject_source: &'static str,
    pub relit: bool,
}

fn model_base(model: &str) -> u32 {
    let m = model.to_ascii_lowercase();
    if m.starts_with("sd15") || m.starts_with("sd21") || m == "sd" { 512 } else { 1024 }
}

/// Build an RGBA cutout from a matte (`rgb` + `alpha`, same dims).
fn cutout_from_matte(rgb: &RgbImage, alpha: &GrayImage) -> RgbaImage {
    let (w, h) = rgb.dimensions();
    let mut out = RgbaImage::new(w, h);
    for (x, y, p) in rgb.enumerate_pixels() {
        let a = alpha.get_pixel(x.min(alpha.width() - 1), y.min(alpha.height() - 1)).0[0];
        out.put_pixel(x, y, image::Rgba([p.0[0], p.0[1], p.0[2], a]));
    }
    out
}

/// Compile a `lighting` rig into an IC-Light prompt (rig phrase + key direction + warmth + a studio tail).
fn lighting_prompt(spec: &ProductSpec) -> String {
    let l = spec.lighting.clone().unwrap_or_default();
    if let Some(p) = l.prompt.as_deref().filter(|s| !s.trim().is_empty()) {
        return p.to_string();
    }
    let rig = match l.rig.as_deref().map(|s| s.to_ascii_lowercase()).as_deref() {
        Some("softbox") => "soft even softbox lighting",
        Some("beauty") => "beauty-dish lighting from the front",
        Some("rim") => "dramatic rim light from behind",
        Some("hard") => "hard directional key light with crisp shadows",
        Some("flat") => "flat even lighting",
        _ => "professional three-point studio lighting",
    };
    let dir = match l.key_dir.as_deref().map(|s| s.to_ascii_lowercase()).as_deref() {
        Some("top-left") | Some("left") => ", key light from the top left",
        Some("top-right") | Some("right") => ", key light from the top right",
        Some("top") => ", key light from above",
        _ => "",
    };
    let warmth = match l.warmth.unwrap_or(0.0) {
        w if w > 0.25 => ", warm golden tone",
        w if w < -0.25 => ", cool daylight tone",
        _ => "",
    };
    format!("{rig}{dir}{warmth}, studio product photography, clean seamless background, sharp focus")
}

/// Resolve the subject to an RGBA cutout, running models only as needed. Returns the cutout, a source
/// label, whether it was relit, and whether any model ran.
async fn resolve_subject(spec: &ProductSpec, opts: &RenderOpts, tmp: &Path) -> Result<(RgbaImage, &'static str, bool, bool)> {
    let subj = spec.subject.clone().unwrap_or_default();
    let model = spec.model.as_deref().unwrap_or("sdxl").to_string();
    let device = |sel: Option<&str>| crate::api::device(sel.unwrap_or("auto"));
    let mut used_model = false;

    // ---- 1. base cutout (cutout | photo→matte | prompt→gen→matte) ----
    let cutout_path = opts.subject.as_ref().map(|p| p.to_string_lossy().to_string()).or_else(|| spec.subject_image().map(str::to_string));
    let (mut cutout, source): (RgbaImage, &'static str) = if let Some(p) = cutout_path {
        (image::open(&p).with_context(|| format!("reading subject {p}"))?.to_rgba8(), "cutout")
    } else if let Some(photo) = subj.photo.as_deref().filter(|s| !s.trim().is_empty()) {
        let dev = device(opts.device.as_deref())?;
        let (rgb, alpha) = crate::pipelines::matting::matte(Path::new(photo), &dev).await.with_context(|| format!("matting photo {photo}"))?;
        used_model = true;
        (cutout_from_matte(&rgb, &alpha), "photo")
    } else if let Some(prompt) = subj.prompt.as_deref().filter(|s| !s.trim().is_empty()) {
        let base = model_base(&model);
        let full = format!("{prompt}, isolated on a plain white background, product photography, centered, the whole product visible");
        let mut g = crate::api::Generate::new(&model)
            .prompt(full)
            .negative("multiple products, cropped, busy background, text, watermark, reflection, shadow")
            .size(base, base)
            .steps(spec.steps.unwrap_or(30))
            .seed(spec.seed.unwrap_or(0))
            .count(1);
        if let Some(d) = &opts.device {
            g = g.device(d);
        }
        let imgs = g.run().await.context("generating subject")?;
        let gen_path = tmp.join("subject_gen.png");
        imgs.first().context("subject generation produced no image")?.save(&gen_path)?;
        let dev = device(opts.device.as_deref())?;
        let (rgb, alpha) = crate::pipelines::matting::matte(&gen_path, &dev).await.context("matting the generated subject")?;
        used_model = true;
        (cutout_from_matte(&rgb, &alpha), "prompt")
    } else {
        anyhow::bail!("no subject — set `subject.image` (a cutout), `subject.photo`, or `subject.prompt`");
    };

    // ---- 2. relight to the rig (optional) ----
    let mut relit = false;
    if opts.relight {
        let dev = device(opts.device.as_deref())?;
        let src_png = tmp.join("subject_pre_relight.png");
        cutout.save(&src_png)?; // IC-Light mattes the alpha off internally; the cutout is fine as input
        let prompt = lighting_prompt(spec);
        let pipeline = crate::pipelines::ic_light::Pipeline::load(dev.clone()).await.context("loading IC-Light for relight")?;
        let (buf, w, h) = pipeline
            .relight(&src_png, &prompt, "harsh shadows, blown highlights, color cast, lowres", 512, 512, spec.steps.unwrap_or(20), 2.0, spec.seed.unwrap_or(0))
            .context("relighting subject")?;
        let relit_rgb = RgbImage::from_raw(w, h, buf).context("relit buffer → image")?;
        let relit_png = tmp.join("subject_relit.png");
        relit_rgb.save(&relit_png)?;
        // re-matte the relit frame (relight composites on grey) to recover a clean cutout.
        let (rgb2, alpha2) = crate::pipelines::matting::matte(&relit_png, &dev).await.context("re-matting the relit subject")?;
        cutout = cutout_from_matte(&rgb2, &alpha2);
        used_model = true;
        relit = true;
    }

    Ok((cutout, source, relit, used_model))
}

/// The full render: resolve the subject (models as needed) → ground + composite → write shot + sidecar.
pub async fn render_spec(spec: &ProductSpec, out: &Path, opts: &RenderOpts) -> Result<Report> {
    let plan: Plan = compose::resolve(spec);
    let tmp = tempfile::tempdir().context("temp dir for product render")?;
    let (cutout, source, relit, used_model) = resolve_subject(spec, opts, tmp.path()).await?;
    let shot = compose::compose(&plan, &DynamicImage::ImageRgba8(cutout));
    shot.save(out).with_context(|| format!("writing {}", out.display()))?;
    let sidecar = out.with_extension("meta.json");
    std::fs::write(&sidecar, compose::meta_json(spec, &plan)).with_context(|| format!("writing {}", sidecar.display()))?;
    Ok(Report { shot: out.to_path_buf(), sidecar, w: plan.w, h: plan.h, weight_free: !used_model, subject_source: source, relit })
}
