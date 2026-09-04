//! `plakat naturalize` (RFC QUALITY-1) — the weight-free analog post-pass: film grain + chromatic
//! aberration + vignette + bloom + a desaturating film grade over any image, to break the digital-clean,
//! over-saturated "AI-generated" fingerprint. No GPU.
//!
//! Etch bar (RFC §Etch preservation): plakat's own provenance is carried forward — the L0 JSON sidecar +
//! the PNG text chunks are copied onto the output so `doctor --if-plakat` still finds it. (A proper
//! re-etch — re-embedding L1 into the new pixels with a `parent` chain — lands in P2; `--no-reetch` writes
//! a clean, un-etched output.)

use anyhow::{Context, Result};
use clap::Args;
use console::style;
use std::path::{Path, PathBuf};

use crate::naturalize::{self, Params};

#[derive(Args, Debug, Clone)]
pub struct NaturalizeArgs {
    /// Input image (or a directory for batch). Not required with `--list-presets` / `--export-lut` /
    /// `--save-preset`.
    #[arg(required_unless_present_any = ["list_presets", "export_lut", "save_preset"])]
    pub input: Option<PathBuf>,
    /// Output image (or directory for batch). Not required with `--report` / `--list-presets` /
    /// `--export-lut` / `--save-preset`.
    #[arg(long, required_unless_present_any = ["report", "list_presets", "export_lut", "save_preset"])]
    pub out: Option<PathBuf>,
    /// Strength bundle: `subtle` (default) | `photo` | `painting` | **`auto`** (tune to the source model
    /// read from metadata) | a library preset (see `--list-presets`). All aim at contemporary realism.
    #[arg(long)]
    pub preset: Option<String>,
    /// Override film-grain amount (0..1).
    #[arg(long)]
    pub grain: Option<f32>,
    /// Override chromatic-aberration amount (0..1).
    #[arg(long)]
    pub aberration: Option<f32>,
    /// Override vignette amount (0..1).
    #[arg(long)]
    pub vignette: Option<f32>,
    /// Override highlight bloom (0..1).
    #[arg(long)]
    pub bloom: Option<f32>,
    /// Override desaturation toward luminance (0..1).
    #[arg(long)]
    pub desaturate: Option<f32>,
    /// Override warm film lift in the shadows (0..1).
    #[arg(long)]
    pub warm: Option<f32>,
    /// Override radial defocus (0..1).
    #[arg(long)]
    pub defocus: Option<f32>,
    /// **Watercolor paper / pigment authenticity** (0..1) — model real wet-on-wet media: paper tooth
    /// (pigment settles in the valleys) + granulation speckle + edge pooling. Applied only where there is
    /// pigment (washes), so bare paper / photos are untouched. For genuine watercolour/ink-wash art;
    /// auto-applied at 0.6 when the medium is wet (`--medium watercolor` or auto-detected). `--paper 0`
    /// disables.
    #[arg(long)]
    pub paper: Option<f32>,
    /// **Brush strokes** (0..1) — synthesize human-like, gradient-aligned strokes for a painting
    /// medium: watercolor edge-pooling + bleed, oil Kuwahara-flatten + directional strokes +
    /// impasto, gouache flat strokes, ink hatching, pastel chalk. Opt-in — fires only when a
    /// painting `--medium` is set (or auto-detected via `--auto-medium`). Default `0.6` when a
    /// medium is present; `--brush-strength 0` disables. Weight-free, follows the forms.
    #[arg(long = "brush-strength", value_name = "N")]
    pub brush_strength: Option<f32>,
    /// **Brush-stroke size** (multiplier, default 1.0) — `<1` = finer/denser strokes (less blocky,
    /// more detail), `>1` = broader daubs / palette-knife. The knob for "too chunky". Try `0.5`.
    #[arg(long = "brush-scale", value_name = "N")]
    pub brush_scale: Option<f32>,
    /// Run CLIP medium-detection on a **weight-free** run (it otherwise only runs for model corrections),
    /// so auto-paper can fire for detected watercolour/ink-wash art without naming `--medium`.
    #[arg(long = "auto-medium", default_value_t = false)]
    pub auto_medium: bool,
    /// **Analyze** the image and print a de-slop scorecard (AI-tell drivers + detected medium + a
    /// recommended recipe) instead of processing it. Add `--json` for a structured report.
    #[arg(long, default_value_t = false)]
    pub report: bool,
    /// With `--report`, emit JSON instead of the table.
    #[arg(long, default_value_t = false)]
    pub json: bool,
    /// Export the fixed film **grade** (desaturate + warm, from the preset / `--desaturate` / `--warm`) as a
    /// standard **`.cube`** 3-D LUT to this path, for DaVinci Resolve / Premiere / OBS. **With an input
    /// image** the LUT bakes that image's PER-IMAGE colour grade (WB + auto-levels + vibrance + grade);
    /// with no input it's the fixed film grade. Spatial unsharp/micro are colour-LUT-excluded either way.
    #[arg(long = "export-lut", value_name = "PATH.cube")]
    pub export_lut: Option<PathBuf>,
    /// `.cube` LUT cube size (per-axis; default 33).
    #[arg(long = "lut-size", value_name = "N")]
    pub lut_size: Option<usize>,
    /// List the named preset library (built-in + your saved presets) and exit.
    #[arg(long = "list-presets", default_value_t = false)]
    pub list_presets: bool,
    /// **Save** the resolved knob spec as a named user preset (`~/.config/plakat/naturalize.presets`),
    /// reusable via `--preset <name>` (user presets shadow the built-ins), then exit.
    #[arg(long = "save-preset", value_name = "NAME")]
    pub save_preset: Option<String>,
    /// **Auto-region focuses** — detect subjects and de-slop each in its own profile: faces→`people`/micro,
    /// a sky band→`sky`, the rest→the base. Composited with feathered seams. Face detection needs a model.
    #[arg(long = "auto-regions", default_value_t = false)]
    pub auto_regions: bool,
    /// Manual **region focus** (repeatable): `x0,y0,x1,y1:<spec>` in normalized 0..1 coords, e.g.
    /// `--region "0,0,1,0.4:sky=1"`. The spec is a naturalize spec applied to that (feathered) rectangle.
    #[arg(long = "region", value_name = "X0,Y0,X1,Y1:SPEC", help_heading = "Content focus")]
    pub regions: Vec<String>,
    /// Override the **quality-improvement** ("de-slop") strength (0..1) — gray-world white balance +
    /// robust auto-levels + vibrance + unsharp, run FIRST to make the colours & detail genuinely better
    /// before any analog look. `0` disables it. Defaults come from the preset (subtle 0.55 … photo 0.70).
    #[arg(long)]
    pub polish: Option<f32>,
    /// Override the **micro-texture** strength (0..1+) — fine pore / micro-wrinkle detail added only to the
    /// unnaturally-smooth regions (variance-gated, mid-tones), the fix for plastic AI skin. High on
    /// `--people`; set explicitly for other smooth surfaces.
    #[arg(long)]
    pub micro: Option<f32>,

    // ---- content focus qualifiers (RFC QUALITY-1): pre-tune the pass to a subject's AI tell. `N` is the
    // blend weight (0 = off, 1 = midpoint of the preset and that subject's de-AI profile, >1 = stronger).
    /// Focus for **people / portraits** (plastic skin) — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub people: Option<f32>,
    /// Focus for **skies** (banding / too-smooth) — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub sky: Option<f32>,
    /// Focus for **vegetation / foliage** (cloud-like repeating mush) — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub vegetation: Option<f32>,
    /// Focus for **cityscapes** (razor-clean geometry) — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub cityscape: Option<f32>,
    /// Focus for **landscape / scenery** (atmosphere) — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub landscape: Option<f32>,
    /// Focus for **seascape** surface — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub sea: Option<f32>,
    /// Focus for **riverscape** surface — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub river: Option<f32>,
    /// Focus for **mechanical apparatus / transports** surface — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub mechanics: Option<f32>,
    /// Focus for **household / indoor** scenes — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub household: Option<f32>,
    /// Focus for **animals** (fur/feather over-smoothness) — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub animal: Option<f32>,
    /// Focus for **food** (plastic sheen / oversaturation) — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub food: Option<f32>,
    /// Focus for **interior / architectural render** (flat CGI light) — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub interior: Option<f32>,
    /// Focus for **textile / fabric** (smooth sheen) — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub textile: Option<f32>,
    /// Focus for **foliage macro / close-up botanical** — weight N.
    #[arg(long = "foliage-macro", value_name = "N", help_heading = "Content focus")]
    pub foliage_macro: Option<f32>,

    // ---- corrective focuses (model-backed: img2img / inpaint, NOT the analog pass) ----
    /// Fix **geometry** (incoherent structure / joinery) via img2img — weight N.
    #[arg(long, value_name = "N", help_heading = "Corrective (needs a model)")]
    pub geometry: Option<f32>,
    /// Fix **anatomy** (proportions / hands) via img2img — weight N.
    #[arg(long, value_name = "N", help_heading = "Corrective (needs a model)")]
    pub anatomy: Option<f32>,
    /// Make **lookalike faces distinct** (detect + inpaint each duplicate) — weight N.
    #[arg(long, value_name = "N", help_heading = "Corrective (needs a model)")]
    pub no_twins: Option<f32>,
    /// **Face-protected repair** (0..1) — the art-safe structural fix. Detects faces and PROTECTS them
    /// (never regenerated → soft artistic faces survive, no uncanny valley), then gently repaints the rest
    /// IN-STYLE to attempt broken hands/feet/limbs. Preserves character where whole-image `--geometry`
    /// would wreck it. Pair with `--style`/`--medium` to hold the medium. Needs a model + faces.
    #[arg(long, value_name = "N", help_heading = "Corrective (needs a model)")]
    pub repair: Option<f32>,
    /// What `--repair` may touch: `figures` (default — only the figures' bodies, faces AND background
    /// preserved) · `non-face` (all non-face pixels, background regenerates) · `full` (whole image).
    #[arg(long = "repair-scope", value_name = "SCOPE", default_value = "figures", help_heading = "Corrective (needs a model)")]
    pub repair_scope: String,
    /// **Painterly repaint** (model-backed, 6.27) — re-paint the whole image via img2img anchored to
    /// the medium (`--medium`/`--style`, or CLIP auto-detected), so the DIFFUSION MODEL builds the
    /// brush strokes **and** the form together (real watercolor washes / oil impasto, not a surface
    /// filter — and it also rebuilds melted forms). The generation-side complement to the weight-free
    /// `--medium` pass. Opt-in; needs a model. Works for **any** medium (watercolor included).
    #[arg(long, default_value_t = false, help_heading = "Corrective (needs a model)")]
    pub repaint: bool,
    /// `--repaint` denoise strength (0..1). Low (~0.3–0.4) keeps the composition + adds media
    /// character; higher repaints more (more strokes, more form change). Default 0.38.
    #[arg(long = "repaint-strength", value_name = "N", help_heading = "Corrective (needs a model)")]
    pub repaint_strength: Option<f32>,
    /// Model alias for `--repaint` (default: `--model`). A painterly checkpoint gives the most
    /// authentic media.
    #[arg(long = "repaint-model", value_name = "ALIAS", help_heading = "Corrective (needs a model)")]
    pub repaint_model: Option<String>,
    /// An optional painterly LoRA for `--repaint` (`spec[:scale]`, where `spec` is a real Hub id or local
    /// path you have, `scale` defaults 0.8) — pushes the medium harder than the prompt anchor alone. Omit it
    /// for a pure prompt-anchored repaint (the prompt medium already works well). If the LoRA can't load,
    /// plakat retries WITHOUT it rather than reverting to the baseline.
    #[arg(long = "repaint-lora", value_name = "SPEC", help_heading = "Corrective (needs a model)")]
    pub repaint_lora: Option<String>,
    /// Art **style/medium** to preserve during model corrections (`--repair`/`--geometry`/`--anatomy`),
    /// e.g. `--style "vintage watercolor storybook illustration"`. Anchors the re-paint to the source
    /// medium instead of drifting to photoreal (the cause of art regressions).
    #[arg(long, value_name = "TEXT", help_heading = "Corrective (needs a model)")]
    pub style: Option<String>,
    /// Shorthand for common art media (expands to a `--style` string): `watercolor` | `oil` | `ink` |
    /// `gouache` | `pencil` | `acrylic` | `pastel` | `comic`. `--style` overrides this.
    #[arg(long, value_name = "MEDIUM", help_heading = "Corrective (needs a model)")]
    pub medium: Option<String>,
    /// **De-clutter** — remove named nonsensical slop objects (OWL-ViT + inpaint) BEFORE the geometry
    /// fix. Comma-separated, e.g. `--declutter "overhead wires,cables"`. The only thing that kills a
    /// *compositional* hallucination (floating catenary wires, phantom rails) that img2img can't fix in
    /// place. Best-effort: an undetected target is skipped. Needs a model.
    #[arg(long, value_name = "OBJECTS", help_heading = "Corrective (needs a model)")]
    pub declutter: Option<String>,

    /// Model for the corrective img2img / inpaint passes.
    #[arg(long, default_value = "sdxl", help_heading = "Corrective (needs a model)")]
    pub model: String,
    /// Steps for the corrective passes.
    #[arg(long, default_value_t = 24, help_heading = "Corrective (needs a model)")]
    pub refine_steps: usize,
    /// Device for the corrective passes.
    #[arg(long, default_value = "auto", help_heading = "Corrective (needs a model)")]
    pub device: String,

    /// Remove a **ghost signature** smudge from a corner (`br`/`bl`/`tr`/`tl`) — a weight-free
    /// content-aware dissolve. Foreign-artifact only; never touches plakat's own etch.
    #[arg(long, value_name = "CORNER")]
    pub designature: Option<String>,
    /// Strength of the ghost-signature dissolve (0..1).
    #[arg(long, default_value_t = 0.9)]
    pub designature_strength: f32,

    /// Write a clean, un-etched output — do NOT carry plakat provenance forward.
    #[arg(long)]
    pub no_reetch: bool,
}

/// Whether a `--declutter` target names thin-line clutter (wires/cables/lines) — routed to the weight-free
/// [`naturalize::wire_mask`] detector instead of OWL-ViT, which can't see thin lines.
fn is_wire_query(q: &str) -> bool {
    let q = q.to_ascii_lowercase();
    ["wire", "cable", "power line", "overhead line", "catenary", "telephone line", "electric line", "tram line", "wiring"]
        .iter()
        .any(|k| q.contains(k))
}

/// Resolve the art style to preserve during model corrections: explicit `--style` wins; else expand a
/// `--medium` preset to a descriptive style string; else `None`.
use naturalize::is_wet_media;

fn resolve_style(style: Option<&str>, medium: Option<&str>) -> Option<String> {
    if let Some(s) = style {
        if !s.trim().is_empty() {
            return Some(s.to_string());
        }
    }
    let m = medium?.trim().to_ascii_lowercase();
    let s = match m.as_str() {
        "watercolor" | "watercolour" => "soft wet-on-wet watercolor illustration, natural pigment granulation, paper texture",
        "oil" => "oil painting, visible brush strokes, impasto texture, canvas",
        "ink" => "ink drawing, pen and ink linework, cross-hatching",
        "gouache" => "gouache painting, matte opaque pigment, flat washes",
        "pencil" => "graphite pencil sketch, soft shading, paper tooth",
        "acrylic" => "acrylic painting, bold brushwork",
        "pastel" => "soft pastel drawing, chalky pigment, blended tones",
        "comic" => "comic book illustration, clean ink lines, cel shading",
        other => return Some(other.to_string()), // pass unknown media through verbatim
    };
    Some(s.to_string())
}

/// Read the source model name from an image's generation metadata (plakat JSON sidecar, else the A1111
/// `parameters` tEXt chunk) for `--preset auto` (RFC QUALITY-8 P1). `None` if no metadata / no model.
fn read_source_model(input: &Path) -> Option<String> {
    let sidecar = input.with_extension("json");
    if let Ok(s) = std::fs::read_to_string(&sidecar) {
        if let Ok(m) = serde_json::from_str::<crate::imaging::metadata::GenerationMetadata>(&s) {
            if !m.model.trim().is_empty() {
                return Some(m.model);
            }
        }
    }
    if let Ok(Some(text)) = crate::imaging::io::read_parameters_chunk(input) {
        if let Some(m) = crate::cli::clone::parse_a1111(&text) {
            if !m.model.trim().is_empty() {
                return Some(m.model);
            }
        }
    }
    None
}

/// Whether the input looks like a **genuine camera photograph** (EXIF camera Make/Model/ISO present) —
/// RFC QUALITY-8 P2. De-slop shouldn't fight a real photo, so `--preset auto` gives these a gentle touch.
/// EXIF reading needs the `photos` feature; without it there's no signal → always `false` (normal de-slop).
#[cfg(feature = "photos")]
fn is_real_photo(input: &Path) -> bool {
    match crate::photos::exif::read_exif(input) {
        Ok(r) => r.camera_make.is_some() || r.camera_model.is_some() || r.iso.is_some(),
        Err(_) => false,
    }
}
#[cfg(not(feature = "photos"))]
fn is_real_photo(_input: &Path) -> bool {
    false
}

/// Resolve the base [`Params`] for `--preset`, including **`auto`** (RFC QUALITY-8 P1): read the source
/// model from metadata → its tuned preset; no model → an analysis-driven fallback; else the named/base
/// preset via [`naturalize::base_params`].
fn resolve_base_params(a: &NaturalizeArgs, input: Option<&Path>) -> Result<Params> {
    if a.preset.as_deref() == Some("auto") {
        // A genuine camera photo (EXIF) → gentle touch: de-slop must not fight a real photograph.
        if input.map(is_real_photo).unwrap_or(false) {
            println!("  {} auto preset · real photo (EXIF camera) → gentle touch", style("de-slop").cyan());
            return Ok(naturalize::from_spec("subtle polish=0.5 desaturate=0 grain=0.04 micro=0.05 aberration=0"));
        }
        let model = input.and_then(read_source_model);
        if let Some(spec) = model.as_deref().and_then(naturalize::model_preset) {
            let mut p = naturalize::from_spec(spec);
            // Adaptive: a very-smooth (plastic) image — portraits especially — wants more micro-texture
            // than the model default, regardless of which model made it.
            let mut note = String::new();
            if let Some(inp) = input {
                if let Ok(img) = image::open(inp) {
                    let s = naturalize::analyze(&img.to_rgb8()).smoothness_tell;
                    if s > 0.85 {
                        p.micro = (p.micro + 0.2).min(1.0);
                        note = format!(" (+micro→{:.2}, very smooth)", p.micro);
                    }
                }
            }
            println!("  {} auto preset · model '{}' → {spec}{note}", style("de-slop").cyan(), model.as_deref().unwrap_or("?"));
            return Ok(p);
        }
        // No recognised model — pick photo vs painting-ish from the image's own tells (analysis).
        if let Some(inp) = input {
            if let Ok(img) = image::open(inp) {
                let an = naturalize::analyze(&img.to_rgb8());
                let preset = if an.smoothness_tell > 0.6 { "photo" } else { "subtle" };
                println!("  {} auto preset · no model metadata → analysis ({preset}; ai-tell {:.2})", style("de-slop").cyan(), an.ai_tell);
                return Ok(naturalize::base_params(Some(preset)).unwrap_or_else(|| naturalize::base_params(Some("photo")).unwrap()));
            }
        }
        return Ok(naturalize::base_params(Some("photo")).unwrap());
    }
    naturalize::base_params(a.preset.as_deref())
        .with_context(|| format!("unknown preset `{}` (subtle|photo|painting|auto, or a library preset — see `--list-presets`)", a.preset.as_deref().unwrap_or("")))
}

/// Whether the input is a video/animation container (QUALITY-6 P2).
fn is_video(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("mp4" | "mov" | "webm" | "mkv" | "avi" | "gif" | "m4v")
    )
}

/// Build the **weight-free** naturalize params (preset + focuses + overrides) and the paper amount from the
/// args — the part safe to run per-video-frame (no model). Shared by the still and video paths.
#[allow(clippy::type_complexity)]
fn weightfree_params(a: &NaturalizeArgs, input: Option<&Path>) -> Result<(Params, Option<f32>, Option<(naturalize::brush::Medium, f32)>)> {
    let base = resolve_base_params(a, input)?;
    let focuses: Vec<(naturalize::Focus, f32)> = [
        (naturalize::Focus::People, a.people), (naturalize::Focus::Sky, a.sky), (naturalize::Focus::Vegetation, a.vegetation),
        (naturalize::Focus::Cityscape, a.cityscape), (naturalize::Focus::Landscape, a.landscape), (naturalize::Focus::Sea, a.sea),
        (naturalize::Focus::River, a.river), (naturalize::Focus::Mechanics, a.mechanics), (naturalize::Focus::Household, a.household),
        (naturalize::Focus::Animal, a.animal), (naturalize::Focus::Food, a.food), (naturalize::Focus::Interior, a.interior),
        (naturalize::Focus::Textile, a.textile), (naturalize::Focus::FoliageMacro, a.foliage_macro),
    ].into_iter().filter_map(|(f, n)| n.filter(|v| *v > 0.0).map(|v| (f, v))).collect();
    let mut p = naturalize::blend_focus(base, &focuses);
    if let Some(v) = a.grain { p.grain = v; }
    if let Some(v) = a.aberration { p.aberration = v; }
    if let Some(v) = a.vignette { p.vignette = v; }
    if let Some(v) = a.bloom { p.bloom = v; }
    if let Some(v) = a.desaturate { p.desaturate = v; }
    if let Some(v) = a.warm { p.warm = v; }
    if let Some(v) = a.defocus { p.defocus = v; }
    if let Some(v) = a.polish { p.polish = v; }
    if let Some(v) = a.micro { p.micro = v; }
    let art_style = resolve_style(a.style.as_deref(), a.medium.as_deref());
    let paper_amt = a.paper.or_else(|| art_style.as_deref().filter(|s| is_wet_media(s)).map(|_| 0.6));
    let brush = art_style
        .as_deref()
        .and_then(naturalize::brush::Medium::detect)
        .map(|m| (m, a.brush_strength.unwrap_or(0.6)))
        .filter(|(_, s)| *s > 0.0);
    Ok((p, paper_amt, brush))
}

/// QUALITY-6 P2: de-slop every frame of a video/animation and re-encode. Weight-free only — the analog
/// pass's noise is seeded per-pixel (frame-invariant), so the grain/paper texture **sits still** while the
/// image moves (no flicker). Output container follows the `--out` extension (mp4 / webm / gif).
async fn naturalize_video(a: &NaturalizeArgs, input: &Path) -> Result<()> {
    let out = a.out.clone().context("--out is required")?;
    crate::imaging::video::ffmpeg_version().context("video de-slop needs ffmpeg on PATH")?;
    let (p, paper_amt, brush) = weightfree_params(a, Some(input))?;
    let tmp = tempfile::tempdir().context("temp dir for video frames")?;
    let frames = crate::imaging::video::extract_frames(input, tmp.path())?;
    anyhow::ensure!(!frames.is_empty(), "no frames extracted from {}", input.display());
    let fps = crate::imaging::video::probe_fps(input);
    println!("naturalize video: {} frames @ {fps}fps → {} (weight-free, frame-invariant grain)", frames.len(), out.display());
    for f in &frames {
        let img = image::open(f).with_context(|| format!("reading frame {}", f.display()))?.to_rgb8();
        let mut o = naturalize::apply(&img, &p);
        if let Some((m, bs)) = brush {
            o = naturalize::brush::apply_brush(&o, m, bs, a.brush_scale.unwrap_or(1.0));
        }
        if let Some(pv) = paper_amt.filter(|v| *v > 0.0) {
            o = naturalize::paper_texture(&o, pv);
        }
        o.save(f).with_context(|| format!("writing frame {}", f.display()))?;
    }
    let pattern = tmp.path().join("frame_%06d.png");
    let pattern = pattern.to_str().context("non-UTF8 frame pattern")?;
    match out.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
        Some("gif") => crate::cli::animate::write_gif(&frames, &out, (1000 / fps.max(1)) as u16)?,
        Some("webm") => crate::imaging::video::frames_to_webm(pattern, &out, fps)?,
        _ => crate::imaging::video::frames_to_mp4(pattern, &out, fps)?,
    }
    println!("{} {}", style("wrote").green(), out.display());
    Ok(())
}

/// Build the per-region focus list (QUALITY-6 P3): manual `--region` rectangles + `--auto-regions`
/// (sky band + SCRFD people). Empty when neither is requested.
async fn build_regions(a: &NaturalizeArgs, input: &Path, img: &image::RgbImage, base: &Params) -> Result<Vec<(image::GrayImage, Params)>> {
    let (w, h) = (img.width(), img.height());
    let feather = (w.min(h) as f32) * 0.04;
    let mut regions: Vec<(image::GrayImage, Params)> = Vec::new();
    for spec in &a.regions {
        let (rect, focus) = spec.split_once(':').with_context(|| format!("--region needs `x0,y0,x1,y1:spec`, got {spec:?}"))?;
        let c: Vec<f32> = rect.split(',').map(|v| v.trim().parse::<f32>()).collect::<std::result::Result<_, _>>().with_context(|| format!("bad region rect {rect:?}"))?;
        anyhow::ensure!(c.len() == 4, "region rect needs 4 coords (x0,y0,x1,y1): {rect:?}");
        regions.push((naturalize::feathered_rect(w, h, c[0], c[1], c[2], c[3], feather), naturalize::from_spec(focus)));
    }
    if a.auto_regions {
        let sky = naturalize::sky_mask(img);
        if sky.pixels().any(|p| p.0[0] > 20) {
            regions.push((sky, naturalize::blend_focus(*base, &[(naturalize::Focus::Sky, 1.0)])));
        }
        if let Some(people) = detect_people_mask(input, img, &a.device).await {
            regions.push((people, naturalize::blend_focus(*base, &[(naturalize::Focus::People, 1.0)])));
        }
    }
    Ok(regions)
}

/// A feathered mask over the people in the frame (SCRFD faces → projected body boxes) for `--auto-regions`.
/// Best-effort — `None` if no detector / no faces.
async fn detect_people_mask(input: &Path, img: &image::RgbImage, device: &str) -> Option<image::GrayImage> {
    let scrfd = crate::pipelines::scrfd::resolve_scrfd_weights().await.ok().flatten()?;
    let dev = crate::api::device(device).ok()?;
    let det = crate::pipelines::scrfd::SCRFDDetector::load(&scrfd, crate::pipelines::scrfd::SCRFDConfig::default(), &dev, candle_core::DType::F32).ok()?;
    let mut faces = det.detect(input).ok()?;
    faces.retain(|f| f.score >= 0.35);
    if faces.is_empty() {
        return None;
    }
    let (w, h) = (img.width(), img.height());
    let mut raw = vec![0f32; (w * h) as usize];
    for f in &faces {
        let (fw, fh) = (f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]);
        let cx = (f.bbox[0] + f.bbox[2]) * 0.5;
        let m = naturalize::feathered_rect(w, h, (cx - 1.1 * fw) / w as f32, (f.bbox[1] - 0.2 * fh) / h as f32, (cx + 1.1 * fw) / w as f32, (f.bbox[3] + 5.0 * fh) / h as f32, (w.min(h) as f32) * 0.03);
        for (i, px) in m.pixels().enumerate() {
            raw[i] = raw[i].max(px.0[0] as f32 / 255.0);
        }
    }
    let mut out = image::GrayImage::new(w, h);
    for (i, px) in out.pixels_mut().enumerate() {
        px.0[0] = (raw[i] * 255.0) as u8;
    }
    Some(out)
}

/// QUALITY-7 P1: scan a folder and print a ranked scorecard (worst-AI first) + an aggregate summary.
/// Weight-free (no CLIP medium probe — kept fast for large folders).
fn folder_report(a: &NaturalizeArgs, input: &Path) -> Result<()> {
    let exts = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"];
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(input)
        .with_context(|| format!("reading dir {}", input.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().and_then(|x| x.to_str()).map(|e| exts.contains(&e.to_ascii_lowercase().as_str())).unwrap_or(false))
        .collect();
    files.sort();
    anyhow::ensure!(!files.is_empty(), "no images in {}", input.display());
    let mut rows: Vec<(std::path::PathBuf, naturalize::Analysis)> = Vec::new();
    for f in &files {
        match image::open(f) {
            Ok(im) => rows.push((f.clone(), naturalize::analyze(&im.to_rgb8()))),
            Err(e) => tracing::warn!(target: "plakat", "skipping {}: {e}", f.display()),
        }
    }
    anyhow::ensure!(!rows.is_empty(), "no readable images in {}", input.display());
    rows.sort_by(|x, y| y.1.ai_tell.partial_cmp(&x.1.ai_tell).unwrap_or(std::cmp::Ordering::Equal));
    let mean = rows.iter().map(|r| r.1.ai_tell).sum::<f32>() / rows.len() as f32;
    let over = rows.iter().filter(|r| r.1.ai_tell > 0.5).count();
    let (msat, msmooth) = (rows.iter().map(|r| r.1.saturation).sum::<f32>() / rows.len() as f32, rows.iter().map(|r| r.1.smoothness_tell).sum::<f32>() / rows.len() as f32);
    let dominant = if msat >= msmooth { "oversaturation" } else { "over-smoothness" };

    if a.json {
        let items: Vec<String> = rows.iter().map(|(p, an)| format!("  {{\"path\": {:?}, \"ai_tell\": {:.4}, \"saturation\": {:.4}, \"smoothness_tell\": {:.4}}}", p.display().to_string(), an.ai_tell, an.saturation, an.smoothness_tell)).collect();
        println!("{{\n  \"count\": {}, \"mean_ai_tell\": {:.4}, \"over_0.5\": {}, \"dominant_tell\": {:?},\n  \"images\": [\n{}\n  ]\n}}", rows.len(), mean, over, dominant, items.join(",\n"));
    } else {
        let bar = |v: f32| { let n = (v * 16.0).round() as usize; format!("{}{}", "█".repeat(n.min(16)), "░".repeat(16 - n.min(16))) };
        println!("\n  {} — {} image(s) in {}", style("folder scorecard").bold(), rows.len(), input.display());
        println!("  {:<20} {:<18} {:>5} {:>5} {:>6}", "", "AI-tell", "sat", "smth", "file");
        for (p, an) in &rows {
            let mark = if an.ai_tell > 0.5 { style("●").red() } else { style("●").green() };
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("?");
            println!("  {mark} {} {:.2}  {:.2}  {:.2}  {}", bar(an.ai_tell), an.ai_tell, an.saturation, an.smoothness_tell, style(name).dim());
        }
        println!("\n  summary: mean AI-tell {:.2} · {} of {} over 0.5 · dominant tell: {}\n", mean, over, rows.len(), style(dominant).cyan());
    }
    Ok(())
}

pub async fn run(a: NaturalizeArgs) -> Result<()> {
    // QUALITY-7 P3: list the named preset library and exit.
    if a.list_presets {
        println!("\n  {} — `naturalize --preset <name>`", style("preset library").bold());
        for (name, spec, desc) in naturalize::preset_library() {
            println!("  {:<10} {}\n             {}", style(name).green(), style(spec).dim(), desc);
        }
        let user = naturalize::load_user_presets();
        if !user.is_empty() {
            println!("\n  {} (shadow the built-ins)", style("your saved presets").bold());
            for (name, spec) in &user {
                println!("  {:<10} {}", style(name).cyan(), style(spec).dim());
            }
        }
        println!("\n  {:<10} {}\n", style("(base)").dim(), "subtle · photo · painting");
        return Ok(());
    }

    // QUALITY-9 P2: save the resolved knob spec as a named user preset, then exit.
    if let Some(name) = a.save_preset.clone() {
        let (p, paper, _brush) = weightfree_params(&a, a.input.as_deref())?;
        let spec = naturalize::to_spec(&p, paper.unwrap_or(0.0));
        let path = naturalize::save_user_preset(&name, &spec).context("saving the preset")?;
        println!("{} preset {} = {}\n  → {} (use it with --preset {name})", style("saved").green(), style(&name).cyan(), style(&spec).dim(), path.display());
        return Ok(());
    }

    // QUALITY-7 P2 + QUALITY-9 P1: export a .cube LUT. With an input image, bake the PER-IMAGE colour
    // grade (WB + auto-levels + vibrance fitted on that image, closing the 6.16 fixed-grade limit); with
    // no input, fall back to the fixed film grade. Spatial unsharp/micro are colour-LUT-excluded either way.
    if let Some(lut) = a.export_lut.clone() {
        let (p, _, _) = weightfree_params(&a, a.input.as_deref())?;
        let size = a.lut_size.unwrap_or(33);
        let (cube, kind) = if let Some(inp) = a.input.as_deref() {
            let src = image::open(inp).with_context(|| format!("reading {}", inp.display()))?.to_rgb8();
            (naturalize::export_cube_image(&src, p.polish, p.desaturate, p.warm, size), "per-image colour")
        } else {
            (naturalize::export_cube(p.desaturate, p.warm, size), "fixed grade")
        };
        std::fs::write(&lut, cube).with_context(|| format!("writing {}", lut.display()))?;
        println!(
            "{} {} ({size}³ .cube — {kind}: polish {:.2}, desaturate {:.2}, warm {:.2})",
            style("wrote").green(), lut.display(), p.polish, p.desaturate, p.warm
        );
        return Ok(());
    }

    let input = a.input.clone().context("an input image is required")?;

    // QUALITY-6 P2: a video/animation input → de-slop every frame + re-encode (weight-free surface pass).
    if is_video(&input) {
        anyhow::ensure!(!a.report, "--report analyses a still image — extract a frame first");
        return naturalize_video(&a, &input).await;
    }

    // QUALITY-7 P1: a directory + --report → a ranked folder scorecard (worst-AI first) + summary.
    if input.is_dir() && a.report {
        return folder_report(&a, &input);
    }

    // QUALITY-5 P3: batch — a directory input de-slops every image into the `--out` directory (same
    // filenames). Model-backed passes reload per image (best-effort convenience, not a resident server).
    if input.is_dir() {
        let exts = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"];
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&input)
            .with_context(|| format!("reading dir {}", input.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && p.extension().and_then(|x| x.to_str()).map(|e| exts.contains(&e.to_ascii_lowercase().as_str())).unwrap_or(false))
            .collect();
        files.sort();
        anyhow::ensure!(!files.is_empty(), "no images in {}", input.display());
        let out_dir = a.out.clone(); // None is only valid alongside --report (per-file report, no output)
        if let Some(d) = &out_dir {
            std::fs::create_dir_all(d).with_context(|| format!("creating out dir {}", d.display()))?;
            println!("naturalize batch: {} image(s) → {}", files.len(), d.display());
        }
        for f in &files {
            let name = f.file_name().context("image has no filename")?;
            let mut sub = a.clone();
            sub.input = Some(f.clone());
            sub.out = out_dir.as_ref().map(|d| d.join(name));
            println!("· {}", f.display());
            if let Err(e) = Box::pin(run(sub)).await {
                tracing::warn!(target: "plakat", "naturalize {}: {e} — continuing", f.display());
            }
        }
        return Ok(());
    }

    // QUALITY-6 P1: --report — analyze + print a de-slop scorecard, don't process.
    if a.report {
        let img = image::open(&input).with_context(|| format!("reading {}", input.display()))?.to_rgb8();
        let an = naturalize::analyze(&img);
        let medium = naturalize::refine::detect_medium_label(&input, Some(&a.device)).await;
        let recipe = naturalize::recommend(&an, medium.as_ref().map(|(m, _)| m.as_str()));
        let cmd = format!("plakat naturalize {} --out OUT.png {}", input.display(), recipe.join(" "));
        if a.json {
            let med = medium.as_ref().map(|(m, s)| format!("{{\"name\": {m:?}, \"score\": {s:.4}}}")).unwrap_or_else(|| "null".into());
            let flags: Vec<String> = recipe.iter().map(|f| format!("{f:?}")).collect();
            println!(
                "{{\n  \"input\": {:?},\n  \"ai_tell\": {:.4},\n  \"saturation\": {:.4},\n  \"smoothness_tell\": {:.4},\n  \"contrast\": {:.4},\n  \"medium\": {},\n  \"recommend\": [{}]\n}}",
                input.display().to_string(), an.ai_tell, an.saturation, an.smoothness_tell, an.contrast, med, flags.join(", ")
            );
        } else {
            let bar = |v: f32| { let n = (v * 20.0).round() as usize; format!("{}{}", "█".repeat(n.min(20)), "░".repeat(20 - n.min(20))) };
            println!("\n  {} — {}×{}", style("naturalize scorecard").bold(), img.width(), img.height());
            println!("  {}  AI-tell        {} {:.2}", if an.ai_tell > 0.5 { style("●").red() } else { style("●").green() }, bar(an.ai_tell), an.ai_tell);
            println!("     oversaturation {} {:.2}", bar(an.saturation), an.saturation);
            println!("     over-smoothness{} {:.2}", bar(an.smoothness_tell), an.smoothness_tell);
            println!("     contrast       {} {:.2}{}", bar(an.contrast), an.contrast, if an.contrast < 0.14 { "  (washed/muddy)" } else { "" });
            if let Some((m, s)) = &medium {
                println!("     medium         {} ({:.2})", style(m).cyan(), s);
            }
            println!("\n  {} {}", style("recommended:").green(), style(&cmd).dim());
            println!();
        }
        return Ok(());
    }

    let out_path = a.out.clone().context("--out is required")?;
    let base = resolve_base_params(&a, Some(&input))?;
    // content focus: blend the preset toward each active subject's de-AI profile, THEN apply explicit
    // per-param overrides (which always win).
    let focuses: Vec<(naturalize::Focus, f32)> = [
        (naturalize::Focus::People, a.people),
        (naturalize::Focus::Sky, a.sky),
        (naturalize::Focus::Vegetation, a.vegetation),
        (naturalize::Focus::Cityscape, a.cityscape),
        (naturalize::Focus::Landscape, a.landscape),
        (naturalize::Focus::Sea, a.sea),
        (naturalize::Focus::River, a.river),
        (naturalize::Focus::Mechanics, a.mechanics),
        (naturalize::Focus::Household, a.household),
        (naturalize::Focus::Animal, a.animal),
        (naturalize::Focus::Food, a.food),
        (naturalize::Focus::Interior, a.interior),
        (naturalize::Focus::Textile, a.textile),
        (naturalize::Focus::FoliageMacro, a.foliage_macro),
    ]
    .into_iter()
    .filter_map(|(f, n)| n.filter(|v| *v > 0.0).map(|v| (f, v)))
    .collect();
    let mut p: Params = naturalize::blend_focus(base, &focuses);
    if let Some(v) = a.grain {
        p.grain = v;
    }
    if let Some(v) = a.aberration {
        p.aberration = v;
    }
    if let Some(v) = a.vignette {
        p.vignette = v;
    }
    if let Some(v) = a.bloom {
        p.bloom = v;
    }
    if let Some(v) = a.desaturate {
        p.desaturate = v;
    }
    if let Some(v) = a.warm {
        p.warm = v;
    }
    if let Some(v) = a.defocus {
        p.defocus = v;
    }
    if let Some(v) = a.polish {
        p.polish = v;
    }
    if let Some(v) = a.micro {
        p.micro = v;
    }

    let tmp = tempfile::tempdir().context("temp dir for naturalize refine")?;
    let mut art_style = resolve_style(a.style.as_deref(), a.medium.as_deref());

    // Auto medium-detection (RFC QUALITY-4 P2 / QUALITY-5 P1): CLIP zero-shot the source medium so a
    // re-paint holds it (no photoreal drift) AND so auto-paper (below) can fire. Runs when a model
    // correction is requested, or when `--auto-medium` opts a weight-free run in — never when the style is
    // already named.
    let wants_model = a.repair.unwrap_or(0.0) > 0.0 || a.geometry.unwrap_or(0.0) > 0.0 || a.anatomy.unwrap_or(0.0) > 0.0;
    if art_style.is_none() && (wants_model || a.auto_medium) {
        if let Some(m) = naturalize::refine::detect_medium(&input, Some(&a.device)).await {
            println!("  {} auto-detected medium → {m}", style("de-slop").cyan());
            art_style = Some(m);
        }
    }

    // Auto-paper (QUALITY-5 P1): watercolor/gouache/ink-wash art → apply --paper at the recommended 0.6 by
    // default (unless the user set --paper explicitly, incl. `--paper 0` to disable).
    let paper_amt = a.paper.or_else(|| {
        art_style.as_deref().filter(|s| is_wet_media(s)).map(|_| 0.6)
    });

    // Brush strokes (naturalize media pass): synthesize human-like, gradient-aligned strokes for a
    // painting medium — fires when `--medium`/`--style` names one, or `--auto-medium` detected it.
    // Opt-in: no painting medium → no brush. Strength 0.6 by default; `--brush-strength 0` disables.
    // Skipped under `--repaint` — the model already painted the strokes (they're alternatives).
    let brush = (!a.repaint)
        .then(|| {
            art_style
                .as_deref()
                .and_then(naturalize::brush::Medium::detect)
                .map(|m| (m, a.brush_strength.unwrap_or(0.6)))
                .filter(|(_, s)| *s > 0.0)
        })
        .flatten();

    // 0. Face-protected repair (model-backed, art-safe) runs FIRST when requested: protect the faces,
    //    gently repaint the rest IN-STYLE to attempt broken limbs — the character-preserving alternative
    //    to whole-image --geometry on figure art.
    let mut current_input = input.clone();
    if let Some(n) = a.repair.filter(|v| *v > 0.0) {
        let strength = (0.3 * n).clamp(0.12, 0.6);
        let scope = naturalize::refine::RepairScope::parse(&a.repair_scope)
            .with_context(|| format!("unknown --repair-scope `{}` (figures|non-face|full)", a.repair_scope))?;
        let repaired = tmp.path().join("repaired.png");
        match naturalize::refine::repair_protected(&input, &repaired, strength, art_style.as_deref(), scope, &a.model, Some(&a.device), a.refine_steps, tmp.path()).await {
            Ok(true) => {
                println!("  {} face-protected repair (scope {:?}, strength {strength:.2}{})", style("de-slop").green(), scope, art_style.as_deref().map(|s| format!(", style: {s}")).unwrap_or_default());
                current_input = repaired;
            }
            Ok(false) => println!("  {} repair skipped (no faces / no detector) — try --geometry", style("de-slop").yellow()),
            Err(e) => tracing::warn!(target: "plakat", "naturalize --repair: {e}"),
        }
    }

    // 1. Corrective refine (model-backed): fix structure (geometry / anatomy) via whole-image img2img.
    //    (Character-destructive on cohesive art — prefer --repair there; fine for photoreal / non-figure.)
    let corrective = naturalize::refine::Corrective {
        geometry: a.geometry.unwrap_or(0.0),
        anatomy: a.anatomy.unwrap_or(0.0),
        no_twins: a.no_twins.unwrap_or(0.0),
    };
    current_input = if corrective.any() {
        let refined = tmp.path().join("refined.png");
        naturalize::refine::refine(&current_input, &refined, &corrective, &a.model, Some(&a.device), a.refine_steps, tmp.path()).await?;
        refined
    } else {
        current_input
    };

    // 2. De-clutter runs AFTER the geometry fix — the geometry img2img REGENERATES the scene (and would
    //    re-hallucinate removed wires), so clutter removal must be the LAST model step. Remove named
    //    compositional slop (floating wires, phantom rails) that img2img can't fix in place.
    if let Some(spec) = a.declutter.as_deref() {
        let targets: Vec<&str> = spec.split(',').map(str::trim).filter(|t| !t.is_empty()).collect();
        if !targets.is_empty() {
            let device = crate::api::device(&a.device).context("resolving device for --declutter")?;
            for (i, q) in targets.iter().enumerate() {
                let out = tmp.path().join(format!("declutter_{i}.png"));
                // Wire-like queries → weight-free sky-gated wire detector + inpaint (OWL-ViT is blind to
                // thin lines). Everything else → OWL-ViT open-vocab detection + inpaint.
                if is_wire_query(q) {
                    let img = image::open(&current_input).with_context(|| format!("reading {}", current_input.display()))?.to_rgb8();
                    let mask = naturalize::wire_mask(&img, 0.6);
                    let coverage = mask.pixels().filter(|p| p.0[0] > 127).count();
                    if coverage == 0 {
                        println!("  {} no thin-line structure detected for '{q}' — skipped", style("de-slop").yellow());
                        continue;
                    }
                    match crate::cli::remove::inpaint_masked(&current_input, &mask, &out, "sdxl-inpaint", &device, a.refine_steps).await {
                        Ok(()) => {
                            println!("  {} removed thin-line clutter ('{q}', {coverage}px)", style("de-slop").green());
                            current_input = out;
                        }
                        Err(e) => tracing::warn!(target: "plakat", "naturalize --declutter '{q}' (wire): {e}"),
                    }
                } else {
                    match crate::cli::remove::declutter_one(&current_input, q, &out, "sdxl-inpaint", &device, a.refine_steps).await {
                        Ok(true) => {
                            println!("  {} decluttered '{q}'", style("de-slop").green());
                            current_input = out;
                        }
                        Ok(false) => println!("  {} '{q}' not found — skipped", style("de-slop").yellow()),
                        Err(e) => tracing::warn!(target: "plakat", "naturalize --declutter '{q}': {e}"),
                    }
                }
            }
        }
    }

    // Painterly repaint (6.27): the model paints the image IN the medium — strokes/washes + form
    // together (real watercolor, oil impasto, …), the generation-side complement to the weight-free
    // `--medium` pass. Runs last in the model stage (over the cleaned image).
    let mut repainted_ok = false;
    if a.repaint {
        let repainted = tmp.path().join("repainted.png");
        // Condition the repaint on the input's own embedded prompt (if any) so it keeps the subjects.
        let scene_prompt = scene_prompt_from_png(&input);
        // img2img is UNet-only — fall back to SDXL when the model is a transformer family (unless the user
        // named --repaint-model), so `--model sd35 --repaint` paints instead of erroring.
        let rmodel: String = match a.repaint_model.as_deref() {
            Some(m) => m.to_string(),
            None if img2img_capable(&a.model) => a.model.clone(),
            None => {
                println!("  {} repaint: '{}' can't run img2img — using sdxl (set --repaint-model to override)", style("de-slop").yellow(), a.model);
                "sdxl".to_string()
            }
        };
        let rmodel = rmodel.as_str();
        let strength = a.repaint_strength.unwrap_or(0.38);
        let style_note = art_style.as_deref().map(|s| format!(", style: {s}")).unwrap_or_default();
        let mut done = false;
        match naturalize::refine::repaint(&current_input, &repainted, art_style.as_deref(), strength, rmodel, a.repaint_lora.as_deref(), Some(&a.device), a.refine_steps, scene_prompt.as_deref()).await {
            Ok(()) => {
                println!("  {} painterly repaint (strength {strength:.2}{style_note})", style("de-slop").green());
                current_input = repainted;
                done = true;
            }
            Err(e) => {
                // A bad `--repaint-lora` (missing / unresolvable) is the usual cause. Rather than silently
                // fall all the way back to the un-repainted baseline (which then only gets the analog grain
                // on top → looks like "nothing happened"), retry WITHOUT the LoRA so the base model still
                // paints the medium — and say so LOUDLY on stdout, not just a hidden log line.
                if a.repaint_lora.is_some() {
                    println!("  {} --repaint-lora `{}` failed to load ({e}) — retrying repaint without it",
                        style("de-slop").yellow(), a.repaint_lora.as_deref().unwrap_or(""));
                    match naturalize::refine::repaint(&current_input, &repainted, art_style.as_deref(), strength, rmodel, None, Some(&a.device), a.refine_steps, scene_prompt.as_deref()).await {
                        Ok(()) => {
                            println!("  {} painterly repaint (strength {strength:.2}{style_note}, no LoRA)", style("de-slop").green());
                            current_input = repainted;
                            done = true;
                        }
                        Err(e2) => tracing::warn!(target: "plakat", "naturalize --repaint (no-lora retry): {e2}"),
                    }
                } else {
                    tracing::warn!(target: "plakat", "naturalize --repaint: {e}");
                }
            }
        }
        if !done {
            println!("  {} painterly repaint SKIPPED (model/img2img failed) — the analog pass is running on the \
                      un-repainted image, so expect the baseline look. Check the model download / --device.",
                style("de-slop").yellow());
        }
        repainted_ok = done;
    }

    let src_for_analog = current_input;

    let mut img = image::open(&src_for_analog).with_context(|| format!("reading {}", src_for_analog.display()))?.to_rgb8();
    let ai_before = naturalize::ai_tell_score(&img);

    // A successful `--repaint` is the TERMINAL media render: the model already painted the medium
    // (real washes / strokes), so slathering the weight-free analog pass (grain/bloom/aberration) AND
    // auto-paper on top only double-textures it — visible speckle/pebble over a finished watercolor.
    // Skip the post-pass after a good repaint UNLESS the user explicitly opted a weight-free knob in
    // (then they clearly want it stacked). Auto-paper (fired implicitly by the medium) is NOT an opt-in.
    let explicit_post = a.preset.is_some()
        || a.grain.is_some() || a.aberration.is_some() || a.vignette.is_some() || a.bloom.is_some()
        || a.desaturate.is_some() || a.warm.is_some() || a.defocus.is_some()
        || a.polish.is_some() || a.micro.is_some()
        || a.paper.is_some() || a.designature.is_some()
        || a.auto_regions || !a.regions.is_empty()
        || !focuses.is_empty() || a.brush_strength.is_some();
    let skip_post = repainted_ok && !explicit_post;
    if skip_post {
        println!("  {} repaint is terminal — skipping the analog/paper post-pass (the model already painted the medium; add --grain/--paper/etc. to stack it)", style("de-slop").cyan());
    }

    // ghost-signature removal (weight-free) before the analog pass.
    if let Some(cs) = a.designature.as_deref() {
        let corner = naturalize::Corner::parse(cs).with_context(|| format!("unknown corner `{cs}` (br|bl|tr|tl)"))?;
        img = naturalize::designature(&img, corner, a.designature_strength);
    }
    // QUALITY-6 P3: per-region focuses (manual --region + --auto-regions) → composite each region's own
    // de-slop with feathered seams. Falls through to the plain whole-frame apply when there are none.
    let regions = if skip_post { Vec::new() } else { build_regions(&a, &input, &img, &p).await? };
    let mut out = if skip_post {
        img.clone()
    } else if regions.is_empty() {
        naturalize::apply(&img, &p)
    } else {
        println!("  {} {} region focus(es)", style("de-slop").cyan(), regions.len());
        naturalize::apply_with_regions(&img, &p, &regions)
    };
    // Brush strokes before paper, so the paper tooth/granulation rides the finished strokes.
    if let Some((m, bs)) = brush {
        out = naturalize::brush::apply_brush(&out, m, bs, a.brush_scale.unwrap_or(1.0));
        println!("  {} {} brush strokes (strength {bs:.2})", style("de-slop").green(), m.label());
    }
    // Watercolor-paper / pigment authenticity (RFC QUALITY-4) — opt-in, for genuine watercolour/ink-wash
    // art (fixes the "simulated media" tell). Runs last so tooth/granulation ride the finished pixels.
    // Suppressed when the repaint is terminal (the model's own paper/granulation is already there).
    if !skip_post {
        if let Some(pv) = paper_amt.filter(|v| *v > 0.0) {
            out = naturalize::paper_texture(&out, pv);
            let how = if a.paper.is_some() { "" } else { " (auto: wet media)" };
            println!("  {} watercolor paper/pigment (amount {pv:.2}{how})", style("de-slop").green());
        }
    }
    let ai_after = naturalize::ai_tell_score(&out);

    // Etch bar (QUALITY-2 P2): if the input was plakat-etched, re-etch the output — re-embed L1 into the
    // new pixels + chain the original as `parent` — so `doctor --if-plakat` resolves it as a derivative.
    // Otherwise plain-save and (for non-etched images) carry any metadata sidecar/chunks forward.
    let mut reetched: Option<crate::etch::EtchId> = None;
    let mut fresh: Option<crate::etch::EtchId> = None;
    let mut carried = false;
    if !a.no_reetch {
        reetched = crate::etch::reetch(&input, out.as_raw(), out.width(), out.height(), &out_path).unwrap_or(None);
    }
    if reetched.is_none() {
        // No plakat parent to chain. If the user explicitly asked for `--etch`, freshly etch this output
        // (plakat produced *this* naturalized image) — same claim `generate --etch` makes. Otherwise
        // plain-save and carry any existing provenance forward.
        if !a.no_reetch && crate::etch::active().is_some() {
            fresh = crate::etch::fresh_etch(out.as_raw(), out.width(), out.height(), &out_path, None).ok();
        }
        if fresh.is_none() {
            out.save(&out_path).with_context(|| format!("writing {}", out_path.display()))?;
            if !a.no_reetch {
                carried = carry_provenance(&input, &out_path).unwrap_or(false);
            }
        }
    }

    let preset_label = a.preset.as_deref().unwrap_or("subtle");
    let focus_note = if focuses.is_empty() {
        String::new()
    } else {
        let names: Vec<String> = focuses.iter().map(|(f, n)| format!("{}:{n}", format!("{f:?}").to_ascii_lowercase())).collect();
        format!(" · focus {}", names.join(","))
    };
    println!("{} {}  (naturalize · {preset_label}{focus_note})", style("wrote").green(), out_path.display());
    let _ = ai_before;
    println!("  {} AI-tell {:.3} (0=human … 1=AI; a batch-ranking heuristic)", style("score").cyan(), ai_after);
    if let Some(id) = reetched {
        println!("  {} re-etched (fresh L1 in the new pixels, id {:016x}, source chained as parent) — `doctor --if-plakat` verifies it, provenance preserved.", style("etch").green(), id.0);
    } else if let Some(id) = fresh {
        println!("  {} etched (fresh L0+L1, id {:016x}, no parent — plakat produced this naturalized image) — `doctor --if-plakat` verifies it.", style("etch").green(), id.0);
    } else if carried {
        println!("  {} metadata carried forward (input not plakat-etched).", style("etch").cyan());
    }
    Ok(())
}

/// Carry plakat's L0 provenance from `src` to `dst`: copy the JSON sidecar and splice the PNG text chunks
/// (`parameters` / `etch`). Returns true if anything was carried. Best-effort.
fn carry_provenance(src: &Path, dst: &Path) -> Result<bool> {
    let mut any = false;
    // 1. JSON sidecar (the carrier `doctor --if-plakat` reads).
    let src_side = crate::imaging::io::sidecar_path(src);
    if src_side.exists() {
        let dst_side = crate::imaging::io::sidecar_path(dst);
        if src_side != dst_side {
            std::fs::copy(&src_side, &dst_side).with_context(|| format!("copying sidecar to {}", dst_side.display()))?;
        }
        any = true;
    }
    // 2. PNG text chunks (verbatim splice — CRCs stay valid). Only for PNG→PNG.
    if src.extension().and_then(|s| s.to_str()) == Some("png") && dst.extension().and_then(|s| s.to_str()) == Some("png") {
        if splice_png_text_chunks(src, dst).unwrap_or(false) {
            any = true;
        }
    }
    Ok(any)
}

const PNG_SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

/// Extract every `tEXt`/`zTXt`/`iTXt` chunk (raw: length+type+data+CRC) from PNG `bytes`.
pub(crate) fn extract_text_chunks(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    if bytes.len() < 8 || bytes[..8] != PNG_SIG {
        return out;
    }
    let mut i = 8usize;
    while i + 8 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize;
        let ty = &bytes[i + 4..i + 8];
        let end = i + 12 + len;
        if end > bytes.len() {
            break;
        }
        if matches!(ty, b"tEXt" | b"zTXt" | b"iTXt") {
            out.push(bytes[i..end].to_vec());
        }
        if ty == b"IEND" {
            break;
        }
        i = end;
    }
    out
}

/// Insert raw text `chunks` into the PNG at `path`, right after `IHDR` (verbatim, CRCs already valid).
pub(crate) fn inject_text_chunks(path: &Path, chunks: &[Vec<u8>]) -> Result<()> {
    if chunks.is_empty() {
        return Ok(());
    }
    let bytes = std::fs::read(path)?;
    if bytes.len() < 8 || bytes[..8] != PNG_SIG {
        return Ok(());
    }
    let ihdr_len = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let ihdr_end = 8 + 12 + ihdr_len;
    if ihdr_end > bytes.len() {
        return Ok(());
    }
    let mut out = Vec::with_capacity(bytes.len() + chunks.iter().map(|c| c.len()).sum::<usize>());
    out.extend_from_slice(&bytes[..ihdr_end]);
    for c in chunks {
        out.extend_from_slice(c);
    }
    out.extend_from_slice(&bytes[ihdr_end..]);
    std::fs::write(path, out).with_context(|| format!("re-writing {} with text chunks", path.display()))?;
    Ok(())
}

/// Copy the `tEXt`/`zTXt`/`iTXt` chunks from `src` into `dst`. Returns whether any were carried.
fn splice_png_text_chunks(src: &Path, dst: &Path) -> Result<bool> {
    let chunks = extract_text_chunks(&std::fs::read(src)?);
    if chunks.is_empty() {
        return Ok(false);
    }
    inject_text_chunks(dst, &chunks)?;
    Ok(true)
}

/// Apply the analog naturalize pass to `path` IN PLACE from a compact spec, preserving the PNG text
/// chunks (the L0 etch carrier; the JSON sidecar is a separate file and is left untouched). Used by
/// `generate --naturalize` and the scenario `naturalize:` field. Weight-free.
/// Async spec applier used by the scenario / `generate --naturalize` post-passes. Handles the
/// model-backed **`repaint=`** token (6.27 — the prose/scenario parity of `plakat naturalize --repaint`):
/// repaint the image in the named `medium=`, then — following the CLI "repaint is terminal" rule — stack
/// the weight-free analog/brush/paper pass ONLY if the spec explicitly names one. With no `repaint=` it
/// falls straight through to the sync [`apply_inplace`] (the pure weight-free + brush path). Metadata is
/// preserved across the repaint.
/// Whether a model alias can run the img2img (UNet) pipeline — SD1.5/2.1/SDXL yes; the transformer
/// families (SD3/3.5, Flux, PixArt, Stable Cascade, Sana) no. Used to pick the naturalize repaint model.
fn img2img_capable(model: &str) -> bool {
    let m = model.to_lowercase();
    !(m.contains("flux")
        || m.contains("sd3")
        || m.contains("sd35")
        || m.contains("pixart")
        || m.contains("cascade")
        || m.contains("wuerstchen")
        || m.contains("würstchen")
        || m.contains("sana"))
}

/// Batch naturalize post-pass for many outputs sharing ONE spec (the scenario naturalize field). When the
/// spec has a `repaint=`, the img2img model is loaded ONCE and reused across every image — instead of the
/// per-file `repaint_batch` which reloaded it each time. Weight-free specs fall through to the cheap
/// per-file path. Best-effort per image (a failure is reported, others continue).
pub async fn repaint_batch(paths: &[std::path::PathBuf], spec: &str, model: &str, device: Option<&str>, steps: usize, keep_original: bool) {
    let repaint_s = crate::naturalize::repaint_from_spec(spec).filter(|v| *v > 0.0);
    let repair_s = crate::naturalize::repair_from_spec(spec).filter(|v| *v > 0.0);
    // Nothing model-backed → the cheap per-file weight-free path.
    if repaint_s.is_none() && repair_s.is_none() {
        for f in paths {
            if let Err(e) = apply_inplace(f, spec, keep_original) {
                crate::ui::progress::println(&format!("  ! naturalize {}: {e}", f.display()));
            }
        }
        return;
    }
    // Resolve the img2img model (repaint AND repair need a UNet family; SDXL fallback for transformers),
    // style + LoRA, once for the batch.
    let medium = crate::naturalize::spec_token(spec, "medium");
    // `repaint-style="..."` is a full custom anchor (brushwork descriptors and all) that overrides the
    // medium's canned anchor — the lever for finer/coarser strokes without touching the medium keyword.
    let style = crate::naturalize::spec_token(spec, "repaint-style")
        .map(str::to_string)
        .or_else(|| resolve_style(None, medium));
    let lora = crate::naturalize::spec_token(spec, "repaint-lora");
    let rmodel = match crate::naturalize::spec_token(spec, "repaint-model") {
        Some(m) => m.to_string(),
        None if img2img_capable(model) => model.to_string(),
        None => {
            crate::ui::progress::println(&format!("  {} '{model}' can't run img2img — using sdxl for the model pass (set `repaint-model=` to override)", console::style("de-slop").yellow()));
            "sdxl".to_string()
        }
    };
    // Load the repaint pipeline ONCE (only when repaint is requested — repair loads its own per image).
    let repaint_pipe = if repaint_s.is_some() {
        match crate::naturalize::refine::load_repaint_pipeline(&rmodel, lora, device).await {
            Ok((pipeline, dev)) => {
                crate::ui::progress::println(&format!("  {} repaint model {rmodel} loaded once for {} image(s)", console::style("de-slop").green(), paths.len()));
                Some((pipeline, dev))
            }
            Err(e) => {
                crate::ui::progress::println(&format!("  ! naturalize repaint: {e:#}"));
                None
            }
        }
    } else {
        None
    };
    for f in paths {
        if let Err(e) = model_pass_one(f, spec, &rmodel, device, steps, repaint_s, repair_s, style.as_deref(), medium, repaint_pipe.as_ref(), keep_original).await {
            crate::ui::progress::println(&format!("  ! naturalize {}: {e:#}", f.display()));
        }
    }
}

/// One image of a batch model-pass: optional face-protected **repair** (fix anatomy first) → optional
/// **repaint** (stylize, shared pipeline) → optional weight-free/paper stack → save + preserve metadata.
#[allow(clippy::too_many_arguments)]
async fn model_pass_one(
    path: &Path,
    spec: &str,
    rmodel: &str,
    device: Option<&str>,
    steps: usize,
    repaint_s: Option<f32>,
    repair_s: Option<f32>,
    style: Option<&str>,
    medium: Option<&str>,
    repaint_pipe: Option<&(crate::pipelines::portrait::Pipeline, candle_core::Device)>,
    keep_original: bool,
) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let chunks = extract_text_chunks(&bytes);
    let tmp = tempfile::tempdir().context("temp dir for naturalize model pass")?;

    // 0. Repair (face-protected anatomy fix) FIRST — on the base render, before any stylize. Writes to a
    //    temp that becomes the input to the repaint (or the final image if there's no repaint).
    let mut src = path.to_path_buf();
    if let Some(rstrength) = repair_s {
        let repaired = tmp.path().join("repaired.png");
        let scope = crate::naturalize::refine::RepairScope::parse(
            crate::naturalize::spec_token(spec, "repair-scope").unwrap_or("figures"),
        )
        .unwrap_or(crate::naturalize::refine::RepairScope::Figures);
        match crate::naturalize::refine::repair_protected(&src, &repaired, rstrength, style, scope, rmodel, device, steps, tmp.path()).await {
            Ok(true) => {
                crate::ui::progress::println(&format!("  {} repaired anatomy → {} (scope {scope:?}, strength {rstrength:.2})", console::style("de-slop").green(), path.display()));
                src = repaired;
            }
            Ok(false) => crate::ui::progress::println(&format!("  {} repair: no faces detected in {} — skipped", console::style("de-slop").yellow(), path.display())),
            Err(e) => crate::ui::progress::println(&format!("  ! repair {}: {e:#}", path.display())),
        }
    }

    // 1. Repaint (stylize) — using the shared pipeline; conditioned on the scene prompt from the PNG.
    let repainted;
    let out_src: &Path = if let (Some(strength), Some((pipeline, dev))) = (repaint_s, repaint_pipe) {
        repainted = tmp.path().join("repaint.png");
        let scene_prompt = scene_prompt_from_png(path);
        crate::naturalize::refine::repaint_with_pipeline(pipeline, dev, &src, &repainted, style, strength, steps, scene_prompt.as_deref()).await?;
        crate::ui::progress::println(&format!(
            "  {} repainted → {} ({}{})",
            console::style("de-slop").green(),
            path.display(),
            medium.unwrap_or("painterly"),
            if scene_prompt.is_some() { ", scene-conditioned" } else { "" }
        ));
        &repainted
    } else {
        &src // repair-only (or nothing): the weight-free stack runs on the repaired/original image
    };

    let mut out = image::open(out_src).with_context(|| format!("reading {}", out_src.display()))?.to_rgb8();
    if crate::naturalize::spec_wants_weightfree(spec) {
        out = crate::naturalize::apply(&out, &crate::naturalize::from_spec(spec));
        if let Some(m) = crate::naturalize::brush::medium_from_spec(spec) {
            let bs = crate::naturalize::brush::brush_from_spec(spec).unwrap_or(0.6);
            if bs > 0.0 {
                out = crate::naturalize::brush::apply_brush(&out, m, bs, crate::naturalize::brush::scale_from_spec(spec).unwrap_or(1.0));
            }
        }
        if let Some(pv) = crate::naturalize::paper_from_spec(spec).filter(|v| *v > 0.0) {
            out = crate::naturalize::paper_texture(&out, pv);
        }
    }
    let dest = naturalized_dest(path, keep_original);
    out.save(&dest).with_context(|| format!("writing {}", dest.display()))?;
    inject_text_chunks(&dest, &chunks)?;
    if keep_original {
        crate::ui::progress::println(&format!("  {} kept pre-naturalize → naturalized to {}", console::style("de-slop").green(), dest.display()));
    }
    Ok(())
}

/// The POSITIVE generation prompt embedded in a plakat PNG (the A1111 `parameters` tEXt chunk), if any —
/// everything before the `Negative prompt:` line. Lets a repaint condition on the scene it's repainting.
fn scene_prompt_from_png(path: &Path) -> Option<String> {
    let chunk = crate::imaging::io::read_parameters_chunk(path).ok().flatten()?;
    let positive = chunk.split("\nNegative prompt:").next().unwrap_or(&chunk).trim();
    (!positive.is_empty()).then(|| positive.to_string())
}

/// Destination for a naturalize write: the same path (overwrite in place) unless `keep_original`, in which
/// case a `<stem>.natural.<ext>` sibling in the same directory — so the pre-naturalize render is preserved
/// next to the naturalized one (`keep-prenaturalize: true`).
pub(crate) fn naturalized_dest(path: &Path, keep_original: bool) -> std::path::PathBuf {
    if !keep_original {
        return path.to_path_buf();
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("png");
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    path.with_file_name(format!("{stem}.natural.{ext}"))
}

pub fn apply_inplace(path: &Path, spec: &str, keep_original: bool) -> Result<()> {
    let params = crate::naturalize::from_spec(spec);
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let chunks = extract_text_chunks(&bytes);
    let img = image::load_from_memory(&bytes).with_context(|| format!("decoding {}", path.display()))?.to_rgb8();
    let mut out = crate::naturalize::apply(&img, &params);
    // 6.27: brush strokes for a painting medium named in the spec (`medium=oil brush=0.7`) — before
    // paper, so tooth/granulation ride the strokes. Syncs the CLI feature to prose/scenario specs.
    if let Some(m) = crate::naturalize::brush::medium_from_spec(spec) {
        let bs = crate::naturalize::brush::brush_from_spec(spec).unwrap_or(0.6);
        if bs > 0.0 {
            out = crate::naturalize::brush::apply_brush(&out, m, bs, crate::naturalize::brush::scale_from_spec(spec).unwrap_or(1.0));
        }
    }
    // QUALITY-5 P3: spec parity for --paper (`generate --naturalize "... paper=0.6"`, scenario field).
    if let Some(pv) = crate::naturalize::paper_from_spec(spec).filter(|v| *v > 0.0) {
        out = crate::naturalize::paper_texture(&out, pv);
    }
    let dest = naturalized_dest(path, keep_original);
    out.save(&dest).with_context(|| format!("writing {}", dest.display()))?;
    inject_text_chunks(&dest, &chunks)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::naturalized_dest;
    use std::path::Path;

    #[test]
    fn naturalized_dest_in_place_vs_sibling() {
        let p = Path::new("/out/run/scene/plakat-sd3-denoise-24.png");
        // keep_original = false → overwrite in place.
        assert_eq!(naturalized_dest(p, false), p.to_path_buf());
        // keep_original = true → `<stem>.natural.png` sibling in the SAME directory, original untouched.
        assert_eq!(
            naturalized_dest(p, true),
            Path::new("/out/run/scene/plakat-sd3-denoise-24.natural.png").to_path_buf()
        );
        // Preserves a non-png extension.
        assert_eq!(
            naturalized_dest(Path::new("a/b.webp"), true),
            Path::new("a/b.natural.webp").to_path_buf()
        );
    }
}
