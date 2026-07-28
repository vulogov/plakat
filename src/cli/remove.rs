//! `plakat remove` — erase an object and fill the hole seamlessly.
//!
//! A one-shot wrapper over the existing selection + inpaint stack:
//!
//! 1. Build a selection mask over the object — SAM `--point X,Y` (repeatable, `:bg` to carve),
//!    `--depth-band LO,HI`, and/or a `--box X0,Y0,X1,Y1` rectangle. Multiple sources intersect.
//! 2. Grow + feather the mask so the inpaint has a margin and a soft seam.
//! 3. Hand the original image + mask to the inpaint flow (`img2img --mask`, strength 1.0) with a
//!    background-continuation prompt — the masked (white) region is regenerated, the rest is kept.
//!
//! Text targeting (`--what "the trash can"`) is wired in Phase 3 (OWL-ViT); until then, select with
//! `--point` / `--box` / `--depth-band`.

use anyhow::{Context, Result};
use candle_core::Device;
use clap::Args as ClapArgs;
use image::{GrayImage, ImageBuffer, Luma};
use std::path::PathBuf;

use crate::cli::img2img::Img2ImgArgs;
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::scheduler::SchedulerKind;

#[derive(ClapArgs, Debug)]
pub struct RemoveArgs {
    /// Path to the source image. Any format the `image` crate reads.
    pub input: PathBuf,

    /// Prompt point over the object, repeatable. `X,Y` selects (foreground); append `:bg` to
    /// exclude. Normalised (0–1) unless any value exceeds 1 (then pixels). Same grammar as
    /// `plakat segment --point`.
    #[arg(help_heading = "Selection", long = "point", value_name = "X,Y[:bg]")]
    pub points: Vec<String>,

    /// Rectangular selection `X0,Y0,X1,Y1` (top-left, bottom-right). Normalised (0–1) unless any
    /// value exceeds 1 (then pixels). Intersects with `--point` / `--depth-band` when combined.
    #[arg(help_heading = "Selection", long = "box", value_name = "X0,Y0,X1,Y1")]
    pub bbox: Option<String>,

    /// Select by DEPTH band `LO,HI` (0–1, near→far via Depth-Anything-V2). Same as
    /// `plakat segment --depth-band`.
    #[arg(help_heading = "Selection", long = "depth-band", value_name = "LO,HI")]
    pub depth_band: Option<String>,

    /// Open-vocabulary text target — name the object (OWL-ViT detects it, SAM refines the mask).
    #[arg(help_heading = "Selection", long = "what", value_name = "TEXT")]
    pub what: Option<String>,

    /// With `--what`, use the raw detection rectangle instead of SAM-refining it to the object outline.
    #[arg(help_heading = "Selection", long = "box-only", default_value_t = false)]
    pub box_only: bool,

    /// Prompt for what should fill the hole. Empty = a plausible background continuation. Describing
    /// the surrounding scene (e.g. "cobblestone street") improves the fill.
    #[arg(help_heading = "Prompt & text", long, default_value = "")]
    pub prompt: String,

    /// Negative prompt — discourage the removed object reappearing.
    #[arg(help_heading = "Prompt & text", long, default_value = "")]
    pub negative: String,

    /// Grow the mask outward by this many pixels before inpainting — gives the fill a margin so the
    /// object's edge/shadow doesn't survive.
    #[arg(help_heading = "Selection", long, default_value_t = 8, value_name = "PX")]
    pub grow: u32,

    /// Feather radius (px) on the mask edge — softens the inpaint↔preserve seam.
    #[arg(help_heading = "Selection", long = "mask-feather", default_value_t = 8, value_name = "PX")]
    pub feather: u32,

    /// Inpaint model. Defaults to `sdxl-inpaint`; any `--mask` consumer works (SD 1.5/SDXL inpaint
    /// UNets, `flux-fill-dev`, `sana`, or a vanilla UNet for RePaint-style masked img2img).
    #[arg(help_heading = "Model & sampler", long, default_value = "sdxl-inpaint")]
    pub model: String,

    /// Denoising steps.
    #[arg(help_heading = "Model & sampler", long, default_value_t = 28)]
    pub steps: usize,

    /// Classifier-free guidance scale.
    #[arg(help_heading = "Model & sampler", long, default_value_t = 7.5)]
    pub guidance: f64,

    /// Base seed.
    #[arg(help_heading = "Model & sampler", long)]
    pub seed: Option<u64>,

    /// Scheduler. `default` follows the model's preferred scheduler.
    #[arg(help_heading = "Model & sampler", long, default_value = "default")]
    pub scheduler: SchedulerKind,

    /// Number of variations to generate. Each gets a fresh seed.
    #[arg(help_heading = "Size & output", long, short = 'n', default_value_t = 1)]
    pub count: u32,

    /// Output directory.
    #[arg(help_heading = "Size & output", long, default_value = "./out")]
    pub out: PathBuf,

    /// `--import <album>` / `--import-move`: land the result in a photo album.
    #[command(flatten)]
    pub import: crate::cli::import::ImportArgs,
}

pub async fn run(args: RemoveArgs, device: Device) -> Result<()> {
    let (w, h) = image::image_dimensions(&args.input)
        .with_context(|| format!("reading dimensions of {}", args.input.display()))?;

    // ---- build the object mask from whichever selection sources were given ----
    let point_mask = if args.points.is_empty() {
        None
    } else {
        let points = crate::cli::segment::parse_points(&args.points)?;
        Some(crate::pipelines::sam::build_selection_mask(&args.input, &points, &device).await?)
    };
    let box_mask = match args.bbox.as_deref() {
        Some(s) => Some(box_to_mask(s, w, h)?),
        None => None,
    };
    // `--what`: OWL-ViT open-vocabulary detection → SAM-refined object mask (or the raw rectangle
    // with `--box-only`).
    let what_mask = match args.what.as_deref() {
        Some(query) => Some(detect_object_mask(&args.input, query, &device, w, h, !args.box_only).await?),
        None => None,
    };
    let depth_mask = match args.depth_band.as_deref() {
        Some(s) => {
            let (lo, hi) = crate::cli::segment::parse_band(s)?;
            let depth = crate::pipelines::depth::DepthPipeline::load(device.clone())
                .await?
                .depth_map(&args.input, w, h)?;
            Some(crate::pipelines::sam::depth_band_to_mask(&depth, w, h, lo, hi))
        }
        None => None,
    };

    let mut mask = None;
    for m in [point_mask, box_mask, depth_mask, what_mask].into_iter().flatten() {
        mask = Some(match mask {
            None => m,
            Some(prev) => crate::pipelines::sam::intersect_masks(&prev, &m),
        });
    }
    let mask = mask.context(
        "no selection: pass --what \"<object>\", --point X,Y, --box X0,Y0,X1,Y1, and/or --depth-band LO,HI",
    )?;

    // Grow + feather, then write the mask to a tempdir the inpaint pass reads by path.
    let tmp_dir = tempfile::Builder::new().prefix("plakat-remove-").tempdir()?;
    let mask_path = tmp_dir.path().join("mask.png");
    crate::pipelines::sam::finish_mask(mask, false, args.grow, args.feather, &mask_path)?;

    crate::ui::progress::println(&format!(
        "Remove: {}×{} object mask (grow={} feather={}) → inpaint fill",
        w, h, args.grow, args.feather
    ));

    // Hand the ORIGINAL image + object mask to the inpaint flow (white = fill, black = preserve).
    let img2img_args = Img2ImgArgs {
        input: args.input.clone(),
        prompt: args.prompt,
        negative: args.negative,
        mask: Some(mask_path.clone()),
        mask_feather: 0, // already feathered on the GrayImage above
        mask_invert: false,
        strength: Some(1.0),
        model: args.model,
        size: None,
        count: args.count,
        steps: args.steps,
        guidance: args.guidance,
        decoder_guidance: 1.1,
        faithful: false,
        seed: args.seed,
        scheduler: args.scheduler,
        loras: Vec::<LoraSpec>::new(),
        lora_scale: 1.0,
        out: args.out,
        import: Default::default(),
        control: None,
        control_image: None,
        control_from: None,
        control_strength: 1.0,
        control_start: 0.0,
        control_end: 1.0,
        control_specs: Vec::new(),
        artefacts: Vec::new(),
        artefact_library: None,
        artefact_blend: false,
        artefact_blend_strength: 0.0,
        smart_zones: false,
        wildcard_dir: None,
        tiled: false,
        tile_size: 1024,
        tile_stride: 768,
        grid: false,
        grid_cols: None,
        grid_padding: 0,
        kontext_bucket: false,
        negative_preset: None,
        look: None,
        genre: None,
        offline: false,
        aspect: None,
        base: 1024,
    };
    crate::cli::img2img::run(img2img_args, device).await
}

/// OWL-ViT text → object mask, shared by `remove --what` and `replace-bg --keep`. Detects the
/// best-scoring box for `query`, then (when `refine`) tightens it to the object outline with SAM:
/// prompt SAM at the box center, intersect with the box rectangle to clip over-selection, and fall
/// back to the plain rectangle if SAM under-selects (< 2% of the image).
pub(crate) async fn detect_object_mask(
    image: &std::path::Path,
    query: &str,
    device: &Device,
    w: u32,
    h: u32,
    refine: bool,
) -> Result<GrayImage> {
    let spin = crate::ui::progress::spinner(&format!("Detecting \"{query}\" (OWL-ViT)"));
    let owl = crate::pipelines::owlvit::OwlViT::load_pretrained(device).await?;
    let det = owl.detect(image, query, 0.1)?.with_context(|| {
        format!("OWL-ViT found no \"{query}\" (try a plainer noun, or select with --point/--box)")
    })?;
    spin.finish_with_message(format!(
        "✓ detected \"{query}\" @ [{:.0},{:.0},{:.0},{:.0}] (score {:.2})",
        det.x0, det.y0, det.x1, det.y1, det.score
    ));
    let rect = rect_mask(det.x0, det.y0, det.x1, det.y1, w, h);
    if !refine {
        return Ok(rect);
    }
    // SAM refine: a foreground point at the box center, plus background points just OUTSIDE the box
    // edges (MobileSAM over-selects from a lone point — the bg points tell it the object doesn't
    // extend past its detected box). Then ∩ the box to clip anything that still leaks out.
    let pt = |x: f32, y: f32, fg: bool| crate::pipelines::sam::PointPrompt {
        x: (x.clamp(0.0, (w - 1) as f32)) as f64,
        y: (y.clamp(0.0, (h - 1) as f32)) as f64,
        foreground: fg,
    };
    let (cx, cy) = ((det.x0 + det.x1) / 2.0, (det.y0 + det.y1) / 2.0);
    let m = 6.0; // margin (px) outside the box for the background hints
    let prompts = vec![
        pt(cx, cy, true),
        pt(cx, det.y0 - m, false), // above
        pt(cx, det.y1 + m, false), // below
        pt(det.x0 - m, cy, false), // left
        pt(det.x1 + m, cy, false), // right
    ];
    let sam_mask = match crate::pipelines::sam::build_selection_mask(image, &prompts, device).await {
        Ok(m) => m,
        Err(_) => return Ok(rect), // SAM failed → the rectangle still works
    };
    let refined = crate::pipelines::sam::intersect_masks(&sam_mask, &rect);
    let white = refined.pixels().filter(|p| p.0[0] > 127).count();
    // Fall back only if SAM essentially collapsed (center missed the object).
    if (white as f32) < 0.005 * (w as f32 * h as f32) {
        Ok(rect)
    } else {
        Ok(refined)
    }
}

/// Build a white-filled rectangle mask from pixel corners (clamped to the image). Used by `--what`
/// (OWL-ViT detection returns pixel boxes).
fn rect_mask(x0: f32, y0: f32, x1: f32, y1: f32, w: u32, h: u32) -> GrayImage {
    let cx0 = (x0.round().max(0.0) as u32).min(w);
    let cy0 = (y0.round().max(0.0) as u32).min(h);
    let cx1 = (x1.round().max(0.0) as u32).min(w);
    let cy1 = (y1.round().max(0.0) as u32).min(h);
    let mut mask: GrayImage = ImageBuffer::from_pixel(w, h, Luma([0]));
    for y in cy0..cy1 {
        for x in cx0..cx1 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    mask
}

/// Parse `X0,Y0,X1,Y1` (normalised 0–1, or pixels if any value > 1) into a white-filled rectangle
/// mask (`GrayImage`, 255 inside the box, 0 outside) at the image's resolution.
pub(crate) fn box_to_mask(s: &str, w: u32, h: u32) -> Result<GrayImage> {
    let parts: Vec<f32> = s
        .split(',')
        .map(|p| p.trim().parse::<f32>())
        .collect::<Result<_, _>>()
        .map_err(|_| anyhow::anyhow!("bad --box '{s}': expected X0,Y0,X1,Y1 numbers"))?;
    if parts.len() != 4 {
        anyhow::bail!("bad --box '{s}': expected 4 comma-separated values X0,Y0,X1,Y1");
    }
    let normalised = parts.iter().all(|&v| v <= 1.0);
    let to_px = |v: f32, span: u32| -> u32 {
        let px = if normalised { v * span as f32 } else { v };
        px.round().clamp(0.0, span as f32) as u32
    };
    let x0 = to_px(parts[0], w).min(w.saturating_sub(1));
    let y0 = to_px(parts[1], h).min(h.saturating_sub(1));
    let x1 = to_px(parts[2], w).min(w);
    let y1 = to_px(parts[3], h).min(h);
    if x1 <= x0 || y1 <= y0 {
        anyhow::bail!("bad --box '{s}': X1>X0 and Y1>Y0 required (got px {x0},{y0},{x1},{y1})");
    }
    let mut mask: GrayImage = ImageBuffer::from_pixel(w, h, Luma([0]));
    for y in y0..y1 {
        for x in x0..x1 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_to_mask_normalised_fills_rect() {
        let m = box_to_mask("0.25,0.5,0.75,1.0", 100, 40).unwrap();
        assert_eq!(m.dimensions(), (100, 40));
        assert_eq!(m.get_pixel(50, 30).0[0], 255); // inside
        assert_eq!(m.get_pixel(10, 10).0[0], 0); // outside (left + above)
        assert_eq!(m.get_pixel(80, 30).0[0], 0); // outside (right)
    }

    #[test]
    fn box_to_mask_pixels_when_over_one() {
        let m = box_to_mask("10,10,30,30", 100, 100).unwrap();
        assert_eq!(m.get_pixel(20, 20).0[0], 255);
        assert_eq!(m.get_pixel(5, 5).0[0], 0);
    }

    #[test]
    fn box_to_mask_rejects_degenerate() {
        assert!(box_to_mask("0.5,0.5,0.5,0.5", 100, 100).is_err());
        assert!(box_to_mask("1,2,3", 100, 100).is_err());
    }
}
