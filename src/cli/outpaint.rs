//! `plakat outpaint` — extend an image past its borders.
//!
//! The command is a thin wrapper over the existing inpaint flow
//! (`plakat img2img --mask …`). Steps:
//!
//! 1. Read the input image and per-side padding (`--left`, `--right`,
//!    `--top`, `--bottom`, or `--expand` for all four).
//! 2. Snap padding so the final canvas matches the inpaint model's
//!    VAE / patch constraint (mult of 8 for SD, mult of 16 for Flux).
//! 3. Allocate a new canvas of the expanded size and replicate the
//!    input's edge pixels into the new region — gives the inpaint
//!    model a low-frequency hint at the seam instead of a hard cliff
//!    against pure gray/black, which often confuses the denoise.
//! 4. Build a single-channel mask: white where the new region sits,
//!    black where the original image went.
//! 5. Write both to a tempdir and hand off to `img2img::run` with
//!    `--mask`, the standard inpaint pipeline.
//!
//! Supports any model the inpaint flow does: SD 1.5 / SDXL inpaint
//! UNets (default `sdxl-inpaint`) and Flux.1-Fill-dev (`--model
//! flux-fill-dev`). RePaint-style outpaint on a non-inpaint UNet (e.g.
//! plain `sdxl`) also works — the inpaint flow handles both.

use anyhow::{Context, Result};
use candle_core::Device;
use clap::Args as ClapArgs;
use image::{imageops, GenericImageView, GrayImage, ImageBuffer, Rgb, RgbImage};
use std::path::PathBuf;

use crate::cli::img2img::Img2ImgArgs;
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::scheduler::SchedulerKind;

#[derive(ClapArgs, Debug)]
pub struct OutpaintArgs {
    /// Path to the source image. Any format the `image` crate reads.
    pub input: PathBuf,

    /// Text prompt describing what the expanded canvas should contain.
    /// The inpaint UNet sees the prompt at every denoise step, so this
    /// should describe the **whole** scene including the new region —
    /// e.g. "wide landscape, mountains and a river in the distance".
    #[arg(help_heading = "Prompt & text", long)]
    pub prompt: String,

    /// Pixels to extend on the left side. 0 = no expansion left.
    #[arg(long, default_value_t = 0, value_name = "PX")]
    pub left: u32,

    /// Pixels to extend on the right side.
    #[arg(long, default_value_t = 0, value_name = "PX")]
    pub right: u32,

    /// Pixels to extend on the top.
    #[arg(long, default_value_t = 0, value_name = "PX")]
    pub top: u32,

    /// Pixels to extend on the bottom.
    #[arg(long, default_value_t = 0, value_name = "PX")]
    pub bottom: u32,

    /// Shorthand: extend all four sides equally. Conflicts with the
    /// per-side flags.
    #[arg(
        long,
        conflicts_with_all = ["left", "right", "top", "bottom"],
        value_name = "PX",
    )]
    pub expand: Option<u32>,

    /// Negative prompt (things to discourage in the expansion).
    #[arg(help_heading = "Prompt & text", long, default_value = "")]
    pub negative: String,

    /// Inpaint model alias or HF repo id. Defaults to `sdxl-inpaint`.
    /// `flux-fill-dev` routes through the Flux pipeline; SD 1.5 /
    /// SDXL inpaint go through the SD path. Vanilla `sdxl` / `sd15`
    /// also work — those are RePaint-style masked img2img.
    #[arg(help_heading = "Model & sampler", long, default_value = "sdxl-inpaint")]
    pub model: String,

    /// Number of variations to generate. Each gets a fresh seed.
    #[arg(help_heading = "Size & output", long, short = 'n', default_value_t = 1)]
    pub count: u32,

    /// Denoising steps.
    #[arg(help_heading = "Model & sampler", long, default_value_t = 28)]
    pub steps: usize,

    /// Classifier-free guidance scale.
    #[arg(help_heading = "Model & sampler", long, default_value_t = 7.5)]
    pub guidance: f64,

    /// Base seed. Subsequent --count outputs use seed+1, seed+2, ...
    #[arg(help_heading = "Model & sampler", long)]
    pub seed: Option<u64>,

    /// Scheduler. `default` follows the model's preferred scheduler.
    #[arg(help_heading = "Model & sampler", long, default_value = "default")]
    pub scheduler: SchedulerKind,

    /// LoRA spec(s). Repeatable — same grammar as `plakat generate --lora`.
    #[arg(help_heading = "LoRA & embeddings", long = "lora")]
    pub loras: Vec<LoraSpec>,

    /// LoRA weight scale multiplier.
    #[arg(help_heading = "LoRA & embeddings", long, default_value_t = 1.0)]
    pub lora_scale: f32,

    /// **v0.25**: art-medium preset. See `plakat generate --look`
    /// for the full list. Composes the prompt + suggests sampler /
    /// steps / guidance, and auto-discovers a matching LoRA when
    /// `--loras` is empty.
    #[arg(help_heading = "Style & look", long = "look", value_name = "NAME")]
    pub look: Option<String>,

    /// **v0.25**: subject-domain preset (`anime`).
    #[arg(help_heading = "Style & look", long = "genre", value_name = "NAME")]
    pub genre: Option<String>,

    /// **v0.25**: skip remote LoRA discovery (use cache + local
    /// scan only).
    #[arg(help_heading = "Model & sampler", long, default_value_t = false)]
    pub offline: bool,

    /// Feather radius (pixels) on the mask edge. Softens the boundary
    /// between the preserved input and the inpainted expansion.
    #[arg(long = "mask-feather", default_value_t = 16, value_name = "PX")]
    pub mask_feather: u32,

    /// Output directory.
    #[arg(help_heading = "Size & output", long, default_value = "./out")]
    pub out: PathBuf,

    /// `--import <album>` / `--import-move`: land the outpainted image in a photo album.
    #[command(flatten)]
    pub import: crate::cli::import::ImportArgs,

    /// v0.18 phase 2: with `--count N > 1`, also write a single
    /// `plakat-inpaint-grid-<base-seed>.png` combining all N outputs
    /// in a near-square layout. Forwarded to the underlying
    /// `plakat img2img` pipeline (outpaint always runs through the
    /// inpaint dispatch).
    #[arg(help_heading = "Size & output", long = "grid", default_value_t = false)]
    pub grid: bool,

    /// v0.18 phase 2: column count for `--grid`. Default is
    /// `ceil(sqrt(count))`. Ignored when `--grid` is off.
    #[arg(help_heading = "Size & output", long = "grid-cols", value_name = "N")]
    pub grid_cols: Option<usize>,

    /// v0.18 phase 2: padding (px) between grid cells. Default 0.
    /// Ignored when `--grid` is off.
    #[arg(help_heading = "Size & output", long = "grid-padding", default_value_t = 0, value_name = "PX")]
    pub grid_padding: u32,
}

pub async fn run(args: OutpaintArgs, device: Device) -> Result<()> {
    let (left, right, top, bottom) = match args.expand {
        Some(n) => (n, n, n, n),
        None => (args.left, args.right, args.top, args.bottom),
    };
    if left == 0 && right == 0 && top == 0 && bottom == 0 {
        anyhow::bail!(
            "Outpaint needs at least one of --left, --right, --top, --bottom (or --expand) > 0."
        );
    }

    // Snap-multiple is determined by the inpaint model's VAE / patch
    // constraint. Flux needs 16; SD needs 8. We over-pad rather than
    // under-pad — the user asked for *at least* this much expansion.
    let snap = if args.model.to_lowercase().contains("flux") {
        16
    } else {
        8
    };
    let snap_up = |n: u32| -> u32 {
        if n == 0 {
            0
        } else {
            ((n + snap - 1) / snap) * snap
        }
    };
    let left = snap_up(left);
    let right = snap_up(right);
    let top = snap_up(top);
    let bottom = snap_up(bottom);

    // Load input.
    let input = image::open(&args.input)
        .with_context(|| format!("opening input {}", args.input.display()))?;
    let (in_w, in_h) = input.dimensions();
    let input_rgb = input.to_rgb8();

    let new_w = in_w + left + right;
    let new_h = in_h + top + bottom;
    // VAE constraint applies to the final canvas too. The padding-snap
    // above guarantees this when in_w / in_h are themselves snap-
    // aligned, but the original input may not be — bail loud rather
    // than silently truncate user content.
    if in_w % snap != 0 || in_h % snap != 0 {
        anyhow::bail!(
            "Outpaint: input image is {in_w}x{in_h}, not divisible by {snap} (the model's \
             VAE / patch constraint). Resize the input to a multiple of {snap} before \
             outpainting, or pass --expand with a value that brings both dims to a multiple."
        );
    }

    // Allocate canvas + replicate-fill border. Replicate is the
    // cheapest fill that gives the inpaint UNet a smooth low-frequency
    // hint at the seam; a hard cliff against gray/black often biases
    // the denoise toward "wall" or "edge" content.
    let canvas = build_replicate_canvas(&input_rgb, left, top, new_w, new_h);

    // Build mask: white (255) where the new region sits, black (0)
    // where the original image went. Img2img treats white as
    // "inpaint" by default — same convention SD inpaint UNets train
    // on.
    let mask = build_outpaint_mask(in_w, in_h, left, top, new_w, new_h);

    // Persist both to a tempdir so the img2img CLI can read them as
    // paths. Tempdir stays alive until end of `run` so the files
    // exist while img2img::run loads them.
    let tmp_dir =
        tempfile::Builder::new().prefix("plakat-outpaint-").tempdir()?;
    let canvas_path = tmp_dir.path().join("canvas.png");
    let mask_path = tmp_dir.path().join("mask.png");
    canvas
        .save(&canvas_path)
        .with_context(|| format!("writing outpaint canvas {}", canvas_path.display()))?;
    mask.save(&mask_path)
        .with_context(|| format!("writing outpaint mask {}", mask_path.display()))?;

    crate::ui::progress::println(&format!(
        "Outpaint canvas: {in_w}x{in_h} → {new_w}x{new_h} (left={left} right={right} \
         top={top} bottom={bottom}), snap={snap}"
    ));

    // Hand off to the inpaint flow.
    let img2img_args = Img2ImgArgs {
        input: canvas_path.clone(),
        prompt: args.prompt,
        negative: args.negative,
        mask: Some(mask_path.clone()),
        mask_feather: args.mask_feather,
        mask_invert: false,
        // Outpaint always wants full inpaint strength — the new
        // region has no original content to preserve. The img2img
        // flow defaults --strength to 1.0 when --mask is set, but
        // pin it explicitly so a future default change can't drift.
        strength: Some(1.0),
        model: args.model,
        size: None, // use the canvas's native dims
        count: args.count,
        steps: args.steps,
        guidance: args.guidance,
        decoder_guidance: 1.1,
        faithful: false,
        seed: args.seed,
        scheduler: args.scheduler,
        loras: args.loras,
        lora_scale: args.lora_scale,
        out: args.out,
        // Import happens once at the outpaint dispatch level — the nested inpaint pass must not
        // re-import, so leave its flag unset.
        import: Default::default(),
        // ControlNet / artefact compositing don't compose with the
        // simple outpaint wrapper in this phase — pass through empty.
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
        // v0.16 phase 5: outpaint's prompt is built from the user's
        // outpaint args — already concrete. No wildcard expansion
        // needed at this level.
        wildcard_dir: None,
        // v0.16 phase 10: outpaint runs full-canvas img2img (no
        // tiled). Outpaint's "stretch the canvas + inpaint the
        // new region" recipe doesn't compose with the per-tile
        // velocity blend — drop here, surface clearly if the
        // user wants tiled.
        tiled: false,
        tile_size: 1024,
        tile_stride: 768,
        // v0.18 phase 2: pass --grid through into the img2img layer
        // — the inpaint dispatch composes the grid on the
        // `plakat-inpaint-{seed}.png` files outpaint produces.
        grid: args.grid,
        grid_cols: args.grid_cols,
        grid_padding: args.grid_padding,
        // v0.18 phase 2b: outpaint never routes through Kontext.
        kontext_bucket: false,
        // v0.19: outpaint inherits the same default-negative model
        // as img2img inpaint; we don't surface --negative-preset
        // at the outpaint CLI yet (outpaint scope is narrow).
        negative_preset: None,
        // v0.25 phase 6: forward --look / --genre / --offline so
        // the img2img dispatch applies them. Override-only-if-
        // user-didn't-pass semantics + auto-LoRA discovery flow
        // identical to a direct `plakat img2img` invocation.
        look: args.look,
        genre: args.genre,
        offline: args.offline,
        // v0.18: outpaint controls dimension changes via per-
        // side --left/--right/--top/--bottom/--expand flags, so
        // --aspect isn't surfaced on the outpaint CLI. Pass None +
        // the existing 1024 base default through so the underlying
        // img2img resolver falls back to the input dims (the
        // standard outpaint behaviour).
        aspect: None,
        base: 1024,
    };
    crate::cli::img2img::run(img2img_args, device).await
}

/// Allocate a new RGB canvas of `(new_w, new_h)` and copy the input
/// at offset `(left, top)`. Border regions are filled by *replicating*
/// the nearest edge pixel from the input — gives a smooth low-frequency
/// continuation that the inpaint UNet can refine over instead of having
/// to invent content against a flat gray slab.
pub(crate) fn build_replicate_canvas(
    input: &RgbImage,
    left: u32,
    top: u32,
    new_w: u32,
    new_h: u32,
) -> RgbImage {
    let (in_w, in_h) = input.dimensions();
    // Start from the input as a centred sub-region; fill the rest by
    // sampling the nearest input pixel for each output pixel.
    let mut out: RgbImage = ImageBuffer::from_pixel(new_w, new_h, Rgb([128, 128, 128]));
    for y in 0..new_h {
        for x in 0..new_w {
            // Map (x, y) on the new canvas to (sx, sy) on the input,
            // clamping to the input's bounds so border pixels replicate.
            let sx = if x < left {
                0
            } else if x >= left + in_w {
                in_w - 1
            } else {
                x - left
            };
            let sy = if y < top {
                0
            } else if y >= top + in_h {
                in_h - 1
            } else {
                y - top
            };
            let p = *input.get_pixel(sx, sy);
            out.put_pixel(x, y, p);
        }
    }
    // Overlay the actual input on top of the replicated background so
    // the unchanged pixels are bit-exact (replicate sampling for
    // in-bounds output positions is a no-op, but copying explicitly
    // here makes the invariant easier to reason about).
    imageops::overlay(&mut out, input, left as i64, top as i64);
    out
}

/// Build an 8-bit grayscale mask of the outpaint region: 255 in the
/// expanded border, 0 over the preserved input area. Img2img's
/// `--mask` reads this as inpaint=255, preserve=0 — the SD inpaint
/// UNet's standard convention.
pub(crate) fn build_outpaint_mask(
    in_w: u32,
    in_h: u32,
    left: u32,
    top: u32,
    new_w: u32,
    new_h: u32,
) -> GrayImage {
    let mut mask: GrayImage = ImageBuffer::from_pixel(new_w, new_h, image::Luma([255]));
    // Punch a black rectangle over the preserved input — that's the
    // region img2img leaves untouched.
    for y in top..top + in_h {
        for x in left..left + in_w {
            mask.put_pixel(x, y, image::Luma([0]));
        }
    }
    let _ = (in_w, in_h); // for readability; bounds already encoded above
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_up_rounds_up_to_multiple() {
        // Reproduce the snap helper inline because it's a closure in `run`.
        let snap = 8u32;
        let snap_up = |n: u32| if n == 0 { 0 } else { ((n + snap - 1) / snap) * snap };
        assert_eq!(snap_up(0), 0);
        assert_eq!(snap_up(1), 8);
        assert_eq!(snap_up(8), 8);
        assert_eq!(snap_up(9), 16);
        assert_eq!(snap_up(127), 128);
    }

    #[test]
    fn mask_is_black_over_input_white_in_border() {
        // 100x100 input, expand 50 left, 0 elsewhere → mask 150x100,
        // with a 50-wide white strip on the left and the rest black.
        let m = build_outpaint_mask(100, 100, 50, 0, 150, 100);
        assert_eq!(m.dimensions(), (150, 100));
        // White strip (inpaint region).
        assert_eq!(m.get_pixel(0, 0).0[0], 255);
        assert_eq!(m.get_pixel(49, 50).0[0], 255);
        // Black over the preserved input.
        assert_eq!(m.get_pixel(50, 50).0[0], 0);
        assert_eq!(m.get_pixel(149, 99).0[0], 0);
    }

    #[test]
    fn canvas_preserves_input_pixels_at_offset() {
        let mut input: RgbImage = ImageBuffer::new(4, 4);
        input.put_pixel(0, 0, Rgb([10, 20, 30]));
        input.put_pixel(3, 3, Rgb([40, 50, 60]));
        let canvas = build_replicate_canvas(&input, 2, 1, 8, 6);
        assert_eq!(canvas.dimensions(), (8, 6));
        // Input lands at (left=2, top=1).
        assert_eq!(canvas.get_pixel(2, 1).0, [10, 20, 30]);
        assert_eq!(canvas.get_pixel(5, 4).0, [40, 50, 60]);
    }

    #[test]
    fn canvas_replicates_edge_in_border() {
        // Single-colour input → entire canvas should be that colour
        // (replicate fill of a constant is the same constant).
        let input: RgbImage = ImageBuffer::from_pixel(4, 4, Rgb([100, 100, 100]));
        let canvas = build_replicate_canvas(&input, 4, 4, 12, 12);
        for y in 0..12 {
            for x in 0..12 {
                assert_eq!(canvas.get_pixel(x, y).0, [100, 100, 100], "edge replicate at ({x}, {y})");
            }
        }
    }
}
