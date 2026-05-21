//! Procedurally draws a sample input image and a matching mask
//! for the IMG2IMG tutorial.
//!
//! Outputs:
//!   examples/tutorials/IMG2IMG/inputs/landscape.png   — a simple
//!     landscape: sky gradient, hill silhouette, sun disc.
//!   examples/tutorials/IMG2IMG/inputs/sky_mask.png    — grayscale
//!     mask that covers the sky region only (white above the horizon,
//!     black below). Used to demonstrate inpaint mode.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example draw_img2img_sample
//! ```

use anyhow::{Context, Result};
use image::{GrayImage, Luma, Rgb, RgbImage};
use std::path::Path;

const W: u32 = 512;
const H: u32 = 512;
const HORIZON_Y: u32 = 320; // ~62% down — low horizon, lots of sky

fn main() -> Result<()> {
    let root = Path::new("examples/tutorials/IMG2IMG/inputs");
    std::fs::create_dir_all(root)
        .with_context(|| format!("creating {}", root.display()))?;

    draw_landscape(&root.join("landscape.png"))?;
    draw_sky_mask(&root.join("sky_mask.png"))?;

    eprintln!("✓ wrote landscape.png + sky_mask.png into {}", root.display());
    Ok(())
}

/// Sky gradient + hill silhouette + sun disc. Deliberately graphic
/// so img2img has obvious things to transform.
fn draw_landscape(path: &Path) -> Result<()> {
    let mut img = RgbImage::new(W, H);

    for y in 0..H {
        for x in 0..W {
            let pixel = if y >= horizon_at(x) {
                // Ground: green→darker-green gradient bottom.
                let depth = (y - HORIZON_Y) as f32 / (H - HORIZON_Y) as f32;
                let r = lerp(80.0, 30.0, depth) as u8;
                let g = lerp(140.0, 70.0, depth) as u8;
                let b = lerp(50.0, 30.0, depth) as u8;
                Rgb([r, g, b])
            } else {
                // Sky: cyan-blue at top → pale near horizon.
                let depth = y as f32 / HORIZON_Y as f32;
                let r = lerp(120.0, 230.0, depth) as u8;
                let g = lerp(170.0, 220.0, depth) as u8;
                let b = lerp(220.0, 210.0, depth) as u8;
                Rgb([r, g, b])
            };
            img.put_pixel(x, y, pixel);
        }
    }

    // Sun disc, upper-right.
    let (sun_cx, sun_cy, sun_r) = (380.0, 110.0, 36.0);
    for y in 0..H {
        for x in 0..W {
            let dx = x as f32 - sun_cx;
            let dy = y as f32 - sun_cy;
            let d2 = dx * dx + dy * dy;
            if d2 < sun_r * sun_r {
                let edge = sun_r - d2.sqrt();
                let alpha = (edge * 0.5).clamp(0.0, 1.0);
                blend(&mut img, x, y, Rgb([255, 230, 180]), alpha);
            }
        }
    }

    img.save(path)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Grayscale mask: white everywhere above the hill silhouette, black
/// below. The mask is a slightly softened version of the same hill
/// outline used for the landscape; combined with --mask-feather=8
/// at inpaint time, that gives a clean inpaint along the horizon
/// without a hard step.
fn draw_sky_mask(path: &Path) -> Result<()> {
    let mut img = GrayImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let v: u8 = if y < horizon_at(x) { 255 } else { 0 };
            img.put_pixel(x, y, Luma([v]));
        }
    }
    img.save(path)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Horizon line as a function of x — slightly undulating so the
/// landscape doesn't look like graph paper.
fn horizon_at(x: u32) -> u32 {
    let phase = (x as f32 / W as f32) * std::f32::consts::TAU * 1.5;
    let wobble = (phase.sin() * 12.0) + (phase * 2.0).cos() * 6.0;
    let y = HORIZON_Y as f32 + wobble;
    y.clamp(0.0, (H - 1) as f32) as u32
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn blend(img: &mut RgbImage, x: u32, y: u32, c: Rgb<u8>, alpha: f32) {
    if x >= img.width() || y >= img.height() {
        return;
    }
    let p = img.get_pixel_mut(x, y);
    p[0] = ((c[0] as f32 * alpha) + (p[0] as f32 * (1.0 - alpha))) as u8;
    p[1] = ((c[1] as f32 * alpha) + (p[1] as f32 * (1.0 - alpha))) as u8;
    p[2] = ((c[2] as f32 * alpha) + (p[2] as f32 * (1.0 - alpha))) as u8;
}
