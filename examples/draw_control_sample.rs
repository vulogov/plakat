//! Procedurally draws a sample depth map for the CONTROLNET tutorial.
//!
//! The output is a grayscale image where pixel brightness encodes depth
//! (white = near, black = far) — the convention Depth-Anything-V2 and
//! every ControlNet-Depth checkpoint expect on the input.
//!
//! Composition:
//! * Black far-distance horizon.
//! * A soft vertical gradient on the ground (brighter toward the
//!   bottom = closer to the camera).
//! * A central foreground "subject" disc, fully white.
//! * Two smaller mid-distance silhouettes (left and right) at
//!   intermediate brightness.
//!
//! Outputs:
//!   examples/tutorials/CONTROL/inputs/scene_depth.png
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example draw_control_sample
//! ```

use anyhow::{Context, Result};
use image::{GrayImage, Luma};
use std::path::Path;

const W: u32 = 512;
const H: u32 = 512;
const HORIZON_Y: f32 = 220.0;

fn main() -> Result<()> {
    let root = Path::new("examples/tutorials/CONTROL/inputs");
    std::fs::create_dir_all(root)
        .with_context(|| format!("creating {}", root.display()))?;

    let path = root.join("scene_depth.png");
    draw_depth(&path)?;
    eprintln!("✓ wrote {}", path.display());
    Ok(())
}

fn draw_depth(path: &Path) -> Result<()> {
    let mut img = GrayImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let y_f = y as f32;
            // Sky / far: dark.
            // Ground: brightness ramps up as y → H (closer).
            let v = if y_f < HORIZON_Y {
                // Sky band — almost black, with a tiny linear gradient so it
                // isn't pitch black at the horizon.
                lerp(0.0, 0.05, y_f / HORIZON_Y)
            } else {
                // Ground band — depth increases (brighter) toward the
                // bottom of the frame.
                lerp(0.20, 0.55, (y_f - HORIZON_Y) / (H as f32 - HORIZON_Y))
            };
            img.put_pixel(x, y, Luma([clamp_u8(v * 255.0)]));
        }
    }

    // Mid-distance bumps (medium grey) to suggest small hills.
    fill_disc(&mut img, 160.0, 235.0, 28.0, 0.30);
    fill_disc(&mut img, 360.0, 240.0, 32.0, 0.32);

    // Central foreground subject — bright disc on the ground.
    fill_disc(&mut img, 256.0, 360.0, 75.0, 0.95);

    img.save(path)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn clamp_u8(x: f32) -> u8 {
    x.clamp(0.0, 255.0).round() as u8
}

fn fill_disc(img: &mut GrayImage, cx: f32, cy: f32, r: f32, depth_value: f32) {
    let r2 = r * r;
    let target = clamp_u8(depth_value * 255.0);
    for y in 0..img.height() {
        for x in 0..img.width() {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let d2 = dx * dx + dy * dy;
            if d2 < r2 {
                let edge = (r - d2.sqrt()).clamp(0.0, 1.5) / 1.5;
                let existing = img.get_pixel(x, y).0[0] as f32;
                let blended = lerp(existing, target as f32, edge);
                img.put_pixel(x, y, Luma([clamp_u8(blended)]));
            }
        }
    }
}
