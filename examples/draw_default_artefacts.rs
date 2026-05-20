//! Procedural generator for plakat's bundled artefact library.
//!
//! Emits a small set of stylized silhouette PNGs (sun, moon, oak,
//! pine, cottage, cloud) into `assets/artefact_library/` along with a
//! matching `library.json`. The silhouettes are deliberately simple
//! and uniform in style — placeholder-quality, but with clean alpha
//! channels, true CC0 provenance (we drew them with pixel arithmetic),
//! and consistent visual language across the set.
//!
//! For production use, replace any of these with your own PNGs and
//! re-run `plakat artefact list` to confirm the new files are picked
//! up. The bundled set is intentionally minimal.
//!
//! Run:
//!
//! ```sh
//! cargo run --release --example draw_default_artefacts
//! ```

use std::path::Path;

use anyhow::{Context, Result};
use image::{Rgba, RgbaImage};

const W: u32 = 512;
const H: u32 = 512;

const TRANSPARENT: Rgba<u8> = Rgba([0, 0, 0, 0]);

fn main() -> Result<()> {
    let root = Path::new("assets/artefact_library");
    std::fs::create_dir_all(root.join("trees"))?;
    std::fs::create_dir_all(root.join("sky"))?;
    std::fs::create_dir_all(root.join("houses"))?;

    draw_sun(&root.join("sky/sun.png"))?;
    draw_moon(&root.join("sky/moon.png"))?;
    draw_cloud(&root.join("sky/cloud.png"))?;
    draw_oak(&root.join("trees/oak.png"))?;
    draw_pine(&root.join("trees/pine.png"))?;
    draw_cottage(&root.join("houses/cottage.png"))?;

    // Emit the library.json catalog last so it lines up with whatever
    // files this run produced.
    write_library_json(&root.join("library.json"))?;
    eprintln!("✓ wrote 6 artefact PNGs + library.json into {}", root.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers: shape primitives operating on an RgbaImage.
// ---------------------------------------------------------------------------

fn new_canvas() -> RgbaImage {
    RgbaImage::from_pixel(W, H, TRANSPARENT)
}

fn fill_circle(img: &mut RgbaImage, cx: f32, cy: f32, r: f32, color: Rgba<u8>) {
    let r2 = r * r;
    for y in 0..img.height() {
        for x in 0..img.width() {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let d2 = dx * dx + dy * dy;
            if d2 <= r2 {
                // Soft anti-aliased edge over the last ~1.5 px.
                let edge = r - d2.sqrt();
                let alpha = (edge * 1.5).clamp(0.0, 1.0);
                blend_pixel(img, x, y, color, alpha);
            }
        }
    }
}

fn fill_rect(img: &mut RgbaImage, x0: u32, y0: u32, x1: u32, y1: u32, color: Rgba<u8>) {
    for y in y0..y1.min(img.height()) {
        for x in x0..x1.min(img.width()) {
            blend_pixel(img, x, y, color, 1.0);
        }
    }
}

/// Triangle with base on the bottom: apex at (cx, top), base from
/// (cx-half_base, base_y) to (cx+half_base, base_y).
fn fill_triangle_down(
    img: &mut RgbaImage,
    cx: f32,
    top_y: f32,
    base_y: f32,
    half_base: f32,
    color: Rgba<u8>,
) {
    let height = base_y - top_y;
    let y_start = top_y.floor() as i32;
    let y_end = base_y.ceil() as i32;
    for y in y_start..=y_end {
        if y < 0 || y >= img.height() as i32 {
            continue;
        }
        let t = ((y as f32 - top_y) / height).clamp(0.0, 1.0);
        let half = half_base * t;
        let x_left = cx - half;
        let x_right = cx + half;
        let xs = x_left.floor() as i32;
        let xe = x_right.ceil() as i32;
        for x in xs..=xe {
            if x < 0 || x >= img.width() as i32 {
                continue;
            }
            blend_pixel(img, x as u32, y as u32, color, 1.0);
        }
    }
}

fn blend_pixel(img: &mut RgbaImage, x: u32, y: u32, color: Rgba<u8>, alpha_mul: f32) {
    let existing = img.get_pixel(x, y).0;
    let src_a = (color[3] as f32 / 255.0) * alpha_mul;
    if src_a <= 0.0 {
        return;
    }
    let inv = 1.0 - src_a;
    let r = (color[0] as f32 * src_a + existing[0] as f32 * inv).round() as u8;
    let g = (color[1] as f32 * src_a + existing[1] as f32 * inv).round() as u8;
    let b = (color[2] as f32 * src_a + existing[2] as f32 * inv).round() as u8;
    let a_out = (existing[3] as f32 * inv + 255.0 * src_a).round() as u8;
    img.put_pixel(x, y, Rgba([r, g, b, a_out]));
}

// ---------------------------------------------------------------------------
// Individual artefact drawings.
// ---------------------------------------------------------------------------

fn draw_sun(path: &Path) -> Result<()> {
    let mut img = new_canvas();
    let yellow = Rgba([248, 196, 64, 255]);
    // 8 outward rays before the disc so the disc covers their inner ends.
    let cx = W as f32 / 2.0;
    let cy = H as f32 / 2.0;
    let n_rays = 12;
    for i in 0..n_rays {
        let theta = (i as f32 / n_rays as f32) * std::f32::consts::TAU;
        let inner = 150.0;
        let outer = 230.0;
        let (sx, sy) = (cx + theta.cos() * inner, cy + theta.sin() * inner);
        let (ex, ey) = (cx + theta.cos() * outer, cy + theta.sin() * outer);
        draw_thick_line(&mut img, sx, sy, ex, ey, 18.0, yellow);
    }
    // Disc on top.
    fill_circle(&mut img, cx, cy, 130.0, yellow);
    img.save(path).with_context(|| format!("writing {}", path.display()))
}

fn draw_moon(path: &Path) -> Result<()> {
    let mut img = new_canvas();
    let white = Rgba([238, 232, 220, 255]);
    let cx = W as f32 / 2.0;
    let cy = H as f32 / 2.0;
    // Full disc, then "erase" by overdrawing with transparent disc
    // shifted right (crescent).
    fill_circle(&mut img, cx, cy, 180.0, white);
    // Cut the crescent by drawing transparent over the right-shifted bite.
    erase_circle(&mut img, cx + 65.0, cy - 20.0, 175.0);
    img.save(path).with_context(|| format!("writing {}", path.display()))
}

fn draw_cloud(path: &Path) -> Result<()> {
    let mut img = new_canvas();
    let white = Rgba([245, 245, 250, 250]);
    // Overlapping circles, centered roughly along the middle of canvas.
    let cy = H as f32 / 2.0;
    let centers = [
        (W as f32 * 0.25, cy + 30.0, 80.0),
        (W as f32 * 0.40, cy - 10.0, 110.0),
        (W as f32 * 0.55, cy - 40.0, 120.0),
        (W as f32 * 0.70, cy + 0.0, 100.0),
        (W as f32 * 0.80, cy + 35.0, 75.0),
    ];
    for (cx, cy_, r) in centers {
        fill_circle(&mut img, cx, cy_, r, white);
    }
    img.save(path).with_context(|| format!("writing {}", path.display()))
}

fn draw_oak(path: &Path) -> Result<()> {
    let mut img = new_canvas();
    let trunk = Rgba([90, 60, 40, 255]);
    let foliage = Rgba([60, 110, 70, 255]);

    let cx = W as f32 / 2.0;
    let trunk_w = 50.0;
    let trunk_top = H as f32 * 0.55;
    let trunk_bot = H as f32 * 0.95;
    fill_rect(
        &mut img,
        (cx - trunk_w / 2.0) as u32,
        trunk_top as u32,
        (cx + trunk_w / 2.0) as u32,
        trunk_bot as u32,
        trunk,
    );
    // Rounded foliage: three overlapping circles forming a leafy cap.
    fill_circle(&mut img, cx - 80.0, H as f32 * 0.40, 110.0, foliage);
    fill_circle(&mut img, cx + 80.0, H as f32 * 0.40, 110.0, foliage);
    fill_circle(&mut img, cx, H as f32 * 0.30, 130.0, foliage);
    img.save(path).with_context(|| format!("writing {}", path.display()))
}

fn draw_pine(path: &Path) -> Result<()> {
    let mut img = new_canvas();
    let trunk = Rgba([90, 60, 40, 255]);
    let foliage = Rgba([45, 95, 55, 255]);

    let cx = W as f32 / 2.0;
    let trunk_w = 40.0;
    let trunk_top = H as f32 * 0.85;
    let trunk_bot = H as f32 * 0.95;
    fill_rect(
        &mut img,
        (cx - trunk_w / 2.0) as u32,
        trunk_top as u32,
        (cx + trunk_w / 2.0) as u32,
        trunk_bot as u32,
        trunk,
    );
    // Three stacked triangles, narrower at the top.
    fill_triangle_down(&mut img, cx, H as f32 * 0.60, H as f32 * 0.85, 130.0, foliage);
    fill_triangle_down(&mut img, cx, H as f32 * 0.35, H as f32 * 0.65, 105.0, foliage);
    fill_triangle_down(&mut img, cx, H as f32 * 0.10, H as f32 * 0.40, 80.0, foliage);
    img.save(path).with_context(|| format!("writing {}", path.display()))
}

fn draw_cottage(path: &Path) -> Result<()> {
    let mut img = new_canvas();
    let wall = Rgba([220, 200, 165, 255]);
    let roof = Rgba([130, 60, 50, 255]);
    let door = Rgba([90, 50, 40, 255]);
    let window = Rgba([180, 190, 180, 255]);

    let cx = W as f32 / 2.0;
    // Walls
    let wall_w = 280.0;
    let wall_h = 200.0;
    let wall_top = H as f32 * 0.50;
    let wall_bot = wall_top + wall_h;
    fill_rect(
        &mut img,
        (cx - wall_w / 2.0) as u32,
        wall_top as u32,
        (cx + wall_w / 2.0) as u32,
        wall_bot as u32,
        wall,
    );
    // Roof: triangle on top.
    fill_triangle_down(
        &mut img,
        cx,
        H as f32 * 0.30,
        wall_top,
        180.0,
        roof,
    );
    // Door (in the middle of the wall)
    let door_w = 50.0;
    let door_h = 90.0;
    fill_rect(
        &mut img,
        (cx - door_w / 2.0) as u32,
        (wall_bot - door_h) as u32,
        (cx + door_w / 2.0) as u32,
        wall_bot as u32,
        door,
    );
    // Two windows flanking the door
    let win = 40.0;
    for offset in [-90.0, 90.0] {
        let wx = cx + offset;
        fill_rect(
            &mut img,
            (wx - win / 2.0) as u32,
            (wall_top + 40.0) as u32,
            (wx + win / 2.0) as u32,
            (wall_top + 40.0 + win) as u32,
            window,
        );
    }
    img.save(path).with_context(|| format!("writing {}", path.display()))
}

/// Subtract a circle from the canvas — set all pixels inside the
/// circle to fully transparent. Used to carve a crescent from a moon.
fn erase_circle(img: &mut RgbaImage, cx: f32, cy: f32, r: f32) {
    let r2 = r * r;
    for y in 0..img.height() {
        for x in 0..img.width() {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            if dx * dx + dy * dy <= r2 {
                img.put_pixel(x, y, TRANSPARENT);
            }
        }
    }
}

/// Bresenham-ish thick line. Used for sun rays. Coarse but adequate
/// for chunky silhouette shapes.
fn draw_thick_line(img: &mut RgbaImage, x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32, color: Rgba<u8>) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    let steps = len.ceil() as i32;
    let half = thickness / 2.0;
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let x = x0 + dx * t;
        let y = y0 + dy * t;
        for dy in -(half as i32)..=(half as i32) {
            for dx in -(half as i32)..=(half as i32) {
                let px = (x + dx as f32) as i32;
                let py = (y + dy as f32) as i32;
                if px < 0 || py < 0 || px >= img.width() as i32 || py >= img.height() as i32 {
                    continue;
                }
                blend_pixel(img, px as u32, py as u32, color, 1.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// library.json
// ---------------------------------------------------------------------------

fn write_library_json(path: &Path) -> Result<()> {
    let json = r#"{
  "schema_version": 1,
  "artefacts": [
    {
      "name": "sun",
      "category": "celestial",
      "path": "sky/sun.png",
      "natural_zone": "sky",
      "natural_size_pct": 0.7,
      "anchor": "center",
      "license": "CC0",
      "tags": ["sky", "weather", "day"]
    },
    {
      "name": "moon",
      "category": "celestial",
      "path": "sky/moon.png",
      "natural_zone": "sky",
      "natural_size_pct": 0.7,
      "anchor": "center",
      "license": "CC0",
      "tags": ["sky", "weather", "night"]
    },
    {
      "name": "cloud",
      "category": "weather",
      "path": "sky/cloud.png",
      "natural_zone": "sky",
      "natural_size_pct": 0.8,
      "anchor": "center",
      "license": "CC0",
      "tags": ["sky", "weather"]
    },
    {
      "name": "oak",
      "category": "tree",
      "path": "trees/oak.png",
      "natural_zone": "middle_plan",
      "natural_size_pct": 0.95,
      "anchor": "bottom_center",
      "license": "CC0",
      "tags": ["nature", "vegetation"]
    },
    {
      "name": "pine",
      "category": "tree",
      "path": "trees/pine.png",
      "natural_zone": "middle_plan",
      "natural_size_pct": 0.95,
      "anchor": "bottom_center",
      "license": "CC0",
      "tags": ["nature", "vegetation", "conifer"]
    },
    {
      "name": "cottage",
      "category": "building",
      "path": "houses/cottage.png",
      "natural_zone": "close_plan",
      "natural_size_pct": 0.8,
      "anchor": "bottom_center",
      "license": "CC0",
      "tags": ["architecture", "rural"]
    }
  ]
}
"#;
    std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))
}
