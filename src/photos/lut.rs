//! Minimal `.cube` 3D-LUT loader + trilinear apply — a non-AI colour grade (film looks, etc.).
//!
//! Supports the common Adobe/Resolve `.cube` format: a `LUT_3D_SIZE N` header followed by N³ RGB
//! triplets (0..1), red varying fastest. `TITLE` / `DOMAIN_*` / comment lines are ignored.

use std::path::Path;

use anyhow::{Context, Result};
use image::DynamicImage;

/// A parsed 3D LUT: `size`³ entries, indexed `r + g*size + b*size*size`.
pub struct Lut {
    pub size: usize,
    data: Vec<[f32; 3]>,
}

/// Load and validate a `.cube` file.
pub fn load_cube(path: &Path) -> Result<Lut> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut size = 0usize;
    let mut data: Vec<[f32; 3]> = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        if let Some(rest) = l.strip_prefix("LUT_3D_SIZE") {
            size = rest.trim().parse().unwrap_or(0);
            continue;
        }
        if l.starts_with("TITLE") || l.starts_with("DOMAIN_") || l.starts_with("LUT_1D") {
            continue;
        }
        let nums: Vec<f32> = l.split_whitespace().filter_map(|t| t.parse().ok()).collect();
        if nums.len() == 3 {
            data.push([nums[0], nums[1], nums[2]]);
        }
    }
    anyhow::ensure!(size >= 2, "no LUT_3D_SIZE in {}", path.display());
    anyhow::ensure!(
        data.len() == size * size * size,
        "cube has {} entries, expected {}³ in {}",
        data.len(),
        size,
        path.display()
    );
    Ok(Lut { size, data })
}

impl Lut {
    fn at(&self, r: usize, g: usize, b: usize) -> [f32; 3] {
        self.data[r + g * self.size + b * self.size * self.size]
    }

    /// Trilinearly interpolate the LUT at normalised `(r, g, b)` in 0..1.
    fn sample(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        let n = self.size - 1;
        let coord = |v: f32| v.clamp(0.0, 1.0) * n as f32;
        let (fr, fg, fb) = (coord(r), coord(g), coord(b));
        let (r0, g0, b0) = (fr.floor() as usize, fg.floor() as usize, fb.floor() as usize);
        let (r1, g1, b1) = ((r0 + 1).min(n), (g0 + 1).min(n), (b0 + 1).min(n));
        let (dr, dg, db) = (fr - r0 as f32, fg - g0 as f32, fb - b0 as f32);
        let mut out = [0.0f32; 3];
        for (i, &(rr, gg, bb, w)) in [
            (r0, g0, b0, (1.0 - dr) * (1.0 - dg) * (1.0 - db)),
            (r1, g0, b0, dr * (1.0 - dg) * (1.0 - db)),
            (r0, g1, b0, (1.0 - dr) * dg * (1.0 - db)),
            (r1, g1, b0, dr * dg * (1.0 - db)),
            (r0, g0, b1, (1.0 - dr) * (1.0 - dg) * db),
            (r1, g0, b1, dr * (1.0 - dg) * db),
            (r0, g1, b1, (1.0 - dr) * dg * db),
            (r1, g1, b1, dr * dg * db),
        ]
        .iter()
        .enumerate()
        {
            let _ = i;
            let c = self.at(rr, gg, bb);
            for k in 0..3 {
                out[k] += c[k] * w;
            }
        }
        out
    }
}

/// Apply a LUT to an image.
pub fn apply(img: &DynamicImage, lut: &Lut) -> DynamicImage {
    let mut rgb = img.to_rgb8();
    for p in rgb.pixels_mut() {
        let s = lut.sample(p.0[0] as f32 / 255.0, p.0[1] as f32 / 255.0, p.0[2] as f32 / 255.0);
        p.0 = [
            (s[0].clamp(0.0, 1.0) * 255.0).round() as u8,
            (s[1].clamp(0.0, 1.0) * 255.0).round() as u8,
            (s[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        ];
    }
    DynamicImage::ImageRgb8(rgb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn identity_cube_roundtrips() {
        let dir = std::env::temp_dir().join(format!("plakat-lut-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A 2×2×2 identity LUT (red fastest).
        let mut s = String::from("LUT_3D_SIZE 2\n");
        for b in 0..2 {
            for g in 0..2 {
                for r in 0..2 {
                    s.push_str(&format!("{} {} {}\n", r as f32, g as f32, b as f32));
                }
            }
        }
        let p = dir.join("id.cube");
        std::fs::write(&p, s).unwrap();
        let lut = load_cube(&p).unwrap();
        assert_eq!(lut.size, 2);
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4, 4, Rgb([40u8, 130, 210])));
        let out = apply(&img, &lut).to_rgb8();
        // Identity LUT ≈ unchanged (±1 for rounding).
        for c in 0..3 {
            assert!(out.get_pixel(0, 0).0[c].abs_diff([40, 130, 210][c]) <= 1);
        }
        // An inverted 2-point LUT flips values.
        let inv = "LUT_3D_SIZE 2\n1 1 1\n0 1 1\n1 0 1\n0 0 1\n1 1 0\n0 1 0\n1 1 0\n0 0 0\n";
        let ip = dir.join("inv.cube");
        std::fs::write(&ip, inv).unwrap();
        assert!(load_cube(&ip).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
