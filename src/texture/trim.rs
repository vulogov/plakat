//! Track A (ROADMAP 6.5.0) — **trim sheets**: compose several sub-materials into one **atlas** (stacked
//! horizontal bands, each a strip that tiles along its run axis) + a UV-region sidecar so an engine can
//! map faces to bands. The standard way games texture pipes / trims / panels / edges from one material.
//! Weight-free: the bands are pre-rendered [`Material`] sets; this just composites their channels.

use crate::texture::derive::Material;
use image::{GrayImage, RgbImage};
use serde::{Deserialize, Serialize};

/// A trim-sheet spec — an ordered list of bands stacked top→bottom into a square atlas.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TrimSpec {
    pub schema: Option<String>,
    /// Atlas edge size in px (square). Default 1024.
    pub size: Option<u32>,
    /// Channel filename convention for the export (`plakat` | `unity` | `unreal`).
    pub naming: Option<String>,
    pub bands: Vec<TrimBand>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrimBand {
    /// Path to a material directory (contains `albedo.png`, …).
    pub material: String,
    /// The band's share of the atlas height (a fraction; bands are normalised to sum to 1). Default: an
    /// equal split.
    pub height: Option<f32>,
    /// How the sub-material fills the band's run: `x` (default, tiles horizontally) | `y` | `none` (stretch).
    pub tile: Option<String>,
    /// A label carried into the UV sidecar.
    pub label: Option<String>,
}

impl TrimSpec {
    pub fn from_hjson(text: &str) -> Result<Self, deser_hjson::Error> {
        deser_hjson::from_str(text)
    }
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        Self::from_hjson(&text).map_err(|e| anyhow::anyhow!("parsing {}: {e}", path.display()))
    }
}

/// How a band's sub-material fills its strip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimAxis {
    /// Tile horizontally (the strip repeats along U) — the trim-sheet default.
    X,
    /// Tile vertically within the band.
    Y,
    /// Stretch to fill (no repeat).
    None,
}

impl TrimAxis {
    pub fn parse(s: &str) -> TrimAxis {
        match s.to_ascii_lowercase().as_str() {
            "y" | "vertical" => TrimAxis::Y,
            "none" | "stretch" => TrimAxis::None,
            _ => TrimAxis::X,
        }
    }
}

/// One band's UV rectangle in the atlas (top-left origin) — the sidecar entry an engine maps faces to.
#[derive(Debug, Clone, Serialize)]
pub struct TrimRegion {
    pub label: String,
    pub u0: f32,
    pub v0: f32,
    pub u1: f32,
    pub v1: f32,
    pub tile: String,
}

/// A pre-loaded band ready to composite.
pub struct BandInput {
    pub material: Material,
    pub height: f32,
    pub tile: TrimAxis,
    pub label: String,
}

// --- strip builders (fill a band's WxH slot from a sub-material channel) --------------------------

fn strip_rgb(src: &RgbImage, w: u32, h: u32, tile: TrimAxis) -> RgbImage {
    use image::imageops::{resize, FilterType::Lanczos3};
    match tile {
        TrimAxis::None => resize(src, w, h, Lanczos3),
        TrimAxis::X => {
            let sq = resize(src, h.max(1), h.max(1), Lanczos3); // one tile = band height square
            RgbImage::from_fn(w, h, |x, y| *sq.get_pixel(x % h.max(1), y))
        }
        TrimAxis::Y => {
            let sq = resize(src, w.max(1), w.max(1), Lanczos3);
            RgbImage::from_fn(w, h, |x, y| *sq.get_pixel(x, y % w.max(1)))
        }
    }
}

fn strip_gray(src: &GrayImage, w: u32, h: u32, tile: TrimAxis) -> GrayImage {
    use image::imageops::{resize, FilterType::Lanczos3};
    match tile {
        TrimAxis::None => resize(src, w, h, Lanczos3),
        TrimAxis::X => {
            let sq = resize(src, h.max(1), h.max(1), Lanczos3);
            GrayImage::from_fn(w, h, |x, y| *sq.get_pixel(x % h.max(1), y))
        }
        TrimAxis::Y => {
            let sq = resize(src, w.max(1), w.max(1), Lanczos3);
            GrayImage::from_fn(w, h, |x, y| *sq.get_pixel(x, y % w.max(1)))
        }
    }
}

/// Per-band pixel heights that sum EXACTLY to `total`, from normalised fractions (last band absorbs the
/// rounding remainder so no gap/overlap).
fn band_heights_px(bands: &[BandInput], total: u32) -> Vec<u32> {
    let sum: f32 = bands.iter().map(|b| b.height.max(0.0)).sum();
    let sum = if sum <= 0.0 { bands.len() as f32 } else { sum };
    let mut px: Vec<u32> = bands.iter().map(|b| ((b.height.max(0.0) / sum) * total as f32).round() as u32).collect();
    let acc: u32 = px.iter().sum();
    if let Some(last) = px.last_mut() {
        *last = (*last as i64 + total as i64 - acc as i64).max(1) as u32; // absorb the remainder
    }
    px
}

/// Compose bands (top→bottom) into one square atlas [`Material`] + the UV-region sidecar. The atlas
/// tiles along U (each band's run axis) but NOT along V (the bands are distinct) — expected for a trim.
pub fn compose(bands: &[BandInput], size: u32) -> (Material, Vec<TrimRegion>) {
    let (w, h) = (size, size);
    let heights = band_heights_px(bands, h);
    let mut albedo = RgbImage::new(w, h);
    let mut normal = RgbImage::from_pixel(w, h, image::Rgb([128, 128, 255]));
    let mut height = GrayImage::new(w, h);
    let mut roughness = GrayImage::from_pixel(w, h, image::Luma([153]));
    let mut metallic = GrayImage::new(w, h);
    let mut ao = GrayImage::from_pixel(w, h, image::Luma([255]));
    let mut regions = Vec::new();
    let mut y0 = 0u32;
    for (band, &bh) in bands.iter().zip(&heights) {
        let bh = bh.min(h.saturating_sub(y0)).max(1);
        let m = &band.material;
        blit_rgb(&mut albedo, &strip_rgb(&m.albedo, w, bh, band.tile), y0);
        blit_rgb(&mut normal, &strip_rgb(&m.normal, w, bh, band.tile), y0);
        blit_gray(&mut height, &strip_gray(&m.height, w, bh, band.tile), y0);
        blit_gray(&mut roughness, &strip_gray(&m.roughness, w, bh, band.tile), y0);
        blit_gray(&mut metallic, &strip_gray(&m.metallic, w, bh, band.tile), y0);
        blit_gray(&mut ao, &strip_gray(&m.ao, w, bh, band.tile), y0);
        regions.push(TrimRegion {
            label: band.label.clone(),
            u0: 0.0,
            v0: y0 as f32 / h as f32,
            u1: 1.0,
            v1: (y0 + bh) as f32 / h as f32,
            tile: match band.tile {
                TrimAxis::X => "x",
                TrimAxis::Y => "y",
                TrimAxis::None => "none",
            }
            .into(),
        });
        y0 += bh;
    }
    (Material { albedo, height, normal, roughness, metallic, ao, anisotropy: None }, regions)
}

fn blit_rgb(dst: &mut RgbImage, src: &RgbImage, y0: u32) {
    let (w, h) = dst.dimensions();
    for y in 0..src.height().min(h.saturating_sub(y0)) {
        for x in 0..src.width().min(w) {
            dst.put_pixel(x, y0 + y, *src.get_pixel(x, y));
        }
    }
}
fn blit_gray(dst: &mut GrayImage, src: &GrayImage, y0: u32) {
    let (w, h) = dst.dimensions();
    for y in 0..src.height().min(h.saturating_sub(y0)) {
        for x in 0..src.width().min(w) {
            dst.put_pixel(x, y0 + y, *src.get_pixel(x, y));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::compile::ChannelSource;
    use image::Rgb;

    fn mat(fill: u8) -> Material {
        Material::derive(RgbImage::from_pixel(32, 32, Rgb([fill, fill, fill])), None, 1.0, true, 1.0, &ChannelSource::Scalar(0.5), &ChannelSource::Scalar(0.0))
    }

    #[test]
    fn two_equal_bands_split_the_atlas_and_carry_their_content() {
        let bands = vec![
            BandInput { material: mat(40), height: 0.5, tile: TrimAxis::X, label: "top".into() },
            BandInput { material: mat(200), height: 0.5, tile: TrimAxis::X, label: "bot".into() },
        ];
        let (atlas, regions) = compose(&bands, 64);
        assert_eq!(atlas.albedo.dimensions(), (64, 64));
        // top band ≈ material A (40), bottom ≈ material B (200).
        assert!((atlas.albedo.get_pixel(10, 16).0[0] as i32 - 40).abs() <= 4, "top band = A");
        assert!((atlas.albedo.get_pixel(10, 48).0[0] as i32 - 200).abs() <= 4, "bottom band = B");
        // UV sidecar: two stacked full-width regions.
        assert_eq!(regions.len(), 2);
        assert!((regions[0].v0 - 0.0).abs() < 1e-6 && (regions[0].v1 - 0.5).abs() < 0.02);
        assert!((regions[1].v0 - 0.5).abs() < 0.02 && (regions[1].v1 - 1.0).abs() < 1e-6);
        assert_eq!((regions[0].u0, regions[0].u1), (0.0, 1.0));
    }

    #[test]
    fn heights_sum_exactly_to_the_atlas() {
        let bands = vec![
            BandInput { material: mat(10), height: 0.3, tile: TrimAxis::X, label: "a".into() },
            BandInput { material: mat(20), height: 0.3, tile: TrimAxis::X, label: "b".into() },
            BandInput { material: mat(30), height: 0.4, tile: TrimAxis::X, label: "c".into() },
        ];
        let px = band_heights_px(&bands, 100);
        assert_eq!(px.iter().sum::<u32>(), 100, "bands fill the atlas with no gap/overlap");
    }
}
