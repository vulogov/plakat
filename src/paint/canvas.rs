//! The pigment canvas (RFC PAINT-1 §8.1). The canvas does NOT carry RGB — it carries, per pixel, a **pigment
//! concentration vector** over the active palette plus paint **height**, **wetness**, and a static **tooth**.
//! sRGB is computed only on output, by Kubelka-Munk mixing the concentration vector. This is what makes mixing,
//! glazing, pickup, and mud *physical* rather than faked: a heavy stroke shifts the concentration ratio and
//! dominates (opaque), a thin one barely perturbs it and the ground shows through (glaze).
//!
//! Colour is a function of the concentration *ratio*, so laying more of the same pigment doesn't darken —
//! only introducing other pigment does. Total concentration is tracked as **saturation** (how full the tooth
//! is), which throttles further deposit so paint build-up is bounded.

use image::RgbImage;

use crate::paint::color::Srgb;
use crate::paint::palette::Palette;
use crate::paint::pigment;

/// A thin priming's worth of ground pigment — low enough that an opaque stroke covers it.
pub const GROUND_CONC: f32 = 0.25;

/// A pigment canvas over a fixed palette basis.
#[derive(Clone)]
pub struct Canvas {
    pub w: u32,
    pub h: u32,
    palette: Palette,
    /// Per-pixel concentration over the palette: `w*h*n`, row-major, pixel-major.
    conc: Vec<f32>,
    /// Per-pixel paint height (impasto), row-major.
    pub height: Vec<f32>,
    /// Per-pixel wet pigment available for pickup, row-major (0 = dry).
    pub wetness: Vec<f32>,
    /// Per-pixel static surface tooth in `[0,1]` (how much the brush catches), row-major.
    pub tooth: Vec<f32>,
    n: usize,
}

impl Canvas {
    /// A canvas primed with a ground: the given concentration vector at every pixel (length = palette size).
    /// `tooth` is uniform. `wetness`/`height` start at zero.
    pub fn with_ground(w: u32, h: u32, palette: Palette, ground: &[f32], tooth: f32) -> Self {
        let n = palette.pigments.len();
        let px = (w * h) as usize;
        let mut g = vec![0f32; n];
        for i in 0..n.min(ground.len()) {
            g[i] = ground[i].max(0.0);
        }
        let mut conc = Vec::with_capacity(px * n);
        for _ in 0..px {
            conc.extend_from_slice(&g);
        }
        Self { w, h, palette, conc, height: vec![0.0; px], wetness: vec![0.0; px], tooth: vec![tooth.clamp(0.0, 1.0); px], n }
    }

    /// A canvas primed with a WHITE ground (the lightest pigment in the palette), i.e. bare paper / gessoed
    /// panel. The default for opaque media before an imprimatura. The ground carries only a MODEST
    /// concentration — a thin priming — so an opaque stroke's deposit overcomes it (covers), rather than the
    /// substrate contributing forever in the concentration-ratio mix.
    pub fn white(w: u32, h: u32, palette: Palette, tooth: f32) -> Self {
        let white = lightest_pigment(&palette);
        let mut ground = vec![0f32; palette.pigments.len()];
        ground[white] = GROUND_CONC;
        Self::with_ground(w, h, palette, &ground, tooth)
    }

    /// A canvas primed with a TONED ground (imprimatura): the given tone, solved into the palette. Opaque media
    /// work on a mid-tone so light passages SHOW (light paint on a white ground is invisible) and the picture
    /// reads as one keyed surface rather than sparse marks on paper.
    pub fn toned(w: u32, h: u32, palette: Palette, tone: Srgb, tooth: f32) -> Self {
        let m = crate::paint::mixer::solve_mixture(&palette, tone, 3);
        let mut ground = vec![0f32; palette.pigments.len()];
        for (&i, &wt) in m.pigments.iter().zip(m.weights.iter()) {
            ground[i] = wt * GROUND_CONC;
        }
        Self::with_ground(w, h, palette, &ground, tooth)
    }

    pub fn palette(&self) -> &Palette {
        &self.palette
    }
    pub fn n_pigments(&self) -> usize {
        self.n
    }

    #[inline]
    fn idx(&self, x: u32, y: u32) -> usize {
        (y as usize * self.w as usize + x as usize) * self.n
    }

    /// The concentration vector at a pixel.
    pub fn conc_at(&self, x: u32, y: u32) -> &[f32] {
        let i = self.idx(x, y);
        &self.conc[i..i + self.n]
    }

    /// Total pigment concentration at a pixel — the "saturation" that throttles further deposit.
    pub fn saturation_at(&self, x: u32, y: u32) -> f32 {
        self.conc_at(x, y).iter().sum()
    }

    /// Deposit a concentration delta at a pixel (adds to the accumulated pigment) and raise its height.
    pub fn deposit(&mut self, x: u32, y: u32, delta: &[f32], height: f32) {
        let i = self.idx(x, y);
        for c in 0..self.n {
            self.conc[i + c] += delta.get(c).copied().unwrap_or(0.0).max(0.0);
        }
        let p = y as usize * self.w as usize + x as usize;
        self.height[p] += height.max(0.0);
    }

    /// WIPE / scrape back at a pixel (RFC PAINT-1 §8.7): remove a `strength` fraction of the accumulated
    /// pigment and height, exposing the ground/earlier work beneath (Sargent scraping a head, Turner wiping,
    /// watercolour lifting). A structural move, not error correction.
    pub fn wipe(&mut self, x: u32, y: u32, strength: f32) {
        let s = strength.clamp(0.0, 1.0);
        let i = self.idx(x, y);
        for c in 0..self.n {
            self.conc[i + c] *= 1.0 - s;
        }
        // Restore the primer (white ground) proportionally, so wiping EXPOSES the ground rather than just
        // scaling the same mixture down (which, being a ratio, wouldn't change the colour).
        let white = lightest_pigment(&self.palette);
        self.conc[i + white] += GROUND_CONC * s;
        let p = y as usize * self.w as usize + x as usize;
        self.height[p] *= 1.0 - s;
    }

    /// The sRGB colour at a pixel — Kubelka-Munk mix of its concentration vector over the palette.
    pub fn color_at(&self, x: u32, y: u32) -> Srgb {
        pigment::mix(self.palette.pigments, self.conc_at(x, y))
    }

    /// A copy of the current canvas state — for timelapse frames.
    pub fn snapshot(&self) -> Canvas {
        self.clone()
    }

    /// The impasto height as a 16-bit grayscale image, normalised to the canvas's peak height (0 if flat).
    pub fn height_image(&self) -> image::ImageBuffer<image::Luma<u16>, Vec<u16>> {
        let peak = self.height.iter().copied().fold(0.0_f32, f32::max);
        let scale = if peak > 0.0 { 65535.0 / peak } else { 0.0 };
        image::ImageBuffer::from_fn(self.w, self.h, |x, y| {
            let v = self.height[y as usize * self.w as usize + x as usize] * scale;
            image::Luma([v.round().clamp(0.0, 65535.0) as u16])
        })
    }

    /// Render the whole canvas to an sRGB image (no impasto relight — that is a later output stage).
    pub fn to_image(&self) -> RgbImage {
        let mut img = RgbImage::new(self.w, self.h);
        for y in 0..self.h {
            for x in 0..self.w {
                let c = self.color_at(x, y);
                img.put_pixel(x, y, image::Rgb(c));
            }
        }
        img
    }
}

/// The index of the lightest pigment in a palette (highest CIELAB L*), used as "white"/ground.
pub fn lightest_pigment(palette: &Palette) -> usize {
    use crate::paint::color::srgb_to_lab;
    (0..palette.pigments.len())
        .max_by(|&a, &b| srgb_to_lab(palette.pigments[a].masstone).l.partial_cmp(&srgb_to_lab(palette.pigments[b].masstone).l).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0)
}

/// The index of the darkest pigment in a palette (lowest CIELAB L*) — the ink for a density medium.
pub fn darkest_pigment(palette: &Palette) -> usize {
    use crate::paint::color::srgb_to_lab;
    (0..palette.pigments.len())
        .min_by(|&a, &b| srgb_to_lab(palette.pigments[a].masstone).l.partial_cmp(&srgb_to_lab(palette.pigments[b].masstone).l).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::color::{delta_e76, srgb_to_lab};
    use crate::paint::palette;

    #[test]
    fn white_ground_reads_white() {
        let c = Canvas::white(4, 4, palette::ZORN, 0.7);
        let col = c.color_at(1, 1);
        assert!(col[0] > 235 && col[1] > 235 && col[2] > 225, "white ground is white: {col:?}");
    }

    #[test]
    fn a_heavy_deposit_dominates_the_ground() {
        // Zorn: [ochre, cad-red, black, white]. Ground is white; deposit a lot of red.
        let mut c = Canvas::white(2, 2, palette::ZORN, 0.7);
        c.deposit(0, 0, &[0.0, 6.0, 0.0, 0.0], 0.5);
        let col = c.color_at(0, 0);
        let d = delta_e76(srgb_to_lab(col), srgb_to_lab(palette::CADMIUM_RED.masstone));
        assert!(d < 25.0, "a heavy deposit reads mostly as the pigment: {col:?} (ΔE {d})");
        assert!(c.height[0] > 0.4, "height accumulated");
    }

    #[test]
    fn wipe_scrapes_pigment_and_height_back() {
        let mut c = Canvas::white(2, 2, palette::ZORN, 0.7);
        c.deposit(0, 0, &[0.0, 1.2, 0.0, 0.0], 0.6); // a red pass + impasto
        let before = srgb_to_lab(c.color_at(0, 0)).l;
        let h_before = c.height[0];
        c.wipe(0, 0, 0.8);
        assert!(srgb_to_lab(c.color_at(0, 0)).l > before + 3.0, "wiping back lightens toward the ground");
        assert!(c.height[0] < h_before, "impasto height scraped down");
    }

    #[test]
    fn a_thin_deposit_shifts_less_than_a_thick_one() {
        // Opacity is emergent: a thin glaze perturbs the ground less than a covering deposit of the same
        // pigment (both shift it — a strong tinter like red visibly tints even thin).
        let base = srgb_to_lab(Canvas::white(1, 1, palette::ZORN, 0.7).color_at(0, 0));
        let mut thin = Canvas::white(1, 1, palette::ZORN, 0.7);
        thin.deposit(0, 0, &[0.0, 0.05, 0.0, 0.0], 0.0);
        let mut thick = Canvas::white(1, 1, palette::ZORN, 0.7);
        thick.deposit(0, 0, &[0.0, 1.0, 0.0, 0.0], 0.0);
        let d_thin = delta_e76(base, srgb_to_lab(thin.color_at(0, 0)));
        let d_thick = delta_e76(base, srgb_to_lab(thick.color_at(0, 0)));
        assert!(d_thin > 0.0, "a glaze does tint");
        assert!(d_thin < d_thick, "thin glaze shifts less than a covering deposit ({d_thin} < {d_thick})");
    }
}
