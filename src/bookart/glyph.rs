//! 6.1.0 (B2): glyph-driven initials (RFC BOOKART-1 §6.5) — the one *intentional-text* path. Rasterise
//! a real letter (any script, incl. Cyrillic) from a supplied TrueType/OpenType font via `ab_glyph`,
//! so a historiated initial is built *around* a legible letterform instead of a diffusion model
//! hallucinating fake glyphs. Behind the `shaped-labels` feature (the same `ab_glyph` dep the map's
//! shaped labels use); without it, `initial` falls back to the decorative composite/diffusion path.

#![cfg(feature = "shaped-labels")]

use anyhow::{Context, Result};
use image::{GrayImage, Luma};

/// Rasterise the first character of `letter` as bold black ink on a white `w×h` `GrayImage`, centred
/// and scaled to fill ~82% of the cell. The result feeds the finisher/frame like any procedural gray.
pub fn render_initial(letter: &str, font_path: &std::path::Path, w: u32, h: u32) -> Result<GrayImage> {
    use ab_glyph::{Font, FontVec, ScaleFont};

    let ch = letter.chars().next().context("initial: empty glyph string")?;
    let bytes = std::fs::read(font_path).with_context(|| format!("reading font {}", font_path.display()))?;
    let font = FontVec::try_from_vec(bytes).map_err(|e| anyhow::anyhow!("parsing font {}: {e}", font_path.display()))?;

    // Scale the em so the glyph's own bounds fill ~82% of the cell. Measure once at a reference px,
    // then rescale by the limiting dimension.
    let target = (w.min(h) as f32) * 0.82;
    let ref_px = 256.0_f32;
    let (rw, rh) = glyph_bounds(&font, ch, ref_px).context("initial: glyph has no outline (missing from the font?)")?;
    let scale = (target / rw.max(rh).max(1.0)) * ref_px;
    let px = scale.clamp(8.0, (h as f32) * 2.0);

    let sf = font.as_scaled(px);
    let mut glyph = sf.scaled_glyph(ch);
    glyph.position = ab_glyph::point(0.0, 0.0);
    let outline = font.outline_glyph(glyph).context("initial: glyph has no outline")?;
    let bb = outline.px_bounds();
    // Centre the glyph bbox in the cell.
    let ox = (w as f32 - bb.width()) / 2.0 - bb.min.x;
    let oy = (h as f32 - bb.height()) / 2.0 - bb.min.y;

    let mut img = GrayImage::from_pixel(w, h, Luma([255]));
    outline.draw(|gx, gy, cov| {
        if cov <= 0.02 {
            return;
        }
        let (px, py) = ((bb.min.x + ox) as i32 + gx as i32, (bb.min.y + oy) as i32 + gy as i32);
        if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
            return;
        }
        let p = img.get_pixel_mut(px as u32, py as u32);
        let v = (255.0 * (1.0 - cov.clamp(0.0, 1.0))).round() as u8;
        p.0[0] = p.0[0].min(v);
    });
    Ok(img)
}

/// The px bounds (w, h) of a glyph outline at `px`, for the fit computation.
fn glyph_bounds(font: &impl ab_glyph::Font, ch: char, px: f32) -> Option<(f32, f32)> {
    use ab_glyph::ScaleFont;
    let sf = font.as_scaled(px);
    let mut g = sf.scaled_glyph(ch);
    g.position = ab_glyph::point(0.0, 0.0);
    let o = font.outline_glyph(g)?;
    let bb = o.px_bounds();
    Some((bb.width(), bb.height()))
}
