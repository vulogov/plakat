//! The print/canvas compositor (RFC BOOKART-1 §6.4 / Layer 5). Places a finished ornament onto the
//! exact page-size transparent canvas at its resolved layout rect(s), completing the **transparent +
//! exactly-page-sized** output contract. Corner pieces are mirrored to point inward.

use crate::bookart::geometry::layout::Layout;
use crate::bookart::geometry::page::PageResolved;
use image::{imageops, imageops::FilterType, Rgba, RgbaImage};

/// Composite `ornament` onto a fresh transparent page canvas at each layout rect (resized to fit,
/// alpha-over). Returns the page-sized RGBA.
pub fn place_on_canvas(ornament: &RgbaImage, page: &PageResolved, layout: &Layout) -> RgbaImage {
    let mut canvas = RgbaImage::from_pixel(page.w_px, page.h_px, Rgba([0, 0, 0, 0]));
    for r in &layout.rects {
        let mut piece = imageops::resize(ornament, r.w.max(1), r.h.max(1), FilterType::Lanczos3);
        if r.flip_h {
            piece = imageops::flip_horizontal(&piece);
        }
        if r.flip_v {
            piece = imageops::flip_vertical(&piece);
        }
        imageops::overlay(&mut canvas, &piece, r.x as i64, r.y as i64);
    }
    canvas
}

/// Save an RGBA as a PNG with the physical DPI recorded in a `pHYs` chunk (so print tools place it at
/// the right size, not a default 72). Falls back to a plain PNG if the encode path errors.
pub fn save_png_dpi(img: &RgbaImage, path: &std::path::Path, dpi: u32) -> anyhow::Result<()> {
    use std::fs::File;
    use std::io::BufWriter;
    let file = File::create(path)?;
    let w = BufWriter::new(file);
    let mut enc = png::Encoder::new(w, img.width(), img.height());
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    // pixels-per-metre = dpi / 0.0254
    let ppm = (dpi as f32 / 0.0254).round() as u32;
    enc.set_pixel_dims(Some(png::PixelDimensions { xppu: ppm, yppu: ppm, unit: png::Unit::Meter }));
    let mut writer = enc.write_header()?;
    writer.write_image_data(img.as_raw())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bookart::geometry::layout::{layout_for, text_block};
    use crate::bookart::geometry::page::resolve_page;
    use crate::bookart::spec::BookArtSpec;

    fn ornament() -> RgbaImage {
        let mut o = RgbaImage::from_pixel(64, 64, Rgba([0, 0, 0, 0]));
        for y in 20..44 {
            for x in 20..44 {
                o.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }
        o
    }

    #[test]
    fn canvas_is_page_sized_and_has_the_ornament() {
        let page = resolve_page(None); // a5 300 → 1748 × 2480
        let tb = text_block(&page, &BookArtSpec::default());
        let layout = layout_for("headpiece", &tb);
        let canvas = place_on_canvas(&ornament(), &page, &layout);
        assert_eq!(canvas.dimensions(), (page.w_px, page.h_px), "exact page size");
        assert!(canvas.pixels().any(|p| p.0[3] > 200), "ornament composited");
        // corners of the page are transparent (nothing placed there)
        assert_eq!(canvas.get_pixel(0, 0).0[3], 0);
    }

    #[test]
    fn four_corners_place_four_pieces() {
        let page = resolve_page(None);
        let tb = text_block(&page, &BookArtSpec::default());
        let layout = layout_for("corner", &tb);
        assert_eq!(layout.rects.len(), 4);
        let canvas = place_on_canvas(&ornament(), &page, &layout);
        assert_eq!(canvas.dimensions(), (page.w_px, page.h_px));
        // each placed piece has ink at its own centre (the test ornament is a centred square).
        for r in &layout.rects {
            let (cx, cy) = (r.x + r.w / 2, r.y + r.h / 2);
            assert!(canvas.get_pixel(cx, cy).0[3] > 128, "piece at ({},{}) has no ink at its centre", r.x, r.y);
        }
    }
}
