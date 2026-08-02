//! Page / print model (RFC BOOKART-1 §6.4). B0 slice: named sizes → mm → exact pixels at a DPI, with
//! orientation and a `custom` escape. The text-block / margin / gutter / bleed geometry and ornament
//! layout land in B2 (`layout.rs`); this is the canvas half the resolver needs today.

use crate::bookart::spec::Page;

/// Named page size → (width_mm, height_mm) in portrait. Trade = 6×9″, mass-market = 4.25×6.87″.
pub fn named_size_mm(name: &str) -> Option<(f32, f32)> {
    Some(match name {
        "a4" => (210.0, 297.0),
        "a5" => (148.0, 210.0),
        "a6" => (105.0, 148.0),
        "b5" => (176.0, 250.0),
        "letter" | "us-letter" => (215.9, 279.4),
        "legal" | "us-legal" => (215.9, 355.6),
        "trade" => (152.4, 228.6),
        "mass-market" | "mass" => (107.95, 174.625),
        _ => return None,
    })
}

/// The full advertised size vocabulary (for `--help`, lint suggestions, and `new`).
pub const SIZE_VOCAB: &[&str] = &["a4", "a5", "a6", "b5", "letter", "legal", "trade", "mass-market", "custom"];

pub const DEFAULT_DPI: u32 = 300;

/// A resolved canvas: exact pixels + the physical size + DPI (written to PNG `pHYs` / the SVG later).
#[derive(Debug, Clone, PartialEq)]
pub struct PageResolved {
    pub w_px: u32,
    pub h_px: u32,
    pub dpi: u32,
    pub w_mm: f32,
    pub h_mm: f32,
    pub bleed_mm: f32,
    pub size_name: String,
}

fn mm_to_px(mm: f32, dpi: u32) -> u32 {
    ((mm / 25.4) * dpi as f32).round().max(1.0) as u32
}

/// Resolve a `Page` (or `None`) to a concrete canvas. Defaults: `a5`, 300 DPI, portrait, 3 mm bleed.
pub fn resolve_page(page: Option<&Page>) -> PageResolved {
    let dpi = page.and_then(|p| p.dpi).unwrap_or(DEFAULT_DPI).clamp(72, 1200);
    let size_name = page.and_then(|p| p.size.clone()).unwrap_or_else(|| "a5".into());
    let bleed_mm = page.and_then(|p| p.bleed_mm).unwrap_or(3.0).max(0.0);

    let (mut w_mm, mut h_mm) = if size_name == "custom" {
        let c = page.and_then(|p| p.custom.as_ref());
        (
            c.and_then(|c| c.w_mm).unwrap_or(148.0).max(1.0),
            c.and_then(|c| c.h_mm).unwrap_or(210.0).max(1.0),
        )
    } else {
        named_size_mm(&size_name).unwrap_or((148.0, 210.0)) // unknown name → a5 fallback; lint flags it
    };

    if page.and_then(|p| p.orientation.as_deref()) == Some("landscape") {
        std::mem::swap(&mut w_mm, &mut h_mm);
    }

    PageResolved { w_px: mm_to_px(w_mm, dpi), h_px: mm_to_px(h_mm, dpi), dpi, w_mm, h_mm, bleed_mm, size_name }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a4_at_300_is_2480x3508() {
        let p = resolve_page(Some(&Page { size: Some("a4".into()), dpi: Some(300), ..Default::default() }));
        assert_eq!((p.w_px, p.h_px), (2480, 3508));
    }

    #[test]
    fn default_is_a5_300_portrait() {
        let p = resolve_page(None);
        assert_eq!(p.size_name, "a5");
        assert_eq!(p.dpi, 300);
        assert_eq!((p.w_px, p.h_px), (1748, 2480));
    }

    #[test]
    fn landscape_swaps() {
        let port = resolve_page(Some(&Page { size: Some("a5".into()), ..Default::default() }));
        let land = resolve_page(Some(&Page { size: Some("a5".into()), orientation: Some("landscape".into()), ..Default::default() }));
        assert_eq!((port.w_px, port.h_px), (land.h_px, land.w_px));
    }

    #[test]
    fn custom_size() {
        let p = resolve_page(Some(&Page {
            size: Some("custom".into()),
            dpi: Some(300),
            custom: Some(crate::bookart::spec::CustomSize { w_mm: Some(100.0), h_mm: Some(100.0) }),
            ..Default::default()
        }));
        assert_eq!((p.w_px, p.h_px), (1181, 1181));
    }
}
