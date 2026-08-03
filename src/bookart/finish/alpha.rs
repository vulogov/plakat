//! Transparency — the B/W-native model (RFC BOOKART-1 §7.2). Ink darkness *is* opacity, so a binarised
//! grayscale ornament → an RGBA where the ink is `tint` and the alpha comes from the luminance. The
//! curve was tuned in the G0.5 probe (`examples/bookart_alpha_probe.rs`): `white_cut ≈ 0.07, γ ≈ 0.70`
//! zeroes page-haze while keeping thin/grey lines opaque. Pure; no matting model, no halo.

use image::{GrayImage, Rgba, RgbaImage};

/// The frozen `luminance` defaults (G0.5).
pub const WHITE_CUT: f32 = 0.07;
pub const GAMMA: f32 = 0.70;

/// luminance → alpha: `clamp((1 − L − white_cut)/(1 − white_cut))^γ`.
pub fn luminance_alpha(l: u8, white_cut: f32, gamma: f32) -> u8 {
    let cov = (255.0 - l as f32) / 255.0;
    let cov = ((cov - white_cut) / (1.0 - white_cut)).clamp(0.0, 1.0);
    (cov.powf(gamma) * 255.0).round() as u8
}

/// Parse an ink tint: `black` | `white` | `sepia` | `#rrggbb`. Unknown → black.
pub fn parse_tint(s: &str) -> [u8; 3] {
    match s.trim().to_ascii_lowercase().as_str() {
        "black" => [0, 0, 0],
        "white" => [255, 255, 255],
        "sepia" => [80, 54, 28],
        hex if hex.starts_with('#') && hex.len() == 7 => {
            let p = |a, b| u8::from_str_radix(&hex[a..b], 16).unwrap_or(0);
            [p(1, 3), p(3, 5), p(5, 7)]
        }
        _ => [0, 0, 0],
    }
}

/// A radial-ish edge falloff for `fade`/vignette: alpha scales down over the outer `fade` fraction of
/// the half-diagonal. `fade == 0` → no change.
fn edge_falloff(x: u32, y: u32, w: u32, h: u32, fade: f32) -> f32 {
    if fade <= 0.0 {
        return 1.0;
    }
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let d = (((x as f32 - cx) / cx).powi(2) + ((y as f32 - cy) / cy).powi(2)).sqrt(); // 0 centre .. ~1.41 corner
    let start = 1.0 - fade.clamp(0.0, 1.0);
    ((1.0 - (d - start) / (1.0 - start).max(1e-3)).clamp(0.0, 1.0)).min(1.0)
}

/// Compose a binarised grayscale ornament into a transparent RGBA. `mode`: `luminance` (default) |
/// `threshold` | `fade` | `matte`. **`matte` needs U2Net (weights) and is wired at render time — here it
/// falls back to `luminance`** (pure). `tint` recolours the ink; `fade` feathers the edges.
pub fn to_transparent(gray: &GrayImage, mode: &str, tint: [u8; 3], fade: f32) -> RgbaImage {
    let (w, h) = (gray.width(), gray.height());
    let mut out = RgbaImage::new(w, h);
    for (x, y, p) in gray.enumerate_pixels() {
        let l = p.0[0];
        let mut a = match mode {
            "threshold" => {
                // crisp cut with a soft anti-alias ramp around mid-grey
                let (t, ramp) = (160.0f32, 24.0f32);
                (((t + ramp - l as f32) / (2.0 * ramp)).clamp(0.0, 1.0) * 255.0).round() as u8
            }
            _ => luminance_alpha(l, WHITE_CUT, GAMMA), // luminance | fade | matte(fallback)
        };
        if mode == "fade" || fade > 0.0 {
            a = (a as f32 * edge_falloff(x, y, w, h, fade)).round() as u8;
        }
        out.put_pixel(x, y, Rgba([tint[0], tint[1], tint[2], a]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn ink_is_opaque_paper_is_transparent() {
        assert_eq!(luminance_alpha(0, WHITE_CUT, GAMMA), 255); // black ink → opaque
        assert_eq!(luminance_alpha(255, WHITE_CUT, GAMMA), 0); // white paper → transparent
        assert_eq!(luminance_alpha(250, WHITE_CUT, GAMMA), 0); // near-white → still transparent (white_cut)
    }

    #[test]
    fn tint_parsing() {
        assert_eq!(parse_tint("black"), [0, 0, 0]);
        assert_eq!(parse_tint("sepia"), [80, 54, 28]);
        assert_eq!(parse_tint("#ff8000"), [255, 128, 0]);
        assert_eq!(parse_tint("nonsense"), [0, 0, 0]);
    }

    #[test]
    fn transparent_carries_tint_and_alpha() {
        let mut g = GrayImage::from_pixel(8, 8, Luma([255]));
        g.put_pixel(4, 4, Luma([0]));
        let rgba = to_transparent(&g, "luminance", [10, 20, 30], 0.0);
        assert_eq!(rgba.get_pixel(0, 0).0[3], 0, "paper transparent");
        let ink = rgba.get_pixel(4, 4).0;
        assert_eq!(ink[3], 255, "ink opaque");
        assert_eq!([ink[0], ink[1], ink[2]], [10, 20, 30], "ink tinted");
    }
}
