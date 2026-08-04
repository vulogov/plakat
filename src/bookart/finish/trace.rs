//! 6.1.0 (B1): raster→SVG trace for the diffusion/composite tiers (RFC BOOKART-1 §7.5). The procedural
//! tier is born-vector (`finish/vector.rs`); the pixel tiers can only be *traced*. vtracer
//! (MIT/Apache-2.0, resolved in G0.1) turns a finished transparent ornament into a compact set of
//! filled B/W paths. Behind the `bookart-trace` feature — vtracer pulls an older image stack that
//! would bloat the base build, so it stays opt-in (the PNG is always the deliverable).

use anyhow::{Context, Result};
use image::RgbaImage;

/// Trace a finished ornament (transparent, B/W ink) into an SVG string.
///
/// The ornament is flattened onto white first (ink→black on white) so vtracer's **Binary** preset
/// keys cleanly on the ink; the emitted `<path>` fills are then recoloured to `tint` (the SVG paper
/// stays transparent — no background rect is drawn). The physical page size (mm, from `dpi`) is set as
/// the SVG `width`/`height` so a layout tool places it at the right print size.
pub fn trace_rgba(img: &RgbaImage, tint: [u8; 3], dpi: u32) -> Result<String> {
    let (w, h) = img.dimensions();
    // Flatten onto white so the tracer sees black ink on a solid field (alpha-keying a mostly
    // transparent page is unreliable). Luminance-weighted alpha compositing over #fff.
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for p in img.pixels() {
        let [r, g, b, a] = p.0;
        let a = a as f32 / 255.0;
        let over = |c: u8| ((c as f32 * a) + 255.0 * (1.0 - a)).round() as u8;
        rgba.extend_from_slice(&[over(r), over(g), over(b), 255]);
    }
    let ci = vtracer::ColorImage { pixels: rgba, width: w as usize, height: h as usize };
    let cfg = vtracer::Config::from_preset(vtracer::Preset::Bw);
    let svg = vtracer::convert(ci, cfg).map_err(|e| anyhow::anyhow!("vtracer trace failed: {e}"))?;
    let raw = svg.to_string();
    Ok(retint_and_size(&raw, w, h, dpi, tint))
}

/// vtracer emits `<svg ... width="Wpx" height="Hpx"><path ... fill="rgb(...)" .../>…`. Two touch-ups:
/// stamp the physical print size (mm) on the root, and force every `fill=` to the ink `tint` (Binary
/// mode fills with whatever dark colour it clustered — normalise to the requested ink).
fn retint_and_size(svg: &str, w: u32, h: u32, dpi: u32, tint: [u8; 3]) -> String {
    let mm = |px: u32| px as f32 * 25.4 / dpi as f32;
    let fill = format!("fill=\"#{:02x}{:02x}{:02x}\"", tint[0], tint[1], tint[2]);
    let mut out = String::with_capacity(svg.len() + 128);
    for line in svg.lines() {
        if line.trim_start().starts_with("<svg") {
            // Replace the pixel width/height with physical mm + keep a px viewBox.
            out.push_str(&format!(
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.2}mm\" height=\"{:.2}mm\" viewBox=\"0 0 {w} {h}\">\n",
                mm(w), mm(h)
            ));
            continue;
        }
        // Force the ink colour on every traced path.
        if let Some(idx) = line.find("fill=\"") {
            let end = line[idx + 6..].find('"').map(|e| idx + 6 + e + 1).unwrap_or(line.len());
            let mut l = String::with_capacity(line.len());
            l.push_str(&line[..idx]);
            l.push_str(&fill);
            l.push_str(&line[end..]);
            out.push_str(&l);
            out.push('\n');
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Load a raster file and trace it (the `bookart vectorize <raster>` path). Any format the `image`
/// crate reads; alpha honoured (flattened onto white).
pub fn trace_file(path: &std::path::Path, tint: [u8; 3], dpi: u32) -> Result<String> {
    let img = image::open(path).with_context(|| format!("reading {}", path.display()))?.to_rgba8();
    trace_rgba(&img, tint, dpi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn traces_a_bilevel_ornament_to_svg_paths() {
        // A transparent page with a solid black square → at least one traced path.
        let mut img = RgbaImage::from_pixel(64, 64, Rgba([0, 0, 0, 0]));
        for y in 16..48 {
            for x in 16..48 {
                img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }
        let svg = trace_rgba(&img, [17, 17, 17], 300).unwrap();
        assert!(svg.contains("<svg"), "has an svg root");
        assert!(svg.contains("<path"), "traced at least one path");
        assert!(svg.contains("width=\"5.42mm\""), "physical print size stamped: {}", &svg[..svg.find('>').unwrap()]);
        assert!(svg.contains("fill=\"#111111\""), "paths retinted to the ink colour");
    }
}
