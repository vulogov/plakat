//! A material map by words: `fur: hare; glass: lamp lens` → for every point of the picture, what it is made of.
//!
//! Used by `plakat paint --medium durer --materials …`. The artist names the materials and what carries them;
//! the masks are found for him, by the two models `plakat remove --what` already uses: OWL-ViT finds every
//! instance of the thing named, MobileSAM outlines each. No new weights, nothing to draw.
//!
//! The map is one byte a pixel — 0 for "nothing said", else a kind's id — and is saved as a picture in the
//! kinds' colours, so it can be looked at, retouched and given back as a file.

use anyhow::{Context, Result, bail};
use candle_core::Device;
use image::{Rgb, RgbImage};
use std::path::Path;

/// The kinds a plate cuts differently: name, id, colour in the saved map.
pub const KINDS: [(&str, u8, [u8; 3]); 3] = [("fur", 1, [200, 120, 40]), ("glass", 2, [70, 160, 230]), ("metal", 3, [170, 170, 180])];

/// How surely the detector must see a thing: firmly, or — when nothing is — at least faintly.
const FIRM: f32 = 0.1;
const FAINT: f32 = 0.03;

/// `kind: thing, thing; kind: thing` → `(kind id, thing)` in the order written, which is also their rank: where
/// two masks overlap the one written first keeps the pixel.
pub fn parse(spec: &str) -> Result<Vec<(u8, String)>> {
    let mut out = Vec::new();
    for part in spec.split(';').map(str::trim).filter(|p| !p.is_empty()) {
        let (kind, things) = part.split_once(':').with_context(|| format!("--materials: `{part}` is not `kind: thing, thing`"))?;
        let kind = kind.trim().to_lowercase();
        let Some(&(_, id, _)) = KINDS.iter().find(|k| k.0 == kind) else {
            bail!("--materials: unknown kind `{kind}` (one of: {})", KINDS.map(|k| k.0).join(", "));
        };
        let before = out.len();
        out.extend(things.split(',').map(str::trim).filter(|t| !t.is_empty()).map(|t| (id, t.to_string())));
        if out.len() == before {
            bail!("--materials: `{kind}` names nothing");
        }
    }
    if out.is_empty() {
        bail!("--materials: nothing named");
    }
    Ok(out)
}

/// The map for `image`, at the image's own size: `(w, h, kind per pixel)`. A thing that is not found is reported
/// and left out; the rest of the map stands.
pub async fn material_map(image: &Path, things: &[(u8, String)], device: &Device) -> Result<(u32, u32, Vec<u8>)> {
    let (w, h) = image::image_dimensions(image).with_context(|| format!("opening {}", image.display()))?;
    let owl = crate::pipelines::owlvit::OwlViT::load_pretrained(device).await?;
    let mut map = vec![0u8; (w * h) as usize];
    for (kind, thing) in things {
        // A whole object answers firmly; a PART of one (a lamp on a machine) answers faintly. Take the firm
        // answers when there are any, else the faint ones that stand with the best of them.
        let mut dets = owl.detect_all(image, thing, FAINT, 8)?;
        let best = dets.iter().map(|d| d.score).fold(0f32, f32::max);
        dets.retain(|d| d.score >= if best >= FIRM { FIRM } else { best * 0.6 });
        if dets.is_empty() {
            println!("·  materials: no \"{thing}\" found — left out (try a plainer noun)");
            continue;
        }
        let mut claimed = 0usize;
        for det in &dets {
            // A point at the box's centre says what; points just outside its sides say where it ends.
            let pt = |x: f32, y: f32, fg: bool| crate::pipelines::sam::PointPrompt { x: x.clamp(0.0, (w - 1) as f32) as f64, y: y.clamp(0.0, (h - 1) as f32) as f64, foreground: fg };
            let (cx, cy, m) = ((det.x0 + det.x1) / 2.0, (det.y0 + det.y1) / 2.0, 6.0);
            let prompts = [pt(cx, cy, true), pt(cx, det.y0 - m, false), pt(cx, det.y1 + m, false), pt(det.x0 - m, cy, false), pt(det.x1 + m, cy, false)];
            let Ok(mask) = crate::pipelines::sam::build_selection_mask(image, &prompts, device).await else { continue };
            for y in det.y0.max(0.0) as u32..(det.y1.ceil() as u32).min(h) {
                for x in det.x0.max(0.0) as u32..(det.x1.ceil() as u32).min(w) {
                    let i = (y * w + x) as usize;
                    if map[i] == 0 && mask.get_pixel(x, y).0[0] > 127 {
                        map[i] = *kind;
                        claimed += 1;
                    }
                }
            }
        }
        println!("·  materials: \"{thing}\" — {} found{}, {:.1}% of the picture", dets.len(), if best < FIRM { " (faintly)" } else { "" }, claimed as f32 * 100.0 / (w * h) as f32);
    }
    Ok((w, h, map))
}

pub fn to_png(map: &[u8], w: u32, h: u32) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| Rgb(KINDS.iter().find(|k| k.1 == map[(y * w + x) as usize]).map_or([0, 0, 0], |k| k.2)))
}

/// A saved map read back: every pixel is the kind whose colour it is nearest, or nothing where it is near black.
pub fn from_png(img: &RgbImage) -> Vec<u8> {
    img.pixels()
        .map(|p| {
            let far = |c: [u8; 3]| (0..3).map(|j| (p.0[j] as i32 - c[j] as i32).pow(2)).sum::<i32>();
            KINDS.iter().map(|k| (far(k.2), k.1)).chain([(far([0, 0, 0]), 0)]).min().map_or(0, |(_, id)| id)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spec_names_kinds_and_things_in_order() {
        let got = parse("fur: hare, rabbit; glass: lamp lens").unwrap();
        assert_eq!(got, vec![(1, "hare".to_string()), (1, "rabbit".to_string()), (2, "lamp lens".to_string())]);
        assert!(parse("velvet: curtain").is_err(), "an unknown kind is refused");
        assert!(parse("fur:").is_err() && parse("").is_err() && parse("hare").is_err());
    }

    #[test]
    fn a_material_map_survives_the_picture() {
        let map = vec![0u8, 1, 2, 3];
        assert_eq!(from_png(&to_png(&map, 2, 2)), map);
    }
}
