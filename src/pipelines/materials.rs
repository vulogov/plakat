//! A material map by words: `fur: hare; glass: lamp lens; skin` → for every point of the picture, what it is
//! made of.
//!
//! Used by `plakat paint --medium durer --materials …`. The artist names the materials and what carries them;
//! the masks are found for him, by the two models `plakat remove --what` already uses: OWL-ViT finds every
//! instance of the thing named, MobileSAM outlines each. `skin` needs no thing named: the faces are found by
//! the face detector (SCRFD, the one `--preserve-face` uses) and outlined the same way. No new weights,
//! nothing to draw.
//!
//! The map is one byte a pixel — 0 for "nothing said", else a kind's id — and is saved as a picture in the
//! kinds' colours, so it can be looked at, retouched and given back as a file.

use anyhow::{Context, Result, bail};
use candle_core::Device;
use image::{Rgb, RgbImage};
use std::path::Path;

/// The kinds a plate cuts differently: name, id, colour in the saved map.
pub const KINDS: [(&str, u8, [u8; 3]); 4] = [("fur", 1, [200, 120, 40]), ("glass", 2, [70, 160, 230]), ("metal", 3, [170, 170, 180]), ("skin", 4, [230, 180, 150])];
const SKIN: u8 = 4;

/// How surely the detector must see a thing: firmly, or — when nothing is — at least faintly.
const FIRM: f32 = 0.1;
const FAINT: f32 = 0.03;
/// How far a face's box is widened before it is outlined: the detector's box runs from the brows to the chin,
/// the skin from the hairline to the neck. Up, to each side, down — as a share of the box.
const FACE_ROOM: [f32; 3] = [0.35, 0.2, 0.3];

/// `kind: thing, thing; kind: thing` → `(kind id, thing)` in the order written, which is also their rank: where
/// two masks overlap the one written first keeps the pixel. `skin`, with or without things after it, also
/// gives `(skin, "")` — the faces, found by the face detector, not by a word.
pub fn parse(spec: &str) -> Result<Vec<(u8, String)>> {
    let mut out = Vec::new();
    for part in spec.split(';').map(str::trim).filter(|p| !p.is_empty()) {
        let (kind, things) = part.split_once(':').unwrap_or((part, ""));
        let kind = kind.trim().to_lowercase();
        let Some(&(_, id, _)) = KINDS.iter().find(|k| k.0 == kind) else {
            if things.is_empty() {
                bail!("--materials: `{part}` is not `kind: thing, thing` (kinds: {})", KINDS.map(|k| k.0).join(", "));
            }
            bail!("--materials: unknown kind `{kind}` (one of: {})", KINDS.map(|k| k.0).join(", "));
        };
        let before = out.len();
        if id == SKIN {
            out.push((id, String::new()));
        }
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
    let mut map = vec![0u8; (w * h) as usize];
    let mut owl = None;
    for (kind, thing) in things {
        // The boxes the thing is in, and whether they were seen only faintly.
        let (boxes, faint): (Vec<[f32; 4]>, bool) = if thing.is_empty() {
            (faces(image, device).await?, false)
        } else {
            let owl = match &owl {
                Some(o) => o,
                None => owl.insert(crate::pipelines::owlvit::OwlViT::load_pretrained(device).await?),
            };
            // A whole object answers firmly; a PART of one (a lamp on a machine) answers faintly. Take the firm
            // answers when there are any, else the faint ones that stand with the best of them.
            let mut dets = owl.detect_all(image, thing, FAINT, 8)?;
            let best = dets.iter().map(|d| d.score).fold(0f32, f32::max);
            dets.retain(|d| d.score >= if best >= FIRM { FIRM } else { best * 0.6 });
            (dets.iter().map(|d| [d.x0, d.y0, d.x1, d.y1]).collect(), best < FIRM)
        };
        let what = if thing.is_empty() { "faces" } else { thing.as_str() };
        if boxes.is_empty() {
            println!("·  materials: no {} found — left out{}", if thing.is_empty() { "face".to_string() } else { format!("\"{thing}\"") }, if thing.is_empty() { "" } else { " (try a plainer noun)" });
            continue;
        }
        let (found, mut claimed) = (boxes.len(), 0usize);
        for [x0, y0, x1, y1] in boxes {
            // A point at the box's centre says what; points just outside its sides say where it ends.
            let pt = |x: f32, y: f32, fg: bool| crate::pipelines::sam::PointPrompt { x: x.clamp(0.0, (w - 1) as f32) as f64, y: y.clamp(0.0, (h - 1) as f32) as f64, foreground: fg };
            let (cx, cy, m) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, 6.0);
            let prompts = [pt(cx, cy, true), pt(cx, y0 - m, false), pt(cx, y1 + m, false), pt(x0 - m, cy, false), pt(x1 + m, cy, false)];
            // A face the outliner cannot read is still a face: its box is taken whole.
            let mask = match crate::pipelines::sam::build_selection_mask(image, &prompts, device).await {
                Ok(mask) => Some(mask),
                Err(_) if thing.is_empty() => None,
                Err(_) => continue,
            };
            for y in y0.max(0.0) as u32..(y1.ceil() as u32).min(h) {
                for x in x0.max(0.0) as u32..(x1.ceil() as u32).min(w) {
                    let i = (y * w + x) as usize;
                    if map[i] == 0 && mask.as_ref().map_or(true, |m| m.get_pixel(x, y).0[0] > 127) {
                        map[i] = *kind;
                        claimed += 1;
                    }
                }
            }
        }
        println!("·  materials: {what} — {found} found{}, {:.1}% of the picture", if faint { " (faintly)" } else { "" }, claimed as f32 * 100.0 / (w * h) as f32);
    }
    Ok((w, h, map))
}

/// The faces in `image`, by the face detector, each box widened by [`FACE_ROOM`] to take in the forehead and
/// the neck: `[x0, y0, x1, y1]` in the picture's pixels. Empty when none is found or the detector is not
/// to be had (the default weights could not be fetched) — the latter is said.
async fn faces(image: &Path, device: &Device) -> Result<Vec<[f32; 4]>> {
    let Some(weights) = crate::pipelines::scrfd::resolve_scrfd_weights().await.context("resolving the face-detector weights")? else {
        println!("·  materials: the face detector's weights are not available — skin left out");
        return Ok(Vec::new());
    };
    let det = crate::pipelines::scrfd::SCRFDDetector::load(&weights, crate::pipelines::scrfd::SCRFDConfig::default(), device, candle_core::DType::F32).context("loading the face detector")?;
    let (w, h) = image::image_dimensions(image)?;
    Ok(det
        .detect(image)
        .context("detecting faces")?
        .iter()
        .map(|f| {
            let [x0, y0, x1, y1] = f.bbox;
            let (bw, bh) = (x1 - x0, y1 - y0);
            [(x0 - bw * FACE_ROOM[1]).max(0.0), (y0 - bh * FACE_ROOM[0]).max(0.0), (x1 + bw * FACE_ROOM[1]).min(w as f32), (y1 + bh * FACE_ROOM[2]).min(h as f32)]
        })
        .collect())
}

/// The FEATURES of the faces in `image`, by the same detector: for each face its eyes, the tip of its nose and
/// the corners of its mouth (left eye, right eye, nose, left mouth, right mouth), as shares of the picture's
/// width and height. A far face is drawn from these — a stroke where each feature is — not traced from a few
/// dozen pixels of shading. Empty when none is found or the detector is not to be had.
pub async fn face_features(image: &Path, device: &Device) -> Result<Vec<[[f32; 2]; 5]>> {
    let Some(weights) = crate::pipelines::scrfd::resolve_scrfd_weights().await.context("resolving the face-detector weights")? else {
        return Ok(Vec::new());
    };
    let det = crate::pipelines::scrfd::SCRFDDetector::load(&weights, crate::pipelines::scrfd::SCRFDConfig::default(), device, candle_core::DType::F32).context("loading the face detector")?;
    let (w, h) = image::image_dimensions(image)?;
    Ok(det.detect(image).context("detecting faces")?.iter().map(|f| f.landmarks.map(|p| [p[0] / w as f32, p[1] / h as f32])).collect())
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
        assert!(parse("fur:").is_err() && parse("").is_err() && parse("hare").is_err() && parse("fur").is_err());
    }

    #[test]
    fn skin_is_the_faces_and_whatever_else_is_named() {
        assert_eq!(parse("skin").unwrap(), vec![(4, String::new())]);
        assert_eq!(parse("fur: hare; skin: hands").unwrap(), vec![(1, "hare".to_string()), (4, String::new()), (4, "hands".to_string())]);
    }

    #[test]
    fn a_material_map_survives_the_picture() {
        let map = vec![0u8, 1, 2, 3, 4, 0];
        assert_eq!(from_png(&to_png(&map, 3, 2)), map);
    }
}
