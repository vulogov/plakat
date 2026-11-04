//! THE SIGHT (7.17.0, set 1; Cargo feature `aqua-sight`): the semantic brain's eye for `--medium watercolour`.
//!
//! The water engine (`Canvas::aqua_inject_with` / `aqua_settle_full`) is final; what it still lacked was a brain
//! that SEES. The parser's focus (`aqua_engine_pass`) was the subject's matte when the plan had found a face and,
//! for every other picture, the top third of the Sobel gradient energy — a landscape, a ship, a tunnel painted by
//! where the pixels were busy, not by what they were. The sight is a per-pixel SALIENCY map `[0, 1]` fused from
//! the vision models the studio already runs — U²-Net's salient-object matte, Depth-Anything-V2's relative depth,
//! the SCRFD face mask — and the sheet's own low-level cue (the centre-weighted contrast field), handed to the
//! painter as `PaintParams::sight` and blended into the engine's focus by the key `aqua_sight`. The engine is
//! untouched: the sight changes where and how much the brush goes, never how the water behaves. The map never
//! enters the score — the brain's decisions are baked into every touch record, so the replay stays exact without
//! it. Off this feature (or at the key 0) the field stays `None` and every sheet is byte-identical; no other
//! medium reads it.
//!
//! The fusion (`fuse`) is a pure function, testable without weights: the cues combine as a NOISY-OR,
//! `S = 1 − Π(1 − w·c²)`, so any one strong cue makes a pixel salient and the cues never cancel each other — each
//! cue as its CONFIDENCE, squared, so a half-sure matte counts a quarter (U²-Net's soft tail over the water and the
//! waves' contrast no longer add up to a focus on the sea); the
//! result is normalised by its 99th percentile, clamped to `[0, 1]`, and a face is salient outright (the max with
//! the face mask). The weights: the matte 1, the face 1, the NEAR depth ½ (what stands ABOVE THE GROUND PLANE: a
//! row's far quarter is the plane — the sea, the floor, the road recede row by row — and the depth a pixel stands
//! nearer than its row's plane, rescaled to the 99th percentile, is the tier; `near_from_depth`), the low-level
//! contrast ½.

use anyhow::{Context, Result};
use image::{GrayImage, RgbImage};
use std::path::Path;

/// The matte's weight in the noisy-OR.
pub const W_OBJECT: f32 = 1.0;
/// The near depth's weight.
pub const W_NEAR: f32 = 0.5;
/// The low-level contrast field's weight.
pub const W_LOW: f32 = 0.5;

/// The sight's maps, every one `w × h` row-major in `[0, 1]`.
#[derive(Debug, Clone)]
pub struct SightMaps {
    pub w: u32,
    pub h: u32,
    /// The fused saliency — what the painter reads.
    pub saliency: Vec<f32>,
    /// U²-Net's salient-object matte (1 = the object), when it ran and found a usable object.
    pub object: Option<Vec<f32>>,
    /// Depth-Anything-V2's relative depth (1 = nearest), when it ran.
    pub depth: Option<Vec<f32>>,
    /// The near tier — what stands above the ground plane (`near_from_depth`), when the depth ran.
    pub near: Option<Vec<f32>>,
    /// The low-level cue: the centre-weighted contrast field (`painter::focal_field`).
    pub low: Vec<f32>,
    /// The names of the cues that went into the fusion, for the report line.
    pub sources: Vec<&'static str>,
}

fn percentile(values: &[f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[((sorted.len() as f32 - 1.0) * q.clamp(0.0, 1.0)) as usize]
}

/// A salient-object matte is usable when the object is neither a speck nor the whole frame: its coverage above
/// one half is between 2% and 98% of the pixels (the matting path's own gate).
pub fn object_usable(mask: &[f32]) -> bool {
    let covered = mask.iter().filter(|&&m| m > 0.5).count();
    let total = mask.len().max(1);
    covered > total / 50 && covered < total * 49 / 50
}

/// The NEAR tier of a relative depth map (1 = nearest): the depth above its median, rescaled so the 90th
/// percentile is 1 — the foreground alone, the middle distance fading in, the far nothing.
pub fn near_from_depth(depth: &[f32], w: usize, h: usize) -> Vec<f32> {
    if w == 0 || h == 0 || depth.len() != w * h {
        return vec![0.0; depth.len()];
    }
    // THE GROUND PLANE: each row's 25th percentile of depth (its far quarter), smoothed over a 32nd of the height —
    // the sea at a row is the plane, a hull at that row stands nearer than it. (The global median made the whole
    // foreground sea the near tier.) A row more than three quarters object keeps its object as the plane; the
    // matte has it.
    let plane_raw: Vec<f32> = (0..h).map(|y| percentile(&depth[y * w..(y + 1) * w], 0.25)).collect();
    let r = (h / 32).max(1);
    let plane: Vec<f32> = (0..h)
        .map(|y| {
            let (lo, hi) = (y.saturating_sub(r), (y + r).min(h - 1));
            plane_raw[lo..=hi].iter().sum::<f32>() / (hi - lo + 1) as f32
        })
        .collect();
    // Rescaled to the 99th percentile of what stands above the plane (a small boat is a twentieth of the frame; the
    // 90th would be zero); under a fiftieth of the range nothing stands above it.
    let above: Vec<f32> = depth.iter().enumerate().map(|(i, d)| (d - plane[i / w]).max(0.0)).collect();
    let p99 = percentile(&above, 0.99);
    if p99 < 0.02 {
        return vec![0.0; depth.len()];
    }
    above.iter().map(|a| (a / p99).clamp(0.0, 1.0)).collect()
}

/// The fusion: the noisy-OR of the weighted cues, normalised by the 99th percentile, a face salient outright.
/// Every map `w × h`; a cue of the wrong length is ignored.
pub fn fuse(w: u32, h: u32, object: Option<&[f32]>, near: Option<&[f32]>, face: Option<&[f32]>, low: &[f32]) -> Vec<f32> {
    let n = w as usize * h as usize;
    fn fit<'a>(m: Option<&'a [f32]>, n: usize) -> Option<&'a [f32]> {
        m.filter(|m| m.len() == n)
    }
    let (object, near, face) = (fit(object, n), fit(near, n), fit(face, n));
    let low_ok = low.len() == n;
    let mut s: Vec<f32> = (0..n)
        .map(|i| {
            let mut keep = 1.0f32;
            // Each cue as its confidence — squared: a half-sure cue counts a quarter.
            let conf = |v: f32| { let v = v.clamp(0.0, 1.0); v * v };
            if let Some(o) = object {
                keep *= 1.0 - W_OBJECT * conf(o[i]);
            }
            if let Some(d) = near {
                keep *= 1.0 - W_NEAR * conf(d[i]);
            }
            if low_ok {
                keep *= 1.0 - W_LOW * conf(low[i]);
            }
            1.0 - keep
        })
        .collect();
    let hi = percentile(&s, 0.99).max(1e-3);
    for v in &mut s {
        *v = (*v / hi).clamp(0.0, 1.0);
    }
    if let Some(f) = face {
        for (v, f) in s.iter_mut().zip(f) {
            *v = v.max(f.clamp(0.0, 1.0));
        }
    }
    s
}

/// The low-level cue: the centre-weighted contrast field of the picture itself.
pub fn low_level(img: &RgbImage) -> Vec<f32> {
    crate::paint::painter::focal_field(img)
}

fn to_gray(v: &[f32], w: u32, h: u32) -> GrayImage {
    GrayImage::from_fn(w, h, |x, y| image::Luma([(v[(y * w + x) as usize].clamp(0.0, 1.0) * 255.0).round() as u8]))
}

impl SightMaps {
    /// The share of the frame the fused map calls salient (above one half), in percent.
    pub fn salient_percent(&self) -> usize {
        self.saliency.iter().filter(|&&v| v >= 0.5).count() * 100 / self.saliency.len().max(1)
    }

    /// Write the maps as grey PNGs into `dir` (`sight_saliency.png`, `sight_object.png`, `sight_depth.png`,
    /// `sight_low.png`) — the review's montage.
    pub fn dump(&self, dir: &Path) {
        let _ = to_gray(&self.saliency, self.w, self.h).save(dir.join("sight_saliency.png"));
        let _ = to_gray(&self.low, self.w, self.h).save(dir.join("sight_low.png"));
        if let Some(o) = &self.object {
            let _ = to_gray(o, self.w, self.h).save(dir.join("sight_object.png"));
        }
        if let Some(d) = &self.depth {
            let _ = to_gray(d, self.w, self.h).save(dir.join("sight_depth.png"));
        }
        if let Some(d) = &self.near {
            let _ = to_gray(d, self.w, self.h).save(dir.join("sight_near.png"));
        }
    }
}

/// Build the sight for a picture: the models run on the auto device, each cue optional — a model that fails to
/// load or finds nothing usable is left out with a note, and the fusion goes on with the cues that came. `img`
/// is the picture at the paint size `w × h`; `input` its path (the depth pipeline reads the file); `face` the
/// painter's face mask when one was found. Fails only when not even the low-level cue can be built.
pub async fn build(input: &Path, img: &RgbImage, face: Option<&[f32]>) -> Result<(SightMaps, Vec<String>)> {
    let (w, h) = img.dimensions();
    let n = w as usize * h as usize;
    let mut notes: Vec<String> = Vec::new();
    let mut sources: Vec<&'static str> = Vec::new();
    let device = crate::device::select("auto").context("selecting the device for the sight")?;
    let object: Option<Vec<f32>> = match crate::pipelines::matting::Matter::load(&device).await {
        Ok(matter) => match matter.matte(img) {
            Ok(alpha) => {
                let m: Vec<f32> = alpha.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
                if m.len() == n && object_usable(&m) {
                    sources.push("matte");
                    Some(m)
                } else {
                    notes.push("the matte found no usable object (under 2% or over 98% of the frame)".into());
                    None
                }
            }
            Err(e) => {
                notes.push(format!("the matte failed ({e})"));
                None
            }
        },
        Err(e) => {
            notes.push(format!("U²-Net did not load ({e})"));
            None
        }
    };
    let depth: Option<Vec<f32>> = match crate::pipelines::depth::DepthPipeline::load(device.clone()).await {
        Ok(pipe) => match pipe.depth_map(input, w, h) {
            Ok(d) if d.len() == n => {
                sources.push("depth");
                Some(d)
            }
            Ok(_) => {
                notes.push("the depth map came at the wrong size".into());
                None
            }
            Err(e) => {
                notes.push(format!("the depth failed ({e})"));
                None
            }
        },
        Err(e) => {
            notes.push(format!("Depth-Anything did not load ({e})"));
            None
        }
    };
    let near = depth.as_deref().map(|d| near_from_depth(d, w as usize, h as usize));
    let low = low_level(img);
    if low.len() != n {
        anyhow::bail!("the contrast field came at the wrong size");
    }
    sources.push("contrast");
    let face = face.filter(|f| f.len() == n && f.iter().any(|v| *v > 0.35));
    if face.is_some() {
        sources.push("faces");
    }
    let saliency = fuse(w, h, object.as_deref(), near.as_deref(), face, &low);
    Ok((SightMaps { w, h, saliency, object, depth, near, low, sources }, notes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(w: u32, h: u32, x0: u32, x1: u32) -> Vec<f32> {
        (0..w * h).map(|i| if (i % w) >= x0 && (i % w) < x1 { 1.0 } else { 0.0 }).collect()
    }

    #[test]
    fn the_matte_makes_its_object_salient_and_the_rest_dim() {
        let (w, h) = (40u32, 20u32);
        let low = vec![0.2f32; (w * h) as usize];
        let object = block(w, h, 10, 20);
        let s = fuse(w, h, Some(&object), None, None, &low);
        assert_eq!(s.len(), (w * h) as usize);
        assert!(s.iter().all(|v| (0.0..=1.0).contains(v)), "in [0, 1]");
        let inside = s[(5 * w + 15) as usize];
        let outside = s[(5 * w + 30) as usize];
        assert!(inside >= 0.99, "the object salient: {inside}");
        assert!(outside < 0.15, "the ground dim: {outside}");
    }

    #[test]
    fn the_cues_never_cancel_and_a_face_is_salient_outright() {
        let (w, h) = (40u32, 20u32);
        let low: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 / w as f32).collect();
        let near = block(w, h, 0, 10);
        let alone = fuse(w, h, None, None, None, &low);
        let with_near = fuse(w, h, None, Some(&near), None, &low);
        // The normalisation rescales the whole map, so a cue is judged by the ORDER it makes: the near pixel rises
        // against a far one, and the far pixels keep their order among themselves.
        let (i_in, i_out, j_out) = ((3 * w + 2) as usize, (3 * w + 20) as usize, (3 * w + 35) as usize);
        assert!(with_near[i_in] - with_near[i_out] > alone[i_in] - alone[i_out], "the near tier lifts its pixels against the rest");
        assert!(with_near[i_in] > alone[i_in], "…and above where it stood");
        assert!(with_near[j_out] > with_near[i_out], "the far pixels keep their order");
        let face = block(w, h, 30, 32);
        let with_face = fuse(w, h, None, None, Some(&face), &low);
        assert!(with_face[(3 * w + 31) as usize] >= 1.0 - 1e-6, "a face is salient outright");
    }

    #[test]
    fn the_near_tier_is_what_stands_above_the_ground_plane() {
        // A sea receding row by row (the depth rising down the frame, flat across a row) is the plane everywhere:
        // no pixel of it is near, however close its row. A hull standing nearer than its rows is the near tier.
        let (w, h) = (40usize, 40usize);
        let sea: Vec<f32> = (0..w * h).map(|i| (i / w) as f32 / (h - 1) as f32).collect();
        let near = near_from_depth(&sea, w, h);
        assert!(near.iter().all(|v| *v == 0.0), "the plane alone is nowhere near");
        let hull: Vec<f32> = sea.iter().enumerate().map(|(i, d)| if (i % w) >= 10 && (i % w) < 25 && (i / w) >= 20 && (i / w) < 28 { d + 0.3 } else { *d }).collect();
        let near = near_from_depth(&hull, w, h);
        assert!(near[24 * w + 15] > 0.9, "the hull stands above its row's plane: {}", near[24 * w + 15]);
        assert!(near[24 * w + 35] < 1e-3 && near[35 * w + 15] < 1e-3, "the sea beside and below it is the plane: {} {}", near[24 * w + 35], near[35 * w + 15]);
        assert!(near_from_depth(&sea, 7, 3).iter().all(|v| *v == 0.0), "a map of the wrong size is nowhere near");
        assert!(object_usable(&block(40, 20, 10, 20)));
        assert!(!object_usable(&block(40, 20, 0, 40)), "the whole frame is no object");
        assert!(!object_usable(&vec![0.0; 800]), "nothing is no object");
    }

    #[test]
    fn a_cue_of_the_wrong_length_is_ignored() {
        let (w, h) = (8u32, 4u32);
        let low = vec![0.5f32; 32];
        let wrong = vec![1.0f32; 7];
        assert_eq!(fuse(w, h, Some(&wrong), Some(&wrong), Some(&wrong), &low), fuse(w, h, None, None, None, &low));
    }
}
