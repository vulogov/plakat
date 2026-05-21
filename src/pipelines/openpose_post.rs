//! OpenPose post-processing: heatmap+PAF → skeleton image.
//!
//! Ported from lllyasviel's `util.py` + `body.py`. The full reference
//! supports multi-scale inference and Gaussian-smoothed peak finding;
//! we ship a simplified single-scale variant with raw-heatmap NMS to
//! keep the dependency surface to candle + image alone. Quality is
//! adequate for ControlNet conditioning (which is "where do the body
//! parts roughly go", not pixel-perfect detection).
//!
//! Simplifications relative to lllyasviel's reference:
//!   * Single-scale forward (the user-supplied detect_resolution),
//!     no scale-search pyramid.
//!   * NMS over raw heatmap rather than `gaussian_filter(σ=3)`. The
//!     net's heatmap output is already smooth (1/8-resolution); the
//!     extra smoothing in the reference exists for high-resolution
//!     multi-scale averaging which we don't do.
//!   * Greedy bipartite matching (sort candidate connections by
//!     score, assign each peak to at most one connection per limb)
//!     instead of Hungarian / Munkres. The reference itself notes
//!     this as a common acceptable approximation.
//!
//! Output: an RGB image with a coloured skeleton on a black background,
//! matching the ControlNet-OpenPose training data convention.

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};

// 18 body keypoints (0-indexed): nose, neck, R/L shoulders, elbows,
// wrists, hips, knees, ankles, R/L eyes, R/L ears. Matches lllyasviel's
// body_25-derived ordering.
pub const NUM_KEYPOINTS: usize = 18;
const NUM_LIMBS: usize = 19;

/// Each limb connects two keypoints (0-indexed). Mirrors lllyasviel's
/// `limbSeq` with 1-indexing dropped.
const LIMB_SEQ: [[usize; 2]; NUM_LIMBS] = [
    [1, 2],   [1, 5],   [2, 3],   [3, 4],   [5, 6],
    [6, 7],   [1, 8],   [8, 9],   [9, 10],  [1, 11],
    [11, 12], [12, 13], [1, 0],   [0, 14],  [14, 16],
    [0, 15],  [15, 17], [2, 16],  [5, 17],
];

/// PAF channel pair feeding each limb. Mirrors lllyasviel's `mapIdx`
/// minus 1-indexing offset (the heatmap channels are 0..18, then
/// 19..56 are the PAF pairs, but the model emits PAFs as a separate
/// tensor of shape (1, 38, h, w); so the indices below are 0..38).
const PAF_IDX: [[usize; 2]; NUM_LIMBS] = [
    [12, 13], [20, 21], [14, 15], [16, 17], [22, 23],
    [24, 25], [0, 1],   [2, 3],   [4, 5],   [6, 7],
    [8, 9],   [10, 11], [28, 29], [30, 31], [34, 35],
    [32, 33], [36, 37], [18, 19], [26, 27],
];

/// Distinct colours for each limb. Matches lllyasviel's `colors`.
const LIMB_COLORS: [[u8; 3]; NUM_LIMBS] = [
    [255, 0, 0],     [255, 85, 0],   [255, 170, 0],  [255, 255, 0],  [170, 255, 0],
    [85, 255, 0],    [0, 255, 0],    [0, 255, 85],   [0, 255, 170],  [0, 255, 255],
    [0, 170, 255],   [0, 85, 255],   [0, 0, 255],    [85, 0, 255],   [170, 0, 255],
    [255, 0, 255],   [255, 0, 170],  [255, 0, 85],   [128, 64, 128],
];

/// A peak found via heatmap NMS.
#[derive(Debug, Clone, Copy)]
pub struct Peak {
    pub x: f32,
    pub y: f32,
    pub score: f32,
}

/// Run NMS on each of the 18 keypoint heatmaps. `heatmap` is row-major
/// `(channels, height, width)` with `channels = 19` (keypoints 0..=17
/// + background channel 18 we ignore here).
///
/// A pixel is a peak if its value is greater than each of its 8
/// neighbours AND above `threshold`. Linear-time scan; no Gaussian
/// smoothing (see module comment).
pub fn find_peaks(
    heatmap: &[f32],
    h: usize,
    w: usize,
    threshold: f32,
) -> Vec<Vec<Peak>> {
    let mut out = Vec::with_capacity(NUM_KEYPOINTS);
    for kp in 0..NUM_KEYPOINTS {
        let base = kp * h * w;
        let mut peaks = Vec::new();
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let v = heatmap[base + y * w + x];
                if v < threshold {
                    continue;
                }
                let mut is_max = true;
                'nbr: for dy in 0..=2_isize {
                    for dx in 0..=2_isize {
                        if dy == 1 && dx == 1 {
                            continue;
                        }
                        let ny = (y as isize + dy - 1) as usize;
                        let nx = (x as isize + dx - 1) as usize;
                        if heatmap[base + ny * w + nx] > v {
                            is_max = false;
                            break 'nbr;
                        }
                    }
                }
                if is_max {
                    peaks.push(Peak {
                        x: x as f32,
                        y: y as f32,
                        score: v,
                    });
                }
            }
        }
        out.push(peaks);
    }
    out
}

/// One scored candidate connection between a peak in `kp_a` and a
/// peak in `kp_b`. Carries indices into `peaks_per_kp[kp_a]` and
/// `peaks_per_kp[kp_b]`.
#[derive(Debug, Clone, Copy)]
pub struct CandidateConn {
    pub i_a: usize,
    pub i_b: usize,
    pub score: f32,
}

/// Score every candidate connection via the standard PAF line
/// integral: sample `n_samples` points along the line from peak A to
/// peak B, dot each sample's PAF vector with the unit A→B vector,
/// and check at least 80% of samples score above `paf_threshold`.
///
/// `paf` is `(38, h, w)` row-major.
pub fn score_connections(
    paf: &[f32],
    h: usize,
    w: usize,
    peaks_per_kp: &[Vec<Peak>],
    paf_threshold: f32,
    min_pass_fraction: f32,
    n_samples: usize,
) -> Vec<Vec<CandidateConn>> {
    let mut out = Vec::with_capacity(NUM_LIMBS);
    for limb in 0..NUM_LIMBS {
        let [a_kp, b_kp] = LIMB_SEQ[limb];
        let [px_idx, py_idx] = PAF_IDX[limb];
        let peaks_a = &peaks_per_kp[a_kp];
        let peaks_b = &peaks_per_kp[b_kp];
        let mut cands: Vec<CandidateConn> = Vec::new();
        for (i_a, pa) in peaks_a.iter().enumerate() {
            for (i_b, pb) in peaks_b.iter().enumerate() {
                let dx = pb.x - pa.x;
                let dy = pb.y - pa.y;
                let len = (dx * dx + dy * dy).sqrt();
                if len < 1.0 {
                    continue;
                }
                let ux = dx / len;
                let uy = dy / len;
                let mut sum = 0.0_f32;
                let mut pass = 0_usize;
                for s in 0..n_samples {
                    let t = s as f32 / (n_samples - 1).max(1) as f32;
                    let sx = pa.x + dx * t;
                    let sy = pa.y + dy * t;
                    let xi = (sx.round() as isize).clamp(0, w as isize - 1) as usize;
                    let yi = (sy.round() as isize).clamp(0, h as isize - 1) as usize;
                    let vx = paf[px_idx * h * w + yi * w + xi];
                    let vy = paf[py_idx * h * w + yi * w + xi];
                    let dotp = vx * ux + vy * uy;
                    sum += dotp;
                    if dotp > paf_threshold {
                        pass += 1;
                    }
                }
                let avg = sum / n_samples as f32;
                let pass_frac = pass as f32 / n_samples as f32;
                if pass_frac >= min_pass_fraction && avg > 0.0 {
                    cands.push(CandidateConn {
                        i_a,
                        i_b,
                        // Boost score by both line-integral mean and
                        // the per-peak heatmap response so a strong
                        // peak with a so-so PAF still wins over a
                        // weak peak.
                        score: avg + pa.score.min(pb.score),
                    });
                }
            }
        }
        // Greedy bipartite matching: sort by score desc; assign
        // each peak at most once per limb.
        cands.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let mut used_a = vec![false; peaks_a.len()];
        let mut used_b = vec![false; peaks_b.len()];
        let mut chosen = Vec::new();
        for c in cands {
            if !used_a[c.i_a] && !used_b[c.i_b] {
                used_a[c.i_a] = true;
                used_b[c.i_b] = true;
                chosen.push(c);
            }
        }
        out.push(chosen);
    }
    out
}

/// Render the detected skeleton onto a black background at
/// `(width, height)`. Coordinates in `peaks_per_kp` and
/// `connections_per_limb` are in heatmap space; we scale by
/// `stride` (the downsample factor between input image and
/// heatmap — 8 for OpenPose).
pub fn draw_skeleton(
    width: usize,
    height: usize,
    stride: usize,
    peaks_per_kp: &[Vec<Peak>],
    connections_per_limb: &[Vec<CandidateConn>],
) -> Result<RgbImage> {
    let mut img = RgbImage::new(width as u32, height as u32);
    // Background already black (RgbImage::new zero-fills).
    let s = stride as f32;

    // Draw limbs first so keypoint dots paint on top.
    for (limb, conns) in connections_per_limb.iter().enumerate() {
        let [a_kp, b_kp] = LIMB_SEQ[limb];
        let color = Rgb(LIMB_COLORS[limb]);
        for c in conns {
            let pa = peaks_per_kp[a_kp][c.i_a];
            let pb = peaks_per_kp[b_kp][c.i_b];
            draw_line(
                &mut img,
                pa.x * s,
                pa.y * s,
                pb.x * s,
                pb.y * s,
                color,
                4,
            );
        }
    }

    // Keypoint dots.
    for (kp, peaks) in peaks_per_kp.iter().enumerate() {
        let color = Rgb(LIMB_COLORS[kp % NUM_LIMBS]);
        for p in peaks {
            draw_filled_circle(&mut img, p.x * s, p.y * s, 4, color);
        }
    }
    Ok(img)
}

/// Simple anti-aliased-ish line drawer using Bresenham-style stepping
/// + radial fill for thickness. Pure stdlib + image.
fn draw_line(img: &mut RgbImage, x0: f32, y0: f32, x1: f32, y1: f32, color: Rgb<u8>, thickness: i32) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let steps = (dx.abs().max(dy.abs())).ceil() as usize;
    if steps == 0 {
        draw_filled_circle(img, x0, y0, thickness, color);
        return;
    }
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let x = x0 + dx * t;
        let y = y0 + dy * t;
        draw_filled_circle(img, x, y, thickness, color);
    }
}

fn draw_filled_circle(img: &mut RgbImage, cx: f32, cy: f32, radius: i32, color: Rgb<u8>) {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let rsq = (radius * radius) as f32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if ((dx * dx + dy * dy) as f32) > rsq {
                continue;
            }
            let x = cx as i32 + dx;
            let y = cy as i32 + dy;
            if x < 0 || y < 0 || x >= w || y >= h {
                continue;
            }
            img.put_pixel(x as u32, y as u32, color);
        }
    }
}

/// End-to-end "heatmap + PAF → skeleton RGB image" for one frame.
///
/// `heatmap` is `(19, h, w)` row-major, `paf` is `(38, h, w)` row-major,
/// both at the model's 1/8 spatial resolution (heatmap_h = input_h/8).
/// `out_w` / `out_h` are the desired final skeleton image size — the
/// keypoint coordinates scale linearly via `stride = 8` and then the
/// output is resized via the caller after this returns the
/// heatmap-resolution image.
pub fn render_skeleton(
    heatmap: &[f32],
    paf: &[f32],
    map_h: usize,
    map_w: usize,
    out_w: u32,
    out_h: u32,
    stride: usize,
) -> Result<RgbImage> {
    // Tunables — match lllyasviel's `thre1` / `thre2`.
    const PEAK_THRESHOLD: f32 = 0.1;
    const PAF_THRESHOLD: f32 = 0.05;
    const PAF_MIN_PASS_FRACTION: f32 = 0.8;
    const PAF_SAMPLES: usize = 10;

    let peaks = find_peaks(heatmap, map_h, map_w, PEAK_THRESHOLD);
    let conns = score_connections(
        paf,
        map_h,
        map_w,
        &peaks,
        PAF_THRESHOLD,
        PAF_MIN_PASS_FRACTION,
        PAF_SAMPLES,
    );
    let img = draw_skeleton(out_w as usize, out_h as usize, stride, &peaks, &conns)
        .context("drawing OpenPose skeleton")?;
    Ok(img)
}
