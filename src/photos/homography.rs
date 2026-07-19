//! Feature-matched panorama stitching (the "true" stitcher). Unlike the translation aligner in
//! [`super::stitch`], this estimates a full **projective homography** between overlapping frames — so
//! it corrects rotation and perspective, not just a shift. Pipeline: FAST corners (imageproc) →
//! normalised-patch descriptors → NCC match with a Lowe ratio test → RANSAC homography (normalised
//! 4-point DLT + Gaussian elimination, refit on the inliers) → warp (imageproc) + feathered blend.
//!
//! It's deliberately self-contained (a small LCG for RANSAC, our own 3×3 linear algebra) and returns
//! `None` when it can't find a confident model, so the caller falls back to the simpler stitchers
//! rather than emitting a mangled canvas.

use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use imageproc::corners::corners_fast9;
use imageproc::geometric_transformations::{warp_into, Interpolation, Projection};

/// Max corners kept per image (top-scoring), and the descriptor patch radius.
const MAX_CORNERS: usize = 320;
const PATCH_R: i32 = 4; // 9×9 patch
const RANSAC_ITERS: usize = 800;
const INLIER_PX: f32 = 3.0; // reprojection tolerance (in the downscaled detection frame)
const DETECT_MAX_DIM: u32 = 700; // downscale long side for detection

/// A deterministic xorshift RNG — RANSAC needs randomness but we want reproducible stitches/tests.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A keypoint with a mean/σ-normalised patch descriptor, in **detection-frame** coordinates.
struct Feature {
    x: f32,
    y: f32,
    desc: Vec<f32>,
}

fn to_gray_scaled(img: &RgbImage) -> (GrayImage, f32) {
    let (w, h) = (img.width(), img.height());
    let long = w.max(h);
    let scale = if long > DETECT_MAX_DIM { long as f32 / DETECT_MAX_DIM as f32 } else { 1.0 };
    let gray = image::imageops::grayscale(img);
    if scale > 1.0 {
        let (nw, nh) = ((w as f32 / scale) as u32, (h as f32 / scale) as u32);
        (image::imageops::resize(&gray, nw.max(1), nh.max(1), image::imageops::FilterType::Triangle), scale)
    } else {
        (gray, 1.0)
    }
}

fn features(img: &RgbImage) -> (Vec<Feature>, f32) {
    let (gray, scale) = to_gray_scaled(img);
    let (w, h) = (gray.width() as i32, gray.height() as i32);
    let mut corners = corners_fast9(&gray, 20);
    corners.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    corners.truncate(MAX_CORNERS);
    let mut feats = Vec::with_capacity(corners.len());
    for c in corners {
        let (cx, cy) = (c.x as i32, c.y as i32);
        if cx - PATCH_R < 0 || cy - PATCH_R < 0 || cx + PATCH_R >= w || cy + PATCH_R >= h {
            continue;
        }
        let mut patch = Vec::with_capacity(81);
        for dy in -PATCH_R..=PATCH_R {
            for dx in -PATCH_R..=PATCH_R {
                patch.push(gray.get_pixel((cx + dx) as u32, (cy + dy) as u32).0[0] as f32);
            }
        }
        let mean = patch.iter().sum::<f32>() / patch.len() as f32;
        let var = patch.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / patch.len() as f32;
        let sd = var.sqrt().max(1e-3);
        for v in &mut patch {
            *v = (*v - mean) / sd;
        }
        feats.push(Feature { x: c.x as f32, y: c.y as f32, desc: patch });
    }
    (feats, scale)
}

/// NCC of two σ-normalised patches (== normalised dot product / N).
fn ncc(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / a.len() as f32
}

/// Match features by best-NCC with a Lowe ratio test. Returns `(add_pt, base_pt)` in detection frame.
fn match_features(fa: &[Feature], fb: &[Feature]) -> Vec<((f32, f32), (f32, f32))> {
    let mut out = Vec::new();
    for a in fa {
        let (mut best, mut second, mut best_j) = (-2.0f32, -2.0f32, usize::MAX);
        for (j, b) in fb.iter().enumerate() {
            let s = ncc(&a.desc, &b.desc);
            if s > best {
                second = best;
                best = s;
                best_j = j;
            } else if s > second {
                second = s;
            }
        }
        // Keep a fairly permissive set of candidates — RANSAC filters the outliers, so recall matters
        // more than precision here (a strict ratio test starves it on self-similar scenes).
        if best_j != usize::MAX && best > 0.6 && best - second > 0.02 {
            out.push(((a.x, a.y), (fb[best_j].x, fb[best_j].y)));
        }
    }
    out
}

// ---- 3×3 linear algebra (row-major) ----------------------------------------------------------

fn mat3_mul(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut m = [0f32; 9];
    for r in 0..3 {
        for c in 0..3 {
            m[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
        }
    }
    m
}

fn mat3_inv(m: &[f32; 9]) -> Option<[f32; 9]> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-12 {
        return None;
    }
    let id = 1.0 / det;
    Some([
        (m[4] * m[8] - m[5] * m[7]) * id,
        (m[2] * m[7] - m[1] * m[8]) * id,
        (m[1] * m[5] - m[2] * m[4]) * id,
        (m[5] * m[6] - m[3] * m[8]) * id,
        (m[0] * m[8] - m[2] * m[6]) * id,
        (m[2] * m[3] - m[0] * m[5]) * id,
        (m[3] * m[7] - m[4] * m[6]) * id,
        (m[1] * m[6] - m[0] * m[7]) * id,
        (m[0] * m[4] - m[1] * m[3]) * id,
    ])
}

/// Apply a 3×3 homography to a point (with the homogeneous divide).
fn apply_h(h: &[f32; 9], x: f32, y: f32) -> (f32, f32) {
    let w = h[6] * x + h[7] * y + h[8];
    let w = if w.abs() < 1e-12 { 1e-12 } else { w };
    ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w)
}

/// Hartley isotropic normalisation: translate to the centroid, scale so the mean distance is √2.
/// Returns the 3×3 similarity transform.
fn normalize(pts: &[(f32, f32)]) -> [f32; 9] {
    let n = pts.len().max(1) as f32;
    let (cx, cy) = (pts.iter().map(|p| p.0).sum::<f32>() / n, pts.iter().map(|p| p.1).sum::<f32>() / n);
    let mean_d = pts.iter().map(|p| ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt()).sum::<f32>() / n;
    let s = if mean_d > 1e-6 { std::f32::consts::SQRT_2 / mean_d } else { 1.0 };
    [s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0]
}

/// Solve an `n×n` linear system `A x = b` by Gaussian elimination with partial pivoting.
fn solve_linear(mut a: Vec<Vec<f32>>, mut b: Vec<f32>) -> Option<Vec<f32>> {
    let n = b.len();
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in 0..n {
            if r != col {
                let f = a[r][col] / a[col][col];
                for c in col..n {
                    a[r][c] -= f * a[col][c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    Some((0..n).map(|i| b[i] / a[i][i]).collect())
}

/// Estimate the homography mapping `src → dst` from ≥4 correspondences (normalised DLT via the
/// least-squares normal equations). Returns a row-major 3×3 with `h33 = 1`.
fn solve_homography(src: &[(f32, f32)], dst: &[(f32, f32)]) -> Option<[f32; 9]> {
    if src.len() < 4 {
        return None;
    }
    let ns = normalize(src);
    let nd = normalize(dst);
    let sp: Vec<(f32, f32)> = src.iter().map(|&(x, y)| apply_h(&ns, x, y)).collect();
    let dp: Vec<(f32, f32)> = dst.iter().map(|&(x, y)| apply_h(&nd, x, y)).collect();
    // Build the 2n×8 system for h = [h11..h32], then the 8×8 normal equations AᵀA h = Aᵀb.
    let mut rows: Vec<[f32; 8]> = Vec::with_capacity(sp.len() * 2);
    let mut rhs: Vec<f32> = Vec::with_capacity(sp.len() * 2);
    for (&(x, y), &(xp, yp)) in sp.iter().zip(&dp) {
        rows.push([x, y, 1.0, 0.0, 0.0, 0.0, -x * xp, -y * xp]);
        rhs.push(xp);
        rows.push([0.0, 0.0, 0.0, x, y, 1.0, -x * yp, -y * yp]);
        rhs.push(yp);
    }
    let mut ata = vec![vec![0f32; 8]; 8];
    let mut atb = vec![0f32; 8];
    for (row, &r) in rows.iter().zip(&rhs) {
        for i in 0..8 {
            atb[i] += row[i] * r;
            for j in 0..8 {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    let h = solve_linear(ata, atb)?;
    let hn = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0];
    // Denormalise: H = Nd⁻¹ · Ĥ · Ns.
    let nd_inv = mat3_inv(&nd)?;
    Some(mat3_mul(&mat3_mul(&nd_inv, &hn), &ns))
}

/// RANSAC homography (`src → dst`). Returns the model refit on its inliers, or `None`.
fn ransac(matches: &[((f32, f32), (f32, f32))]) -> Option<[f32; 9]> {
    if matches.len() < 8 {
        return None; // too few to trust
    }
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let (mut best_h, mut best_inl) = (None, 0usize);
    for _ in 0..RANSAC_ITERS {
        // Sample 4 distinct matches.
        let mut idx = [0usize; 4];
        for k in 0..4 {
            loop {
                let c = rng.below(matches.len());
                if !idx[..k].contains(&c) {
                    idx[k] = c;
                    break;
                }
            }
        }
        let src: Vec<_> = idx.iter().map(|&i| matches[i].0).collect();
        let dst: Vec<_> = idx.iter().map(|&i| matches[i].1).collect();
        let Some(h) = solve_homography(&src, &dst) else { continue };
        let inl = matches
            .iter()
            .filter(|((sx, sy), (dx, dy))| {
                let (px, py) = apply_h(&h, *sx, *sy);
                (px - dx).powi(2) + (py - dy).powi(2) < INLIER_PX * INLIER_PX
            })
            .count();
        if inl > best_inl {
            best_inl = inl;
            best_h = Some(h);
        }
    }
    // Require a solid consensus, then refit on all inliers for accuracy.
    let h = best_h?;
    if best_inl < 12 {
        return None;
    }
    let (src, dst): (Vec<_>, Vec<_>) = matches
        .iter()
        .filter(|((sx, sy), (dx, dy))| {
            let (px, py) = apply_h(&h, *sx, *sy);
            (px - dx).powi(2) + (py - dy).powi(2) < INLIER_PX * INLIER_PX
        })
        .map(|&(s, d)| (s, d))
        .unzip();
    solve_homography(&src, &dst).or(Some(h))
}

/// Estimate the homography that maps `add`'s pixels into `base`'s coordinate frame (full-res).
pub fn estimate(base: &RgbImage, add: &RgbImage) -> Option<[f32; 9]> {
    let (fb, sb) = features(base);
    let (fa, sa) = features(add);
    // Match add→base, then lift the resulting point homography from detection frame to full-res:
    //   H_full = S_base · Ĥ · S_add⁻¹,  where S scales detection→full (a uniform scale here).
    let matches = match_features(&fa, &fb);
    let h = ransac(&matches)?;
    let s_add = [sa, 0.0, 0.0, 0.0, sa, 0.0, 0.0, 0.0, 1.0]; // full→detect for add is 1/sa; detect→full is sa
    let s_add_inv = [1.0 / sa, 0.0, 0.0, 0.0, 1.0 / sa, 0.0, 0.0, 0.0, 1.0];
    let s_base = [sb, 0.0, 0.0, 0.0, sb, 0.0, 0.0, 0.0, 1.0];
    let _ = s_add;
    Some(mat3_mul(&mat3_mul(&s_base, &h), &s_add_inv))
}

/// 4-point perspective rectify: warp the picked quad (`pts` = TL, TR, BR, BL in per-mille of the
/// image) so it fills the frame — straightens a photographed plane (a document, a painting, a wall).
/// A degenerate quad returns the image unchanged.
pub fn rectify(img: &DynamicImage, pts: [[i32; 2]; 4]) -> DynamicImage {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let (wf, hf) = ((w.max(1) - 1) as f32, (h.max(1) - 1) as f32);
    let src: Vec<(f32, f32)> =
        pts.iter().map(|p| (p[0] as f32 / 1000.0 * wf, p[1] as f32 / 1000.0 * hf)).collect();
    let dst = vec![(0.0, 0.0), (wf, 0.0), (wf, hf), (0.0, hf)];
    let Some(hm) = solve_homography(&src, &dst) else {
        return img.clone();
    };
    let Some(proj) = Projection::from_matrix(hm) else {
        return img.clone();
    };
    let mut out = ImageBuffer::from_pixel(w, h, Rgb([0u8, 0, 0]));
    warp_into(&rgb, &proj, Interpolation::Bilinear, Rgb([0u8, 0, 0]), &mut out);
    DynamicImage::ImageRgb8(out)
}

/// Warp a same-size white mask through `proj` into a canvas the size of `out`, returning coverage.
fn coverage(src_w: u32, src_h: u32, proj: &Projection, cw: u32, ch: u32) -> GrayImage {
    let white = ImageBuffer::from_pixel(src_w, src_h, Luma([255u8]));
    let mut out = ImageBuffer::from_pixel(cw, ch, Luma([0u8]));
    warp_into(&white, proj, Interpolation::Bilinear, Luma([0u8]), &mut out);
    out
}

/// Stitch `add` onto `base` via a homography, returning the blended panorama, or `None` if no
/// confident model was found (→ caller falls back to a simpler stitch).
pub fn stitch_pair(base: &RgbImage, add: &RgbImage) -> Option<RgbImage> {
    let h = estimate(base, add)?;
    let proj_h = Projection::from_matrix(h)?;
    // Bounds: base occupies [0,bw]×[0,bh]; add's corners map through H.
    let (bw, bh) = (base.width() as f32, base.height() as f32);
    let (aw, ah) = (add.width() as f32, add.height() as f32);
    let corners = [(0.0, 0.0), (aw, 0.0), (aw, ah), (0.0, ah)];
    let mapped: Vec<(f32, f32)> = corners.iter().map(|&(x, y)| apply_h(&h, x, y)).collect();
    let min_x = mapped.iter().map(|p| p.0).fold(0.0f32, f32::min).min(0.0);
    let min_y = mapped.iter().map(|p| p.1).fold(0.0f32, f32::min).min(0.0);
    let max_x = mapped.iter().map(|p| p.0).fold(bw, f32::max).max(bw);
    let max_y = mapped.iter().map(|p| p.1).fold(bh, f32::max).max(bh);
    let cw = (max_x - min_x).ceil().max(1.0) as u32;
    let ch = (max_y - min_y).ceil().max(1.0) as u32;
    // Guard against a degenerate/huge canvas (a bad homography).
    if cw > 20_000 || ch > 20_000 || cw as f32 > (bw + aw) * 3.0 || ch as f32 > (bh + ah) * 3.0 {
        return None;
    }
    let t = Projection::translate(-min_x, -min_y); // A-frame → canvas
    // Warp base (translation only) and add (H then translation) into the canvas.
    let mut canvas_b = ImageBuffer::from_pixel(cw, ch, Rgb([0u8, 0, 0]));
    warp_into(base, &t, Interpolation::Bilinear, Rgb([0u8, 0, 0]), &mut canvas_b);
    let mut canvas_a = ImageBuffer::from_pixel(cw, ch, Rgb([0u8, 0, 0]));
    warp_into(add, &proj_h.and_then(t), Interpolation::Bilinear, Rgb([0u8, 0, 0]), &mut canvas_a);
    // Feather weights = each coverage mask blurred, so seams cross-fade.
    let cov_b = image::imageops::blur(&coverage(base.width(), base.height(), &t, cw, ch), 6.0);
    let cov_a = image::imageops::blur(&coverage(add.width(), add.height(), &proj_h.and_then(t), cw, ch), 6.0);
    let mut out = ImageBuffer::from_pixel(cw, ch, Rgb([245u8, 245, 245]));
    for y in 0..ch {
        for x in 0..cw {
            let wa = cov_a.get_pixel(x, y).0[0] as f32;
            let wb = cov_b.get_pixel(x, y).0[0] as f32;
            let sum = wa + wb;
            if sum < 1.0 {
                continue; // neither image covers this pixel → keep background
            }
            let (pa, pb) = (canvas_a.get_pixel(x, y).0, canvas_b.get_pixel(x, y).0);
            let px = std::array::from_fn(|i| ((pa[i] as f32 * wa + pb[i] as f32 * wb) / sum) as u8);
            out.put_pixel(x, y, Rgb(px));
        }
    }
    Some(out)
}

/// Stitch a sequence of frames left→right by folding each onto the growing canvas via homography.
/// Any frame that fails to register is placed edge-to-edge (so the result degrades, never mangles).
pub fn stitch(imgs: &[RgbImage]) -> RgbImage {
    if imgs.is_empty() {
        return RgbImage::new(1, 1);
    }
    let mut canvas = imgs[0].clone();
    for add in &imgs[1..] {
        canvas = match stitch_pair(&canvas, add) {
            Some(c) => c,
            None => super::stitch::concat_h(&canvas, add),
        };
    }
    canvas
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a feature-rich synthetic image: distinctive scattered squares on a light field, so FAST
    /// finds repeatable, locally-unique corners (a checkerboard would be too self-similar to match).
    fn feature_image(w: u32, h: u32) -> RgbImage {
        let mut img = RgbImage::from_pixel(w, h, Rgb([200, 200, 200]));
        let mut s: u32 = 0x1234_5678;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        for _ in 0..80 {
            let x = (rnd() % (w - 20)) as i64;
            let y = (rnd() % (h - 20)) as i64;
            let sz = 5 + (rnd() % 12);
            let g = (rnd() % 210) as u8;
            let col = Rgb([g, g.wrapping_add(40), 255u8.wrapping_sub(g)]);
            for dy in 0..sz as i64 {
                for dx in 0..sz as i64 {
                    let (px, py) = (x + dx, y + dy);
                    if px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h {
                        img.put_pixel(px as u32, py as u32, col);
                    }
                }
            }
        }
        img
    }

    /// A synthetic perspective pair: warp a feature-rich image by a known homography, then recover it.
    #[test]
    fn recovers_a_known_homography_and_stitches() {
        let base = feature_image(220, 180);
        // A mild projective warp (rotation + perspective).
        let h = [0.98f32, -0.12, 18.0, 0.10, 0.97, 6.0, 0.0002, 0.0001, 1.0];
        let proj = Projection::from_matrix(h).unwrap();
        let mut add = ImageBuffer::from_pixel(220, 180, Rgb([0u8, 0, 0]));
        warp_into(&base, &proj.invert(), Interpolation::Bilinear, Rgb([0u8, 0, 0]), &mut add);
        // estimate(base, add) should approximate `h` (maps add→base).
        let est = estimate(&base, &add).expect("homography recovered");
        // Check a few points map close to the ground-truth H.
        for &(x, y) in &[(50.0, 40.0), (150.0, 120.0), (100.0, 80.0)] {
            let (gx, gy) = apply_h(&h, x, y);
            let (ex, ey) = apply_h(&est, x, y);
            assert!((gx - ex).abs() < 6.0 && (gy - ey).abs() < 6.0, "point ({x},{y}) off: gt ({gx:.1},{gy:.1}) est ({ex:.1},{ey:.1})");
        }
        let pano = stitch_pair(&base, &add).expect("stitched");
        assert!(pano.width() >= 220 && pano.height() >= 180);
    }

    #[test]
    fn unrelated_frames_return_none() {
        let a = RgbImage::from_fn(120, 90, |x, y| Rgb([(x % 200) as u8, (y % 200) as u8, 40]));
        let b = RgbImage::from_pixel(120, 90, Rgb([10, 20, 30])); // flat → no corners/matches
        assert!(stitch_pair(&a, &b).is_none());
    }
}
