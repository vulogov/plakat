//! The camera statistics of a picture — what a photograph has that a render has not, read as numbers.
//!
//! A photograph's NOISE grows with the signal (the photons) and has a structure (the grain's clumps, the
//! demosaic's correlation); a render's is flat, white or absent. A photograph's SHARPNESS varies over the
//! frame — a plane of focus, a lens softer at the edge; a render's is one sharpness everywhere. A photograph's
//! HIGHLIGHTS clip or roll off; its corners darken; its colour fringes at the corners' edges; a JPEG leaves its
//! 8×8 grid. None of these is a judgement — a studio shot at f/16 is sharp everywhere and a scanned plate has
//! no JPEG grid — so the readings are printed with what they usually mean and left to the reader; the critic
//! uses the few that follow from a plan (the clipping, the key, the subject's sharpness, the invented share).
//!
//! The noise, the fringes and the blockiness are read at the picture's own pixels (a resize would correlate the
//! noise and erase the grid); the sharpness over blocks of a thirty-second of the long side.

use super::Field;
use image::RgbImage;

/// The readings of one picture.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Readings {
    pub width: u32,
    pub height: u32,
    pub luma_mean: f32,
    pub luma_std: f32,
    pub chroma_mean: f32,
    /// The noise σ in the calm areas, in the shadows (luma under 0.35) and in the lights (over 0.6).
    pub noise_dark: f32,
    pub noise_light: f32,
    /// The lag-1 autocorrelation of the noise residual in the calm areas (white ≈ 0; a grain or a demosaic > 0.2).
    pub noise_corr: f32,
    /// The σ of the colour residual (R−B) in the calm areas.
    pub chroma_noise: f32,
    /// The calm share of the frame the noise was read on.
    pub calm_share: f32,
    /// The coefficient of variation of the sharpness over the textured blocks (a plane of focus > 0.3).
    pub sharp_cv: f32,
    /// The sharpness of the edge blocks over the centre blocks'.
    pub sharp_edge: f32,
    /// The sharpness of the focus region over the rest (when a focus mask was given).
    pub sharp_focus: Option<f32>,
    /// The correlation of the blocks' sharpness with their depth (when a depth was given): a plane of focus on
    /// the near ground reads positive, on the far ground negative, a render near 0.
    pub sharp_depth: Option<f32>,
    pub clip_share: f32,
    pub dark_share: f32,
    /// The corners' luma over the centre's, both heavily blurred.
    pub vignette: f32,
    /// The colour fringe at the strong edges: the signed mean radial offset in pixels of the blue image against the red (outward > 0; the noise's offsets cancel), at
    /// the corners and at the centre (a lateral aberration grows outward; a render reads near 0 both).
    pub fringe_corner: f32,
    pub fringe_centre: f32,
    /// The difference across the 8-pixel boundaries over the difference inside them (a JPEG > 1.05).
    pub blockiness: f32,
}

fn luma8(p: &image::Rgb<u8>) -> f32 {
    (0.2126 * p.0[0] as f32 + 0.7152 * p.0[1] as f32 + 0.0722 * p.0[2] as f32) / 255.0
}

/// Read a picture's camera statistics. `focus` (a 0..1 mask) and `depth` (near = 1) are at the picture's size.
pub fn measure(img: &RgbImage, focus: Option<&Field>, depth: Option<&Field>) -> Readings {
    let (w, h) = img.dimensions();
    let n = (w as usize) * (h as usize);
    if n == 0 {
        return Readings::default();
    }
    let l: Vec<f32> = img.pixels().map(luma8).collect();
    let rb: Vec<f32> = img.pixels().map(|p| (p.0[0] as f32 - p.0[2] as f32) / 255.0).collect();
    let chroma: Vec<f32> = img.pixels().map(|p| (p.0.iter().max().unwrap() - p.0.iter().min().unwrap()) as f32 / 255.0).collect();
    let luma_mean = l.iter().sum::<f32>() / n as f32;
    let luma_std = (l.iter().map(|v| (v - luma_mean).powi(2)).sum::<f32>() / n as f32).sqrt();
    let chroma_mean = chroma.iter().sum::<f32>() / n as f32;
    let at = |v: &[f32], x: i64, y: i64| v[(y.clamp(0, h as i64 - 1) as usize) * w as usize + x.clamp(0, w as i64 - 1) as usize];
    // The residual (the pixel less its 3×3 mean) and the calm mask (the local σ of the 3×3 means under 0.02).
    let mut res = vec![0.0f32; n];
    let mut res_rb = vec![0.0f32; n];
    let mut mean3 = vec![0.0f32; n];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let mut s = 0.0;
            let mut s2 = 0.0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    s += at(&l, x + dx, y + dy);
                    s2 += at(&rb, x + dx, y + dy);
                }
            }
            let i = y as usize * w as usize + x as usize;
            mean3[i] = s / 9.0;
            res[i] = l[i] - s / 9.0;
            res_rb[i] = rb[i] - s2 / 9.0;
        }
    }
    let mut calm = vec![false; n];
    let r = 3i64;
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let (mut s, mut s2, mut c) = (0.0, 0.0, 0.0);
            for dy in -r..=r {
                for dx in -r..=r {
                    let v = at(&mean3, x + dx, y + dy);
                    s += v;
                    s2 += v * v;
                    c += 1.0;
                }
            }
            let var = (s2 / c - (s / c).powi(2)).max(0.0);
            calm[y as usize * w as usize + x as usize] = var.sqrt() < 0.02;
        }
    }
    let mut dark = (0.0f64, 0usize);
    let mut light = (0.0f64, 0usize);
    let mut corr = (0.0f64, 0.0f64);
    let mut crb = (0.0f64, 0usize);
    let mut calm_n = 0usize;
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = y * w as usize + x;
            if !calm[i] {
                continue;
            }
            calm_n += 1;
            let rr = res[i] as f64;
            if l[i] < 0.35 {
                dark.0 += rr * rr;
                dark.1 += 1;
            } else if l[i] > 0.6 {
                light.0 += rr * rr;
                light.1 += 1;
            }
            crb.0 += (res_rb[i] as f64).powi(2);
            crb.1 += 1;
            if x + 1 < w as usize && calm[i + 1] {
                corr.0 += rr * res[i + 1] as f64;
                corr.1 += rr * rr;
            }
        }
    }
    let sd = |a: (f64, usize)| if a.1 > 50 { (a.0 / a.1 as f64).sqrt() as f32 } else { 0.0 };
    let noise_dark = sd(dark);
    let noise_light = sd(light);
    let noise_corr = if corr.1 > 1e-12 { (corr.0 / corr.1) as f32 } else { 0.0 };
    let chroma_noise = sd(crb);
    let calm_share = calm_n as f32 / n as f32;
    // The sharpness over blocks: the fine band's share of the gradient energy — the gradient of the picture
    // against the gradient of its coarse copy (a 7×7 box twice, ≈ σ 2.8), bounded and free of the block's
    // contrast: a blurred block near 0, a sharp one toward 1.
    let lb = {
        let box7 = |v: &[f32]| -> Vec<f32> {
            let mut t = vec![0.0f32; n];
            for y in 0..h as i64 {
                for x in 0..w as i64 {
                    let mut s = 0.0;
                    for dx in -3..=3 {
                        s += at(v, x + dx, y);
                    }
                    t[y as usize * w as usize + x as usize] = s / 7.0;
                }
            }
            let mut o = vec![0.0f32; n];
            for y in 0..h as i64 {
                for x in 0..w as i64 {
                    let mut s = 0.0;
                    for dy in -3..=3 {
                        s += at(&t, x, y + dy);
                    }
                    o[y as usize * w as usize + x as usize] = s / 7.0;
                }
            }
            o
        };
        box7(&box7(&l))
    };
    let long = w.max(h) as f32;
    let block = ((long / 32.0).round() as u32).max(4);
    let (bw, bh) = ((w / block).max(1), (h / block).max(1));
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let half_diag = (cx * cx + cy * cy).sqrt();
    let mut blocks: Vec<(f32, f32, f32, f32)> = Vec::new(); // (sharpness, radius, focus share, depth)
    for by in 0..bh {
        for bx in 0..bw {
            let (x0, y0) = (bx * block, by * block);
            let (x1, y1) = ((x0 + block).min(w), (y0 + block).min(h));
            let (mut s, mut s2, mut g, mut gc, mut c, mut fsum, mut dsum) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y * w + x) as usize;
                    let v = l[i];
                    s += v;
                    s2 += v * v;
                    let gx = at(&l, x as i64 + 1, y as i64) - at(&l, x as i64 - 1, y as i64);
                    let gy = at(&l, x as i64, y as i64 + 1) - at(&l, x as i64, y as i64 - 1);
                    g += (gx * gx + gy * gy).sqrt();
                    let cx_ = at(&lb, x as i64 + 1, y as i64) - at(&lb, x as i64 - 1, y as i64);
                    let cy_ = at(&lb, x as i64, y as i64 + 1) - at(&lb, x as i64, y as i64 - 1);
                    gc += (cx_ * cx_ + cy_ * cy_).sqrt();
                    c += 1.0;
                    if let Some(f) = focus {
                        fsum += f.v[i];
                    }
                    if let Some(d) = depth {
                        dsum += d.v[i];
                    }
                }
            }
            let std = (s2 / c - (s / c).powi(2)).max(0.0).sqrt();
            // A flat block (a sky, a wall) says nothing of the focus; a blurred one does, so the gate is low.
            if std < 0.015 || gc / c < 0.0005 {
                continue;
            }
            // The fine band's share of the gradient, bounded: a blurred block near 0, a sharp one toward 1.
            let sharp = (g - gc) / (g + gc + 1e-6);
            let rad = (((x0 + x1) as f32 / 2.0 - cx).powi(2) + ((y0 + y1) as f32 / 2.0 - cy).powi(2)).sqrt() / half_diag;
            blocks.push((sharp, rad, fsum / c, dsum / c));
        }
    }
    let (mut sharp_cv, mut sharp_edge, mut sharp_focus, mut sharp_depth) = (0.0, 1.0, None, None);
    if blocks.len() >= 4 {
        let m = blocks.iter().map(|b| b.0).sum::<f32>() / blocks.len() as f32;
        let sd = (blocks.iter().map(|b| (b.0 - m).powi(2)).sum::<f32>() / blocks.len() as f32).sqrt();
        sharp_cv = if m > 1e-6 { sd / m } else { 0.0 };
        let mean_of = |pred: &dyn Fn(&(f32, f32, f32, f32)) -> bool| {
            let v: Vec<f32> = blocks.iter().filter(|b| pred(b)).map(|b| b.0).collect();
            if v.is_empty() {
                None
            } else {
                Some(v.iter().sum::<f32>() / v.len() as f32)
            }
        };
        if let (Some(e), Some(c)) = (mean_of(&|b| b.1 > 0.6), mean_of(&|b| b.1 < 0.4)) {
            if c > 1e-6 {
                sharp_edge = e / c;
            }
        }
        if focus.is_some() {
            if let (Some(a), Some(b)) = (mean_of(&|b| b.2 > 0.5), mean_of(&|b| b.2 <= 0.5)) {
                if b > 1e-6 {
                    sharp_focus = Some(a / b);
                }
            }
        }
        if depth.is_some() {
            let md = blocks.iter().map(|b| b.3).sum::<f32>() / blocks.len() as f32;
            let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
            for b in &blocks {
                sxy += (b.0 - m) * (b.3 - md);
                sxx += (b.0 - m).powi(2);
                syy += (b.3 - md).powi(2);
            }
            if sxx > 1e-9 && syy > 1e-9 {
                sharp_depth = Some(sxy / (sxx * syy).sqrt());
            }
        }
    }
    // The highlights and the shadows.
    let clip_share = l.iter().filter(|v| **v >= 0.985).count() as f32 / n as f32;
    let dark_share = l.iter().filter(|v| **v <= 0.03).count() as f32 / n as f32;
    // The vignette: the corners over the centre, on means of boxes.
    let box_mean = |x0: u32, y0: u32, x1: u32, y1: u32| {
        let (mut s, mut c) = (0.0, 0.0);
        for y in y0..y1.min(h) {
            for x in x0..x1.min(w) {
                s += l[(y * w + x) as usize];
                c += 1.0;
            }
        }
        if c > 0.0 {
            s / c
        } else {
            0.0
        }
    };
    let (dw, dh) = ((w / 10).max(1), (h / 10).max(1));
    let corners = (box_mean(0, 0, dw, dh) + box_mean(w - dw, 0, w, dh) + box_mean(0, h - dh, dw, h) + box_mean(w - dw, h - dh, w, h)) / 4.0;
    let centre = box_mean(w / 2 - w / 5, h / 2 - h / 5, w / 2 + w / 5, h / 2 + h / 5);
    let vignette = if centre > 0.02 { corners / centre } else { 1.0 };
    // The fringes: at the strong edges, the RADIAL OFFSET of the blue image against the red — the offset (in
    // pixels, −2..2) at which the blue gradient along the radius best matches the red one, by a parabola over the
    // correlation at the integer offsets; a colour edge without aberration reads 0, a lateral aberration grows
    // toward the corners.
    let mut grads: Vec<f32> = Vec::with_capacity(n / 4);
    let step = ((n / 200_000) as u32).max(1);
    for y in (2..h.saturating_sub(2)).step_by(step as usize) {
        for x in (2..w.saturating_sub(2)).step_by(step as usize) {
            let gx = at(&l, x as i64 + 1, y as i64) - at(&l, x as i64 - 1, y as i64);
            let gy = at(&l, x as i64, y as i64 + 1) - at(&l, x as i64, y as i64 - 1);
            grads.push((gx * gx + gy * gy).sqrt());
        }
    }
    grads.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let strong = grads.get(grads.len() * 9 / 10).copied().unwrap_or(1.0).max(0.05);
    let chan = |c: usize, x: f32, y: f32| -> f32 {
        let (x0, y0) = (x.floor(), y.floor());
        let (tx, ty) = (x - x0, y - y0);
        let g = |xi: i64, yi: i64| img.get_pixel(xi.clamp(0, w as i64 - 1) as u32, yi.clamp(0, h as i64 - 1) as u32).0[c] as f32 / 255.0;
        let (xi, yi) = (x0 as i64, y0 as i64);
        (g(xi, yi) * (1.0 - tx) + g(xi + 1, yi) * tx) * (1.0 - ty) + (g(xi, yi + 1) * (1.0 - tx) + g(xi + 1, yi + 1) * tx) * ty
    };
    let (mut fc, mut fcn, mut fm, mut fmn) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for y in (3..h.saturating_sub(3)).step_by(step as usize) {
        for x in (3..w.saturating_sub(3)).step_by(step as usize) {
            let (xi, yi) = (x as i64, y as i64);
            let gx = at(&l, xi + 1, yi) - at(&l, xi - 1, yi);
            let gy = at(&l, xi, yi + 1) - at(&l, xi, yi - 1);
            let gl = (gx * gx + gy * gy).sqrt();
            if gl < strong {
                continue;
            }
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let r = (dx * dx + dy * dy).sqrt();
            if r < 1.0 {
                continue;
            }
            let (ux, uy) = (dx / r, dy / r);
            let rad = r / half_diag;
            // The red gradient along the radius at this pixel; the blue's at offsets −2..2 along it.
            let gr = chan(0, x as f32 + ux, y as f32 + uy) - chan(0, x as f32 - ux, y as f32 - uy);
            if gr.abs() < 0.02 {
                continue;
            }
            let mut best = (0i32, f32::NEG_INFINITY);
            let mut corr = [0.0f32; 5];
            for (k, s_) in (-2i32..=2).enumerate() {
                let (ox, oy) = (x as f32 + ux * s_ as f32, y as f32 + uy * s_ as f32);
                let gb = chan(2, ox + ux, oy + uy) - chan(2, ox - ux, oy - uy);
                corr[k] = gr * gb;
                if corr[k] > best.1 {
                    best = (s_, corr[k]);
                }
            }
            if best.1 <= 0.0 {
                continue;
            }
            let k = (best.0 + 2) as usize;
            let off = if k > 0 && k < 4 {
                let (c0, c1, c2) = (corr[k - 1], corr[k], corr[k + 1]);
                let d = c0 - 2.0 * c1 + c2;
                best.0 as f32 + if d.abs() > 1e-9 { (0.5 * (c0 - c2) / d).clamp(-0.5, 0.5) } else { 0.0 }
            } else {
                best.0 as f32
            };
            // Signed: a lateral aberration pulls the blue one way along every radius and its mean keeps its
            // sign; the noise's random offsets cancel (their absolute mean would read a fringe on a clean frame).
            if rad > 0.6 {
                fc += off;
                fcn += 1.0;
            } else if rad < 0.4 {
                fm += off;
                fmn += 1.0;
            }
        }
    }
    let fringe_corner = if fcn > 0.0 { fc / fcn } else { 0.0 };
    let fringe_centre = if fmn > 0.0 { fm / fmn } else { 0.0 };
    // The JPEG grid: the horizontal and vertical differences at the 8-pixel boundaries over those inside.
    let (mut bnd, mut bn, mut inn, mut inn_n) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for y in 0..h as usize {
        for x in 1..w as usize {
            let d = (l[y * w as usize + x] - l[y * w as usize + x - 1]).abs();
            if x % 8 == 0 {
                bnd += d;
                bn += 1.0;
            } else if x % 8 == 4 {
                inn += d;
                inn_n += 1.0;
            }
        }
    }
    for y in 1..h as usize {
        for x in 0..w as usize {
            let d = (l[y * w as usize + x] - l[(y - 1) * w as usize + x]).abs();
            if y % 8 == 0 {
                bnd += d;
                bn += 1.0;
            } else if y % 8 == 4 {
                inn += d;
                inn_n += 1.0;
            }
        }
    }
    let blockiness = if inn_n > 0.0 && bn > 0.0 && inn / inn_n > 1e-6 { (bnd / bn) / (inn / inn_n) } else { 1.0 };
    Readings { width: w, height: h, luma_mean, luma_std, chroma_mean, noise_dark, noise_light, noise_corr, chroma_noise, calm_share, sharp_cv, sharp_edge, sharp_focus, sharp_depth, clip_share, dark_share, vignette, fringe_corner, fringe_centre, blockiness }
}

impl Readings {
    /// The readings as lines, each with what it usually means.
    pub fn lines(&self) -> Vec<String> {
        let mut o = Vec::new();
        o.push(format!("size {}×{}, key {:.2}, spread {:.2}, chroma {:.2}", self.width, self.height, self.luma_mean, self.luma_std, self.chroma_mean));
        let slope = if self.noise_dark > 1e-5 { self.noise_light / self.noise_dark } else { 0.0 };
        o.push(format!(
            "noise: σ {:.4} in the shadows, {:.4} in the lights{} on {:.0}% calm ground — a sensor's noise grows with the signal, a render's is flat or absent",
            self.noise_dark,
            self.noise_light,
            if slope > 0.0 { format!(" (×{slope:.2})") } else { String::new() },
            self.calm_share * 100.0
        ));
        o.push(format!("noise structure: lag-1 correlation {:.2} (white noise ≈ 0; a grain or a demosaic over 0.2), colour noise σ {:.4}", self.noise_corr, self.chroma_noise));
        o.push(format!(
            "sharpness: varies {:.2} over the textured blocks (a plane of focus over 0.3, one sharpness under 0.15), the edge {:.2} of the centre{}{}",
            self.sharp_cv,
            self.sharp_edge,
            self.sharp_focus.map(|v| format!(", the focus {v:.2}× the rest")).unwrap_or_default(),
            self.sharp_depth.map(|v| format!(", with the depth {v:+.2}")).unwrap_or_default()
        ));
        o.push(format!("highlights: {:.1}% clipped, {:.1}% black; the corners {:.2} of the centre (the vignette)", self.clip_share * 100.0, self.dark_share * 100.0, self.vignette));
        o.push(format!("fringes at the strong edges: the blue {:+.2} px off the red along the radius at the corners, {:+.2} px at the centre (a lateral aberration grows outward; a clean frame reads near 0); JPEG grid {:.2} (over 1.05 the 8×8 blocks show)", self.fringe_corner, self.fringe_centre, self.blockiness));
        o
    }

    /// The readings as a markdown table, one column per picture.
    pub fn table(rows: &[(String, Readings)]) -> String {
        let mut o = format!("| reading | {} |\n|---|{}\n", rows.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(" | "), "---|".repeat(rows.len()));
        let line = |name: &str, f: &dyn Fn(&Readings) -> String| format!("| {name} | {} |\n", rows.iter().map(|(_, r)| f(r)).collect::<Vec<_>>().join(" | "));
        o.push_str(&line("size", &|r| format!("{}×{}", r.width, r.height)));
        o.push_str(&line("key (mean luma)", &|r| format!("{:.3}", r.luma_mean)));
        o.push_str(&line("spread (luma σ)", &|r| format!("{:.3}", r.luma_std)));
        o.push_str(&line("chroma", &|r| format!("{:.3}", r.chroma_mean)));
        o.push_str(&line("noise σ, shadows", &|r| format!("{:.4}", r.noise_dark)));
        o.push_str(&line("noise σ, lights", &|r| format!("{:.4}", r.noise_light)));
        o.push_str(&line("noise lag-1 correlation", &|r| format!("{:.2}", r.noise_corr)));
        o.push_str(&line("colour noise σ", &|r| format!("{:.4}", r.chroma_noise)));
        o.push_str(&line("calm ground", &|r| format!("{:.0}%", r.calm_share * 100.0)));
        o.push_str(&line("sharpness variation (cv)", &|r| format!("{:.2}", r.sharp_cv)));
        o.push_str(&line("sharpness, edge / centre", &|r| format!("{:.2}", r.sharp_edge)));
        o.push_str(&line("sharpness, focus / rest", &|r| r.sharp_focus.map(|v| format!("{v:.2}")).unwrap_or_else(|| "—".into())));
        o.push_str(&line("sharpness ~ depth", &|r| r.sharp_depth.map(|v| format!("{v:+.2}")).unwrap_or_else(|| "—".into())));
        o.push_str(&line("clipped", &|r| format!("{:.1}%", r.clip_share * 100.0)));
        o.push_str(&line("black", &|r| format!("{:.1}%", r.dark_share * 100.0)));
        o.push_str(&line("vignette (corners / centre)", &|r| format!("{:.2}", r.vignette)));
        o.push_str(&line("fringe, corners", &|r| format!("{:.3}", r.fringe_corner)));
        o.push_str(&line("fringe, centre", &|r| format!("{:.3}", r.fringe_centre)));
        o.push_str(&line("JPEG grid", &|r| format!("{:.2}", r.blockiness)));
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photo::{hash_gauss, Frame};

    fn checker_with_flat(w: u32, h: u32) -> Frame {
        let mut f = Frame::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                // The left half a checker (texture), the right half a flat mid grey.
                let v = if x < w / 2 { if ((x / 3) + (y / 3)) % 2 == 0 { 0.7 } else { 0.2 } } else { 0.4 };
                f.px[i] = [v; 3];
            }
        }
        f
    }

    #[test]
    fn a_clean_render_reads_no_noise_and_a_noisy_one_reads_its_noise_and_its_structure() {
        let f = checker_with_flat(128, 96);
        let clean = measure(&f.to_srgb(), None, None);
        assert!(clean.noise_dark < 0.002 && clean.noise_light < 0.002, "{clean:?}");
        assert!(clean.calm_share > 0.3);
        let mut noisy = f.clone();
        for y in 0..96 {
            for x in 0..128 {
                let i = (y * 128 + x) as usize;
                let g = 0.02 * hash_gauss(x, y, 5);
                for c in 0..3 {
                    noisy.px[i][c] += g;
                }
            }
        }
        let r = measure(&noisy.to_srgb(), None, None);
        assert!(r.noise_dark > 0.004 || r.noise_light > 0.004, "{r:?}");
        assert!(r.noise_corr.abs() < 0.2, "white noise is uncorrelated: {}", r.noise_corr);
        // A blurred noise is structured.
        let b = crate::photo::gaussian(&noisy, 1.2);
        let r2 = measure(&b.to_srgb(), None, None);
        assert!(r2.noise_corr > 0.3, "{}", r2.noise_corr);
        assert_eq!(r.lines().len(), 6);
        let t = Readings::table(&[("A".into(), r.clone()), ("B".into(), r2)]);
        assert!(t.lines().count() == 2 + 19, "{}", t.lines().count());
    }

    #[test]
    fn a_half_blurred_frame_reads_a_varying_sharpness_and_a_clipped_one_its_clip() {
        let (w, h) = (320u32, 192u32);
        let mut f = Frame::new(w, h);
        for y in 0..h {
            for x in 0..w {
                // A random texture: the same statistics in every block.
                f.px[(y * w + x) as usize] = [(0.5 + 0.35 * hash_gauss(x, y, 3)).clamp(0.0, 1.0); 3];
            }
        }
        let b = crate::photo::gaussian(&f, 3.0);
        let mut half = f.clone();
        for y in 0..h {
            for x in w / 2..w {
                half.px[(y * w + x) as usize] = b.px[(y * w + x) as usize];
            }
        }
        let uniform = measure(&f.to_srgb(), None, None);
        let mut focus = Field::new(w, h, 0.0);
        for y in 0..h {
            for x in 0..w / 2 {
                focus.v[(y * w + x) as usize] = 1.0;
            }
        }
        let varied = measure(&half.to_srgb(), Some(&focus), None);
        assert!(uniform.sharp_cv < 0.1, "{}", uniform.sharp_cv);
        assert!(varied.sharp_cv > 0.15, "{}", varied.sharp_cv);
        assert!(varied.sharp_focus.unwrap() > 1.3, "{:?}", varied.sharp_focus);
        let clipped = f.map(|p| [p[0] * 3.0, p[1] * 3.0, p[2] * 3.0]);
        let r = measure(&clipped.to_srgb(), None, None);
        assert!(r.clip_share > 0.4, "{}", r.clip_share);
    }
}
