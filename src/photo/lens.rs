//! The lens and the shutter.
//!
//! THE LENS: the distortion and the lateral chromatic aberration (one resampling, the red and the blue images
//! scaled apart toward the corners), the aerial haze by the depth, the DEPTH OF FIELD — the circle of
//! confusion of every pixel from the aperture, the format's focal length, the focus distance and the pixel's
//! metric depth, with the field's curvature, the diffraction and the resolving power added in quadrature, laid
//! as a per-pixel blur interpolated between Gaussian levels; the Petzval's tangential swirl at the edges, the
//! anamorphic's standing ovals; the pictorial soft focus; the veiling flare and the bloom; the vignette.
//!
//! THE SHUTTER: the exposure time — the hand's shake past 1/30, the moving things (the water, the foliage, the
//! sky, where the stuff was read) smeared past a second, the figures ghosted past four seconds: the scene
//! behind them grown into their place and the figure laid half over it.

use super::plan::PhotoPlan;
use super::{gaussian, gaussian_field, gaussian_xy, luma, Aux, Field, Frame};

/// The focus distance in metres by the plan's `focus`, given the metric depth.
pub fn focus_distance(plan: &PhotoPlan, aux: &Aux, metric: &Field) -> (f32, String) {
    if plan.focus_distance > 0.0 {
        return (plan.focus_distance, "given".into());
    }
    let mode = plan.focus.trim().to_ascii_lowercase();
    let eyes = || {
        aux.faces.iter().max_by(|a, b| ((a.0[2] - a.0[0]).partial_cmp(&(b.0[2] - b.0[0]))).unwrap_or(std::cmp::Ordering::Equal)).map(|(_, e)| {
            let (x, y) = ((e[0][0] + e[1][0]) / 2.0, (e[0][1] + e[1][1]) / 2.0);
            metric.sample(x, y)
        })
    };
    let subject = || {
        aux.matte.as_ref().and_then(|m| {
            let mut v: Vec<f32> = m.v.iter().zip(&metric.v).filter(|(a, _)| **a > 0.5).map(|(_, z)| *z).collect();
            if v.len() < 16 {
                return None;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Some(v[v.len() / 2])
        })
    };
    let percentile = |p: f32| {
        let mut v = metric.v.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f32 * p) as usize]
    };
    let centre = || metric.sample(metric.w as f32 / 2.0, metric.h as f32 / 2.0);
    match mode.as_str() {
        "eyes" => eyes().map(|z| (z, "the eyes".to_string())).unwrap_or_else(|| (centre(), "the centre (no eyes found)".into())),
        "subject" => subject().map(|z| (z, "the subject".to_string())).unwrap_or_else(|| (centre(), "the centre (no subject found)".into())),
        "near" => (percentile(0.05), "the near ground".into()),
        "far" => (percentile(0.95), "the far ground".into()),
        "centre" | "center" => (centre(), "the centre".into()),
        s if s.contains(',') => {
            let mut it = s.split(',').map(|t| t.trim().parse::<f32>().unwrap_or(0.5));
            let (fx, fy) = (it.next().unwrap_or(0.5), it.next().unwrap_or(0.5));
            (metric.sample(fx * metric.w as f32, fy * metric.h as f32), format!("the point {fx:.2},{fy:.2}"))
        }
        _ => {
            if let Some(z) = eyes() {
                (z, "the eyes".into())
            } else if let Some(z) = subject() {
                (z, "the subject".into())
            } else {
                (centre(), "the centre".into())
            }
        }
    }
}

/// The per-pixel blur σ in pixels from the plan's lens over the metric depth (None without an aperture and
/// without a curvature, a diffraction or a resolving-power blur).
pub fn blur_field(plan: &PhotoPlan, aux: &Aux, w: u32, h: u32, focal_eq: f32, notes: &mut Vec<String>) -> Option<Field> {
    let long = w.max(h) as f32;
    let s_mm = plan.format_width_mm();
    let f_mm = focal_eq * s_mm / 36.0;
    let n = plan.aperture;
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let half_diag = (cx * cx + cy * cy).sqrt();
    let sigma_res = plan.resolution.max(0.0) * long / 1000.0;
    let sigma_diff = if plan.diffraction > 0.0 && n > 0.0 { plan.diffraction * 0.5 * 1.22 * 0.00055 * n / s_mm * long } else { 0.0 };
    let dof = if n > 0.0 { aux.metric.as_ref().map(|m| (m, focus_distance(plan, aux, m))) } else { None };
    if dof.is_none() && plan.field_curve <= 0.0 && sigma_res <= 0.0 && sigma_diff <= 0.0 {
        if n > 0.0 {
            notes.push("lens: no depth was read — the aperture stands down, everything sharp".into());
        }
        return None;
    }
    let mut out = Field::new(w, h, 0.0);
    if let Some((_, (zf, how))) = &dof {
        notes.push(format!("lens: {focal_eq:.0} mm-eq ({f_mm:.0} mm on {}) at f/{n:.1}, focused at {zf:.2} m ({how})", plan.format));
    }
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let rn = (((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt() / half_diag).min(1.0);
            let mut s2 = sigma_res * sigma_res + sigma_diff * sigma_diff;
            if let Some((m, (zf, _))) = &dof {
                let z = m.v[i].max(0.05) * 1000.0;
                let zf = zf.max(0.05) * 1000.0;
                let coc = f_mm * f_mm * (z - zf).abs() / (n * z * (zf - f_mm).max(1.0));
                let sigma = coc / s_mm * long * 0.5;
                s2 += sigma * sigma;
            }
            if plan.field_curve > 0.0 {
                let s = plan.field_curve * 6.0 * rn * rn * long / 1000.0;
                s2 += s * s;
            }
            out.v[i] = s2.sqrt().min(long / 12.0);
        }
    }
    Some(out)
}

/// A per-pixel Gaussian blur: the frame blurred at doubling σ levels and each pixel interpolated between the
/// two levels around its own σ (in log σ). `aniso` > 0 stretches every level vertically (the anamorphic).
pub fn variable_blur(frame: &Frame, sigma: &Field, aniso: f32) -> Frame {
    let max = sigma.v.iter().cloned().fold(0.0f32, f32::max);
    if max < 0.35 {
        return frame.clone();
    }
    let mut levels: Vec<(f32, Frame)> = vec![(0.0, frame.clone())];
    let mut s = 0.5;
    while s < max * 1.05 {
        let f = if aniso > 0.0 { gaussian_xy(frame, s / (1.0 + aniso).sqrt(), s * (1.0 + aniso).sqrt()) } else { gaussian(frame, s) };
        levels.push((s, f));
        s *= 2.0;
    }
    levels.push((s, if aniso > 0.0 { gaussian_xy(frame, s / (1.0 + aniso).sqrt(), s * (1.0 + aniso).sqrt()) } else { gaussian(frame, s) }));
    let mut out = Frame::new(frame.w, frame.h);
    for i in 0..out.px.len() {
        let want = sigma.v[i];
        if want < 0.25 {
            out.px[i] = frame.px[i];
            continue;
        }
        let mut k = 1;
        while k + 1 < levels.len() && levels[k].0 < want {
            k += 1;
        }
        let (s0, f0) = (&levels[k - 1].0, &levels[k - 1].1);
        let (s1, f1) = (&levels[k].0, &levels[k].1);
        let t = if *s0 <= 0.0 { (want / s1).clamp(0.0, 1.0) } else { ((want.ln() - s0.ln()) / (s1.ln() - s0.ln())).clamp(0.0, 1.0) };
        let (a, b) = (f0.px[i], f1.px[i]);
        out.px[i] = [a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t, a[2] * (1.0 - t) + b[2] * t];
    }
    out
}

/// The distortion and the lateral chromatic aberration, one resampling.
fn distort_ca(frame: &Frame, distort: f32, ca: f32) -> Frame {
    if distort == 0.0 && ca == 0.0 {
        return frame.clone();
    }
    let (w, h) = (frame.w, frame.h);
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let half_diag = (cx * cx + cy * cy).sqrt();
    let mut out = Frame::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
            let rn = (dx * dx + dy * dy).sqrt() / half_diag;
            let base = 1.0 + distort * rn * rn;
            let shift = ca * 2.0 * (w.max(h) as f32 / 4000.0) * rn * rn / half_diag.max(1.0);
            let fr = base * (1.0 + shift);
            let fb = base * (1.0 - shift);
            let r = frame.sample(cx + dx * fr, cy + dy * fr)[0];
            let g = frame.sample(cx + dx * base, cy + dy * base)[1];
            let b = frame.sample(cx + dx * fb, cy + dy * fb)[2];
            out.px[(y * w + x) as usize] = [r, g, b];
        }
    }
    out
}

/// The sky's colour: the mean of the brightest twentieth of the top third, or a pale grey-blue.
fn sky_colour(frame: &Frame) -> [f32; 3] {
    let top = (frame.h / 3).max(1);
    let mut v: Vec<(f32, [f32; 3])> = (0..top * frame.w).map(|i| (luma(frame.px[i as usize]), frame.px[i as usize])).collect();
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let n = (v.len() / 20).max(1);
    let mut acc = [0.0f32; 3];
    for (_, p) in v.iter().take(n) {
        for c in 0..3 {
            acc[c] += p[c];
        }
    }
    let m = [acc[0] / n as f32, acc[1] / n as f32, acc[2] / n as f32];
    if luma(m) < 0.3 {
        [0.62, 0.68, 0.78]
    } else {
        m
    }
}

/// The lens stage.
pub fn run(frame: &Frame, plan: &PhotoPlan, aux: &Aux, seed: u32, notes: &mut Vec<String>) -> Frame {
    let _ = seed;
    let (w, h) = (frame.w, frame.h);
    let long = w.max(h) as f32;
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let half_diag = (cx * cx + cy * cy).sqrt();
    let mut f = distort_ca(frame, plan.distort, plan.ca);
    // The haze: the far ground lifted toward the sky.
    if plan.haze > 0.0 {
        if let Some(d) = &aux.depth {
            let sky = sky_colour(&f);
            for i in 0..f.px.len() {
                let far = (1.0 - d.v[i]).clamp(0.0, 1.0);
                let t = plan.haze * 0.8 * far * far;
                for c in 0..3 {
                    f.px[i][c] = f.px[i][c] * (1.0 - t) + sky[c] * t;
                }
            }
        }
    }
    // The depth of field, the curvature, the diffraction, the resolving power.
    let focal_eq = if plan.focal > 0.0 { plan.focal } else if plan.source_focal > 0.0 { plan.source_focal } else { 35.0 };
    if let Some(sigma) = blur_field(plan, aux, w, h, focal_eq, notes) {
        f = variable_blur(&f, &sigma, plan.anamorphic.max(0.0));
        // The swirl: a tangential smear growing with the radius and the local defocus.
        if plan.swirl > 0.0 {
            let src = f.clone();
            for y in 0..h {
                for x in 0..w {
                    let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                    let r = (dx * dx + dy * dy).sqrt();
                    let rn = r / half_diag;
                    let i = (y * w + x) as usize;
                    let len = plan.swirl * 5.0 * rn * rn * (long / 1000.0) * (1.0 + sigma.v[i].min(8.0));
                    if len < 0.5 || r < 1.0 {
                        continue;
                    }
                    let (tx, ty) = (-dy / r, dx / r);
                    let mut acc = [0.0f32; 3];
                    let taps = 7;
                    for k in 0..taps {
                        let t = (k as f32 / (taps - 1) as f32 - 0.5) * 2.0 * len;
                        let p = src.sample(x as f32 + 0.5 + tx * t, y as f32 + 0.5 + ty * t);
                        for c in 0..3 {
                            acc[c] += p[c] / taps as f32;
                        }
                    }
                    f.px[i] = acc;
                }
            }
        }
    }
    // The soft focus: the sharp image under a glowing blurred one.
    if plan.soft > 0.0 {
        let b = gaussian(&f, long / 150.0);
        let k = plan.soft * 0.6;
        for i in 0..f.px.len() {
            for c in 0..3 {
                let glow = (b.px[i][c] - 0.4).max(0.0);
                f.px[i][c] = f.px[i][c] * (1.0 - k) + b.px[i][c] * k + glow * plan.soft * 0.35;
            }
        }
    }
    // The flare: the bloom of the highlights and a veil over the frame.
    if plan.flare > 0.0 {
        let hi = f.map(|p| [(p[0] - 0.8).max(0.0), (p[1] - 0.8).max(0.0), (p[2] - 0.8).max(0.0)]);
        let bloom = if plan.anamorphic > 0.0 { gaussian_xy(&hi, long / 12.0, 1.5) } else { gaussian(&hi, long / 60.0) };
        let veil = {
            let n = hi.px.len() as f32;
            let m: f32 = hi.px.iter().map(|p| luma(*p)).sum::<f32>() / n;
            m.min(0.2)
        };
        let tint = if plan.anamorphic > 0.0 { [0.5, 0.7, 1.0] } else { [1.0, 0.95, 0.85] };
        for i in 0..f.px.len() {
            for c in 0..3 {
                f.px[i][c] += plan.flare * (0.7 * bloom.px[i][c] * tint[c] + 0.4 * veil * tint[c]);
            }
        }
    }
    // The vignette.
    if plan.vignette > 0.0 {
        for y in 0..h {
            for x in 0..w {
                let rn = (((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt() / half_diag).min(1.0);
                let v = 1.0 - plan.vignette * 0.75 * rn.powf(2.5);
                let i = (y * w + x) as usize;
                for c in 0..3 {
                    f.px[i][c] *= v;
                }
            }
        }
    }
    f
}

/// A directional line blur of `len` pixels along (dx, dy), weighted by a mask (None = everywhere).
fn line_blur(frame: &Frame, dx: f32, dy: f32, len: f32, mask: Option<&Field>) -> Frame {
    if len < 0.5 {
        return frame.clone();
    }
    let taps = ((len * 2.0).ceil() as usize).clamp(3, 33);
    let mut out = frame.clone();
    for y in 0..frame.h {
        for x in 0..frame.w {
            let i = (y * frame.w + x) as usize;
            let m = mask.map(|m| m.v[i]).unwrap_or(1.0);
            if m < 0.02 {
                continue;
            }
            let l = len * m;
            let mut acc = [0.0f32; 3];
            for k in 0..taps {
                let t = (k as f32 / (taps - 1) as f32 - 0.5) * 2.0 * l;
                let p = frame.sample(x as f32 + 0.5 + dx * t, y as f32 + 0.5 + dy * t);
                for c in 0..3 {
                    acc[c] += p[c] / taps as f32;
                }
            }
            out.px[i] = acc;
        }
    }
    out
}

/// The shutter stage.
pub fn shutter(frame: &Frame, plan: &PhotoPlan, aux: &Aux, seed: u32, notes: &mut Vec<String>) -> Frame {
    let t = plan.exposure_time;
    if t <= 0.0 {
        return frame.clone();
    }
    let long = frame.w.max(frame.h) as f32;
    let mut f = frame.clone();
    let mut what = Vec::new();
    // The hand's shake past 1/30 s.
    if plan.shake > 0.0 && t > 1.0 / 30.0 {
        let stops = (t * 30.0).log2().max(0.0);
        let len = plan.shake * stops * 1.5 * long / 1000.0;
        let ang = super::hash01(1, 2, seed ^ 0x51) * std::f32::consts::TAU;
        f = line_blur(&f, ang.cos(), ang.sin(), len, None);
        what.push(format!("shake {len:.1} px"));
    }
    // The moving things past a second, where the stuff was read.
    if plan.motion > 0.0 && t >= 0.5 {
        if let Some(m) = &aux.moving {
            let stops = (t * 2.0).log2().max(0.0) + 1.0;
            let len = plan.motion * stops * 6.0 * long / 1000.0;
            let soft = gaussian_field(m, long / 40.0);
            f = line_blur(&f, 1.0, 0.15, len, Some(&soft));
            let g = gaussian(&f, len * 0.25);
            for i in 0..f.px.len() {
                let k = (soft.v[i] * plan.motion).clamp(0.0, 1.0);
                for c in 0..3 {
                    f.px[i][c] = f.px[i][c] * (1.0 - k) + g.px[i][c] * k;
                }
            }
            what.push(format!("the moving things smeared {len:.0} px over {:.0}% of the frame", m.mean() * 100.0));
        } else {
            what.push("no stuff was read — the motion stands down".into());
        }
    }
    // The figures ghosted past four seconds: the scene grown into their place, the figure half over it.
    if plan.ghost > 0.0 && t >= 2.0 {
        if let Some(m) = &aux.matte {
            if m.v.iter().filter(|v| **v > 0.5).count() > 16 {
                let k = (plan.ghost * ((t / 2.0).log2() * 0.5).clamp(0.2, 1.0)).clamp(0.0, 0.9);
                let mut behind = f.clone();
                let holes = Field { w: m.w, h: m.h, v: m.v.iter().map(|v| if *v > 0.5 { 1.0 } else { 0.0 }).collect() };
                let metric = aux.metric.clone().unwrap_or_else(|| Field::new(m.w, m.h, 1.0));
                super::camera::propagate(&mut behind, &holes, &metric);
                let soft = gaussian_field(m, 2.0);
                for i in 0..f.px.len() {
                    let a = soft.v[i].clamp(0.0, 1.0) * k;
                    for c in 0..3 {
                        f.px[i][c] = f.px[i][c] * (1.0 - a) + behind.px[i][c] * a;
                    }
                }
                what.push(format!("the figures ghosted {:.0}%", k * 100.0));
            }
        }
    }
    if !what.is_empty() {
        notes.push(format!("shutter: {t} s — {}", what.join(", ")));
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_depth_of_field_blurs_the_far_ground_and_leaves_the_focus_sharp() {
        let (w, h) = (96u32, 64u32);
        let mut f = Frame::new(w, h);
        let mut metric = Field::new(w, h, 0.0);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                // A checker everywhere; the left half at 1.5 m (the focus), the right half at 20 m.
                let c = if ((x / 2) + (y / 2)) % 2 == 0 { 0.9 } else { 0.1 };
                f.px[i] = [c; 3];
                metric.v[i] = if x < w / 2 { 1.5 } else { 20.0 };
            }
        }
        let aux = Aux { metric: Some(metric), ..Default::default() };
        let mut plan = PhotoPlan::default();
        plan.aperture = 1.4;
        plan.focus_distance = 1.5;
        plan.focal = 85.0;
        let mut notes = Vec::new();
        let out = run(&f, &plan, &aux, 1, &mut notes);
        let energy = |fr: &Frame, x0: u32, x1: u32| {
            let mut e = 0.0;
            for y in 1..h - 1 {
                for x in x0..x1 {
                    e += (fr.px[(y * w + x) as usize][0] - fr.px[(y * w + x + 1) as usize][0]).abs();
                }
            }
            e
        };
        let (near_in, near_out) = (energy(&f, 4, w / 2 - 8), energy(&out, 4, w / 2 - 8));
        let (far_in, far_out) = (energy(&f, w / 2 + 8, w - 4), energy(&out, w / 2 + 8, w - 4));
        assert!(near_out > near_in * 0.8, "the focus stays sharp: {near_out} vs {near_in}");
        assert!(far_out < far_in * 0.3, "the far ground blurs: {far_out} vs {far_in}");
        assert!(notes[0].contains("f/1.4"), "{notes:?}");
    }

    #[test]
    fn the_vignette_darkens_the_corners_and_the_barrel_pulls_the_edge_in() {
        let (w, h) = (64u32, 64u32);
        let f = Frame { w, h, px: vec![[0.5; 3]; 64 * 64] };
        let mut plan = PhotoPlan::default();
        plan.vignette = 1.0;
        let out = run(&f, &plan, &Aux::default(), 1, &mut Vec::new());
        assert!(out.px[0][0] < 0.35 && (out.px[(32 * 64 + 32) as usize][0] - 0.5).abs() < 0.02);
        // A white bar at the right edge; barrel distortion samples further out, so the bar thins at the edge.
        let mut g = Frame::new(w, h);
        for y in 0..h {
            for x in 0..w {
                g.px[(y * w + x) as usize] = [if x >= 56 { 1.0 } else { 0.0 }; 3];
            }
        }
        let mut plan = PhotoPlan::default();
        plan.distort = 0.3;
        let out = run(&g, &plan, &Aux::default(), 1, &mut Vec::new());
        // The bar now begins before column 56 on the middle row: the edge's content pulled in.
        assert!(out.px[(32 * 64 + 54) as usize][0] > 0.5 && out.px[(32 * 64 + 50) as usize][0] < 0.5, "{} {}", out.px[(32 * 64 + 54) as usize][0], out.px[(32 * 64 + 50) as usize][0]);
    }
}
