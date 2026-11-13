//! The emulsion or the sensor, and the digital pipeline after it.
//!
//! THE EMULSION / THE SENSOR: the exposure in stops; the spectral sensitivity — a colour emulsion through its
//! dyes' matrix, a monochrome one by its weights (the panchromatic near the eye's, the orthochromatic blind to
//! red, the wet plate blind to all but blue, the infrared bright in the leaves); the characteristic curve (a
//! digital sensor's line to a hard clip, a film's toe and shoulder, a plate's short scale, a paper's compressed
//! one) with its contrast and latitude; the GRAIN — its size from the micrometres on the negative and the
//! format's width in pixels (a 35 mm frame's grain a pixel wide at 4000 px, an 8×10's under a tenth), its
//! structure a value noise at that cell with a fine noise under it, its weight on the mid-tones; the sensor's
//! photon noise growing with the root of the signal and the read noise under it, then the Bayer mosaic and its
//! demosaic (the colour noise, the zipper at the edges); the halation of a film's base; the autochrome's starch
//! reseau.
//!
//! THE DIGITAL: the on-camera flash by the depth, the red eye on the faces, the HDR flattening, the face
//! smoothing, the phone's portrait mode by the matte, the sharpening and its halo, the JPEG (a real round trip
//! through the encoder), the date stamp.

use super::plan::PhotoPlan;
use super::{gaussian, gaussian_field, hash01, hash_gauss, luma, value_noise, Aux, Field, Frame};

/// The monochrome weights of a spectral sensitivity.
pub fn mono_weights(spectral: &str) -> Option<[f32; 3]> {
    match spectral.trim().to_ascii_lowercase().as_str() {
        "pan" | "panchromatic" => Some([0.25, 0.60, 0.15]),
        "ortho" | "orthochromatic" => Some([0.03, 0.55, 0.42]),
        "blue" | "wet-plate" | "collodion" => Some([0.0, 0.2, 0.8]),
        "infrared" | "ir" => Some([0.5, 0.45, 0.05]),
        _ => None,
    }
}

/// The characteristic curve on linear light (mid grey 0.18 stays 0.18 at gamma 1).
pub fn curve(x: f32, plan: &PhotoPlan, gamma: f32) -> f32 {
    let x = x.max(0.0);
    // The contrast about mid grey, in log exposure.
    let x = if (gamma - 1.0).abs() > 1e-3 { 0.18 * (x / 0.18).powf(gamma) } else { x };
    match plan.curve.trim().to_ascii_lowercase().as_str() {
        "film" | "paper" | "plate" => {
            // A finished picture is exposed for its whites: the curve's shape (a toe, a soft shoulder whose
            // softness is the latitude) is laid so the picture's white lands at 0.95; what goes over 1 rolls off.
            let plate = plan.curve.trim().eq_ignore_ascii_case("plate");
            let lat = if plan.latitude > 0.0 { plan.latitude } else if plate { 3.5 } else { 7.0 };
            let wp = 2f32.powf(lat / 2.0);
            let f = |v: f32| v * (1.0 + v / (wp * wp)) / (1.0 + v);
            let mut y = f(x) / f(1.0) * 0.95;
            // The white pinned at 0.95 and the mid grey at 0.18: the shape's lift of the mid-tones taken out by
            // a power, so the toe and the shoulder stay and the exposure does not move.
            let y18 = f(0.18) / f(1.0) * 0.95;
            let pw = (0.18f32 / 0.95).ln() / (y18 / 0.95).ln();
            if y <= 0.95 {
                y = 0.95 * (y / 0.95).powf(pw);
            }
            if y > 0.95 {
                // Over the picture's white the curve rolls into its shoulder, never reaching 1.
                y = 0.95 + 0.05 * (1.0 - (-(y - 0.95) / 0.05).exp());
            }
            let toe = if plate { 0.4 + plan.toe } else { plan.toe };
            if toe > 0.0 {
                let t = toe * 0.03;
                y = y * y / (y + t);
            }
            if plan.shoulder > 0.0 {
                let k = 1.0 - plan.shoulder * 0.5;
                if y > k {
                    y = k + (1.0 - k) * (1.0 - (-(y - k) / (1.0 - k)).exp());
                }
            }
            y.min(1.0)
        }
        _ => {
            if plan.shoulder > 0.0 {
                let k = 1.0 - plan.shoulder * 0.5;
                if x > k {
                    return k + (1.0 - k) * (1.0 - (-(x - k) / (1.0 - k)).exp());
                }
            }
            x.min(1.0)
        }
    }
}

/// The Bayer mosaic and a bilinear demosaic: the colour noise and the zipper a sensor's colour has.
pub fn demosaic(frame: &Frame) -> Frame {
    let (w, h) = (frame.w as i64, frame.h as i64);
    let chan = |x: i64, y: i64| -> usize {
        match (x & 1, y & 1) {
            (0, 0) => 0,
            (1, 1) => 2,
            _ => 1,
        }
    };
    let mosaic: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| frame.px[(y * w + x) as usize][chan(x, y)]).collect();
    // The border reflected with its parity kept, so a neighbour is always the channel it should be.
    let refl = |v: i64, n: i64| if v < 0 { (-v).min(n - 1) } else if v >= n { (2 * n - 2 - v).max(0) } else { v };
    let at = |x: i64, y: i64| mosaic[(refl(y, h) * w + refl(x, w)) as usize];
    let mut out = Frame::new(frame.w, frame.h);
    for y in 0..h {
        for x in 0..w {
            let c = chan(x, y);
            let mut px = [0.0f32; 3];
            px[c] = at(x, y);
            match c {
                1 => {
                    // Green: red on the row's neighbours or the column's.
                    let (hz, vt) = ((at(x - 1, y) + at(x + 1, y)) / 2.0, (at(x, y - 1) + at(x, y + 1)) / 2.0);
                    if y & 1 == 0 {
                        px[0] = hz;
                        px[2] = vt;
                    } else {
                        px[0] = vt;
                        px[2] = hz;
                    }
                }
                _ => {
                    px[1] = (at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1)) / 4.0;
                    px[2 - c] = (at(x - 1, y - 1) + at(x + 1, y - 1) + at(x - 1, y + 1) + at(x + 1, y + 1)) / 4.0;
                }
            }
            out.px[(y * w + x) as usize] = px;
        }
    }
    out
}

/// The emulsion / sensor stage.
pub fn run(frame: &Frame, plan: &PhotoPlan, aux: &Aux, seed: u32, notes: &mut Vec<String>) -> Frame {
    let _ = aux;
    let (w, h) = (frame.w, frame.h);
    let long = w.max(h) as f32;
    let n = frame.px.len();
    let mut f = frame.clone();
    let mut what = Vec::new();
    // The exposure.
    let ev = plan.exposure;
    if ev != 0.0 {
        let k = 2f32.powf(ev);
        for p in &mut f.px {
            for c in 0..3 {
                p[c] *= k;
            }
        }
        what.push(format!("{ev:+.1} EV"));
    }
    // The spectral sensitivity.
    let mono = plan.is_mono();
    if mono {
        let wts = mono_weights(&plan.spectral).unwrap_or([0.25, 0.60, 0.15]);
        let ir = matches!(plan.spectral.trim(), "infrared" | "ir");
        for p in &mut f.px {
            let mut v = wts[0] * p[0] + wts[1] * p[1] + wts[2] * p[2];
            if ir {
                v += 1.5 * (p[1] - p[0].max(p[2])).max(0.0);
            }
            *p = [v; 3];
        }
        what.push(format!("{} emulsion", if plan.spectral.trim().is_empty() || plan.spectral.trim() == "colour" { "panchromatic".to_string() } else { plan.spectral.trim().to_string() }));
    } else {
        if plan.matrix.len() == 9 {
            let m = &plan.matrix;
            for p in &mut f.px {
                let q = [m[0] * p[0] + m[1] * p[1] + m[2] * p[2], m[3] * p[0] + m[4] * p[1] + m[5] * p[2], m[6] * p[0] + m[7] * p[1] + m[8] * p[2]];
                *p = q;
            }
        }
        if (plan.saturation - 1.0).abs() > 1e-3 || plan.warm != 0.0 || plan.cross > 0.0 {
            for p in &mut f.px {
                let l = luma(*p);
                for c in 0..3 {
                    p[c] = l + (p[c] - l) * plan.saturation;
                }
                if plan.warm != 0.0 {
                    p[0] += plan.warm * 0.12 * l;
                    p[2] -= plan.warm * 0.12 * l;
                }
                if plan.cross > 0.0 {
                    let k = plan.cross * 0.35;
                    p[0] = p[0].max(0.0).powf(1.0 - k);
                    p[2] = p[2].max(0.0).powf(1.0 + k);
                    p[1] = 0.18 * (p[1].max(0.0) / 0.18).powf(1.0 + k * 0.5);
                }
            }
        }
    }
    // The characteristic curve.
    let gamma = plan.gamma * (1.0 + 0.12 * plan.develop);
    for p in &mut f.px {
        for c in 0..3 {
            p[c] = curve(p[c], plan, gamma);
        }
    }
    what.push(format!("the {} curve γ {gamma:.2}", if plan.curve.trim().is_empty() { "digital" } else { plan.curve.trim() }));
    // The halation.
    if plan.halation > 0.0 {
        let l = f.luma_field();
        let hi = Field { w, h, v: l.v.iter().map(|v| (v - 0.7).max(0.0)).collect() };
        let b = gaussian_field(&hi, long / 90.0);
        let tint = if mono { [0.9, 0.9, 0.9] } else { [1.0, 0.45, 0.08] };
        for i in 0..n {
            for c in 0..3 {
                f.px[i][c] += plan.halation * 0.9 * b.v[i] * tint[c];
            }
        }
    }
    // The grain.
    let iso_k = if plan.iso > 0.0 { (plan.iso / 400.0).powf(0.35).clamp(0.4, 3.0) } else { 1.0 };
    let dev_k = 1.0 + 0.15 * plan.develop.max(0.0);
    if plan.grain > 0.0 {
        let s_mm = plan.format_width_mm();
        let px = plan.grain_size * long / (s_mm * 1000.0);
        // A grain finer than a pixel averages out (σ ∝ its size); a coarser one grows slowly.
        let size_k = px.clamp(0.02, 1.0) * px.max(1.0).sqrt().min(2.0);
        let amp = plan.grain * 0.1 * iso_k * dev_k * size_k;
        let cell = px.max(1.0) * 1.5;
        let coarse = value_noise(w, h, cell, seed ^ 0x6a1);
        let coarse_g = if mono { None } else { Some((value_noise(w, h, cell, seed ^ 0x6a2), value_noise(w, h, cell, seed ^ 0x6a3))) };
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let l = luma(f.px[i]).clamp(0.0, 1.0);
                let wgt = 0.25 + 4.0 * l * (1.0 - l);
                let shared = 0.7 * (coarse.v[i] - 0.5) * 2.0 + 0.3 * hash_gauss(x, y, seed ^ 0x77) * 0.6;
                for c in 0..3 {
                    let own = match (&coarse_g, c) {
                        (Some((g, _)), 1) => (g.v[i] - 0.5) * 2.0,
                        (Some((_, b)), 2) => (b.v[i] - 0.5) * 2.0,
                        _ => shared,
                    };
                    let gval = if mono { shared } else { 0.6 * shared + 0.4 * own };
                    f.px[i][c] *= 1.0 + amp * wgt * gval;
                }
            }
        }
        what.push(format!("grain {:.1} px at {:.0} µm on {}", px, plan.grain_size, plan.format));
    }
    // The sensor's noise and its demosaic.
    if plan.sensor_noise > 0.0 {
        let k = plan.sensor_noise * iso_k;
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                for c in 0..3 {
                    let v = f.px[i][c].max(0.0);
                    let sigma = k * (0.004 + 0.03 * v.sqrt()) * if c == 2 { 1.4 } else { 1.0 };
                    f.px[i][c] = v + sigma * hash_gauss(x, y, seed ^ (0x9100 + c as u32));
                }
            }
        }
        what.push(format!("sensor noise {:.2}", k));
    }
    if plan.demosaic > 0.0 && !mono {
        let d = demosaic(&f);
        let t = plan.demosaic.clamp(0.0, 1.0);
        for i in 0..n {
            for c in 0..3 {
                f.px[i][c] = f.px[i][c] * (1.0 - t) + d.px[i][c] * t;
            }
        }
        what.push("Bayer demosaic".into());
    }
    // The autochrome's reseau.
    if plan.reseau > 0.0 {
        let cell = (long / 700.0).max(1.2);
        let dyes: [[f32; 3]; 3] = [[1.0, 0.55, 0.12], [0.25, 0.85, 0.3], [0.5, 0.3, 0.95]];
        let soft = gaussian(&f, cell * 0.5);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let (gx, gy) = ((x as f32 / cell).floor() as u32, (y as f32 / cell).floor() as u32);
                let jitter = hash01(gx, gy, seed ^ 0x33);
                let k = ((jitter * 3.0) as usize).min(2);
                let dye = dyes[k];
                let mean = (dye[0] + dye[1] + dye[2]) / 3.0;
                let p = soft.px[i];
                let seen = (p[0] * dye[0] + p[1] * dye[1] + p[2] * dye[2]) / (3.0 * mean);
                for c in 0..3 {
                    // The grain's colour at half strength over the grey it passes: a mosaic, not a static.
                    let through = seen * (0.5 + 0.5 * dye[c] / mean) * 0.95;
                    f.px[i][c] = p[c] * (1.0 - plan.reseau) + through * plan.reseau;
                }
            }
        }
        what.push(format!("reseau {cell:.1} px"));
    }
    notes.push(format!("emulsion: {}", what.join(", ")));
    f
}

/// The digital stage.
pub fn digital(frame: &Frame, plan: &PhotoPlan, aux: &Aux, seed: u32, notes: &mut Vec<String>) -> Frame {
    let _ = seed;
    let (w, h) = (frame.w, frame.h);
    let long = w.max(h) as f32;
    let n = frame.px.len();
    let mut f = frame.clone();
    let mut what = Vec::new();
    // The flash: the frontal light falling off with the depth, the shine on the faces.
    if plan.flash > 0.0 {
        if let Some(m) = &aux.metric {
            let mut v: Vec<f32> = m.v.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let zs = v[v.len() / 3].max(0.1);
            for i in 0..n {
                let k = (zs / m.v[i].max(0.1)).powi(2).clamp(0.25, 2.5);
                let t = 1.0 + plan.flash * 0.7 * (k - 1.0);
                for c in 0..3 {
                    f.px[i][c] *= t;
                }
            }
        }
        for (b, _) in &aux.faces {
            let (x0, y0, x1, y1) = (b[0].max(0.0) as u32, b[1].max(0.0) as u32, (b[2].min(w as f32 - 1.0)) as u32, (b[3].min(h as f32 - 1.0)) as u32);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let i = (y * w + x) as usize;
                    let l = luma(f.px[i]);
                    if l > 0.45 {
                        for c in 0..3 {
                            f.px[i][c] += plan.flash * 0.15 * (l - 0.45);
                        }
                    }
                }
            }
        }
        what.push("flash".into());
    }
    // The red eye.
    if plan.redeye > 0.0 {
        for (b, eyes) in &aux.faces {
            let r = ((b[2] - b[0]).max(b[3] - b[1]) * 0.035).max(1.0);
            for e in eyes {
                for y in (e[1] - r - 1.0).max(0.0) as u32..=((e[1] + r + 1.0).min(h as f32 - 1.0)) as u32 {
                    for x in (e[0] - r - 1.0).max(0.0) as u32..=((e[0] + r + 1.0).min(w as f32 - 1.0)) as u32 {
                        let d = ((x as f32 + 0.5 - e[0]).powi(2) + (y as f32 + 0.5 - e[1]).powi(2)).sqrt();
                        let t = ((r + 0.5 - d) / 1.5).clamp(0.0, 1.0) * plan.redeye;
                        let i = (y * w + x) as usize;
                        let red = [0.85, 0.1, 0.08];
                        for c in 0..3 {
                            f.px[i][c] = f.px[i][c] * (1.0 - t) + red[c] * t;
                        }
                    }
                }
            }
        }
        if !aux.faces.is_empty() {
            what.push("red eye".into());
        }
    }
    // The HDR flattening: the shadows of a lit scene lifted, the lights held; the deep shadows (a night sky)
    // left alone, so the stars do not bloom into blotches.
    if plan.hdr > 0.0 {
        let l = gaussian_field(&f.luma_field(), long / 25.0);
        for i in 0..n {
            let lv = l.v[i];
            let k = (0.45 / (lv + 0.05)).powf(0.6).clamp(0.5, 2.2);
            let ramp = ((lv - 0.03) / 0.12).clamp(0.0, 1.0);
            let t = 1.0 + plan.hdr * (k - 1.0) * ramp;
            for c in 0..3 {
                f.px[i][c] *= t;
            }
        }
        what.push("HDR".into());
    }
    // The face smoothing.
    if plan.face_smooth > 0.0 && !aux.faces.is_empty() {
        let mut mask = Field::new(w, h, 0.0);
        let mut side = 0.0f32;
        for (b, _) in &aux.faces {
            side = side.max((b[2] - b[0]).max(b[3] - b[1]));
            for y in b[1].max(0.0) as u32..(b[3].min(h as f32)) as u32 {
                for x in b[0].max(0.0) as u32..(b[2].min(w as f32)) as u32 {
                    mask.v[(y * w + x) as usize] = 1.0;
                }
            }
        }
        let mask = gaussian_field(&mask, side / 8.0);
        let b = gaussian(&f, (side / 40.0).max(1.0));
        for i in 0..n {
            let t = (mask.v[i] * plan.face_smooth * 0.8).clamp(0.0, 1.0);
            for c in 0..3 {
                f.px[i][c] = f.px[i][c] * (1.0 - t) + b.px[i][c] * t;
            }
        }
        what.push("face smoothing".into());
    }
    // The portrait mode: a phone's synthetic depth of field — the blur computed by the depth as a 50 mm-eq lens
    // at f/2 (times the dial) would give it, focused on the face, the subject cut out by the matte's hard edge
    // (the hair wrong). Only with a face over 2% of the frame, as a phone engages it for a person close by.
    if plan.portrait_mode > 0.0 {
        let face_share: f32 = aux.faces.iter().map(|(b, _)| ((b[2] - b[0]) * (b[3] - b[1])).max(0.0)).sum::<f32>() / (w as f32 * h as f32).max(1.0);
        if face_share > 0.02 {
            if let (Some(m), Some(_)) = (&aux.matte, &aux.metric) {
                let mut synth = plan.clone();
                synth.aperture = 2.0 / plan.portrait_mode.max(0.1);
                synth.focus = "eyes".into();
                synth.focus_distance = 0.0;
                synth.field_curve = 0.0;
                synth.diffraction = 0.0;
                synth.resolution = 0.0;
                synth.format = "ff".into();
                if let Some(sigma) = super::lens::blur_field(&synth, aux, w, h, 50.0, &mut Vec::new()) {
                    let b = super::lens::variable_blur(&f, &sigma, 0.0);
                    for i in 0..n {
                        if m.v[i] <= 0.5 {
                            f.px[i] = b.px[i];
                        }
                    }
                    what.push("portrait mode".into());
                }
            }
        }
    }
    // The sharpening and its halo.
    if plan.sharpen > 0.0 {
        let b = gaussian(&f, 1.0 * long / 1500.0 + 0.8);
        for i in 0..n {
            for c in 0..3 {
                f.px[i][c] += plan.sharpen * 1.5 * (f.px[i][c] - b.px[i][c]);
            }
        }
        what.push("sharpening".into());
    }
    // The JPEG: a real round trip.
    if plan.jpeg > 0.0 {
        let q = plan.jpeg.clamp(1.0, 100.0) as u8;
        let srgb = f.to_srgb();
        let mut buf = Vec::new();
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, q);
        if enc.encode_image(&srgb).is_ok() {
            if let Ok(back) = image::load_from_memory_with_format(&buf, image::ImageFormat::Jpeg) {
                f = Frame::from_srgb(&back.to_rgb8());
                what.push(format!("JPEG q{q}"));
            }
        }
    }
    // The date stamp.
    if !plan.date_stamp.trim().is_empty() {
        let text = if plan.date_stamp.trim() == "auto" { "'98  7 14".to_string() } else { plan.date_stamp.trim().to_string() };
        let mut srgb = f.to_srgb();
        let scale = ((long / 400.0).round() as u32).max(1);
        let tw = crate::paint::measure::text_width(&text, scale);
        let x = w.saturating_sub(tw + 6 * scale);
        let y = h.saturating_sub(7 * scale + 6 * scale);
        crate::paint::measure::draw_text(&mut srgb, x, y, &text, scale, [255, 150, 40]);
        f = Frame::from_srgb(&srgb);
        what.push("date stamp".into());
    }
    if !what.is_empty() {
        notes.push(format!("digital: {}", what.join(", ")));
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ortho_emulsion_darkens_red_and_the_film_curve_holds_the_lights() {
        let mut f = Frame::new(4, 1);
        f.px[0] = [0.8, 0.1, 0.1];
        f.px[1] = [0.1, 0.1, 0.8];
        f.px[2] = [3.0, 3.0, 3.0];
        f.px[3] = [0.18, 0.18, 0.18];
        let mut plan = PhotoPlan::default();
        plan.spectral = "ortho".into();
        plan.curve = "film".into();
        let out = run(&f, &plan, &Aux::default(), 1, &mut Vec::new());
        assert!(out.px[0][0] < out.px[1][0], "red darker than blue: {} {}", out.px[0][0], out.px[1][0]);
        assert!(out.px[2][0] < 1.0 && out.px[2][0] > 0.8, "the lights roll off: {}", out.px[2][0]);
        assert!((out.px[3][0] - 0.18).abs() < 0.1, "mid grey stays near: {}", out.px[3][0]);
        // A digital curve clips.
        plan.curve = "digital".into();
        plan.spectral = "colour".into();
        let out = run(&f, &plan, &Aux::default(), 1, &mut Vec::new());
        assert_eq!(out.px[2][0], 1.0);
    }

    #[test]
    fn the_grain_is_coarser_on_a_small_format_and_the_demosaic_keeps_a_flat_colour() {
        let f = Frame { w: 64, h: 64, px: vec![[0.4, 0.4, 0.4]; 64 * 64] };
        let mut plan = PhotoPlan::default();
        plan.grain = 1.0;
        plan.grain_size = 20.0;
        plan.format = "35mm".into();
        let sd = |fr: &Frame| {
            let m: f32 = fr.px.iter().map(|p| p[0]).sum::<f32>() / fr.px.len() as f32;
            (fr.px.iter().map(|p| (p[0] - m).powi(2)).sum::<f32>() / fr.px.len() as f32).sqrt()
        };
        let small = run(&f, &plan, &Aux::default(), 1, &mut Vec::new());
        plan.format = "8x10".into();
        let large = run(&f, &plan, &Aux::default(), 1, &mut Vec::new());
        assert!(sd(&small) > sd(&large) * 1.5, "{} vs {}", sd(&small), sd(&large));
        assert!(sd(&small) > 0.0003, "{}", sd(&small));
        let g = Frame { w: 16, h: 16, px: vec![[0.3, 0.5, 0.7]; 256] };
        let d = demosaic(&g);
        for p in &d.px {
            assert!((p[0] - 0.3).abs() < 1e-5 && (p[1] - 0.5).abs() < 1e-5 && (p[2] - 0.7).abs() < 1e-5, "{p:?}");
        }
        let mut plan = PhotoPlan::default();
        plan.jpeg = 30.0;
        let mut notes = Vec::new();
        let j = digital(&g, &plan, &Aux::default(), 1, &mut notes);
        assert_eq!(j.w, 16);
        assert!(notes[0].contains("JPEG q30"), "{notes:?}");
    }
}
