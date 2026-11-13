//! The process, the carrier and the age.
//!
//! THE PROCESS: a monochrome print's TONE — the shadows' colour and the paper's, the value between them
//! (sepia, selenium, gold, cyanotype, platinum, albumen, the plate's cold silver, the collodion's cream); the
//! print's density range (`black`, `white`); the hand tint, the picture's own colour laid over the print as a
//! wash.
//!
//! THE CARRIER: the mirror of a direct positive; the mottle of a hand-poured plate and the sheen of its silver
//! in the shadows; the frame or the border — the brass mat's oval of a daguerreotype, the chipped dark edge of a
//! tintype, the card's mount, the Polaroid's and the Instax's white, a print's rim, a slide's mount.
//!
//! THE AGE: the fade (the blacks up, the lights down, the dyes gone toward a hue), the yellowing, the silvering
//! from the edges, the scratches, the dust and the fibres, the foxing — each drawn from the seed, each a dial,
//! the master `age` setting a profile for the dials left at 0.

use super::plan::PhotoPlan;
use super::{fbm, hash01, luma, value_noise, Frame};

/// A toning: the colour of the deepest shadow and of the paper's white (linear light).
pub fn tone_colours(tone: &str) -> Option<([f32; 3], [f32; 3])> {
    match tone.trim().to_ascii_lowercase().as_str() {
        "sepia" => Some(([0.020, 0.011, 0.005], [0.92, 0.84, 0.70])),
        "selenium" => Some(([0.006, 0.005, 0.009], [0.88, 0.88, 0.90])),
        "gold" => Some(([0.010, 0.006, 0.008], [0.90, 0.86, 0.84])),
        "cyanotype" => Some(([0.005, 0.020, 0.080], [0.90, 0.92, 0.95])),
        "platinum" => Some(([0.015, 0.013, 0.011], [0.88, 0.85, 0.78])),
        "palladium" => Some(([0.020, 0.015, 0.010], [0.90, 0.85, 0.74])),
        "albumen" => Some(([0.020, 0.012, 0.006], [0.90, 0.80, 0.58])),
        "salt" => Some(([0.025, 0.015, 0.008], [0.86, 0.78, 0.62])),
        "silver" => Some(([0.012, 0.013, 0.017], [0.88, 0.90, 0.94])),
        "collodion" | "tintype" => Some(([0.010, 0.009, 0.007], [0.78, 0.74, 0.62])),
        "warm" => Some(([0.007, 0.006, 0.005], [0.95, 0.92, 0.86])),
        "cold" | "neutral" => Some(([0.005, 0.005, 0.006], [0.94, 0.95, 0.97])),
        _ => None,
    }
}

/// The process stage: the tone, the density range, the hand tint.
pub fn run(frame: &Frame, plan: &PhotoPlan, seed: u32, notes: &mut Vec<String>) -> Frame {
    let _ = seed;
    let mono = plan.is_mono();
    let mut f = frame.clone();
    let mut what = Vec::new();
    if mono {
        if let Some((shadow, paper)) = tone_colours(&plan.tone) {
            let a = plan.tone_amount.clamp(0.0, 1.0);
            for p in &mut f.px {
                let v = p[0].clamp(0.0, 1.0);
                for c in 0..3 {
                    let toned = shadow[c] * (1.0 - v) + paper[c] * v;
                    p[c] = v * (1.0 - a) + toned * a;
                }
            }
            what.push(format!("{} toned", plan.tone.trim()));
        }
        if plan.tint > 0.0 {
            let k = plan.tint * 0.6;
            for (p, o) in f.px.iter_mut().zip(&frame.px) {
                let l = luma(*o);
                for c in 0..3 {
                    p[c] += k * (o[c] - l);
                }
            }
            what.push("hand tinted".into());
        }
    }
    if plan.black > 0.0 || plan.white < 1.0 {
        let (b, w) = (plan.black.clamp(0.0, 0.9), plan.white.clamp(0.1, 1.0));
        for p in &mut f.px {
            for c in 0..3 {
                p[c] = b + (w - b) * p[c].clamp(0.0, 1.0);
            }
        }
        what.push(format!("the print's range {b:.2}..{w:.2}"));
    }
    if !what.is_empty() {
        notes.push(format!("process: {}", what.join(", ")));
    }
    f
}

/// The carrier stage.
pub fn carrier(frame: &Frame, plan: &PhotoPlan, seed: u32, notes: &mut Vec<String>) -> Frame {
    let (w, h) = (frame.w, frame.h);
    let long = w.max(h) as f32;
    let short = w.min(h) as f32;
    let mut f = frame.clone();
    let mut what = Vec::new();
    if plan.mirror > 0.5 {
        for y in 0..h {
            for x in 0..w / 2 {
                let (i, j) = ((y * w + x) as usize, (y * w + (w - 1 - x)) as usize);
                f.px.swap(i, j);
            }
        }
        what.push("laterally reversed".into());
    }
    if plan.mottle > 0.0 {
        let m = fbm(w, h, long / 5.0, 3, seed ^ 0x40);
        let streak = value_noise(w, h, long / 3.0, seed ^ 0x41);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let s = streak.v[i];
                let k = 1.0 + plan.mottle * (0.3 * (m.v[i] - 0.5) * 2.0 + 0.1 * (s - 0.5) * 2.0 * (y as f32 / h as f32));
                for c in 0..3 {
                    f.px[i][c] *= k.max(0.0);
                }
            }
        }
        what.push("the plate's mottle".into());
    }
    if plan.sheen > 0.0 {
        let silver = [0.85, 0.9, 1.0];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let v = luma(f.px[i]).clamp(0.0, 1.0);
                let grad = 0.12 + 0.12 * (y as f32 / h as f32) + 0.06 * (x as f32 / w as f32);
                let k = 0.3 * plan.sheen * (1.0 - v).powi(4) * grad;
                for c in 0..3 {
                    f.px[i][c] += k * silver[c];
                }
            }
        }
        what.push("the silver's sheen".into());
    }
    let kind = plan.carrier.trim().to_ascii_lowercase();
    if plan.frame > 0.0 && kind != "none" && !kind.is_empty() {
        let b = (plan.frame * short).round().max(1.0);
        let (bottom, colour, oval, chipped): (f32, [f32; 3], bool, bool) = match kind.as_str() {
            "plate" => (b, [0.46, 0.36, 0.14], true, false),
            "tintype" => (b * 0.6, [0.04, 0.035, 0.03], false, true),
            "glass" => (b * 0.6, [0.02, 0.02, 0.025], false, true),
            "card" => (b * 2.2, [0.86, 0.80, 0.66], false, false),
            "polaroid" => (b * 3.2, [0.94, 0.94, 0.92], false, false),
            "instax" => (b * 2.4, [0.95, 0.95, 0.94], false, false),
            "slide" => (b, [0.03, 0.03, 0.03], false, false),
            _ => (b, [0.93, 0.92, 0.90], false, false),
        };
        let (iw, ih) = ((w as f32 - 2.0 * b).max(8.0) as u32, (h as f32 - b - bottom).max(8.0) as u32);
        let inner = f.resized(iw, ih);
        let mut out = Frame::new(w, h);
        let edge_noise = if chipped { Some(fbm(w, h, long / 12.0, 3, seed ^ 0x42)) } else { None };
        let (cx, cy) = (b + iw as f32 / 2.0, b + ih as f32 / 2.0);
        let (rx, ry) = (iw as f32 / 2.0, ih as f32 / 2.0);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let inside_rect = x as f32 >= b && (x as f32) < b + iw as f32 && y as f32 >= b && (y as f32) < b + ih as f32;
                let inside = if oval {
                    let (dx, dy) = ((x as f32 + 0.5 - cx) / rx, (y as f32 + 0.5 - cy) / ry);
                    inside_rect && dx * dx + dy * dy <= 1.0
                } else if let Some(en) = &edge_noise {
                    let dist = (x as f32).min(w as f32 - x as f32).min(y as f32).min(h as f32 - y as f32);
                    inside_rect && dist > b * 0.4 * (0.4 + en.v[i] * 1.6)
                } else {
                    inside_rect
                };
                if inside {
                    out.px[i] = inner.px[((y as f32 - b) as u32 * iw + (x as f32 - b) as u32) as usize];
                } else {
                    let mut c = colour;
                    if kind == "plate" {
                        let g = 1.0 + 0.25 * ((x as f32 / w as f32) - (y as f32 / h as f32));
                        for k in 0..3 {
                            c[k] *= g;
                        }
                    }
                    if kind == "card" || kind == "polaroid" || kind == "instax" {
                        let t = hash01(x, y, seed ^ 0x43);
                        for k in 0..3 {
                            c[k] *= 1.0 + 0.02 * (t - 0.5);
                        }
                    }
                    out.px[i] = c;
                }
            }
        }
        f = out;
        what.push(format!("on a {kind}, framed"));
    } else if kind != "none" && !kind.is_empty() {
        what.push(format!("on a {kind}"));
    }
    if !what.is_empty() {
        notes.push(format!("carrier: {}", what.join(", ")));
    }
    f
}

/// An RGB from a hue in degrees (full saturation, mid value), linear.
fn hue_rgb(deg: f32) -> [f32; 3] {
    let h = deg.rem_euclid(360.0) / 60.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    let (r, g, b) = match h as u32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    [r, g, b]
}

/// The age stage.
pub fn age(frame: &Frame, plan: &PhotoPlan, seed: u32, notes: &mut Vec<String>) -> Frame {
    let (w, h) = (frame.w, frame.h);
    let long = w.max(h) as f32;
    let n = frame.px.len();
    let paper_carrier = matches!(plan.carrier.trim(), "card" | "print" | "polaroid" | "instax" | "none" | "");
    let pick = |key: f32, default: f32| if plan.age > 0.0 { (if key > 0.0 { key } else { default }) * plan.age } else { key };
    let fade = pick(plan.fade, 0.5);
    let yellowing = pick(plan.yellowing, 0.4);
    let silvering = pick(plan.silvering, if plan.is_mono() { 0.3 } else { 0.0 });
    let scratches = pick(plan.scratches, 0.3);
    let dust = pick(plan.dust, 0.4);
    let foxing = pick(plan.foxing, if paper_carrier { 0.2 } else { 0.0 });
    if fade <= 0.0 && yellowing <= 0.0 && silvering <= 0.0 && scratches <= 0.0 && dust <= 0.0 && foxing <= 0.0 {
        return frame.clone();
    }
    let mut f = frame.clone();
    let mut what = Vec::new();
    if fade > 0.0 {
        let hue = if plan.fade_hue != 0.0 { Some(hue_rgb(plan.fade_hue)) } else { None };
        for p in &mut f.px {
            for c in 0..3 {
                // The blacks come up to a few per cent of linear light at a full fade, the lights lose a little.
                p[c] = p[c] * (1.0 - 0.3 * fade) + 0.06 * fade;
                if let Some(hr) = hue {
                    p[c] += fade * 0.12 * (hr[c] - 0.5);
                }
            }
        }
        what.push(format!("faded {fade:.2}"));
    }
    if yellowing > 0.0 {
        let tint = [1.0, 0.92, 0.72];
        for p in &mut f.px {
            let l = luma(*p).clamp(0.0, 1.0);
            for c in 0..3 {
                p[c] *= 1.0 - yellowing * (1.0 - tint[c]) * (0.4 + 0.6 * l);
            }
        }
        what.push(format!("yellowed {yellowing:.2}"));
    }
    if silvering > 0.0 {
        let nz = fbm(w, h, long / 5.0, 3, seed ^ 0x50);
        let silver = [0.55, 0.60, 0.72];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let dist = (x as f32 / w as f32).min(1.0 - x as f32 / w as f32).min(y as f32 / h as f32).min(1.0 - y as f32 / h as f32);
                let edge = (1.0 - dist / 0.3).clamp(0.0, 1.0);
                let l = luma(f.px[i]).clamp(0.0, 1.0);
                let m = (silvering * edge * (nz.v[i] * 1.6 - 0.3) * (1.0 - l)).clamp(0.0, 1.0);
                for c in 0..3 {
                    f.px[i][c] = f.px[i][c] * (1.0 - m) + silver[c] * (0.5 + 0.5 * l) * m;
                }
            }
        }
        what.push(format!("silvered {silvering:.2}"));
    }
    if scratches > 0.0 {
        let count = (scratches * 40.0 * (n as f32 / 1e6).sqrt().max(0.3)) as u32;
        for k in 0..count {
            let x0 = hash01(k, 1, seed ^ 0x60) * w as f32;
            let y0 = hash01(k, 2, seed ^ 0x60) * h as f32;
            let len = (0.1 + 0.5 * hash01(k, 3, seed ^ 0x60)) * long;
            let ang = std::f32::consts::FRAC_PI_2 + (hash01(k, 4, seed ^ 0x60) - 0.5) * 0.8;
            let light = hash01(k, 5, seed ^ 0x60) > 0.4;
            let steps = len as u32;
            for s in 0..steps {
                let t = s as f32;
                let wob = (t / 17.0).sin() * 1.5;
                let x = x0 + ang.cos() * t + wob;
                let y = y0 + ang.sin() * t;
                if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 {
                    break;
                }
                let i = (y as u32 * w + x as u32) as usize;
                let a = 0.55 * scratches.min(1.0);
                for c in 0..3 {
                    f.px[i][c] = f.px[i][c] * (1.0 - a) + if light { 0.9 } else { 0.03 } * a;
                }
            }
        }
        what.push(format!("{count} scratches"));
    }
    if dust > 0.0 {
        let count = (dust * 300.0 * (n as f32 / 1e6).max(0.05)) as u32;
        for k in 0..count {
            let x = hash01(k, 11, seed ^ 0x70) * w as f32;
            let y = hash01(k, 12, seed ^ 0x70) * h as f32;
            let r = 0.5 + 1.2 * hash01(k, 13, seed ^ 0x70);
            let light = hash01(k, 14, seed ^ 0x70) > 0.5;
            for yy in (y - r - 1.0).max(0.0) as u32..=((y + r + 1.0).min(h as f32 - 1.0)) as u32 {
                for xx in (x - r - 1.0).max(0.0) as u32..=((x + r + 1.0).min(w as f32 - 1.0)) as u32 {
                    let d = ((xx as f32 + 0.5 - x).powi(2) + (yy as f32 + 0.5 - y).powi(2)).sqrt();
                    let a = ((r + 0.5 - d) / 1.0).clamp(0.0, 1.0) * 0.8 * dust.min(1.0);
                    let i = (yy * w + xx) as usize;
                    for c in 0..3 {
                        f.px[i][c] = f.px[i][c] * (1.0 - a) + if light { 0.95 } else { 0.02 } * a;
                    }
                }
            }
        }
        what.push(format!("{count} specks"));
    }
    if foxing > 0.0 {
        let nz = fbm(w, h, long / 8.0, 4, seed ^ 0x80);
        let brown = [0.50, 0.30, 0.12];
        for i in 0..n {
            let m = ((nz.v[i] - 0.62) / 0.15).clamp(0.0, 1.0) * foxing * 0.7;
            let l = luma(f.px[i]).clamp(0.0, 1.0);
            for c in 0..3 {
                f.px[i][c] = f.px[i][c] * (1.0 - m) + brown[c] * (0.5 + 0.5 * l) * m;
            }
        }
        what.push(format!("foxed {foxing:.2}"));
    }
    notes.push(format!("age: {}", what.join(", ")));
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sepia_tone_warms_the_lights_and_the_range_lifts_the_black() {
        let mut f = Frame::new(3, 1);
        f.px[0] = [0.0; 3];
        f.px[1] = [0.5; 3];
        f.px[2] = [1.0; 3];
        let mut plan = PhotoPlan::default();
        plan.mono = 1.0;
        plan.tone = "sepia".into();
        plan.black = 0.1;
        plan.white = 0.9;
        let mut notes = Vec::new();
        let out = run(&f, &plan, 1, &mut notes);
        assert!(out.px[2][0] > out.px[2][2], "the paper is warm: {:?}", out.px[2]);
        assert!(out.px[0][0] >= 0.1 && out.px[2][0] <= 0.9, "{:?}", out.px);
        assert!(notes[0].contains("sepia toned"));
    }

    #[test]
    fn the_mirror_reverses_the_plate_frame_opens_an_oval_and_the_age_leaves_its_marks() {
        let (w, h) = (64u32, 48u32);
        let mut f = Frame::new(w, h);
        for y in 0..h {
            for x in 0..w {
                f.px[(y * w + x) as usize] = [if x < w / 2 { 0.9 } else { 0.1 }; 3];
            }
        }
        let mut plan = PhotoPlan::default();
        plan.mirror = 1.0;
        let out = carrier(&f, &plan, 1, &mut Vec::new());
        assert!(out.px[(10 * w + 2) as usize][0] < 0.5 && out.px[(10 * w + w - 3) as usize][0] > 0.5);
        let mut plan = PhotoPlan::default();
        plan.carrier = "plate".into();
        plan.frame = 0.1;
        let out = carrier(&f, &plan, 1, &mut Vec::new());
        // The corner is the brass mat, the centre the picture.
        assert!((out.px[0][0] - 0.46).abs() < 0.15, "{:?}", out.px[0]);
        let c = out.px[(24 * w + 20) as usize];
        assert!(c[0] > 0.5, "{c:?}");
        let mut plan = PhotoPlan::default();
        plan.age = 0.8;
        let mut notes = Vec::new();
        let aged = age(&f, &plan, 7, &mut notes);
        assert_ne!(aged, f);
        assert!(notes[0].contains("faded") && notes[0].contains("specks"), "{notes:?}");
        // The dark half came up: the fade.
        assert!(aged.px[(24 * w + 60) as usize][0] > 0.105, "{}", aged.px[(24 * w + 60) as usize][0]);
        let again = age(&f, &plan, 7, &mut Vec::new());
        assert_eq!(aged, again, "the same seed, the same marks");
    }
}
