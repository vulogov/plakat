//! The virtual camera: the picture re-photographed from another point with another lens.
//!
//! The picture's depth (relative, near = 1) is given a scale — the subject's distance in metres, read from a
//! face (the eyes 63 mm apart) or given — and every pixel becomes a point in space through the SOURCE lens
//! (`source_focal`, 35 mm-equivalent). The camera is moved (`dolly`, `truck`, `pedestal`) and turned (`pan`,
//! `tilt`, `roll`), the lens changed (`focal`), and the points are projected again. A FORWARD pass with a
//! z-buffer finds which point each new pixel sees; a BACKWARD pass takes that pixel's depth back through the
//! move to the source and samples it bilinearly, checking the depth it lands on is the depth it left from (a
//! pixel that lands on a nearer thing is a hole: the ground the move opened). The holes are filled by
//! `fill`: "propagate" grows the far ground into them (the far pixels weigh more, so a figure's edge does not
//! smear into its background), "neural" leaves them to a model, "none" leaves them black. The share invented
//! is reported and the critic keeps it under `fill_cap`.
//!
//! `keep_subject` ties the focal length and the dolly by the subject's size: the 24 mm portrait re-shot on 85
//! from three times the distance, the face's proportions changing as a lens would change them.

use super::plan::PhotoPlan;
use super::{Aux, Field, Frame};

/// What the camera returns: the frame, the fields moved with it, the holes.
pub struct CameraOut {
    pub frame: Frame,
    pub depth: Option<Field>,
    /// The metric depth after the move (metres), for the lens.
    pub metric: Option<Field>,
    pub matte: Option<Field>,
    pub faces: Vec<([f32; 4], [[f32; 2]; 2])>,
    pub invented: f32,
    pub holes: Option<Field>,
    pub distance: f32,
    pub focal_out: f32,
}

/// Relative depth (near = 1, far = 0) → a factor of the subject's distance: `z = distance · g(rel_s) / g(rel)`,
/// the far ground `range` times the near (the plan's `depth_range`).
#[inline]
fn g(rel: f32, range: f32) -> f32 {
    let a = 1.0 / range.max(1.01);
    a + (1.0 - a) * rel.clamp(0.0, 1.0)
}

/// The subject's distance in metres: from the plan, else a face's eyes, else 3 m.
pub fn subject_distance(plan: &PhotoPlan, aux: &Aux, fx: f32) -> (f32, String) {
    if plan.distance > 0.0 {
        return (plan.distance, "given".into());
    }
    if let Some((_, eyes)) = aux.faces.iter().max_by(|a, b| face_side(&a.0).partial_cmp(&face_side(&b.0)).unwrap_or(std::cmp::Ordering::Equal)) {
        let d = ((eyes[0][0] - eyes[1][0]).powi(2) + (eyes[0][1] - eyes[1][1]).powi(2)).sqrt();
        if d > 2.0 {
            let z = fx * 0.063 / d;
            return (z.clamp(0.2, 50.0), format!("from the eyes {d:.0} px apart"));
        }
    }
    (3.0, "the default".into())
}

fn face_side(b: &[f32; 4]) -> f32 {
    (b[2] - b[0]).max(b[3] - b[1])
}

/// The subject's relative depth: the matte's median, else the largest face's, else the central region's.
fn subject_rel(depth: &Field, aux: &Aux) -> f32 {
    let mut vals: Vec<f32> = Vec::new();
    if let Some(m) = &aux.matte {
        for (i, v) in m.v.iter().enumerate() {
            if *v > 0.5 {
                vals.push(depth.v[i]);
            }
        }
    }
    if vals.len() < 16 {
        if let Some((b, _)) = aux.faces.iter().max_by(|a, c| face_side(&a.0).partial_cmp(&face_side(&c.0)).unwrap_or(std::cmp::Ordering::Equal)) {
            for y in (b[1].max(0.0) as u32)..(b[3].min(depth.h as f32 - 1.0) as u32) {
                for x in (b[0].max(0.0) as u32)..(b[2].min(depth.w as f32 - 1.0) as u32) {
                    vals.push(depth.v[(y * depth.w + x) as usize]);
                }
            }
        }
    }
    if vals.len() < 16 {
        for y in depth.h / 4..depth.h * 3 / 4 {
            for x in depth.w / 4..depth.w * 3 / 4 {
                vals.push(depth.v[(y * depth.w + x) as usize]);
            }
        }
    }
    if vals.is_empty() {
        return 0.5;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals[vals.len() / 2]
}

/// A 3×3 median then a σ 1.5 Gaussian over a depth field.
pub fn smooth_depth(d: &Field) -> Field {
    let (w, h) = (d.w as i64, d.h as i64);
    let mut med = Field::new(d.w, d.h, 0.0);
    for y in 0..h {
        for x in 0..w {
            let mut v = [0.0f32; 9];
            let mut k = 0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    v[k] = d.get(x + dx, y + dy);
                    k += 1;
                }
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            med.v[(y * w + x) as usize] = v[4];
        }
    }
    super::gaussian_field(&med, 1.5)
}

/// A 3×3 rotation, row-major.
type Mat = [[f32; 3]; 3];

fn mul(a: &Mat, b: &Mat) -> Mat {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

fn apply(m: &Mat, p: [f32; 3]) -> [f32; 3] {
    [m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2], m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2], m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2]]
}

fn transpose(m: &Mat) -> Mat {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = m[j][i];
        }
    }
    o
}

/// The world → new-camera rotation for a pan (right +), a tilt (up +) and a roll (clockwise +), degrees;
/// the image's y points down.
fn rotation(pan: f32, tilt: f32, roll: f32) -> Mat {
    let (sp, cp) = pan.to_radians().sin_cos();
    let (st, ct) = tilt.to_radians().sin_cos();
    let (sr, cr) = roll.to_radians().sin_cos();
    // A pan to the right: a point ahead moves to the left of the frame.
    let ry: Mat = [[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]];
    // A tilt up: a point ahead moves down the frame.
    let rx: Mat = [[1.0, 0.0, 0.0], [0.0, ct, st], [0.0, -st, ct]];
    // A roll clockwise: the picture turns counter-clockwise.
    let rz: Mat = [[cr, sr, 0.0], [-sr, cr, 0.0], [0.0, 0.0, 1.0]];
    mul(&rz, &mul(&rx, &ry))
}

/// Is there a camera move or a lens change in the plan?
pub fn moves(plan: &PhotoPlan) -> bool {
    plan.focal > 0.0 || plan.dolly != 0.0 || plan.truck != 0.0 || plan.pedestal != 0.0 || plan.pan != 0.0 || plan.tilt != 0.0 || plan.roll != 0.0 || plan.shift != 0.0
}

/// Run the camera. Without a move, the frame passes through and only the metric depth is derived.
pub fn run(frame: &Frame, plan: &PhotoPlan, aux: &Aux, notes: &mut Vec<String>) -> anyhow::Result<CameraOut> {
    let (w, h) = (frame.w, frame.h);
    let long = w.max(h) as f32;
    let f_s = if plan.source_focal > 0.0 { plan.source_focal } else { 35.0 };
    let fx_s = f_s / 36.0 * long;
    let (distance, how) = subject_distance(plan, aux, fx_s);
    // The depth smoothed (a 3×3 median, then σ 1.5): the model's depth is read at 1024 and resampled, and its
    // jaggedness would become waves on every vertical under a parallax.
    let depth = aux.depth.as_ref().map(smooth_depth);
    let rel_s = depth.as_ref().map(|d| subject_rel(d, aux)).unwrap_or(0.5);
    let range = if plan.depth_range > 0.0 { plan.depth_range } else { 20.0 };
    let metric_of = |rel: f32| distance * g(rel_s, range) / g(rel, range);
    let metric = depth.as_ref().map(|d| Field { w, h, v: d.v.iter().map(|r| metric_of(*r)).collect() });
    if !moves(plan) {
        return Ok(CameraOut { frame: frame.clone(), depth, metric, matte: aux.matte.clone(), faces: aux.faces.clone(), invented: 0.0, holes: None, distance, focal_out: f_s });
    }
    // The lens and the dolly, tied by the subject when asked.
    let mut f_t = if plan.focal > 0.0 { plan.focal } else { f_s };
    let mut dolly = plan.dolly;
    if plan.keep_subject > 0.5 {
        if plan.focal > 0.0 && plan.dolly == 0.0 {
            dolly = distance * (1.0 - f_t / f_s);
        } else if plan.dolly != 0.0 && plan.focal <= 0.0 {
            f_t = f_s * (distance - dolly).max(0.1) / distance;
        }
    }
    if dolly >= distance * 0.9 {
        let held = distance * 0.5;
        notes.push(format!("camera: a dolly of {dolly:.2} m would pass the subject at {distance:.2} m — held at {held:.2} m"));
        dolly = held;
    }
    let fx_t = f_t / 36.0 * long;
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let cy_t = cy + plan.shift * h as f32;
    let t = [plan.truck, -plan.pedestal, dolly];
    let r = rotation(plan.pan, plan.tilt, plan.roll);
    let r_inv = transpose(&r);
    let flat = Field::new(w, h, rel_s);
    // The subject moves as one body: inside its matte the depth is at least its median (a white coat read as
    // the white wall behind it would otherwise tear away with the wall).
    let unified = depth.as_ref().map(|d| {
        let mut d = d.clone();
        if let Some(m) = &aux.matte {
            for i in 0..d.v.len() {
                if m.v[i] > 0.5 && d.v[i] < rel_s {
                    d.v[i] = rel_s;
                }
            }
        }
        d
    });
    let d_rel = unified.as_ref().unwrap_or(&flat);
    if depth.is_none() {
        notes.push("camera: no depth was read — the move is a flat shift, the lens a crop".into());
    }
    // THE FORWARD PASS: a z-buffer of the new frame.
    let n = (w as usize) * (h as usize);
    let mut zbuf = vec![f32::INFINITY; n];
    for v in 0..h {
        for u in 0..w {
            let z = metric_of(d_rel.v[(v * w + u) as usize]);
            let p = [(u as f32 + 0.5 - cx) / fx_s * z, (v as f32 + 0.5 - cy) / fx_s * z, z];
            let q = apply(&r, [p[0] - t[0], p[1] - t[1], p[2] - t[2]]);
            if q[2] <= 0.05 {
                continue;
            }
            let un = cx + fx_t * q[0] / q[2];
            let vn = cy_t + fx_t * q[1] / q[2];
            let m = (fx_t / fx_s) * (z / q[2]);
            let rad = (m.ceil() as i64).clamp(2, 6);
            let (ui, vi) = (un.floor() as i64, vn.floor() as i64);
            for dy in -(rad / 2)..=(rad / 2 + rad % 2) {
                for dx in -(rad / 2)..=(rad / 2 + rad % 2) {
                    let (x, y) = (ui + dx, vi + dy);
                    if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                        continue;
                    }
                    let i = (y as u32 * w + x as u32) as usize;
                    if q[2] < zbuf[i] {
                        zbuf[i] = q[2];
                    }
                }
            }
        }
    }
    // THE BACKWARD PASS: each new pixel's depth taken back through the move to the source.
    let mut out = Frame::new(w, h);
    let mut holes = Field::new(w, h, 1.0);
    let mut depth_out = Field::new(w, h, 0.0);
    let mut metric_out = Field::new(w, h, 0.0);
    let mut matte_out = aux.matte.as_ref().map(|_| Field::new(w, h, 0.0));
    let (mut outside, mut hidden) = (0usize, 0usize);
    let mut past_edge = Field::new(w, h, 0.0);
    let zmax = metric_of(0.0);
    let a0 = 1.0 / range.max(1.01);
    let rel_of = |z: f32| ((distance * g(rel_s, range) / z.max(1e-6) - a0) / (1.0 - a0)).clamp(0.0, 1.0);
    for v in 0..h {
        for u in 0..w {
            let i = (v * w + u) as usize;
            let zq = zbuf[i];
            if !zq.is_finite() {
                // No pixel reached here; taken back at the far ground, does the ray leave the picture? Then it
                // is the band past the picture's edge, not a tear.
                let q = [(u as f32 + 0.5 - cx) / fx_t * zmax, (v as f32 + 0.5 - cy_t) / fx_t * zmax, zmax];
                let p0 = apply(&r_inv, q);
                let p = [p0[0] + t[0], p0[1] + t[1], p0[2] + t[2]];
                if p[2] > 0.05 {
                    let us = cx + fx_s * p[0] / p[2];
                    let vs = cy + fx_s * p[1] / p[2];
                    if us < 0.0 || vs < 0.0 || us > w as f32 || vs > h as f32 {
                        outside += 1;
                        past_edge.v[i] = 1.0;
                    }
                }
                continue;
            }
            let q = [(u as f32 + 0.5 - cx) / fx_t * zq, (v as f32 + 0.5 - cy_t) / fx_t * zq, zq];
            let p0 = apply(&r_inv, q);
            let p = [p0[0] + t[0], p0[1] + t[1], p0[2] + t[2]];
            if p[2] <= 0.05 {
                continue;
            }
            let us = cx + fx_s * p[0] / p[2];
            let vs = cy + fx_s * p[1] / p[2];
            if us < 0.0 || vs < 0.0 || us > w as f32 || vs > h as f32 {
                outside += 1;
                past_edge.v[i] = 1.0;
                continue;
            }
            let z_src = metric_of(d_rel.get(us.floor() as i64, vs.floor() as i64));
            if (z_src - p[2]).abs() > 0.06 * p[2] + 0.02 {
                hidden += 1;
                continue;
            }
            out.px[i] = frame.sample(us, vs);
            holes.v[i] = 0.0;
            metric_out.v[i] = zq;
            depth_out.v[i] = rel_of(zq);
            if let (Some(mo), Some(m)) = (matte_out.as_mut(), aux.matte.as_ref()) {
                mo.v[i] = m.sample(us, vs);
            }
        }
    }
    let fill = plan.fill.trim().to_ascii_lowercase();
    // THE CROP: past the picture's edge the camera sees nothing the picture has; unless the holes are to be
    // shown ("none") or invented by a model ("neural"), the frame is cropped to the rows and columns the picture
    // reached (a row or column over a quarter past the edge is cut; the rest of the band is grown into) and
    // brought back to its size — a small move trims the edge, it does not smear it.
    let mut crop: Option<(u32, u32, u32, u32)> = None;
    if outside > 0 && fill != "none" && fill != "neural" {
        let col = |x: u32| (0..h).filter(|y| past_edge.v[(y * w + x) as usize] > 0.5).count() as f32 / h as f32;
        let row = |y: u32| (0..w).filter(|x| past_edge.v[(y * w + x) as usize] > 0.5).count() as f32 / w as f32;
        let (mut x0, mut x1, mut y0, mut y1) = (0u32, w, 0u32, h);
        while x0 + 1 < x1 && col(x0) > 0.25 {
            x0 += 1;
        }
        while x1 > x0 + 1 && col(x1 - 1) > 0.25 {
            x1 -= 1;
        }
        while y0 + 1 < y1 && row(y0) > 0.25 {
            y0 += 1;
        }
        while y1 > y0 + 1 && row(y1 - 1) > 0.25 {
            y1 -= 1;
        }
        let (cw, ch) = (x1 - x0, y1 - y0);
        if (x0 > 0 || y0 > 0 || x1 < w || y1 < h) && cw >= w / 2 && ch >= h / 2 {
            let sub_frame = |f: &Frame| {
                let mut o = Frame::new(cw, ch);
                for y in 0..ch {
                    for x in 0..cw {
                        o.px[(y * cw + x) as usize] = f.px[((y + y0) * w + x + x0) as usize];
                    }
                }
                o.resized(w, h)
            };
            let sub_field = |f: &Field| {
                let mut o = Field::new(cw, ch, 0.0);
                for y in 0..ch {
                    for x in 0..cw {
                        o.v[(y * cw + x) as usize] = f.v[((y + y0) * w + x + x0) as usize];
                    }
                }
                o.resized(w, h)
            };
            out = sub_frame(&out);
            holes = sub_field(&holes);
            depth_out = sub_field(&depth_out);
            metric_out = sub_field(&metric_out);
            matte_out = matte_out.as_ref().map(sub_field);
            for v in holes.v.iter_mut() {
                *v = if *v > 0.5 { 1.0 } else { 0.0 };
            }
            crop = Some((x0, y0, cw, ch));
        }
    }
    let hole_count = holes.v.iter().filter(|v| **v > 0.5).count();
    let invented = hole_count as f32 / n.max(1) as f32;
    // The holes' depth: the far ground (so the lens blurs them as background), the max of the frame.
    for i in 0..n {
        if holes.v[i] > 0.5 {
            metric_out.v[i] = zmax;
            depth_out.v[i] = 0.0;
        }
    }
    if hole_count > 0 && fill != "none" && fill != "neural" {
        propagate(&mut out, &holes, &metric_out);
    }
    // The faces, moved with the frame (the box's centre through the move, the side by the magnification).
    let faces = aux.faces.iter().filter_map(|(b, eyes)| {
        let (bx, by) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
        let rel = d_rel.sample(bx, by);
        let z = metric_of(rel);
        let p = [(bx - cx) / fx_s * z, (by - cy) / fx_s * z, z];
        let q = apply(&r, [p[0] - t[0], p[1] - t[1], p[2] - t[2]]);
        if q[2] <= 0.05 {
            return None;
        }
        let m = (fx_t / fx_s) * (z / q[2]);
        let (nx, ny) = (cx + fx_t * q[0] / q[2], cy_t + fx_t * q[1] / q[2]);
        let (hw, hh) = ((b[2] - b[0]) / 2.0 * m, (b[3] - b[1]) / 2.0 * m);
        if nx + hw < 0.0 || ny + hh < 0.0 || nx - hw > w as f32 || ny - hh > h as f32 {
            return None;
        }
        let e = |pt: [f32; 2]| [nx + (pt[0] - bx) * m, ny + (pt[1] - by) * m];
        let (b, e) = ([nx - hw, ny - hh, nx + hw, ny + hh], [e(eyes[0]), e(eyes[1])]);
        // Through the crop: the box's corners and the eyes moved and scaled with the frame.
        let (b, e) = match crop {
            Some((x0, y0, cw, ch)) => {
                let (sx, sy) = (w as f32 / cw as f32, h as f32 / ch as f32);
                let c = |pt: [f32; 2]| [(pt[0] - x0 as f32) * sx, (pt[1] - y0 as f32) * sy];
                let (p0, p1) = (c([b[0], b[1]]), c([b[2], b[3]]));
                ([p0[0], p0[1], p1[0], p1[1]], [c(e[0]), c(e[1])])
            }
            None => (b, e),
        };
        Some((b, e))
    }).collect();
    let mut what = Vec::new();
    if (f_t - f_s).abs() > 0.01 {
        what.push(format!("{f_s:.0} mm → {f_t:.0} mm"));
    }
    if dolly != 0.0 {
        what.push(format!("dolly {dolly:+.2} m"));
    }
    if plan.truck != 0.0 {
        what.push(format!("truck {:+.2} m", plan.truck));
    }
    if plan.pedestal != 0.0 {
        what.push(format!("pedestal {:+.2} m", plan.pedestal));
    }
    if plan.pan != 0.0 || plan.tilt != 0.0 || plan.roll != 0.0 {
        what.push(format!("pan {:+.1}° tilt {:+.1}° roll {:+.1}°", plan.pan, plan.tilt, plan.roll));
    }
    if plan.shift != 0.0 {
        what.push(format!("shift {:+.2}", plan.shift));
    }
    if plan.keep_subject > 0.5 {
        what.push("the subject's size kept".into());
    }
    let unseen = hole_count.saturating_sub(hidden + outside);
    let cropped = match crop {
        Some((_, _, cw, ch)) => format!("; the frame cropped to {:.0}%×{:.0}% past the picture's edge and brought back to size", cw as f32 / w as f32 * 100.0, ch as f32 / h as f32 * 100.0),
        None => String::new(),
    };
    notes.push(format!(
        "camera: {} at {distance:.2} m ({how}); invented {:.1}% of the frame ({:.1}% opened behind things, {:.1}% outside the picture, {:.1}% no pixel reached), filled by {fill}{cropped}",
        what.join(", "),
        invented * 100.0,
        hidden as f32 / n as f32 * 100.0,
        outside as f32 / n as f32 * 100.0,
        unseen.min(hole_count) as f32 / n as f32 * 100.0
    ));
    Ok(CameraOut { frame: out, depth: Some(depth_out), metric: Some(metric_out), matte: matte_out, faces, invented, holes: Some(holes), distance, focal_out: f_t })
}

/// Grow the ground into the holes: eight passes of the 3×3 "farthest neighbours" (the background wins over a
/// figure's edge), then a normalised convolution for what is left, the far pixels weighing more, and a
/// softening inside the holes so the fill does not streak.
pub fn propagate(frame: &mut Frame, holes: &Field, metric: &Field) {
    let (w, h) = (frame.w as i64, frame.h as i64);
    let n = (w * h) as usize;
    let mut filled = holes.v.iter().map(|v| *v < 0.5).collect::<Vec<bool>>();
    let mut z = metric.v.clone();
    // Each hole is a disocclusion: its true content lies behind the thing that hid it. The holes are labelled
    // into components and each component's boundary gives the depth it may be filled from — anything clearly
    // behind its NEAR side (a quarter farther than the nearest boundary pixel), so a figure's edge does not
    // creep into the ground it uncovered while the ground on every other side grows in; a hole whose boundary
    // is all at one depth (a tear in a ramp) is filled from any side.
    let mut label = vec![u32::MAX; n];
    let mut need_of: Vec<f32> = Vec::new();
    for start in 0..n {
        if filled[start] || label[start] != u32::MAX {
            continue;
        }
        let id = need_of.len() as u32;
        let mut stack = vec![start];
        label[start] = id;
        let (mut near, mut far) = (f32::INFINITY, 0.0f32);
        while let Some(i) = stack.pop() {
            let (x, y) = ((i as i64) % w, (i as i64) / w);
            for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= w || ny >= h {
                    continue;
                }
                let j = (ny * w + nx) as usize;
                if filled[j] {
                    far = far.max(z[j]);
                    near = near.min(z[j]);
                } else if label[j] == u32::MAX {
                    label[j] = id;
                    stack.push(j);
                }
            }
        }
        need_of.push(if near.is_finite() && far >= near * 1.25 { near * 1.25 } else { 0.0 });
    }
    // The passes over the shrinking list of unfilled pixels: each takes the mean of its valid neighbours behind
    // the near side, until no pass fills anything (the fill advances one pixel a pass from every boundary that
    // qualifies, so the hole's sides meet in its middle).
    let mut remaining: Vec<usize> = (0..n).filter(|i| !filled[*i]).collect();
    let max_passes = (w.max(h) as usize) * 2;
    for _ in 0..max_passes {
        if remaining.is_empty() {
            break;
        }
        let mut writes: Vec<(usize, [f32; 3], f32)> = Vec::new();
        let mut still = Vec::with_capacity(remaining.len());
        for &i in &remaining {
            let (x, y) = ((i as i64) % w, (i as i64) / w);
            let mut far = f32::NEG_INFINITY;
            let mut cands: Vec<(usize, f32)> = Vec::with_capacity(8);
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w || ny >= h || (dx == 0 && dy == 0) {
                        continue;
                    }
                    let j = (ny * w + nx) as usize;
                    if filled[j] {
                        cands.push((j, z[j]));
                        far = far.max(z[j]);
                    }
                }
            }
            let need = need_of.get(label[i] as usize).copied().unwrap_or(0.0);
            if cands.is_empty() || far < need {
                still.push(i);
                continue;
            }
            let mut acc = [0.0f32; 3];
            let mut cnt = 0.0;
            for (j, zj) in &cands {
                if *zj >= need {
                    for c in 0..3 {
                        acc[c] += frame.px[*j][c];
                    }
                    cnt += 1.0;
                }
            }
            writes.push((i, [acc[0] / cnt, acc[1] / cnt, acc[2] / cnt], far));
        }
        if writes.is_empty() {
            break;
        }
        for (i, px, zf) in writes {
            frame.px[i] = px;
            z[i] = zf;
            filled[i] = true;
        }
        remaining = still;
    }
    if filled.iter().all(|f| *f) {
        soften(frame, holes);
        return;
    }
    // The normalised convolution for the holes left, the far pixels weighing more.
    let zmax = z.iter().cloned().fold(0.0f32, f32::max).max(1e-3);
    let weight = Field { w: frame.w, h: frame.h, v: (0..n).map(|i| if filled[i] { (z[i] / zmax).powi(2).max(0.05) } else { 0.0 }).collect() };
    let mut remaining: Vec<usize> = (0..n).filter(|i| !filled[*i]).collect();
    for sigma in [4.0f32, 16.0, 64.0, 256.0] {
        if remaining.is_empty() {
            break;
        }
        let wb = super::gaussian_field(&weight, sigma);
        let mut cols = [Field::new(frame.w, frame.h, 0.0), Field::new(frame.w, frame.h, 0.0), Field::new(frame.w, frame.h, 0.0)];
        for c in 0..3 {
            for i in 0..n {
                cols[c].v[i] = frame.px[i][c] * weight.v[i];
            }
            cols[c] = super::gaussian_field(&cols[c], sigma);
        }
        let mut still = Vec::new();
        for &i in &remaining {
            if wb.v[i] > 1e-4 {
                frame.px[i] = [cols[0].v[i] / wb.v[i], cols[1].v[i] / wb.v[i], cols[2].v[i] / wb.v[i]];
            } else {
                still.push(i);
            }
        }
        remaining = still;
    }
    for &i in &remaining {
        frame.px[i] = [0.0; 3];
    }
    soften(frame, holes);
}

/// A σ=1 blur inside the holes (feathered a pixel out), so the grown ground does not streak.
fn soften(frame: &mut Frame, holes: &Field) {
    let b = super::gaussian(frame, 1.0);
    let m = super::gaussian_field(holes, 1.0);
    for i in 0..frame.px.len() {
        let t = m.v[i].clamp(0.0, 1.0);
        if t > 0.0 {
            for c in 0..3 {
                frame.px[i][c] = frame.px[i][c] * (1.0 - t) + b.px[i][c] * t;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame: a grey far wall, a near red square in the left half.
    fn scene() -> (Frame, Aux) {
        let (w, h) = (64u32, 48u32);
        let mut f = Frame::new(w, h);
        let mut d = Field::new(w, h, 0.0);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let near = (12..28).contains(&x) && (14..34).contains(&y);
                f.px[i] = if near { [0.8, 0.1, 0.1] } else { [0.5, 0.5, 0.5] };
                d.v[i] = if near { 1.0 } else { 0.1 };
            }
        }
        (f, Aux { depth: Some(d), ..Default::default() })
    }

    #[test]
    fn a_pan_to_the_right_moves_the_picture_left_and_a_truck_opens_ground_behind_the_near_thing() {
        let (f, aux) = scene();
        let mut plan = PhotoPlan::default();
        plan.pan = 10.0;
        let mut notes = Vec::new();
        let out = run(&f, &plan, &aux, &mut notes).unwrap();
        // The red square's centre was at x≈20; after a pan right it sits further left.
        let cx = |fr: &Frame| {
            let (mut s, mut n) = (0.0f32, 0.0f32);
            for y in 0..fr.h {
                for x in 0..fr.w {
                    let p = fr.px[(y * fr.w + x) as usize];
                    if p[0] > 0.6 && p[1] < 0.3 {
                        s += x as f32;
                        n += 1.0;
                    }
                }
            }
            s / n.max(1.0)
        };
        assert!(cx(&out.frame) < cx(&f) - 2.0, "{} vs {}", cx(&out.frame), cx(&f));
        // A truck to the right at 3 m with the square at 3 m and the wall ~20 m: ground opens behind the square.
        let mut plan = PhotoPlan::default();
        plan.truck = 0.6;
        plan.distance = 3.0;
        plan.fill = "none".into();
        let mut notes = Vec::new();
        let out = run(&f, &plan, &aux, &mut notes).unwrap();
        assert!(out.invented > 0.01 && out.invented < 0.4, "invented {}", out.invented);
        // With propagate, the holes take the wall's grey, not the square's red.
        plan.fill = "propagate".into();
        let out2 = run(&f, &plan, &aux, &mut notes).unwrap();
        let holes = out.holes.unwrap();
        let (mut grey, mut n) = (0.0, 0.0);
        for i in 0..holes.v.len() {
            if holes.v[i] > 0.5 {
                let p = out2.frame.px[i];
                grey += ((p[0] - 0.5).abs() < 0.15 && (p[1] - 0.5).abs() < 0.15) as u32 as f32;
                n += 1.0;
            }
        }
        assert!(grey / n > 0.8, "the fill is the wall's grey: {}", grey / n);
        assert!(notes.iter().any(|s| s.starts_with("camera:")), "{notes:?}");
        // A truck that looks past the picture's edge: the frame is cropped to what the picture reached and brought
        // back to size (the note says so), no pixel past the edge is left to smear.
        let mut plan = PhotoPlan::default();
        plan.truck = 0.6;
        plan.distance = 3.0;
        let mut notes = Vec::new();
        let out3 = run(&f, &plan, &aux, &mut notes).unwrap();
        assert_eq!((out3.frame.w, out3.frame.h), (f.w, f.h));
        assert!(notes.iter().any(|s| s.contains("cropped")), "{notes:?}");
        assert!(out3.invented <= out.invented + 0.005, "the crop leaves no more holes: {} vs {}", out3.invented, out.invented);
        let mut plan = PhotoPlan::default();
        plan.truck = 0.6;
        plan.distance = 3.0;
        plan.fill = "none".into();
        let mut notes = Vec::new();
        run(&f, &plan, &aux, &mut notes).unwrap();
        assert!(!notes.iter().any(|s| s.contains("cropped")), "fill none shows the holes: {notes:?}");
    }

    #[test]
    fn a_longer_lens_with_the_subject_kept_dollies_back_and_keeps_the_square_s_size() {
        let (f, aux) = scene();
        let size = |fr: &Frame| fr.px.iter().filter(|p| p[0] > 0.6 && p[1] < 0.3).count() as f32;
        let mut plan = PhotoPlan::default();
        plan.focal = 85.0;
        plan.source_focal = 35.0;
        plan.distance = 3.0;
        plan.keep_subject = 1.0;
        // The subject's depth is the matte's.
        let mut m = Field::new(f.w, f.h, 0.0);
        for y in 14..34 {
            for x in 12..28 {
                m.v[(y * f.w + x) as usize] = 1.0;
            }
        }
        let aux = Aux { matte: Some(m), ..aux };
        let mut notes = Vec::new();
        let out = run(&f, &plan, &aux, &mut notes).unwrap();
        let ratio = size(&out.frame) / size(&f);
        assert!((0.75..1.3).contains(&ratio), "size ratio {ratio} — {notes:?}");
        assert!(notes[0].contains("dolly -4.29 m"), "{}", notes[0]);
        // Without the tie, 85 mm is a 2.4× crop: the square grows.
        plan.keep_subject = 0.0;
        let out = run(&f, &plan, &aux, &mut notes).unwrap();
        assert!(size(&out.frame) / size(&f) > 3.0);
        // No move: the frame passes through untouched and the metric depth is derived.
        let out = run(&f, &PhotoPlan::default(), &aux, &mut notes).unwrap();
        assert_eq!(out.frame, f);
        assert!(out.metric.is_some());
    }
}
