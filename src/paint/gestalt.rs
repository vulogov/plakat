//! THE GESTALT PLANNER (7.17.0 set 12; `MediumProfile::aqua_gestalt`): the water engine's score planned by
//! TOPOLOGY, not by a grid of optical debt.
//!
//! The engine's parser (`painter::aqua_engine_pass`) walks a jittered grid over the whole sheet on every session
//! and lays a touch wherever the sheet owes the picture a darkening — a dense, even, grid-shaped reading of the
//! photograph, the same finish everywhere whatever a passage is worth. The keys of sets 1–11 damped that reading
//! by the sight; this module REPLACES it under the key, and leaves the engine's grid to the bit at 0.
//!
//! What it does, in order (`plan`):
//!
//! 1. THE TRIAGE. The sight's saliency `S` bent by `abstraction` (`S^(1 + 3 × key)`: the detail falling from the
//!    focus to the edges the steeper the key) is cut into three ZONES — the FOCUS above 0.7 (every session, wet to
//!    dry, and the drawing after the washes), the MIDDLE GROUND between 0.3 and 0.7 (the wet sessions only, three at
//!    most: no small brush, no dry brush), the BACKGROUND at 0.3 and under (ONE session: the first wet flood, from
//!    the picture's low-pass base alone — its details owe nothing). A pixel's ladder of ceilings runs over the
//!    sessions its zone allows (the painter's debt), so the background is at its tone by the first flood.
//! 2. THE NEGATIVE SPACE. Off the focus, every pixel lighter than `negative` of the picture's own range is PAPER:
//!    no debt, no stroke enters it, no fluid is laid — the paper stays white because nothing touches it. The small
//!    lights of `aqua_sight_whites` join it the same way (cut from the debt, not masked).
//! 3. THE STROKES. On every session a seeded grid of the engine's own pitch supplies the SEEDS; each seed not yet
//!    CLAIMED by a stroke of the session (nor paper, nor out of its zone's sessions) grows a POLYLINE along the
//!    picture's structure — the structure tensor's isophote where it is coherent, the class's own direction where
//!    it is not — both ways, until the sheet's edge, the paper, another zone, another class or a turn of more than
//!    45°; the focus for four radii each way, the middle ground six, the background twelve. On the background of a
//!    WASH class (the sea, the sky) the strokes run in BANDS across the class's direction — abutting, two radii
//!    apart, the water uneven along a band by `aqua_sight_strokes` — and the solver joins them wet. Each stroke
//!    CLAIMS the pixels under its CORE (half the radius about the path; the brush's spread beyond it is the wet edge
//!    the neighbours share, the dab's profile summing to the film); a stroke that finds less than a
//!    fifth of its core unclaimed is not laid (the hand does not go over what it has covered; the footprint's
//!    spill over the zone's or the class's boundary is no claim — the neighbour's strokes claim their own); the pixels it
//!    claims are the MASS it carries (the engine's debt per area, exactly as a grid cell's — the claims of a session
//!    sum to the sheet at most, so a session lays its debt once). The brush lifts off at the ends (`aqua_taper`)
//!    and runs dry toward them (`aqua_sweep`), as the engine's strokes do.
//! 4. THE BUDGET. The first flood is never dropped; every stroke after is weighed by the ground it covers × its
//!    zone's bent saliency × the picture's contrast along it, and what does not fit `budget` is dropped, the
//!    heaviest kept — the finish taken from the finest session upward, never the masses: the sheet LAZY by the
//!    key, the air left in.
//!
//! The planner is pure: maps in, strokes out; nothing of it enters the score — every stroke is a touch record as
//! the engine's are, so the replay is exact without it.

/// A planned stroke: the session it belongs to, its path (one dab or a polyline), the brush's radius, the zone it
/// stands in (0 the background, 1 the middle ground, 2 the focus), the pixels it claims (its mass), its weight for
/// the budget, and the water's unevenness along a band (`wob`, a signed share).
#[derive(Debug, Clone)]
pub(crate) struct GestaltStroke {
    pub session: usize,
    pub path: Vec<[f32; 2]>,
    pub radius: f32,
    pub zone: u8,
    /// The painter class the seed sits in (0 without a class map): the debt under the stroke is read within it.
    pub class: u8,
    pub area: f32,
    pub weight: f32,
    pub wob: f32,
}

/// The plan: the strokes in laying order, the zone of every pixel, the negative space.
#[derive(Debug, Clone, Default)]
pub(crate) struct Gestalt {
    pub strokes: Vec<GestaltStroke>,
    pub zone: Vec<u8>,
    pub negative: Vec<bool>,
}

/// What the planner reads.
pub(crate) struct GestaltInput<'a> {
    pub w: usize,
    pub h: usize,
    /// The sight's saliency per pixel, [0, 1].
    pub sal: &'a [f32],
    /// The picture's luma at the engine's wash scale.
    pub luma: &'a [f32],
    /// The luma of the picture's low-pass base (the contrast's reference).
    pub low_luma: &'a [f32],
    /// The feathered focus (the engine's brush narrows on it).
    pub focus: &'a [f32],
    /// The class per pixel, when the sight has them.
    pub class: Option<&'a [u8]>,
    /// The technique as water per pixel (+1 a wash), when the sight has it.
    pub wash: Option<&'a [f32]>,
    /// The paper as the engine has it (the protected shapes, the paper gate).
    pub paper: &'a [bool],
    /// The small lights to keep as paper (`aqua_sight_whites`), when on.
    pub whites: Option<&'a [bool]>,
    pub sessions: usize,
    pub crisp_from: usize,
    /// The brush's radius per session.
    pub radii: &'a [f32],
    pub class_k: f32,
    pub abstraction: f32,
    pub negative: f32,
    pub budget: usize,
    pub strokes_k: f32,
    pub seed: u64,
}

/// The zones' thresholds on the bent saliency.
pub(crate) const FOCUS_ABOVE: f32 = 0.7;
pub(crate) const MIDDLE_ABOVE: f32 = 0.3;

/// The sessions a zone allows: the background one, the middle ground the wet sessions up to three, the focus all.
pub(crate) fn zone_sessions(zone: u8, sessions: usize, crisp_from: usize) -> usize {
    match zone {
        0 => 1,
        1 => crisp_from.clamp(1, 3).min(sessions.max(1)),
        _ => sessions.max(1),
    }
}

/// The saliency bent by the abstraction key: `S^(1 + 3k)`.
pub(crate) fn bend(s: f32, k: f32) -> f32 {
    s.clamp(0.0, 1.0).powf(1.0 + 3.0 * k.clamp(0.0, 1.0))
}

fn hash(seed: u64, a: u64, b: u64, c: u64) -> f32 {
    let mut z = seed ^ a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ c.wrapping_mul(0x1656_67B1_9E37_79F9);
    z ^= z >> 29;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 32;
    (z & 0xFF_FFFF) as f32 / 16_777_216.0
}

fn sobel(v: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut gx, mut gy) = (vec![0f32; w * h], vec![0f32; w * h]);
    if w < 3 || h < 3 {
        return (gx, gy);
    }
    let at = |x: usize, y: usize| v[y * w + x];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            gx[y * w + x] = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
            gy[y * w + x] = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
        }
    }
    (gx, gy)
}

fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let mut acc = 0f32;
        for x in 0..=r.min(w - 1) {
            acc += row[x];
        }
        for x in 0..w {
            let lo = x as isize - r as isize;
            let hi = x + r;
            if x > 0 {
                if hi < w {
                    acc += row[hi];
                }
                if lo > 0 {
                    acc -= row[(lo - 1) as usize];
                }
            }
            let n = (hi.min(w - 1) as isize - lo.max(0) + 1) as f32;
            tmp[y * w + x] = acc / n;
        }
    }
    let mut out = vec![0f32; w * h];
    for x in 0..w {
        let mut acc = 0f32;
        for y in 0..=r.min(h - 1) {
            acc += tmp[y * w + x];
        }
        for y in 0..h {
            let lo = y as isize - r as isize;
            let hi = y + r;
            if y > 0 {
                if hi < h {
                    acc += tmp[hi * w + x];
                }
                if lo > 0 {
                    acc -= tmp[(lo - 1) as usize * w + x];
                }
            }
            let n = (hi.min(h - 1) as isize - lo.max(0) + 1) as f32;
            out[y * w + x] = acc / n;
        }
    }
    out
}

/// The plan (see the module's note).
pub(crate) fn plan(inp: &GestaltInput) -> Gestalt {
    let (w, h) = (inp.w, inp.h);
    let n = w * h;
    if n == 0 || inp.sal.len() != n || inp.luma.len() != n || inp.radii.len() < inp.sessions {
        return Gestalt::default();
    }
    let long = w.max(h) as f32;
    // 1. THE TRIAGE.
    let s_eff: Vec<f32> = inp.sal.iter().map(|s| bend(*s, inp.abstraction)).collect();
    let zone: Vec<u8> = s_eff.iter().map(|s| if *s > FOCUS_ABOVE { 2 } else if *s > MIDDLE_ABOVE { 1 } else { 0 }).collect();
    // 2. THE NEGATIVE SPACE: off the focus, the picture's own range above the key's share; the small lights with it.
    let negative: Vec<bool> = {
        let mut sorted = inp.luma.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo = sorted[n * 2 / 100];
        let hi = sorted[(n * 98 / 100).min(n - 1)];
        let (lo, hi) = if hi - lo < 0.05 { (0.0, 1.0) } else { (lo, hi) };
        let k = inp.negative.clamp(0.0, 1.0);
        let thr = lo + k * (hi - lo);
        (0..n).map(|i| zone[i] < 2 && ((k > 0.0 && inp.luma[i] >= thr) || inp.whites.is_some_and(|m| m.get(i).copied().unwrap_or(false)))).collect()
    };
    let blocked = |i: usize| inp.paper.get(i).copied().unwrap_or(false) || negative[i];
    // The structure: the tensor over the engine's window, its isophote direction and its coherence; each class's
    // own direction (the tensor summed over the class).
    let sigma = ((long / 48.0).round() as usize).clamp(4, 64);
    let (sx, sy) = sobel(inp.luma, w, h);
    let (mut jxx, mut jyy, mut jxy) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    let (mut cxx, mut cyy, mut cxy) = ([0f32; 256], [0f32; 256], [0f32; 256]);
    for i in 0..n {
        jxx[i] = sx[i] * sx[i];
        jyy[i] = sy[i] * sy[i];
        jxy[i] = sx[i] * sy[i];
        if let Some(cm) = inp.class {
            let c = cm[i] as usize;
            cxx[c] += jxx[i];
            cyy[c] += jyy[i];
            cxy[c] += jxy[i];
        }
    }
    let jxx = box_blur(&jxx, w, h, sigma);
    let jyy = box_blur(&jyy, w, h, sigma);
    let jxy = box_blur(&jxy, w, h, sigma);
    let axis = |a: f32, b: f32, d: f32| -> (f32, f32) {
        // The isophote: across the tensor's major axis.
        if a + d < 1e-9 {
            return (1.0, 0.0);
        }
        let l1 = 0.5 * ((a + d) + ((a - d).powi(2) + 4.0 * b * b).sqrt());
        let (ex, ey) = if b.abs() > 1e-9 { (l1 - d, b) } else if a >= d { (1.0, 0.0) } else { (0.0, 1.0) };
        let m = (ex * ex + ey * ey).sqrt().max(1e-9);
        // (The sign canonical — rightward, else downward — so a band's offset does not depend on which way the
        // tensor's eigenvector happened to point.)
        let (x, y) = (-ey / m, ex / m);
        if x < 0.0 || (x == 0.0 && y < 0.0) { (-x, -y) } else { (x, y) }
    };
    let class_dir: Vec<(f32, f32)> = (0..256).map(|c| axis(cxx[c], cxy[c], cyy[c])).collect();
    let coh = |i: usize| -> f32 {
        let disc = ((jxx[i] - jyy[i]).powi(2) + 4.0 * jxy[i] * jxy[i]).sqrt();
        (disc / (jxx[i] + jyy[i] + 1e-6)).clamp(0.0, 1.0) * ((jxx[i] + jyy[i]) / 1e-4).clamp(0.0, 1.0)
    };
    let at = |x: f32, y: f32| -> usize { (y.max(0.0) as usize).min(h - 1) * w + (x.max(0.0) as usize).min(w - 1) };
    let inside = |x: f32, y: f32| x >= 0.0 && y >= 0.0 && x < w as f32 && y < h as f32;
    let class_of = |i: usize| inp.class.map_or(0u8, |cm| cm[i]);
    let dir_of = |i: usize| -> (f32, f32) { if coh(i) >= 0.3 { axis(jxx[i], jxy[i], jyy[i]) } else { class_dir[class_of(i) as usize] } };
    let contrast = |i: usize| (inp.luma[i].max(1e-3) / inp.low_luma.get(i).copied().unwrap_or(inp.luma[i]).max(1e-3)).ln().abs();
    // 3. THE STROKES, session by session.
    let mut strokes: Vec<GestaltStroke> = Vec::new();
    let mut claimed = vec![false; n];
    // (Each stroke's own footprint counted once per pixel, whatever the dabs' overlap along the path.)
    let mut seen = vec![0u32; n];
    let mut sid = 0u32;
    let dbg = std::env::var_os("PLAKAT_GESTALT_DEBUG").is_some();
    for s in 0..inp.sessions {
        let mut cnt = [0usize; 6]; // seeds, blocked, claimed, closed session, thin, planned
        let r = inp.radii[s].max(1.0);
        claimed.iter_mut().for_each(|c| *c = false);
        let sp = 1.25 * r * 0.65;
        let (nx, ny) = ((w as f32 / sp).ceil() as usize, (h as f32 / sp).ceil() as usize);
        let stride = ((r / 6.0).round() as usize).max(1);
        for j in 0..ny {
            for i in 0..nx {
                let jx = hash(inp.seed, s as u64, i as u64, j as u64 * 2 + 1) - 0.5;
                let jy = hash(inp.seed, s as u64, i as u64, j as u64 * 2 + 2) - 0.5;
                let (cx, cy) = ((i as f32 + 0.5 + 0.6 * jx) * sp, (j as f32 + 0.5 + 0.6 * jy) * sp);
                if !inside(cx, cy) {
                    continue;
                }
                let i0 = at(cx, cy);
                let z = zone[i0];
                cnt[0] += 1;
                if blocked(i0) { cnt[1] += 1; continue; }
                if claimed[i0] { cnt[2] += 1; continue; }
                if s >= zone_sessions(z, inp.sessions, inp.crisp_from) { cnt[3] += 1; continue; }
                let wc = inp.wash.map_or(0.0, |m| m[i0].clamp(-1.0, 1.0));
                let m = inp.focus.get(i0).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                // (The brush on the focus narrower by its sight, as the engine's; not widened by the class — the
                // background has its one session, and a brush of a fifth of the sheet spills a fifth of the sheet.)
                let r_loc = r * (1.0 - 0.35 * m);
                let c0 = class_of(i0);
                // The background of a wash (or of a sheet without classes): bands across the class's direction.
                let band = z == 0 && (inp.wash.is_none() || wc >= 0.5);
                let (sdx, sdy) = if band { class_dir[c0 as usize] } else { dir_of(i0) };
                let mut wob = 0.0f32;
                if band {
                    // (No phase: the band runs through its seed, and the seeds under a laid band are thin — the
                    // claim alone spaces the bands, abutting.)
                    let pitch = 2.0 * r_loc;
                    let across = cx * -sdy + cy * sdx;
                    let band_id = (across / pitch).floor() as i64;
                    let along_id = ((cx * sdx + cy * sdy) / (2.0 * r_loc).max(1.0)).floor() as i64;
                    wob = inp.strokes_k.clamp(0.0, 1.0) * 0.3 * (2.0 * hash(inp.seed, s as u64 + 211, band_id as u64, along_id as u64) - 1.0);
                }
                let max_len = r_loc * match z { 2 => 4.0, 1 => 6.0, _ => if band { 12.0 } else { 8.0 } };
                let step = (0.6 * r_loc).max(1.0);
                let (mut back, mut fwd): (Vec<[f32; 2]>, Vec<[f32; 2]>) = (Vec::new(), Vec::new());
                for (sg, out) in [(-1.0f32, &mut back), (1.0, &mut fwd)] {
                    let (mut x, mut y, mut dx, mut dy, mut len) = (cx, cy, sdx * sg, sdy * sg, 0.0f32);
                    while len + step <= max_len {
                        let (x2, y2) = (x + dx * step, y + dy * step);
                        if !inside(x2, y2) {
                            break;
                        }
                        let i2 = at(x2, y2);
                        if blocked(i2) || zone[i2] != z || class_of(i2) != c0 {
                            break;
                        }
                        if !band {
                            let (vx, vy) = dir_of(i2);
                            let (vx, vy) = if vx * dx + vy * dy < 0.0 { (-vx, -vy) } else { (vx, vy) };
                            if vx * dx + vy * dy < 0.7 {
                                break;
                            }
                            dx = vx;
                            dy = vy;
                        }
                        out.push([x2, y2]);
                        x = x2;
                        y = y2;
                        len += step;
                    }
                }
                back.reverse();
                let mut path = back;
                path.push([cx, cy]);
                path.extend(fwd);
                if path.len() == 1 {
                    // The plain touch where the structure allows no stroke: three dabs along the direction at
                    // 0.6 of the radius, as the engine's grid lays it — a short mark, not a disc.
                    let st = 0.6 * r_loc;
                    path = vec![[cx - sdx * st, cy - sdy * st], [cx, cy], [cx + sdx * st, cy + sdy * st]];
                }
                // The claim: the pixels under the footprint not yet claimed this session are the stroke's mass; a
                // stroke over claimed ground is not laid.
                // The stroke's CORE — half the radius about the path — is what it claims and what it carries; the
                // brush's spread beyond the core is the wet edge the neighbouring strokes share (the dab's profile
                // falls to nothing at the radius, so a sheet tiled by whole footprints is blotches with the paper
                // between; tiled by cores, the spreads overlap two- to threefold and sum to the film).
                let r_core = (0.5 * r_loc).max(0.5);
                let (mut fresh, mut foot) = (0usize, 0usize);
                let mut contrast_sum = 0.0f32;
                sid += 1;
                for d in &path {
                    let x0 = (d[0] - r_core).floor().max(0.0) as usize;
                    let x1 = ((d[0] + r_core).ceil() as usize).min(w - 1);
                    let y0 = (d[1] - r_core).floor().max(0.0) as usize;
                    let y1 = ((d[1] + r_core).ceil() as usize).min(h - 1);
                    // (The samples on ONE lattice over the sheet, a stride apart from its corner — not from each
                    // dab's own corner, or two strokes over one ground would sample different pixels and each
                    // find the other's ground fresh.)
                    let mut yy = y0.div_ceil(stride) * stride;
                    while yy <= y1 {
                        let mut xx = x0.div_ceil(stride) * stride;
                        while xx <= x1 {
                            let (ddx, ddy) = (xx as f32 + 0.5 - d[0], yy as f32 + 0.5 - d[1]);
                            if ddx * ddx + ddy * ddy < r_core * r_core {
                                let q = yy * w + xx;
                                // (The ground of the stroke's own zone and class alone: what the footprint spills
                                // over the boundary is the brush's spill, not ground covered — the neighbour's
                                // strokes claim it themselves.)
                                if zone[q] != z || class_of(q) != c0 || seen[q] == sid {
                                    xx += stride;
                                    continue;
                                }
                                seen[q] = sid;
                                foot += 1;
                                if !claimed[q] && !blocked(q) {
                                    claimed[q] = true;
                                    fresh += 1;
                                    contrast_sum += contrast(q);
                                }
                            }
                            xx += stride;
                        }
                        yy += stride;
                    }
                }
                if foot == 0 || (fresh as f32) < 0.2 * foot as f32 {
                    cnt[4] += 1;
                    continue;
                }
                cnt[5] += 1;
                let area = fresh as f32 * (stride * stride) as f32;
                let mean_contrast = contrast_sum / fresh as f32;
                // The stroke's worth: the ground it covers × the zone's bent saliency × the picture's contrast along
                // it — a wide stroke of a wet session worth more than a fine one of a dry one, so a budget takes the
                // finish away from the finest session upward and never the masses. (The first flood outweighs every
                // stroke after it, whatever their worth: it is never dropped.)
                let weight = if s == 0 { 1.0e12 + area } else { area * s_eff[i0] * (0.05 + mean_contrast) };
                strokes.push(GestaltStroke { session: s, path, radius: r_loc, zone: z, class: c0, area, weight, wob });
            }
        }
        if dbg {
            eprintln!("gestalt session {s}: seeds {} blocked {} claimed {} closed {} thin {} planned {}", cnt[0], cnt[1], cnt[2], cnt[3], cnt[4], cnt[5]);
        }
    }
    // 4. THE BUDGET: the heaviest kept, the laying order restored.
    if inp.budget > 0 && strokes.len() > inp.budget {
        let mut order: Vec<usize> = (0..strokes.len()).collect();
        order.sort_by(|a, b| strokes[*b].weight.partial_cmp(&strokes[*a].weight).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));
        order.truncate(inp.budget);
        order.sort_unstable();
        strokes = order.into_iter().map(|i| strokes[i].clone()).collect();
    }
    Gestalt { strokes, zone, negative }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input<'a>(w: usize, h: usize, sal: &'a [f32], luma: &'a [f32], low: &'a [f32], focus: &'a [f32], paper: &'a [bool], radii: &'a [f32], class: Option<&'a [u8]>, wash: Option<&'a [f32]>) -> GestaltInput<'a> {
        GestaltInput { w, h, sal, luma, low_luma: low, focus, class, wash, paper, whites: None, sessions: radii.len(), crisp_from: radii.len().saturating_sub(2).max(1), radii, class_k: 1.0, abstraction: 0.0, negative: 0.0, budget: 0, strokes_k: 0.5, seed: 7 }
    }

    #[test]
    fn the_triage_cuts_the_zones_and_the_zones_their_sessions() {
        // The focus above 0.7, the middle ground to 0.3, the background under; the abstraction bends the saliency
        // down (0.6 at 0 a middle ground, at 1 — 0.6⁴ ≈ 0.13 — a background); the background one session, the
        // middle ground the wet ones up to three, the focus all.
        assert_eq!(zone_sessions(0, 6, 4), 1);
        assert_eq!(zone_sessions(1, 6, 4), 3);
        assert_eq!(zone_sessions(1, 6, 2), 2);
        assert_eq!(zone_sessions(2, 6, 4), 6);
        assert!(bend(0.6, 0.0) > MIDDLE_ABOVE && bend(0.6, 1.0) < MIDDLE_ABOVE, "{} {}", bend(0.6, 0.0), bend(0.6, 1.0));
        assert!(bend(1.0, 1.0) > FOCUS_ABOVE);
    }

    #[test]
    fn the_plan_floods_the_background_once_and_works_the_focus_on_every_session() {
        // A 128 × 96 sheet: the sight 1 in a box, 0 elsewhere; a light sky over a dark ground with a vertical edge
        // in the box. The first session's strokes cover the sheet; the later sessions' lie in the box alone (the
        // background allows one session); with the negative key the light sky off the focus takes no stroke at
        // all and is marked negative; with a budget the strokes are at most the budget and the first session's all
        // kept; the background bands of a wash run along the sheet (the class's direction).
        let (w, h) = (128usize, 96usize);
        let n = w * h;
        let in_box = |x: usize, y: usize| (40..88).contains(&x) && (24..72).contains(&y);
        let sal: Vec<f32> = (0..n).map(|i| if in_box(i % w, i / w) { 1.0 } else { 0.0 }).collect();
        let luma: Vec<f32> = (0..n).map(|i| { let (x, y) = (i % w, i / w); if y < h / 2 { 0.85 } else if in_box(x, y) && x >= 64 { 0.2 } else { 0.4 } }).collect();
        let low = vec![0.5f32; n];
        let focus = sal.clone();
        let paper = vec![false; n];
        let radii = [12.8f32, 6.4, 3.2, 1.6];
        let class: Vec<u8> = (0..n).map(|i| if i / w < h / 2 { 0 } else { 2 }).collect();
        let wash: Vec<f32> = (0..n).map(|i| if i / w < h / 2 { 1.0 } else { 0.3 }).collect();
        let base = input(w, h, &sal, &luma, &low, &focus, &paper, &radii, Some(&class), Some(&wash));
        let g = plan(&base);
        assert!(g.zone.iter().enumerate().all(|(i, z)| (*z == 2) == in_box(i % w, i / w)), "the box the focus, the rest the background");
        let first: Vec<&GestaltStroke> = g.strokes.iter().filter(|s| s.session == 0).collect();
        let later: Vec<&GestaltStroke> = g.strokes.iter().filter(|s| s.session > 0).collect();
        assert!(!first.is_empty() && !later.is_empty(), "{} / {}", first.len(), later.len());
        assert!(first.iter().any(|s| s.zone == 0) && first.iter().any(|s| s.zone == 2), "the first session floods the background and the focus");
        assert!(later.iter().all(|s| s.zone == 2 && s.path.iter().all(|d| in_box(d[0] as usize, d[1] as usize))), "the later sessions on the focus alone");
        assert!(g.strokes.iter().filter(|s| s.session == 3).count() > 0, "the finest session reaches the focus");
        let bands: Vec<&GestaltStroke> = first.iter().copied().filter(|s| s.zone == 0 && (s.path[0][1] as usize) < h / 2).collect();
        assert!(!bands.is_empty() && bands.iter().all(|s| s.path.iter().all(|d| (d[1] - s.path[0][1]).abs() < 1e-3)), "the sky's bands run along the sheet");
        assert!(bands.iter().any(|s| s.path.len() >= 4), "…as long strokes");
        assert!(g.strokes.iter().all(|s| s.area > 0.0 && s.weight > 0.0));
        for sess in 0..radii.len() {
            let sum: f32 = g.strokes.iter().filter(|s| s.session == sess).map(|s| s.area).sum();
            assert!(sum <= n as f32 * 1.05, "session {sess}: the claims {sum} over the sheet {n} — the strokes lay the debt once");
        }
        let mut neg = input(w, h, &sal, &luma, &low, &focus, &paper, &radii, Some(&class), Some(&wash));
        neg.negative = 0.8;
        let gn = plan(&neg);
        assert!(gn.negative.iter().enumerate().all(|(i, v)| *v == (i / w < h / 2 && !in_box(i % w, i / w))), "the light sky off the focus the negative space, the dark ground and the focus not");
        assert!(gn.strokes.iter().all(|s| s.path.iter().all(|d| (d[1] as usize) >= h / 2 || in_box(d[0] as usize, d[1] as usize))), "no stroke enters the negative space");
        let mut lim = input(w, h, &sal, &luma, &low, &focus, &paper, &radii, Some(&class), Some(&wash));
        lim.budget = first.len() + 3;
        let gl = plan(&lim);
        assert_eq!(gl.strokes.len(), first.len() + 3, "the budget kept");
        assert_eq!(gl.strokes.iter().filter(|s| s.session == 0).count(), first.len(), "the first flood never dropped");
        assert!(gl.strokes.windows(2).all(|p| p[0].session <= p[1].session), "the laying order kept");
    }
}
