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
//!    focus to the edges the steeper the key) is cut into three ZONES — the FOCUS above 0.4 (the flood, the GLAZES,
//!    the dry brushes and the drawing after the washes), the MIDDLE GROUND between 0.15 and 0.4 (the wide brushes
//!    only, three sessions at most: no small brush, no dry brush), the BACKGROUND at 0.15 and under (the first wet
//!    flood, from the picture's low-pass base at the brush's scale, and one glaze at most). Within its zone every
//!    pixel's LADDER — the sessions that touch it — is CONTINUOUS in the picture's local contrast (set 13): the
//!    zone's floor of sessions everywhere (the background's flood, the middle ground's flood, the focus's flood,
//!    glazes and first dry brush), the zone's further sessions where the contrast at the brush's window rises
//!    toward the sheet's ninth decile — a flat hull is glazed and left, its rigging is drawn. The painter's debt
//!    runs each pixel's ceilings over its own ladder.
//!    THE GLAZES (set 13): the first session is the wet flood on the wetted sheet; every session after it is laid
//!    WET-ON-DRY — the sheet dried as the engine dries it, no water laid before the session, the brush wet as the
//!    session's — a half-transparent layer over the flood that holds its shape, the way a painter closes what a
//!    flood left light: the engine's wet laying pays about half of a charge in stops, and that is the watercolour's
//!    own lightness, not a debt to be forced into the puddle. (Set 12 had rehearsed the wet session on a copy and
//!    raised the charge by the measured shortfall: oversaturation and mud.)
//!    THE SHAPES (set 14): off the focus the sheet is cut into SHAPES — the connected regions of one class, the
//!    sky, the hill, the sea, a tree — and the focus is a shape of its own. The painter wets the sheet WITHIN a
//!    shape and never across its edge, so the hill is wet inside and dry against the sky: the flood's edge there
//!    is the wash's own hard edge, the horizon between two wet shapes a line, the figure's edge kept. A shape the
//!    brush does not fit across (its half-width under the radius) is laid as ONE stroke a session along its own
//!    axis, the brush its half-width — the crown of a tree a stroke the crown's length, never discs on a grid. The
//!    background's ladder is TWO washes, the flood and a glaze over the whole shape (the flood pays half a charge;
//!    two washes lay the tone, as they are painted), a third where the contrast rises. A wash is laid as a painter
//!    lays it: the EDGE STROKE first, the brush run along the shape's edge (half the session's brush, a quarter
//!    radius inside, its spill over the edge one even line), then the FILL in straight parallel bands kept a full
//!    radius inside the edge (the glazes' bands follow the form) — the wash's edge is a line, not the round ends
//!    of its bands.
//! 2. THE NEGATIVE SPACE. Off the focus, every pixel lighter than `negative` of the picture's own range is PAPER:
//!    no debt, no stroke enters it, no fluid is laid — the paper stays white because nothing touches it. The small
//!    lights of `aqua_sight_whites` join it the same way (cut from the debt, not masked).
//! 3. THE STROKES. On every session a seeded grid of the engine's own pitch supplies the SEEDS; each seed not yet
//!    CLAIMED by a stroke of the session (nor paper, nor out of its zone's sessions) grows a POLYLINE along the
//!    picture's structure — the structure tensor's isophote where it is coherent, the class's own direction where
//!    it is not — both ways, until the sheet's edge, the paper, another zone, another class or a turn of more than
//!    45°; the focus for four radii each way, the middle ground six, the background twelve. On the background of a
//!    WASH class (the sea, the sky) the strokes run in BANDS — long, abutting, each following the FORM (the
//!    isophote where the picture is coherent, the class's own direction where it is not, a turn of 60° allowed),
//!    the water uneven along a band by `aqua_sight_strokes` (the sweep and the unevenness leave the paper's
//!    slivers between them) — and the solver joins them wet. NO DAUB (set 13): a wet session's stroke is a
//!    movement, never a disc — a stroke the structure cuts short of two radii is traced again straight along its
//!    class's direction, and if the boundary cuts that short too, the brush NARROWS to half the run (a tree, a boat
//!    smaller than the session's brush takes the brush that fits it, down to a quarter of the session's) — under
//!    that nothing is laid there (the paper breathes); the dry brushes keep their short tick. Each stroke
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

/// The plan: the strokes in laying order, the zone of every pixel, its ladder (the sessions that touch it), the
/// negative space.
#[derive(Debug, Clone, Default)]
pub(crate) struct Gestalt {
    pub strokes: Vec<GestaltStroke>,
    pub zone: Vec<u8>,
    /// The sessions that touch the pixel: its zone's floor, and the zone's further sessions by the local contrast.
    pub sessions: Vec<u8>,
    /// The SHAPE every pixel belongs to (set 14): 0 none (paper, the negative space), 1 the focus, from 2 on the
    /// connected regions of one class off the focus — the sky, the hill, the sea, a tree. The sheet is wetted
    /// within a shape, never across its edge.
    pub shape: Vec<u32>,
    /// The pixels of every shape, by its id.
    pub shape_size: Vec<u32>,
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
    pub abstraction: f32,
    pub negative: f32,
    pub budget: usize,
    pub strokes_k: f32,
    pub seed: u64,
}

/// The zones' thresholds on the bent saliency.
pub(crate) const FOCUS_ABOVE: f32 = 0.4;
pub(crate) const MIDDLE_ABOVE: f32 = 0.15;

/// A pixel's LADDER: the sessions that touch it, continuous in its local contrast `c` (0 flat, 1 the sheet's ninth
/// decile and over). The background: the flood and ONE GLAZE over the whole shape (set 14: two washes lay a tone,
/// the flood paying half), a second glaze where the contrast is over half. The middle ground: the two washes, and
/// a third wide session by the contrast. The focus: the flood, the glazes and the first dry brush everywhere, the
/// further dry brushes by the contrast.
pub(crate) fn ladder(zone: u8, c: f32, sessions: usize, crisp_from: usize) -> usize {
    let sessions = sessions.max(1);
    let c = c.clamp(0.0, 1.0);
    let n = match zone {
        0 => (2 + c.round() as usize).min(crisp_from.max(1)),
        1 => {
            let top = crisp_from.clamp(1, 3);
            let floor = top.min(2);
            floor + ((top - floor) as f32 * c).round() as usize
        }
        _ => {
            let floor = (crisp_from + 1).min(sessions);
            floor + ((sessions - floor) as f32 * c).round() as usize
        }
    };
    n.clamp(1, sessions)
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

/// A stroke's CLAIM: the pixels under its core on the session's lattice (a stride apart from the sheet's corner —
/// not from each dab's own corner, or two strokes over one ground would sample different pixels and each find the
/// other's ground fresh), counted once per stroke whatever the dabs' overlap, within the stroke's OWN ground alone
/// (what the footprint spills over the zone's or the class's boundary is the brush's spill, not ground covered).
/// Returns (fresh, foot, the contrast summed over the fresh).
#[allow(clippy::too_many_arguments)]
fn claim_core(path: &[[f32; 2]], r_core: f32, stride: usize, w: usize, h: usize, own: &dyn Fn(usize) -> bool, open: &dyn Fn(usize) -> bool, contrast: &dyn Fn(usize) -> f32, claimed: &mut [bool], seen: &mut [u32], sid: u32) -> (usize, usize, f32) {
    let (mut fresh, mut foot) = (0usize, 0usize);
    let mut contrast_sum = 0.0f32;
    for d in path {
        let x0 = (d[0] - r_core).floor().max(0.0) as usize;
        let x1 = ((d[0] + r_core).ceil() as usize).min(w - 1);
        let y0 = (d[1] - r_core).floor().max(0.0) as usize;
        let y1 = ((d[1] + r_core).ceil() as usize).min(h - 1);
        let mut yy = y0.div_ceil(stride) * stride;
        while yy <= y1 {
            let mut xx = x0.div_ceil(stride) * stride;
            while xx <= x1 {
                let (ddx, ddy) = (xx as f32 + 0.5 - d[0], yy as f32 + 0.5 - d[1]);
                if ddx * ddx + ddy * ddy < r_core * r_core {
                    let q = yy * w + xx;
                    if !own(q) || seen[q] == sid {
                        xx += stride;
                        continue;
                    }
                    seen[q] = sid;
                    foot += 1;
                    if !claimed[q] && open(q) {
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
    (fresh, foot, contrast_sum)
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
    // THE SHAPES (set 14): off the focus, the connected regions of one class (four-connected, the paper and the
    // negative space cutting them), the focus one shape of its own. Each shape's moments — its centroid, its axis
    // and its half-widths — are what a small shape is laid by.
    let class_at = |i: usize| inp.class.map_or(0u8, |cm| cm[i]);
    let mut shape = vec![0u32; n];
    let mut moments: Vec<[f64; 6]> = vec![[0.0; 6]; 2]; // n, Σx, Σy, Σxx, Σyy, Σxy
    {
        let mut stack: Vec<usize> = Vec::new();
        for i0 in 0..n {
            if shape[i0] != 0 || blocked(i0) {
                continue;
            }
            let id = if zone[i0] == 2 { 1 } else { moments.len() as u32 };
            if id > 1 {
                moments.push([0.0; 6]);
            }
            let c = class_at(i0);
            shape[i0] = id;
            stack.push(i0);
            while let Some(i) = stack.pop() {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                let m = &mut moments[id as usize];
                m[0] += 1.0;
                m[1] += x;
                m[2] += y;
                m[3] += x * x;
                m[4] += y * y;
                m[5] += x * y;
                let (xi, yi) = (i % w, i / w);
                let mut push = |j: usize| {
                    if shape[j] == 0 && !blocked(j) && (zone[j] == 2) == (id == 1) && (id == 1 || class_at(j) == c) {
                        shape[j] = id;
                        stack.push(j);
                    }
                };
                if xi > 0 { push(i - 1); }
                if xi + 1 < w { push(i + 1); }
                if yi > 0 { push(i - w); }
                if yi + 1 < h { push(i + w); }
            }
        }
    }
    // The distance of every pixel to its shape's edge (a chamfer of 3-4, in pixels; the sheet's edge is no edge):
    // the brush is kept INSIDE its shape by half its radius, so the wash's edge is the brush's edge, not its spill.
    let dist: Vec<f32> = {
        let inf = i32::MAX / 4;
        let mut d = vec![inf; n];
        for i in 0..n {
            let (x, y) = (i % w, i / w);
            if (x + 1 < w && shape[i + 1] != shape[i]) || (x > 0 && shape[i - 1] != shape[i]) || (y + 1 < h && shape[i + w] != shape[i]) || (y > 0 && shape[i - w] != shape[i]) {
                d[i] = 0;
            }
        }
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let mut v = d[i];
                if x > 0 { v = v.min(d[i - 1] + 3); }
                if y > 0 {
                    v = v.min(d[i - w] + 3);
                    if x > 0 { v = v.min(d[i - w - 1] + 4); }
                    if x + 1 < w { v = v.min(d[i - w + 1] + 4); }
                }
                d[i] = v;
            }
        }
        for y in (0..h).rev() {
            for x in (0..w).rev() {
                let i = y * w + x;
                let mut v = d[i];
                if x + 1 < w { v = v.min(d[i + 1] + 3); }
                if y + 1 < h {
                    v = v.min(d[i + w] + 3);
                    if x + 1 < w { v = v.min(d[i + w + 1] + 4); }
                    if x > 0 { v = v.min(d[i + w - 1] + 4); }
                }
                d[i] = v;
            }
        }
        d.into_iter().map(|v| v as f32 / 3.0).collect()
    };
    // (Each shape's axis: the direction of its major moment, its half-widths twice the standard deviations.)
    let shape_axis: Vec<(f32, f32, f32, f32, f32, f32)> = moments.iter().map(|m| {
        if m[0] < 1.0 {
            return (0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        }
        let (cx, cy) = (m[1] / m[0], m[2] / m[0]);
        let (vxx, vyy, vxy) = (m[3] / m[0] - cx * cx, m[4] / m[0] - cy * cy, m[5] / m[0] - cx * cy);
        let l1 = 0.5 * ((vxx + vyy) + ((vxx - vyy).powi(2) + 4.0 * vxy * vxy).sqrt());
        let l2 = (vxx + vyy - l1).max(0.0);
        let (ex, ey) = if vxy.abs() > 1e-9 { (l1 - vyy, vxy) } else if vxx >= vyy { (1.0, 0.0) } else { (0.0, 1.0) };
        let mm = (ex * ex + ey * ey).sqrt().max(1e-9);
        (cx as f32, cy as f32, (ex / mm) as f32, (ey / mm) as f32, 2.0 * l1.max(0.0).sqrt() as f32, 2.0 * l2.sqrt() as f32)
    }).collect();
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
    // THE LADDER: the local contrast at the tensor's window, against the sheet's ninth decile (the paper and the
    // negative space aside), and every pixel's sessions by its zone and that contrast.
    let sessions_of: Vec<u8> = {
        let raw: Vec<f32> = (0..n).map(contrast).collect();
        let cb = box_blur(&raw, w, h, sigma);
        let mut v: Vec<f32> = (0..n).filter(|&i| !blocked(i)).map(|i| cb[i]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p90 = if v.is_empty() { 1.0 } else { v[(v.len() - 1) * 9 / 10].max(1e-3) };
        (0..n).map(|i| ladder(zone[i], cb[i] / p90, inp.sessions, inp.crisp_from) as u8).collect()
    };
    // 3. THE STROKES, session by session.
    let mut strokes: Vec<GestaltStroke> = Vec::new();
    let mut claimed = vec![false; n];
    // (Each stroke's own footprint counted once per pixel, whatever the dabs' overlap along the path.)
    let mut seen = vec![0u32; n];
    let mut sid = 0u32;
    let dbg = std::env::var_os("PLAKAT_GESTALT_DEBUG").is_some();
    for s in 0..inp.sessions {
        let mut cnt = [0usize; 7]; // seeds, blocked, claimed, closed session, thin, planned, short (no daub)
        let wet = s < inp.crisp_from;
        let r = inp.radii[s].max(1.0);
        claimed.iter_mut().for_each(|c| *c = false);
        let mut shape_done = vec![false; moments.len()];
        let sp = 1.25 * r * 0.65;
        let (nx, ny) = ((w as f32 / sp).ceil() as usize, (h as f32 / sp).ceil() as usize);
        let stride = ((r / 6.0).round() as usize).max(1);
        // THE EDGE STROKES (set 14): a wash is laid as a painter lays it — the brush run ALONG the shape's edge
        // first (half the session's brush, a quarter of the session's radius inside the edge, so its spill over
        // the edge is a quarter radius, one even line), then the fill, which stays a full radius inside. The edge's
        // lattice points (a band of the distance map) chained nearest to nearest within their shape; a shape
        // narrower than the brush takes none (its fill is the small shape's own stroke).
        if wet {
            let r_e = 0.5 * r;
            let d0 = 0.5 * r_e;
            let tol = 0.15 * r;
            let mut pts: Vec<(u32, usize)> = Vec::new();
            let mut yy = 0;
            while yy < h {
                let mut xx = 0;
                while xx < w {
                    let q = yy * w + xx;
                    let sh = shape[q];
                    if sh >= 1 && !blocked(q) && (dist[q] - d0).abs() <= tol && sessions_of[q] as usize > s && shape_axis[sh as usize].5 >= r {
                        pts.push((sh, q));
                    }
                    xx += stride;
                }
                yy += stride;
            }
            pts.sort_unstable();
            let mut used = vec![false; pts.len()];
            let link2 = (2.5 * stride as f32).powi(2);
            let mut lo = 0;
            while lo < pts.len() {
                let sh = pts[lo].0;
                let hi = lo + pts[lo..].iter().take_while(|p| p.0 == sh).count();
                for start in lo..hi {
                    if used[start] {
                        continue;
                    }
                    used[start] = true;
                    let mut chain: Vec<usize> = vec![start];
                    for _end in 0..2 {
                        loop {
                            let last = *chain.last().unwrap();
                            let (lx, ly) = ((pts[last].1 % w) as f32, (pts[last].1 / w) as f32);
                            let mut best = None;
                            let mut bd = link2;
                            for j in lo..hi {
                                if used[j] {
                                    continue;
                                }
                                let (x, y) = ((pts[j].1 % w) as f32, (pts[j].1 / w) as f32);
                                let dd = (x - lx).powi(2) + (y - ly).powi(2);
                                if dd < bd {
                                    bd = dd;
                                    best = Some(j);
                                }
                            }
                            match best {
                                Some(j) => {
                                    used[j] = true;
                                    chain.push(j);
                                }
                                None => break,
                            }
                        }
                        chain.reverse();
                    }
                    if chain.len() < 3 {
                        continue;
                    }
                    let path: Vec<[f32; 2]> = chain.iter().map(|&j| [(pts[j].1 % w) as f32 + 0.5, (pts[j].1 / w) as f32 + 0.5]).collect();
                    let i0 = pts[chain[0]].1;
                    let (z, c0) = (zone[i0], class_of(i0));
                    sid += 1;
                    let own = |q: usize| zone[q] == z && class_of(q) == c0 && sessions_of[q] as usize > s;
                    let (fresh, foot, contrast_sum) = claim_core(&path, (0.5 * r_e).max(0.5), stride, w, h, &own, &|q| !blocked(q), &contrast, &mut claimed, &mut seen, sid);
                    if foot == 0 || (fresh as f32) < 0.2 * foot as f32 {
                        cnt[4] += 1;
                        continue;
                    }
                    cnt[5] += 1;
                    let area = fresh as f32 * (stride * stride) as f32;
                    let mean_contrast = contrast_sum / fresh as f32;
                    let weight = if s == 0 { 1.0e12 + area } else { area * (0.05 + s_eff[i0]) * (0.05 + mean_contrast) };
                    strokes.push(GestaltStroke { session: s, path, radius: r_e, zone: z, class: c0, area, weight, wob: 0.0 });
                }
                lo = hi;
            }
        }
        // The seeds: every small shape seeds itself first (its anchor the centroid, or the nearest point of its
        // axis within it — the grid may miss a tree), then the jittered grid of the engine's pitch.
        let mut seeds: Vec<(f32, f32)> = Vec::new();
        for sh in 2..moments.len() {
            let (ax, ay, dx, dy, half, wid) = shape_axis[sh];
            if wid >= r {
                continue;
            }
            let st = (0.3 * wid).max(1.0);
            let mut d = 0.0f32;
            let mut anchor = None;
            while d <= half.max(st) && anchor.is_none() {
                for sg in [1.0f32, -1.0] {
                    let (x, y) = (ax + sg * dx * d, ay + sg * dy * d);
                    if inside(x, y) && shape[at(x, y)] as usize == sh {
                        anchor = Some((x, y));
                        break;
                    }
                }
                d += st;
            }
            if let Some(a) = anchor {
                seeds.push(a);
            }
        }
        for j in 0..ny {
            for i in 0..nx {
                let jx = hash(inp.seed, s as u64, i as u64, j as u64 * 2 + 1) - 0.5;
                let jy = hash(inp.seed, s as u64, i as u64, j as u64 * 2 + 2) - 0.5;
                let (cx, cy) = ((i as f32 + 0.5 + 0.6 * jx) * sp, (j as f32 + 0.5 + 0.6 * jy) * sp);
                if inside(cx, cy) {
                    seeds.push((cx, cy));
                }
            }
        }
        for (cx, cy) in seeds {
            {
                let i0 = at(cx, cy);
                let z = zone[i0];
                cnt[0] += 1;
                if blocked(i0) { cnt[1] += 1; continue; }
                if claimed[i0] { cnt[2] += 1; continue; }
                if s >= sessions_of[i0] as usize { cnt[3] += 1; continue; }
                let wc = inp.wash.map_or(0.0, |m| m[i0].clamp(-1.0, 1.0));
                let m = inp.focus.get(i0).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                // (The brush on the focus narrower by its sight, as the engine's; not widened by the class — the
                // background has its one session, and a brush of a fifth of the sheet spills a fifth of the sheet.)
                let mut r_loc = r * (1.0 - 0.35 * m);
                let c0 = class_of(i0);
                // The background of a wash (or of a sheet without classes): bands across the class's direction.
                let band = z == 0 && (inp.wash.is_none() || wc >= 0.5);
                let mut wob = 0.0f32;
                if band {
                    // (No phase: the band runs through its seed, and the seeds under a laid band are thin — the
                    // claim alone spaces the bands, abutting. The band's identity for its water's unevenness is
                    // taken across the class's direction, whatever form the band itself follows.)
                    let (bdx, bdy) = class_dir[c0 as usize];
                    let pitch = 2.0 * r_loc;
                    let across = cx * -bdy + cy * bdx;
                    let band_id = (across / pitch).floor() as i64;
                    let along_id = ((cx * bdx + cy * bdy) / (2.0 * r_loc).max(1.0)).floor() as i64;
                    wob = inp.strokes_k.clamp(0.0, 1.0) * 0.3 * (2.0 * hash(inp.seed, s as u64 + 211, band_id as u64, along_id as u64) - 1.0);
                }
                let max_len = r_loc * match z { 2 => 4.0, 1 => 6.0, _ => if band { 12.0 } else { 8.0 } };
                // (The step a third of the radius: the dabs overlap along the stroke as the engine's do, and the
                // trace sees a boundary before it has crossed it; a band bends up to 60°, a stroke 45°.)
                let step = (0.3 * r_loc).max(1.0);
                let turn_min = if band { 0.5 } else { 0.7 };
                // THE BRUSH INSIDE ITS SHAPE (set 14): a wet session's fill never spills over the shape's edge — the
                // brush narrows to the seed's distance from the edge (down to a quarter of the session's brush;
                // under that the place waits for a smaller brush), and the trace stops where the brush would no
                // longer fit; the edge itself is the EDGE STROKE's.
                let sh0 = shape[i0] as usize;
                let small0 = sh0 >= 2 && shape_axis[sh0].5 < r_loc;
                if wet && !small0 && dist[i0] < r_loc {
                    let fit = dist[i0];
                    if fit < 0.25 * r {
                        cnt[6] += 1;
                        continue;
                    }
                    r_loc = fit;
                }
                let r_fit = if wet { r_loc } else { 0.5 * r_loc };
                let open = |i2: usize| !blocked(i2) && zone[i2] == z && class_of(i2) == c0 && sessions_of[i2] as usize > s && dist[i2] >= r_fit;
                let trace = |sdx: f32, sdy: f32, follow: bool| -> Vec<[f32; 2]> {
                    let (mut back, mut fwd): (Vec<[f32; 2]>, Vec<[f32; 2]>) = (Vec::new(), Vec::new());
                    for (sg, out) in [(-1.0f32, &mut back), (1.0, &mut fwd)] {
                        let (mut x, mut y, mut dx, mut dy, mut len) = (cx, cy, sdx * sg, sdy * sg, 0.0f32);
                        while len + step <= max_len {
                            let (x2, y2) = (x + dx * step, y + dy * step);
                            if !inside(x2, y2) {
                                break;
                            }
                            let i2 = at(x2, y2);
                            if !open(i2) {
                                break;
                            }
                            if follow {
                                let (vx, vy) = dir_of(i2);
                                let (vx, vy) = if vx * dx + vy * dy < 0.0 { (-vx, -vy) } else { (vx, vy) };
                                if vx * dx + vy * dy < turn_min {
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
                    path
                };
                let (sdx, sdy) = dir_of(i0);
                // A SMALL SHAPE (set 14): a shape the brush does not fit across (its half-width under the radius) is
                // laid as ONE stroke a session along its own axis — the crown of a tree a stroke the crown's
                // length with the brush of the crown's half-width — never as discs along a grid.
                let sh = shape[i0] as usize;
                let small = sh >= 2 && shape_axis[sh].5 < r_loc;
                let mut path = if small {
                    if shape_done[sh] {
                        cnt[2] += 1;
                        continue;
                    }
                    shape_done[sh] = true;
                    let (_, _, dx, dy, half, wid) = shape_axis[sh];
                    r_loc = wid.max(1.0).min(r_loc);
                    let st = (0.3 * r_loc).max(1.0);
                    let mut back: Vec<[f32; 2]> = Vec::new();
                    let mut fwd: Vec<[f32; 2]> = Vec::new();
                    for (sg, out) in [(-1.0f32, &mut back), (1.0, &mut fwd)] {
                        let mut len = st;
                        while len <= 2.0 * half + st {
                            let (x, y) = (cx + sg * dx * len, cy + sg * dy * len);
                            if !inside(x, y) || shape[at(x, y)] as usize != sh {
                                break;
                            }
                            out.push([x, y]);
                            len += st;
                        }
                    }
                    back.reverse();
                    back.push([cx, cy]);
                    back.extend(fwd);
                    back
                } else if band && s == 0 {
                    // THE FLOOD'S BANDS STRAIGHT (set 14): the first wash is laid in parallel strokes along the
                    // class's direction, as a flat wash is; the glazes after it follow the form.
                    let (bdx, bdy) = class_dir[c0 as usize];
                    trace(bdx, bdy, false)
                } else {
                    trace(sdx, sdy, true)
                };
                let step = if small { (0.3 * r_loc).max(1.0) } else { step };
                // NO DAUB: a wet session's stroke is a movement — short of two radii along the structure, it is traced
                // again straight along the class's direction; short of that too, the place is left (the paper
                // breathes there rather than take a disc). The dry brushes keep the short tick.
                let min_pts = (2.0 * r_loc / step).ceil() as usize + 1;
                if wet && small && path.len() < 3 {
                    cnt[6] += 1;
                    continue;
                }
                if wet && !small && path.len() < min_pts {
                    let (bdx, bdy) = class_dir[c0 as usize];
                    let straight = trace(bdx, bdy, false);
                    if straight.len() > path.len() {
                        path = straight;
                    }
                    if path.len() < min_pts {
                        // THE SMALLER BRUSH: the run the structure allows is the stroke's length, and the brush
                        // narrows to half of it — a tree, a boat, a rock smaller than the session's brush is laid
                        // with the brush that fits it, down to a quarter of the session's; under that, nothing.
                        let fit = 0.5 * step * (path.len() - 1) as f32;
                        if fit < 0.25 * r_loc {
                            cnt[6] += 1;
                            continue;
                        }
                        r_loc = fit;
                    }
                }
                if path.len() == 1 {
                    // (The dry brush's tick: three dabs along the direction at 0.6 of the radius, the end dabs kept
                    // only where the ground is the stroke's own — the tick never reaches over an edge.)
                    let st = 0.6 * r_loc;
                    let mut tick = Vec::with_capacity(3);
                    for sg in [-1.0f32, 1.0] {
                        let (x, y) = (cx + sg * sdx * st, cy + sg * sdy * st);
                        if inside(x, y) && open(at(x, y)) {
                            tick.push([x, y]);
                        }
                        if sg < 0.0 {
                            tick.push([cx, cy]);
                        }
                    }
                    path = tick;
                }
                // The claim: the pixels under the footprint not yet claimed this session are the stroke's mass; a
                // stroke over claimed ground is not laid.
                // The stroke's CORE — half the radius about the path — is what it claims and what it carries; the
                // brush's spread beyond the core is the wet edge the neighbouring strokes share (the dab's profile
                // falls to nothing at the radius, so a sheet tiled by whole footprints is blotches with the paper
                // between; tiled by cores, the spreads overlap two- to threefold and sum to the film).
                let stride = if small { stride.min(((r_loc / 6.0).round() as usize).max(1)) } else { stride };
                let r_core = (0.5 * r_loc).max(0.5);
                sid += 1;
                let own = |q: usize| zone[q] == z && class_of(q) == c0 && sessions_of[q] as usize > s;
                let (fresh, foot, contrast_sum) = claim_core(&path, r_core, stride, w, h, &own, &|q| !blocked(q), &contrast, &mut claimed, &mut seen, sid);
                if foot == 0 || (fresh as f32) < 0.2 * foot as f32 {
                    cnt[4] += 1;
                    continue;
                }
                cnt[5] += 1;
                let area = fresh as f32 * (stride * stride) as f32;
                let mean_contrast = contrast_sum / fresh as f32;
                // The stroke's worth: the ground it covers × the zone's bent saliency × the picture's contrast along
                // it (the saliency and the contrast each over a floor of 0.05: a background's glaze weighs by its ground, not
                // nothing) — a wide stroke of a wet session worth more than a fine one of a dry one, so a budget takes the
                // finish away from the finest session upward and never the masses. (The first flood outweighs every
                // stroke after it, whatever their worth: it is never dropped.)
                let weight = if s == 0 { 1.0e12 + area } else { area * (0.05 + s_eff[i0]) * (0.05 + mean_contrast) };
                strokes.push(GestaltStroke { session: s, path, radius: r_loc, zone: z, class: c0, area, weight, wob });
            }
        }
        if dbg {
            eprintln!("gestalt session {s}: seeds {} blocked {} claimed {} closed {} thin {} planned {} short {}", cnt[0], cnt[1], cnt[2], cnt[3], cnt[4], cnt[5], cnt[6]);
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
    let shape_size: Vec<u32> = moments.iter().map(|m| m[0] as u32).collect();
    Gestalt { strokes, zone, sessions: sessions_of, shape, shape_size, negative }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input<'a>(w: usize, h: usize, sal: &'a [f32], luma: &'a [f32], low: &'a [f32], focus: &'a [f32], paper: &'a [bool], radii: &'a [f32], class: Option<&'a [u8]>, wash: Option<&'a [f32]>) -> GestaltInput<'a> {
        GestaltInput { w, h, sal, luma, low_luma: low, focus, class, wash, paper, whites: None, sessions: radii.len(), crisp_from: radii.len().saturating_sub(2).max(1), radii, abstraction: 0.0, negative: 0.0, budget: 0, strokes_k: 0.5, seed: 7 }
    }

    #[test]
    fn the_triage_cuts_the_zones_and_the_ladder_follows_the_contrast() {
        // The focus above 0.4, the middle ground to 0.15, the background under; the abstraction bends the saliency
        // down (0.6 at 0 a focus, at 1 — 0.6⁴ ≈ 0.13 — a background). The ladder: the background's flood, and a
        // glaze where the contrast is over half; the middle ground's flood and up to three wide sessions by the
        // contrast; the focus's flood, glazes and first dry brush everywhere, the finest brush by the contrast.
        assert_eq!(ladder(0, 0.0, 6, 4), 2);
        assert_eq!(ladder(0, 1.0, 6, 4), 3);
        assert_eq!(ladder(1, 0.0, 6, 4), 2);
        assert_eq!(ladder(1, 0.5, 6, 4), 3);
        assert_eq!(ladder(1, 1.0, 6, 4), 3);
        assert_eq!(ladder(1, 1.0, 6, 2), 2);
        assert_eq!(ladder(0, 0.0, 6, 1), 1, "one wet session: the flood alone");
        assert_eq!(ladder(2, 0.0, 6, 4), 5);
        assert_eq!(ladder(2, 1.0, 6, 4), 6);
        assert_eq!(ladder(2, 0.0, 3, 2), 3);
        assert_eq!(ladder(0, 1.0, 3, 1), 1, "the background's glaze a wet brush or none");
        assert!(bend(0.6, 0.0) > FOCUS_ABOVE && bend(0.6, 1.0) < MIDDLE_ABOVE, "{} {}", bend(0.6, 0.0), bend(0.6, 1.0));
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
        // (The contrast's reference the picture's own blur: the flat sky and ground owe no contrast, the horizon
        // and the box's edges do.)
        let low = box_blur(&luma, w, h, 8);
        let focus = sal.clone();
        let paper = vec![false; n];
        let radii = [12.8f32, 6.4, 3.2, 1.6];
        let class: Vec<u8> = (0..n).map(|i| if i / w < h / 2 { 0 } else { 2 }).collect();
        let wash: Vec<f32> = (0..n).map(|i| if i / w < h / 2 { 1.0 } else { 0.3 }).collect();
        let base = input(w, h, &sal, &luma, &low, &focus, &paper, &radii, Some(&class), Some(&wash));
        let g = plan(&base);
        assert!(g.zone.iter().enumerate().all(|(i, z)| (*z == 2) == in_box(i % w, i / w)), "the box the focus, the rest the background");
        // The ladder: the sky the flood and its glaze (the two wet sessions of four), the box the flood, the glaze
        // and the first dry brush at least, its edges the finest brush.
        assert!((0..n).filter(|&i| !in_box(i % w, i / w)).all(|i| g.sessions[i] == 2), "the background two washes");
        assert!((0..n).filter(|&i| in_box(i % w, i / w)).all(|i| g.sessions[i] >= 3), "the focus three sessions at least");
        assert!((0..n).filter(|&i| in_box(i % w, i / w)).any(|i| g.sessions[i] == 4), "the focus's edges the finest brush");
        let first: Vec<&GestaltStroke> = g.strokes.iter().filter(|s| s.session == 0).collect();
        let later: Vec<&GestaltStroke> = g.strokes.iter().filter(|s| s.session > 0).collect();
        assert!(!first.is_empty() && !later.is_empty(), "{} / {}", first.len(), later.len());
        assert!(first.iter().any(|s| s.zone == 0) && first.iter().any(|s| s.zone == 2), "the first session floods the background and the focus");
        assert!(later.iter().filter(|s| s.session >= 2).all(|s| s.zone == 2 && s.path.iter().all(|d| in_box(d[0] as usize, d[1] as usize))), "the sessions after the glaze on the focus alone");
        assert!(later.iter().any(|s| s.session == 1 && s.zone == 0 && (s.path[0][1] as usize) < 16), "the background's glaze over the whole sky");
        // The shapes: the sky and the ground two shapes, the box the focus's own.
        assert!(g.shape[5 * w + 5] >= 2 && g.shape[(h - 5) * w + 5] >= 2 && g.shape[5 * w + 5] != g.shape[(h - 5) * w + 5] && g.shape[40 * w + 60] == 1, "{} {} {}", g.shape[5 * w + 5], g.shape[(h - 5) * w + 5], g.shape[40 * w + 60]);
        assert!(g.strokes.iter().filter(|s| s.session == 3).count() > 0, "the finest session reaches the focus");
        // No daub: every wet session's stroke runs two radii at least; the dry brushes may tick.
        for s in g.strokes.iter().filter(|s| s.session < base.crisp_from) {
            let len: f32 = s.path.windows(2).map(|p| ((p[1][0] - p[0][0]).powi(2) + (p[1][1] - p[0][1]).powi(2)).sqrt()).sum();
            assert!(len >= 2.0 * s.radius - 1e-3, "session {}: a stroke of {len} under two radii ({})", s.session, s.radius);
        }
        // (The fill's bands — the edge strokes, half the brush, run along the horizon and round the box.)
        let bands: Vec<&GestaltStroke> = first.iter().copied().filter(|s| s.zone == 0 && (s.path[0][1] as usize) < h / 2 && s.radius > 0.6 * radii[0]).collect();
        assert!(first.iter().any(|s| s.zone == 0 && s.radius < 0.6 * radii[0] && s.path.len() >= 3), "the edge strokes along the sky's edge");
        assert!(!bands.is_empty() && bands.iter().all(|s| s.path.iter().all(|d| (d[1] - s.path[0][1]).abs() < 1e-3)), "the sky's bands run along the sheet");
        assert!(bands.iter().any(|s| s.path.len() >= 4), "…as long strokes");
        assert!(g.strokes.iter().all(|s| s.area > 0.0 && s.weight > 0.0));
        for sess in 0..radii.len() {
            let sum: f32 = g.strokes.iter().filter(|s| s.session == sess).map(|s| s.area).sum();
            assert!(sum <= n as f32 * 1.05, "session {sess}: the claims {sum} over the sheet {n} — the strokes lay the debt once");
        }
        // A SMALL SHAPE: a dark crown of foliage, 24 wide and 8 tall, in the sky far from the box — one stroke a
        // session along its axis (horizontal), the brush its half-width, never a round mark.
        let crown = |x: usize, y: usize| { let (dx, dy) = (x as f32 - 20.0, y as f32 - 10.0); dx * dx / 144.0 + dy * dy / 16.0 <= 1.0 };
        let luma_c: Vec<f32> = (0..n).map(|i| if crown(i % w, i / w) { 0.15 } else { luma[i] }).collect();
        let class_c: Vec<u8> = (0..n).map(|i| if crown(i % w, i / w) { 3 } else { class[i] }).collect();
        let low_c = box_blur(&luma_c, w, h, 8);
        let gc = plan(&input(w, h, &sal, &luma_c, &low_c, &focus, &paper, &radii, Some(&class_c), Some(&wash)));
        let crown_strokes: Vec<&GestaltStroke> = gc.strokes.iter().filter(|s| s.class == 3).collect();
        assert!(!crown_strokes.is_empty(), "the crown is laid");
        for s in &crown_strokes {
            assert!(s.path.len() >= 3 && s.path.iter().all(|d| crown(d[0] as usize, d[1] as usize)), "within the crown: {:?}", s.path);
            assert!(s.path.iter().all(|d| (d[1] - 10.0).abs() < 1.5), "along the crown's axis: {:?}", s.path);
            assert!(s.radius <= 4.5 && s.radius >= 1.0, "the brush the crown's half-width: {}", s.radius);
        }
        assert!(crown_strokes.iter().filter(|s| s.session == 0).count() == 1, "one stroke a session");
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
