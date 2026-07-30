//! Procedural detail-overlay generators (RFC §8.3, §8.8). Each is a **pure, byte-stable** function of
//! its detail record + a seed + the estimated scene light — an `RgbaImage` centred on the detail's
//! anchor, ready to alpha-composite (§8.4). No weights, no RNG beyond a seeded splitmix64.
//!
//! What makes a composited mark read as *skin* rather than a sticker (§8.8): a **maturity** colour ramp
//! (fresh pink-red → mature pale/depressed), a **relief** highlight/shadow pair aligned to the scene
//! light, and a believable **edge** (soft / defined / irregular). All three are modelled here; the one
//! stochastic step (harmonisation) is applied later, in the compositing pass, not in these generators.

use image::{Rgba, RgbaImage};

/// Scene light as a unit vector pointing *toward* the light source (image coords, +x right / +y down).
/// The compositor estimates it from the face's own shading; a top-left key light is the default.
#[derive(Debug, Clone, Copy)]
pub struct Light {
    pub dx: f32,
    pub dy: f32,
}

impl Default for Light {
    fn default() -> Self {
        // top-left key light (the studio-portrait default).
        let n = (0.5f32 * 0.5 + 0.5 * 0.5).sqrt();
        Light { dx: -0.5 / n, dy: -0.5 / n }
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
fn rand01(state: &mut u64) -> f32 {
    (splitmix64(state) >> 11) as f32 / (1u64 << 53) as f32
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}
fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// Apply a relief highlight/shadow term: `nx,ny` is the surface-normal tilt at this pixel (from the
/// mark's local gradient), `raised` in `[-1,1]` (>0 raised, <0 depressed). Returns an additive delta.
fn relief_delta(nx: f32, ny: f32, raised: f32, light: Light) -> f32 {
    // dot of the (tilted) normal with the light direction → lit vs shadowed side.
    let d = nx * light.dx + ny * light.dy;
    d * raised * 42.0
}

/// A **mole**: a soft radial patch with a relief-shaded rim. `size_px` is the diameter; `color` the
/// base sRGB; `raised` in `[0,1]` the elevation (drives the highlight/shadow rim).
pub fn mole(size_px: u32, color: [u8; 3], raised: f32, light: Light) -> RgbaImage {
    let s = size_px.max(3);
    let mut img = RgbaImage::new(s, s);
    let (cx, cy) = (s as f32 / 2.0, s as f32 / 2.0);
    let r = s as f32 / 2.0;
    let base = [color[0] as f32, color[1] as f32, color[2] as f32];
    for y in 0..s {
        for x in 0..s {
            let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
            let dist = (dx * dx + dy * dy).sqrt() / r;
            if dist >= 1.0 {
                continue;
            }
            // soft alpha: opaque core, feathered edge.
            let alpha = ((1.0 - dist) / 0.35).clamp(0.0, 1.0);
            // relief: normal tilts outward near the rim → rim catches/loses the light.
            let (nx, ny) = if dist > 1e-3 { (dx / (r * dist).max(1e-3), dy / (r * dist).max(1e-3)) } else { (0.0, 0.0) };
            let d = relief_delta(nx, ny, raised * 2.0 - 0.0, light) * dist; // strongest at the rim
            let px = [base[0] + d, base[1] + d, base[2] + d];
            img.put_pixel(x, y, Rgba([clamp_u8(px[0]), clamp_u8(px[1]), clamp_u8(px[2]), clamp_u8(alpha * 235.0)]));
        }
    }
    img
}

/// A **scar**: a linear stroke of length `len_px`, thickness `width_px`, rotated by `orientation`
/// (radians). `maturity` in `[0,1]` interpolates the colour ramp (0 fresh pink-red → 1 pale/flat) and
/// `relief` in `[0,1]` the raised/depressed highlight-shadow pair along the stroke (§8.8).
pub fn scar(len_px: u32, width_px: u32, orientation: f32, maturity: f32, relief: f32, skin: [u8; 3], light: Light) -> RgbaImage {
    let len = len_px.max(4) as f32;
    let w = width_px.max(2) as f32;
    let pad = (w * 2.0) as u32 + 2;
    let dim = len_px.max(4) + pad * 2;
    let mut img = RgbaImage::new(dim, dim);
    let (cx, cy) = (dim as f32 / 2.0, dim as f32 / 2.0);
    let (ca, sa) = (orientation.cos(), orientation.sin());
    // colour ramp: fresh = skin pushed toward red; mature = skin lightened + desaturated.
    let fresh = [(skin[0] as f32 + 60.0).min(255.0), skin[1] as f32 * 0.7, skin[2] as f32 * 0.7];
    let mature = [(skin[0] as f32 + 25.0).min(255.0), (skin[1] as f32 + 20.0).min(255.0), (skin[2] as f32 + 20.0).min(255.0)];
    let colour = lerp3(fresh, mature, maturity);
    // depressed when mature, raised when fresh: relief sign follows maturity.
    let raised = relief * (1.0 - 2.0 * maturity);
    for y in 0..dim {
        for x in 0..dim {
            // rotate the pixel into stroke-local (u along, v across).
            let (px, py) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
            let u = px * ca + py * sa;
            let v = -px * sa + py * ca;
            if u.abs() > len / 2.0 || v.abs() > w / 2.0 {
                continue;
            }
            // feather across the width + taper at the ends.
            let across = 1.0 - (v.abs() / (w / 2.0)).powf(1.5);
            let along = 1.0 - (u.abs() / (len / 2.0)).powf(3.0);
            let alpha = (across * along).clamp(0.0, 1.0);
            // relief: normal tilts across the stroke (v axis), lit per the across-light dot.
            let nv = (v / (w / 2.0)).clamp(-1.0, 1.0);
            let (nx, ny) = (-nv * sa, nv * ca); // across-axis normal in image space
            let d = relief_delta(nx, ny, raised, light);
            let c = [colour[0] + d, colour[1] + d, colour[2] + d];
            img.put_pixel(x, y, Rgba([clamp_u8(c[0]), clamp_u8(c[1]), clamp_u8(c[2]), clamp_u8(alpha * 210.0)]));
        }
    }
    img
}

/// A **birthmark**: a soft-edged irregular blob. `aspect` (h/w), `edge` character, `intensity` the
/// opacity, `color` the sRGB tone. Edge noise is seeded (byte-stable).
pub fn birthmark(size_px: u32, aspect: f32, edge: &str, intensity: f32, color: [u8; 3], seed: u64) -> RgbaImage {
    let s = size_px.max(4);
    let h = ((s as f32) * aspect.clamp(0.4, 2.5)) as u32;
    let h = h.max(4);
    let mut img = RgbaImage::new(s, h);
    let (cx, cy) = (s as f32 / 2.0, h as f32 / 2.0);
    let (rx, ry) = (s as f32 / 2.0, h as f32 / 2.0);
    let (feather, noise_amp) = match edge {
        "defined" => (0.12, 0.05),
        "irregular" => (0.30, 0.35),
        _ => (0.45, 0.12), // soft (default)
    };
    // per-angle radius perturbation for an irregular boundary.
    let mut st = seed ^ 0xB1A7_C0DE_5EED_1234;
    let harmonics: [(f32, f32); 3] = [
        (1.0 + rand01(&mut st), rand01(&mut st) * std::f32::consts::TAU),
        (2.0 + rand01(&mut st), rand01(&mut st) * std::f32::consts::TAU),
        (3.0 + rand01(&mut st), rand01(&mut st) * std::f32::consts::TAU),
    ];
    let base = [color[0] as f32, color[1] as f32, color[2] as f32];
    for y in 0..h {
        for x in 0..s {
            let (dx, dy) = ((x as f32 + 0.5 - cx) / rx, (y as f32 + 0.5 - cy) / ry);
            let ang = dy.atan2(dx);
            let wobble: f32 = harmonics.iter().map(|&(f, ph)| (ang * f + ph).sin()).sum::<f32>() / 3.0 * noise_amp;
            let dist = (dx * dx + dy * dy).sqrt() * (1.0 + wobble);
            if dist >= 1.0 {
                continue;
            }
            let alpha = ((1.0 - dist) / feather).clamp(0.0, 1.0) * intensity.clamp(0.0, 1.0);
            img.put_pixel(x, y, Rgba([base[0] as u8, base[1] as u8, base[2] as u8, clamp_u8(alpha * 200.0)]));
        }
    }
    img
}

/// A **freckle / pockmark field** (distributional, §8.2/§8.8): a seeded scatter of small dots over the
/// given `mask` (255 = inside region). `density` scales the count; `color` the tone; `seed` fixes it.
/// The returned RGBA is the mask's size, so the compositor places it over the region directly.
pub fn freckle_field(mask: &image::GrayImage, density: f32, color: [u8; 3], seed: u64) -> RgbaImage {
    let (w, h) = mask.dimensions();
    let mut img = RgbaImage::new(w, h);
    // area of the region → number of freckles.
    let area = mask.pixels().filter(|p| p.0[0] > 0).count() as f32;
    let n = (area * density.clamp(0.0, 1.0) * 0.02) as u32;
    let mut st = seed ^ 0xF00D_FACE_1234_5678;
    for _ in 0..n {
        let fx = (rand01(&mut st) * w as f32) as u32;
        let fy = (rand01(&mut st) * h as f32) as u32;
        if fx >= w || fy >= h || mask.get_pixel(fx, fy).0[0] == 0 {
            continue;
        }
        let r = 1 + (rand01(&mut st) * 2.0) as i32;
        // neutral per-dot jitter (equal across channels) so a skin-derived grey stays grey.
        let jitter = (rand01(&mut st) - 0.5) * 24.0;
        let c = [clamp_u8(color[0] as f32 + jitter), clamp_u8(color[1] as f32 + jitter), clamp_u8(color[2] as f32 + jitter)];
        let a = clamp_u8(120.0 + rand01(&mut st) * 80.0);
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy > r * r {
                    continue;
                }
                let (px, py) = (fx as i32 + dx, fy as i32 + dy);
                if px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h && mask.get_pixel(px as u32, py as u32).0[0] > 0 {
                    img.put_pixel(px as u32, py as u32, Rgba([c[0], c[1], c[2], a]));
                }
            }
        }
    }
    img
}

/// Metal tone → base sRGB (§8.3 recolour). Generic tones only (no trade-dress).
pub fn metal_colour(metal: &str) -> [u8; 3] {
    match metal {
        "gold" | "yellow-gold" => [212, 175, 55],
        "rose-gold" => [200, 150, 130],
        "white-gold" | "platinum" => [222, 224, 228],
        "silver" => [200, 204, 210],
        "steel" | "titanium" => [170, 174, 180],
        "black" | "oxidised" => [60, 60, 66],
        "bronze" | "copper" => [176, 120, 78],
        _ => [200, 204, 210], // default silver-ish
    }
}

/// Stone tint → sRGB (§8.3). Generic gem tones.
pub fn stone_colour(stone: &str) -> [u8; 3] {
    match stone {
        "diamond" | "clear" | "crystal" => [235, 240, 245],
        "ruby" | "red" => [190, 40, 60],
        "sapphire" | "blue" => [40, 70, 170],
        "emerald" | "green" => [40, 150, 90],
        "amethyst" | "purple" => [130, 80, 170],
        "onyx" | "black" => [30, 30, 34],
        "pearl" | "white" => [240, 238, 230],
        "turquoise" => [64, 190, 190],
        _ => [220, 225, 230],
    }
}

/// A **jewelry** overlay from a generic procedural shape (§8.3 — original/PD, no trade-dress). `kind`
/// selects stud / hoop / pendant / bar; `metal`/`stone` recolour. Returned RGBA is `size_px` square-ish.
pub fn jewelry(kind: &str, size_px: u32, metal: [u8; 3], stone: Option<[u8; 3]>) -> RgbaImage {
    let s = size_px.max(6);
    let mut img = RgbaImage::new(s, s);
    let (cx, cy) = (s as f32 / 2.0, s as f32 / 2.0);
    let put = |img: &mut RgbaImage, x: i32, y: i32, c: [u8; 3], a: u8| {
        if x >= 0 && y >= 0 && (x as u32) < s && (y as u32) < s {
            img.put_pixel(x as u32, y as u32, Rgba([c[0], c[1], c[2], a]));
        }
    };
    // simple shaded disc (a light rim for metallic read).
    let disc = |img: &mut RgbaImage, ccx: f32, ccy: f32, r: f32, c: [u8; 3]| {
        let ri = r.ceil() as i32;
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                let dist = ((dx * dx + dy * dy) as f32).sqrt() / r;
                if dist <= 1.0 {
                    let shade = 1.0 + (-(dx as f32) - (dy as f32)) / (r * 4.0); // top-left highlight
                    let cc = [clamp_u8(c[0] as f32 * shade), clamp_u8(c[1] as f32 * shade), clamp_u8(c[2] as f32 * shade)];
                    put(img, (ccx + dx as f32) as i32, (ccy + dy as f32) as i32, cc, if dist > 0.85 { 200 } else { 255 });
                }
            }
        }
    };
    match kind {
        "hoop" | "ring" => {
            // an annulus.
            let (ro, ri) = (s as f32 * 0.46, s as f32 * 0.30);
            let rio = ro.ceil() as i32;
            for dy in -rio..=rio {
                for dx in -rio..=rio {
                    let d = ((dx * dx + dy * dy) as f32).sqrt();
                    if d <= ro && d >= ri {
                        let shade = 1.0 + (-(dx as f32) - (dy as f32)) / (ro * 4.0);
                        let cc = [clamp_u8(metal[0] as f32 * shade), clamp_u8(metal[1] as f32 * shade), clamp_u8(metal[2] as f32 * shade)];
                        put(&mut img, (cx + dx as f32) as i32, (cy + dy as f32) as i32, cc, 255);
                    }
                }
            }
        }
        "pendant" | "necklace" | "drop" => {
            // a small bail at top + a stone/metal drop below centre.
            disc(&mut img, cx, cy * 0.5, s as f32 * 0.10, metal);
            disc(&mut img, cx, cy * 1.15, s as f32 * 0.30, stone.unwrap_or(metal));
        }
        "bar" | "barbell" => {
            // a horizontal bar with two ball ends.
            let r = s as f32 * 0.14;
            for x in (r as i32)..(s as i32 - r as i32) {
                put(&mut img, x, cy as i32, metal, 255);
                put(&mut img, x, cy as i32 + 1, metal, 255);
            }
            disc(&mut img, r, cy, r, metal);
            disc(&mut img, s as f32 - r, cy, r, metal);
        }
        // "stud" and default: a metal setting with an optional stone.
        _ => {
            disc(&mut img, cx, cy, s as f32 * 0.46, metal);
            if let Some(st) = stone {
                disc(&mut img, cx, cy, s as f32 * 0.26, st);
            }
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ink(img: &RgbaImage) -> usize {
        img.pixels().filter(|p| p.0[3] > 0).count()
    }
    fn hash(img: &RgbaImage) -> u64 {
        let mut acc: u64 = 1469598103934665603;
        for p in img.pixels() {
            for &c in &p.0 {
                acc = (acc ^ c as u64).wrapping_mul(1099511628211);
            }
        }
        acc
    }

    #[test]
    fn mole_is_a_soft_opaque_centre_with_feathered_edge() {
        let m = mole(20, [90, 55, 40], 0.7, Light::default());
        assert_eq!(m.get_pixel(10, 10).0[3], 235, "opaque core");
        assert_eq!(m.get_pixel(0, 0).0[3], 0, "transparent corner");
        assert!(ink(&m) > 50);
    }

    #[test]
    fn scar_maturity_shifts_colour_red_to_pale() {
        let fresh = scar(40, 4, 0.0, 0.0, 0.5, [200, 150, 130], Light::default());
        let mature = scar(40, 4, 0.0, 1.0, 0.5, [200, 150, 130], Light::default());
        let centre = |i: &RgbaImage| i.get_pixel(i.width() / 2, i.height() / 2).0;
        // fresh scar is redder (higher R−G) than a mature one.
        let redness = |p: [u8; 4]| p[0] as i32 - p[1] as i32;
        assert!(redness(centre(&fresh)) > redness(centre(&mature)), "fresh redder than mature");
    }

    #[test]
    fn birthmark_edges_differ_in_ink() {
        let defined = birthmark(30, 1.0, "defined", 0.8, [120, 80, 70], 1);
        let irregular = birthmark(30, 1.0, "irregular", 0.8, [120, 80, 70], 1);
        // both draw; irregular is byte-different from defined (edge noise).
        assert!(ink(&defined) > 20 && ink(&irregular) > 20);
        assert_ne!(hash(&defined), hash(&irregular));
    }

    #[test]
    fn freckle_field_stays_inside_the_mask_and_is_seed_stable() {
        let mut mask = image::GrayImage::new(64, 64);
        for y in 16..48 {
            for x in 16..48 {
                mask.put_pixel(x, y, image::Luma([255]));
            }
        }
        let a = freckle_field(&mask, 0.8, [110, 70, 55], 7);
        let b = freckle_field(&mask, 0.8, [110, 70, 55], 7);
        assert_eq!(hash(&a), hash(&b), "same seed → identical");
        // no ink outside the mask.
        for (x, y, p) in a.enumerate_pixels() {
            if p.0[3] > 0 {
                assert!(mask.get_pixel(x, y).0[0] > 0, "freckle escaped the region at {x},{y}");
            }
        }
    }

    #[test]
    fn jewelry_recolours_by_metal_and_stone() {
        let gold = jewelry("stud", 24, metal_colour("gold"), Some(stone_colour("ruby")));
        assert!(ink(&gold) > 50);
        // a ruby stud has red near its centre.
        let c = gold.get_pixel(12, 12).0;
        assert!(c[0] > c[2], "ruby centre reads red");
        let hoop = jewelry("hoop", 24, metal_colour("silver"), None);
        assert_eq!(hoop.get_pixel(12, 12).0[3], 0, "hoop centre is hollow");
    }

    #[test]
    fn generators_are_byte_stable() {
        assert_eq!(hash(&mole(16, [90, 55, 40], 0.6, Light::default())), 8991750472580538086);
    }
}
