//! `plakat make-photo` — a picture passed through a PHOTOGRAPHIC CHAIN of a chosen period and style, from the
//! daguerreotype to the computational phone, chosen by `--media`.
//!
//! The chain is the same for every medium; a medium is a RECIPE (`assets/photo/<media>.hjson`) that sets the
//! chain's keys — the lens, the emulsion or the sensor, the process, the carrier, the age, the digital pipeline.
//! Every key is a dial of [`plan::PhotoPlan`], 0 or "none" when the stage is off, and a medium is tuned in its
//! recipe, never in the code. The stages, in the order the light goes through them:
//!
//! 1. **the scene** — the picture's sRGB undone to linear light, the depth (Depth-Anything-V2), the subject's
//!    matte (U²-Net) and the faces (SCRFD) read for the stages that need them;
//! 2. **the camera** ([`camera`]) — a virtual camera: the focal length, a move (dolly, truck, pedestal), a turn
//!    (pan, tilt, roll), the picture re-projected through its depth from the new point, the opened ground filled
//!    and its share reported;
//! 3. **the lens** ([`lens`]) — the depth of field from the aperture and the focus, the field's curvature, the
//!    vignette, the distortion, the chromatic aberration, the flare, the soft focus, the haze;
//! 4. **the shutter** ([`lens::shutter`]) — the exposure time: the moving things smeared, the people ghosted;
//! 5. **the emulsion or the sensor** ([`sensor`]) — the exposure, the spectral sensitivity (orthochromatic,
//!    panchromatic, the colour dyes), the characteristic curve, the latitude, the grain with its structure, the
//!    sensor's noise and its demosaic, the halation, the autochrome's starch reseau;
//! 6. **the process** ([`process`]) — the development, the toning, the print's paper and its density range, the
//!    hand tint;
//! 7. **the carrier** ([`process::carrier`]) — the plate, the tintype, the card, the Polaroid's frame; the mirror;
//! 8. **the age** ([`process::age`]) — the fade, the yellowing, the silvering, the scratches, the dust, the foxing;
//! 9. **the digital** ([`sensor::digital`]) — the JPEG, the sharpening halo, the HDR flattening, the face
//!    smoothing, the phone's portrait mode, the date stamp, the flash and the red eye.
//!
//! [`measure`] reads a picture's CAMERA STATISTICS (the noise and its structure, the sharpness over the frame,
//! the highlights, the vignette, the fringes, the JPEG) — the readings `make-photo measure` prints and the
//! critic ([`director`]) judges a render by. [`neural`] holds the model-backed options (the neural fill, the
//! refine, the relight), each behind a flag and disclosed in the report.
//!
//! Everything here but [`neural`] is deterministic, weight-free and runs on the CPU; the same plan and seed give
//! the same bytes.

pub mod camera;
pub mod director;
pub mod lens;
pub mod measure;
pub mod media;
pub mod neural;
pub mod plan;
pub mod process;
pub mod sensor;

use image::RgbImage;

/// A picture in LINEAR light, three floats per pixel, row-major; values may exceed 1 (the highlights before the
/// sensor clips them) and go under 0 only by a filter's ringing.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub w: u32,
    pub h: u32,
    pub px: Vec<[f32; 3]>,
}

/// A scalar field over a frame (a depth, a matte, a mask), row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub w: u32,
    pub h: u32,
    pub v: Vec<f32>,
}

/// sRGB (0..255) → linear (0..1).
pub fn srgb_to_linear(c: u8) -> f32 {
    let v = c as f32 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear → sRGB (0..255), clamped.
pub fn linear_to_srgb(v: f32) -> u8 {
    let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    let s = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

/// Rec.709 luma of a linear pixel.
pub fn luma(p: [f32; 3]) -> f32 {
    0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
}

impl Frame {
    pub fn new(w: u32, h: u32) -> Self {
        Frame { w, h, px: vec![[0.0; 3]; (w as usize) * (h as usize)] }
    }

    pub fn from_srgb(img: &RgbImage) -> Self {
        let lut: Vec<f32> = (0..=255u8).map(srgb_to_linear).collect();
        let (w, h) = img.dimensions();
        let px = img.pixels().map(|p| [lut[p.0[0] as usize], lut[p.0[1] as usize], lut[p.0[2] as usize]]).collect();
        Frame { w, h, px }
    }

    pub fn to_srgb(&self) -> RgbImage {
        let mut out = RgbImage::new(self.w, self.h);
        for (o, p) in out.pixels_mut().zip(&self.px) {
            o.0 = [linear_to_srgb(p[0]), linear_to_srgb(p[1]), linear_to_srgb(p[2])];
        }
        out
    }

    #[inline]
    pub fn idx(&self, x: u32, y: u32) -> usize {
        (y as usize) * (self.w as usize) + x as usize
    }

    #[inline]
    pub fn get(&self, x: i64, y: i64) -> [f32; 3] {
        let x = x.clamp(0, self.w as i64 - 1) as u32;
        let y = y.clamp(0, self.h as i64 - 1) as u32;
        self.px[self.idx(x, y)]
    }

    /// Bilinear sample at a continuous position (pixel centres at integer + 0.5), edges clamped.
    pub fn sample(&self, x: f32, y: f32) -> [f32; 3] {
        let fx = x - 0.5;
        let fy = y - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let (x0, y0) = (x0 as i64, y0 as i64);
        let a = self.get(x0, y0);
        let b = self.get(x0 + 1, y0);
        let c = self.get(x0, y0 + 1);
        let d = self.get(x0 + 1, y0 + 1);
        let mut o = [0.0; 3];
        for k in 0..3 {
            o[k] = (a[k] * (1.0 - tx) + b[k] * tx) * (1.0 - ty) + (c[k] * (1.0 - tx) + d[k] * tx) * ty;
        }
        o
    }

    pub fn luma_field(&self) -> Field {
        Field { w: self.w, h: self.h, v: self.px.iter().map(|p| luma(*p)).collect() }
    }

    pub fn map(&self, f: impl Fn([f32; 3]) -> [f32; 3]) -> Frame {
        Frame { w: self.w, h: self.h, px: self.px.iter().map(|p| f(*p)).collect() }
    }

    /// Resample to `w`×`h` (bilinear; a box prefilter when shrinking by more than 2).
    pub fn resized(&self, w: u32, h: u32) -> Frame {
        if w == self.w && h == self.h {
            return self.clone();
        }
        let src = if (self.w as f32 / w as f32) > 2.0 || (self.h as f32 / h as f32) > 2.0 {
            box_prefilter(self, ((self.w as f32 / w as f32).max(self.h as f32 / h as f32) / 2.0).floor().max(1.0) as u32)
        } else {
            self.clone()
        };
        let mut out = Frame::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let sx = (x as f32 + 0.5) * src.w as f32 / w as f32;
                let sy = (y as f32 + 0.5) * src.h as f32 / h as f32;
                out.px[(y * w + x) as usize] = src.sample(sx, sy);
            }
        }
        out
    }
}

impl Field {
    pub fn new(w: u32, h: u32, v: f32) -> Self {
        Field { w, h, v: vec![v; (w as usize) * (h as usize)] }
    }

    #[inline]
    pub fn get(&self, x: i64, y: i64) -> f32 {
        let x = x.clamp(0, self.w as i64 - 1) as usize;
        let y = y.clamp(0, self.h as i64 - 1) as usize;
        self.v[y * self.w as usize + x]
    }

    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let fx = x - 0.5;
        let fy = y - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let (x0, y0) = (x0 as i64, y0 as i64);
        (self.get(x0, y0) * (1.0 - tx) + self.get(x0 + 1, y0) * tx) * (1.0 - ty) + (self.get(x0, y0 + 1) * (1.0 - tx) + self.get(x0 + 1, y0 + 1) * tx) * ty
    }

    pub fn resized(&self, w: u32, h: u32) -> Field {
        if w == self.w && h == self.h {
            return self.clone();
        }
        let mut out = Field::new(w, h, 0.0);
        for y in 0..h {
            for x in 0..w {
                let sx = (x as f32 + 0.5) * self.w as f32 / w as f32;
                let sy = (y as f32 + 0.5) * self.h as f32 / h as f32;
                out.v[(y * w + x) as usize] = self.sample(sx, sy);
            }
        }
        out
    }

    pub fn from_gray(img: &image::GrayImage) -> Field {
        Field { w: img.width(), h: img.height(), v: img.pixels().map(|p| p.0[0] as f32 / 255.0).collect() }
    }

    pub fn mean(&self) -> f32 {
        if self.v.is_empty() {
            0.0
        } else {
            self.v.iter().sum::<f32>() / self.v.len() as f32
        }
    }
}

/// A separable Gaussian blur of a frame (σ in pixels; under 0.3 returns a copy). A large σ is run on a
/// reduced copy (the blur commutes with the reduction well enough at σ over 6) and resampled back.
pub fn gaussian(f: &Frame, sigma: f32) -> Frame {
    if sigma < 0.3 {
        return f.clone();
    }
    if sigma > 6.0 && f.w >= 16 && f.h >= 16 {
        let k = (sigma / 3.0).floor().max(2.0);
        let (sw, sh) = (((f.w as f32 / k).ceil() as u32).max(4), ((f.h as f32 / k).ceil() as u32).max(4));
        let small = f.resized(sw, sh);
        return gaussian(&small, sigma / k).resized(f.w, f.h);
    }
    let k = kernel(sigma);
    let r = (k.len() / 2) as i64;
    let (w, h) = (f.w as i64, f.h as i64);
    let mut tmp = Frame::new(f.w, f.h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for (i, kv) in k.iter().enumerate() {
                let p = f.get(x + i as i64 - r, y);
                for c in 0..3 {
                    acc[c] += p[c] * kv;
                }
            }
            tmp.px[(y * w + x) as usize] = acc;
        }
    }
    let mut out = Frame::new(f.w, f.h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for (i, kv) in k.iter().enumerate() {
                let p = tmp.get(x, y + i as i64 - r);
                for c in 0..3 {
                    acc[c] += p[c] * kv;
                }
            }
            out.px[(y * w + x) as usize] = acc;
        }
    }
    out
}

/// A separable Gaussian blur of a field (a large σ on a reduced copy, as [`gaussian`]).
pub fn gaussian_field(f: &Field, sigma: f32) -> Field {
    if sigma < 0.3 {
        return f.clone();
    }
    if sigma > 6.0 && f.w >= 16 && f.h >= 16 {
        let k = (sigma / 3.0).floor().max(2.0);
        let (sw, sh) = (((f.w as f32 / k).ceil() as u32).max(4), ((f.h as f32 / k).ceil() as u32).max(4));
        let small = f.resized(sw, sh);
        return gaussian_field(&small, sigma / k).resized(f.w, f.h);
    }
    let k = kernel(sigma);
    let r = (k.len() / 2) as i64;
    let (w, h) = (f.w as i64, f.h as i64);
    let mut tmp = Field::new(f.w, f.h, 0.0);
    for y in 0..h {
        for x in 0..w {
            tmp.v[(y * w + x) as usize] = k.iter().enumerate().map(|(i, kv)| f.get(x + i as i64 - r, y) * kv).sum();
        }
    }
    let mut out = Field::new(f.w, f.h, 0.0);
    for y in 0..h {
        for x in 0..w {
            out.v[(y * w + x) as usize] = k.iter().enumerate().map(|(i, kv)| tmp.get(x, y + i as i64 - r) * kv).sum();
        }
    }
    out
}

/// An anisotropic Gaussian: σx along the rows, σy down the columns.
pub fn gaussian_xy(f: &Frame, sx: f32, sy: f32) -> Frame {
    if sx < 0.3 && sy < 0.3 {
        return f.clone();
    }
    let kx = kernel(sx.max(0.01));
    let ky = kernel(sy.max(0.01));
    let (rx, ry) = ((kx.len() / 2) as i64, (ky.len() / 2) as i64);
    let (w, h) = (f.w as i64, f.h as i64);
    let mut tmp = Frame::new(f.w, f.h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for (i, kv) in kx.iter().enumerate() {
                let p = f.get(x + i as i64 - rx, y);
                for c in 0..3 {
                    acc[c] += p[c] * kv;
                }
            }
            tmp.px[(y * w + x) as usize] = acc;
        }
    }
    let mut out = Frame::new(f.w, f.h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for (i, kv) in ky.iter().enumerate() {
                let p = tmp.get(x, y + i as i64 - ry);
                for c in 0..3 {
                    acc[c] += p[c] * kv;
                }
            }
            out.px[(y * w + x) as usize] = acc;
        }
    }
    out
}

/// A box mean of radius `r` (the prefilter of a strong reduction).
fn box_prefilter(f: &Frame, r: u32) -> Frame {
    if r == 0 {
        return f.clone();
    }
    let r = r as i64;
    let (w, h) = (f.w as i64, f.h as i64);
    let n = ((2 * r + 1) as f32).powi(2);
    let mut tmp = Frame::new(f.w, f.h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for dx in -r..=r {
                let p = f.get(x + dx, y);
                for c in 0..3 {
                    acc[c] += p[c];
                }
            }
            tmp.px[(y * w + x) as usize] = acc;
        }
    }
    let mut out = Frame::new(f.w, f.h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for dy in -r..=r {
                let p = tmp.get(x, y + dy);
                for c in 0..3 {
                    acc[c] += p[c];
                }
            }
            out.px[(y * w + x) as usize] = [acc[0] / n, acc[1] / n, acc[2] / n];
        }
    }
    out
}

fn kernel(sigma: f32) -> Vec<f32> {
    let r = (sigma * 3.0).ceil().max(1.0) as i64;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    for v in &mut k {
        *v /= s;
    }
    k
}

/// A deterministic hash noise in [0,1) for a pixel and a seed (no RNG state, so every stage can draw its
/// own field without an order dependence).
#[inline]
pub fn hash01(x: u32, y: u32, seed: u32) -> f32 {
    let mut h = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77) ^ seed.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    (h >> 8) as f32 / 16_777_216.0
}

/// A standard normal draw from two hashes (Box–Muller).
#[inline]
pub fn hash_gauss(x: u32, y: u32, seed: u32) -> f32 {
    let u1 = hash01(x, y, seed).max(1e-7);
    let u2 = hash01(x, y, seed ^ 0x5bd1_e995);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// A smooth VALUE NOISE field in [0,1] with a cell size in pixels (bilinear between lattice draws) — the
/// clumps of grain, the mottle of a collodion, the blotches of foxing.
pub fn value_noise(w: u32, h: u32, cell: f32, seed: u32) -> Field {
    let cell = cell.max(1.0);
    let mut out = Field::new(w, h, 0.0);
    for y in 0..h {
        for x in 0..w {
            let gx = x as f32 / cell;
            let gy = y as f32 / cell;
            let (x0, y0) = (gx.floor(), gy.floor());
            let (tx, ty) = (gx - x0, gy - y0);
            let (tx, ty) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
            let (x0, y0) = (x0 as u32, y0 as u32);
            let a = hash01(x0, y0, seed);
            let b = hash01(x0 + 1, y0, seed);
            let c = hash01(x0, y0 + 1, seed);
            let d = hash01(x0 + 1, y0 + 1, seed);
            out.v[(y * w + x) as usize] = (a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty;
        }
    }
    out
}

/// Several octaves of value noise, in [0,1] after normalisation.
pub fn fbm(w: u32, h: u32, cell: f32, octaves: u32, seed: u32) -> Field {
    let mut out = Field::new(w, h, 0.0);
    let mut amp = 1.0;
    let mut total = 0.0;
    for o in 0..octaves.max(1) {
        let n = value_noise(w, h, (cell / (1u32 << o) as f32).max(1.0), seed.wrapping_add(o * 7919));
        for (a, b) in out.v.iter_mut().zip(&n.v) {
            *a += b * amp;
        }
        total += amp;
        amp *= 0.5;
    }
    for a in &mut out.v {
        *a /= total;
    }
    out
}

/// The everything-the-stages-may-need bundle read once from the picture: the depth (near = 1, far = 0,
/// relative), the subject's matte, the faces' boxes and eye points — each optional, each stage standing down
/// without it. All at the frame's size.
#[derive(Clone, Debug, Default)]
pub struct Aux {
    pub depth: Option<Field>,
    /// The metric depth in metres (set by the camera stage).
    pub metric: Option<Field>,
    pub matte: Option<Field>,
    /// The share of moving stuff (water, foliage, sky) per pixel, coarse, from the stuff read over crops.
    pub moving: Option<Field>,
    /// Faces as `(x1, y1, x2, y2, [left eye, right eye])` in frame pixels.
    pub faces: Vec<([f32; 4], [[f32; 2]; 2])>,
    /// What each reader reported (ran, or why not).
    pub sources: Vec<String>,
}

impl Aux {
    /// Resample the fields and scale the boxes to a new frame size.
    pub fn resized(&self, w: u32, h: u32, from: (u32, u32)) -> Aux {
        let sx = w as f32 / from.0.max(1) as f32;
        let sy = h as f32 / from.1.max(1) as f32;
        Aux {
            depth: self.depth.as_ref().map(|d| d.resized(w, h)),
            metric: self.metric.as_ref().map(|d| d.resized(w, h)),
            matte: self.matte.as_ref().map(|m| m.resized(w, h)),
            moving: self.moving.as_ref().map(|m| m.resized(w, h)),
            faces: self.faces.iter().map(|(b, e)| ([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], [[e[0][0] * sx, e[0][1] * sy], [e[1][0] * sx, e[1][1] * sy]])).collect(),
            sources: self.sources.clone(),
        }
    }
}

/// What a render returns beside the picture: the stages' notes, the share of the frame the camera's move
/// invented (filled where the picture had nothing), and the fields after the camera (for the critic).
#[derive(Clone, Debug, Default)]
pub struct Rendered {
    pub image: RgbImage,
    pub notes: Vec<String>,
    pub invented: f32,
    pub aux: Aux,
}

/// The chain after the camera, before the lens: where the neural stages (the fill, the refine, the relight)
/// go in. [`stage`] makes it, the CLI runs the neural stages on its frame, [`finish`] runs the rest.
#[derive(Clone, Debug)]
pub struct Staged {
    pub frame: Frame,
    pub aux: Aux,
    /// The holes the camera's move opened (1 = invented), for the neural fill.
    pub holes: Option<Field>,
    pub invented: f32,
    pub notes: Vec<String>,
    pub seed: u32,
}

/// The size and the camera.
pub fn stage(img: &RgbImage, plan: &plan::PhotoPlan, aux: &Aux, seed: u64) -> anyhow::Result<Staged> {
    let mut notes = Vec::new();
    let seed32 = (seed as u32) ^ ((seed >> 32) as u32);
    let (w0, h0) = img.dimensions();
    let (w, h) = plan.output_size(w0, h0);
    let mut frame = Frame::from_srgb(img);
    if (w, h) != (w0, h0) {
        frame = frame.resized(w, h);
        notes.push(format!("size: {w0}×{h0} → {w}×{h} ({:.1} Mpx, the medium's)", (w as f32 * h as f32) / 1e6));
    }
    let mut aux = aux.resized(w, h, (w0, h0));
    let cam = camera::run(&frame, plan, &aux, &mut notes)?;
    aux.depth = cam.depth.or(aux.depth);
    aux.metric = cam.metric;
    aux.matte = cam.matte.or(aux.matte);
    aux.faces = cam.faces;
    Ok(Staged { frame: cam.frame, aux, holes: cam.holes, invented: cam.invented, notes, seed: seed32 })
}

/// The lens, the shutter, the emulsion, the process, the digital, the carrier, the age.
pub fn finish(st: Staged, plan: &plan::PhotoPlan) -> Rendered {
    let Staged { frame, aux, invented, mut notes, seed, .. } = st;
    let frame = lens::run(&frame, plan, &aux, seed, &mut notes);
    let frame = lens::shutter(&frame, plan, &aux, seed, &mut notes);
    let frame = sensor::run(&frame, plan, &aux, seed, &mut notes);
    let frame = process::run(&frame, plan, seed, &mut notes);
    let frame = sensor::digital(&frame, plan, &aux, seed, &mut notes);
    let frame = process::carrier(&frame, plan, seed, &mut notes);
    let frame = process::age(&frame, plan, seed, &mut notes);
    Rendered { image: frame.to_srgb(), notes, invented, aux }
}

/// Run the whole deterministic chain on `img` by `plan`, with the facts in `aux` and a seed for every
/// stochastic stage (the neural stages are the CLI's, between [`stage`] and [`finish`]).
pub fn render(img: &RgbImage, plan: &plan::PhotoPlan, aux: &Aux, seed: u64) -> anyhow::Result<Rendered> {
    Ok(finish(stage(img, plan, aux, seed)?, plan))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_round_trips_and_the_hash_noise_is_stable() {
        for c in [0u8, 1, 17, 128, 254, 255] {
            assert_eq!(linear_to_srgb(srgb_to_linear(c)), c, "{c}");
        }
        assert_eq!(hash01(3, 4, 5), hash01(3, 4, 5));
        assert_ne!(hash01(3, 4, 5), hash01(4, 3, 5));
        let n = value_noise(32, 32, 8.0, 1);
        assert!(n.v.iter().all(|v| (0.0..=1.0).contains(v)));
        let g: f32 = (0..10_000).map(|i| hash_gauss(i % 100, i / 100, 9)).sum::<f32>() / 10_000.0;
        assert!(g.abs() < 0.05, "mean {g}");
    }

    #[test]
    fn a_blur_keeps_the_mean_and_a_resize_keeps_the_picture() {
        let mut f = Frame::new(20, 20);
        for (i, p) in f.px.iter_mut().enumerate() {
            *p = [(i % 20) as f32 / 19.0, 0.5, 0.2];
        }
        let b = gaussian(&f, 1.5);
        let m0: f32 = f.px.iter().map(|p| p[0]).sum::<f32>() / 400.0;
        let m1: f32 = b.px.iter().map(|p| p[0]).sum::<f32>() / 400.0;
        assert!((m0 - m1).abs() < 0.01, "{m0} {m1}");
        let r = f.resized(10, 10).resized(20, 20);
        let err: f32 = f.px.iter().zip(&r.px).map(|(a, b)| (a[0] - b[0]).abs()).sum::<f32>() / 400.0;
        assert!(err < 0.08, "{err}");
    }
}
