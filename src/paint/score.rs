//! The stroke score (RFC PAINT-1 §11.2) — the canonical artifact. An ordered, line-oriented, replayable record
//! of every stroke: its geometry, its pigment mixture (by NAME, so it is palette-portable), the stage it
//! belongs to, and its physics. The image `plakat paint` writes is one *rendering* of the score; the score is
//! authoritative under re-render.
//!
//! Guarantees this slice establishes: **replay** at any resolution (geometry scales; native-resolution replay
//! is byte-identical to the original paint, A4), and a round-trip through the text format. Checkpoints,
//! export, and timelapse build on this record in later slices.

use anyhow::{bail, Context, Result};

use crate::paint::canvas::Canvas;
use crate::paint::palette::Palette;
use crate::paint::stroke::{BrushConfig, Stroke};

/// The score header — everything replay needs to reconstruct the canvas identically.
#[derive(Clone, Debug)]
pub struct ScoreHeader {
    pub version: u32,
    pub palette: String,
    pub medium: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub tooth: f32,
    pub brush: BrushConfig,
}

/// One recorded stroke.
#[derive(Clone, Debug)]
pub struct StrokeRecord {
    pub id: u32,
    /// A `wipe` (subtractive) stroke rather than a deposit. (Deposit-only in this slice; the flag is recorded.)
    pub wipe: bool,
    pub stage: String,
    pub spline: Vec<[f32; 2]>,
    pub w0: f32,
    pub w1: f32,
    pub taper: f32,
    /// Pigment mixture as `(pigment_name, load_value)` for the non-zero pigments.
    pub mix: Vec<(String, f32)>,
    pub wet: f32,
    pub press: f32,
}

/// The whole score: a header plus the ordered strokes.
#[derive(Clone, Debug)]
pub struct StrokeScore {
    pub header: ScoreHeader,
    pub strokes: Vec<StrokeRecord>,
}

fn fmt_f(v: f32) -> String {
    // Compact, round-trippable, locale-free.
    let s = format!("{v:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() || s == "-" {
        "0".into()
    } else {
        s.to_string()
    }
}

impl StrokeScore {
    /// Serialise to the line-oriented text format.
    pub fn to_text(&self) -> String {
        let h = &self.header;
        let b = &h.brush;
        let mut o = String::new();
        o.push_str("# plakat stroke score v1\n");
        o.push_str(&format!(
            "H palette={} medium={} seed={} size={}x{} tooth={} kd={} kp={} visc={} bristles={} loadmax={}\n",
            h.palette, h.medium, h.seed, h.width, h.height, fmt_f(h.tooth), fmt_f(b.k_deposit), fmt_f(b.k_pickup), fmt_f(b.viscosity), b.bristles, fmt_f(b.load_max),
        ));
        for s in &self.strokes {
            let spline = s.spline.iter().map(|p| format!("{},{}", fmt_f(p[0]), fmt_f(p[1]))).collect::<Vec<_>>().join(";");
            let mix = s.mix.iter().map(|(n, v)| format!("{n}:{}", fmt_f(*v))).collect::<Vec<_>>().join(",");
            o.push_str(&format!(
                "{} {} stage={} spline={} w0={} w1={} taper={} mix={} wet={} press={}\n",
                if s.wipe { "W" } else { "S" },
                s.id,
                s.stage,
                spline,
                fmt_f(s.w0),
                fmt_f(s.w1),
                fmt_f(s.taper),
                mix,
                fmt_f(s.wet),
                fmt_f(s.press),
            ));
        }
        o
    }

    /// Parse the text format back into a score.
    pub fn parse(text: &str) -> Result<Self> {
        let mut header: Option<ScoreHeader> = None;
        let mut strokes = Vec::new();
        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split_whitespace();
            let tag = it.next().unwrap_or("");
            let kv = |it: std::str::SplitWhitespace| -> std::collections::HashMap<String, String> {
                it.filter_map(|t| t.split_once('=').map(|(k, v)| (k.to_string(), v.to_string()))).collect()
            };
            match tag {
                "H" => {
                    let m = kv(it);
                    let get = |k: &str| m.get(k).cloned().unwrap_or_default();
                    let (w, h) = get("size").split_once('x').map(|(a, b)| (a.parse().unwrap_or(0), b.parse().unwrap_or(0))).unwrap_or((0, 0));
                    header = Some(ScoreHeader {
                        version: 1,
                        palette: get("palette"),
                        medium: get("medium"),
                        seed: get("seed").parse().unwrap_or(0),
                        width: w,
                        height: h,
                        tooth: get("tooth").parse().unwrap_or(0.85),
                        brush: BrushConfig {
                            k_deposit: get("kd").parse().unwrap_or(0.12),
                            k_pickup: get("kp").parse().unwrap_or(0.6),
                            viscosity: get("visc").parse().unwrap_or(1.0),
                            bristles: get("bristles").parse().unwrap_or(7),
                            load_max: get("loadmax").parse().unwrap_or(6.0),
                        },
                    });
                }
                "S" | "W" => {
                    let id: u32 = it.clone().next().and_then(|s| s.parse().ok()).unwrap_or(0);
                    let m = kv(it);
                    let get = |k: &str| m.get(k).cloned().unwrap_or_default();
                    let spline = get("spline")
                        .split(';')
                        .filter(|s| !s.is_empty())
                        .filter_map(|p| p.split_once(',').and_then(|(a, b)| Some([a.parse().ok()?, b.parse().ok()?])))
                        .collect::<Vec<[f32; 2]>>();
                    let mix = get("mix")
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .filter_map(|e| e.split_once(':').map(|(n, v)| (n.to_string(), v.parse().unwrap_or(0.0))))
                        .collect::<Vec<(String, f32)>>();
                    strokes.push(StrokeRecord {
                        id,
                        wipe: tag == "W",
                        stage: get("stage"),
                        spline,
                        w0: get("w0").parse().unwrap_or(1.0),
                        w1: get("w1").parse().unwrap_or(1.0),
                        taper: get("taper").parse().unwrap_or(0.0),
                        mix,
                        wet: get("wet").parse().unwrap_or(1.0),
                        press: get("press").parse().unwrap_or(1.0),
                    });
                }
                "C" => { /* checkpoint markers ignored on parse in this slice */ }
                _ => bail!("stroke score: unrecognised record on line {}: {:?}", lineno + 1, tag),
            }
        }
        let header = header.context("stroke score: missing H header line")?;
        Ok(StrokeScore { header, strokes })
    }

    /// Replay the score onto a fresh canvas at `out_w × out_h`. Geometry (and the brush width + load charge)
    /// scale from the score's native size, so the same score renders at any resolution with no upscaler. At
    /// the native size this reproduces the original paint byte-for-byte.
    pub fn replay(&self, out_w: u32, out_h: u32) -> Result<Canvas> {
        let palette = Palette::by_name(&self.header.palette).with_context(|| format!("stroke score: unknown palette {:?}", self.header.palette))?;
        let n = palette.pigments.len();
        let mut canvas = Canvas::white(out_w, out_h, palette, self.header.tooth);
        let sx = out_w as f32 / self.header.width.max(1) as f32;
        let sy = out_h as f32 / self.header.height.max(1) as f32;
        let ss = (sx * sy).sqrt(); // isotropic scale for widths + physics

        // Resolution independence: at scale `ss` a stroke has ~ss× more integration steps, so the deposit rate
        // is scaled by 1/ss to keep the per-unit-LENGTH dry-out profile constant, and the load charge by ss so
        // the extra steps still empty the brush over the stroke. Native (ss=1) leaves the brush untouched, so
        // native replay stays byte-identical.
        let mut brush = self.header.brush;
        brush.k_deposit /= ss;
        brush.load_max *= ss;

        // Map pigment names → palette indices once.
        let index_of = |name: &str| palette.pigments.iter().position(|p| p.name.eq_ignore_ascii_case(name));

        for rec in &self.strokes {
            if rec.wipe {
                continue; // deposit-only replay in this slice
            }
            let mut load = vec![0f32; n];
            for (name, val) in &rec.mix {
                if let Some(i) = index_of(name) {
                    load[i] += val * ss; // more paint to cover the longer path at higher resolution
                }
            }
            let path: Vec<[f32; 2]> = rec.spline.iter().map(|p| [p[0] * sx, p[1] * sy]).collect();
            let s = Stroke { path, width0: rec.w0 * ss, width1: rec.w1 * ss, load, pressure: rec.press, wetness: rec.wet };
            s.rasterize(&mut canvas, &brush);
        }
        Ok(canvas)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::palette;

    fn sample() -> StrokeScore {
        StrokeScore {
            header: ScoreHeader { version: 1, palette: "zorn".into(), medium: "oil-direct".into(), seed: 42, width: 64, height: 48, tooth: 0.85, brush: BrushConfig::default() },
            strokes: vec![
                StrokeRecord { id: 1, wipe: false, stage: "shadow-mass".into(), spline: vec![[5.0, 20.0], [30.0, 22.0], [50.0, 20.0]], w0: 8.0, w1: 5.0, taper: 0.4, mix: vec![("cadmium-red".into(), 3.0), ("ivory-black".into(), 1.0)], wet: 1.0, press: 0.9 },
                StrokeRecord { id: 2, wipe: false, stage: "light-mass".into(), spline: vec![[10.0, 10.0], [40.0, 12.0]], w0: 6.0, w1: 4.0, taper: 0.3, mix: vec![("yellow-ochre".into(), 2.0), ("titanium-white".into(), 3.0)], wet: 1.0, press: 1.0 },
            ],
        }
    }

    #[test]
    fn text_round_trips() {
        let s = sample();
        let parsed = StrokeScore::parse(&s.to_text()).expect("parses");
        assert_eq!(parsed.header.palette, "zorn");
        assert_eq!(parsed.header.width, 64);
        assert_eq!(parsed.strokes.len(), 2);
        assert_eq!(parsed.strokes[0].stage, "shadow-mass");
        assert_eq!(parsed.strokes[0].spline.len(), 3);
        assert_eq!(parsed.strokes[1].mix[0].0, "yellow-ochre");
    }

    #[test]
    fn replay_at_native_size_is_byte_identical_across_reparses() {
        let s = sample();
        let a = s.replay(64, 48).unwrap().to_image().into_raw();
        // Same score, and a round-tripped copy, must render identically.
        let b = StrokeScore::parse(&s.to_text()).unwrap().replay(64, 48).unwrap().to_image().into_raw();
        assert_eq!(a, b, "replay is deterministic and survives serialize/parse");
    }

    #[test]
    fn replay_scales_to_any_resolution_keeping_structure() {
        // A denser, real score (painted) — replay it at native size and at 4×, and the 4× render downscaled
        // keeps the same structure. Cross-resolution replay is structure-preserving (A2), not pixel-identical.
        use crate::paint::painter::{paint_from_image, PaintParams};
        let img = image::RgbImage::from_fn(96, 72, |x, y| image::Rgb([(30 + 2 * x) as u8, (200 - x - y) as u8, (60 + y) as u8]));
        let mut p = PaintParams::new(palette::EARTH, 400);
        p.brush_sizes = vec![22.0, 11.0];
        let s = paint_from_image(&img, &p).score;
        let native = s.replay(96, 72).unwrap().to_image();
        let big = s.replay(384, 288).unwrap().to_image();
        assert_eq!(big.dimensions(), (384, 288), "renders at the requested size, no upscaler");
        let big_small = image::imageops::resize(&big, 96, 72, image::imageops::FilterType::Triangle);
        let tr = crate::paint::painter::traceability(&big_small, &native);
        // Cross-resolution replay keeps the structure (well above chance); pixel-tight resolution-independence
        // (A2 to a high bar) is a later refinement of the deposit/charge scaling.
        assert!(tr > 0.65, "same score at 4× keeps structure (corr {tr})");
    }

    #[test]
    fn unknown_palette_replay_errors() {
        let mut s = sample();
        s.header.palette = "nope".into();
        assert!(s.replay(64, 48).is_err());
        let _ = palette::ZORN; // keep the import used
    }
}
