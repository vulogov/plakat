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
use crate::paint::pigment::Pigment;
use crate::paint::stroke::{BrushConfig, Stroke};

/// Reconstruct a `Palette` from recorded pigment definitions (`name`, sRGB). The `Palette`/`Pigment` types hold
/// `&'static` data, so the reconstructed set is leaked — bounded by the number of palettes a single CLI run
/// replays, which is small.
fn palette_from_defs(name: &str, defs: &[(String, [u8; 3])]) -> Palette {
    let pigs: Vec<Pigment> = defs
        .iter()
        .map(|(n, rgb)| Pigment { name: Box::leak(n.clone().into_boxed_str()), masstone: *rgb })
        .collect();
    Palette { name: Box::leak(name.to_string().into_boxed_str()), pigments: Box::leak(pigs.into_boxed_slice()) }
}

/// The score header — everything replay needs to reconstruct the canvas identically.
#[derive(Clone, Debug)]
pub struct ScoreHeader {
    pub version: u32,
    pub palette: String,
    /// The palette's pigment definitions (`name`, sRGB masstone), so a DERIVED palette (`--palette image`) — or
    /// any palette — replays without the binary knowing it. Empty = resolve `palette` by name (old scores).
    pub pigments: Vec<(String, [u8; 3])>,
    pub medium: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub tooth: f32,
    /// The toned ground (imprimatura), if any — replay must prime the canvas the same way the paint did.
    pub ground: Option<crate::paint::color::Srgb>,
    pub brush: BrushConfig,
    /// Wet-into-wet BLEED strength applied after painting (0 = dry media) — replay reproduces it.
    pub bleed: f32,
    /// INTER-PASS DRYING (0..1): how much the canvas dried between passes during painting. Recorded so a replay
    /// (at any resolution) dries at the same stage boundaries and reproduces the delivered image. 0 = never dried.
    pub dry: f32,
    /// BODY / opacity of the paint film (1 = opaque; lower = transparent) — replay must match.
    pub opacity: f32,
    /// IMPASTO relight strength at output (0 = flat) — the textured oil/knife look, replayed from height.
    pub impasto: f32,
    /// Material finish (§8.5) — the paint's own behaviour, replayed at output.
    pub chroma: f32,
    pub dry_shift: f32,
    pub granulate: f32,
    pub sheen: f32,
    /// EDGE POOLING — darken pigment at wash boundaries (watercolour edge-bloom), applied at output.
    pub edge_pool: f32,
    /// PAPER EDGE — fade to a deckled paper border at output (torn-paper vignette).
    pub paper_edge: f32,
    /// FINISH GRADE (painting-safe tonal grade, applied at output): contrast, warmth (WB), clarity (local contrast).
    pub contrast: f32,
    pub warmth: f32,
    pub clarity: f32,
    /// LIFT — wipe removability (staining), applied during painting.
    pub lift: f32,
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
    /// This stroke's brush character (RFC brush vocabulary): bristle streak and cross-section roundness. Lets
    /// each stroke carry its OWN brush (a flat for the sky, a round for a face) and replay exactly.
    pub streak: f32,
    pub round: f32,
    /// This stroke's own PICKUP (dirty-brush drag), when it differs from the score's base brush — a DETAIL accent
    /// is laid nearly clean so it defines a feature instead of smearing the mass beneath. `None` = the header's
    /// `k_pickup` (old scores, and every non-detail stroke), so replay reproduces the paint exactly.
    pub pickup: Option<f32>,
}

impl ScoreHeader {
    /// The medium's output material finish (§8.5), for `Canvas::to_image_finished`.
    pub fn finish(&self) -> crate::paint::canvas::Finish {
        crate::paint::canvas::Finish {
            impasto: self.impasto,
            chroma: self.chroma,
            dry_shift: self.dry_shift,
            granulate: self.granulate,
            sheen: self.sheen,
            edge_pool: self.edge_pool,
            paper_edge: self.paper_edge,
            contrast: self.contrast,
            warmth: self.warmth,
            clarity: self.clarity,
            seed: self.seed,
        }
    }
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
        let ground = h.ground.map(|g| format!(" ground={},{},{}", g[0], g[1], g[2])).unwrap_or_default();
        o.push_str(&format!(
            "H palette={} medium={} seed={} size={}x{} tooth={} kd={} kp={} visc={} bristles={} loadmax={} streak={} round={} bleed={} dry={} opacity={} impasto={} chroma={} dryshift={} granulate={} sheen={} edgepool={} paperedge={} contrast={} warmth={} clarity={} lift={}{}\n",
            h.palette, h.medium, h.seed, h.width, h.height, fmt_f(h.tooth), fmt_f(b.k_deposit), fmt_f(b.k_pickup), fmt_f(b.viscosity), b.bristles, fmt_f(b.load_max), fmt_f(b.streak), fmt_f(b.round), fmt_f(h.bleed), fmt_f(h.dry), fmt_f(h.opacity), fmt_f(h.impasto), fmt_f(h.chroma), fmt_f(h.dry_shift), fmt_f(h.granulate), fmt_f(h.sheen), fmt_f(h.edge_pool), fmt_f(h.paper_edge), fmt_f(h.contrast), fmt_f(h.warmth), fmt_f(h.clarity), fmt_f(h.lift), ground,
        ));
        // Pigment definitions (self-contained palette) — so a derived/any palette replays without the binary.
        for (name, rgb) in &h.pigments {
            o.push_str(&format!("P {} {} {} {}\n", name, rgb[0], rgb[1], rgb[2]));
        }
        for s in &self.strokes {
            let spline = s.spline.iter().map(|p| format!("{},{}", fmt_f(p[0]), fmt_f(p[1]))).collect::<Vec<_>>().join(";");
            let mix = s.mix.iter().map(|(n, v)| format!("{n}:{}", fmt_f(*v))).collect::<Vec<_>>().join(",");
            o.push_str(&format!(
                "{} {} stage={} spline={} w0={} w1={} taper={} mix={} wet={} press={} streak={} round={}{}\n",
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
                fmt_f(s.streak),
                fmt_f(s.round),
                s.pickup.map(|v| format!(" pickup={}", fmt_f(v))).unwrap_or_default(),
            ));
        }
        o
    }

    /// Parse the text format back into a score.
    pub fn parse(text: &str) -> Result<Self> {
        let mut header: Option<ScoreHeader> = None;
        let mut strokes = Vec::new();
        let mut pigments: Vec<(String, [u8; 3])> = Vec::new();
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
                        pigments: Vec::new(),
                        medium: get("medium"),
                        seed: get("seed").parse().unwrap_or(0),
                        width: w,
                        height: h,
                        tooth: get("tooth").parse().unwrap_or(0.85),
                        ground: m.get("ground").and_then(|g| {
                            let v: Vec<u8> = g.split(',').filter_map(|x| x.parse::<u8>().ok()).collect();
                            (v.len() == 3).then_some([v[0], v[1], v[2]])
                        }),
                        bleed: get("bleed").parse().unwrap_or(0.0),
                        dry: get("dry").parse().unwrap_or(0.0),
                        opacity: get("opacity").parse().unwrap_or(1.0),
                        impasto: get("impasto").parse().unwrap_or(0.0),
                        chroma: get("chroma").parse().unwrap_or(1.0),
                        dry_shift: get("dryshift").parse().unwrap_or(0.0),
                        granulate: get("granulate").parse().unwrap_or(0.0),
                        sheen: get("sheen").parse().unwrap_or(0.0),
                        edge_pool: get("edgepool").parse().unwrap_or(0.0),
                        paper_edge: get("paperedge").parse().unwrap_or(0.0),
                        contrast: get("contrast").parse().unwrap_or(1.0),
                        warmth: get("warmth").parse().unwrap_or(0.0),
                        clarity: get("clarity").parse().unwrap_or(0.0),
                        lift: get("lift").parse().unwrap_or(1.0),
                        brush: BrushConfig {
                            k_deposit: get("kd").parse().unwrap_or(0.12),
                            k_pickup: get("kp").parse().unwrap_or(0.6),
                            viscosity: get("visc").parse().unwrap_or(1.0),
                            bristles: get("bristles").parse().unwrap_or(7),
                            load_max: get("loadmax").parse().unwrap_or(6.0),
                            streak: get("streak").parse().unwrap_or(0.6),
                            round: get("round").parse().unwrap_or(0.7),
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
                        streak: get("streak").parse().unwrap_or(0.6),
                        round: get("round").parse().unwrap_or(0.7),
                        pickup: get("pickup").parse().ok(),
                    });
                }
                "P" => {
                    // Pigment definition: `P <name> <r> <g> <b>`.
                    let name = it.next().unwrap_or("").to_string();
                    let mut rgb = [0u8; 3];
                    for c in rgb.iter_mut() {
                        *c = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                    }
                    if !name.is_empty() {
                        pigments.push((name, rgb));
                    }
                }
                "C" => { /* checkpoint markers ignored on parse in this slice */ }
                _ => bail!("stroke score: unrecognised record on line {}: {:?}", lineno + 1, tag),
            }
        }
        let mut header = header.context("stroke score: missing H header line")?;
        header.pigments = pigments;
        Ok(StrokeScore { header, strokes })
    }

    /// The distinct stage names in first-seen order — the basis for stage separations / a stages contact sheet.
    pub fn stages(&self) -> Vec<String> {
        let mut seen = Vec::new();
        for s in &self.strokes {
            if !seen.iter().any(|x| x == &s.stage) {
                seen.push(s.stage.clone());
            }
        }
        seen
    }

    /// Replay only the strokes matching `keep`, in order, onto a fresh canvas at `out_w × out_h`. The building
    /// block for separations (one stage), a cumulative stages sheet, `--until N`, and frame emission.
    pub fn replay_filtered(&self, out_w: u32, out_h: u32, keep: impl Fn(&StrokeRecord) -> bool) -> Result<Canvas> {
        let (palette, brush, sx, sy, ss) = self.replay_setup(out_w, out_h)?;
        let n = palette.pigments.len();
        let mut canvas = match self.header.ground {
            Some(g) => Canvas::toned(out_w, out_h, palette, g, self.header.tooth),
            None => Canvas::white(out_w, out_h, palette, self.header.tooth),
        }
        .with_opacity(self.header.opacity);
        let index_of = |name: &str| palette.pigments.iter().position(|p| p.name.eq_ignore_ascii_case(name));
        let full = |k: &dyn Fn(&StrokeRecord) -> bool| self.strokes.iter().all(|r| k(r));
        // Inter-pass drying: the paint dried the canvas at each pass boundary; a stage change in the recorded
        // stream marks one. Reproduce it here so replay matches the delivered image (no-op for old scores, dry=0).
        let dry_keep = 1.0 - self.header.dry;
        let mut last_stage: Option<&str> = None;
        for rec in &self.strokes {
            if !keep(rec) {
                continue;
            }
            if self.header.dry > 0.0 {
                if let Some(prev) = last_stage {
                    if prev != rec.stage {
                        canvas.dry(dry_keep);
                    }
                }
                last_stage = Some(&rec.stage);
            }
            let path: Vec<[f32; 2]> = rec.spline.iter().map(|p| [p[0] * sx, p[1] * sy]).collect();
            if rec.wipe {
                // A subtractive WIPE stroke: scrape pigment back (strength recorded in `wet`). LIFT scales how
                // much comes off — oil lifts freely, a staining watercolour barely at all.
                let s = Stroke { path, width0: rec.w0 * ss, width1: rec.w1 * ss, load: Vec::new(), pressure: rec.press, wetness: rec.wet };
                s.wipe(&mut canvas, &brush, rec.wet * self.header.lift);
                continue;
            }
            let mut load = vec![0f32; n];
            for (name, val) in &rec.mix {
                if let Some(i) = index_of(name) {
                    load[i] += val * ss;
                }
            }
            let s = Stroke { path, width0: rec.w0 * ss, width1: rec.w1 * ss, load, pressure: rec.press, wetness: rec.wet };
            // This stroke's own brush character (a flat, a round, …) over the score's base brush physics.
            let sb = BrushConfig { streak: rec.streak, round: rec.round, k_pickup: rec.pickup.unwrap_or(brush.k_pickup), ..brush };
            s.rasterize(&mut canvas, &sb);
        }
        // Wet-into-wet BLEED (the final fusion the paint applied) — only on a FULL replay; a truncated/filtered
        // replay is an intermediate state, before the fusion.
        if self.header.bleed > 0.0 && full(&keep) {
            canvas.bleed(self.header.bleed);
        }
        Ok(canvas)
    }

    /// Replay the whole score in order, emitting a canvas frame every `every` strokes (plus the final frame).
    /// A single O(n) forward pass, so a long timelapse is linear rather than quadratic.
    pub fn replay_frames(&self, out_w: u32, out_h: u32, every: usize) -> Result<Vec<Canvas>> {
        let (palette, brush, sx, sy, ss) = self.replay_setup(out_w, out_h)?;
        let n = palette.pigments.len();
        let mut canvas = match self.header.ground {
            Some(g) => Canvas::toned(out_w, out_h, palette, g, self.header.tooth),
            None => Canvas::white(out_w, out_h, palette, self.header.tooth),
        }
        .with_opacity(self.header.opacity);
        let index_of = |name: &str| palette.pigments.iter().position(|p| p.name.eq_ignore_ascii_case(name));
        let every = every.max(1);
        let mut frames = Vec::new();
        let mut laid = 0usize;
        let dry_keep = 1.0 - self.header.dry;
        let mut last_stage: Option<&str> = None;
        for rec in &self.strokes {
            if self.header.dry > 0.0 {
                if let Some(prev) = last_stage {
                    if prev != rec.stage {
                        canvas.dry(dry_keep);
                    }
                }
                last_stage = Some(&rec.stage);
            }
            if !rec.wipe {
                let mut load = vec![0f32; n];
                for (name, val) in &rec.mix {
                    if let Some(i) = index_of(name) {
                        load[i] += val * ss;
                    }
                }
                let path: Vec<[f32; 2]> = rec.spline.iter().map(|p| [p[0] * sx, p[1] * sy]).collect();
                let s = Stroke { path, width0: rec.w0 * ss, width1: rec.w1 * ss, load, pressure: rec.press, wetness: rec.wet };
                let sb = BrushConfig { streak: rec.streak, round: rec.round, k_pickup: rec.pickup.unwrap_or(brush.k_pickup), ..brush };
                s.rasterize(&mut canvas, &sb);
                laid += 1;
                if laid % every == 0 {
                    frames.push(canvas.snapshot());
                }
            }
        }
        if self.header.bleed > 0.0 {
            canvas.bleed(self.header.bleed); // the final wet fusion, as on the finished painting
        }
        frames.push(canvas); // always end on the finished painting
        Ok(frames)
    }

    /// Resolve the palette, scaled brush, and scale factors shared by every replay path.
    fn replay_setup(&self, out_w: u32, out_h: u32) -> Result<(Palette, BrushConfig, f32, f32, f32)> {
        // Prefer the score's own pigment definitions (self-contained — works for a derived `image` palette or any
        // palette the running binary doesn't know); fall back to a built-in by name for old, pigment-less scores.
        let palette = if !self.header.pigments.is_empty() {
            palette_from_defs(&self.header.palette, &self.header.pigments)
        } else {
            Palette::by_name(&self.header.palette).with_context(|| format!("stroke score: unknown palette {:?}", self.header.palette))?
        };
        let sx = out_w as f32 / self.header.width.max(1) as f32;
        let sy = out_h as f32 / self.header.height.max(1) as f32;
        let ss = (sx * sy).sqrt();
        let mut brush = self.header.brush;
        brush.k_deposit /= ss;
        brush.load_max *= ss;
        Ok((palette, brush, sx, sy, ss))
    }

    /// Replay the score onto a fresh canvas at `out_w × out_h`. Geometry (and the brush width + load charge)
    /// scale from the score's native size, so the same score renders at any resolution with no upscaler. At
    /// the native size this reproduces the original paint byte-for-byte.
    pub fn replay(&self, out_w: u32, out_h: u32) -> Result<Canvas> {
        self.replay_filtered(out_w, out_h, |_| true)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::palette;

    fn sample() -> StrokeScore {
        StrokeScore {
            header: ScoreHeader { version: 1, palette: "zorn".into(), pigments: Vec::new(), medium: "oil-direct".into(), seed: 42, width: 64, height: 48, tooth: 0.85, ground: None, brush: BrushConfig::default(), bleed: 0.0, dry: 0.0, opacity: 1.0, impasto: 0.0, chroma: 1.0, dry_shift: 0.0, granulate: 0.0, sheen: 0.0, edge_pool: 0.0, paper_edge: 0.0, contrast: 1.0, warmth: 0.0, clarity: 0.0, lift: 1.0 },
            strokes: vec![
                StrokeRecord { id: 1, wipe: false, stage: "shadow-mass".into(), spline: vec![[5.0, 20.0], [30.0, 22.0], [50.0, 20.0]], w0: 8.0, w1: 5.0, taper: 0.4, mix: vec![("cadmium-red".into(), 3.0), ("ivory-black".into(), 1.0)], wet: 1.0, press: 0.9, streak: 0.6, round: 0.7, pickup: None },
                StrokeRecord { id: 2, wipe: false, stage: "light-mass".into(), spline: vec![[10.0, 10.0], [40.0, 12.0]], w0: 6.0, w1: 4.0, taper: 0.3, mix: vec![("yellow-ochre".into(), 2.0), ("titanium-white".into(), 3.0)], wet: 1.0, press: 1.0, streak: 0.6, round: 0.7, pickup: None },
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
    fn stages_and_filtered_replay() {
        let s = sample();
        assert_eq!(s.stages(), vec!["shadow-mass".to_string(), "light-mass".to_string()], "distinct stages in order");
        // Replaying only the light-mass stage differs from the full painting (the shadow stroke is absent).
        let full = s.replay(64, 48).unwrap().to_image().into_raw();
        let one = s.replay_filtered(64, 48, |r| r.stage == "light-mass").unwrap().to_image().into_raw();
        assert_ne!(full, one, "a single-stage separation is not the whole painting");
    }

    #[test]
    fn replay_frames_emits_progress_and_ends_finished() {
        let s = sample();
        let frames = s.replay_frames(64, 48, 1).unwrap();
        // 2 strokes, every=1 → a frame after each, plus the final = 3 (with a duplicate final, fine for a reel).
        assert!(frames.len() >= 2, "one frame per stroke + the finish");
        let last = frames.last().unwrap().to_image().into_raw();
        let full = s.replay(64, 48).unwrap().to_image().into_raw();
        assert_eq!(last, full, "the final frame is the finished painting");
    }

    #[test]
    fn replay_applies_a_wipe_record() {
        // A score that lays a dark stroke then WIPES part of it — the wiped band is lighter than without it.
        let base = ScoreHeader { version: 1, palette: "zorn".into(), pigments: Vec::new(), medium: "oil-direct".into(), seed: 1, width: 40, height: 20, tooth: 0.9, ground: None, brush: BrushConfig::default(), bleed: 0.0, dry: 0.0, opacity: 1.0, impasto: 0.0, chroma: 1.0, dry_shift: 0.0, granulate: 0.0, sheen: 0.0, edge_pool: 0.0, paper_edge: 0.0, contrast: 1.0, warmth: 0.0, clarity: 0.0, lift: 1.0 };
        let stroke = StrokeRecord { id: 1, wipe: false, stage: "mass".into(), spline: vec![[2.0, 10.0], [38.0, 10.0]], w0: 10.0, w1: 10.0, taper: 0.0, mix: vec![("ivory-black".into(), 5.0)], wet: 1.0, press: 1.0, streak: 0.6, round: 0.7, pickup: None };
        let no_wipe = StrokeScore { header: base.clone(), strokes: vec![stroke.clone()] };
        let wipe = StrokeRecord { id: 2, wipe: true, stage: "scrape".into(), spline: vec![[18.0, 4.0], [18.0, 16.0]], w0: 8.0, w1: 8.0, taper: 0.0, mix: vec![], wet: 0.9, press: 1.0, streak: 0.6, round: 0.7, pickup: None };
        let with_wipe = StrokeScore { header: base, strokes: vec![stroke, wipe] };
        let a = no_wipe.replay(40, 20).unwrap();
        let b = with_wipe.replay(40, 20).unwrap();
        use crate::paint::color::{delta_e76, srgb_to_lab};
        let lifted = srgb_to_lab(b.color_at(18, 10)).l - srgb_to_lab(a.color_at(18, 10)).l;
        // The wipe thins the paint film so the ground shows through and the band lightens (film-build opacity —
        // subtler than injecting white, but the direction is the point).
        assert!(lifted > 1.0, "the wipe lightened the scraped band (ΔL {lifted})");
        // Away from the wipe, the two are identical.
        assert!(delta_e76(srgb_to_lab(a.color_at(5, 10)), srgb_to_lab(b.color_at(5, 10))) < 1.0, "untouched elsewhere");
    }

    #[test]
    fn unknown_palette_replay_errors() {
        let mut s = sample();
        s.header.palette = "nope".into();
        assert!(s.replay(64, 48).is_err());
        let _ = palette::ZORN; // keep the import used
    }
}
