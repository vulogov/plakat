//! `plakat make-photo` — a picture passed through the photographic chain of a period and a style
//! (`--media`), from the daguerreotype to the computational phone; a render made a photograph
//! (`--media photo`); the camera moved and the lens changed (`--focal`, `--dolly`, …); the director's plan
//! (`--make-plan`) and the critic's loop (`--make-plan-iterate`); the camera statistics of any picture
//! (`make-photo measure`). See `crate::photo`.

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use console::style;
use std::path::{Path, PathBuf};

use crate::paint::director::SceneFacts;
use crate::photo::{self, director, measure, media, plan::PhotoPlan, Aux, Field};

#[derive(Args, Debug, Clone)]
#[command(args_conflicts_with_subcommands = true)]
pub struct MakePhotoArgs {
    #[command(subcommand)]
    pub cmd: Option<PhotoCmd>,
    #[command(flatten)]
    pub from: FromArgs,
}

#[derive(Subcommand, Debug, Clone)]
pub enum PhotoCmd {
    /// The catalogue of media, oldest first.
    List,
    /// The plan's keys with their meaning, in the chain's order.
    Keys,
    /// The camera statistics of pictures: the noise and its structure, the sharpness over the frame, the
    /// highlights, the vignette, the fringes, the JPEG grid — what a photograph has and a render has not.
    Measure(MeasureArgs),
    /// The director's plan alone (`--media`), or the critic's verdict on a rendered picture (`--verdict`).
    Plan(PlanArgs),
}

/// The main command: a picture as a medium.
#[derive(Args, Debug, Clone, Default)]
pub struct FromArgs {
    /// The picture.
    pub input: Option<PathBuf>,
    /// The medium (`make-photo list`); "photo" is the modern camera that makes a render a photograph.
    #[arg(long, visible_alias = "medium", default_value = "photo")]
    pub media: String,
    /// The output (default `<input>.<media>.png`).
    #[arg(short = 'o', long)]
    pub out: Option<PathBuf>,
    /// A plan file (hjson) in place of the medium's recipe (`--set` and the flags still override it).
    #[arg(long, conflicts_with = "make_plan")]
    pub plan: Option<PathBuf>,
    /// The director composes the plan for this picture (its facts read by the vision models through the rules)
    /// and writes it as `<out>.plan.hjson` or to PATH; the picture is then made by it.
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "")]
    pub make_plan: Option<String>,
    /// The critic's loop: make, measure, amend the plan, make again — N more times at most, stopping when the
    /// critic moves nothing; the intermediate pictures kept as `<stem>.i<k>.<ext>`.
    #[arg(long, value_name = "N", default_value_t = 0, requires = "make_plan")]
    pub make_plan_iterate: u32,
    /// Any plan key: `--set grain=0.4 --set tone=sepia`.
    #[arg(long, value_name = "KEY=VALUE", action = clap::ArgAction::Append)]
    pub set: Vec<String>,

    // ---- the camera ----
    /// The lens to re-photograph with (35 mm-equivalent mm); with `--keep-subject` the camera dollies so the
    /// subject keeps its size.
    #[arg(long, help_heading = "The camera")]
    pub focal: Option<f32>,
    /// The lens the picture was taken with (35 mm-equivalent mm); default by the genre.
    #[arg(long, help_heading = "The camera")]
    pub source_focal: Option<f32>,
    /// The distance to the subject in metres (default from a face's eyes, else by the genre).
    #[arg(long, help_heading = "The camera")]
    pub distance: Option<f32>,
    /// The camera moved toward the subject (+) or back (−), metres.
    #[arg(long, help_heading = "The camera", allow_hyphen_values = true)]
    pub dolly: Option<f32>,
    /// The camera moved right (+) or left (−), metres.
    #[arg(long, help_heading = "The camera", allow_hyphen_values = true)]
    pub truck: Option<f32>,
    /// The camera moved up (+) or down (−), metres.
    #[arg(long, help_heading = "The camera", allow_hyphen_values = true)]
    pub pedestal: Option<f32>,
    /// The camera turned right (+) or left (−), degrees.
    #[arg(long, help_heading = "The camera", allow_hyphen_values = true)]
    pub pan: Option<f32>,
    /// The camera tilted up (+) or down (−), degrees.
    #[arg(long, help_heading = "The camera", allow_hyphen_values = true)]
    pub tilt: Option<f32>,
    /// The camera rolled clockwise (+), degrees.
    #[arg(long, help_heading = "The camera", allow_hyphen_values = true)]
    pub roll: Option<f32>,
    /// The lens shifted (rise + / fall −), a fraction of the frame's height.
    #[arg(long, help_heading = "The camera", allow_hyphen_values = true)]
    pub shift: Option<f32>,
    /// Tie the lens and the dolly by the subject's size (the 24→85 portrait).
    #[arg(long, help_heading = "The camera")]
    pub keep_subject: bool,
    /// Keep the picture's own perspective: no re-lensing by the medium's recipe.
    #[arg(long, help_heading = "The camera", conflicts_with = "keep_subject")]
    pub keep_perspective: bool,
    /// How the ground a move opens is filled: propagate | neural | none.
    #[arg(long, help_heading = "The camera")]
    pub fill: Option<String>,
    /// The largest share of the frame a move may invent (the critic scales a move down to it).
    #[arg(long, help_heading = "The camera")]
    pub fill_cap: Option<f32>,

    // ---- the lens and the exposure ----
    /// The f-number (0 = everything sharp).
    #[arg(long, help_heading = "The lens and the exposure")]
    pub aperture: Option<f32>,
    /// The focus: auto | eyes | subject | near | far | centre | x,y.
    #[arg(long, help_heading = "The lens and the exposure")]
    pub focus: Option<String>,
    /// The exposure correction in stops.
    #[arg(long, help_heading = "The lens and the exposure", allow_hyphen_values = true)]
    pub exposure: Option<f32>,
    /// The exposure time in seconds.
    #[arg(long, help_heading = "The lens and the exposure")]
    pub exposure_time: Option<f32>,
    /// The film speed.
    #[arg(long, help_heading = "The lens and the exposure")]
    pub iso: Option<f32>,
    /// The format: 8x10 | 4x5 | 645 | 6x6 | 35mm | ff | aps-c | 110 | phone | plate.
    #[arg(long, help_heading = "The lens and the exposure")]
    pub format: Option<String>,

    // ---- the carrier and the age ----
    /// The age's master dial (0..1).
    #[arg(long, help_heading = "The carrier and the age")]
    pub age: Option<f32>,
    /// The frame or the border as a share of the short side (0 = none).
    #[arg(long, help_heading = "The carrier and the age")]
    pub frame: Option<f32>,
    /// The picture laterally reversed, as a direct positive is.
    #[arg(long, help_heading = "The carrier and the age")]
    pub mirror: bool,
    /// No lateral reversal even where the medium's recipe has it.
    #[arg(long, help_heading = "The carrier and the age", conflicts_with = "mirror")]
    pub no_mirror: bool,

    // ---- the neural stages ----
    /// The model-backed stages to run, each disclosed: fill (the opened ground inpainted), refine (a whole-frame
    /// img2img at the plan's `refine` strength with the medium's prompt, the faces protected), relight (IC-Light
    /// by the plan's `relight` preset).
    #[arg(long, help_heading = "The neural stages", value_delimiter = ',', value_name = "STAGE")]
    pub neural: Vec<String>,
    /// The diffusion model alias for the fill and the refine.
    #[arg(long, help_heading = "The neural stages", default_value = "sdxl", value_name = "ALIAS")]
    pub neural_model: String,
    /// The refine's strength (sets the plan's `refine`): 0.3 refines a photograph (the faces kept), 0.6–0.75
    /// converts a drawing or a painting into a photograph.
    #[arg(long, help_heading = "The neural stages", value_name = "N")]
    pub refine: Option<f32>,
    /// A ControlNet holding the structure through the refine (`depth:strength=0.5`, `canny:strength=0.4`);
    /// repeatable; `none` for no control.
    #[arg(long, help_heading = "The neural stages", value_name = "SPEC", default_value = "depth:strength=0.5", action = clap::ArgAction::Append)]
    pub neural_control: Vec<String>,
    /// The relight preset (sets the plan's `relight`): top, key-left, key-right, rim, softbox, golden-hour,
    /// sunset, moonlight, candlelight, overcast, neon.
    #[arg(long, help_heading = "The neural stages", value_name = "PRESET")]
    pub relight: Option<String>,

    /// The seed for the grain, the dust and the scratches (default 0).
    #[arg(long)]
    pub seed: Option<u64>,
    /// Print the camera statistics of the source and the result.
    #[arg(long)]
    pub report: bool,
    /// Read the pixels only (no vision models: no depth, no faces, no subject, no genre).
    #[arg(long)]
    pub pixels_only: bool,
}

#[derive(Args, Debug, Clone)]
pub struct MeasureArgs {
    /// The pictures (one: its readings; several: a table).
    #[arg(required = true, num_args = 1..)]
    pub inputs: Vec<PathBuf>,
    /// Write the table (markdown) here.
    #[arg(short = 'o', long)]
    pub out: Option<PathBuf>,
    /// Read the depth and the subject too (the vision models), for the sharpness against them.
    #[arg(long)]
    pub models: bool,
}

#[derive(Args, Debug, Clone)]
pub struct PlanArgs {
    /// The picture.
    pub input: PathBuf,
    /// The medium.
    #[arg(long, visible_alias = "medium", default_value = "photo")]
    pub media: String,
    /// Write the plan here (default `<input>.<media>.plan.hjson`; with `--verdict`, the plan is written back).
    #[arg(short = 'o', long)]
    pub out: Option<PathBuf>,
    /// The critic's verdict on this rendered picture, appended to `--plan`.
    #[arg(long, value_name = "PICTURE")]
    pub verdict: Option<PathBuf>,
    /// The plan the picture was made by (with `--verdict`).
    #[arg(long, value_name = "PLAN", requires = "verdict")]
    pub plan: Option<PathBuf>,
    /// Any plan key: `--set grain=0.4`.
    #[arg(long, value_name = "KEY=VALUE", action = clap::ArgAction::Append)]
    pub set: Vec<String>,
    /// Read the pixels only.
    #[arg(long)]
    pub pixels_only: bool,
}

pub async fn run(a: MakePhotoArgs) -> Result<()> {
    match a.cmd {
        Some(PhotoCmd::List) => {
            for l in media::list() {
                println!("{l}");
            }
            Ok(())
        }
        Some(PhotoCmd::Keys) => {
            let mut section = "";
            for (k, doc) in PhotoPlan::KEYS {
                let s = photo::plan::section_of(k);
                if s != section {
                    section = s;
                    println!("\n{}", style(s).bold());
                }
                println!("  {:<14} {}", style(k).cyan(), doc.trim().replace('\n', " "));
            }
            Ok(())
        }
        Some(PhotoCmd::Measure(m)) => run_measure(&m).await,
        Some(PhotoCmd::Plan(p)) => run_plan(&p).await,
        None => run_from(&a.from).await,
    }
}

/// The command line's keys as (key, value) pairs, in the plan's text form.
fn overrides(a: &FromArgs) -> Result<Vec<(String, String)>> {
    let mut o: Vec<(String, String)> = Vec::new();
    let mut num = |k: &str, v: Option<f32>| {
        if let Some(v) = v {
            o.push((k.into(), <f32 as photo::plan::Dial>::to_text(&v)));
        }
    };
    num("focal", a.focal);
    num("source_focal", a.source_focal);
    num("distance", a.distance);
    num("dolly", a.dolly);
    num("truck", a.truck);
    num("pedestal", a.pedestal);
    num("pan", a.pan);
    num("tilt", a.tilt);
    num("roll", a.roll);
    num("shift", a.shift);
    num("fill_cap", a.fill_cap);
    num("aperture", a.aperture);
    num("exposure", a.exposure);
    num("exposure_time", a.exposure_time);
    num("iso", a.iso);
    num("age", a.age);
    num("frame", a.frame);
    num("refine", a.refine);
    if a.keep_subject {
        o.push(("keep_subject".into(), "1".into()));
    }
    if a.keep_perspective {
        o.push(("keep_subject".into(), "0".into()));
        o.push(("focal".into(), "0".into()));
    }
    if a.mirror {
        o.push(("mirror".into(), "1".into()));
    }
    if a.no_mirror {
        o.push(("mirror".into(), "0".into()));
    }
    let mut text = |k: &str, v: &Option<String>| {
        if let Some(v) = v {
            o.push((k.into(), format!("\"{}\"", v.trim())));
        }
    };
    text("fill", &a.fill);
    text("focus", &a.focus);
    text("format", &a.format);
    text("relight", &a.relight);
    for s in &a.set {
        let (k, v) = s.split_once('=').with_context(|| format!("--set {s:?}: expected KEY=VALUE"))?;
        let k = k.trim();
        if !PhotoPlan::KEYS.iter().any(|(kk, _)| *kk == k) {
            anyhow::bail!("--set {k}: no such key (see `make-photo keys`)");
        }
        let v = v.trim();
        let v = if v.parse::<f32>().is_ok() || v.starts_with('[') || v.starts_with('"') { v.to_string() } else { format!("\"{v}\"") };
        o.push((k.to_string(), v));
    }
    Ok(o)
}

/// Read the picture: the pixel facts, and each vision model in turn (the faces, the subject, the depth, the
/// genre and the stuff), none fatal. Returns the fields at the picture's size.
async fn read_scene(path: &Path, img: &image::RgbImage, pixels_only: bool) -> Result<(Aux, SceneFacts)> {
    use crate::paint::director::{stuff_shares, Genre, STUFF_LABELS};
    let (w, h) = img.dimensions();
    let scale = (1024.0 / w.max(h) as f32).min(1.0);
    let (sw, sh) = (((w as f32 * scale).round() as u32).max(1), ((h as f32 * scale).round() as u32).max(1));
    let small = if scale < 1.0 { image::imageops::resize(img, sw, sh, image::imageops::FilterType::Triangle) } else { img.clone() };
    let mut f = SceneFacts::from_pixels(&small);
    f.width = w;
    f.height = h;
    f.structure = crate::paint::plan::structure_of(img);
    let mut aux = Aux::default();
    if pixels_only {
        f.sources.push("the vision models not run (--pixels-only)".into());
        aux.sources = f.sources.clone();
        return Ok((aux, f));
    }
    let device = match crate::device::select("auto") {
        Ok(d) => d,
        Err(e) => {
            f.sources.push(format!("the vision models not run (no device: {e})"));
            aux.sources = f.sources.clone();
            return Ok((aux, f));
        }
    };
    // THE FACES: SCRFD on the picture itself, the boxes and the eyes in its pixels.
    match crate::pipelines::scrfd::resolve_scrfd_weights().await {
        Ok(Some(weights)) => match crate::pipelines::scrfd::SCRFDDetector::load(&weights, crate::pipelines::scrfd::SCRFDConfig::default(), &device, candle_core::DType::F32) {
            Ok(det) => match det.detect(path) {
                Ok(faces) => {
                    let area = (w as f32 * h as f32).max(1.0);
                    let mut share = 0.0;
                    for fc in &faces {
                        if fc.score < 0.5 {
                            continue;
                        }
                        let b = fc.bbox;
                        share += ((b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)) / area;
                        aux.faces.push((b, [fc.landmarks[0], fc.landmarks[1]]));
                    }
                    f.faces = aux.faces.len();
                    f.face_share = share.min(1.0);
                    f.sources.push(if f.faces > 0 { format!("faces: SCRFD found {}", f.faces) } else { "faces: SCRFD found none".into() });
                }
                Err(e) => f.sources.push(format!("faces: the detector failed ({e})")),
            },
            Err(e) => f.sources.push(format!("faces: the detector did not load ({e})")),
        },
        Ok(None) => f.sources.push("faces: the detector's weights are not fetched".into()),
        Err(e) => f.sources.push(format!("faces: {e}")),
    }
    let n = (sw as usize * sh as usize).max(1) as f32;
    // THE SUBJECT: U²-Net's matte.
    match crate::pipelines::matting::Matter::load(&device).await {
        Ok(matter) => match matter.matte(&small) {
            Ok(alpha) => {
                let fld = Field::from_gray(&alpha);
                let (mut cnt, mut sx, mut sy) = (0usize, 0f64, 0f64);
                for (x, y, p) in alpha.enumerate_pixels() {
                    if p.0[0] > 127 {
                        cnt += 1;
                        sx += x as f64;
                        sy += y as f64;
                    }
                }
                f.subject_share = Some(cnt as f32 / n);
                f.subject_centre = (cnt > 0).then(|| ((sx / cnt as f64 / sw as f64) as f32, (sy / cnt as f64 / sh as f64) as f32));
                aux.matte = Some(fld.resized(w, h));
                f.sources.push("subject: U²-Net matte".into());
            }
            Err(e) => f.sources.push(format!("subject: the matter failed ({e})")),
        },
        Err(e) => f.sources.push(format!("subject: the matter did not load ({e})")),
    }
    // THE DEPTH.
    match crate::pipelines::depth::DepthPipeline::load(device.clone()).await {
        Ok(depth) => match depth.depth_map(path, sw, sh) {
            Ok(d) => {
                let (mn, mx) = d.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                let span = (mx - mn).max(1e-6);
                let fld = Field { w: sw, h: sh, v: d.iter().map(|v| (v - mn) / span).collect() };
                f.far_share = Some(fld.v.iter().filter(|v| **v < 0.35).count() as f32 / fld.v.len().max(1) as f32);
                aux.depth = Some(fld.resized(w, h));
                f.sources.push("depth: Depth-Anything-V2 small".into());
            }
            Err(e) => f.sources.push(format!("depth: failed ({e})")),
        },
        Err(e) => f.sources.push(format!("depth: did not load ({e})")),
    }
    // THE GENRE AND THE STUFF, and the moving things' field from the crops.
    match crate::pipelines::clip_embed::ClipEmbedder::load(&device).await {
        Ok(clip) => {
            let mut probe = || -> Result<()> {
                let whole = clip.embed_image(path)?;
                let scores: Vec<(Genre, f32)> = Genre::LABELS.iter().map(|(g, t)| clip.embed_text(t).map(|e| (*g, crate::pipelines::clip_embed::cosine(&whole, &e)))).collect::<Result<_>>()?;
                f.genre = Genre::from_scores(&scores);
                let texts: Vec<Vec<f32>> = STUFF_LABELS.iter().map(|(_, t)| clip.embed_text(&format!("a photo of {t}"))).collect::<Result<_>>()?;
                let side = sw.max(sh) as f32;
                let crop = (side / 4.0).round().max(32.0) as u32;
                let stride = (side / 6.0).round().max(16.0) as u32;
                let tmp = tempfile::Builder::new().prefix("plakat-make-photo-").suffix(".png").tempfile().context("a scratch file for the crops")?;
                let (mut acc, mut cnt) = ([0f32; 4], 0usize);
                let mut moving_sum = Field::new(sw, sh, 0.0);
                let mut moving_cnt = Field::new(sw, sh, 0.0);
                let mut y0 = 0u32;
                while y0 + crop / 2 < sh {
                    let mut x0 = 0u32;
                    while x0 + crop / 2 < sw {
                        let (cw, ch) = (crop.min(sw - x0), crop.min(sh - y0));
                        image::imageops::crop_imm(&small, x0, y0, cw, ch).to_image().save(tmp.path()).context("saving a crop")?;
                        let e = clip.embed_image(tmp.path())?;
                        let cos: Vec<f32> = texts.iter().map(|t| crate::pipelines::clip_embed::cosine(&e, t)).collect();
                        let (sky, water, fol, stone) = stuff_shares(&cos);
                        acc[0] += sky;
                        acc[1] += water;
                        acc[2] += fol;
                        acc[3] += stone;
                        cnt += 1;
                        let mv = (water + 0.6 * fol + 0.4 * sky).clamp(0.0, 1.0);
                        for y in y0..y0 + ch {
                            for x in x0..x0 + cw {
                                let i = (y * sw + x) as usize;
                                moving_sum.v[i] += mv;
                                moving_cnt.v[i] += 1.0;
                            }
                        }
                        x0 += stride;
                    }
                    y0 += stride;
                }
                if cnt > 0 {
                    f.sky_share = Some(acc[0] / cnt as f32);
                    f.water_share = Some(acc[1] / cnt as f32);
                    f.foliage_share = Some(acc[2] / cnt as f32);
                    f.stone_share = Some(acc[3] / cnt as f32);
                    for i in 0..moving_sum.v.len() {
                        if moving_cnt.v[i] > 0.0 {
                            moving_sum.v[i] /= moving_cnt.v[i];
                        }
                    }
                    aux.moving = Some(moving_sum.resized(w, h));
                }
                f.sources.push(format!("genre and stuff: CLIP ViT-L/14 over the picture and {cnt} crops"));
                Ok(())
            };
            if let Err(e) = probe() {
                f.sources.push(format!("CLIP: failed ({e})"));
            }
        }
        Err(e) => f.sources.push(format!("CLIP: did not load ({e})")),
    }
    aux.sources = f.sources.clone();
    Ok((aux, f))
}

/// The focus region for the sharpness reading: the faces (grown by half), else the subject, else the central
/// half of the frame.
fn focus_mask(aux: &Aux, w: u32, h: u32) -> Field {
    let mut m = Field::new(w, h, 0.0);
    if !aux.faces.is_empty() {
        for (b, _) in &aux.faces {
            let (bw, bh) = ((b[2] - b[0]) * 0.25, (b[3] - b[1]) * 0.25);
            for y in (b[1] - bh).max(0.0) as u32..((b[3] + bh).min(h as f32)) as u32 {
                for x in (b[0] - bw).max(0.0) as u32..((b[2] + bw).min(w as f32)) as u32 {
                    m.v[(y * w + x) as usize] = 1.0;
                }
            }
        }
        return m;
    }
    if let Some(mt) = &aux.matte {
        if mt.v.iter().filter(|v| **v > 0.5).count() > 64 {
            for i in 0..m.v.len() {
                m.v[i] = if mt.v[i] > 0.5 { 1.0 } else { 0.0 };
            }
            return m;
        }
    }
    for y in h / 4..h * 3 / 4 {
        for x in w / 4..w * 3 / 4 {
            m.v[(y * w + x) as usize] = 1.0;
        }
    }
    m
}

fn subject_share(aux: &Aux) -> f32 {
    aux.matte.as_ref().map(|m| m.v.iter().filter(|v| **v > 0.5).count() as f32 / m.v.len().max(1) as f32).unwrap_or(0.0)
}

/// One render: the stage, the neural stages the flags name, the finish.
async fn render_once(img: &image::RgbImage, plan: &PhotoPlan, aux: &Aux, seed: u64, a: &FromArgs) -> Result<(photo::Rendered, Vec<String>)> {
    let mut st = photo::stage(img, plan, aux, seed)?;
    let mut neural = Vec::new();
    let wants = |s: &str| a.neural.iter().any(|n| n.trim().eq_ignore_ascii_case(s));
    if wants("fill") && plan.fill.trim() == "neural" && st.invented > 0.0 {
        if let Some(holes) = st.holes.clone() {
            match photo::neural::fill(&st.frame, &holes, &plan.prompt, &a.neural_model, seed).await {
                Ok(f) => {
                    st.frame = f;
                    neural.push(format!("fill: {:.1}% of the frame inpainted by {} ({})", st.invented * 100.0, a.neural_model, if plan.prompt.is_empty() { "a photograph" } else { plan.prompt.as_str() }));
                }
                Err(e) => neural.push(format!("fill: failed ({e:#}) — the ground grown instead")),
            }
            if neural.last().map_or(false, |s| s.contains("failed")) {
                let metric = st.aux.metric.clone().unwrap_or_else(|| Field::new(st.frame.w, st.frame.h, 1.0));
                photo::camera::propagate(&mut st.frame, &holes, &metric);
            }
        }
    } else if plan.fill.trim() == "neural" && st.invented > 0.0 {
        // The plan asks for the neural fill but the flag does not: the ground is grown and it is said.
        if let Some(holes) = st.holes.clone() {
            let metric = st.aux.metric.clone().unwrap_or_else(|| Field::new(st.frame.w, st.frame.h, 1.0));
            photo::camera::propagate(&mut st.frame, &holes, &metric);
            neural.push("fill: the plan says neural but --neural fill was not given — the ground grown instead".into());
        }
    }
    if wants("relight") {
        if plan.relight.trim().is_empty() {
            neural.push("relight: no preset in the plan (`--relight <preset>`)".into());
        } else {
            match photo::neural::relight(&st.frame, plan.relight.trim(), seed).await {
                Ok(f) => {
                    st.frame = f;
                    neural.push(format!("relight: IC-Light, preset {} — the frame is the model's, background included", plan.relight.trim()));
                }
                Err(e) => neural.push(format!("relight: failed ({e:#})")),
            }
        }
    }
    if wants("refine") {
        if plan.refine <= 0.0 {
            neural.push("refine: the plan's `refine` is 0 (`--refine 0.3`)".into());
        } else {
            let controls: Vec<String> = a.neural_control.iter().filter(|c| !c.trim().eq_ignore_ascii_case("none")).cloned().collect();
            match photo::neural::refine(&st.frame, plan.refine, &plan.prompt, &a.neural_model, seed, &controls, plan.neural_px.max(256.0) as u32).await {
                Ok(f) => {
                    st.frame = f;
                    neural.push(format!("refine: img2img at {:.2} by {} with the medium's prompt{}{}", plan.refine, a.neural_model, if controls.is_empty() { String::new() } else { format!(", the structure held by {}", controls.join(", ")) }, if plan.refine <= 0.4 { ", the faces composited back" } else { "" }));
                }
                Err(e) => neural.push(format!("refine: failed ({e:#})")),
            }
        }
    }
    for n in &a.neural {
        if !["fill", "refine", "relight"].iter().any(|k| n.trim().eq_ignore_ascii_case(k)) {
            neural.push(format!("{n}: no such neural stage (fill, refine, relight)"));
        }
    }
    Ok((photo::finish(st, plan), neural))
}

fn save_output(path: &Path, img: &image::RgbImage, medium: &media::Medium) -> Result<()> {
    img.save(path).with_context(|| format!("saving {}", path.display()))?;
    // The picture says what it is: no camera's EXIF is forged, the keywords name the simulation.
    #[cfg(feature = "photos")]
    {
        let fields = crate::photos::exifwrite::MetaFields { keywords: vec!["plakat make-photo".into(), format!("media:{}", medium.name), "simulated photograph".into()], ..Default::default() };
        let _ = crate::photos::exifwrite::write_metadata(path, &fields);
    }
    #[cfg(not(feature = "photos"))]
    let _ = medium;
    Ok(())
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("png");
    path.with_file_name(format!("{stem}{suffix}.{ext}"))
}

async fn run_from(a: &FromArgs) -> Result<()> {
    let input = a.input.clone().context("give the picture: plakat make-photo <image> --media <name>")?;
    let medium = media::by_name(&a.media).with_context(|| format!("unknown medium {:?} — see `plakat make-photo list`", a.media))?;
    let img = image::open(&input).with_context(|| format!("opening {}", input.display()))?.to_rgb8();
    let out = a.out.clone().unwrap_or_else(|| input.with_file_name(format!("{}.{}.png", input.file_stem().and_then(|s| s.to_str()).unwrap_or("out"), medium.name)));
    let ov = overrides(a)?;
    let seed = a.seed.unwrap_or(0);
    let t0 = std::time::Instant::now();
    let (aux, facts) = read_scene(&input, &img, a.pixels_only).await?;
    println!("{}  {} as {} — {} ({}); read in {:.1} s", style("◆").cyan(), input.display(), style(medium.name).bold(), medium.title, medium.years, t0.elapsed().as_secs_f32());
    for s in &facts.sources {
        println!("{}  {s}", style("·").dim());
    }
    // THE PLAN.
    let source_name = input.file_name().and_then(|s| s.to_str()).unwrap_or("the picture").to_string();
    let (mut directed, plan_path): (Option<director::Directed>, Option<PathBuf>) = match (&a.plan, &a.make_plan) {
        (Some(p), _) => {
            let text = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
            let mut d = director::Directed::from_text(&text)?;
            if !ov.is_empty() {
                let moves: Vec<director::Adjust> = ov.iter().map(|(k, v)| director::Adjust { key: k.clone(), value: v.clone(), why: "the command line".into() }).collect();
                d = director::apply_section(&d, "the command line", "the command line", &[], &moves)?;
            }
            (Some(d), Some(p.clone()))
        }
        (None, Some(path)) => {
            let d = director::direct(&facts, medium, &source_name, &ov, &aux)?;
            let p = if path.is_empty() { out.with_file_name(format!("{}.plan.hjson", out.file_stem().and_then(|s| s.to_str()).unwrap_or("out"))) } else { PathBuf::from(path) };
            std::fs::write(&p, &d.text).with_context(|| format!("writing {}", p.display()))?;
            println!("{}  the director's plan → {}", style("✓").green(), p.display());
            for n in &d.notes {
                println!("{}  {n}", style("·").dim());
            }
            (Some(d), Some(p))
        }
        (None, None) => (None, None),
    };
    let mut plan = match &directed {
        Some(d) => d.plan.clone(),
        None => {
            // The director composes the plan on every run; without --make-plan it is only not written.
            let d = director::direct(&facts, medium, &source_name, &ov, &aux)?;
            for n in &d.notes {
                println!("{}  {n}", style("·").dim());
            }
            d.plan
        }
    };
    // THE LOOP (one pass without --make-plan-iterate).
    let n_iter = if a.make_plan.is_some() { a.make_plan_iterate } else { 0 };
    let source_focus = focus_mask(&aux, img.width(), img.height());
    let source_readings = measure::measure(&img, Some(&source_focus), aux.depth.as_ref());
    let mut last_path = out.clone();
    for k in 0..=n_iter {
        let t1 = std::time::Instant::now();
        let (r, neural) = render_once(&img, &plan, &aux, seed, a).await?;
        let path = if k < n_iter { with_suffix(&out, &format!(".i{k}")) } else { out.clone() };
        save_output(&path, &r.image, medium)?;
        last_path = path.clone();
        println!("{}  {} → {} ({}×{}, {:.1} s)", style("✓").green(), if n_iter > 0 { format!("pass {k}") } else { "made".into() }, path.display(), r.image.width(), r.image.height(), t1.elapsed().as_secs_f32());
        for n in &r.notes {
            println!("{}  {n}", style("·").dim());
        }
        for n in &neural {
            println!("{}  neural {n}", style("·").yellow());
        }
        let result_focus = focus_mask(&r.aux, r.image.width(), r.image.height());
        let result_readings = measure::measure(&r.image, Some(&result_focus), r.aux.depth.as_ref());
        if a.report {
            println!("{}", style("the source:").bold());
            for l in source_readings.lines() {
                println!("  {l}");
            }
            println!("{}", style("the result:").bold());
            for l in result_readings.lines() {
                println!("  {l}");
            }
        }
        if n_iter == 0 {
            break;
        }
        // THE CRITIC.
        let outcome = director::Outcome { source: source_readings.clone(), result: result_readings, invented: r.invented, faces_before: aux.faces.len(), faces_after: r.aux.faces.len(), subject_before: subject_share(&aux), subject_after: subject_share(&r.aux) };
        let v = director::verdict(&outcome, &plan);
        let d = directed.take().context("the loop needs the director's plan")?;
        let d2 = director::apply_verdict(&d, path.file_name().and_then(|s| s.to_str()).unwrap_or("the picture"), &v)?;
        if let Some(pp) = &plan_path {
            std::fs::write(pp, &d2.text)?;
        }
        for l in &v.readings {
            println!("{}  critic: {l}", style("·").dim());
        }
        if v.moves.is_empty() {
            println!("{}  the critic moves nothing — the picture stands as pass {k}", style("✓").green());
            if path != out {
                std::fs::copy(&path, &out)?;
                println!("{}  → {}", style("✓").green(), out.display());
            }
            break;
        }
        for m in &v.moves {
            println!("{}  critic moves {}: {} — {}", style("·").yellow(), m.key, m.value, m.why);
        }
        plan = d2.plan.clone();
        directed = Some(d2);
        if k == n_iter {
            println!("{}  the loop's last pass; the critic still had moves (above) — they are in the plan", style("·").dim());
        }
    }
    let _ = last_path;
    Ok(())
}

async fn run_measure(m: &MeasureArgs) -> Result<()> {
    let mut rows = Vec::new();
    for p in &m.inputs {
        let img = image::open(p).with_context(|| format!("opening {}", p.display()))?.to_rgb8();
        let (aux, _) = if m.models { read_scene(p, &img, false).await? } else { (Aux::default(), SceneFacts::default()) };
        let focus = focus_mask(&aux, img.width(), img.height());
        let r = measure::measure(&img, Some(&focus), aux.depth.as_ref());
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string();
        println!("{}  {}", style("◆").cyan(), style(&name).bold());
        for l in r.lines() {
            println!("  {l}");
        }
        rows.push((name, r));
    }
    if rows.len() > 1 || m.out.is_some() {
        let t = measure::Readings::table(&rows);
        if let Some(o) = &m.out {
            std::fs::write(o, &t).with_context(|| format!("writing {}", o.display()))?;
            println!("{}  table → {}", style("✓").green(), o.display());
        } else {
            println!("\n{t}");
        }
    }
    Ok(())
}

async fn run_plan(p: &PlanArgs) -> Result<()> {
    let medium = media::by_name(&p.media).with_context(|| format!("unknown medium {:?} — see `plakat make-photo list`", p.media))?;
    let img = image::open(&p.input).with_context(|| format!("opening {}", p.input.display()))?.to_rgb8();
    if let Some(sheet) = &p.verdict {
        let plan_path = p.plan.clone().context("--verdict needs --plan <the plan the picture was made by>")?;
        let text = std::fs::read_to_string(&plan_path)?;
        let d = director::Directed::from_text(&text)?;
        let (aux, _) = read_scene(&p.input, &img, p.pixels_only).await?;
        let result = image::open(sheet).with_context(|| format!("opening {}", sheet.display()))?.to_rgb8();
        let (raux, _) = read_scene(sheet, &result, p.pixels_only).await?;
        let sf = focus_mask(&aux, img.width(), img.height());
        let rf = focus_mask(&raux, result.width(), result.height());
        let outcome = director::Outcome {
            source: measure::measure(&img, Some(&sf), aux.depth.as_ref()),
            result: measure::measure(&result, Some(&rf), raux.depth.as_ref()),
            invented: 0.0,
            faces_before: aux.faces.len(),
            faces_after: raux.faces.len(),
            subject_before: subject_share(&aux),
            subject_after: subject_share(&raux),
        };
        let v = director::verdict(&outcome, &d.plan);
        let d2 = director::apply_verdict(&d, sheet.file_name().and_then(|s| s.to_str()).unwrap_or("the picture"), &v)?;
        let out = p.out.clone().unwrap_or(plan_path);
        std::fs::write(&out, &d2.text)?;
        for l in &v.readings {
            println!("{}  {l}", style("·").dim());
        }
        for m in &v.moves {
            println!("{}  moves {}: {} — {}", style("·").yellow(), m.key, m.value, m.why);
        }
        if v.moves.is_empty() {
            println!("{}  the critic moves nothing", style("✓").green());
        }
        println!("{}  verdict → {}", style("✓").green(), out.display());
        return Ok(());
    }
    let (aux, facts) = read_scene(&p.input, &img, p.pixels_only).await?;
    let mut ov = Vec::new();
    for s in &p.set {
        let (k, v) = s.split_once('=').with_context(|| format!("--set {s:?}: expected KEY=VALUE"))?;
        let v = v.trim();
        let v = if v.parse::<f32>().is_ok() || v.starts_with('[') || v.starts_with('"') { v.to_string() } else { format!("\"{v}\"") };
        ov.push((k.trim().to_string(), v));
    }
    let source_name = p.input.file_name().and_then(|s| s.to_str()).unwrap_or("the picture").to_string();
    let d = director::direct(&facts, medium, &source_name, &ov, &aux)?;
    let out = p.out.clone().unwrap_or_else(|| p.input.with_file_name(format!("{}.{}.plan.hjson", p.input.file_stem().and_then(|s| s.to_str()).unwrap_or("out"), medium.name)));
    std::fs::write(&out, &d.text).with_context(|| format!("writing {}", out.display()))?;
    for s in &facts.sources {
        println!("{}  {s}", style("·").dim());
    }
    for n in &d.notes {
        println!("{}  {n}", style("·").dim());
    }
    println!("{}  plan → {}", style("✓").green(), out.display());
    Ok(())
}
