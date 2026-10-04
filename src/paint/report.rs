//! The ANALYSIS artefact (RFC PAINT-3, P0): a Markdown report of a paint run — every parameter it used
//! with the comment that explains it, what the run FOUND (faces, matte, hair, budget, passes) and what it
//! MEASURED on the finished canvas (height per plane, stroke lengths, dry share, pigment usage).
//!
//! Everything here is read from the run's own facts: the plan text (its comments are the art director's
//! reasoning), the command line, the resolved parameters, the stroke score, the pass stats, the masks and
//! the canvas. Nothing is invented; a measurement that cannot be made is left out, not decorated.

use std::sync::Mutex;

use crate::paint::canvas::Canvas;
use crate::paint::painter::{PaintParams, PassStat};
use crate::paint::score::StrokeScore;

/// The "found" lines the CLI prints while it prepares a run (faces, matte, semantic regions, hair, HDR,
/// budget) — recorded here so the report can repeat them. One run per process.
static NOTES: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Record a finding (the CLI calls this beside its terminal line).
pub fn note(msg: impl Into<String>) {
    if let Ok(mut n) = NOTES.lock() {
        n.push(msg.into());
    }
}

/// The findings recorded so far (in order).
pub fn notes() -> Vec<String> {
    NOTES.lock().map(|n| n.clone()).unwrap_or_default()
}

/// The reference rows for the dials (`Documentation/PAINT_CONTROLS.md`, compiled in), so a non-default
/// dial can be explained without opening the docs.
const CONTROLS: &str = include_str!("../../Documentation/PAINT_CONTROLS.md");

/// What the report needs from the CLI.
pub struct RunInfo<'a> {
    pub source: &'a std::path::Path,
    pub output: &'a std::path::Path,
    pub width: u32,
    pub height: u32,
    /// The plan file's text, verbatim (comments and all), when a plan was used.
    pub plan_text: Option<&'a str>,
    pub plan_path: Option<&'a std::path::Path>,
    pub argv: &'a [String],
    pub seconds: f64,
}

/// A measured number with its label.
struct Measure {
    label: &'static str,
    value: String,
}

fn f2(v: f32) -> String {
    let s = format!("{v:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Where a key's value came from: named in the plan text, on the command line, or the default.
fn source_of(key: &str, plan_text: Option<&str>, argv: &[String]) -> &'static str {
    let flag = format!("--{}", key.replace('_', "-"));
    if argv.iter().any(|a| a == &flag || a.starts_with(&format!("{flag}="))) {
        return "cli";
    }
    if let Some(t) = plan_text {
        let needle = format!("{key}:");
        if t.lines().any(|l| l.trim_start().starts_with(&needle)) {
            return "plan";
        }
    }
    "default"
}

/// The PAINT_CONTROLS row for a key (the table cell that explains it), if there is one.
fn control_row(key: &str) -> Option<String> {
    let needle = format!("| `{key}` ");
    let needle2 = format!("| `{key}`|");
    CONTROLS.lines().find(|l| l.starts_with(&needle) || l.starts_with(&needle2)).map(|l| {
        // The explanation is the last cell.
        let cells: Vec<&str> = l.split('|').map(|c| c.trim()).filter(|c| !c.is_empty()).collect();
        cells.last().map(|s| s.to_string()).unwrap_or_default()
    })
}

/// The length of a stroke's recorded path in pixels.
fn path_len(spline: &[[f32; 2]]) -> f32 {
    spline.windows(2).map(|w| ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt()).sum()
}

/// Mean of a mask-weighted field over the pixels the mask selects (`None` when it selects nothing).
fn masked_mean(field: &[f32], mask: &[f32], thr: f32, invert: bool) -> Option<f32> {
    let (mut s, mut n) = (0f64, 0usize);
    for (v, m) in field.iter().zip(mask) {
        let inside = *m > thr;
        if inside != invert {
            s += *v as f64;
            n += 1;
        }
    }
    (n > 0).then(|| (s / n as f64) as f32)
}

/// The analysis, as Markdown.
pub fn analysis_markdown(info: &RunInfo, params: &PaintParams, result_score: &StrokeScore, canvas: &Canvas, stats: &[PassStat], strokes_laid: usize, source: Option<&image::RgbImage>) -> String {
    let mut o = String::new();
    let h = &result_score.header;
    let title = info.source.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "painting".into());

    // ---- 1. Run ----
    o.push_str(&format!("# plakat paint — analysis of “{title}”\n\n"));
    o.push_str("| | |\n|---|---|\n");
    o.push_str(&format!("| source | `{}` ({}×{}) |\n", info.source.display(), info.width, info.height));
    o.push_str(&format!("| output | `{}` · score `{}` |\n", info.output.display(), info.output.with_extension("strokes").display()));
    o.push_str(&format!("| medium | **{}** · palette `{}` · seed {} |\n", h.medium, h.palette, h.seed));
    o.push_str(&format!("| strokes | {} laid of a budget of {} |\n", strokes_laid, params.budget));
    o.push_str(&format!("| time | {:.1} s in the painter |\n", info.seconds));
    o.push_str(&format!("| plakat | {} |\n", env!("CARGO_PKG_VERSION")));
    o.push_str(&format!("| command | `{}` |\n\n", info.argv.join(" ")));

    // ---- 2. Plan ----
    o.push_str("## The plan\n\n");
    match (info.plan_text, info.plan_path) {
        (Some(t), p) => {
            if let Some(p) = p {
                o.push_str(&format!("`{}`, verbatim — the comments are the art director's reasoning:\n\n", p.display()));
            }
            o.push_str("```hjson\n");
            o.push_str(t.trim_end());
            o.push_str("\n```\n\n");
        }
        _ => o.push_str("No plan file: the technique's defaults and the command line only.\n\n"),
    }

    // ---- 2b. Resolved parameters ----
    o.push_str("### Resolved parameters\n\n");
    o.push_str("Every dial the painter read, with where its value came from (`plan` = named in the plan, `cli` = on the command line, `default` = the technique's own).\n\n");
    o.push_str("| key | value | from | what it does |\n|---|---|---|---|\n");
    let b = &params.brush;
    let rows: Vec<(&str, String)> = vec![
        ("medium", params.medium.clone()),
        ("palette", h.palette.clone()),
        ("style", format!("{:?}", params.style).to_lowercase()),
        ("technique", format!("{:?}", params.technique).to_lowercase()),
        ("armature", params.armature_side.map(|v| v.to_string()).unwrap_or_else(|| "—".into())),
        ("armature_body", params.armature_body_side.map(|v| v.to_string()).unwrap_or_else(|| "—".into())),
        ("armature_face", params.armature_face_side.map(|v| v.to_string()).unwrap_or_else(|| "—".into())),
        ("min_brush", f2(params.min_brush)),
        ("brush_sizes", params.brush_sizes.iter().map(|v| f2(*v)).collect::<Vec<_>>().join(", ")),
        ("budget", params.budget.to_string()),
        ("commit_shadows", f2(params.commit_shadows)),
        ("stroke_width", f2(params.stroke_width)),
        ("stroke_length", f2(params.stroke_len)),
        ("detail_len", f2(params.detail_len)),
        ("detail_restate", f2(params.detail_restate)),
        ("detail_texture", f2(params.detail_texture)),
        ("fine_lines", f2(params.fine_lines)),
        ("coverage", f2(params.coverage)),
        ("haze", f2(params.haze)),
        ("bleed", f2(params.bleed)),
        ("diffuse", f2(params.diffuse)),
        ("dry", f2(params.dry)),
        ("opacity", f2(params.opacity)),
        ("pickup", f2(b.k_pickup)),
        ("impasto", f2(params.impasto)),
        ("impasto_map", f2(params.impasto_map)),
        ("ridges", f2(b.ridges)),
        ("skip", f2(b.skip)),
        ("weave", f2(params.weave)),
        ("sheen", f2(params.sheen)),
        ("chroma", f2(params.chroma)),
        ("dry_shift", f2(params.dry_shift)),
        ("granulate", f2(params.granulate)),
        ("edge_pool", f2(params.edge_pool)),
        ("paper_edge", f2(params.paper_edge)),
        ("contrast", f2(params.contrast)),
        ("warmth", f2(params.warmth)),
        ("clarity", f2(params.clarity)),
        ("lift", f2(params.lift)),
        ("splatter", f2(params.splatter)),
        ("leak", f2(params.leak)),
        ("rigger", f2(params.rigger)),
        ("hotspot", f2(params.hotspot)),
        ("preserve_face", f2(params.preserve_face)),
        ("focus_detail", f2(params.focus_detail)),
        ("saliency", f2(params.saliency)),
        ("bristles", b.bristles.to_string()),
        ("deposit", f2(b.k_deposit)),
        ("viscosity", f2(b.viscosity)),
        ("streak", f2(b.streak)),
        ("round", f2(b.round)),
    ];
    for (k, v) in &rows {
        let from = source_of(k, info.plan_text, info.argv);
        let what = if from == "default" { String::new() } else { control_row(k).unwrap_or_default() };
        o.push_str(&format!("| `{k}` | {v} | {from} | {what} |\n"));
    }
    // Plan keys the painter's dial table does not carry (the CLI's own switches: hdr, value_key, the
    // armature tiers, semantic…): listed from the plan text so nothing named in the plan goes unreported.
    if let Some(t) = info.plan_text {
        for line in t.lines() {
            let l = line.trim();
            if l.starts_with('#') || !l.contains(':') {
                continue;
            }
            let (k, v) = l.split_once(':').unwrap();
            let k = k.trim();
            let v = v.split('#').next().unwrap_or("").trim().trim_matches('"');
            if k.is_empty() || k.contains(' ') || rows.iter().any(|(rk, _)| *rk == k) || k == "analysis" || k == "analysis_insights" || k == "outcome" {
                continue;
            }
            o.push_str(&format!("| `{k}` | {v} | plan | {} |\n", control_row(k).unwrap_or_default()));
        }
    }
    if let Some((s, r, m, g, e)) = h.flow {
        o.push_str(&format!("| `flow` | strength {} · radius {} px · rim {} · grain {} · selective {} | recipe | the watercolour's fluid stage, run after every broad pass |\n", f2(s), f2(r), f2(m), f2(g), f2(e)));
    }
    if h.transmittance {
        o.push_str("| `film` | transmittance | recipe | the paint as a transparent film over the paper (Beer–Lambert), not an opaque layer |\n");
    }
    o.push('\n');

    // ---- 3. What the run found ----
    o.push_str("## What the run found\n\n");
    let notes = notes();
    if notes.is_empty() {
        o.push_str("(nothing recorded)\n\n");
    } else {
        for n in &notes {
            o.push_str(&format!("- {}\n", n.trim()));
        }
        o.push('\n');
    }
    let px = (info.width as usize) * (info.height as usize);
    let pct = |m: &Option<Vec<f32>>, thr: f32| m.as_ref().map(|m| (m.iter().filter(|v| **v > thr).count() * 100 / px.max(1)) as i64);
    o.push_str("| mask | covers |\n|---|---|\n");
    o.push_str(&format!("| faces | {} |\n", pct(&params.face_mask, 0.35).map(|p| format!("{p}% of the sheet")).unwrap_or_else(|| "none".into())));
    o.push_str(&format!("| subject (matte) | {} |\n", pct(&params.subject_mask, 0.5).map(|p| format!("{p}%")).unwrap_or_else(|| "none".into())));
    o.push_str(&format!("| hair / fur | {} |\n", pct(&params.hair_mask, 0.3).map(|p| format!("{p}%")).unwrap_or_else(|| "none".into())));
    o.push_str(&format!("| depth map | {} |\n\n", if params.depth.is_some() { "yes" } else { "no" }));

    // ---- 4. The passes ----
    o.push_str("## The passes\n\n");
    o.push_str("| stage | brush | strokes | per s | seconds | mean length | mean width | dry marks |\n|---|---|---|---|---|---|---|---|\n");
    for s in stats {
        let recs: Vec<&crate::paint::score::StrokeRecord> = result_score.strokes.iter().filter(|r| r.stage == s.stage && !r.wipe).collect();
        let (mut len, mut wid, mut dry) = (0f32, 0f32, 0usize);
        for r in &recs {
            len += path_len(&r.spline);
            wid += r.w0;
            if r.wet < 0.62 {
                dry += 1;
            }
        }
        let n = recs.len().max(1) as f32;
        let rate = if s.seconds > 0.0 { s.strokes as f64 / s.seconds } else { 0.0 };
        o.push_str(&format!(
            "| {} | {:.0} px | {} | {:.0} | {:.1} | {:.1} px | {:.1} px | {}% |\n",
            s.stage, s.radius, s.strokes, rate, s.seconds, len / n, wid / n, if recs.is_empty() { 0 } else { dry * 100 / recs.len() }
        ));
    }
    // THE CAPS a reader (or a model) cannot see from the counts alone: why the budget was not spent, and
    // how long a mark of each brush can be at all.
    if strokes_laid < params.budget {
        o.push_str(&format!(
            "\n**Why {} of {} strokes:** the budget is a ceiling, not a target. A pass lays a mark only where the canvas is still notably wrong (the restate gate), where the picture has structure for a fine brush, and where a plane allows that brush — so a picture whose masses are right after the mid passes leaves the budget unspent. Raising `fill`, `budget` or the density does not change this; `detail_restate` (lower = restate more) and `fine_lines` do.\n",
            strokes_laid, params.budget
        ));
    }
    o.push_str("\n**Length caps:** a mark may run at most ~2.2× its brush radius per half (then the colour-drift and hard-edge stops end it sooner), scaled by `stroke_length` and, on the fine passes, by `detail_len`: ");
    let caps: Vec<String> = stats.iter().filter(|s| s.strokes > 0).map(|s| format!("{} ≤ ~{:.0} px", s.stage, s.radius * 2.2 * 2.0 * params.stroke_len)).collect();
    o.push_str(&caps.join(" · "));
    o.push_str(". A 2–4 px brush cannot lay a long mark whatever the dials; its share of short marks is a fact of its size.\n");
    let stage_names: Vec<&str> = result_score.strokes.iter().map(|r| r.stage.as_str()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    let extra: Vec<&str> = stage_names.iter().copied().filter(|n| !stats.iter().any(|s| s.stage == *n)).collect();
    if !extra.is_empty() {
        o.push_str(&format!("\nOther stages in the score (laid outside the ladder): {}.\n", extra.iter().map(|s| format!("`{s}`")).collect::<Vec<_>>().join(", ")));
    }
    o.push('\n');

    // ---- 5. Measurements ----
    o.push_str("## Measurements\n\n");
    let mut ms: Vec<Measure> = Vec::new();
    // Height per plane, % of the peak.
    let peak = canvas.height.iter().copied().fold(0f32, f32::max);
    if peak > 1e-6 {
        let hp: Vec<f32> = canvas.height.iter().map(|v| v / peak * 100.0).collect();
        let whole = hp.iter().sum::<f32>() / hp.len().max(1) as f32;
        ms.push(Measure { label: "paint height, whole sheet (mean, % of peak)", value: format!("{whole:.1} — heights are relative to this run's own peak; compare RATIOS between runs, not these percentages") });
        if let Some(fm) = &params.face_mask {
            if let Some(v) = masked_mean(&hp, fm, 0.35, false) {
                ms.push(Measure { label: "paint height on the faces", value: format!("{v:.1}") });
            }
        }
        let mut face_h = None;
        if let Some(fm) = &params.face_mask {
            face_h = masked_mean(&hp, fm, 0.35, false);
        }
        if let Some(sm) = &params.subject_mask {
            let subj = masked_mean(&hp, sm, 0.5, false);
            let bg = masked_mean(&hp, sm, 0.5, true);
            if let Some(v) = subj {
                ms.push(Measure { label: "paint height on the subject", value: format!("{v:.1}") });
            }
            if let Some(v) = bg {
                ms.push(Measure { label: "paint height on the background", value: format!("{v:.1}") });
            }
            if let (Some(sj), Some(b)) = (subj, bg) {
                ms.push(Measure { label: "RATIO subject : background (the map's intent is > 1)", value: format!("{:.2}", sj / b.max(1e-3)) });
            }
            if let (Some(f), Some(sj)) = (face_h, subj) {
                ms.push(Measure { label: "RATIO faces : subject (the map lays faces thin: < 1)", value: format!("{:.2}", f / sj.max(1e-3)) });
            }
        }
        // Lights and shadows by the SOURCE picture's value (what the painter read), the top and bottom
        // fifths — and smooth vs broken SURFACES by the source's local value range at the finest brush's
        // scale (the painter's own busy measure). The finished picture's value is relit and painted, so
        // it is a weaker witness; it is used only when the source is not at hand.
        let (luma, witness) = match source {
            Some(img) if img.width() as usize * img.height() as usize == hp.len() => (crate::paint::painter::luma_map(img), "source"),
            _ => {
                let img = canvas.to_image();
                (img.pixels().map(|p| (0.299 * p.0[0] as f32 + 0.587 * p.0[1] as f32 + 0.114 * p.0[2] as f32) / 255.0).collect::<Vec<f32>>(), "finished picture")
            }
        };
        let mut sorted = luma.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo = sorted[sorted.len() / 5];
        let hi = sorted[sorted.len() * 4 / 5];
        let lights: Vec<f32> = luma.iter().map(|l| if *l >= hi { 1.0 } else { 0.0 }).collect();
        let shadows: Vec<f32> = luma.iter().map(|l| if *l <= lo { 1.0 } else { 0.0 }).collect();
        if let Some(v) = masked_mean(&hp, &lights, 0.5, false) {
            ms.push(Measure { label: "paint height in the lights (top fifth of the source by value)", value: format!("{v:.1} (by the {witness})") });
        }
        if let Some(v) = masked_mean(&hp, &shadows, 0.5, false) {
            ms.push(Measure { label: "paint height in the shadows (bottom fifth)", value: format!("{v:.1}") });
        }
        if let (Some(l), Some(sh)) = (masked_mean(&hp, &lights, 0.5, false), masked_mean(&hp, &shadows, 0.5, false)) {
            ms.push(Measure { label: "RATIO lights : shadows (the map's intent is > 1; without the map darks come out thickest, they carry more pigment)", value: format!("{:.2}", l / sh.max(1e-3)) });
        }
        if witness == "source" {
            let r = (params.min_brush * 0.75).round().max(2.0) as usize;
            let fine = crate::paint::painter::local_range(&luma, info.width as usize, info.height as usize, r);
            let busy: Vec<f32> = fine.iter().map(|f| ((f - 0.08) / 0.12).clamp(0.0, 1.0)).collect();
            if let (Some(b), Some(sm)) = (masked_mean(&hp, &busy, 0.5, false), masked_mean(&hp, &busy, 0.2, true)) {
                ms.push(Measure { label: "paint height on broken surfaces (beard, bark, cobbles) / on smooth ones (skin, sky, glass)", value: format!("{b:.1} / {sm:.1}") });
                ms.push(Measure { label: "RATIO broken : smooth (the map's intent is > 1)", value: format!("{:.2}", b / sm.max(1e-3)) });
            }
            let covered = busy.iter().filter(|b| **b > 0.5).count() * 100 / busy.len().max(1);
            ms.push(Measure { label: "share of the sheet that is broken surface at the finest brush's scale", value: format!("{covered}%") });
        }
    }
    // Fine-stroke lengths: the finest stage of the ladder.
    if let Some(finest) = stats.iter().filter(|s| s.strokes > 0).min_by(|a, b| a.radius.partial_cmp(&b.radius).unwrap_or(std::cmp::Ordering::Equal)) {
        let lens: Vec<f32> = result_score.strokes.iter().filter(|r| r.stage == finest.stage && !r.wipe).map(|r| path_len(&r.spline)).collect();
        if !lens.is_empty() {
            let n = lens.len() as f32;
            let short = lens.iter().filter(|l| **l < 10.0).count() as f32 / n * 100.0;
            let long = lens.iter().filter(|l| **l > 20.0).count() as f32 / n * 100.0;
            ms.push(Measure { label: "finest marks under 10 px / over 20 px", value: format!("{short:.0}% / {long:.0}% (stage `{}`)", finest.stage) });
        }
    }
    // Dry share overall.
    let painted: Vec<&crate::paint::score::StrokeRecord> = result_score.strokes.iter().filter(|r| !r.wipe).collect();
    if !painted.is_empty() {
        let dry = painted.iter().filter(|r| r.wet < 0.62).count() * 100 / painted.len();
        ms.push(Measure { label: "marks laid dry (wetness < 0.62)", value: format!("{dry}%") });
    }
    if !ms.is_empty() {
        o.push_str("| measure | value |\n|---|---|\n");
        for m in &ms {
            o.push_str(&format!("| {} | {} |\n", m.label, m.value));
        }
        o.push('\n');
    }
    // Pigment usage.
    let mut usage: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
    for r in &painted {
        for (name, w) in &r.mix {
            *usage.entry(name.as_str()).or_insert(0.0) += *w as f64;
        }
    }
    if !usage.is_empty() {
        let total: f64 = usage.values().sum();
        let mut ranked: Vec<(&str, f64)> = usage.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        o.push_str("### Pigments by use\n\n| pigment | masstone | share |\n|---|---|---|\n");
        for (name, w) in ranked.iter().take(12) {
            let mass = h.pigments.iter().find(|(n, _)| n == name).map(|(_, m)| format!("#{:02x}{:02x}{:02x}", m[0], m[1], m[2])).unwrap_or_else(|| "—".into());
            o.push_str(&format!("| {name} | `{mass}` | {:.1}% |\n", w / total * 100.0));
        }
        o.push('\n');
    }

    // ---- 6. Dials glossary ----
    let non_default: Vec<&(&str, String)> = rows.iter().filter(|(k, _)| source_of(k, info.plan_text, info.argv) != "default").collect();
    if !non_default.is_empty() {
        o.push_str("## The dials, explained\n\n");
        for (k, v) in non_default {
            if let Some(row) = control_row(k) {
                if !row.is_empty() {
                    o.push_str(&format!("- **`{k}` = {v}** — {row}\n"));
                }
            }
        }
        o.push('\n');
    }
    o.push_str("---\n*Every number above is a fact of this run: the plan, the command line, the stroke score and the finished canvas. Regenerate it from the `.strokes` with `plakat paint replay`.*\n");
    o
}

/// The system prompt for the INSIGHTS pass (RFC PAINT-3 §2b), versioned with the binary.
pub const INSIGHTS_SYSTEM: &str = include_str!("../../assets/prompts/paint_insights.md");

/// Whether a provider name sends the text off the machine (a hosted API) — said in the terminal and in
/// the report, never hidden.
pub fn provider_is_hosted(provider: &str) -> bool {
    let p = provider.to_lowercase();
    p == "deepseek" || p == "gemini" || (p == "auto" && {
        let cfg = crate::config::Config::load().ok();
        cfg.as_ref().is_some_and(|c| c.deepseek_api_key.is_some() || c.gemini_api_key.is_some())
    })
}

/// The INSIGHTS pass: the analysis report (facts) through the configured LLM with [`INSIGHTS_SYSTEM`],
/// returning the two sections to append (`## Insights`, `## Recommendations`) plus a provenance line.
/// The model gets the report and the medium's dial glossary — never the picture (P0.5).
pub async fn insights(provider: &str, analysis_md: &str, medium: &str) -> anyhow::Result<String> {
    let glossary = CONTROLS.lines().filter(|l| l.starts_with("| `")).collect::<Vec<_>>().join("\n");
    let user = format!(
        "Medium: {medium}\n\n# RUN REPORT (facts)\n\n{analysis_md}\n\n# DIAL GLOSSARY (the only dials you may recommend)\n\n{glossary}\n"
    );
    let label = crate::prompt::resolve_provider_label(provider);
    let out = crate::prompt::complete(provider, INSIGHTS_SYSTEM, &user, &crate::prompt::EnhanceArgs::default()).await?;
    let out = out.trim();
    // Keep only from the first section heading: a chatty model's preamble is not a finding.
    let body = out.find("## Insights").map(|i| &out[i..]).unwrap_or(out);
    let where_ = if provider_is_hosted(provider) { "a hosted provider — the report text left this machine" } else { "run locally — nothing left this machine" };
    Ok(format!("\n{body}\n\n*Insights by `{label}` ({where_}). They reason only from the report above; they have not seen the picture.*\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_of_reads_the_plan_and_the_command_line() {
        let plan = "medium: oil-direct\nimpasto: 0.80\n# weave: 0.4 (commented out)\n";
        let argv = vec!["plakat".to_string(), "paint".into(), "--impasto-map".into(), "1".into(), "--sheen=0.3".into()];
        assert_eq!(source_of("impasto", Some(plan), &argv), "plan");
        assert_eq!(source_of("impasto_map", Some(plan), &argv), "cli");
        assert_eq!(source_of("sheen", Some(plan), &argv), "cli");
        assert_eq!(source_of("weave", Some(plan), &argv), "default");
    }

    #[test]
    fn the_insights_prompt_pins_its_rules() {
        for must in ["ONLY from the report", "faces must stay recognizable", "## Insights", "## Recommendations", "key: value"] {
            assert!(INSIGHTS_SYSTEM.contains(must), "system prompt lost: {must}");
        }
        assert!(provider_is_hosted("deepseek") && provider_is_hosted("gemini"));
        assert!(!provider_is_hosted("ollama") && !provider_is_hosted("local") && !provider_is_hosted("ollama:qwen2.5"));
    }

    #[test]
    fn the_controls_table_explains_a_dial() {
        let row = control_row("ridges").expect("ridges is documented");
        assert!(row.contains("relief"), "{row}");
        assert!(control_row("no_such_dial").is_none());
    }

    #[test]
    fn the_report_carries_the_run_and_the_passes() {
        use crate::paint::painter::{paint_from_image, PaintParams};
        use crate::paint::palette;
        note("face: 1 found (test)");
        let img = image::RgbImage::from_fn(64, 48, |x, y| image::Rgb([(40 + 3 * x) as u8, (200 - x - y) as u8, (60 + y) as u8]));
        let mut p = PaintParams::new(palette::EARTH, 300);
        p.brush_sizes = vec![14.0, 7.0];
        p.impasto = 0.5;
        let r = paint_from_image(&img, &p);
        let argv = vec!["plakat".to_string(), "paint".into(), "from".into(), "x.png".into(), "--impasto".into(), "0.5".into()];
        let info = RunInfo { source: std::path::Path::new("x.png"), output: std::path::Path::new("out/x_paint.png"), width: 64, height: 48, plan_text: Some("medium: oil-direct\nridges: 0.8\n"), plan_path: None, argv: &argv, seconds: r.seconds };
        let md = analysis_markdown(&info, &p, &r.score, &r.canvas, &r.stats, r.strokes, Some(&img));
        assert!(md.contains("# plakat paint — analysis of “x”"));
        assert!(md.contains("```hjson\nmedium: oil-direct"));
        assert!(md.contains("| `impasto` | 0.5 | cli |"));
        assert!(md.contains("| `ridges` | 0 | plan |"));
        assert!(md.contains("face: 1 found (test)"));
        assert!(md.contains("## The passes"));
        assert!(md.contains("### Pigments by use"));
        assert!(md.contains("paint height, whole sheet"));
    }
}
