//! `plakat persona` — controllable synthetic-person composition (RFC PERSONA-1, the 5.0.0 flagship).
//!
//! P0 first slice: `new` (scaffold a spec) + `lint` (validate without weights). The resolver, geometry,
//! details, casting, scorecard and TUI subcommands land across later phases (ROADMAP_5.0.0). Fully
//! additive — nothing here touches existing behaviour.

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use console::style;
use std::path::PathBuf;

use crate::persona::lint::{self, Level};
use crate::persona::PersonaSpec;

#[derive(Args, Debug)]
pub struct PersonaArgs {
    #[command(subcommand)]
    pub cmd: PersonaCmd,
}

#[derive(Subcommand, Debug)]
pub enum PersonaCmd {
    /// Scaffold a new persona spec (a valid, partial `PersonaSpec` HJSON you then edit or `--tui`).
    New(NewArgs),
    /// Validate a persona spec — schema, scalar ranges, contradictions, the age gate. No weights, no
    /// network. Exits non-zero on any error so it can gate CI.
    Lint(LintArgs),
    /// Show what a spec resolves to for a model family: the salience-ranked prompt-routed attributes,
    /// the compiled positive/negative prompt, and (on CLIP) which attributes the token budget dropped.
    Show(ShowArgs),
    /// Measure a rendered image against a spec (the scorecard, RFC §12). P1: the landmark probe —
    /// SCRFD detects the face, PIPNet-98 aligns it, and geometric ratio metrics are reported.
    Verify(VerifyArgs),
}

#[derive(Args, Debug)]
pub struct NewArgs {
    /// Output path for the new spec (`.hjson`).
    pub out: PathBuf,
    /// How much of the schema to scaffold: `quick` (identity + core structure), `standard`, `full`.
    #[arg(long, default_value = "quick")]
    pub depth: String,
    /// Slug name for the persona (also the People-library folder name).
    #[arg(long, default_value = "unnamed")]
    pub name: String,
    /// Apparent age in years (must be >= 18; see §23.1).
    #[arg(long, default_value_t = 30)]
    pub age: u32,
}

#[derive(Args, Debug)]
pub struct LintArgs {
    /// Path to the persona spec (`.hjson`).
    pub spec: PathBuf,
}

#[derive(Args, Debug)]
pub struct ShowArgs {
    /// Path to the persona spec (`.hjson`).
    pub spec: PathBuf,
    /// Model family to compile for (its encoder class shapes the prompt). Default `sdxl`.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
}

#[derive(Args, Debug)]
pub struct VerifyArgs {
    /// Path to the persona spec (`.hjson`).
    pub spec: PathBuf,
    /// The rendered image to measure against the spec.
    #[arg(long)]
    pub image: PathBuf,
}

pub async fn run(args: PersonaArgs) -> Result<()> {
    match args.cmd {
        PersonaCmd::New(a) => run_new(a),
        PersonaCmd::Lint(a) => run_lint(a),
        PersonaCmd::Show(a) => run_show(a),
        PersonaCmd::Verify(a) => run_verify(a).await,
    }
}

async fn run_verify(a: VerifyArgs) -> Result<()> {
    use crate::persona::scorecard;
    let spec = PersonaSpec::load(&a.spec)?;
    // Measurement runs on CPU (the aligner is tiny + deterministic; avoids GPU contention).
    let device = candle_core::Device::Cpu;
    let (detector, pipnet) = scorecard::load_probes(&device).await?;
    let metrics = scorecard::measure_landmarks(&a.image, &detector, &pipnet)?;
    let Some(m) = metrics else {
        println!("{}  no face detected in {}", style("✗").red(), a.image.display());
        return Ok(());
    };
    println!("{}  {}  (face score {:.2})", style("scorecard").bold(), a.image.display(), m.detection_score);
    println!("\n{}", style("landmark metrics (WFLW-98):").bold());
    println!("  interpupillary / face-width   {:.3}   ({})", m.interpupillary_over_facewidth, style("eyes.spacing").dim());
    println!("  mouth-width / face-width      {:.3}   ({})", m.mouth_over_facewidth, style("mouth.width").dim());
    println!("  face aspect (h/w)             {:.3}   ({})", m.face_aspect, style("face.width⁻¹").dim());

    // region_color probe: measured iris / skin CIELAB, and ΔE to the spec's eyes.color when given.
    let colors = scorecard::measure_colors(&m);
    println!("\n{}", style("region colour (CIELAB):").bold());
    let fmt = |l: Option<[f32; 3]>| l.map(|v| format!("L{:.0} a{:+.0} b{:+.0}", v[0], v[1], v[2])).unwrap_or_else(|| "—".into());
    let eye_target = spec.eyes.as_ref().and_then(|e| e.color.as_ref()).and_then(|c| match c {
        crate::persona::spec::Color::Named(n) => scorecard::color_name_to_lab(n),
        crate::persona::spec::Color::Lab { lab } => Some(*lab),
    });
    match (colors.iris, eye_target) {
        (Some(iris), Some(t)) => println!(
            "  iris  {}   → ΔE {:.1} to eyes.color target   {}",
            fmt(Some(iris)),
            scorecard::delta_e(iris, t),
            if scorecard::delta_e(iris, t) < 20.0 { style("✓").green() } else { style("✗").red() }
        ),
        (iris, _) => println!("  iris  {}   ({})", fmt(iris), style("eyes.color").dim()),
    }
    println!("  skin  {}   ({})", fmt(colors.skin), style("skin.tone").dim());

    // Detail sub-score: for each spec mark with an anchor, go to where it should be and probe.
    let marks = spec.marks.as_deref().unwrap_or(&[]);
    let positional: Vec<_> = marks.iter().filter(|mk| mk.anchor.is_some()).collect();
    if !positional.is_empty() {
        println!("\n{}", style("localized details (local_anomaly):").bold());
        for (i, mk) in positional.iter().enumerate() {
            let anchor = mk.anchor.as_ref().unwrap();
            let kind = mk.kind.as_deref().unwrap_or("mark");
            let Some(pos) = scorecard::resolve_anchor(anchor, &m) else {
                println!(
                    "  {} {kind}[{i}]: anchor {:?} not in the WFLW-98 vocabulary",
                    style("?").yellow(),
                    anchor.landmark.as_deref().or(anchor.region.as_deref()).unwrap_or("(none)")
                );
                continue;
            };
            let radius = mk.size.unwrap_or(0.05) * m.face_w; // mark size is a fraction of face width
            let r = scorecard::local_anomaly(&m.crop, pos, radius.max(0.02));
            let mark_glyph = if r.presence > 0.5 { style("✓").green() } else { style("✗").red() };
            println!(
                "  {mark_glyph} {kind}[{i}] @ {}  presence {:.2}  position-error {:.3}",
                style(anchor.landmark.as_deref().or(anchor.region.as_deref()).unwrap_or("?")).dim(),
                r.presence,
                r.position_error
            );
        }
    }
    println!("\n{}", style("note: scalar attribute-scoring needs the P4 calibration prior; raw metrics shown.").dim());
    Ok(())
}

fn run_show(a: ShowArgs) -> Result<()> {
    use crate::persona::compile::{self, EncoderClass};
    use crate::persona::lexicon::Lexicon;
    let spec = PersonaSpec::load(&a.spec)?;
    let lex = Lexicon::skeleton();
    let resolved = compile::resolve(&spec, &lex);
    let compiled = compile::compile_for_model(&spec, &lex, &a.model);

    println!("{}  {}  (model {}, encoder {})", style("persona").bold(), a.spec.display(), a.model, style(compiled.class).cyan());
    let _ = EncoderClass::from_model(&a.model);
    println!("\n{}", style("resolved attributes (salience-ranked):").bold());
    if resolved.is_empty() {
        println!("  (none — spec is empty or all-unknown)");
    }
    for r in &resolved {
        let emitted = compiled.emitted.iter().any(|p| p == &r.path);
        let mark = if r.phrase.is_empty() {
            style("neg").yellow()
        } else if emitted {
            style("✓").green()
        } else {
            style("dropped").red()
        };
        let phrase = if r.phrase.is_empty() { "(negative only)" } else { &r.phrase };
        println!("  {mark} {:<22} sal {:.2}  {}", style(&r.path).dim(), r.salience, phrase);
    }
    if !compiled.dropped.is_empty() {
        println!("\n{} {}", style("dropped by budget:").red(), compiled.dropped.join(", "));
    }
    println!("\n{}\n  {}", style("positive:").bold().green(), if compiled.positive.is_empty() { "(empty)" } else { &compiled.positive });
    println!("{}\n  {}", style("negative:").bold().yellow(), if compiled.negative.is_empty() { "(empty)" } else { &compiled.negative });
    Ok(())
}

fn run_new(a: NewArgs) -> Result<()> {
    if a.out.exists() {
        anyhow::bail!("{} already exists — refusing to overwrite", a.out.display());
    }
    let body = scaffold(&a.depth, &a.name, a.age)?;
    // Sanity: the scaffold must itself load + lint clean.
    let spec = PersonaSpec::from_hjson(&body)
        .with_context(|| "internal: generated scaffold did not parse")?;
    let findings = lint::lint(&spec);
    if lint::has_errors(&findings) {
        anyhow::bail!("internal: generated scaffold did not lint clean: {findings:?}");
    }
    if let Some(parent) = a.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    std::fs::write(&a.out, body).with_context(|| format!("writing {}", a.out.display()))?;
    println!("{}  persona scaffold ({})  →  {}", style("✓").green(), a.depth, a.out.display());
    println!("   edit it, or run `plakat persona lint {}`", a.out.display());
    Ok(())
}

fn run_lint(a: LintArgs) -> Result<()> {
    let spec = PersonaSpec::load(&a.spec)?;
    let findings = lint::lint(&spec);
    let (mut errs, mut warns, mut infos) = (0u32, 0u32, 0u32);
    for f in &findings {
        let (tag, styled) = match f.level {
            Level::Error => {
                errs += 1;
                ("error", style("error").red().bold())
            }
            Level::Warning => {
                warns += 1;
                ("warn", style("warn").yellow())
            }
            Level::Info => {
                infos += 1;
                ("info", style("info").cyan())
            }
        };
        let _ = tag;
        println!("  {styled} {}: {}", style(&f.path).bold(), f.message);
    }
    if findings.is_empty() {
        println!("{}  {} — clean", style("✓").green(), a.spec.display());
    } else {
        println!(
            "{}  {errs} error(s), {warns} warning(s), {infos} info",
            if errs > 0 { style("✗").red() } else { style("✓").green() },
        );
    }
    if errs > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// Build a valid, partial persona scaffold at the requested depth. HJSON is brace-based; quoteless
/// strings run to end-of-line, so every value sits alone on its line.
fn scaffold(depth: &str, name: &str, age: u32) -> Result<String> {
    let mut s = String::new();
    s.push_str("{\n  schema: persona/1\n\n");
    s.push_str("  identity: {\n");
    s.push_str(&format!("    name: {name}\n"));
    s.push_str(&format!("    apparent_age: {age}\n"));
    s.push_str("    sex: androgynous          # female | male | androgynous\n");
    s.push_str("  }\n\n");
    s.push_str("  # Scalars are 0..1 where 0.5 is the model's own prior; leave a field OUT to mean\n");
    s.push_str("  # \"unknown\" (an explicit 0.5 asserts \"deliberately average\"). See PERSONA.md.\n\n");
    s.push_str("  face: {\n    # shape: oval             # oval|round|square|heart|diamond|oblong|triangular\n");
    s.push_str("    # width: 0.5\n  }\n\n");
    s.push_str("  eyes: {\n    # color: hazel\n    # spacing: 0.5\n  }\n");

    if depth == "standard" || depth == "full" {
        s.push_str("\n  nose: {\n    # profile: straight\n  }\n");
        s.push_str("\n  mouth: {\n    # width: 0.5\n  }\n");
        s.push_str("\n  skin: {\n    # tone: fitzpatrick-3     # fitzpatrick-1..6\n  }\n");
        s.push_str("\n  hair: {\n    # color: auburn\n    # length: shoulder        # buzz|crop|short|chin|shoulder|long|very-long\n  }\n");
        s.push_str("\n  figure: {\n    # height_cm: 170\n    # build: mesomorph\n  }\n");
    }
    if depth == "full" {
        s.push_str("\n  teeth: {\n    # visibility: auto\n    # alignment: even\n  }\n");
        // Collections: an empty list ASSERTS \"none\"; omit the key entirely to mean \"undecided\".
        s.push_str("\n  # marks: []                 # [] asserts \"no marks\"; omit for \"undecided\"\n");
        s.push_str("  # piercings: []\n");
        s.push_str("\n  jewelry: {\n    # identity_locked: false\n    # items: []\n  }\n");
        s.push_str("\n  defaults: {\n    # expression: neutral\n    # framing: headshot\n  }\n");
    }
    if !matches!(depth, "quick" | "standard" | "full") {
        anyhow::bail!("unknown --depth {depth:?} (quick|standard|full)");
    }
    s.push_str("\n  provenance: {\n    method: manual\n    lexicon_version: \"1.0\"\n  }\n}\n");
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffolds_lint_clean_at_every_depth() {
        for depth in ["quick", "standard", "full"] {
            let body = scaffold(depth, "alice", 34).unwrap();
            let spec = PersonaSpec::from_hjson(&body)
                .unwrap_or_else(|e| panic!("depth {depth} did not parse: {e}\n{body}"));
            let f = lint::lint(&spec);
            assert!(!lint::has_errors(&f), "depth {depth} did not lint clean: {f:?}");
            assert_eq!(spec.schema_version(), Some(1));
            assert_eq!(spec.identity.unwrap().apparent_age, Some(34));
        }
    }

    #[test]
    fn scaffold_rejects_bad_depth() {
        assert!(scaffold("deep", "x", 30).is_err());
    }
}
