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

pub fn run(args: PersonaArgs) -> Result<()> {
    match args.cmd {
        PersonaCmd::New(a) => run_new(a),
        PersonaCmd::Lint(a) => run_lint(a),
    }
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
