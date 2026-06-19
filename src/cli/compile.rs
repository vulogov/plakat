//! `plakat compile` — compile a prose `prompts.txt` into a `scenario` HJSON.
//!
//! Write scenes as natural-language paragraphs with optional `key: value`
//! commands; compile rewrites each through the LLM provider stack (family-aware)
//! and emits a ready-to-run scenario. `--no-enhance --no-negative` is fully
//! deterministic (no LLM) — the path the corpus proof exercises.

use anyhow::{Context, Result, bail};
use clap::Args as ClapArgs;
use console::style;
use std::io::Read;
use std::path::PathBuf;

use crate::compile::{self, CompileOpts};

#[derive(ClapArgs, Debug)]
pub struct CompileArgs {
    /// Input `prompts.txt` (`-` reads stdin).
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Output scenario HJSON. Default: `<input-stem>.hjson` (`-` = stdout; also
    /// the default when reading stdin).
    #[arg(long, value_name = "PATH")]
    pub out: Option<PathBuf>,

    /// LLM provider (reuses the `--enhance` stack): `deepseek`/`gemini`/`local`/
    /// `local:<alias>`/`auto`.
    #[arg(long = "compile-provider", default_value = "auto")]
    pub provider: String,

    /// Model used to pick the family prompt profile when no block names a model.
    #[arg(long, default_value = "sdxl")]
    pub model: String,

    /// Override the positive-enhancement system prompt (file).
    #[arg(long = "compile-system", value_name = "PATH")]
    pub system: Option<PathBuf>,

    /// Skip the positive LLM call; assemble the prompt verbatim.
    #[arg(long = "no-enhance", default_value_t = false)]
    pub no_enhance: bool,

    /// Suppress the negative LLM call; seed terms pass through verbatim.
    #[arg(long = "no-negative", default_value_t = false)]
    pub no_negative: bool,

    /// Validate the input (unknown commands, misplaced `skip:`) and exit; no LLM.
    #[arg(long, default_value_t = false)]
    pub lint: bool,

    /// Print a per-block summary (family, LLM call count) without calling the LLM.
    #[arg(long = "dry-run", default_value_t = false)]
    pub dry_run: bool,

    /// Read/write the two-namespace LLM disk cache (`positive/` + `negative/`).
    #[arg(long = "compile-cache", default_value_t = false)]
    pub compile_cache: bool,

    /// Clear the compile cache and exit: `all` (default), `positive`, or `negative`.
    #[arg(long = "compile-cache-clear", value_name = "WHICH", num_args = 0..=1, default_missing_value = "all")]
    pub cache_clear: Option<String>,

    /// Inverse: read a scenario HJSON (the INPUT) and emit a `prompts.txt`.
    #[arg(long, default_value_t = false)]
    pub decompile: bool,

    /// Compare the freshly-compiled scenario against an existing HJSON; print the
    /// per-task add/change/remove diff instead of writing output.
    #[arg(long, value_name = "PATH")]
    pub diff: Option<PathBuf>,
}

fn read_input(path: &std::path::Path) -> Result<String> {
    if path.as_os_str() == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s).context("reading stdin")?;
        Ok(s)
    } else {
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))
    }
}

pub async fn run(args: CompileArgs) -> Result<()> {
    // --compile-cache-clear: wipe the cache and exit (before reading input).
    if let Some(which) = &args.cache_clear {
        let ns = match which.as_str() {
            "positive" => Some(compile::cache::POSITIVE),
            "negative" => Some(compile::cache::NEGATIVE),
            "all" | "" => None,
            other => bail!("--compile-cache-clear: expected all|positive|negative, got `{other}`"),
        };
        let n = compile::cache::clear(ns);
        println!("{}  cleared {n} compile cache entries", style("✓").green());
        return Ok(());
    }

    let input = read_input(&args.input)?;

    // --decompile: the INPUT is a scenario HJSON → emit a prompts.txt.
    if args.decompile {
        let txt = compile::scenario_read::decompile(&input)?;
        match &args.out {
            Some(p) if p.as_os_str() != "-" => {
                std::fs::write(p, &txt).with_context(|| format!("writing {}", p.display()))?;
                println!("{}  decompiled → {}", style("✓").green(), p.display());
            }
            _ => print!("{txt}"),
        }
        return Ok(());
    }

    let stdin_input = args.input.as_os_str() == "-";
    let input_name = if stdin_input {
        "<stdin>".to_string()
    } else {
        args.input.file_name().and_then(|n| n.to_str()).unwrap_or("prompts.txt").to_string()
    };

    // --lint: validate and exit (non-zero on issues, for CI).
    if args.lint {
        let issues = compile::lint(&input)?;
        if issues.is_empty() {
            println!("{}  no issues", style("✓").green());
            return Ok(());
        }
        for i in &issues {
            eprintln!("{}  {i}", style("✗").red());
        }
        bail!("compile --lint: {} issue(s)", issues.len());
    }

    // --dry-run: parse + resolve + summarize, no LLM.
    if args.dry_run {
        let doc = compile::parser::parse(&input)?;
        let resolved = compile::resolver::resolve(&doc, &args.model)?;
        println!("{}  compile dry-run · {input_name} · provider {}", style("◆").cyan(), args.provider);
        let mut calls = 0usize;
        for s in &resolved.scenes {
            if s.skip {
                println!("  - {} [skipped]", s.name);
                continue;
            }
            let pos = if args.no_enhance { 0 } else { 1 };
            let neg = if args.no_negative { 0 } else { 1 };
            calls += pos + neg;
            println!(
                "  - {} · family {} · {} LLM call(s)",
                style(&s.name).bold(),
                s.family.label(),
                pos + neg
            );
        }
        println!("  total: {} scene(s) · {calls} LLM call(s)", resolved.scenes.iter().filter(|s| !s.skip).count());
        return Ok(());
    }

    let system_override = match &args.system {
        Some(p) => Some(std::fs::read_to_string(p).with_context(|| format!("reading --compile-system {}", p.display()))?),
        None => None,
    };

    let hjson = compile::compile_to_string(
        &input,
        &CompileOpts {
            provider: args.provider.clone(),
            default_model: args.model.clone(),
            no_enhance: args.no_enhance,
            no_negative: args.no_negative,
            system_override,
            cache: args.compile_cache,
            input_name,
        },
    )
    .await?;

    // --diff: compare against an existing scenario instead of writing.
    if let Some(existing) = &args.diff {
        let prev = std::fs::read_to_string(existing)
            .with_context(|| format!("reading --diff target {}", existing.display()))?;
        let report = compile::scenario_read::diff(&hjson, &prev)?;
        print!("{report}");
        return Ok(());
    }

    // Resolve output target: --out, else stdout for stdin, else <stem>.hjson.
    let out: Option<PathBuf> = match &args.out {
        Some(p) if p.as_os_str() == "-" => None,
        Some(p) => Some(p.clone()),
        None if stdin_input => None,
        None => Some(args.input.with_extension("hjson")),
    };

    match out {
        None => {
            print!("{hjson}");
            Ok(())
        }
        Some(path) => {
            std::fs::write(&path, &hjson).with_context(|| format!("writing {}", path.display()))?;
            println!("{}  compiled → {}", style("✓").green(), path.display());
            Ok(())
        }
    }
}
