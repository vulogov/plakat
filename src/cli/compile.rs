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

#[derive(ClapArgs, Debug, Clone)]
pub struct CompileArgs {
    /// Re-compile whenever the input file changes (dev loop; pair with `--no-enhance` for instant). D3.
    #[arg(help_heading = "Compile", long, default_value_t = false)]
    pub watch: bool,
    /// Input `prompts.txt` (`-` reads stdin).
    #[arg(help_heading = "Compile", value_name = "INPUT")]
    pub input: PathBuf,

    /// Output scenario HJSON. Default: `<input-stem>.hjson` (`-` = stdout; also
    /// the default when reading stdin).
    #[arg(help_heading = "Size & output", long, value_name = "PATH")]
    pub out: Option<PathBuf>,

    /// LLM provider (reuses the `--enhance` stack): `deepseek`/`gemini`/`local`/
    /// `local:<alias>`/`auto`.
    #[arg(help_heading = "Enhancement", long = "compile-provider", default_value = "auto")]
    pub provider: String,

    /// Model used to pick the family prompt profile when no block names a model.
    #[arg(help_heading = "Model & sampler", long, default_value = "sdxl")]
    pub model: String,

    /// Override the positive-enhancement system prompt (file).
    #[arg(help_heading = "Enhancement", long = "compile-system", value_name = "PATH")]
    pub system: Option<PathBuf>,

    /// Skip the positive LLM call; assemble the prompt verbatim.
    #[arg(help_heading = "Enhancement", long = "no-enhance", default_value_t = false)]
    pub no_enhance: bool,

    /// Suppress the negative LLM call; seed terms pass through verbatim.
    #[arg(help_heading = "Enhancement", long = "no-negative", default_value_t = false)]
    pub no_negative: bool,

    /// Validate the input (unknown commands, misplaced `skip:`) and exit; no LLM.
    #[arg(help_heading = "Compile", long, default_value_t = false)]
    pub lint: bool,

    /// *(6.22 E1)* Validate that the INPUT is a loadable scenario HJSON (deserialises + known task types)
    /// and exit. A fast, no-model CI check for hand-written or compiled scenarios.
    #[arg(help_heading = "Compile", long, default_value_t = false)]
    pub check: bool,

    /// *(6.22 E3)* Print the resolved model family + the exact LLM system prompt per scene and exit; no
    /// LLM call. Debug why a prompt gets enhanced a certain way.
    #[arg(help_heading = "Compile", long, default_value_t = false)]
    pub explain: bool,

    /// Print a per-block summary (family, LLM call count) without calling the LLM.
    #[arg(help_heading = "Compile", long = "dry-run", default_value_t = false)]
    pub dry_run: bool,

    /// Max concurrent scenes when calling the LLM. `0` = auto (per provider:
    /// deepseek 3, gemini 5, local/auto 1).
    #[arg(help_heading = "Compile", long = "compile-parallel", value_name = "N", default_value_t = 1)]
    pub parallel: usize,

    /// Read/write the two-namespace LLM disk cache (`positive/` + `negative/`).
    #[arg(help_heading = "Compile", long = "compile-cache", default_value_t = false)]
    pub compile_cache: bool,

    /// Clear the compile cache and exit: `all` (default), `positive`, or `negative`.
    #[arg(help_heading = "Compile", long = "compile-cache-clear", value_name = "WHICH", num_args = 0..=1, default_missing_value = "all")]
    pub cache_clear: Option<String>,

    /// Inverse: read a scenario HJSON (the INPUT) and emit a `prompts.txt`.
    #[arg(help_heading = "Compile", long, default_value_t = false)]
    pub decompile: bool,

    /// Compare the freshly-compiled scenario against an existing HJSON; print the
    /// per-task add/change/remove diff instead of writing output.
    #[arg(help_heading = "Compile", long, value_name = "PATH")]
    pub diff: Option<PathBuf>,

    // ---- COMPILE-2: Tera template pre-pass (needs `--features templates`) ----
    /// Force the Tera template pre-pass regardless of file extension.
    #[arg(help_heading = "Templating", long, default_value_t = false)]
    pub template: bool,

    /// Inject a template variable `KEY=VALUE` (repeatable; highest precedence).
    #[arg(help_heading = "Templating", long = "var", value_name = "KEY=VALUE")]
    pub var: Vec<String>,

    /// Load template variables from a JSON or TOML file (repeatable; later wins).
    #[arg(help_heading = "Templating", long = "vars", value_name = "PATH")]
    pub vars: Vec<PathBuf>,

    /// Import env vars with PREFIX into the template context (prefix stripped, key
    /// lowercased: `PLAKAT_MODEL` → `{{ model }}`). Repeatable.
    #[arg(help_heading = "Templating", long = "vars-env", value_name = "PREFIX")]
    pub vars_env: Vec<String>,

    /// Write the rendered `prompts.txt` (before parsing) to PATH (`-` = stdout).
    #[arg(help_heading = "Templating", long = "dump-rendered", value_name = "PATH")]
    pub dump_rendered: Option<PathBuf>,

    /// Render the template, write it, and exit — no parse, no LLM.
    #[arg(help_heading = "Templating", long = "dump-rendered-only", default_value_t = false)]
    pub dump_rendered_only: bool,
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
    if args.watch {
        return watch(args).await;
    }
    run_inner(args).await
}

/// D3 (6.22.0) — `--watch`: compile once, then re-compile whenever the input file's mtime changes
/// (poll-based, no extra deps). Ctrl-C to stop. Needs a file input (not stdin).
async fn watch(mut args: CompileArgs) -> Result<()> {
    args.watch = false;
    anyhow::ensure!(args.input.as_os_str() != "-", "--watch needs a file input (not stdin)");
    let path = args.input.clone();
    let mtime = |p: &std::path::Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    println!("{}  watching {} — Ctrl-C to stop", style("👀").cyan(), path.display());
    let mut last = mtime(&path);
    loop {
        if let Err(e) = run_inner(args.clone()).await {
            eprintln!("{}  {e:#}", style("compile error:").red());
        }
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(400)).await;
            let now = mtime(&path);
            if now != last && now.is_some() {
                last = now;
                break;
            }
        }
    }
}

async fn run_inner(args: CompileArgs) -> Result<()> {
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

    let mut input = read_input(&args.input)?;
    // C4 (FACESWAP-4): inline `@include <path>` lines before anything else, relative to the input's dir
    // (CWD for stdin), so prose sets can be split across files.
    {
        let base = if args.input.as_os_str() == "-" {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        } else {
            args.input.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."))
        };
        if input.contains("@include") {
            input = compile::parser::expand_includes(&input, &base, 0)?;
        }
    }

    // --decompile: the INPUT is a scenario HJSON → emit a prompts.txt.
    // E1 (6.22): validate the INPUT is a loadable scenario HJSON, then exit (a no-model CI check).
    if args.check {
        crate::cli::scenario::validate_hjson(&input).context("scenario check failed")?;
        println!("{}  scenario is valid (loads · known task types)", style("✓").green());
        return Ok(());
    }

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

    // COMPILE-2: Tera template pre-pass (feature-gated). Fires BEFORE the parser —
    // a `.tera`/`.j2`/… input (or --template) renders to a prompts.txt string that
    // everything below then treats normally.
    let path = if stdin_input { None } else { Some(args.input.as_path()) };
    let input = if compile::should_use_template(path, args.template) {
        let mut vars = Vec::with_capacity(args.var.len());
        for s in &args.var {
            match s.split_once('=') {
                Some((k, v)) => vars.push((k.to_string(), v.to_string())),
                None => bail!("--var must be KEY=VALUE, got `{s}`"),
            }
        }
        let topts = compile::TemplateOpts {
            vars,
            vars_files: args.vars.clone(),
            env_prefixes: args.vars_env.clone(),
        };
        let rendered = compile::template::render(&input, path, &topts)?;
        if let Some(p) = &args.dump_rendered {
            if p.as_os_str() == "-" {
                print!("{rendered}");
            } else {
                std::fs::write(p, &rendered).with_context(|| format!("writing {}", p.display()))?;
                println!("{}  rendered → {}", style("✓").green(), p.display());
            }
        }
        if args.dump_rendered_only {
            if args.dump_rendered.is_none() {
                print!("{rendered}");
            }
            return Ok(());
        }
        rendered
    } else {
        input
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

    // --explain (E1/E3): resolve each scene and print the family + the exact system prompt; no LLM.
    if args.explain {
        let doc = compile::parser::parse(&input)?;
        let resolved = compile::resolver::resolve(&doc, &args.model)?;
        let sys_override = match &args.system {
            Some(p) => Some(std::fs::read_to_string(p).with_context(|| format!("reading --compile-system {}", p.display()))?),
            None => None,
        };
        for s in resolved.scenes.iter().filter(|s| !s.skip) {
            println!("{} scene {:?} · family {:?}", style("──").cyan(), s.name, s.family);
            println!("{}", style("[positive system]").dim());
            println!("{}\n", compile::assembler::positive_system(s, sys_override.as_deref(), &[]));
            // Also show the NEGATIVE system prompt + the resolved `negative:` seed terms, so a
            // surprising negative is diagnosable without a model call. Empty seeds → the auto-
            // negative is fully LLM-authored; non-empty seeds MUST survive into the output.
            println!("{}", style("[negative system]").dim());
            println!("{}", compile::assembler::negative_system(s));
            let seeds = s.negative_seeds.trim();
            println!(
                "{} negative seeds: {}\n",
                style("↳").cyan(),
                if seeds.is_empty() { "(none — auto-negative is fully LLM-authored)" } else { seeds }
            );
        }
        return Ok(());
    }

    // --dry-run: parse + resolve + summarize, no LLM.
    if args.dry_run {
        let doc = compile::parser::parse(&input)?;
        let resolved = compile::resolver::resolve(&doc, &args.model)?;
        println!("{}  compile dry-run · {input_name} · provider {}", style("◆").cyan(), args.provider);
        let (mut calls, mut tokens) = (0usize, 0usize);
        for s in &resolved.scenes {
            if s.skip {
                println!("  - {} [skipped]", s.name);
                continue;
            }
            let pos = if args.no_enhance { 0 } else { 1 };
            let neg = if args.no_negative { 0 } else { 1 };
            calls += pos + neg;
            // Rough token estimate: ~1 token per 4 chars of input + a typical
            // output budget per call (positive ~120, negative ~50).
            let assembled = compile::assembler::assemble_input(s);
            let est = assembled.len() / 4 + pos * 120 + neg * 50;
            tokens += est;
            println!(
                "  - {} · family {} · {} LLM call(s) · ~{est} tok",
                style(&s.name).bold(),
                s.family.label(),
                pos + neg
            );
        }
        let n_scenes = resolved.scenes.iter().filter(|s| !s.skip).count();
        println!(
            "  total: {n_scenes} scene(s) · {calls} LLM call(s) · ~{tokens} tokens (rough; cost depends on provider)"
        );
        return Ok(());
    }

    let system_override = match &args.system {
        Some(p) => Some(std::fs::read_to_string(p).with_context(|| format!("reading --compile-system {}", p.display()))?),
        None => None,
    };

    let (hjson, warnings) = compile::compile_to_string(
        &input,
        &CompileOpts {
            provider: args.provider.clone(),
            default_model: args.model.clone(),
            no_enhance: args.no_enhance,
            no_negative: args.no_negative,
            system_override,
            cache: args.compile_cache,
            parallel: args.parallel,
            input_name,
        },
    )
    .await?;

    // 6.26.2: surface per-scene diligence warnings (budget overflow / dropped style) to stderr —
    // never silently drop. These don't change the emitted scenario; they tell the user to act.
    for warning in &warnings {
        eprintln!("{}  {warning}", style("⚠").yellow().bold());
    }

    // C2 (FACESWAP-4): validate the emitted scenario is loadable (deserialises + known task types) before
    // writing — so a compiled scenario is guaranteed runnable, not just well-formed text.
    crate::cli::scenario::validate_hjson(&hjson).context("compiled scenario failed validation")?;

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
