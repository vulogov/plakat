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

    /// *(6.29)* Analyse the prose (any language) and write two files instead of compiling: a strategy file
    /// `composition_<NNN>.txt` (which pipeline suits the scene — pure sd / skeleton / skeleton+regional — plus
    /// scene-tuned naturalize), and `<stem>_optimized.txt` — a cleaner, less hallucination-prone rewrite of the
    /// prose IN ITS ORIGINAL LANGUAGE (all relationships/figures kept, a tailored negative) that `@include`s the
    /// strategy file. Advisory starting points to review + `@include` in your real prose.
    #[arg(help_heading = "Compile", long = "make-composition", default_value_t = false)]
    pub make_composition: bool,

    /// *(6.29)* Which DRAFT model `--make-composition` should pick when it selects a control-generate strategy
    /// (SKELETON / REGIONAL). The strategist emits `control-generate: <this>` and the model's native
    /// `control-generate-size` (1024×1024 for `sdxl`/any `xl` model, else 512×512). Default `sdxl`. Use
    /// `sd15` for the strongest pose control on seated/unusual poses.
    #[arg(help_heading = "Compile", long = "composition-model", value_name = "MODEL", default_value = "sdxl")]
    pub composition_model: String,

    /// *(6.29)* Analyse the prose (any language) and print a FEASIBILITY REPORT instead of compiling — a
    /// grade (N/10) plus the specific risks that make generation fail (over-stuffing, fused vehicle+trailer,
    /// person+cargo fusion, rare object names, negation-in-positive, hard poses, count ambiguity, …) each with
    /// a concrete fix. The polish-loop companion to `--make-composition`: iterate the prose until it's low-risk
    /// BEFORE spending tokens on a generation run. Writes nothing, generates nothing.
    #[arg(help_heading = "Compile", long, default_value_t = false)]
    pub analyze: bool,

    /// *(6.29)* With `--analyze`: also AUTO-APPLY the safe text fixes. Traces each offending phrase to the
    /// exact source `@include` file it lives in, backs that file up to `<file>.<N>` first, edits it in place
    /// (preserving each phrase's language — Russian stays Russian, English stays English), and reports what
    /// changed where. Structural issues (split the scene, control-preimage, config conflicts) are reported for
    /// you to handle, never auto-edited. No-op without `--analyze`.
    #[arg(help_heading = "Compile", long, default_value_t = false)]
    pub fix: bool,

    /// Compare the freshly-compiled scenario against an existing HJSON; print the
    /// per-task add/change/remove diff instead of writing output.
    #[arg(help_heading = "Compile", long, value_name = "PATH")]
    pub diff: Option<PathBuf>,

    /// *(6.30 smysl)* Also write a `<stem>.smysl` provenance sidecar beside the compiled scenario: the
    /// authored components + relations as smysl `@claim`/`@rel` units (the AI→AI→Human corpus). Opt-in;
    /// the HJSON output is byte-identical. Reviewable, and the substrate `--trace` queries.
    #[arg(help_heading = "Compile", long, default_value_t = false)]
    pub smysl: bool,

    /// *(6.30 smysl)* Answer "why is this phrase in the prompt?" — no LLM, no generation. Resolves the
    /// scene to its smysl claims/relations (plus any `<stem>.smysl` fix-corpus beside the input) and
    /// prints every unit that mentions PHRASE with its confidence + provenance (grounds chain, relations).
    #[arg(help_heading = "Compile", long, value_name = "PHRASE")]
    pub trace: Option<String>,

    /// *(6.30 smysl-optimize)* AUTOMATIC improve loop: compile the first scene, render + aesthetically score
    /// it, then let an LLM regenerator propose one prompt edit at a time — keeping only edits that raise the
    /// score, and using the `<stem>.smysl` corpus as a TABU list so it never re-tries a spent move. Stops on
    /// a plateau / the pass budget / a dry proposer, always reporting the best prompt. Renders images (needs a
    /// model); budget it with `--improve-passes`.
    #[arg(help_heading = "Compile", long, default_value_t = false)]
    pub improve: bool,

    /// *(6.30)* `--improve` pass budget — the maximum number of edits to try. Each pass renders + scores.
    #[arg(help_heading = "Compile", long = "improve-passes", value_name = "N", default_value_t = 6)]
    pub improve_passes: usize,

    /// *(6.30)* Model `--improve` renders with while scoring (defaults to `--model`). Use `sd15` for fast,
    /// cheap passes; `sdxl` for a truer preview of the final look.
    #[arg(help_heading = "Compile", long = "improve-model", value_name = "MODEL")]
    pub improve_model: Option<String>,

    /// *(6.30)* Base seed `--improve` renders at, so every pass scores the *same* composition(s) and the rank
    /// delta reflects the prompt edit, not a fresh roll. With `--improve-seeds > 1`, the seed set is
    /// `SEED, SEED+1, …`.
    #[arg(help_heading = "Compile", long = "improve-seed", value_name = "SEED", default_value_t = 1000)]
    pub improve_seed: u64,

    /// *(6.30)* Improve EVERY non-skipped scene, not just the first. Each composition is optimized
    /// independently (own prompt, negative, best, tabu). Budget-heavy — roughly scenes × (1+passes) × seeds
    /// renders — so it prints the render count up front. Implies `--improve`.
    #[arg(help_heading = "Compile", long = "improve-all", default_value_t = false)]
    pub improve_all: bool,

    /// *(6.30)* Improve only the named scene/composition instead of the first. Implies `--improve`.
    #[arg(help_heading = "Compile", long = "improve-scene", value_name = "NAME")]
    pub improve_scene: Option<String>,

    /// *(6.30 polish)* CORROBORATION — how many seeds to render + score per candidate, averaged into its
    /// rank. Aesthetic score is noisy, so `1` seed can teach the tabu list garbage; `2`–`3` makes a
    /// kept/rejected verdict robust, at N× the render cost. Every candidate is judged on the SAME seed set.
    #[arg(help_heading = "Compile", long = "improve-seeds", value_name = "N", default_value_t = 2)]
    pub improve_seeds: u32,

    /// *(6.30 polish)* The noise floor: a candidate is *kept* only if its (corroborated) rank beats the
    /// best-so-far by more than this. Raise it if the loop keeps chasing insignificant wobble.
    #[arg(help_heading = "Compile", long = "improve-min-gain", value_name = "F", default_value_t = 0.1)]
    pub improve_min_gain: f32,

    /// *(6.30 polish)* Stop after this many passes in a row with no gain above `--improve-min-gain`.
    #[arg(help_heading = "Compile", long = "improve-plateau", value_name = "K", default_value_t = 3)]
    pub improve_plateau: usize,

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
            // Show the DETERMINISTIC negative (seeds + curated quality set, deduped/capped) — no model call,
            // no hallucinated content exclusions.
            println!("{}", style("[negative (deterministic)]").dim());
            println!("{}", compile::assembler::auto_negative(s));
            let seeds = s.negative_seeds.trim();
            println!(
                "{} negative seeds: {}\n",
                style("↳").cyan(),
                if seeds.is_empty() { "(none — quality terms only)" } else { seeds }
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

    // --trace "<phrase>": explain why a phrase is in the prompt (no LLM, no generation). Resolves the
    // scene to smysl units + folds in any `<stem>.smysl` fix-corpus beside the input, then reports.
    if let Some(phrase) = &args.trace {
        let corpus = if stdin_input { None } else { Some(args.input.with_extension("smysl")) };
        let report = compile::trace_prose(&input, &args.model, phrase, corpus.as_deref())?;
        println!("{}", report.trim_end());
        return Ok(());
    }

    // --improve[-all|-scene]: the automatic aesthetic improve loop (Phase D). Renders + scores, needs a model.
    if args.improve || args.improve_all || args.improve_scene.is_some() {
        anyhow::ensure!(!stdin_input, "--improve needs a file input (not stdin)");
        return improve_cmd(&args, &input).await;
    }

    let system_override = match &args.system {
        Some(p) => Some(std::fs::read_to_string(p).with_context(|| format!("reading --compile-system {}", p.display()))?),
        None => None,
    };

    // --analyze: print a feasibility report (grade + risks + fixes) and stop. The polish-loop companion to
    // --make-composition — iterate the prose until it's low-risk before spending a generation run.
    if args.analyze {
        let opts = CompileOpts {
            provider: args.provider.clone(),
            default_model: args.model.clone(),
            no_enhance: args.no_enhance,
            no_negative: args.no_negative,
            system_override: system_override.clone(),
            cache: args.compile_cache,
            parallel: args.parallel,
            input_name: input_name.clone(),
        };
        // Feed the critic the prior polish history (resolved/open findings) so it doesn't re-flag risks
        // earlier passes already fixed.
        let prior_corpus = if stdin_input {
            None
        } else {
            std::fs::read_to_string(args.input.with_extension("smysl")).ok()
        };
        let report = compile::analyze_prose(&input, &opts, prior_corpus.as_deref()).await?;
        println!("{}", report.trim());
        if args.fix {
            anyhow::ensure!(!stdin_input, "--fix needs a file input (not stdin) so it can edit + back up the source");
            println!("\n{}  applying safe auto-fixes to the source prose…", style("--fix").cyan());
            let fixreport = compile::apply_fixes(&args.input, &opts, &report).await?;
            println!("{}", fixreport.trim());
            println!("\n{}  re-run `--analyze` to confirm the grade improved.", style("→").dim());
        }
        return Ok(());
    }

    // --make-composition: analyse the prose with two LLM passes and write a strategy file + an optimized
    // prose that @includes it. Analysis-only: emit the pair and stop (the author then compiles the optimized).
    if args.make_composition {
        anyhow::ensure!(!stdin_input, "--make-composition needs a file input (not stdin)");
        let opts = CompileOpts {
            provider: args.provider.clone(),
            default_model: args.model.clone(),
            no_enhance: args.no_enhance,
            no_negative: args.no_negative,
            system_override,
            cache: args.compile_cache,
            parallel: args.parallel,
            input_name: input_name.clone(),
        };
        let (comp, opt) = compile::make_composition(&input, &args.input, &opts, &args.composition_model).await?;
        println!("{}  strategy    → {}", style("✓").green(), comp.display());
        println!("{}  optimized   → {}  (@includes {})", style("✓").green(), opt.display(),
            comp.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        println!("   review both, then: {} scenario {}", style("plakat compile").dim(), opt.display());
        return Ok(());
    }

    let (hjson, warnings, trace, provenance) = compile::compile_to_string(
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

    // 6.27: show WHAT the pipeline did per scene (translate, compose, weights, enhance, negative, fit) —
    // to stderr so piping the HJSON is unaffected. Header lines (no indent) are cyan, steps dim.
    for line in &trace {
        if line.starts_with("  ") {
            eprintln!("{}", style(line).dim());
        } else {
            eprintln!("{} {line}", style("◆").cyan());
        }
    }

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
            // --smysl: write the provenance sidecar beside the scenario (best-effort; opt-in).
            if args.smysl {
                let sidecar = path.with_extension("smysl");
                // Finalize onto any corpus already born at `--fix` (input-side) or a prior compile.
                let prior = if args.input.as_os_str() == "-" { None } else { Some(args.input.with_extension("smysl")) };
                let merge_with = prior.as_deref().filter(|p| p.exists()).or(Some(sidecar.as_path()));
                // The prior corpus (before this run) — for linking recompilation drift.
                let prior_text = merge_with.and_then(|p| std::fs::read_to_string(p).ok()).unwrap_or_default();
                match compile::compose_scene_smysl(&input, &args.model, merge_with) {
                    Ok(doc) => {
                        // Fold in the compile's provenance — budget-pack decisions AND the prose→emitted
                        // transformation (the "trace BUDGETS" + "trace the ENHANCE step" halves).
                        let doc = if provenance.trim().is_empty() {
                            doc
                        } else {
                            crate::smysl::merge_surface(&doc, &provenance)
                        };
                        // Recompilation drift: Supersedes edges from this run's emitted-prompt claims to the
                        // prior ones for the same prose — so the corpus versions the enhancer's rewrites.
                        let sup = crate::smysl::supersedes_edges(&provenance, &prior_text);
                        let doc = if sup.is_empty() { doc } else { crate::smysl::merge_relations(&doc, sup) };
                        std::fs::write(&sidecar, &doc)
                            .with_context(|| format!("writing {}", sidecar.display()))?;
                        println!("{}  smysl       → {}", style("✓").green(), sidecar.display());
                    }
                    Err(e) => eprintln!("{}  smysl sidecar skipped: {e:#}", style("⚠").yellow()),
                }
            }
            Ok(())
        }
    }
}

// ===========================================================================
//  smysl-optimize Phase D — the `compile --improve` automatic aesthetic loop.
//  The live ImproveStep: an LLM regenerator proposes one prompt edit, each
//  candidate is rendered + aesthetically scored, and the tested controller
//  (compile::improve) keeps only improvements while the smysl corpus is the
//  tabu memory so no spent move is re-tried.
// ===========================================================================

/// System prompt for the regenerator — proposes ONE aesthetic edit as JSON. Steered toward concrete
/// aesthetic *levers* (light, composition, depth, palette, mood) and explicitly away from meaningless
/// synonym swaps, which a weak model otherwise loves and which never move the image.
const REGEN_SYSTEM: &str = "You optimize a text-to-image PROMPT for higher AESTHETIC quality of the \
    RENDERED IMAGE — lighting, composition, colour harmony, depth, mood, and focal clarity. Propose exactly \
    ONE small VERBATIM edit: replace an existing substring with a better one, WITHOUT changing the subject \
    or adding new subjects.\n\
    PREFER an edit that adds a concrete aesthetic lever a photographer or painter would reach for, e.g.:\n\
    - LIGHT: 'golden hour', 'soft volumetric light', 'rim lighting', 'warm key light with cool fill'\n\
    - COMPOSITION: 'rule-of-thirds framing', 'strong leading lines', 'balanced negative space'\n\
    - DEPTH: 'shallow depth of field, soft bokeh', 'atmospheric depth, layered fog'\n\
    - PALETTE / MOOD: 'muted complementary palette', 'rich cinematic colour grade'\n\
    AVOID trivial SYNONYM swaps that do not change the image (e.g. 'wet'->'damp', 'street'->'lane') — they \
    waste a render. Make the edit MEAN something visually.\n\
    Output ONLY a JSON object {\"old\":\"…\",\"new\":\"…\"} where \"old\" is an EXACT substring of the \
    prompt; output {} if you have no confident improvement. NEVER propose an edit the message lists as \
    already tried, and never reverse one.";

/// The live improve step: render + aesthetic-score for `rank`, LLM regenerator for `propose`.
struct LiveStep {
    model: String,
    negative: String,
    width: u32,
    height: u32,
    /// The corroboration seed set — every candidate is rendered + scored on ALL of these and averaged, so a
    /// kept/rejected verdict is robust to the aesthetic score's per-seed noise.
    seeds: Vec<u64>,
    provider: String,
    eargs: crate::prompt::EnhanceArgs,
    scorer: crate::pipelines::aesthetic::AestheticScorer,
    /// The RENDER pipeline, loaded ONCE and reused for every render (SD-family). `None` → the one-shot
    /// `api::Generate` fallback (reloads per render) for non-SD-family models (sd35/Flux).
    pipe: Option<crate::pipelines::t2i::Pipeline>,
    tmp: std::path::PathBuf,
    last_rank: f32,
}

impl crate::compile::improve::ImproveStep for LiveStep {
    fn propose<'a>(
        &'a mut self,
        prompt: &'a str,
        tabu: &'a [(String, String)],
    ) -> crate::compile::improve::StepFut<'a, Option<(String, String)>> {
        Box::pin(async move {
            eprintln!("    · asking the regenerator for one edit…");
            let hint = crate::smysl::tabu_hint(tabu);
            let user = format!(
                "IMAGE PROMPT (current aesthetic score {:.2}):\n{prompt}\n\n{hint}\n\nPropose ONE small \
                 verbatim edit to raise the aesthetic quality.",
                self.last_rank,
            );
            let out = crate::prompt::complete(&self.provider, REGEN_SYSTEM, &user, &self.eargs).await.ok()?;
            let delta = parse_delta(&out, prompt);
            match &delta {
                Some((o, n)) => eprintln!("    · proposed: \u{201C}{o}\u{201D} \u{2192} \u{201C}{n}\u{201D}"),
                None => eprintln!("    · no confident edit proposed"),
            }
            delta
        })
    }

    fn rank<'a>(&'a mut self, prompt: &'a str) -> crate::compile::improve::StepFut<'a, anyhow::Result<f32>> {
        Box::pin(async move {
            eprintln!(
                "    · rendering + scoring ({}, {}\u{00D7}{}, {} seed{})…",
                self.model,
                self.width,
                self.height,
                self.seeds.len(),
                if self.seeds.len() == 1 { "" } else { "s" },
            );
            // Corroboration: render + score EVERY seed, average — robust to per-seed noise.
            let mut scores = Vec::with_capacity(self.seeds.len());
            for (i, &seed) in self.seeds.iter().enumerate() {
                let path = self.tmp.join(format!("improve-{i}.png"));
                if let Some(pipe) = &self.pipe {
                    // Resident SD-family pipeline — loaded ONCE (no per-render model reload).
                    crate::cli::scenario::draft_generate(
                        pipe,
                        prompt,
                        self.negative.as_str(),
                        self.width,
                        self.height,
                        seed,
                        &path,
                        &[],
                        &[],
                    )?;
                } else {
                    // Fallback: one-shot render (reloads the model each call) for non-SD-family models.
                    let images = crate::api::Generate::new(self.model.as_str())
                        .prompt(prompt)
                        .negative(self.negative.as_str())
                        .size(self.width, self.height)
                        .seed(seed)
                        .count(1)
                        .run()
                        .await?;
                    let img = images
                        .into_iter()
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--improve: render produced no image"))?;
                    img.save(&path)?;
                }
                scores.push(self.scorer.score_path(&path)?);
            }
            let mean = scores.iter().sum::<f32>() / scores.len().max(1) as f32;
            self.last_rank = mean;
            if scores.len() > 1 {
                let spread = scores.iter().cloned().fold(f32::MIN, f32::max)
                    - scores.iter().cloned().fold(f32::MAX, f32::min);
                eprintln!("    · aesthetic {mean:.2} (mean of {}, spread {spread:.2})", scores.len());
            } else {
                eprintln!("    · aesthetic {mean:.2}");
            }
            Ok(mean)
        })
    }
}

/// Extract a `{"old":…, "new":…}` delta from the regenerator's reply. Returns `None` (no fresh move) when
/// the JSON is empty/absent, `old` is not a verbatim substring of the prompt, or the edit is a no-op.
fn parse_delta(out: &str, prompt: &str) -> Option<(String, String)> {
    let (a, b) = (out.find('{')?, out.rfind('}')?);
    if b < a {
        return None;
    }
    #[derive(serde::Deserialize)]
    struct D {
        #[serde(default)]
        old: String,
        #[serde(default)]
        new: String,
    }
    let d: D = serde_json::from_str(&out[a..=b]).ok()?;
    let (old, new) = (d.old.trim(), d.new.trim());
    if old.is_empty() || old == new || !prompt.contains(old) {
        return None;
    }
    Some((old.to_string(), new.to_string()))
}

/// Merge the loop's tried deltas into the `<stem>.smysl` corpus as tabu moves (accumulate, content-hash
/// dedup) so the next `--improve`/`--fix` run starts already knowing them.
fn persist_improve_corpus(corpus: &std::path::Path, moves: &[(String, String, String)]) -> Result<()> {
    let (mut recs, mut labels) = if corpus.exists() {
        let text = std::fs::read_to_string(corpus)?;
        match smysl_core::surface::parse_surface(&text) {
            Ok(p) => (p.records, p.labels),
            Err(_) => (Vec::new(), std::collections::BTreeMap::new()),
        }
    } else {
        (Vec::new(), std::collections::BTreeMap::new())
    };
    let (nr, nl) = crate::smysl::fixes_to_records(moves, &[])?;
    crate::smysl::merge_records(&mut recs, &mut labels, nr, nl);
    std::fs::write(corpus, crate::smysl::records_to_surface(&recs, &labels))
        .with_context(|| format!("--improve: writing corpus {}", corpus.display()))?;
    Ok(())
}

/// `compile --improve`: run the automatic aesthetic improve loop and report the best prompt found.
async fn improve_cmd(args: &CompileArgs, input: &str) -> Result<()> {
    let input_name = args.input.file_name().and_then(|n| n.to_str()).unwrap_or("prompts.txt").to_string();
    let opts = CompileOpts {
        provider: args.provider.clone(),
        default_model: args.model.clone(),
        no_enhance: args.no_enhance,
        no_negative: args.no_negative,
        system_override: None,
        cache: args.compile_cache,
        parallel: args.parallel,
        input_name,
    };

    // 1. Compile every scene, then SELECT the scope: a named scene, all scenes, or (default) the first.
    let all = compile::all_prompts(input, &opts).await?;
    anyhow::ensure!(!all.is_empty(), "--improve: no renderable scenes in {}", args.input.display());
    let selected: Vec<(String, String, String)> = if let Some(name) = &args.improve_scene {
        let hit: Vec<_> = all.iter().filter(|(n, _, _)| n.eq_ignore_ascii_case(name.trim())).cloned().collect();
        anyhow::ensure!(
            !hit.is_empty(),
            "--improve-scene: no scene named {name:?} — scenes are: {}",
            all.iter().map(|(n, _, _)| n.as_str()).collect::<Vec<_>>().join(", "),
        );
        hit
    } else if args.improve_all {
        all
    } else {
        all.into_iter().take(1).collect() // default: the first scene
    };

    // 2. Load the aesthetic scorer ONCE (the expensive load) + resolve the render model / size / seeds.
    let model = args.improve_model.clone().unwrap_or_else(|| args.model.clone());
    let res = crate::capability::native_res(&model);
    let device = crate::device::select("auto")?;
    let scorer = crate::pipelines::aesthetic::AestheticScorer::load(&device)
        .await
        .context("--improve: loading the aesthetic scorer (CLIP ViT-L/14 + LAION predictor)")?;
    // Load the RENDER model ONCE and keep it resident — reloading the SD core per render was the bug. SD
    // family (sd15/sdxl/…) loads here; other families (sd35/Flux) fall back to the one-shot per-render path.
    let pipe = match crate::pipelines::t2i::Pipeline::load(crate::pipelines::t2i::LoadRequest {
        model: model.clone(),
        device: device.clone(),
        loras: Vec::new(),
        lora_scale: 1.0,
        use_refiner: false,
        embeddings: Vec::new(),
        vae_cache: None,
    })
    .await
    {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!(
                "{}  {model} is not an SD-family model for the resident render loop ({e}) — falling back to \
                 per-render loading (slow). Use --improve-model sdxl or sd15 for the fast path.",
                style("⚠").yellow(),
            );
            None
        }
    };
    let tmp = std::env::temp_dir().join(format!("plakat-improve-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).with_context(|| format!("--improve: creating {}", tmp.display()))?;
    let n_seeds = args.improve_seeds.max(1) as u64;
    let seeds: Vec<u64> = (0..n_seeds).map(|i| args.improve_seed + i).collect();
    let corpus_path = args.input.with_extension("smysl");

    // Up-front cost, so a fan-out over many scenes is never a surprise.
    let est_renders = selected.len() * (1 + args.improve_passes) * seeds.len();
    println!(
        "{}  improving {} scene(s) — up to ~{} renders · model {model} · {} seed{} from {} · {} pass(es) · keep-gain {:.2}",
        style("◆").cyan(),
        selected.len(),
        est_renders,
        seeds.len(),
        if seeds.len() == 1 { "" } else { "s" },
        args.improve_seed,
        args.improve_passes,
        args.improve_min_gain,
    );

    // 3. One LiveStep, reused across scenes (scorer loaded once); negative + last_rank reset per scene.
    let mut step = LiveStep {
        model,
        negative: String::new(),
        width: res,
        height: res,
        seeds,
        provider: args.provider.clone(),
        eargs: crate::prompt::EnhanceArgs::default(),
        scorer,
        pipe,
        tmp: tmp.clone(),
        last_rank: 0.0,
    };

    let mut results: Vec<(String, compile::improve::ImproveOutcome)> = Vec::new();
    for (name, prompt0, negative) in &selected {
        println!("\n{} scene {name}", style("──").cyan());
        // Each scene is an INDEPENDENT optimization: its own negative, its own baseline, its own tabu seeded
        // from the (shared, content-addressed) corpus — a scene's edits never collide with another's.
        step.negative = negative.clone();
        step.last_rank = 0.0;
        let seed_tabu = std::fs::read_to_string(&corpus_path)
            .ok()
            .map(|t| crate::smysl::prior_fixes(&t))
            .unwrap_or_default();
        let out = compile::improve::run_improve(
            &mut step,
            prompt0,
            seed_tabu,
            args.improve_passes,
            args.improve_plateau.max(1),
            args.improve_min_gain,
        )
        .await?;
        // Persist this scene's tried deltas into the shared corpus (tabu memory for next time).
        let moves = out.corpus_moves();
        if !moves.is_empty() {
            if let Err(e) = persist_improve_corpus(&corpus_path, &moves) {
                eprintln!("{}  smysl corpus not updated: {e:#}", style("⚠").yellow());
            }
        }
        results.push((name.clone(), out));
    }
    let _ = std::fs::remove_dir_all(&tmp);

    // 4. Report per scene.
    for (name, out) in &results {
        println!("\n{} scene {name}", style("✓").green());
        print!("{}", compile::improve::format_report(out));
        println!("  best prompt (aesthetic {:.2}):\n{}", out.best_rank, out.best_prompt);
    }
    Ok(())
}
