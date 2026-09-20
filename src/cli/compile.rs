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

    /// *(6.34)* LAYERED-aware compilation. A scene with several INDEPENDENT foreground subjects (≥2 hero
    /// figures that don't physically interact — the fusion-prone case) is auto-decomposed into a LAYERED-1
    /// plan: a backdrop plus one subject layer per figure, each geometrically placed. compile emits a
    /// `type: layered` task and writes the derived plan as a `<scene>.layered.hjson` sidecar next to the
    /// scenario. Deterministic (no LLM); needs a file output (`--out`), not stdout. Off by default.
    #[arg(help_heading = "Compile", long, default_value_t = false)]
    pub layered: bool,

    /// *(6.29)* With `--analyze`: also AUTO-APPLY the safe text fixes. Traces each offending phrase to the
    /// exact source `@include` file it lives in, backs that file up to `<file>.<N>` first, edits it in place
    /// (preserving each phrase's language — Russian stays Russian, English stays English), and reports what
    /// changed where. Structural issues (split the scene, control-preimage, config conflicts) are reported for
    /// you to handle, never auto-edited. No-op without `--analyze`.
    #[arg(help_heading = "Compile", long, default_value_t = false)]
    pub fix: bool,

    /// *(6.32, Thread C)* With `--fix`, also fold MEASURED AESTHETIC WINS from the smysl corpus into the
    /// prose — but only a win whose original phrase appears VERBATIM in the source (an exact map). A prompt
    /// win that doesn't map is left as a critic suggestion, never force-applied. Opt-in: the prose is yours.
    #[arg(help_heading = "Compile", long = "fix-wins", default_value_t = false)]
    pub fix_wins: bool,

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

    /// *(6.32)* Print a human DIGEST of the prose's `<stem>.smysl` corpus — no LLM, no render: per-scene best
    /// aesthetic rank, the measured wins, the rejected (tabu) moves, and resolved vs open findings. "The
    /// corpus IS the process," made legible.
    #[arg(help_heading = "Compile", long = "smysl-report", default_value_t = false)]
    pub smysl_report: bool,

    /// *(6.32)* Write the smysl corpus digest as a self-contained, styled HTML page to this path (shareable
    /// provenance report). Same data as `--smysl-report`; no LLM, no render.
    #[arg(help_heading = "Compile", long = "smysl-html")]
    pub smysl_html: Option<PathBuf>,

    /// *(6.32)* PREFLIGHT: compile (no render) and run deterministic feasibility checks — LoRA trigger present
    /// in the prompt, budget fit, sane steps/guidance, non-empty prompt, region sanity. Prints PASS/WARN/FAIL
    /// per scene and exits non-zero on any FAIL, so it gates a render run in CI. No LLM.
    #[arg(help_heading = "Compile", long = "preflight", default_value_t = false)]
    pub preflight: bool,

    /// *(6.32)* MATRIX: expand every scene along one or more axes into the cartesian product of variations
    /// → an N-task scenario. `--matrix "weather=clear,storm,fog"` appends each value to the prompt;
    /// `--matrix "set.model=sdxl,sd35"` sets a per-scene directive. Repeatable. Cells are named
    /// `<scene>__<axis>-<value>…`. Run the result with `plakat scenario` (a grid/contact of all cells).
    #[arg(help_heading = "Compile", long = "matrix")]
    pub matrix: Vec<String>,

    /// *(6.32)* Seed a fresh compile from the SETTINGS that scored best in the `<stem>.smysl` corpus — the
    /// model/steps/guidance/scheduler of the highest-ranked `--improve` render — applied per scene where the
    /// scene doesn't already set them. "The corpus configures the compile," the knobs as well as the words.
    #[arg(help_heading = "Compile", long = "smysl-defaults", default_value_t = false)]
    pub smysl_defaults: bool,

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

    /// *(6.30)* Improve only the named scene(s)/composition(s) instead of the first. Repeatable and
    /// comma-separated: `--improve-scene bench --improve-scene garden` or `--improve-scene bench,garden`.
    /// Implies `--improve`.
    #[arg(help_heading = "Compile", long = "improve-scene", value_name = "NAME", value_delimiter = ',')]
    pub improve_scene: Vec<String>,

    /// *(6.30)* Skip a scene whose baseline aesthetic rank already meets this target — no passes are spent on
    /// it (the baseline still renders, so the gate is prose-aware). Applies to every improved scene.
    #[arg(help_heading = "Compile", long = "improve-target", value_name = "SCORE")]
    pub improve_target: Option<f32>,

    /// *(6.30)* Consult the smysl corpus: skip any scene whose baseline already matches or beats the best rank
    /// a prior `--improve` run recorded for it. Per-scene target from the corpus; scenes with no history are
    /// improved normally. Combine with `--improve-target` to also apply an absolute floor.
    #[arg(help_heading = "Compile", long = "improve-skip-good")]
    pub improve_skip_good: bool,

    /// *(6.30)* Keep the candidate images `--improve` renders (normally scored then deleted) in a directory
    /// so you can SEE the trajectory. Saved as `<stem>_improve/<scene>/call<NN>-seed<S>-rank<R>.png`.
    /// Optionally give a directory; default is `<input-stem>_improve/` beside the input.
    #[arg(help_heading = "Compile", long = "keep-compiled-images", value_name = "DIR", num_args = 0..=1, default_missing_value = "")]
    pub keep_compiled_images: Option<String>,

    /// *(6.32)* Score PLAIN t2i during `--improve` instead of through the scenario's LoRA stack. By default
    /// the improve loop renders with the scene's LoRAs (the finish `scenario` ships), so the aesthetic
    /// verdict matches what you'll print; `--improve-plain` isolates prompt effects from the LoRA look
    /// (faster, no LoRA load).
    #[arg(help_heading = "Compile", long = "improve-plain", default_value_t = false)]
    pub improve_plain: bool,

    /// *(6.32 ③)* Fast-draft proxy: render each `--improve` candidate at this many steps for the SEARCH
    /// (default 30 = full). Lower (e.g. 12–16) makes `--improve-all` much cheaper; the winning PROMPT is
    /// written and `scenario` renders it at full quality. Validate a lower count still ranks candidates the
    /// same for your model before trusting a deep cut.
    #[arg(help_heading = "Compile", long = "improve-draft-steps", value_name = "N", default_value_t = 30)]
    pub improve_draft_steps: usize,

    /// *(6.32)* MULTI-OBJECTIVE improve: weight for CLIP text↔image ADHERENCE added to each candidate's rank
    /// (`aesthetic + weight × adherence × 10`), so a winning edit must stay FAITHFUL to the prompt, not just
    /// look prettier. 0 (default) = aesthetic-only. Try 0.3–0.6. Loads the CLIP text tower once.
    #[arg(help_heading = "Compile", long = "improve-adherence", value_name = "W", default_value_t = 0.0)]
    pub improve_adherence: f32,

    /// *(6.32 ②)* Improve the REAL finish: img2img each candidate from this init image (a structure draft /
    /// reference frame) instead of a fresh t2i, so the aesthetic verdict scores your composition's *finish*,
    /// not a random layout. Rendered on a resident pipeline (no per-candidate reload). SD-family only.
    #[arg(help_heading = "Compile", long = "improve-init", value_name = "IMAGE")]
    pub improve_init: Option<PathBuf>,

    /// *(6.32 ②)* img2img strength for `--improve-init` — how hard the finish repaints the init. Lower keeps
    /// more of the composition; `0.5`–`0.7` is the usual finish range.
    #[arg(help_heading = "Compile", long = "improve-init-strength", value_name = "F", default_value_t = 0.6)]
    pub improve_init_strength: f32,

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
        // Thread B: show the same corpus-derived "avoid rejected phrasings" tail the real enhance appends.
        let explain_corpus =
            if stdin_input { None } else { std::fs::read_to_string(args.input.with_extension("smysl")).ok() };
        let avoid_hint = explain_corpus
            .as_deref()
            .map(|c| crate::smysl::enhance_avoid_hint(&crate::smysl::rejected_phrasings(c)))
            .unwrap_or_default();
        for s in resolved.scenes.iter().filter(|s| !s.skip) {
            println!("{} scene {:?} · family {:?}", style("──").cyan(), s.name, s.family);
            println!("{}", style("[positive system]").dim());
            println!("{}{}\n", compile::assembler::positive_system(s, sys_override.as_deref(), &[]), avoid_hint);
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

    // --smysl-report: a human digest of the prose's corpus (no LLM, no render).
    if args.smysl_report {
        anyhow::ensure!(!stdin_input, "--smysl-report needs a file input (the corpus is <stem>.smysl beside it)");
        let corpus_path = args.input.with_extension("smysl");
        let text = std::fs::read_to_string(&corpus_path).unwrap_or_default();
        println!("{}  smysl corpus for {}", style("◆").cyan(), corpus_path.display());
        print!("{}", crate::smysl::corpus_report(&text));
        return Ok(());
    }

    // --smysl-html: the same corpus digest as a self-contained HTML page (shareable provenance report).
    if let Some(out) = &args.smysl_html {
        anyhow::ensure!(!stdin_input, "--smysl-html needs a file input (the corpus is <stem>.smysl beside it)");
        let corpus_path = args.input.with_extension("smysl");
        let text = std::fs::read_to_string(&corpus_path).unwrap_or_default();
        let html = crate::smysl::corpus_report_html(&text, &corpus_path.display().to_string());
        std::fs::write(out, &html).with_context(|| format!("writing {}", out.display()))?;
        println!("{} {}  (smysl corpus \u{2192} HTML)", style("wrote").green(), out.display());
        return Ok(());
    }

    // --preflight: compile (no render) and run deterministic feasibility checks that gate a render run.
    if args.preflight {
        anyhow::ensure!(!stdin_input, "--preflight needs a file input (not stdin)");
        let sys = match &args.system {
            Some(p) => Some(std::fs::read_to_string(p).with_context(|| format!("reading --compile-system {}", p.display()))?),
            None => None,
        };
        let opts = CompileOpts {
            provider: args.provider.clone(),
            default_model: args.model.clone(),
            no_enhance: args.no_enhance,
            no_negative: args.no_negative,
            system_override: sys,
            cache: args.compile_cache,
            parallel: args.parallel,
            input_name: input_name.clone(),
            corpus_text: std::fs::read_to_string(args.input.with_extension("smysl")).ok(),
        };
        let doc = compile::compile_doc(&input, &opts).await?;
        return preflight_report(&doc, &args.input.display().to_string());
    }

    // --improve[-all|-scene]: the automatic aesthetic improve loop (Phase D). Renders + scores, needs a model.
    if args.improve || args.improve_all || !args.improve_scene.is_empty() {
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
            corpus_text: None,
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
            let fixreport = compile::apply_fixes(&args.input, &opts, &report, args.fix_wins).await?;
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
            corpus_text: None,
        };
        let (comp, opt) = compile::make_composition(&input, &args.input, &opts, &args.composition_model).await?;
        println!("{}  strategy    → {}", style("✓").green(), comp.display());
        println!("{}  optimized   → {}  (@includes {})", style("✓").green(), opt.display(),
            comp.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        println!("   review both, then: {} scenario {}", style("plakat compile").dim(), opt.display());
        return Ok(());
    }

    let opts = CompileOpts {
        provider: args.provider.clone(),
        default_model: args.model.clone(),
        no_enhance: args.no_enhance,
        no_negative: args.no_negative,
        system_override,
        cache: args.compile_cache,
        parallel: args.parallel,
        input_name,
        // Thread B: consult the corpus so the enhancer avoids prior rejected phrasings.
        corpus_text: if stdin_input {
            None
        } else {
            std::fs::read_to_string(args.input.with_extension("smysl")).ok()
        },
    };

    let needs_doc = !args.matrix.is_empty() || args.smysl_defaults || args.layered;
    let (hjson, layered_plans, warnings, trace, provenance): (String, Vec<(String, String)>, Vec<String>, Vec<String>, String) = if !needs_doc {
        let (h, w, t, p) = compile::compile_to_string(&input, &opts).await?;
        (h, Vec::new(), w, t, p)
    } else {
        // Doc-level post-processing: apply corpus-learned defaults, then expand the matrix.
        let mut doc = compile::compile_doc(&input, &opts).await?;
        if args.smysl_defaults {
            let corpus = if stdin_input {
                String::new()
            } else {
                std::fs::read_to_string(args.input.with_extension("smysl")).unwrap_or_default()
            };
            let defaults = crate::smysl::learned_defaults(&corpus);
            if defaults.is_empty() {
                eprintln!("{}  --smysl-defaults: no ranked settings in the corpus yet (run --improve first)", style("⚠").yellow());
            } else {
                apply_learned_defaults(&mut doc.scenes, &defaults);
                let shown: Vec<String> = defaults.iter().map(|(k, v)| format!("{k}={v}")).collect();
                eprintln!("{} smysl-defaults: applied {}", style("◆").cyan(), shown.join(" · "));
            }
        }
        if !args.matrix.is_empty() {
            let axes = parse_matrix(&args.matrix)?;
            let cells = expand_matrix(&mut doc, &axes)?;
            eprintln!("{} matrix: {} axis(es) → {} cell(s)", style("◆").cyan(), axes.len(), cells);
        }
        doc.recompute_provenance();
        let plans = if args.layered { doc.layered_plans() } else { Vec::new() };
        (doc.emit(args.layered), plans, doc.warnings.clone(), doc.trace.clone(), doc.provenance.clone())
    };

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
            anyhow::ensure!(
                layered_plans.is_empty(),
                "--layered writes plan sidecars next to the scenario, so it needs a file output — pass --out <file> instead of stdout"
            );
            print!("{hjson}");
            Ok(())
        }
        Some(path) => {
            std::fs::write(&path, &hjson).with_context(|| format!("writing {}", path.display()))?;
            println!("{}  compiled → {}", style("✓").green(), path.display());
            // --layered: write each auto-derived plan sidecar next to the scenario (the tasks reference
            // them by `layered: { plan: <name> }`).
            let plan_dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
            for (name, content) in &layered_plans {
                let p = plan_dir.join(name);
                std::fs::write(&p, content).with_context(|| format!("writing layered plan {}", p.display()))?;
                println!("{}  layered plan → {}", style("✓").green(), p.display());
            }
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
    /// The scenario's LoRA specs (raw `source[:scale]` strings) — so the fallback `api::Generate` path
    /// (non-SD-family) renders through the same LoRA stack the resident pipeline loaded. Empty = plain t2i.
    loras: Vec<String>,
    /// *(6.32 ③)* Steps per improve render — a fast-draft proxy for the SEARCH (`--improve-draft-steps`).
    /// Lower = faster candidate renders; the winning PROMPT is what's written, and `scenario` renders it at
    /// full quality. Default 30 (full).
    draft_steps: usize,
    /// *(6.32 ②)* `--improve-init`: when set, each candidate is img2img'd from `init` at `strength` on the
    /// RESIDENT `portrait` pipeline (scores the real finish, not a fresh t2i). Takes precedence over `pipe`.
    img2img: Option<Img2imgMode>,
    /// `--keep-compiled-images`: archive every scored candidate under here (`None` → discard). Images land in
    /// `<keep_dir>/<scene_name>/call<NN>-seed<S>-rank<R>.png` — the whole render trajectory, not just the best.
    keep_dir: Option<std::path::PathBuf>,
    /// Current scene (subdir under `keep_dir`) and a per-scene monotonic call counter (`0` = baseline render).
    scene_name: String,
    call_idx: usize,
    /// *(6.32)* Multi-objective: when set, each candidate's rank ADDS `adherence_weight × (CLIP text↔image
    /// adherence × 10)` to the aesthetic score, so a winning edit must stay FAITHFUL to the prompt, not just
    /// look prettier. `None` = aesthetic-only (the default).
    adherence: Option<crate::pipelines::clip_adherence::ClipAdherence>,
    adherence_weight: f32,
}

/// The resident img2img mode for `--improve-init` — a loaded `portrait::Pipeline` + the fixed init + strength.
struct Img2imgMode {
    pipe: crate::pipelines::portrait::Pipeline,
    init: std::path::PathBuf,
    strength: f32,
    model: String,
    device: candle_core::Device,
    loras: Vec<crate::pipelines::lora::LoraSpec>,
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
                if let Some(m) = &self.img2img {
                    // ② --improve-init: img2img each candidate from the FIXED init on the resident portrait
                    // pipeline — scores the real finish over your composition, not a fresh t2i layout.
                    let out_dir = self.tmp.join(format!("i2i-{i}"));
                    let _ = std::fs::remove_dir_all(&out_dir);
                    std::fs::create_dir_all(&out_dir)?;
                    let req = crate::pipelines::img2img::Request {
                        prompt: prompt.to_string(),
                        negative: self.negative.clone(),
                        model: m.model.clone(),
                        device: m.device.clone(),
                        loras: m.loras.clone(),
                        lora_scale: 1.0,
                        input: m.init.clone(),
                        mask: None,
                        mask_feather: 0,
                        mask_invert: false,
                        width: self.width,
                        height: self.height,
                        count: 1,
                        steps: self.draft_steps,
                        guidance: 7.5,
                        scheduler: crate::pipelines::scheduler::SchedulerKind::default(),
                        strength: m.strength,
                        seed: Some(seed),
                        out_dir: out_dir.clone(),
                        controls: Vec::new(),
                    };
                    crate::pipelines::img2img::run_with_pipeline(&m.pipe, &req).await?;
                    let produced = std::fs::read_dir(&out_dir)?
                        .filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .find(|p| p.extension().map(|x| x == "png").unwrap_or(false))
                        .ok_or_else(|| anyhow::anyhow!("--improve-init: img2img produced no image"))?;
                    std::fs::copy(&produced, &path)?;
                } else if let Some(pipe) = &self.pipe {
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
                        self.draft_steps,
                    )?;
                } else {
                    // Fallback: one-shot render (reloads the model each call) for non-SD-family models.
                    let mut req = crate::api::Generate::new(self.model.as_str())
                        .prompt(prompt)
                        .negative(self.negative.as_str())
                        .size(self.width, self.height)
                        .seed(seed)
                        .steps(self.draft_steps)
                        .count(1);
                    // Same LoRA stack as the resident path — score the real finish, not plain t2i. Each raw
                    // spec re-parses in build_loras; passing its own scale keeps `source:scale` intact.
                    for spec in &self.loras {
                        let scale = spec.parse::<crate::pipelines::lora::LoraSpec>().map(|l| l.scale).unwrap_or(1.0);
                        req = req.lora(spec.as_str(), scale);
                    }
                    let images = req.run().await?;
                    let img = images
                        .into_iter()
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--improve: render produced no image"))?;
                    img.save(&path)?;
                }
                let aesthetic = self.scorer.score_path(&path)?;
                // Multi-objective: fold in CLIP text↔image adherence so wins stay faithful, not just pretty.
                let score = if let Some(adh) = &self.adherence {
                    let emb = self.scorer.image_embedding(&path)?;
                    let a = adh.adherence(&emb, prompt)?;
                    aesthetic + self.adherence_weight * a * 10.0
                } else {
                    aesthetic
                };
                scores.push(score);
                // `--keep-compiled-images`: archive this candidate (best-effort — never fail a render over it).
                if let Some(kd) = &self.keep_dir {
                    let safe: String = self
                        .scene_name
                        .chars()
                        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
                        .collect();
                    let dir = kd.join(if safe.is_empty() { "scene".to_string() } else { safe });
                    let dest = dir.join(format!("call{:02}-seed{seed}-rank{score:.2}.png", self.call_idx));
                    if let Err(e) = std::fs::create_dir_all(&dir).and_then(|_| std::fs::copy(&path, &dest).map(|_| ())) {
                        eprintln!("    · keep-images: could not save {}: {e}", dest.display());
                    }
                }
            }
            self.call_idx += 1;
            let mean = scores.iter().sum::<f32>() / scores.len().max(1) as f32;
            self.last_rank = mean;
            let label = if self.adherence.is_some() { "objective (aesthetic+adherence)" } else { "aesthetic" };
            if scores.len() > 1 {
                let spread = scores.iter().cloned().fold(f32::MIN, f32::max)
                    - scores.iter().cloned().fold(f32::MAX, f32::min);
                eprintln!("    · {label} {mean:.2} (mean of {}, spread {spread:.2})", scores.len());
            } else {
                eprintln!("    · {label} {mean:.2}");
            }
            Ok(mean)
        })
    }
}

/// Parse `--matrix KEY=v1,v2,…` flags into ordered axes.
fn parse_matrix(flags: &[String]) -> Result<Vec<(String, Vec<String>)>> {
    let mut axes = Vec::new();
    for f in flags {
        let (k, v) = f.split_once('=').ok_or_else(|| anyhow::anyhow!("--matrix expects KEY=v1,v2,… (got {f:?})"))?;
        let key = k.trim().to_string();
        anyhow::ensure!(!key.is_empty(), "--matrix: empty axis name in {f:?}");
        let vals: Vec<String> = v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        anyhow::ensure!(!vals.is_empty(), "--matrix axis {key:?} has no values");
        axes.push((key, vals));
    }
    Ok(axes)
}

/// Lowercase kebab slug for a cell name segment.
fn slug(s: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for c in s.trim().to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// Apply one axis value to a cloned scene: `set.KEY` sets a per-scene directive (model is a typed field),
/// anything else appends to the prompt.
fn apply_axis(c: &mut crate::compile::emitter::CompiledScene, key: &str, val: &str) {
    if let Some(k) = key.strip_prefix("set.") {
        if k.eq_ignore_ascii_case("model") {
            c.scene.model_for_family = Some(val.to_string());
        } else {
            c.scene.passthrough.retain(|(pk, _)| !pk.eq_ignore_ascii_case(k));
            c.scene.passthrough.push((k.to_string(), val.to_string()));
        }
    } else if !c.prompt.to_lowercase().contains(&val.to_lowercase()) {
        if !c.prompt.trim().is_empty() && !c.prompt.trim_end().ends_with(',') {
            c.prompt.push_str(", ");
        }
        c.prompt.push_str(val);
    }
}

/// Expand every scene in `doc` along the axes into the cartesian product of cells; returns the cell count.
fn expand_matrix(doc: &mut crate::compile::CompiledDoc, axes: &[(String, Vec<String>)]) -> Result<usize> {
    let mut combos: Vec<Vec<(&str, &str)>> = vec![vec![]];
    for (k, vals) in axes {
        let mut next = Vec::new();
        for c in &combos {
            for v in vals {
                let mut cc = c.clone();
                cc.push((k.as_str(), v.as_str()));
                next.push(cc);
            }
        }
        combos = next;
    }
    let cells = combos.len();
    anyhow::ensure!(cells <= 128, "matrix has {cells} cells (> 128) — narrow the axes");

    let base = std::mem::take(&mut doc.scenes);
    let mut out = Vec::with_capacity(base.len().saturating_mul(cells));
    for scene in &base {
        for combo in &combos {
            let mut c = scene.clone();
            let mut suffix = String::new();
            for (key, val) in combo {
                apply_axis(&mut c, key, val);
                suffix.push_str(&format!("__{}-{}", slug(key.trim_start_matches("set.")), slug(val)));
            }
            let bn = if c.scene.name.trim().is_empty() { "scene".to_string() } else { c.scene.name.clone() };
            c.scene.name = format!("{bn}{suffix}");
            c.scene.name_auto = false;
            out.push(c);
        }
    }
    doc.scenes = out;
    Ok(cells)
}

/// Apply corpus-learned settings to each scene that doesn't already set them. Typed knobs (model / steps /
/// guidance / scheduler) go to their fields; anything else becomes a per-scene passthrough directive.
fn apply_learned_defaults(scenes: &mut [crate::compile::emitter::CompiledScene], defaults: &[(String, String)]) {
    for c in scenes.iter_mut() {
        let sc = &mut c.scene;
        for (k, v) in defaults {
            match k.as_str() {
                "model" => {
                    if sc.model_for_family.is_none() {
                        sc.model_for_family = Some(v.clone());
                    }
                }
                "steps" => {
                    if sc.steps.is_none() {
                        sc.steps = v.parse().ok();
                    }
                }
                "guidance" => {
                    if sc.guidance.is_none() {
                        sc.guidance = v.parse().ok();
                    }
                }
                "scheduler" => {
                    if sc.scheduler.as_deref().unwrap_or("").trim().is_empty() {
                        sc.scheduler = Some(v.clone());
                    }
                }
                _ => {
                    if !sc.passthrough.iter().any(|(pk, _)| pk.eq_ignore_ascii_case(k)) {
                        sc.passthrough.push((k.clone(), v.clone()));
                    }
                }
            }
        }
    }
}

/// Run deterministic preflight checks over a compiled doc and print a PASS/WARN/FAIL report. Returns an
/// error (non-zero exit) when any scene has a hard FAIL — so it gates a render run in CI. No LLM, no render.
fn preflight_report(doc: &crate::compile::CompiledDoc, input: &str) -> Result<()> {
    let (mut warns, mut fails) = (0usize, 0usize);
    println!("{}  preflight — {} ({} scene(s))", style("◆").cyan(), input, doc.scenes.len());
    let pass = style("✓").green();
    let warn = style("⚠").yellow();
    let fail = style("✗").red();

    for c in &doc.scenes {
        let s = &c.scene;
        println!("  {}:", style(format!("\"{}\"", s.name)).bold());
        let mut line = |ok: i8, msg: String| match ok {
            1 => println!("    {pass} {msg}"),
            0 => {
                warns += 1;
                println!("    {warn} {msg}");
            }
            _ => {
                fails += 1;
                println!("    {fail} {msg}");
            }
        };

        if s.skip {
            line(1, "scene is marked skip — not rendered".into());
            continue;
        }
        // Prompt present.
        if c.prompt.trim().is_empty() {
            line(-1, "prompt is EMPTY — nothing to render".into());
        } else {
            line(1, format!("prompt present ({} chars)", c.prompt.chars().count()));
        }
        // LoRA trigger must appear in the prompt or the LoRA never activates.
        let trig = s.lora_trigger.trim();
        if !trig.is_empty() {
            if c.prompt.to_lowercase().contains(&trig.to_lowercase()) {
                line(1, format!("LoRA trigger {trig:?} present in the prompt"));
            } else {
                line(-1, format!("LoRA trigger {trig:?} ABSENT from the prompt — the LoRA won't activate"));
            }
        }
        // Sane steps / guidance.
        if let Some(st) = s.steps {
            if !(1..=150).contains(&st) {
                line(0, format!("steps = {st} is outside the usual 1–150"));
            }
        }
        if let Some(g) = s.guidance {
            if !(0.0..=30.0).contains(&g) {
                line(0, format!("guidance = {g} is outside the usual 0–30"));
            }
        }
        // Budget / style-drop diligence warnings surfaced by the compiler.
        for w in &c.warnings {
            line(0, format!("compile: {w}"));
        }
        // Region sanity: a `w=<num>` weight should be in [0,1].
        for r in &s.regions {
            for tok in r.split_whitespace() {
                if let Some(v) = tok.trim_matches(|c| c == ',' || c == ';').strip_prefix("w=").and_then(|n| n.parse::<f32>().ok()) {
                    if !(0.0..=1.0).contains(&v) {
                        line(0, format!("region weight {v} outside 0–1 ({r:?})"));
                    }
                }
            }
        }
    }

    let verdict = if fails > 0 {
        style(format!("FAIL — {fails} failure(s), {warns} warning(s)")).red().bold()
    } else if warns > 0 {
        style(format!("PASS with {warns} warning(s)")).yellow()
    } else {
        style("PASS — all checks green".into()).green().bold()
    };
    println!("\n{}  {}", style("→").dim(), verdict);
    anyhow::ensure!(fails == 0, "preflight failed: {fails} hard check(s) — fix before rendering");
    Ok(())
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

/// Record a scene's best achieved aesthetic rank into the `.smysl` corpus (merge, content-hash dedup) so a
/// later `--improve-skip-good` run can read it back via `smysl::prior_scene_rank` and skip an already-good scene.
fn persist_scene_rank(corpus: &std::path::Path, scene: &str, rank: f32, settings: &[(String, String)]) -> Result<()> {
    let (recs, labels) = crate::smysl::scene_rank_records(scene, rank, settings)?;
    let snapshot = crate::smysl::records_to_surface(&recs, &labels);
    let base = std::fs::read_to_string(corpus).unwrap_or_default();
    std::fs::write(corpus, crate::smysl::merge_surface(&base, &snapshot))
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
        // Thread B: the baseline enhance consults the corpus (avoid re-introducing rejected phrasings).
        corpus_text: std::fs::read_to_string(args.input.with_extension("smysl")).ok(),
    };

    // 1. ENHANCE — compile every scene to its prompt (the translation pass). Keep the WHOLE compiled doc, not
    //    just the prompts: the winning prompt is written back into `doc.scenes[i].prompt` and the doc is
    //    re-emitted, so `compile --improve` produces an improved HJSON (the improvement lands in the artifact,
    //    not just the console). The smysl corpus drives the loop (tabu + skip gate) — it IS the process.
    let mut doc = compile::compile_doc(input, &opts).await?;
    anyhow::ensure!(!doc.scenes.is_empty(), "--improve: no renderable scenes in {}", args.input.display());
    // SELECT the scope: named scene(s), all scenes, or (default) the first — as INDICES into doc.scenes.
    let scene_names: Vec<String> = doc.scenes.iter().map(|c| c.scene.name.clone()).collect();
    let selected: Vec<usize> = if !args.improve_scene.is_empty() {
        // One or more named scenes (repeatable / comma-separated). Every requested name must resolve.
        let wanted: Vec<String> =
            args.improve_scene.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        for w in &wanted {
            anyhow::ensure!(
                scene_names.iter().any(|n| n.eq_ignore_ascii_case(w)),
                "--improve-scene: no scene named {w:?} — scenes are: {}",
                scene_names.join(", "),
            );
        }
        (0..doc.scenes.len())
            .filter(|&i| wanted.iter().any(|w| scene_names[i].eq_ignore_ascii_case(w)))
            .collect()
    } else if args.improve_all {
        (0..doc.scenes.len()).collect()
    } else {
        vec![0] // default: the first scene
    };

    // 2. Load the aesthetic scorer ONCE (the expensive load) + resolve the render model / size / seeds.
    let model = args.improve_model.clone().unwrap_or_else(|| args.model.clone());
    let res = crate::capability::native_res(&model);
    let device = crate::device::select("auto")?;
    let scorer = crate::pipelines::aesthetic::AestheticScorer::load(&device)
        .await
        .context("--improve: loading the aesthetic scorer (CLIP ViT-L/14 + LAION predictor)")?;
    // Multi-objective (6.32): load the CLIP text tower for prompt-adherence when --improve-adherence > 0.
    let adherence = if args.improve_adherence > 0.0 {
        let a = crate::pipelines::clip_adherence::ClipAdherence::load(&device)
            .await
            .context("--improve-adherence: loading the CLIP text tower")?;
        eprintln!("{} multi-objective: aesthetic + {:.2} × adherence", style("◆").cyan(), args.improve_adherence);
        Some(a)
    } else {
        None
    };
    // Thread A (6.32): score the REAL finish — load the scenario's LoRA stack into the improve pipeline
    // (unless --improve-plain), so the aesthetic verdict matches what `scenario` will render. Compile LoRAs
    // are scenario-global (`doc.globals.loras`); the activation token is already prepended into each compiled
    // prompt, so nothing else in the render path changes.
    let lora_specs: Vec<String> = if args.improve_plain { Vec::new() } else { doc.globals.loras.clone() };
    let lora_stack: Vec<crate::pipelines::lora::LoraSpec> = lora_specs
        .iter()
        .map(|x| x.parse())
        .collect::<Result<Vec<_>>>()
        .context("--improve: parsing the scenario LoRA specs")?;
    if !lora_stack.is_empty() {
        println!(
            "{}  improve renders through {} LoRA(s): {} (use --improve-plain for bare t2i)",
            style("◆").cyan(),
            lora_stack.len(),
            lora_specs.join(", "),
        );
    }
    if args.improve_draft_steps < 30 {
        println!(
            "{}  fast-draft search at {} steps (winner prompt renders at full quality in `scenario`)",
            style("◆").cyan(),
            args.improve_draft_steps.max(1),
        );
    }
    // ② --improve-init: load the RESIDENT portrait pipeline (img2img-capable) once; each candidate is
    // img2img'd from the fixed init. When set, the plain-t2i pipe is not loaded (the finish is what we score).
    let img2img: Option<Img2imgMode> = if let Some(init) = &args.improve_init {
        anyhow::ensure!(init.exists(), "--improve-init: image {} not found", init.display());
        let strength = args.improve_init_strength.clamp(0.05, 1.0);
        println!(
            "{}  improve scores the FINISH: img2img from {} at strength {strength:.2} (resident, no reload)",
            style("◆").cyan(),
            init.display(),
        );
        let p = crate::pipelines::portrait::Pipeline::load(crate::pipelines::portrait::LoadRequest {
            model: model.clone(),
            device: device.clone(),
            loras: lora_stack.clone(),
            lora_scale: 1.0,
            identity: None,
            shared_clip_h: None,
        })
        .await
        .with_context(|| format!("--improve-init: loading the img2img pipeline for {model}"))?;
        Some(Img2imgMode {
            pipe: p,
            init: init.clone(),
            strength,
            model: model.clone(),
            device: device.clone(),
            loras: lora_stack.clone(),
        })
    } else {
        None
    };
    // Load the RENDER model ONCE and keep it resident — reloading the SD core per render was the bug. SD
    // family (sd15/sdxl/…) loads here; other families (sd35/Flux) fall back to the one-shot per-render path.
    // Skipped in --improve-init mode (the portrait pipeline above is the renderer).
    let pipe = if img2img.is_some() {
        None
    } else {
        match crate::pipelines::t2i::Pipeline::load(crate::pipelines::t2i::LoadRequest {
        model: model.clone(),
        device: device.clone(),
        loras: lora_stack.clone(),
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
        }
    };
    let tmp = std::env::temp_dir().join(format!("plakat-improve-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).with_context(|| format!("--improve: creating {}", tmp.display()))?;
    let n_seeds = args.improve_seeds.max(1) as u64;
    let seeds: Vec<u64> = (0..n_seeds).map(|i| args.improve_seed + i).collect();
    let corpus_path = args.input.with_extension("smysl");

    // `--keep-compiled-images [DIR]`: where to archive the candidates. Bare flag → `<input-stem>_improve/`
    // beside the input; an explicit value is used verbatim. Created up front so a save can't be the first error.
    let keep_dir: Option<std::path::PathBuf> = args.keep_compiled_images.as_ref().map(|d| {
        if d.trim().is_empty() {
            let stem = args.input.file_stem().and_then(|s| s.to_str()).unwrap_or("compile");
            let parent = args.input.parent().filter(|p| !p.as_os_str().is_empty());
            parent.unwrap_or_else(|| std::path::Path::new(".")).join(format!("{stem}_improve"))
        } else {
            std::path::PathBuf::from(d)
        }
    });
    if let Some(kd) = &keep_dir {
        std::fs::create_dir_all(kd)
            .with_context(|| format!("--keep-compiled-images: creating {}", kd.display()))?;
        println!("{}  keeping every scored candidate under {}/", style("◆").cyan(), kd.display());
    }

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
        loras: lora_specs,
        draft_steps: args.improve_draft_steps.max(1),
        img2img,
        keep_dir,
        scene_name: String::new(),
        call_idx: 0,
        adherence,
        adherence_weight: args.improve_adherence,
    };

    let mut results: Vec<(String, compile::improve::ImproveOutcome)> = Vec::new();
    for &i in &selected {
        let name = doc.scenes[i].scene.name.clone();
        let prompt0 = doc.scenes[i].prompt.clone();
        let negative = doc.scenes[i].negative.clone();
        println!("\n{} scene {name}", style("──").cyan());
        // Each scene is an INDEPENDENT optimization: its own negative, its own baseline (its ENHANCED prompt),
        // its own tabu seeded from the (shared, content-addressed) corpus — a scene's edits never collide.
        step.negative = negative.clone();
        step.last_rank = 0.0;
        step.scene_name = name.clone();
        step.call_idx = 0;
        // Read the corpus ONCE for this scene: its prior fix-moves (tabu) AND its prior best rank (the gate).
        let corpus_text = std::fs::read_to_string(&corpus_path).ok();
        let seed_tabu =
            corpus_text.as_deref().map(crate::smysl::prior_fixes).unwrap_or_default();
        // The quality gate: skip passes if the baseline already meets the target. `--improve-target` is an
        // absolute floor; `--improve-skip-good` pulls a per-scene target from the corpus's prior best. With
        // both, the baseline must clear the HIGHER bar (max) — skip only if it satisfies both.
        let prior_best = if args.improve_skip_good {
            corpus_text.as_deref().and_then(|t| crate::smysl::prior_scene_rank(t, &name))
        } else {
            None
        };
        // Fresh baselines are NOISY — Metal renders aren't bit-reproducible and the aesthetic score drifts
        // ~0.2 run-to-run. So the `--improve-skip-good` bar (a noisy stored best vs a noisy fresh baseline)
        // gets a tolerance band; without it, a scene we already optimized re-marches just because this run's
        // baseline landed a hair lower. An explicit `--improve-target` stays an EXACT absolute bar (the user
        // named a number). With both, the higher bar wins.
        const SKIP_TOL: f32 = 0.25;
        let target = match (args.improve_target, prior_best) {
            (Some(a), Some(b)) => Some(a.max(b - SKIP_TOL)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b - SKIP_TOL),
            (None, None) => None,
        };
        if let Some(t) = target {
            let src = match (args.improve_target, prior_best) {
                (Some(_), Some(b)) => format!("target / prior best {b:.2} \u{2212}{SKIP_TOL} tol"),
                (Some(_), None) => "target".to_string(),
                (None, Some(b)) => format!("prior best {b:.2} \u{2212}{SKIP_TOL} tol from smysl"),
                (None, None) => String::new(),
            };
            println!("    · gate: skip if baseline \u{2265} {t:.2} ({src})");
        }
        let out = compile::improve::run_improve(
            &mut step,
            &prompt0,
            seed_tabu,
            args.improve_passes,
            args.improve_plateau.max(1),
            args.improve_min_gain,
            target,
        )
        .await?;
        // WRITE THE WINNER BACK into the compiled scene, so the emitted HJSON carries the improved prompt.
        // (For a skipped/already-good scene, best_prompt IS the baseline — a harmless identity override.)
        doc.scenes[i].prompt = out.best_prompt.clone();
        // Persist this scene's tried deltas into the shared corpus (tabu memory for next time).
        let moves = out.corpus_moves();
        if !moves.is_empty() {
            if let Err(e) = persist_improve_corpus(&corpus_path, &moves) {
                eprintln!("{}  smysl corpus not updated: {e:#}", style("⚠").yellow());
            }
        }
        // Record this scene's best rank AND the settings that achieved it — so `--improve-skip-good` can
        // gate it, and `--smysl-defaults` can later learn the best-scoring knobs (not just the words).
        let settings = {
            let sc = &doc.scenes[i].scene;
            let mut v: Vec<(String, String)> = Vec::new();
            let model = args
                .improve_model
                .clone()
                .or_else(|| sc.model_for_family.clone())
                .unwrap_or_else(|| format!("{:?}", sc.family).to_lowercase());
            v.push(("model".into(), model));
            if let Some(st) = sc.steps {
                v.push(("steps".into(), st.to_string()));
            }
            if let Some(g) = sc.guidance {
                v.push(("guidance".into(), format!("{g}")));
            }
            if let Some(s) = &sc.scheduler {
                if !s.trim().is_empty() {
                    v.push(("scheduler".into(), s.trim().to_string()));
                }
            }
            v
        };
        if let Err(e) = persist_scene_rank(&corpus_path, &name, out.best_rank, &settings) {
            eprintln!("{}  smysl rank not recorded: {e:#}", style("⚠").yellow());
        }
        results.push((name.clone(), out));
    }
    let _ = std::fs::remove_dir_all(&tmp);

    // 4. Report per scene.
    let mut skipped = 0usize;
    for (name, out) in &results {
        let already_good = out.stop == compile::improve::StopReason::AlreadyGood;
        if already_good {
            skipped += 1;
            println!("\n{} scene {name} — already good (aesthetic {:.2}), skipped", style("↷").dim(), out.best_rank);
            continue;
        }
        println!("\n{} scene {name}", style("✓").green());
        print!("{}", compile::improve::format_report(out));
        println!("  best prompt (aesthetic {:.2}):\n{}", out.best_rank, out.best_prompt);
    }
    if skipped > 0 {
        println!(
            "\n{}  {skipped} of {} scene(s) already met the target — passes skipped.",
            style("◆").cyan(),
            results.len(),
        );
    }

    // 5. EMIT — write the improved scenario. The winning prompts are already in `doc.scenes[*].prompt`, so
    //    re-emitting produces an HJSON that carries the improvement into the artifact you render.
    doc.recompute_provenance(); // provenance now tracks prose → WINNING prompt
    let hjson = doc.emit(args.layered);
    crate::cli::scenario::validate_hjson(&hjson).context("improved scenario failed validation")?;
    let out_path: PathBuf = match &args.out {
        Some(p) if p.as_os_str() != "-" => p.clone(),
        _ => args.input.with_extension("hjson"),
    };
    std::fs::write(&out_path, &hjson).with_context(|| format!("writing {}", out_path.display()))?;
    println!("\n{}  compiled (improved) → {}", style("✓").green(), out_path.display());
    if args.layered {
        let plan_dir = out_path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        for (name, content) in doc.layered_plans() {
            let p = plan_dir.join(&name);
            std::fs::write(&p, &content).with_context(|| format!("writing layered plan {}", p.display()))?;
            println!("{}  layered plan → {}", style("✓").green(), p.display());
        }
    }

    // The smysl corpus is the PROCESS memory (tabu + per-scene rank) and already lives at <stem>.smysl. Under
    // --smysl, also fold in the scene-claims + prose→winning-prompt provenance so the sidecar is complete.
    if args.smysl {
        let sidecar = args.input.with_extension("smysl");
        match compile::compose_scene_smysl(input, &args.model, Some(sidecar.as_path())) {
            Ok(base) => {
                let doc_txt = if doc.provenance.trim().is_empty() {
                    base
                } else {
                    crate::smysl::merge_surface(&base, &doc.provenance)
                };
                if let Err(e) = std::fs::write(&sidecar, &doc_txt) {
                    eprintln!("{}  smysl sidecar not updated: {e:#}", style("⚠").yellow());
                } else {
                    println!("{}  smysl       → {}", style("✓").green(), sidecar.display());
                }
            }
            Err(e) => eprintln!("{}  smysl sidecar skipped: {e:#}", style("⚠").yellow()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod matrix_tests {
    use super::{parse_matrix, slug};

    #[test]
    fn parse_matrix_axes() {
        let axes = parse_matrix(&["weather=clear, storm ,fog".into(), "set.steps=20,40".into()]).unwrap();
        assert_eq!(axes[0].0, "weather");
        assert_eq!(axes[0].1, vec!["clear", "storm", "fog"]);
        assert_eq!(axes[1].0, "set.steps");
        assert_eq!(axes[1].1, vec!["20", "40"]);
        assert!(parse_matrix(&["noequals".into()]).is_err(), "missing = is an error");
        assert!(parse_matrix(&["k=".into()]).is_err(), "no values is an error");
    }

    #[test]
    fn slug_kebabs() {
        assert_eq!(slug("Golden Hour!"), "golden-hour");
        assert_eq!(slug("sd3.5"), "sd3-5");
        assert_eq!(slug("  storm  "), "storm");
    }

    #[test]
    fn learned_defaults_apply_to_typed_fields_and_respect_overrides() {
        use crate::compile::emitter::CompiledScene;
        use crate::compile::resolver::ResolvedScene;
        let scene = |steps: Option<usize>| CompiledScene {
            scene: ResolvedScene { steps, ..Default::default() },
            prompt: "a cat".into(),
            negative: String::new(),
            structure_prompt: None,
            control_generate_max_figures: None,
            warnings: vec![],
            trace: vec![],
            pack: None,
        };
        let mut scenes = vec![scene(None), scene(Some(50))];
        let defaults = vec![("model".into(), "sd35".into()), ("steps".into(), "40".into()), ("guidance".into(), "5.5".into())];
        super::apply_learned_defaults(&mut scenes, &defaults);
        // Scene 0 had no steps → learns 40; scene 1 already set 50 → kept.
        assert_eq!(scenes[0].scene.model_for_family.as_deref(), Some("sd35"));
        assert_eq!(scenes[0].scene.steps, Some(40));
        assert_eq!(scenes[0].scene.guidance, Some(5.5));
        assert_eq!(scenes[1].scene.steps, Some(50), "explicit scene setting is not overridden");
    }
}
