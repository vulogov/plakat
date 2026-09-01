//! `plakat compile` — turn a prose `prompts.txt` into a `scenario` HJSON.
//!
//! A `prompts.txt` is blank-line-separated **blocks**. Each block is free-text
//! lines (the description) plus `key: value` **command** lines. The first block
//! is the **global** block iff it has no free text; its commands become the
//! scenario's global defaults. Every other block becomes one scenario **task**.
//!
//! Pipeline (COMPILE-1):
//!   parse → resolve (global↔scene inheritance + model-family) → assemble
//!   (header+text+footer, system prompts) → LLM (positive + auto-negative) →
//!   emit (JSON→HJSON post-pass; `deser-hjson` is deserialize-only).
//!
//! The LLM stage reuses the `--enhance` provider stack (`src/prompt`); with
//! `--no-enhance` the assembled text is used verbatim, making the whole pass
//! deterministic (the corpus gate).

pub mod parser;
pub mod resolver;
pub mod assembler;
pub mod emitter;
pub mod cache;
pub mod scenario_read;

// COMPILE-2: the Tera template pre-pass is feature-gated. When `templates` is on,
// `template` renders `.tera`/`.j2`/… inputs to a `prompts.txt` string before the
// parser; when off, `template_stub` returns a "recompile with --features templates"
// error. Both expose the same `render(input, input_path, opts)` signature.
#[cfg(feature = "templates")]
pub mod template;
#[cfg(not(feature = "templates"))]
pub mod template_stub;
#[cfg(not(feature = "templates"))]
pub use template_stub as template;

/// Inputs for the Tera pre-pass (always compiled, so the CLI layer is
/// feature-agnostic).
#[derive(Debug, Default)]
pub struct TemplateOpts {
    /// `--var KEY=VALUE` pairs (highest precedence).
    pub vars: Vec<(String, String)>,
    /// `--vars <PATH>` JSON/TOML files (later files win).
    pub vars_files: Vec<std::path::PathBuf>,
    /// `--vars-env <PREFIX>` env-var imports (prefix stripped, key lowercased).
    pub env_prefixes: Vec<String>,
}

/// Whether the input should go through the Tera pre-pass: `--template` forces it,
/// else a `.tera`/`.j2`/`.jinja`/`.jinja2` extension triggers it.
pub fn should_use_template(path: Option<&std::path::Path>, force: bool) -> bool {
    if force {
        return true;
    }
    matches!(
        path.and_then(|p| p.extension()).and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).as_deref(),
        Some("tera" | "j2" | "jinja" | "jinja2")
    )
}

/// How repeated occurrences of a command within one block combine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Merge {
    /// Join all values with `, ` (header, footer, negative, style, persona).
    Concatenate,
    /// Each occurrence appends one list entry (lora, tag).
    AccumulateList,
    /// The last occurrence wins (model, seed, count, …).
    LastWins,
}

/// Where a command's value goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandKind {
    /// Shapes or feeds the LLM call (header/footer/negative/style/translate/persona).
    Prompt,
    /// Passes straight to scenario HJSON, no LLM (model/lora/seed/…).
    Scenario,
}

/// One known command's metadata.
#[derive(Clone, Copy, Debug)]
pub struct CommandSpec {
    pub key: &'static str,
    pub kind: CommandKind,
    pub merge: Merge,
}

/// The full command table. Unknown commands are a lint error (E-C2).
pub const COMMANDS: &[CommandSpec] = &[
    // ---- prompt commands (shape/feed the LLM) ----
    CommandSpec { key: "header",    kind: CommandKind::Prompt,   merge: Merge::Concatenate },
    CommandSpec { key: "footer",    kind: CommandKind::Prompt,   merge: Merge::Concatenate },
    CommandSpec { key: "negative",  kind: CommandKind::Prompt,   merge: Merge::Concatenate },
    CommandSpec { key: "style",     kind: CommandKind::Prompt,   merge: Merge::Concatenate },
    CommandSpec { key: "translate", kind: CommandKind::Prompt,   merge: Merge::LastWins },
    CommandSpec { key: "persona",   kind: CommandKind::Prompt,   merge: Merge::Concatenate },
    // ---- scenario commands (straight to HJSON) ----
    CommandSpec { key: "model",     kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "lora",      kind: CommandKind::Scenario, merge: Merge::AccumulateList },
    CommandSpec { key: "seed",      kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "count",     kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "size",      kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "steps",     kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "guidance",  kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "scheduler", kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "refine",    kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "name",      kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "tag",       kind: CommandKind::Scenario, merge: Merge::AccumulateList },
    CommandSpec { key: "weather",   kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "skip",      kind: CommandKind::Scenario, merge: Merge::LastWins },
    // ---- MAP-4: a `type: map` block compiles to a scenario `map` task ----
    CommandSpec { key: "type",          kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "map-spec",      kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "map-style",     kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "map-paint",     kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "map-scale",     kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "map-tiles",     kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "map-sd-model",  kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "map-sd-lora",   kind: CommandKind::Scenario, merge: Merge::AccumulateList },
    CommandSpec { key: "map-provider",  kind: CommandKind::Scenario, merge: Merge::LastWins },
];

/// Look up a command spec by key.
pub fn command_spec(key: &str) -> Option<&'static CommandSpec> {
    COMMANDS.iter().find(|c| c.key == key)
}

/// SD model family — drives the family-specific LLM system-prompt section.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ModelFamily {
    Sd15,
    Sdxl,
    Flux,
    #[default]
    Unknown,
}

impl ModelFamily {
    /// Short label for HJSON comments / dry-run output.
    pub fn label(self) -> &'static str {
        match self {
            ModelFamily::Sd15 => "SD15",
            ModelFamily::Sdxl => "SDXL",
            ModelFamily::Flux => "Flux",
            ModelFamily::Unknown => "Unknown",
        }
    }
}

/// Options for [`compile_to_string`].
pub struct CompileOpts {
    /// LLM provider (`deepseek`/`gemini`/`local`/`local:<alias>`/`auto`).
    pub provider: String,
    /// `--model` fallback for family classification when no block names a model.
    pub default_model: String,
    /// `--no-enhance`: skip the positive LLM call; use assembled text verbatim.
    pub no_enhance: bool,
    /// `--no-negative`: skip the negative LLM call; pass seed terms verbatim.
    pub no_negative: bool,
    /// `--compile-system` override for the positive system prompt.
    pub system_override: Option<String>,
    /// `--compile-cache`: read/write the two-namespace disk cache.
    pub cache: bool,
    /// `--compile-parallel`: max concurrent scenes. `0` = auto (per-provider).
    pub parallel: usize,
    /// Name shown in the output's header comment (kept deterministic).
    pub input_name: String,
}

/// Resolve the concurrency: an explicit value wins; `0` auto-picks per provider
/// (API providers parallelize well; the in-process `local` LLM is mutex-guarded,
/// so 1).
fn effective_parallelism(requested: usize, provider: &str) -> usize {
    if requested >= 1 {
        return requested;
    }
    match provider.to_ascii_lowercase().as_str() {
        "deepseek" => 3,
        "gemini" => 5,
        _ => 1, // local / auto / unknown → serial
    }
}

/// Compile one scene end-to-end (translate → positive → negative). Never errors —
/// every LLM step falls back (verbatim / seed terms), so scenes are independent
/// and parallelizable.
async fn compile_one_scene(
    scene: &resolver::ResolvedScene,
    opts: &CompileOpts,
    eargs: &crate::prompt::EnhanceArgs,
) -> emitter::CompiledScene {
    // 0) translate the body to English first (LLM, unless --no-enhance).
    let body = match (&scene.translate, opts.no_enhance) {
        (Some(lang), false) if !lang.trim().is_empty() => {
            let sys = format!(
                "You are a translator. Translate the user's text from {lang} to English. \
                 Output ONLY the translation — no notes, no quotes, no markdown."
            );
            cached_call(&opts.provider, &sys, scene.free_text.trim(), cache::POSITIVE, opts.cache, eargs)
                .await
                .unwrap_or_else(|| scene.free_text.clone())
        }
        _ => scene.free_text.clone(),
    };

    // 1) personas → loaded fragments.
    let persona_fragments: Vec<String> = scene.personas.iter().map(|n| load_persona(n)).collect();

    let assembled = assembler::assemble_with_body(scene, &body);

    // 2) positive.
    let prompt = if opts.no_enhance || assembled.is_empty() {
        assembled.clone()
    } else {
        let sys = assembler::positive_system(scene, opts.system_override.as_deref(), &persona_fragments);
        cached_call(&opts.provider, &sys, &assembled, cache::POSITIVE, opts.cache, eargs)
            .await
            .map(|p| assembler::clean(&p))
            .unwrap_or_else(|| {
                tracing::warn!(target: "plakat", "compile: positive enhance failed for '{}', using verbatim", scene.name);
                assembled.clone()
            })
    };

    // 3) negative — from the final positive prompt (RFC step 9).
    let negative = if opts.no_negative {
        scene.negative_seeds.clone()
    } else {
        let nsys = assembler::negative_system(scene);
        let llm = cached_call(&opts.provider, &nsys, &prompt, cache::NEGATIVE, opts.cache, eargs)
            .await
            .map(|n| assembler::clean(&n));
        // Enforce the seed contract: `negative_system` tells the model the user's explicit
        // `negative:` terms MUST appear verbatim. A weak model sometimes ignores that and returns
        // a paraphrased *positive* instead (which, as a negative, would actively suppress the very
        // scene we want). When the output drops the user's seeds, distrust it entirely and fall
        // back to the seeds — never let a rogue auto-negative override an explicit `negative:`.
        match llm {
            Some(neg) if negative_honors_seeds(&neg, &scene.negative_seeds) => neg,
            _ => scene.negative_seeds.clone(),
        }
    };

    // 4) diligence warnings (6.26.2): budget overflow / dropped style. Style is only checked when
    // the enhancer actually ran (verbatim `--no-enhance` never injects the style directive).
    let warnings = assembler::scene_warnings(
        &scene.name,
        &scene.styles,
        &prompt,
        scene.family,
        !opts.no_enhance && !assembled.is_empty(),
    );

    emitter::CompiledScene { scene: scene.clone(), prompt, negative, warnings }
}

/// Whether an LLM-generated negative honours the user's explicit `negative:` seed terms — each
/// comma-separated term must appear (case-insensitively) in the output. Empty seeds means the user
/// gave no negative, so there's no contract to enforce and the LLM output is always accepted.
fn negative_honors_seeds(neg: &str, seeds: &str) -> bool {
    let neg_l = neg.to_lowercase();
    seeds
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .all(|t| neg_l.contains(&t.to_lowercase()))
}

/// One provider call, optionally cached. Returns the trimmed output, or None on
/// empty/failed (callers fall back to verbatim / seed terms).
async fn cached_call(
    provider: &str,
    system: &str,
    user: &str,
    namespace: &str,
    cache_on: bool,
    eargs: &crate::prompt::EnhanceArgs,
) -> Option<String> {
    let key = if cache_on { Some(cache::key(&[provider, system, user])) } else { None };
    if let Some(k) = &key {
        if let Some(hit) = cache::lookup(namespace, k) {
            return Some(hit);
        }
    }
    let out = match crate::prompt::complete(provider, system, user, eargs).await {
        Ok(v) if !v.trim().is_empty() => v.trim().to_string(),
        _ => return None,
    };
    if let Some(k) = &key {
        cache::store(namespace, k, &out);
    }
    Some(out)
}

/// Load a persona fragment from `~/.config/plakat/personas/<name>`; on miss the
/// name itself is used as the fragment (with a warn) so the prompt still gets a
/// persona cue.
fn load_persona(name: &str) -> String {
    let path = std::env::var_os("HOME")
        .map(|h| std::path::Path::new(&h).join(".config/plakat/personas").join(name));
    match path.and_then(|p| std::fs::read_to_string(p).ok()) {
        Some(content) => content,
        None => {
            tracing::warn!(target: "plakat", "compile: persona '{name}' not found in ~/.config/plakat/personas — using the name as the cue");
            name.to_string()
        }
    }
}

/// Compile a `prompts.txt` string into a scenario HJSML string. With
/// `no_enhance && no_negative` the whole pass is deterministic (the corpus gate).
/// Compile to the scenario HJSON plus any per-scene diligence warnings (6.26.2) — budget
/// overflow / dropped style — for the CLI to surface. The warnings never change the output.
pub async fn compile_to_string(input: &str, opts: &CompileOpts) -> anyhow::Result<(String, Vec<String>)> {
    let doc = parser::parse(input)?;
    let resolved = resolver::resolve(&doc, &opts.default_model)?;
    let eargs = crate::prompt::EnhanceArgs::default();

    let active: Vec<&resolver::ResolvedScene> = resolved.scenes.iter().filter(|s| !s.skip).collect();
    if active.is_empty() {
        anyhow::bail!("compile: every scene was skipped (skip: true)");
    }

    // Scenes are independent → run up to N concurrently. `buffered` preserves
    // input order, so the emitted task order is deterministic regardless of N.
    let n = effective_parallelism(opts.parallel, &opts.provider);
    let compiled: Vec<emitter::CompiledScene> = if n <= 1 {
        let mut v = Vec::with_capacity(active.len());
        for s in active.iter().copied() {
            v.push(compile_one_scene(s, opts, &eargs).await);
        }
        v
    } else {
        use futures_util::stream::{self, StreamExt};
        stream::iter(active.iter().copied().map(|s| compile_one_scene(s, opts, &eargs)))
            .buffered(n)
            .collect()
            .await
    };

    let warnings: Vec<String> = compiled.iter().flat_map(|c| c.warnings.clone()).collect();
    let hjson = emitter::emit(&resolved.globals, &compiled, &opts.input_name, &opts.provider);
    Ok((hjson, warnings))
}

/// Lint a `prompts.txt` without calling the LLM (E-C2): unknown commands and
/// misplaced `skip:` in the global block. Returns human-readable issues.
pub fn lint(input: &str) -> anyhow::Result<Vec<String>> {
    let doc = parser::parse(input)?;
    let mut issues = Vec::new();
    if let Some(g) = &doc.global {
        for (k, _) in &g.commands {
            if command_spec(k).is_none() {
                issues.push(format!("global block: unknown command `{k}:`"));
            }
            if k == "skip" {
                issues.push("global block: `skip:` is per-scene only".to_string());
            }
        }
    }
    // D2 (6.22.0): duplicate task names collide (scenario uses names as ids).
    let mut seen_names: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (i, s) in doc.scenes.iter().enumerate() {
        for (k, _) in &s.commands {
            if command_spec(k).is_none() {
                issues.push(format!(
                    "scene #{} (line {}): unknown command `{k}:`",
                    i + 1,
                    s.line_start
                ));
            }
        }
        // D2: duplicate command keys that don't allow repeats (e.g. two `seed:` lines).
        let mut keys_here: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for (k, _) in &s.commands {
            let repeatable = matches!(k.as_str(), "style" | "persona" | "lora") || k.contains('-');
            let n = keys_here.entry(k.as_str()).or_insert(0);
            *n += 1;
            if *n == 2 && !repeatable {
                issues.push(format!("scene #{} (line {}): command `{k}:` repeated (last wins — likely a mistake)", i + 1, s.line_start));
            }
        }
        // D2: duplicate task name across scenes.
        if let Some(name) = s.values("name").next() {
            if let Some(prev) = seen_names.insert(name.to_string(), i + 1) {
                issues.push(format!("scene #{}: duplicate task name {name:?} (already used by scene #{prev})", i + 1));
            }
        }
    }
    // E4 (6.22): model / scheduler typo checks — soft (a custom `org/repo` model is allowed).
    let known_models = crate::hf::all_known_aliases();
    let mut check = |where_: &str, block: Option<&parser::Block>| {
        let Some(b) = block else { return };
        for m in b.values("model") {
            if !m.is_empty() && !m.contains('/') && !known_models.iter().any(|a| *a == m) {
                issues.push(format!("{where_}: unknown model alias `{m}` (not a known alias or an `org/repo`)"));
            }
        }
        for sc in b.values("scheduler") {
            if !sc.is_empty() && sc.parse::<crate::pipelines::scheduler::SchedulerKind>().is_err() {
                issues.push(format!("{where_}: unknown scheduler `{sc}`"));
            }
        }
    };
    check("global block", doc.global.as_ref());
    for (i, s) in doc.scenes.iter().enumerate() {
        check(&format!("scene #{}", i + 1), Some(s));
    }
    Ok(issues)
}

/// Classify a model name into a family (priority: flux → xl → 1.5 → unknown).
pub fn classify_model(name: &str) -> ModelFamily {
    let n = name.to_ascii_lowercase();
    if n.contains("flux") {
        ModelFamily::Flux
    } else if n.contains("sdxl") || n.contains("xl") {
        ModelFamily::Sdxl
    } else if n.contains("sd15") || n.contains("1-5") || n.contains("1.5") || n.contains("sd35") || n.contains("sd3") {
        // sd15 / sd21 / sd35 all use comma-or-prose CLIP-ish prompting; the SD15
        // profile is the safe default for the non-XL, non-Flux SD family.
        ModelFamily::Sd15
    } else {
        ModelFamily::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_seed_contract_rejects_a_rogue_llm_negative() {
        let seeds = "blurry, low quality, watermark, bad anatomy, extra limbs";
        // The reported bug: the model returned a paraphrased POSITIVE as the negative —
        // none of the seed terms present → contract violated → must be rejected (→ fall back).
        let rogue = "A sunlit medieval street, timber-framed houses with stained glass and blue roofs";
        assert!(!negative_honors_seeds(rogue, seeds));
        // A cooperative negative that keeps the seeds (any case) + adds terms → accepted.
        let good = "Blurry, low quality, watermark, bad anatomy, extra limbs, jpeg artifacts, deformed hands";
        assert!(negative_honors_seeds(good, seeds));
        // Dropping even one required term fails the contract.
        assert!(!negative_honors_seeds("blurry, low quality, watermark, extra limbs", seeds));
        // No explicit negative → no contract → any LLM output is accepted.
        assert!(negative_honors_seeds("anything at all", ""));
    }

    #[test]
    fn lint_flags_duplicate_task_names_and_repeats() {
        // Two tasks named "dup" + a repeated non-repeatable command.
        let issues = lint("model: sdxl\n\nname: dup\nseed: 1\nseed: 2\nA tundra.\n\nname: dup\nA harbor.\n").unwrap();
        assert!(issues.iter().any(|i| i.contains("duplicate task name")), "dup name flagged: {issues:?}");
        assert!(issues.iter().any(|i| i.contains("`seed:` repeated")), "repeat flagged: {issues:?}");
        // A clean doc lints without issues.
        assert!(lint("model: sdxl\n\nname: a\nA tundra.\n\nname: b\nA harbor.\n").unwrap().is_empty());
    }

    #[test]
    fn lint_flags_unknown_model_and_scheduler() {
        let issues = lint("model: sdxl\n\nname: a\nmodel: modle-typo\nscheduler: dmp++\nA tundra.\n").unwrap();
        assert!(issues.iter().any(|i| i.contains("unknown model alias")), "model typo: {issues:?}");
        assert!(issues.iter().any(|i| i.contains("unknown scheduler")), "scheduler typo: {issues:?}");
        // A real `org/repo` model + a known scheduler are allowed.
        assert!(lint("model: sdxl\n\nname: a\nmodel: my-org/custom-sd\nscheduler: euler-a\nA tundra.\n").unwrap().is_empty());
    }

    #[test]
    fn classifies_model_families() {
        assert_eq!(classify_model("flux-dev"), ModelFamily::Flux);
        assert_eq!(classify_model("sdxl"), ModelFamily::Sdxl);
        assert_eq!(classify_model("stable-diffusion-xl-base"), ModelFamily::Sdxl);
        assert_eq!(classify_model("sd15"), ModelFamily::Sd15);
        assert_eq!(classify_model("sd35-medium"), ModelFamily::Sd15);
        assert_eq!(classify_model("some-unknown-thing"), ModelFamily::Unknown);
        // flux wins over a stray "xl"-less name; xl wins over 1.5 substrings.
        assert_eq!(classify_model("flux-xl-weird"), ModelFamily::Flux);
    }

    #[tokio::test]
    async fn no_enhance_no_negative_is_deterministic() {
        let input = "model: sdxl\nnegative: blurry\n\nheader: wide shot,\nA frozen tundra.\nfooter: 8k\nseed: 7\n";
        let opts = CompileOpts {
            provider: "auto".into(),
            default_model: "sdxl".into(),
            no_enhance: true,
            no_negative: true,
            system_override: None,
            cache: false,
            parallel: 0,
            input_name: "t.txt".into(),
        };
        let a = compile_to_string(input, &opts).await.unwrap().0;
        let b = compile_to_string(input, &opts).await.unwrap().0;
        assert_eq!(a, b, "deterministic with no LLM");
        assert!(a.contains("prompt: \"wide shot, A frozen tundra., 8k\""));
        assert!(a.contains("negative: \"blurry\""));
        assert!(a.contains("seed: 7"));
        // Must parse as the same HJSON `scenario` consumes.
        let _: serde_json::Value = deser_hjson::from_str(&a).expect("compiled HJSON parses");
    }

    #[test]
    fn parallelism_auto_picks_per_provider() {
        assert_eq!(effective_parallelism(4, "deepseek"), 4, "explicit wins");
        assert_eq!(effective_parallelism(0, "deepseek"), 3);
        assert_eq!(effective_parallelism(0, "gemini"), 5);
        assert_eq!(effective_parallelism(0, "local"), 1);
        assert_eq!(effective_parallelism(0, "auto"), 1);
    }

    #[test]
    fn lint_flags_unknown_commands() {
        let issues = lint("model: sdxl\nstyl: oops\n\nA scene.\nbogus: x\n").unwrap();
        assert_eq!(issues.len(), 2);
        assert!(issues.iter().any(|i| i.contains("styl")));
        assert!(issues.iter().any(|i| i.contains("bogus")));
        assert!(lint("model: sdxl\n\nA clean scene.\n").unwrap().is_empty());
    }

    #[test]
    fn command_table_lookup() {
        assert_eq!(command_spec("header").unwrap().merge, Merge::Concatenate);
        assert_eq!(command_spec("lora").unwrap().merge, Merge::AccumulateList);
        assert_eq!(command_spec("model").unwrap().kind, CommandKind::Scenario);
        assert_eq!(command_spec("style").unwrap().kind, CommandKind::Prompt);
        assert!(command_spec("bogus").is_none());
    }
}
