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

/// Salt mixed into every compile LLM cache key. Bump on any change to how the LLM is called (system
/// prompts, weight/negative handling) so stale — possibly wrong — cache entries are invalidated.
/// v2: `auto` now honours caller system prompts; negatives are deterministic (no LLM).
const CACHE_VERSION: &str = "compile-v2";
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
    // 6.26.x: reusable prompt pieces. `component.<name>:` (global) defines a fragment;
    // `composition:` (per-scene, comma-list of `component.<name>` refs) assembles them into the
    // prompt (before the block's own prose). `component.*` keys are matched by prefix (below).
    CommandSpec { key: "composition", kind: CommandKind::Prompt, merge: Merge::Concatenate },
    // ---- scenario commands (straight to HJSON) ----
    CommandSpec { key: "model",     kind: CommandKind::Scenario, merge: Merge::LastWins },
    CommandSpec { key: "lora",      kind: CommandKind::Scenario, merge: Merge::AccumulateList },
    CommandSpec { key: "loras",     kind: CommandKind::Scenario, merge: Merge::AccumulateList },
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
    // 6.26.x parity: regional prompting — repeatable `region: X0,Y0,X1,Y1[,w=][,feather=]:prompt`
    // → the task's `regions: [...]` array.
    CommandSpec { key: "region",    kind: CommandKind::Scenario, merge: Merge::AccumulateList },
    // 6.27.0 parity finish: repeatable `redux:` (→ task `redux-images: [...]`, Flux Redux refs) and
    // `control:` (compact `kind:image:strength` → the `controls: [{…}]` object array). `scene:` is
    // the per-task axis reference (`weather:` already above); `scene.<n>:`/`weather.<n>:` (global)
    // define the axes and are matched by prefix in `is_known_command`.
    CommandSpec { key: "redux",     kind: CommandKind::Scenario, merge: Merge::AccumulateList },
    CommandSpec { key: "control",   kind: CommandKind::Scenario, merge: Merge::AccumulateList },
    CommandSpec { key: "scene",     kind: CommandKind::Scenario, merge: Merge::LastWins },
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

/// 6.26.x parity: common **scalar** scenario fields that compile passes straight through to the
/// emitted HJSON (global or per-scene). These aren't shaped by the LLM — they're recognised (so
/// `--lint` accepts them) and written verbatim (type-inferred) by the emitter. The generic
/// `set.<key>: value` form covers anything not listed here (the long tail / future fields).
pub const PASSTHROUGH_KEYS: &[&str] = &[
    // sizing / device
    "aspect", "base", "device", "offline", "fast", "lcm",
    // post-process
    "naturalize", "upscale", "restore-faces", "restore-faces-model", "restore-faces-strength",
    // refiner + LoRA scale
    "refiner", "refine-strength", "refiner-frac", "lora-scale",
    // quality knobs (the guidance bundle)
    "pag-scale", "guidance-rescale", "freeu", "freeu-params", "dynamic-threshold",
    // style presets + per-task style/image refs
    "look", "genre", "style-ref", "style-strength", "concept-image",
    // img2img / inpaint (per-scene)
    "init-image", "strength", "mask", "mask-feather", "mask-invert", "outpaint",
    // animate (video)
    "format", "frames", "window-size", "window-overlap", "motion-lora", "motion-lora-scale", "gif-delay-ms",
    // flux / quant / advanced
    "kontext-bucket", "quantize-t5", "flux-quant-level", "t5-quant-level", "smart-zones",
];

/// Whether `key` is a pass-through scenario field: a known [`PASSTHROUGH_KEYS`] entry, or the
/// generic `set.<key>` form (the tail). The name after `set.` is the literal scenario key.
pub fn is_passthrough_key(key: &str) -> bool {
    PASSTHROUGH_KEYS.contains(&key) || key.strip_prefix("set.").is_some_and(|k| !k.is_empty())
}

/// The scenario key a pass-through directive writes: `set.<key>` → `<key>`; a bare known key → itself.
pub fn passthrough_target(key: &str) -> Option<&str> {
    if let Some(k) = key.strip_prefix("set.") {
        (!k.is_empty()).then_some(k)
    } else if PASSTHROUGH_KEYS.contains(&key) {
        Some(key)
    } else {
        None
    }
}

/// Whether `key` is a recognised command — a fixed [`COMMANDS`] key, a `component.<name>`
/// definition, or a pass-through scenario field (`set.<key>` or a known scalar). Used by the lint.
pub fn is_known_command(key: &str) -> bool {
    command_spec(key).is_some()
        || key.strip_prefix("component.").is_some_and(|n| !n.is_empty())
        // 6.27.0: `scene.<name>:` / `weather.<name>:` define the scenario's scene/weather axes.
        || key.strip_prefix("scene.").is_some_and(|n| !n.is_empty())
        || key.strip_prefix("weather.").is_some_and(|n| !n.is_empty())
        || is_passthrough_key(key)
}

/// SD model family — drives the family-specific LLM system-prompt section.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ModelFamily {
    Sd15,
    Sdxl,
    /// SD3 / SD3.5 — CLIP-ish prose prompting like SD15, but the T5-XXL text encoder carries a much
    /// larger token budget, so the 77-token CLIP cap does NOT apply.
    Sd3,
    /// Stable Cascade (Würstchen v3) — CLIP text encoders; descriptive prompting, and it does NOT honour
    /// A1111 `(term:N)` attention weights (plakat's Cascade pipeline has no weight parser).
    Cascade,
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
            ModelFamily::Sd3 => "SD3",
            ModelFamily::Cascade => "Cascade",
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
    // Attention weights are TRANSLATED and kept INLINE. For each `(phrase:N)` we translate the phrase on its
    // own (a reliable, unambiguous ask), substitute the English `(phrase_en:N)` back at its ORIGINAL position,
    // THEN enhance — so the model sees the emphasis inline, in English, on the same tokens the prose uses
    // (source-language weights would emphasise tokens an English-trained CLIP/T5 can't represent). A safety
    // net re-adds — in English — any span a weak enhancer still drops. `--no-enhance` keeps the user's
    // verbatim text (wording + weights untouched, no translation).

    // 1) personas + assemble with the ORIGINAL body (weights + wording intact; the enhancer translates it).
    let persona_fragments: Vec<String> = scene.personas.iter().map(|n| load_persona(n)).collect();
    let assembled = assembler::assemble_with_body(scene, &scene.free_text);

    // 2) translate every weighted phrase to English → phrase(src)→(English, weight) map (enhance path only).
    let spans = dedup_spans(assembler::extract_weight_spans(&assembled));
    let mut en_map: std::collections::HashMap<String, (String, f32)> = std::collections::HashMap::new();
    let mut weight_note: Option<String> = None;
    if !opts.no_enhance {
        let mut failed = 0usize;
        for (phrase, w) in &spans {
            let en = match &scene.translate {
                Some(lang) if !lang.trim().is_empty() => translate_phrase(phrase, lang, &opts.provider, opts.cache, eargs).await,
                _ => Some(phrase.clone()), // no `translate:` → already the target language
            };
            match en {
                Some(t) => {
                    en_map.insert(phrase.clone(), (t, *w));
                }
                None => {
                    failed += 1;
                    en_map.insert(phrase.clone(), (phrase.clone(), *w)); // keep source-language rather than drop
                }
            }
        }
        if failed > 0 {
            weight_note = Some(format!(
                "scene '{}': could not translate {failed} attention-weighted phrase(s) — they stay in the source \
                 language; try a different `--compile-provider`",
                scene.name
            ));
        }
    }

    // Cascade's text encoders don't honour `(term:N)` weights, so keeping them inline would just add noisy
    // punctuation tokens — for Cascade we strip to the plain phrase and let prose reinforcement (2d) carry
    // the emphasis. Every other family keeps the inline weight (their CLIP/T5 encoders apply it).
    let keep_weights = !matches!(scene.family, ModelFamily::Cascade);

    // 3) substitute the English weighted spans back INLINE, at their original positions (enhance path only;
    //    `--no-enhance` keeps the source text verbatim). The prose around them is still source-language here —
    //    the enhancer translates it and only has to KEEP the already-English `(phrase:N)` spans.
    let prepared = if opts.no_enhance {
        assembled.clone()
    } else {
        assembler::rewrite_weight_spans(&assembled, |p, w| {
            let en = en_map.get(p).map(|(t, _)| t.as_str()).unwrap_or(p);
            if keep_weights {
                format!("({}:{})", en, w)
            } else {
                en.to_string()
            }
        })
    };

    // 4) positive enhance.
    let mut prompt = if opts.no_enhance || prepared.is_empty() {
        assembled.clone()
    } else {
        let sys = assembler::positive_system(scene, opts.system_override.as_deref(), &persona_fragments);
        cached_call(&opts.provider, &sys, &prepared, cache::POSITIVE, opts.cache, eargs)
            .await
            .map(|p| assembler::clean(&p))
            .unwrap_or_else(|| {
                tracing::warn!(target: "plakat", "compile: positive enhance failed for '{}', using verbatim", scene.name);
                prepared.clone()
            })
    };

    // 5) safety net: re-add — in English, as a short tail — any weighted span the enhancer dropped, so no
    //    emphasis is ever silently lost. The ones it KEPT stay inline (matched by substring), so there's no
    //    duplication. Ordered by `spans` for deterministic output.
    let mut weights_tailed = 0usize;
    if !opts.no_enhance && !en_map.is_empty() && keep_weights {
        let have: Vec<String> =
            assembler::extract_weight_spans(&prompt).into_iter().map(|(p, _)| p.to_lowercase()).collect();
        let missing: Vec<String> = spans
            .iter()
            .filter_map(|(src, _)| en_map.get(src))
            .filter(|(en, _)| {
                let e = en.trim().to_lowercase();
                !e.is_empty() && !have.iter().any(|h| h.contains(&e) || e.contains(h))
            })
            .map(|(en, w)| format!("({}:{})", en.trim().trim_end_matches(['.', ',']), w))
            .collect();
        weights_tailed = missing.len();
        if !missing.is_empty() {
            let sep = if prompt.trim_end().ends_with(',') || prompt.trim().is_empty() { " " } else { ", " };
            prompt = format!("{}{sep}{}", prompt.trim_end(), missing.join(", "));
            if weight_note.is_none() {
                weight_note = Some(format!(
                    "scene '{}': the enhancer dropped {} inline weight(s) — re-added them (English) at the end",
                    scene.name,
                    missing.len()
                ));
            }
        }
    }

    // 2c) fit-to-budget: if the finished prompt overflows the model's effective token budget, condense it
    // (model-specific) to fit while preserving subjects/style/weights, rather than only warning. Only when
    // the enhancer ran (verbatim `--no-enhance` is the user's exact wording — never rewrite that).
    let mut fit_note: Option<String> = None;
    if !opts.no_enhance && !assembled.is_empty() {
        let (fitted, note) = fit_to_budget(&prompt, scene.family, &scene.name, &opts.provider, opts.cache, eargs).await;
        prompt = fitted;
        fit_note = note;
    }

    // 2d) SD3/Flux prose reinforcement: these T5-driven families honour prose >> numeric weights, so a
    // heavily-weighted concept still loses to strong priors (a green sky, an orange sun). Restate the
    // weighted concepts as a short prose intensifier appended to the prompt — the inline `(term:N)` weights
    // stay for the CLIP encoders. Runs after fit so it isn't condensed away; the SD3/Flux budgets (256/300)
    // leave ample room for the clause. Only on the enhance path.
    let mut reinforced = false;
    if !opts.no_enhance {
        if let Some(clause) = assembler::prose_reinforcement(&prompt, scene.family) {
            let sep = if prompt.trim_end().ends_with(['.', ',']) || prompt.trim().is_empty() { " " } else { ". " };
            prompt = format!("{}{sep}{clause}", prompt.trim_end());
            reinforced = true;
        }
    }

    // 3) negative — DETERMINISTIC, not LLM-generated. Asking a model to "write a negative" makes it
    // hallucinate scene-specific content exclusions the user never asked for (weapons, blood, colours…) and
    // can even run away into repetition. Instead: the user's `negative:` seeds (verbatim) plus a curated,
    // generic QUALITY negative — deduped and capped. `--no-negative` keeps just the seeds.
    let negative = if opts.no_negative {
        scene.negative_seeds.clone()
    } else {
        assembler::auto_negative(scene)
    };

    // 4) name upgrade (6.26.2): when the name was auto-derived AND the LLM enhanced the prompt,
    // slug a MEANINGFUL name from the English prompt (so a `translate:` scene named `scene_1`
    // becomes e.g. `a_clean_medieval_western_european_street`). Explicit `name:` is left untouched;
    // if the prompt still yields no ASCII slug, the sequential `scene_N` stands.
    let mut out_scene = scene.clone();
    if scene.name_auto && !opts.no_enhance {
        if let Some(better) = resolver::slug_from_text(&prompt) {
            out_scene.name = better;
        }
    }

    // 5) diligence warnings (6.26.2): budget overflow / dropped style. Style is only checked when
    // the enhancer actually ran (verbatim `--no-enhance` never injects the style directive).
    let mut warnings = assembler::scene_warnings(
        &out_scene.name,
        &scene.styles,
        &prompt,
        scene.family,
        !opts.no_enhance && !assembled.is_empty(),
    );

    // Informational TRACE (6.27): the steps taken for this scene, so compilation isn't a black box.
    let mut trace: Vec<String> = Vec::new();
    let enhanced = !opts.no_enhance && !assembled.is_empty();
    if !scene.composition_text.trim().is_empty() {
        trace.push("composed prompt from components".to_string());
    }
    if enhanced {
        if let Some(lang) = scene.translate.as_deref().filter(|l| !l.trim().is_empty()) {
            trace.push(format!("translated from {lang} → English"));
        }
    }
    if !spans.is_empty() {
        if !enhanced {
            trace.push(format!("{} attention weight(s) kept verbatim (--no-enhance)", spans.len()));
        } else if !keep_weights {
            trace.push(format!(
                "{} attention weight(s) stripped — {} ignores `(term:N)`; emphasis applied via prose",
                spans.len(),
                scene.family.label()
            ));
        } else {
            let inline = spans.len().saturating_sub(weights_tailed);
            trace.push(format!(
                "{} attention weight(s): {inline} translated inline{}",
                spans.len(),
                if weights_tailed > 0 { format!(", {weights_tailed} re-added at end (enhancer dropped)") } else { String::new() }
            ));
        }
    }
    if enhanced {
        trace.push(format!("positive enhanced via {}", crate::prompt::resolve_provider_label(&opts.provider)));
    } else {
        trace.push("positive kept verbatim (--no-enhance)".to_string());
    }
    trace.push(format!(
        "~{} tokens (budget ~{}, {})",
        assembler::estimate_tokens(&prompt),
        assembler::family_token_budget(scene.family),
        scene.family.label()
    ));
    if fit_note.is_some() {
        trace.push("condensed to fit the token budget".to_string());
    }
    if reinforced {
        trace.push("reinforced weighted concepts as prose (SD3/Flux honour prose > weights)".to_string());
    }
    trace.push(if opts.no_negative {
        "negative: your seeds only (--no-negative)".to_string()
    } else {
        format!("negative: {} terms (seeds + curated quality)", negative.split(',').filter(|t| !t.trim().is_empty()).count())
    });
    if out_scene.name != scene.name {
        trace.push(format!("named from prompt → {}", out_scene.name));
    }

    // Attention-weight note (from step 2b): success = weights were re-applied deterministically; failure =
    // re-translation failed and the (corrected) advice stands. Either way it's surfaced, not silent.
    if let Some(note) = weight_note {
        warnings.push(note);
    }
    // Fit-to-budget note (from step 2c): the prompt was condensed to fit the model's token budget.
    if let Some(note) = fit_note {
        warnings.push(note);
    }

    emitter::CompiledScene { scene: out_scene, prompt, negative, warnings, trace }
}

/// Deduplicate weight spans by (phrase, weight), preserving first-seen order — so a phrase repeated across
/// components is translated + re-injected once.
fn dedup_spans(spans: Vec<(String, f32)>) -> Vec<(String, f32)> {
    let mut seen = std::collections::HashSet::new();
    spans
        .into_iter()
        .filter(|(p, w)| seen.insert((p.to_lowercase(), w.to_bits())))
        .collect()
}

/// Model-specific **fit-to-budget** (RFC step): when the finished prompt exceeds the family's effective
/// token budget, ask the LLM to condense it to fit — preserving every distinct subject, the style, and the
/// attention weights — then GUARANTEE the weights survived by re-appending any the fit pass dropped.
/// Returns `(prompt, Some(note))` when it condensed the prompt to within budget (the note is user-facing);
/// `(prompt, None)` when it already fit, the fit call failed, or it condensed but still couldn't reach
/// budget — in the last case the (smaller) condensed prompt is returned and `scene_warnings` reports the
/// remaining overflow, so there's exactly one message either way.
async fn fit_to_budget(
    prompt: &str,
    family: ModelFamily,
    scene_name: &str,
    provider: &str,
    cache_on: bool,
    eargs: &crate::prompt::EnhanceArgs,
) -> (String, Option<String>) {
    let budget = assembler::family_token_budget(family);
    let before = assembler::estimate_tokens(prompt);
    if before <= budget {
        return (prompt.to_string(), None);
    }
    let spans = dedup_spans(assembler::extract_weight_spans(prompt));
    let sys = format!(
        "You compress text-to-image prompts to a token budget for the {label} model. Rewrite the prompt to \
         fit within about {budget} CLIP tokens. PRESERVE every attention-weight span `(phrase:number)` \
         EXACTLY — keep the parentheses and the number unchanged. Keep every distinct visual subject and the \
         overall style; cut only filler, repetition and redundant adjectives. Output ONLY the rewritten prompt.",
        label = family.label()
    );
    let fitted = match cached_call(provider, &sys, prompt, cache::POSITIVE, cache_on, eargs).await {
        Some(f) => assembler::clean(&f),
        None => return (prompt.to_string(), None), // fit call failed — keep original; scene_warnings flags it
    };
    // Guarantee the weights survived the compression: only re-append if the fit pass lost them ALL (matches
    // step 2b's all-or-nothing — avoids duplicating weights the fit pass kept).
    let mut out = fitted;
    if !spans.is_empty() && assembler::weight_span_count(&out) == 0 {
        let tail = spans.iter().map(|(p, w)| format!("({}:{})", p.trim(), w)).collect::<Vec<_>>().join(", ");
        let sep = if out.trim_end().ends_with(',') || out.trim().is_empty() { " " } else { ", " };
        out = format!("{}{sep}{tail}", out.trim_end());
    }
    let after = assembler::estimate_tokens(&out);
    if after <= budget {
        let note = format!(
            "scene '{scene_name}': prompt was ~{before} tokens (over the {label} ~{budget}-token budget) — condensed to ~{after} to fit, weights preserved",
            label = family.label()
        );
        (out, Some(note))
    } else {
        // Smaller than the original but still over — hand back the condensed text; `scene_warnings` reports
        // the residual overflow (single message), which is honest.
        (out, None)
    }
}

/// Translate a single short phrase to English (used to re-apply attention weights the enhancer flattened).
/// One phrase per call — an unambiguous request a model handles reliably. Returns None on empty/failed.
async fn translate_phrase(
    phrase: &str,
    lang: &str,
    provider: &str,
    cache_on: bool,
    eargs: &crate::prompt::EnhanceArgs,
) -> Option<String> {
    let sys = format!(
        "Translate the following short phrase from {lang} into English. Return the English translation and \
         nothing else — no quotes, no notes, no trailing punctuation, no markdown, no {lang} text."
    );
    cached_call(provider, &sys, phrase.trim(), cache::POSITIVE, cache_on, eargs)
        .await
        .map(|t| t.trim().trim_matches('"').trim_end_matches(['.', ',']).to_string())
        .filter(|t| !t.is_empty())
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
    // A version salt in the cache key: bump `CACHE_VERSION` whenever LLM-call semantics change so old
    // entries can't return a now-wrong result (e.g. translations cached by a run before `auto` honoured the
    // system prompt). Changing it invalidates every entry at once.
    let key = if cache_on { Some(cache::key(&[CACHE_VERSION, provider, system, user])) } else { None };
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
pub async fn compile_to_string(input: &str, opts: &CompileOpts) -> anyhow::Result<(String, Vec<String>, Vec<String>)> {
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
    let mut compiled: Vec<emitter::CompiledScene> = if n <= 1 {
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

    // De-duplicate AUTO-derived names (two scenes can slug to the same words) — a numeric suffix
    // keeps each task's output directory distinct. Explicit `name:` values are never touched (a
    // real collision there is the user's to fix, and `--lint` flags it).
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for c in &mut compiled {
        if !c.scene.name_auto {
            continue;
        }
        let count = seen.entry(c.scene.name.clone()).or_insert(0);
        *count += 1;
        if *count > 1 {
            c.scene.name = format!("{}_{}", c.scene.name, *count);
        }
    }

    let warnings: Vec<String> = compiled.iter().flat_map(|c| c.warnings.clone()).collect();

    // Per-scene informational trace (6.27): a header line + the scene's steps, so the CLI can show what the
    // pipeline did. Uses the FINAL (deduped) scene name.
    let mut trace: Vec<String> = Vec::new();
    for c in &compiled {
        trace.push(format!("scene '{}' · {} · {}", c.scene.name, c.scene.family.label(), c.scene.model_for_family.as_deref().unwrap_or("(default)")));
        for step in &c.trace {
            trace.push(format!("  {step}"));
        }
    }

    let hjson = emitter::emit(&resolved.globals, &compiled, &opts.input_name, &opts.provider);
    Ok((hjson, warnings, trace))
}

/// Lint a `prompts.txt` without calling the LLM (E-C2): unknown commands and
/// misplaced `skip:` in the global block. Returns human-readable issues.
pub fn lint(input: &str) -> anyhow::Result<Vec<String>> {
    let doc = parser::parse(input)?;
    let mut issues = Vec::new();
    if let Some(g) = &doc.global {
        for (k, _) in &g.commands {
            if !is_known_command(k) {
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
            if !is_known_command(k) {
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
            let repeatable = matches!(k.as_str(), "style" | "persona" | "lora" | "loras") || k.contains('-');
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
    } else if n.contains("cascade") || n.contains("wuerstchen") || n.contains("würstchen") {
        ModelFamily::Cascade
    } else if n.contains("sdxl") || n.contains("xl") {
        ModelFamily::Sdxl
    } else if n.contains("sd35") || n.contains("sd3") {
        // SD3 / SD3.5: prose prompting like SD15, but the T5-XXL encoder gives a MUCH larger token
        // budget — so it gets its own family (77-token CLIP cap must not be imposed on it).
        ModelFamily::Sd3
    } else if n.contains("sd15") || n.contains("1-5") || n.contains("1.5") || n.contains("sd21") || n.contains("2-1") {
        // sd15 / sd21 use comma-or-prose CLIP-ish prompting; the SD15 profile is the safe default for
        // the non-XL, non-Flux, non-SD3 stable-diffusion family.
        ModelFamily::Sd15
    } else {
        ModelFamily::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(classify_model("sd35-medium"), ModelFamily::Sd3);
        assert_eq!(classify_model("sd35"), ModelFamily::Sd3);
        assert_eq!(classify_model("sd3"), ModelFamily::Sd3);
        assert_eq!(classify_model("stable-cascade"), ModelFamily::Cascade);
        assert_eq!(classify_model("cascade"), ModelFamily::Cascade);
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

    #[tokio::test]
    async fn scene_weather_axes_redux_control_emit_and_load() {
        // 6.27.0: axes from prose + per-task refs + redux + control object-array all emit AND the
        // scenario the compiler produces actually LOADS (deser + known task types).
        let input = "model: sdxl\nscene.morning: soft dawn\nweather.rain: heavy rain\n\nscene: morning\nweather: rain\nredux: a.jpg\ncontrol: depth:h.png:0.8\nA street.\n";
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
        let out = compile_to_string(input, &opts).await.unwrap().0;
        assert!(out.contains("{ name: \"morning\", prompt: \"soft dawn\" }"), "scene axis: {out}");
        assert!(out.contains("scene: morning") && out.contains("weather: rain"), "task refs: {out}");
        assert!(out.contains("redux-images: [\"a.jpg\"]"), "redux: {out}");
        assert!(out.contains("{ kind: \"depth\", image: \"h.png\", strength: 0.8 }"), "control: {out}");
        // The emitted scenario is loadable (this is what the negative/parity work guarantees).
        crate::cli::scenario::validate_hjson(&out).expect("compiled scenario with axes/redux/control loads");
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
