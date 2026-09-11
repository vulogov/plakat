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
    // 6.28: explicit relationship layer. `relate: <component> <relationship> <component>` (repeatable)
    // declares how two named objects relate (`relate: tram on rails`) → an English grounding clause built
    // from the components' (translated) descriptions + a verb phrase, prepended to the prose.
    CommandSpec { key: "relate", kind: CommandKind::Prompt, merge: Merge::AccumulateList },
    // 6.28: the 3-tier figure model. `foreground:` = the 1–2 DELIBERATE hero components (full OpenPose+region
    // treatment; sets control-generate-max-figures). `background:` = specific people painted in the base (no
    // skeleton). `crowd:` = free-text unspecified group (pure atmosphere); `crowd-density:` = sparse/moderate/
    // dense. Comma-lists of `component.<name>` (foreground/background) or free text (crowd).
    CommandSpec { key: "foreground", kind: CommandKind::Prompt, merge: Merge::Concatenate },
    CommandSpec { key: "background", kind: CommandKind::Prompt, merge: Merge::Concatenate },
    CommandSpec { key: "crowd", kind: CommandKind::Prompt, merge: Merge::Concatenate },
    CommandSpec { key: "crowd-density", kind: CommandKind::Prompt, merge: Merge::LastWins },
    // 6.28: `lora-trigger:` — a LoRA's activation token(s). PREPENDED verbatim to the FINAL prompt (never
    // sent through enhance/translate/fit, which would rewrite or drop a non-semantic trigger), and its tokens
    // are RESERVED from the fit budget so the total still fits. Global + per-task (concatenated).
    CommandSpec { key: "lora-trigger", kind: CommandKind::Prompt, merge: Merge::Concatenate },
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
    "naturalize", "relight", "relight-when", "upscale", "restore-faces", "restore-faces-model", "restore-faces-strength",
    // auto-ranking (generate → rank → cull → regenerate; process only survivors)
    "ranking",
    // output layout — keep every pass of a repeated run side by side
    "unique-files", "keep-prenaturalize",
    // SD3 ControlNet opt-in (memory-heavy; off by default)
    "sd3controlnet",
    // 6.28 two-pass structure: SDXL/composition draft → task-model img2img finish
    "control-generate", "control-preimage",
    "control-generate-strength", "control-generate-count", "control-generate-min-score",
    "control-generate-tries", "control-generate-size", "control-generate-mode",
    "control-generate-opportunistic", "control-generate-figure-shuffle", "control-generate-finish",
    "control-generate-seed", "control-generate-max-figures", "control-generate-inpaint-figures",
    "control-generate-regional",
    // 6.29: hand-declarable pose/contact/object hints (normally auto-generated from relates, but a user may
    // set them directly — e.g. a `carrying` pose that has no relate verb). Passed through VERBATIM.
    "control-generate-figure-poses", "control-generate-figure-contacts", "control-generate-objects",
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
        // Compile MANAGES enhancement itself: it enhances at compile time (unless `--no-enhance`) and always
        // emits `enhance: false` per task so the scenario won't re-enhance. A prose `enhance:`/`enhancer:` is
        // accepted (no lint error) but ignored — the value has no effect on compile's own enhancement.
        || matches!(key, "enhance" | "enhancer")
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
/// The SYSTEM prompt for the `control-generate` STRUCTURE draft. It is consumed by **SDXL** (77-token CLIP),
/// not sd35's long-context T5 — so it must be CONCISE, NOT verbose prose. It STRIPS every style/medium word
/// (the finish owns style), but it MUST keep each subject's ACTION and spatial RELATIONSHIP (the verbs that
/// define pose and interaction) — condensing those away turns "leaning on a cane" into a disconnected cane.
const STRUCTURE_SYSTEM: &str = "You rewrite a scene into a CONCISE SDXL prompt for a composition/layout image \
    (an img2img / ControlNet base for SDXL, whose text encoder holds only ~75 tokens). Output ONE compact \
    English prompt, aim ~90 tokens, as short comma-separated phrases. Order:\n\
    1) the scene/setting in a few words;\n\
    2) EACH subject as a terse phrase that KEEPS its action and interaction — '[position] [subject] [key \
    attributes], [what they are DOING]'. You MUST preserve the verb/relationship (holding X, carrying X, \
    leaning on X, sitting on X, stepping out of X, standing beside X, talking to X, reaching for X): these \
    define the POSE and the composition. Do NOT reduce a subject to a bag of nouns — a '[verb]-ing on/with X' \
    phrase must stay a phrase, never collapse to a stray noun 'X'. Keep position to ONE word \
    (foreground/left/right/centre/background) plus near/far if relevant;\n\
    3) concrete environment/colour facts and any (weighted:N) spans verbatim (e.g. '(pale green sky:1.4), \
    (orange sun:1.4)');\n\
    4) end with 'correct anatomy, natural proportions, distinct separated figures'.\n\
    STRIP every style/medium/mood/rendering word (impressionist, painting, painterly, soft focus, loose \
    brushwork, watercolour, oil, muted, atmospheric, delicate, soft, 'without detailed portraits') — NEVER \
    include any. Be compact, but never at the cost of an action or relationship. Translate to English. Output \
    ONLY the prompt, no preamble, no quotes.";

async fn compile_one_scene(
    scene: &resolver::ResolvedScene,
    opts: &CompileOpts,
    eargs: &crate::prompt::EnhanceArgs,
    wants_structure: bool,
) -> emitter::CompiledScene {
    // Attention weights are TRANSLATED and kept INLINE. For each `(phrase:N)` we translate the phrase on its
    // own (a reliable, unambiguous ask), substitute the English `(phrase_en:N)` back at its ORIGINAL position,
    // THEN enhance — so the model sees the emphasis inline, in English, on the same tokens the prose uses
    // (source-language weights would emphasise tokens an English-trained CLIP/T5 can't represent). A safety
    // net re-adds — in English — any span a weak enhancer still drops. `--no-enhance` keeps the user's
    // verbatim text (wording + weights untouched, no translation).

    // 1) personas + assemble with the ORIGINAL body (weights + wording intact; the enhancer translates it).
    // 6.28 `relate:` — build an English grounding clause from the declared object relationships (component
    // descriptions translated, verb → phrase, first mention full / repeat by name) and PREPEND it to the
    // prose, so it flows through the same translate+enhance pipeline. Explicit + language-agnostic.
    let persona_fragments: Vec<String> = scene.personas.iter().map(|n| load_persona(n)).collect();
    // 3-tier figures (foreground heroes with folded attach-objects · background people · crowd) +
    // PLACE-relation grounding + the prose. Attach relations are already folded into each figure's desc.
    let figures = build_figures_clause(scene);
    let grounding = build_grounding(scene, opts, eargs).await;
    let body = [figures, grounding, scene.free_text.trim().to_string()]
        .into_iter()
        .filter(|p| !p.trim().is_empty())
        .collect::<Vec<_>>()
        .join(". ");
    let assembled = assembler::assemble_with_body(scene, &body);

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
    let mut positive_failed = false;
    let mut prompt = if opts.no_enhance || prepared.is_empty() {
        assembled.clone()
    } else {
        let sys = assembler::positive_system(scene, opts.system_override.as_deref(), &persona_fragments);
        match cached_call(&opts.provider, &sys, &prepared, cache::POSITIVE, opts.cache, eargs).await {
            Some(p) => assembler::clean(&p),
            None => {
                positive_failed = true;
                tracing::warn!(target: "plakat", "compile: positive enhance failed for '{}', using verbatim", scene.name);
                prepared.clone()
            }
        }
    };

    // 4b) 6.28 STRUCTURE prompt for `control-generate`: enhance the SAME content prose (`prepared`) but with
    // a composition-focused, style-STRIPPED directive instead of the user's style — so SDXL's structure
    // draft is a clean layout base (figures placed, environment present), not a soft-focus rendering that
    // fights the finish. Only when the scenario uses control-generate (avoids a needless LLM call otherwise).
    let structure_prompt: Option<String> = if wants_structure && !opts.no_enhance && !prepared.is_empty() {
        // A DEDICATED style-stripping system prompt (not the style-carrying positive one) + persona
        // fragments so named characters keep their identity while all style/medium words are removed.
        let mut ssys = STRUCTURE_SYSTEM.to_string();
        for frag in &persona_fragments {
            ssys.push_str("\n\nCharacter (keep identity, drop any style words): ");
            ssys.push_str(frag);
        }
        cached_call(&opts.provider, &ssys, &prepared, cache::STRUCTURE, opts.cache, eargs)
            .await
            .map(|p| assembler::clean(&p))
    } else {
        None
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

    // fit-to-budget runs LAST (step 2e below), AFTER the prose reinforcement — otherwise the reinforcement
    // clauses append after a fit and push the prompt back over budget (the ~256→260 overflow bug).
    let mut fit_note: Option<String> = None;

    // 2d) SD3/Flux prose reinforcement: these T5-driven families honour prose >> numeric weights, so a
    // heavily-weighted concept still loses to strong priors (a green sky, an orange sun). Restate the
    // weighted concepts as a short prose intensifier appended to the prompt — the inline `(term:N)` weights
    // stay for the CLIP encoders. Runs after fit so it isn't condensed away; the SD3/Flux budgets (256/300)
    // leave ample room for the clause. Only on the enhance path.
    // 6.28: generic relationship grounding — detect object-to-object relationships on the ORIGINAL prompt
    // (before any clause is appended; the grounding clause itself contains markers). Drives a short
    // coherent-placement clause (prose families) + generic relationship-violation negatives below.
    let has_rel = assembler::has_relationships(&prompt) || !scene.relations.is_empty();
    let mut reinforced = false;
    if !opts.no_enhance {
        if let Some(clause) = assembler::prose_reinforcement(&prompt, scene.family) {
            let sep = if prompt.trim_end().ends_with(['.', ',']) || prompt.trim().is_empty() { " " } else { ". " };
            prompt = format!("{}{sep}{clause}", prompt.trim_end());
            reinforced = true;
        }
        if has_rel {
            if let Some(clause) = assembler::relationship_reinforcement(scene.family) {
                let sep = if prompt.trim_end().ends_with(['.', ',']) || prompt.trim().is_empty() { " " } else { ". " };
                prompt = format!("{}{sep}{clause}", prompt.trim_end());
            }
        }
    }

    // 2e) fit-to-budget — LAST, so the enhanced + reinforced prompt is condensed to the model's effective
    // token budget (preserving every distinct subject, the style, and the inline weights) rather than only
    // warning. Runs after the reinforcement clauses so they can't push it back over budget. Reserves room for
    // the `lora-trigger:` that gets prepended below.
    let trigger = scene.lora_trigger.trim();
    let trigger_reserve = if trigger.is_empty() { 0 } else { assembler::estimate_tokens(trigger) + 2 };
    if !opts.no_enhance && !assembled.is_empty() {
        let (fitted, note) = fit_to_budget(&prompt, scene.family, &scene.name, &opts.provider, opts.cache, eargs, trigger_reserve).await;
        prompt = fitted;
        fit_note = note;
    }
    // 2f) PREPEND the LoRA trigger verbatim — after enhance/fit/reinforce, so a non-semantic activation token
    // can't be rewritten or dropped, and the budget already reserved its tokens (2e) so the total fits.
    if !trigger.is_empty() {
        let sep = if prompt.trim().is_empty() { "" } else { ", " };
        prompt = format!("{trigger}{sep}{}", prompt.trim());
    }

    // 3) negative — HYBRID: a deterministic base (the user's `negative:` seeds + a curated QUALITY set)
    // ALWAYS, plus — on the enhance path — a FEW LLM scene-specific DEFECT terms. Bounded so the old
    // failures can't recur: the whole thing is deduped + capped (no runaway), the LLM is told to add only
    // defects (never content), and `strip_terms_in_positive` HARD-drops any suggestion echoing the positive
    // prompt (so it can never negate a wanted green sky / orange sun). Flux ignores negatives → seeds only;
    // `--no-negative` → seeds only; `--no-enhance` → deterministic (no LLM).
    let mut scene_negatives = false;
    let negative = if opts.no_negative {
        scene.negative_seeds.clone()
    } else if opts.no_enhance || matches!(scene.family, ModelFamily::Flux) {
        assembler::auto_negative(scene)
    } else {
        let sys = assembler::negative_scene_system();
        let llm = cached_call(&opts.provider, sys, &prompt, cache::NEGATIVE, opts.cache, eargs)
            .await
            .map(|t| assembler::clean(&t))
            .map(|t| assembler::strip_terms_in_positive(&t, &prompt))
            .filter(|t| !t.trim().is_empty());
        match llm {
            Some(scene_terms) => {
                scene_negatives = true;
                assembler::merge_negative_terms(&[&scene.negative_seeds, &scene_terms, assembler::QUALITY_NEGATIVE], 40)
            }
            None => assembler::auto_negative(scene),
        }
    };
    // 6.28: weight-free relationship grounding (negative side) — when the scene describes object
    // relationships, merge generic relationship-violation terms (floating / detached / merging), deduped
    // and capped. Object-agnostic. Flux ignores negatives; `--no-negative` stays seeds-only.
    let negative = if has_rel && !opts.no_negative && !matches!(scene.family, ModelFamily::Flux) {
        assembler::merge_negative_terms(&[&negative, assembler::RELATIONSHIP_NEGATIVE], 48)
    } else {
        negative
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
    // 6.29: the DECLARED-pose descriptions must match the runtime planner's figure labels, which are built
    // from the ENGLISHED structure-prompt. So translate each pose's description the same way the grounding
    // clause is (source-language descs would share no tokens with the English labels → no pose applied).
    if !opts.no_enhance {
        if let Some(lang) = scene.translate.clone().filter(|l| !l.trim().is_empty()) {
            for (desc, _pose) in out_scene.figure_poses.iter_mut() {
                if let Some(en) = translate_phrase(desc, &lang, &opts.provider, opts.cache, eargs).await {
                    *desc = en;
                }
            }
            // Contact descriptions must match the runtime figure labels too (English planner labels).
            for (a, b) in out_scene.figure_contacts.iter_mut() {
                if let Some(en) = translate_phrase(a, &lang, &opts.provider, opts.cache, eargs).await {
                    *a = en;
                }
                if let Some(en) = translate_phrase(b, &lang, &opts.provider, opts.cache, eargs).await {
                    *b = en;
                }
            }
            // Object descriptions likewise match the English planner object labels + drive the region prompt.
            for (_name, desc) in out_scene.objects.iter_mut() {
                if let Some(en) = translate_phrase(desc, &lang, &opts.provider, opts.cache, eargs).await {
                    *desc = en;
                }
            }
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
        let label = crate::prompt::resolve_provider_label(&opts.provider);
        if positive_failed {
            trace.push(format!(
                "positive enhance FAILED via {label} — kept verbatim (see the WARN above: bad key / \
                 blocked / truncated; try a different --compile-provider or --no-enhance)"
            ));
        } else {
            trace.push(format!("positive enhanced via {label}"));
        }
    } else {
        trace.push("positive kept verbatim (--no-enhance)".to_string());
    }
    trace.push(format!(
        "~{} tokens (budget ~{}, {})",
        assembler::estimate_tokens(&prompt),
        assembler::family_token_budget(scene.family),
        scene.family.label()
    ));
    if reinforced {
        trace.push("reinforced weighted concepts as prose (SD3/Flux honour prose > weights)".to_string());
    }
    let neg_count = negative.split(',').filter(|t| !t.trim().is_empty()).count();
    trace.push(if opts.no_negative {
        "negative: your seeds only (--no-negative)".to_string()
    } else if scene_negatives {
        format!("negative: {neg_count} terms (seeds + scene-specific defects + curated quality)")
    } else {
        format!("negative: {neg_count} terms (seeds + curated quality)")
    });
    if out_scene.name != scene.name {
        trace.push(format!("named from prompt → {}", out_scene.name));
    }

    // Attention-weight note (from step 2b): success = weights were re-applied deterministically; failure =
    // re-translation failed and the (corrected) advice stands. Either way it's surfaced, not silent.
    if let Some(note) = weight_note {
        warnings.push(note);
    }
    // Fit-to-budget note: a SUCCESSFUL condense (the prompt now fits) is INFO, not a ⚠ warning — only a
    // fit that couldn't reach budget is surfaced as a warning (by the diligence check below).
    if let Some(note) = fit_note {
        trace.push(note);
    }

    if structure_prompt.is_some() {
        trace.push("structure prompt (composition-focused) built for control-generate".to_string());
    }
    // A `foreground:` list sets the deliberate-figure count for the control-generate cap.
    let control_generate_max_figures = (!scene.foreground.is_empty()).then_some(scene.foreground.len());
    emitter::CompiledScene { scene: out_scene, prompt, negative, structure_prompt, control_generate_max_figures, warnings, trace }
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
    // Tokens RESERVED for text prepended/appended outside the fit (e.g. a `lora-trigger:`), so the total
    // still fits the model's budget once that text is added back.
    reserve: usize,
) -> (String, Option<String>) {
    let budget = assembler::family_token_budget(family).saturating_sub(reserve).max(16);
    let before = assembler::estimate_tokens(prompt);
    if before <= budget {
        return (prompt.to_string(), None);
    }
    let spans = dedup_spans(assembler::extract_weight_spans(prompt));
    // Two attempts: if the first condense still overshoots (LLMs are imprecise about token counts), retry
    // with a firmer, lower target so the prompt actually fits rather than overflowing with only a warning.
    let mut fitted = String::new();
    let mut current = prompt.to_string();
    for attempt in 0..2 {
        let target = if attempt == 0 { budget } else { budget.saturating_sub(budget / 8).max(32) };
        let sys = format!(
            "You compress text-to-image prompts to a token budget for the {label} model. Rewrite the prompt to \
             fit within AT MOST {target} CLIP tokens — but keep as MUCH of the detail as fits; do NOT \
             over-shorten (aim close to the budget, not far under it). PRESERVE every attention-weight span \
             `(phrase:number)` EXACTLY — keep the parentheses and the number unchanged. Keep every distinct \
             visual subject and the overall style; cut only filler, repetition and redundant adjectives. \
             Output ONLY the rewritten prompt.",
            label = family.label()
        );
        fitted = match cached_call(provider, &sys, &current, cache::POSITIVE, cache_on, eargs).await {
            Some(f) => assembler::clean(&f),
            None if attempt == 0 => return (prompt.to_string(), None), // fit call failed — keep original
            None => break,                                             // retry failed — keep the first fit
        };
        if assembler::estimate_tokens(&fitted) <= budget {
            break;
        }
        current = fitted.clone(); // still over — condense the condensed once more
    }
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

/// 6.28: build the English grounding clause for a scene's `relate:` relationships. Each object is
/// introduced once by its (translated) description — `the <english desc>` — and referred to by its short
/// name on repeat (`the tram`), so a chain of relations reads naturally. Verb → phrase via
/// [`assembler::relation_phrase`]. Empty when the scene declares no relations. Translation runs on the
/// enhance path only (mirrors the weighted-phrase loop); under `--no-enhance` the source description is
/// used verbatim.
/// 6.28: the 3-tier figure clause — foreground heroes (attach-objects already folded in), specific
/// background people, and the non-deterministic crowd — assembled ahead of the prose so the scene names its
/// subjects explicitly. Empty when the scene uses none of the tiers (a plain prose/composition scene).
fn build_figures_clause(scene: &resolver::ResolvedScene) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !scene.foreground.is_empty() {
        let fg = scene.foreground.iter().map(|(_, d)| d.trim()).collect::<Vec<_>>().join("; ");
        parts.push(format!("In the foreground: {fg}"));
    }
    if !scene.background.is_empty() {
        let bg = scene.background.iter().map(|(_, d)| d.trim()).collect::<Vec<_>>().join("; ");
        parts.push(format!("In the background: {bg}"));
    }
    if !scene.crowd.trim().is_empty() {
        let density = scene.crowd_density.as_deref().map(|d| format!("{} ", d.trim())).unwrap_or_default();
        parts.push(format!("further back, {density}{}", scene.crowd.trim()));
    }
    parts.join(". ")
}

async fn build_grounding(
    scene: &resolver::ResolvedScene,
    opts: &CompileOpts,
    eargs: &crate::prompt::EnhanceArgs,
) -> String {
    if scene.relations.is_empty() {
        return String::new();
    }
    let mut introduced: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut clauses: Vec<String> = Vec::new();
    // PLACE relations only — ATTACH relations (holding / leaning-on) are folded into the figure's own
    // description at resolve time, so re-stating them here would double the held object.
    for rel in scene.relations.iter().filter(|rel| !resolver::is_attach_verb(&rel.verb)) {
        let a = grounding_ref(&rel.a, &rel.a_desc, &mut introduced, scene, opts, eargs).await;
        let b = grounding_ref(&rel.b, &rel.b_desc, &mut introduced, scene, opts, eargs).await;
        clauses.push(format!("{a} {} {b}", assembler::relation_phrase(&rel.verb)));
    }
    clauses.join("; ")
}

/// A reference to one related object: full (translated) description on first mention, short name on repeat.
async fn grounding_ref(
    name: &str,
    desc: &str,
    introduced: &mut std::collections::HashSet<String>,
    scene: &resolver::ResolvedScene,
    opts: &CompileOpts,
    eargs: &crate::prompt::EnhanceArgs,
) -> String {
    if !introduced.insert(name.to_string()) {
        return format!("the {name}");
    }
    let en = match &scene.translate {
        Some(lang) if !opts.no_enhance && !lang.trim().is_empty() => {
            translate_phrase(desc, lang, &opts.provider, opts.cache, eargs)
                .await
                .unwrap_or_else(|| desc.to_string())
        }
        _ => desc.to_string(),
    };
    // Language-agnostic: use the description as-is. Prepending an English "the" both double-articles
    // English ("the a merchant") AND mixes languages on non-English prose ("the бородатый торговец").
    en.trim().to_string()
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
        Ok(_) => {
            tracing::warn!(target: "plakat", "compile: {provider} returned empty for {namespace} — falling back");
            return None;
        }
        Err(e) => {
            // Surface the real reason (HTTP status, SAFETY/MAX_TOKENS finishReason, bad key) instead of a
            // silent verbatim fallback — this is what made a failed enhance look like a success.
            tracing::warn!(target: "plakat", "compile: {provider} call failed for {namespace}: {e}");
            return None;
        }
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

/// `--make-composition` STRATEGIST system prompt: read a scene (ANY language) and pick the generation
/// pipeline best suited to it, emitting plakat directive lines only. Encodes the tool-selection lesson —
/// pure diffusion for common compositions, skeleton-only for posed few-figure scenes, skeleton+regional for
/// many distinct figures. No scene-specific hardcoding: the LLM maps scene features onto the categories.
const STRATEGIST_SYSTEM: &str = "You are a GENERATION STRATEGIST for the plakat text-to-image compiler. You \
    are given a scene prose file (possibly NOT in English). Decide the best generation PIPELINE for it and \
    output ONLY plakat directive lines and `#` comment lines. NEVER translate or restate the scene.\n\
    Judge from the scene: how many DISTINCT people are individually described; their poses (standing / sitting \
    / lying / kneeling / leaning / mid-action); whether distinct figures' garments and colours could be \
    painted onto the WRONG body; whether it is a common everyday composition or a hard multi-figure layout.\n\
    Pick EXACTLY ONE strategy and emit its directives verbatim:\n\
    # PURE — a common/intimate composition the diffusion model renders well ALONE: a single figure, a \
    portrait, a couple, a landscape; standing OR sitting naturally; no attribute-crossing risk. Control tools \
    would FIGHT the model here, so emit NO control-generate directives — instead:\n\
    count: 6\n\
    ranking: on by=vision threshold=5 coach=on coach-stuck=on min=2 max-tries=10\n\
    # SKELETON — 1 to 3 figures in an UNUSUAL / non-standing pose (seated, lying, kneeling, leaning, action) \
    where correct ANATOMY/POSE is the priority and attribute-crossing is NOT a real risk. Pose control is \
    strongest WITHOUT the regional layer:\n\
    control-generate: sd15\n\
    control-generate-mode: wireframe\n\
    control-generate-regional: false\n\
    control-generate-size: 512x512\n\
    control-generate-min-score: 4\n\
    # REGIONAL — MANY (3 or more) distinct figures, usually standing, whose garments MUST stay on the right \
    body. Regional binds attributes per box:\n\
    control-generate: sdxl\n\
    control-generate-mode: wireframe\n\
    control-generate-regional: true\n\
    control-generate-size: 1024x1024\n\
    control-generate-max-figures: 2\n\
    control-generate-inpaint-figures: true\n\
    control-generate-min-score: 5\n\
    Additionally, ALWAYS emit ONE `naturalize:` line tuned to THIS scene: infer the art MEDIUM from the prose \
    (watercolor / oil / ink-wash / gouache / pencil — or omit it for a photoreal scene) and set `medium=<that>`; \
    a gentler `repaint` (~0.25) for detailed realistic scenes, a stronger one (~0.4) for loose impressionist \
    ones; keep `brush`, `scale`, `paper`, `repair` modest. Example: `naturalize: repaint=0.35 medium=watercolor \
    brush=0.3 scale=0.55 paper=0.3 repair=0.15`.\n\
    Rules: the plakat prose format uses UNQUOTED values — write `control-generate: sd15`, and NEVER wrap any \
    value in quotes. START with one line `# strategy: PURE|SKELETON|REGIONAL — <one-line reason>`. For \
    REGIONAL set control-generate-max-figures to the count of DELIBERATE foreground HERO figures (usually 2; \
    the rest go to the background). Output ONLY `#` comment lines and directive lines. No prose paragraphs, \
    no code fences.";

/// `--make-composition` OPTIMIZER system prompt: rewrite the prose CLEANER + less hallucination-prone while
/// KEEPING the original language and every structural directive (relationships / figures), plus a tailored
/// negative. The companion of the strategist — together they turn a rough prose into a ready-to-run pair.
const OPTIMIZER_SYSTEM: &str = "You are a PROMPT EDITOR for the plakat text-to-image compiler. You are given a \
    plakat prose file that MAY be non-English. Produce a CLEANER, less hallucination-prone version.\n\
    HARD RULES:\n\
    - KEEP the ORIGINAL LANGUAGE of every human-readable description. Do NOT translate the scene text.\n\
    - PRESERVE every structural directive with its key and references intact — `component.<name>:`, \
    `composition:`, `relate:`, `foreground:`, `background:`, `crowd:`, `crowd-density:`, `name:`, `persona:`, \
    `style:`, `translate:`, `lora-trigger:`. NEVER drop a relationship or a figure. You may fix an obvious \
    mistake, but every relate/foreground/background line must survive.\n\
    - DO NOT emit any generation-STRATEGY directive — no `control-generate-*`, `naturalize:`, `ranking:`, \
    `count:`, `size:`, `steps:` — the included composition file owns all of those. KEEP infrastructure + \
    content: `model:`, `loras:`, `lora-trigger:`, `quantize-t5:`, `t5-quant-level:`, `style:`, `translate:`, \
    `persona:`, and every component / scene line.\n\
    IMPROVE:\n\
    - Rewrite the free-text scene sentences to be concrete and unambiguous, in the SAME language.\n\
    - Remove NEGATION from every positive description and component: never write 'without X' / 'no X' / \
    '(без X)' (the model paints X). Describe what the subject DOES have, and MOVE the absence into the \
    negative.\n\
    - Make figure COUNT explicit where it matters (e.g. 'a couple', 'two people').\n\
    - Drop duplicated or redundant phrases and stray attention weights.\n\
    NEGATIVE: output exactly ONE `negative:` line, in ENGLISH, tailored to THIS scene — fold in every \
    'without/no' you moved out of the positive, plus concise anatomy/quality guards (bad anatomy, deformed \
    hands, extra fingers, extra limbs, blurry, lowres, low quality). Keep it focused, not a wall of tags.\n\
    Output ONLY the rewritten prose file content (directives + optimized free text + the negative line), \
    same structure, in the same language, with NO code fences and NO commentary.";

/// `--analyze` FEASIBILITY CRITIC: judge how reliably a diffusion model can render the scene in ONE image and
/// flag the specific failure modes learned the hard way (over-stuffing, coupled-object fusion, person+cargo
/// fusion, rare object names, negation-in-positive, hard poses, count ambiguity, attribute-crossing). The
/// polish-loop companion to the strategist — it tells the author what to fix BEFORE any tokens are generated.
const CRITIC_SYSTEM: &str = "You are a FEASIBILITY CRITIC for the plakat text-to-image compiler. You are given \
    a plakat prose scene (which MAY be non-English). Judge how RELIABLY a diffusion model (an SDXL draft handed \
    to an SD3.5 finish) can render it in ONE image, and flag the SPECIFIC risks that make generation fail — so \
    the author fixes them BEFORE spending tokens generating.\n\
    Output a short report, nothing else:\n\
    - FIRST line exactly: `Feasibility: N/10 — LOW|MODERATE|HIGH RISK`, where N is 1–10 (10 = a common, easy \
    composition a model nails first try; 1 = many stacked hard problems). The RISK word is the INVERSE of the \
    score: N of 8–10 → `LOW RISK`, N of 4–7 → `MODERATE RISK`, N of 1–3 → `HIGH RISK`. (So a 2/10 is HIGH RISK, \
    an 8/10 is LOW RISK — do NOT invert this.)\n\
    - Then a list, WORST FIRST, of the concrete risks THIS scene has. Each risk is two lines: `⚠ <RISK> — \
    <the exact phrase / element in THIS scene that triggers it>` then `   → <specific, actionable fix>`.\n\
    - Then `✓ <thing>` lines for what is already fine.\n\
    Judge ONLY against these known failure modes, and flag ONLY the ones this scene actually has:\n\
    - OVER-STUFFED: more than ~2 distinct described PEOPLE plus a complex OBJECT plus a detailed BACKGROUND all \
    competing in one frame → the model drops or fuses elements. Fix: split into separate scenes, or demote \
    extras to background / a crowd.\n\
    - FUSION (coupled objects): a vehicle plus a SEPARATE towed trailer/cart, or any two hitched/coupled \
    objects → the model renders ONE fused body, not two. Fix: a `control-preimage:` from a reference image — \
    region placement will NOT reliably separate two coupled wheeled vehicles.\n\
    - FUSION (person + cargo): a person carrying/holding a LARGE flat or bulky object → the cargo fuses INTO \
    the body. Fix: make the carried item small and clearly in the hands ('a few books under one arm').\n\
    - RARE / INVENTED OBJECT NAME: an obscure or made-up term (e.g. 'steam locomobile') → the model \
    hallucinates. Fix: the closest WELL-KNOWN term (e.g. 'steam traction engine / road locomotive').\n\
    - NEGATION-IN-POSITIVE: 'without X' / 'no X' / '(без X)' in a POSITIVE description → the model PAINTS X. \
    Fix: describe what IS present; move the absence to the negative.\n\
    - HARD POSE ON SDXL: seated / lying / kneeling / crouching figures → SDXL's OpenPose control is weak on \
    non-standing poses. Fix: `--composition-model sd15` for the draft.\n\
    - COUNT AMBIGUITY: vague plurals ('several', 'some people') → unpredictable counts. Fix: state exact \
    numbers.\n\
    - ATTRIBUTE-CROSSING: multiple distinct figures whose garments/colours could swap onto the wrong body. \
    Fix: `control-generate-regional: true`.\n\
    - TOKEN BLOAT: an extremely long, detail-stuffed prompt → the finish model drops details. Fix: cut \
    secondary detail.\n\
    Be specific — QUOTE the offending phrases from the scene. Be honest: if the scene stacks several hard \
    problems, say so and grade it LOW. Output ONLY the report — no preamble, no code fences.";

/// `plakat compile --analyze`: run the feasibility critic over the prose (any language) and return its report
/// (a grade + specific risks + fixes). Pure analysis — no compile, no generation. The polish-loop companion to
/// [`make_composition`]: iterate the prose until the grade is acceptable before spending a generation run.
pub async fn analyze_prose(input: &str, opts: &CompileOpts) -> anyhow::Result<String> {
    use anyhow::Context;
    let eargs = crate::prompt::EnhanceArgs::default();
    let report = crate::prompt::complete(&opts.provider, CRITIC_SYSTEM, input, &eargs)
        .await
        .context("compile --analyze: critic LLM call failed")?;
    let report = strip_code_fences(&report);
    anyhow::ensure!(!report.trim().is_empty(), "compile --analyze: critic returned nothing");
    Ok(report)
}

/// `--analyze --fix` AUTO-FIXER: propose SAFE, high-confidence VERBATIM text edits that reduce the failure
/// risks the critic finds, plus notes for the structural ones a machine must not touch. Emits JSON so the
/// edits can be located in the exact source `@include` file and applied with a backup.
const FIXER_SYSTEM: &str = "You are an AUTO-FIXER for the plakat text-to-image compiler. You are given a plakat \
    prose scene (which MAY be non-English). Propose SAFE, high-confidence TEXT edits that reduce \
    generation-failure risk, plus a list of STRUCTURAL changes only the author can make.\n\
    Output ONLY a JSON object: {\"edits\":[{\"old\":\"…\",\"new\":\"…\",\"why\":\"…\"}],\"manual\":[\"…\"]}.\n\
    RULES for each edit:\n\
    - \"old\" MUST be an EXACT, VERBATIM substring copied from the scene (so it can be found and replaced). \
    Keep it SHORT and UNIQUE — just the offending phrase, not a whole sentence.\n\
    - \"new\" is the improved replacement in the EXACT SAME LANGUAGE as \"old\". The scene commonly MIXES \
    languages (e.g. Russian component descriptions + English negative terms) — NEVER translate: fix a Russian \
    phrase in Russian, an English phrase in English, matching the script/language of the text you replace.\n\
    - Propose ONLY edits you are CONFIDENT improve reliability, of these kinds:\n\
      * RARE/INVENTED object name → the closest well-known term.\n\
      * NEGATION-in-positive ('without X'/'no X'/'(без X)') → describe what IS present instead.\n\
      * person+cargo fusion → shrink a big carried item to small + hand-held.\n\
      * COUNT ambiguity → an exact number.\n\
      * TOKEN bloat → cut ONE secondary detail clause (set \"new\" to empty to delete it).\n\
    - Do NOT edit component NAMES, `relate:`/`foreground:`/`objects:` directives, or any structural syntax — \
    only human-readable descriptive text and negative terms.\n\
    \"manual\" = short notes for what you CANNOT safely auto-apply (splitting the over-stuffed scene into \
    separate images, using a control-preimage for two coupled wheeled vehicles, a conflicting control config). \
    Output ONLY the JSON — no prose, no code fences.";

#[derive(serde::Deserialize)]
struct FixEdit {
    old: String,
    new: String,
    #[serde(default)]
    why: String,
}
#[derive(serde::Deserialize, Default)]
struct FixPlan {
    #[serde(default)]
    edits: Vec<FixEdit>,
    #[serde(default)]
    manual: Vec<String>,
}

/// Next free versioned backup path `<file>.<N>` (1, 2, …).
fn next_backup_path(f: &std::path::Path) -> std::path::PathBuf {
    let mut n = 1u32;
    loop {
        let cand = std::path::PathBuf::from(format!("{}.{n}", f.display()));
        if !cand.exists() {
            return cand;
        }
        n += 1;
    }
}

fn trunc(s: &str) -> String {
    let s = s.trim().replace('\n', " ");
    if s.chars().count() > 60 {
        format!("{}…", s.chars().take(59).collect::<String>())
    } else {
        s
    }
}

/// `plakat compile --analyze --fix`: run the auto-fixer, apply its SAFE text edits to the exact source
/// `@include` file each phrase lives in (backing that file up to `<file>.<N>` FIRST), and return a report of
/// what changed where — plus the structural items that still need the author. Never touches a file it can't
/// locate the phrase in, and never edits the same phrase in two files (reports it instead).
pub async fn apply_fixes(
    input_path: &std::path::Path,
    opts: &CompileOpts,
    risks: &str,
) -> anyhow::Result<String> {
    use anyhow::Context;
    let eargs = crate::prompt::EnhanceArgs::default();
    // 1. Gather the source files + the expanded prose the fixer reasons over.
    let files = parser::collect_source_files(input_path, 0)?;
    let raw = std::fs::read_to_string(input_path)
        .with_context(|| format!("--fix: reading {}", input_path.display()))?;
    let base = input_path.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| std::path::PathBuf::from("."));
    let expanded = if raw.contains("@include") { parser::expand_includes(&raw, &base, 0)? } else { raw };

    // 2. Fixer LLM → plan. Feed it the critic's findings so it fixes exactly what `--analyze` flagged (not a
    // second, possibly-disagreeing judgement). Extract the JSON object, tolerating any stray wrapping.
    let user = format!(
        "SCENE PROSE:\n{expanded}\n\nCRITIC FINDINGS (produce verbatim edits that fix the auto-fixable ones):\n{}",
        risks.trim()
    );
    let out = crate::prompt::complete(&opts.provider, FIXER_SYSTEM, &user, &eargs)
        .await
        .context("--fix: fixer LLM call failed")?;
    let out = strip_code_fences(&out);
    let json = match (out.find('{'), out.rfind('}')) {
        (Some(a), Some(b)) if b >= a => &out[a..=b],
        _ => out.as_str(),
    };
    let plan: FixPlan = serde_json::from_str(json)
        .with_context(|| format!("--fix: fixer returned invalid JSON:\n{out}"))?;

    // 3. Apply each text edit to the ONE source file that contains it verbatim, backing it up first.
    let mut backups: std::collections::HashMap<std::path::PathBuf, std::path::PathBuf> =
        std::collections::HashMap::new();
    let mut report = String::new();
    let mut applied = 0usize;
    for e in &plan.edits {
        if e.old.trim().is_empty() || e.old == e.new {
            continue;
        }
        let hits: Vec<&std::path::PathBuf> = files
            .iter()
            .filter(|f| std::fs::read_to_string(f).map(|c| c.contains(&e.old)).unwrap_or(false))
            .collect();
        match hits.as_slice() {
            [f] => {
                let f = (*f).to_path_buf();
                let bak = backups
                    .entry(f.clone())
                    .or_insert_with(|| {
                        let b = next_backup_path(&f);
                        let _ = std::fs::copy(&f, &b);
                        b
                    })
                    .clone();
                let text = std::fs::read_to_string(&f)?;
                let new_text = text.replacen(&e.old, &e.new, 1);
                std::fs::write(&f, &new_text)
                    .with_context(|| format!("--fix: writing {}", f.display()))?;
                report.push_str(&format!(
                    "✓ {}  (backup → {})\n    {}: “{}” → “{}”\n",
                    f.display(),
                    bak.display(),
                    if e.why.trim().is_empty() { "fix" } else { e.why.trim() },
                    trunc(&e.old),
                    if e.new.trim().is_empty() { "(removed)".into() } else { trunc(&e.new) },
                ));
                applied += 1;
            }
            [] => report.push_str(&format!("• skipped — phrase not found verbatim: “{}”\n", trunc(&e.old))),
            _ => report.push_str(&format!(
                "• skipped — “{}” appears in >1 file (edit by hand)\n",
                trunc(&e.old)
            )),
        }
    }
    if applied == 0 {
        report.push_str("(no auto-applicable text fixes)\n");
    }
    if !plan.manual.is_empty() {
        report.push_str("\nNeeds your hand (not auto-fixable):\n");
        for m in &plan.manual {
            report.push_str(&format!("  ⚠ {}\n", m.trim()));
        }
    }
    Ok(report)
}

/// Strip a leading/trailing Markdown code fence (```…```), which chat LLMs add despite instructions.
fn strip_code_fences(s: &str) -> String {
    let t = s.trim();
    let t = t.strip_prefix("```").map(|r| r.splitn(2, '\n').nth(1).unwrap_or("")).unwrap_or(t);
    let t = t.trim_end().strip_suffix("```").unwrap_or(t);
    t.trim().to_string()
}

/// Strip surrounding quotes from directive VALUES (`key: "v"` → `key: v`) — the plakat prose format is
/// UNQUOTED, so a quoted value would parse WITH the quotes. Leaves `#` comments and already-unquoted lines
/// (including free-text prose) untouched; only removes a matched pair wrapping the whole value.
fn unquote_directive_values(s: &str) -> String {
    s.lines()
        .map(|line| {
            if line.trim_start().starts_with('#') {
                return line.to_string();
            }
            if let Some(colon) = line.find(':') {
                let (key, rest) = line.split_at(colon + 1);
                let val = rest.trim();
                let inner = val
                    .strip_prefix('"')
                    .and_then(|v| v.strip_suffix('"'))
                    .or_else(|| val.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')));
                if let Some(inner) = inner {
                    return format!("{key} {inner}");
                }
            }
            line.to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Next free `composition_NNN.txt` index in `dir` (scans existing ones; starts at 1).
fn next_composition_index(dir: &std::path::Path) -> u32 {
    let mut max = 0u32;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if let Some(rest) = name.strip_prefix("composition_").and_then(|r| r.strip_suffix(".txt")) {
                if let Ok(n) = rest.parse::<u32>() {
                    max = max.max(n);
                }
            }
        }
    }
    max + 1
}

/// `plakat compile --make-composition`: analyse a prose scene (any language) with two LLM passes and write
/// (a) `composition_<NNN>.txt` — generation-strategy directives to `@include`, and (b) `<stem>_optimized.txt`
/// — a cleaner, less hallucination-prone rewrite of the prose (SAME language, all relationships/figures kept,
/// a tailored negative), whose FIRST line `@include`s the composition file. Advisory: the author reviews both.
pub async fn make_composition(
    input: &str,
    input_path: &std::path::Path,
    opts: &CompileOpts,
    control_model: &str,
) -> anyhow::Result<(std::path::PathBuf, std::path::PathBuf)> {
    use anyhow::Context;
    let eargs = crate::prompt::EnhanceArgs::default();
    let dir = input_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    // 1) STRATEGIST → the composition-strategy directives. Pin the DRAFT MODEL (+ its native size) the
    // strategist must use for any control-generate strategy, so it can't emit an SDXL/512 (or SD1.5/1024)
    // mismatch. Default is sdxl (1024²); `--composition-model sd15` swaps to the strong-pose 512² draft.
    let cg_size = if control_model.to_lowercase().contains("xl") { "1024x1024" } else { "512x512" };
    let strategist_sys = format!(
        "{STRATEGIST_SYSTEM}\n\
         DRAFT MODEL (override): whenever you emit control-generate directives, use EXACTLY \
         `control-generate: {control_model}` and `control-generate-size: {cg_size}` (that model's native \
         resolution) — never a different draft model or a mismatched size."
    );
    let strategy = crate::prompt::complete(&opts.provider, &strategist_sys, input, &eargs)
        .await
        .context("make-composition: strategist LLM call failed")?;
    let strategy = unquote_directive_values(&strip_code_fences(&strategy));
    anyhow::ensure!(!strategy.trim().is_empty(), "make-composition: strategist returned nothing");
    let n = next_composition_index(&dir);
    let comp_name = format!("composition_{n:03}.txt");
    let comp_path = dir.join(&comp_name);
    let comp_body = format!(
        "# plakat auto-strategy — generated by `compile --make-composition`\n# source: {}\n# review + tweak, then it is @included by the optimized prose.\n{}\n",
        opts.input_name,
        strategy.trim()
    );
    std::fs::write(&comp_path, &comp_body)
        .with_context(|| format!("writing {}", comp_path.display()))?;

    // 2) OPTIMIZER → cleaner prose in the original language; prepend the @include of the strategy.
    let optimized = crate::prompt::complete(&opts.provider, OPTIMIZER_SYSTEM, input, &eargs)
        .await
        .context("make-composition: optimizer LLM call failed")?;
    let optimized = unquote_directive_values(&strip_code_fences(&optimized));
    anyhow::ensure!(!optimized.trim().is_empty(), "make-composition: optimizer returned nothing");
    let stem = input_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "prompts".into());
    let opt_path = dir.join(format!("{stem}_optimized.txt"));
    let opt_body = format!("@include {comp_name}\n\n{}\n", optimized.trim());
    std::fs::write(&opt_path, &opt_body)
        .with_context(|| format!("writing {}", opt_path.display()))?;

    Ok((comp_path, opt_path))
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
    // Live progress: each scene fires several LLM calls (translate + positive + negative), which used to
    // leave the screen frozen until the final trace. Emit a header + a per-scene ✓ as each completes.
    let total = active.len();
    let label = crate::prompt::resolve_provider_label(&opts.provider);
    crate::ui::progress::println(&format!(
        "  compiling {total} scene(s) via {label}{}…",
        if n > 1 { format!(" ({n} in parallel)") } else { String::new() }
    ));
    // Does this scenario use `control-generate`? If so, each scene also gets a style-stripped STRUCTURE
    // prompt (one extra LLM call). Checked once across globals + scenes so we don't pay it otherwise.
    let is_on = |v: &str| {
        let v = v.trim();
        !v.is_empty()
            && !v.eq_ignore_ascii_case("off")
            && !v.eq_ignore_ascii_case("false")
            && !v.eq_ignore_ascii_case("none")
    };
    let wants_structure = resolved.globals.passthrough.iter().any(|(k, v)| k == "control-generate" && is_on(v))
        || active.iter().any(|s| s.passthrough.iter().any(|(k, v)| k == "control-generate" && is_on(v)));
    let done = std::sync::atomic::AtomicUsize::new(0);
    let tick = |name: &str| {
        let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        crate::ui::progress::println(&format!("  {} {k}/{total} · {name}", console::style("✓").green()));
    };
    let mut compiled: Vec<emitter::CompiledScene> = if n <= 1 {
        let mut v = Vec::with_capacity(active.len());
        for s in active.iter().copied() {
            let r = compile_one_scene(s, opts, &eargs, wants_structure).await;
            tick(&s.name);
            v.push(r);
        }
        v
    } else {
        use futures_util::stream::{self, StreamExt};
        let eargs_ref = &eargs;
        let tick_ref = &tick;
        stream::iter(active.iter().copied().map(|s| async move {
            let r = compile_one_scene(s, opts, eargs_ref, wants_structure).await;
            tick_ref(&s.name);
            r
        }))
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
            // A command may repeat when its merge kind accumulates/concatenates (lora, loras, region, redux,
            // control, style, persona, header, footer, composition, negative) — deriving it from the spec
            // keeps this correct as commands are added. Passthrough (`-`) keys and `component.`/axis keys
            // also repeat. Only LastWins scalars (seed, count, …) are flagged when duplicated.
            let repeatable = command_spec(k).is_some_and(|s| matches!(s.merge, Merge::AccumulateList | Merge::Concatenate))
                || k.contains('-')
                || k.starts_with("component.")
                || k.starts_with("scene.")
                || k.starts_with("weather.");
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
    fn unquote_directive_values_strips_only_wrapping_quotes() {
        let src = "# strategy: SKELETON — reason\ncontrol-generate: \"sd15\"\ncontrol-generate-size: \"512x512\"\n\
                   control-generate-regional: false\nnaturalize: \"repaint=0.3 medium=oil\"\n\
                   a free-text line: with a colon but no quotes\ncomponent.x: 'single quoted'";
        let out = unquote_directive_values(src);
        assert!(out.contains("control-generate: sd15"), "double quotes stripped");
        assert!(out.contains("control-generate-size: 512x512"));
        assert!(out.contains("naturalize: repaint=0.3 medium=oil"), "value with spaces unwrapped");
        assert!(out.contains("component.x: single quoted"), "single quotes stripped too");
        assert!(out.contains("# strategy: SKELETON — reason"), "comment untouched");
        assert!(out.contains("control-generate-regional: false"), "bare value untouched");
        assert!(out.contains("a free-text line: with a colon but no quotes"), "unquoted free text untouched");
        assert!(!out.contains('"'), "no double quotes remain");
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
