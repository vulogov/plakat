//! Resolve a parsed [`Document`] into globals + scenes, applying each command's
//! merge strategy across the global→scene inheritance and classifying the model
//! family per scene.

use anyhow::Result;

use super::parser::{Block, Document};
use super::{ModelFamily, classify_model};

/// Resolved scenario-global defaults (→ HJSON top level). `model` and `loras`
/// can only live here — scenarios share one pre-loaded pipeline.
#[derive(Debug, Clone, Default)]
pub struct ResolvedGlobals {
    pub model: Option<String>,
    pub family: ModelFamily,
    pub loras: Vec<String>,
    pub seed: Option<u64>,
    pub count: Option<u32>,
    pub size: Option<String>,
    pub steps: Option<usize>,
    pub guidance: Option<f64>,
    pub scheduler: Option<String>,
    pub refine: Option<usize>,
    pub negative_seeds: String,
}

/// One resolved scene → one scenario task (+ the prompt-shaping inputs for the
/// LLM stage).
#[derive(Debug, Clone, Default)]
pub struct ResolvedScene {
    pub name: String,
    /// The model this scene's prompt is *written for* (family classification).
    /// The scenario still runs on the global model (shared pipeline).
    pub family: ModelFamily,
    pub model_for_family: Option<String>,
    // prompt shaping (LLM inputs)
    pub header: String,
    pub footer: String,
    pub free_text: String,
    pub styles: Vec<String>,
    pub personas: Vec<String>,
    pub translate: Option<String>,
    pub negative_seeds: String,
    // scenario per-task fields
    pub seed: Option<u64>,
    pub count: Option<u32>,
    pub size: Option<String>,
    pub steps: Option<usize>,
    pub guidance: Option<f64>,
    pub scheduler: Option<String>,
    pub refine: Option<usize>,
    pub tags: Vec<String>,
    pub skip: bool,
    // MAP-4: a `type: map` block compiles to a scenario `map` task. These mirror
    // the scenario `map-*` fields; all on the deterministic path (no LLM).
    pub task_type: Option<String>,
    pub map_spec: Option<String>,
    pub map_style: Option<String>,
    pub map_paint: Option<String>,
    pub map_scale: Option<String>,
    pub map_tiles: Option<String>,
    pub map_sd_model: Option<String>,
    pub map_sd_lora: Vec<String>,
    pub map_provider: Option<String>,
    // 6.1.0 A3: a `type: bookart` block compiles to a scenario `bookart` task. The prose (if any) is the
    // ornament prompt; `bookart-*` directives fill the spec.
    pub bookart_origin: Option<String>,
    pub bookart_technique: Option<String>,
    pub bookart_type: Option<String>,
    pub bookart_page: Option<String>,
    pub bookart_svg: Option<String>,
    // 6.3.0 B7: a `type: texture` block compiles to a scenario `texture` task. The prose (if any) is the
    // material prompt; `texture-*` directives fill the spec.
    pub texture_from: Option<String>,
    pub texture_size: Option<String>,
    pub texture_upscale: Option<String>,
    pub texture_seamless: Option<String>,
    pub texture_height: Option<String>,
    // 6.8.0 P4: a `type: comic` block compiles to a scenario `comic` task. `comic-spec-file` points at a
    // ComicSpec; otherwise the prose (if any) becomes a single-panel page.
    pub comic_spec_file: Option<String>,
    // 6.9.0 P4: a `type: product` block compiles to a scenario `product` task. `product-spec-file` points
    // at a ProductSpec; otherwise the prose (if any) becomes the subject prompt.
    pub product_spec_file: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Resolved {
    pub globals: ResolvedGlobals,
    pub scenes: Vec<ResolvedScene>,
}

/// Concatenate-merge: `global + scene` joined with `, `. An **empty** scene
/// occurrence resets (drops) inherited global values for this block.
fn concat(global: &[&str], scene: &[&str]) -> String {
    let reset = scene.iter().any(|v| v.is_empty());
    let mut parts: Vec<&str> = Vec::new();
    if !reset {
        parts.extend(global.iter().copied().filter(|v| !v.is_empty()));
    }
    parts.extend(scene.iter().copied().filter(|v| !v.is_empty()));
    parts.join(", ")
}

/// Last-wins merge: scene's last value, else global's last value.
fn last_wins<'a>(global: &[&'a str], scene: &[&'a str]) -> Option<&'a str> {
    scene.last().or_else(|| global.last()).copied()
}

/// List-accumulate merge: all global entries then all scene entries.
fn list(global: &[&str], scene: &[&str]) -> Vec<String> {
    global.iter().chain(scene.iter()).map(|s| s.to_string()).collect()
}

fn vals<'a>(b: Option<&'a Block>, key: &str) -> Vec<&'a str> {
    b.map(|b| b.values(key).collect()).unwrap_or_default()
}

fn parse_opt<T: std::str::FromStr>(v: Option<&str>, what: &str) -> Result<Option<T>>
where
    T::Err: std::fmt::Display,
{
    match v {
        None => Ok(None),
        Some(s) => s
            .parse::<T>()
            .map(Some)
            .map_err(|e| anyhow::anyhow!("compile: bad `{what}: {s}` — {e}")),
    }
}

/// Parse an optional finite f64. `"inf"`/`"NaN"` parse fine as f64 but emit an invalid
/// HJSON/JSON number (no inf/NaN literal) that makes the compiled scenario fail to load,
/// so reject them here.
fn parse_finite_f64(v: Option<&str>, what: &str) -> Result<Option<f64>> {
    let parsed: Option<f64> = parse_opt(v, what)?;
    if let Some(f) = parsed {
        if !f.is_finite() {
            anyhow::bail!("compile: `{what}` must be a finite number (got {f})");
        }
    }
    Ok(parsed)
}

/// Auto scene name from the first 6 words of the description, slugified.
fn auto_name(free_text: &str, idx: usize) -> String {
    let slug: String = free_text
        .split_whitespace()
        .take(6)
        .map(|w| {
            w.chars()
                .filter(|c| c.is_ascii_alphanumeric())
                .collect::<String>()
                .to_ascii_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join("_");
    if slug.is_empty() {
        format!("scene_{}", idx + 1)
    } else {
        slug
    }
}

/// Resolve the document. `default_model` is the `--model` CLI fallback (used for
/// family classification when neither scene nor global names a model).
pub fn resolve(doc: &Document, default_model: &str) -> Result<Resolved> {
    let g = doc.global.as_ref();

    let g_model = last_wins(&[], &vals(g, "model")).map(str::to_string);
    let globals = ResolvedGlobals {
        family: classify_model(g_model.as_deref().unwrap_or(default_model)),
        model: g_model.clone(),
        loras: list(&[], &vals(g, "lora")),
        seed: parse_opt(last_wins(&[], &vals(g, "seed")), "seed")?,
        count: parse_opt(last_wins(&[], &vals(g, "count")), "count")?,
        size: last_wins(&[], &vals(g, "size")).map(str::to_string),
        steps: parse_opt(last_wins(&[], &vals(g, "steps")), "steps")?,
        guidance: parse_finite_f64(last_wins(&[], &vals(g, "guidance")), "guidance")?,
        scheduler: last_wins(&[], &vals(g, "scheduler")).map(str::to_string),
        refine: parse_opt(last_wins(&[], &vals(g, "refine")), "refine")?,
        negative_seeds: concat(&[], &vals(g, "negative")),
    };

    let mut scenes = Vec::with_capacity(doc.scenes.len());
    for (i, s) in doc.scenes.iter().enumerate() {
        let free_text = s.free_text.join(" ");
        let model_for_family = last_wins(&vals(g, "model"), &vals(Some(s), "model")).map(str::to_string);
        let family = classify_model(model_for_family.as_deref().unwrap_or(default_model));
        let name = last_wins(&[], &vals(Some(s), "name"))
            .map(str::to_string)
            .unwrap_or_else(|| auto_name(&free_text, i));
        let skip = matches!(last_wins(&[], &vals(Some(s), "skip")), Some("true" | "yes" | "1"));

        scenes.push(ResolvedScene {
            name,
            family,
            model_for_family,
            header: concat(&vals(g, "header"), &vals(Some(s), "header")),
            footer: concat(&vals(g, "footer"), &vals(Some(s), "footer")),
            free_text,
            styles: list(&vals(g, "style"), &vals(Some(s), "style")),
            personas: list(&vals(g, "persona"), &vals(Some(s), "persona")),
            translate: last_wins(&[], &vals(Some(s), "translate")).map(str::to_string),
            negative_seeds: concat(&vals(g, "negative"), &vals(Some(s), "negative")),
            seed: parse_opt(last_wins(&[], &vals(Some(s), "seed")), "seed")?,
            count: parse_opt(last_wins(&[], &vals(Some(s), "count")), "count")?,
            size: last_wins(&[], &vals(Some(s), "size")).map(str::to_string),
            steps: parse_opt(last_wins(&[], &vals(Some(s), "steps")), "steps")?,
            guidance: parse_finite_f64(last_wins(&[], &vals(Some(s), "guidance")), "guidance")?,
            scheduler: last_wins(&[], &vals(Some(s), "scheduler")).map(str::to_string),
            refine: parse_opt(last_wins(&[], &vals(Some(s), "refine")), "refine")?,
            tags: list(&[], &vals(Some(s), "tag")),
            skip,
            // MAP-4: global→scene inheritance for the map directives.
            task_type: last_wins(&vals(g, "type"), &vals(Some(s), "type")).map(str::to_string),
            map_spec: last_wins(&vals(g, "map-spec"), &vals(Some(s), "map-spec")).map(str::to_string),
            map_style: last_wins(&vals(g, "map-style"), &vals(Some(s), "map-style")).map(str::to_string),
            map_paint: last_wins(&vals(g, "map-paint"), &vals(Some(s), "map-paint")).map(str::to_string),
            map_scale: last_wins(&vals(g, "map-scale"), &vals(Some(s), "map-scale")).map(str::to_string),
            map_tiles: last_wins(&vals(g, "map-tiles"), &vals(Some(s), "map-tiles")).map(str::to_string),
            map_sd_model: last_wins(&vals(g, "map-sd-model"), &vals(Some(s), "map-sd-model")).map(str::to_string),
            map_sd_lora: list(&vals(g, "map-sd-lora"), &vals(Some(s), "map-sd-lora")),
            map_provider: last_wins(&vals(g, "map-provider"), &vals(Some(s), "map-provider")).map(str::to_string),
            bookart_origin: last_wins(&vals(g, "bookart-origin"), &vals(Some(s), "bookart-origin")).map(str::to_string),
            bookart_technique: last_wins(&vals(g, "bookart-technique"), &vals(Some(s), "bookart-technique")).map(str::to_string),
            bookart_type: last_wins(&vals(g, "bookart-type"), &vals(Some(s), "bookart-type")).map(str::to_string),
            bookart_page: last_wins(&vals(g, "bookart-page"), &vals(Some(s), "bookart-page")).map(str::to_string),
            bookart_svg: last_wins(&vals(g, "bookart-svg"), &vals(Some(s), "bookart-svg")).map(str::to_string),
            texture_from: last_wins(&vals(g, "texture-from"), &vals(Some(s), "texture-from")).map(str::to_string),
            texture_size: last_wins(&vals(g, "texture-size"), &vals(Some(s), "texture-size")).map(str::to_string),
            texture_upscale: last_wins(&vals(g, "texture-upscale"), &vals(Some(s), "texture-upscale")).map(str::to_string),
            texture_seamless: last_wins(&vals(g, "texture-seamless"), &vals(Some(s), "texture-seamless")).map(str::to_string),
            texture_height: last_wins(&vals(g, "texture-height"), &vals(Some(s), "texture-height")).map(str::to_string),
            comic_spec_file: last_wins(&vals(g, "comic-spec-file"), &vals(Some(s), "comic-spec-file")).map(str::to_string),
            product_spec_file: last_wins(&vals(g, "product-spec-file"), &vals(Some(s), "product-spec-file")).map(str::to_string),
        });
    }
    Ok(Resolved { globals, scenes })
}

#[cfg(test)]
mod tests {
    use super::super::parser::parse;
    use super::*;

    fn resolve_str(s: &str) -> Resolved {
        resolve(&parse(s).unwrap(), "sdxl").unwrap()
    }

    #[test]
    fn concatenate_merges_global_and_scene() {
        let r = resolve_str("footer: global-a\nfooter: global-b\n\nfooter: scene-x\nA scene.\n");
        assert_eq!(r.scenes[0].footer, "global-a, global-b, scene-x");
    }

    #[test]
    fn empty_header_resets_inherited_global() {
        let r = resolve_str("header: global,\n\nheader:\nheader: local,\nA scene.\n");
        assert_eq!(r.scenes[0].header, "local,", "empty header drops the global");
    }

    #[test]
    fn loras_accumulate_global_plus_scene() {
        let r = resolve_str("lora: a:0.5\n\nlora: b:0.8\nlora: c:0.3\nA scene.\n");
        assert_eq!(r.globals.loras, vec!["a:0.5"]);
        // scene loras live on the scene (the emitter folds them per the format).
        let r2 = resolve_str("A scene.\nlora: b:0.8\nlora: c:0.3\n");
        assert_eq!(r2.scenes[0].tags, Vec::<String>::new());
        assert_eq!(r2.globals.loras, Vec::<String>::new());
    }

    #[test]
    fn last_wins_for_scalars_and_family() {
        let r = resolve_str("model: sd15\nseed: 1\n\nmodel: flux-dev\nseed: 2\nA scene.\n");
        assert_eq!(r.globals.model.as_deref(), Some("sd15"));
        assert_eq!(r.globals.family, ModelFamily::Sd15);
        assert_eq!(r.scenes[0].family, ModelFamily::Flux, "scene model wins for family");
        assert_eq!(r.scenes[0].seed, Some(2));
    }

    #[test]
    fn auto_names_from_first_words() {
        let r = resolve_str("A vast frozen tundra stretching to the far horizon.\n");
        assert_eq!(r.scenes[0].name, "a_vast_frozen_tundra_stretching_to");
    }

    #[test]
    fn bad_number_errors() {
        let err = resolve(&parse("seed: not-a-number\n\nA scene.\n").unwrap(), "sdxl").unwrap_err();
        assert!(err.to_string().contains("bad `seed"), "got: {err}");
    }
}
