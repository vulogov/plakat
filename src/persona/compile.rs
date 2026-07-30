//! The persona compiler (RFC §9) — a **pure** function from `(PersonaSpec, Lexicon)` to per-encoder
//! prompts. P0 slice: resolve known+prompt-routed attributes → salience-rank → emit per encoder class,
//! with a token budget on CLIP. Deterministic and byte-stable; no weights, no I/O.
//!
//! Not yet modelled (later P0 / phases): the geometry/detail conditioning (composited details leave the
//! prompt entirely, §8.9), the manifestation gate for dentition (§8.6 — no teeth attrs in the skeleton
//! lexicon yet), region-headed grouping (an experiment, §27.1), and per-encoder phrasing variants
//! (the skeleton uses one phrasing + an encoder-specific connective).

use crate::persona::lexicon::{LexEntry, Lexicon};
use crate::persona::spec::{Color, PersonaSpec};

/// Encoder class of a model family (§9.3). Determines prompt shape + token budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderClass {
    Clip,       // SD 1.5 / 2.1
    ClipDual,   // SDXL
    ClipTriple, // SD 3.5
    T5,         // PixArt-Σ / Flux
    Gemma,      // Sana
}

impl EncoderClass {
    /// Best-effort mapping from a model alias to its encoder class.
    pub fn from_model(alias: &str) -> Self {
        let a = alias.to_lowercase();
        if a.contains("sana") {
            EncoderClass::Gemma
        } else if a.contains("pixart") || a.contains("flux") {
            EncoderClass::T5
        } else if a.contains("sd3") || a.contains("sd35") || a.contains("stable-diffusion-3") {
            EncoderClass::ClipTriple
        } else if a.contains("sdxl") || a.contains("xl") {
            EncoderClass::ClipDual
        } else {
            EncoderClass::Clip
        }
    }

    /// Approximate persona token budget (words). PURE-CLIP families (SD 1.5/2.1, SDXL) are bounded by the
    /// 77-token CLIP limit; at ~1.7 tokens/word for short attribute phrases that is ~44 words, less a
    /// short subject clause + a little scene headroom → ~40. SD3.5 (ClipTriple) also carries a **T5-XXL**
    /// encoder that attends to long prompts, so like PixArt/Flux (T5) and Sana (Gemma) it is effectively
    /// unbounded. NB: attributes dropped from the TEXT still drive the geometry conditioning — the budget
    /// only trims what the prompt can hold, it does not discard the attribute.
    fn word_budget(self) -> Option<usize> {
        match self {
            EncoderClass::Clip | EncoderClass::ClipDual => Some(40),
            EncoderClass::ClipTriple | EncoderClass::T5 | EncoderClass::Gemma => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            EncoderClass::Clip => "clip",
            EncoderClass::ClipDual => "clip_dual",
            EncoderClass::ClipTriple => "clip_triple",
            EncoderClass::T5 => "t5",
            EncoderClass::Gemma => "gemma",
        }
    }
}

/// An extracted attribute value.
enum AttrVal {
    Scalar(f32),
    Enum(String),
    Color(Color),
}

/// A resolved, prompt-routed attribute (known + not the model prior).
#[derive(Debug, Clone)]
pub struct ResolvedAttr {
    pub path: String,
    pub section: String,
    pub phrase: String,
    pub salience: f32,
    pub negative: Option<String>,
}

/// The compiled prompt for one encoder class.
#[derive(Debug, Clone)]
pub struct Compiled {
    pub class: &'static str,
    pub positive: String,
    pub negative: String,
    pub emitted: Vec<String>,
    pub dropped: Vec<String>,
}

/// Resolve the spec against the lexicon into salience-ranked, prompt-routed attributes.
pub fn resolve(spec: &PersonaSpec, lex: &Lexicon) -> Vec<ResolvedAttr> {
    let mut out = Vec::new();
    // Deterministic iteration: sort the lexicon paths.
    let mut paths: Vec<&String> = lex.entries.keys().collect();
    paths.sort();
    for path in paths {
        let entry = &lex.entries[path];
        let Some(val) = attr_value(spec, path) else {
            continue; // unknown — contributes nothing (§6.4)
        };
        if let Some(ra) = resolve_one(path, entry, val) {
            out.push(ra);
        }
    }
    // Stable order: salience desc, then path for ties.
    out.sort_by(|a, b| {
        b.salience
            .partial_cmp(&a.salience)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });
    out
}

fn resolve_one(path: &str, entry: &LexEntry, val: AttrVal) -> Option<ResolvedAttr> {
    let weight = entry.control_weight() * entry.class_weight();
    let mk = |phrase: String, salience: f32, negative: Option<String>| ResolvedAttr {
        path: path.to_string(),
        section: entry.section.clone(),
        phrase,
        salience,
        negative,
    };
    match val {
        AttrVal::Scalar(v) => {
            // An explicit prior (0.5) contributes nothing to the prompt (§6.4).
            let dev = (v - 0.5).abs() * 2.0;
            if dev < 1e-4 {
                return None;
            }
            let (phrase, anti) = if v < 0.5 {
                (entry.low.clone(), entry.high.clone())
            } else {
                (entry.high.clone(), entry.low.clone())
            };
            phrase.map(|p| mk(p, dev * weight, anti))
        }
        AttrVal::Enum(v) => {
            if v == "none" {
                // e.g. facial_hair.style: none → a negative exclusion, no positive.
                return entry
                    .none_negative
                    .clone()
                    .map(|neg| mk(String::new(), 0.0, Some(neg)));
            }
            if v == "auto" {
                return None;
            }
            // Per-value phrasing (e.g. skin.tone fitzpatrick → neutral words) wins over the template.
            if let Some(map) = &entry.values {
                if let Some(phrase) = map.get(&v) {
                    if phrase.is_empty() {
                        return None; // this enum value deliberately emits nothing
                    }
                    return Some(mk(fix_article(phrase), weight, None));
                }
            }
            let t = entry.template.as_deref().unwrap_or("{}");
            Some(mk(fix_article(&t.replace("{}", &v)), weight, None))
        }
        AttrVal::Color(c) => match c {
            Color::Named(name) if name != "auto" => {
                let t = entry.template.as_deref().unwrap_or("{}");
                Some(mk(fix_article(&t.replace("{}", &name)), weight, None))
            }
            // Lab colours drive geometry/scorecard, not the prompt (skeleton); skip in text.
            _ => None,
        },
    }
}

/// Emit the positive + negative prompt for `class`, applying the CLIP token budget.
pub fn emit(resolved: &[ResolvedAttr], class: EncoderClass) -> Compiled {
    let mut emitted = Vec::new();
    let mut dropped = Vec::new();
    let mut negatives: Vec<String> = Vec::new();
    let mut phrases: Vec<String> = Vec::new();
    let mut used = 0usize;
    let budget = class.word_budget();

    for a in resolved {
        // Pure-negative attributes (facial_hair: none) contribute only to the negative list.
        if a.phrase.is_empty() {
            if let Some(n) = &a.negative {
                negatives.push(n.clone());
            }
            continue;
        }
        let cost = a.phrase.split_whitespace().count();
        if let Some(b) = budget {
            if used + cost > b {
                dropped.push(a.path.clone());
                continue;
            }
        }
        used += cost;
        phrases.push(a.phrase.clone());
        emitted.push(a.path.clone());
        if let Some(n) = &a.negative {
            negatives.push(n.clone());
        }
    }

    let positive = match class {
        EncoderClass::Clip | EncoderClass::ClipDual | EncoderClass::ClipTriple => phrases.join(", "),
        EncoderClass::T5 => {
            if phrases.is_empty() {
                String::new()
            } else {
                format!("A portrait of a person with {}.", join_natural(&phrases))
            }
        }
        EncoderClass::Gemma => {
            if phrases.is_empty() {
                String::new()
            } else {
                format!("A detailed portrait of a person who has {}.", join_natural(&phrases))
            }
        }
    };

    // Deduplicate negatives on a normalised basis, preserving order.
    let mut seen = std::collections::HashSet::new();
    let negative = negatives
        .into_iter()
        .flat_map(|n| n.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>())
        .filter(|s| !s.is_empty() && seen.insert(s.to_lowercase()))
        .collect::<Vec<_>>()
        .join(", ");

    Compiled { class: class.label(), positive, negative, emitted, dropped }
}

/// Add asserted-empty-collection negatives (§9.4): `marks: []` → moles/scars, `piercings: []` →
/// piercings/earrings. Absent collections (unknown) contribute nothing.
pub fn collection_negatives(spec: &PersonaSpec) -> Vec<String> {
    let mut n = Vec::new();
    if spec.marks.as_ref().is_some_and(|m| m.is_empty()) {
        n.push("moles, freckles, scars, blemishes".to_string());
    }
    if spec.piercings.as_ref().is_some_and(|p| p.is_empty()) {
        n.push("piercings, earrings".to_string());
    }
    n
}

/// Fix the `a`/`an` article when a template's `a ` lands before a vowel-initial word.
fn fix_article(phrase: &str) -> String {
    if let Some(rest) = phrase.strip_prefix("a ") {
        if rest.chars().next().is_some_and(|c| "aeiou".contains(c.to_ascii_lowercase())) {
            return format!("an {rest}");
        }
    }
    phrase.to_string()
}

fn join_natural(phrases: &[String]) -> String {
    match phrases.len() {
        0 => String::new(),
        1 => phrases[0].clone(),
        _ => {
            let (last, head) = phrases.split_last().unwrap();
            format!("{}, and {}", head.join(", "), last)
        }
    }
}

/// Compile a spec for a model alias — the top-level entry point (`persona show`).
pub fn compile_for_model(spec: &PersonaSpec, lex: &Lexicon, model: &str) -> Compiled {
    let class = EncoderClass::from_model(model);
    let resolved = resolve(spec, lex);
    let mut c = emit(&resolved, class);

    // Ground the render with age + sex + a photographic framing. Without it the model invents the age
    // (an adult persona drifts to an idealised young face) and the framing, and the bare attribute list
    // reads as an illustration. CLIP: a photographic lead at the front (highest salience). T5/Gemma:
    // slot the age+sex noun into the natural-language template.
    let noun = subject_noun(spec);
    c.positive = match class {
        EncoderClass::Clip | EncoderClass::ClipDual | EncoderClass::ClipTriple => {
            let lead = photo_lead(spec);
            if c.positive.is_empty() {
                format!("{lead} of {noun}")
            } else {
                format!("{lead} of {noun}, {}", c.positive)
            }
        }
        EncoderClass::T5 | EncoderClass::Gemma => {
            if c.positive.contains("a person") {
                c.positive.replacen("a person", &noun, 1)
            } else if c.positive.is_empty() {
                format!("A portrait photograph of {noun}.")
            } else {
                format!("A portrait photograph of {noun}, {}", c.positive)
            }
        }
    };

    // Fold in the asserted-empty-collection negatives + anti-uncanny negatives.
    // exclude child/underage drift explicitly so the render matches the adult spec.
    let mut negs = collection_negatives(spec);
    if spec.identity.is_some() {
        // Anti-uncanny only — fight the plastic/illustration look the bare attribute list tends to
        // produce. The subject's AGE is set by the positive clause (`a 31-year-old woman`), so a
        // legitimately-authored age renders correctly; no age is suppressed here.
        for n in ["cgi", "3d render", "illustration", "plastic skin", "airbrushed", "doll"] {
            negs.push(n.into());
        }
    }
    if !negs.is_empty() {
        let joined = negs.join(", ");
        c.negative = if c.negative.is_empty() { joined } else { format!("{}, {}", c.negative, joined) };
    }
    c
}

/// A concise per-person descriptor for a crowd/scene prompt (§14.3) — `a 31-year-old woman with auburn
/// hair`. Age + sex + the one or two most identifying surface cues (hair colour, facial hair), WITHOUT
/// the headshot framing (which conflicts with a group scene). Used by multiperson render.
pub fn crowd_descriptor(spec: &PersonaSpec) -> String {
    let id = spec.identity.as_ref();
    let noun = match id.and_then(|i| i.sex.as_deref()) {
        Some("female") => "woman",
        Some("male") => "man",
        _ => "person",
    };
    let mut d = match id.and_then(|i| i.apparent_age) {
        Some(age) => format!("a {age}-year-old {noun}"),
        None => format!("a {noun}"),
    };
    if let Some(Color::Named(c)) = spec.hair.as_ref().and_then(|h| h.color.as_ref()) {
        d.push_str(&format!(" with {c} hair"));
    }
    if let Some(style) = spec.facial_hair.as_ref().and_then(|f| f.style.as_deref()) {
        if style != "none" {
            d.push_str(&format!(" and a {}", style.replace('-', " ")));
        }
    }
    d
}

/// The subject noun phrase — `a 31-year-old woman` (age + sex, neutral, no valence). The age grounds
/// the render at the authored age and stops the drift to an idealised young face.
fn subject_noun(spec: &PersonaSpec) -> String {
    let id = spec.identity.as_ref();
    let noun = match id.and_then(|i| i.sex.as_deref()) {
        Some("female") => "woman",
        Some("male") => "man",
        _ => "person",
    };
    match id.and_then(|i| i.apparent_age) {
        Some(age) => format!("a {age}-year-old {noun}"),
        None => format!("a {noun}"),
    }
}

/// The photographic lead for CLIP — grounds the render as a real photograph (fighting the illustration
/// / cgi look) and carries the framing. `headshot` → `a portrait photograph` (a bare "headshot" tends
/// to an unnaturally tight crop).
fn photo_lead(spec: &PersonaSpec) -> &'static str {
    match spec.defaults.as_ref().and_then(|d| d.framing.as_deref()) {
        Some("half-body") | Some("waist-up") => "a waist-up portrait photograph",
        Some("full-body") | Some("full-length") => "a full-body photograph",
        _ => "a portrait photograph",
    }
}

/// A dentition-focused prompt for the mouth-region inpaint (RFC §8.7). Built from the `teeth` block
/// (alignment · shade · features · appliance). `None` when no teeth attributes are set.
pub fn dentition_prompt(spec: &PersonaSpec) -> Option<String> {
    let t = spec.teeth.as_ref()?;
    let mut words: Vec<String> = Vec::new();
    if let Some(al) = t.alignment.as_deref() {
        match al {
            "crowded" => words.push("crowded".into()),
            "gapped" | "diastema" => words.push("gapped".into()),
            "even" | "straight" => words.push("straight even".into()),
            "protruding" => words.push("protruding".into()),
            _ => {}
        }
    }
    if let Some(s) = t.shade {
        if s < 0.35 {
            words.push("bright white".into());
        } else if s > 0.65 {
            words.push("slightly yellowed".into());
        }
    }
    let mut prompt = format!("{} teeth", words.join(" ")).trim().to_string();
    if prompt == "teeth" && t.alignment.is_none() && t.shade.is_none() && t.features.is_none() && t.appliance.is_none() {
        return None; // teeth block present but empty of describable attributes
    }
    // discrete features + appliance.
    let mut extras: Vec<String> = Vec::new();
    if let Some(feats) = &t.features {
        for f in feats {
            match f.kind.as_deref() {
                Some("chip") | Some("chipped") => extras.push("a chipped tooth".into()),
                Some("gold-crown") | Some("gold") => extras.push("a gold tooth".into()),
                Some("missing") => extras.push("a missing tooth".into()),
                Some(other) => extras.push(format!("a {other} tooth")),
                None => {}
            }
        }
    }
    if let Some(app) = t.appliance.as_deref() {
        match app {
            "braces" => extras.push("metal braces".into()),
            "clear-aligner" | "retainer" => extras.push("a clear retainer".into()),
            other if other != "none" => extras.push(other.to_string()),
            _ => {}
        }
    }
    if !extras.is_empty() {
        prompt = format!("{prompt}, {}", extras.join(", "));
    }
    Some(prompt)
}

fn attr_value(spec: &PersonaSpec, path: &str) -> Option<AttrVal> {
    use AttrVal::*;
    match path {
        "face.shape" => spec.face.as_ref()?.shape.clone().map(Enum),
        "face.width" => spec.face.as_ref()?.width.map(Scalar),
        "eyes.shape" => spec.eyes.as_ref()?.shape.clone().map(Enum),
        "eyes.spacing" => spec.eyes.as_ref()?.spacing.map(Scalar),
        "eyes.color" => spec.eyes.as_ref()?.color.clone().map(Color),
        "eyes.brow.thickness" => spec.eyes.as_ref()?.brow.as_ref()?.thickness.map(Scalar),
        "nose.profile" => spec.nose.as_ref()?.profile.clone().map(Enum),
        "mouth.width" => spec.mouth.as_ref()?.width.map(Scalar),
        "hair.color" => spec.hair.as_ref()?.color.clone().map(Color),
        "hair.length" => spec.hair.as_ref()?.length.clone().map(Enum),
        "hair.texture" => spec.hair.as_ref()?.texture.clone().map(Enum),
        "facial_hair.style" => spec.facial_hair.as_ref()?.style.clone().map(Enum),
        "figure.build" => spec.figure.as_ref()?.build.clone().map(Enum),
        // extended coverage
        "face.jaw.width" => spec.face.as_ref()?.jaw.as_ref()?.width.map(Scalar),
        "face.jaw.definition" => spec.face.as_ref()?.jaw.as_ref()?.definition.clone().map(Enum),
        "face.chin.projection" => spec.face.as_ref()?.chin.as_ref()?.projection.map(Scalar),
        "face.cheekbones.prominence" => spec.face.as_ref()?.cheekbones.as_ref()?.prominence.map(Scalar),
        "eyes.canthal_tilt" => spec.eyes.as_ref()?.canthal_tilt.map(Scalar),
        "eyes.brow.arch" => spec.eyes.as_ref()?.brow.as_ref()?.arch.clone().map(Enum),
        "nose.length" => spec.nose.as_ref()?.length.map(Scalar),
        "mouth.lower_lip" => spec.mouth.as_ref()?.lower_lip.map(Scalar),
        "mouth.cupids_bow" => spec.mouth.as_ref()?.cupids_bow.clone().map(Enum),
        "skin.tone" => spec.skin.as_ref()?.tone.clone().map(Enum),
        "skin.undertone" => spec.skin.as_ref()?.undertone.clone().map(Enum),
        "skin.texture" => spec.skin.as_ref()?.texture.map(Scalar),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(s: &str) -> PersonaSpec {
        PersonaSpec::from_hjson(s).unwrap()
    }

    #[test]
    fn dentition_prompt_from_teeth_block() {
        assert!(dentition_prompt(&spec("{ schema: \"persona/1\" }")).is_none(), "no teeth → none");
        let p = dentition_prompt(&spec("{ schema: \"persona/1\"\n teeth: { alignment: \"crowded\", shade: 0.8 } }")).unwrap();
        assert!(p.contains("crowded") && p.contains("yellowed") && p.contains("teeth"), "{p}");
        let b = dentition_prompt(&spec("{ schema: \"persona/1\"\n teeth: { appliance: \"braces\" } }")).unwrap();
        assert!(b.contains("metal braces"), "{b}");
    }

    #[test]
    fn encoder_class_mapping() {
        assert_eq!(EncoderClass::from_model("sd15"), EncoderClass::Clip);
        assert_eq!(EncoderClass::from_model("sdxl"), EncoderClass::ClipDual);
        assert_eq!(EncoderClass::from_model("sana-600m"), EncoderClass::Gemma);
        assert_eq!(EncoderClass::from_model("flux-dev"), EncoderClass::T5);
    }

    #[test]
    fn explicit_prior_scalar_emits_nothing() {
        let lex = Lexicon::skeleton();
        let s = spec("{\n  eyes: {\n    spacing: 0.5\n  }\n}\n");
        let r = resolve(&s, &lex);
        assert!(r.iter().all(|a| a.path != "eyes.spacing"), "0.5 (prior) must not emit");
    }

    #[test]
    fn scalar_pole_and_anti() {
        let lex = Lexicon::skeleton();
        let s = spec("{\n  eyes: {\n    spacing: 0.85\n  }\n}\n");
        let r = resolve(&s, &lex);
        let a = r.iter().find(|a| a.path == "eyes.spacing").unwrap();
        assert_eq!(a.phrase, "wide-set eyes");
        assert_eq!(a.negative.as_deref(), Some("close-set eyes")); // opposite pole → negative
    }

    #[test]
    fn facial_hair_none_becomes_negative_only() {
        let lex = Lexicon::skeleton();
        let s = spec("{\n  facial_hair: {\n    style: none\n  }\n}\n");
        let c = emit(&resolve(&s, &lex), EncoderClass::Clip);
        assert!(!c.positive.contains("beard"));
        assert!(c.negative.contains("beard"));
    }

    #[test]
    fn clip_budget_drops_low_salience() {
        // Many attributes on a tiny budget → the least salient are dropped and recorded.
        let lex = Lexicon::skeleton();
        let s = spec(
            "{\n  face: {\n    shape: oval\n    width: 0.9\n  }\n  eyes: {\n    shape: almond\n    spacing: 0.9\n    color: hazel\n  }\n  nose: {\n    profile: aquiline\n  }\n  mouth: {\n    width: 0.9\n  }\n  hair: {\n    color: auburn\n    length: shoulder\n    texture: wavy\n  }\n  figure: {\n    build: mesomorph\n  }\n}\n",
        );
        let r = resolve(&s, &lex);
        let clip = emit(&r, EncoderClass::Clip);
        let gemma = emit(&r, EncoderClass::Gemma);
        // Gemma (unbounded) emits everything; clip drops some under budget.
        assert!(gemma.dropped.is_empty());
        assert!(clip.emitted.len() <= gemma.emitted.len());
        // Structural high-deviation attrs (eyes.spacing 0.9, face.width 0.9) outrank weak figure.build.
        assert!(clip.emitted.iter().any(|p| p == "eyes.spacing"));
    }

    #[test]
    fn end_to_end_prompt_and_asserted_empty_negatives() {
        let lex = Lexicon::skeleton();
        let s = spec(
            "{\n  eyes: {\n    color: hazel\n    spacing: 0.8\n  }\n  hair: {\n    color: auburn\n    length: shoulder\n  }\n  marks: []\n}\n",
        );
        let c = compile_for_model(&s, &lex, "sd15");
        assert!(c.positive.contains("hazel eyes"));
        assert!(c.positive.contains("auburn hair"));
        assert!(c.positive.contains("wide-set eyes"));
        assert!(c.negative.contains("moles")); // marks: [] → asserted-empty negative
    }

    #[test]
    fn t5_and_gemma_are_natural_language() {
        let lex = Lexicon::skeleton();
        let s = spec("{\n  eyes: {\n    color: hazel\n  }\n  hair: {\n    color: auburn\n  }\n}\n");
        let t5 = compile_for_model(&s, &lex, "flux-dev");
        assert!(t5.positive.starts_with("A portrait of a person with "));
        assert!(t5.positive.ends_with('.'));
    }

    /// Structural corpus (§24, §25 P0): a fixed reference spec compiles to exact, byte-stable prompts
    /// per encoder class. This is the deterministic-compiler regression gate — any drift is caught here.
    #[test]
    fn structural_corpus_golden() {
        let lex = Lexicon::skeleton();
        let s = spec(
            "{\n  schema: persona/1\n  identity: {\n    name: alice\n    apparent_age: 34\n  }\n\
             face: {\n    shape: oval\n  }\n  eyes: {\n    color: hazel\n    shape: almond\n    spacing: 0.78\n  }\n\
             nose: {\n    profile: aquiline\n  }\n  hair: {\n    color: auburn\n    length: shoulder\n    texture: wavy\n  }\n\
             facial_hair: {\n    style: none\n  }\n  marks: []\n}\n",
        );
        let clip = compile_for_model(&s, &lex, "sd15");
        assert_eq!(
            clip.positive,
            "a portrait photograph of a 34-year-old person, hazel eyes, almond eyes, auburn hair, shoulder-length hair, wavy hair, wide-set eyes, oval-shaped face, an aquiline nose"
        );
        assert_eq!(
            clip.negative,
            "close-set eyes, beard, stubble, moustache, goatee, facial hair, moles, freckles, scars, blemishes, cgi, 3d render, illustration, plastic skin, airbrushed, doll"
        );
        let t5 = compile_for_model(&s, &lex, "flux-dev");
        assert_eq!(
            t5.positive,
            "A portrait of a 34-year-old person with hazel eyes, almond eyes, auburn hair, shoulder-length hair, wavy hair, wide-set eyes, oval-shaped face, and an aquiline nose."
        );
    }

    #[test]
    fn per_value_phrasing_maps_neutrally() {
        // skin.tone fitzpatrick → neutral tone words (§7.4), never the raw enum value.
        let lex = Lexicon::skeleton();
        let s = spec("{\n  skin: {\n    tone: fitzpatrick-3\n  }\n}\n");
        let r = resolve(&s, &lex);
        let a = r.iter().find(|a| a.path == "skin.tone").unwrap();
        assert_eq!(a.phrase, "light skin");
        assert!(!a.phrase.contains("fitzpatrick"));
        // a `values` entry mapping to "" (mouth.cupids_bow flat) emits nothing.
        let flat = spec("{\n  mouth: {\n    cupids_bow: flat\n  }\n}\n");
        assert!(resolve(&flat, &lex).iter().all(|a| a.path != "mouth.cupids_bow"));
    }

    #[test]
    fn deterministic() {
        let lex = Lexicon::skeleton();
        let s = spec("{\n  eyes: {\n    color: hazel\n    spacing: 0.7\n  }\n  hair: {\n    color: auburn\n  }\n}\n");
        let a = compile_for_model(&s, &lex, "sdxl");
        let b = compile_for_model(&s, &lex, "sdxl");
        assert_eq!(a.positive, b.positive);
        assert_eq!(a.negative, b.negative);
    }
}
