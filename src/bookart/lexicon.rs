//! Layer 0b — the origin × technique × ornament lexicon (RFC BOOKART-1 §8). B0 slice: a **built-in**
//! vocabulary + defaults (origin prompt scaffolds, technique descriptors + binarisers, per-ornament-type
//! default tier/symmetry/aspect). Later phases load overrides from `assets/bookart/lexicon.hjson`; the
//! built-in is always the fallback, so the resolver is never data-dependent to *function*.
//!
//! Two orthogonal axes (§8): **origin** (a tradition — russian/english/…) and **technique** (a drawing
//! method — line/woodcut/…). They compose: `russian × woodcut` and `russian × line` are both reachable.

/// Illustration traditions. `generic` is the always-available, LoRA-free path (G0.3): it works for any
/// technique via prompt + lineart-CN + the finisher, so v1 never blocks on a trained LoRA.
pub const ORIGINS: &[&str] = &["russian", "english", "japanese", "american", "european", "chinese", "generic"];

/// Drawing methods (orthogonal to origin). Drives the LoRA choice *and* the finisher binariser.
pub const TECHNIQUES: &[&str] = &["line", "woodcut", "engraving", "stipple", "cross-hatch", "silhouette", "ink-wash", "scratchboard"];

/// The ornament vocabulary (RFC §4).
pub const ORNAMENTS: &[&str] =
    &["headpiece", "tailpiece", "initial", "corner", "border", "divider", "fleuron", "dinkus", "vignette", "frontispiece", "colophon", "endpaper", "marginalia"];

pub const TRANSPARENCY_MODES: &[&str] = &["luminance", "threshold", "matte", "fade"];
pub const TIERS: &[&str] = &["auto", "procedural", "diffusion", "composite"];
pub const SYMMETRIES: &[&str] = &["bilateral", "radial", "frieze", "none"]; // `radial`/`frieze` accept a `:N`/`:GROUP` suffix

/// Origin → a prompt scaffold (tradition cue) + its default technique + default motifs. `generic` is the
/// neutral line-art path. Trained LoRAs (russian/english/japanese) attach in B4; the scaffold stands
/// alone until then.
pub fn origin_scaffold(origin: &str) -> (&'static str, &'static str, &'static [&'static str]) {
    match origin {
        "russian" => ("in the tradition of Russian folk book illustration, Bilibin lubok ornament", "woodcut", &["firebird", "oak-leaf", "vine"]),
        "english" => ("in the tradition of English pen illustration, Beardsley line and Morris foliate ornament", "line", &["rose", "vine", "peacock"]),
        "japanese" => ("in the tradition of Japanese sumi brush and ukiyo-e line illustration", "line", &["wave", "crane", "pine"]),
        "american" => ("in the tradition of American golden-age book illustration, bold woodcut", "woodcut", &["oak-leaf", "star"]),
        "european" => ("in the tradition of European engraving, Dürer and Doré line", "engraving", &["acanthus", "laurel"]),
        "chinese" => ("in the tradition of Chinese baimiao outline and woodblock illustration", "line", &["lotus", "crane", "cloud"]),
        _ => ("clean black ink book illustration, ornamental", "line", &["leaf", "scroll"]),
    }
}

/// The binariser a technique wants in the finisher (§7.1). Names are stable identifiers the B1 finisher
/// dispatches on.
pub fn technique_binariser(technique: &str) -> &'static str {
    match technique {
        "woodcut" => "threshold-bold",
        "engraving" => "engrave-invert",
        "stipple" => "dither",
        "cross-hatch" => "xdog",
        "silhouette" => "matte-solid",
        "ink-wash" => "halftone",
        "scratchboard" => "threshold-invert",
        _ => "xdog", // line
    }
}

/// A short technique prompt cue.
pub fn technique_prompt(technique: &str) -> &'static str {
    match technique {
        "woodcut" => "bold woodcut, high contrast",
        "engraving" => "fine engraving, cross-hatched linework",
        "stipple" => "stippled, dotted shading",
        "cross-hatch" => "cross-hatched pen lines",
        "silhouette" => "solid black silhouette",
        "ink-wash" => "sumi ink wash",
        "scratchboard" => "white lines on black, scratchboard",
        _ => "clean black ink line art, no shading",
    }
}

/// Is an ornament type *pictorial* (wants diffusion) vs *geometric* (wants procedural)?
pub fn ornament_pictorial(kind: &str) -> bool {
    matches!(kind, "vignette" | "frontispiece" | "marginalia" | "colophon")
}

/// The default render tier for an ornament type (§5.3): geometric → procedural, pictorial → diffusion,
/// framed-pictorial → composite.
pub fn default_tier(kind: &str) -> &'static str {
    match kind {
        "border" | "corner" | "divider" | "fleuron" | "dinkus" | "endpaper" => "procedural",
        "vignette" | "frontispiece" | "marginalia" | "colophon" => "diffusion",
        "headpiece" | "tailpiece" | "initial" => "composite",
        _ => "diffusion",
    }
}

/// The default symmetry for an ornament type.
pub fn default_symmetry(kind: &str) -> &'static str {
    match kind {
        "headpiece" | "border" | "divider" | "fleuron" | "tailpiece" => "bilateral",
        "colophon" | "endpaper" => "radial:8",
        _ => "none", // corner is placed 4× by layout; vignette/frontispiece/initial are free
    }
}

/// The origins that ship a trained sd15 LoRA hosted at `vulogov98/plakat-bookart` (B4). The others in
/// [`ORIGINS`] are valid vocabulary but render on the generic scaffold path (no LoRA) until B5 trains
/// them — so we must NOT attach a `<origin>-sd15.safetensors` that doesn't exist (it would 404).
pub const HOSTED_LORA_ORIGINS: &[&str] = &["russian", "english", "japanese"];

// ---------------------------------------------------------------------------------------------------
// 6.1.0 (B3): optional `assets/bookart/lexicon.hjson` override. The built-in lexicon above is always
// the fallback; an override can add new traditions (or re-scaffold an existing one) without a rebuild.
// ---------------------------------------------------------------------------------------------------

/// One origin entry in the override file. Every field but `name` is optional — a missing field falls
/// back to the built-in [`origin_scaffold`] for that origin (or the neutral default for a brand-new one).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct OriginOverride {
    pub name: String,
    #[serde(default)]
    pub scaffold: Option<String>,
    #[serde(default)]
    pub default_technique: Option<String>,
    #[serde(default)]
    pub motif: Option<Vec<String>>,
    /// `true` if a `<name>-sd15.safetensors` LoRA is hosted at `vulogov98/plakat-bookart`.
    #[serde(default)]
    pub hosted_lora: bool,
}

/// The override document (`assets/bookart/lexicon.hjson`). Additive: today just custom origins.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct LexiconOverride {
    #[serde(default)]
    pub origins: Vec<OriginOverride>,
}

/// The override file path — `assets/bookart/lexicon.hjson`, overridable via `PLAKAT_BOOKART_LEXICON`.
pub fn override_path() -> std::path::PathBuf {
    std::env::var_os("PLAKAT_BOOKART_LEXICON")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("assets/bookart/lexicon.hjson"))
}

/// Load + cache the override once per process. Absent file → `None` (built-in only). A malformed file
/// warns and is ignored (the built-in lexicon always works, so a bad override never breaks rendering).
pub fn lexicon_override() -> Option<&'static LexiconOverride> {
    static OVERRIDE: std::sync::OnceLock<Option<LexiconOverride>> = std::sync::OnceLock::new();
    OVERRIDE
        .get_or_init(|| {
            let path = override_path();
            let text = std::fs::read_to_string(&path).ok()?;
            match deser_hjson::from_str::<LexiconOverride>(&text) {
                Ok(l) => Some(l),
                Err(e) => {
                    tracing::warn!(target: "plakat", "bookart lexicon override {} ignored (parse error): {e}", path.display());
                    None
                }
            }
        })
        .as_ref()
}

fn find_override(origin: &str) -> Option<&'static OriginOverride> {
    lexicon_override()?.origins.iter().find(|o| o.name.eq_ignore_ascii_case(origin))
}

/// Origin scaffold resolved through the override (owned): the override's fields win, missing ones fall
/// back to the built-in [`origin_scaffold`]. The resolver uses this so a custom tradition renders.
pub fn origin_scaffold_dyn(origin: &str) -> (String, String, Vec<String>) {
    let (bs, bt, bm) = origin_scaffold(origin);
    if let Some(o) = find_override(origin) {
        return (
            o.scaffold.clone().unwrap_or_else(|| bs.to_string()),
            o.default_technique.clone().unwrap_or_else(|| bt.to_string()),
            o.motif.clone().unwrap_or_else(|| bm.iter().map(|s| s.to_string()).collect()),
        );
    }
    (bs.to_string(), bt.to_string(), bm.iter().map(|s| s.to_string()).collect())
}

/// Does this origin have a hosted sd15 LoRA to attach (built-in list, or an override that declares one)?
pub fn has_hosted_lora(origin: &str) -> bool {
    if let Some(o) = find_override(origin) {
        return o.hosted_lora;
    }
    HOSTED_LORA_ORIGINS.contains(&origin)
}

/// All known origins — built-in [`ORIGINS`] plus any the override adds (de-duped, built-ins first).
pub fn all_origins() -> Vec<String> {
    let mut v: Vec<String> = ORIGINS.iter().map(|s| s.to_string()).collect();
    if let Some(ov) = lexicon_override() {
        for o in &ov.origins {
            if !v.iter().any(|x| x.eq_ignore_ascii_case(&o.name)) {
                v.push(o.name.clone());
            }
        }
    }
    v
}

/// Nearest known vocabulary entry (case-insensitive edit-distance) — for lint suggestions.
pub fn nearest<'a>(word: &str, vocab: &[&'a str]) -> Option<&'a str> {
    let w = word.to_ascii_lowercase();
    vocab
        .iter()
        .map(|&c| (levenshtein(&w, &c.to_ascii_lowercase()), c))
        .filter(|&(d, c)| d <= 3.max(c.len() / 2))
        .min_by_key(|&(d, _)| d)
        .map(|(_, c)| c)
}

fn levenshtein(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_and_symmetry_defaults() {
        assert_eq!(default_tier("border"), "procedural");
        assert_eq!(default_tier("vignette"), "diffusion");
        assert_eq!(default_tier("headpiece"), "composite");
        assert_eq!(default_symmetry("headpiece"), "bilateral");
        assert_eq!(default_symmetry("corner"), "none");
    }

    #[test]
    fn nearest_suggests() {
        assert_eq!(nearest("woodcutt", TECHNIQUES), Some("woodcut"));
        assert_eq!(nearest("russsian", ORIGINS), Some("russian"));
        assert_eq!(nearest("zzzzzzzz", ORIGINS), None);
    }

    #[test]
    fn only_trained_origins_report_a_hosted_lora() {
        // The correctness guard: attaching a `<origin>-sd15.safetensors` that doesn't exist would 404.
        for o in ["russian", "english", "japanese"] {
            assert!(has_hosted_lora(o), "{o} ships a LoRA");
        }
        for o in ["american", "european", "chinese", "generic"] {
            assert!(!has_hosted_lora(o), "{o} has no hosted LoRA (scaffold path)");
        }
    }

    #[test]
    fn override_hjson_parses_custom_origins() {
        let text = r#"{
            origins: [
                { name: "byzantine", scaffold: "Byzantine illuminated manuscript ornament", default_technique: "engraving", motif: ["cross", "vine"], hosted_lora: false }
                { name: "russian", hosted_lora: true }
            ]
        }"#;
        let ov: LexiconOverride = deser_hjson::from_str(text).unwrap();
        assert_eq!(ov.origins.len(), 2);
        let byz = &ov.origins[0];
        assert_eq!(byz.name, "byzantine");
        assert_eq!(byz.default_technique.as_deref(), Some("engraving"));
        assert_eq!(byz.motif.as_ref().unwrap(), &vec!["cross".to_string(), "vine".to_string()]);
        assert!(!byz.hosted_lora);
        // A partial entry (only name + flag) leaves the rest to the built-in fallback.
        assert!(ov.origins[1].scaffold.is_none());
    }
}
