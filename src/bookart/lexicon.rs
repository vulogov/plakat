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
}
