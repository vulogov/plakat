//! Bundled negative-prompt presets. Saves users from copy-pasting
//! the same `blurry, low quality, watermark, ...` line into every
//! invocation. Each preset is hand-tuned for one of the common
//! aesthetics plakat targets.
//!
//! Usage at the CLI:
//!
//! ```text
//! --negative-preset photo               (preset alone)
//! --negative "ugly hands"               (user-only — existing behaviour)
//! --negative-preset photo \
//!   --negative "ugly hands"             (combine via comma-join)
//! ```
//!
//! Preset alone wins when both flags are set; the user's
//! `--negative` is appended after the preset so explicit negatives
//! still apply on top. CLIP weights each token equally — adding a
//! few generic negatives on top of a curated preset doesn't dilute
//! the preset's signal.
//!
//! The registry is deliberately small (4 entries). The point is to
//! ship sane defaults for the four common targets; users who want
//! exotic negatives still write them inline.

/// (alias, negative-prompt) pairs. Order shown in
/// `supported_names()` for diagnostics.
pub const PRESETS: &[(&str, &str)] = &[
    (
        "photo",
        // Realistic photography target. The "blurry" / "low quality"
        // / "jpeg artifacts" trio suppresses common SD failure modes
        // at photo-style prompts; "watermark" / "signature" keep
        // logo / artist-mark overlays out.
        "blurry, low quality, oversaturated, jpeg artifacts, \
         watermark, signature, deformed, bad anatomy",
    ),
    (
        "painting",
        // Painted-art target (oil / watercolor / acrylic). Drops
        // "deformed" + "bad anatomy" because painterly styles
        // accept some anatomical liberty; keeps quality-suppressors
        // and overlay-suppressors.
        "low quality, blurry, jpeg artifacts, watermark, signature, \
         frame, picture frame, border",
    ),
    (
        "anime",
        // Anime / illustration target. Includes the
        // hand-and-finger-specific negatives that anime users tend
        // to fight the most. Note: paste these alongside a positive
        // prompt that includes "masterpiece, best quality" — that
        // pairing is the convention.
        "lowres, bad anatomy, bad hands, missing fingers, extra digit, \
         fewer digits, cropped, worst quality, low quality, \
         jpeg artifacts, signature, watermark, username, blurry",
    ),
    (
        "cinematic",
        // Film / poster aesthetic. Aimed at moody compositions
        // where the photo preset's oversaturation negative would be
        // too aggressive (cinematic outputs LOVE color saturation
        // when used purposefully).
        "blurry, low quality, watermark, signature, deformed, \
         bad anatomy, ugly",
    ),
];

/// Look up a preset by name. Case-insensitive. Returns `None`
/// for unregistered names — callers should pair with
/// `supported_names()` for a friendly diagnostic.
pub fn resolve(name: &str) -> Option<&'static str> {
    PRESETS
        .iter()
        .find(|(alias, _)| alias.eq_ignore_ascii_case(name))
        .map(|(_, neg)| *neg)
}

/// Comma-joined list of every registered preset name. Used in
/// error messages so a fat-fingered `--negative-preset` doesn't
/// leave the user grepping the source.
pub fn supported_names() -> String {
    PRESETS
        .iter()
        .map(|(alias, _)| *alias)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Combine a preset (resolved by name) with the user's existing
/// `--negative` value. Empty fields are dropped; both non-empty
/// → preset first, user appended after a `", "` join. Returns
/// `Err` only when the preset name is unknown (a typo is worth
/// bailing on, not silently ignoring).
pub fn combine(
    preset_name: Option<&str>,
    user_negative: &str,
) -> Result<String, anyhow::Error> {
    let preset = match preset_name {
        None => return Ok(user_negative.to_string()),
        Some(n) => resolve(n).ok_or_else(|| {
            anyhow::anyhow!(
                "--negative-preset {n:?} not recognised. Supported: {}",
                supported_names()
            )
        })?,
    };
    if user_negative.trim().is_empty() {
        Ok(preset.to_string())
    } else {
        Ok(format!("{preset}, {user_negative}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_names_are_unique_and_lowercase() {
        let mut names: Vec<&str> = PRESETS.iter().map(|(n, _)| *n).collect();
        names.sort();
        let mut dedup = names.clone();
        dedup.dedup();
        assert_eq!(names.len(), dedup.len(), "preset name collision");
        for n in &names {
            assert_eq!(
                *n,
                n.to_ascii_lowercase(),
                "preset names must be lowercase for canonical match"
            );
        }
    }

    #[test]
    fn registry_negatives_are_non_empty() {
        for (alias, neg) in PRESETS {
            assert!(!neg.trim().is_empty(), "{alias} preset has empty negative");
        }
    }

    #[test]
    fn resolve_case_insensitive() {
        assert!(resolve("photo").is_some());
        assert!(resolve("PHOTO").is_some());
        assert!(resolve("Photo").is_some());
        assert!(resolve("Painting").is_some());
        assert!(resolve("anime").is_some());
        assert!(resolve("cinematic").is_some());
    }

    #[test]
    fn resolve_unknown_returns_none() {
        assert!(resolve("not-a-preset").is_none());
    }

    #[test]
    fn supported_names_includes_each_preset() {
        let s = supported_names();
        for (alias, _) in PRESETS {
            assert!(s.contains(alias), "supported_names missing {alias}");
        }
    }

    #[test]
    fn combine_no_preset_passes_through() {
        let out = combine(None, "blurry").unwrap();
        assert_eq!(out, "blurry");
    }

    #[test]
    fn combine_preset_only_returns_preset() {
        let out = combine(Some("photo"), "").unwrap();
        assert!(out.starts_with("blurry, low quality"));
    }

    #[test]
    fn combine_both_concatenates() {
        let out = combine(Some("photo"), "ugly hands").unwrap();
        assert!(out.starts_with("blurry, low quality"));
        assert!(out.ends_with("ugly hands"));
        assert!(out.contains(", ugly hands"));
    }

    #[test]
    fn combine_unknown_preset_bails_with_supported_list() {
        let err = combine(Some("not-a-preset"), "").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not recognised"));
        assert!(msg.contains("photo"));
    }

    #[test]
    fn combine_whitespace_only_negative_treated_as_empty() {
        let out = combine(Some("photo"), "   ").unwrap();
        // Whitespace-only `--negative` collapses to "empty"; result
        // is the preset alone with no trailing comma-whitespace.
        assert!(!out.ends_with(", "));
        assert!(!out.contains("   "));
    }
}
