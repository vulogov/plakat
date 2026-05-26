//! Bundled negative-prompt presets plus an opt-in user catalog
//! under `<plakat-config-dir>/negative-presets/*.txt` (v0.20 #6).
//! Saves users from copy-pasting the same `blurry, low quality,
//! watermark, ...` line into every invocation.
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
//! The built-in registry is deliberately small (4 entries). User
//! files under `<plakat-config-dir>/negative-presets/<name>.txt`
//! extend (and can override) it — the lookup checks user files
//! first, then falls through to [`PRESETS`].

use std::borrow::Cow;
use std::path::{Path, PathBuf};

/// (alias, negative-prompt) pairs. Order shown in
/// [`supported_names()`] for diagnostics.
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

/// v0.20 #6: user-preset directory. Returns `None` if the platform
/// has no resolvable config dir (very rare — only happens on
/// stripped systems without `$HOME`).
///
/// Uses the same `directories::ProjectDirs` lookup as
/// [`crate::config::config_path`] so the user catalog lives next
/// to `config.toml`:
/// * Linux:   `~/.config/plakat/negative-presets/`
/// * macOS:   `~/Library/Application Support/ai.plakat.plakat/negative-presets/`
/// * Windows: `%APPDATA%\plakat\plakat\config\negative-presets\`
pub fn user_preset_dir() -> Option<PathBuf> {
    directories::ProjectDirs::from("ai", "plakat", "plakat")
        .map(|d| d.config_dir().join("negative-presets"))
}

/// Reject preset names that aren't simple identifiers. Anything
/// that could escape the user-preset directory (slashes, `..`) or
/// produce a surprising filename gets rejected; the resulting
/// `<dir>/<name>.txt` is then guaranteed to live inside the
/// catalog.
fn is_safe_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Read `<dir>/<name>.txt` if it exists. Trims trailing whitespace
/// (so a file authored in $EDITOR with the usual trailing newline
/// doesn't appear in the final negative as `..., bar, \n`).
/// Returns `Ok(None)` when the file isn't present — that's the
/// "fall through to built-ins" path.
fn read_user_preset(dir: &Path, name: &str) -> std::io::Result<Option<String>> {
    if !is_safe_name(name) {
        return Ok(None);
    }
    let path = dir.join(format!("{name}.txt"));
    match std::fs::read_to_string(&path) {
        Ok(s) => {
            let trimmed = s.trim().to_string();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                Ok(Some(trimmed))
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// List preset names defined under `dir` (file stems of `*.txt`
/// entries that pass [`is_safe_name`]). Sorted, deduplicated.
/// Returns an empty vec if the directory doesn't exist — that's
/// the unconfigured-system path.
fn list_user_presets_in(dir: &Path) -> Vec<String> {
    let entries = match std::fs::read_dir(dir) {
        Ok(it) => it,
        Err(_) => return Vec::new(),
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("txt") {
                return None;
            }
            let stem = p.file_stem().and_then(|s| s.to_str())?;
            if !is_safe_name(stem) {
                return None;
            }
            Some(stem.to_string())
        })
        .collect();
    names.sort();
    names.dedup();
    names
}

/// Public listing of user-defined preset names (for diagnostics +
/// `plakat doctor`-style introspection).
pub fn list_user_presets() -> Vec<String> {
    match user_preset_dir() {
        Some(d) => list_user_presets_in(&d),
        None => Vec::new(),
    }
}

/// Internal resolver used by both [`resolve`] and tests. Allows
/// the user-preset directory to be overridden via `dir` for
/// hermetic tests.
fn resolve_in(name: &str, dir: Option<&Path>) -> Option<Cow<'static, str>> {
    // User files win — that's the override path. A user who
    // wants stricter `photo` than the built-in drops a file at
    // `<dir>/photo.txt` and gets it without renaming anything.
    if let Some(d) = dir {
        if let Ok(Some(body)) = read_user_preset(d, name) {
            return Some(Cow::Owned(body));
        }
    }
    PRESETS
        .iter()
        .find(|(alias, _)| alias.eq_ignore_ascii_case(name))
        .map(|(_, neg)| Cow::Borrowed(*neg))
}

/// Look up a preset by name. Case-insensitive. Returns `None`
/// for unregistered names — callers should pair with
/// [`supported_names()`] for a friendly diagnostic.
///
/// v0.20 #6: user files under [`user_preset_dir()`] take
/// precedence over built-in [`PRESETS`].
pub fn resolve(name: &str) -> Option<Cow<'static, str>> {
    resolve_in(name, user_preset_dir().as_deref())
}

/// Comma-joined list of every registered preset name. Includes
/// both built-ins and user-defined entries; user names get a
/// trailing `(user)` annotation so the source is unambiguous in
/// error output.
pub fn supported_names() -> String {
    let user = list_user_presets();
    let built_ins: Vec<&str> = PRESETS.iter().map(|(n, _)| *n).collect();

    let mut parts: Vec<String> = built_ins.iter().map(|s| (*s).to_string()).collect();
    for name in &user {
        // Avoid double-listing when a user file shadows a built-in.
        // Mark the built-in as "(overridden)" in that case — useful
        // signal in error output.
        let lower = name.to_ascii_lowercase();
        if let Some(idx) = parts.iter().position(|p| p.eq_ignore_ascii_case(&lower)) {
            parts[idx] = format!("{name} (user override)");
        } else {
            parts.push(format!("{name} (user)"));
        }
    }
    parts.join(", ")
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
        Ok(preset.into_owned())
    } else {
        Ok(format!("{preset}, {user_negative}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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

    // v0.20 #6 — user-preset catalog.

    #[test]
    fn name_safety_rejects_traversal_and_specials() {
        assert!(is_safe_name("photo"));
        assert!(is_safe_name("my_neg"));
        assert!(is_safe_name("v1-strict"));
        assert!(!is_safe_name(""));
        assert!(!is_safe_name(".."));
        assert!(!is_safe_name("../etc/passwd"));
        assert!(!is_safe_name("a/b"));
        assert!(!is_safe_name("a.b"));
        assert!(!is_safe_name("a b"));
        assert!(!is_safe_name("a\nb"));
    }

    #[test]
    fn user_file_resolves_when_built_in_absent() {
        let tmp = tempdir().unwrap();
        std::fs::write(
            tmp.path().join("anatomy.txt"),
            "extra fingers, bad hands, missing teeth\n",
        )
        .unwrap();
        let got = resolve_in("anatomy", Some(tmp.path())).unwrap();
        assert_eq!(got.as_ref(), "extra fingers, bad hands, missing teeth");
    }

    #[test]
    fn user_file_overrides_built_in_preset() {
        let tmp = tempdir().unwrap();
        std::fs::write(tmp.path().join("photo.txt"), "STRICTER PHOTO NEG\n")
            .unwrap();
        let got = resolve_in("photo", Some(tmp.path())).unwrap();
        assert_eq!(got.as_ref(), "STRICTER PHOTO NEG");
    }

    #[test]
    fn missing_user_file_falls_through_to_built_in() {
        let tmp = tempdir().unwrap();
        // No file in the dir — built-in `photo` should still resolve.
        let got = resolve_in("photo", Some(tmp.path())).unwrap();
        assert!(got.as_ref().contains("blurry"));
    }

    #[test]
    fn empty_user_file_falls_through_to_built_in() {
        // A user who emptied their override should get the
        // built-in back rather than the empty string (which CLIP
        // would treat as "no negatives").
        let tmp = tempdir().unwrap();
        std::fs::write(tmp.path().join("photo.txt"), "   \n\n").unwrap();
        let got = resolve_in("photo", Some(tmp.path())).unwrap();
        assert!(got.as_ref().contains("blurry"));
    }

    #[test]
    fn user_file_with_traversal_name_ignored() {
        // The lookup never gets to `is_safe_name` for this call
        // (resolve_in does its own is_safe_name check via
        // read_user_preset). The point of the test is that a
        // caller passing a hostile name doesn't escape the dir.
        let tmp = tempdir().unwrap();
        // Pretend the attacker placed a file with a hostile name.
        let escape = tmp.path().join("../escape.txt");
        let _ = std::fs::write(&escape, "should never be read");
        let got = resolve_in("../escape", Some(tmp.path()));
        assert!(got.is_none(), "traversal name resolved!");
    }

    #[test]
    fn list_user_presets_in_dir() {
        let tmp = tempdir().unwrap();
        std::fs::write(tmp.path().join("strict.txt"), "x").unwrap();
        std::fs::write(tmp.path().join("loose.txt"), "y").unwrap();
        std::fs::write(tmp.path().join("README.md"), "not a preset").unwrap();
        std::fs::write(tmp.path().join("bad name.txt"), "ignored").unwrap();
        let names = list_user_presets_in(tmp.path());
        assert_eq!(names, vec!["loose".to_string(), "strict".to_string()]);
    }

    #[test]
    fn list_user_presets_missing_dir_returns_empty() {
        let tmp = tempdir().unwrap();
        let missing = tmp.path().join("does-not-exist");
        assert!(list_user_presets_in(&missing).is_empty());
    }

    #[test]
    fn read_user_preset_trims_trailing_whitespace() {
        let tmp = tempdir().unwrap();
        std::fs::write(tmp.path().join("p.txt"), "foo, bar\n\n  ").unwrap();
        let got = read_user_preset(tmp.path(), "p").unwrap().unwrap();
        assert_eq!(got, "foo, bar");
    }
}
