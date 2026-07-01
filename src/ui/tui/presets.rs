//! Named generation presets (RFC — 1.21.0 workflow) — save the current model +
//! LoRA stack + negative prompt under a name and re-apply it in one step from the
//! command palette. Persisted as `<workspace>/presets.json`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One saved bundle: a base model plus the LoRA stack (each with its weight) and the
/// session negative prompt.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Preset {
    pub name: String,
    /// Base-model alias (e.g. `sdxl`).
    pub model: String,
    /// Applied LoRAs as (path, weight).
    #[serde(default)]
    pub loras: Vec<(PathBuf, f32)>,
    #[serde(default)]
    pub negative: String,
    /// Generation size override (w, h) for this model; `None` = the model's native square.
    #[serde(default)]
    pub size: Option<(u32, u32)>,
    /// Denoise steps override; `None` = the workspace default.
    #[serde(default)]
    pub steps: Option<usize>,
    /// Guidance (CFG) override; `None` = the workspace default.
    #[serde(default)]
    pub guidance: Option<f64>,
}

impl Preset {
    /// A compact one-line summary for the palette / listing.
    pub fn summary(&self) -> String {
        let loras = match self.loras.len() {
            0 => "no LoRAs".to_string(),
            1 => "1 LoRA".to_string(),
            n => format!("{n} LoRAs"),
        };
        let mut s = format!("{} · {loras}", self.model);
        if let Some((w, h)) = self.size {
            s.push_str(&format!(" · {w}×{h}"));
        }
        if let Some(st) = self.steps {
            s.push_str(&format!(" · {st} steps"));
        }
        if let Some(g) = self.guidance {
            s.push_str(&format!(" · cfg {g:.1}"));
        }
        s
    }
}

/// The workspace's preset file.
fn presets_path(root: &Path) -> PathBuf {
    root.join("presets.json")
}

/// Load all presets for a workspace (empty when the file is absent or unreadable —
/// presets are a convenience, never load-bearing).
pub fn load(root: &Path) -> Vec<Preset> {
    match std::fs::read(presets_path(root)) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

/// Persist the full preset list, pretty-printed. Returns an error string on failure.
pub fn save(root: &Path, presets: &[Preset]) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(presets).map_err(|e| e.to_string())?;
    std::fs::write(presets_path(root), bytes).map_err(|e| e.to_string())
}

/// Insert-or-replace a preset by name (case-insensitive) and persist. Returns the new
/// full list on success.
pub fn upsert(root: &Path, preset: Preset) -> Result<Vec<Preset>, String> {
    let mut all = load(root);
    if let Some(slot) = all.iter_mut().find(|p| p.name.eq_ignore_ascii_case(&preset.name)) {
        *slot = preset;
    } else {
        all.push(preset);
    }
    all.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    save(root, &all)?;
    Ok(all)
}

/// Find a preset by name (case-insensitive).
pub fn find(root: &Path, name: &str) -> Option<Preset> {
    load(root).into_iter().find(|p| p.name.eq_ignore_ascii_case(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(label: &str) -> PathBuf {
        // A per-test dir under the OS temp root (no Instant/rand → pid + a label keep
        // parallel tests from colliding on the same file).
        let dir = std::env::temp_dir().join(format!("plakat-presets-test-{}-{label}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::remove_file(presets_path(&dir));
        dir
    }

    #[test]
    fn upsert_load_and_find_round_trip() {
        let root = tmp("roundtrip");
        assert!(load(&root).is_empty());
        let p = Preset {
            name: "portrait".into(),
            model: "sdxl".into(),
            loras: vec![(PathBuf::from("/l/film.safetensors"), 0.8)],
            negative: "blurry".into(),
            size: Some((1024, 768)),
            steps: Some(30),
            guidance: Some(6.5),
        };
        let all = upsert(&root, p.clone()).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(find(&root, "PORTRAIT").as_ref(), Some(&p), "case-insensitive find");
        assert!(find(&root, "nope").is_none());
    }

    #[test]
    fn upsert_replaces_same_name_and_sorts() {
        let root = tmp("replace");
        upsert(&root, Preset { name: "zeta".into(), model: "sd15".into(), ..Default::default() }).unwrap();
        upsert(&root, Preset { name: "alpha".into(), model: "sd15".into(), ..Default::default() }).unwrap();
        // Replace "zeta" with a different model.
        let all = upsert(&root, Preset { name: "ZETA".into(), model: "sdxl".into(), ..Default::default() }).unwrap();
        assert_eq!(all.len(), 2, "same name (any case) replaces, not appends");
        assert_eq!(all[0].name, "alpha", "sorted by name");
        assert_eq!(all.iter().find(|p| p.name == "ZETA").unwrap().model, "sdxl");
    }

    #[test]
    fn summary_pluralizes_loras() {
        let mk = |n: usize| Preset {
            name: "x".into(),
            model: "sdxl".into(),
            loras: vec![(PathBuf::from("/a"), 1.0); n],
            ..Default::default()
        };
        assert!(mk(0).summary().ends_with("no LoRAs"));
        assert!(mk(1).summary().ends_with("1 LoRA"));
        assert!(mk(3).summary().ends_with("3 LoRAs"));
        // Recipe fields append to the summary when present.
        let full = Preset { name: "y".into(), model: "sdxl".into(), size: Some((1024, 768)), steps: Some(30), guidance: Some(6.5), ..Default::default() };
        let sum = full.summary();
        assert!(sum.contains("1024×768") && sum.contains("30 steps") && sum.contains("cfg 6.5"));
    }
}
