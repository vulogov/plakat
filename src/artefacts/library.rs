//! Artefact library: the on-disk catalog of available cutouts.
//!
//! Mirrors the style catalog's shape: a directory containing
//! `library.json` (routing metadata) plus the PNG files referenced
//! from it. Loaded once at process start and looked up by name when
//! the user references an artefact.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::anchor::Anchor;
use super::zones::ZoneRef;

/// One artefact entry from `library.json`. After `load()`, `path` is
/// resolved to absolute (library_dir + relative path).
#[derive(Debug, Clone)]
pub struct Artefact {
    pub name: String,
    pub category: String,
    pub path: PathBuf,
    pub natural_zone: ZoneRef,
    pub natural_size_pct: f32,
    pub anchor: Anchor,
    pub license: Option<String>,
    pub license_url: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawLibrary {
    schema_version: u32,
    #[serde(default)]
    artefacts: Vec<RawArtefact>,
}

#[derive(Debug, Deserialize)]
struct RawArtefact {
    name: String,
    #[serde(default = "default_category")]
    category: String,
    path: PathBuf,
    natural_zone: ZoneRef,
    #[serde(default = "default_natural_size_pct")]
    natural_size_pct: f32,
    #[serde(default)]
    anchor: Option<Anchor>,
    #[serde(default)]
    license: Option<String>,
    #[serde(default)]
    license_url: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

fn default_category() -> String {
    "uncategorized".to_string()
}
fn default_natural_size_pct() -> f32 {
    0.7
}

/// The runtime artefact library — loaded from disk, indexed by name
/// for `O(1)` lookup, with `order` preserving the JSON order for
/// stable `plakat artefact list` output.
pub struct ArtefactLibrary {
    pub root: PathBuf,
    pub artefacts: HashMap<String, Artefact>,
    pub order: Vec<String>,
}

impl ArtefactLibrary {
    pub fn load(library_dir: &Path) -> Result<Self> {
        let json_path = library_dir.join("library.json");
        let raw_text = std::fs::read_to_string(&json_path)
            .with_context(|| format!("reading {}", json_path.display()))?;
        let raw: RawLibrary = serde_json::from_str(&raw_text)
            .with_context(|| format!("parsing {}", json_path.display()))?;

        if raw.schema_version != 1 {
            bail!(
                "artefact library at {} uses schema_version={}, plakat supports 1",
                json_path.display(),
                raw.schema_version
            );
        }

        let mut artefacts: HashMap<String, Artefact> = HashMap::with_capacity(raw.artefacts.len());
        let mut order: Vec<String> = Vec::with_capacity(raw.artefacts.len());

        for ra in raw.artefacts {
            if !ra.natural_size_pct.is_finite() || ra.natural_size_pct <= 0.0 {
                bail!(
                    "artefact {:?}: natural_size_pct must be positive + finite, got {}",
                    ra.name,
                    ra.natural_size_pct
                );
            }
            // Resolve path relative to library directory.
            let abs_path = if ra.path.is_absolute() {
                ra.path.clone()
            } else {
                library_dir.join(&ra.path)
            };

            if artefacts.contains_key(&ra.name) {
                bail!("artefact library has duplicate name {:?}", ra.name);
            }
            order.push(ra.name.clone());
            artefacts.insert(
                ra.name.clone(),
                Artefact {
                    name: ra.name,
                    category: ra.category,
                    path: abs_path,
                    natural_zone: ra.natural_zone,
                    natural_size_pct: ra.natural_size_pct,
                    anchor: ra.anchor.unwrap_or_default(),
                    license: ra.license,
                    license_url: ra.license_url,
                    tags: ra.tags,
                },
            );
        }

        Ok(Self {
            root: library_dir.to_owned(),
            artefacts,
            order,
        })
    }

    /// Look up an artefact by name. Returns a clear "did you mean"
    /// hint when the name doesn't exist (cheap edit-distance match).
    pub fn get(&self, name: &str) -> Result<&Artefact> {
        if let Some(a) = self.artefacts.get(name) {
            return Ok(a);
        }
        let candidates = self.suggest_similar(name, 3);
        if candidates.is_empty() {
            bail!(
                "unknown artefact {:?} (library has {} entries; run `plakat artefact list`)",
                name,
                self.order.len()
            );
        } else {
            bail!(
                "unknown artefact {:?}; closest matches: [{}]",
                name,
                candidates.join(", ")
            );
        }
    }

    /// Cheap typo-suggestion: simple substring match + length-prefix
    /// match. Returns up to `max` candidate names.
    fn suggest_similar(&self, query: &str, max: usize) -> Vec<String> {
        let q = query.to_lowercase();
        let mut hits: Vec<&str> = self
            .order
            .iter()
            .filter(|n| {
                let nl = n.to_lowercase();
                nl.contains(&q) || q.contains(&nl)
            })
            .map(String::as_str)
            .collect();
        hits.sort();
        hits.into_iter().take(max).map(String::from).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_library_dir(json: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let mut f = std::fs::File::create(dir.path().join("library.json")).unwrap();
        f.write_all(json.as_bytes()).unwrap();
        dir
    }

    #[test]
    fn loads_minimal_library() {
        let dir = temp_library_dir(
            r#"
            {
                "schema_version": 1,
                "artefacts": [
                    {
                        "name": "oak",
                        "category": "tree",
                        "path": "trees/oak.png",
                        "natural_zone": "middle_plan",
                        "natural_size_pct": 0.7,
                        "anchor": "bottom_center"
                    }
                ]
            }
            "#,
        );
        let lib = ArtefactLibrary::load(dir.path()).unwrap();
        assert_eq!(lib.order, vec!["oak"]);
        let oak = lib.get("oak").unwrap();
        assert_eq!(oak.category, "tree");
        assert_eq!(oak.anchor, Anchor::BOTTOM_CENTER);
        assert!(oak.path.is_absolute());
    }

    #[test]
    fn rejects_duplicate_names() {
        let dir = temp_library_dir(
            r#"
            {
                "schema_version": 1,
                "artefacts": [
                    {"name": "x", "path": "a.png", "natural_zone": "sky"},
                    {"name": "x", "path": "b.png", "natural_zone": "sky"}
                ]
            }
            "#,
        );
        assert!(ArtefactLibrary::load(dir.path()).is_err());
    }

    #[test]
    fn fractional_anchor_in_library() {
        let dir = temp_library_dir(
            r#"
            {
                "schema_version": 1,
                "artefacts": [
                    {
                        "name": "sun",
                        "path": "sun.png",
                        "natural_zone": "sky",
                        "anchor": { "x": 0.5, "y": 0.3 }
                    }
                ]
            }
            "#,
        );
        let lib = ArtefactLibrary::load(dir.path()).unwrap();
        let sun = lib.get("sun").unwrap();
        assert_eq!(sun.anchor, Anchor { x: 0.5, y: 0.3 });
    }

    #[test]
    fn lookup_suggests_close_matches() {
        let dir = temp_library_dir(
            r#"
            {
                "schema_version": 1,
                "artefacts": [
                    {"name": "oak", "path": "a.png", "natural_zone": "sky"},
                    {"name": "pine", "path": "b.png", "natural_zone": "sky"}
                ]
            }
            "#,
        );
        let lib = ArtefactLibrary::load(dir.path()).unwrap();
        let err = lib.get("oa").unwrap_err().to_string();
        assert!(err.contains("closest matches"), "got: {err}");
        assert!(err.contains("oak"), "got: {err}");
    }
}
