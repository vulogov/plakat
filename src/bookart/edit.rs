//! The class-aware edit loop (RFC BOOKART-1 §9 analog / ROADMAP B9). Changing an attribute should cost
//! only what it must: recolouring the ink is a **post** op on the finished PNG; a new page size is a
//! **re-raster**; a new origin/motif/prompt is a full **re-gen**. `diff` classifies every changed field
//! so a user (or a future incremental pipeline) knows the cheapest action. Pure — operates on the raw
//! HJSON values, so it also catches unknown/forward-compat keys.

use serde_json::Value;
use std::collections::BTreeMap;

/// What re-work a changed field forces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditClass {
    /// A finished-image operation — recolour tint / re-symmetrise. No sampler, no re-raster.
    Post,
    /// Re-place the ornament onto a new page canvas (new size / DPI / margins).
    Reraster,
    /// A full re-render (the sampler / the binariser threshold / the geometry all change).
    Regen,
}

impl EditClass {
    pub fn label(self) -> &'static str {
        match self {
            EditClass::Post => "post",
            EditClass::Reraster => "re-raster",
            EditClass::Regen => "re-gen",
        }
    }
    fn rank(self) -> u8 {
        match self {
            EditClass::Post => 0,
            EditClass::Reraster => 1,
            EditClass::Regen => 2,
        }
    }
}

/// Classify a dotted field path into the cheapest edit that realises a change to it.
pub fn classify(path: &str) -> EditClass {
    // strip any array index suffix for matching (`motif[0]` → `motif`)
    let head = path.split('[').next().unwrap_or(path);
    match head {
        "ink.color" | "output.tint" | "ornament.symmetry" => EditClass::Post,
        p if p.starts_with("page") => EditClass::Reraster,
        // everything else — origin/technique/motif/prompt/type/tier/frame/ink.weight/transparency/
        // fade/taper/glyph, transparent — needs a fresh render.
        _ => EditClass::Regen,
    }
}

/// A classified change between two specs.
#[derive(Debug, Clone, PartialEq)]
pub struct Change {
    pub path: String,
    pub old: Option<String>,
    pub new: Option<String>,
    pub class: EditClass,
}

/// Flatten a JSON value to `dotted.path -> scalar-string` leaves.
fn flatten(v: &Value, prefix: &str, out: &mut BTreeMap<String, String>) {
    match v {
        Value::Object(m) => {
            for (k, val) in m {
                let p = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                flatten(val, &p, out);
            }
        }
        Value::Array(a) => {
            for (i, val) in a.iter().enumerate() {
                flatten(val, &format!("{prefix}[{i}]"), out);
            }
        }
        leaf => {
            out.insert(prefix.to_string(), leaf.to_string());
        }
    }
}

/// The classified diff between two specs (as parsed HJSON values). Sorted by path.
pub fn diff(old: &Value, new: &Value) -> Vec<Change> {
    let (mut a, mut b) = (BTreeMap::new(), BTreeMap::new());
    flatten(old, "", &mut a);
    flatten(new, "", &mut b);
    let mut paths: Vec<&String> = a.keys().chain(b.keys()).collect();
    paths.sort();
    paths.dedup();
    paths
        .into_iter()
        .filter_map(|p| {
            let (ov, nv) = (a.get(p), b.get(p));
            (ov != nv).then(|| Change { path: p.clone(), old: ov.cloned(), new: nv.cloned(), class: classify(p) })
        })
        .collect()
}

/// The most-expensive action any change requires (what a re-run must do overall).
pub fn worst(changes: &[Change]) -> Option<EditClass> {
    changes.iter().map(|c| c.class).max_by_key(|c| c.rank())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn val(s: &str) -> Value {
        deser_hjson::from_str(s).unwrap()
    }

    #[test]
    fn classify_paths() {
        assert_eq!(classify("output.tint"), EditClass::Post);
        assert_eq!(classify("ornament.symmetry"), EditClass::Post);
        assert_eq!(classify("page.size"), EditClass::Reraster);
        assert_eq!(classify("page.dpi"), EditClass::Reraster);
        assert_eq!(classify("origin"), EditClass::Regen);
        assert_eq!(classify("ink.weight"), EditClass::Regen);
        assert_eq!(classify("motif[0]"), EditClass::Regen);
    }

    #[test]
    fn diff_finds_and_classifies_changes() {
        let old = val(r#"{"origin":"russian","output":{"tint":"black"},"page":{"size":"a5"}}"#);
        let new = val(r#"{"origin":"english","output":{"tint":"sepia"},"page":{"size":"a4"}}"#);
        let d = diff(&old, &new);
        let by = |p: &str| d.iter().find(|c| c.path == p).unwrap().class;
        assert_eq!(by("origin"), EditClass::Regen);
        assert_eq!(by("output.tint"), EditClass::Post);
        assert_eq!(by("page.size"), EditClass::Reraster);
        assert_eq!(worst(&d), Some(EditClass::Regen));
    }

    #[test]
    fn tint_only_change_is_post() {
        let old = val(r#"{"origin":"russian","output":{"tint":"black"}}"#);
        let new = val(r#"{"origin":"russian","output":{"tint":"sepia"}}"#);
        let d = diff(&old, &new);
        assert_eq!(d.len(), 1);
        assert_eq!(worst(&d), Some(EditClass::Post));
    }
}
