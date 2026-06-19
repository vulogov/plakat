//! Two-namespace SHA-256 disk cache for `compile` LLM calls (opt-in via
//! `--compile-cache`). `positive/` keys on (provider, model, system, input);
//! `negative/` keys on (provider, model, system, **enhanced positive**, seeds),
//! so changing the positive prompt correctly invalidates the negative entry.
//! Sits next to the enhance cache (`…/plakat/compile/<ns>/`).

use sha2::{Digest, Sha256};
use std::path::PathBuf;

pub const POSITIVE: &str = "positive";
pub const NEGATIVE: &str = "negative";

fn root(namespace: &str) -> PathBuf {
    crate::llm::cache::cache_dir()
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
        .join("compile")
        .join(namespace)
}

/// SHA-256 hex over length-prefixed parts (length-prefixing prevents
/// `"ab"+"c"` colliding with `"a"+"bc"`).
pub fn key(parts: &[&str]) -> String {
    let mut h = Sha256::new();
    for p in parts {
        h.update((p.len() as u64).to_le_bytes());
        h.update(p.as_bytes());
    }
    let bytes = h.finalize();
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Cache hit → the stored text; miss / I/O error → None (a broken cache falls
/// through to a fresh call rather than crashing).
pub fn lookup(namespace: &str, key: &str) -> Option<String> {
    std::fs::read_to_string(root(namespace).join(format!("{key}.txt"))).ok()
}

/// Best-effort store (atomic-ish: write then rename). Errors are swallowed — a
/// cache write failure must not fail the compile.
pub fn store(namespace: &str, key: &str, value: &str) {
    let dir = root(namespace);
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let final_path = dir.join(format!("{key}.txt"));
    let tmp = dir.join(format!("{key}.tmp"));
    if std::fs::write(&tmp, value).is_ok() {
        let _ = std::fs::rename(&tmp, &final_path);
    }
}

/// Delete one namespace (`positive`/`negative`) or both (`None`). Returns the
/// number of `.txt` entries removed.
pub fn clear(namespace: Option<&str>) -> usize {
    let nss: &[&str] = match namespace {
        Some(POSITIVE) => &[POSITIVE],
        Some(NEGATIVE) => &[NEGATIVE],
        _ => &[POSITIVE, NEGATIVE],
    };
    let mut n = 0;
    for ns in nss {
        if let Ok(entries) = std::fs::read_dir(root(ns)) {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) == Some("txt") && std::fs::remove_file(&p).is_ok() {
                    n += 1;
                }
            }
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_stable_and_length_prefixed() {
        assert_eq!(key(&["a", "b"]), key(&["a", "b"]));
        assert_ne!(key(&["ab", "c"]), key(&["a", "bc"]), "no boundary collision");
        assert_eq!(key(&["x"]).len(), 64);
    }
}
