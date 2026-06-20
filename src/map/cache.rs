//! SHA-256 disk cache for parsed MapSpecs (`--map-cache`). Keyed on (provider,
//! system, description). Sits at `…/plakat/map/`, next to the enhance + compile
//! caches.

use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn root() -> PathBuf {
    crate::llm::cache::cache_dir()
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
        .join("map")
}

/// SHA-256 hex over length-prefixed parts.
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

pub fn lookup(key: &str) -> Option<String> {
    std::fs::read_to_string(root().join(format!("{key}.json"))).ok()
}

pub fn store(key: &str, value: &str) {
    let dir = root();
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let tmp = dir.join(format!("{key}.tmp"));
    if std::fs::write(&tmp, value).is_ok() {
        let _ = std::fs::rename(&tmp, dir.join(format!("{key}.json")));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_stable_and_length_prefixed() {
        assert_eq!(key(&["a", "b"]), key(&["a", "b"]));
        assert_ne!(key(&["ab", "c"]), key(&["a", "bc"]));
        assert_eq!(key(&["x"]).len(), 64);
    }
}
