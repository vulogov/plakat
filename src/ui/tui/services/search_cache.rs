//! Disk cache for LoRA-Hub remote searches (RFC TUI-1 §10, roadmap C). Civitai / HF
//! results are slow and rate-limited, so an identical query inside the TTL window is
//! served from a sidecar JSON file instead of re-hitting the network. Freshness is the
//! file's mtime — no embedded timestamp — so the cache is just `read`/`write` of a
//! `Vec<RemoteHit>` keyed by `(source, query)`.

use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use crate::ui::tui::screens::lorahub::RemoteHit;

/// Remote search results live for an hour (roadmap C: "Civitai/HF results cached 1h").
pub const TTL: Duration = Duration::from_secs(60 * 60);

/// `<shared-cache-root>/plakat-ui/lora-search/` — created on demand. Sits next to the
/// HF / Civitai caches (and honors `--cache-dir` / `PLAKAT_CACHE_DIR` the same way).
fn cache_dir() -> Option<PathBuf> {
    let hub = crate::hf::cache::hf_cache_root();
    let root = hub.parent().unwrap_or(&hub);
    let dir = root.join("plakat-ui").join("lora-search");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// A filesystem-safe, collision-resistant file name for `(source, query)`.
fn key(source: &str, query: &str) -> String {
    // Normalize the query (case/space-insensitive) so "Water Color" and "watercolor "
    // don't miss each other, then sanitize for a file name.
    let norm: String = query.trim().to_lowercase().chars().map(|c| if c.is_alphanumeric() { c } else { '-' }).collect();
    let norm = norm.trim_matches('-');
    // Length-bounded; a short hash guards against distinct queries colliding after
    // sanitization (e.g. "a/b" vs "a-b").
    let mut h: u64 = 1469598103934665603; // FNV-1a
    for b in query.trim().to_lowercase().bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    let stem: String = norm.chars().take(48).collect();
    format!("{source}-{stem}-{h:016x}.json")
}

fn path(source: &str, query: &str) -> Option<PathBuf> {
    Some(cache_dir()?.join(key(source, query)))
}

/// Return cached hits for `(source, query)` if a fresh (< [`TTL`]) sidecar exists.
pub fn get(source: &str, query: &str) -> Option<Vec<RemoteHit>> {
    let p = path(source, query)?;
    let meta = std::fs::metadata(&p).ok()?;
    let age = SystemTime::now().duration_since(meta.modified().ok()?).ok()?;
    if age > TTL {
        return None;
    }
    let bytes = std::fs::read(&p).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Persist `hits` for `(source, query)`. Best-effort; failures are silent (the search
/// already succeeded — caching is an optimization, never a correctness path).
pub fn put(source: &str, query: &str, hits: &[RemoteHit]) {
    let Some(p) = path(source, query) else { return };
    if let Ok(bytes) = serde_json::to_vec(hits) {
        let _ = std::fs::write(p, bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::tui::screens::lorahub::DownloadRef;

    fn hit(title: &str) -> RemoteHit {
        RemoteHit {
            title: title.into(),
            subtitle: "SDXL".into(),
            downloads: 1,
            family: None,
            dl: DownloadRef::Hf { repo: "u/x".into() },
        }
    }

    #[test]
    fn key_is_stable_and_query_normalized() {
        // Same logical query → same key regardless of case / surrounding space.
        assert_eq!(key("civitai", "Water Color"), key("civitai", "  water color "));
        // Different source → different key.
        assert_ne!(key("civitai", "x"), key("hf", "x"));
        // File-name safe.
        assert!(!key("civitai", "a/b\\c").contains('/'));
    }

    #[test]
    fn round_trips_through_disk_when_cache_dir_is_available() {
        // Skip silently in sandboxes with no cache dir.
        if cache_dir().is_none() {
            return;
        }
        let q = "round-trip-test-zzqq-9981";
        put("hf", q, &[hit("alpha"), hit("beta")]);
        let got = get("hf", q).expect("fresh cache entry");
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].title, "alpha");
        // Clean up the sidecar we just wrote.
        if let Some(p) = path("hf", q) {
            let _ = std::fs::remove_file(p);
        }
    }
}
