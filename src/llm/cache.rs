//! On-disk cache for prompt-enhancer outputs. Keyed by a SHA-256
//! over `(model_alias, system_prompt, user_prompt, temperature,
//! max_new_tokens)` — every input that materially affects the
//! generated text. Cache hits skip the LLM forward entirely;
//! cache misses run the model and write the result on success
//! (refusals + empty outputs are not cached).
//!
//! Why opt-in via `--enhance-cache` rather than default-on:
//! greedy decoding is already reproducible, but users iterating
//! on the system prompt or model alias would otherwise see stale
//! hits from a previous run. Opt-in keeps the developer-loop
//! ergonomics predictable. Scenarios that re-enhance the same
//! prompts dozens of times across edits benefit; one-off
//! invocations don't pay the disk-write cost.
//!
//! Cache directory: `${PLAKAT_CACHE_DIR or HF_HOME or
//! ~/.cache/huggingface}/../plakat/enhance/`. We deliberately sit
//! alongside the HF cache rather than inside it — `plakat models
//! rm` shouldn't accidentally evict enhanced-prompt cache entries.
//!
//! Cache entries are plain UTF-8 text (the rewritten prompt), one
//! per file. Atomic write via tempfile-then-rename so a crashed
//! process can't leave a partial entry.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Inputs that participate in the cache key. Any field change
/// invalidates the cache entry — change the system prompt, change
/// the model alias, or change temperature, and the next call
/// becomes a cache miss.
pub struct CacheKey<'a> {
    pub alias: &'a str,
    pub system: &'a str,
    pub user: &'a str,
    pub temperature: f64,
    pub max_new_tokens: usize,
}

impl<'a> CacheKey<'a> {
    /// Hex-encoded SHA-256 of the key inputs joined by
    /// length-prefixed framing — `<u64 LE len><bytes>` per field —
    /// so that `\0` (or any other byte) appearing inside (e.g.)
    /// the user prompt can't collide with a re-arrangement of
    /// field contents that would happen to produce the same
    /// concatenation if a plain delimiter were used.
    pub fn digest(&self) -> String {
        let mut hasher = Sha256::new();
        let temp_str = format!("{:.6}", self.temperature);
        let max_str = format!("{}", self.max_new_tokens);
        for field in [
            self.alias.as_bytes(),
            self.system.as_bytes(),
            self.user.as_bytes(),
            temp_str.as_bytes(),
            max_str.as_bytes(),
        ] {
            hasher.update((field.len() as u64).to_le_bytes());
            hasher.update(field);
        }
        hex(&hasher.finalize())
    }
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

/// Resolve the cache directory. Sits alongside (not inside) the HF
/// cache so `plakat models rm` doesn't evict enhanced prompts.
pub fn cache_dir() -> PathBuf {
    let hf_root = crate::hf::cache::hf_cache_root();
    // hf_cache_root() typically returns `<base>/huggingface/hub`;
    // we want a sibling `plakat/enhance` dir. Go up two levels.
    let plakat_root = hf_root
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("plakat"))
        .unwrap_or_else(|| {
            // Conservative fallback: $XDG_CACHE_HOME (or ~/.cache)
            // + /plakat. Same layout `dirs::cache_dir()` would
            // return on Linux; spelled out to avoid the extra
            // dep just for one path lookup.
            let xdg = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from);
            let home_cache = std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join(".cache"));
            xdg.or(home_cache)
                .unwrap_or_else(|| PathBuf::from("."))
                .join("plakat")
        });
    plakat_root.join("enhance")
}

/// Look up a cache entry. Returns the cached text on hit, `None`
/// on miss (including I/O errors — a broken cache should fall
/// through to a fresh enhance, not crash).
pub fn lookup(key: &CacheKey) -> Option<String> {
    let path = cache_dir().join(format!("{}.txt", key.digest()));
    if !path.exists() {
        return None;
    }
    std::fs::read_to_string(&path).ok().and_then(|s| {
        let trimmed = s.trim_end_matches('\n').to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}

/// Store an enhance result. Best-effort: failures (e.g.
/// permission-denied on the cache dir) emit a warn log and the
/// caller continues with the in-memory result. Never bubble a
/// cache-write error up to the user.
pub fn store(key: &CacheKey, enhanced: &str) -> Result<()> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating cache dir {}", dir.display()))?;
    let final_path = dir.join(format!("{}.txt", key.digest()));
    // Tempfile-then-rename for atomicity. tempfile guarantees a
    // unique name in the target directory.
    let mut tmp = tempfile::NamedTempFile::new_in(&dir)
        .with_context(|| format!("opening tempfile in {}", dir.display()))?;
    use std::io::Write;
    tmp.write_all(enhanced.as_bytes())
        .with_context(|| "writing cache contents")?;
    tmp.write_all(b"\n")?;
    tmp.persist(&final_path)
        .map_err(|e| anyhow::anyhow!("persisting cache entry: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_changes_with_each_field() {
        let base = CacheKey {
            alias: "qwen2.5-1.5b",
            system: "rewrite prompts",
            user: "a fox",
            temperature: 0.0,
            max_new_tokens: 96,
        };
        let d0 = base.digest();

        let alt_alias = CacheKey {
            alias: "smollm2-360m",
            ..base
        };
        let alt_system = CacheKey {
            system: "different system",
            ..base
        };
        let alt_user = CacheKey {
            user: "a cat",
            ..base
        };
        let alt_temp = CacheKey {
            temperature: 0.5,
            ..base
        };
        let alt_tokens = CacheKey {
            max_new_tokens: 64,
            ..base
        };
        for other in [
            alt_alias.digest(),
            alt_system.digest(),
            alt_user.digest(),
            alt_temp.digest(),
            alt_tokens.digest(),
        ] {
            assert_ne!(d0, other);
        }
    }

    #[test]
    fn digest_is_deterministic() {
        let key = CacheKey {
            alias: "qwen2.5-1.5b",
            system: "S",
            user: "U",
            temperature: 0.7,
            max_new_tokens: 96,
        };
        let a = key.digest();
        let b = key.digest();
        assert_eq!(a, b);
        // SHA-256 hex output is 64 chars.
        assert_eq!(a.len(), 64);
        // Round-trip parse: all hex digits.
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn digest_handles_null_byte_inputs_safely() {
        // User prompts containing the internal \0 delimiter
        // mustn't collide with split fields. Two prompts that
        // would tokenize identically if naively joined still
        // produce distinct digests.
        let a = CacheKey {
            alias: "m",
            system: "s\0extra",
            user: "u",
            temperature: 0.0,
            max_new_tokens: 96,
        };
        let b = CacheKey {
            alias: "m",
            system: "s",
            user: "extra\0u",
            temperature: 0.0,
            max_new_tokens: 96,
        };
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn round_trip_store_then_lookup() {
        // `hf::cache::set_override` is OnceLock-backed — other tests
        // in the same `cargo test` process may have locked in a
        // different override before this test runs, so we can't
        // reliably point the cache_dir at a tempdir. Instead, use
        // an absurdly-unique key so the on-disk slot can't be
        // shared with another test's leftover state, then clean
        // up after.
        let unique_user = format!(
            "round-trip-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let key = CacheKey {
            alias: "qwen2.5-1.5b",
            system: "S",
            user: &unique_user,
            temperature: 0.0,
            max_new_tokens: 96,
        };
        // Miss initially (digest is fresh; no leftover file).
        assert!(lookup(&key).is_none());
        store(&key, "enhanced text here").unwrap();
        assert_eq!(lookup(&key).as_deref(), Some("enhanced text here"));
        // Clean up so a stale file doesn't accumulate in the cache.
        let path = cache_dir().join(format!("{}.txt", key.digest()));
        let _ = std::fs::remove_file(&path);
    }
}
