//! Presence heartbeats for shared libraries (RFC PHOTOS 3.10). Each running `plakat photos` writes a
//! tiny JSON heartbeat under `<root>/.plakat_presence/`, refreshed periodically and removed on exit.
//! Reading the directory (and ignoring stale entries) tells you which instances are live on the same
//! library — so `:who` can list your collaborators and the status bar can show a count.
//!
//! Fully offline and best-effort: a crashed instance's heartbeat simply ages out (its `at` timestamp
//! falls outside the freshness window). Never an error path that blocks editing.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// How long (seconds) a heartbeat is considered live after its last refresh.
pub const TTL_SECS: u64 = 90;

/// One instance's heartbeat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Presence {
    pub who: String,
    pub pid: u32,
    /// Unix epoch seconds of the last refresh.
    pub at: u64,
    /// The album this instance currently has open (relative to the library root), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub album: Option<String>,
}

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The presence directory under a library root.
pub fn dir(root: &Path) -> PathBuf {
    root.join(".plakat_presence")
}

/// A stable per-instance filename (`<pid>-<sanitized who>.json`).
fn file_name(who: &str, pid: u32) -> String {
    let safe: String = who.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    format!("{pid}-{safe}.json")
}

/// Write / refresh this instance's heartbeat (with the open album, relative to root). Best-effort.
pub fn heartbeat(root: &Path, who: &str, pid: u32, album: Option<String>) {
    let d = dir(root);
    if std::fs::create_dir_all(&d).is_err() {
        return;
    }
    let p = Presence { who: who.to_string(), pid, at: now_epoch(), album };
    if let Ok(json) = serde_json::to_string(&p) {
        let _ = std::fs::write(d.join(file_name(who, pid)), json);
    }
}

/// Remove this instance's heartbeat (call on exit). Best-effort.
pub fn depart(root: &Path, who: &str, pid: u32) {
    let _ = std::fs::remove_file(dir(root).join(file_name(who, pid)));
}

/// The live instances (heartbeats refreshed within [`TTL_SECS`]), most-recent first. Stale entries
/// are skipped (and opportunistically removed).
pub fn live(root: &Path) -> Vec<Presence> {
    let d = dir(root);
    let now = now_epoch();
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(&d) else { return out };
    for e in rd.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("json") {
            continue;
        }
        match std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<Presence>(&t).ok()) {
            Some(p) if now.saturating_sub(p.at) <= TTL_SECS => out.push(p),
            _ => {
                let _ = std::fs::remove_file(&path); // stale or unparseable → clean up
            }
        }
    }
    out.sort_by(|a, b| b.at.cmp(&a.at));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heartbeat_shows_up_live_and_departs() {
        let root = std::env::temp_dir().join(format!("plakat-presence-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();

        heartbeat(&root, "alice@box", 111, Some("Iceland".into()));
        heartbeat(&root, "bob@box", 222, None);
        let live_now = live(&root);
        let names: Vec<String> = live_now.iter().map(|p| p.who.clone()).collect();
        assert!(names.iter().any(|n| n == "alice@box") && names.iter().any(|n| n == "bob@box"), "both live: {names:?}");
        assert_eq!(
            live_now.iter().find(|p| p.who == "alice@box").unwrap().album.as_deref(),
            Some("Iceland"),
            "album context carried"
        );

        depart(&root, "alice@box", 111);
        let after: Vec<String> = live(&root).into_iter().map(|p| p.who).collect();
        assert_eq!(after, vec!["bob@box".to_string()], "alice departed");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn stale_heartbeats_age_out() {
        let root = std::env::temp_dir().join(format!("plakat-presence-stale-{}", std::process::id()));
        let d = dir(&root);
        std::fs::create_dir_all(&d).unwrap();
        // Hand-write a heartbeat well outside the TTL.
        let old = Presence { who: "ghost@box".into(), pid: 9, at: now_epoch().saturating_sub(TTL_SECS + 60), album: None };
        std::fs::write(d.join(file_name("ghost@box", 9)), serde_json::to_string(&old).unwrap()).unwrap();
        assert!(live(&root).is_empty(), "stale entry is not live");
        let _ = std::fs::remove_dir_all(&root);
    }
}
