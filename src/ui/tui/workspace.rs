//! Workspace — the working context for `plakat ui` (RFC TUI-1 §3).
//!
//! A workspace is a directory marked by `plakat-workspace.hjson`; every screen
//! resolves paths relative to its root. `plakat ui` finds the nearest workspace
//! at/above the cwd (or honours `--workspace`), and *always* creates one (running
//! a tiny wizard) when none is found — there is no degraded mode. The pure parts
//! (marker search, config (de)serialization, directory + `.gitignore` creation)
//! are unit-tested; the interactive wizard is a thin I/O layer over them.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

/// The file whose presence marks a directory as a workspace root.
pub const MARKER: &str = "plakat-workspace.hjson";

/// Subdirectories created for a fresh workspace.
const DIRS: &[&str] = &[
    "people", "scenarios", "loras", "refs", "prompts", "chat", "out", ".plakat_cache",
];

/// Project config persisted to `plakat-workspace.hjson`. Every field serde-defaults
/// so a partial / hand-trimmed file still loads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceConfig {
    #[serde(default = "d_name")]
    pub name: String,
    #[serde(default)]
    pub created: String,
    #[serde(default = "d_model")]
    pub default_model: String,
    #[serde(default = "d_identity")]
    pub default_identity: String,
    #[serde(default = "d_steps")]
    pub default_steps: usize,
    #[serde(default = "d_guidance")]
    pub default_guidance: f64,
    #[serde(default = "d_size")]
    pub default_size: String,
    #[serde(default = "d_provider")]
    pub layout_provider: String,
    #[serde(default = "d_provider")]
    pub enhancer: String,
    #[serde(default = "d_out")]
    pub out_dir: String,
    #[serde(default = "d_people")]
    pub people_dir: String,
    #[serde(default = "d_scenarios")]
    pub scenarios_dir: String,
    #[serde(default = "d_loras")]
    pub loras_dir: String,
    #[serde(default = "d_prompts")]
    pub prompts_dir: String,
    #[serde(default = "d_chat")]
    pub chat_dir: String,
    #[serde(default)]
    pub global_lora_dirs: Vec<String>,
    #[serde(default = "d_preview")]
    pub preview_every_n_steps: usize,
}

fn d_name() -> String { "Untitled Project".into() }
fn d_model() -> String { "sdxl".into() }
fn d_identity() -> String { "plus-face-sdxl".into() }
fn d_steps() -> usize { 35 }
fn d_guidance() -> f64 { 7.5 }
fn d_size() -> String { "1280x768".into() }
fn d_provider() -> String { "auto".into() }
fn d_out() -> String { "out".into() }
fn d_people() -> String { "people".into() }
fn d_scenarios() -> String { "scenarios".into() }
fn d_loras() -> String { "loras".into() }
fn d_prompts() -> String { "prompts".into() }
fn d_chat() -> String { "chat".into() }
fn d_preview() -> usize { 5 }

impl Default for WorkspaceConfig {
    fn default() -> Self {
        Self {
            name: d_name(), created: String::new(), default_model: d_model(),
            default_identity: d_identity(), default_steps: d_steps(),
            default_guidance: d_guidance(), default_size: d_size(),
            layout_provider: d_provider(), enhancer: d_provider(), out_dir: d_out(),
            people_dir: d_people(), scenarios_dir: d_scenarios(), loras_dir: d_loras(),
            prompts_dir: d_prompts(), chat_dir: d_chat(), global_lora_dirs: Vec::new(),
            preview_every_n_steps: d_preview(),
        }
    }
}

/// A resolved workspace: its root directory + config. Path accessors join the
/// configured (usually default) subdir names onto the root.
#[derive(Debug, Clone)]
pub struct Workspace {
    pub root: PathBuf,
    pub config: WorkspaceConfig,
}

impl Workspace {
    pub fn out_dir(&self) -> PathBuf { self.root.join(&self.config.out_dir) }
    pub fn people_dir(&self) -> PathBuf { self.root.join(&self.config.people_dir) }
    pub fn scenarios_dir(&self) -> PathBuf { self.root.join(&self.config.scenarios_dir) }
    pub fn loras_dir(&self) -> PathBuf { self.root.join(&self.config.loras_dir) }
    pub fn prompts_dir(&self) -> PathBuf { self.root.join(&self.config.prompts_dir) }
    pub fn chat_dir(&self) -> PathBuf { self.root.join(&self.config.chat_dir) }
    pub fn cache_dir(&self) -> PathBuf { self.root.join(".plakat_cache") }

    /// Load a workspace from an existing root (must contain the marker).
    pub fn load(root: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(root.join(MARKER))
            .with_context(|| format!("reading {}", root.join(MARKER).display()))?;
        let config: WorkspaceConfig = deser_hjson::from_str(&text)
            .with_context(|| format!("parsing {MARKER}"))?;
        Ok(Self { root: root.to_path_buf(), config })
    }
}

/// Walk up from `start` to the filesystem root, returning the first directory
/// containing the marker. Pure (no creation).
pub fn find_marker(start: &Path) -> Option<PathBuf> {
    let mut dir: Option<&Path> = Some(start);
    while let Some(d) = dir {
        if d.join(MARKER).is_file() {
            return Some(d.to_path_buf());
        }
        dir = d.parent();
    }
    None
}

/// Create a workspace at `dir`: write the marker, the subdirectories, and a
/// `.gitignore` (only if absent — never clobbers an existing one). Existing files
/// are left untouched (the migration path creates only what's missing). Pure of
/// any prompting — callers build the `config` first.
pub fn create(dir: &Path, config: &WorkspaceConfig) -> Result<Workspace> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating workspace dir {}", dir.display()))?;
    std::fs::write(dir.join(MARKER), workspace_hjson(config))
        .with_context(|| format!("writing {MARKER}"))?;
    for sub in DIRS {
        std::fs::create_dir_all(dir.join(sub))
            .with_context(|| format!("creating {sub}/"))?;
    }
    let gi = dir.join(".gitignore");
    if !gi.exists() {
        std::fs::write(&gi, GITIGNORE).context("writing .gitignore")?;
    }
    Ok(Workspace { root: dir.to_path_buf(), config: config.clone() })
}

/// Heuristic: is this an existing plakat project (scenarios / out / loras) without
/// a workspace marker? Drives the migration prompt.
pub fn detect_existing(dir: &Path) -> Vec<String> {
    let mut found = Vec::new();
    let has_hjson = |sub: &str| {
        std::fs::read_dir(dir.join(sub))
            .map(|rd| rd.flatten().any(|e| e.path().extension().is_some_and(|x| x == "hjson")))
            .unwrap_or(false)
    };
    if has_hjson("scenarios") { found.push("scenarios/".into()); }
    if dir.join("out").is_dir() { found.push("out/".into()); }
    if dir.join("loras").is_dir() { found.push("loras/".into()); }
    found
}

/// Today's date as `YYYY-MM-DD` (via humantime's RFC3339; no chrono dep).
pub fn today() -> String {
    humantime::format_rfc3339_seconds(std::time::SystemTime::now())
        .to_string()
        .chars()
        .take(10)
        .collect()
}

/// Resolve the workspace for this launch (RFC §3): `--workspace` wins (created if
/// it has no marker); else the nearest marker at/above `cwd`; else run the creation
/// wizard in `cwd`. `interactive` gates the stdin prompts (false → silent defaults,
/// for tests / non-tty).
pub fn resolve_or_create(arg: Option<PathBuf>, cwd: &Path, interactive: bool) -> Result<Workspace> {
    if let Some(dir) = arg {
        return match find_marker(&dir) {
            Some(root) if root == dir => Workspace::load(&root),
            _ => wizard(&dir, interactive),
        };
    }
    match find_marker(cwd) {
        Some(root) => Workspace::load(&root),
        None => wizard(cwd, interactive),
    }
}

/// Interactive creation / migration wizard (runs before raw mode). Asks ≤3
/// questions with defaults; on a non-interactive stream it silently takes the
/// defaults so `plakat ui` always ends up with a usable workspace.
fn wizard(dir: &Path, interactive: bool) -> Result<Workspace> {
    let existing = detect_existing(dir);
    let mut cfg = WorkspaceConfig { created: today(), ..Default::default() };
    let default_name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("project")
        .to_string();
    cfg.name = default_name.clone();

    if interactive {
        if !existing.is_empty() {
            eprintln!(
                "plakat ui — No workspace found. Existing plakat files detected:\n  {}\n",
                existing.join("\n  ")
            );
            eprintln!("Adopting this directory as a workspace (existing files unchanged).\n");
        } else {
            eprintln!("plakat ui — No workspace found. Creating one in {}\n", dir.display());
        }
        cfg.name = prompt("Workspace name", &default_name)?;
        cfg.default_model = prompt("Default model", &cfg.default_model)?;
        cfg.layout_provider = prompt("Default LLM provider", &cfg.layout_provider)?;
        cfg.enhancer = cfg.layout_provider.clone();
    }

    let ws = create(dir, &cfg)?;
    if interactive {
        eprintln!("\nWorkspace ready at {}. Launching plakat ui…\n", ws.root.display());
    }
    Ok(ws)
}

/// One stdin prompt with a default (empty input → default).
fn prompt(label: &str, default: &str) -> Result<String> {
    eprint!("{label} [{default}]: ");
    std::io::stderr().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    let trimmed = line.trim();
    Ok(if trimmed.is_empty() { default.to_string() } else { trimmed.to_string() })
}

/// Render the `plakat-workspace.hjson` (a commented HJSON; values interpolated).
fn workspace_hjson(c: &WorkspaceConfig) -> String {
    let globals = if c.global_lora_dirs.is_empty() {
        "[\n    \"~/.plakat/loras\"\n  ]".to_string()
    } else {
        let items = c.global_lora_dirs.iter().map(|d| format!("    \"{d}\"")).collect::<Vec<_>>().join("\n");
        format!("[\n{items}\n  ]")
    };
    format!(
        "{{\n\
        \x20 // plakat workspace — created by `plakat ui` (RFC TUI-1). Edit freely.\n\
        \x20 //\n\
        \x20 // NOTE: downloaded weights are NOT cached here — they use the GLOBAL plakat\n\
        \x20 // cache ($PLAKAT_CACHE_DIR / $HF_HOME / --cache-dir), shared across every\n\
        \x20 // workspace, exactly as the CLI does:\n\
        \x20 //   HuggingFace models  -> <plakat-cache>/hub/\n\
        \x20 //   Civitai LoRAs       -> <plakat-cache>/civitai/\n\
        \x20 // This directory holds only PROJECT files (people, scenarios, prompts, chat,\n\
        \x20 // generated images). `global_lora_dirs` below is for extra loose LoRA files\n\
        \x20 // you keep yourself; the workspace `loras/` dir is for project-specific ones.\n\
        \x20 name: \"{name}\"\n\
        \x20 created: \"{created}\"\n\n\
        \x20 // Default generation settings (override the global config per workspace).\n\
        \x20 default_model: {model}\n\
        \x20 default_identity: {identity}\n\
        \x20 default_steps: {steps}\n\
        \x20 default_guidance: {guidance}\n\
        \x20 default_size: \"{size}\"\n\n\
        \x20 // LLM providers.\n\
        \x20 layout_provider: {provider}\n\
        \x20 enhancer: {enhancer}\n\n\
        \x20 // Directory layout (relative to this file).\n\
        \x20 out_dir: {out}\n\
        \x20 people_dir: {people}\n\
        \x20 scenarios_dir: {scenarios}\n\
        \x20 loras_dir: {loras}\n\
        \x20 prompts_dir: {prompts}\n\
        \x20 chat_dir: {chat}\n\n\
        \x20 // GLOBAL (shared) LoRA dirs, searched in addition to the workspace loras/.\n\
        \x20 // This is your cross-workspace LoRA cache.\n\
        \x20 global_lora_dirs: {globals}\n\n\
        \x20 // Decode an inline preview every N denoise steps.\n\
        \x20 preview_every_n_steps: {preview}\n\
        }}\n",
        name = c.name, created = c.created, model = c.default_model,
        identity = c.default_identity, steps = c.default_steps, guidance = c.default_guidance,
        size = c.default_size, provider = c.layout_provider, enhancer = c.enhancer,
        out = c.out_dir, people = c.people_dir, scenarios = c.scenarios_dir,
        loras = c.loras_dir, prompts = c.prompts_dir, chat = c.chat_dir,
        globals = globals, preview = c.preview_every_n_steps,
    )
}

/// The generated `.gitignore` (RFC §3): cache + generated images + derived
/// encodings excluded; reference photos deliberately left to the user.
const GITIGNORE: &str = "\
# plakat workspace — generated by plakat ui
# Cache (ephemeral, regenerated automatically)
.plakat_cache/

# Generated images (large binaries — exclude from version control)
out/

# Encoding cache (derived from reference photos, regenerated on demand)
people/*/encoding/
people/*/encoding_tests/

# Keep everything else:
#  - plakat-workspace.hjson (project config)
#  - people/*/person.hjson (person definitions)
#  - people/*/refs/ (reference photos — YOU decide whether to commit these)
#  - scenarios/ (scenario HJSON files)
#  - prompts/ (prompt templates)
#  - chat/ (chat session history)
";

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("plakat-ws-test-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn config_round_trips_through_hjson() {
        let cfg = WorkspaceConfig { name: "My Proj".into(), created: "2026-06-26".into(), ..Default::default() };
        let text = workspace_hjson(&cfg);
        let back: WorkspaceConfig = deser_hjson::from_str(&text).unwrap();
        assert_eq!(back.name, "My Proj");
        assert_eq!(back.created, "2026-06-26");
        assert_eq!(back.default_model, "sdxl");
        assert_eq!(back.preview_every_n_steps, 5);
        assert_eq!(back.global_lora_dirs, vec!["~/.plakat/loras".to_string()]);
    }

    #[test]
    fn create_writes_marker_dirs_and_gitignore() {
        let dir = tmp("create");
        let ws = create(&dir, &WorkspaceConfig::default()).unwrap();
        assert!(dir.join(MARKER).is_file());
        for sub in DIRS {
            assert!(dir.join(sub).is_dir(), "{sub} created");
        }
        let gi = std::fs::read_to_string(dir.join(".gitignore")).unwrap();
        assert!(gi.contains(".plakat_cache/") && gi.contains("out/"));
        // the written marker reloads
        let reloaded = Workspace::load(&ws.root).unwrap();
        assert_eq!(reloaded.config.name, ws.config.name);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn create_does_not_clobber_existing_gitignore() {
        let dir = tmp("gitignore");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(".gitignore"), "custom\n").unwrap();
        create(&dir, &WorkspaceConfig::default()).unwrap();
        assert_eq!(std::fs::read_to_string(dir.join(".gitignore")).unwrap(), "custom\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_marker_walks_up_parents() {
        let dir = tmp("walk");
        let nested = dir.join("a/b/c");
        std::fs::create_dir_all(&nested).unwrap();
        create(&dir, &WorkspaceConfig::default()).unwrap();
        assert_eq!(find_marker(&nested), Some(dir.clone()));
        assert_eq!(find_marker(&dir), Some(dir.clone()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_creates_when_absent_non_interactive() {
        let dir = tmp("resolve");
        std::fs::create_dir_all(&dir).unwrap();
        let ws = resolve_or_create(None, &dir, false).unwrap();
        assert_eq!(ws.root, dir);
        assert!(dir.join(MARKER).is_file());
        // second resolve loads the same one (no re-create / no prompt)
        let again = resolve_or_create(None, &dir, false).unwrap();
        assert_eq!(again.config.name, ws.config.name);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn detect_existing_flags_a_plakat_dir() {
        let dir = tmp("migrate");
        std::fs::create_dir_all(dir.join("scenarios")).unwrap();
        std::fs::write(dir.join("scenarios/x.hjson"), "{}").unwrap();
        std::fs::create_dir_all(dir.join("out")).unwrap();
        let found = detect_existing(&dir);
        assert!(found.contains(&"scenarios/".to_string()));
        assert!(found.contains(&"out/".to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn today_is_iso_date() {
        let t = today();
        assert_eq!(t.len(), 10);
        assert_eq!(t.matches('-').count(), 2); // YYYY-MM-DD
    }
}
