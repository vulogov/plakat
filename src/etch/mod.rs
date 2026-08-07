//! Provenance etching (RFC ETCH-1). `--etch` writes a 64-bit [`EtchId`] into every image plakat produces
//! by four independent evidence layers (L0 manifest · L1 pixel · L2 latent · L3 fingerprint), and
//! `plakat doctor --if-plakat` reads whatever survived into a *graded* verdict — not a boolean.
//!
//! ETCH does NOT promise bit recovery through high-strength regeneration (a denoiser removes an
//! off-manifold mark as a side effect of working); it promises graded attribution that degrades
//! `exact id → generated → probable derivative → no evidence` rather than off a cliff. See RFC ETCH-1.
//!
//! This module is **always compiled** (not behind a cargo feature) so the `--no-default-features` CI gate
//! covers it; the runtime `--etch` flag (default OFF) is the opt-in.

pub mod detect;
pub mod manifest;
pub mod payload;

use std::path::PathBuf;
use std::sync::{OnceLock, RwLock};

/// A 64-bit provenance identifier (rendered as 16 lowercase hex nibbles).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EtchId(pub u64);

impl EtchId {
    pub fn hex(&self) -> String {
        format!("{:016x}", self.0)
    }
    pub fn parse_hex(s: &str) -> Option<EtchId> {
        let s = s.trim().trim_start_matches("0x");
        if s.len() == 16 {
            u64::from_str_radix(s, 16).ok().map(EtchId)
        } else {
            None
        }
    }
}

/// Which evidence layers to write (default: all applicable to the command).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    L0,
    L1,
    L2,
    L3,
}

impl Layer {
    pub fn parse_list(s: &str) -> Vec<Layer> {
        s.split(',')
            .filter_map(|t| match t.trim().to_ascii_lowercase().as_str() {
                "l0" => Some(Layer::L0),
                "l1" => Some(Layer::L1),
                "l2" => Some(Layer::L2),
                "l3" => Some(Layer::L3),
                _ => None,
            })
            .collect()
    }
}

/// Runtime etch configuration, built once from the global `--etch*` flags.
#[derive(Debug, Clone)]
pub struct EtchConfig {
    pub enabled: bool,
    /// Key for `EtchId` derivation + carrier PRNG (public constant by default).
    pub key: String,
    /// Explicit `EtchId` override (`--etch-id`).
    pub id_override: Option<EtchId>,
    /// Requested layers; empty = all applicable.
    pub layers: Vec<Layer>,
    /// L1 embedding strength `0..=1`.
    pub strength: f32,
    /// L3 fingerprint store (`None` = disabled / `--etch-db none`).
    pub db: Option<PathBuf>,
}

/// The default public key — a published constant so any stock plakat build can verify. (Public key ⇒ the
/// carrier is public and subtractable; this is a provenance signal, not a defence against a remover.)
pub const PUBLIC_KEY: &str = "plakat-etch-public-v1";

impl Default for EtchConfig {
    fn default() -> Self {
        Self { enabled: false, key: PUBLIC_KEY.to_string(), id_override: None, layers: Vec::new(), strength: 0.35, db: None }
    }
}

impl EtchConfig {
    /// Is this layer requested? (empty layer list = all applicable.)
    pub fn wants(&self, l: Layer) -> bool {
        self.layers.is_empty() || self.layers.contains(&l)
    }
}

/// The default L3 store: `$PLAKAT_HOME/etchdb` (or `~/.plakat/etchdb`). `None` if no home is resolvable.
pub fn default_db() -> Option<PathBuf> {
    if let Ok(h) = std::env::var("PLAKAT_HOME") {
        return Some(PathBuf::from(h).join("etchdb"));
    }
    dirs_home().map(|h| h.join(".plakat").join("etchdb"))
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from)
}

static CONFIG: OnceLock<EtchConfig> = OnceLock::new();
static PARENT: RwLock<Option<EtchId>> = RwLock::new(None);

/// Install the process-wide etch config (from the CLI globals). First write wins.
pub fn set_config(cfg: EtchConfig) {
    let _ = CONFIG.set(cfg);
}

/// The active config *when etching is enabled*, else `None`.
pub fn active() -> Option<&'static EtchConfig> {
    CONFIG.get().filter(|c| c.enabled)
}

/// Record the source image's `EtchId` so a plakat-internal derivation (img2img/outpaint/relight/…) can
/// chain it into the output's `parent` (RFC L0). Cleared with `None`.
pub fn set_parent(id: Option<EtchId>) {
    if let Ok(mut p) = PARENT.write() {
        *p = id;
    }
}

/// The current derivation parent, if any.
pub fn parent() -> Option<EtchId> {
    PARENT.read().ok().and_then(|p| *p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imaging::metadata::GenerationMetadata;

    #[test]
    fn etch_id_hex_roundtrips() {
        let id = EtchId(0x9f2c4a17b3e08d5c);
        assert_eq!(id.hex(), "9f2c4a17b3e08d5c");
        assert_eq!(EtchId::parse_hex("9f2c4a17b3e08d5c"), Some(id));
        assert_eq!(EtchId::parse_hex("0x9f2c4a17b3e08d5c"), Some(id));
        assert_eq!(EtchId::parse_hex("nothex"), None);
    }

    #[test]
    fn layer_list_parses() {
        assert_eq!(Layer::parse_list("l0, L2 ,l3"), vec![Layer::L0, Layer::L2, Layer::L3]);
    }

    // The full write→read loop: enabling etch writes the L0 chunk + sidecar; `detect` recovers it. This
    // test owns the process-global config (the only test that enables it).
    #[test]
    fn l0_write_read_end_to_end() {
        set_config(EtchConfig { enabled: true, key: "testkey".into(), id_override: None, layers: vec![], strength: 0.35, db: None });
        let meta = GenerationMetadata::new("a red poster", "sdxl", 7, 30, 7.0, "ddim", 4, 4);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.png");
        let buf = vec![128u8; 4 * 4 * 3];
        crate::imaging::io::save_rgb_u8_with_metadata(&buf, 4, 4, &path, &meta).unwrap();

        let report = detect::verify(&path, false);
        assert_eq!(report.verdict, detect::Verdict::Generated);
        let expected = payload::derive_id("testkey", &manifest::canonical_manifest(&meta));
        assert_eq!(report.id, Some(expected), "recovered id == derived id");
        // sidecar carries the etch object too (plakat's `<image>.png.json` convention).
        let side = std::fs::read_to_string(crate::imaging::io::sidecar_path(&path)).unwrap();
        assert!(side.contains("\"etch\""), "sidecar has the etch object");
        assert!(side.contains(&expected.hex()));
    }
}

/// The layers this build implements AND the config wants. Grows per phase (Phase 1: L0 only).
fn active_layers(cfg: &EtchConfig) -> Vec<Layer> {
    [Layer::L0].into_iter().filter(|l| cfg.wants(*l)).collect()
}

/// The `EtchId` for this render — the `--etch-id` override, else derived from the recipe + key.
pub fn render_id(cfg: &EtchConfig, metadata: &crate::imaging::metadata::GenerationMetadata) -> EtchId {
    cfg.id_override.unwrap_or_else(|| payload::derive_id(&cfg.key, &manifest::canonical_manifest(metadata)))
}

/// The L0 `etch` manifest JSON for this render, if etching is enabled and L0 is requested. Called at save
/// time (the `imaging::io` hook) — written into the PNG `tEXt` chunk + the JSON sidecar.
pub fn l0_manifest_json(metadata: &crate::imaging::metadata::GenerationMetadata) -> Option<String> {
    let cfg = active()?;
    let layers = active_layers(cfg);
    if !layers.contains(&Layer::L0) {
        return None;
    }
    let id = render_id(cfg, metadata);
    Some(manifest::EtchManifest::new(id, &layers, parent()).to_json())
}
