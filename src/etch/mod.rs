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
pub mod fingerprint;
pub mod manifest;
pub mod payload;
pub mod pixel;

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
    fn write_read_end_to_end_l0_and_l1() {
        // Owns the process-global config (the only test that enables it). A 256² image so L1 survives too.
        set_config(EtchConfig { enabled: true, key: "testkey".into(), id_override: None, layers: vec![], strength: 0.35, db: None });
        let (w, h) = (512u32, 512u32); // native == canonical grid so L1 is exact (real gens are ≥512)
        let meta = GenerationMetadata::new("a red poster", "sdxl", 7, 30, 7.0, "ddim", w, h);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.png");
        // a mildly textured image (not flat — DCT needs some signal).
        let buf: Vec<u8> = (0..(w * h) as usize)
            .flat_map(|i| {
                let g = (120 + (i % 37)) as u8;
                [g, g.saturating_sub(4), g.saturating_add(3)]
            })
            .collect();
        crate::imaging::io::save_rgb_u8_with_metadata(&buf, w, h, &path, &meta).unwrap();

        let report = detect::verify(&path, "testkey", false);
        assert_eq!(report.verdict, detect::Verdict::Generated);
        let expected = payload::derive_id("testkey", &manifest::canonical_manifest(&meta));
        assert_eq!(report.id, Some(expected), "recovered id == derived id (L0)");
        // L0 sidecar carries the etch object (plakat's `<image>.png.json` convention).
        let side = std::fs::read_to_string(crate::imaging::io::sidecar_path(&path)).unwrap();
        assert!(side.contains("\"etch\"") && side.contains(&expected.hex()));
        // L1 pixel etch also survived the PNG round-trip and decodes the SAME id.
        assert_eq!(report.l1.state, "present", "L1 recovered: {}", report.l1.detail);
    }
}

/// The layers this build implements AND the config wants. Grows per phase (now L0 + L1 + L3).
fn active_layers(cfg: &EtchConfig) -> Vec<Layer> {
    [Layer::L0, Layer::L1, Layer::L3].into_iter().filter(|l| cfg.wants(*l)).collect()
}

// L3 (6.7.0 Phase 3): images are enqueued at save time (sync) and fingerprinted in one batch at the end
// of the run — so CLIP loads at most once, and the sync save path stays sync. Gated on the encoder being
// cached (never a surprise multi-GB download during a render).
static L3_QUEUE: RwLock<Vec<(PathBuf, EtchId)>> = RwLock::new(Vec::new());

/// Enqueue a just-saved image for L3 fingerprinting, if etching is on and L3 is requested + a db is set.
/// Sync + cheap (records a path); the CLIP embed happens later in [`l3_flush`].
pub fn l3_enqueue(path: &std::path::Path, metadata: &crate::imaging::metadata::GenerationMetadata) {
    let Some(cfg) = active() else { return };
    if !active_layers(cfg).contains(&Layer::L3) || cfg.db.is_none() {
        return;
    }
    let id = render_id(cfg, metadata);
    if let Ok(mut q) = L3_QUEUE.write() {
        q.push((path.to_path_buf(), id));
    }
}

/// Fingerprint every enqueued image into the store — called once at the end of the run. Best-effort;
/// skipped (with a notice) if the CLIP encoder isn't already cached. Loads CLIP once for the batch.
pub async fn l3_flush(device: &candle_core::Device) {
    let batch: Vec<(PathBuf, EtchId)> = match L3_QUEUE.write() {
        Ok(mut q) => std::mem::take(&mut *q),
        Err(_) => return,
    };
    if batch.is_empty() {
        return;
    }
    let Some(db) = active().and_then(|c| c.db.clone()) else { return };
    if !crate::pipelines::clip_embed::ClipEmbedder::is_cached() {
        tracing::info!(target: "plakat", "etch L3: CLIP encoder not cached — {} image(s) not fingerprinted (L0/L1 still written; run a CLIP feature once to enable)", batch.len());
        return;
    }
    let embedder = match crate::pipelines::clip_embed::ClipEmbedder::load(device).await {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(target: "plakat", "etch L3: CLIP load failed: {e:#}");
            return;
        }
    };
    let store = match fingerprint::Store::open(&db) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(target: "plakat", "etch L3: store open failed: {e:#}");
            return;
        }
    };
    let mut stored = 0;
    for (path, id) in &batch {
        match embedder.embed_image(path).and_then(|v| store.add(*id, &v)) {
            Ok(()) => stored += 1,
            Err(e) => tracing::warn!(target: "plakat", "etch L3: {} → {e:#}", path.display()),
        }
    }
    tracing::info!(target: "plakat", "etch L3: fingerprinted {stored} image(s) → {}", db.display());
}

/// The verifier's key (from `--etch-key`), else the public constant — for reading L1/L3 back.
pub fn effective_key() -> String {
    CONFIG.get().map(|c| c.key.clone()).unwrap_or_else(|| PUBLIC_KEY.to_string())
}

/// The L3 store to query on verify (from `--etch-db`, else the default), regardless of `--etch` enabled.
pub fn effective_db() -> Option<PathBuf> {
    match CONFIG.get() {
        Some(c) => c.db.clone(),
        None => default_db(),
    }
}

/// L1 (6.7.0 Phase 2): embed the pixel etch into an RGB buffer, if etching is on and L1 is requested.
/// Returns the marked buffer (the caller writes it). `alpha` (0 = transparent) excludes those regions.
pub fn l1_embed_rgb(rgb: &[u8], w: u32, h: u32, alpha: Option<&[u8]>, metadata: &crate::imaging::metadata::GenerationMetadata) -> Option<Vec<u8>> {
    let cfg = active()?;
    if !active_layers(cfg).contains(&Layer::L1) {
        return None;
    }
    let id = render_id(cfg, metadata);
    Some(pixel::embed(rgb, w as usize, h as usize, id, &cfg.key, cfg.strength, alpha))
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
