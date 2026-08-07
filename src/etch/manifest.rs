//! L0 — the manifest layer: an `etch` object carried in the PNG `tEXt` chunk and the `<base>.json`
//! sidecar. Free, exact when it survives, dies to any metadata strip.

use super::{EtchId, Layer};
use crate::imaging::metadata::GenerationMetadata;
use serde::{Deserialize, Serialize};

pub const ETCH_VERSION: u32 = 1;

/// The `etch` manifest object (RFC L0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EtchManifest {
    pub v: u32,
    /// 16-hex `EtchId`.
    pub id: String,
    pub tool: String,
    pub tool_version: String,
    /// Layers this image carries, e.g. `["L0","L1"]`.
    pub layers: Vec<String>,
    /// The source image's `EtchId` when plakat performed the derivation; else `null`.
    #[serde(default)]
    pub parent: Option<String>,
}

impl EtchManifest {
    pub fn new(id: EtchId, layers: &[Layer], parent: Option<EtchId>) -> Self {
        Self {
            v: ETCH_VERSION,
            id: id.hex(),
            tool: "plakat".into(),
            tool_version: env!("CARGO_PKG_VERSION").to_string(),
            layers: layers.iter().map(|l| format!("{l:?}")).collect(),
            parent: parent.map(|p| p.hex()),
        }
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
    pub fn parse(s: &str) -> Option<EtchManifest> {
        serde_json::from_str(s).ok()
    }
    pub fn etch_id(&self) -> Option<EtchId> {
        EtchId::parse_hex(&self.id)
    }
    pub fn parent_id(&self) -> Option<EtchId> {
        self.parent.as_deref().and_then(EtchId::parse_hex)
    }
}

/// Deterministic serialization of the generation recipe → the `canonical_manifest` fed to
/// `payload::derive_id`. Stable field order; the recipe that identifies the render (RFC §"EtchId").
pub fn canonical_manifest(m: &GenerationMetadata) -> String {
    format!(
        "prompt={}\nnegative={}\nmodel={}\nseed={}\nsteps={}\nguidance={}\nscheduler={}\nsize={}x{}\nloras={}",
        m.prompt.trim(),
        m.negative.trim(),
        m.model,
        m.seed,
        m.steps,
        m.guidance,
        m.scheduler,
        m.width,
        m.height,
        m.loras.join(","),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_roundtrips_json() {
        let m = EtchManifest::new(EtchId(0x9f2c4a17b3e08d5c), &[Layer::L0, Layer::L1], Some(EtchId(0x1122334455667788)));
        let j = m.to_json();
        let back = EtchManifest::parse(&j).unwrap();
        assert_eq!(back.id, "9f2c4a17b3e08d5c");
        assert_eq!(back.layers, vec!["L0", "L1"]);
        assert_eq!(back.parent_id(), Some(EtchId(0x1122334455667788)));
        assert_eq!(back.etch_id(), Some(EtchId(0x9f2c4a17b3e08d5c)));
    }
}
