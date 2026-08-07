//! Evidence fusion → a graded verdict (RFC §"Verdicts"). Phase 1 fuses L0 only (the manifest); later
//! phases add L1 (pixel), L2 (latent), L3 (fingerprint). The verdict is graded, never a boolean, and
//! `no-evidence` is explicitly *not* proof of non-plakat origin.

use super::manifest::EtchManifest;
use super::EtchId;
use std::path::Path;

/// The graded verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Generated,
    Derived,
    ProbableDerivative,
    Inconclusive,
    NoEvidence,
}

impl Verdict {
    pub fn slug(&self) -> &'static str {
        match self {
            Verdict::Generated => "generated",
            Verdict::Derived => "derived",
            Verdict::ProbableDerivative => "probable-derivative",
            Verdict::Inconclusive => "inconclusive",
            Verdict::NoEvidence => "no-evidence",
        }
    }
    pub fn meaning(&self) -> &'static str {
        match self {
            Verdict::Generated => "plakat produced this image",
            Verdict::Derived => "plakat produced an ancestor",
            Verdict::ProbableDerivative => "semantically matches a known plakat output",
            Verdict::Inconclusive => "weak/conflicting evidence — do not rely on this either way",
            Verdict::NoEvidence => "no evidence found (absence of evidence, not evidence of absence)",
        }
    }
}

/// One layer's outcome in the report.
#[derive(Debug, Clone)]
pub struct LayerStatus {
    pub state: &'static str, // present | absent | partial | match | skipped | unavailable
    pub detail: String,
}
impl LayerStatus {
    fn absent(detail: &str) -> Self {
        Self { state: "absent", detail: detail.into() }
    }
    fn skipped(detail: &str) -> Self {
        Self { state: "skipped", detail: detail.into() }
    }
}

/// The fused report for one image.
#[derive(Debug, Clone)]
pub struct Report {
    pub verdict: Verdict,
    pub id: Option<EtchId>,
    pub parent: Option<EtchId>,
    pub l0: LayerStatus,
    pub l1: LayerStatus,
    pub l2: LayerStatus,
    pub l3: LayerStatus,
    pub note: Option<String>,
}

/// Read the L0 `etch` manifest from a PNG `tEXt` chunk, falling back to the `<base>.json` sidecar's
/// `etch` field. Offline, no model.
pub fn read_l0(path: &Path) -> Option<EtchManifest> {
    if let Some(chunk) = read_etch_chunk(path) {
        if let Some(m) = EtchManifest::parse(&chunk) {
            return Some(m);
        }
    }
    // sidecar (`<image>.png.json`, plakat's convention) with an `etch` object.
    let side = crate::imaging::io::sidecar_path(path);
    let text = std::fs::read_to_string(&side).ok()?;
    let val: serde_json::Value = serde_json::from_str(&text).ok()?;
    serde_json::from_value(val.get("etch")?.clone()).ok()
}

fn read_etch_chunk(path: &Path) -> Option<String> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let reader = decoder.read_info().ok()?;
    reader
        .info()
        .uncompressed_latin1_text
        .iter()
        .find(|c| c.keyword == "etch")
        .map(|c| c.text.clone())
}

/// Verify an image. **Phase 1: L0 only** (later phases run L1/L2/L3 and fuse). `run_l2` (the model-loading
/// DDIM path) is honoured in Phase 4; here it only affects the L2 status line.
pub fn verify(path: &Path, run_l2: bool) -> Report {
    let l2 = if run_l2 {
        LayerStatus::skipped("L2 not yet implemented")
    } else {
        LayerStatus::skipped("--verify not given")
    };
    match read_l0(path) {
        Some(m) => {
            let id = m.etch_id();
            let parent = m.parent_id();
            let note = parent.map(|p| format!("derivation chain: parent {}", p.hex()));
            Report {
                verdict: Verdict::Generated,
                id,
                parent,
                l0: LayerStatus { state: "present", detail: format!("manifest v{}, tool {} {}", m.v, m.tool, m.tool_version) },
                l1: LayerStatus::skipped("L1 not yet implemented"),
                l2,
                l3: LayerStatus::skipped("L3 not yet implemented"),
                note,
            }
        }
        None => Report {
            verdict: Verdict::NoEvidence,
            id: None,
            parent: None,
            l0: LayerStatus::absent("stripped or never written"),
            l1: LayerStatus::skipped("L1 not yet implemented"),
            l2,
            l3: LayerStatus::skipped("L3 not yet implemented"),
            note: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::etch::manifest::EtchManifest;
    use crate::etch::Layer;

    fn write_png_with_etch(path: &Path, etch: Option<&str>) {
        let file = std::fs::File::create(path).unwrap();
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), 2, 2);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        if let Some(e) = etch {
            enc.add_text_chunk("etch".to_string(), e.to_string()).unwrap();
        }
        let mut w = enc.write_header().unwrap();
        w.write_image_data(&[0u8; 12]).unwrap();
    }

    #[test]
    fn verify_reads_l0_from_the_png_chunk_to_generated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.png");
        let m = EtchManifest::new(EtchId(0xdeadbeefcafef00d), &[Layer::L0], Some(EtchId(0x1122334455667788)));
        write_png_with_etch(&path, Some(&m.to_json()));
        let r = verify(&path, false);
        assert_eq!(r.verdict, Verdict::Generated);
        assert_eq!(r.id, Some(EtchId(0xdeadbeefcafef00d)));
        assert_eq!(r.parent, Some(EtchId(0x1122334455667788)));
        assert_eq!(r.l0.state, "present");
        assert!(r.note.as_deref().unwrap().contains("1122334455667788"));
    }

    #[test]
    fn verify_falls_back_to_the_sidecar_then_no_evidence() {
        let dir = tempfile::tempdir().unwrap();
        // no chunk, but a sidecar with an etch object → still found.
        let path = dir.path().join("b.png");
        write_png_with_etch(&path, None);
        let m = EtchManifest::new(EtchId(0x0102030405060708), &[Layer::L0], None);
        std::fs::write(crate::imaging::io::sidecar_path(&path), format!("{{\"etch\":{}}}", m.to_json())).unwrap();
        let r = verify(&path, false);
        assert_eq!(r.verdict, Verdict::Generated);
        assert_eq!(r.id, Some(EtchId(0x0102030405060708)));
        // a bare image with neither → no-evidence.
        let bare = dir.path().join("c.png");
        write_png_with_etch(&bare, None);
        assert_eq!(verify(&bare, false).verdict, Verdict::NoEvidence);
    }
}
