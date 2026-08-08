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

/// Load an image as RGB + optional alpha (0 = transparent) + dims. `None` if unreadable.
fn read_image(path: &Path) -> Option<(Vec<u8>, Option<Vec<u8>>, usize, usize)> {
    let img = image::open(path).ok()?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    let alpha = match &img {
        image::DynamicImage::ImageRgba8(_) | image::DynamicImage::ImageLumaA8(_) => {
            Some(img.to_rgba8().pixels().map(|p| p.0[3]).collect())
        }
        _ => None,
    };
    Some((img.to_rgb8().into_raw(), alpha, w, h))
}

/// Verify an image — fuse L0 (manifest) + L1 (pixel etch). `key` is the verifier's key (public by
/// default). `run_l2` gates the model-loading L2 path (Phase 4). Offline for L0/L1.
pub fn verify(path: &Path, key: &str, run_l2: bool) -> Report {
    let l0 = read_l0(path);
    // L1 — extract the pixel etch (offline).
    let l1_res = read_image(path).and_then(|(rgb, alpha, w, h)| super::pixel::extract(&rgb, w, h, key, alpha.as_deref()));
    let l2 = if run_l2 {
        LayerStatus::skipped("L2 not yet implemented")
    } else {
        LayerStatus::skipped("--verify not given")
    };
    let l3 = LayerStatus::skipped("L3 not yet implemented");

    // Fuse. A confident L1 decode (low p) or a present-and-consistent L0 ⇒ generated.
    let l1_strong = l1_res.as_ref().map(|r| r.p_value < 1e-6).unwrap_or(false);
    let l1_status = match &l1_res {
        Some(r) if r.p_value < 1e-6 => LayerStatus { state: "present", detail: format!("{}/{} tiles, p = {:.1e}", r.tiles_ok, r.tiles_total, r.p_value) },
        Some(r) => LayerStatus { state: "partial", detail: format!("{}/{} tiles, p = {:.1e}", r.tiles_ok, r.tiles_total, r.p_value) },
        None => LayerStatus::absent("no pixel etch recovered"),
    };
    let l1_id = l1_res.as_ref().map(|r| r.id);
    let l0_id = l0.as_ref().and_then(|m| m.etch_id());
    let parent = l0.as_ref().and_then(|m| m.parent_id());

    let (verdict, id, note) = match (&l0, l1_strong) {
        (Some(m), _) => {
            let id = l0_id;
            let note = parent.map(|p| format!("derivation chain: parent {}", p.hex()));
            // if L1 also decoded a different id, flag it.
            let note = match (l1_id, id) {
                (Some(a), Some(b)) if a != b => Some(format!("L0 id {} but L1 decoded {} — inconsistent", b.hex(), a.hex())),
                _ => note,
            };
            let _ = m;
            (Verdict::Generated, id, note)
        }
        (None, true) => (Verdict::Generated, l1_id, Some("recovered from the pixel etch (L1); L0 manifest was stripped".into())),
        (None, false) => match &l1_res {
            // a weak/partial L1 with no L0 — some evidence, not conclusive.
            Some(r) if r.tiles_ok > 0 => (Verdict::Inconclusive, Some(r.id), Some("weak L1 signal — treat as inconclusive".into())),
            _ => (Verdict::NoEvidence, None, None),
        },
    };
    let l0_status = match &l0 {
        Some(m) => LayerStatus { state: "present", detail: format!("manifest v{}, tool {} {}", m.v, m.tool, m.tool_version) },
        None => LayerStatus::absent("stripped or never written"),
    };
    Report { verdict, id, parent, l0: l0_status, l1: l1_status, l2, l3, note }
}

/// Fuse an L3 fingerprint match (from the store, at generation-time semantics) into the report. Called by
/// the doctor after loading CLIP + querying the store (Phase 3). `avail`: whether L3 could even run.
pub fn fuse_l3(mut report: Report, l3: Option<super::fingerprint::Match>, avail: &str) -> Report {
    use super::fingerprint::{classify, L3Strength};
    match l3 {
        Some(m) => {
            let strength = classify(m.cosine);
            report.l3 = LayerStatus { state: if strength == L3Strength::Strong { "match" } else if strength == L3Strength::Probable { "weak-match" } else { "no-match" }, detail: format!("cosine {:.3} → {}", m.cosine, m.id.hex()) };
            match (report.verdict, strength) {
                // already conclusive from L0/L1 → L3 only confirms; flag an id mismatch.
                (Verdict::Generated, L3Strength::Strong) => {
                    if report.id.map(|id| id != m.id).unwrap_or(false) {
                        report.note = Some(format!("L3 matched a different id ({}) than L0/L1 — check the edit chain", m.id.hex()));
                    }
                }
                // no bit-level evidence but a strong semantic match ⇒ a probable derivative of a known output.
                (Verdict::NoEvidence | Verdict::Inconclusive, L3Strength::Strong) => {
                    report.verdict = Verdict::ProbableDerivative;
                    report.id = Some(m.id);
                    report.note = Some("no L0/L1 bits, but a strong CLIP match to a known plakat output".into());
                }
                (Verdict::NoEvidence | Verdict::Inconclusive, L3Strength::Probable) => {
                    report.verdict = Verdict::ProbableDerivative;
                    report.id = Some(m.id);
                    report.note = Some("weak semantic match — treat with caution".into());
                }
                // partial L1 + L3 match ⇒ derived (a light generative edit).
                _ if report.l1.state == "partial" && strength != L3Strength::None => {
                    report.verdict = Verdict::Derived;
                    report.note = Some("partial pixel etch + a semantic match — consistent with a light generative edit".into());
                }
                _ => {}
            }
        }
        None => {
            report.l3 = LayerStatus { state: "unavailable", detail: avail.to_string() };
        }
    }
    report
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
        let r = verify(&path, "k", false);
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
        let r = verify(&path, "k", false);
        assert_eq!(r.verdict, Verdict::Generated);
        assert_eq!(r.id, Some(EtchId(0x0102030405060708)));
        // a bare image with neither → no-evidence.
        let bare = dir.path().join("c.png");
        write_png_with_etch(&bare, None);
        assert_eq!(verify(&bare, "k", false).verdict, Verdict::NoEvidence);
    }
}
