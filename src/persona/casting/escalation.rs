//! The region-escalation ladder (RFC §14.1). Feature pixel-area determines which mechanisms function
//! at all: a full-body render puts the face at a small fraction of the frame, where face-reference
//! adapters degrade, swapping artefacts, and a four-pixel mole is not representable. The renderer
//! therefore branches on *measured* area and refines undersized regions at native resolution.
//!
//! This module is the pure decision half — measure area, decide whether a region needs escalation
//! against committed thresholds. The crop→refine→composite itself (reusing the adetailer path) is
//! driven by the render CLI. Thresholds are committed constants, exposed as flags.

/// A region the ladder can escalate (§14.1), in the fixed order it is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationRegion {
    Face,
    Mouth,
    Hand,
}

impl EscalationRegion {
    pub fn label(self) -> &'static str {
        match self {
            EscalationRegion::Face => "face",
            EscalationRegion::Mouth => "mouth",
            EscalationRegion::Hand => "hand",
        }
    }
    /// The default minimum area fraction (region bbox / frame) below which the region is refined.
    /// Committed constants (§14.1); the render CLI exposes them as flags.
    pub fn default_threshold(self) -> f32 {
        match self {
            // below ~9% of the frame, identity conditioning + swapping start to degrade.
            EscalationRegion::Face => 0.09,
            // teeth need a materially larger relative area to be representable at all.
            EscalationRegion::Mouth => 0.006,
            // hands are the least reliable; escalate generously when jewelry rides on them.
            EscalationRegion::Hand => 0.012,
        }
    }
}

/// Area of a `[x0, y0, x1, y1]` bbox as a fraction of the `frame_w × frame_h` frame, clamped `[0,1]`.
pub fn area_fraction(bbox: [f32; 4], frame_w: u32, frame_h: u32) -> f32 {
    let w = (bbox[2] - bbox[0]).max(0.0);
    let h = (bbox[3] - bbox[1]).max(0.0);
    let frame = (frame_w as f32 * frame_h as f32).max(1.0);
    (w * h / frame).clamp(0.0, 1.0)
}

/// The outcome of measuring one region against its threshold.
#[derive(Debug, Clone, Copy)]
pub struct EscalationDecision {
    pub region: EscalationRegion,
    pub area_fraction: f32,
    pub threshold: f32,
    /// True when the region is smaller than its threshold → crop + refine at native resolution.
    pub escalate: bool,
}

/// Decide whether `region` (at `area_fraction` of the frame) needs escalation, using `threshold`
/// (pass `region.default_threshold()` for the committed default).
pub fn decide(region: EscalationRegion, area_fraction: f32, threshold: f32) -> EscalationDecision {
    EscalationDecision { region, area_fraction, threshold, escalate: area_fraction < threshold }
}

/// The refinement crop for a region: the bbox expanded by `margin` (fraction of the bbox), clamped to
/// the frame. Matches the "crop with margin" step of the ladder so the refine has context to blend.
pub fn refine_crop(bbox: [f32; 4], margin: f32, frame_w: u32, frame_h: u32) -> [u32; 4] {
    let w = bbox[2] - bbox[0];
    let h = bbox[3] - bbox[1];
    let x0 = (bbox[0] - w * margin).max(0.0);
    let y0 = (bbox[1] - h * margin).max(0.0);
    let x1 = (bbox[2] + w * margin).min(frame_w as f32);
    let y1 = (bbox[3] + h * margin).min(frame_h as f32);
    [x0 as u32, y0 as u32, x1.ceil() as u32, y1.ceil() as u32]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_fraction_basics() {
        // a 100×100 box in a 200×200 frame = 0.25.
        assert!((area_fraction([0.0, 0.0, 100.0, 100.0], 200, 200) - 0.25).abs() < 1e-4);
        // degenerate / inverted box → 0.
        assert_eq!(area_fraction([50.0, 50.0, 10.0, 10.0], 200, 200), 0.0);
    }

    #[test]
    fn full_frame_face_does_not_escalate_tiny_face_does() {
        let big = area_fraction([20.0, 20.0, 180.0, 180.0], 200, 200); // 0.64
        let small = area_fraction([90.0, 90.0, 110.0, 110.0], 200, 200); // 0.01
        let t = EscalationRegion::Face.default_threshold();
        assert!(!decide(EscalationRegion::Face, big, t).escalate, "a big face is fine");
        assert!(decide(EscalationRegion::Face, small, t).escalate, "a tiny face escalates");
    }

    #[test]
    fn thresholds_are_ordered_face_gt_hand_gt_mouth() {
        // the face needs the most relative area; the mouth the least (it is a sub-region).
        assert!(EscalationRegion::Face.default_threshold() > EscalationRegion::Hand.default_threshold());
        assert!(EscalationRegion::Hand.default_threshold() > EscalationRegion::Mouth.default_threshold());
    }

    #[test]
    fn refine_crop_expands_and_clamps() {
        // a centred box expands by the margin...
        let c = refine_crop([80.0, 80.0, 120.0, 120.0], 0.25, 200, 200);
        assert_eq!(c, [70, 70, 130, 130]);
        // ...but clamps at the frame edges.
        let edge = refine_crop([0.0, 0.0, 40.0, 40.0], 0.5, 200, 200);
        assert_eq!(edge, [0, 0, 60, 60]);
    }
}
