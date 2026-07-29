//! Layer 3 — the identity anchor / casting (RFC §11). Geometry, details and prompts constrain what a
//! face looks like; they do not by themselves produce a face stable enough to recognise across renders.
//! The **cast** does: render N candidates, composite the persona's details, score them, keep the best,
//! and validate that they are one person (ArcFace coherence) — then store the reference set that every
//! later render anchors to.
//!
//! This is the first heavy-weights phase (SCRFD + ArcFace + the generate pipeline + the swap bridge).
//! The deterministic core — reference-set storage + coherence math (`reference`) — is testable without
//! weights; the orchestration (`cast` / `render`) lives in the CLI and runs models.

pub mod attribution;
pub mod escalation;
pub mod reference;

pub use attribution::{assign, containment, iou, Assignment, ATTRIBUTION_CONFIDENCE_MIN};
pub use escalation::{area_fraction, decide, refine_crop, EscalationDecision, EscalationRegion};
pub use reference::{
    centroid, cosine, compute_coherence, Coherence, Reference, ReferenceSet, COHERENCE_THRESHOLD,
};
