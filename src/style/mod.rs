// Spike: this module's surface is unused until the `plakat style detect`
// CLI lands in commit 4. Remove these allows once that commit lands.
#![allow(dead_code, unused_imports)]

//! Art-style detection and transfer.
//!
//! Two halves:
//!
//! * **Detect** — encode a reference photo through CLIP-H and cosine-match
//!   it against the bundled catalog of exemplar embeddings. See
//!   [`detect_style`].
//! * **Transfer** — resolve the detected style to one or more LoRAs +
//!   trigger tokens for the active base model, threaded into the normal
//!   generation pipeline. The resolve API is not implemented in the
//!   spike — see the design notes for the planned shape.
//!
//! The catalog (`catalog.json` + `exemplars.safetensors`) is built
//! offline by `examples/spike_catalog.rs` (spike) / a dedicated build
//! tool (post-spike). Runtime only loads it.
//!
//! ## Spike scope
//!
//! Only the detection half ships in the spike. Sufficient surface to
//! prove the cosine pipeline carries enough style signal end-to-end
//! before committing to the rest of the design.

pub mod catalog;
pub mod detect;
pub mod encode;

pub use catalog::{
    Aggregation, BaseModel, DetectionPolicy, DetectionResult, LoadedStyle, LoraEntry, ModelEntry,
    StyleCatalog, StyleMatch,
};
pub use detect::detect_style;
pub use encode::encode_reference_photo;
