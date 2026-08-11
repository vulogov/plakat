//! `plakat product` — studio product-shots / packshots from a subject (RFC PRODUCT-1). Sibling of
//! `texture` / `bookart` / `comic`; stands on `relight` (IC-Light) + `matting` + `compose`. The
//! weight-free half — sweep + grounding (contact shadow + reflection from the alpha) + composite — needs
//! no GPU; only relight + subject-generation (P2) need a model.
//!
//! Always compiled (the deterministic front half is weight-free).

pub mod compose;
pub mod ground;
pub mod lint;
pub mod render;
pub mod spec;

pub use compose::{resolve, Plan};
pub use spec::{ProductSpec, SCHEMA_VERSION};
