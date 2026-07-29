//! Layer-2.5 — the localized-detail subsystem (RFC §8). Marks, jewelry and dentition are small,
//! positional, high-contrast, and identical-across-renders — which defeats text conditioning (§2.1.4)
//! and needs its own machinery: **anchored anatomically, realised by compositing, verified locally**.
//!
//! Build order (ROADMAP_5.0.0 P3): procedural overlay generators (this + `overlay`) → the compositing
//! pass (`composite`) → the `persona composite` CLI. The deterministic core (overlay generation +
//! compositing without harmonisation) is byte-stable and CI-testable without weights (§5.2); the one
//! stochastic step, harmonisation, is an optional masked-img2img pass layered on top.

pub mod composite;
pub mod overlay;

pub use composite::{composite_details, composite_details_opts, CompositeResult, Culled};
pub use overlay::Light;
