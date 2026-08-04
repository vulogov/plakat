//! `plakat bookart` — controllable black-and-white book-ornament composition (RFC BOOKART-1).
//!
//! A `BookArtSpec` HJSON document → deterministic resolver → ornament geometry + hybrid render router +
//! B/W finisher (transparency + optional vector) → exact-page-sized transparent PNG. Sibling in shape to
//! `persona`: this module holds the **deterministic front half** — schema, lexicon, the pure resolver,
//! the page model, and lint. No weights, no I/O beyond reading the spec, byte-stable (§5.2).
//!
//! Build status (ROADMAP_BOOKART_1):
//!   * B0 — spec + lexicon + resolver + page model + lint (this).
//!   * later phases (finisher, symmetry, procedural, diffusion, scorecard, kit, manuscript) are siblings.
//!
//! Fully additive; nothing here changes existing behaviour.

pub mod compile;
pub mod edit;
pub mod finish;
pub mod geometry;
#[cfg(feature = "shaped-labels")]
pub mod glyph;
pub mod kit;
pub mod lexicon;
pub mod lint;
pub mod manuscript;
pub mod procedural;
pub mod render;
pub mod scenario_task;
pub mod scorecard;
pub mod spec;

pub use compile::{resolve, RenderPlan};
pub use finish::finish_ornament;
pub use render::{render_spec, RenderOpts, Rendered};
pub use scorecard::{score, Scorecard};
pub use spec::{BookArtSpec, SCHEMA_VERSION};
