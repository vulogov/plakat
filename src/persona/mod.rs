//! `plakat persona` — controllable synthetic-person composition (RFC PERSONA-1, the 5.0.0 flagship).
//!
//! A `PersonaSpec` HJSON document → deterministic resolver → per-family conditioning + geometric map +
//! landmark-anchored localized details + identity reference set + measurement scorecard. This module
//! holds the **deterministic compiler half** (Layer 0/0b/1's front matter): the schema, the lexicon,
//! the pure resolver, and the per-encoder-class emitters — no weights, no I/O, byte-stable (§5.2).
//!
//! Build status (ROADMAP_5.0.0, cut line P0–P8):
//!   * P0 — spec + lint (this) → lexicon + resolver + emitters + salience (follow-on).
//!   * later phases (scorecard, geometry, details, casting, TUI) live in sibling modules.
//!
//! Fully additive; nothing here changes existing behaviour.

pub mod compile;
pub mod lexicon;
pub mod lint;
pub mod spec;

pub use spec::{PersonaSpec, SCHEMA_VERSION};
