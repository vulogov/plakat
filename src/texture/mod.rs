//! `plakat texture` — seamless PBR material synthesis (RFC TEXTURE-1).
//!
//! A `TextureSpec` HJSON document → deterministic resolver → seamless (circular-conv) generation →
//! channel derivation → measured, engine-ready export. Sibling in shape to `bookart`/`persona`: this
//! module holds the **deterministic front half** first — schema, the pure resolver, and lint — with
//! the derivation core, scorecard, preview, export, seamless engine, and generation landing across
//! later phases. No weights, byte-stable; fully additive.
//!
//! Build status (ROADMAP_TEXTURE_1):
//!   * B0 — spec + resolver + lint (this).
//!   * B1+ — derivation + scorecard + preview + export + seamless engine + generation (siblings).

pub mod compile;
pub mod derive;
pub mod export;
pub mod lint;
pub mod preview;
pub mod render;
pub mod scorecard;
pub mod seamless;
pub mod spec;

pub use compile::{resolve, ChannelSource, HeightSource, RenderPlan};
pub use derive::Material;
pub use preview::Shape;
pub use scorecard::{score, Scorecard};
pub use spec::{TextureSpec, SCHEMA_VERSION};
