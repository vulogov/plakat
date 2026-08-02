//! The ornament geometry engine (RFC BOOKART-1 §6.2–6.4). Pure, no weights, byte-stable.
//!
//! B0 slice: the [`page`] model (named sizes → exact pixels at DPI). B2 adds `layout` (per-ornament-type
//! placement against the text block) and `symmetry` (fundamental-domain + replicate).

pub mod page;

pub use page::{resolve_page, PageResolved, SIZE_VOCAB};
