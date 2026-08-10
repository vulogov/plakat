//! `plakat comic` — multi-panel comic pages from a script (RFC COMIC-1). Sibling of `bookart` (page
//! layout) + `persona` (character identity across panels). The weight-free half — spec, panel-layout
//! engine, page composite, balloons/lettering — needs no GPU; only per-panel scene art (P3) needs a model.
//!
//! Always compiled (the deterministic front half is weight-free).

pub mod balloon;
pub mod layout;
pub mod lint;
pub mod page;
pub mod render;
pub mod scenario_task;
pub mod spec;

pub use layout::{resolve, Plan, PanelRect};
pub use spec::{ComicSpec, SCHEMA_VERSION};
