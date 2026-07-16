//! History semantic search — re-exports the crate-level [`crate::textsearch`] ranker (moved there
//! in v3.0 so `plakat photos` can share it without depending on the `ui` feature). Kept as a module
//! so the History screen's `services::semantic::rank` path is unchanged.

pub use crate::textsearch::rank;
