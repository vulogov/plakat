//! Progress-callback plumbing for the (potentially slow) fractal generators.
//!
//! Each renderer periodically calls `prog(done, total)` from its work loop — including
//! from rayon worker threads — so the trait object is `Sync`. The CLI wires this to an
//! `indicatif` progress bar; tests and the library `render_spec` pass [`silent`].

/// A progress sink: `(units_done, units_total)`. Called from worker threads → `Sync`.
pub type ProgressFn<'a> = &'a (dyn Fn(u64, u64) + Sync);

/// A no-op progress sink for callers that don't want a bar.
pub fn silent() -> impl Fn(u64, u64) + Sync {
    |_, _| {}
}
