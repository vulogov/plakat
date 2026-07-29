//! Layer 5 — calibration (RFC §13). Converts the schema from a suggestion box into an instrument: a
//! slow, offline, **per-family** process whose outputs are committed tables (`assets/persona/
//! calibration/<family>.hjson`). Three products: **priors** (the meaning of `0.5`, §13.1), **response
//! curves** (requested→realised transfer functions the compiler pre-distorts through, §13.2), and
//! **grades** (measured controllability, §13.3). Tables record their measurement identity and go stale
//! deterministically (§13.4).
//!
//! Build order (ROADMAP_5.0.0 P4): fit/grade math (`fit`) → table schema + loader + staleness +
//! bootstrap (`table`) → compiler/scorecard wiring → the `persona calibrate` harness.

pub mod fit;
pub mod table;

pub use fit::{eval, fit, grade_from, predistort, Grade, ResponseCurve};
pub use table::{CalibrationTable, MeasurementIdentity, Prior};
