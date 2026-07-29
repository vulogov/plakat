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

use std::collections::BTreeMap;

/// Assemble a table from measured (or seeded) sweep data: fit a `ResponseCurve` per attribute from its
/// `(requested, realised)` samples + per-step variance. The one place the fit is applied to real data.
#[allow(clippy::too_many_arguments)]
pub fn assemble(
    family: String,
    identity: MeasurementIdentity,
    priors: BTreeMap<String, Prior>,
    samples: BTreeMap<String, (Vec<(f32, f32)>, f32)>,
    harmonise: BTreeMap<String, f32>,
    spontaneous_detail_rate: f32,
    prompted_detail_hit_rate: f32,
) -> CalibrationTable {
    let curves = samples.into_iter().map(|(k, (s, var))| (k, fit(s, var))).collect();
    CalibrationTable { family, identity, priors, curves, harmonise, spontaneous_detail_rate, prompted_detail_hit_rate }
}

/// Pre-distort a geometry value map in place through the family's response-curve inverses (§13.2), so
/// that a requested scalar *lands* at that value once the model has realised the conditioning. Values
/// with no curve for their attribute pass through unchanged. Returns the attributes it corrected.
pub fn predistort_geometry(values: &mut BTreeMap<String, f32>, table: &CalibrationTable) -> Vec<String> {
    let mut corrected = Vec::new();
    for (path, v) in values.iter_mut() {
        if let Some(curve) = table.curves.get(path) {
            let pd = predistort(curve, *v);
            if (pd - *v).abs() > 1e-4 {
                corrected.push(format!("{path} {v:.2}→{pd:.2}"));
            }
            *v = pd;
        }
    }
    corrected
}
