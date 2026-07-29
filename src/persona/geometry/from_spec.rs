//! Bridge from a `PersonaSpec` to the geometry engine's inputs (the resolver → Layer-2 join).
//!
//! Kept separate from the pure geometry core (`basis`/`raster`/`figure`) so those stay a function of
//! plain values only (the §5.2 determinism contract, testable without the spec types). This module is
//! the one place that knows the spec's field layout.

use super::figure::{Build, FigureParams};
use crate::persona::spec::PersonaSpec;
use std::collections::BTreeMap;

/// Pull the geometric scalar attributes (lexicon paths → `[0,1]`) the deformation basis consumes.
/// Absent spec fields are simply omitted (→ the mean template contributes for that attribute).
pub fn geometry_values(spec: &PersonaSpec) -> BTreeMap<String, f32> {
    let mut v = BTreeMap::new();
    let mut put = |k: &str, val: Option<f32>| {
        if let Some(x) = val.filter(|x| x.is_finite()) {
            v.insert(k.to_string(), x);
        }
    };
    if let Some(f) = &spec.face {
        put("face.width", f.width);
        put("face.jaw.width", f.jaw.as_ref().and_then(|j| j.width));
        put("face.chin.projection", f.chin.as_ref().and_then(|c| c.projection));
        put("face.cheekbones.prominence", f.cheekbones.as_ref().and_then(|c| c.prominence));
        put("face.asymmetry", f.asymmetry);
    }
    if let Some(e) = &spec.eyes {
        put("eyes.spacing", e.spacing);
        put("eyes.canthal_tilt", e.canthal_tilt);
        put("eyes.brow.thickness", e.brow.as_ref().and_then(|b| b.thickness));
    }
    if let Some(n) = &spec.nose {
        put("nose.length", n.length);
    }
    if let Some(m) = &spec.mouth {
        put("mouth.width", m.width);
        put("mouth.lower_lip", m.lower_lip);
    }
    v
}

/// Whether the mouth should manifest the open (dentition-visible) aperture variant (§8.7). True when
/// teeth are asserted visible or the expression is an open-mouth smile/laugh.
pub fn open_mouth(spec: &PersonaSpec) -> bool {
    let teeth_visible = spec
        .teeth
        .as_ref()
        .and_then(|t| t.visibility.as_deref())
        .is_some_and(|v| matches!(v, "visible" | "full" | "smile" | "wide"));
    let expr_open = spec
        .defaults
        .as_ref()
        .and_then(|d| d.expression.as_deref())
        .is_some_and(|e| matches!(e, "smile" | "laugh" | "grin" | "open"));
    teeth_visible || expr_open
}

/// Build `FigureParams` from the spec's `figure` block. Returns `None` when the block is absent
/// (no figure geometry requested). Missing sub-fields fall back to the neutral defaults.
pub fn figure_params(spec: &PersonaSpec) -> Option<FigureParams> {
    let f = spec.figure.as_ref()?;
    let d = FigureParams::default();
    // shoulder↔waist taper: broad shoulders and/or narrow waist push it up.
    let sw = match (f.shoulders, f.waist) {
        (None, None) => 0.5,
        (s, w) => (0.5 + (s.unwrap_or(0.5) - 0.5) * 0.5 + (0.5 - w.unwrap_or(0.5)) * 0.5).clamp(0.0, 1.0),
    };
    Some(FigureParams {
        height_cm: f.height_cm.filter(|x| x.is_finite()).unwrap_or(d.height_cm),
        build: f.build.as_deref().and_then(Build::parse).unwrap_or(d.build),
        shoulder_waist: sw,
        limb_length: f.limb_length.filter(|x| x.is_finite()).unwrap_or(d.limb_length),
        musculature: f.musculature.filter(|x| x.is_finite()).unwrap_or(d.musculature),
    })
}

#[cfg(test)]
mod tests {
    use super::super::{resolve, resolve_figure};
    use super::*;

    fn spec_from(hjson: &str) -> PersonaSpec {
        PersonaSpec::from_hjson(hjson).unwrap()
    }

    #[test]
    fn pulls_geometric_scalars_from_the_spec() {
        let s = spec_from("{ schema: persona/1\n face: { width: 0.8 }\n eyes: { spacing: 0.3, canthal_tilt: 0.7 }\n nose: { length: 0.6 } }");
        let v = geometry_values(&s);
        assert_eq!(v.get("face.width"), Some(&0.8));
        assert_eq!(v.get("eyes.spacing"), Some(&0.3));
        assert_eq!(v.get("nose.length"), Some(&0.6));
        assert!(!v.contains_key("mouth.width"), "unset attrs omitted");
    }

    #[test]
    fn open_mouth_from_teeth_or_expression() {
        // (HJSON: quote string values that share a line with a closing brace — quoteless runs to EOL.)
        assert!(open_mouth(&spec_from("{ schema: \"persona/1\"\n teeth: { visibility: \"visible\" } }")));
        assert!(open_mouth(&spec_from("{ schema: \"persona/1\"\n defaults: { expression: \"smile\" } }")));
        assert!(!open_mouth(&spec_from("{ schema: \"persona/1\"\n defaults: { expression: \"neutral\" } }")));
    }

    #[test]
    fn figure_params_absent_when_no_block() {
        assert!(figure_params(&spec_from("{ schema: \"persona/1\" }")).is_none());
        let p = figure_params(&spec_from("{ schema: \"persona/1\"\n figure: { height_cm: 185, build: \"athletic\" } }")).unwrap();
        assert_eq!(p.height_cm, 185.0);
        assert_eq!(p.build, Build::Mesomorph);
    }

    #[test]
    fn spec_to_geometry_is_byte_stable() {
        // The corpus: a fixed spec → resolved landmarks → hash. Pins the whole spec→geometry path.
        let s = spec_from(
            "{ schema: \"persona/1\"\n face: { width: 0.7, chin: { projection: 0.6 } }\n eyes: { spacing: 0.65, canthal_tilt: 0.4 }\n nose: { length: 0.55 }\n mouth: { width: 0.6, lower_lip: 0.7 } }",
        );
        let d = resolve(&geometry_values(&s), open_mouth(&s), 2026);
        let mut acc: u64 = 1469598103934665603;
        for &(x, y) in d.landmarks.iter() {
            for val in [x, y] {
                acc = (acc ^ (val * 100_000.0).round() as i64 as u64).wrapping_mul(1099511628211);
            }
        }
        assert_eq!(acc, 954449522269940096, "spec→geometry output changed — update the golden intentionally");
        // and the figure path is stable too.
        let fp = figure_params(&spec_from("{ schema: \"persona/1\"\n figure: { build: \"endomorph\", shoulders: 0.4, waist: 0.7 } }")).unwrap();
        let fig = resolve_figure(&fp, 2026);
        assert!(fig.joints.iter().all(|p| p[0].is_finite() && p[1].is_finite()));
    }
}
