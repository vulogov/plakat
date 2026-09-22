//! The PAINTING PLAN (RFC PAINT-1 §5/§12.2). The model/analysis stage acts as an ART DIRECTOR: it inspects a
//! provided or generated image and produces a *plan* — how a painter would approach it — and the deterministic,
//! weight-free stroke engine then executes that plan. No pixel of the deliverable comes from a model (G1); the
//! plan only decides STRUCTURE (focal region, per-region armature resolution, value key, reserve), never surface.
//!
//! This is the first, deterministic analyzer (face + measured contrast + medium). A model-driven art director
//! (VLM / low-res diffusion block-in) can later produce the same `PaintPlan` behind this interface unchanged.

use serde::Deserialize;

/// A painting plan: the structural decisions the paint stage executes. Serialisable to HJSON so it is
/// inspectable and hand-editable — the art director's written plan.
#[derive(Debug, Clone, Deserialize)]
pub struct PaintPlan {
    /// The medium to paint in.
    #[serde(default = "default_medium")]
    pub medium: String,
    /// Palette name, or `image` to derive from the reference.
    #[serde(default = "default_palette")]
    pub palette: String,
    /// Fidelity register.
    #[serde(default = "default_style")]
    pub style: String,
    /// COARSE body/background armature resolution (px) — structure, not detail (RFC §1.1).
    #[serde(default = "default_armature")]
    pub armature: u32,
    /// FINE focal (face) armature resolution (px), or `null` for a uniform armature (RFC §5.2).
    #[serde(default)]
    pub armature_face: Option<u32>,
    /// MID subject-body armature resolution (px) — the three-tier plan: background coarsest, body mid, face fine.
    /// Needs a subject matte (U2Net), run automatically when set (RFC §5.2 multi-region).
    #[serde(default)]
    pub armature_body: Option<u32>,
    /// AERIAL PERSPECTIVE strength (0..1): veil the BACKGROUND (matte-derived) so the subject advances (§5.5).
    #[serde(default)]
    pub recede: f32,
    /// Value-key strength (tonal-range expansion) — higher for a flat, low-contrast reference.
    #[serde(default)]
    pub value_key: f32,
    /// Reserve threshold for surface-white media (paper whites), or `null` for the medium default.
    #[serde(default)]
    pub reserve: Option<f32>,
    /// Stroke budget, or `null` for the size-derived default.
    #[serde(default)]
    pub budget: Option<usize>,
    /// Human-readable analysis notes (why these numbers) — informational, ignored by the paint stage.
    #[serde(default)]
    pub notes: Vec<String>,
}

fn default_medium() -> String {
    "watercolour".into()
}
fn default_palette() -> String {
    "image".into()
}
fn default_style() -> String {
    "impressionist".into()
}
fn default_armature() -> u32 {
    72
}

impl Default for PaintPlan {
    fn default() -> Self {
        Self {
            medium: default_medium(),
            palette: default_palette(),
            style: default_style(),
            armature: default_armature(),
            armature_face: None,
            armature_body: None,
            recede: 0.0,
            value_key: 0.0,
            reserve: None,
            budget: None,
            notes: Vec::new(),
        }
    }
}

impl PaintPlan {
    /// Parse a plan from HJSON.
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        Ok(deser_hjson::from_str(text)?)
    }

    /// Serialise to HJSON (stable key order, with the analysis notes as comments).
    pub fn to_hjson(&self) -> String {
        let mut o = String::new();
        o.push_str("# plakat painting plan (RFC PAINT-1) — the art director's structural decisions.\n");
        for n in &self.notes {
            o.push_str(&format!("# {n}\n"));
        }
        o.push_str(&format!("medium: {}\n", self.medium));
        o.push_str(&format!("palette: {}\n", self.palette));
        o.push_str(&format!("style: {}\n", self.style));
        o.push_str(&format!("armature: {}\n", self.armature));
        if let Some(af) = self.armature_face {
            o.push_str(&format!("armature_face: {af}\n"));
        }
        if let Some(ab) = self.armature_body {
            o.push_str(&format!("armature_body: {ab}\n"));
        }
        if self.recede > 1e-3 {
            o.push_str(&format!("recede: {:.2}\n", self.recede));
        }
        o.push_str(&format!("value_key: {:.2}\n", self.value_key));
        if let Some(r) = self.reserve {
            o.push_str(&format!("reserve: {r:.2}\n"));
        }
        if let Some(b) = self.budget {
            o.push_str(&format!("budget: {b}\n"));
        }
        o
    }
}

/// The measured signals the analyzer reads from the reference. Kept separate so the (GPU) face detection and the
/// (CPU) statistics are gathered by the caller and handed in — the analyzer itself is a pure function.
pub struct Analysis {
    /// Number of faces detected (the focal subjects).
    pub faces: usize,
    /// Global luma standard deviation in [0,1] — a proxy for tonal contrast (low = flat/foggy).
    pub luma_stddev: f32,
    /// The medium requested (drives the reserve default).
    pub medium: String,
    /// The palette requested (usually `image`).
    pub palette: String,
    /// Canvas short side (px) — scales the budget.
    pub short_side: u32,
    /// Long side (px) — for budget area.
    pub long_side: u32,
    /// Whether the medium reserves the paper white (watercolour / ink).
    pub surface_white: bool,
}

/// Turn measured signals into a plan — the deterministic art director. The rules encode the tuning that had to be
/// done by hand: paint from a coarse armature (never trace), give a detected FACE a finer armature so it stays
/// crisp while the rest becomes washes, expand a flat reference's values, and reserve the paper for wet media.
pub fn plan_from(a: &Analysis) -> PaintPlan {
    let mut notes = Vec::new();

    // A COARSE base armature so the engine invents the surface instead of tracing (RFC §1.1). Larger canvases can
    // carry a touch more structure without tracing. (Reduced further for the background when a subject is present.)
    let armature = if a.short_side >= 900 { 88 } else { 72 };
    notes.push("paint from a coarse armature — structure, not pixels (no tracing)".into());

    // A detected FACE is the focal region: give it a FINE armature so features stay crisp while the beard / hair /
    // background become washes (RFC §5.2). With a subject present, use a THREE-TIER plan — a coarser background, a
    // mid subject body, and the fine face — plus a touch of aerial recession so the subject advances.
    let (armature_face, armature_body, recede, armature) = if a.faces > 0 {
        notes.push(format!("{} face(s) → three-tier armature: background {}px · body 104px · face 200px", a.faces, (armature as f32 * 0.6).round() as u32));
        notes.push("subject matte (U2Net) → body/background split; background recedes (aerial perspective)".into());
        // The background can go coarser than the default when the body/face carry the structure — a calmer ground.
        ((Some(200)), Some(104), 0.28, (armature as f32 * 0.6).round().max(36.0) as u32)
    } else {
        notes.push("no face → uniform coarse armature".into());
        (None, None, 0.0, armature)
    };

    // VALUE KEY from measured contrast: a flat, foggy reference (low stddev) needs more tonal expansion for real
    // darks and lights; a punchy reference needs little. Map stddev∈[~0.10,0.28] → value_key∈[0.9,0.2].
    let value_key = ((0.28 - a.luma_stddev) / (0.28 - 0.10) * 0.7 + 0.2).clamp(0.0, 0.95);
    notes.push(format!("luma σ {:.3} → value-key {:.2} (expand a flat reference's tonal range)", a.luma_stddev, value_key));

    // Reserve the paper for surface-white media so lights read as paper, not paint.
    let reserve = a.surface_white.then_some(0.9);
    if reserve.is_some() {
        notes.push("surface-white medium → reserve 0.90 (keep the paper for the lights)".into());
    }

    // Budget scales with area so density is consistent; capped so a big canvas doesn't run away.
    let area = a.short_side as u64 * a.long_side as u64;
    let budget = ((area / 40) as usize).clamp(4000, 14000);
    notes.push(format!("budget {budget} strokes (area-scaled)"));

    PaintPlan {
        medium: a.medium.clone(),
        palette: a.palette.clone(),
        style: "impressionist".into(),
        armature,
        armature_face,
        armature_body,
        recede,
        value_key,
        reserve,
        budget: Some(budget),
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_reference_gets_more_value_key_than_punchy() {
        let base = Analysis { faces: 1, luma_stddev: 0.12, medium: "watercolour".into(), palette: "image".into(), short_side: 512, long_side: 682, surface_white: true };
        let flat = plan_from(&base);
        let punchy = plan_from(&Analysis { luma_stddev: 0.26, ..base_like(&base) });
        assert!(flat.value_key > punchy.value_key, "a flat reference is keyed harder ({} vs {})", flat.value_key, punchy.value_key);
        assert_eq!(flat.armature_face, Some(200), "a detected face gets a focal armature");
        assert_eq!(flat.armature_body, Some(104), "a subject gets a mid body armature (three-tier)");
        assert!(flat.recede > 0.0 && flat.armature < 72, "background recedes and goes coarser with a subject");
        assert_eq!(flat.reserve, Some(0.9), "watercolour reserves the paper");
    }

    #[test]
    fn no_face_means_uniform_armature() {
        let a = Analysis { faces: 0, luma_stddev: 0.2, medium: "oil-direct".into(), palette: "zorn".into(), short_side: 512, long_side: 512, surface_white: false };
        let plan = plan_from(&a);
        assert_eq!(plan.armature_face, None);
        assert_eq!(plan.armature_body, None, "no subject → no body tier");
        assert_eq!(plan.reserve, None, "an opaque medium does not reserve paper");
    }

    #[test]
    fn hjson_round_trips() {
        let a = Analysis { faces: 1, luma_stddev: 0.15, medium: "watercolour".into(), palette: "image".into(), short_side: 512, long_side: 682, surface_white: true };
        let plan = plan_from(&a);
        let parsed = PaintPlan::parse(&plan.to_hjson()).expect("parses");
        assert_eq!(parsed.armature, plan.armature);
        assert_eq!(parsed.armature_face, plan.armature_face);
        assert_eq!(parsed.medium, "watercolour");
    }

    fn base_like(a: &Analysis) -> Analysis {
        Analysis { faces: a.faces, luma_stddev: a.luma_stddev, medium: a.medium.clone(), palette: a.palette.clone(), short_side: a.short_side, long_side: a.long_side, surface_white: a.surface_white }
    }
}
