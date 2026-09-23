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
    /// SEMANTIC tiers (RFC §5.2): detect parts (hair/beard) with OWL-ViT and give them their own armature tier —
    /// e.g. a coarse wash for the beard. Runs the detector when true.
    #[serde(default)]
    pub semantic: bool,
    /// FAMILY SEPARATION (RFC §3.3): partition light/shadow masses so the painting reads solid, not washed.
    #[serde(default)]
    pub families: bool,
    /// COMMIT SHADOWS (0..1, RFC §3.3): paint the dark value masses decisively (a solid value backbone).
    #[serde(default)]
    pub commit_shadows: f32,
    /// SILHOUETTE (0..1, RFC §5/§7): mark the subject boundary so a light subject reads by its edge.
    #[serde(default)]
    pub silhouette: f32,
    /// How the silhouette edge is marked: `line` / `colour` / `knife` / `lost` (RFC §7 edge craft).
    #[serde(default)]
    pub silhouette_mode: Option<String>,
    /// SAM precise masks (RFC §5): use MobileSAM for a precise subject/face mask (sharp silhouette, face-shaped focal).
    #[serde(default)]
    pub sam: bool,
    /// Value-key strength (tonal-range expansion) — higher for a flat, low-contrast reference.
    #[serde(default)]
    pub value_key: f32,
    /// Shadow floor (0..0.45): the value re-key lifts the SHADOW family into a narrow band above this value instead
    /// of stretching darks to black (RFC §3.3 — a shadow mass is a solid dark, never crushed).
    #[serde(default = "default_shadow_floor")]
    pub shadow_floor: f32,
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
fn default_shadow_floor() -> f32 {
    0.16
}

fn default_style() -> String {
    // LEGIBLE resolves features on the fine passes (it has a detail tier); IMPRESSIONIST has none, so an
    // analyzer that defaulted to impressionist produced masses-only mush. Legible is the right default; a user
    // who wants loose masses still asks for `--style impressionist` or writes it into the plan.
    "legible".into()
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
            semantic: false,
            families: false,
            commit_shadows: 0.0,
            silhouette: 0.0,
            silhouette_mode: None,
            sam: false,
            value_key: 0.0,
            shadow_floor: 0.16,
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
        if self.semantic {
            o.push_str("semantic: true\n");
        }
        if self.families {
            o.push_str("families: true\n");
        }
        if self.commit_shadows > 1e-3 {
            o.push_str(&format!("commit_shadows: {:.2}\n", self.commit_shadows));
        }
        if self.silhouette > 1e-3 {
            o.push_str(&format!("silhouette: {:.2}\n", self.silhouette));
            o.push_str(&format!("silhouette_mode: {}\n", self.silhouette_mode.as_deref().unwrap_or("line")));
        }
        if self.sam {
            o.push_str("sam: true\n");
        }
        o.push_str(&format!("value_key: {:.2}\n", self.value_key));
        o.push_str(&format!("shadow_floor: {:.2}\n", self.shadow_floor));
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

    // The base armature (structure resolution). With the STRUCTURE-PRESERVING armature (edge-preserving + value
    // masses, not a blur) this can be fairly FINE without tracing — finer retains modelling, and density comes
    // from the budget, not from over-coarsening. (The background goes a bit coarser when a subject is present.)
    let armature = if a.short_side >= 900 { 150 } else { 120 };
    notes.push("structure-preserving armature (value masses + edges), fine enough to keep modelling".into());

    // A detected FACE is the focal region: give it a FINE armature so features stay crisp while the beard / hair /
    // background become washes (RFC §5.2). With a subject present, use a THREE-TIER plan — a coarser background, a
    // mid subject body, and the fine face — plus a touch of aerial recession so the subject advances.
    let (armature_face, armature_body, recede, armature) = if a.faces > 0 {
        notes.push(format!("{} face(s) → three-tier armature: background {}px · body 210px · face 300px", a.faces, (armature as f32 * 0.85).round().max(90.0) as u32));
        notes.push("subject matte (U2Net) → body/background split; background recedes (aerial perspective)".into());
        // The background can go coarser than the default when the body/face carry the structure — a calmer ground.
        notes.push("semantic tiers (OWL-ViT): hair/beard → a slightly coarser tier".into());
        notes.push("SAM precise masks → sharp silhouette + face-shaped focal (finer face armature)".into());
        // FINE armatures build DENSITY: a fine (detailed) armature keeps the restate passes finding structure to
        // paint, so the budget is used and the painting reads rich — not a sparse wash. The structure-preserving
        // armature keeps this from tracing. Mild recession only (heavy recede erases soft periphery).
        // Background stays only a touch coarser than the body (0.85), not the old aggressive 0.62 that starved it.
        // No aerial veil by default: measured on a real scene it lifted mean luma +0.04 and moved the painting
        // AWAY from the target structure (SSIM); the reference's own depth cues carry the recession. `--recede`
        // stays available for a deliberate atmospheric treatment.
        ((Some(300)), Some(210), 0.0, (armature as f32 * 0.85).round().max(90.0) as u32)
    } else {
        notes.push("no face → uniform coarse armature".into());
        (None, None, 0.0, armature)
    };
    let semantic = a.faces > 0;
    // FAMILY SEPARATION — group light/shadow masses so the painting reads SOLID, not a washed photographic average.
    let families = true;
    notes.push("family separation (light/shadow masses) → solid, not washed".into());
    // COMMIT SHADOWS — paint the dark masses decisively (a solid value backbone), the direct fix for a pale wash.
    let commit_shadows = 0.85;
    notes.push("commit shadows 0.85 → decisive dark masses (value backbone, not a wash)".into());
    // SILHOUETTE — OFF by default: an auto-drawn contour line reads as tacked-on on most subjects (it is only
    // wanted deliberately). Opt in with `--silhouette <n>` / `--silhouette-mode`; the plan leaves it disabled.
    let (silhouette, silhouette_mode) = (0.0, None);
    // SAM precise masks when a subject is present — a sharp silhouette and a face-shaped focal region.
    let sam = a.faces > 0;

    // VALUE KEY from measured contrast: a flat, foggy reference (low stddev) needs more tonal expansion for real
    // darks and lights; a punchy reference needs little. Map stddev∈[~0.10,0.28] → value_key∈[0.9,0.2].
    // Expand ONLY a genuinely flat/foggy reference (low luma σ). A normal-contrast reference gets no stretch: the
    // stretch lightened a faithful painting by +0.04 mean luma for nothing — a painter re-keys a fog, not a
    // clear day. σ ≥ 0.20 → 0; σ 0.08 → 0.7.
    let value_key = ((0.20 - a.luma_stddev) / (0.20 - 0.08) * 0.7).clamp(0.0, 0.7);
    notes.push(format!("luma σ {:.3} → value-key {:.2} (expand a flat reference's tonal range)", a.luma_stddev, value_key));
    // SHADOW FLOOR (RFC §3.3): the re-key keeps the shadow family a narrow, LIFTED band — a solid dark mass, not a
    // crush to black. 0.16 ≈ a deep but readable dark; raise for a high-key picture, lower for a nocturne.
    let shadow_floor = 0.16;
    notes.push("shadow floor 0.16 → shadow family lifted off black (solid dark masses, RFC §3.3)".into());

    // Reserve the paper ONLY for the brightest highlights — a lower cutoff starves a light subject (a white
    // beard/shirt) into sparse, washed paper. 0.92 paints the light masses and keeps only the true whites as paper.
    let reserve = a.surface_white.then_some(0.92);
    if reserve.is_some() {
        notes.push("surface-white medium → reserve 0.92 (paper only for true highlights; paint the light masses)".into());
    }

    // Budget scales with area for DENSITY — the earlier low budgets starved the painting into a sparse, washed
    // look. Match a rich block-in (~1 stroke per ~9 px, like the 30k reference on 512²), capped so a big canvas
    // stays sane. This is the single biggest lever the analyzer was getting wrong.
    let area = a.short_side as u64 * a.long_side as u64;
    let budget = ((area / 9) as usize).clamp(12000, 34000);
    notes.push(format!("budget {budget} strokes (dense — area/9, matches a rich block-in)"));

    PaintPlan {
        medium: a.medium.clone(),
        palette: a.palette.clone(),
        style: "legible".into(),
        armature,
        armature_face,
        armature_body,
        recede,
        semantic,
        families,
        commit_shadows,
        silhouette,
        silhouette_mode,
        sam,
        value_key,
        shadow_floor,
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
        assert_eq!(flat.armature_face, Some(300), "a detected face gets a fine focal armature (SAM precise focal)");
        assert_eq!(flat.armature_body, Some(210), "a subject gets a mid body armature (three-tier)");
        // No aerial veil by default (measured: it lifted mean luma +0.04 and hurt structural agreement with the
        // target); the background still goes a touch coarser than the base armature with a subject present.
        assert!(flat.recede == 0.0 && flat.armature < 120, "no default veil; background a touch coarser with a subject ({} / {})", flat.recede, flat.armature);
        assert_eq!(flat.reserve, Some(0.92), "watercolour reserves the paper only for true highlights");
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
