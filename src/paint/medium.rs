//! Medium profiles and stage-schedule generation (RFC PAINT-1 §6).
//!
//! A stage schedule is not a property of the painter — it is a consequence of the medium's **irreversibility
//! structure**. Declare the invariants (opacity, where white comes from, reversibility, how many stages the
//! medium tolerates, whether it splits into light/shadow families) and the schedule *derives itself*. There
//! are no per-medium code paths; a new medium is a new row of invariants.
//!
//! The load-bearing rule the generator encodes: **the irreversible operation goes last** — opaque highlights
//! in oil, the darkest darks in watercolour.

/// The order value is built in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueDirection {
    DarkToLight,
    LightToDark,
    MidOut,
}

/// How the paint covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opacity {
    Opaque,
    Transparent,
    Mixed,
}

/// Where the lightest value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhiteSource {
    /// An opaque white pigment laid on top (oil, gouache).
    Pigment,
    /// The bare paper/panel, reserved (watercolour, ink).
    Surface,
}

/// How long the paint stays workable — the axis stages are separated along.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Reversibility {
    None,
    Minimal,
    Hours,
    Days,
}

/// Whether the medium is painted as two families (light/shadow) or one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Families {
    Split,
    Unified,
}

/// How marks build value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkModel {
    /// Pigment concentration (a loaded brush).
    Continuous,
    /// Mark density (hatching / stippling).
    Density,
}

/// Whether the canvas finishes uniformly or plane-by-plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishPolicy {
    Uniform,
    BackToFront,
}

/// The invariant set that generates a stage schedule and drives the stroke physics.
#[derive(Clone, Copy, Debug)]
pub struct MediumProfile {
    pub name: &'static str,
    pub value_direction: ValueDirection,
    pub opacity: Opacity,
    pub white_source: WhiteSource,
    pub reversibility: Reversibility,
    /// Max stages before the medium degrades (mud / over-working).
    pub stage_budget: usize,
    /// Pickup coefficient for the brush (0 = no pickup, 1 = strong).
    pub pickup: f32,
    pub families: Families,
    pub subtractive: bool,
    pub mark_model: MarkModel,
    pub finish_policy: FinishPolicy,
}

// ── The medium table (§6.2). P1 executes oil-direct + gouache; the rest are declared for the generator and
//    land as later phases wire their physics. ────────────────────────────────────────────────────────────
pub const OIL_DIRECT: MediumProfile = MediumProfile {
    name: "oil-direct",
    value_direction: ValueDirection::MidOut,
    opacity: Opacity::Opaque,
    white_source: WhiteSource::Pigment,
    reversibility: Reversibility::Hours,
    stage_budget: 1,
    pickup: 0.65,
    families: Families::Split,
    subtractive: true,
    mark_model: MarkModel::Continuous,
    finish_policy: FinishPolicy::Uniform,
};

pub const OIL_INDIRECT: MediumProfile = MediumProfile {
    name: "oil-indirect",
    value_direction: ValueDirection::DarkToLight,
    opacity: Opacity::Mixed,
    white_source: WhiteSource::Pigment,
    reversibility: Reversibility::Days,
    stage_budget: 8,
    pickup: 0.4,
    families: Families::Split,
    subtractive: true,
    mark_model: MarkModel::Continuous,
    finish_policy: FinishPolicy::Uniform,
};

pub const GOUACHE: MediumProfile = MediumProfile {
    name: "gouache",
    value_direction: ValueDirection::MidOut,
    opacity: Opacity::Opaque,
    white_source: WhiteSource::Pigment,
    reversibility: Reversibility::Minimal,
    stage_budget: 3,
    pickup: 0.8,
    families: Families::Split,
    subtractive: false,
    mark_model: MarkModel::Continuous,
    finish_policy: FinishPolicy::BackToFront,
};

pub const WATERCOLOUR: MediumProfile = MediumProfile {
    name: "watercolour",
    value_direction: ValueDirection::LightToDark,
    opacity: Opacity::Transparent,
    white_source: WhiteSource::Surface,
    reversibility: Reversibility::Minimal,
    stage_budget: 3,
    pickup: 0.15,
    families: Families::Unified,
    subtractive: false,
    mark_model: MarkModel::Continuous,
    finish_policy: FinishPolicy::BackToFront,
};

pub const INK_WASH: MediumProfile = MediumProfile {
    name: "ink-wash",
    value_direction: ValueDirection::LightToDark,
    opacity: Opacity::Transparent,
    white_source: WhiteSource::Surface,
    reversibility: Reversibility::None,
    stage_budget: 1,
    pickup: 0.9,
    families: Families::Unified,
    subtractive: false,
    mark_model: MarkModel::Continuous,
    finish_policy: FinishPolicy::BackToFront,
};

pub const PEN_INK: MediumProfile = MediumProfile {
    name: "pen-ink",
    value_direction: ValueDirection::LightToDark,
    opacity: Opacity::Transparent,
    white_source: WhiteSource::Surface,
    reversibility: Reversibility::None,
    stage_budget: 1,
    pickup: 0.0,
    families: Families::Unified,
    subtractive: false,
    mark_model: MarkModel::Density,
    finish_policy: FinishPolicy::Uniform,
};

pub const TEMPERA: MediumProfile = MediumProfile {
    name: "tempera",
    value_direction: ValueDirection::DarkToLight,
    opacity: Opacity::Transparent,
    white_source: WhiteSource::Pigment,
    reversibility: Reversibility::None,
    stage_budget: 40,
    pickup: 0.05,
    families: Families::Split,
    subtractive: false,
    mark_model: MarkModel::Density,
    finish_policy: FinishPolicy::Uniform,
};

/// Every declared medium.
pub const ALL: &[MediumProfile] = &[OIL_DIRECT, OIL_INDIRECT, GOUACHE, WATERCOLOUR, INK_WASH, PEN_INK, TEMPERA];

/// The media P1 can execute (opaque continuous).
pub const P1_EXECUTABLE: &[&str] = &["oil-direct", "gouache"];
/// The media executable through P2 — adds watercolour (reservation) and pen-ink (density).
pub const P2_EXECUTABLE: &[&str] = &["oil-direct", "gouache", "watercolour", "pen-ink"];
/// Every executable medium (P3 adds indirect oil, ink wash, tempera — they reuse the opaque-continuous,
/// transparent-reserve, and density paths respectively; tempera's density marks are monochrome for now).
pub const EXECUTABLE: &[&str] = &["oil-direct", "oil-indirect", "gouache", "watercolour", "ink-wash", "pen-ink", "tempera"];

impl MediumProfile {
    /// Look up a medium by name (case-insensitive).
    pub fn by_name(name: &str) -> Option<MediumProfile> {
        ALL.iter().find(|m| m.name.eq_ignore_ascii_case(name.trim())).copied()
    }
    /// Whether P1 can render this medium.
    pub fn is_p1_executable(&self) -> bool {
        P1_EXECUTABLE.contains(&self.name)
    }
    /// Whether the engine can render this medium.
    pub fn is_executable(&self) -> bool {
        EXECUTABLE.contains(&self.name)
    }
}

/// A stage of a painting — a paint layer separated from its neighbours by a state change (§3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Plan the un-painted whites (surface-white media only).
    ReservePlan,
    /// Lay the toned ground / imprimatura.
    Ground,
    /// Establish the value structure in the medium's build direction.
    Value(ValueDirection),
    /// The thin, transparent shadow family.
    ShadowMass,
    /// The opaque light family.
    LightMass,
    /// A colour pass (dead-colour / glaze), numbered when a medium affords several.
    Colour(usize),
    /// The transition band at the terminator.
    Halftone,
    /// The final small dark/bright notes.
    Accents,
    /// The opaque highlights — the irreversible operation, always last for a pigment-white medium.
    Highlights,
}

impl Stage {
    /// A short stable slug for the score / reports.
    pub fn slug(&self) -> String {
        match self {
            Stage::ReservePlan => "reserve-plan".into(),
            Stage::Ground => "ground".into(),
            Stage::Value(_) => "value".into(),
            Stage::ShadowMass => "shadow-mass".into(),
            Stage::LightMass => "light-mass".into(),
            Stage::Colour(i) => format!("colour-{i}"),
            Stage::Halftone => "halftone".into(),
            Stage::Accents => "accents".into(),
            Stage::Highlights => "highlights".into(),
        }
    }
}

/// Generate the stage schedule for a medium (§6.4). The schedule is a pure function of the invariants.
pub fn generate_schedule(m: &MediumProfile) -> Vec<Stage> {
    let mut stages: Vec<Stage> = Vec::new();
    if m.white_source == WhiteSource::Surface {
        stages.push(Stage::ReservePlan);
    }
    if m.opacity != Opacity::Transparent {
        stages.push(Stage::Ground);
    }
    stages.push(Stage::Value(m.value_direction));
    if m.families == Families::Split {
        stages.push(Stage::ShadowMass);
        stages.push(Stage::LightMass);
    }
    // Colour stages fill whatever the stage budget affords beyond the structural stages already placed.
    let colour = (m.stage_budget as i64 - stages.len() as i64).max(0) as usize;
    for i in 0..colour {
        stages.push(Stage::Colour(i + 1));
    }
    if m.reversibility >= Reversibility::Hours {
        stages.push(Stage::Halftone);
    }
    stages.push(Stage::Accents);
    // The irreversible operation goes last.
    if m.white_source == WhiteSource::Pigment {
        stages.push(Stage::Highlights);
    }
    stages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_and_p1_executability() {
        assert_eq!(MediumProfile::by_name("Oil-Direct").unwrap().name, "oil-direct");
        assert!(OIL_DIRECT.is_p1_executable() && GOUACHE.is_p1_executable());
        assert!(!WATERCOLOUR.is_p1_executable(), "watercolour lands in P2");
        assert!(MediumProfile::by_name("nope").is_none());
    }

    #[test]
    fn surface_white_media_reserve_first() {
        let s = generate_schedule(&WATERCOLOUR);
        assert_eq!(s.first(), Some(&Stage::ReservePlan), "white from paper → plan the reserve before stroke one");
        assert!(!s.contains(&Stage::Ground), "transparent → no opaque ground");
        assert_ne!(s.last(), Some(&Stage::Highlights), "no opaque highlights when white is the surface");
    }

    #[test]
    fn pigment_white_media_end_on_highlights() {
        for m in [OIL_DIRECT, GOUACHE, OIL_INDIRECT] {
            let s = generate_schedule(&m);
            assert_eq!(s.last(), Some(&Stage::Highlights), "{}: irreversible highlights last", m.name);
            assert_eq!(s.first(), Some(&Stage::Ground), "{}: opaque → ground first", m.name);
        }
    }

    #[test]
    fn split_family_media_have_both_masses() {
        let s = generate_schedule(&OIL_DIRECT);
        assert!(s.contains(&Stage::ShadowMass) && s.contains(&Stage::LightMass), "split → two families: {s:?}");
        // Shadow before light (thin darks, then opaque lights).
        let si = s.iter().position(|x| *x == Stage::ShadowMass).unwrap();
        let li = s.iter().position(|x| *x == Stage::LightMass).unwrap();
        assert!(si < li, "shadow mass precedes light mass");
    }

    #[test]
    fn indirect_oil_affords_colour_stages() {
        // stage_budget 8 leaves room for colour passes beyond the structural stages.
        let s = generate_schedule(&OIL_INDIRECT);
        assert!(s.iter().any(|x| matches!(x, Stage::Colour(_))), "big stage budget → colour passes: {s:?}");
        assert!(s.contains(&Stage::Halftone), "reversible (days) → a halftone pass");
    }
}
