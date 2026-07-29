//! WFLW-98 landmark topology (RFC §10.1, **frozen as topology v1** by the Phase-G gating decision —
//! NOT the RFC's originally-assumed 106-pt InsightFace set; see `Documentation/PERSONA_GATING.md`).
//!
//! This module is the single source of truth for the 98-point index layout and the **named anchor
//! region** vocabulary (§8.2/§10.1) that `anchor.landmark` draws from. The geometry engine, the
//! scorecard probes, and the detail subsystem all resolve anchors through here.
//!
//! Coordinate convention throughout the persona geometry engine: a normalised **face-box** in
//! `[0,1]×[0,1]`, `x` increasing to the **image right**, `y` increasing **downward** (image
//! convention). The face is authored symmetric about `x = 0.5`. "Right"/"left" in landmark names are
//! the **subject's** right/left — the subject's right eye is on the image left (small `x`).

/// Total landmark count (WFLW).
pub const NUM_LANDMARKS: usize = 98;

// --- Index ranges (the frozen WFLW-98 layout) ---

/// Jaw + face outline, ear-to-ear under the chin. `0` = subject's-right temple, `16` = chin bottom
/// centre, `32` = subject's-left temple.
pub const CONTOUR: std::ops::Range<usize> = 0..33;
/// Chin bottom centre (contour midpoint).
pub const CHIN: usize = 16;

/// Subject's-right eyebrow, `33` (outer) .. `41` (inner); `37` ≈ centre.
pub const BROW_RIGHT: std::ops::Range<usize> = 33..42;
/// Subject's-left eyebrow, `42` (inner) .. `50` (outer); `46` ≈ centre.
pub const BROW_LEFT: std::ops::Range<usize> = 42..51;
pub const BROW_RIGHT_OUTER: usize = 33;
pub const BROW_RIGHT_INNER: usize = 41;
pub const BROW_LEFT_INNER: usize = 42;
pub const BROW_LEFT_OUTER: usize = 50;

/// Nose: bridge `51`..`54` (top→down centreline) + base/nostril line `55`..`59`. `57` = tip.
pub const NOSE: std::ops::Range<usize> = 51..60;
pub const NOSE_BRIDGE_TOP: usize = 51;
pub const NOSE_TIP: usize = 57;

/// Subject's-right eye contour, 8 pts. `60` = outer corner, `64` = inner corner.
pub const EYE_RIGHT: std::ops::Range<usize> = 60..68;
/// Subject's-left eye contour, 8 pts. `68` = inner corner, `72` = outer corner.
pub const EYE_LEFT: std::ops::Range<usize> = 68..76;
pub const EYE_RIGHT_OUTER: usize = 60;
pub const EYE_RIGHT_INNER: usize = 64;
pub const EYE_LEFT_INNER: usize = 68;
pub const EYE_LEFT_OUTER: usize = 72;

/// Outer lip contour, 12 pts. `76` = right corner, `82` = left corner (closed loop).
pub const LIP_OUTER: std::ops::Range<usize> = 76..88;
/// Inner lip contour, 8 pts — the **mouth aperture** (§8.7). `88` = right inner corner.
pub const LIP_INNER: std::ops::Range<usize> = 88..96;
pub const MOUTH_CORNER_RIGHT: usize = 76;
pub const MOUTH_CORNER_LEFT: usize = 82;
pub const LIP_INNER_RIGHT: usize = 88;
pub const LIP_INNER_LEFT: usize = 92;

/// Pupil centres (WFLW's two extra points beyond the 68-set).
pub const PUPIL_RIGHT: usize = 96;
pub const PUPIL_LEFT: usize = 97;

/// A named anchor region → the landmark indices whose centroid defines it (§8.2/§10.1).
///
/// Regions with a single index sit *on* a landmark; multi-index regions are a centroid (e.g. a cheek
/// is the pupil↔mouth-corner midpoint). The `offset` field on `Anchor` then nudges in face-normalised
/// units. This is the frozen vocabulary for `anchor.landmark` / `anchor.region`.
pub fn named_region(name: &str) -> Option<&'static [usize]> {
    Some(match name {
        // --- brows ---
        "right-brow-outer" => &[BROW_RIGHT_OUTER],
        "right-brow-inner" => &[BROW_RIGHT_INNER],
        "right-brow-mid" => &[37],
        "left-brow-inner" => &[BROW_LEFT_INNER],
        "left-brow-outer" => &[BROW_LEFT_OUTER],
        "left-brow-mid" => &[46],
        "glabella" => &[BROW_RIGHT_INNER, BROW_LEFT_INNER], // between the inner brows
        "forehead-centre" => &[37, 46],                     // brow centres — offset upward via `offset`
        // --- nose ---
        "nose-bridge" => &[52],
        "nose-tip" => &[NOSE_TIP],
        "septum" => &[NOSE_TIP], // sub-nasal; offset downward
        "right-nostril" => &[55],
        "left-nostril" => &[59],
        // --- eyes ---
        "right-eye-outer" => &[EYE_RIGHT_OUTER],
        "right-eye-inner" => &[EYE_RIGHT_INNER],
        "left-eye-inner" => &[EYE_LEFT_INNER],
        "left-eye-outer" => &[EYE_LEFT_OUTER],
        "right-under-eye" => &[PUPIL_RIGHT], // offset downward
        "left-under-eye" => &[PUPIL_LEFT],
        // --- cheeks / nasolabial (pupil↔mouth-corner geometry) ---
        "right-cheek" => &[PUPIL_RIGHT, MOUTH_CORNER_RIGHT],
        "left-cheek" => &[PUPIL_LEFT, MOUTH_CORNER_LEFT],
        "right-cheekbone" => &[PUPIL_RIGHT, 2], // pupil + outer jaw upper
        "left-cheekbone" => &[PUPIL_LEFT, 30],
        "right-nasolabial-upper" => &[NOSE_TIP, MOUTH_CORNER_RIGHT],
        "left-nasolabial-upper" => &[NOSE_TIP, MOUTH_CORNER_LEFT],
        // --- mouth / philtrum ---
        "philtrum" => &[NOSE_TIP, 79], // sub-nasal to upper-lip centre
        "upper-lip-centre" => &[79],
        "lower-lip-centre" => &[85],
        "right-mouth-corner" => &[MOUTH_CORNER_RIGHT],
        "left-mouth-corner" => &[MOUTH_CORNER_LEFT],
        // --- jaw / chin ---
        "right-jaw-mid" => &[6],
        "left-jaw-mid" => &[26],
        "right-jaw-angle" => &[4],
        "left-jaw-angle" => &[28],
        "chin" | "chin-crease" => &[CHIN],
        // --- ears (approximate; contour temple endpoints, offset outward) ---
        "right-lobe" | "right-helix" => &[0],
        "left-lobe" | "left-helix" => &[32],
        _ => return None,
    })
}

/// Is `name` a recognised anchor region? (lint + `geometry --anchors` use this.)
pub fn is_named_region(name: &str) -> bool {
    named_region(name).is_some()
}

/// The full anchor vocabulary, for `--help`, lint suggestions, and the corpus. Kept in sync with
/// `named_region` by the `anchor_vocab_all_resolve` test.
pub const ANCHOR_VOCAB: &[&str] = &[
    "right-brow-outer", "right-brow-inner", "right-brow-mid",
    "left-brow-inner", "left-brow-outer", "left-brow-mid",
    "glabella", "forehead-centre",
    "nose-bridge", "nose-tip", "septum", "right-nostril", "left-nostril",
    "right-eye-outer", "right-eye-inner", "left-eye-inner", "left-eye-outer",
    "right-under-eye", "left-under-eye",
    "right-cheek", "left-cheek", "right-cheekbone", "left-cheekbone",
    "right-nasolabial-upper", "left-nasolabial-upper",
    "philtrum", "upper-lip-centre", "lower-lip-centre",
    "right-mouth-corner", "left-mouth-corner",
    "right-jaw-mid", "left-jaw-mid", "right-jaw-angle", "left-jaw-angle",
    "chin", "chin-crease",
    "right-lobe", "right-helix", "left-lobe", "left-helix",
];

/// How the 98 points connect when drawn (mesh / wireframe rasterisation, §10.3). Each entry is a
/// contiguous index range and whether it closes into a loop.
pub struct DrawGroup {
    pub name: &'static str,
    pub range: std::ops::Range<usize>,
    pub closed: bool,
}

/// The feature polylines/loops, in draw order. Contour + brows + nose are open polylines; eyes + both
/// lips are closed loops. Pupils are drawn as points, not part of any polyline.
pub const DRAW_GROUPS: &[DrawGroup] = &[
    DrawGroup { name: "contour", range: 0..33, closed: false },
    DrawGroup { name: "brow-right", range: 33..42, closed: false },
    DrawGroup { name: "brow-left", range: 42..51, closed: false },
    DrawGroup { name: "nose-bridge", range: 51..55, closed: false },
    DrawGroup { name: "nose-base", range: 55..60, closed: false },
    DrawGroup { name: "eye-right", range: 60..68, closed: true },
    DrawGroup { name: "eye-left", range: 68..76, closed: true },
    DrawGroup { name: "lip-outer", range: 76..88, closed: true },
    DrawGroup { name: "lip-inner", range: 88..96, closed: true },
];

/// Feature groups that bound a **filled region mask** (§10.3): the polygon of these indices is filled.
/// `face` uses the contour plus the two brow arcs to close the top.
pub const MASK_EYE_RIGHT: std::ops::Range<usize> = 60..68;
pub const MASK_EYE_LEFT: std::ops::Range<usize> = 68..76;
pub const MASK_MOUTH: std::ops::Range<usize> = 76..88;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_groups_cover_the_contours() {
        // Every non-pupil index belongs to exactly one draw group.
        let mut seen = [false; NUM_LANDMARKS];
        for g in DRAW_GROUPS {
            for i in g.range.clone() {
                assert!(!seen[i], "index {i} in two groups");
                seen[i] = true;
            }
        }
        for (i, &s) in seen.iter().enumerate() {
            let is_pupil = i == PUPIL_RIGHT || i == PUPIL_LEFT;
            assert_eq!(s, !is_pupil, "index {i} coverage");
        }
    }

    #[test]
    fn ranges_partition_the_98_points() {
        // Every index 0..98 belongs to exactly one contiguous feature range (+ the two pupils).
        assert_eq!(CONTOUR, 0..33);
        assert_eq!(BROW_RIGHT.start, 33);
        assert_eq!(BROW_LEFT.end, 51);
        assert_eq!(NOSE, 51..60);
        assert_eq!(EYE_RIGHT.start, 60);
        assert_eq!(EYE_LEFT.end, 76);
        assert_eq!(LIP_OUTER, 76..88);
        assert_eq!(LIP_INNER, 88..96);
        assert_eq!(PUPIL_RIGHT, 96);
        assert_eq!(PUPIL_LEFT, 97);
        assert_eq!(PUPIL_LEFT, NUM_LANDMARKS - 1);
    }

    #[test]
    fn corner_indices_are_ordered() {
        assert!(MOUTH_CORNER_RIGHT < MOUTH_CORNER_LEFT);
        assert!(EYE_RIGHT_OUTER < EYE_RIGHT_INNER);
        assert!(BROW_RIGHT_OUTER < BROW_RIGHT_INNER);
    }

    #[test]
    fn anchor_vocab_all_resolve() {
        // Every advertised anchor name resolves, and to in-range indices.
        for &name in ANCHOR_VOCAB {
            let idxs = named_region(name).unwrap_or_else(|| panic!("{name} missing"));
            assert!(!idxs.is_empty(), "{name} empty");
            assert!(idxs.iter().all(|&i| i < NUM_LANDMARKS), "{name} out of range");
        }
        assert!(named_region("not-a-region").is_none());
    }
}
