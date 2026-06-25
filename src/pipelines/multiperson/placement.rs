//! Multiperson placement model — convert a per-persona *relative location*
//! (words, not pixels) into a screen region + an orientation phrase.
//!
//! Three independent axes:
//!   * **position** (horizontal): left … center … right → the region's x-centroid
//!   * **distance** (depth): closer / mid / farther → the region's y + scale, so
//!     a "closer" persona is larger and lower (foreground), "farther" smaller and
//!     higher (background)
//!   * **facing** (orientation): front / side / back → a prompt phrase
//!     ("facing the viewer" / "in profile" / "seen from behind"); does not move
//!     the region, it conditions how the persona is drawn within it
//!
//! Parsing is order-insensitive and forgiving: `"left closer front"`,
//! `"right, far, profile"`, `"center back"` all work. Unrecognised tokens are
//! ignored (the LLM auto-placer or a default fills the rest).

use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    Left,
    CenterLeft,
    Center,
    CenterRight,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    Closer,
    Mid,
    Farther,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Facing {
    Front,
    Side,
    Back,
}

/// A fully-resolved placement for one persona.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub position: Position,
    pub distance: Distance,
    pub facing: Facing,
}

impl Default for Placement {
    fn default() -> Self {
        Self { position: Position::Center, distance: Distance::Mid, facing: Facing::Front }
    }
}

impl Position {
    /// Horizontal centroid in [0,1].
    pub fn cx(self) -> f32 {
        match self {
            Position::Left => 0.20,
            Position::CenterLeft => 0.35,
            Position::Center => 0.50,
            Position::CenterRight => 0.65,
            Position::Right => 0.80,
        }
    }
    fn parse(tok: &str) -> Option<Position> {
        Some(match tok {
            "left" | "l" => Position::Left,
            "center-left" | "center_left" | "centerleft" | "left-center" | "midleft" => {
                Position::CenterLeft
            }
            "center" | "centre" | "middle" | "c" | "mid-h" => Position::Center,
            "center-right" | "center_right" | "centerright" | "right-center" | "midright" => {
                Position::CenterRight
            }
            "right" | "r" => Position::Right,
            _ => return None,
        })
    }
}

impl Distance {
    /// Vertical centroid in [0,1] (closer = lower on screen).
    pub fn cy(self) -> f32 {
        match self {
            Distance::Closer => 0.70,
            Distance::Mid => 0.55,
            Distance::Farther => 0.34,
        }
    }
    /// Region half-extent (width, height) as a fraction of the canvas.
    pub fn spread(self) -> (f32, f32) {
        match self {
            Distance::Closer => (0.42, 0.78),  // foreground: large
            Distance::Mid => (0.34, 0.66),
            Distance::Farther => (0.24, 0.50), // background: small
        }
    }
    fn parse(tok: &str) -> Option<Distance> {
        // `back` is handled by Facing (parsed first), so it never reaches here.
        Some(match tok {
            "closer" | "close" | "near" | "nearer" | "foreground" | "fg" => Distance::Closer,
            "mid" | "middle-distance" | "medium" | "default" => Distance::Mid,
            "farther" | "further" | "far" | "distant" | "background" | "bg" => Distance::Farther,
            _ => return None,
        })
    }
}

impl Facing {
    /// Prompt phrase appended to this persona's region prompt.
    pub fn phrase(self) -> &'static str {
        match self {
            Facing::Front => "facing the viewer, front view",
            Facing::Side => "in profile, side view",
            Facing::Back => "seen from behind, back turned to the viewer",
        }
    }
    fn parse(tok: &str) -> Option<Facing> {
        Some(match tok {
            "front" | "facing" | "frontal" | "toward-viewer" | "toward_viewer" => Facing::Front,
            "side" | "profile" | "sideways" => Facing::Side,
            "back" | "behind" | "away" | "back-turned" | "rear" => Facing::Back,
            _ => return None,
        })
    }
}

impl Placement {
    /// Parse a free-form `at:` string (e.g. `"left closer front"`). Order-
    /// insensitive; comma/space separated; unknown tokens ignored. Any axis not
    /// mentioned keeps its default (center / mid / front). `"back"` is treated
    /// as FACING (back-turned), not distance — use `"far"`/`"farther"`/
    /// `"background"` for depth.
    pub fn parse(s: &str) -> Result<Placement> {
        let mut p = Placement::default();
        for raw in s.split([',', ' ', '/', ';']).filter(|t| !t.is_empty()) {
            let tok = raw.trim().to_ascii_lowercase();
            if let Some(pos) = Position::parse(&tok) {
                p.position = pos;
            } else if let Some(face) = Facing::parse(&tok) {
                // Facing wins ties with distance (e.g. "back" → facing).
                p.facing = face;
            } else if let Some(dist) = Distance::parse(&tok) {
                p.distance = dist;
            }
            // unknown tokens are silently ignored
        }
        Ok(p)
    }

    /// Screen region `[x0, y0, x1, y1]` (normalised, clamped to [0,1]).
    pub fn bbox(self) -> [f32; 4] {
        let (sw, sh) = self.distance.spread();
        let (cx, cy) = (self.position.cx(), self.distance.cy());
        let (hw, hh) = (sw * 0.5, sh * 0.5);
        [
            (cx - hw).clamp(0.0, 1.0),
            (cy - hh).clamp(0.0, 1.0),
            (cx + hw).clamp(0.0, 1.0),
            (cy + hh).clamp(0.0, 1.0),
        ]
    }

    /// The orientation phrase for this placement's facing.
    pub fn facing_phrase(self) -> &'static str {
        self.facing.phrase()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_three_axes_order_insensitive() {
        let a = Placement::parse("left closer front").unwrap();
        assert_eq!(a, Placement { position: Position::Left, distance: Distance::Closer, facing: Facing::Front });
        let b = Placement::parse("profile, right, far").unwrap();
        assert_eq!(b, Placement { position: Position::Right, distance: Distance::Farther, facing: Facing::Side });
        // "back" is facing, not distance:
        let c = Placement::parse("center back").unwrap();
        assert_eq!(c.facing, Facing::Back);
        assert_eq!(c.distance, Distance::Mid);
    }

    #[test]
    fn defaults_when_axis_omitted() {
        let p = Placement::parse("left").unwrap();
        assert_eq!(p.position, Position::Left);
        assert_eq!(p.distance, Distance::Mid);
        assert_eq!(p.facing, Facing::Front);
    }

    #[test]
    fn closer_is_lower_larger_than_farther() {
        let close = Placement { position: Position::Center, distance: Distance::Closer, facing: Facing::Front };
        let far = Placement { position: Position::Center, distance: Distance::Farther, facing: Facing::Front };
        let cb = close.bbox();
        let fb = far.bbox();
        // closer centroid is lower on screen (larger y) and the box is taller/wider
        assert!(close.distance.cy() > far.distance.cy());
        assert!((cb[2] - cb[0]) > (fb[2] - fb[0]), "closer wider");
        assert!((cb[3] - cb[1]) > (fb[3] - fb[1]), "closer taller");
    }

    #[test]
    fn position_maps_left_to_right() {
        assert!(Position::Left.cx() < Position::Center.cx());
        assert!(Position::Center.cx() < Position::Right.cx());
    }

    #[test]
    fn facing_phrases_differ() {
        assert_ne!(Facing::Front.phrase(), Facing::Side.phrase());
        assert_ne!(Facing::Side.phrase(), Facing::Back.phrase());
    }

    #[test]
    fn bbox_clamped_in_unit_square() {
        for s in ["left closer", "right closer", "center closer", "left far", "right far"] {
            let b = Placement::parse(s).unwrap().bbox();
            assert!(b.iter().all(|&v| (0.0..=1.0).contains(&v)), "{s}: {b:?}");
            assert!(b[0] < b[2] && b[1] < b[3], "{s}: degenerate {b:?}");
        }
    }
}
