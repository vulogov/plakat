//! Where in the target zone an artefact "attaches" — the point on the
//! artefact that aligns to the placement point in the zone.
//!
//! Two forms:
//!
//! * **Named anchor**: 9-point grid (`top_left`, `top_center`,
//!   `top_right`, `center_left`, `center`, `center_right`,
//!   `bottom_left`, `bottom_center`, `bottom_right`). Covers the
//!   common cases — trees anchor to the ground (`bottom_center`),
//!   suns float in the middle of the sky (`center`).
//! * **Fractional anchor**: `{ x: 0.3, y: 0.7 }` — coordinates as
//!   fractions of the artefact's own width/height from its top-left.
//!   Useful when none of the 9-point options land where you want
//!   (e.g., a sun whose "natural attach point" is its rays, not its
//!   visual center).
//!
//! Both forms are serializable as the same JSON union via serde
//! untagged enum — strings parse as named anchors, objects parse as
//! fractional.

use anyhow::{Result, bail};
use serde::Deserialize;
use std::str::FromStr;

/// The fractional position on the artefact that aligns to the
/// zone's placement point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anchor {
    /// Fraction from artefact's left edge (0.0 = left, 1.0 = right).
    pub x: f32,
    /// Fraction from artefact's top edge (0.0 = top, 1.0 = bottom).
    pub y: f32,
}

impl Anchor {
    pub const TOP_LEFT: Self = Self { x: 0.0, y: 0.0 };
    pub const TOP_CENTER: Self = Self { x: 0.5, y: 0.0 };
    pub const TOP_RIGHT: Self = Self { x: 1.0, y: 0.0 };
    pub const CENTER_LEFT: Self = Self { x: 0.0, y: 0.5 };
    pub const CENTER: Self = Self { x: 0.5, y: 0.5 };
    pub const CENTER_RIGHT: Self = Self { x: 1.0, y: 0.5 };
    pub const BOTTOM_LEFT: Self = Self { x: 0.0, y: 1.0 };
    pub const BOTTOM_CENTER: Self = Self { x: 0.5, y: 1.0 };
    pub const BOTTOM_RIGHT: Self = Self { x: 1.0, y: 1.0 };

    fn named(name: &str) -> Option<Self> {
        Some(match name {
            "top_left" | "top-left" => Self::TOP_LEFT,
            "top_center" | "top-center" | "top_centre" | "top" => Self::TOP_CENTER,
            "top_right" | "top-right" => Self::TOP_RIGHT,
            "center_left" | "centre_left" | "center-left" | "left" => Self::CENTER_LEFT,
            "center" | "centre" | "middle" => Self::CENTER,
            "center_right" | "centre_right" | "center-right" | "right" => Self::CENTER_RIGHT,
            "bottom_left" | "bottom-left" => Self::BOTTOM_LEFT,
            "bottom_center" | "bottom-center" | "bottom_centre" | "bottom" => {
                Self::BOTTOM_CENTER
            }
            "bottom_right" | "bottom-right" => Self::BOTTOM_RIGHT,
            _ => return None,
        })
    }
}

impl Default for Anchor {
    fn default() -> Self {
        Self::CENTER
    }
}

impl FromStr for Anchor {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::named(s.trim()).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown anchor name {s:?} (try top_left, top_center, top_right, \
                 center_left, center, center_right, bottom_left, bottom_center, \
                 bottom_right)"
            )
        })
    }
}

// Custom serde to accept either `"bottom_center"` or `{ x: 0.5, y: 0.8 }`.
impl<'de> Deserialize<'de> for Anchor {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Named(String),
            Fractional {
                x: f32,
                y: f32,
            },
        }
        match Raw::deserialize(d)? {
            Raw::Named(s) => s.parse().map_err(serde::de::Error::custom),
            Raw::Fractional { x, y } => {
                if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) {
                    return Err(serde::de::Error::custom(format!(
                        "fractional anchor out of [0, 1]: ({x}, {y})"
                    )));
                }
                if !x.is_finite() || !y.is_finite() {
                    return Err(serde::de::Error::custom(
                        "fractional anchor must be finite",
                    ));
                }
                Ok(Anchor { x, y })
            }
        }
    }
}

/// Validate a parsed fractional anchor (separate from serde because
/// the CLI also constructs these).
pub fn validate_fractional(x: f32, y: f32) -> Result<Anchor> {
    if !x.is_finite() || !y.is_finite() {
        bail!("fractional anchor must be finite, got ({x}, {y})");
    }
    if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) {
        bail!("fractional anchor out of [0, 1], got ({x}, {y})");
    }
    Ok(Anchor { x, y })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_named() {
        let a: Anchor = "bottom_center".parse().unwrap();
        assert_eq!(a, Anchor::BOTTOM_CENTER);

        let a: Anchor = "center".parse().unwrap();
        assert_eq!(a, Anchor::CENTER);

        // Hyphens, alternative spellings.
        assert_eq!("top-left".parse::<Anchor>().unwrap(), Anchor::TOP_LEFT);
        assert_eq!("centre".parse::<Anchor>().unwrap(), Anchor::CENTER);
    }

    #[test]
    fn rejects_unknown_anchor_names() {
        assert!("garbage".parse::<Anchor>().is_err());
        assert!("".parse::<Anchor>().is_err());
    }

    #[test]
    fn fractional_validation_rejects_out_of_range() {
        assert!(validate_fractional(0.5, 0.5).is_ok());
        assert!(validate_fractional(1.0, 0.0).is_ok());
        assert!(validate_fractional(-0.1, 0.5).is_err());
        assert!(validate_fractional(0.5, 1.1).is_err());
        assert!(validate_fractional(f32::NAN, 0.5).is_err());
    }
}
