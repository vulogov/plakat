//! Zone definitions for artefact placement.
//!
//! An image is partitioned into a 4×3 grid of named zones:
//!
//! ```text
//!                  left     center    right
//!               +--------+--------+--------+
//!         sky   |        |        |        |  ← 0      .. h/4
//!               +--------+--------+--------+
//!     far_plan  |        |        |        |  ← h/4    .. h/2
//!               +--------+--------+--------+
//!   middle_plan |        |        |        |  ← h/2    .. 3h/4
//!               +--------+--------+--------+
//!    close_plan |        |        |        |  ← 3h/4   .. h
//!               +--------+--------+--------+
//!                 0..w/3  w/3..2w/3  2w/3..w
//! ```
//!
//! Zone references can name a depth band alone (full width), a horizontal
//! band alone (full height), or the intersection of both. Defaults can
//! be overridden per scenario via the `zones:` HJSON field — see
//! [`ZoneOverrides`].
//!
//! v3 plan: replace the rigid grid with depth/segmentation-aware zones
//! derived from the generated image itself (e.g. via MiDaS depth or SAM
//! segmentation). The runtime API here (`ZoneRef::resolve` returning a
//! [`Rect`]) is the abstraction boundary v3 will reuse — only the
//! resolver internals change.

use anyhow::{Result, bail};
use serde::Deserialize;
use std::str::FromStr;

/// Pixel-coordinate rectangle, inclusive of `(x0, y0)`, exclusive of
/// `(x1, y1)`. Same orientation as the `image` crate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl Rect {
    pub fn width(&self) -> u32 {
        self.x1.saturating_sub(self.x0)
    }
    pub fn height(&self) -> u32 {
        self.y1.saturating_sub(self.y0)
    }
}

/// Depth band — vertical extent from "deep background" (sky) to
/// "right in front of you" (close_plan).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Depth {
    Sky,
    FarPlan,
    MiddlePlan,
    ClosePlan,
}

impl Depth {
    fn slug(self) -> &'static str {
        match self {
            Self::Sky => "sky",
            Self::FarPlan => "far_plan",
            Self::MiddlePlan => "middle_plan",
            Self::ClosePlan => "close_plan",
        }
    }
}

/// Horizontal band — left / center / right slice of the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Horizontal {
    Left,
    Center,
    Right,
}

impl Horizontal {
    fn slug(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Center => "center",
            Self::Right => "right",
        }
    }
}

/// Either a single band or an intersection of both. Parsed from
/// strings like `"sky"`, `"middle_plan/left"`, `"center"`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZoneRef {
    pub depth: Option<Depth>,
    pub horizontal: Option<Horizontal>,
}

impl ZoneRef {
    pub fn full() -> Self {
        Self {
            depth: None,
            horizontal: None,
        }
    }

    /// Pretty-print form, mirroring the parse grammar.
    pub fn display(&self) -> String {
        match (self.depth, self.horizontal) {
            (Some(d), Some(h)) => format!("{}/{}", d.slug(), h.slug()),
            (Some(d), None) => d.slug().to_string(),
            (None, Some(h)) => h.slug().to_string(),
            (None, None) => "full".to_string(),
        }
    }

    /// Resolve to pixel-coordinate rect given output dimensions and
    /// optional band-extent overrides.
    pub fn resolve(&self, width: u32, height: u32, overrides: &ZoneOverrides) -> Rect {
        let (y0, y1) = match self.depth {
            Some(d) => overrides.depth_band(d, height),
            None => (0, height),
        };
        let (x0, x1) = match self.horizontal {
            Some(h) => overrides.horizontal_band(h, width),
            None => (0, width),
        };
        Rect { x0, y0, x1, y1 }
    }
}

impl FromStr for ZoneRef {
    type Err = anyhow::Error;
    /// Grammar:
    ///   * `"sky"` | `"far_plan"` | `"middle_plan"` | `"close_plan"`
    ///   * `"left"` | `"center"` | `"right"`
    ///   * `"<depth>/<horizontal>"` (e.g., `"middle_plan/left"`)
    ///   * `"<horizontal>/<depth>"` (reverse order also accepted)
    fn from_str(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split('/').filter(|p| !p.is_empty()).collect();
        if parts.is_empty() {
            bail!("empty zone reference");
        }

        let parse_band = |token: &str| -> (Option<Depth>, Option<Horizontal>) {
            let d = match token {
                "sky" => Some(Depth::Sky),
                "far_plan" | "far" => Some(Depth::FarPlan),
                "middle_plan" | "middle" | "mid" => Some(Depth::MiddlePlan),
                "close_plan" | "close" | "near" => Some(Depth::ClosePlan),
                _ => None,
            };
            if d.is_some() {
                return (d, None);
            }
            let h = match token {
                "left" => Some(Horizontal::Left),
                "center" | "centre" | "middle" => Some(Horizontal::Center),
                "right" => Some(Horizontal::Right),
                _ => None,
            };
            (None, h)
        };

        let mut depth = None;
        let mut horizontal = None;
        for token in parts {
            let (d, h) = parse_band(token);
            if let Some(d) = d {
                if depth.is_some() {
                    bail!("zone reference has two depth bands");
                }
                depth = Some(d);
            } else if let Some(h) = h {
                if horizontal.is_some() {
                    bail!("zone reference has two horizontal bands");
                }
                horizontal = Some(h);
            } else {
                bail!(
                    "unknown zone token {:?} (valid: sky, far_plan, middle_plan, close_plan, left, center, right)",
                    token
                );
            }
        }
        Ok(Self { depth, horizontal })
    }
}

// Custom serde: ZoneRef in HJSON is a string like "middle_plan/left".
impl<'de> Deserialize<'de> for ZoneRef {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Per-zone extent overrides (normalized 0..=1). Missing bands fall
/// back to the rigid default grid.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ZoneOverrides {
    #[serde(default)]
    pub sky: Option<[f32; 2]>,
    #[serde(default, rename = "far_plan")]
    pub far_plan: Option<[f32; 2]>,
    #[serde(default, rename = "middle_plan")]
    pub middle_plan: Option<[f32; 2]>,
    #[serde(default, rename = "close_plan")]
    pub close_plan: Option<[f32; 2]>,
    #[serde(default)]
    pub left: Option<[f32; 2]>,
    #[serde(default)]
    pub center: Option<[f32; 2]>,
    #[serde(default)]
    pub right: Option<[f32; 2]>,
}

impl ZoneOverrides {
    /// Return `(y0_px, y1_px)` for the given depth band on a `height`-pixel image.
    fn depth_band(&self, d: Depth, height: u32) -> (u32, u32) {
        let h = height as f32;
        let raw = match d {
            Depth::Sky => self.sky.unwrap_or([0.0, 0.25]),
            Depth::FarPlan => self.far_plan.unwrap_or([0.25, 0.50]),
            Depth::MiddlePlan => self.middle_plan.unwrap_or([0.50, 0.75]),
            Depth::ClosePlan => self.close_plan.unwrap_or([0.75, 1.00]),
        };
        ((raw[0] * h).round() as u32, (raw[1] * h).round().min(h) as u32)
    }

    /// Return `(x0_px, x1_px)` for the given horizontal band on a `width`-pixel image.
    fn horizontal_band(&self, h: Horizontal, width: u32) -> (u32, u32) {
        let w = width as f32;
        let raw = match h {
            Horizontal::Left => self.left.unwrap_or([0.0, 1.0 / 3.0]),
            Horizontal::Center => self.center.unwrap_or([1.0 / 3.0, 2.0 / 3.0]),
            Horizontal::Right => self.right.unwrap_or([2.0 / 3.0, 1.0]),
        };
        ((raw[0] * w).round() as u32, (raw[1] * w).round().min(w) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_named_zones() {
        let z: ZoneRef = "sky".parse().unwrap();
        assert_eq!(z.depth, Some(Depth::Sky));
        assert_eq!(z.horizontal, None);

        let z: ZoneRef = "middle_plan/left".parse().unwrap();
        assert_eq!(z.depth, Some(Depth::MiddlePlan));
        assert_eq!(z.horizontal, Some(Horizontal::Left));

        // Reverse order also works.
        let z: ZoneRef = "right/close_plan".parse().unwrap();
        assert_eq!(z.depth, Some(Depth::ClosePlan));
        assert_eq!(z.horizontal, Some(Horizontal::Right));

        // Bare horizontal band.
        let z: ZoneRef = "center".parse().unwrap();
        assert_eq!(z.depth, None);
        assert_eq!(z.horizontal, Some(Horizontal::Center));
    }

    #[test]
    fn rejects_invalid_zones() {
        assert!("garbage".parse::<ZoneRef>().is_err());
        assert!("sky/far_plan".parse::<ZoneRef>().is_err()); // two depths
        assert!("left/right".parse::<ZoneRef>().is_err()); // two horizontals
        assert!("".parse::<ZoneRef>().is_err());
    }

    #[test]
    fn resolves_default_grid_rects() {
        let ov = ZoneOverrides::default();

        // sky on 800x400 → (0, 0) to (800, 100)
        let z: ZoneRef = "sky".parse().unwrap();
        let r = z.resolve(800, 400, &ov);
        assert_eq!(r, Rect { x0: 0, y0: 0, x1: 800, y1: 100 });

        // middle_plan/left on 600x400 → (0, 200) to (200, 300)
        let z: ZoneRef = "middle_plan/left".parse().unwrap();
        let r = z.resolve(600, 400, &ov);
        assert_eq!(r, Rect { x0: 0, y0: 200, x1: 200, y1: 300 });

        // close_plan/right on 900x600 → (600, 450) to (900, 600)
        let z: ZoneRef = "close_plan/right".parse().unwrap();
        let r = z.resolve(900, 600, &ov);
        assert_eq!(r, Rect { x0: 600, y0: 450, x1: 900, y1: 600 });
    }

    #[test]
    fn override_changes_band_extent() {
        let ov = ZoneOverrides {
            sky: Some([0.0, 0.40]),
            ..Default::default()
        };
        let z: ZoneRef = "sky".parse().unwrap();
        let r = z.resolve(800, 400, &ov);
        assert_eq!(r, Rect { x0: 0, y0: 0, x1: 800, y1: 160 });
    }
}
