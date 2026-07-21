//! Perceptual (CIE Lab) gradient interpolation for escape-time coloring.
//!
//! Linear interpolation in *sRGB* drags saturated hues through muddy grey (red→blue
//! passes through brown); interpolating in **Lab** keeps the path perceptually straight.
//! We convert each stop sRGB→Lab once, `mix` in Lab, then convert back for output.
//! RFC FRACTALS-1, Phase 1 — the `palette 0.7` dependency exists for exactly this.

use anyhow::{Context, Result};
use palette::{IntoColor, Lab, Mix, Srgb};

use super::spec::PaletteSpec;

/// A ready-to-sample gradient: N Lab stops plus a solid interior color.
#[derive(Debug, Clone)]
pub struct Palette {
    stops: Vec<Lab>,
    interior: [u8; 3],
    cycles: f64,
    offset: f64,
    cyclic: bool,
}

/// Parse `#rrggbb` (or `rrggbb`) into 8-bit RGB.
pub fn parse_hex(s: &str) -> Result<[u8; 3]> {
    let h = s.trim().trim_start_matches('#');
    if h.len() != 6 || !h.chars().all(|c| c.is_ascii_hexdigit()) {
        anyhow::bail!("expected a #rrggbb hex color, got {s:?}");
    }
    let r = u8::from_str_radix(&h[0..2], 16).context("hex red")?;
    let g = u8::from_str_radix(&h[2..4], 16).context("hex green")?;
    let b = u8::from_str_radix(&h[4..6], 16).context("hex blue")?;
    Ok([r, g, b])
}

fn rgb_to_lab([r, g, b]: [u8; 3]) -> Lab {
    let srgb = Srgb::new(r, g, b).into_format::<f32>();
    srgb.into_color()
}

fn lab_to_rgb(lab: Lab) -> [u8; 3] {
    let srgb: Srgb = lab.into_color();
    let (r, g, b) = srgb.into_format::<u8>().into_components();
    [r, g, b]
}

/// The built-in named presets → their `#rrggbb` stop lists.
pub fn preset_stops(name: &str) -> Option<Vec<&'static str>> {
    let stops: Vec<&'static str> = match name.trim().to_ascii_lowercase().as_str() {
        "fire" => vec!["#000000", "#3b0a00", "#a81600", "#ff5a00", "#ffce3a", "#fffce0"],
        "ice" => vec!["#000814", "#003559", "#0678be", "#4cc9f0", "#caf0f8", "#ffffff"],
        "electric" => vec!["#000000", "#1a0033", "#5f00ba", "#b100e8", "#ff2fd6", "#eaffff"],
        "neon" => vec!["#05010d", "#ff006e", "#fb5607", "#ffbe0b", "#8338ec", "#3a86ff"],
        "pastel" => vec!["#2b2d42", "#8d99ae", "#edf2f4", "#ffcad4", "#f4acb7", "#9d8189"],
        "monochrome" | "mono" | "grayscale" | "greyscale" => {
            vec!["#000000", "#3a3a3a", "#7a7a7a", "#bcbcbc", "#ffffff"]
        }
        "midnight" => vec!["#03071e", "#001233", "#023e8a", "#0096c7", "#48cae4", "#ade8f4"],
        "earth" => vec!["#1b1a17", "#3e2723", "#795548", "#a1887f", "#c8b6a6", "#f1dec9"],
        _ => return None,
    };
    Some(stops)
}

impl Palette {
    /// Build a sampleable palette from its spec. Explicit `stops` win over `preset`;
    /// an unknown preset name is an error (surfaced early, before the render).
    pub fn from_spec(spec: &PaletteSpec) -> Result<Self> {
        let hex_stops: Vec<[u8; 3]> = if !spec.stops.is_empty() {
            spec.stops
                .iter()
                .map(|s| parse_hex(s))
                .collect::<Result<_>>()?
        } else {
            let names = preset_stops(&spec.preset).with_context(|| {
                format!(
                    "unknown palette preset {:?} (want: fire | ice | electric | neon | pastel | \
                     monochrome | midnight | earth — or supply explicit stops)",
                    spec.preset
                )
            })?;
            names.iter().map(|s| parse_hex(s)).collect::<Result<_>>()?
        };
        if hex_stops.is_empty() {
            anyhow::bail!("palette has no color stops");
        }
        let stops = hex_stops.into_iter().map(rgb_to_lab).collect();
        let interior = parse_hex(&spec.interior)?;
        Ok(Palette {
            stops,
            interior,
            cycles: spec.cycles,
            offset: spec.offset,
            cyclic: spec.cyclic,
        })
    }

    /// The interior (non-escaping) color.
    pub fn interior(&self) -> [u8; 3] {
        self.interior
    }

    /// Sample the gradient at normalized position `t ∈ [0,1]` (the escape fraction),
    /// after applying `cycles`/`offset` and `cyclic` wrap/clamp.
    pub fn sample(&self, t: f64) -> [u8; 3] {
        let mut u = t * self.cycles + self.offset;
        u = if self.cyclic {
            u.rem_euclid(1.0)
        } else {
            u.clamp(0.0, 1.0)
        };
        let n = self.stops.len();
        if n == 1 {
            return lab_to_rgb(self.stops[0]);
        }
        let seg = u * (n - 1) as f64;
        let i = (seg.floor() as usize).min(n - 2);
        let f = (seg - i as f64) as f32;
        let mixed = self.stops[i].mix(self.stops[i + 1], f);
        lab_to_rgb(mixed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hex_forms() {
        assert_eq!(parse_hex("#ff8000").unwrap(), [255, 128, 0]);
        assert_eq!(parse_hex("00ff00").unwrap(), [0, 255, 0]);
        assert!(parse_hex("#fff").is_err());
        assert!(parse_hex("#gggggg").is_err());
    }

    #[test]
    fn endpoints_match_stops() {
        let spec = PaletteSpec {
            stops: vec!["#000000".into(), "#ffffff".into()],
            ..PaletteSpec::default()
        };
        let pal = Palette::from_spec(&spec).unwrap();
        // Lab→sRGB round-trip is exact at the pure black / white endpoints.
        assert_eq!(pal.sample(0.0), [0, 0, 0]);
        assert_eq!(pal.sample(1.0), [255, 255, 255]);
        // The midpoint is a real grey (monotone), not an endpoint.
        let mid = pal.sample(0.5);
        assert!(mid[0] > 100 && mid[0] < 200);
        assert_eq!(mid[0], mid[1]);
        assert_eq!(mid[1], mid[2]);
    }

    #[test]
    fn clamp_vs_cyclic() {
        let spec = PaletteSpec {
            stops: vec!["#000000".into(), "#ffffff".into()],
            cyclic: false,
            ..PaletteSpec::default()
        };
        let clamp = Palette::from_spec(&spec).unwrap();
        assert_eq!(clamp.sample(2.0), [255, 255, 255]); // clamps past the end

        let spec_cyc = PaletteSpec { cyclic: true, ..spec };
        let cyc = Palette::from_spec(&spec_cyc).unwrap();
        assert_eq!(cyc.sample(2.0), cyc.sample(0.0)); // wraps
    }

    #[test]
    fn presets_resolve() {
        for name in ["fire", "ice", "electric", "neon", "pastel", "monochrome", "midnight", "earth"] {
            let spec = PaletteSpec { preset: name.into(), ..PaletteSpec::default() };
            assert!(Palette::from_spec(&spec).is_ok(), "preset {name} failed");
        }
        let bad = PaletteSpec { preset: "nope".into(), stops: vec![], ..PaletteSpec::default() };
        assert!(Palette::from_spec(&bad).is_err());
    }
}
