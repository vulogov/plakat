//! Runtime form of an artefact placement — what the compositing
//! pipeline actually consumes. Built by combining a library entry
//! (`Artefact`) with per-invocation overrides (zone, scale, offset,
//! anchor, flip, alpha).

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::str::FromStr;

use super::anchor::Anchor;
use super::library::{Artefact, ArtefactLibrary};
use super::zones::{Rect, ZoneOverrides, ZoneRef};

/// A user-facing artefact specification before resolution against the
/// library. The CLI `--artefact NAME[@ZONE[:SCALE]]` grammar and the
/// scenario `artefacts:` field both produce these.
#[derive(Debug, Clone)]
pub struct ArtefactSpec {
    pub name: String,
    /// `None` = use library's natural zone.
    pub zone: Option<ZoneRef>,
    /// Multiplier on `natural_size_pct`. `None` = 1.0.
    pub scale: Option<f32>,
    /// Fractional offset within the zone (dx, dy) as `[−1, 1]` shifts.
    /// `None` = no offset (or auto-stagger if multiple in same zone).
    pub offset: Option<[f32; 2]>,
    /// Overrides the library's natural anchor.
    pub anchor: Option<Anchor>,
    /// Horizontal flip.
    pub flip: bool,
    /// Multiplied into the artefact's per-pixel alpha at composite
    /// time. Range `[0, 1]`. `None` = 1.0 (fully opaque).
    pub alpha: Option<f32>,
}

impl ArtefactSpec {
    pub fn from_name(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            zone: None,
            scale: None,
            offset: None,
            anchor: None,
            flip: false,
            alpha: None,
        }
    }
}

impl FromStr for ArtefactSpec {
    type Err = anyhow::Error;
    /// Grammar:
    ///   * `NAME`                — natural zone, default scale.
    ///   * `NAME@ZONE`           — explicit zone.
    ///   * `NAME@ZONE:SCALE`     — explicit zone + scale.
    ///
    /// Scale is a positive float, multiplied into the library's
    /// `natural_size_pct`. Offset / anchor / flip / alpha require the
    /// full-object form (HJSON scenarios) — they're impractical in
    /// CLI shorthand.
    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.is_empty() {
            bail!("empty artefact spec");
        }
        // Split into `name` and the optional `@zone[:scale]` tail.
        let (name, rest) = match s.split_once('@') {
            Some((n, r)) => (n.trim(), Some(r.trim())),
            None => (s, None),
        };
        if name.is_empty() {
            bail!("artefact spec is missing a name (got {:?})", s);
        }

        let (zone, scale) = match rest {
            Some(r) => {
                // r is either `ZONE` or `ZONE:SCALE`. Splitting on the
                // last ':' is the safe pattern.
                let (zone_s, scale) = match r.rsplit_once(':') {
                    Some((z, sc)) => {
                        let parsed = sc.parse::<f32>().with_context(|| {
                            format!("parsing scale {sc:?} in artefact spec {s:?}")
                        })?;
                        if !parsed.is_finite() || parsed <= 0.0 {
                            bail!(
                                "artefact scale must be positive + finite, got {parsed}"
                            );
                        }
                        (z, Some(parsed))
                    }
                    None => (r, None),
                };
                let zone: ZoneRef = zone_s
                    .parse()
                    .with_context(|| format!("parsing zone {zone_s:?} in artefact spec {s:?}"))?;
                (Some(zone), scale)
            }
            None => (None, None),
        };

        Ok(Self {
            name: name.to_owned(),
            zone,
            scale,
            offset: None,
            anchor: None,
            flip: false,
            alpha: None,
        })
    }
}

/// Full-object form for HJSON scenarios. Accepts every override field.
#[derive(Debug, Clone, Deserialize)]
pub struct ArtefactSpecObject {
    pub name: String,
    #[serde(default)]
    pub zone: Option<ZoneRef>,
    #[serde(default)]
    pub scale: Option<f32>,
    #[serde(default)]
    pub offset: Option<[f32; 2]>,
    #[serde(default)]
    pub anchor: Option<Anchor>,
    #[serde(default)]
    pub flip: bool,
    #[serde(default)]
    pub alpha: Option<f32>,
}

/// HJSON spec for an artefact: either a shorthand string (CLI grammar)
/// or a full object with overrides.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ArtefactSpecEntry {
    Shorthand(String),
    Full(ArtefactSpecObject),
}

impl ArtefactSpecEntry {
    pub fn to_spec(&self) -> Result<ArtefactSpec> {
        match self {
            Self::Shorthand(s) => s.parse(),
            Self::Full(o) => Ok(ArtefactSpec {
                name: o.name.clone(),
                zone: o.zone,
                scale: o.scale,
                offset: o.offset,
                anchor: o.anchor,
                flip: o.flip,
                alpha: o.alpha,
            }),
        }
    }
}

/// An artefact specification resolved against a library entry and an
/// output image's dimensions. This is what the compositor consumes.
#[derive(Debug, Clone)]
pub struct ResolvedArtefact {
    /// The library entry (resolved file path lives here).
    pub artefact: Artefact,
    /// Pixel-coordinate rect of the target zone.
    pub zone: Rect,
    /// Final scale fraction = `library.natural_size_pct * spec.scale.unwrap_or(1.0)`.
    pub scale_fraction: f32,
    /// Final offset (dx, dy) in fractional zone units. Auto-stagger
    /// fills this in when multiple artefacts share a zone and none
    /// supplied an explicit offset.
    pub offset: [f32; 2],
    /// Anchor in use (spec override, else library default).
    pub anchor: Anchor,
    /// Horizontal flip.
    pub flip: bool,
    /// Final alpha multiplier (`spec.alpha.unwrap_or(1.0)`).
    pub alpha: f32,
}

/// Resolve a list of specs against the library + image dimensions.
/// Auto-stagger applies when two or more artefacts share the same
/// zone *and* neither supplied an explicit offset.
pub fn resolve_specs(
    specs: &[ArtefactSpec],
    library: &ArtefactLibrary,
    image_width: u32,
    image_height: u32,
    zone_overrides: &ZoneOverrides,
) -> Result<Vec<ResolvedArtefact>> {
    use std::collections::HashMap;

    // First pass: look up the library entry and apply scale/anchor/flip.
    let mut intermediate: Vec<(ArtefactSpec, Artefact, Rect)> = Vec::with_capacity(specs.len());
    for spec in specs {
        let entry = library
            .get(&spec.name)
            .with_context(|| format!("resolving artefact {:?}", spec.name))?;
        let zone_ref = spec.zone.unwrap_or(entry.natural_zone);
        let rect = zone_ref.resolve(image_width, image_height, zone_overrides);
        if rect.width() == 0 || rect.height() == 0 {
            bail!(
                "artefact {:?}: zone {} resolved to an empty rect on {}x{} canvas",
                spec.name,
                zone_ref.display(),
                image_width,
                image_height
            );
        }
        intermediate.push((spec.clone(), entry.clone(), rect));
    }

    // Auto-stagger pass: group by (zone rect) and distribute horizontal
    // offsets within each group (only entries without an explicit offset).
    let mut groups: HashMap<(u32, u32, u32, u32), Vec<usize>> = HashMap::new();
    for (i, (_spec, _art, rect)) in intermediate.iter().enumerate() {
        let key = (rect.x0, rect.y0, rect.x1, rect.y1);
        groups.entry(key).or_default().push(i);
    }
    let mut auto_offsets: HashMap<usize, [f32; 2]> = HashMap::new();
    for (_, indices) in groups.iter() {
        // Only entries WITHOUT an explicit offset participate.
        let auto: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|i| intermediate[*i].0.offset.is_none())
            .collect();
        if auto.len() < 2 {
            continue;
        }
        // Stagger N artefacts horizontally across the zone:
        //   positions = (-(N-1)/2, ..., -1, 0, 1, ..., (N-1)/2) × step
        // where step is the per-step horizontal fraction (smaller for
        // larger N).
        let n = auto.len() as f32;
        let step = (0.6_f32 / n.max(1.0)).min(0.25); // never wider than ±0.25
        for (rank, &idx) in auto.iter().enumerate() {
            let dx = (rank as f32 - (n - 1.0) / 2.0) * step;
            auto_offsets.insert(idx, [dx, 0.0]);
        }
    }

    // Final pass: build ResolvedArtefact, applying any auto-stagger.
    let mut out: Vec<ResolvedArtefact> = Vec::with_capacity(intermediate.len());
    for (i, (spec, art, rect)) in intermediate.into_iter().enumerate() {
        let scale_mult = spec.scale.unwrap_or(1.0);
        if !scale_mult.is_finite() || scale_mult <= 0.0 {
            bail!(
                "artefact {:?}: scale must be positive + finite, got {}",
                spec.name,
                scale_mult
            );
        }
        let scale_fraction = art.natural_size_pct * scale_mult;
        let offset = spec
            .offset
            .or_else(|| auto_offsets.get(&i).copied())
            .unwrap_or([0.0, 0.0]);
        let anchor = spec.anchor.unwrap_or(art.anchor);
        let alpha = spec.alpha.unwrap_or(1.0);
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            bail!(
                "artefact {:?}: alpha must be finite in [0, 1], got {}",
                spec.name,
                alpha
            );
        }

        out.push(ResolvedArtefact {
            artefact: art,
            zone: rect,
            scale_fraction,
            offset,
            anchor,
            flip: spec.flip,
            alpha,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artefacts::library::Artefact;
    use std::path::PathBuf;

    #[test]
    fn parses_shorthand_grammar() {
        // Bare name.
        let s: ArtefactSpec = "oak".parse().unwrap();
        assert_eq!(s.name, "oak");
        assert!(s.zone.is_none());
        assert!(s.scale.is_none());

        // Name + zone.
        let s: ArtefactSpec = "oak@middle_plan/left".parse().unwrap();
        assert_eq!(s.name, "oak");
        assert!(s.zone.is_some());
        assert!(s.scale.is_none());

        // Name + zone + scale.
        let s: ArtefactSpec = "sun@sky/right:0.6".parse().unwrap();
        assert_eq!(s.name, "sun");
        assert!(s.zone.is_some());
        assert_eq!(s.scale, Some(0.6));
    }

    #[test]
    fn rejects_bad_shorthand() {
        assert!("".parse::<ArtefactSpec>().is_err());
        assert!("@sky".parse::<ArtefactSpec>().is_err()); // no name
        assert!("oak@garbage".parse::<ArtefactSpec>().is_err());
        assert!("oak@sky:negative".parse::<ArtefactSpec>().is_err());
        assert!("oak@sky:-0.5".parse::<ArtefactSpec>().is_err());
        assert!("oak@sky:0".parse::<ArtefactSpec>().is_err()); // not positive
    }

    // Helper: make a fake library with one artefact.
    fn fake_library(name: &str, natural_zone_str: &str) -> ArtefactLibrary {
        let mut lib = ArtefactLibrary {
            root: PathBuf::from("/tmp"),
            artefacts: Default::default(),
            order: vec![name.to_string()],
        };
        let natural_zone: ZoneRef = natural_zone_str.parse().unwrap();
        lib.artefacts.insert(
            name.to_string(),
            Artefact {
                name: name.to_string(),
                category: "test".to_string(),
                path: PathBuf::from(format!("/tmp/{name}.png")),
                natural_zone,
                natural_size_pct: 0.5,
                anchor: Anchor::BOTTOM_CENTER,
                license: None,
                license_url: None,
                tags: vec![],
            },
        );
        lib
    }

    #[test]
    fn resolves_natural_zone_when_unspecified() {
        let lib = fake_library("oak", "middle_plan");
        let specs = vec![ArtefactSpec::from_name("oak")];
        let r = resolve_specs(&specs, &lib, 800, 400, &ZoneOverrides::default()).unwrap();
        assert_eq!(r.len(), 1);
        // middle_plan on 800x400 = (0, 200) → (800, 300)
        assert_eq!(
            r[0].zone,
            Rect {
                x0: 0,
                y0: 200,
                x1: 800,
                y1: 300
            }
        );
    }

    #[test]
    fn explicit_zone_overrides_natural() {
        let lib = fake_library("oak", "middle_plan");
        let specs = vec!["oak@sky/right".parse().unwrap()];
        let r = resolve_specs(&specs, &lib, 900, 400, &ZoneOverrides::default()).unwrap();
        assert_eq!(
            r[0].zone,
            Rect {
                x0: 600,
                y0: 0,
                x1: 900,
                y1: 100
            }
        );
    }

    #[test]
    fn auto_stagger_spreads_two_artefacts_in_same_zone() {
        let lib = fake_library("oak", "middle_plan");
        let specs = vec![
            ArtefactSpec::from_name("oak"),
            ArtefactSpec::from_name("oak"),
        ];
        let r = resolve_specs(&specs, &lib, 800, 400, &ZoneOverrides::default()).unwrap();
        assert_eq!(r.len(), 2);
        // Two unweighted entries in the same zone → first dx negative, second positive.
        assert!(r[0].offset[0] < 0.0, "first should stagger left: {:?}", r[0].offset);
        assert!(r[1].offset[0] > 0.0, "second should stagger right: {:?}", r[1].offset);
        // Equal magnitudes, opposite signs.
        assert!((r[0].offset[0] + r[1].offset[0]).abs() < 1e-5);
    }

    #[test]
    fn explicit_offset_disables_auto_stagger() {
        let lib = fake_library("oak", "middle_plan");
        let mut s1 = ArtefactSpec::from_name("oak");
        let mut s2 = ArtefactSpec::from_name("oak");
        s1.offset = Some([0.1, 0.0]);
        s2.offset = Some([0.2, 0.0]);
        let r = resolve_specs(&[s1, s2], &lib, 800, 400, &ZoneOverrides::default()).unwrap();
        assert_eq!(r[0].offset, [0.1, 0.0]);
        assert_eq!(r[1].offset, [0.2, 0.0]);
    }

    #[test]
    fn unknown_name_errors_with_suggestion() {
        let lib = fake_library("oak", "middle_plan");
        let specs = vec![ArtefactSpec::from_name("oaks")];
        let err = resolve_specs(&specs, &lib, 800, 400, &ZoneOverrides::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("oak"), "got: {err}");
    }
}
