//! The calibration table (RFC §13): schema, HJSON loader, staleness (§13.4), and the committed
//! bootstrap. One table per family in `assets/persona/calibration/<family>.hjson`. Each records the
//! **measurement identity** it was produced under, so a mismatch surfaces a staleness warning rather
//! than a silent wrong answer.

use super::fit::{Grade, ResponseCurve};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// The current environment identity — what a freshly-measured table *would* record. Staleness compares
/// a loaded table against this.
pub const CURRENT_ALIGNER: &str = "pipnet-wflw98";
pub const CURRENT_TOPOLOGY: &str = "wflw-98";
pub const CURRENT_LEXICON: &str = "1.0";

/// What a table was measured under (§13.1/§13.4). Changes here invalidate the table.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MeasurementIdentity {
    pub population: u32,
    pub prompt: String,
    pub sampler: String,
    pub steps: u32,
    pub size: u32,
    pub aligner: String,
    pub topology: String,
    pub lexicon_version: String,
    /// `true` = a bootstrap seed (lexicon defaults + a single-render baseline), NOT a real sweep.
    pub provisional: bool,
}

impl Default for MeasurementIdentity {
    fn default() -> Self {
        MeasurementIdentity {
            population: 0,
            prompt: String::new(),
            sampler: String::new(),
            steps: 0,
            size: 0,
            aligner: CURRENT_ALIGNER.into(),
            topology: CURRENT_TOPOLOGY.into(),
            lexicon_version: CURRENT_LEXICON.into(),
            provisional: true,
        }
    }
}

/// The distribution of one landmark metric in the family's prior population (§13.1). `median` is the
/// meaning of `0.5`; `p5`/`p95` bound the usable range.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Prior {
    pub median: f32,
    pub p5: f32,
    pub p95: f32,
}

impl Prior {
    /// Map a realised metric value to a normalised `[0,1]` scalar against this prior (median→0.5).
    pub fn normalise(&self, metric: f32) -> f32 {
        if metric <= self.median {
            let span = (self.median - self.p5).max(1e-6);
            (0.5 - 0.5 * (self.median - metric) / span).clamp(0.0, 1.0)
        } else {
            let span = (self.p95 - self.median).max(1e-6);
            (0.5 + 0.5 * (metric - self.median) / span).clamp(0.0, 1.0)
        }
    }
}

// --- on-disk DTOs (serde) → runtime types ---

#[derive(Debug, Deserialize)]
struct CurveDto {
    grade: String,
    #[serde(default)]
    slope: f32,
    #[serde(default)]
    variance: f32,
    #[serde(default)]
    samples: Vec<[f32; 2]>,
}

#[derive(Debug, Deserialize)]
struct TableDto {
    family: String,
    identity: MeasurementIdentity,
    #[serde(default)]
    priors: BTreeMap<String, Prior>,
    #[serde(default)]
    curves: BTreeMap<String, CurveDto>,
    #[serde(default)]
    harmonise: BTreeMap<String, f32>,
    #[serde(default)]
    spontaneous_detail_rate: f32,
    #[serde(default)]
    prompted_detail_hit_rate: f32,
}

/// A loaded, per-family calibration table.
#[derive(Debug, Clone)]
pub struct CalibrationTable {
    pub family: String,
    pub identity: MeasurementIdentity,
    pub priors: BTreeMap<String, Prior>,
    pub curves: BTreeMap<String, ResponseCurve>,
    /// Harmonisation strength per detail kind (§13.2).
    pub harmonise: BTreeMap<String, f32>,
    pub spontaneous_detail_rate: f32,
    pub prompted_detail_hit_rate: f32,
}

impl CalibrationTable {
    /// Parse a table from HJSON text.
    pub fn from_hjson(text: &str) -> Result<CalibrationTable> {
        let dto: TableDto = deser_hjson::from_str(text).context("parsing calibration HJSON")?;
        let curves = dto
            .curves
            .into_iter()
            .map(|(k, c)| {
                let grade = Grade::parse(&c.grade).unwrap_or(Grade::Moderate);
                let samples: Vec<(f32, f32)> = c.samples.iter().map(|p| (p[0], p[1])).collect();
                (k, ResponseCurve { samples, slope: c.slope, variance: c.variance, grade })
            })
            .collect();
        Ok(CalibrationTable {
            family: dto.family,
            identity: dto.identity,
            priors: dto.priors,
            curves,
            harmonise: dto.harmonise,
            spontaneous_detail_rate: dto.spontaneous_detail_rate,
            prompted_detail_hit_rate: dto.prompted_detail_hit_rate,
        })
    }

    /// Load the committed table for `family` from the bundled assets, or `None` if none exists.
    pub fn bundled(family: &str) -> Option<CalibrationTable> {
        let text = bundled_text(family)?;
        CalibrationTable::from_hjson(text).ok()
    }

    /// Load from an explicit path.
    pub fn load(path: &Path) -> Result<CalibrationTable> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        CalibrationTable::from_hjson(&text)
    }

    /// The per-family grade for an attribute, falling back to `None` if not calibrated here.
    pub fn grade(&self, attr: &str) -> Option<Grade> {
        self.curves.get(attr).map(|c| c.grade)
    }

    /// Serialise the table back to HJSON (what `persona calibrate` writes). Round-trips through
    /// `from_hjson`.
    pub fn to_hjson(&self) -> String {
        let id = &self.identity;
        let mut s = String::new();
        s.push_str(&format!("{{\n  # PERSONA-1 calibration table — {} (RFC §13). Generated by `persona calibrate`.\n", self.family));
        s.push_str(&format!("  family: {:?}\n\n", self.family));
        s.push_str("  identity: {\n");
        s.push_str(&format!("    population: {}\n", id.population));
        s.push_str(&format!("    prompt: {:?}\n", id.prompt));
        s.push_str(&format!("    sampler: {:?}\n", id.sampler));
        s.push_str(&format!("    steps: {}\n    size: {}\n", id.steps, id.size));
        s.push_str(&format!("    aligner: {:?}\n    topology: {:?}\n    lexicon_version: {:?}\n", id.aligner, id.topology, id.lexicon_version));
        s.push_str(&format!("    provisional: {}\n  }}\n\n", id.provisional));
        s.push_str("  priors: {\n");
        for (k, p) in &self.priors {
            s.push_str(&format!("    {k:?}: {{ median: {:.4}, p5: {:.4}, p95: {:.4} }}\n", p.median, p.p5, p.p95));
        }
        s.push_str("  }\n\n  curves: {\n");
        for (k, c) in &self.curves {
            s.push_str(&format!("    {k:?}: {{\n      grade: {:?}\n      slope: {:.4}\n      variance: {:.4}\n", c.grade.as_str(), c.slope, c.variance));
            let pts: Vec<String> = c.samples.iter().map(|(a, b)| format!("[{a:.3}, {b:.3}]")).collect();
            s.push_str(&format!("      samples: [ {} ]\n    }}\n", pts.join(", ")));
        }
        s.push_str("  }\n\n  harmonise: {\n");
        for (k, v) in &self.harmonise {
            s.push_str(&format!("    {k}: {v:.3}\n"));
        }
        s.push_str("  }\n\n");
        s.push_str(&format!("  spontaneous_detail_rate: {:.3}\n", self.spontaneous_detail_rate));
        s.push_str(&format!("  prompted_detail_hit_rate: {:.3}\n}}\n", self.prompted_detail_hit_rate));
        s
    }

    /// Staleness reasons vs the current environment (§13.4). Empty = fresh. `provisional` always
    /// reports, since a bootstrap is not a real measurement.
    pub fn staleness(&self) -> Vec<String> {
        let mut out = Vec::new();
        let id = &self.identity;
        if id.provisional {
            out.push("provisional bootstrap (not a measured sweep) — run `persona calibrate`".into());
        }
        if id.aligner != CURRENT_ALIGNER {
            out.push(format!("aligner changed ({} → {CURRENT_ALIGNER})", id.aligner));
        }
        if id.topology != CURRENT_TOPOLOGY {
            out.push(format!("landmark topology changed ({} → {CURRENT_TOPOLOGY})", id.topology));
        }
        if id.lexicon_version != CURRENT_LEXICON {
            out.push(format!("lexicon changed ({} → {CURRENT_LEXICON}) — bases/gains may differ", id.lexicon_version));
        }
        out
    }
}

/// The bundled bootstrap tables, by family (`include_str!`). Provisional until a real sweep replaces
/// them. Unknown families fall back to `sdxl`'s shape at the call site.
fn bundled_text(family: &str) -> Option<&'static str> {
    Some(match family {
        "sd15" | "sd21" => include_str!("../../../assets/persona/calibration/sd15.hjson"),
        "sdxl" | "sdxl-lightning" | "pony" => include_str!("../../../assets/persona/calibration/sdxl.hjson"),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prior_maps_median_to_half() {
        let p = Prior { median: 0.37, p5: 0.30, p95: 0.44 };
        assert!((p.normalise(0.37) - 0.5).abs() < 1e-6);
        assert!(p.normalise(0.30) <= 0.01);
        assert!(p.normalise(0.44) >= 0.99);
        assert!(p.normalise(0.5) >= 0.99); // beyond p95 clamps
    }

    #[test]
    fn bundled_tables_parse_and_are_marked_provisional() {
        for fam in ["sd15", "sdxl"] {
            let t = CalibrationTable::bundled(fam).unwrap_or_else(|| panic!("no bundled {fam}"));
            assert_eq!(t.family, fam);
            assert!(!t.priors.is_empty(), "{fam} has priors");
            assert!(!t.curves.is_empty(), "{fam} has curves");
            // bootstrap → staleness reports the provisional flag.
            assert!(t.staleness().iter().any(|s| s.contains("provisional")));
            // grades round-trip.
            assert!(t.grade("eyes.spacing").is_some());
        }
    }

    #[test]
    fn to_hjson_round_trips() {
        let t = CalibrationTable::bundled("sdxl").unwrap();
        let text = t.to_hjson();
        let back = CalibrationTable::from_hjson(&text).expect("re-parse");
        assert_eq!(back.family, t.family);
        assert_eq!(back.priors.len(), t.priors.len());
        assert_eq!(back.curves.len(), t.curves.len());
        assert_eq!(back.grade("eyes.spacing"), t.grade("eyes.spacing"));
        assert!((back.priors["eyes.spacing"].median - t.priors["eyes.spacing"].median).abs() < 1e-3);
    }

    #[test]
    fn staleness_flags_a_topology_change() {
        let text = r#"{ family: "x"
          identity: { aligner: "pipnet-wflw98", topology: "old-106", lexicon_version: "1.0", provisional: false }
          priors: {}
          curves: {} }"#;
        let t = CalibrationTable::from_hjson(text).unwrap();
        assert!(t.staleness().iter().any(|s| s.contains("topology changed")));
        assert!(!t.staleness().iter().any(|s| s.contains("provisional")));
    }
}
