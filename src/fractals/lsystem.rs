//! L-system — Lindenmayer rewriting + turtle line drawing (RFC FRACTALS-1, Phase 3).
//!
//! An axiom string is rewritten `iterations` times by a set of production rules, then a
//! turtle walks the final string drawing line segments: `F`/`G` draw forward, `f`/`g`
//! move, `+`/`-` turn by the configured angle, `[`/`]` push/pop state, `|` reverse.
//! Other letters (`X`, `Y`, `A`, `B`…) are rewrite-only placeholders. The path is fit to
//! the canvas (aspect-preserving) and colored as a gradient along its length.

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_line_segment_mut;
use std::collections::HashMap;
use std::f64::consts::PI;

use super::palette::Palette;
use super::plot::{Bounds, Fit};
use super::progress::ProgressFn;
use super::spec::{FractalSpec, LsystemSpec};

/// Cap on the expanded string length — the grammar grows exponentially.
const MAX_EXPANDED: usize = 40_000_000;
/// Cap on drawn segments (memory + time).
const MAX_SEGMENTS: usize = 8_000_000;

/// A resolved grammar ready to expand + draw.
#[derive(Debug, Clone)]
pub struct Grammar {
    pub axiom: String,
    pub rules: HashMap<char, String>,
    pub angle_deg: f64,
    pub iterations: u32,
    pub start_deg: f64,
}

/// Resolve the grammar from a spec: an explicit `axiom` wins, else the named preset.
pub fn resolve_grammar(spec: &FractalSpec) -> Result<Grammar> {
    let ls = &spec.lsystem;
    if !ls.axiom.is_empty() {
        let mut rules = HashMap::new();
        for r in &ls.rules {
            let (k, v) = parse_rule(r)?;
            rules.insert(k, v);
        }
        return Ok(Grammar {
            axiom: ls.axiom.clone(),
            rules,
            angle_deg: ls.angle,
            iterations: ls.iterations,
            start_deg: ls.start_angle,
        });
    }
    preset_grammar(&ls.preset, ls).with_context(|| {
        format!(
            "unknown L-system preset {:?} (want: koch | koch-snowflake | sierpinski | dragon | \
             hilbert | gosper | plant | bush — or supply an explicit axiom + rules)",
            ls.preset
        )
    })
}

fn parse_rule(r: &str) -> Result<(char, String)> {
    let (lhs, rhs) = r.split_once('=').with_context(|| format!("rule {r:?} must be `X=...`"))?;
    let lhs = lhs.trim();
    let mut chars = lhs.chars();
    let k = chars.next().with_context(|| format!("rule {r:?} has an empty LHS"))?;
    if chars.next().is_some() {
        anyhow::bail!("rule {r:?} LHS must be a single symbol");
    }
    Ok((k, rhs.trim().to_string()))
}

fn rules_of(pairs: &[(char, &str)]) -> HashMap<char, String> {
    pairs.iter().map(|(k, v)| (*k, v.to_string())).collect()
}

/// The built-in L-system presets. Uses the spec's `angle` / `iterations` / `start_angle`
/// only when the caller overrode the preset's own defaults would be nicer, but here the
/// preset fully specifies the grammar (the spec's `iterations` still applies as depth).
pub fn preset_grammar(name: &str, ls: &LsystemSpec) -> Option<Grammar> {
    // (axiom, rules, angle, default_depth, start_angle)
    let (axiom, rules, angle, depth, start): (&str, Vec<(char, &str)>, f64, u32, f64) =
        match name.trim().to_ascii_lowercase().as_str() {
            "koch" => ("F", vec![('F', "F+F--F+F")], 60.0, 4, 0.0),
            "koch-snowflake" | "snowflake" => {
                ("F--F--F", vec![('F', "F+F--F+F")], 60.0, 4, 0.0)
            }
            "sierpinski" | "sierpinski-triangle" => {
                ("F-G-G", vec![('F', "F-G+F+G-F"), ('G', "GG")], 120.0, 6, 0.0)
            }
            "dragon" => ("F", vec![('F', "F+G"), ('G', "F-G")], 90.0, 12, 0.0),
            "hilbert" => (
                "X",
                vec![('X', "+YF-XFX-FY+"), ('Y', "-XF+YFY+FX-")],
                90.0,
                6,
                0.0,
            ),
            "gosper" => (
                "F",
                vec![('F', "F-G--G+F++FF+G-"), ('G', "+F-GG--G-F++F+G")],
                60.0,
                4,
                0.0,
            ),
            "plant" => ("X", vec![('X', "F+[[X]-X]-F[-FX]+X"), ('F', "FF")], 25.0, 5, 65.0),
            "bush" => ("F", vec![('F', "FF+[+F-F-F]-[-F+F+F]")], 22.5, 4, 90.0),
            _ => return None,
        };
    // The preset carries a sensible default depth; a non-default spec depth overrides it.
    let iterations = if ls.iterations == LsystemSpec::default().iterations {
        depth
    } else {
        ls.iterations
    };
    Some(Grammar {
        axiom: axiom.to_string(),
        rules: rules_of(&rules),
        angle_deg: angle,
        iterations,
        start_deg: start,
    })
}

/// Expand the axiom by the grammar's rules `iterations` times (bounded).
pub fn expand(g: &Grammar) -> Result<String> {
    let mut s = g.axiom.clone();
    for _ in 0..g.iterations {
        let mut next = String::with_capacity(s.len() * 2);
        for ch in s.chars() {
            match g.rules.get(&ch) {
                Some(rep) => next.push_str(rep),
                None => next.push(ch),
            }
            if next.len() > MAX_EXPANDED {
                anyhow::bail!(
                    "L-system expanded past {MAX_EXPANDED} symbols — lower lsystem.iterations"
                );
            }
        }
        s = next;
    }
    Ok(s)
}

/// One drawn line segment in model space.
type Seg = (f32, f32, f32, f32);

/// Walk the turtle over the expanded string, collecting segments + their bounds.
fn turtle(s: &str, g: &Grammar) -> Result<(Vec<Seg>, Bounds)> {
    let ang = g.angle_deg * PI / 180.0;
    let mut heading = g.start_deg * PI / 180.0;
    let (mut x, mut y) = (0.0f64, 0.0f64);
    let mut stack: Vec<(f64, f64, f64)> = Vec::new();
    let mut segs: Vec<Seg> = Vec::new();
    let mut bounds = Bounds::empty();
    bounds.include(x, y);

    for ch in s.chars() {
        match ch {
            'F' | 'G' => {
                let nx = x + heading.cos();
                let ny = y + heading.sin();
                segs.push((x as f32, y as f32, nx as f32, ny as f32));
                x = nx;
                y = ny;
                bounds.include(x, y);
                if segs.len() > MAX_SEGMENTS {
                    anyhow::bail!(
                        "L-system produced more than {MAX_SEGMENTS} segments — lower iterations"
                    );
                }
            }
            'f' | 'g' => {
                x += heading.cos();
                y += heading.sin();
                bounds.include(x, y);
            }
            '+' => heading += ang,
            '-' => heading -= ang,
            '|' => heading += PI,
            '[' => stack.push((x, y, heading)),
            ']' => {
                if let Some((sx, sy, sh)) = stack.pop() {
                    x = sx;
                    y = sy;
                    heading = sh;
                }
            }
            _ => {}
        }
    }
    Ok((segs, bounds))
}

/// Draw one segment, thickened to `width` px by parallel offsets.
fn draw_thick(img: &mut RgbImage, s: Seg, color: Rgb<u8>, width: u32) {
    let (x0, y0, x1, y1) = s;
    if width <= 1 {
        draw_line_segment_mut(img, (x0, y0), (x1, y1), color);
        return;
    }
    let (dx, dy) = (x1 - x0, y1 - y0);
    let len = (dx * dx + dy * dy).sqrt().max(1e-6);
    let (px, py) = (-dy / len, dx / len); // unit perpendicular
    let half = (width as f32 - 1.0) / 2.0;
    let mut k = -half;
    while k <= half + 1e-4 {
        let (ox, oy) = (px * k, py * k);
        draw_line_segment_mut(img, (x0 + ox, y0 + oy), (x1 + ox, y1 + oy), color);
        k += 1.0;
    }
}

/// Render the L-system to a packed `RGB8` buffer.
pub fn render(spec: &FractalSpec, palette: &Palette, prog: ProgressFn) -> Result<Vec<u8>> {
    prog(0, 3);
    let g = resolve_grammar(spec)?;
    let expanded = expand(&g)?;
    prog(1, 3);
    let (segs, bounds) = turtle(&expanded, &g)?;
    if !bounds.is_valid() || segs.is_empty() {
        anyhow::bail!("L-system drew no segments (does the grammar produce F/G?)");
    }
    prog(2, 3);

    let interior = palette.interior();
    let mut img = RgbImage::from_pixel(spec.width, spec.height, Rgb(interior));
    let fit = Fit::new(&bounds, spec.width, spec.height, spec.lsystem.margin, spec.zoom);
    let total = segs.len().max(1) as f64;
    let width = spec.lsystem.line_width.max(1);
    for (i, &(x0, y0, x1, y1)) in segs.iter().enumerate() {
        let (px0, py0) = fit.map_f(x0 as f64, y0 as f64);
        let (px1, py1) = fit.map_f(x1 as f64, y1 as f64);
        let color = Rgb(palette.sample(i as f64 / total));
        draw_thick(&mut img, (px0 as f32, py0 as f32, px1 as f32, py1 as f32), color, width);
    }
    prog(3, 3);
    Ok(img.into_raw())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractals::spec::{FractalKind, PaletteSpec};

    fn spec_for(preset: &str) -> FractalSpec {
        FractalSpec {
            kind: FractalKind::Lsystem,
            width: 96,
            height: 96,
            lsystem: LsystemSpec { preset: preset.into(), ..LsystemSpec::default() },
            ..FractalSpec::default()
        }
    }

    #[test]
    fn all_presets_resolve_and_render() {
        let pal = Palette::from_spec(&PaletteSpec::default()).unwrap();
        for p in ["koch", "koch-snowflake", "sierpinski", "dragon", "hilbert", "gosper", "plant", "bush"] {
            let spec = spec_for(p);
            let buf = render(&spec, &pal, &|_, _| {}).unwrap();
            assert_eq!(buf.len(), 96 * 96 * 3, "{p}");
            assert!(buf.chunks(3).any(|px| px != &buf[0..3]), "{p} drew nothing");
        }
    }

    #[test]
    fn expansion_grows() {
        let g = resolve_grammar(&spec_for("koch")).unwrap();
        let s0 = g.axiom.len();
        let s = expand(&g).unwrap();
        assert!(s.len() > s0);
    }

    #[test]
    fn custom_grammar_parses() {
        let spec = FractalSpec {
            kind: FractalKind::Lsystem,
            width: 64,
            height: 64,
            lsystem: LsystemSpec {
                axiom: "F".into(),
                rules: vec!["F=F+F-F-F+F".into()],
                angle: 90.0,
                iterations: 3,
                ..LsystemSpec::default()
            },
            ..FractalSpec::default()
        };
        let g = resolve_grammar(&spec).unwrap();
        assert_eq!(g.rules.len(), 1);
        let pal = Palette::from_spec(&PaletteSpec::default()).unwrap();
        assert!(render(&spec, &pal, &|_, _| {}).is_ok());
    }

    #[test]
    fn bad_rule_errors() {
        let spec = FractalSpec {
            kind: FractalKind::Lsystem,
            lsystem: LsystemSpec { axiom: "F".into(), rules: vec!["no-equals".into()], ..LsystemSpec::default() },
            ..FractalSpec::default()
        };
        assert!(resolve_grammar(&spec).is_err());
    }
}
