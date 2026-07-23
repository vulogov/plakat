//! Fractal → ControlNet conditioning source for `plakat generate --control-fractal`
//! (RFC FRACTALS-1 → 4.3 "ecosystem", Phase 2).
//!
//! The inverse of `--fractal-paint`: instead of the fractals command painting a scene, the
//! *generate* command renders a fractal and uses its **structure** (canny / lineart / depth,
//! auto per family) as ControlNet conditioning for any prompt.

use anyhow::Result;
use std::path::Path;

use super::spec::{FractalKind, FractalSpec};

/// Resolve a `--control-fractal` argument to a spec. Accepts, in order:
///   * a path to a fractal spec HJSON/JSON file,
///   * a `kind` or `kind:preset` shorthand (e.g. `flame`, `ifs:barnsley-fern`,
///     `raymarch:menger`),
///   * otherwise, prose via the offline keyword mapper.
pub fn resolve(src: &str) -> Result<FractalSpec> {
    let s = src.trim();
    if Path::new(s).is_file() {
        return FractalSpec::load(Path::new(s));
    }
    // `kind` or `kind:preset`
    let (kind_str, preset) = match s.split_once(':') {
        Some((k, p)) => (k, Some(p.trim())),
        None => (s, None),
    };
    if let Ok(kind) = FractalKind::parse(kind_str) {
        let mut spec = FractalSpec { kind, ..FractalSpec::default() };
        if let Some(p) = preset {
            match kind {
                FractalKind::Ifs => spec.ifs.preset = p.to_string(),
                FractalKind::Lsystem => spec.lsystem.preset = p.to_string(),
                FractalKind::Flame => spec.flame.preset = p.to_string(),
                FractalKind::Attractor => spec.attractor.preset = p.to_string(),
                FractalKind::Raymarch => spec.raymarch.shape = p.to_string(),
                _ => {}
            }
        }
        return Ok(spec);
    }
    // Prose fallback (always yields a valid spec).
    Ok(super::prompt::spec_from_prose(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_kind() {
        assert_eq!(resolve("burning-ship").unwrap().kind, FractalKind::BurningShip);
        assert_eq!(resolve("flame").unwrap().kind, FractalKind::Flame);
    }

    #[test]
    fn kind_with_preset() {
        let s = resolve("ifs:barnsley-fern").unwrap();
        assert_eq!(s.kind, FractalKind::Ifs);
        assert_eq!(s.ifs.preset, "barnsley-fern");
        let r = resolve("raymarch:menger").unwrap();
        assert_eq!(r.kind, FractalKind::Raymarch);
        assert_eq!(r.raymarch.shape, "menger");
    }

    #[test]
    fn prose_fallback() {
        // Not a path, not a kind → keyword mapper (icy → julia/ice).
        let s = resolve("an icy julia set").unwrap();
        assert_eq!(s.kind, FractalKind::Julia);
        assert_eq!(s.palette.preset, "ice");
    }
}
