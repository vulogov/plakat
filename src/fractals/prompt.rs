//! Prose → `FractalSpec` (RFC FRACTALS-1, Phase 8) — deterministic, offline.
//!
//! Maps a natural-language description to a starting spec by keyword: family, mood
//! (palette), coloring, symmetry, and depth. Not an LLM — it always works offline and is
//! fully reproducible; CLI flags then override any field. (An LLM-backed `prompt::complete`
//! path, like `plakat map`'s parser, is a future enhancement.)

use super::spec::{Coloring, FractalKind, FractalSpec};

/// Build a starting spec from a prose description.
pub fn spec_from_prose(text: &str) -> FractalSpec {
    let t = text.to_lowercase();
    let has = |kw: &str| t.contains(kw);
    let mut s = FractalSpec::default();

    // ── Family (most specific first) ──────────────────────────────────────────
    if has("burning ship") || has("burning-ship") {
        s.kind = FractalKind::BurningShip;
        s.center = [-0.4, -0.5];
    } else if has("mandelbulb") || has("3d") || has("raymarch") {
        s.kind = FractalKind::Raymarch;
    } else if has("mandelbox") {
        s.kind = FractalKind::Raymarch;
        s.raymarch.shape = "mandelbox".into();
    } else if has("menger") || has("sponge") {
        s.kind = FractalKind::Raymarch;
        s.raymarch.shape = "menger".into();
    } else if has("nebula") || has("buddhabrot") {
        s.kind = FractalKind::Buddhabrot;
    } else if has("lorenz") {
        s.kind = FractalKind::Attractor;
        s.attractor.preset = "lorenz".into();
    } else if has("clifford") || has("de jong") || has("dejong") || has("attractor") {
        s.kind = FractalKind::Attractor;
        if has("clifford") { s.attractor.preset = "clifford".into(); }
        else if has("de jong") || has("dejong") { s.attractor.preset = "dejong".into(); }
    } else if has("flame") {
        s.kind = FractalKind::Flame;
    } else if has("fern") {
        s.kind = FractalKind::Ifs;
        s.ifs.preset = "barnsley-fern".into();
    } else if has("koch") || has("snowflake") {
        s.kind = FractalKind::Lsystem;
        s.lsystem.preset = "koch-snowflake".into();
    } else if has("dragon") {
        s.kind = FractalKind::Lsystem;
        s.lsystem.preset = "dragon".into();
    } else if has("plant") || has("tree") || has("bush") || has("branch") {
        s.kind = FractalKind::Lsystem;
        s.lsystem.preset = "plant".into();
    } else if has("hilbert") {
        s.kind = FractalKind::Lsystem;
        s.lsystem.preset = "hilbert".into();
    } else if has("sierpinski") || has("sierpiński") {
        s.kind = FractalKind::Ifs;
        s.ifs.preset = "sierpinski".into();
    } else if has("newton") {
        s.kind = FractalKind::Newton;
        s.power = 3.0;
        s.center = [0.0, 0.0];
        s.zoom = 0.6;
        s.coloring = Coloring::Angle;
    } else if has("tricorn") || has("mandelbar") {
        s.kind = FractalKind::Tricorn;
        s.center = [0.0, 0.0];
    } else if has("phoenix") {
        s.kind = FractalKind::Phoenix;
        s.center = [0.0, 0.0];
    } else if has("julia") {
        s.kind = FractalKind::Julia;
        s.center = [0.0, 0.0];
    }
    // else: Mandelbrot (the default).

    // ── Mood → palette ────────────────────────────────────────────────────────
    let palette = if has("neon") {
        Some("neon")
    } else if has("fiery") || has("fire") || has("lava") || has("molten") || has("hot") || has("ember") {
        Some("fire")
    } else if has("icy") || has("ice") || has("frost") || has("frozen") || has("glacial") {
        Some("ice")
    } else if has("electric") || has("vivid") || has("psychedelic") {
        Some("electric")
    } else if has("pastel") || has("soft") || has("gentle") {
        Some("pastel")
    } else if has("monochrome") || has("grayscale") || has("greyscale") || has("black and white") {
        Some("monochrome")
    } else if has("midnight") || has("cosmic") || has("deep space") || has("nocturnal") {
        Some("midnight")
    } else if has("earth") || has("natural") || has("organic") || has("forest") || has("autumn") {
        Some("earth")
    } else {
        None
    };
    if let Some(p) = palette {
        s.palette.preset = p.into();
    }

    // ── Coloring hints (only for the escape families) ─────────────────────────
    if s.kind.is_escape_time() {
        if has("stripe") || has("banded") || has("bands") {
            s.coloring = Coloring::Stripe;
        } else if has("filament") || has("thin") || has("wireframe") || has("web") {
            s.coloring = Coloring::Distance;
        } else if has("equalized") || has("balanced") {
            s.coloring = Coloring::Histogram;
        }
    }

    // ── Symmetry (flame) ──────────────────────────────────────────────────────
    if s.kind == FractalKind::Flame && (has("symmetric") || has("kaleidoscope") || has("mandala")) {
        s.flame.symmetry = 6;
    }

    // ── Depth / detail ────────────────────────────────────────────────────────
    if has("deep zoom") || has("deep") || has("zoomed") {
        s.zoom *= 40.0;
        s.max_iter = s.max_iter.max(1200);
    }
    if has("intricate") || has("detailed") || has("ornate") || has("complex") {
        s.max_iter = (s.max_iter as f64 * 1.6) as u32;
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_family_and_mood() {
        let s = spec_from_prose("a fiery burning ship, intricate deep zoom");
        assert_eq!(s.kind, FractalKind::BurningShip);
        assert_eq!(s.palette.preset, "fire");
        assert!(s.zoom > 1.0);
        assert!(s.max_iter >= 1200);
    }

    #[test]
    fn icy_julia_with_stripes() {
        let s = spec_from_prose("an icy julia set with stripes");
        assert_eq!(s.kind, FractalKind::Julia);
        assert_eq!(s.palette.preset, "ice");
        assert_eq!(s.coloring, Coloring::Stripe);
    }

    #[test]
    fn cosmic_nebula() {
        let s = spec_from_prose("a cosmic nebula");
        assert_eq!(s.kind, FractalKind::Buddhabrot);
        assert_eq!(s.palette.preset, "midnight");
    }

    #[test]
    fn kaleidoscope_flame_is_symmetric() {
        let s = spec_from_prose("a neon kaleidoscope flame");
        assert_eq!(s.kind, FractalKind::Flame);
        assert_eq!(s.palette.preset, "neon");
        assert_eq!(s.flame.symmetry, 6);
    }

    #[test]
    fn plain_text_defaults_to_mandelbrot() {
        let s = spec_from_prose("something interesting");
        assert_eq!(s.kind, FractalKind::Mandelbrot);
        // Always yields a valid spec.
        assert!(s.validate().is_ok());
    }

    #[test]
    fn mandelbulb_prose() {
        let s = spec_from_prose("a golden 3d mandelbulb sculpture");
        assert_eq!(s.kind, FractalKind::Raymarch);
    }
}
