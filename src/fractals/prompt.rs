//! Prose → `FractalSpec` (RFC FRACTALS-1, Phase 8) — deterministic, offline.
//!
//! This shapes the **fractal itself** (Track A), NOT an AI scene. It reads *fractal*
//! keywords from the text — family (mandelbrot, julia, flame, fern…), mood → palette
//! (fiery, icy, cosmic…), coloring (stripes, filaments), symmetry, depth — and for any
//! text that names no family it derives a **distinctive fractal from a hash of the words**,
//! so different phrases give different art (never the same default twice). Fully
//! deterministic (no LLM); CLI flags override any field afterward.
//!
//! To paint a *scene* from a description ("a winding forest path"), that text belongs in
//! `--fractal-prompt` with `--fractal-paint` — see `ai_pass`.

use std::f64::consts::TAU;

use super::spec::{Coloring, FractalKind, FractalSpec};

/// A small deterministic string hash (FNV-1a) — seeds the "distinctive fractal from any
/// text" behavior without any RNG.
fn text_hash(t: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in t.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Whether the text explicitly names a fractal family (vs. leaving it to the hash).
pub fn names_a_family(text: &str) -> bool {
    let t = text.to_lowercase();
    [
        "burning ship", "burning-ship", "mandelbulb", "3d", "raymarch", "mandelbox", "menger",
        "sponge", "nebula", "buddhabrot", "lorenz", "clifford", "de jong", "dejong", "attractor",
        "flame", "fern", "koch", "snowflake", "dragon", "plant", "tree", "bush", "branch",
        "hilbert", "sierpinski", "sierpiński", "newton", "tricorn", "mandelbar", "phoenix",
        "julia", "mandelbrot",
    ]
    .iter()
    .any(|k| t.contains(k))
}

/// Build a starting spec from a prose description.
pub fn spec_from_prose(text: &str) -> FractalSpec {
    let t = text.to_lowercase();
    let has = |kw: &str| t.contains(kw);
    let mut s = FractalSpec::default();
    let hash = text_hash(&t);
    s.seed = hash; // stochastic families vary per phrase

    // ── Family (most specific first) ──────────────────────────────────────────
    if has("mandelbrot") {
        s.kind = FractalKind::Mandelbrot;
    } else if has("burning ship") || has("burning-ship") {
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
    } else {
        // No family named → a distinctive connected Julia set derived from the text hash.
        // (Julia constants on the 0.7885·e^{iθ} circle are always connected and pretty, so
        // any phrase yields something worth looking at — and a different one each time.)
        s.kind = FractalKind::Julia;
        let theta = (hash % 1_000_000) as f64 / 1_000_000.0 * TAU;
        s.julia_c = [0.7885 * theta.cos(), 0.7885 * theta.sin()];
        s.center = [0.0, 0.0];
        s.zoom = 1.15;
    }

    // ── Mood → palette (else a hash-picked palette so every phrase gets a color) ──
    const PALETTES: &[&str] =
        &["fire", "ice", "electric", "neon", "pastel", "monochrome", "midnight", "earth"];
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
    s.palette.preset = palette
        .map(str::to_string)
        .unwrap_or_else(|| PALETTES[(hash as usize / 7) % PALETTES.len()].to_string());

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
    fn explicit_mandelbrot_is_honored() {
        let s = spec_from_prose("a classic mandelbrot");
        assert_eq!(s.kind, FractalKind::Mandelbrot);
        assert!(names_a_family("a classic mandelbrot"));
    }

    #[test]
    fn unrecognized_text_gives_a_distinctive_julia() {
        // No family named → a hash-derived Julia (not the identical default Mandelbrot).
        let a = spec_from_prose("winding path in the forest");
        assert_eq!(a.kind, FractalKind::Julia);
        assert!(a.validate().is_ok());
        assert!(!names_a_family("winding path in the forest"));
        // "forest" still steers the palette.
        assert_eq!(a.palette.preset, "earth");
        // Different phrases → different fractals (not a fixed default).
        let b = spec_from_prose("a quiet mountain lake");
        assert_ne!(a.julia_c, b.julia_c);
        assert_ne!(a.seed, b.seed);
        // Deterministic: same phrase → same spec.
        assert_eq!(a, spec_from_prose("winding path in the forest"));
    }

    #[test]
    fn no_palette_keyword_still_gets_a_color() {
        let s = spec_from_prose("abcdef ghijkl");
        assert!(!s.palette.preset.is_empty());
    }

    #[test]
    fn mandelbulb_prose() {
        let s = spec_from_prose("a golden 3d mandelbulb sculpture");
        assert_eq!(s.kind, FractalKind::Raymarch);
    }
}
