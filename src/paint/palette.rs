//! Named limited palettes (RFC PAINT-1 §8.3). A palette is a small set of pigments (4–8); every mixture in a
//! painting is solved from ONE palette, which is what generates colour harmony for free. Masstones are
//! approximate real-pigment values — good enough for the subtractive mixer; a measured-spectra upgrade is
//! independent of this table.

use crate::paint::pigment::Pigment;

macro_rules! pig {
    ($name:literal, $r:literal, $g:literal, $b:literal) => {
        Pigment { name: $name, masstone: [$r, $g, $b] }
    };
}

// ── The pigment cabinet ──────────────────────────────────────────────────────────────────────────────────
pub const TITANIUM_WHITE: Pigment = pig!("titanium-white", 252, 251, 248);
pub const IVORY_BLACK: Pigment = pig!("ivory-black", 28, 28, 30);
pub const YELLOW_OCHRE: Pigment = pig!("yellow-ochre", 190, 145, 60);
pub const CADMIUM_YELLOW: Pigment = pig!("cadmium-yellow", 250, 200, 20);
pub const LEMON_YELLOW: Pigment = pig!("lemon-yellow", 235, 224, 70);
pub const CADMIUM_RED: Pigment = pig!("cadmium-red", 200, 45, 35);
pub const QUINACRIDONE_ROSE: Pigment = pig!("quinacridone-rose", 170, 30, 80);
pub const ALIZARIN_CRIMSON: Pigment = pig!("alizarin-crimson", 130, 22, 42);
pub const ULTRAMARINE_BLUE: Pigment = pig!("ultramarine-blue", 40, 50, 140);
pub const PHTHALO_BLUE: Pigment = pig!("phthalo-blue", 20, 52, 110);
pub const CERULEAN_BLUE: Pigment = pig!("cerulean-blue", 42, 110, 170);
pub const BURNT_SIENNA: Pigment = pig!("burnt-sienna", 120, 60, 35);
pub const RAW_UMBER: Pigment = pig!("raw-umber", 82, 66, 50);
pub const TERRE_VERTE: Pigment = pig!("terre-verte", 96, 110, 80);
pub const VIRIDIAN: Pigment = pig!("viridian", 22, 110, 90);

/// A named limited palette.
#[derive(Clone, Copy, Debug)]
pub struct Palette {
    pub name: &'static str,
    pub pigments: &'static [Pigment],
}

// ── The palettes ─────────────────────────────────────────────────────────────────────────────────────────
/// Zorn — the classic four: yellow ochre, cadmium red, black, white. A whole figure from a warm limited set.
pub const ZORN: Palette = Palette { name: "zorn", pigments: &[YELLOW_OCHRE, CADMIUM_RED, IVORY_BLACK, TITANIUM_WHITE] };
/// Split-primary — a warm and cool of each primary, plus white. Reaches most of the gamut.
pub const SPLIT_PRIMARY: Palette = Palette {
    name: "split-primary",
    pigments: &[CADMIUM_YELLOW, LEMON_YELLOW, CADMIUM_RED, QUINACRIDONE_ROSE, ULTRAMARINE_BLUE, PHTHALO_BLUE, TITANIUM_WHITE],
};
/// Verdaccio — the tempera greenish underpainting set: ochre, black, green earth, white.
pub const VERDACCIO: Palette = Palette { name: "verdaccio", pigments: &[YELLOW_OCHRE, IVORY_BLACK, TERRE_VERTE, TITANIUM_WHITE] };
/// Earth — the muted earths: raw umber, burnt sienna, yellow ochre, black, white.
pub const EARTH: Palette = Palette { name: "earth", pigments: &[RAW_UMBER, BURNT_SIENNA, YELLOW_OCHRE, IVORY_BLACK, TITANIUM_WHITE] };
/// Limited-landscape — ultramarine, burnt sienna, ochre, cadmium yellow, white. Sky, earth, and their greys.
pub const LIMITED_LANDSCAPE: Palette =
    Palette { name: "limited-landscape", pigments: &[ULTRAMARINE_BLUE, BURNT_SIENNA, YELLOW_OCHRE, CADMIUM_YELLOW, TITANIUM_WHITE] };
/// Sumi — ink and paper: black and white only. For ink wash and grisaille.
pub const SUMI: Palette = Palette { name: "sumi", pigments: &[IVORY_BLACK, TITANIUM_WHITE] };

/// Every built-in palette.
pub const ALL: &[Palette] = &[ZORN, SPLIT_PRIMARY, VERDACCIO, EARTH, LIMITED_LANDSCAPE, SUMI];

impl Palette {
    /// Look a palette up by name (case-insensitive).
    pub fn by_name(name: &str) -> Option<Palette> {
        ALL.iter().find(|p| p.name.eq_ignore_ascii_case(name.trim())).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_is_case_insensitive_and_lists_all() {
        assert_eq!(Palette::by_name("Zorn").unwrap().name, "zorn");
        assert_eq!(Palette::by_name("  sumi ").unwrap().pigments.len(), 2);
        assert!(Palette::by_name("nope").is_none());
        assert!(ALL.len() >= 6, "at least six built-ins");
    }

    #[test]
    fn every_palette_has_a_white_or_light_and_a_dark() {
        use crate::paint::color::srgb_to_lab;
        for p in ALL {
            let ls: Vec<f32> = p.pigments.iter().map(|pig| srgb_to_lab(pig.masstone).l).collect();
            let hi = ls.iter().cloned().fold(0.0_f32, f32::max);
            let lo = ls.iter().cloned().fold(100.0_f32, f32::min);
            assert!(hi - lo > 40.0, "palette {} spans value (hi {hi} lo {lo})", p.name);
        }
    }
}
