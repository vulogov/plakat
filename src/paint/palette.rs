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
// Extended cabinet — for the scene/mood presets below.
pub const PAYNES_GREY: Pigment = pig!("paynes-grey", 60, 70, 85);
pub const INDIGO: Pigment = pig!("indigo", 34, 40, 68);
pub const DIOXAZINE_PURPLE: Pigment = pig!("dioxazine-purple", 70, 40, 90);
pub const CADMIUM_ORANGE: Pigment = pig!("cadmium-orange", 235, 130, 40);
pub const GOLD_OCHRE: Pigment = pig!("gold-ochre", 210, 160, 70);
pub const SAP_GREEN: Pigment = pig!("sap-green", 90, 130, 60);
pub const OLIVE_GREEN: Pigment = pig!("olive-green", 110, 110, 60);
pub const DEEP_GREEN: Pigment = pig!("deep-green", 30, 70, 45);
pub const TEAL: Pigment = pig!("teal", 30, 120, 120);
pub const SKY_BLUE: Pigment = pig!("sky-blue", 135, 180, 215);
pub const LAVENDER: Pigment = pig!("lavender", 120, 100, 160);
pub const ROSE_MADDER: Pigment = pig!("rose-madder", 220, 150, 150);
pub const SAND: Pigment = pig!("sand", 205, 180, 140);
pub const SLATE: Pigment = pig!("slate", 90, 100, 110);

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

// ── Scene / mood presets ─────────────────────────────────────────────────────────────────────────────────
// Time-of-day and weather/biome palettes: pick the one that matches the LIGHT and MOOD of the scene, and the
// whole painting inherits that harmony. Each spans value (a light and a dark) so it can carry a full image.
/// Early morning — cool, soft, pale first light with a warm touch.
pub const EARLY_MORNING: Palette = Palette { name: "early-morning", pigments: &[SKY_BLUE, LAVENDER, ROSE_MADDER, CERULEAN_BLUE, YELLOW_OCHRE, PAYNES_GREY, TITANIUM_WHITE] };
/// Bright day — clear high-key daylight: clean blues, greens and warm accents.
pub const BRIGHT_DAY: Palette = Palette { name: "bright-day", pigments: &[CERULEAN_BLUE, ULTRAMARINE_BLUE, SAP_GREEN, CADMIUM_YELLOW, CADMIUM_RED, BURNT_SIENNA, TITANIUM_WHITE] };
/// Early evening — golden hour: warm gold and orange over cool shadow.
pub const EARLY_EVENING: Palette = Palette { name: "early-evening", pigments: &[GOLD_OCHRE, CADMIUM_ORANGE, CADMIUM_RED, QUINACRIDONE_ROSE, ULTRAMARINE_BLUE, RAW_UMBER, TITANIUM_WHITE] };
/// Late evening — dim warm dusk sinking into purple.
pub const LATE_EVENING: Palette = Palette { name: "late-evening", pigments: &[DIOXAZINE_PURPLE, ALIZARIN_CRIMSON, BURNT_SIENNA, INDIGO, GOLD_OCHRE, IVORY_BLACK, TITANIUM_WHITE] };
/// Night — deep cool dark: indigo, blue and violet with sparse light.
pub const NIGHT: Palette = Palette { name: "night", pigments: &[INDIGO, ULTRAMARINE_BLUE, PAYNES_GREY, PHTHALO_BLUE, DIOXAZINE_PURPLE, IVORY_BLACK, TITANIUM_WHITE] };
/// Rain — desaturated cool greys, muted blues and greens.
pub const RAIN: Palette = Palette { name: "rain", pigments: &[PAYNES_GREY, SLATE, CERULEAN_BLUE, TERRE_VERTE, RAW_UMBER, IVORY_BLACK, TITANIUM_WHITE] };
/// Storm — dark and dramatic: deep indigo, green-black and a bruised warm.
pub const STORM: Palette = Palette { name: "storm", pigments: &[INDIGO, PAYNES_GREY, DEEP_GREEN, BURNT_SIENNA, DIOXAZINE_PURPLE, IVORY_BLACK, TITANIUM_WHITE] };
/// Desert — warm sand and ochre under a clean sky.
pub const DESERT: Palette = Palette { name: "desert", pigments: &[SAND, GOLD_OCHRE, BURNT_SIENNA, CADMIUM_ORANGE, CERULEAN_BLUE, RAW_UMBER, TITANIUM_WHITE] };
/// Forest — earthy greens and browns.
pub const FOREST: Palette = Palette { name: "forest", pigments: &[SAP_GREEN, OLIVE_GREEN, TERRE_VERTE, BURNT_SIENNA, RAW_UMBER, YELLOW_OCHRE, TITANIUM_WHITE] };
/// Rainforest — saturated humid greens and deep shadow.
pub const RAINFOREST: Palette = Palette { name: "rainforest", pigments: &[VIRIDIAN, SAP_GREEN, DEEP_GREEN, TEAL, GOLD_OCHRE, RAW_UMBER, TITANIUM_WHITE] };
/// Snow — cool whites with blue shadow.
pub const SNOW: Palette = Palette { name: "snow", pigments: &[TITANIUM_WHITE, CERULEAN_BLUE, PAYNES_GREY, LAVENDER, SKY_BLUE, SLATE, IVORY_BLACK] };
/// Vivid colours — saturated primaries and secondaries at full strength.
pub const VIVID_COLORS: Palette = Palette { name: "vivid-colors", pigments: &[CADMIUM_YELLOW, CADMIUM_ORANGE, CADMIUM_RED, QUINACRIDONE_ROSE, ULTRAMARINE_BLUE, PHTHALO_BLUE, VIRIDIAN, TITANIUM_WHITE] };
/// Rich colours — deep jewel tones.
pub const RICH_COLORS: Palette = Palette { name: "rich-colors", pigments: &[ALIZARIN_CRIMSON, DIOXAZINE_PURPLE, PHTHALO_BLUE, VIRIDIAN, BURNT_SIENNA, GOLD_OCHRE, IVORY_BLACK, TITANIUM_WHITE] };
/// Muted colours — greyed, desaturated harmony.
pub const MUTED_COLORS: Palette = Palette { name: "muted-colors", pigments: &[YELLOW_OCHRE, TERRE_VERTE, RAW_UMBER, PAYNES_GREY, ROSE_MADDER, SLATE, TITANIUM_WHITE] };

/// Every built-in palette.
pub const ALL: &[Palette] = &[
    ZORN, SPLIT_PRIMARY, VERDACCIO, EARTH, LIMITED_LANDSCAPE, SUMI,
    EARLY_MORNING, BRIGHT_DAY, EARLY_EVENING, LATE_EVENING, NIGHT, RAIN, STORM,
    DESERT, FOREST, RAINFOREST, SNOW, VIVID_COLORS, RICH_COLORS, MUTED_COLORS,
];

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
