//! The catalogue of media: a name, its years, a title, and its recipe (`assets/photo/<name>.hjson`, shipped in
//! the binary). A medium is tuned in its recipe, never in the code; the chain is the same for all.

/// One medium of the catalogue.
#[derive(Clone, Copy, Debug)]
pub struct Medium {
    pub name: &'static str,
    pub years: &'static str,
    pub title: &'static str,
    pub recipe: &'static str,
}

macro_rules! media {
    ($( $name:literal, $years:literal, $title:literal; )*) => {
        /// Every medium, in the order of the catalogue (oldest first; the modern camera last).
        pub const MEDIA: &[Medium] = &[
            $( Medium { name: $name, years: $years, title: $title, recipe: include_str!(concat!("../../assets/photo/", $name, ".hjson")) }, )*
        ];
    };
}

media! {
    "daguerreotype", "1839–1860", "the silver plate — a direct positive, laterally reversed, the Petzval's draw, the mirror in the shadows, the brass mat";
    "calotype", "1841–1855", "Talbot's paper negative and its salt print — the fibre in every edge, a long exposure in the open air";
    "ambrotype", "1854–1865", "the wet-collodion positive on glass — cream lights, a short scale, the pour in the corners, cased";
    "tintype", "1856–1900", "collodion on lacquered iron — the fairground's portrait, dark and warm, the plate clipped by the snips";
    "albumen", "1855–1895", "the glass negative printed out on albumen paper — the cabinet card, purple-brown on cream, mounted";
    "cyanotype", "1842–1900s", "iron salts in the sun — Prussian blue on paper";
    "platinum", "1880–1920", "the platinotype — the long soft scale in the paper's fibre, the pictorialists' print";
    "ortho-press", "1890–1925", "the orthochromatic press plate — blind to red, the sky white, the lips dark, the magnesium flash";
    "pictorialist", "1890–1915", "the photograph that wanted to be a painting — the soft-focus lens, the glowing lights, the bromoil print";
    "autochrome", "1907–1935", "the Lumières' colour — a mosaic of dyed starch grains, muted, warm, a long exposure";
    "box-brownie", "1900–1940", "the snapshot — a meniscus lens, soft corners, a contact print the size of a hand";
    "kodachrome", "1936–2009", "the slide — saturated reds and blues, clean dark shadows, fine grain";
    "agfacolor", "1936–1970", "the other colour — cooler, softer, the greens toward olive, the family album";
    "ektachrome", "1946–2012", "the E-6 slide — cool and clean, the magazine's page, cyan in the shadows";
    "tri-x", "1954–", "the press and the street — 400 pushed to 1600, the grain a fabric, the blacks blocked";
    "hp5", "1976–", "Ilford's 400 — the longer scale, the documentary and the art school";
    "large-format", "1930–", "the 4×5 press and view camera — a sheet of film, stopped down, the verticals standing";
    "contact-8x10", "1900–", "the 8×10 negative laid on the paper — no enlarger, no grain, every hair";
    "polaroid-sx70", "1972–", "the integral print — the dyes migrating, pastel and warm, the square in its white frame";
    "polaroid-600", "1981–", "the party's instant — the flash in every face, cyan shadows, magenta lights";
    "c-print-70s", "1970s", "the chromogenic print fifty years on — the cyan gone, the print leaning magenta, the blacks lifted";
    "holga", "1982–", "the toy camera — a plastic meniscus, the corners gone, light leaks, cross-processed colour";
    "point-and-shoot", "1985–2005", "the compact — the flash on in every room, the red eyes, the orange date in the corner";
    "disposable", "1986–2010", "the cardboard camera — fixed focus, an 800 film, the flash's hard light and the black beyond it";
    "early-digital", "1999–2005", "the first consumer digital — 1.3 megapixels, the purple fringe, the hard clip, the JPEG blocks";
    "dslr-2008", "2006–2012", "the first affordable DSLR — an APS-C sensor at 800, the kit zoom, the in-camera sharpening";
    "phone-2016", "2014–2018", "the phone before the computational era — everything sharp, the shadows smeared, the HDR";
    "phone-2023", "2020–", "the computational camera — the HDR flattening, the faces smoothed, the portrait mode's cut-out bokeh";
    "cinema-anamorphic", "1953–", "a frame from the widescreen — the oval bokeh, the blue flare, the halation, the letterbox grade";
    "instax", "1998–", "the small instant of the 2010s — pale pastel, the flash always on, pinned to the wall";
    "infrared", "1930s–", "the infrared film — the leaves white as snow, the sky black, the glow in the lights";
    "photo", "2020s", "the modern camera — a full-frame body, the sensor's own noise and demosaic, the aperture's depth, the camera's JPEG; a render made a photograph";
}

/// A medium by name (case-insensitive; "camera", "modern" and "realistic" are "photo").
pub fn by_name(name: &str) -> Option<&'static Medium> {
    let n = name.trim().to_ascii_lowercase();
    let n = match n.as_str() {
        "camera" | "modern" | "realistic" | "real" | "dslr" | "mirrorless" => "photo".to_string(),
        "trix" | "tri_x" => "tri-x".to_string(),
        "polaroid" => "polaroid-sx70".to_string(),
        "phone" | "smartphone" | "iphone" => "phone-2023".to_string(),
        "8x10" => "contact-8x10".to_string(),
        "4x5" => "large-format".to_string(),
        "anamorphic" | "cinema" => "cinema-anamorphic".to_string(),
        "ir" => "infrared".to_string(),
        _ => n,
    };
    MEDIA.iter().find(|m| m.name == n)
}

/// The catalogue as lines for the terminal.
pub fn list() -> Vec<String> {
    MEDIA.iter().map(|m| format!("{:<18} {:<10} {}", m.name, m.years, m.title)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photo::plan::PhotoPlan;

    #[test]
    fn every_medium_s_recipe_parses_with_no_key_twice_and_names_only_dials_the_chain_has() {
        assert!(MEDIA.len() >= 30);
        for m in MEDIA {
            let p = PhotoPlan::parse(m.recipe).unwrap_or_else(|e| panic!("{}: {e}", m.name));
            assert!(!p.prompt.is_empty(), "{} names its prompt", m.name);
            let entries = crate::paint::director::parse_entries(m.recipe);
            let mut seen = std::collections::BTreeSet::new();
            for e in &entries {
                if let crate::paint::director::Entry::Key { key, .. } = e {
                    assert!(seen.insert(key.clone()), "{}: {key} twice", m.name);
                    assert!(PhotoPlan::KEYS.iter().any(|(k, _)| k == key), "{}: {key} is no dial", m.name);
                }
            }
        }
        assert!(by_name("Daguerreotype").is_some());
        assert_eq!(by_name("camera").unwrap().name, "photo");
        assert!(by_name("nosuch").is_none());
        assert_eq!(list().len(), MEDIA.len());
    }

    #[test]
    fn every_medium_renders_a_small_picture_and_the_same_seed_gives_the_same_bytes() {
        let (w, h) = (96u32, 64u32);
        let mut img = image::RgbImage::new(w, h);
        for (x, y, p) in img.enumerate_pixels_mut() {
            let v = ((x * 7 + y * 3) % 97) as f32 / 97.0;
            p.0 = [(v * 255.0) as u8, ((1.0 - v) * 200.0) as u8, ((x as f32 / w as f32) * 255.0) as u8];
        }
        let mut depth = crate::photo::Field::new(w, h, 0.0);
        for y in 0..h {
            for x in 0..w {
                depth.v[(y * w + x) as usize] = 1.0 - y as f32 / h as f32;
            }
        }
        let aux = crate::photo::Aux { depth: Some(depth), ..Default::default() };
        for m in MEDIA {
            let mut plan = PhotoPlan::parse(m.recipe).unwrap();
            plan.megapixels = 0.0; // the medium's size is tested on its own; here the chain
            let a = crate::photo::render(&img, &plan, &aux, 7).unwrap_or_else(|e| panic!("{}: {e}", m.name));
            let b = crate::photo::render(&img, &plan, &aux, 7).unwrap();
            assert_eq!(a.image.as_raw(), b.image.as_raw(), "{}: deterministic", m.name);
            let (ow, oh) = plan.output_size(w, h);
            assert_eq!(a.image.dimensions(), (ow, oh), "{}", m.name);
            assert!(!a.notes.is_empty(), "{}", m.name);
        }
    }
}
