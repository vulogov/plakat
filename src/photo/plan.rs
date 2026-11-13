//! The photographic plan — every dial of the chain, one key each, read from and written to hjson. A medium's
//! recipe sets them; the director adjusts them with a reason; the command line overrides them. 0 (or "none",
//! or an empty string) is OFF for every stage that can be off, so the all-zero plan passes a picture through
//! unchanged but for the rounding of linear light.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A dial's text form (for the plan file, the report and the director's moves).
pub trait Dial: Sized {
    fn to_text(&self) -> String;
    fn from_text(s: &str) -> anyhow::Result<Self>;
}

impl Dial for f32 {
    fn to_text(&self) -> String {
        if (self.fract()).abs() < 1e-6 && self.abs() < 1e6 {
            format!("{}", *self as i64)
        } else {
            let s = format!("{:.4}", self);
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        }
    }
    fn from_text(s: &str) -> anyhow::Result<Self> {
        s.trim().parse::<f32>().map_err(|e| anyhow::anyhow!("{s:?} is not a number: {e}"))
    }
}

impl Dial for String {
    fn to_text(&self) -> String {
        format!("\"{}\"", self.replace('"', "'"))
    }
    fn from_text(s: &str) -> anyhow::Result<Self> {
        Ok(s.trim().trim_matches('"').to_string())
    }
}

impl Dial for Vec<f32> {
    fn to_text(&self) -> String {
        format!("[{}]", self.iter().map(|v| v.to_text()).collect::<Vec<_>>().join(", "))
    }
    fn from_text(s: &str) -> anyhow::Result<Self> {
        s.trim().trim_start_matches('[').trim_end_matches(']').split(',').filter(|t| !t.trim().is_empty()).map(|t| f32::from_text(t)).collect()
    }
}

macro_rules! plan_fields {
    ($( $(#[doc = $doc:expr])* $name:ident : $ty:ty = $default:expr ; )*) => {
        /// The plan: see the module doc. Each field's doc is its line in `make-photo plan --keys`.
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(default, deny_unknown_fields)]
        pub struct PhotoPlan {
            $( $(#[doc = $doc])* pub $name: $ty, )*
        }

        impl Default for PhotoPlan {
            fn default() -> Self {
                PhotoPlan { $( $name: $default.into(), )* }
            }
        }

        impl PhotoPlan {
            /// Every key with its doc, in the chain's order.
            pub const KEYS: &'static [(&'static str, &'static str)] = &[
                $( (stringify!($name), concat!($($doc, "\n",)*)), )*
            ];

            /// A dial's value as text.
            pub fn get(&self, key: &str) -> Option<String> {
                match key {
                    $( stringify!($name) => Some(Dial::to_text(&self.$name)), )*
                    _ => None,
                }
            }

            /// Set a dial from text (the director's moves, the command line).
            pub fn set(&mut self, key: &str, value: &str) -> anyhow::Result<()> {
                match key {
                    $( stringify!($name) => { self.$name = Dial::from_text(value)?; Ok(()) } )*
                    _ => anyhow::bail!("no such key {key:?}"),
                }
            }

            /// A dial as a number, if it is one.
            pub fn num(&self, key: &str) -> Option<f32> {
                self.get(key).and_then(|v| v.parse::<f32>().ok())
            }
        }
    };
}

plan_fields! {
    // ---- THE CAMERA ----
    /// The lens to re-photograph with, in 35 mm-equivalent millimetres; 0 = the picture's own perspective (no
    /// move). With `keep_subject` the camera dollies so the subject keeps its size — the 24→85 portrait.
    focal: f32 = 0.0;
    /// The lens the picture was taken with (35 mm-equivalent), the perspective the re-projection starts from;
    /// 0 = by the genre (the director) or 35.
    source_focal: f32 = 0.0;
    /// The distance to the subject in metres, the scene's scale for a move; 0 = read from a face (the eyes 63 mm
    /// apart), else by the genre.
    distance: f32 = 0.0;
    /// The scene's depth as a ratio, the far ground over the near: the model's relative depth is stretched over
    /// it (a studio portrait spans 4, a street 20, a landscape 50); 0 = by the genre (the director), else 20.
    depth_range: f32 = 0.0;
    /// The camera moved toward the subject (+) or back (−), metres.
    dolly: f32 = 0.0;
    /// The camera moved to the right (+) or the left (−), metres.
    truck: f32 = 0.0;
    /// The camera moved up (+) or down (−), metres.
    pedestal: f32 = 0.0;
    /// The camera turned to the right (+) or the left (−), degrees.
    pan: f32 = 0.0;
    /// The camera tilted up (+) or down (−), degrees (the keystone of an architecture shot).
    tilt: f32 = 0.0;
    /// The camera rolled clockwise (+), degrees.
    roll: f32 = 0.0;
    /// The lens shifted (rise + / fall −) as a fraction of the frame's height: the verticals kept vertical while
    /// the framing climbs — the architectural camera.
    shift: f32 = 0.0;
    /// 1 = `focal` and `dolly` tied by the subject's size: give one and the other follows.
    keep_subject: f32 = 0.0;
    /// How the ground a move opens is filled: "propagate" (the background grown into the hole, weight-free),
    /// "neural" (inpainted by a model, disclosed), "none" (left black).
    fill: String = "propagate";
    /// The largest share of the frame a move may invent; the critic scales a move down to it.
    fill_cap: f32 = 0.08;
    // ---- THE LENS ----
    /// The f-number; 0 = a pinhole, everything sharp. The depth of field follows from it, the focal length and
    /// the focus distance through the circle of confusion on the format's sensor.
    aperture: f32 = 0.0;
    /// Where the focus is: "auto" (the eyes, else the subject, else the centre), "subject", "eyes", "near",
    /// "far", "centre", or "x,y" in frame fractions.
    focus: String = "auto";
    /// The focus distance in metres, overriding `focus`; 0 = from `focus`.
    focus_distance: f32 = 0.0;
    /// The field's curvature: the edges fall out of focus (0..1).
    field_curve: f32 = 0.0;
    /// The vignette (0..1): the cos⁴ fall-off and the mechanical darkening of the corners.
    vignette: f32 = 0.0;
    /// The distortion: barrel (+) or pincushion (−), about −0.3..0.3.
    distort: f32 = 0.0;
    /// The lateral chromatic aberration: the red and blue images scaled apart toward the corners (0..1).
    ca: f32 = 0.0;
    /// The purple fringe of an early digital sensor at the clipped edges (0..1).
    fringe: f32 = 0.0;
    /// The veiling flare and the bloom of the highlights (0..1).
    flare: f32 = 0.0;
    /// The soft focus of a pictorial lens: the sharp image under a glowing blurred one (0..1).
    soft: f32 = 0.0;
    /// The swirl of a Petzval: the blur turning tangential toward the edges (0..1).
    swirl: f32 = 0.0;
    /// The anamorphic squeeze: the bokeh ovals standing, the flare a horizontal streak (0..1).
    anamorphic: f32 = 0.0;
    /// The diffraction at a small aperture: 1 = physical (the Airy disc on the format's pixel pitch).
    diffraction: f32 = 0.0;
    /// The aerial haze: the far ground lifted toward the sky's colour by its depth (0..1).
    haze: f32 = 0.0;
    /// The lens's and the emulsion's resolving power as a blur, in pixels at a 1000-pixel long side (0 = perfect).
    resolution: f32 = 0.0;
    // ---- THE SHUTTER ----
    /// The exposure time in seconds; over 1/30 the camera shakes, over 1 the water and the foliage smear, over 4
    /// the people ghost.
    exposure_time: f32 = 0.0;
    /// The hand's shake as a blur (0..1), scaled by the exposure time.
    shake: f32 = 0.0;
    /// The moving things (water, foliage, sky) smeared by a long exposure (0..1).
    motion: f32 = 0.0;
    /// The people's ghosting on a long exposure: the figures half there (0..1).
    ghost: f32 = 0.0;
    // ---- THE EMULSION / THE SENSOR ----
    /// The exposure correction in stops.
    exposure: f32 = 0.0;
    /// The film speed; 0 = the medium's base. The grain and the sensor's noise scale with it.
    iso: f32 = 0.0;
    /// The format: "8x10", "4x5", "645", "6x6", "35mm", "ff", "aps-c", "110", "phone", "plate". It sets the
    /// sensor's width for the depth of field, the grain's size in pixels and the diffraction.
    format: String = "35mm";
    /// The spectral sensitivity: "colour", "pan" (panchromatic), "ortho" (the reds dark, the skies white),
    /// "blue" (the wet plate: the blues white, the freckles black), "infrared".
    spectral: String = "colour";
    /// 1 = a monochrome picture (by `spectral` when it is not "colour").
    mono: f32 = 0.0;
    /// The characteristic curve: "digital" (linear to a hard clip), "film" (a toe and a shoulder), "plate" (a
    /// short scale, blocked shadows, blown lights), "paper" (a print's compressed scale).
    curve: String = "digital";
    /// The curve's contrast: 1 = normal, the slope of the straight line.
    gamma: f32 = 1.0;
    /// The toe: how slowly the shadows come up (0..1).
    toe: f32 = 0.0;
    /// The shoulder: how slowly the lights roll off (0..1).
    shoulder: f32 = 0.0;
    /// The latitude in stops: the range of exposure the curve holds before the shoulder; 0 = the curve's own.
    latitude: f32 = 0.0;
    /// The colour response as a 3×3 matrix (row-major, the dyes' cross-talk); empty = identity.
    matrix: Vec<f32> = Vec::<f32>::new();
    /// The saturation: 1 = as the scene; the dyes' chroma.
    saturation: f32 = 1.0;
    /// The warm cast (+) or the cool one (−), about −0.3..0.3.
    warm: f32 = 0.0;
    /// The cross-processing: the curves of the channels pulled apart (0..1).
    cross: f32 = 0.0;
    /// The grain (0..1): its amount; its structure from `grain_size` and the format, its weight on the
    /// mid-tones.
    grain: f32 = 0.0;
    /// The grain's size in micrometres on the negative (a fine film 8, a fast one 20, a plate 4).
    grain_size: f32 = 10.0;
    /// The sensor's noise (0..1): the photon noise growing with the root of the signal, a read noise under it.
    sensor_noise: f32 = 0.0;
    /// 1 = the colour read through a Bayer mosaic and demosaiced: the colour noise, the zipper at the edges.
    demosaic: f32 = 0.0;
    /// The halation: the red halo of a film's base around the lights (0..1).
    halation: f32 = 0.0;
    /// The autochrome's starch reseau: the picture seen through a mosaic of orange, green and violet grains
    /// (0..1).
    reseau: f32 = 0.0;
    // ---- THE PROCESS ----
    /// The development pushed (+) or pulled (−), stops: the contrast and the grain follow.
    develop: f32 = 0.0;
    /// The toning of a monochrome print: "none", "sepia", "selenium", "gold", "cyanotype", "platinum", "albumen",
    /// "silver" (the plate's cold grey), "collodion".
    tone: String = "none";
    /// The toning's strength (0..1).
    tone_amount: f32 = 1.0;
    /// The print's black: the deepest density as a value (0 = black, a platinum print 0.08, a tintype 0.12).
    black: f32 = 0.0;
    /// The print's white: the paper's brightness (1 = white, an albumen print 0.92).
    white: f32 = 1.0;
    /// The hand tint: the picture's own colour laid over the monochrome print as a wash (0..1).
    tint: f32 = 0.0;
    // ---- THE CARRIER ----
    /// What the picture is on: "none", "plate" (the daguerreotype's silver), "tintype", "glass", "card", "polaroid",
    /// "instax", "print", "slide".
    carrier: String = "none";
    /// The frame or the border as a share of the short side (0 = none).
    frame: f32 = 0.0;
    /// 1 = the picture laterally reversed, as a daguerreotype or a tintype is.
    mirror: f32 = 0.0;
    /// The mottle of a hand-coated plate: the collodion's uneven pour (0..1).
    mottle: f32 = 0.0;
    /// The plate's sheen: the mirror of the silver showing through the shadows (0..1).
    sheen: f32 = 0.0;
    // ---- THE AGE ----
    /// The age's master dial (0..1): scales the fade, the yellowing, the silvering, the scratches, the dust and
    /// the foxing.
    age: f32 = 0.0;
    /// The fade: the blacks lifted, the lights lost (0..1).
    fade: f32 = 0.0;
    /// The hue the fade turns to, degrees (a C-print's cyan loss 330 — magenta; 0 = neutral).
    fade_hue: f32 = 0.0;
    /// The yellowing of the paper and the varnish (0..1).
    yellowing: f32 = 0.0;
    /// The silvering: the metallic bloom creeping in from a plate's or a print's edges (0..1).
    silvering: f32 = 0.0;
    /// The scratches on the emulsion (0..1).
    scratches: f32 = 0.0;
    /// The dust and the specks (0..1).
    dust: f32 = 0.0;
    /// The foxing: the brown blotches of a damp paper (0..1).
    foxing: f32 = 0.0;
    // ---- THE DIGITAL ----
    /// The JPEG quality the picture went through (1..100); 0 = never compressed.
    jpeg: f32 = 0.0;
    /// The camera's sharpening and its halo (0..1).
    sharpen: f32 = 0.0;
    /// The HDR flattening of the computational camera: the shadows lifted, the lights held, the picture flat
    /// (0..1).
    hdr: f32 = 0.0;
    /// The phone's face smoothing (0..1).
    face_smooth: f32 = 0.0;
    /// The phone's portrait mode: the background blurred by the matte, not the depth — the hair's edge wrong
    /// (0..1).
    portrait_mode: f32 = 0.0;
    /// The date stamp in the corner: "" = none, "auto" = a date of the medium's time, or the text.
    date_stamp: String = "";
    /// The on-camera flash (0..1): the frontal light falling off with the depth, the shine on the skin.
    flash: f32 = 0.0;
    /// The red eye of a flash close to the lens (0..1), on the faces found.
    redeye: f32 = 0.0;
    /// The picture's size in megapixels; 0 = the picture's own.
    megapixels: f32 = 0.0;
    // ---- THE NEURAL (each behind `--neural`, each disclosed) ----
    /// The diffusion model of the fill, the conversion and the refine, by alias: "sd35-medium" (the default — a
    /// flow model holds the composition without a ControlNet and a light pass over its own output stays clean),
    /// "sd35-large", "kandinsky5" (as good, seven times slower), "sdxl" (ControlNets; a light pass over a
    /// conversion stipples the sky), "sd15".
    neural_model: String = "sd35-medium";
    /// The conversion's strength (0..1): a drawing, a watercolour, a painting re-rendered as a NEUTRAL photograph
    /// of the same scene BEFORE the medium's pass — a heavy denoise (0.6–0.8) with the structure held by
    /// `convert_control` and the picture described by `scene`; 0 = none. The director sets it from what CLIP reads
    /// the source to be (`--neural convert` runs it).
    convert: f32 = 0.0;
    /// The conversion's strength outside the subject's matte (0 = the same as `convert`): the ground held closer
    /// to the picture than the figure, or further from it; two passes blended by the matte.
    convert_ground: f32 = 0.0;
    /// The ControlNets holding the structure through the conversion, specs separated by spaces: the soft edges
    /// (HED) hold the contours and the composition — the hat's shape, the branches, the house, the boat; "depth:…",
    /// "lineart:…", "canny:…"; "" = none. On SDXL the controls' strengths summed over about 0.6 tile the frame
    /// with the latent's grid (depth at 0.7, soft edges at 0.7, soft edges 0.5 with depth 0.3 — all seen).
    convert_control: String = "softedge:strength=0.5";
    /// The refine's strength (0..1): a whole-frame img2img at this strength with `prompt`, the faces protected at
    /// 0.4 and under. A stylisation: 0.15–0.3 over a photograph or a converted picture.
    refine: f32 = 0.0;
    /// The ControlNets through the refine ("depth:strength=0.5"; "" = none).
    refine_control: String = "depth:strength=0.5";
    /// The medium's description for the refine and the fill ("a daguerreotype, 1850s, …").
    prompt: String = "";
    /// What the picture shows, for the prompts ("a bearded man on a dirt path, wooden cottages, the sea"): the
    /// director writes CLIP's reading (the genre, the stuff), `--scene` says it exactly.
    scene: String = "";
    /// The refine's working size, the long side in pixels: the model draws at its own size and the result is
    /// brought back to the frame's.
    neural_px: f32 = 1024.0;
    /// The denoising steps a pass actually runs at its strength (the schedule's total is steps/strength, 24 to
    /// 100): a light refine on a short schedule runs six steps and stipples the sky; twenty real steps do not.
    neural_steps: f32 = 20.0;
    /// The classifier-free guidance of the conversion (the prompt's pull; 5.5 draws the scene the words say).
    convert_guidance: f32 = 5.5;
    /// The guidance of the refine: a light pass at a strong guidance can only add high-frequency detail — a
    /// stipple over the sky; a low guidance (2–3) lets the medium's words tint rather than draw.
    refine_guidance: f32 = 3.0;
    /// Words added to every pass's negative prompt ("" = none): what the model must not draw.
    negative: String = "";
    /// The relight preset ("" = off): "top", "key-left", "key-right", "rim", "softbox", "golden-hour", "sunset",
    /// "moonlight", "candlelight", "overcast", "neon".
    relight: String = "";
}

impl PhotoPlan {
    /// Parse hjson (a key read twice is refused).
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        Ok(deser_hjson::from_str(text)?)
    }

    /// The plan as hjson, every key on its line with its doc's first sentence, grouped by stage.
    pub fn to_hjson(&self) -> String {
        let mut o = String::from("{\n");
        let mut section = "";
        for (k, doc) in Self::KEYS {
            let s = section_of(k);
            if s != section {
                section = s;
                o.push_str(&format!("\n# ---- {} ----\n", s));
            }
            let first = doc.lines().next().unwrap_or("").trim();
            o.push_str(&format!("{k}: {}   # {first}\n", self.get(k).unwrap_or_default()));
        }
        o.push_str("}\n");
        o
    }

    /// The keys that differ from the default, as text.
    pub fn non_default(&self) -> BTreeMap<String, String> {
        let d = PhotoPlan::default();
        Self::KEYS.iter().filter_map(|(k, _)| {
            let v = self.get(k)?;
            (v != d.get(k)?).then(|| (k.to_string(), v))
        }).collect()
    }

    /// The output size for a source of `w`×`h` (the medium's megapixels, or the source's).
    pub fn output_size(&self, w: u32, h: u32) -> (u32, u32) {
        if self.megapixels <= 0.0 {
            return (w, h);
        }
        let want = self.megapixels * 1e6;
        let have = w as f32 * h as f32;
        let s = (want / have).sqrt();
        if (s - 1.0).abs() < 0.02 {
            return (w, h);
        }
        (((w as f32 * s).round() as u32).max(8), ((h as f32 * s).round() as u32).max(8))
    }

    /// The sensor's or the negative's width in millimetres for the format.
    pub fn format_width_mm(&self) -> f32 {
        format_width_mm(&self.format)
    }

    /// True when the picture is monochrome by this plan.
    pub fn is_mono(&self) -> bool {
        self.mono > 0.5 || !matches!(self.spectral.trim(), "colour" | "color" | "")
    }
}

/// The stage a key belongs to (by its position in KEYS).
pub fn section_of(key: &str) -> &'static str {
    const CAMERA: &[&str] = &["focal", "source_focal", "distance", "dolly", "truck", "pedestal", "pan", "tilt", "roll", "shift", "keep_subject", "fill", "fill_cap"];
    const LENS: &[&str] = &["aperture", "focus", "focus_distance", "field_curve", "vignette", "distort", "ca", "fringe", "flare", "soft", "swirl", "anamorphic", "diffraction", "haze", "resolution"];
    const SHUTTER: &[&str] = &["exposure_time", "shake", "motion", "ghost"];
    const SENSOR: &[&str] = &["exposure", "iso", "format", "spectral", "mono", "curve", "gamma", "toe", "shoulder", "latitude", "matrix", "saturation", "warm", "cross", "grain", "grain_size", "sensor_noise", "demosaic", "halation", "reseau"];
    const PROCESS: &[&str] = &["develop", "tone", "tone_amount", "black", "white", "tint"];
    const CARRIER: &[&str] = &["carrier", "frame", "mirror", "mottle", "sheen"];
    const AGE: &[&str] = &["age", "fade", "fade_hue", "yellowing", "silvering", "scratches", "dust", "foxing"];
    const DIGITAL: &[&str] = &["jpeg", "sharpen", "hdr", "face_smooth", "portrait_mode", "date_stamp", "flash", "redeye", "megapixels"];
    if CAMERA.contains(&key) {
        "THE CAMERA"
    } else if LENS.contains(&key) {
        "THE LENS"
    } else if SHUTTER.contains(&key) {
        "THE SHUTTER"
    } else if SENSOR.contains(&key) {
        "THE EMULSION / THE SENSOR"
    } else if PROCESS.contains(&key) {
        "THE PROCESS"
    } else if CARRIER.contains(&key) {
        "THE CARRIER"
    } else if AGE.contains(&key) {
        "THE AGE"
    } else if DIGITAL.contains(&key) {
        "THE DIGITAL"
    } else {
        "THE NEURAL"
    }
}

/// The width of a format's frame in millimetres.
pub fn format_width_mm(format: &str) -> f32 {
    match format.trim().to_ascii_lowercase().as_str() {
        "8x10" => 254.0,
        "5x7" => 178.0,
        "4x5" | "plate" => 127.0,
        "whole-plate" => 216.0,
        "half-plate" => 165.0,
        "quarter-plate" => 108.0,
        "6x9" => 84.0,
        "6x7" => 70.0,
        "6x6" => 56.0,
        "645" => 56.0,
        "35mm" | "ff" | "135" | "" => 36.0,
        "half-frame" => 24.0,
        "aps-c" => 23.5,
        "m43" => 17.3,
        "110" => 13.0,
        "1inch" => 13.2,
        "phone" => 7.0,
        "phone-2005" | "early-digital" => 5.0,
        "polaroid" => 79.0,
        "instax" => 62.0,
        "disc" => 8.0,
        _ => 36.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_plan_round_trips_through_hjson_and_every_key_is_known() {
        let p = PhotoPlan::default();
        let text = p.to_hjson();
        let back = PhotoPlan::parse(&text).expect("the written plan parses");
        assert_eq!(p, back);
        assert!(PhotoPlan::KEYS.len() > 80);
        for (k, doc) in PhotoPlan::KEYS {
            assert!(p.get(k).is_some(), "{k} readable");
            assert!(!doc.trim().is_empty(), "{k} documented");
        }
        assert!(PhotoPlan::parse("{ nosuch: 1 }").is_err(), "an unknown key is refused");
        assert!(PhotoPlan::parse("{ grain: 0.2\n grain: 0.3 }").is_err(), "a duplicate key is refused");
    }

    #[test]
    fn a_dial_set_from_text_reads_back_and_the_non_defaults_are_listed() {
        let mut p = PhotoPlan::default();
        p.set("grain", "0.35").unwrap();
        p.set("tone", "sepia").unwrap();
        p.set("matrix", "[1, 0, 0, 0, 1, 0, 0, 0, 1]").unwrap();
        assert_eq!(p.grain, 0.35);
        assert_eq!(p.tone, "sepia");
        assert_eq!(p.matrix.len(), 9);
        let nd = p.non_default();
        assert_eq!(nd.len(), 3, "{nd:?}");
        assert_eq!(nd["grain"], "0.35");
        assert!(p.set("grain", "abc").is_err());
        assert!(p.set("nosuch", "1").is_err());
        assert_eq!(p.output_size(4000, 3000), (4000, 3000));
        p.megapixels = 1.2;
        let (w, h) = p.output_size(4000, 3000);
        assert!((w as f32 * h as f32 / 1e6 - 1.2).abs() < 0.02, "{w}×{h}");
        assert_eq!(format_width_mm("35mm"), 36.0);
        assert_eq!(section_of("grain"), "THE EMULSION / THE SENSOR");
    }
}
