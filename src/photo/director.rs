//! The director and the critic of a photograph.
//!
//! THE DIRECTOR composes the plan for THIS picture as THIS medium: the medium's house recipe
//! (`assets/photo/<media>.hjson`) and the picture's facts (`paint::director::SceneFacts` — the key, the calm
//! and busy fields, the chroma, the points of light, the faces, the subject, the far ground, the genre and the
//! stuff) through RULES, each universal, each written into the plan with its fact: the distance and the source
//! lens by the genre, the focus on the eyes or the subject, a scene stopped down, a nocturne's long exposure, a
//! high key held back, the water smeared when the shutter is open, the street's figures ghosted while a sitter
//! is braced, the far ground hazed, the lamps blooming, the sky held on an emulsion that cannot see it. A rule
//! moves no dial the recipe holds at 0 but the camera's own (the distance, the lens, the focus, the exposure).
//!
//! THE CRITIC reads the rendered picture's camera statistics against the source's and the plan's intent and
//! moves the dials that follow from them, bounded: the clipping and the black to the exposure, the invented
//! share to the camera's move, the subject's softness to the focus or the aperture, the subject's size to the
//! distance, a lost face to the fill. Every reading is written into the plan as a comment, every moved line is
//! kept as a comment, so the plan tells its own story.

use super::measure::Readings;
use super::plan::PhotoPlan;
use super::Aux;
use crate::paint::director::{entries_text, parse_entries, Entry, Genre, SceneFacts};

/// One move of one dial, with its reason.
#[derive(Clone, Debug, PartialEq)]
pub struct Adjust {
    pub key: String,
    pub value: String,
    pub why: String,
}

/// The plan as composed: its text (the three sections, every line with its reason), the plan it parses to,
/// and the notes for the terminal.
#[derive(Clone, Debug)]
pub struct Directed {
    pub text: String,
    pub plan: PhotoPlan,
    pub notes: Vec<String>,
}

/// A table of moves over a plan: a dial set once per pass, the last reason kept.
struct Table<'a> {
    plan: PhotoPlan,
    base: &'a PhotoPlan,
    moves: Vec<Adjust>,
}

impl<'a> Table<'a> {
    fn new(base: &'a PhotoPlan) -> Self {
        Table { plan: base.clone(), base, moves: Vec::new() }
    }
    fn on(&self, key: &str) -> bool {
        self.base.num(key).map_or(false, |v| v > 0.0)
    }
    fn set(&mut self, key: &str, value: &str, why: String) {
        if self.plan.set(key, value).is_ok() {
            self.moves.retain(|m| m.key != key);
            self.moves.push(Adjust { key: key.into(), value: self.plan.get(key).unwrap_or_default(), why });
        }
    }
    fn num(&mut self, key: &str, value: f32, why: String) {
        let s = <f32 as super::plan::Dial>::to_text(&value);
        self.set(key, &s, why);
    }
    /// Multiply a dial the recipe uses, capped.
    fn scale(&mut self, key: &str, factor: f32, cap: f32, why: String) {
        if let Some(v) = self.plan.num(key) {
            if v > 0.0 {
                self.num(key, (v * factor).min(cap), why);
            }
        }
    }
    /// Raise a dial the recipe uses to at least `to`.
    fn raise(&mut self, key: &str, to: f32, why: String) {
        if let Some(v) = self.plan.num(key) {
            if v > 0.0 && v < to {
                self.num(key, to, why);
            }
        }
    }
    /// Add to a dial, within bounds (the exposure).
    fn add(&mut self, key: &str, delta: f32, lo: f32, hi: f32, why: String) {
        if let Some(v) = self.plan.num(key) {
            self.num(key, (v + delta).clamp(lo, hi), why);
        }
    }
}

fn genre(f: &SceneFacts, min: f32) -> Option<Genre> {
    f.genre.and_then(|(g, c)| (c > min).then_some(g))
}

/// The largest face's relative depth inside its box: the 5th and 95th percentiles and the median (near = 1).
fn face_depth(aux: &Aux) -> Option<(f32, f32, f32)> {
    let d = aux.depth.as_ref()?;
    let (b, _) = aux.faces.iter().max_by(|a, c| ((a.0[2] - a.0[0]) * (a.0[3] - a.0[1])).partial_cmp(&((c.0[2] - c.0[0]) * (c.0[3] - c.0[1]))).unwrap_or(std::cmp::Ordering::Equal))?;
    let mut v = Vec::new();
    for y in (b[1].max(0.0) as u32)..(b[3].min(d.h as f32) as u32) {
        for x in (b[0].max(0.0) as u32)..(b[2].min(d.w as f32) as u32) {
            v.push(d.v[(y * d.w + x) as usize]);
        }
    }
    if v.len() < 64 {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |p: f32| v[((v.len() - 1) as f32 * p) as usize];
    Some((at(0.05), at(0.95), at(0.5)))
}

/// The picture in words from the facts — the genre, the people, the night — for the model's prompts. Generic
/// by design: what CLIP can vouch for; the exact description is the user's (`scene`). The stuff is NOT named: a
/// share says water is in the picture, not where, and "water" put a sea on a sandy path.
pub fn scene_words(f: &SceneFacts) -> String {
    let mut w: Vec<String> = Vec::new();
    match genre(f, 0.3) {
        Some(Genre::Portrait) => w.push("a person's face, a portrait".into()),
        Some(Genre::Figure) => w.push(if f.faces > 1 { format!("{} people, full figure", f.faces) } else { "a person, full figure".into() }),
        Some(Genre::Landscape) => w.push("a landscape".into()),
        Some(Genre::Townscape) => w.push("a town street with buildings".into()),
        Some(Genre::StillLife) => w.push("objects on a table".into()),
        Some(Genre::Interior) => w.push("a room's interior".into()),
        Some(Genre::Animal) => w.push("an animal".into()),
        _ => {}
    }
    if f.faces > 0 && !matches!(genre(f, 0.3), Some(Genre::Portrait | Genre::Figure)) {
        w.push(if f.faces > 1 { format!("{} people", f.faces) } else { "a person".into() });
    }
    if f.luma_mean < 0.3 {
        w.push("at night".into());
    }
    w.join(", ")
}

/// The director's rules: the adjustments for the picture over the recipe. `aux` carries the fields the facts
/// summarise (the depth inside a face box sets the portrait's aperture).
pub fn rules(f: &SceneFacts, recipe: &PhotoPlan, aux: &Aux) -> Vec<Adjust> {
    let mut t = Table::new(recipe);
    let g = genre(f, 0.3);
    let gname = f.genre.map(|(g, c)| format!("{} ({c:.2})", g.name())).unwrap_or_else(|| "no genre read".into());
    let scenic = matches!(g, Some(Genre::Landscape | Genre::Townscape));
    // The scene's depth: the far ground so many times the near, by the genre.
    let depth_range = if recipe.depth_range > 0.0 {
        recipe.depth_range
    } else {
        let (r, why) = match g {
            Some(Genre::Portrait) => (4.0, "a portrait: a room's depth, the far wall a few times the face's distance"),
            Some(Genre::StillLife) => (3.0, "a still life: a table's depth"),
            Some(Genre::Figure) | Some(Genre::Animal) => (10.0, "a figure in its place: the ground behind it ten times its distance"),
            Some(Genre::Interior) => (8.0, "an interior: the far wall eight times the near"),
            Some(Genre::Townscape) => (20.0, "a street: its far end twenty times the near"),
            Some(Genre::Landscape) => (50.0, "a landscape: the horizon fifty times the foreground"),
            _ => (20.0, "assumed"),
        };
        t.num("depth_range", r, format!("{why} ({gname})"));
        r
    };
    // THE CONVERSION: a painting or a drawing is a picture of a scene, not the scene — the model re-renders it as
    // a neutral photograph before the medium's pass, which then comes light. A photograph or a render needs no
    // conversion: the physics is its conversion (`--convert` to force one).
    if recipe.convert <= 0.0 {
        if let Some((word, c)) = &aux.rendering {
            if (word == "painting" || word == "drawing") && *c > 0.4 {
                let fam = super::neural::family_of(&recipe.neural_model);
                let held = if fam.takes_controls() { if recipe.convert_control.trim().is_empty() { "nothing".to_string() } else { recipe.convert_control.trim().to_string() } } else { format!("the flow's init ({} takes no ControlNet)", fam.name()) };
                t.num("convert", 0.7, format!("CLIP reads the source as a {word} ({c:.2}): converted into a neutral photograph of the same scene before the medium's pass (`--neural convert`) by {}, the structure held by {held}", recipe.neural_model));
                if recipe.refine <= 0.0 {
                    if fam.takes_controls() {
                        t.num("refine", 0.0, format!("no refine over the conversion on {}: a light pass of the SD family over its own output stipples every smooth field (`--refine 0.15` at most); the medium is the physical chain's", recipe.neural_model));
                    } else {
                        t.num("refine", 0.25, format!("the medium's pass over the converted picture, light — the material, not the content; clean on {} (`--neural refine`)", recipe.neural_model));
                    }
                }
            }
        }
    }
    // THE SCENE for the prompts: what CLIP read, in words; `--scene` says it exactly.
    if recipe.scene.trim().is_empty() {
        let words = scene_words(f);
        if !words.is_empty() {
            t.set("scene", &words, "the picture in CLIP's words for the model's prompts; `--scene` says it exactly".into());
        }
    }
    // The distance: a face sets it in the camera; else the genre.
    if recipe.distance <= 0.0 && f.faces == 0 {
        let d = match g {
            Some(Genre::Portrait) => Some(1.5),
            Some(Genre::Figure) => Some(4.0),
            Some(Genre::Townscape) => Some(12.0),
            Some(Genre::Landscape) => Some(60.0),
            Some(Genre::StillLife) => Some(0.8),
            Some(Genre::Interior) => Some(4.0),
            Some(Genre::Animal) => Some(8.0),
            _ => None,
        };
        if let Some(d) = d {
            t.num("distance", d, format!("the genre read {gname} — such a picture is taken from about {d} m (no face to read the distance from)"));
        }
    }
    // The source lens by the genre.
    let source_focal = if recipe.source_focal > 0.0 {
        recipe.source_focal
    } else {
        let (mm, why) = match g {
            Some(Genre::Portrait) if f.face_share > 0.15 => (85.0, "a head shot: a short tele, the face drawn without a wide lens's bulge"),
            Some(Genre::Portrait) => (50.0, "a portrait is usually taken on a normal to short tele lens"),
            Some(Genre::Figure) => (35.0, "a figure in its place: a moderate wide"),
            Some(Genre::Townscape) | Some(Genre::Landscape) => (28.0, "a scene: a wide lens"),
            Some(Genre::StillLife) => (60.0, "a still life: a normal lens, close"),
            Some(Genre::Interior) => (24.0, "an interior: a wide lens"),
            Some(Genre::Animal) => (135.0, "an animal: a tele lens"),
            _ => (35.0, "assumed; give --source-focal if the lens is known"),
        };
        t.num("source_focal", mm, format!("{why} ({gname})"));
        mm
    };
    // The medium re-lenses by the genre when its recipe ties the subject.
    if recipe.keep_subject > 0.5 && recipe.focal <= 0.0 {
        let mm = match g {
            Some(Genre::Portrait) => Some((85.0, "the studio's portrait lens, the face drawn as a long lens draws it")),
            Some(Genre::Figure) => Some((50.0, "a figure on a normal lens")),
            Some(Genre::StillLife) => Some((90.0, "a still life on a long lens, the perspective flattened")),
            Some(Genre::Animal) => Some((135.0, "an animal on a tele")),
            _ => None,
        };
        if let Some((mm, why)) = mm {
            t.num("focal", mm, format!("{why} ({gname}); the camera dollies to keep the subject's size"));
        }
    }
    // The focus.
    if recipe.focus.trim() == "auto" {
        let (where_, why) = if f.faces > 0 {
            ("eyes", format!("{} face(s) found — the focus on the eyes", f.faces))
        } else if f.subject_share.map_or(false, |s| s > 0.05) {
            ("subject", format!("the subject holds {:.0}% of the frame — the focus on it", f.subject_share.unwrap_or(0.0) * 100.0))
        } else if scenic {
            ("far", format!("a scene ({gname}) — the focus on the far ground"))
        } else {
            ("centre", "no face, no subject — the focus at the centre".to_string())
        };
        t.set("focus", where_, why);
    }
    // A scene is stopped down; a portrait's lens is opened.
    if scenic && recipe.aperture > 0.0 {
        let a = recipe.aperture;
        let to = (a * 2.0).max(8.0).min(22.0);
        if to > a {
            t.num("aperture", to, format!("a scene ({gname}) is shot stopped down, f/{a:.1} → f/{to:.1}"));
        }
    } else if f.faces > 0 && f.face_share > 0.08 && recipe.aperture > 0.0 {
        // A portrait's lens is opened as far as the face's depth allows: the face's near and far (the model's
        // depth inside its box over the scene's range, at the distance the eyes give) must sit inside the depth
        // of field — the circle of confusion 0.03 mm with the 35 mm-equivalent lens (the format scales the
        // lens and the circle alike, the f-number is the same on a plate) — and the f-number follows:
        // N = f²·Δz / (2·c·z²). A tight head shot asks the studio's f/11; a face in a street opens to f/2.8.
        let a = recipe.aperture;
        let fx = source_focal / 36.0 * f.width.max(f.height) as f32;
        let mut p = recipe.clone();
        p.source_focal = source_focal;
        let (z, how) = super::camera::subject_distance(&p, aux, fx);
        let to = match face_depth(aux) {
            Some((lo, hi, med)) => {
                let gg = |r: f32| (1.0 / depth_range) + (1.0 - 1.0 / depth_range) * r.clamp(0.0, 1.0);
                let (z_far, z_near) = (z * gg(med) / gg(lo), z * gg(med) / gg(hi));
                let dz_mm = ((z_far - z_near) * 1000.0).max(1.0);
                let c = 0.03;
                let n = source_focal * source_focal * dz_mm / (2.0 * c * (z * 1000.0).powi(2));
                // The format: for the same framing a plate's lens is as many times longer as the plate is wider,
                // and its depth of field as many times thinner at the same f-number — the plate's portrait was
                // stopped to f/32–f/64 (and its exposure ran long). The f-number scales with the crop factor,
                // between f/2 (a phone's f/1.6) and f/16 on 35 mm, f/64 on the largest plate.
                let crop = recipe.format_width_mm() / 36.0;
                let (lo, hi) = ((2.0 * crop).clamp(1.6, 2.0), (16.0 * crop).clamp(16.0, 64.0));
                let to = (n * crop).clamp(lo, hi);
                t.num("aperture", to, format!("a portrait (the face {:.0}% of the frame): the face spans {:.0} cm at {z:.2} m ({how}) on {source_focal:.0} mm-eq — the depth of field asks f/{n:.1} on 35 mm, f/{:.1} on this format; f/{a:.1} → f/{to:.1}", f.face_share * 100.0, dz_mm / 10.0, n * crop));
                to
            }
            None => {
                let to = (a * 0.6).max(2.0);
                if to < a {
                    t.num("aperture", to, format!("a portrait (the face {:.0}% of the frame, no depth read inside it): the lens opened, f/{a:.1} → f/{to:.1}", f.face_share * 100.0));
                }
                to
            }
        };
        let _ = to;
    }
    // A nocturne.
    if f.luma_mean < 0.3 {
        if recipe.exposure_time > 0.0 {
            let tt = (recipe.exposure_time * 8.0).min(60.0);
            t.num("exposure_time", tt, format!("a nocturne (key {:.2}) on this emulsion: the exposure runs long, {} s → {tt} s", f.luma_mean, recipe.exposure_time));
            if matches!(g, Some(Genre::Figure | Genre::Townscape)) {
                t.raise("ghost", 0.5, "the figures of a night street cannot hold still that long".into());
            }
        }
        if recipe.iso > 0.0 && recipe.sensor_noise > 0.0 {
            t.scale("iso", 4.0, 12800.0, format!("a nocturne (key {:.2}): the night pushes the ISO", f.luma_mean));
        }
        if t.on("flare") || t.on("halation") {
            t.scale("halation", 1.5, 1.0, "a nocturne: the lamps bloom on the emulsion".into());
            t.scale("flare", 1.3, 1.0, "a nocturne: the lamps flare".into());
        }
        if recipe.hdr > 0.0 {
            t.scale("hdr", 0.3, 1.0, format!("a nocturne (key {:.2}): the HDR stands down, the night mode holds the dark", f.luma_mean));
        }
    }
    // A high key held back.
    if f.luma_mean > 0.72 {
        t.add("exposure", -0.3, -3.0, 3.0, format!("a high key ({:.2}): the exposure held back so the lights do not clip", f.luma_mean));
    }
    // The water smeared when the shutter is open.
    if f.water_share.map_or(false, |s| s > 0.2) && recipe.exposure_time >= 0.5 {
        t.raise("motion", 0.7, format!("water over {:.0}% of the frame and a {} s exposure: the water smooths", f.water_share.unwrap_or(0.0) * 100.0, recipe.exposure_time));
    }
    // A sitter is braced; a street is not.
    // A sitter is braced: the head clamp of the long exposure's portrait — nothing in the frame moves.
    if f.faces > 0 && f.face_share > 0.08 && recipe.exposure_time >= 1.0 {
        if recipe.motion > 0.0 {
            t.num("motion", 0.0, format!("a portrait on a {} s exposure: the sitter braced in the head clamp, nothing moves", recipe.exposure_time));
        }
        if recipe.ghost > 0.0 {
            t.num("ghost", 0.0, "a sitter does not ghost".into());
        }
    }
    if matches!(g, Some(Genre::Figure | Genre::Townscape)) && recipe.exposure_time >= 2.0 {
        t.raise("ghost", 0.4, format!("a {} on a {} s exposure: the people ghost (a sitter is braced, a street is not)", gname, recipe.exposure_time));
    }
    // The far ground hazed.
    if f.far_share.map_or(false, |s| s > 0.45) && scenic {
        t.scale("haze", 1.5, 1.0, format!("the far ground holds {:.0}% of the frame ({gname}): the haze deepens", f.far_share.unwrap_or(0.0) * 100.0));
    }
    // The lamps bloom.
    if f.points_per_mpx > 100.0 && f.luma_mean >= 0.3 {
        t.scale("halation", 1.5, 1.0, format!("{:.0} points of light per Mpx on a dark ground: they bloom", f.points_per_mpx));
        t.scale("flare", 1.3, 1.0, format!("{:.0} points of light per Mpx: they flare", f.points_per_mpx));
    }
    // The sky held on an emulsion blind to it.
    if f.sky_share.map_or(false, |s| s > 0.3) && matches!(recipe.spectral.trim(), "ortho" | "blue" | "orthochromatic" | "collodion" | "wet-plate") {
        t.add("exposure", -0.3, -3.0, 3.0, format!("sky over {:.0}% of the frame on a {} emulsion: the sky goes white, the exposure is held for it", f.sky_share.unwrap_or(0.0) * 100.0, recipe.spectral.trim()));
    }
    // The red eye of a flash in the faces.
    if f.faces > 0 && recipe.flash > 0.0 {
        t.raise("redeye", 0.6, format!("{} face(s) and an on-camera flash: the red eye", f.faces));
    }
    t.moves
}

/// A note the rules make without a move.
pub fn remarks(f: &SceneFacts, recipe: &PhotoPlan) -> Vec<String> {
    let mut o = Vec::new();
    if f.chroma_mean < 0.06 && !recipe.is_mono() {
        o.push(format!("a near-monochrome picture (chroma {:.2}): the colour emulsion will show little colour", f.chroma_mean));
    }
    if f.light.1 > 0.25 {
        o.push(format!("the light comes from {:.0}° ({}) with coherence {:.2}; `--neural relight` could restage it, the chain keeps it", f.light.0, crate::paint::director::compass(f.light.0), f.light.1));
    }
    o
}

/// The plan's header: the picture's facts as comments.
fn header(f: &SceneFacts, source: &str, media: &super::media::Medium) -> String {
    let mut o = format!("# plakat make-photo — the plan for {source} as {} — {} ({})\n#\n# THE PICTURE:\n", media.name, media.title, media.years);
    for l in f.describe() {
        o.push_str("#   ");
        o.push_str(&l);
        o.push('\n');
    }
    if !f.sources.is_empty() {
        o.push_str("#   read by: ");
        o.push_str(&f.sources.join("; "));
        o.push('\n');
    }
    o
}

/// Comment out, in `entries`, every key `keys` names, marking who moved it.
fn retire(entries: &mut [Entry], keys: &[String], mover: &str) {
    for e in entries.iter_mut() {
        if let Entry::Key { key, value, comment } = e {
            if keys.contains(key) {
                let c = comment.as_ref().map(|c| format!("   # {c}")).unwrap_or_default();
                *e = Entry::Comment(format!("{key}: {value}{c}   (moved below by {mover})"));
            }
        }
    }
}

/// Append a section of moves to a plan's text, retiring the earlier lines of the same keys, and re-parse.
pub fn apply_section(d: &Directed, title: &str, mover: &str, comments: &[String], moves: &[Adjust]) -> anyhow::Result<Directed> {
    let keys: Vec<String> = moves.iter().map(|m| m.key.clone()).collect();
    let mut entries = parse_entries(&d.text);
    retire(&mut entries, &keys, mover);
    let mut text = entries_text(&entries);
    // The braces were dropped by the parser; the text is rebuilt with them.
    text = text.trim_end().to_string();
    text.push_str(&format!("\n\n# ---- {title} ----\n"));
    for c in comments {
        text.push_str("# ");
        text.push_str(c);
        text.push('\n');
    }
    if moves.is_empty() {
        text.push_str("# (nothing moved)\n");
    }
    for m in moves {
        text.push_str(&format!("{}: {}   # {}\n", m.key, m.value, m.why));
    }
    let full = wrap(&text);
    let plan = PhotoPlan::parse(&full)?;
    Ok(Directed { text: full, plan, notes: d.notes.clone() })
}

/// The text with its root braces (the parser's entries drop them).
fn wrap(body: &str) -> String {
    let b = body.trim_matches(|c| c == '\n');
    format!("{{\n{b}\n}}\n")
}

/// Compose the plan for `source` as `media` from the picture's facts; `overrides` are the command line's
/// (key, value) pairs, written as their own section after the director's.
pub fn direct(f: &SceneFacts, media: &super::media::Medium, source: &str, overrides: &[(String, String)], aux: &Aux) -> anyhow::Result<Directed> {
    let recipe = PhotoPlan::parse(media.recipe).map_err(|e| anyhow::anyhow!("the recipe of {} does not parse: {e}", media.name))?;
    let mut text = header(f, source, media);
    text.push_str(&format!("\n# ---- the house recipe: {} ----\n", media.name));
    let entries = parse_entries(media.recipe);
    text.push_str(&entries_text(&entries));
    let d0 = Directed { text: wrap(&text), plan: recipe.clone(), notes: Vec::new() };
    let moves = rules(f, &recipe, aux);
    let mut comments = remarks(f, &recipe);
    if moves.is_empty() {
        comments.push("the picture asks nothing of the recipe".into());
    }
    let mut d = apply_section(&d0, "the director's adjustments", "the director", &comments, &moves)?;
    d.notes = moves.iter().map(|m| format!("{}: {} — {}", m.key, m.value, m.why)).collect();
    if !overrides.is_empty() {
        let ov: Vec<Adjust> = overrides.iter().map(|(k, v)| Adjust { key: k.clone(), value: v.clone(), why: "the command line".into() }).collect();
        d = apply_section(&d, "the command line", "the command line", &[], &ov)?;
        for o in overrides {
            d.notes.push(format!("{}: {} — the command line", o.0, o.1));
        }
    }
    Ok(d)
}

impl Directed {
    /// A plan read back from its text.
    pub fn from_text(text: &str) -> anyhow::Result<Directed> {
        let plan = PhotoPlan::parse(text)?;
        Ok(Directed { text: text.to_string(), plan, notes: Vec::new() })
    }
}

/// What the critic sees of one render against its source.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub source: Readings,
    pub result: Readings,
    pub invented: f32,
    pub faces_before: usize,
    pub faces_after: usize,
    /// The subject's share of the frame before and after (0 when no matte).
    pub subject_before: f32,
    pub subject_after: f32,
}

/// The critic's verdict: the readings and the moves.
#[derive(Clone, Debug, Default)]
pub struct Verdict {
    pub readings: Vec<String>,
    pub moves: Vec<Adjust>,
}

/// Judge a render by its outcome and the plan it was made by.
pub fn verdict(o: &Outcome, plan: &PhotoPlan) -> Verdict {
    let (s, r) = (&o.source, &o.result);
    let mut t = Table::new(plan);
    let mut readings = Vec::new();
    readings.push(format!("key {:.2} → {:.2}, spread {:.2} → {:.2}, clipped {:.1}% → {:.1}%, black {:.1}% → {:.1}%", s.luma_mean, r.luma_mean, s.luma_std, r.luma_std, s.clip_share * 100.0, r.clip_share * 100.0, s.dark_share * 100.0, r.dark_share * 100.0));
    readings.push(format!("noise σ {:.4}/{:.4} → {:.4}/{:.4} (shadows/lights), structure {:.2} → {:.2}; sharpness variation {:.2} → {:.2}{}", s.noise_dark, s.noise_light, r.noise_dark, r.noise_light, s.noise_corr, r.noise_corr, s.sharp_cv, r.sharp_cv, r.sharp_focus.map(|v| format!(", the focus {v:.2}× the rest")).unwrap_or_default()));
    if o.invented > 0.0 || o.faces_before > 0 || o.subject_before > 0.0 {
        readings.push(format!("invented {:.1}% of the frame (cap {:.0}%); faces {} → {}; the subject {:.1}% → {:.1}% of the frame", o.invented * 100.0, plan.fill_cap * 100.0, o.faces_before, o.faces_after, o.subject_before * 100.0, o.subject_after * 100.0));
    }
    let printed = plan.black > 0.0 || plan.white < 1.0 || plan.curve.trim() == "plate";
    let mut exposure_moved = false;
    // The clipping and the black to the exposure.
    if r.clip_share > (0.08f32).max(s.clip_share + 0.03) {
        t.add("exposure", -0.3, -3.0, 3.0, format!("{:.1}% clipped against {:.1}% in the source", r.clip_share * 100.0, s.clip_share * 100.0));
        exposure_moved = true;
    } else if r.dark_share > (0.5f32).max(s.dark_share + 0.2) && !printed {
        t.add("exposure", 0.3, -3.0, 3.0, format!("{:.1}% black against {:.1}% in the source", r.dark_share * 100.0, s.dark_share * 100.0));
        exposure_moved = true;
    }
    // The key, when the medium does not shift it by design.
    if !exposure_moved && !printed && (r.luma_mean - s.luma_mean).abs() > 0.15 {
        let d = if r.luma_mean > s.luma_mean { -0.3 } else { 0.3 };
        t.add("exposure", d, -3.0, 3.0, format!("the key moved {:.2} → {:.2}", s.luma_mean, r.luma_mean));
    }
    // The invented share to the move.
    if o.invented > plan.fill_cap && plan.fill_cap > 0.0 {
        let k = (plan.fill_cap / o.invented).clamp(0.2, 0.95);
        for key in ["dolly", "truck", "pedestal"] {
            if let Some(v) = plan.num(key) {
                if v != 0.0 {
                    t.num(key, v * k, format!("invented {:.1}% of the frame over the cap {:.0}% — the move scaled ×{k:.2}", o.invented * 100.0, plan.fill_cap * 100.0));
                }
            }
        }
        for key in ["pan", "tilt"] {
            if let Some(v) = plan.num(key) {
                if v != 0.0 {
                    t.num(key, v * 0.7, format!("invented {:.1}% over the cap — the turn eased", o.invented * 100.0));
                }
            }
        }
        if plan.focal > 0.0 && plan.keep_subject > 0.5 && plan.dolly == 0.0 {
            let f0 = plan.source_focal.max(1.0);
            let f = f0 + (plan.focal - f0) * k;
            t.num("focal", f, format!("invented {:.1}% over the cap — the lens change eased toward the picture's own", o.invented * 100.0));
        }
    }
    // The subject's softness to the focus or the aperture.
    if plan.aperture > 0.0 {
        if let Some(sf) = r.sharp_focus {
            if sf < 0.9 {
                let mode = plan.focus.trim();
                if mode == "auto" || mode == "centre" || mode == "center" || mode == "far" || mode == "near" {
                    let to = if o.faces_after > 0 { "eyes" } else { "subject" };
                    t.set("focus", to, format!("the focus region reads {sf:.2}× the rest's sharpness — the focus set on the {to}"));
                } else {
                    t.scale("aperture", 1.4, 22.0, format!("the focus region reads {sf:.2}× the rest's sharpness at f/{:.1} — the depth of field too thin, stopped down", plan.aperture));
                }
            }
        }
    }
    // The subject's size to the distance.
    if plan.keep_subject > 0.5 && o.subject_before > 0.01 && o.subject_after > 0.0 {
        let ratio = o.subject_after / o.subject_before;
        if !(0.8..=1.25).contains(&ratio) && plan.distance > 0.0 {
            let d = (plan.distance * ratio.sqrt()).clamp(plan.distance * 0.5, plan.distance * 2.0);
            t.num("distance", d, format!("the subject's share came out ×{ratio:.2} — the distance re-read {:.2} → {d:.2} m", plan.distance));
        }
    }
    // A lost face to the fill or the move.
    if o.faces_after < o.faces_before {
        if plan.fill.trim() == "neural" {
            t.set("fill", "propagate", format!("a face was lost ({} → {}) — the neural fill stands down", o.faces_before, o.faces_after));
        } else if super::camera::moves(plan) {
            for key in ["dolly", "truck", "pedestal", "pan", "tilt"] {
                if let Some(v) = plan.num(key) {
                    if v != 0.0 {
                        t.num(key, v * 0.5, format!("a face was lost ({} → {}) — the move halved", o.faces_before, o.faces_after));
                    }
                }
            }
        }
    }
    Verdict { readings, moves: t.moves }
}

/// Append the verdict to the plan as its own section.
pub fn apply_verdict(d: &Directed, label: &str, v: &Verdict) -> anyhow::Result<Directed> {
    apply_section(d, &format!("the critic's verdict on {label}"), "the critic", &v.readings, &v.moves)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> SceneFacts {
        SceneFacts { width: 1200, height: 800, luma_mean: 0.45, luma_stddev: 0.2, chroma_mean: 0.15, flat_share: 0.3, busy_share: 0.2, ..Default::default() }
    }

    #[test]
    fn a_portraits_aperture_follows_the_faces_depth() {
        // A head shot (the face 30% of the frame, the eyes 400 px apart at 1200 px on an 85) whose face spans a
        // tenth of a room's depth range: the depth of field asks a studio's stop, the f-number rises above the
        // recipe's; a flat face (no span) opens the lens as before.
        let photo = super::super::media::by_name("photo").unwrap();
        let recipe = PhotoPlan::parse(photo.recipe).unwrap();
        let mut f = facts();
        f.faces = 1;
        f.face_share = 0.3;
        f.genre = Some((Genre::Portrait, 0.8));
        let (w, h) = (1200u32, 800u32);
        let mut d = super::super::Field::new(w, h, 0.0);
        for y in 200..600 {
            for x in 300..900 {
                d.v[(y * w + x) as usize] = 0.85 + 0.1 * ((x - 300) as f32 / 600.0);
            }
        }
        let face = ([300.0, 200.0, 900.0, 600.0], [[450.0, 350.0], [850.0, 350.0]]);
        let aux = Aux { depth: Some(d), faces: vec![face], ..Default::default() };
        let m = rules(&f, &recipe, &aux);
        let find = |k: &str| m.iter().find(|a| a.key == k).map(|a| a.value.clone());
        assert_eq!(find("depth_range").as_deref(), Some("4"), "{m:?}");
        assert_eq!(find("source_focal").as_deref(), Some("85"), "{m:?}");
        let ap: f32 = find("aperture").expect("the aperture is set").parse().unwrap();
        assert!(ap > recipe.aperture, "stopped down for the face's depth: f/{ap} vs f/{}", recipe.aperture);
        assert!(m.iter().any(|a| a.key == "aperture" && a.why.contains("depth of field asks")), "{m:?}");
        let mut flat = super::super::Field::new(w, h, 0.0);
        for y in 200..600 {
            for x in 300..900 {
                flat.v[(y * w + x) as usize] = 0.9;
            }
        }
        let aux = Aux { depth: Some(flat), faces: vec![face], ..Default::default() };
        let m = rules(&f, &recipe, &aux);
        let ap: f32 = m.iter().find(|a| a.key == "aperture").map(|a| a.value.parse().unwrap()).unwrap();
        assert!(ap < recipe.aperture, "a flat face opens the lens: f/{ap}");
    }

    #[test]
    fn a_painting_is_converted_before_the_mediums_pass_and_a_photograph_is_not() {
        let photo = super::super::media::by_name("albumen").unwrap();
        let recipe = PhotoPlan::parse(photo.recipe).unwrap();
        let mut f = facts();
        f.faces = 2;
        f.genre = Some((Genre::Figure, 0.7));
        f.foliage_share = Some(0.4);
        f.sky_share = Some(0.1);
        let aux = Aux { rendering: Some(("painting".into(), 0.8)), ..Default::default() };
        let m = rules(&f, &recipe, &aux);
        let find = |k: &str| m.iter().find(|a| a.key == k).map(|a| a.value.clone());
        assert_eq!(find("convert").as_deref(), Some("0.7"), "{m:?}");
        assert_eq!(find("refine").as_deref(), Some("0.25"), "a flow model's light pass is clean: {m:?}");
        let mut sdxl = recipe.clone();
        sdxl.neural_model = "sdxl".into();
        let m2 = rules(&f, &sdxl, &aux);
        assert_eq!(m2.iter().find(|a| a.key == "refine").map(|a| a.value.as_str()), Some("0"), "the SD family's light pass over a conversion stipples: {m2:?}");
        let scene = find("scene").expect("the scene is written");
        assert_eq!(scene, "\"2 people, full figure\"", "the genre and the people, no stuff");
        let aux = Aux { rendering: Some(("photograph".into(), 0.9)), ..Default::default() };
        let m = rules(&f, &recipe, &aux);
        assert!(!m.iter().any(|a| a.key == "convert" || a.key == "refine"), "a photograph is not converted: {m:?}");
        let mut r2 = recipe.clone();
        r2.scene = "two boys by a fence".into();
        let m = rules(&f, &r2, &aux);
        assert!(!m.iter().any(|a| a.key == "scene"), "the plan's own scene stands: {m:?}");
    }

    #[test]
    fn a_portrait_is_focused_on_the_eyes_and_a_scene_is_stopped_down_and_hazed() {
        let media = super::super::media::by_name("daguerreotype").unwrap();
        let recipe = PhotoPlan::parse(media.recipe).unwrap();
        let mut f = facts();
        f.faces = 1;
        f.face_share = 0.12;
        f.genre = Some((Genre::Portrait, 0.6));
        let m = rules(&f, &recipe, &Aux::default());
        let find = |k: &str| m.iter().find(|a| a.key == k).map(|a| a.value.clone());
        assert_eq!(find("focus").as_deref(), Some("\"eyes\""));
        assert_eq!(find("focal").as_deref(), Some("85"), "the daguerreotype re-lenses a portrait: {m:?}");
        assert!(find("distance").is_none(), "a face reads the distance in the camera");
        let photo = super::super::media::by_name("photo").unwrap();
        let recipe = PhotoPlan::parse(photo.recipe).unwrap();
        let mut f = facts();
        f.genre = Some((Genre::Landscape, 0.7));
        f.far_share = Some(0.6);
        f.sky_share = Some(0.4);
        let m = rules(&f, &recipe, &Aux::default());
        let find = |k: &str| m.iter().find(|a| a.key == k).map(|a| a.value.clone());
        assert_eq!(find("focus").as_deref(), Some("\"far\""));
        assert!(find("aperture").map(|v| v.parse::<f32>().unwrap()).unwrap() > recipe.aperture);
        assert_eq!(find("distance").as_deref(), Some("60"));
        assert!(find("focal").is_none(), "the modern camera keeps the picture's lens");
        let d = direct(&f, photo, "scene.png", &[("grain".into(), "0.4".into())], &Aux::default()).unwrap();
        assert_eq!(d.plan.grain, 0.4);
        assert!(d.text.contains("the command line") && d.text.contains("the director's adjustments"));
        let back = Directed::from_text(&d.text).unwrap();
        assert_eq!(back.plan, d.plan);
    }

    #[test]
    fn the_critic_moves_the_exposure_on_a_clip_and_eases_a_move_over_the_cap() {
        let photo = super::super::media::by_name("photo").unwrap();
        let mut plan = PhotoPlan::parse(photo.recipe).unwrap();
        plan.dolly = 1.0;
        plan.pan = 4.0;
        let o = Outcome {
            source: Readings { clip_share: 0.01, luma_mean: 0.5, ..Default::default() },
            result: Readings { clip_share: 0.15, luma_mean: 0.55, sharp_focus: Some(1.3), ..Default::default() },
            invented: 0.2,
            ..Default::default()
        };
        let v = verdict(&o, &plan);
        let find = |k: &str| v.moves.iter().find(|a| a.key == k).map(|a| a.value.clone());
        assert_eq!(find("exposure").as_deref(), Some("-0.3"));
        assert_eq!(find("dolly").as_deref(), Some("0.4"), "{:?}", v.moves);
        assert_eq!(find("pan").as_deref(), Some("2.8"));
        assert!(find("focus").is_none(), "the focus region was the sharper");
        // A soft focus region with the focus on auto: the focus set on the subject.
        let o2 = Outcome { result: Readings { sharp_focus: Some(0.6), luma_mean: 0.5, ..Default::default() }, source: Readings { luma_mean: 0.5, ..Default::default() }, ..Default::default() };
        let v2 = verdict(&o2, &plan);
        assert_eq!(v2.moves.iter().find(|a| a.key == "focus").map(|a| a.value.as_str()), Some("\"subject\""));
        // Nothing to move: an empty verdict, written as such.
        let o3 = Outcome { result: Readings { luma_mean: 0.5, ..Default::default() }, source: Readings { luma_mean: 0.5, ..Default::default() }, ..Default::default() };
        let v3 = verdict(&o3, &plan);
        assert!(v3.moves.is_empty());
        let d = Directed { text: PhotoPlan::default().to_hjson(), plan: PhotoPlan::default(), notes: vec![] };
        let d2 = apply_verdict(&d, "out.png", &v).unwrap();
        assert_eq!(d2.plan.exposure, -0.3);
        assert!(d2.text.contains("(moved below by the critic)"));
        let d3 = apply_verdict(&d2, "out.i1.png", &v3).unwrap();
        assert!(d3.text.contains("(nothing moved)"));
    }
}
