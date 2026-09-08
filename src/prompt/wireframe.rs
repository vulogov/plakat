//! 6.28: procedural / LLM-planned WIREFRAME generation for `control-generate`.
//!
//! A wireframe is a 2-D skeletal outline — boxes and simple figure glyphs showing WHERE each element goes,
//! styling kept to a minimum — NOT a rendered image. Diffusion can't produce this reliably (it renders
//! scenes), so we build it deterministically: an LLM plans a bounding-box layout for the scene's elements,
//! and we draw them as clean black line-art on white. That wireframe then drives a Canny/Scribble ControlNet
//! so the composition is PLACED by construction, not guessed by the renderer.

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_circle_mut, draw_hollow_rect_mut, draw_line_segment_mut};
use imageproc::rect::Rect;
use serde::Deserialize;

/// One planned element with a normalized bounding box (`x`,`y` = top-left, all in `0..1`).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LayoutElement {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub kind: String,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// Which way a person looks: `front` / `left` / `right` / `away` (so two people talking face each other,
    /// not the viewer). Empty = front. Only meaningful for `person`.
    #[serde(default)]
    pub facing: String,
    /// What a person is DOING: `standing` / `holding` / `leaning` / `walking` / `gesturing` — shapes the
    /// skeleton's arms/legs so figures aren't identical frontal mannequins. Empty = standing.
    #[serde(default)]
    pub pose: String,
}

/// The layout-planner system prompt: turn a scene into a JSON array of placed boxes.
const PLANNER_SYSTEM: &str = "You are a composition LAYOUT PLANNER for a picture. Given a scene, output ONLY \
    a JSON array of its main elements. Each element is an object: {\"label\": string, \"kind\": one of \
    \"person\" | \"building\" | \"object\" | \"sun\" | \"ground\", \"x\": number, \"y\": number, \"w\": \
    number, \"h\": number, \"facing\": string, \"pose\": string}. x, y, w, h are FRACTIONS of the image in \
    0..1, where (x, y) is the TOP-LEFT of the box.\n\
    EVERY element's \"label\" MUST be its FULL visual description taken WORD-FOR-WORD from the scene — copy \
    the scene's OWN attributes (a person's sex/age, garments and their stated colours, held objects and \
    interactions; an object's stated material and colour; the sun's stated colour, size and height) and \
    invent or assume NOTHING the scene did not say. The label is painted VERBATIM, so a bare noun loses the \
    attributes the scene gave, but an ADDED attribute (a colour the scene never stated) paints something \
    wrong. If the scene names an element's colour, copy that exact colour; if it names none, add none. Always \
    the scene's own words, never a default.\n\
    POSITIONING — the author may not be technical, so they describe WHERE things go in plain words. When the \
    scene states a position — upper / lower, left / right / centre, foreground / background, near / far, \
    \"in the doorway\", \"beside the stall\" — you MUST place that element THERE (translate the words into x, \
    y, w, h). Only choose a position yourself for elements the scene leaves unplaced.\n\
    For each PERSON also give:\n\
    - \"facing\": which way they LOOK — one of front / left / right / away. Make it natural to the scene: \
    two people talking FACE EACH OTHER (one left, one right), not the viewer; someone leaving a doorway may \
    face front or their direction of travel. Avoid everyone facing front.\n\
    - \"pose\": what they are DOING — one of standing / holding / leaning / walking / gesturing (e.g. a \
    basket-carrier = holding, a person on a cane = leaning, a talker = gesturing).\n\
    Convey DEPTH with SIZE and BASELINE — do NOT put every figure on the same line at the same size: a figure \
    CLOSER to the viewer has a TALLER box sitting LOWER (its bottom, y+h, near 0.90-0.98); a figure FARTHER \
    away has a SHORTER box sitting HIGHER (feet up toward the horizon — bottom y+h around 0.55-0.78, height \
    maybe half the nearest figure's). Use the scene's cues for who is near vs far; a foreground subject should \
    clearly be larger and lower than a background one.\n\
    Rules: sky/sun near the top; buildings frame the left and right edges; keep every distinct FIGURE as its \
    own NON-OVERLAPPING box at the position the scene states. Include every person and key object named. \
    Output ONLY the JSON array — no prose, no code fences.";

/// Ask an LLM to plan the layout for `scene_prompt`. Returns the placed elements (may be empty on a bad reply).
pub async fn plan_layout(provider: &str, scene_prompt: &str) -> Result<Vec<LayoutElement>> {
    let resp = crate::prompt::complete(provider, PLANNER_SYSTEM, scene_prompt, &crate::prompt::EnhanceArgs::default())
        .await
        .context("layout planner LLM call")?;
    let json = extract_json_array(&resp).context("layout planner returned no JSON array")?;
    let elems: Vec<LayoutElement> = serde_json::from_str(&json)
        .with_context(|| format!("parsing layout JSON: {json}"))?;
    Ok(elems
        .into_iter()
        .filter(|e| e.w > 0.0 && e.h > 0.0)
        .collect())
}

/// Generate up to `tries` layout plans and keep the best by a programmatic **completeness** score (a vision
/// judge can't reliably read abstract stick-figure line-art, but the structured plan is easy to score):
/// more DISTINCT, NON-OVERLAPPING figures + at least some environment = better. Re-planning is varied with a
/// hint so the LLM doesn't return the same arrangement. Returns the best plan, or an error if none parsed.
pub async fn plan_best_layout(provider: &str, scene_prompt: &str, tries: usize) -> Result<Vec<LayoutElement>> {
    let mut best: Option<(Vec<LayoutElement>, f32)> = None;
    let mut attempts = 0usize;
    for t in 0..tries.max(1) {
        let user = if t == 0 {
            scene_prompt.to_string()
        } else {
            format!("{scene_prompt}\n\n(layout variation {t}: a DIFFERENT but valid arrangement; keep every figure separated)")
        };
        let elems = match plan_layout(provider, &user).await {
            Ok(e) if !e.is_empty() => e,
            _ => continue,
        };
        attempts += 1;
        let score = score_layout(&elems);
        if best.as_ref().is_none_or(|(_, b)| score > *b) {
            best = Some((elems, score));
        }
    }
    let _ = attempts;
    best.map(|(e, _)| e).context("layout planner produced no usable plan")
}

/// System prompt for [`scene_background`].
const BACKGROUND_SYSTEM: &str = "You rewrite a scene description into a BACKGROUND-ONLY setting. Keep the \
    place, architecture, streets, ground, sky, sun, clouds, weather, lighting, colours, atmosphere and art \
    style/medium EXACTLY as written. REMOVE every person, figure, animal and anything they wear or hold — no \
    people at all. The result is an EMPTY setting (a stage with no actors). Do not add new elements. Output \
    ONLY the rewritten description, no preamble, no quotes.";

/// Produce a FIGURE-FREE version of the scene — the setting, architecture, sky, lighting and art style with
/// every person removed. Used as the BASE prompt for regional generation: the regions own the figures, so a
/// base that still names people (an "old man", a "merchant") bleeds those attributes into a neighbouring
/// region's box (the "bearded woman in a red dress" failure). Best-effort — the caller falls back to the full
/// prompt if this errors or comes back empty.
pub async fn scene_background(provider: &str, scene_prompt: &str) -> Result<String> {
    let out = crate::prompt::complete(provider, BACKGROUND_SYSTEM, scene_prompt, &crate::prompt::EnhanceArgs::default())
        .await
        .context("scene-background LLM call")?;
    let out = out.trim().trim_matches('"').trim().to_string();
    if out.is_empty() {
        anyhow::bail!("scene-background returned empty");
    }
    Ok(out)
}

/// Completeness score for a plan: reward distinct figures, penalise overlapping figure boxes, small bonus
/// for having environment (buildings/sun/objects) so the scene isn't figures-in-a-void.
fn score_layout(elems: &[LayoutElement]) -> f32 {
    let persons: Vec<&LayoutElement> = elems.iter().filter(|e| e.kind.eq_ignore_ascii_case("person")).collect();
    let mut overlaps = 0usize;
    for i in 0..persons.len() {
        for j in (i + 1)..persons.len() {
            if boxes_overlap(persons[i], persons[j]) {
                overlaps += 1;
            }
        }
    }
    let has_env = elems.iter().any(|e| !e.kind.eq_ignore_ascii_case("person"));
    persons.len() as f32 * 2.0 - overlaps as f32 * 1.5 + if has_env { 1.0 } else { 0.0 }
}

fn boxes_overlap(a: &LayoutElement, b: &LayoutElement) -> bool {
    let ax2 = a.x + a.w;
    let ay2 = a.y + a.h;
    let bx2 = b.x + b.w;
    let by2 = b.y + b.h;
    a.x < bx2 && b.x < ax2 && a.y < by2 && b.y < ay2
}

/// Pull the first `[ … ]` array out of an LLM reply (tolerates preamble / code fences).
fn extract_json_array(s: &str) -> Option<String> {
    let start = s.find('[')?;
    let end = s.rfind(']')?;
    (end > start).then(|| s[start..=end].to_string())
}

/// COCO-18 RELAXED standing pose (contrapposto): keypoint positions as fractions of a person's bounding box
/// (x: 0=left…1=right, y: 0=top…1=bottom). Order: nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip,
/// Rknee, Rank, Lhip, Lknee, Lank, Reye, Leye, Rear, Lear. Weight on the figure's RIGHT leg (straight,
/// vertical); LEFT leg eased out and bent; elbows bent so forearms angle in (not stiff hanging arms); head
/// tilted slightly. `posed_keypoints` mirrors this per figure so they aren't clones.
const POSE_TEMPLATE: [(f32, f32); 18] = [
    (0.515, 0.085), // nose (slight tilt)
    (0.50, 0.185),  // neck
    (0.375, 0.20),  // R shoulder
    (0.33, 0.35),   // R elbow (out)
    (0.40, 0.49),   // R wrist (forearm angled in → bent elbow)
    (0.62, 0.195),  // L shoulder
    (0.675, 0.35),  // L elbow (out)
    (0.61, 0.49),   // L wrist (angled in)
    (0.45, 0.54),   // R hip (weight side, slightly higher)
    (0.45, 0.76),   // R knee (straight, vertical)
    (0.455, 0.985), // R ankle
    (0.575, 0.555), // L hip (relaxed, slightly lower)
    (0.605, 0.75),  // L knee (eased out, bent)
    (0.585, 0.985), // L ankle
    (0.485, 0.065), // R eye
    (0.55, 0.065),  // L eye
    (0.45, 0.075),  // R ear
    (0.575, 0.075), // L ear
];

/// OpenPose limb connections (0-indexed keypoint pairs) — the standard `limbSeq` the annotators/ControlNets
/// are trained on, with the canonical per-limb colours.
const POSE_LIMBS: [(usize, usize, [u8; 3]); 17] = [
    (1, 2, [255, 0, 0]),
    (1, 5, [255, 85, 0]),
    (2, 3, [255, 170, 0]),
    (3, 4, [255, 255, 0]),
    (5, 6, [170, 255, 0]),
    (6, 7, [85, 255, 0]),
    (1, 8, [0, 255, 0]),
    (8, 9, [0, 255, 85]),
    (9, 10, [0, 255, 170]),
    (1, 11, [0, 255, 255]),
    (11, 12, [0, 170, 255]),
    (12, 13, [0, 85, 255]),
    (1, 0, [0, 0, 255]),
    (0, 14, [85, 0, 255]),
    (14, 16, [170, 0, 255]),
    (0, 15, [255, 0, 255]),
    (15, 17, [255, 0, 170]),
];

/// Deform the relaxed template by a figure's `pose` (arms/legs) and `facing` (turn), and `mirror` the stance
/// (weight leg + arm swing flip) per figure so they aren't identical clones. Heuristic, but varies the pose.
fn posed_keypoints(facing: &str, pose: &str, mirror: bool) -> [(f32, f32); 18] {
    let mut k = POSE_TEMPLATE;
    // GENERIC posture categories only (the LLM maps the scene's specifics — a basket, a cane, a conversation
    // — onto these; this code knows nothing scene-specific). Match the canonical category the planner emits.
    match pose.trim().to_lowercase().as_str() {
        "holding" => {
            k[3] = (0.36, 0.40); k[4] = (0.45, 0.56); // forearms in, hands low-front
            k[6] = (0.64, 0.40); k[7] = (0.55, 0.56);
        }
        "leaning" => {
            k[3] = (0.30, 0.38); k[4] = (0.25, 0.62); // one arm down-out (support at the side)
            k[6] = (0.67, 0.36); k[7] = (0.69, 0.53);
        }
        "gesturing" => {
            k[3] = (0.31, 0.30); k[4] = (0.23, 0.22); // one arm raised
        }
        "walking" => {
            k[9] = (0.44, 0.74); k[10] = (0.36, 0.98); // mid-stride legs
            k[12] = (0.56, 0.80); k[13] = (0.63, 0.98);
        }
        _ => {} // standing / unspecified → the neutral template
    }
    // facing: turn the torso (narrower, head offset) so the figure looks aside, not at the camera.
    let dir = if facing.eq_ignore_ascii_case("left") {
        -1.0
    } else if facing.eq_ignore_ascii_case("right") {
        1.0
    } else {
        0.0
    };
    if dir != 0.0 {
        for pt in k.iter_mut() {
            pt.0 = 0.5 + (pt.0 - 0.5) * 0.55; // compress width → turned torso
        }
        for idx in [0usize, 14, 15, 16, 17] {
            k[idx].0 += 0.07 * dir; // shift head/eyes toward the facing side
        }
    }
    // Mirror the whole stance (flip x + swap L/R keypoint pairs so limb colours stay side-correct) — gives
    // adjacent figures the opposite weight leg / arm swing instead of being clones.
    if mirror {
        for pt in k.iter_mut() {
            pt.0 = 1.0 - pt.0;
        }
        for (a, b) in [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (14, 15), (16, 17)] {
            k.swap(a, b);
        }
    }
    k
}

/// Render the PERSON elements as an OpenPose skeleton image (coloured joints + limbs on BLACK) — the format
/// the OpenPose ControlNet is trained on, so SDXL renders a real human at each figure's box, in the figure's
/// planned pose/facing. Non-person elements are ignored (environment comes from the prompt).
pub fn render_openpose(elements: &[LayoutElement], w: u32, h: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(w, h, Rgb([0, 0, 0]));
    for (i, e) in elements.iter().filter(|e| e.kind.eq_ignore_ascii_case("person")).enumerate() {
        let bx = e.x.clamp(0.0, 1.0) * w as f32;
        let by = e.y.clamp(0.0, 1.0) * h as f32;
        let bw = e.w.clamp(0.02, 1.0) * w as f32;
        let bh = e.h.clamp(0.02, 1.0) * h as f32;
        // Alternate the mirrored stance per figure so adjacent people aren't identical.
        let kp = posed_keypoints(&e.facing, &e.pose, i % 2 == 1);
        let pt = |k: usize| -> (f32, f32) {
            let (fx, fy) = kp[k];
            (bx + fx * bw, by + fy * bh)
        };
        let thick = (bw.min(bh) / 22.0).clamp(2.0, 8.0);
        for (a, b, c) in POSE_LIMBS {
            draw_thick_line(&mut img, pt(a), pt(b), thick, Rgb(c));
        }
        for k in 0..18 {
            let (x, y) = pt(k);
            imageproc::drawing::draw_filled_circle_mut(&mut img, (x as i32, y as i32), (thick * 0.8) as i32, Rgb([255, 255, 255]));
        }
    }
    img
}

/// Draw a filled thick line (several parallel offsets) so limbs read at ControlNet resolution.
fn draw_thick_line(img: &mut RgbImage, a: (f32, f32), b: (f32, f32), thick: f32, c: Rgb<u8>) {
    let t = thick.max(1.0) as i32;
    for dx in -t / 2..=t / 2 {
        for dy in -t / 2..=t / 2 {
            draw_line_segment_mut(img, (a.0 + dx as f32, a.1 + dy as f32), (b.0 + dx as f32, b.1 + dy as f32), c);
        }
    }
}

/// Render the FULL planned layout as black-on-white line-art (all elements incl. person stick-figures) —
/// a human-inspectable schematic. The OpenPose skeleton + structures line-art are what drive the ControlNets.
pub fn render_wireframe(elements: &[LayoutElement], w: u32, h: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));
    let ink = Rgb([0u8, 0, 0]);
    for e in elements {
        let x0 = (e.x.clamp(0.0, 1.0) * w as f32) as i32;
        let y0 = (e.y.clamp(0.0, 1.0) * h as f32) as i32;
        let bw = ((e.w.clamp(0.0, 1.0) * w as f32) as i32).max(2);
        let bh = ((e.h.clamp(0.0, 1.0) * h as f32) as i32).max(2);
        match e.kind.to_lowercase().as_str() {
            "person" => draw_person(&mut img, x0, y0, bw, bh, ink),
            "building" => draw_building(&mut img, x0, y0, bw, bh, ink),
            "sun" => {
                let r = (bw.min(bh) / 2).max(2);
                draw_hollow_circle_mut(&mut img, (x0 + bw / 2, y0 + bh / 2), r, ink);
            }
            "ground" | "sky" => {} // background regions carry no strong edge — skip
            _ => draw_hollow_box(&mut img, x0, y0, bw, bh, ink), // generic object
        }
    }
    img
}

fn line(img: &mut RgbImage, a: (i32, i32), b: (i32, i32), ink: Rgb<u8>) {
    draw_line_segment_mut(img, (a.0 as f32, a.1 as f32), (b.0 as f32, b.1 as f32), ink);
}

fn draw_hollow_box(img: &mut RgbImage, x: i32, y: i32, w: i32, h: i32, ink: Rgb<u8>) {
    if w <= 0 || h <= 0 {
        return;
    }
    draw_hollow_rect_mut(img, Rect::at(x, y).of_size(w as u32, h as u32), ink);
}

/// A simple stick-figure silhouette inside the box (head circle + spine + arms + legs) — enough for a
/// Canny/Scribble ControlNet to place a standing person there.
fn draw_person(img: &mut RgbImage, x: i32, y: i32, w: i32, h: i32, ink: Rgb<u8>) {
    let cx = x + w / 2;
    let head_r = (w.min(h) / 8).max(2);
    let head_cy = y + head_r + h / 20;
    draw_hollow_circle_mut(img, (cx, head_cy), head_r, ink);
    let neck = head_cy + head_r;
    let hip = y + (h as f32 * 0.62) as i32;
    let feet = y + h - 1;
    // spine
    line(img, (cx, neck), (cx, hip), ink);
    // shoulders + arms
    let shoulder = neck + (h as f32 * 0.06) as i32;
    line(img, (cx - w / 3, shoulder + h / 8), (cx, shoulder), ink);
    line(img, (cx, shoulder), (cx + w / 3, shoulder + h / 8), ink);
    // legs
    line(img, (cx, hip), (cx - w / 4, feet), ink);
    line(img, (cx, hip), (cx + w / 4, feet), ink);
}

/// A house outline: body rectangle + a simple roof triangle.
fn draw_building(img: &mut RgbImage, x: i32, y: i32, w: i32, h: i32, ink: Rgb<u8>) {
    let roof = y + h / 4;
    draw_hollow_box(img, x, roof, w, h - h / 4, ink);
    line(img, (x, roof), (x + w / 2, y), ink);
    line(img, (x + w / 2, y), (x + w, roof), ink);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_array_tolerates_fences_and_preamble() {
        let s = "Here is the layout:\n```json\n[{\"label\":\"a\",\"kind\":\"person\",\"x\":0.1,\"y\":0.2,\"w\":0.2,\"h\":0.6}]\n```";
        let j = extract_json_array(s).expect("finds array");
        let v: Vec<LayoutElement> = serde_json::from_str(&j).expect("parses");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].kind, "person");
    }

    #[test]
    fn score_prefers_more_distinct_non_overlapping_figures() {
        let p = |x: f32| LayoutElement { label: "p".into(), kind: "person".into(), x, y: 0.4, w: 0.15, h: 0.5, ..Default::default() };
        let env = LayoutElement { label: "sun".into(), kind: "sun".into(), x: 0.4, y: 0.05, w: 0.2, h: 0.2, ..Default::default() };
        // 3 separated figures + env beats 1 figure, and beats 3 overlapping figures.
        let three_sep = vec![p(0.05), p(0.4), p(0.75), env.clone()];
        let one = vec![p(0.4), env.clone()];
        let three_overlap = vec![p(0.4), p(0.42), p(0.44), env];
        assert!(score_layout(&three_sep) > score_layout(&one));
        assert!(score_layout(&three_sep) > score_layout(&three_overlap));
    }

    #[test]
    fn posed_keypoints_vary_by_facing_and_pose() {
        let front = posed_keypoints("front", "standing", false);
        // Facing right shifts the head/nose to the right and narrows the torso vs a frontal template.
        let right = posed_keypoints("right", "standing", false);
        assert!(right[0].0 > front[0].0, "nose shifts toward the facing side");
        assert!((right[2].0 - right[5].0).abs() < (front[2].0 - front[5].0).abs(), "turned torso is narrower");
        // 'gesturing' raises a wrist (smaller y) vs standing.
        let gest = posed_keypoints("front", "gesturing", false);
        assert!(gest[4].1 < front[4].1, "gesturing raises the wrist");
        // Unknown pose falls back to the neutral template.
        assert_eq!(posed_keypoints("front", "loitering", false), front);
        // Mirroring flips the stance across the centre (nose lands on the opposite side of 0.5).
        let mirrored = posed_keypoints("front", "standing", true);
        assert!((mirrored[0].0 - 0.5).signum() != (front[0].0 - 0.5).signum() || (front[0].0 - 0.5).abs() < 1e-3);
    }

    #[test]
    fn render_wireframe_draws_ink_on_white() {
        let elems = vec![
            LayoutElement { label: "woman".into(), kind: "person".into(), x: 0.4, y: 0.4, w: 0.2, h: 0.55, ..Default::default() },
            LayoutElement { label: "house".into(), kind: "building".into(), x: 0.0, y: 0.1, w: 0.3, h: 0.8, ..Default::default() },
            LayoutElement { label: "sun".into(), kind: "sun".into(), x: 0.4, y: 0.05, w: 0.2, h: 0.2, ..Default::default() },
        ];
        let img = render_wireframe(&elems, 256, 256);
        // white background dominant, but some black ink drawn.
        let ink = img.pixels().filter(|p| p.0 == [0, 0, 0]).count();
        assert!(ink > 0, "wireframe should contain ink");
        assert!(ink < (256 * 256) as usize / 2, "wireframe is mostly white (skeletal)");
    }
}
