//! The LAYERED-1 **layer plan** (RFC §The layer plan) — the HJSON a layered render is authored in:
//! a global look, an optional finish prompt, a backdrop, and an ordered set of subject layers, each with a
//! box (or placement words) and a depth. Weight-free parsing + resolution (defaults, `place:` → box); the
//! lint rules and size classifier live in [`super::lint`].

use serde::Deserialize;

use crate::pipelines::multiperson::placement::Placement;

/// The whole plan.
#[derive(Deserialize, Default, Clone, Debug)]
pub struct LayerPlan {
    #[serde(default)]
    pub version: Option<u32>,
    /// Output size, e.g. `"1216x832"`. CLI `--page`/`--size` may override.
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub global: Global,
    /// The finish prompt (the model actually renders this). Optional — see the finish-prompt precedence.
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub backdrop: Option<Backdrop>,
    #[serde(default)]
    pub layers: Vec<Layer>,
    #[serde(default)]
    pub draft: DraftCfg,
    #[serde(default)]
    pub repair: Option<RepairCfg>,
    /// Lift policy: `auto`/`full`/`crop`/`keep`/`off`.
    #[serde(default)]
    pub lift: Option<String>,
}

/// The global look — anchored into the drafts (palette + light) and carried to the finish (all three).
#[derive(Deserialize, Default, Clone, Debug)]
pub struct Global {
    #[serde(default)]
    pub palette: Option<String>,
    #[serde(default)]
    pub light: Option<String>,
    /// Medium/technique — sent to the FINISH only, never to drafts (only low frequencies survive).
    #[serde(default)]
    pub medium: Option<String>,
}

/// The environment layer (rendered full-canvas; anchored at a lower weight than subjects).
#[derive(Deserialize, Default, Clone, Debug)]
pub struct Backdrop {
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub weight: Option<f32>,
    #[serde(default)]
    pub window: Option<f32>,
}

/// One subject layer.
#[derive(Deserialize, Default, Clone, Debug)]
pub struct Layer {
    pub id: String,
    #[serde(default)]
    pub prompt: Option<String>,
    /// Explicit normalised box `[x0,y0,x1,y1]` in `[0,1]`. Takes precedence over `place`.
    #[serde(default, rename = "box")]
    pub bbox: Option<[f32; 4]>,
    /// Placement words (`"center-left mid front"`) resolved via the multiperson placer. `box` wins if both.
    #[serde(default)]
    pub place: Option<String>,
    /// Size token (`small`/`medium`/`large`) scaling the `place` spread. May also appear inside `place`.
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub depth: Option<f32>,
    #[serde(default)]
    pub weight: Option<f32>,
    #[serde(default)]
    pub window: Option<f32>,
    /// `anchored`/`hinted`/`lifted` — computed by the classifier, overridable here.
    #[serde(default)]
    pub class: Option<String>,
    /// OWL-ViT verification query; defaults to the prompt's head noun phrase.
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
}

/// Draft-tier config.
#[derive(Deserialize, Default, Clone, Debug)]
pub struct DraftCfg {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
}

/// Repair (S4) config.
#[derive(Deserialize, Default, Clone, Debug)]
pub struct RepairCfg {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub rounds: Option<u32>,
    #[serde(default)]
    pub strength: Option<f32>,
}

/// Parse a plan from HJSON text (permissive; the same `deser_hjson` the rest of the studio uses).
pub fn parse(text: &str) -> anyhow::Result<LayerPlan> {
    deser_hjson::from_str(text).map_err(|e| anyhow::anyhow!("parsing layer plan: {e}"))
}

/// Parse a `<w>x<h>` size string (e.g. `"1216x832"`) into pixel dims. `None` (or malformed) → the fallback.
pub fn parse_size(size: Option<&str>, fallback: (u32, u32)) -> (u32, u32) {
    let Some(s) = size else { return fallback };
    let s = s.trim().to_ascii_lowercase();
    let mut it = s.split(['x', '×', '*']);
    match (it.next().and_then(|a| a.trim().parse::<u32>().ok()), it.next().and_then(|b| b.trim().parse::<u32>().ok())) {
        (Some(w), Some(h)) if w > 0 && h > 0 => (w, h),
        _ => fallback,
    }
}

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// The size-token scale applied to a `place` spread.
fn size_scale(token: &str) -> f32 {
    match token.trim().to_ascii_lowercase().as_str() {
        "small" | "s" | "tiny" => 0.65,
        "large" | "l" | "big" => 1.4,
        _ => 1.0,
    }
}

/// Find a size token in an explicit `size:` field or inside a `place` string.
fn size_token(place: &str, size: Option<&str>) -> f32 {
    if let Some(sz) = size {
        return size_scale(sz);
    }
    for tok in place.split([',', ' ', '/', ';']).filter(|t| !t.is_empty()) {
        let s = size_scale(tok);
        if s != 1.0 {
            return s;
        }
    }
    1.0
}

/// Resolve placement words → a normalised box `[x0,y0,x1,y1]` using the multiperson placer's centroids +
/// distance spread (a half-extent), scaled by any size token. Unknown tokens are ignored by the placer.
pub fn place_to_box(place: &str, size: Option<&str>) -> [f32; 4] {
    let p = Placement::parse(place).unwrap_or_default();
    let (cx, cy) = (p.position.cx(), p.distance.cy());
    let (hw, hh) = p.distance.spread();
    let sc = size_token(place, size);
    let (hw, hh) = (hw * sc, hh * sc);
    [clamp01(cx - hw), clamp01(cy - hh), clamp01(cx + hw), clamp01(cy + hh)]
}

/// The resolved box for a layer: explicit `box` if present, else `place` words, else a centred default.
pub fn layer_box(layer: &Layer) -> [f32; 4] {
    if let Some(b) = layer.bbox {
        return [clamp01(b[0]), clamp01(b[1]), clamp01(b[2]), clamp01(b[3])];
    }
    if let Some(place) = &layer.place {
        return place_to_box(place, layer.size.as_deref());
    }
    // No box and no placement → a centred mid box.
    [0.3, 0.25, 0.7, 0.85]
}

/// The resolved depth `[0,1]` (0 = nearest): explicit `depth`, else derived from a `place` distance, else 0.5.
pub fn layer_depth(layer: &Layer) -> f32 {
    if let Some(d) = layer.depth {
        return d.clamp(0.0, 1.0);
    }
    if let Some(place) = &layer.place {
        use crate::pipelines::multiperson::placement::Distance;
        let p = Placement::parse(place).unwrap_or_default();
        return match p.distance {
            Distance::Closer => 0.25,
            Distance::Mid => 0.5,
            Distance::Farther => 0.8,
        };
    }
    0.5
}

/// The shorter side of a layer's box in OUTPUT PIXELS — the size the classifier keys on.
pub fn box_short_side_px(layer: &Layer, out_w: u32, out_h: u32) -> u32 {
    let b = layer_box(layer);
    let w = ((b[2] - b[0]).max(0.0) * out_w as f32).round() as u32;
    let h = ((b[3] - b[1]).max(0.0) * out_h as f32).round() as u32;
    w.min(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_plan_and_defaults() {
        let text = r#"{
          version: 1
          size: "1216x832"
          global: { palette: "muted autumn", light: "overcast", medium: "oil painting" }
          prompt: "a rainy market square"
          backdrop: { prompt: "cobblestones", weight: 0.6, window: 0.25 }
          layers: [
            { id: "stall", prompt: "a fruit stall", box: [0.05, 0.35, 0.45, 0.90], depth: 0.6 }
            { id: "boy", prompt: "a boy", place: "center-left mid front", depth: 0.35 }
          ]
          draft: { model: "sdxl", seed: 7 }
        }"#;
        let plan = parse(text).unwrap();
        assert_eq!(plan.version, Some(1));
        assert_eq!(parse_size(plan.size.as_deref(), (512, 512)), (1216, 832));
        assert_eq!(plan.global.medium.as_deref(), Some("oil painting"));
        assert_eq!(plan.layers.len(), 2);
        assert_eq!(plan.layers[0].id, "stall");
        // Explicit box passes through.
        assert_eq!(layer_box(&plan.layers[0]), [0.05, 0.35, 0.45, 0.90]);
        // `place` resolves to a box centred left-of-centre; depth is explicit.
        let b = layer_box(&plan.layers[1]);
        assert!(b[0] < b[2] && b[1] < b[3], "valid box: {b:?}");
        assert!((b[0] + b[2]) / 2.0 < 0.5, "center-left → centroid left of centre");
        assert_eq!(layer_depth(&plan.layers[1]), 0.35);
        assert_eq!(plan.draft.model.as_deref(), Some("sdxl"));
    }

    #[test]
    fn place_box_and_depth_from_distance() {
        // A `small` background object → a smaller box, higher depth (farther).
        let far = Layer { id: "crow".into(), place: Some("right far".into()), size: Some("small".into()), ..Default::default() };
        let near = Layer { id: "dog".into(), place: Some("right closer".into()), ..Default::default() };
        let fb = layer_box(&far);
        let nb = layer_box(&near);
        let farea = (fb[2] - fb[0]) * (fb[3] - fb[1]);
        let narea = (nb[2] - nb[0]) * (nb[3] - nb[1]);
        assert!(farea < narea, "small/far box ({farea}) < closer box ({narea})");
        assert!(layer_depth(&far) > layer_depth(&near), "farther has greater depth");
        assert!((layer_depth(&near) - 0.25).abs() < 1e-6, "closer → 0.25");
    }

    #[test]
    fn size_parsing_and_short_side() {
        assert_eq!(parse_size(Some("1216x832"), (0, 0)), (1216, 832));
        assert_eq!(parse_size(Some("768×768"), (0, 0)), (768, 768));
        assert_eq!(parse_size(Some("garbage"), (512, 512)), (512, 512));
        assert_eq!(parse_size(None, (640, 640)), (640, 640));
        let l = Layer { id: "x".into(), bbox: Some([0.0, 0.0, 0.5, 0.25]), ..Default::default() };
        // shorter side = 0.25·height of 800 = 200 vs 0.5·1000 = 500 → 200.
        assert_eq!(box_short_side_px(&l, 1000, 800), 200);
    }
}
