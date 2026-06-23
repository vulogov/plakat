//! Scene-aware auto-placement. When a persona has no explicit `at:` location,
//! an LLM analyses the scene and assigns a placement (position + distance +
//! facing) that fits naturally and avoids the zones already taken by pinned
//! personas. Reuses the `--enhance` provider stack (deepseek / gemini / local).
//!
//! The LLM only ever sees the **scene clause** (no style/weather/per-person
//! prompts). On any failure it falls back to a geometric spread across the free
//! horizontal positions — the feature never hard-fails on the model.

use anyhow::Result;
use candle_core::Device;
use serde::Deserialize;

use super::placement::{Distance, Facing, Placement, Position};

const SYSTEM: &str = "\
You arrange people within an image. Given a scene description, how many people \
still need a spot, and which spots are already taken, assign each remaining \
person a placement that fits the scene naturally and does NOT collide with a \
taken spot.

Output ONLY a JSON array (no prose, no markdown fences), one object per person \
to place, in order:
[{\"position\":\"...\",\"distance\":\"...\",\"facing\":\"...\"}]

position: one of left, center-left, center, center-right, right
distance: one of closer, mid, farther   (closer = foreground/larger)
facing:   one of front, side, back       (front = facing the viewer)

Spread people out; don't stack two at the same position+distance unless the \
scene clearly implies it (e.g. a tight group). Pick facing/distance from the \
scene's logic (a listener may face side toward a speaker; a background figure \
is farther).";

#[derive(Deserialize)]
struct FigureJson {
    position: Option<String>,
    distance: Option<String>,
    facing: Option<String>,
}

/// Assign placements to `n` unplaced personas, avoiding `occupied` zones.
/// Returns exactly `n` placements (LLM result, padded/clamped, or geometric
/// fallback). `provider` is the layout LLM alias; empty / "none" → geometric.
pub async fn auto_place(
    provider: &str,
    device: &Device,
    scene_clause: &str,
    n: usize,
    occupied: &[Placement],
    seed: u64,
) -> Vec<Placement> {
    if n == 0 {
        return Vec::new();
    }
    if provider.is_empty() || provider.eq_ignore_ascii_case("none") {
        return geometric_fallback(n, occupied);
    }
    match llm_place(provider, device, scene_clause, n, occupied, seed).await {
        Ok(v) if !v.is_empty() => fit_to_count(v, n, occupied),
        Ok(_) => {
            tracing::info!(target: "plakat", "multiperson: layout LLM returned no figures; geometric placement");
            geometric_fallback(n, occupied)
        }
        Err(e) => {
            tracing::warn!(target: "plakat", "multiperson: layout LLM failed ({e}); geometric placement");
            geometric_fallback(n, occupied)
        }
    }
}

async fn llm_place(
    provider: &str,
    device: &Device,
    scene_clause: &str,
    n: usize,
    occupied: &[Placement],
    seed: u64,
) -> Result<Vec<Placement>> {
    let taken = if occupied.is_empty() {
        "none".to_string()
    } else {
        occupied
            .iter()
            .map(|p| format!("{}/{}", pos_word(p.position), dist_word(p.distance)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let user = format!(
        "Scene: {scene_clause}\nPeople to place: {n}\nSpots already taken: {taken}\nReturn the JSON array."
    );
    let opts = crate::llm::EnhanceOpts { seed, temperature: 0.0, max_new_tokens: 256 };
    let raw = crate::llm::enhance(provider, device.clone(), SYSTEM, &user, opts)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let json = extract_json_array(&raw);
    let figs: Vec<FigureJson> = serde_json::from_str(&json)
        .map_err(|e| anyhow::anyhow!("parse layout JSON: {e} (raw: {})", raw.chars().take(200).collect::<String>()))?;
    Ok(figs.into_iter().map(figure_to_placement).collect())
}

fn figure_to_placement(f: FigureJson) -> Placement {
    let mut p = Placement::default();
    if let Some(s) = &f.position {
        if let Ok(pp) = Placement::parse(s) {
            p.position = pp.position;
        }
    }
    if let Some(s) = &f.distance {
        if let Ok(pp) = Placement::parse(s) {
            p.distance = pp.distance;
        }
    }
    if let Some(s) = &f.facing {
        if let Ok(pp) = Placement::parse(s) {
            p.facing = pp.facing;
        }
    }
    p
}

/// Make the LLM result exactly `n` long: truncate extras, pad shortfall with
/// geometric placements for the remaining count.
fn fit_to_count(mut v: Vec<Placement>, n: usize, occupied: &[Placement]) -> Vec<Placement> {
    if v.len() > n {
        v.truncate(n);
    } else if v.len() < n {
        let mut taken = occupied.to_vec();
        taken.extend(v.iter().copied());
        let extra = geometric_fallback(n - v.len(), &taken);
        v.extend(extra);
    }
    v
}

/// Distribute `n` people across the free horizontal positions, mid distance,
/// facing front. Skips positions already occupied where possible.
fn geometric_fallback(n: usize, occupied: &[Placement]) -> Vec<Placement> {
    const ORDER: [Position; 5] = [
        Position::Left,
        Position::Right,
        Position::Center,
        Position::CenterLeft,
        Position::CenterRight,
    ];
    let used: Vec<Position> = occupied.iter().map(|p| p.position).collect();
    let mut free: Vec<Position> = ORDER.into_iter().filter(|p| !used.contains(p)).collect();
    // If everything's taken (or n exceeds free slots), reuse the full order.
    if free.len() < n {
        for p in ORDER {
            if !free.contains(&p) {
                free.push(p);
            }
        }
    }
    (0..n)
        .map(|i| Placement {
            position: free[i % free.len()],
            distance: Distance::Mid,
            facing: Facing::Front,
        })
        .collect()
}

fn pos_word(p: Position) -> &'static str {
    match p {
        Position::Left => "left",
        Position::CenterLeft => "center-left",
        Position::Center => "center",
        Position::CenterRight => "center-right",
        Position::Right => "right",
    }
}
fn dist_word(d: Distance) -> &'static str {
    match d {
        Distance::Closer => "closer",
        Distance::Mid => "mid",
        Distance::Farther => "farther",
    }
}

/// Pull the first JSON array out of an LLM response (strip fences / prose).
fn extract_json_array(s: &str) -> String {
    let s = s.trim();
    let s = s.strip_prefix("```json").or_else(|| s.strip_prefix("```")).unwrap_or(s);
    let s = s.strip_suffix("```").unwrap_or(s);
    match (s.find('['), s.rfind(']')) {
        (Some(a), Some(b)) if b > a => s[a..=b].to_string(),
        _ => s.trim().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric_spreads_and_avoids_taken() {
        let taken = vec![Placement { position: Position::Left, ..Default::default() }];
        let got = geometric_fallback(2, &taken);
        assert_eq!(got.len(), 2);
        // should not reuse Left while free slots remain
        assert!(got.iter().all(|p| p.position != Position::Left));
    }

    #[test]
    fn fit_truncates_and_pads() {
        let three = vec![
            Placement { position: Position::Left, ..Default::default() },
            Placement { position: Position::Center, ..Default::default() },
            Placement { position: Position::Right, ..Default::default() },
        ];
        assert_eq!(fit_to_count(three.clone(), 2, &[]).len(), 2);
        assert_eq!(fit_to_count(three[..1].to_vec(), 3, &[]).len(), 3);
    }

    #[test]
    fn extract_json_array_strips_fences_and_prose() {
        let raw = "Here you go:\n```json\n[{\"position\":\"left\"}]\n```";
        assert_eq!(extract_json_array(raw), "[{\"position\":\"left\"}]");
    }

    #[test]
    fn figure_json_maps_to_placement() {
        let f = FigureJson {
            position: Some("right".into()),
            distance: Some("closer".into()),
            facing: Some("side".into()),
        };
        let p = figure_to_placement(f);
        assert_eq!(p.position, Position::Right);
        assert_eq!(p.distance, Distance::Closer);
        assert_eq!(p.facing, Facing::Side);
    }
}
