//! Reverse-geocoding gazetteer (place names from coordinates). We keep the runtime **offline**, but —
//! like model weights — the place database is **downloaded once and cached**: a small Natural Earth
//! populated-places table (public domain) fetched from its GitHub mirror, parsed to a compact list,
//! and written to the XDG cache. After the first fetch it's read from disk with no network.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// A populated place: name + country + coordinates.
#[derive(Serialize, Deserialize, Clone)]
pub struct Place {
    pub name: String,
    pub country: String,
    pub lon: f64,
    pub lat: f64,
}

/// Natural Earth 1:50m populated places (≈1.2k cities) — public domain, ~200 KB of GeoJSON.
const URL: &str =
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_populated_places_simple.geojson";

/// Cached gazetteer path: XDG `<cache>/plakat/photos/geo/places.json`.
pub fn gazetteer_path() -> PathBuf {
    directories::ProjectDirs::from("", "", "plakat")
        .map(|d| d.cache_dir().join("photos").join("geo").join("places.json"))
        .unwrap_or_else(|| std::env::temp_dir().join("plakat-geo-places.json"))
}

/// Read the cached gazetteer (fast, offline). `None` if it hasn't been fetched yet.
pub fn load_gazetteer() -> Option<Vec<Place>> {
    let data = std::fs::read(gazetteer_path()).ok()?;
    serde_json::from_slice(&data).ok()
}

/// Download + cache the gazetteer (needs network the FIRST time only). Returns the parsed places.
pub async fn fetch_gazetteer() -> Result<Vec<Place>> {
    let bytes = reqwest::get(URL).await.context("fetching gazetteer")?.bytes().await.context("reading gazetteer")?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).context("parsing gazetteer GeoJSON")?;
    let mut places = Vec::new();
    if let Some(features) = v.get("features").and_then(|f| f.as_array()) {
        for f in features {
            let coords = f.pointer("/geometry/coordinates").and_then(|c| c.as_array());
            let (Some(lon), Some(lat)) = (
                coords.and_then(|c| c.first()).and_then(serde_json::Value::as_f64),
                coords.and_then(|c| c.get(1)).and_then(serde_json::Value::as_f64),
            ) else {
                continue;
            };
            let props = f.get("properties");
            let pick = |keys: &[&str]| -> String {
                keys.iter()
                    .find_map(|k| props.and_then(|p| p.get(*k)).and_then(|x| x.as_str()))
                    .unwrap_or("")
                    .to_string()
            };
            let name = pick(&["name", "nameascii", "NAME"]);
            if name.is_empty() {
                continue;
            }
            let country = pick(&["adm0name", "sov0name", "ADM0NAME"]);
            places.push(Place { name, country, lon, lat });
        }
    }
    if places.is_empty() {
        anyhow::bail!("gazetteer contained no places");
    }
    let path = gazetteer_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, serde_json::to_vec(&places)?);
    Ok(places)
}

/// The closest place to `(lon, lat)` by planar degree distance (fine at city scale).
pub fn nearest(places: &[Place], lon: f64, lat: f64) -> Option<&Place> {
    places.iter().min_by(|a, b| {
        let da = (a.lon - lon).powi(2) + (a.lat - lat).powi(2);
        let db = (b.lon - lon).powi(2) + (b.lat - lat).powi(2);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_picks_the_closest() {
        let places = vec![
            Place { name: "Tokyo".into(), country: "Japan".into(), lon: 139.7, lat: 35.7 },
            Place { name: "London".into(), country: "UK".into(), lon: -0.1, lat: 51.5 },
        ];
        assert_eq!(nearest(&places, 140.0, 35.0).unwrap().name, "Tokyo");
        assert_eq!(nearest(&places, 2.0, 48.0).unwrap().name, "London");
    }
}
