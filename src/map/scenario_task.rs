//! MAP-4 — the **`map` scenario task**. Lets a `plakat scenario` batch (and, via
//! compile, a `prompts.txt`) produce a map alongside renders/animations. This is
//! the delegate the scenario loop calls; it keeps the big `cli/scenario.rs`
//! dispatch thin — source the spec (load or LLM-parse), then render linework
//! (deterministic, no GPU) or paint with SD (the 1.6 `render_sd` path).

use anyhow::{Context, Result};
use candle_core::Device;
use std::path::{Path, PathBuf};

use super::parser::{self, ParseOpts};
use super::render::Style;
use super::render_sd::{self, SdOptions};
use super::spec::{MapSpec, SPEC_VERSION};

/// Everything a `map` scenario task needs, already merged (scenario ⊕ task).
#[derive(Debug, Clone)]
pub struct MapTaskCfg {
    /// Prose world description (the task `prompt`). Ignored when `spec_path` is set.
    pub description: String,
    /// A committed `MapSpec` JSON to load (skips the LLM) — the deterministic path.
    pub spec_path: Option<PathBuf>,
    /// `parchment` | `inked` | `blueprint`.
    pub style: String,
    /// Paint with SD (img2img + Canny) vs the deterministic linework render.
    pub paint: bool,
    /// LLM provider for the prose→spec parse (reuses the `--enhance` stack).
    pub provider: String,
    /// `--map-scale` alias, e.g. `region`.
    pub scale: Option<String>,
    /// `--map-tiles` `CxR`.
    pub tiles: Option<String>,
    /// SD backbone for `paint` (any plakat model).
    pub sd_model: String,
    /// LoRA specs for `paint`; empty = the model's default (fantasy-map for SDXL).
    pub sd_loras: Vec<String>,
    /// SHA-256 cache the parsed spec.
    pub cache: bool,
}

impl Default for MapTaskCfg {
    fn default() -> Self {
        MapTaskCfg {
            description: String::new(),
            spec_path: None,
            style: "parchment".into(),
            paint: false,
            provider: "auto".into(),
            scale: None,
            tiles: None,
            sd_model: "sdxl".into(),
            sd_loras: Vec::new(),
            cache: false,
        }
    }
}

impl MapTaskCfg {
    /// Resolve the LoRA set: explicit list (with `none` → none) else the model's
    /// default (fantasy-map for SDXL-family, none otherwise).
    fn resolved_loras(&self) -> Vec<String> {
        if self.sd_loras.is_empty() {
            render_sd::default_loras_for_model(&self.sd_model)
        } else if self.sd_loras.iter().any(|s| s.eq_ignore_ascii_case("none")) {
            Vec::new()
        } else {
            self.sd_loras.clone()
        }
    }
}

/// Source the spec for a map task: load `spec_path` (no LLM) or parse the prose.
/// Pure of rendering — used by both the run path and `--dry-run` validation.
pub async fn source_spec(cfg: &MapTaskCfg) -> Result<MapSpec> {
    let (grid, tier) = super::resolve_scale(cfg.tiles.as_deref(), cfg.scale.as_deref())?;
    if let Some(p) = &cfg.spec_path {
        let text = std::fs::read_to_string(p)
            .with_context(|| format!("reading map task --map-spec {}", p.display()))?;
        let mut m: MapSpec =
            serde_json::from_str(&text).with_context(|| format!("parsing MapSpec {}", p.display()))?;
        if m.version != SPEC_VERSION {
            tracing::warn!(target: "plakat", "map task: spec version {} (expected {}) — best-effort", m.version, SPEC_VERSION);
        }
        if let Some(g) = grid {
            m.tile_grid = g;
        }
        if let Some(t) = tier {
            m.scale_tier = t;
        }
        Ok(m)
    } else {
        if cfg.description.trim().is_empty() {
            anyhow::bail!("map task: provide a description (prompt) or a `map-spec` path");
        }
        parser::parse(
            &cfg.description,
            &ParseOpts {
                provider: cfg.provider.clone(),
                system_override: None,
                tile_grid: grid,
                scale_tier: tier,
                cache: cfg.cache,
            },
        )
        .await
    }
}

/// Run a `map` scenario task: source the spec, then render to `out_dir/map.png`.
/// `dry_run` validates the spec source without rendering. Returns the output path.
pub async fn run_map_task(
    cfg: &MapTaskCfg,
    seed: u64,
    device: Device,
    out_dir: &Path,
    dry_run: bool,
) -> Result<PathBuf> {
    let spec = source_spec(cfg).await?;
    let style = Style::named(&cfg.style)?;
    let out = out_dir.join("map.png");
    if dry_run {
        return Ok(out);
    }
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("creating map task out dir {}", out_dir.display()))?;

    if cfg.paint {
        let opts = SdOptions {
            model: cfg.sd_model.clone(),
            loras: cfg.resolved_loras(),
            ..SdOptions::default()
        };
        render_sd::render_sd(&spec, seed, style, &opts, device, &out)
            .await
            .context("map task: SD painted render")?;
    } else {
        super::save_map_image(&spec, seed, style, &out).context("map task: linework render")?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn island_cfg() -> MapTaskCfg {
        MapTaskCfg {
            spec_path: Some(PathBuf::from("corpus/map/island.spec.json")),
            ..MapTaskCfg::default()
        }
    }

    #[tokio::test]
    async fn sources_committed_spec_without_llm() {
        let m = source_spec(&island_cfg()).await.unwrap();
        assert_eq!(m.name, "The Isle of Vethûn");
        assert_eq!(m.landmarks.len(), 4);
    }

    #[tokio::test]
    async fn dry_run_validates_without_rendering() {
        let out = run_map_task(&island_cfg(), 42, Device::Cpu, Path::new("/tmp/plakat-noexist-xyz"), true)
            .await
            .unwrap();
        assert!(out.ends_with("map.png"));
        assert!(!out.exists(), "dry-run must not write a file");
    }

    #[tokio::test]
    async fn linework_render_writes_a_png() {
        let dir = std::env::temp_dir().join("plakat-map-task-test");
        let _ = std::fs::remove_dir_all(&dir);
        let out = run_map_task(&island_cfg(), 42, Device::Cpu, &dir, false).await.unwrap();
        assert!(out.exists(), "linework render writes map.png");
        let img = image::open(&out).unwrap();
        assert!(img.width() > 0 && img.height() > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lora_resolution_matches_cli_semantics() {
        let mut c = MapTaskCfg { sd_model: "sdxl".into(), ..MapTaskCfg::default() };
        assert_eq!(c.resolved_loras(), vec!["Muapi/fantasy-map".to_string()]);
        c.sd_model = "sd15".into();
        assert!(c.resolved_loras().is_empty(), "non-SDXL → no default LoRA");
        c.sd_loras = vec!["none".into()];
        assert!(c.resolved_loras().is_empty(), "explicit none disables");
        c.sd_loras = vec!["user/custom".into()];
        assert_eq!(c.resolved_loras(), vec!["user/custom".to_string()]);
    }
}
