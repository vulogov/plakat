//! `fractal` scenario task (RFC FRACTALS-1 → 4.3 "ecosystem", Phase 1).
//!
//! Lets a `plakat scenario` HJSON file batch-render fractals — Track A, composition grids,
//! animations, or AI-painted — alongside `generate` / `map` / `multiperson` tasks, with the
//! same seed / out conventions. Mirrors `map::scenario_task`.

use anyhow::{Context, Result};
use candle_core::Device;
use serde::Deserialize;
use std::path::{Path, PathBuf};

use super::spec::{Coloring, FractalKind, TrapShape};
use super::FractalSpec;

/// Per-task fractal configuration (a `fractal:` block in a scenario task).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct FractalTaskCfg {
    /// Load a full FractalSpec from an HJSON/JSON file (overrides still apply).
    pub spec: Option<String>,
    /// Prose → spec (offline keyword mapper).
    pub from: Option<String>,
    pub kind: Option<String>,
    pub center: Option<String>,
    pub zoom: Option<f64>,
    pub iter: Option<u32>,
    pub size: Option<String>,
    pub palette: Option<String>,
    pub coloring: Option<String>,
    pub supersample: Option<u32>,
    pub seed: Option<u64>,
    #[serde(rename = "ifs-preset")]
    pub ifs_preset: Option<String>,
    #[serde(rename = "lsystem-preset")]
    pub lsystem_preset: Option<String>,
    #[serde(rename = "flame-preset")]
    pub flame_preset: Option<String>,
    #[serde(rename = "attractor-preset")]
    pub attractor_preset: Option<String>,
    #[serde(rename = "raymarch-shape")]
    pub raymarch_shape: Option<String>,
    #[serde(rename = "trap-image")]
    pub trap_image: Option<String>,
    // composition
    pub compose: Option<String>,
    pub grid: Option<String>,
    // animation
    pub animate: Option<String>,
    pub frames: Option<u32>,
    pub fps: Option<u32>,
    // AI paint (Track B)
    pub paint: Option<bool>,
    pub prompt: Option<String>,
    pub negative: Option<String>,
    #[serde(rename = "paint-mode")]
    pub paint_mode: Option<String>,
    #[serde(rename = "sd-model")]
    pub sd_model: Option<String>,
    #[serde(rename = "sd-strength")]
    pub sd_strength: Option<f32>,
    #[serde(rename = "sd-control-strength")]
    pub sd_control_strength: Option<f32>,
}

fn parse_pair(s: &str, what: &str) -> Result<[f64; 2]> {
    let p: Vec<&str> = s.split(',').map(str::trim).collect();
    if p.len() != 2 {
        anyhow::bail!("{what} must be `RE,IM` (got {s:?})");
    }
    Ok([p[0].parse().with_context(|| format!("{what} re"))?, p[1].parse().with_context(|| format!("{what} im"))?])
}

fn parse_wh(s: &str, sep: &[char], what: &str) -> Result<(u32, u32)> {
    let p: Vec<&str> = s.split(sep).map(str::trim).collect();
    if p.len() != 2 {
        anyhow::bail!("{what} must be two numbers (got {s:?})");
    }
    Ok((p[0].parse().with_context(|| format!("{what} a"))?, p[1].parse().with_context(|| format!("{what} b"))?))
}

/// Build the FractalSpec from the task config (base + overrides), applying `seed` as the
/// default when the task didn't set one.
pub fn build_spec(cfg: &FractalTaskCfg, seed: u64) -> Result<FractalSpec> {
    let mut s = if let Some(p) = &cfg.spec {
        FractalSpec::load(Path::new(p))?
    } else if let Some(prose) = &cfg.from {
        super::prompt::spec_from_prose(prose)
    } else {
        FractalSpec::default()
    };

    s.seed = cfg.seed.unwrap_or(seed);
    if let Some(k) = &cfg.kind {
        s.kind = FractalKind::parse(k)?;
    }
    if let Some(c) = &cfg.center {
        s.center = parse_pair(c, "center")?;
        let parts: Vec<&str> = c.split(',').map(str::trim).collect();
        s.center_hi = [parts[0].to_string(), parts[1].to_string()];
    }
    if let Some(z) = cfg.zoom {
        s.zoom = z;
    }
    if let Some(n) = cfg.iter {
        s.max_iter = n;
    }
    if let Some(sz) = &cfg.size {
        let (w, h) = parse_wh(sz, &['x', 'X', '×'], "size")?;
        s.width = w;
        s.height = h;
    }
    if let Some(p) = &cfg.palette {
        s.palette.preset = p.clone();
        s.palette.stops.clear();
    }
    if let Some(c) = &cfg.coloring {
        s.coloring = Coloring::parse(c)?;
    }
    if let Some(ss) = cfg.supersample {
        s.supersample = ss;
    }
    if let Some(p) = &cfg.ifs_preset {
        s.ifs.preset = p.clone();
    }
    if let Some(p) = &cfg.lsystem_preset {
        s.lsystem.preset = p.clone();
    }
    if let Some(p) = &cfg.flame_preset {
        s.flame.preset = p.clone();
    }
    if let Some(p) = &cfg.attractor_preset {
        s.attractor.preset = p.clone();
    }
    if let Some(sh) = &cfg.raymarch_shape {
        s.raymarch.shape = sh.clone();
    }
    if let Some(img) = &cfg.trap_image {
        s.trap_image = img.clone();
        s.coloring = Coloring::Image;
        s.trap.shape = TrapShape::Point;
    }
    // Paint (Track B).
    if cfg.paint == Some(true) || cfg.prompt.is_some() {
        s.ai.enabled = true;
    }
    if let Some(m) = &cfg.paint_mode {
        s.ai.mode = m.clone();
    }
    if let Some(p) = &cfg.prompt {
        s.ai.prompt = p.clone();
    }
    if let Some(n) = &cfg.negative {
        s.ai.negative = n.clone();
    }
    if let Some(m) = &cfg.sd_model {
        s.ai.model = m.clone();
    }
    if let Some(v) = cfg.sd_strength {
        s.ai.strength = v;
    }
    if let Some(v) = cfg.sd_control_strength {
        s.ai.control_strength = v;
    }

    s.validate()?;
    Ok(s)
}

/// Run one fractal scenario task → `<out_dir>/fractal.{png,mp4,gif}` (+ `.painted.png`).
/// Track A / composition / animation need no GPU; the paint pass uses `device`.
pub async fn run_fractal_task(
    cfg: &FractalTaskCfg,
    seed: u64,
    device: Device,
    out_dir: &Path,
    dry_run: bool,
) -> Result<PathBuf> {
    let spec = build_spec(cfg, seed)?;
    let silent = super::progress::silent();

    // Determine the primary output path.
    let out = if cfg.animate.is_some() {
        out_dir.join("fractal.mp4")
    } else if spec.ai.enabled && cfg.compose.is_none() {
        out_dir.join("fractal.painted.png")
    } else {
        out_dir.join("fractal.png")
    };
    if dry_run {
        return Ok(out);
    }
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("creating fractal task out dir {}", out_dir.display()))?;

    if let Some(mode_s) = &cfg.animate {
        let mode = super::animation::AnimMode::parse(mode_s)?;
        let frames = cfg.frames.unwrap_or(120);
        let fps = cfg.fps.unwrap_or(30);
        super::animation::render_animation(&spec, mode, frames, fps, &out, &silent)?;
        return Ok(out);
    }

    if let Some(mode_s) = &cfg.compose {
        let mode = super::compose::ComposeMode::parse(mode_s)?;
        let (rows, cols) = match &cfg.grid {
            Some(g) => parse_wh(g, &['x', 'X', '×'], "grid")?,
            None => (4, 4),
        };
        let r = super::compose::compose(&spec, mode, rows, cols, &silent)?;
        let png = out_dir.join("fractal.png");
        super::image_io::save_png_with_spec(&r.pixels, r.width, r.height, &spec, &png)?;
        return Ok(png);
    }

    // Single fractal (Track A), optionally painted.
    let png = out_dir.join("fractal.png");
    super::render_to_file(&spec, &png)?;
    if spec.ai.enabled {
        super::ai_pass::run_ai_pass(&spec, &png, &out, device).await?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_spec_applies_overrides_and_seed() {
        let cfg = FractalTaskCfg {
            kind: Some("julia".into()),
            center: Some("0.0,0.0".into()),
            zoom: Some(2.0),
            palette: Some("ice".into()),
            coloring: Some("stripe".into()),
            ..Default::default()
        };
        let s = build_spec(&cfg, 42).unwrap();
        assert_eq!(s.kind, FractalKind::Julia);
        assert_eq!(s.zoom, 2.0);
        assert_eq!(s.palette.preset, "ice");
        assert_eq!(s.coloring, Coloring::Stripe);
        assert_eq!(s.seed, 42); // scenario seed used when cfg.seed is None
    }

    #[test]
    fn prompt_enables_paint() {
        let cfg = FractalTaskCfg { prompt: Some("a forest".into()), ..Default::default() };
        assert!(build_spec(&cfg, 0).unwrap().ai.enabled);
    }

    #[tokio::test]
    async fn dry_run_reports_path_without_rendering() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = FractalTaskCfg { kind: Some("mandelbrot".into()), ..Default::default() };
        let out = run_fractal_task(&cfg, 0, Device::Cpu, dir.path(), true).await.unwrap();
        assert!(out.ends_with("fractal.png"));
        assert!(!out.exists()); // dry-run wrote nothing
    }

    #[tokio::test]
    async fn renders_a_track_a_fractal() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = FractalTaskCfg {
            kind: Some("mandelbrot".into()),
            size: Some("48x48".into()),
            ..Default::default()
        };
        let out = run_fractal_task(&cfg, 7, Device::Cpu, dir.path(), false).await.unwrap();
        assert!(out.exists());
        // The saved PNG carries its spec.
        assert!(crate::fractals::spec::read_spec_chunk(&out).unwrap().is_some());
    }
}
