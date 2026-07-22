//! Fractal animation → video (RFC FRACTALS-1 → 4.2 "depth", Phase B).
//!
//! Renders a sequence of Track-A frames (interpolating zoom / Julia constant / a parameter)
//! and encodes them to MP4 (ffmpeg) or animated GIF (pure-Rust `image`). Deep zoom-ins
//! automatically use the perturbation path once a frame passes the f64 limit.

use anyhow::{Context, Result};
use std::f64::consts::TAU;
use std::path::{Path, PathBuf};

use super::progress::ProgressFn;
use super::spec::{FractalKind, FractalSpec};

/// What to animate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimMode {
    /// Geometric zoom-in from a wide view to `spec.zoom`, centered on `spec.center`.
    Zoom,
    /// Julia constant swept around the `0.7885·e^{iθ}` circle (a seamless morphing loop).
    JuliaSweep,
    /// The `power` exponent swept `2 → 6 → 2` (a seamless multibrot morph loop).
    ParamSweep,
}

impl AnimMode {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "zoom" | "zoom-in" => AnimMode::Zoom,
            "julia-sweep" | "julia" => AnimMode::JuliaSweep,
            "param-sweep" | "param" | "power-sweep" => AnimMode::ParamSweep,
            other => anyhow::bail!(
                "unknown animation {other:?} (want: zoom | julia-sweep | param-sweep)"
            ),
        })
    }

    /// Sweeps loop seamlessly (use `i/frames`); zoom runs end-to-end (use `i/(frames-1)`).
    fn loops(self) -> bool {
        matches!(self, AnimMode::JuliaSweep | AnimMode::ParamSweep)
    }
}

/// The spec for frame `i` of `frames`.
fn frame_spec(base: &FractalSpec, mode: AnimMode, i: u32, frames: u32) -> FractalSpec {
    let denom = if mode.loops() { frames.max(1) } else { frames.saturating_sub(1).max(1) };
    let t = i as f64 / denom as f64; // 0..1 (exclusive of 1 for loops)
    let mut s = base.clone();
    match mode {
        AnimMode::Zoom => {
            // Geometric zoom from 1× to the target zoom, centered on the base center.
            let end = base.zoom.max(1.0);
            s.zoom = end.powf(t); // 1 → end
        }
        AnimMode::JuliaSweep => {
            s.kind = FractalKind::Julia;
            let theta = TAU * t;
            s.julia_c = [0.7885 * theta.cos(), 0.7885 * theta.sin()];
            s.center = [0.0, 0.0];
            if s.zoom < 1e-6 {
                s.zoom = 1.15;
            }
        }
        AnimMode::ParamSweep => {
            // 2 → 6 → 2 (cosine ease, seamless).
            s.power = 2.0 + 4.0 * (0.5 - 0.5 * (TAU * t).cos());
            if base.kind == FractalKind::Mandelbrot {
                s.kind = FractalKind::Multibrot;
            }
        }
    }
    s
}

/// Render `frames` frames and encode to `out` (`.mp4` / `.gif` / `.webm` by extension).
pub fn render_animation(
    base: &FractalSpec,
    mode: AnimMode,
    frames: u32,
    fps: u32,
    out: &Path,
    prog: ProgressFn,
) -> Result<()> {
    if frames < 2 {
        anyhow::bail!("animation needs at least 2 frames");
    }
    let ext = out
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    if ext == "mp4" || ext == "webm" {
        // Fail early with a friendly message rather than after rendering every frame.
        crate::imaging::video::ffmpeg_version()
            .context("ffmpeg is required for .mp4 / .webm output (use a .gif path, or install ffmpeg)")?;
    }

    let scratch = tempfile::tempdir().context("creating animation scratch dir")?;
    let mut frame_paths: Vec<PathBuf> = Vec::with_capacity(frames as usize);
    for i in 0..frames {
        let fs = frame_spec(base, mode, i, frames);
        let path = scratch.path().join(format!("frame-{i:04}.png"));
        super::render_to_file(&fs, &path)
            .with_context(|| format!("rendering animation frame {i}"))?;
        frame_paths.push(path);
        prog((i + 1) as u64, frames as u64);
    }

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    match ext.as_str() {
        "gif" => {
            let delay_ms = (1000 / fps.max(1)).clamp(1, 65_535) as u16;
            crate::cli::animate::write_gif(&frame_paths, out, delay_ms)
        }
        "webm" => {
            let pattern = frame_pattern(scratch.path());
            crate::imaging::video::frames_to_webm(&pattern, out, fps)
        }
        _ => {
            let pattern = frame_pattern(scratch.path());
            crate::imaging::video::frames_to_mp4(&pattern, out, fps)
        }
    }
}

fn frame_pattern(dir: &Path) -> String {
    dir.join("frame-%04d.png").to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_parse() {
        assert_eq!(AnimMode::parse("zoom").unwrap(), AnimMode::Zoom);
        assert_eq!(AnimMode::parse("julia").unwrap(), AnimMode::JuliaSweep);
        assert_eq!(AnimMode::parse("power-sweep").unwrap(), AnimMode::ParamSweep);
        assert!(AnimMode::parse("nope").is_err());
    }

    #[test]
    fn zoom_interpolates_geometrically() {
        let base = FractalSpec { zoom: 1e6, ..FractalSpec::default() };
        let first = frame_spec(&base, AnimMode::Zoom, 0, 60);
        let last = frame_spec(&base, AnimMode::Zoom, 59, 60);
        assert!((first.zoom - 1.0).abs() < 1e-6, "first frame is 1x, got {}", first.zoom);
        assert!((last.zoom - 1e6).abs() / 1e6 < 1e-6, "last frame is target zoom");
        // Monotonic increase.
        let mid = frame_spec(&base, AnimMode::Zoom, 30, 60);
        assert!(mid.zoom > first.zoom && mid.zoom < last.zoom);
    }

    #[test]
    fn julia_sweep_loops_around_the_circle() {
        let base = FractalSpec::default();
        let a = frame_spec(&base, AnimMode::JuliaSweep, 0, 8);
        let b = frame_spec(&base, AnimMode::JuliaSweep, 2, 8);
        assert_eq!(a.kind, FractalKind::Julia);
        assert_ne!(a.julia_c, b.julia_c);
        // c stays on the 0.7885 circle.
        let r = (a.julia_c[0].powi(2) + a.julia_c[1].powi(2)).sqrt();
        assert!((r - 0.7885).abs() < 1e-9);
    }

    #[test]
    fn param_sweep_morphs_power_and_loops() {
        let base = FractalSpec::default();
        let start = frame_spec(&base, AnimMode::ParamSweep, 0, 60);
        let mid = frame_spec(&base, AnimMode::ParamSweep, 30, 60);
        assert!((start.power - 2.0).abs() < 1e-9);
        assert!(mid.power > 5.0); // peaks near 6 at the midpoint
        assert_eq!(start.kind, FractalKind::Multibrot); // base mandelbrot → multibrot
    }

    #[test]
    fn end_to_end_gif_is_written() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("anim.gif");
        let base = FractalSpec { width: 48, height: 48, zoom: 100.0, ..FractalSpec::default() };
        render_animation(&base, AnimMode::Zoom, 4, 10, &out, &|_, _| {}).unwrap();
        assert!(out.exists());
        assert!(std::fs::metadata(&out).unwrap().len() > 0);
    }
}
