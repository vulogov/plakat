//! v0.26 phase 5: video output formats for `plakat animate`
//! (and AnimateDiff in particular).
//!
//! ## Format coverage
//!
//! - **GIF** — pure-Rust via the `image` crate's existing
//!   GIF encoder. Already-shipped in v0.20; this module documents
//!   the format-enum surface but the GIF writer lives in
//!   [`crate::cli::animate::write_gif`].
//! - **MP4 / WebM** — ffmpeg as an external subprocess. plakat
//!   doesn't bundle ffmpeg; users must have it on `$PATH`. The
//!   `is_available` check surfaces a clear diagnostic when it's
//!   missing.
//! - **PNG frames** — already-shipped: every animate run writes
//!   `<out>/frame-NNNN.png`.
//!
//! ## Why ffmpeg-via-Command vs a Rust crate
//!
//! Pure-Rust MP4/WebM encoders exist but the H.264 / VP9 / AV1
//! codec story is complicated: `mp4` (Rust) is container-only,
//! still wants pre-encoded H.264 streams; AV1 encoders (`rav1e`)
//! are slow at default settings. ffmpeg has the codec landscape
//! covered and most users already have it installed for other
//! reasons. The tradeoff vs bundling a Rust codec stack: one
//! more dep + slower binary growth.

use anyhow::{Context, Result};
use std::path::Path;
use std::process::{Command, Stdio};
use std::str::FromStr;

/// Output format for animation runs. Default is `Frames` —
/// always write per-frame PNGs. `Gif` writes an animated GIF too.
/// `Mp4` / `Webm` invoke ffmpeg to encode a video. `All` writes
/// every format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Per-frame PNGs only (default). `<out>/frame-NNNN.png`.
    Frames,
    /// PNGs + animated GIF. `<out>/animation.gif`.
    Gif,
    /// PNGs + MP4 via ffmpeg. `<out>/animation.mp4`. Requires
    /// ffmpeg on `$PATH`.
    Mp4,
    /// PNGs + WebM via ffmpeg. `<out>/animation.webm`. Requires
    /// ffmpeg on `$PATH`.
    Webm,
    /// PNGs + GIF + MP4 + WebM. Requires ffmpeg for the last two.
    All,
}

impl FromStr for Format {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_lowercase().as_str() {
            "frames" | "png" | "pngs" => Self::Frames,
            "gif" => Self::Gif,
            "mp4" | "h264" => Self::Mp4,
            "webm" | "vp9" => Self::Webm,
            "all" => Self::All,
            other => anyhow::bail!(
                "unknown --format {other:?} (expected: frames | gif | mp4 | webm | all)"
            ),
        })
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Frames => "frames",
            Self::Gif => "gif",
            Self::Mp4 => "mp4",
            Self::Webm => "webm",
            Self::All => "all",
        })
    }
}

impl Format {
    /// True when this format requires the GIF encoder (pure Rust).
    pub fn needs_gif(self) -> bool {
        matches!(self, Self::Gif | Self::All)
    }
    /// True when this format requires ffmpeg (`Mp4` / `Webm` / `All`).
    pub fn needs_ffmpeg(self) -> bool {
        matches!(self, Self::Mp4 | Self::Webm | Self::All)
    }
    /// True when this format requires MP4 output.
    pub fn needs_mp4(self) -> bool {
        matches!(self, Self::Mp4 | Self::All)
    }
    /// True when this format requires WebM output.
    pub fn needs_webm(self) -> bool {
        matches!(self, Self::Webm | Self::All)
    }
}

/// Check whether ffmpeg is on `$PATH`. Returns the version
/// string on success; `Err` with a friendly install hint when
/// it's missing.
pub fn ffmpeg_version() -> Result<String> {
    let out = Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context(
            "running `ffmpeg -version` — `ffmpeg` not found on PATH. Install via:\n  \
             macOS:   brew install ffmpeg\n  \
             Ubuntu:  apt install ffmpeg\n  \
             Windows: scoop install ffmpeg",
        )?;
    if !out.status.success() {
        anyhow::bail!(
            "ffmpeg -version exited with {} — stderr: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim_end(),
        );
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    // First line is like "ffmpeg version 6.1.1 ..."
    let first = stdout.lines().next().unwrap_or("").trim();
    Ok(first.to_string())
}

/// Encode a PNG frame sequence into MP4 via ffmpeg. Uses
/// libx264 (yuv420p) for broad compatibility.
///
/// `frame_pattern`: ffmpeg-style format string for the frame
/// paths, e.g. `"./out/frame-%04d.png"`. The frame numbering
/// must be zero-padded contiguous starting at 0.
///
/// `fps`: frames per second (8 by default for AnimateDiff).
pub fn frames_to_mp4(
    frame_pattern: &str,
    out_path: &Path,
    fps: u32,
) -> Result<()> {
    let status = Command::new("ffmpeg")
        .args([
            "-y", // overwrite output
            "-framerate",
            &fps.to_string(),
            "-i",
            frame_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out_path.to_str().context("non-UTF8 mp4 path")?,
        ])
        .status()
        .with_context(|| format!("invoking ffmpeg for MP4 → {}", out_path.display()))?;
    if !status.success() {
        anyhow::bail!(
            "ffmpeg failed (exit {}) writing {}",
            status,
            out_path.display()
        );
    }
    Ok(())
}

/// Encode PNG frames into WebM (libvpx-vp9). Slower than MP4
/// but better compression at modern bitrates.
pub fn frames_to_webm(
    frame_pattern: &str,
    out_path: &Path,
    fps: u32,
) -> Result<()> {
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate",
            &fps.to_string(),
            "-i",
            frame_pattern,
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            "0",   // CRF mode (no bitrate target)
            "-crf",
            "30", // visually lossless-ish for the vp9 default
            out_path.to_str().context("non-UTF8 webm path")?,
        ])
        .status()
        .with_context(|| format!("invoking ffmpeg for WebM → {}", out_path.display()))?;
    if !status.success() {
        anyhow::bail!(
            "ffmpeg failed (exit {}) writing {}",
            status,
            out_path.display()
        );
    }
    Ok(())
}

/// v0.30 phase 2: extract every frame of an input video into
/// `out_dir` as zero-padded PNGs (`frame_000001.png`, ...) via
/// ffmpeg. Returns the sorted list of extracted paths. The caller
/// is responsible for keeping `out_dir` alive (typically a
/// `tempfile::TempDir`).
///
/// This extracts all frames — for very long inputs that's wasteful,
/// but for the typical AnimateDiff control-video use case (a few
/// seconds of footage feeding a 16–32 frame animation) it's
/// simplest and most predictable. Callers should subsample via
/// [`pick_evenly_spaced`] to get the exact frame count they need.
pub fn extract_frames(input: &Path, out_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let pattern = out_dir.join("frame_%06d.png");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            input.to_str().context("non-UTF8 input video path")?,
            "-vsync",
            "0", // pass through raw decode timing (no duplication)
            "-f",
            "image2",
            pattern.to_str().context("non-UTF8 output frame pattern")?,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("invoking ffmpeg to decode {}", input.display()))?;
    if !status.status.success() {
        anyhow::bail!(
            "ffmpeg decode failed (exit {}) on {} — stderr: {}",
            status.status,
            input.display(),
            String::from_utf8_lossy(&status.stderr).trim_end(),
        );
    }
    // Collect, sort, return. ffmpeg numbers `frame_000001.png` onward.
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(out_dir)
        .with_context(|| format!("reading frame dir {}", out_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("frame_") && n.ends_with(".png"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        anyhow::bail!(
            "ffmpeg decoded 0 frames from {} — is this a valid video?",
            input.display()
        );
    }
    Ok(paths)
}

/// v0.30 phase 2: pick `n` frames from a pool of available frames,
/// evenly spaced. Used to map a raw video frame sequence onto a
/// fixed animate frame budget regardless of the input video's
/// length.
///
/// Behaviour:
/// - `n == 0`: returns empty.
/// - `available.is_empty()`: returns empty (caller should validate).
/// - `available.len() >= n`: returns `n` paths sampled at
///   `idx = i * len / n` (`i` in `0..n`) — first/last anchored.
/// - `available.len() < n`: returns all available frames followed
///   by repeats of the last frame to pad to `n`. (The common case
///   is a short looping reference; repeating the tail is the
///   least surprising default.)
pub fn pick_evenly_spaced(available: &[std::path::PathBuf], n: usize) -> Vec<std::path::PathBuf> {
    if n == 0 || available.is_empty() {
        return Vec::new();
    }
    if available.len() >= n {
        (0..n)
            .map(|i| available[(i * available.len()) / n].clone())
            .collect()
    } else {
        let mut out: Vec<_> = available.iter().cloned().collect();
        let last = available
            .last()
            .expect("available non-empty by branch")
            .clone();
        while out.len() < n {
            out.push(last.clone());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_parses_common_spellings() {
        assert_eq!(Format::from_str("frames").unwrap(), Format::Frames);
        assert_eq!(Format::from_str("png").unwrap(), Format::Frames);
        assert_eq!(Format::from_str("PNGS").unwrap(), Format::Frames);
        assert_eq!(Format::from_str("gif").unwrap(), Format::Gif);
        assert_eq!(Format::from_str("mp4").unwrap(), Format::Mp4);
        assert_eq!(Format::from_str("h264").unwrap(), Format::Mp4);
        assert_eq!(Format::from_str("webm").unwrap(), Format::Webm);
        assert_eq!(Format::from_str("vp9").unwrap(), Format::Webm);
        assert_eq!(Format::from_str("all").unwrap(), Format::All);
    }

    #[test]
    fn format_rejects_unknown() {
        let err = Format::from_str("av1").unwrap_err();
        assert!(err.to_string().contains("unknown"));
    }

    #[test]
    fn format_dependency_flags() {
        // PNG frames + GIF need no ffmpeg.
        assert!(!Format::Frames.needs_ffmpeg());
        assert!(!Format::Gif.needs_ffmpeg());
        // MP4 / WebM / All do.
        assert!(Format::Mp4.needs_ffmpeg());
        assert!(Format::Webm.needs_ffmpeg());
        assert!(Format::All.needs_ffmpeg());
        // GIF requirement.
        assert!(Format::Gif.needs_gif());
        assert!(Format::All.needs_gif());
        assert!(!Format::Frames.needs_gif());
        assert!(!Format::Mp4.needs_gif());
        // Specific codec flags.
        assert!(Format::Mp4.needs_mp4());
        assert!(Format::All.needs_mp4());
        assert!(!Format::Webm.needs_mp4());
        assert!(Format::Webm.needs_webm());
        assert!(Format::All.needs_webm());
        assert!(!Format::Mp4.needs_webm());
    }

    #[test]
    fn format_display_round_trips() {
        for f in [
            Format::Frames,
            Format::Gif,
            Format::Mp4,
            Format::Webm,
            Format::All,
        ] {
            let s = f.to_string();
            assert_eq!(Format::from_str(&s).unwrap(), f);
        }
    }

    /// ffmpeg availability check — `#[ignore]` because it depends
    /// on the host environment.
    #[test]
    #[ignore]
    fn ffmpeg_is_available_on_dev_machine() {
        let v = ffmpeg_version().expect("ffmpeg installed");
        assert!(v.starts_with("ffmpeg version"));
    }

    // --------------------------------------------------------------
    // v0.30 phase 2: pick_evenly_spaced — pure logic, no ffmpeg.
    // --------------------------------------------------------------

    fn mk_paths(n: usize) -> Vec<std::path::PathBuf> {
        (0..n)
            .map(|i| std::path::PathBuf::from(format!("/tmp/frame_{i:06}.png")))
            .collect()
    }

    #[test]
    fn pick_evenly_spaced_zero_count_returns_empty() {
        let pool = mk_paths(100);
        assert!(pick_evenly_spaced(&pool, 0).is_empty());
    }

    #[test]
    fn pick_evenly_spaced_empty_pool_returns_empty() {
        let empty: Vec<std::path::PathBuf> = Vec::new();
        assert!(pick_evenly_spaced(&empty, 16).is_empty());
    }

    #[test]
    fn pick_evenly_spaced_equal_size_returns_all_in_order() {
        let pool = mk_paths(16);
        let picked = pick_evenly_spaced(&pool, 16);
        assert_eq!(picked, pool);
    }

    #[test]
    fn pick_evenly_spaced_larger_pool_subsamples() {
        // 100 input frames → 4 output frames at 0, 25, 50, 75.
        let pool = mk_paths(100);
        let picked = pick_evenly_spaced(&pool, 4);
        assert_eq!(picked.len(), 4);
        assert_eq!(picked[0], pool[0]);
        assert_eq!(picked[1], pool[25]);
        assert_eq!(picked[2], pool[50]);
        assert_eq!(picked[3], pool[75]);
    }

    #[test]
    fn pick_evenly_spaced_short_pool_pads_with_last_frame() {
        // 3 frames into 8 slots: 3 originals + 5 copies of frame 2.
        let pool = mk_paths(3);
        let picked = pick_evenly_spaced(&pool, 8);
        assert_eq!(picked.len(), 8);
        assert_eq!(picked[0], pool[0]);
        assert_eq!(picked[1], pool[1]);
        assert_eq!(picked[2], pool[2]);
        for i in 3..8 {
            assert_eq!(picked[i], pool[2], "slot {i} should repeat last frame");
        }
    }

    #[test]
    fn pick_evenly_spaced_singleton_pool_fills_with_repeat() {
        let pool = mk_paths(1);
        let picked = pick_evenly_spaced(&pool, 4);
        assert_eq!(picked, vec![pool[0].clone(); 4]);
    }
}
