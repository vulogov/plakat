//! `plakat faceswap` — swap the face(s) in an existing image (or a folder / video) with source face(s).
//!
//! The standalone verb for the proven [`FaceSwapper`] engine (SCRFD 5-point align → ArcFace identity →
//! `inswapper_128` → colour-matched feather paste-back). Unlike `persona` / `multiperson` it does **not**
//! generate a scene — it edits media you already have.
//!
//! 6.21.0 (RFC FACESWAP-3) depth: `--dry-run`/`--preview` (inspect detections), a **directory** input
//! (batch), **repeatable `--source`** (per-face different identities by size rank), and **video** input
//! (swap every frame). inswapper weights are **non-commercial** (InsightFace) — gated behind opt-in.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use console::style;
use image::RgbImage;

use crate::pipelines::faceswap::FaceSwapper;
use crate::pipelines::scrfd::Face;

#[derive(clap::Args, Debug)]
pub struct FaceswapArgs {
    /// Scene to edit: an image, a **directory** of images (batch), or a **video** (swap every frame).
    #[arg(value_name = "SCENE")]
    pub scene: PathBuf,
    /// Source face photo (its largest face is used). **Repeatable**: with K sources, the i-th source
    /// swaps the i-th largest face (source[0] → largest). Not needed with `--dry-run`.
    #[arg(long, value_name = "FACE")]
    pub source: Vec<PathBuf>,
    /// With ONE source: which detected face to swap (0 = largest). Ignored with `--all` / multiple sources.
    #[arg(long, default_value_t = 0)]
    pub face: usize,
    /// With ONE source: swap EVERY detected face with it.
    #[arg(long, default_value_t = false)]
    pub all: bool,
    /// Output path (image), or output directory (batch). Default: `<scene>_swapped.<ext>`.
    #[arg(long, value_name = "PATH")]
    pub out: Option<PathBuf>,
    /// Run a light ADetailer detail pass on the result (crisper, but can slightly drift identity).
    #[arg(long, default_value_t = false)]
    pub restore: bool,
    /// Detect faces and print them (index · bbox · score), largest-first — no swap, no source needed.
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,
    /// Also write the scene with numbered/colour-coded detection boxes to this path (implies `--dry-run`).
    #[arg(long, value_name = "PATH")]
    pub preview: Option<PathBuf>,
}

/// Extensions treated as video (routed to the per-frame path). Mirrors `naturalize`.
fn is_video(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).as_deref(),
        Some("mp4" | "mov" | "webm" | "mkv" | "avi" | "gif" | "m4v")
    )
}

/// Decide **which face gets which source** (pure — the mapping logic, unit-tested). Returns
/// `(face_index, source_index)` pairs. Rules:
/// * K > 1 sources → map source `i` to the i-th largest face (`min(K, n_faces)` pairs);
/// * 1 source + `all` → every face ← source 0;
/// * 1 source + `--face N` → just face N ← source 0 (errors if N is out of range).
fn plan_swaps(n_faces: usize, n_sources: usize, face: usize, all: bool) -> Result<Vec<(usize, usize)>> {
    anyhow::ensure!(n_faces > 0, "no face detected");
    anyhow::ensure!(n_sources > 0, "no --source given");
    if n_sources > 1 {
        Ok((0..n_sources.min(n_faces)).map(|i| (i, i)).collect())
    } else if all {
        Ok((0..n_faces).map(|i| (i, 0)).collect())
    } else {
        anyhow::ensure!(face < n_faces, "--face {face} out of range: {n_faces} face(s) detected");
        Ok(vec![(face, 0)])
    }
}

/// Swap a single loaded scene against precomputed source `latents`, per `plan_swaps`. Returns the edited
/// image + the number of faces swapped.
fn swap_scene(swapper: &FaceSwapper, scene: &RgbImage, faces: &[Face], latents: &[Tensor], face: usize, all: bool) -> Result<(RgbImage, usize)> {
    let plan = plan_swaps(faces.len(), latents.len(), face, all)?;
    let mut out = scene.clone();
    for &(fi, si) in &plan {
        out = swapper.swap_into(&out, faces[fi].landmarks, &latents[si]).with_context(|| format!("swapping face {fi}"))?;
    }
    Ok((out, plan.len()))
}

/// Draw colour-coded detection boxes (index-ordered) onto a copy of `scene` — the `--preview` image.
fn draw_preview(scene: &RgbImage, faces: &[Face]) -> RgbImage {
    const PALETTE: [[u8; 3]; 6] = [[255, 60, 60], [60, 200, 60], [80, 140, 255], [255, 200, 40], [220, 80, 220], [40, 220, 220]];
    let mut img = scene.clone();
    let (w, h) = img.dimensions();
    for (i, f) in faces.iter().enumerate() {
        let c = image::Rgb(PALETTE[i % PALETTE.len()]);
        let x0 = f.bbox[0].max(0.0) as u32;
        let y0 = f.bbox[1].max(0.0) as u32;
        let x1 = (f.bbox[2].min((w - 1) as f32)) as u32;
        let y1 = (f.bbox[3].min((h - 1) as f32)) as u32;
        // 2-px border; a filled i+1-height tick in the top-left corner marks the index.
        for t in 0..2 {
            for x in x0..=x1.min(w - 1) {
                if y0 + t < h { img.put_pixel(x, y0 + t, c); }
                if y1 >= t { img.put_pixel(x, y1 - t, c); }
            }
            for y in y0..=y1.min(h - 1) {
                if x0 + t < w { img.put_pixel(x0 + t, y, c); }
                if x1 >= t { img.put_pixel(x1 - t, y, c); }
            }
        }
        for k in 0..=(i as u32) {
            for dx in 0..6 {
                let (px, py) = (x0 + dx, y0 + k * 3);
                if px < w && py < h { img.put_pixel(px, py, c); }
            }
        }
    }
    img
}

pub async fn run(args: FaceswapArgs, device: Device) -> Result<()> {
    // ── dry-run / preview: inspect detections only (no source, no inswapper download) ──
    if args.dry_run || args.preview.is_some() {
        let scrfd = crate::pipelines::scrfd::resolve_scrfd_weights()
            .await?
            .context("face-swap needs SCRFD weights (none resolved)")?;
        let mut det = crate::pipelines::scrfd::SCRFDDetector::load(&scrfd, crate::pipelines::scrfd::SCRFDConfig::default(), &device, DType::F32)?;
        det.score_threshold = 0.35;
        let mut faces = det.detect(&args.scene).context("detecting faces")?;
        faces.sort_by(|a, b| {
            let area = |f: &Face| (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]);
            area(b).partial_cmp(&area(a)).unwrap_or(std::cmp::Ordering::Equal)
        });
        println!("{}  {} face(s) in {}:", style("✓").green(), faces.len(), args.scene.display());
        for (i, f) in faces.iter().enumerate() {
            println!("  [{i}] bbox {:.0},{:.0} → {:.0},{:.0}  score {:.2}", f.bbox[0], f.bbox[1], f.bbox[2], f.bbox[3], f.score);
        }
        if let Some(p) = &args.preview {
            let scene = image::open(&args.scene)?.to_rgb8();
            draw_preview(&scene, &faces).save(p).with_context(|| format!("writing preview {}", p.display()))?;
            println!("  {} preview → {}", style("↳").cyan(), p.display());
        }
        return Ok(());
    }

    anyhow::ensure!(!args.source.is_empty(), "faceswap needs --source <face> (or use --dry-run to inspect)");
    let swapper = FaceSwapper::load_resolved(&device, DType::F32)
        .await
        .context("loading face-swap models (SCRFD + ArcFace + inswapper)")?;
    let latents: Vec<Tensor> = args
        .source
        .iter()
        .map(|s| swapper.source_latent(s).with_context(|| format!("embedding source {}", s.display())))
        .collect::<Result<_>>()?;

    if is_video(&args.scene) {
        return faceswap_video(&swapper, &latents, &args, device).await;
    }
    if args.scene.is_dir() {
        return faceswap_batch(&swapper, &latents, &args, device).await;
    }

    // ── single image ──
    let faces = swapper.detect(&args.scene).context("detecting faces in the scene")?;
    anyhow::ensure!(!faces.is_empty(), "no face detected in {} (try a clearer / larger face)", args.scene.display());
    let scene = image::open(&args.scene).with_context(|| format!("opening {}", args.scene.display()))?.to_rgb8();
    let (out_img, n) = swap_scene(&swapper, &scene, &faces, &latents, args.face, args.all)?;

    let out = args.out.clone().unwrap_or_else(|| default_out(&args.scene));
    write_image(&out_img, &out)?;
    if args.restore {
        run_restore(&out, device).await?;
    }
    println!("{}  swapped {n} face(s) → {}", style("✓").green(), out.display());
    license_note();
    Ok(())
}

/// Batch: `<scene>` is a directory → swap every image into the `--out` directory.
async fn faceswap_batch(swapper: &FaceSwapper, latents: &[Tensor], args: &FaceswapArgs, device: Device) -> Result<()> {
    let out_dir = args.out.clone().unwrap_or_else(|| args.scene.join("swapped"));
    std::fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&args.scene)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && is_image(p))
        .collect();
    entries.sort();
    anyhow::ensure!(!entries.is_empty(), "no images found in {}", args.scene.display());
    let (mut ok, mut noface, mut err) = (0u32, 0u32, 0u32);
    for path in &entries {
        let faces = match swapper.detect(path) {
            Ok(f) if !f.is_empty() => f,
            Ok(_) => { noface += 1; continue; }
            Err(_) => { err += 1; continue; }
        };
        let scene = match image::open(path) { Ok(i) => i.to_rgb8(), Err(_) => { err += 1; continue; } };
        match swap_scene(swapper, &scene, &faces, latents, args.face, args.all) {
            Ok((img, _)) => {
                let out = out_dir.join(default_out(path).file_name().unwrap());
                if write_image(&img, &out).is_ok() {
                    if args.restore { let _ = run_restore(&out, device.clone()).await; }
                    ok += 1;
                } else { err += 1; }
            }
            Err(_) => err += 1,
        }
    }
    println!("{}  batch: {ok} swapped · {noface} no-face · {err} failed → {}", style("✓").green(), out_dir.display());
    license_note();
    Ok(())
}

/// Video: swap every frame (detect per frame — faces move) and re-encode. Reuses `imaging::video`.
async fn faceswap_video(swapper: &FaceSwapper, latents: &[Tensor], args: &FaceswapArgs, device: Device) -> Result<()> {
    crate::imaging::video::ffmpeg_version().context("video face-swap needs ffmpeg on PATH")?;
    let out = args.out.clone().unwrap_or_else(|| default_out(&args.scene));
    let tmp = tempfile::tempdir().context("temp dir for video frames")?;
    let frames = crate::imaging::video::extract_frames(&args.scene, tmp.path())?;
    anyhow::ensure!(!frames.is_empty(), "no frames extracted from {}", args.scene.display());
    let fps = crate::imaging::video::probe_fps(&args.scene);
    println!("faceswap video: {} frames @ {fps}fps → {}", frames.len(), out.display());
    let (mut swapped_frames, mut noface) = (0u32, 0u32);
    for f in &frames {
        let faces = swapper.detect(f).unwrap_or_default();
        if faces.is_empty() {
            noface += 1;
            continue; // leave the frame as-is (no face this frame)
        }
        let scene = image::open(f)?.to_rgb8();
        if let Ok((img, _)) = swap_scene(swapper, &scene, &faces, latents, args.face, args.all) {
            img.save(f)?; // overwrite the extracted frame in place
            swapped_frames += 1;
        }
    }
    let pattern = tmp.path().join("frame_%06d.png");
    let pattern = pattern.to_string_lossy();
    match out.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).as_deref() {
        Some("webm") => crate::imaging::video::frames_to_webm(&pattern, &out, fps)?,
        Some("gif") => crate::imaging::video::frames_to_mp4(&pattern, &out, fps)?, // ffmpeg picks by ext
        _ => crate::imaging::video::frames_to_mp4(&pattern, &out, fps)?,
    }
    let _ = device;
    println!("{}  video: {swapped_frames} frames swapped · {noface} had no face → {}", style("✓").green(), out.display());
    license_note();
    Ok(())
}

async fn run_restore(path: &Path, device: Device) -> Result<()> {
    let cfg = crate::pipelines::adetailer::Config { device, ..crate::pipelines::adetailer::Config::defaults() };
    crate::pipelines::adetailer::refine_files(&cfg, &[path.to_path_buf()], None).await.context("restore pass")?;
    Ok(())
}

fn default_out(scene: &Path) -> PathBuf {
    let stem = scene.file_stem().and_then(|s| s.to_str()).unwrap_or("scene");
    let ext = scene.extension().and_then(|s| s.to_str()).unwrap_or("png");
    scene.with_file_name(format!("{stem}_swapped.{ext}"))
}

fn is_image(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "bmp" | "tiff")
    )
}

fn write_image(img: &RgbImage, out: &Path) -> Result<()> {
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    img.save(out).with_context(|| format!("writing {}", out.display()))
}

fn license_note() {
    println!("  {} inswapper is non-commercial (InsightFace) — personal / research use", style("note").dim());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_swaps_single_source_variants() {
        // --face N → just that face.
        assert_eq!(plan_swaps(3, 1, 1, false).unwrap(), vec![(1, 0)]);
        // --all → every face ← source 0.
        assert_eq!(plan_swaps(3, 1, 0, true).unwrap(), vec![(0, 0), (1, 0), (2, 0)]);
        // out-of-range face errors.
        assert!(plan_swaps(2, 1, 5, false).is_err());
    }

    #[test]
    fn plan_swaps_maps_multiple_sources_by_rank() {
        // 2 sources, 3 faces → source i → face i (largest-first); extra face untouched.
        assert_eq!(plan_swaps(3, 2, 0, false).unwrap(), vec![(0, 0), (1, 1)]);
        // more sources than faces → clamp to n_faces.
        assert_eq!(plan_swaps(1, 3, 0, false).unwrap(), vec![(0, 0)]);
    }

    #[test]
    fn plan_swaps_guards_empty() {
        assert!(plan_swaps(0, 1, 0, false).is_err(), "no faces");
        assert!(plan_swaps(2, 0, 0, false).is_err(), "no sources");
    }

    #[test]
    fn is_video_and_is_image_route_correctly() {
        assert!(is_video(Path::new("clip.MP4")));
        assert!(is_video(Path::new("a.gif")));
        assert!(!is_video(Path::new("photo.png")));
        assert!(is_image(Path::new("photo.JPG")));
        assert!(!is_image(Path::new("clip.mp4")));
    }
}
