//! `plakat faceswap` — swap the face(s) in an existing image (or a folder / video) with source face(s).
//!
//! The standalone verb for the proven [`FaceSwapper`] engine (SCRFD 5-point align → ArcFace identity →
//! `inswapper_128` → colour-matched feather paste-back). Unlike `persona` / `multiperson` it does **not**
//! generate a scene — it edits media you already have.
//!
//! 6.21.0 (RFC FACESWAP-3) depth: `--dry-run`/`--preview` (inspect detections), a **directory** input
//! (batch), **repeatable `--source`** (per-face identities matched by ArcFace recognition, not size rank),
//! `--source-face N`, small-face sharpen, and **video** input (swap every frame). inswapper weights are
//! **non-commercial** (InsightFace) — gated behind opt-in.

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
    /// Source face photo. **Repeatable**: with K sources, each is matched to its closest detected face by
    /// ArcFace **recognition** (identity-follows-face, robust to size order). Not needed with `--dry-run`.
    #[arg(long, value_name = "FACE")]
    pub source: Vec<PathBuf>,
    /// With ONE source: which detected face to swap (0 = largest). Ignored with `--all` / multiple sources.
    #[arg(long, default_value_t = 0)]
    pub face: usize,
    /// With ONE source: swap EVERY detected face with it.
    #[arg(long, default_value_t = false)]
    pub all: bool,
    /// Which face in each SOURCE photo is the identity (0 = largest). For multi-face source photos.
    #[arg(long, default_value_t = 0)]
    pub source_face: usize,
    /// Output path (image), or output directory (batch). Default: `<scene>_swapped.<ext>`.
    #[arg(long, value_name = "PATH")]
    pub out: Option<PathBuf>,
    /// Run a light ADetailer detail pass on the result (crisper, but can slightly drift identity).
    /// The pass only touches detected face boxes (feathered), so the rest of the image is preserved.
    #[arg(long, default_value_t = false)]
    pub restore: bool,
    /// Paste-back edge feather in px (default 16). Larger = softer seam, smaller = crisper edge (S2).
    #[arg(long, value_name = "PX")]
    pub feather: Option<f32>,
    /// Disable the skin-tone colour match (paste the raw swap; S2).
    #[arg(long = "no-color-match", default_value_t = false)]
    pub no_color_match: bool,
    /// Multi-source face→source mapping: `identity` (ArcFace recognition, default) or `rank` (by size).
    #[arg(long = "match", value_name = "MODE", default_value = "identity")]
    pub match_by: String,
    /// Print which source matched which face (+ cosine) for a multi-source swap (S2).
    #[arg(long, default_value_t = false)]
    pub report: bool,
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

/// A resolved source: its swap `latent` + its recognition `emb`edding (for R1 matching).
struct Source {
    latent: Tensor,
    emb: Vec<f32>,
}

/// Cosine-similarity matrix `sim[s][f]` between source embeddings and face embeddings (all unit-norm →
/// cosine = dot). Pure.
fn cosine_matrix(sources: &[Vec<f32>], faces: &[Vec<f32>]) -> Vec<Vec<f32>> {
    sources
        .iter()
        .map(|s| faces.iter().map(|f| s.iter().zip(f).map(|(a, b)| a * b).sum::<f32>()).collect())
        .collect()
}

/// **Greedy recognition assignment** (RFC FACESWAP-3 R1, pure/tested): given `sim[s][f]` cosine, assign
/// each source to its most-similar still-free face (highest similarity first). Returns `(face, source)`
/// pairs — so identities follow the FACE, not its size rank. Stops when sources or faces run out.
fn match_sources_to_faces(sim: &[Vec<f32>]) -> Vec<(usize, usize)> {
    let n_src = sim.len();
    let n_face = sim.first().map(|r| r.len()).unwrap_or(0);
    let (mut used_s, mut used_f) = (vec![false; n_src], vec![false; n_face]);
    let mut pairs = Vec::new();
    for _ in 0..n_src.min(n_face) {
        let mut best: Option<(f32, usize, usize)> = None;
        for s in 0..n_src {
            if used_s[s] {
                continue;
            }
            for f in 0..n_face {
                if used_f[f] {
                    continue;
                }
                if best.map(|(b, _, _)| sim[s][f] > b).unwrap_or(true) {
                    best = Some((sim[s][f], s, f));
                }
            }
        }
        let Some((_, s, f)) = best else { break };
        used_s[s] = true;
        used_f[f] = true;
        pairs.push((f, s));
    }
    pairs
}

/// Swap a loaded scene against precomputed `sources`. With >1 source, faces are matched to sources by
/// **recognition** (R1: ArcFace cosine, identity-follows-face); with one source, by `--face`/`--all`.
/// Returns the edited image + the number of faces swapped.
fn swap_scene(swapper: &FaceSwapper, scene: &RgbImage, faces: &[Face], sources: &[Source], face: usize, all: bool, match_by: &str, report: bool) -> Result<(RgbImage, usize)> {
    let plan = if sources.len() > 1 && match_by != "rank" {
        let face_embs: Vec<Vec<f32>> = faces
            .iter()
            .map(|f| Ok(swapper.face_embedding(scene, f.landmarks)?.flatten_all()?.to_vec1::<f32>()?))
            .collect::<Result<_>>()?;
        let src_embs: Vec<Vec<f32>> = sources.iter().map(|s| s.emb.clone()).collect();
        let sim = cosine_matrix(&src_embs, &face_embs);
        let pairs = match_sources_to_faces(&sim);
        if report {
            for &(f, s) in &pairs {
                println!("  {} source {s} → face {f}  (cos {:.3})", style("match").cyan(), sim[s][f]);
            }
        }
        pairs
    } else {
        plan_swaps(faces.len(), sources.len(), face, all)?
    };
    let mut out = scene.clone();
    for &(fi, si) in &plan {
        out = swapper.swap_into(&out, faces[fi].landmarks, &sources[si].latent).with_context(|| format!("swapping face {fi}"))?;
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
    anyhow::ensure!(matches!(args.match_by.as_str(), "identity" | "rank"), "--match must be `identity` or `rank`");
    let mut swapper = FaceSwapper::load_resolved(&device, DType::F32)
        .await
        .context("loading face-swap models (SCRFD + ArcFace + inswapper)")?;
    if let Some(f) = args.feather {
        swapper.feather = f; // S2 --feather
    }
    if args.no_color_match {
        swapper.color_match = false; // S2 --no-color-match
    }
    let sources: Vec<Source> = args
        .source
        .iter()
        .map(|s| {
            Ok(Source {
                latent: swapper.source_latent_n(s, args.source_face).with_context(|| format!("embedding source {}", s.display()))?,
                emb: swapper.source_embedding_n(s, args.source_face)?.flatten_all()?.to_vec1::<f32>()?,
            })
        })
        .collect::<Result<_>>()?;

    if is_video(&args.scene) {
        return faceswap_video(&swapper, &sources, &args, device).await;
    }
    if args.scene.is_dir() {
        return faceswap_batch(&swapper, &sources, &args, device).await;
    }

    // ── single image ──
    let faces = swapper.detect(&args.scene).context("detecting faces in the scene")?;
    anyhow::ensure!(!faces.is_empty(), "no face detected in {} (try a clearer / larger face)", args.scene.display());
    let scene = image::open(&args.scene).with_context(|| format!("opening {}", args.scene.display()))?.to_rgb8();
    let (out_img, n) = swap_scene(&swapper, &scene, &faces, &sources, args.face, args.all, &args.match_by, args.report)?;

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
async fn faceswap_batch(swapper: &FaceSwapper, sources: &[Source], args: &FaceswapArgs, device: Device) -> Result<()> {
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
        match swap_scene(swapper, &scene, &faces, sources, args.face, args.all, &args.match_by, false) {
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
async fn faceswap_video(swapper: &FaceSwapper, sources: &[Source], args: &FaceswapArgs, device: Device) -> Result<()> {
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
        if let Ok((img, _)) = swap_scene(swapper, &scene, &faces, sources, args.face, args.all, &args.match_by, false) {
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
    img.save(out).with_context(|| format!("writing {}", out.display()))?;
    // S4 (FACESWAP-3): honor the global `--etch` — this is an image plakat produced. A fresh, parentless
    // etch (L0 manifest + L1 pixel mark) is minted over the swapped pixels.
    if crate::etch::active().is_some() {
        let _ = crate::etch::fresh_etch(img.as_raw(), img.width(), img.height(), out, None);
    }
    Ok(())
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
    fn recognition_match_follows_identity_not_size() {
        // sources s0,s1; faces f0,f1. s0 looks like f1, s1 looks like f0 (identity crossed vs size order).
        let sim = vec![vec![0.1, 0.9], vec![0.85, 0.2]]; // sim[s][f]
        let pairs = match_sources_to_faces(&sim); // (face, source)
        assert!(pairs.contains(&(1, 0)), "s0 → f1 (its match)");
        assert!(pairs.contains(&(0, 1)), "s1 → f0 (its match)");
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn recognition_match_is_greedy_highest_first() {
        // global best is (s0,f0)=0.95; s1 is then forced onto the only free face f1.
        let sim = vec![vec![0.95, 0.9], vec![0.8, 0.1]];
        let pairs = match_sources_to_faces(&sim);
        assert_eq!(pairs, vec![(0, 0), (1, 1)]);
        // fewer faces than sources → stop when faces run out.
        assert_eq!(match_sources_to_faces(&[vec![0.5], vec![0.9]]), vec![(0, 1)]);
    }

    #[test]
    fn cosine_matrix_is_dot_of_unit_vectors() {
        let s = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let f = vec![vec![1.0, 0.0], vec![0.7071, 0.7071]];
        let m = cosine_matrix(&s, &f);
        assert!((m[0][0] - 1.0).abs() < 1e-4, "identical → 1");
        assert!((m[0][1] - 0.7071).abs() < 1e-3, "45° → ~0.707");
        assert!((m[1][0] - 0.0).abs() < 1e-4, "orthogonal → 0");
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
