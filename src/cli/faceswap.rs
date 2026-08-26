//! `plakat faceswap` — swap the face(s) in an existing image with a source face.
//!
//! The standalone verb for the proven [`FaceSwapper`] engine (SCRFD 5-point align →
//! ArcFace identity → `inswapper_128` → colour-matched feather paste-back). Unlike
//! `persona` / `multiperson` it does **not** generate a scene — it edits an image you
//! already have. inswapper weights are **non-commercial** (InsightFace) — gated behind
//! explicit opt-in (auto-resolved hosted safetensors / env overrides).

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use console::style;

use crate::pipelines::{adetailer, faceswap::FaceSwapper};

#[derive(clap::Args, Debug)]
pub struct FaceswapArgs {
    /// Scene image whose face(s) to replace.
    #[arg(value_name = "SCENE")]
    pub scene: PathBuf,
    /// Source face photo — the identity to swap IN (its largest face is used).
    #[arg(long, value_name = "FACE")]
    pub source: PathBuf,
    /// Which detected face to swap (0 = largest). Ignored with `--all`.
    #[arg(long, default_value_t = 0)]
    pub face: usize,
    /// Swap EVERY detected face with the source identity.
    #[arg(long, default_value_t = false)]
    pub all: bool,
    /// Output path (default: `<scene>_swapped.png`).
    #[arg(long, value_name = "PATH")]
    pub out: Option<PathBuf>,
    /// Run a light ADetailer detail pass on the result (crisper, but can slightly drift identity).
    #[arg(long, default_value_t = false)]
    pub restore: bool,
}

pub async fn run(args: FaceswapArgs, device: Device) -> Result<()> {
    let swapper = FaceSwapper::load_resolved(&device, DType::F32)
        .await
        .context("loading face-swap models (SCRFD + ArcFace + inswapper)")?;

    let faces = swapper.detect(&args.scene).context("detecting faces in the scene")?;
    anyhow::ensure!(
        !faces.is_empty(),
        "no face detected in {} (try a clearer / larger face)",
        args.scene.display()
    );

    let latent = swapper.source_latent(&args.source).context("embedding the source face")?;
    let mut scene = image::open(&args.scene)
        .with_context(|| format!("opening {}", args.scene.display()))?
        .to_rgb8();

    let targets: Vec<usize> = if args.all {
        (0..faces.len()).collect()
    } else {
        anyhow::ensure!(
            args.face < faces.len(),
            "--face {} out of range: {} face(s) detected",
            args.face,
            faces.len()
        );
        vec![args.face]
    };
    for &i in &targets {
        scene = swapper
            .swap_into(&scene, faces[i].landmarks, &latent)
            .with_context(|| format!("swapping face {i}"))?;
    }

    let out = args.out.clone().unwrap_or_else(|| {
        let stem = args.scene.file_stem().and_then(|s| s.to_str()).unwrap_or("scene");
        args.scene.with_file_name(format!("{stem}_swapped.png"))
    });
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    scene.save(&out).with_context(|| format!("writing {}", out.display()))?;

    if args.restore {
        let cfg = adetailer::Config { device, ..adetailer::Config::defaults() };
        adetailer::refine_files(&cfg, &[out.clone()], None)
            .await
            .context("restore pass")?;
    }

    println!("{}  swapped {} face(s) → {}", style("✓").green(), targets.len(), out.display());
    println!(
        "  {} inswapper is non-commercial (InsightFace) — personal / research use",
        style("note").dim()
    );
    Ok(())
}
