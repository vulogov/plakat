//! v0.16 phase 8: "Hires fix" workflow.
//!
//! SD 1.5 / SDXL were trained at fixed working resolutions (512² /
//! 1024²) — sampling much past that introduces the classic
//! "multi-head problem": the model can't track global composition
//! across more tokens than it saw at train time and produces
//! repeated faces, doubled limbs, malformed crowds.
//!
//! Hires fix is the standard mitigation:
//!
//! 1. **Generate at the trained resolution** (the t2i pass already
//!    did this — the user passes `--size 768x768` for SD 1.5 / 1024²
//!    for SDXL).
//! 2. **Upscale** the result classically (Lanczos) or via
//!    Real-ESRGAN. Lanczos is fast + sharp; ESRGAN reconstructs
//!    high-frequency detail at extra compute cost.
//! 3. **img2img the upscaled image** at moderate strength (default
//!    0.5) so the model refines small-scale detail without losing
//!    the composition it just got right at native res.
//!
//! Same SdCore reuse pattern as ADetailer + artefact-blend — the
//! `shared_core` arg lets the caller hand in the SdCore t2i just
//! loaded, avoiding a second multi-GB load.

use anyhow::{Context, Result};
use candle_core::Device;
use std::path::PathBuf;
use std::sync::Arc;

use crate::imaging::upscale::{self, Method};
use crate::pipelines::img2img;
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::portrait::{self, LoadRequest};
use crate::pipelines::scheduler::SchedulerKind;
use crate::pipelines::sd_core::SdCore;

/// Configuration for one batch of hires-fix refinement passes.
pub struct Config {
    /// SD model alias / repo. Must match the model the input files
    /// were generated with — hires fix reuses the same SdCore when
    /// `shared_core` is supplied. Diverging models produce stylistic
    /// drift between the t2i pass and the refine pass.
    pub model: String,
    /// LoRA stack to apply during the refine img2img. Typically the
    /// same stack the t2i pass used.
    pub loras: Vec<LoraSpec>,
    pub lora_scale: f32,
    /// Prompt for the refine pass. Defaults to the t2i prompt.
    pub prompt: String,
    pub negative: String,
    /// Upscale factor (e.g. `2.0` for 768² → 1536²). Classical
    /// upscalers honour this directly; ML upscalers use their
    /// native scale and ignore this value with a warning.
    pub scale: f32,
    /// Upscaler method. `Lanczos3` is fast + sharp; `RealEsrganX2`
    /// / `RealEsrganX4` reconstruct high-frequency detail at extra
    /// compute cost.
    pub upscaler: Method,
    /// img2img strength on the upscaled image. `0.5` (default)
    /// preserves the t2i composition + adds refinement; `0.7+`
    /// allows more reinterpretation.
    pub strength: f32,
    /// Step count of the refine pass.
    pub steps: usize,
    /// CFG guidance for the refine pass.
    pub guidance: f64,
    /// Scheduler for the refine pass.
    pub scheduler: SchedulerKind,
    pub device: Device,
}

/// Run hires fix over `files` in place. Each file is read, upscaled,
/// img2img-refined, and written back. Returns the count of files
/// successfully refined.
pub async fn refine_files(
    cfg: &Config,
    files: &[PathBuf],
    shared_core: Option<Arc<SdCore>>,
) -> Result<usize> {
    if files.is_empty() {
        return Ok(0);
    }

    // ESRGAN pipeline is lazy — only built when the user picked an
    // ML upscaler. Reused across files within one batch so the
    // weights load once.
    let esrgan = if cfg.upscaler.is_ml() {
        Some(
            upscale::EsrganPipeline::load(cfg.upscaler, &cfg.device)
                .await
                .context("loading Real-ESRGAN for hires fix")?,
        )
    } else {
        None
    };

    let pipeline = match shared_core {
        Some(core) => portrait::Pipeline::from_core(core),
        None => portrait::Pipeline::load(LoadRequest {
            model: cfg.model.clone(),
            device: cfg.device.clone(),
            loras: cfg.loras.clone(),
            lora_scale: cfg.lora_scale,
            identity: None,
            shared_clip_h: None,
        })
        .await
        .context("loading SD pipeline for hires fix")?,
    };

    let tmpdir = tempfile::Builder::new()
        .prefix("plakat-hires-")
        .tempdir()
        .context("creating hires-fix tempdir")?;

    let mut refined = 0usize;
    for (idx, path) in files.iter().enumerate() {
        refine_one(cfg, esrgan.as_ref(), path, &pipeline, tmpdir.path(), idx)
            .await
            .with_context(|| format!("hires fix on {}", path.display()))?;
        refined += 1;
    }
    Ok(refined)
}

async fn refine_one(
    cfg: &Config,
    esrgan: Option<&upscale::EsrganPipeline>,
    target: &std::path::Path,
    pipeline: &portrait::Pipeline,
    tmpdir: &std::path::Path,
    idx: usize,
) -> Result<()> {
    // 1. Upscale the original t2i output into a tmp file. ML and
    // classical paths land at different scales — ML is fixed
    // (2x/4x by model); classical honours `cfg.scale`.
    let upscaled = tmpdir.join(format!("hires_upscaled_{idx}.png"));
    let (in_w, in_h, out_w, out_h) = if let Some(p) = esrgan {
        p.upscale_file(target, &upscaled)?
    } else {
        upscale::upscale(target, &upscaled, cfg.scale, cfg.upscaler)?
    };
    tracing::debug!(
        target: "plakat",
        "hires upscale: {} {}x{} → {}x{} via {:?}",
        target.display(),
        in_w, in_h, out_w, out_h, cfg.upscaler
    );

    // Snap to multiples of 8 (VAE downsample). image's resize gives
    // exact dims; we may need to crop to /8 before passing to img2img.
    let snap_w = (out_w / 8) * 8;
    let snap_h = (out_h / 8) * 8;
    if snap_w == 0 || snap_h == 0 {
        anyhow::bail!(
            "hires fix: upscaled image {}x{} too small for VAE (need ≥8 on each axis)",
            out_w, out_h
        );
    }
    let snapped = if snap_w != out_w || snap_h != out_h {
        let snapped_path = tmpdir.join(format!("hires_snapped_{idx}.png"));
        let img = image::open(&upscaled)?.to_rgb8();
        let snapped_img = image::imageops::crop_imm(&img, 0, 0, snap_w, snap_h).to_image();
        snapped_img.save(&snapped_path)?;
        snapped_path
    } else {
        upscaled.clone()
    };

    // 2. img2img the upscaled image at moderate strength. Same
    // model + LoRAs as the t2i pass (caller's responsibility to
    // configure `Config.model` / `Config.loras` consistently).
    let req = img2img::Request {
        prompt: cfg.prompt.clone(),
        negative: cfg.negative.clone(),
        model: cfg.model.clone(),
        device: cfg.device.clone(),
        loras: cfg.loras.clone(),
        lora_scale: cfg.lora_scale,
        input: snapped.clone(),
        mask: None,
        mask_feather: 0,
        mask_invert: false,
        width: snap_w,
        height: snap_h,
        count: 1,
        steps: cfg.steps,
        guidance: cfg.guidance,
        scheduler: cfg.scheduler,
        strength: cfg.strength,
        // Seed-passthrough: portrait/img2img will roll its own RNG.
        // Reproducibility-conscious callers can extract the seed
        // from the input filename and pass it explicitly in a future
        // wiring; the current contract is "refine in place".
        seed: None,
        out_dir: tmpdir.to_path_buf(),
        controls: Vec::new(),
    };
    img2img::run_with_pipeline(pipeline, &req)
        .await
        .with_context(|| format!("img2img refine of hires-upscaled {}", target.display()))?;

    // 3. Find the img2img output and move it over the original.
    // img2img names its output `plakat-img2img-<seed>.png` so we
    // pick the freshest file with that prefix.
    let refined = find_freshest(tmpdir, "plakat-img2img-")?;
    std::fs::copy(&refined, target)
        .with_context(|| format!("copying refined {} → {}", refined.display(), target.display()))?;
    // Clean up so the next file's `find_freshest` is unambiguous.
    let _ = std::fs::remove_file(&refined);
    let _ = std::fs::remove_file(&upscaled);
    if snapped != upscaled {
        let _ = std::fs::remove_file(&snapped);
    }
    Ok(())
}

/// Find the freshest file in `dir` whose name starts with `prefix`.
/// "Freshest" = highest modified time. Used by `refine_one` to
/// pick out the just-written img2img output.
fn find_freshest(dir: &std::path::Path, prefix: &str) -> Result<PathBuf> {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if !name.starts_with(prefix) {
            continue;
        }
        let mtime = entry.metadata()?.modified().unwrap_or(std::time::UNIX_EPOCH);
        match best.as_ref() {
            Some((t, _)) if *t >= mtime => {}
            _ => best = Some((mtime, path)),
        }
    }
    best.map(|(_, p)| p).ok_or_else(|| {
        anyhow::anyhow!(
            "no img2img output found in {} (prefix={prefix})",
            dir.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_freshest_picks_newest_match() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join("plakat-img2img-1.png"), b"old").unwrap();
        // Ensure mtime ordering even on filesystems with second-level
        // resolution.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        std::fs::write(dir.join("plakat-img2img-2.png"), b"new").unwrap();
        std::fs::write(dir.join("other.png"), b"distract").unwrap();
        let p = find_freshest(dir, "plakat-img2img-").unwrap();
        assert!(p.ends_with("plakat-img2img-2.png"), "got {}", p.display());
    }

    #[test]
    fn find_freshest_errors_when_no_match() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("unrelated.png"), b"x").unwrap();
        let err = find_freshest(tmp.path(), "plakat-img2img-").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("no img2img output"), "got {msg}");
    }
}
