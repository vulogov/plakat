//! Photos-local resident ML worker (AI-track Phase A). A dedicated background thread owns the loaded
//! pipelines (the SDXL img2img backbone, IC-Light, Real-ESRGAN) so successive ML edits **reuse** the
//! weights instead of reloading per op, and drives them off the event-loop thread — progress + cancel
//! flow back over a [`GenMessage`] channel the manager drains **inline** each tick (no more suspending
//! the TUI). It has zero dependency on `plakat ui`: it mirrors the shape of that crate's `ModelService`
//! but is photos-scoped and only wires the three ops the photos ML menu exposes.
//!
//! The loaded pipelines are not `Send`; they live entirely on the worker thread and never cross it.
//! Only the job parameters (in) and [`GenMessage`]s (out) move between threads. Async pipeline loads
//! run via `Handle::block_on` on this thread — never on the manager's async event loop.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result};
use candle_core::Device;
use tokio::runtime::Handle;

use crate::imaging::upscale::{EsrganPipeline, Method};
use crate::pipelines::gen_channel::{CancelFlag, ChannelHook, GenMessage};
use crate::pipelines::scheduler::SchedulerKind;
use crate::pipelines::{ic_light, img2img, portrait, t2i};

use super::mledit::{dest_path, MlOp};

/// The SDXL backbone used for img2img (matches the old `MlJob` default — best quality on 24 GB).
const IMG2IMG_MODEL: &str = "sdxl";

/// A job handed to the worker: the op, its source image, the album to land the result in, and the
/// channel + cancel flag the manager watches.
pub struct Job {
    pub op: MlOp,
    pub input: PathBuf,
    pub album: PathBuf,
    pub tx: Sender<GenMessage>,
    pub cancel: CancelFlag,
}

enum Cmd {
    Run(Job),
    Shutdown,
}

/// Handle to the background worker. Dropping it stops the thread (after the in-flight job).
pub struct MlWorker {
    cmd_tx: Sender<Cmd>,
    handle: Option<JoinHandle<()>>,
}

impl MlWorker {
    /// Spawn the worker. `rt` is the manager's tokio handle (used for `block_on` of the async
    /// pipeline loads, on this fresh thread).
    pub fn spawn(rt: Handle) -> Self {
        let (cmd_tx, cmd_rx) = channel::<Cmd>();
        let handle = std::thread::Builder::new()
            .name("plakat-photos-ml".into())
            .spawn(move || worker_loop(cmd_rx, rt))
            .expect("spawn photos ML worker thread");
        MlWorker { cmd_tx, handle: Some(handle) }
    }

    /// Queue a job (non-blocking). Returns `false` if the worker thread is gone.
    pub fn submit(&self, job: Job) -> bool {
        self.cmd_tx.send(Cmd::Run(job)).is_ok()
    }
}

impl Drop for MlWorker {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(Cmd::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Pipelines held resident across jobs. Each is loaded lazily on first use and reused thereafter.
#[derive(Default)]
struct Resident {
    /// (alias, backbone) for img2img. Reloaded only if a different alias is requested (today it's
    /// always `sdxl`, so it loads once).
    sd: Option<(String, t2i::Pipeline)>,
    ic: Option<ic_light::Pipeline>,
    esrgan: Option<EsrganPipeline>,
}

fn worker_loop(rx: Receiver<Cmd>, rt: Handle) {
    // Resolve the device once. If it fails, every job reports the error (rather than the thread dying).
    let device = crate::api::device("auto");
    // Arm the OOM watchdog for the worker's whole lifetime (inert unless Metal + enabled). If a load
    // or denoise drives unified memory to the cliff, the guard hard-exits cleanly — the abort hook
    // (registered in `run_with`) restores the terminal and prints the reason. This is the "guard
    // fired" path: the process can't survive to send a channel message, so the report is the guard's
    // own stderr line the user sees in their shell. The soft path below catches the common cases first.
    let _mem_guard = device.as_ref().ok().map(|d| crate::memwatch::MemoryGuard::start(d, "plakat photos"));
    let mut res = Resident::default();
    while let Ok(cmd) = rx.recv() {
        let job = match cmd {
            Cmd::Shutdown => break,
            Cmd::Run(job) => job,
        };
        let dev = match &device {
            Ok(d) => d.clone(),
            Err(e) => {
                let _ = job.tx.send(GenMessage::Error { message: format!("no compute device: {e:#}") });
                continue;
            }
        };
        // Soft OOM preflight: if the kernel already reports CRITICAL memory pressure, refuse and
        // report to the UI *before* attempting the heavy allocation that would trip the hard-exit
        // guard. (We trust only the kernel pressure signal — free-RAM under-reports reclaimable pages
        // on macOS and would cry wolf, per the memwatch guard's own reasoning.)
        if let Err(e) = oom_preflight() {
            let _ = job.tx.send(GenMessage::Error { message: format!("{e:#}") });
            continue;
        }
        // Catch a pipeline panic so one bad op reports to the UI instead of silently killing the
        // worker (which would disconnect the channel and strand future jobs). On a panic the resident
        // pipelines may be in a half-built state, so drop them — the next job reloads cleanly.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_job(&mut res, &rt, &dev, &job)
        }));
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let _ = job.tx.send(GenMessage::Error { message: format!("{e:#}") });
            }
            Err(_) => {
                res = Resident::default();
                let _ = job.tx.send(GenMessage::Error {
                    message: "the ML pipeline crashed (out of memory or an internal error) — models freed, try again".into(),
                });
            }
        }
    }
}

/// Soft out-of-memory preflight. Returns an error (reported to the UI) when the kernel's own
/// memory-pressure level is already CRITICAL — attempting a multi-GB load in that state risks the
/// watchdog hard-exiting the whole process, so we refuse gracefully and keep the manager alive.
fn oom_preflight() -> Result<()> {
    if crate::hw::mem_pressure() == crate::hw::Pressure::Critical {
        anyhow::bail!(
            "out of memory: system pressure is CRITICAL — free RAM (close apps / `sudo purge`) and retry"
        );
    }
    Ok(())
}

fn run_job(res: &mut Resident, rt: &Handle, device: &Device, job: &Job) -> Result<()> {
    std::fs::create_dir_all(&job.album).ok();
    let out = dest_path(&job.album, &job.input, job.op.suffix());
    match &job.op {
        MlOp::Img2img { prompt } => {
            // Ensure the backbone is resident (reuse if the alias matches).
            if res.sd.as_ref().map(|(a, _)| a != IMG2IMG_MODEL).unwrap_or(true) {
                let lr = t2i::LoadRequest {
                    model: IMG2IMG_MODEL.into(),
                    device: device.clone(),
                    loras: Vec::new(),
                    lora_scale: 1.0,
                    use_refiner: false,
                    embeddings: Vec::new(),
                    vae_cache: None,
                };
                let pipe = rt.block_on(t2i::Pipeline::load(lr)).context("loading SDXL for img2img")?;
                res.sd = Some((IMG2IMG_MODEL.into(), pipe));
            }
            let (_, pipe) = res.sd.as_ref().expect("backbone loaded above");
            let refine = portrait::Pipeline::from_core(pipe.core());
            let seed = rand::random::<u64>();
            let req = img2img::Request {
                prompt: prompt.clone(),
                negative: String::new(),
                model: IMG2IMG_MODEL.into(),
                device: device.clone(),
                loras: Vec::new(),
                lora_scale: 1.0,
                input: job.input.clone(),
                mask: None,
                mask_feather: 0,
                mask_invert: false,
                width: 0, // keep input dims
                height: 0,
                count: 1,
                steps: 20,
                guidance: 7.5,
                scheduler: SchedulerKind::default(),
                strength: 0.5,
                seed: Some(seed),
                out_dir: job.album.clone(),
                controls: Vec::new(),
            };
            let mut hook = ChannelHook::new(job.tx.clone(), job.cancel.clone(), 0);
            rt.block_on(img2img::run_with_pipeline_hooked(&refine, &req, Some(&mut hook)))
                .context("img2img")?;
            if job.cancel.is_cancelled() {
                let _ = job.tx.send(GenMessage::Done { output: out, cancelled: true });
                return Ok(());
            }
            // The pipeline names its output `plakat-img2img-<seed>.png`; move it to our variant name.
            let produced = job.album.join(format!("plakat-img2img-{seed}.png"));
            adopt(&produced, &out)?;
            let _ = job.tx.send(GenMessage::Done { output: out, cancelled: false });
        }
        MlOp::Relight { prompt } => {
            if res.ic.is_none() {
                res.ic = Some(rt.block_on(ic_light::Pipeline::load(device.clone())).context("loading IC-Light")?);
            }
            let ic = res.ic.as_ref().expect("ic-light loaded above");
            // IC-Light has no per-step hook; emit a single indeterminate tick so the bar isn't blank.
            let _ = job.tx.send(GenMessage::Progress {
                step: 0,
                total: 1,
                elapsed: Duration::ZERO,
                steps_per_sec: 0.0,
            });
            let seed = rand::random::<u64>();
            let (pixels, w, h) = ic.relight(&job.input, prompt, "", 512, 512, 20, 2.0, seed, crate::pipelines::ic_light::Backdrop::Flat).context("relight")?;
            let img = image::RgbImage::from_raw(w, h, pixels)
                .ok_or_else(|| anyhow::anyhow!("relight buffer size mismatch"))?;
            img.save(&out).with_context(|| format!("saving {}", out.display()))?;
            let _ = job.tx.send(GenMessage::Done { output: out, cancelled: false });
        }
        MlOp::Upscale => {
            // Photos always upscales ×4; load the ESRGAN model once and hold it resident.
            if res.esrgan.is_none() {
                let pipe = rt
                    .block_on(EsrganPipeline::load(Method::RealEsrganX4, device))
                    .context("loading Real-ESRGAN")?;
                res.esrgan = Some(pipe);
            }
            let pipe = res.esrgan.as_ref().expect("esrgan loaded above");
            let _ = job.tx.send(GenMessage::Progress {
                step: 0,
                total: 1,
                elapsed: Duration::ZERO,
                steps_per_sec: 0.0,
            });
            pipe.upscale_file(&job.input, &out).context("upscale")?;
            let _ = job.tx.send(GenMessage::Done { output: out, cancelled: false });
        }
    }
    Ok(())
}

/// Move `produced` to `dest` (rename, falling back to copy+remove across filesystems).
fn adopt(produced: &Path, dest: &Path) -> Result<()> {
    if produced == dest {
        return Ok(());
    }
    if std::fs::rename(produced, dest).is_err() {
        std::fs::copy(produced, dest).with_context(|| format!("placing {}", dest.display()))?;
        let _ = std::fs::remove_file(produced);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The worker spawns and shuts down cleanly (Drop sends Shutdown + joins) without loading any
    /// model — a lifecycle smoke test that must not hang.
    #[test]
    fn spawn_and_shutdown_is_clean() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let worker = MlWorker::spawn(rt.handle().clone());
        drop(worker); // Shutdown + join; would hang if the loop didn't honour Shutdown.
    }

    #[test]
    fn adopt_renames_into_place() {
        let dir = std::env::temp_dir().join(format!("plakat-mlworker-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("produced.png");
        let dst = dir.join("final.png");
        std::fs::write(&src, b"pixels").unwrap();
        adopt(&src, &dst).unwrap();
        assert!(!src.exists() && dst.exists(), "moved to dest");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
