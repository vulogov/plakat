//! `ModelService` — a dedicated background thread that owns the loaded model and
//! serialises load/unload (RFC TUI-1 §16). A multi-GB load can't run on the event
//! loop without freezing the UI, and the loaded pipeline isn't moved across threads
//! (so no `Send` gymnastics): one thread owns it, receives [`ModelCommand`]s, and
//! reports [`ModelMessage`]s the `App` drains each tick. The same thread will run
//! generations later (Metal device exclusivity for free).
//!
//! This increment loads the SD-family t2i pipeline (sd15 / sd21 / sdxl). Other
//! families return a friendly "not yet wired" message rather than a cryptic error.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use std::path::PathBuf;

use candle_core::Device;
use tokio::runtime::Handle;

use crate::pipelines::gen_channel::{CancelFlag, ChannelHook, GenMessage};
use crate::pipelines::{img2img, portrait, t2i};
use crate::preset::discovery::BaseFamily;

/// Parameters for a Chat generation (the model thread builds the GenRequest).
pub struct GenJob {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub guidance: f64,
    pub seed: u64,
    pub out_dir: PathBuf,
    pub preview_every: usize,
    /// `Some(path)` → conversational refinement: img2img over this image at
    /// `strength`, reusing the loaded weights. `None` → fresh txt2img.
    pub init_image: Option<PathBuf>,
    pub strength: f32,
    /// `Some(provider)` → AI-enhance the prompt (`/enhance`) before generating.
    pub enhance: Option<String>,
    pub tx: Sender<GenMessage>,
    pub cancel: CancelFlag,
}

/// A request to the model thread.
pub enum ModelCommand {
    Load { alias: String, loras: Vec<crate::pipelines::lora::LoraSpec> },
    Unload,
    Generate(GenJob),
    Shutdown,
}

/// A status update from the model thread.
#[derive(Debug, Clone)]
pub enum ModelMessage {
    /// A load has begun (the multi-GB download/map runs in the background).
    LoadStarted(String),
    /// Loaded; `used_gb` is the system memory in use just after the load.
    Loaded { alias: String, used_gb: f64 },
    /// Unloaded — no model resident.
    Unloaded,
    /// A load/unload failed (or the model family isn't wired yet).
    Error(String),
}

/// Whether the t2i loader handles this model's family. Pure — unit-tested. Returns
/// `Err(message)` with friendly guidance for families not yet wired in the TUI.
pub fn t2i_load_check(alias: &str) -> Result<(), String> {
    match BaseFamily::from_model_arg(alias) {
        BaseFamily::Sd15 | BaseFamily::Sd21 | BaseFamily::Sdxl => Ok(()),
        other => Err(format!(
            "'{alias}' is a {other:?} model — the TUI loads SD-family models \
             (sd15 / sd21 / sdxl) in this release; {other:?} support is a follow-up."
        )),
    }
}

/// Handle to the model thread. Drop signals shutdown and joins.
pub struct ModelService {
    cmd_tx: Sender<ModelCommand>,
    msg_rx: Receiver<ModelMessage>,
    handle: Option<JoinHandle<()>>,
}

impl ModelService {
    /// Spawn the model thread. `rt` is the app's tokio runtime handle, used to
    /// `block_on` the async pipeline load from this (non-runtime) thread.
    pub fn spawn(device: Device, rt: Handle) -> Self {
        let (cmd_tx, cmd_rx) = channel::<ModelCommand>();
        let (msg_tx, msg_rx) = channel::<ModelMessage>();
        let handle = std::thread::Builder::new()
            .name("plakat-model-svc".into())
            .spawn(move || model_loop(cmd_rx, msg_tx, device, rt))
            .expect("spawn model service");
        Self { cmd_tx, msg_rx, handle: Some(handle) }
    }

    /// Load `alias`, merging `loras` into the weights (LoRA application is load-time).
    pub fn load(&self, alias: impl Into<String>, loras: Vec<crate::pipelines::lora::LoraSpec>) {
        let _ = self.cmd_tx.send(ModelCommand::Load { alias: alias.into(), loras });
    }

    pub fn unload(&self) {
        let _ = self.cmd_tx.send(ModelCommand::Unload);
    }

    /// Dispatch a generation to the model thread. Returns the channel of
    /// `GenMessage`s (Progress/Preview/Done/Error) to drain + the cancel flag.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: String,
        negative: String,
        width: u32,
        height: u32,
        steps: usize,
        guidance: f64,
        seed: u64,
        out_dir: PathBuf,
        preview_every: usize,
        init_image: Option<PathBuf>,
        strength: f32,
        enhance: Option<String>,
    ) -> (std::sync::mpsc::Receiver<GenMessage>, CancelFlag) {
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = CancelFlag::new();
        let job = GenJob {
            prompt, negative, width, height, steps, guidance, seed, out_dir, preview_every,
            init_image, strength, enhance,
            tx, cancel: cancel.clone(),
        };
        let _ = self.cmd_tx.send(ModelCommand::Generate(job));
        (rx, cancel)
    }

    /// Non-blocking drain of one status message (called from the event-loop tick).
    pub fn try_recv(&self) -> Option<ModelMessage> {
        self.msg_rx.try_recv().ok()
    }
}

impl Drop for ModelService {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(ModelCommand::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn model_loop(
    cmd_rx: Receiver<ModelCommand>,
    msg_tx: Sender<ModelMessage>,
    device: Device,
    rt: Handle,
) {
    // The loaded pipeline lives here and never leaves this thread.
    let mut loaded: Option<(String, t2i::Pipeline)> = None;
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            ModelCommand::Load { alias, loras } => {
                if let Err(msg) = t2i_load_check(&alias) {
                    let _ = msg_tx.send(ModelMessage::Error(msg));
                    continue;
                }
                let _ = msg_tx.send(ModelMessage::LoadStarted(alias.clone()));
                // Free the current model first — unified memory means we can't hold
                // two large models at once.
                loaded = None;
                let req = t2i::LoadRequest {
                    model: alias.clone(),
                    device: device.clone(),
                    loras,
                    lora_scale: 1.0,
                    use_refiner: false,
                    embeddings: Vec::new(),
                    vae_cache: None,
                };
                match rt.block_on(t2i::Pipeline::load(req)) {
                    Ok(p) => {
                        let used = (crate::hw::total_ram_gb() - crate::hw::available_ram_gb()).max(0.0);
                        loaded = Some((alias.clone(), p));
                        let _ = msg_tx.send(ModelMessage::Loaded { alias, used_gb: used });
                    }
                    Err(e) => {
                        let _ = msg_tx.send(ModelMessage::Error(format!("load {alias}: {e:#}")));
                    }
                }
            }
            ModelCommand::Unload => {
                loaded = None;
                let _ = msg_tx.send(ModelMessage::Unloaded);
            }
            ModelCommand::Generate(job) => {
                let Some((alias, pipeline)) = &loaded else {
                    let _ = job.tx.send(GenMessage::Error {
                        message: "no model loaded — load one in Models (Ctrl-2)".into(),
                    });
                    continue;
                };

                // ── /enhance: expand the prompt with the configured LLM first. ──
                // On failure, fall back to the original prompt (the run still goes).
                let mut prompt = job.prompt.clone();
                if let Some(provider) = &job.enhance {
                    let label = crate::prompt::resolve_provider_label(provider);
                    crate::ui::progress::println(&format!("  ✨ enhancing prompt via {label} …"));
                    match rt.block_on(crate::prompt::enhance(provider, &prompt)) {
                        Ok(enhanced) => {
                            prompt = enhanced.clone();
                            let _ = job.tx.send(GenMessage::Enhanced { prompt: enhanced });
                        }
                        Err(e) => {
                            crate::ui::progress::println(&format!(
                                "  enhance failed ({e:#}); using the original prompt"
                            ));
                        }
                    }
                }

                // ── Conversational refinement: img2img over the previous image. ──
                // Reuses the loaded weights via portrait::from_core (no reload). The
                // img2img body isn't StepHook-wired, so progress flows to the Output
                // pane via the rerouted `ui::progress` (no inline preview / cancel).
                if let Some(init) = job.init_image.clone() {
                    let _ = std::fs::create_dir_all(&job.out_dir);
                    let refine_pipe = portrait::Pipeline::from_core(pipeline.core());
                    let req = img2img::Request {
                        prompt: prompt.clone(),
                        negative: job.negative.clone(),
                        model: alias.clone(),
                        device: device.clone(),
                        loras: Vec::new(),
                        lora_scale: 1.0,
                        input: init,
                        mask: None,
                        mask_feather: 0,
                        mask_invert: false,
                        width: job.width,
                        height: job.height,
                        count: 1,
                        steps: job.steps,
                        guidance: job.guidance,
                        scheduler: crate::pipelines::scheduler::SchedulerKind::default(),
                        strength: job.strength,
                        seed: Some(job.seed),
                        out_dir: job.out_dir.clone(),
                        controls: Vec::new(),
                    };
                    match rt.block_on(img2img::run_with_pipeline(&refine_pipe, &req)) {
                        Ok(()) => {
                            let produced = job.out_dir.join(format!("plakat-img2img-{}.png", job.seed));
                            let out = keep_unique(&produced, &job.out_dir, job.seed);
                            embed_chat_recipe(
                                &out, alias, &prompt, &job.negative, job.seed, job.steps,
                                job.guidance, Some("img2img"), Some(job.strength),
                            );
                            let _ = job.tx.send(GenMessage::Done {
                                output: out,
                                cancelled: job.cancel.is_cancelled(),
                            });
                        }
                        Err(e) => {
                            let _ = job.tx.send(GenMessage::Error { message: format!("{e:#}") });
                        }
                    }
                    continue;
                }

                let req = t2i::GenRequest {
                    prompt: prompt.clone(),
                    negative: job.negative.clone(),
                    width: job.width,
                    height: job.height,
                    count: 1,
                    steps: job.steps,
                    guidance: job.guidance,
                    seed: Some(job.seed),
                    out_dir: job.out_dir.clone(),
                    scheduler: crate::pipelines::scheduler::SchedulerKind::default(),
                    refine: None,
                    refine_strength: 0.3,
                    refiner_frac: None,
                    clip_skip: 1,
                    metadata: None,
                    preview_every: None,
                    preview_size: None,
                    output_format: crate::imaging::io::OutputFormat::Png,
                };
                let mut hook = ChannelHook::new(job.tx.clone(), job.cancel.clone(), job.preview_every);
                let result = pipeline.generate_hooked(&req, &[], Some(&mut hook));
                match result {
                    Ok(()) => {
                        let produced = job.out_dir.join(format!("plakat-{}.png", job.seed));
                        let out = keep_unique(&produced, &job.out_dir, job.seed);
                        embed_chat_recipe(
                            &out, alias, &prompt, &job.negative, job.seed, job.steps,
                            job.guidance, None, None,
                        );
                        let _ = job.tx.send(GenMessage::Done {
                            output: out,
                            cancelled: job.cancel.is_cancelled(),
                        });
                    }
                    Err(e) => {
                        let _ = job.tx.send(GenMessage::Error { message: format!("{e:#}") });
                    }
                }
            }
            ModelCommand::Shutdown => break,
        }
    }
}

/// Rename `produced` (the pipeline's `plakat-<seed>.png`) to the next free
/// `plakat-<seed>-<n>.png` so successive Chat turns at the SAME (stable) seed don't
/// overwrite each other — each step is kept as its own file. Falls back to the
/// produced path if the rename can't happen.
fn keep_unique(produced: &std::path::Path, out_dir: &std::path::Path, seed: u64) -> PathBuf {
    let mut n = 1u32;
    let target = loop {
        let p = out_dir.join(format!("plakat-{seed}-{n}.png"));
        if !p.exists() {
            break p;
        }
        n += 1;
    };
    match std::fs::rename(produced, &target) {
        Ok(()) => target,
        Err(_) => produced.to_path_buf(),
    }
}

/// Embed the generation recipe into the final PNG (A1111 `parameters` tEXt chunk +
/// JSON sidecar) so the History screen can show it and continue from it. Best-effort
/// — a failure never affects the generation. Re-encodes the saved PNG (lossless),
/// which keeps the txt2img and img2img paths uniform: img2img's pipeline writes no
/// metadata, and txt2img's would land at the pre-rename name + orphan its sidecar.
#[allow(clippy::too_many_arguments)]
fn embed_chat_recipe(
    path: &std::path::Path,
    model: &str,
    prompt: &str,
    negative: &str,
    seed: u64,
    steps: usize,
    guidance: f64,
    mode: Option<&str>,
    strength: Option<f32>,
) {
    let Ok(img) = image::open(path) else { return };
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let mut meta = crate::imaging::metadata::GenerationMetadata::new(
        prompt, model, seed, steps, guidance, "default", w, h,
    );
    meta.negative = negative.to_string();
    meta.mode = mode.map(|m| m.to_string());
    meta.strength = strength;
    let _ = crate::imaging::io::save_rgb_u8_with_metadata(rgb.as_raw(), w, h, path, &meta);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_unique_increments_per_seed_without_overwriting() {
        let d = std::env::temp_dir().join("plakat-keepunique-test");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let seed = 99u64;

        // First turn: pipeline produced plakat-99.png → renamed to plakat-99-1.png.
        let produced = d.join("plakat-99.png");
        std::fs::write(&produced, b"a").unwrap();
        let out1 = keep_unique(&produced, &d, seed);
        assert_eq!(out1, d.join("plakat-99-1.png"));
        assert!(out1.exists());
        assert!(!produced.exists(), "the produced file was moved, not copied");

        // Second turn at the SAME seed → plakat-99-2.png (no overwrite).
        std::fs::write(&produced, b"b").unwrap();
        let out2 = keep_unique(&produced, &d, seed);
        assert_eq!(out2, d.join("plakat-99-2.png"));
        assert!(out1.exists() && out2.exists(), "both steps are kept");

        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn embed_chat_recipe_writes_a_readable_parameters_chunk() {
        let d = std::env::temp_dir().join("plakat-embedmeta-test");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("plakat-7-1.png");
        // A 2x1 PNG with no metadata.
        crate::imaging::io::save_rgb_u8(&[1, 2, 3, 4, 5, 6], 2, 1, &p).unwrap();
        assert!(crate::imaging::io::read_parameters_chunk(&p).unwrap().is_none());

        embed_chat_recipe(&p, "sd15", "a red fox", "blurry", 7, 28, 7.5, Some("img2img"), Some(0.6));

        let params = crate::imaging::io::read_parameters_chunk(&p).unwrap().expect("recipe embedded");
        assert!(params.contains("a red fox"), "prompt round-trips: {params:?}");
        assert!(params.contains("blurry"), "negative round-trips: {params:?}");
        // The JSON sidecar is written too.
        assert!(p.with_extension("json").exists(), "sidecar written");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn sd_family_passes_the_load_check() {
        assert!(t2i_load_check("sdxl").is_ok());
        assert!(t2i_load_check("sd15").is_ok());
        assert!(t2i_load_check("sd21").is_ok());
    }

    #[test]
    fn other_families_get_a_friendly_error() {
        let e = t2i_load_check("flux-dev").unwrap_err();
        assert!(e.contains("Flux"));
        assert!(e.contains("follow-up"));
        assert!(t2i_load_check("sd35-medium").is_err());
        assert!(t2i_load_check("pixart-sigma").is_err());
    }
}
