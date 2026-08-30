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
use crate::pipelines::{cascade, img2img, multiperson, pixart, portrait, sd3, t2i};
use crate::pipelines::lora::LoraSpec;
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
    /// 6.25.0 P1 — subseed / variation-seed (`/subseed` in Chat). `None` = off.
    pub subseed: Option<u64>,
    pub subseed_strength: f32,
    pub out_dir: PathBuf,
    pub preview_every: usize,
    /// `Some(path)` → conversational refinement: img2img over this image at
    /// `strength`, reusing the loaded weights. `None` → fresh txt2img.
    pub init_image: Option<PathBuf>,
    pub strength: f32,
    /// `Some(path)` → inpaint: only the mask's white pixels change (Canvas mask).
    pub mask: Option<PathBuf>,
    /// `Some(provider)` → AI-enhance the prompt (`/enhance`) before generating.
    pub enhance: Option<String>,
    /// 6.25.0 P3 in the ui (`/region`): when non-empty, run regional prompting
    /// (`generate_regional`) instead of the plain txt2img denoise. Each region's prompt
    /// applies in its box, blended over the base prompt. No per-step preview in this path.
    pub regions: Vec<crate::pipelines::tiled::RegionSpec>,
    pub tx: Sender<GenMessage>,
    pub cancel: CancelFlag,
}

/// A request to the model thread.
pub enum ModelCommand {
    Load { alias: String, loras: Vec<crate::pipelines::lora::LoraSpec> },
    Unload,
    Generate(GenJob),
    /// In-process scenario run. The model thread **drops the loaded Chat pipeline first**
    /// (frees memory deterministically — same thread), then runs the scenario, so only
    /// one model is ever resident (no double-load OOM on unified memory).
    RunScenario {
        args: crate::cli::scenario::ScenarioArgs,
        events: Sender<crate::cli::scenario::ScenarioEvent>,
        done: Sender<Result<(), String>>,
    },
    /// In-process portrait (People `G`) — same free-then-run discipline.
    RunPortrait {
        req: portrait::Request,
        produced: PathBuf,
        done: Sender<Result<PathBuf, String>>,
    },
    /// In-process multiperson scene (People `G` with ≥2 marked).
    RunMultiperson {
        req: multiperson::MultipersonRequest,
        produced: PathBuf,
        done: Sender<Result<PathBuf, String>>,
    },
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

/// Which UI loader handles a model. SD-family, SD3/3.5, PixArt-Σ, and Stable Cascade
/// are wired; Flux is still CLI-only (a follow-up).
enum UiFamily {
    Sd,
    Sd3,
    PixArt,
    Cascade,
}

/// Resolve a model alias to its UI loader, or a friendly error for unwired families.
/// Pure — unit-tested.
fn ui_family(alias: &str) -> Result<UiFamily, String> {
    match BaseFamily::from_model_arg(alias) {
        BaseFamily::Sd15 | BaseFamily::Sd21 | BaseFamily::Sdxl => Ok(UiFamily::Sd),
        BaseFamily::Sd3 => Ok(UiFamily::Sd3),
        BaseFamily::PixArt => Ok(UiFamily::PixArt),
        BaseFamily::StableCascade => Ok(UiFamily::Cascade),
        other => Err(format!(
            "'{alias}' is a {other:?} model — the TUI loads SD-family (sd15 / sd21 / \
             sdxl), SD3/3.5, PixArt, and Cascade; {other:?} support is a follow-up."
        )),
    }
}

/// Whether the UI can load this model (SD-family or SD3). Used by the App's startup
/// auto-load gate. Pure — unit-tested.
pub fn t2i_load_check(alias: &str) -> Result<(), String> {
    ui_family(alias).map(|_| ())
}

/// A loaded model — the UI holds SD-family and SD3 pipelines (each persistent, so
/// refines are fast). Lives on the model thread and never crosses threads.
enum Loaded {
    Sd(t2i::Pipeline),
    Sd3(sd3::Pipeline),
    // PixArt / Cascade have no persistent pipeline (their `run()` loads per call), so
    // we hold only the applied LoRA set and load-per-generation. Slower, but usable.
    PixArt { loras: Vec<LoraSpec> },
    Cascade { loras: Vec<LoraSpec> },
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
        subseed: Option<u64>,
        subseed_strength: f32,
        out_dir: PathBuf,
        preview_every: usize,
        init_image: Option<PathBuf>,
        strength: f32,
        mask: Option<PathBuf>,
        enhance: Option<String>,
        regions: Vec<crate::pipelines::tiled::RegionSpec>,
    ) -> (std::sync::mpsc::Receiver<GenMessage>, CancelFlag) {
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = CancelFlag::new();
        let job = GenJob {
            prompt, negative, width, height, steps, guidance, seed, subseed, subseed_strength,
            out_dir, preview_every,
            init_image, strength, mask, enhance, regions,
            tx, cancel: cancel.clone(),
        };
        let _ = self.cmd_tx.send(ModelCommand::Generate(job));
        (rx, cancel)
    }

    /// Run a scenario on the model thread (frees the loaded Chat model first). `events`
    /// receives the live per-task [`ScenarioEvent`]s; the returned receiver yields the
    /// terminal result.
    pub fn run_scenario(
        &self,
        args: crate::cli::scenario::ScenarioArgs,
        events: Sender<crate::cli::scenario::ScenarioEvent>,
    ) -> Receiver<Result<(), String>> {
        let (done, rx) = channel();
        let _ = self.cmd_tx.send(ModelCommand::RunScenario { args, events, done });
        rx
    }

    /// Run a portrait on the model thread (frees the loaded Chat model first). `produced`
    /// is the output path returned on success.
    pub fn run_portrait(&self, req: portrait::Request, produced: PathBuf) -> Receiver<Result<PathBuf, String>> {
        let (done, rx) = channel();
        let _ = self.cmd_tx.send(ModelCommand::RunPortrait { req, produced, done });
        rx
    }

    /// Run a multiperson scene on the model thread (frees the loaded Chat model first).
    pub fn run_multiperson(&self, req: multiperson::MultipersonRequest, produced: PathBuf) -> Receiver<Result<PathBuf, String>> {
        let (done, rx) = channel();
        let _ = self.cmd_tx.send(ModelCommand::RunMultiperson { req, produced, done });
        rx
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
    // OOM watchdog for the whole UI session: every TUI generation (Chat, portrait,
    // multiperson, and any in-process scenario) runs on THIS thread + device, so one
    // guard covers them all. On sustained critical unified-memory pressure it aborts
    // plakat cleanly (the OS then reclaims all memory, incl. Metal buffers) rather than
    // letting the host crash. No-op on CUDA / `PLAKAT_OOM_GUARD_GB=0`.
    let _mem_guard = crate::memwatch::MemoryGuard::start(&device, "plakat ui");
    // The loaded pipeline lives here and never leaves this thread.
    let mut loaded: Option<(String, Loaded)> = None;
    // Whether the resident pipeline was loaded with NO LoRAs — only a vanilla base can be
    // safely handed to a scenario run for reuse (a LoRA'd pipeline would generate wrong).
    let mut loaded_vanilla = false;
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            ModelCommand::Load { alias, loras } => {
                let is_vanilla = loras.is_empty();
                let family = match ui_family(&alias) {
                    Ok(f) => f,
                    Err(msg) => {
                        let _ = msg_tx.send(ModelMessage::Error(msg));
                        continue;
                    }
                };
                let _ = msg_tx.send(ModelMessage::LoadStarted(alias.clone()));
                // Sweep any stale hf-hub `.lock` left by an interrupted download (Ctrl-C
                // or an OOM-guard hard-exit) — otherwise a cached model can hang/refuse to
                // load. Safe here: this thread loads serially, no download is in flight.
                let swept = crate::hf::download::clean_stale_locks(&alias);
                if swept > 0 {
                    crate::ui::progress::println(&format!("  cleared {swept} stale download lock(s) for {alias}"));
                }
                // Free the current model first — unified memory means we can't hold
                // two large models at once.
                loaded = None;
                // PixArt / Cascade have no persistent pipeline — their `run()` loads
                // per generation, so "loading" just records the alias + applied LoRAs
                // (the first gen shows the real download/load in the Output pane).
                let result = match family {
                    UiFamily::Sd => rt
                        .block_on(t2i::Pipeline::load(t2i::LoadRequest {
                            model: alias.clone(),
                            device: device.clone(),
                            loras,
                            lora_scale: 1.0,
                            use_refiner: false,
                            embeddings: Vec::new(),
                            vae_cache: None,
                        }))
                        .map(Loaded::Sd),
                    UiFamily::Sd3 => rt
                        .block_on(sd3::Pipeline::load(sd3::LoadRequest {
                            variant: sd3_variant(&alias),
                            repo: resolve_repo(&alias),
                            device: device.clone(),
                            loras,
                            lora_scale: 1.0,
                            controlnets: Vec::new(),
                            embeddings: Vec::new(),
                        }))
                        .map(Loaded::Sd3),
                    UiFamily::PixArt => Ok(Loaded::PixArt { loras }),
                    UiFamily::Cascade => Ok(Loaded::Cascade { loras }),
                };
                match result {
                    Ok(p) => {
                        let used = (crate::hw::total_ram_gb() - crate::hw::available_ram_gb()).max(0.0);
                        loaded = Some((alias.clone(), p));
                        loaded_vanilla = is_vanilla;
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
                if loaded.is_none() {
                    let _ = job.tx.send(GenMessage::Error {
                        message: "no model loaded — load one in Models (Ctrl-2)".into(),
                    });
                    continue;
                }

                // ── /enhance: expand the prompt with the configured LLM first. ──
                // On failure, fall back to the original prompt (the run still goes).
                // Family-agnostic (no pipeline access), so it runs before dispatch.
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

                let (alias, model) = loaded.as_mut().expect("loaded checked above");
                let pipeline = match model {
                    Loaded::Sd(p) => p,
                    // ── SD3 / 3.5: one generate_hooked handles txt2img / img2img /
                    //    inpaint (it branches on init_image + mask). ──
                    Loaded::Sd3(sd3_pipe) => {
                        let _ = std::fs::create_dir_all(&job.out_dir);
                        let req = sd3::GenRequest {
                            prompt: prompt.clone(),
                            negative: job.negative.clone(),
                            width: job.width,
                            height: job.height,
                            count: 1,
                            steps: Some(job.steps),
                            guidance: Some(job.guidance),
                            seed: Some(job.seed),
                            out_dir: job.out_dir.clone(),
                            init_image: job.init_image.clone(),
                            mask: job.mask.clone(),
                            mask_feather: 0,
                            mask_invert: false,
                            strength: job.init_image.as_ref().map(|_| job.strength),
                            tiled: None,
                            regions: Vec::new(),
                            controlnet_conditioning: Vec::new(),
                            output_format: crate::imaging::io::OutputFormat::Png,
                        };
                        let mut hook = ChannelHook::new(job.tx.clone(), job.cancel.clone(), job.preview_every);
                        match sd3_pipe.generate_hooked(&req, Some(&mut hook)) {
                            Ok(()) => {
                                // sd3 names plakat-sd3-{mode}-{seed}.png.
                                let mode = match (job.init_image.is_some(), job.mask.is_some()) {
                                    (true, true) => "inpaint",
                                    (true, false) => "img2img",
                                    _ => "denoise",
                                };
                                let produced = job.out_dir.join(format!("plakat-sd3-{mode}-{}.png", job.seed));
                                let out = keep_unique(&produced, &job.out_dir, job.seed);
                                embed_chat_recipe(
                                    &out, alias, &prompt, &job.negative, job.seed, job.steps, job.guidance,
                                    job.init_image.as_ref().map(|_| mode),
                                    job.init_image.as_ref().map(|_| job.strength),
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
                    // ── PixArt-Σ: load-per-gen txt2img (hooked → live preview/cancel).
                    //    No img2img — `init_image` is ignored (prompt-evolve carries the
                    //    edits). ──
                    Loaded::PixArt { loras } => {
                        let _ = std::fs::create_dir_all(&job.out_dir);
                        let req = pixart::RunRequest {
                            model: alias.clone(),
                            device: device.clone(),
                            prompt: prompt.clone(),
                            negative: job.negative.clone(),
                            width: job.width,
                            height: job.height,
                            steps: job.steps,
                            guidance: job.guidance,
                            seed: Some(job.seed),
                            scheduler: crate::pipelines::scheduler::SchedulerKind::default(),
                            out_dir: job.out_dir.clone(),
                            count: 1,
                            loras: loras.clone(),
                            lora_scale: 1.0,
                        };
                        let mut hook = ChannelHook::new(job.tx.clone(), job.cancel.clone(), job.preview_every);
                        match rt.block_on(pixart::run_hooked(req, Some(&mut hook))) {
                            Ok(()) => {
                                let produced = job.out_dir.join(format!("plakat-pixart-{}.png", job.seed));
                                let out = keep_unique(&produced, &job.out_dir, job.seed);
                                embed_chat_recipe(&out, alias, &prompt, &job.negative, job.seed, job.steps, job.guidance, None, None);
                                let _ = job.tx.send(GenMessage::Done { output: out, cancelled: job.cancel.is_cancelled() });
                            }
                            Err(e) => {
                                let _ = job.tx.send(GenMessage::Error { message: format!("{e:#}") });
                            }
                        }
                        continue;
                    }
                    // ── Stable Cascade: load-per-gen txt2img (hooked). Square output;
                    //    Stage C/B step split mirrors the CLI default. ──
                    Loaded::Cascade { loras } => {
                        let _ = std::fs::create_dir_all(&job.out_dir);
                        let stage_c_steps = (job.steps * 2).div_ceil(3).max(1);
                        let stage_b_steps = job.steps.saturating_sub(stage_c_steps).max(1);
                        let req = cascade::RunRequest {
                            model: alias.clone(),
                            device: device.clone(),
                            prompt: prompt.clone(),
                            negative: job.negative.clone(),
                            output_dim: job.width,
                            image_prompt: None,
                            stage_c_steps,
                            stage_b_steps,
                            guidance: job.guidance,
                            decoder_guidance: 1.1,
                            seed: Some(job.seed),
                            scheduler: crate::pipelines::scheduler::SchedulerKind::default(),
                            out_dir: job.out_dir.clone(),
                            count: 1,
                            loras: loras.clone(),
                            lora_scale: 1.0,
                            control_spec: None,
                            controlnet_weights: None,
                        };
                        let mut hook = ChannelHook::new(job.tx.clone(), job.cancel.clone(), job.preview_every);
                        match rt.block_on(cascade::run_hooked(req, Some(&mut hook))) {
                            Ok(()) => {
                                let produced = job.out_dir.join(format!("plakat-cascade-{}.png", job.seed));
                                let out = keep_unique(&produced, &job.out_dir, job.seed);
                                embed_chat_recipe(&out, alias, &prompt, &job.negative, job.seed, job.steps, job.guidance, None, None);
                                let _ = job.tx.send(GenMessage::Done { output: out, cancelled: job.cancel.is_cancelled() });
                            }
                            Err(e) => {
                                let _ = job.tx.send(GenMessage::Error { message: format!("{e:#}") });
                            }
                        }
                        continue;
                    }
                };

                // ── SD-family conversational refinement: img2img over the previous
                //    image. Reuses the loaded weights via portrait::from_core (no
                //    reload), StepHook-wired for live preview + cancel (A1). ──
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
                        mask: job.mask.clone(),
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
                    // Live preview + cancel during the refine (RFC §0-R0-3).
                    let mut hook = ChannelHook::new(job.tx.clone(), job.cancel.clone(), job.preview_every);
                    let result = rt.block_on(img2img::run_with_pipeline_hooked(
                        &refine_pipe,
                        &req,
                        Some(&mut hook),
                    ));
                    match result {
                        Ok(()) => {
                            // The pipeline names by mode: a mask → "inpaint", else
                            // "img2img". Match it or we'd rename/open a missing file
                            // and the UI would show the stale previous image.
                            let mode = if job.mask.is_some() { "inpaint" } else { "img2img" };
                            let produced = job.out_dir.join(format!("plakat-{mode}-{}.png", job.seed));
                            let out = keep_unique(&produced, &job.out_dir, job.seed);
                            embed_chat_recipe(
                                &out, alias, &prompt, &job.negative, job.seed, job.steps,
                                job.guidance, Some(mode), Some(job.strength),
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
                    subseed: job.subseed,
                    subseed_strength: job.subseed_strength,
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
                // 6.25.0 P3: regional prompting takes a distinct denoise path (base + per-region
                // MultiDiffusion) with no per-step hook — the Chat status stays "Generating"
                // until Done. Plain txt2img keeps the live preview hook.
                let result = if job.regions.is_empty() {
                    let mut hook = ChannelHook::new(job.tx.clone(), job.cancel.clone(), job.preview_every);
                    pipeline.generate_hooked(&req, &[], Some(&mut hook))
                } else {
                    pipeline.generate_regional(&req, &job.regions)
                };
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
            ModelCommand::RunScenario { args, events, done } => {
                // Hand a *vanilla* resident SD pipeline to the runner so a matching
                // all-SD scenario reuses it instead of reloading the same weights (saves
                // a full load). The runner drops it if the scenario's model/LoRAs/refiner
                // don't match — so this can never change output. Any other resident model
                // (LoRA'd, or a non-SD family) is just freed, as before.
                let preloaded = match loaded.take() {
                    Some((alias, Loaded::Sd(pipe))) if loaded_vanilla => {
                        let _ = msg_tx.send(ModelMessage::Unloaded);
                        Some((alias, pipe))
                    }
                    other => {
                        if other.is_some() {
                            let _ = msg_tx.send(ModelMessage::Unloaded);
                        }
                        None
                    }
                };
                loaded_vanilla = false;
                let result = rt
                    .block_on(crate::cli::scenario::run_with_events(args, Some(events), preloaded))
                    .map_err(|e| format!("{e:#}"));
                let _ = done.send(result);
            }
            ModelCommand::RunPortrait { req, produced, done } => {
                free_loaded(&mut loaded, &msg_tx);
                let result = rt.block_on(portrait::run(req)).map(|_| produced).map_err(|e| format!("{e:#}"));
                let _ = done.send(result);
            }
            ModelCommand::RunMultiperson { req, produced, done } => {
                free_loaded(&mut loaded, &msg_tx);
                let result = rt.block_on(multiperson::run(req)).map(|_| produced).map_err(|e| format!("{e:#}"));
                let _ = done.send(result);
            }
            ModelCommand::Shutdown => break,
        }
    }
}

/// Drop the resident Chat pipeline (if any) before an in-process heavy run, so only one
/// model is ever in memory; tell the UI it's now unloaded.
fn free_loaded(loaded: &mut Option<(String, Loaded)>, msg_tx: &Sender<ModelMessage>) {
    if loaded.take().is_some() {
        let _ = msg_tx.send(ModelMessage::Unloaded);
    }
}

/// Resolve a model alias to its HF repo (an explicit `org/name` passes through).
fn resolve_repo(alias: &str) -> String {
    if alias.contains('/') {
        alias.to_string()
    } else {
        crate::hf::resolve_alias(alias).to_string()
    }
}

/// Map an SD3 alias to its `sd3::Variant` (caller already gated to an SD3 family).
fn sd3_variant(alias: &str) -> sd3::Variant {
    match t2i::Variant::detect(alias) {
        t2i::Variant::Sd3Medium => sd3::Variant::Sd3Medium,
        t2i::Variant::Sd35Medium => sd3::Variant::Sd35Medium,
        t2i::Variant::Sd35Large => sd3::Variant::Sd35Large,
        t2i::Variant::Sd35LargeTurbo => sd3::Variant::Sd35LargeTurbo,
        _ => sd3::Variant::Sd35Medium,
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
    fn run_scenario_executes_on_the_model_thread() {
        // The in-process runner routes a scenario through the model thread (which frees
        // any loaded Chat model first). A dry-run exercises the command path end-to-end
        // without loading a model. Uses a real multi-thread runtime handle.
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
        let svc = ModelService::spawn(Device::Cpu, rt.handle().clone());

        let d = std::env::temp_dir().join("plakat-modelsvc-scenario-test");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let file = d.join("s.hjson");
        std::fs::write(
            &file,
            r#"{"model":"stable-diffusion-v1-5/stable-diffusion-v1-5","size":"512x512","enhancer":"local","scene":[{"name":"","prompt":"p"}],"weather":[{"name":"","prompt":"c"}],"tasks":[{"name":"alpha","prompt":"a"}]}"#,
        )
        .unwrap();
        let args = crate::cli::scenario::ScenarioArgs {
            file,
            dry_run: true,
            resume: false,
            force: false,
            only: Vec::new(),
            limit: 0,
            json_summary: None,
            out_override: None,
        };
        let (etx, erx) = channel();
        let done = svc.run_scenario(args, etx);

        // Wait (bounded) for the terminal result; the model thread runs run_with_events.
        let mut result = None;
        for _ in 0..600 {
            if let Ok(r) = done.try_recv() {
                result = Some(r);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(matches!(result, Some(Ok(()))), "dry-run scenario ran on the model thread: {result:?}");
        // The per-task events flowed back on the events channel.
        let evs: Vec<_> = std::iter::from_fn(|| erx.try_recv().ok()).collect();
        assert!(!evs.is_empty(), "scenario events were forwarded");
        let _ = std::fs::remove_dir_all(&d);
    }

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
        // The JSON sidecar is written too (named `<image>.json`).
        assert!(crate::imaging::io::sidecar_path(&p).exists(), "sidecar written");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn sd_family_passes_the_load_check() {
        assert!(t2i_load_check("sdxl").is_ok());
        assert!(t2i_load_check("sd15").is_ok());
        assert!(t2i_load_check("sd21").is_ok());
    }

    #[test]
    fn non_sd_families_map_to_their_loaders() {
        assert!(matches!(ui_family("sdxl"), Ok(UiFamily::Sd)));
        assert!(matches!(ui_family("sd35-medium"), Ok(UiFamily::Sd3)));
        assert!(matches!(ui_family("pixart"), Ok(UiFamily::PixArt)));
        assert!(matches!(ui_family("stable-cascade"), Ok(UiFamily::Cascade)));
        // All four are loadable in the UI now.
        for a in ["sdxl", "sd35-medium", "pixart", "stable-cascade"] {
            assert!(t2i_load_check(a).is_ok(), "{a} should load");
        }
    }

    #[test]
    fn flux_still_gets_a_friendly_cli_only_error() {
        let e = t2i_load_check("flux-dev").unwrap_err();
        assert!(e.contains("Flux"));
        assert!(e.contains("follow-up"));
    }
}
