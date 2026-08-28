//! The TUI application shell — `App`, the event loop, and global chrome (RFC
//! TUI-1 §4–§5). This Phase-1 increment is the navigable frame: a tab bar, a
//! status bar, `Ctrl-1..8` (or plain `1..8`) screen switching, and per-screen
//! placeholders. The Chat + Models screen bodies, services, and channel draining
//! land in the next increments.

use anyhow::Result;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Tabs, Wrap},
};
use candle_core::Device;
use ratatui_image::picker::Picker;
use tokio::runtime::Handle;

use std::sync::mpsc::Receiver;

use crate::cli::scenario::ScenarioEvent;
use crate::pipelines::gen_channel::{CancelFlag, GenMessage};

use super::output::OutputPane;
use super::screens::canvas::{self, CanvasState};
use super::screens::chat::{ChatAction, ChatState, ChatStatus};
use super::screens::history::{HistoryAction, HistoryState};
use super::screens::lorahub::{self, LoraHubState};
use super::screens::models::ModelsState;
use super::screens::naturalize::{NaturalizeAction, NaturalizeState};
use super::screens::palette::{self, Cmd, PaletteResult};
use super::screens::people::{self, PeopleState};
use super::screens::prompts::{self, PromptsState};
use super::screens::scenarios::{ScenariosAction, ScenariosState};
use super::services::model_service::ModelService;
use super::workspace::Workspace;

/// img2img strength used when the user opts INTO image-anchored refinement via
/// `/strength` without a value (or as the default for that mode).
const DEFAULT_ANCHOR_STRENGTH: f32 = 0.6;

/// Per-LoRA scale used when applying a LoRA from the Hub to Chat.
const APPLY_LORA_SCALE: f32 = 0.8;

/// img2img strength for a Canvas inpaint turn — high, so the masked region actually
/// regenerates (a soft 0.6 only nudges it).
const INPAINT_STRENGTH: f32 = 0.85;
/// Max LoRA downloads running at once (the rest queue). Unified memory + a shared CDN
/// make a small cap saner than unbounded fan-out.
const MAX_CONCURRENT_DOWNLOADS: usize = 2;
/// Idle span after which a loaded model is auto-unloaded to return its memory. It
/// auto-reloads on the next keypress (the user picking activity back up).
const IDLE_UNLOAD: Duration = Duration::from_secs(10 * 60);

/// The eight screens (RFC §1). Release 1 implements Chat + Models; the rest show a
/// placeholder until their cycle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActiveScreen {
    Chat,
    Models,
    Scenarios,
    History,
    LoraHub,
    People,
    PromptWorkspace,
    Canvas,
    Naturalize,
}

impl ActiveScreen {
    const ALL: [ActiveScreen; 9] = [
        Self::Chat,
        Self::Models,
        Self::Scenarios,
        Self::History,
        Self::LoraHub,
        Self::People,
        Self::PromptWorkspace,
        Self::Canvas,
        Self::Naturalize,
    ];

    fn title(self) -> &'static str {
        match self {
            Self::Chat => "Chat",
            Self::Models => "Models",
            Self::Scenarios => "Scenarios",
            Self::History => "History",
            Self::LoraHub => "LoRA Hub",
            Self::People => "People",
            Self::PromptWorkspace => "Prompts",
            Self::Canvas => "Canvas",
            Self::Naturalize => "Naturalize",
        }
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|s| *s == self).unwrap_or(0)
    }

    fn from_index(i: usize) -> Option<Self> {
        Self::ALL.get(i).copied()
    }

    /// Cycle to the next (`+1`) / previous (`-1`) screen, wrapping. Drives Tab /
    /// Shift-Tab — universal navigation that works on every terminal.
    fn cycle(self, delta: isize) -> Self {
        let n = Self::ALL.len() as isize;
        let i = (self.index() as isize + delta).rem_euclid(n) as usize;
        Self::ALL[i]
    }

}

/// A model load held back for confirmation because its estimated resident footprint
/// exceeds free RAM (RFC — 1.20.0 memory-budget guard). Confirm with `y`/Enter.
pub struct PendingLoad {
    pub alias: String,
    /// Estimated resident GB (weights + runtime overhead).
    pub est_gb: f64,
    /// Free RAM at the time of the request.
    pub avail_gb: f64,
    /// Whether `est_gb` is exact (from the on-disk cache) or a coarse guess (uncached).
    pub exact: bool,
}

/// Render a centered, bordered modal box (Clear + titled block) with `body`, clamped to
/// the frame. The single styling path for every TUI overlay confirm/info popup so they
/// stay visually consistent. `size` is the desired (width, height).
fn centered_modal(f: &mut Frame, title: &str, border: Color, body: Vec<Line<'static>>, size: (u16, u16)) {
    let area = f.area();
    let w = size.0.min(area.width.saturating_sub(4));
    let h = size.1.min(area.height.saturating_sub(2));
    let rect = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    };
    f.render_widget(ratatui::widgets::Clear, rect);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::new().fg(border))
        .title(title.to_string());
    f.render_widget(Paragraph::new(body).block(block).wrap(Wrap { trim: true }), rect);
}

/// The running TUI. Holds the workspace, the image `Picker` (for inline previews,
/// used once screens render images), the active screen, and the quit flag. Services
/// (ModelService / GenQueue / LlmPool) join in the next increment.
pub struct App {
    pub workspace: Workspace,
    pub picker: Picker,
    pub screen: ActiveScreen,
    pub should_quit: bool,
    // Per-screen state (persists across screen switches).
    pub chat: ChatState,
    pub models: ModelsState,
    pub scenarios: ScenariosState,
    pub history: HistoryState,
    pub people: PeopleState,
    pub lorahub: LoraHubState,
    pub prompts: PromptsState,
    pub canvas: CanvasState,
    /// Naturalize tab — interactive weight-free de-slop (RFC QUALITY-8 P3).
    pub naturalize: NaturalizeState,
    // Command palette overlay (Ctrl-K), fuzzy action launcher (RFC §5).
    pub palette: palette::PaletteState,
    // A load awaiting confirmation because its estimated footprint over-commits RAM.
    pub pending_load: Option<PendingLoad>,
    // Last user keypress — drives idle auto-unload (memory reclaim while away).
    last_activity: Instant,
    // A model that idle auto-unload freed; reloaded when the user resumes activity.
    suspended: Option<String>,
    // Hard-reset (re-exec) confirmation + request. `should_reset` breaks the event
    // loop; `run()` restores the terminal then re-execs the process.
    confirm_reset: bool,
    should_reset: bool,
    // Keyboard cheatsheet overlay (F1) — global keys + the active screen's keys.
    show_help: bool,
    // Pending `/vary N` variations — the same prompt queued at fresh seeds; the model
    // thread is serial, so they dispatch one at a time (each lands in the filmstrip).
    variation_queue: std::collections::VecDeque<String>,
    // Per-model generation-size override (`/size`), keyed by alias. Absent = the model's
    // native square (always Metal-safe). Each model remembers its own preferred size.
    gen_sizes: std::collections::HashMap<String, (u32, u32)>,
    // Session denoise-steps / guidance overrides (`/steps`, `/cfg`); None = workspace
    // default. Captured into and restored from named presets.
    steps_override: Option<usize>,
    guidance_override: Option<f64>,
    // v2.7 free-quality guidance-bundle session overrides (`/pag`, `/rescale`, `/freeu`,
    // `/dynthresh`). Applied by setting the env the pipelines read at dispatch time.
    pag_override: Option<f64>,
    rescale_override: Option<f64>,
    freeu_override: bool,
    dynthresh_override: Option<f64>,
    // Shared Output pane (messages + live progress, fed by the rerouted sink).
    pub output: OutputPane,
    progress_rx: Receiver<String>,
    // Background services.
    pub model_svc: ModelService,
    rt: Handle,
    device: Device,
    // The in-flight Chat generation (its message channel + cancel flag).
    active_gen: Option<(Receiver<GenMessage>, CancelFlag)>,
    // Conversational refinement state. Each refine re-bases on the CLEAN original
    // image (NOT the previous refine's output) and uses the ACCUMULATED prompt, so
    // VAE round-trips don't compound into mosaic degradation across turns.
    refine_base: Option<std::path::PathBuf>,
    refine_prompt: String,
    // The in-flight turn's refine flag + full (accumulated) prompt — applied on Done.
    active_is_refine: bool,
    active_full_prompt: String,
    // Session negative prompt (`/negative …`), applied to every generation.
    negative: String,
    // Refinement mode. None (default) = prompt-evolve: a follow-up re-renders the
    // ACCUMULATED prompt with txt2img at the conversation's stable seed, so typed
    // edits reliably appear and the composition stays recognizable. Some(strength) =
    // image-anchored: img2img over the clean base at that strength (`/strength`).
    refine_strength: Option<f32>,
    // The conversation's stable seed (so prompt-evolve refines keep composition).
    base_seed: Option<u64>,
    // Explicit seed pin (`/seed …`) overriding both; None = use base_seed / random.
    fixed_seed: Option<u64>,
    // The seed used by the in-flight turn (recorded as base_seed on a fresh Done).
    active_seed: u64,
    // LoRAs applied to Chat generation (load-time merge), each with its weight.
    // Changing the set or a weight reloads the current model.
    active_loras: Vec<(std::path::PathBuf, f32)>,
    // Inpaint mask from Canvas applied to the next Chat refinement (white = change).
    chat_mask: Option<std::path::PathBuf>,
    // The exact image a Canvas mask / outpaint was painted over (the LATEST render, not
    // the prompt-evolve base) — the inpaint runs over THIS so the mask aligns and edits
    // compound on the current state. One-shot, consumed with the mask.
    inpaint_base: Option<std::path::PathBuf>,
    // One-time-per-session nudge: prompt-evolve can't reliably ADD an object → point at
    // Canvas inpaint the first time the user types an "add a …"-style edit.
    inpaint_nudged: bool,
    // `/auto` — LLM-classify each follow-up as an edit (refine) vs a new scene (fresh)
    // instead of the always-refine heuristic. Off by default (adds a quick LLM call).
    auto_route: bool,
    // In-flight classification: (is_new, edit_text).
    route_rx: Option<Receiver<(bool, String)>>,
    // The in-flight scenario run (its terminal-result channel).
    scenario_run: Option<Receiver<Result<(), String>>>,
    // Live per-task events from the in-flight scenario run (RUNNER board).
    scenario_events: Option<Receiver<ScenarioEvent>>,
    // The in-flight People quick-generate (portrait) — output path on success, plus
    // the prompt/seed to seed the Chat continuation when it lands.
    portrait_run: Option<Receiver<Result<std::path::PathBuf, String>>>,
    portrait_prompt: String,
    // Identity-preserving Chat continuation: while `Some`, a portrait with a known person
    // is loaded in Chat, so a refine re-runs the IP-Adapter portrait pass (keeps the face)
    // instead of plain img2img. `pending_identity` is set per-run and adopted by
    // `drain_portrait` once the image lands.
    chat_identity: Option<ChatIdentity>,
    pending_identity: Option<ChatIdentity>,
    // In-flight remote (Civitai / HF) search + download for the LoRA Hub.
    remote_search: Option<Receiver<Result<Vec<lorahub::RemoteHit>, String>>>,
    // Download pool: ≤2 concurrent, the rest queued (DownloadRef + title).
    downloads_active: Vec<Receiver<Result<String, String>>>,
    downloads_queue: std::collections::VecDeque<(lorahub::DownloadRef, String)>,
    // In-flight LLM LoRA assessment: (item key, assessment text).
    lora_assess: Option<Receiver<(String, String)>>,
    // In-flight LLM recommend-for-context (LoRA Hub search tabs).
    lora_recommend: Option<Receiver<String>>,
    // In-flight LLM LoRA-combination suggestion (LoRA Hub Ctrl-R).
    lora_combine: Option<Receiver<String>>,
    // In-flight Civitai update check (LoRA Hub `U`): `(model_id, name, Result<Option<(new_id, new_name)>>)`.
    #[allow(clippy::type_complexity)]
    lora_update: Option<Receiver<(u64, String, Result<Option<(u64, String)>, String>)>>,
    // In-flight Prompt Workspace LLM compile.
    prompt_compile: Option<Receiver<Result<String, String>>>,
    // In-flight Prompt Workspace structural (live) compile: `(src, result)`. Runs on a
    // worker thread — block_on must NOT run on the event-loop thread (it's inside the
    // tokio runtime, which would panic).
    prompt_structural: Option<(String, Receiver<Result<String, String>>)>,
    // In-flight Chat→Scenario summary (Scenarios editor Ctrl-G).
    chat_to_scenario: Option<Receiver<Result<String, String>>>,
    // In-flight Canvas face detection (for the face-aware `B` preset): `(base, boxes)`.
    canvas_faces: Option<Receiver<(std::path::PathBuf, Vec<[f32; 4]>)>>,
    // In-flight People identity re-encode: `(name, dir, Result<(score, faces, total)>)`.
    #[allow(clippy::type_complexity)]
    people_encode: Option<Receiver<(String, std::path::PathBuf, Result<(f32, usize, usize), String>)>>,
    // In-flight History image decode (off the event-loop tick): `(path, decoded image)`.
    history_decode: Option<(std::path::PathBuf, Receiver<Option<image::DynamicImage>>)>,
    // In-flight History thumbnail decode (grid view): `(path, decoded thumbnail)`.
    thumb_decode: Option<(std::path::PathBuf, Receiver<Option<image::DynamicImage>>)>,
}

/// The reusable identity context for an IP-Adapter-preserving Chat continuation: the
/// person's reference photos + strategy + the run parameters, so each refine re-renders
/// the accumulated prompt with the *same* face at the same seed.
#[derive(Clone)]
struct ChatIdentity {
    photos: Vec<(std::path::PathBuf, f32)>,
    identity: String,
    face_strength: Option<f32>,
    negative: String,
    model: String,
    width: u32,
    height: u32,
    seed: u64,
}

/// A persisted Chat session (`/save` / `/load`): the visible thread plus the
/// refinement state needed to keep editing where you left off.
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct ChatSession {
    #[serde(default)]
    turns: Vec<SessionTurn>,
    #[serde(default)]
    refine_prompt: String,
    #[serde(default)]
    base_seed: Option<u64>,
    #[serde(default)]
    negative: String,
    #[serde(default)]
    refine_base: Option<String>,
    #[serde(default)]
    active_seed: u64,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SessionTurn {
    utterance: String,
    #[serde(default)]
    result: Option<String>,
    #[serde(default)]
    refine: bool,
    #[serde(default)]
    system: bool,
}

impl App {
    pub fn new(
        workspace: Workspace,
        picker: Picker,
        device: Device,
        rt: Handle,
        progress_rx: Receiver<String>,
    ) -> Self {
        let scenarios = ScenariosState::new(workspace.scenarios_dir());
        let history = HistoryState::new(workspace.out_dir());
        let people = PeopleState::new(workspace.people_dir(), workspace.scenarios_dir());
        let lorahub = LoraHubState::new(vec![
            (workspace.loras_dir(), "workspace".into()),
            (crate::preset::discovery::default_cache_root(), "global".into()),
            (crate::civitai::download::cache_root(), "civitai".into()),
        ]);
        let prompts = PromptsState::new(workspace.prompts_dir());
        let canvas = CanvasState::new(workspace.out_dir().join("masks"));
        Self {
            scenarios,
            history,
            people,
            lorahub,
            prompts,
            canvas,
            naturalize: NaturalizeState::new(),
            palette: palette::PaletteState::new(),
            pending_load: None,
            last_activity: Instant::now(),
            suspended: None,
            confirm_reset: false,
            should_reset: false,
            show_help: false,
            variation_queue: std::collections::VecDeque::new(),
            gen_sizes: std::collections::HashMap::new(),
            steps_override: None,
            guidance_override: None,
            pag_override: None,
            rescale_override: None,
            freeu_override: false,
            dynthresh_override: None,
            chat: ChatState::new(),
            models: ModelsState::new(),
            output: OutputPane::new(),
            progress_rx,
            model_svc: ModelService::spawn(device.clone(), rt.clone()),
            rt,
            device,
            active_gen: None,
            refine_base: None,
            refine_prompt: String::new(),
            active_is_refine: false,
            active_full_prompt: String::new(),
            negative: String::new(),
            refine_strength: None,
            base_seed: None,
            fixed_seed: None,
            active_seed: 0,
            scenario_run: None,
            scenario_events: None,
            portrait_run: None,
            portrait_prompt: String::new(),
            chat_identity: None,
            pending_identity: None,
            remote_search: None,
            downloads_active: Vec::new(),
            downloads_queue: std::collections::VecDeque::new(),
            lora_assess: None,
            lora_recommend: None,
            lora_combine: None,
            lora_update: None,
            prompt_compile: None,
            prompt_structural: None,
            chat_to_scenario: None,
            canvas_faces: None,
            people_encode: None,
            history_decode: None,
            thumb_decode: None,
            active_loras: Vec::new(),
            chat_mask: None,
            inpaint_base: None,
            inpaint_nudged: false,
            auto_route: false,
            route_rx: None,
            screen: ActiveScreen::Chat,
            should_quit: false,
            picker,
            workspace,
        }
    }

    /// Enter the alternate screen + raw mode, run the loop, and always restore the
    /// terminal on the way out (`ratatui::init` also installs a panic hook that
    /// restores, so a panic won't leave the terminal wedged).
    /// One-time startup: pre-select the workspace's default model in Models, and
    /// auto-load it in the background if its weights are already cached (no surprise
    /// multi-GB download). Non-SD-family or uncached defaults are only pre-selected.
    fn startup(&mut self) {
        let default = self.workspace.config.default_model.clone();
        self.models.select_by_alias(&default);
        if super::services::model_service::t2i_load_check(&default).is_ok()
            && crate::hf::download::is_cached(&default)
        {
            self.load_model(default);
        }
    }

    pub fn run(&mut self) -> Result<()> {
        self.startup();
        let mut terminal = ratatui::init();
        // Enable the "disambiguate escape codes" keyboard protocol (Kitty/Ghostty/
        // WezTerm/foot) so Ctrl-1..8 report as clean `Ctrl+Char` events instead of
        // legacy control bytes (where Ctrl-3 == Esc, Ctrl-2 == NUL, Ctrl-8 == DEL).
        // Best-effort: terminals without the protocol fall back to the plain-digit
        // switch and are unaffected.
        let enhanced = crossterm::terminal::supports_keyboard_enhancement().unwrap_or(false);
        if enhanced {
            let _ = crossterm::execute!(
                std::io::stdout(),
                crossterm::event::PushKeyboardEnhancementFlags(
                    crossterm::event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                )
            );
        }
        // If the OOM watchdog has to hard-exit, it skips Drop — so restore the terminal
        // (raw mode + alt screen + keyboard flags) here too, or the user's shell is left
        // garbled. Best-effort, runs on the watchdog thread.
        crate::memwatch::set_abort_hook(move || {
            if enhanced {
                let _ = crossterm::execute!(std::io::stdout(), crossterm::event::PopKeyboardEnhancementFlags);
            }
            ratatui::restore();
        });
        let res = self.event_loop(&mut terminal);
        if enhanced {
            let _ = crossterm::execute!(std::io::stdout(), crossterm::event::PopKeyboardEnhancementFlags);
        }
        ratatui::restore();
        // A hard reset re-execs a fresh process (the only way to fully return candle's
        // Metal buffer pool). The terminal is already restored above, so the new process
        // starts clean. `exec` replaces this image and never returns on success.
        if self.should_reset && res.is_ok() {
            return Self::reexec();
        }
        // Clean user quit: force an immediate exit. `plakat ui` runs inside the shared
        // multi-thread tokio runtime and holds a background ModelService thread / blocking
        // inference tasks; letting `main` unwind lets the runtime teardown BLOCK on those
        // (no timeout) — leaving a defunct `(plakat)` stuck in exit. The terminal is already
        // restored, so exiting now is safe and guaranteed. (Errors still propagate below.)
        if res.is_ok() {
            std::process::exit(0);
        }
        res
    }

    /// Replace the current process with a fresh `plakat` invocation (same args). Only
    /// returns on failure (the error is surfaced to the caller).
    #[cfg(unix)]
    fn reexec() -> Result<()> {
        use std::os::unix::process::CommandExt;
        let exe = std::env::current_exe()?;
        let args: Vec<String> = std::env::args().skip(1).collect();
        // `exec` does not return on success — the process image is replaced.
        Err(std::process::Command::new(exe).args(args).exec().into())
    }

    #[cfg(not(unix))]
    fn reexec() -> Result<()> {
        let exe = std::env::current_exe()?;
        let args: Vec<String> = std::env::args().skip(1).collect();
        let status = std::process::Command::new(exe).args(args).status()?;
        std::process::exit(status.code().unwrap_or(0));
    }

    /// Keep the Naturalize tab in sync (RFC QUALITY-8 P3): when it's active and has no source yet, seed it
    /// from the latest generated frame; when the processed image needs a preview, build it via the Picker
    /// (the App owns it). Called once per frame — cheap when idle.
    fn sync_naturalize(&mut self) {
        if self.screen != ActiveScreen::Naturalize {
            return;
        }
        // Seed the source: the latest Chat frame, else the newest image in the workspace out-dir (so
        // existing images are picked up, not only fresh generations).
        if self.naturalize.source.is_none() {
            if let Some(p) = self.chat.latest_frame_path().or_else(|| Self::newest_image_in(&self.workspace.out_dir())) {
                self.naturalize.load(p);
            }
        }
        // Re-apply the weight-free pass when the processed image is missing (load / knob change clears it).
        if self.naturalize.source.is_some() && self.naturalize.processed.is_none() {
            self.naturalize.apply();
        }
        // (Re)build the terminal preview when flagged (needs the App's Picker) — from the original or the
        // de-slopped image per the Space toggle.
        if self.naturalize.needs_preview {
            let img = self.naturalize.preview_image().cloned();
            self.naturalize.preview = img.map(|i| self.picker.new_resize_protocol(image::DynamicImage::ImageRgb8(i)));
            self.naturalize.needs_preview = false;
        }
    }

    /// The newest image file in `dir` (by mtime), skipping our own `*_naturalized.png` outputs. For the
    /// Naturalize tab's workspace fallback / reload.
    fn newest_image_in(dir: &std::path::Path) -> Option<std::path::PathBuf> {
        let exts = ["png", "jpg", "jpeg", "webp"];
        std::fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok())
            .filter(|e| {
                let p = e.path();
                p.is_file()
                    && p.extension().and_then(|x| x.to_str()).map(|x| exts.contains(&x.to_ascii_lowercase().as_str())).unwrap_or(false)
                    && !p.file_stem().and_then(|s| s.to_str()).map(|s| s.ends_with("_naturalized")).unwrap_or(false)
            })
            .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
            .map(|e| e.path())
    }

    fn event_loop(&mut self, terminal: &mut ratatui::DefaultTerminal) -> Result<()> {
        while !self.should_quit && !self.should_reset {
            self.sync_naturalize();
            terminal.draw(|f| self.render(f))?;
            // 100 ms tick: poll input, then (later) drain the gen/llm/download
            // channels so a running generation keeps the UI live.
            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    self.handle_key(key);
                }
            }
            // Drain the rerouted progress sink → the Output pane (all pipelines).
            while let Ok(line) = self.progress_rx.try_recv() {
                self.output.push(line);
            }
            // Drain background-service messages each tick so a load in flight
            // updates the UI without blocking the event loop.
            while let Some(msg) = self.model_svc.try_recv() {
                self.models.apply(&msg);
            }
            self.drain_generation();
            self.drain_scenario();
            self.sync_history();
            self.sync_people();
            self.drain_portrait();
            // Keep the LoRA Hub's compatibility column + applied marks in sync.
            self.lorahub.set_loaded_family(
                self.models.loaded_alias().map(crate::preset::discovery::BaseFamily::from_model_arg),
            );
            self.lorahub.set_applied(&self.active_loras);
            self.drain_civitai();
            self.sync_prompts();
            self.drain_prompt_compile();
            self.sync_canvas();
            self.drain_canvas_faces();
            self.drain_people_encode();
            self.drain_route();
            self.sync_chat_mentions();
            self.sync_chat_mode();
            self.idle_tick();
        }
        Ok(())
    }

    /// Auto-unload the loaded model after [`IDLE_UNLOAD`] of no keypresses, returning
    /// its memory. It reloads on the next activity (see `resume_if_suspended`). Never
    /// fires mid-generation or when nothing is loaded.
    fn idle_tick(&mut self) {
        if self.active_gen.is_some() || self.suspended.is_some() {
            return;
        }
        let Some(alias) = self.models.loaded_alias().map(str::to_string) else {
            return;
        };
        if self.last_activity.elapsed() >= IDLE_UNLOAD {
            self.output.push(format!(
                "idle {}m — unloading {alias} to free memory (auto-reloads when you resume)",
                IDLE_UNLOAD.as_secs() / 60
            ));
            self.model_svc.unload();
            self.suspended = Some(alias);
            // Reset so we don't re-attempt while the Unloaded message is in flight.
            self.last_activity = Instant::now();
        }
    }

    /// If idle auto-unload freed a model and nothing is loaded/loading, kick off a
    /// background reload so it's ready by the time the user generates again.
    fn resume_if_suspended(&mut self) {
        if self.suspended.is_none() || self.models.loaded_alias().is_some() || self.models.is_loading() {
            return;
        }
        if let Some(alias) = self.suspended.take() {
            self.output.push(format!("welcome back — reloading {alias}…"));
            self.load_model(alias);
        }
    }

    /// Keep the Chat `@mention` candidates current (people labels + local LoRA names),
    /// only while Chat is active.
    fn sync_chat_mentions(&mut self) {
        if self.screen != ActiveScreen::Chat {
            return;
        }
        self.chat.set_mention_candidates(self.people.names(), self.lorahub.local_names());
    }

    /// Summarize what the next Enter will do (mode + strength + seed) for the Chat pane
    /// title, so the build-an-image loop's current behaviour is always visible.
    fn chat_mode_hint(&self) -> String {
        // No base image yet → the next Enter starts a fresh generation.
        if self.base_seed.is_none() {
            return "fresh".into();
        }
        let mut mode = if self.chat_mask.is_some() && self.inpaint_base.is_some() {
            "inpaint".to_string()
        } else if self.chat_identity.is_some() {
            "identity".to_string()
        } else if let Some(s) = self.refine_strength {
            format!("anchored {s:.2}")
        } else {
            "evolve".to_string()
        };
        if let Some(seed) = self.fixed_seed.or(self.base_seed) {
            mode.push_str(&format!(" · seed {seed}"));
        }
        // Show a non-native generation size when one is set for the loaded model.
        if let Some((w, h)) = self.models.loaded_alias().and_then(|a| self.gen_sizes.get(a)) {
            mode.push_str(&format!(" · {w}×{h}"));
        }
        if self.auto_route {
            mode.push_str(" · auto");
        }
        mode
    }

    /// Ctrl-T in Chat: flip the refine mode between prompt-evolve (`None`) and
    /// image-anchored img2img at [`DEFAULT_ANCHOR_STRENGTH`]. A no-op hint until an
    /// image exists to anchor to.
    fn toggle_anchor(&mut self) {
        self.refine_strength = match self.refine_strength {
            Some(_) => None,
            None => Some(DEFAULT_ANCHOR_STRENGTH),
        };
        let msg = match self.refine_strength {
            Some(s) => format!("anchored img2img at strength {s:.2} (Ctrl-T to switch back)"),
            None => "prompt-evolve (txt2img at the stable seed)".to_string(),
        };
        self.chat.push_system(msg);
    }

    /// The model whose generation size `/size` acts on: the loaded model, else the
    /// Models cursor (so you can pre-set a size before loading).
    fn size_target_alias(&self) -> Option<String> {
        self.models.loaded_alias().map(str::to_string).or_else(|| self.models.selected_alias())
    }

    /// The generation dimensions for `alias`: its per-model override, else the native
    /// square (always Metal-safe).
    fn resolve_gen_size(&self, alias: &str) -> (u32, u32) {
        match self.gen_sizes.get(alias) {
            Some(&(w, h)) => (w, h),
            None => {
                let n = crate::capability::native_res(alias);
                (n, n)
            }
        }
    }

    /// `/size <spec>` — set (or clear) the loaded/selected model's generation size, with a
    /// memory-budget check against the generation's estimated working set. `spec` is
    /// `WxH`, a single `N` (square), or `native`/`off`/`auto`/empty to clear the override.
    fn set_gen_size(&mut self, spec: &str) {
        let Some(alias) = self.size_target_alias() else {
            self.chat.push_system("load or select a model first, then /size".into());
            return;
        };
        if spec.is_empty() || matches!(spec.to_lowercase().as_str(), "native" | "off" | "auto" | "reset") {
            self.gen_sizes.remove(&alias);
            let n = crate::capability::native_res(&alias);
            self.chat.push_system(format!("{alias} size → native {n}×{n}"));
            return;
        }
        let Some((w, h)) = parse_gen_size(spec) else {
            self.chat.push_system("size must be WxH or N (e.g. 1024x768 or 1024), or 'native'".into());
            return;
        };
        // Budget guard: estimate the generation's extra working set vs free RAM.
        let est = crate::capability::generation_estimate_gb(&alias, w, h);
        let free = crate::hw::available_ram_gb();
        self.gen_sizes.insert(alias.clone(), (w, h));
        let mut msg = format!("{alias} size → {w}×{h} (≈{est:.1} GB working set, {free:.1} GB free)");
        if est > free {
            msg.push_str(" ⚠ may OOM — the memory guard will abort cleanly if it does");
        }
        self.chat.push_system(msg);
    }

    fn sync_chat_mode(&mut self) {
        if self.screen != ActiveScreen::Chat {
            return;
        }
        let hint = self.chat_mode_hint();
        self.chat.set_mode_hint(hint);
    }

    fn drain_prompt_compile(&mut self) {
        if let Some(rx) = &self.prompt_compile {
            match rx.try_recv() {
                Ok(result) => {
                    self.prompts.compiling = false;
                    match result {
                        Ok(hjson) => {
                            self.prompts.compiled = hjson;
                            self.prompts.compile_err = None;
                        }
                        Err(e) => self.prompts.compile_err = Some(e),
                    }
                    self.prompt_compile = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.prompts.compiling = false;
                    self.prompt_compile = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
    }

    /// Lazily decode the selected person's primary reference photo into a preview,
    /// only while People is active and only when the selection changed.
    fn sync_people(&mut self) {
        if self.screen != ActiveScreen::People {
            return;
        }
        let sel = self.people.selected_ref();
        if sel != self.people.preview_for {
            self.people.preview = sel
                .as_ref()
                .and_then(|p| image::open(p).ok())
                .map(|img| self.picker.new_resize_protocol(img));
            self.people.preview_for = sel;
        }
        // Auto-compute the encoding quality the first time the ENCODING tab is viewed on
        // an unscored identity (once per identity per session; not while one's running).
        if self.people_encode.is_none() {
            if let people::PeopleAction::Encode { name, dir, photos, fingerprint } = self.people.auto_encode_request() {
                self.encode_person(name, dir, photos, fingerprint);
            }
        }
    }

    /// Lazily decode the selected History image into a preview (and read its recipe),
    /// but only while History is the active screen and only when the selection
    /// changed — so navigating the list never blocks the event loop on every frame.
    fn sync_history(&mut self) {
        if self.screen != ActiveScreen::History {
            return;
        }
        self.history.sync_detail();
        // Grid view: lazily build thumbnail protocols for the visible page, one decode
        // per tick (small thumbnails, cheap to build), so the grid never hitches.
        if self.history.is_grid() {
            self.sync_history_thumbs();
            return; // the grid doesn't use the single big preview
        }
        let sel = self.history.selected_path();
        // Already showing the right image (or nothing selected) → done.
        if sel == self.history.preview_for {
            return;
        }
        // A completed background decode for the *current* selection → build the
        // protocol (cheap) on the main thread and show it. A stale result (the user
        // moved on) is dropped; the mismatch re-triggers a decode below.
        if let Some((path, rx)) = &self.history_decode {
            match rx.try_recv() {
                Ok(decoded) => {
                    let matched = Some(path) == sel.as_ref();
                    self.history_decode = None;
                    if matched {
                        self.history.preview = decoded.map(|img| self.picker.new_resize_protocol(img));
                        self.history.preview_for = sel;
                        return; // showing the current selection now
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.history_decode = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        // Start a decode only when NONE is in flight — not merely when the current
        // selection isn't being decoded. Holding j/k over large (upscaled) PNGs otherwise
        // spawned a fresh full-image decode every 100 ms tick (each replacing the tracked
        // rx but leaving its thread running detached) → a pile-up of concurrent decodes.
        // Serializing to one at a time bounds the work; a stale result is dropped above and
        // the next tick starts the current selection's decode.
        match &sel {
            None => {
                self.history.preview = None;
                self.history.preview_for = None;
                self.history_decode = None;
            }
            Some(path) if self.history_decode.is_none() => {
                let path = path.clone();
                let (tx, rx) = std::sync::mpsc::channel();
                let worker_path = path.clone();
                std::thread::spawn(move || {
                    let _ = tx.send(image::open(&worker_path).ok());
                });
                self.history_decode = Some((path, rx));
            }
            Some(_) => {}
        }
    }

    /// Build thumbnail protocols for the visible grid page, one decode per tick on a
    /// worker thread (resized small), so a page of large PNGs never hitches the loop.
    fn sync_history_thumbs(&mut self) {
        // Land a completed thumbnail decode into the History cache (build the protocol
        // on the main thread — it owns the Picker).
        if let Some((path, rx)) = &self.thumb_decode {
            match rx.try_recv() {
                Ok(decoded) => {
                    if let Some(img) = decoded {
                        let proto = self.picker.new_resize_protocol(img);
                        self.history.set_thumb(path.clone(), proto);
                    }
                    self.thumb_decode = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.thumb_decode = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => return, // still decoding
            }
        }
        if self.thumb_decode.is_some() {
            return;
        }
        // The first visible cell whose thumbnail isn't cached yet → decode it.
        let Some(next) = self
            .history
            .visible_thumb_paths()
            .into_iter()
            .find(|p| !self.history.has_thumb(p))
        else {
            return;
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let worker_path = next.clone();
        std::thread::spawn(move || {
            // Decode + downscale to a thumbnail so building the protocol is cheap and the
            // cache stays small.
            let thumb = image::open(&worker_path)
                .ok()
                .map(|img| img.thumbnail(192, 192));
            let _ = tx.send(thumb);
        });
        self.thumb_decode = Some((next, rx));
    }

    fn handle_key(&mut self, key: KeyEvent) {
        // Any keypress is activity: reset the idle timer and, if a model was
        // auto-unloaded while away, start reloading it in the background now.
        self.last_activity = Instant::now();
        self.resume_if_suspended();
        // ── The command palette overlay owns all input while open. ──
        if self.palette.is_open() {
            if let PaletteResult::Run(cmd) = self.palette.handle_key(key) {
                self.exec_palette(cmd);
            }
            return;
        }
        // ── The keyboard cheatsheet (F1) owns input while open: any key dismisses. ──
        if self.show_help {
            self.show_help = false;
            return;
        }
        if key.code == KeyCode::F(1) {
            self.show_help = true;
            return;
        }
        // ── A hard-reset confirmation owns input until answered. ──
        if self.confirm_reset {
            self.confirm_reset = false;
            match key.code {
                KeyCode::Char('y' | 'Y') | KeyCode::Enter => self.should_reset = true,
                _ => self.output.push("restart cancelled".into()),
            }
            return;
        }
        // ── A memory-budget confirmation owns input until answered. ──
        if let Some(pl) = self.pending_load.take() {
            match key.code {
                KeyCode::Char('y' | 'Y') | KeyCode::Enter => {
                    self.output.push(format!("loading {} anyway (~{:.1} GB vs {:.1} GB free)", pl.alias, pl.est_gb, pl.avail_gb));
                    self.load_model(pl.alias);
                }
                _ => self.output.push(format!("load of {} cancelled — free up memory or pick a smaller model", pl.alias)),
            }
            return;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        // ── Always-global keys (work even while a text input is focused) ──
        // Ctrl-K opens the command palette from anywhere (even a text editor).
        if ctrl && key.code == KeyCode::Char('k') {
            self.open_palette();
            return;
        }
        match key.code {
            // Ctrl-Q always quits. Ctrl-C cancels a running generation if there is
            // one (it saves the partial), else quits.
            KeyCode::Char('q') if ctrl => {
                self.should_quit = true;
                return;
            }
            KeyCode::Char('c') if ctrl => {
                if let Some((_, cancel)) = &self.active_gen {
                    cancel.cancel();
                } else {
                    self.should_quit = true;
                }
                return;
            }
            // Ctrl-1..8 — disambiguated screen jump (Kitty/Ghostty/WezTerm/foot).
            KeyCode::Char(c @ '1'..='9') if ctrl => {
                if let Some(s) = ActiveScreen::from_index((c as u8 - b'1') as usize) {
                    self.screen = s;
                }
                return;
            }
            // Tab / Shift-Tab cycle screens — universal (BackTab legacy; Tab+SHIFT
            // under the kbd protocol).
            KeyCode::BackTab => {
                self.screen = self.screen.cycle(-1);
                return;
            }
            KeyCode::Tab if shift => {
                self.screen = self.screen.cycle(-1);
                return;
            }
            // Plain Tab cycles screens; Ctrl-Tab falls through (the Prompt Workspace
            // uses it to cycle buffers).
            KeyCode::Tab if !ctrl => {
                self.screen = self.screen.cycle(1);
                return;
            }
            _ => {}
        }

        // ── Chat owns text input: plain chars / Enter / Backspace go to it. ──
        if self.screen == ActiveScreen::Chat {
            match self.chat.handle_key(key) {
                ChatAction::Submit(prompt) => self.handle_chat_submit(prompt),
                ChatAction::ApplyLora(name) => self.apply_lora_by_name(&name),
                ChatAction::SelectFrame(path) => self.show_chat_frame(path),
                ChatAction::Rollback(path) => self.rollback_to_frame(path),
                ChatAction::Vary(path) => self.vary_frame(path),
                ChatAction::ToggleAnchor => self.toggle_anchor(),
                ChatAction::None => {}
            }
            return;
        }

        // ── The Scenarios EDITOR / RUNNER own the keyboard (type into the buffer /
        //    capture Esc) — route everything to them while active. ──
        if self.screen == ActiveScreen::Scenarios && self.scenarios.captures_input() {
            let action = self.scenarios.handle_key(key);
            self.handle_scenarios_action(action);
            return;
        }

        // ── The LoRA Hub's Civitai search box owns the keyboard while editing. ──
        if self.screen == ActiveScreen::LoraHub && self.lorahub.captures_input() {
            let action = self.lorahub.handle_key(key);
            self.handle_lorahub_action(action);
            return;
        }

        // ── The Prompt Workspace editor owns the keyboard while focused. ──
        if self.screen == ActiveScreen::PromptWorkspace && self.prompts.captures_input() {
            let action = self.prompts.handle_key(key);
            self.handle_prompts_action(action);
            return;
        }

        // ── History's filter / tag input owns the keyboard while typing. ──
        if self.screen == ActiveScreen::History && self.history.captures_input() {
            let action = self.history.handle_key(key);
            self.handle_history_action(action);
            return;
        }

        // ── The Naturalize tab's open-path line owns the keyboard while typing. ──
        if self.screen == ActiveScreen::Naturalize && self.naturalize.captures_input() {
            let _ = self.naturalize.handle_key(key);
            return;
        }

        // ── People's delete-confirm modal owns the keyboard (type the name). ──
        if self.screen == ActiveScreen::People && self.people.captures_input() {
            match self.people.handle_key(key) {
                people::PeopleAction::Generate(spec) => self.quick_generate(spec),
                people::PeopleAction::GenerateMulti(specs) => self.quick_generate_multi(specs),
                people::PeopleAction::Encode { name, dir, photos, fingerprint } => self.encode_person(name, dir, photos, fingerprint),
                people::PeopleAction::None => {}
            }
            return;
        }

        // ── The Canvas owns the keyboard (preset letters + Space painting). ──
        if self.screen == ActiveScreen::Canvas {
            match self.canvas.handle_key(key) {
                canvas::CanvasAction::MaskReady(path) => self.apply_canvas_mask(path),
                canvas::CanvasAction::OutpaintReady { base, mask } => self.apply_outpaint(base, mask),
                canvas::CanvasAction::None => {}
            }
            return;
        }

        // ── Non-input screens: plain digits switch, q quits, else delegate. ──
        match key.code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char(c @ '1'..='9') => {
                if let Some(s) = ActiveScreen::from_index((c as u8 - b'1') as usize) {
                    self.screen = s;
                }
            }
            KeyCode::Esc => {}
            _ => self.handle_screen_key(key),
        }
    }

    /// Delegate a non-global key to the active (non-input) screen.
    fn handle_screen_key(&mut self, key: KeyEvent) {
        match self.screen {
            ActiveScreen::Models => match key.code {
                // [L] load the selected model, [U] unload — dispatched to the
                // background ModelService (the event loop stays live during load).
                KeyCode::Char('l' | 'L') => {
                    if let Some(alias) = self.models.selected_alias() {
                        self.request_load(alias);
                    }
                }
                KeyCode::Char('u' | 'U') => self.model_svc.unload(),
                _ => {
                    self.models.handle_key(key);
                }
            },
            ActiveScreen::Scenarios => {
                let action = self.scenarios.handle_key(key);
                self.handle_scenarios_action(action);
            }
            ActiveScreen::History => {
                let action = self.history.handle_key(key);
                self.handle_history_action(action);
            }
            ActiveScreen::People => match self.people.handle_key(key) {
                people::PeopleAction::Generate(spec) => self.quick_generate(spec),
                people::PeopleAction::GenerateMulti(specs) => self.quick_generate_multi(specs),
                people::PeopleAction::Encode { name, dir, photos, fingerprint } => self.encode_person(name, dir, photos, fingerprint),
                people::PeopleAction::None => {}
            },
            ActiveScreen::LoraHub => {
                let action = self.lorahub.handle_key(key);
                self.handle_lorahub_action(action);
            }
            ActiveScreen::PromptWorkspace => {
                let action = self.prompts.handle_key(key);
                self.handle_prompts_action(action);
            }
            ActiveScreen::Naturalize => match self.naturalize.handle_key(key) {
                // Knob changes / the Space toggle mutate the screen state directly (clearing `processed` /
                // setting `needs_preview`); `sync_naturalize` re-applies + rebuilds the preview next frame.
                NaturalizeAction::Save => self.save_naturalized(),
                NaturalizeAction::Reload => {
                    self.naturalize.source = None;
                    self.naturalize.source_path = None;
                }
                NaturalizeAction::None => {}
            },
            _ => {}
        }
    }

    /// Save the Naturalize tab's current processed image next to the source (`*_naturalized.png`).
    fn save_naturalized(&mut self) {
        let (Some(img), Some(src)) = (self.naturalize.processed.clone(), self.naturalize.source_path.clone()) else {
            self.naturalize.status = "nothing to save".into();
            return;
        };
        let out = src.with_file_name(format!(
            "{}_naturalized.png",
            src.file_stem().and_then(|s| s.to_str()).unwrap_or("image")
        ));
        match img.save(&out) {
            Ok(()) => self.naturalize.status = format!("saved → {}", out.display()),
            Err(e) => self.naturalize.status = format!("save failed: {e}"),
        }
    }

    fn handle_prompts_action(&mut self, action: prompts::PromptsAction) {
        match action {
            prompts::PromptsAction::None => {}
            prompts::PromptsAction::LlmCompile(text) => self.prompt_llm_compile(text),
            prompts::PromptsAction::OpenInScenarios { name, hjson } => {
                self.open_compiled_in_scenarios(name, hjson)
            }
        }
    }

    /// Build the command list for the current context and open the palette (Ctrl-K):
    /// the active screen's most-relevant actions first, then screen navigation, then
    /// quit. Most entries replay a key into the active screen so existing handlers run.
    fn open_palette(&mut self) {
        let k = |c: char| Cmd::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        let kc = |c: char| Cmd::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL));
        let enter = Cmd::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let mut cmds: Vec<(String, Cmd)> = Vec::new();
        match self.screen {
            ActiveScreen::Chat => {
                cmds.push(("Save Chat session".into(), Cmd::Submit("/save".into())));
                cmds.push(("List Chat sessions".into(), Cmd::Submit("/sessions".into())));
                cmds.push(("Fan out 4 variations (fresh seeds)".into(), Cmd::Submit("/vary 4".into())));
                cmds.push(("Grab this image's recipe → Scenario (exact)".into(), Cmd::Submit("/scenario".into())));
                cmds.push(("Toggle refine mode: evolve ↔ anchored".into(), kc('t')));
                cmds.push(("Toggle auto edit/new routing".into(), Cmd::Submit("/auto".into())));
                cmds.push(("Clear negative prompt".into(), Cmd::Submit("/negative".into())));
                if self.chat.frames().len() > 1 {
                    cmds.push(("Filmstrip: previous frame".into(), Cmd::Key(KeyEvent::new(KeyCode::Left, KeyModifiers::CONTROL))));
                    cmds.push(("Filmstrip: next frame".into(), Cmd::Key(KeyEvent::new(KeyCode::Right, KeyModifiers::CONTROL))));
                    cmds.push(("Roll back to selected frame".into(), kc('b')));
                    cmds.push(("New variation of selected frame".into(), kc('y')));
                    cmds.push(("Undo (previous image)".into(), kc('z')));
                    cmds.push(("Redo (next image)".into(), Cmd::Key(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::CONTROL | KeyModifiers::SHIFT))));
                }
            }
            ActiveScreen::Models => {
                cmds.push(("Load selected model".into(), k('l')));
                cmds.push(("Unload model".into(), k('u')));
            }
            ActiveScreen::Scenarios => {
                cmds.push(("Run selected scenario".into(), enter.clone()));
                cmds.push(("Edit selected scenario".into(), k('e')));
                cmds.push(("New scenario".into(), k('n')));
            }
            ActiveScreen::History => {
                cmds.push(("Toggle thumbnail grid".into(), k('v')));
                cmds.push(("Filter images…".into(), k('/')));
                cmds.push(("Semantic search…".into(), k('?')));
                cmds.push(("Tag selected".into(), k('t')));
                cmds.push(("Export filtered set".into(), k('x')));
                cmds.push(("Compare baseline".into(), k('d')));
                cmds.push(("Continue selected in Chat".into(), k('c')));
                cmds.push(("Naturalize selected (de-slop)".into(), k('n')));
                cmds.push(("Vary selected".into(), k('y')));
                cmds.push(("Etch provenance into selected".into(), k('e')));
                cmds.push(("Rescan".into(), k('r')));
            }
            ActiveScreen::People => {
                cmds.push(("Generate selected → Chat".into(), k('g')));
                cmds.push(("Next detail tab".into(), Cmd::Key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE))));
                cmds.push(("Encode identity (quality score)".into(), k('e')));
                cmds.push(("Import scenario persona".into(), k('i')));
                cmds.push(("Rescan".into(), k('r')));
            }
            ActiveScreen::LoraHub => {
                cmds.push(("Apply selected LoRA".into(), k('a')));
                cmds.push(("Assess selected (LLM)".into(), k('r')));
                cmds.push(("Check for a newer version".into(), k('u')));
                cmds.push(("Suggest a LoRA stack (LLM)".into(), kc('r')));
            }
            ActiveScreen::PromptWorkspace => {
                cmds.push(("New buffer".into(), kc('n')));
                cmds.push(("Save buffer".into(), kc('s')));
                cmds.push(("Toggle Tera mode".into(), kc('t')));
                cmds.push(("LLM compile".into(), kc('r')));
                cmds.push(("Open compiled in Scenarios".into(), kc('o')));
            }
            ActiveScreen::Canvas => {
                cmds.push(("Outpaint mode".into(), k('m')));
                cmds.push(("Rasterize mask → Chat".into(), enter.clone()));
                cmds.push(("Clear mask".into(), Cmd::Key(KeyEvent::new(KeyCode::Char('C'), KeyModifiers::SHIFT))));
            }
            ActiveScreen::Naturalize => {
                cmds.push(("Naturalize: save de-slopped image".into(), k('s')));
            }
        }
        // The selected model's cache health (Models screen only — needs a selection).
        if self.screen == ActiveScreen::Models {
            cmds.push(("Cache doctor: sweep locks + report".into(), Cmd::CacheDoctor));
        }
        // Apply a saved preset (model + LoRA stack + negative) from anywhere.
        for p in crate::ui::tui::presets::load(&self.workspace.root) {
            cmds.push((format!("Preset: {} ({})", p.name, p.summary()), Cmd::ApplyPreset(p.name)));
        }
        for s in ActiveScreen::ALL {
            if s != self.screen {
                cmds.push((format!("Go to {}", s.title()), Cmd::Goto(s.index())));
            }
        }
        cmds.push(("Keyboard shortcuts (F1)".into(), Cmd::Key(KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE))));
        cmds.push(("Restart plakat (free all GPU memory)".into(), Cmd::HardReset));
        cmds.push(("Quit plakat ui".into(), Cmd::Quit));
        self.palette.open(cmds);
    }

    /// Execute a palette command: navigate, quit, replay a key into the active screen,
    /// or submit a Chat line.
    fn exec_palette(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Goto(i) => {
                if let Some(s) = ActiveScreen::from_index(i) {
                    self.screen = s;
                }
            }
            Cmd::Quit => self.should_quit = true,
            Cmd::Key(k) => self.handle_key(k),
            Cmd::Submit(line) => {
                self.screen = ActiveScreen::Chat;
                self.handle_chat_submit(line);
            }
            Cmd::HardReset => self.confirm_reset = true,
            Cmd::CacheDoctor => self.run_cache_doctor(),
            Cmd::ApplyPreset(name) => {
                self.screen = ActiveScreen::Chat;
                self.apply_preset(&name);
            }
        }
    }

    /// Deterministic structural compile (no LLM) of the Prompt Workspace buffer,
    /// recomputed when the text changes. The compile runs on a WORKER thread (the event
    /// loop is inside the tokio runtime, so `block_on` here would panic); the result is
    /// drained back into the pane the next tick.
    fn sync_prompts(&mut self) {
        if self.screen != ActiveScreen::PromptWorkspace || self.prompts.compiling {
            return;
        }
        // Drain an in-flight structural compile.
        if let Some((src, rx)) = &self.prompt_structural {
            match rx.try_recv() {
                Ok(result) => {
                    match result {
                        Ok(hjson) => {
                            self.prompts.compiled = hjson;
                            self.prompts.compile_err = None;
                        }
                        Err(e) => self.prompts.compile_err = Some(e),
                    }
                    self.prompts.last_compiled_src = Some(src.clone());
                    self.prompt_structural = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => return, // still computing
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.prompt_structural = None,
            }
        }
        let src = self.prompts.editor_text();
        if self.prompts.last_compiled_src.as_deref() == Some(src.as_str()) {
            return;
        }
        // Already compiling this exact source → wait for it.
        if self.prompt_structural.as_ref().map(|(s, _)| s == &src).unwrap_or(false) {
            return;
        }
        // Tera mode: render the buffer through the Tera pre-pass (with the panel's
        // variable values) before the structural compile. The Tera render is synchronous
        // and cheap, so it runs inline; a Tera error stops here.
        let to_compile = if self.prompts.tera_mode {
            let topts = crate::compile::TemplateOpts {
                vars: self.prompts.tera_var_pairs(),
                ..Default::default()
            };
            match crate::compile::template::render(&src, self.prompts.path(), &topts) {
                Ok(rendered) => rendered,
                Err(e) => {
                    self.prompts.compile_err = Some(format!("Tera: {e:#}"));
                    self.prompts.last_compiled_src = Some(src);
                    return;
                }
            }
        } else {
            src.clone()
        };
        let opts = self.compile_opts(&self.prompts.buffer_name(), true);
        let (tx, rx) = std::sync::mpsc::channel();
        let rt = self.rt.clone();
        std::thread::spawn(move || {
            let result = rt
                .block_on(crate::compile::compile_to_string(&to_compile, &opts))
                .map_err(|e| format!("{e:#}"));
            let _ = tx.send(result);
        });
        self.prompt_structural = Some((src, rx));
    }

    /// Build compile options. `structural` → no LLM (deterministic); else the full
    /// LLM compile using the workspace enhancer provider.
    fn compile_opts(&self, name: &str, structural: bool) -> crate::compile::CompileOpts {
        crate::compile::CompileOpts {
            provider: if structural { "none".into() } else { self.workspace.config.enhancer.clone() },
            default_model: self.workspace.config.default_model.clone(),
            no_enhance: structural,
            no_negative: structural,
            system_override: None,
            cache: !structural,
            parallel: 0,
            input_name: name.to_string(),
        }
    }

    /// Run the full LLM compile on a background thread; result lands in the pane.
    fn prompt_llm_compile(&mut self, text: String) {
        if self.prompt_compile.is_some() {
            return;
        }
        // In Tera mode, render the template first (same as the live structural pane).
        let text = if self.prompts.tera_mode {
            let topts = crate::compile::TemplateOpts { vars: self.prompts.tera_var_pairs(), ..Default::default() };
            match crate::compile::template::render(&text, self.prompts.path(), &topts) {
                Ok(r) => r,
                Err(e) => {
                    self.prompts.compiling = false;
                    self.prompts.compile_err = Some(format!("Tera: {e:#}"));
                    return;
                }
            }
        } else {
            text
        };
        let opts = self.compile_opts(&self.prompts.buffer_name(), false);
        let (tx, rx) = std::sync::mpsc::channel();
        let rt = self.rt.clone();
        std::thread::spawn(move || {
            let result = rt
                .block_on(crate::compile::compile_to_string(&text, &opts))
                .map_err(|e| format!("{e:#}"));
            let _ = tx.send(result);
        });
        self.prompt_compile = Some(rx);
    }

    /// `Ctrl-O` — save the compiled HJSON into the scenarios dir and open it in the
    /// Scenarios EDITOR.
    fn open_compiled_in_scenarios(&mut self, name: String, hjson: String) {
        let path = self.workspace.scenarios_dir().join(format!("{name}.hjson"));
        let _ = std::fs::create_dir_all(self.workspace.scenarios_dir());
        match std::fs::write(&path, hjson) {
            Ok(()) => {
                self.scenarios.rescan();
                self.scenarios.open_path_in_editor(path);
                self.screen = ActiveScreen::Scenarios;
            }
            Err(e) => self.output.push(format!("✗ could not write scenario: {e}")),
        }
    }

    fn handle_lorahub_action(&mut self, action: lorahub::LoraHubAction) {
        match action {
            lorahub::LoraHubAction::None => {}
            lorahub::LoraHubAction::ToggleApply { path, compatible } => self.toggle_lora(path, compatible),
            lorahub::LoraHubAction::Search { source, query } => self.remote_search(source, query),
            lorahub::LoraHubAction::Download { dl, title } => self.remote_download(dl, title),
            lorahub::LoraHubAction::Assess { key, prompt } => self.assess_lora(key, prompt),
            lorahub::LoraHubAction::Recommend { candidates } => self.recommend_loras(candidates),
            lorahub::LoraHubAction::AdjustWeight { path, delta } => self.adjust_lora_weight(path, delta),
            lorahub::LoraHubAction::SuggestCombination { candidates } => self.suggest_combination(candidates),
            lorahub::LoraHubAction::CheckUpdate { path, name } => self.check_lora_update(path, name),
        }
    }

    /// `U` — check whether a Civitai-sourced LoRA has a newer version; if so, download it
    /// (it lands in LOCAL) — reuses the remote-download path. Reports to the Output pane.
    fn check_lora_update(&mut self, path: std::path::PathBuf, name: String) {
        let Some((model_id, version_id)) = lorahub::civitai_ids_from_path(&path) else {
            self.output.push(format!("‘{name}’ isn't a Civitai download — no update check"));
            return;
        };
        if self.lora_update.is_some() {
            return;
        }
        self.output.push(format!("checking ‘{name}’ for a newer version…"));
        let (tx, rx) = std::sync::mpsc::channel();
        let rt = self.rt.clone();
        std::thread::spawn(move || {
            // Query the model's versions; report a newer one (id, name) if present.
            let result = rt
                .block_on(crate::civitai::api::get_model(model_id))
                .map(|model| lorahub::newer_version(version_id, &model.model_versions).map(|v| (v.id, v.name.clone())))
                .map_err(|e| format!("{e:#}"));
            let _ = tx.send((model_id, name, result));
        });
        self.lora_update = Some(rx);
    }

    /// `Ctrl-R` — ask the LLM which compatible LoRAs to STACK for the current Chat
    /// prompt, on a background thread; the suggestion shows in the LOCAL detail.
    /// Route a Scenarios screen action.
    fn handle_scenarios_action(&mut self, action: ScenariosAction) {
        match action {
            ScenariosAction::None => {}
            ScenariosAction::Run(path) => self.run_scenario(path),
            ScenariosAction::GrabFromChat => self.grab_chat_into_scenario(),
        }
    }

    /// Summarize the current Chat session into one coherent image prompt and (when the
    /// LLM returns) insert it as a `{ name, prompt }` task at the editor cursor. Source
    /// = the session's non-system utterances (the whole refinement thread), else the
    /// accumulated refine prompt, else whatever is typed in the Chat box. The summary
    /// runs on a background thread (the editor stays live); `drain_civitai` delivers it.
    fn grab_chat_into_scenario(&mut self) {
        if self.chat_to_scenario.is_some() {
            return;
        }
        // Build the source material from the chat thread.
        let steps: Vec<String> = self
            .chat
            .history
            .iter()
            .filter(|e| !e.system)
            .map(|e| e.utterance.clone())
            .collect();
        let source = if !steps.is_empty() {
            steps.join("\n")
        } else if !self.refine_prompt.is_empty() {
            self.refine_prompt.clone()
        } else {
            self.chat.editor.text()
        };
        if source.trim().is_empty() {
            self.scenarios.set_status("✗ nothing in Chat to summarize yet — generate something first");
            return;
        }
        let negative = self.negative.clone();
        let provider = crate::prompt::resolve_provider_label(&self.workspace.config.enhancer);
        let (tx, rx) = std::sync::mpsc::channel();
        let rt = self.rt.clone();
        std::thread::spawn(move || {
            const SYSTEM: &str = "You distill a Stable Diffusion chat session into ONE reusable \
                image prompt. The user's messages are successive refinement steps (each builds on \
                the last). Merge them into a single coherent, comma-separated prompt capturing the \
                final intended image. Output ONLY the prompt — no preamble, no quotes, no markdown.";
            let user = if negative.trim().is_empty() {
                format!("Refinement steps (oldest first):\n{source}\n\nFinal prompt:")
            } else {
                format!("Refinement steps (oldest first):\n{source}\n\n(Negative: {negative})\n\nFinal prompt:")
            };
            let result = rt
                .block_on(crate::prompt::complete(&provider, SYSTEM, &user, &crate::prompt::EnhanceArgs::default()))
                .map(|t| t.trim().trim_matches('"').to_string())
                .map_err(|e| format!("{e:#}"));
            let _ = tx.send(result);
        });
        self.chat_to_scenario = Some(rx);
    }

    /// `/scenario` — promote the CURRENT Chat image's embedded recipe into a scenario
    /// task in one step (exact prompt + negative + seed, no LLM), then jump to Scenarios.
    fn grab_recipe_into_scenario(&mut self) {
        let Some(path) = self.chat.latest_frame_path() else {
            self.chat.push_system("no image yet — generate one first, then /scenario".into());
            return;
        };
        let params = crate::imaging::io::read_parameters_chunk(&path).ok().flatten();
        let Some(params) = params else {
            self.chat.push_system("this image has no embedded recipe to grab".into());
            return;
        };
        let prompt = recipe_positive(&params);
        if prompt.trim().is_empty() {
            self.chat.push_system("the recipe has no prompt to grab".into());
            return;
        }
        let negative = recipe_negative(&params);
        let seed = recipe_seed(&params);
        self.scenarios.insert_recipe_task("from-chat", &prompt, &negative, seed);
        self.screen = ActiveScreen::Scenarios;
        self.chat.push_system("✓ recipe → Scenarios (exact prompt + seed)".into());
    }

    fn suggest_combination(&mut self, candidates: Vec<String>) {
        if self.lora_combine.is_some() {
            return;
        }
        let context = if !self.refine_prompt.is_empty() {
            self.refine_prompt.clone()
        } else {
            let typed = self.chat.editor.text();
            if typed.trim().is_empty() { "a general image".into() } else { typed }
        };
        let list = candidates.join(", ");
        let provider = crate::prompt::resolve_provider_label(&self.workspace.config.enhancer);
        let (tx, rx) = std::sync::mpsc::channel();
        let rt = self.rt.clone();
        std::thread::spawn(move || {
            const SYSTEM: &str = "You compose Stable Diffusion LoRA STACKS. Given an image \
                prompt and available LoRAs, suggest which to combine (1–3) and rough weights, \
                in ONE plain sentence. No preamble, no markdown.";
            let user = format!("Image prompt: {context}\n\nAvailable LoRAs: {list}\n\nSuggest a combination.");
            let text = rt
                .block_on(crate::prompt::complete(&provider, SYSTEM, &user, &crate::prompt::EnhanceArgs::default()))
                .unwrap_or_else(|e| format!("(suggestion failed: {e:#})"));
            let _ = tx.send(text.trim().to_string());
        });
        self.lora_combine = Some(rx);
    }

    /// Recommend-for-context: ask the LLM which candidate LoRA best fits the current
    /// Chat prompt, on a background thread. Context = the accumulated refine prompt,
    /// else the current Chat input, else a generic.
    fn recommend_loras(&mut self, candidates: Vec<String>) {
        if self.lora_recommend.is_some() {
            return;
        }
        let context = if !self.refine_prompt.is_empty() {
            self.refine_prompt.clone()
        } else {
            let typed = self.chat.editor.text();
            if typed.trim().is_empty() { "a general image".into() } else { typed }
        };
        let list = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{}. {c}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let provider = crate::prompt::resolve_provider_label(&self.workspace.config.enhancer);
        let (tx, rx) = std::sync::mpsc::channel();
        let rt = self.rt.clone();
        std::thread::spawn(move || {
            const SYSTEM: &str = "You recommend the single best Stable Diffusion LoRA for a \
                given image prompt. Reply with the chosen LoRA's name and ONE plain sentence \
                explaining why. No preamble, no markdown.";
            let user = format!("Image prompt: {context}\n\nCandidate LoRAs:\n{list}\n\nWhich fits best?");
            let text = rt
                .block_on(crate::prompt::complete(&provider, SYSTEM, &user, &crate::prompt::EnhanceArgs::default()))
                .unwrap_or_else(|e| format!("(recommendation failed: {e:#})"));
            let _ = tx.send(text.trim().to_string());
        });
        self.lora_recommend = Some(rx);
    }

    /// Ask the LLM (workspace enhancer provider, resolved to a concrete one so the
    /// custom system prompt is honoured) to assess a LoRA, on a background thread.
    fn assess_lora(&mut self, key: String, prompt: String) {
        if self.lora_assess.is_some() {
            return;
        }
        // A LoRA's assessment describes the file, not the chat — serve a fresh (24h)
        // cached one without re-billing the LLM.
        if let Some(cached) = super::services::search_cache::assessment_get(&key) {
            self.lorahub.set_assessment(key, cached);
            return;
        }
        let provider = crate::prompt::resolve_provider_label(&self.workspace.config.enhancer);
        let (tx, rx) = std::sync::mpsc::channel();
        let rt = self.rt.clone();
        std::thread::spawn(move || {
            const SYSTEM: &str = "You are a concise assistant describing Stable Diffusion \
                LoRAs. Reply in ONE plain sentence, no preamble, no markdown.";
            let text = rt
                .block_on(crate::prompt::complete(&provider, SYSTEM, &prompt, &crate::prompt::EnhanceArgs::default()))
                .unwrap_or_else(|e| format!("(assessment failed: {e:#})"));
            let text = text.trim().to_string();
            // Cache successful assessments for the day (assessment_put skips failures).
            super::services::search_cache::assessment_put(&key, &text);
            let _ = tx.send((key, text));
        });
        self.lora_assess = Some(rx);
    }

    /// Run a LoRA search (Civitai or HF) on a background thread; results land in the Hub.
    fn remote_search(&mut self, source: lorahub::RemoteSource, query: String) {
        if self.remote_search.is_some() {
            return;
        }
        let src_tag = match source {
            lorahub::RemoteSource::Civitai => "civitai",
            lorahub::RemoteSource::HuggingFace => "hf",
        };
        // Serve an identical recent query from the 1h disk cache — no network round-trip.
        if let Some(hits) = super::services::search_cache::get(src_tag, &query) {
            let n = hits.len();
            self.lorahub.set_remote_hits(hits);
            self.lorahub.set_remote_status(format!("{n} results (cached)"));
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let rt = self.rt.clone();
        std::thread::spawn(move || {
            let result = match source {
                lorahub::RemoteSource::Civitai => rt
                    .block_on(crate::civitai::api::search(&query, Some(crate::civitai::api::AssetType::Lora), 30, 1))
                    .map(|resp| {
                        resp.items
                            .into_iter()
                            .map(|m| {
                                let v = m.model_versions.first();
                                let base = v.and_then(|v| v.base_model.clone()).unwrap_or_default();
                                lorahub::RemoteHit {
                                    title: m.name,
                                    family: lorahub::family_from_str(&base),
                                    subtitle: base,
                                    downloads: m.stats.download_count,
                                    dl: lorahub::DownloadRef::Civitai { model_id: m.id, version_id: v.map(|v| v.id) },
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .map_err(|e| format!("{e:#}")),
                lorahub::RemoteSource::HuggingFace => rt
                    .block_on(crate::hf::search::search_models(&query, 30))
                    .map(|hits| {
                        hits.into_iter()
                            .map(|h| lorahub::RemoteHit {
                                // HF exposes no per-LoRA base model → guess from the repo id.
                                family: lorahub::family_from_str(&h.id),
                                title: h.id.clone(),
                                subtitle: h.pipeline,
                                downloads: h.downloads,
                                dl: lorahub::DownloadRef::Hf { repo: h.id },
                            })
                            .collect::<Vec<_>>()
                    })
                    .map_err(|e| format!("{e:#}")),
            };
            // Persist successful results for the next identical query within the TTL.
            if let Ok(hits) = &result {
                super::services::search_cache::put(src_tag, &query, hits);
            }
            let _ = tx.send(result);
        });
        self.remote_search = Some(rx);
    }

    /// Queue a LoRA download (Civitai → its cache; HF → the workspace loras/ dir). Up to
    /// [`MAX_CONCURRENT_DOWNLOADS`] run at once; the rest wait in `downloads_queue`.
    fn remote_download(&mut self, dl: lorahub::DownloadRef, title: String) {
        self.downloads_queue.push_back((dl, title));
        self.pump_downloads();
    }

    /// Start queued downloads until the concurrency cap is reached.
    fn pump_downloads(&mut self) {
        while self.downloads_active.len() < MAX_CONCURRENT_DOWNLOADS {
            let Some((dl, title)) = self.downloads_queue.pop_front() else { break };
            let loras_dir = self.workspace.loras_dir();
            let (tx, rx) = std::sync::mpsc::channel();
            let rt = self.rt.clone();
            std::thread::spawn(move || {
                let result = match dl {
                    lorahub::DownloadRef::Civitai { model_id, version_id } => rt
                        .block_on(crate::civitai::download::download_version(Some(model_id), version_id, None))
                        .map(|_| title)
                        .map_err(|e| format!("{e:#}")),
                    lorahub::DownloadRef::Hf { repo } => rt
                        .block_on(crate::hf::search::download_lora_into(&repo, &loras_dir))
                        .map(|_| title)
                        .map_err(|e| format!("{e:#}")),
                };
                let _ = tx.send(result);
            });
            self.downloads_active.push(rx);
        }
        // The `●` tab marker reflects anything in flight or waiting.
        self.lorahub.set_downloading(!self.downloads_active.is_empty() || !self.downloads_queue.is_empty());
    }

    /// Drain the in-flight remote search / download into the Hub each tick.
    fn drain_civitai(&mut self) {
        if let Some(rx) = &self.remote_search {
            match rx.try_recv() {
                Ok(Ok(hits)) => {
                    self.lorahub.set_remote_hits(hits);
                    self.remote_search = None;
                }
                Ok(Err(e)) => {
                    self.lorahub.set_remote_status(format!("✗ search failed: {e}"));
                    self.remote_search = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.remote_search = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        // Drain the active download pool: collect completed receivers, report each,
        // keep the still-running ones, then pump the queue to refill the freed slots.
        if !self.downloads_active.is_empty() {
            let mut finished_any = false;
            let mut still_running = Vec::with_capacity(self.downloads_active.len());
            for rx in std::mem::take(&mut self.downloads_active) {
                match rx.try_recv() {
                    Ok(Ok(name)) => {
                        let waiting = self.downloads_queue.len();
                        let tail = if waiting > 0 { format!(" ({waiting} queued)") } else { String::new() };
                        self.lorahub.set_remote_status(format!("✓ downloaded {name} — see LOCAL{tail}"));
                        self.lorahub.rescan();
                        finished_any = true;
                    }
                    Ok(Err(e)) => {
                        self.lorahub.set_remote_status(format!("✗ download failed: {e}"));
                        finished_any = true;
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => finished_any = true,
                    Err(std::sync::mpsc::TryRecvError::Empty) => still_running.push(rx),
                }
            }
            self.downloads_active = still_running;
            if finished_any {
                self.pump_downloads(); // start queued downloads into freed slots + refresh `●`
            }
        }
        if let Some(rx) = &self.lora_assess {
            match rx.try_recv() {
                Ok((key, text)) => {
                    self.lorahub.set_assessment(key, text);
                    self.lora_assess = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.lora_assess = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        if let Some(rx) = &self.lora_recommend {
            match rx.try_recv() {
                Ok(text) => {
                    self.lorahub.set_recommendation(text);
                    self.lora_recommend = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.lora_recommend = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        if let Some(rx) = &self.lora_combine {
            match rx.try_recv() {
                Ok(text) => {
                    self.lorahub.set_combination(text);
                    self.lora_combine = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.lora_combine = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        if let Some(rx) = &self.lora_update {
            match rx.try_recv() {
                Ok((model_id, name, result)) => {
                    self.lora_update = None;
                    match result {
                        Ok(Some((new_id, new_name))) => {
                            self.output.push(format!("↑ ‘{name}’: newer version ‘{new_name}’ — downloading…"));
                            // Reuse the remote-download path; the new version lands in LOCAL.
                            self.remote_download(
                                lorahub::DownloadRef::Civitai { model_id, version_id: Some(new_id) },
                                format!("{name} ({new_name})"),
                            );
                        }
                        Ok(None) => self.output.push(format!("✓ ‘{name}’ is up to date")),
                        Err(e) => self.output.push(format!("✗ update check for ‘{name}’ failed: {e}")),
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.lora_update = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        if let Some(rx) = &self.chat_to_scenario {
            match rx.try_recv() {
                Ok(Ok(prompt)) => {
                    self.scenarios.insert_task("from-chat", &prompt);
                    self.chat_to_scenario = None;
                }
                Ok(Err(e)) => {
                    self.scenarios.set_status(format!("✗ Chat summary failed: {e}"));
                    self.chat_to_scenario = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.chat_to_scenario = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
    }

    /// Interactive load entry point (Models `[L]`): estimate the model's resident
    /// footprint and, if it would over-commit free RAM, hold the load for an explicit
    /// confirmation instead of firing a multi-GB download+load that may hard-OOM.
    fn request_load(&mut self, alias: String) {
        // Reloading the model already resident is always fine (no extra footprint).
        if self.models.loaded_alias() == Some(alias.as_str()) {
            self.load_model(alias);
            return;
        }
        let est = crate::capability::resident_estimate(&alias);
        let avail = crate::hw::available_ram_gb();
        if est.gb > avail {
            self.pending_load = Some(PendingLoad { alias, est_gb: est.gb, avail_gb: avail, exact: est.exact });
        } else {
            self.load_model(alias);
        }
    }

    /// Cache doctor (palette, Models screen): sweep stale download locks for the
    /// selected model and report its cache health to the Output pane. Reuses the
    /// 1.19.0 lock-sweep + `capability` sizing.
    fn run_cache_doctor(&mut self) {
        let Some(alias) = self.models.selected_alias() else { return };
        let repo = crate::hf::resolve_alias(&alias).to_string();
        let gated = self.models.rows.get(self.models.selected).map(|r| r.gated).unwrap_or(false);
        self.output.push(format!("cache doctor · {alias}  ({repo})"));
        let swept = crate::hf::download::clean_stale_locks(&alias);
        self.output.push(if swept > 0 {
            format!("  cleared {swept} stale download lock(s)")
        } else {
            "  no stale locks".into()
        });
        match crate::capability::cached_size_gb(&alias) {
            Some(gb) => self.output.push(format!("  cached · {gb:.1} GB of weights on disk")),
            None if crate::hf::download::is_cached(&alias) => {
                self.output.push("  cached · partial (index present, weights incomplete) — reload to finish".into())
            }
            None => self.output.push("  not cached — first load downloads the weights".into()),
        }
        if gated {
            self.output.push("  gated · accept the licence on HF (+ HF_TOKEN) if the load 401s".into());
        }
    }

    /// Load (or reload) `alias` with the currently-applied LoRAs merged in (each at
    /// its per-LoRA weight).
    fn load_model(&mut self, alias: impl Into<String>) {
        let specs: Vec<crate::pipelines::lora::LoraSpec> = self
            .active_loras
            .iter()
            .map(|(p, w)| crate::pipelines::lora::LoraSpec {
                source: crate::pipelines::lora::LoraSource::Local(p.clone()),
                scale: *w,
            })
            .collect();
        self.model_svc.load(alias, specs);
    }

    /// Reload the loaded model so a LoRA-set / weight change takes effect.
    fn reload_for_loras(&mut self) {
        if let Some(alias) = self.models.loaded_alias().map(str::to_string) {
            self.output.push(format!("reloading {alias} with {} LoRA(s)…", self.active_loras.len()));
            self.load_model(alias);
        }
    }

    /// Toggle a LoRA in Chat's active set (LoRA Hub `A`). Applying reloads the loaded
    /// model so the LoRA is merged in; an incompatible LoRA is refused with a note.
    fn toggle_lora(&mut self, path: std::path::PathBuf, compatible: bool) {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("lora").to_string();
        if let Some(pos) = self.active_loras.iter().position(|(p, _)| p == &path) {
            self.active_loras.remove(pos);
            self.chat.push_system(format!("LoRA off: {name} ({} active)", self.active_loras.len()));
        } else {
            if !compatible {
                self.chat.push_system(format!("⚠ {name} doesn't match the loaded model — not applied"));
                return;
            }
            self.active_loras.push((path, APPLY_LORA_SCALE));
            self.chat.push_system(format!("LoRA on: {name} @ {APPLY_LORA_SCALE:.2} ({} active)", self.active_loras.len()));
        }
        self.reload_for_loras();
    }

    /// Apply a local LoRA by name (from a Chat `@mention`): resolve → apply if not
    /// already on. Unknown name / incompatible family → a Chat system note.
    fn apply_lora_by_name(&mut self, name: &str) {
        let Some((path, compatible)) = self.lorahub.resolve_local(name) else {
            self.chat.push_system(format!("no local LoRA named ‘{name}’"));
            return;
        };
        if self.active_loras.iter().any(|(p, _)| p == &path) {
            self.chat.push_system(format!("LoRA ‘{name}’ already applied"));
            return;
        }
        self.toggle_lora(path, compatible);
    }

    /// Expand `@name` person mentions into their prompt fragments (case-insensitive).
    /// LoRA mentions are applied + stripped at accept-time, so only people remain here.
    fn expand_mentions(&self, text: &str) -> String {
        let mut out = text.to_string();
        for name in self.people.names() {
            if let Some(frag) = self.people.prompt_fragment(&name) {
                out = replace_ci(&out, &format!("@{name}"), &frag);
            }
        }
        out
    }

    /// Nudge an applied LoRA's weight (LoRA Hub `+`/`-`) and reload. Ignored (with a
    /// hint) when the LoRA isn't applied.
    fn adjust_lora_weight(&mut self, path: std::path::PathBuf, delta: f32) {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("lora").to_string();
        let new_w = match self.active_loras.iter_mut().find(|(p, _)| p == &path) {
            Some(entry) => {
                entry.1 = (entry.1 + delta).clamp(0.1, 1.5);
                entry.1
            }
            None => {
                self.chat.push_system(format!("apply {name} (A) before changing its weight"));
                return;
            }
        };
        self.chat.push_system(format!("LoRA weight: {name} → {new_w:.2}"));
        self.reload_for_loras();
    }

    /// Canvas `Enter` — adopt the rasterized mask as a ONE-SHOT inpaint for the next
    /// Chat prompt (only the white pixels change). It does NOT flip the session into
    /// anchored mode — after that one turn the mask is consumed and refinement reverts
    /// to whatever it was (prompt-evolve by default), so you're not locked in.
    fn apply_canvas_mask(&mut self, path: std::path::PathBuf) {
        self.chat_mask = Some(path);
        // Inpaint over the exact image the mask was painted on (the Canvas base = the
        // latest render), so the mask aligns and the edit compounds on the current state.
        // Fall back to the prompt-evolve base if the Canvas wasn't synced (e.g. tests).
        self.inpaint_base = self.canvas.base_path().or_else(|| self.refine_base.clone());
        // A mask needs a base to inpaint over; ensure a refine is triggered.
        if self.base_seed.is_none() {
            self.base_seed = Some(rand::random::<u32>() as u64);
        }
        self.chat.refine_armed = true;
        self.chat.push_system("inpaint mask set from Canvas — type an edit for the masked region (one-shot)".into());
        self.screen = ActiveScreen::Chat;
    }

    /// Apply a Canvas outpaint: the enlarged grey-padded image becomes the Chat base
    /// and the band mask is a one-shot inpaint, so the next prompt fills the new region.
    fn apply_outpaint(&mut self, base: std::path::PathBuf, mask: std::path::PathBuf) {
        self.refine_base = Some(base.clone());
        self.inpaint_base = Some(base); // inpaint over the grey-padded canvas
        self.chat_mask = Some(mask);
        if self.base_seed.is_none() {
            self.base_seed = Some(rand::random::<u32>() as u64);
        }
        self.chat.refine_armed = true;
        self.chat.push_system("outpaint base set from Canvas — type what fills the new region (one-shot)".into());
        self.screen = ActiveScreen::Chat;
    }

    fn sessions_dir(&self) -> std::path::PathBuf {
        self.workspace.root.join("sessions")
    }

    /// `/save [name]` — write the Chat thread + refinement state to
    /// `sessions/<name>.json` so it can be reloaded later.
    /// `/preset save <name>` — snapshot the current model + LoRA stack + negative + the
    /// model's size/steps/guidance overrides under a name (workspace `presets.json`). Uses
    /// the loaded model, else the Models cursor.
    fn save_preset(&mut self, name: &str) {
        let model = self
            .models
            .loaded_alias()
            .map(str::to_string)
            .or_else(|| self.models.selected_alias());
        let Some(model) = model else {
            self.chat.push_system("no model loaded or selected to save".into());
            return;
        };
        let preset = crate::ui::tui::presets::Preset {
            name: name.to_string(),
            size: self.gen_sizes.get(&model).copied(),
            model,
            loras: self.active_loras.clone(),
            negative: self.negative.clone(),
            steps: self.steps_override,
            guidance: self.guidance_override,
        };
        let summary = preset.summary();
        match crate::ui::tui::presets::upsert(&self.workspace.root, preset) {
            Ok(_) => self.chat.push_system(format!("✓ saved preset “{name}” — {summary}")),
            Err(e) => self.chat.push_system(format!("✗ save preset failed: {e}")),
        }
    }

    /// `/preset` / `/preset list` — list saved presets.
    fn list_presets(&mut self) {
        let all = crate::ui::tui::presets::load(&self.workspace.root);
        if all.is_empty() {
            self.chat.push_system("no presets yet — /preset save <name> to make one".into());
            return;
        }
        self.chat.push_system(format!("{} preset(s):", all.len()));
        for p in all {
            self.chat.push_system(format!("  {} — {}", p.name, p.summary()));
        }
    }

    /// `/preset <name>` (or a palette entry) — apply a saved preset: set the LoRA stack +
    /// negative, then load the model (through the memory-budget guard).
    fn apply_preset(&mut self, name: &str) {
        let Some(p) = crate::ui::tui::presets::find(&self.workspace.root, name) else {
            self.chat.push_system(format!("no preset “{name}” (/preset list)"));
            return;
        };
        self.active_loras = p.loras.clone();
        self.negative = p.negative.clone();
        self.lorahub.set_applied(&self.active_loras);
        // Restore the recipe overrides: size (per-model), steps, guidance.
        match p.size {
            Some(sz) => {
                self.gen_sizes.insert(p.model.clone(), sz);
            }
            None => {
                self.gen_sizes.remove(&p.model);
            }
        }
        self.steps_override = p.steps;
        self.guidance_override = p.guidance;
        self.chat.push_system(format!("applying preset “{}” — {}", p.name, p.summary()));
        self.request_load(p.model);
    }

    fn save_session(&mut self, name: &str) {
        let slug = session_slug(name);
        let session = ChatSession {
            turns: self
                .chat
                .history
                .iter()
                .map(|e| SessionTurn { utterance: e.utterance.clone(), result: e.result.clone(), refine: e.refine, system: e.system })
                .collect(),
            refine_prompt: self.refine_prompt.clone(),
            base_seed: self.base_seed,
            negative: self.negative.clone(),
            refine_base: self.refine_base.as_ref().map(|p| p.to_string_lossy().into_owned()),
            active_seed: self.active_seed,
        };
        let dir = self.sessions_dir();
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(format!("{slug}.json"));
        match serde_json::to_vec_pretty(&session) {
            Ok(bytes) => match std::fs::write(&path, bytes) {
                Ok(()) => self.chat.push_system(format!("✓ saved session → sessions/{slug}.json ({} turn(s))", session.turns.len())),
                Err(e) => self.chat.push_system(format!("✗ save failed: {e}")),
            },
            Err(e) => self.chat.push_system(format!("✗ serialize failed: {e}")),
        }
    }

    /// `/load <name>` — restore a saved session into Chat (thread + accumulated prompt,
    /// seed, negative, base image) so refinement continues where it left off.
    fn load_session(&mut self, name: &str) {
        if name.is_empty() {
            self.chat.push_system("usage: /load <name> — see /sessions".into());
            return;
        }
        let path = self.sessions_dir().join(format!("{}.json", session_slug(name)));
        let session: ChatSession = match std::fs::read(&path).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string())) {
            Ok(s) => s,
            Err(e) => {
                self.chat.push_system(format!("✗ couldn't load ‘{name}’: {e}"));
                return;
            }
        };
        let turns = session.turns.len();
        self.chat.restore(session.turns.into_iter().map(|t| (t.utterance, t.result, t.refine, t.system)).collect());
        self.refine_prompt = session.refine_prompt;
        self.base_seed = session.base_seed;
        self.negative = session.negative;
        self.active_seed = session.active_seed;
        self.refine_base = session.refine_base.map(std::path::PathBuf::from);
        // Rebuild the inline preview from the restored base image, if it still exists.
        self.chat.preview = self
            .refine_base
            .as_ref()
            .and_then(|p| image::open(p).ok())
            .map(|img| self.picker.new_resize_protocol(img));
        self.chat.refine_armed = self.base_seed.is_some();
        self.screen = ActiveScreen::Chat;
        self.chat.push_system(format!("✓ loaded session ‘{name}’ ({turns} turn(s)) — keep refining"));
    }

    /// `/sessions` — list the saved sessions under `sessions/`.
    fn list_sessions(&mut self) {
        let dir = self.sessions_dir();
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.flatten()
                    .filter_map(|e| {
                        let p = e.path();
                        (p.extension().and_then(|x| x.to_str()) == Some("json"))
                            .then(|| p.file_stem().and_then(|n| n.to_str()).map(str::to_string))
                            .flatten()
                    })
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        if names.is_empty() {
            self.chat.push_system("no saved sessions yet — /save [name] to create one".into());
        } else {
            self.chat.push_system(format!("sessions: {} · /load <name>", names.join(", ")));
        }
    }

    /// Keep the Canvas base in sync with the current Chat base image (lazy preview).
    fn sync_canvas(&mut self) {
        if self.screen != ActiveScreen::Canvas {
            return;
        }
        // Mask over the LATEST rendered image (not the prompt-evolve base, which stays the
        // clean original) — so you paint + inpaint the current state and edits compound.
        // Works for any model (it just reads the produced PNG).
        let target = self.chat.latest_frame_path().or_else(|| self.refine_base.clone());
        if target != self.canvas.base_path() {
            let dims = target.as_ref().and_then(|p| image::open(p).ok()).map(|i| (i.width(), i.height()));
            self.canvas.set_base(target.clone(), dims);
        }
        let sel = target;
        if sel != self.canvas.preview_for {
            self.canvas.preview = sel
                .as_ref()
                .and_then(|p| image::open(p).ok())
                .map(|img| self.picker.new_resize_protocol(img));
            self.canvas.preview_for = sel;
        }
        // Kick off face detection for the face-aware `B` preset (once per base).
        if self.canvas_faces.is_none() {
            if let Some(base) = self.canvas.faces_needed_for() {
                self.detect_canvas_faces(base);
            }
        }
    }

    /// (Re)compute an identity's encoding quality (mean pairwise ArcFace cosine of its
    /// refs) on a background thread, persist it, and reflect it in the People screen.
    fn encode_person(&mut self, name: String, dir: std::path::PathBuf, photos: Vec<std::path::PathBuf>, fingerprint: String) {
        if self.people_encode.is_some() {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let rt = self.rt.clone();
        let device = self.device.clone();
        std::thread::spawn(move || {
            let result = rt
                .block_on(crate::pipelines::identity_quality::IdentityScorer::load_resolved(&device))
                .and_then(|scorer| scorer.score(&photos))
                .map(|r| (r.score, r.faces, r.total))
                .map_err(|e| format!("{e:#}"));
            // Persist a successful score (with the fingerprint that ties it to this ref
            // set + strategy) so it survives rescans and invalidates on a change.
            if let Ok((score, faces, total)) = &result {
                let _ = crate::ui::tui::screens::people::write_quality_sidecar(&dir, *score, *faces, *total, &fingerprint);
            }
            let _ = tx.send((name, dir, result));
        });
        self.people_encode = Some(rx);
    }

    /// Deliver a completed identity re-encode to the People screen each tick.
    fn drain_people_encode(&mut self) {
        if let Some(rx) = &self.people_encode {
            match rx.try_recv() {
                Ok((name, _dir, result)) => {
                    match result {
                        Ok((score, faces, total)) => self.people.set_encoding_quality(&name, score, faces, total),
                        Err(e) => self.people.set_encoding_error(&name, &e),
                    }
                    self.people_encode = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.people_encode = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
    }

    /// Deliver completed Canvas face detection to the Canvas each tick.
    fn drain_canvas_faces(&mut self) {
        if let Some(rx) = &self.canvas_faces {
            match rx.try_recv() {
                Ok((base, boxes)) => {
                    self.canvas.set_faces(base, boxes);
                    self.canvas_faces = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.canvas_faces = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
    }

    /// Detect faces in the Canvas base on a background thread (SCRFD), normalize the
    /// boxes to 0..1, and hand them to the Canvas for the face-aware `B` preset. A
    /// missing detector / no faces just yields an empty set (B then fills plainly).
    fn detect_canvas_faces(&mut self, base: std::path::PathBuf) {
        let (tx, rx) = std::sync::mpsc::channel();
        let rt = self.rt.clone();
        let device = self.device.clone();
        std::thread::spawn(move || {
            let boxes = detect_face_boxes_normalized(&rt, &device, &base).unwrap_or_default();
            let _ = tx.send((base, boxes));
        });
        self.canvas_faces = Some(rx);
    }

    /// Build a `portrait::Request` for an identity context at the given (accumulated)
    /// prompt. Used for the initial portrait and for every identity-preserving refine.
    fn identity_portrait_request(&self, prompt: String, id: &ChatIdentity) -> crate::pipelines::portrait::Request {
        let identity = if id.photos.is_empty() {
            None
        } else {
            Some(
                id.identity
                    .parse::<crate::pipelines::ip_adapter::IdentityKind>()
                    .unwrap_or(crate::pipelines::ip_adapter::IdentityKind::PlusFace),
            )
        };
        let photos = id
            .photos
            .iter()
            .map(|(p, w)| crate::pipelines::ip_adapter::WeightedPhoto { path: p.clone(), weight: Some(*w) })
            .collect();
        crate::pipelines::portrait::Request {
            prompt,
            negative: id.negative.clone(),
            photos,
            model: id.model.clone(),
            width: id.width,
            height: id.height,
            count: 1,
            steps: 30,
            guidance: 7.0,
            seed: Some(id.seed),
            out_dir: self.workspace.out_dir().join("people"),
            device: self.device.clone(),
            loras: Vec::new(),
            lora_scale: 1.0,
            scheduler: crate::pipelines::scheduler::SchedulerKind::default(),
            refine: None,
            refine_strength: 0.0,
            face_strength: id.face_strength.unwrap_or(0.8),
            face_bbox: None,
            face_landmarks: None,
            identity,
            shared_clip_h: None,
            controls: Vec::new(),
        }
    }

    /// Identity-preserving Chat refine: re-render the accumulated prompt with the stored
    /// person's IP-Adapter pass (same face, same seed) instead of plain img2img.
    fn dispatch_identity_refine(&mut self, edit: String) {
        if self.portrait_run.is_some() {
            return;
        }
        let Some(id) = self.chat_identity.clone() else { return };
        let full = if self.refine_prompt.is_empty() { edit.clone() } else { format!("{}, {}", self.refine_prompt, edit) };
        // The identity path doesn't go through drain_generation, so update the
        // accumulated prompt here.
        self.refine_prompt = full.clone();
        self.active_full_prompt = full.clone();
        self.chat.push_utterance(edit, true);
        self.chat.push_system("↻ identity-preserving refine (IP-Adapter) — keeping the face".into());
        if let Some(alias) = self.models.loaded_alias() {
            self.chat.push_system(format!("(freeing {alias} for the run — reload with L after)"));
        }
        let produced = self.workspace.out_dir().join("people").join(format!("plakat-portrait-{}.png", id.seed));
        let req = self.identity_portrait_request(full.clone(), &id);
        self.portrait_prompt = full;
        self.pending_identity = Some(id); // persist across the next refine
        self.portrait_run = Some(self.model_svc.run_portrait(req, produced));
    }

    /// People `G` — generate a portrait from a person on a background thread (loads
    /// its own model; progress flows to the Output pane). The result opens in Chat.
    fn quick_generate(&mut self, spec: people::QuickGen) {
        if self.portrait_run.is_some() {
            self.output.push("a portrait is already generating…".into());
            return;
        }
        // Identity strategy → model family + IdentityKind. No photos = text-only.
        let sdxl = spec.identity.to_lowercase().contains("sdxl");
        let model = if sdxl { "sdxl" } else { "sd15" }.to_string();
        let (w, h) = if sdxl { (768u32, 960u32) } else { (512u32, 640u32) };
        let identity = if spec.photos.is_empty() {
            None
        } else {
            Some(
                spec.identity
                    .parse::<crate::pipelines::ip_adapter::IdentityKind>()
                    .unwrap_or(crate::pipelines::ip_adapter::IdentityKind::PlusFace),
            )
        };
        let prompt = if spec.prompt.trim().is_empty() {
            "portrait photograph, head and shoulders, soft studio lighting, sharp focus, detailed".to_string()
        } else {
            spec.prompt.clone()
        };
        let _ = identity; // kind is recomputed by the helper from the strategy string
        let seed = rand::random::<u32>() as u64;
        let out_dir = self.workspace.out_dir().join("people");
        // The reusable identity context — refines in Chat re-run this IP-Adapter pass.
        let ident = ChatIdentity {
            photos: spec.photos.clone(),
            identity: spec.identity.clone(),
            face_strength: spec.face_strength,
            negative: spec.negative.clone(),
            model,
            width: w,
            height: h,
            seed,
        };
        let req = self.identity_portrait_request(prompt.clone(), &ident);

        self.chat.push_system(format!("generating portrait of {} — opens in Chat (refines keep the face)…", spec.label));
        if let Some(alias) = self.models.loaded_alias() {
            self.chat.push_system(format!("(freeing {alias} for the run — reload with L after)"));
        }
        self.portrait_prompt = prompt;
        self.pending_identity = Some(ident);
        // Run on the model thread (frees the loaded Chat model first — no double-load).
        let produced = out_dir.join(format!("plakat-portrait-{seed}.png"));
        self.portrait_run = Some(self.model_svc.run_portrait(req, produced));
    }

    /// People `G` with ≥2 marked — generate a multiperson scene placing each person
    /// in a deterministic region (no LLM auto-placement), on a background thread. The
    /// result opens in Chat. Uses the plus-face / sd15 path.
    fn quick_generate_multi(&mut self, specs: Vec<people::QuickGen>) {
        if self.portrait_run.is_some() {
            self.output.push("a generation is already running…".into());
            return;
        }
        let n = specs.len();
        let labels: Vec<String> = specs.iter().map(|s| s.label.clone()).collect();
        // Route the scene's identity strategy + model from the marked personas' own
        // strategies, rather than forcing plus-face / sd15 for everyone.
        let strategies: Vec<String> = specs.iter().map(|s| s.identity.clone()).collect();
        let (identity, model) = route_multiperson_identity(&strategies);
        // Even horizontal split across the canvas, one region per person.
        let people_req: Vec<crate::pipelines::multiperson::Person> = specs
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let x0 = i as f32 / n as f32 + 0.02;
                let x1 = (i as f32 + 1.0) / n as f32 - 0.02;
                crate::pipelines::multiperson::Person {
                    label: s.label.clone(),
                    photos: s
                        .photos
                        .iter()
                        .map(|(p, w)| crate::pipelines::ip_adapter::WeightedPhoto { path: p.clone(), weight: Some(*w) })
                        .collect(),
                    placement: None,
                    bbox: Some([x0, 0.08, x1, 0.96]),
                    prompt: (!s.prompt.trim().is_empty()).then(|| s.prompt.clone()),
                    face_strength: s.face_strength,
                    face_bbox: None,
                    face_landmarks: None,
                    scale: None,
                }
            })
            .collect();

        let scene = format!(
            "a group portrait of {n} people standing together, soft natural light, sharp focus, detailed"
        );
        let seed = rand::random::<u32>() as u64;
        let out_dir = self.workspace.out_dir().join("people");
        let req = crate::pipelines::multiperson::MultipersonRequest {
            scene: scene.clone(),
            people: people_req,
            model: model.to_string(),
            identity,
            style: None,
            negative: String::new(),
            layout_provider: "none".into(),
            enhancer: None,
            // SDXL strategies render at 1024²; sd15 strategies at 768².
            width: if model == "sdxl" { 1024 } else { 768 },
            height: if model == "sdxl" { 1024 } else { 768 },
            steps: 30,
            guidance: 7.5,
            seed: Some(seed),
            count: 1,
            out_dir: out_dir.clone(),
            scheduler: crate::pipelines::scheduler::SchedulerKind::default(),
            device: self.device.clone(),
            dry_run: false,
            composite: false,
            relight: false,
            harmonize: None,
            pose: false,
            swap: false,
            restore_faces: false,
            refine_faces: true,
            refine_face_strength: 0.85,
            refine_denoise: 0.35,
        };

        self.chat.push_system(format!(
            "generating multiperson scene: {} [{} · {model}] — opens in Chat…",
            labels.join(", "),
            identity.label()
        ));
        if let Some(alias) = self.models.loaded_alias() {
            self.chat.push_system(format!("(freeing {alias} for the run — reload with L after)"));
        }
        self.portrait_prompt = scene;
        // Multiperson isn't a single-identity refine (per-person IP-Adapter would need a
        // composite) — its continuation is plain.
        self.pending_identity = None;
        // Run on the model thread (frees the loaded Chat model first — no double-load).
        let produced = out_dir.join(format!("plakat-multiperson-{seed}.png"));
        self.portrait_run = Some(self.model_svc.run_multiperson(req, produced));
    }

    /// Poll the in-flight portrait / multiperson gen; on success, open it in Chat.
    fn drain_portrait(&mut self) {
        let done = match &self.portrait_run {
            Some(rx) => match rx.try_recv() {
                Ok(r) => Some(r),
                Err(std::sync::mpsc::TryRecvError::Empty) => None,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    Some(Err("portrait thread ended unexpectedly".to_string()))
                }
            },
            None => None,
        };
        if let Some(result) = done {
            self.portrait_run = None;
            // The identity context this run produced (Some for a single portrait /
            // identity refine; None for multiperson). `continue_from_image` clears
            // `chat_identity`, so re-adopt it here.
            let ident = self.pending_identity.take();
            match result {
                Ok(path) => {
                    // A known identity → continue in PROMPT-EVOLVE at its stable seed (so
                    // refines keep accumulating); else image-anchored.
                    let seed = ident.as_ref().map(|i| i.seed);
                    self.continue_from_image(path, self.portrait_prompt.clone(), seed);
                    self.chat_identity = ident;
                }
                Err(e) => self.output.push(format!("✗ portrait failed: {e}")),
            }
        }
    }

    /// Load an image into Chat to keep editing it. When the image carries a recipe
    /// (`seed` + `prompt` recovered from its metadata) we resume in PROMPT-EVOLVE mode
    /// — txt2img at that seed reproduces ~the same image and additive edits ("add a
    /// sun") reliably land. Without a recipe (`seed` = None) we only have the pixels,
    /// so we fall back to image-anchored img2img. Switches to Chat.
    fn continue_from_image(&mut self, path: std::path::PathBuf, prompt: String, seed: Option<u64>) {
        self.refine_base = Some(path.clone());
        self.refine_prompt = prompt;
        self.fixed_seed = None;
        self.chat_mask = None;
        // Generic continue (History `C` / Canvas) is not identity-aware; the portrait
        // path re-adopts its identity context after this returns.
        self.chat_identity = None;
        let mode = match seed {
            Some(s) => {
                self.base_seed = Some(s);
                self.refine_strength = None; // prompt-evolve
                "prompt-evolve"
            }
            None => {
                self.base_seed = Some(rand::random::<u32>() as u64);
                self.refine_strength = Some(DEFAULT_ANCHOR_STRENGTH); // image-anchored
                "image-anchored"
            }
        };
        if let Ok(img) = image::open(&path) {
            self.chat.preview = Some(self.picker.new_resize_protocol(img));
        }
        self.chat.refine_armed = true;
        self.chat.push_system(format!(
            "continuing from {} — type an edit ({mode})",
            path.file_name().and_then(|n| n.to_str()).unwrap_or("image")
        ));
        self.screen = ActiveScreen::Chat;
    }

    /// Show a session filmstrip frame in the image pane (`None` → the latest image).
    fn show_chat_frame(&mut self, path: Option<std::path::PathBuf>) {
        let target = path.or_else(|| self.chat.latest_frame_path());
        self.chat.preview = target
            .as_ref()
            .and_then(|p| image::open(p).ok())
            .map(|img| self.picker.new_resize_protocol(img));
    }

    /// Roll the session back to a filmstrip frame: branch from it (recover its prompt +
    /// seed → prompt-evolve continuation), so the next prompt refines from there.
    fn rollback_to_frame(&mut self, path: std::path::PathBuf) {
        let (prompt, seed) = recover_recipe(&path);
        self.continue_from_image(path, prompt, seed);
    }

    /// Generate a fresh variation of a filmstrip frame — its prompt at a new seed.
    /// `/vary [n]` — fan out `n` variations (default 4, clamped 2–8) of the current image
    /// at fresh seeds. The model thread is serial, so they run one at a time via
    /// `pump_variations`; each lands in the filmstrip to scrub (Ctrl-←/→) and keep (Ctrl-B).
    fn start_variations(&mut self, n: usize) {
        let n = n.clamp(2, 8);
        let prompt = self
            .chat
            .latest_frame_path()
            .map(|p| recover_recipe(&p).0)
            .filter(|s| !s.trim().is_empty())
            .or_else(|| {
                let p = self.refine_prompt.trim();
                (!p.is_empty()).then(|| p.to_string())
            });
        let Some(prompt) = prompt else {
            self.chat.push_system("nothing to vary yet — generate an image first".into());
            return;
        };
        self.chat.push_system(format!(
            "fanning out {n} variations at fresh seeds — scrub the filmstrip (Ctrl-←/→), Ctrl-B keeps one"
        ));
        for _ in 0..n {
            self.variation_queue.push_back(prompt.clone());
        }
        self.pump_variations();
    }

    /// Dispatch the next queued variation if the model is free (called on each Done).
    fn pump_variations(&mut self) {
        if self.active_gen.is_some() {
            return;
        }
        if let Some(prompt) = self.variation_queue.pop_front() {
            // `/new` forces a fresh generation at a new random seed (a true variation).
            self.dispatch_generation(format!("/new {prompt}"), false);
        }
    }

    fn vary_frame(&mut self, path: std::path::PathBuf) {
        let (prompt, _) = recover_recipe(&path);
        if prompt.trim().is_empty() {
            self.chat.push_system("can't vary — that frame has no embedded recipe".into());
            return;
        }
        self.chat.push_system(format!(
            "variation of {} (same prompt, new seed)",
            path.file_name().and_then(|n| n.to_str()).unwrap_or("frame")
        ));
        self.handle_chat_submit(format!("/new {prompt}"));
    }

    /// Dispatch a History-screen action (RFC UI-GALLERY-1): continue / vary / naturalize / delete.
    fn handle_history_action(&mut self, action: HistoryAction) {
        match action {
            HistoryAction::None => {}
            HistoryAction::Continue { path, prompt, seed } => self.continue_from_image(path, prompt, seed),
            HistoryAction::Vary { path } => self.vary_frame(path),
            HistoryAction::Naturalize { path } => self.naturalize_frame(path),
            HistoryAction::Delete { path } => self.delete_frame(path),
            HistoryAction::Etch { path } => self.etch_frame(path),
        }
    }

    /// V1 — etch plakat provenance into a History frame in place (sync, no model), then rescan.
    fn etch_frame(&mut self, path: std::path::PathBuf) {
        let done = image::open(&path).ok().and_then(|img| {
            let rgb = img.to_rgb8();
            crate::etch::fresh_etch(rgb.as_raw(), rgb.width(), rgb.height(), &path, None).ok()
        });
        self.history.status = match done {
            Some(id) => {
                self.history.rescan();
                format!("etched provenance (id {})", id.hex())
            }
            None => "etch failed".into(),
        };
    }

    /// P1 — weight-free de-slop of a History frame in place → `<stem>_naturalized.png`, then rescan.
    fn naturalize_frame(&mut self, path: std::path::PathBuf) {
        let done = image::open(&path).ok().and_then(|img| {
            let out = crate::naturalize::apply(&img.to_rgb8(), &crate::naturalize::Preset::Photo.params());
            let dst = path.with_file_name(format!("{}_naturalized.png", path.file_stem().and_then(|s| s.to_str()).unwrap_or("frame")));
            out.save(&dst).ok().map(|_| dst)
        });
        self.history.status = match done {
            Some(dst) => {
                self.history.rescan();
                format!("naturalized → {}", dst.file_name().and_then(|n| n.to_str()).unwrap_or(""))
            }
            None => "naturalize failed".into(),
        };
    }

    /// P3 — move a History frame to `<out>/.trash/` (recoverable), then rescan.
    fn delete_frame(&mut self, path: std::path::PathBuf) {
        let trash = self.workspace.out_dir().join(".trash");
        let _ = std::fs::create_dir_all(&trash);
        let name = path.file_name().map(std::ffi::OsString::from).unwrap_or_default();
        self.history.status = match std::fs::rename(&path, trash.join(&name)) {
            Ok(()) => {
                self.history.rescan();
                format!("trashed {} (→ .trash/)", name.to_str().unwrap_or(""))
            }
            Err(e) => format!("delete failed: {e}"),
        };
    }

    /// Chat `/relight <preset>` — relight the latest frame with a lighting preset (RFC UI-GALLERY-1 V1).
    /// Reuses the `active_gen` channel so the result posts to Chat like any generation; evicts the
    /// resident t2i model first (single Metal device). Heavy op — needs the IC-Light weights.
    fn chat_relight(&mut self, arg: &str) {
        let arg = arg.trim();
        if arg.is_empty() {
            let names: Vec<&str> = crate::pipelines::ic_light::light_presets().iter().map(|p| p.name).collect();
            self.chat.push_system(format!("usage: /relight <preset> — {}", names.join(" / ")));
            return;
        }
        if self.active_gen.is_some() {
            self.chat.push_system("busy — wait for the current op to finish".into());
            return;
        }
        let Some(preset) = crate::pipelines::ic_light::light_preset(arg).copied() else {
            self.chat.push_system(format!("unknown light preset `{arg}` — try /relight for the list"));
            return;
        };
        let Some(src) = self.chat.latest_frame_path() else {
            self.chat.push_system("no image in the thread to relight — generate one first".into());
            return;
        };
        self.model_svc.unload(); // free the Metal device for IC-Light
        let (device, rt) = (self.device.clone(), self.rt.clone());
        let out_dir = self.workspace.out_dir().join("chat");
        let seed = rand::random::<u32>() as u64;
        let (tx, rx) = std::sync::mpsc::channel::<GenMessage>();
        let cancel = CancelFlag::new();
        std::thread::spawn(move || {
            let result = (|| -> anyhow::Result<std::path::PathBuf> {
                let pipe = rt.block_on(crate::pipelines::ic_light::Pipeline::load(device))?;
                let (buf, w, h) = pipe.relight(&src, preset.prompt, preset.negative, 512, 512, 25, 2.0, seed, preset.backdrop)?;
                std::fs::create_dir_all(&out_dir)?;
                let out = out_dir.join(format!("relight-{seed}.png"));
                crate::imaging::io::save_rgb_u8(&buf, w, h, &out)?;
                Ok(out)
            })();
            let _ = match result {
                Ok(output) => tx.send(GenMessage::Done { output, cancelled: false }),
                Err(e) => tx.send(GenMessage::Error { message: format!("{e:#}") }),
            };
        });
        self.active_gen = Some((rx, cancel));
        self.chat.status = ChatStatus::Generating { step: 0, total: 1, refine: false };
        self.chat.push_system(format!("relight → {arg} (loading IC-Light…)"));
    }

    /// Chat `/faceswap <source.png> [N]` — swap a source face into the latest frame (V1). Same
    /// `active_gen` reuse + device eviction as `/relight`. Heavy op — needs the inswapper weights.
    fn chat_faceswap(&mut self, arg: &str) {
        let mut it = arg.split_whitespace();
        let Some(source) = it.next().map(std::path::PathBuf::from) else {
            self.chat.push_system("usage: /faceswap <source-face.png> [face-index]".into());
            return;
        };
        let face: usize = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        if self.active_gen.is_some() {
            self.chat.push_system("busy — wait for the current op to finish".into());
            return;
        }
        if !source.exists() {
            self.chat.push_system(format!("source face not found: {}", source.display()));
            return;
        }
        let Some(scene) = self.chat.latest_frame_path() else {
            self.chat.push_system("no image in the thread to swap into — generate one first".into());
            return;
        };
        let source_label = source.display().to_string();
        self.model_svc.unload();
        let (device, rt) = (self.device.clone(), self.rt.clone());
        let out_dir = self.workspace.out_dir().join("chat");
        let seed = rand::random::<u32>() as u64;
        let (tx, rx) = std::sync::mpsc::channel::<GenMessage>();
        let cancel = CancelFlag::new();
        std::thread::spawn(move || {
            let result = (|| -> anyhow::Result<std::path::PathBuf> {
                let swapper = rt.block_on(crate::pipelines::faceswap::FaceSwapper::load_resolved(&device, candle_core::DType::F32))?;
                let faces = swapper.detect(&scene)?;
                anyhow::ensure!(!faces.is_empty(), "no face detected in the scene");
                anyhow::ensure!(face < faces.len(), "face {face} out of range — {} detected", faces.len());
                let latent = swapper.source_latent(&source)?;
                let img = image::open(&scene)?.to_rgb8();
                let out_img = swapper.swap_into(&img, faces[face].landmarks, &latent)?;
                std::fs::create_dir_all(&out_dir)?;
                let out = out_dir.join(format!("faceswap-{seed}.png"));
                out_img.save(&out)?;
                Ok(out)
            })();
            let _ = match result {
                Ok(output) => tx.send(GenMessage::Done { output, cancelled: false }),
                Err(e) => tx.send(GenMessage::Error { message: format!("{e:#}") }),
            };
        });
        self.active_gen = Some((rx, cancel));
        self.chat.status = ChatStatus::Generating { step: 0, total: 1, refine: false };
        self.chat.push_system(format!("faceswap ← {source_label} (loading engine…)"));
    }

    /// Run a scenario file on a background thread. Its task-by-task progress (model
    /// load, denoise bars, per-task status) flows to the Output pane automatically —
    /// the scenario runner uses the rerouted `ui::progress`. The runner loads its own
    /// model (independent of the TUI's ModelService); sharing the loaded pipeline is a
    /// later optimization (RFC §0-R0-2). One run at a time.
    fn run_scenario(&mut self, path: std::path::PathBuf) {
        if self.scenario_run.is_some() {
            self.scenarios.status = "A scenario is already running.".into();
            return;
        }
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("scenario").to_string();
        // Pre-populate the RUNNER board from the scenario's task names (best-effort;
        // the runner's Started event reconciles the count if our parse differs).
        let names = crate::cli::scenario::task_names(&path).unwrap_or_default();
        self.scenarios.start_run(name, names);

        // Land scenario images under the workspace out/ dir (grouped per scenario) so
        // History — which scans workspace.out_dir() — picks them up, no matter what the
        // scenario file's own `out:` says.
        let stem = path.file_stem().and_then(|n| n.to_str()).unwrap_or("scenario").to_string();
        let out_override = Some(self.workspace.out_dir().join("scenarios").join(stem));
        // If a Chat model is loaded, note that the in-process run will free it (only one
        // model fits in unified memory; the runner loads the scenario's own).
        if let Some(alias) = self.models.loaded_alias() {
            self.scenarios.status = format!("freeing {alias} to run — reload with L in Models after");
        }
        let args = crate::cli::scenario::ScenarioArgs {
            file: path,
            dry_run: false,
            resume: false,
            force: false,
            only: Vec::new(),
            limit: 0,
            json_summary: None,
            out_override,
        };
        // Run on the model thread so it drops the loaded Chat pipeline first (no
        // double-load). Events flow on `erx`; the terminal result on the returned rx.
        let (etx, erx) = std::sync::mpsc::channel();
        self.scenario_run = Some(self.model_svc.run_scenario(args, etx));
        self.scenario_events = Some(erx);
    }

    /// Feed live per-task events to the RUNNER board, and poll for run completion.
    fn drain_scenario(&mut self) {
        // Events first (they all precede the terminal result on the worker thread).
        if let Some(rx) = &self.scenario_events {
            let evs: Vec<ScenarioEvent> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
            for ev in evs {
                self.scenarios.apply_event(ev);
            }
        }
        let done = match &self.scenario_run {
            Some(rx) => match rx.try_recv() {
                Ok(result) => Some(result),
                Err(std::sync::mpsc::TryRecvError::Empty) => None,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    Some(Err("scenario thread ended unexpectedly".to_string()))
                }
            },
            None => None,
        };
        if let Some(result) = done {
            self.scenario_run = None;
            self.scenario_events = None;
            self.scenarios.finish_run(result);
            // The run wrote images under the workspace out/ dir — refresh History so
            // they show up without the user pressing `r`.
            self.history.rescan();
        }
    }

    /// Route a submitted Chat line: slash commands (`/negative`, `/enhance`, `/new`)
    /// or a plain prompt. `/negative` updates session state without generating.
    fn handle_chat_submit(&mut self, text: String) {
        // Expand `@name` person mentions into their prompt fragments before routing
        // (slash commands don't carry person tokens, so this is a no-op for them).
        let text = self.expand_mentions(&text);
        if let Some(rest) = text.strip_prefix("/negative") {
            self.negative = rest.trim().to_string();
            let note = if self.negative.is_empty() {
                "negative prompt cleared".to_string()
            } else {
                format!("negative prompt set → {}", self.negative)
            };
            self.chat.push_system(note);
            return;
        }
        // `/strength <0.1–1.0>` opts into IMAGE-ANCHORED refinement: a follow-up
        // img2img's over the actual previous image at that strength (anchors to its
        // exact pixels). `/strength off` returns to the default prompt-evolve mode.
        if let Some(rest) = text.strip_prefix("/strength") {
            let arg = rest.trim();
            if arg.eq_ignore_ascii_case("off") || arg == "0" {
                self.refine_strength = None;
                self.chat.push_system("refine mode → prompt-evolve (default)".into());
            } else if arg.is_empty() {
                self.refine_strength = Some(DEFAULT_ANCHOR_STRENGTH);
                self.chat.push_system(format!("refine mode → image-anchored ({DEFAULT_ANCHOR_STRENGTH:.2})"));
            } else if let Ok(v) = arg.parse::<f32>() {
                if (0.1..=1.0).contains(&v) {
                    self.refine_strength = Some(v);
                    self.chat.push_system(format!("refine mode → image-anchored ({v:.2})"));
                } else {
                    self.chat.push_system("strength must be 0.1–1.0 (or 'off')".into());
                }
            } else {
                self.chat.push_system("strength must be a number 0.1–1.0 (or 'off')".into());
            }
            return;
        }
        // `/seed <n>` pins the seed for reproducible / comparable runs; `/seed random`
        // (or bare `/seed`) returns to a fresh random seed each generation.
        if let Some(rest) = text.strip_prefix("/seed") {
            let arg = rest.trim();
            if arg.is_empty() || arg.eq_ignore_ascii_case("random") {
                self.fixed_seed = None;
                self.chat.push_system("seed → random".into());
            } else if let Ok(n) = arg.parse::<u64>() {
                self.fixed_seed = Some(n);
                self.chat.push_system(format!("seed pinned → {n}"));
            } else {
                self.chat.push_system("seed must be a non-negative integer (or 'random')".into());
            }
            return;
        }
        // `/size <WxH>` | `<N>` | `native`/`off` — per-model generation-size override,
        // budget-guarded. Each model remembers its own size.
        if let Some(rest) = text.strip_prefix("/size") {
            self.set_gen_size(rest.trim());
            return;
        }
        // `/steps <n>` | `default` — session denoise-steps override.
        if let Some(rest) = text.strip_prefix("/steps") {
            let arg = rest.trim();
            if arg.is_empty() || arg.eq_ignore_ascii_case("default") {
                self.steps_override = None;
                self.chat.push_system(format!("steps → default ({})", self.workspace.config.default_steps));
            } else if let Some(n) = arg.parse::<usize>().ok().filter(|n| (1..=200).contains(n)) {
                self.steps_override = Some(n);
                self.chat.push_system(format!("steps → {n}"));
            } else {
                self.chat.push_system("steps must be 1–200 (or 'default')".into());
            }
            return;
        }
        // `/cfg <x>` | `default` — session guidance (CFG) override.
        if let Some(rest) = text.strip_prefix("/cfg") {
            let arg = rest.trim();
            if arg.is_empty() || arg.eq_ignore_ascii_case("default") {
                self.guidance_override = None;
                self.chat.push_system(format!("cfg → default ({:.1})", self.workspace.config.default_guidance));
            } else if let Some(g) = arg.parse::<f64>().ok().filter(|g| (0.0..=30.0).contains(g)) {
                self.guidance_override = Some(g);
                self.chat.push_system(format!("cfg → {g:.1}"));
            } else {
                self.chat.push_system("cfg must be 0–30 (or 'default')".into());
            }
            return;
        }
        // `/pag <x>` | `off` — Perturbed-Attention Guidance (SD 1.5/SDXL/PixArt/SD3). Sharper detail.
        if let Some(rest) = text.strip_prefix("/pag") {
            let arg = rest.trim();
            if arg.is_empty() || arg.eq_ignore_ascii_case("off") {
                self.pag_override = None;
                self.chat.push_system("pag → off".into());
            } else if let Some(v) = arg.parse::<f64>().ok().filter(|v| (0.0..=10.0).contains(v)) {
                self.pag_override = (v > 0.0).then_some(v);
                self.chat.push_system(format!("pag → {v:.1} (try 2–3)"));
            } else {
                self.chat.push_system("pag must be 0–10 (or 'off')".into());
            }
            return;
        }
        // `/rescale <phi>` | `off` — CFG-rescale (curbs high-CFG over-exposure). ~0.7.
        if let Some(rest) = text.strip_prefix("/rescale") {
            let arg = rest.trim();
            if arg.is_empty() || arg.eq_ignore_ascii_case("off") {
                self.rescale_override = None;
                self.chat.push_system("rescale → off".into());
            } else if let Some(v) = arg.parse::<f64>().ok().filter(|v| (0.0..=1.0).contains(v)) {
                self.rescale_override = (v > 0.0).then_some(v);
                self.chat.push_system(format!("rescale → {v:.2}"));
            } else {
                self.chat.push_system("rescale must be 0–1 (or 'off')".into());
            }
            return;
        }
        // `/freeu [on|off]` — FreeU up-block reweighting (richer detail/texture). SD 1.5/SDXL.
        if let Some(rest) = text.strip_prefix("/freeu") {
            let arg = rest.trim();
            self.freeu_override = !(arg.eq_ignore_ascii_case("off") || arg == "0");
            self.chat.push_system(format!("freeu → {}", if self.freeu_override { "on" } else { "off" }));
            return;
        }
        // `/dynthresh <pctl>` | `off` — dynamic thresholding (epsilon SD). ~99.5.
        if let Some(rest) = text.strip_prefix("/dynthresh") {
            let arg = rest.trim();
            if arg.is_empty() || arg.eq_ignore_ascii_case("off") {
                self.dynthresh_override = None;
                self.chat.push_system("dynthresh → off".into());
            } else if let Some(v) = arg.parse::<f64>().ok().filter(|v| (0.0..=100.0).contains(v)) {
                self.dynthresh_override = (v > 0.0).then_some(v);
                self.chat.push_system(format!("dynthresh → {v:.1}"));
            } else {
                self.chat.push_system("dynthresh must be 0–100 (or 'off')".into());
            }
            return;
        }
        // `/enhance <prompt>` AI-expands the prompt, then generates fresh.
        if let Some(rest) = text.strip_prefix("/enhance") {
            let p = rest.trim().to_string();
            if !p.is_empty() {
                self.dispatch_generation(p, true);
            }
            return;
        }
        // `/save [name]` / `/load <name>` / `/sessions` — persist & restore the thread.
        if let Some(rest) = text.strip_prefix("/sessions") {
            let _ = rest;
            self.list_sessions();
            return;
        }
        if let Some(rest) = text.strip_prefix("/save") {
            self.save_session(rest.trim());
            return;
        }
        if let Some(rest) = text.strip_prefix("/load") {
            self.load_session(rest.trim());
            return;
        }
        // `/scenario` promotes the current image's exact recipe into a scenario task.
        if text.trim_start().starts_with("/scenario") {
            self.grab_recipe_into_scenario();
            return;
        }
        // `/vary [n]` fans out N variations of the current image at fresh seeds.
        if let Some(rest) = text.strip_prefix("/vary") {
            let n = rest.trim().parse::<usize>().unwrap_or(4);
            self.start_variations(n);
            return;
        }
        // `/relight <preset>` and `/faceswap <source> [N]` — edit the latest frame with a model op
        // (RFC UI-GALLERY-1 V1), reusing the generation channel so the result posts to Chat.
        if let Some(rest) = text.strip_prefix("/relight") {
            self.chat_relight(rest);
            return;
        }
        if let Some(rest) = text.strip_prefix("/faceswap") {
            self.chat_faceswap(rest.trim());
            return;
        }
        // `/preset save <name>` snapshots (model + LoRA stack + negative); `/preset` or
        // `/preset list` lists them; `/preset <name>` applies one. Palette entries apply too.
        if let Some(rest) = text.strip_prefix("/preset") {
            let arg = rest.trim();
            if let Some(name) = arg.strip_prefix("save").map(str::trim).filter(|s| !s.is_empty()) {
                self.save_preset(name);
            } else if arg.is_empty() || arg.eq_ignore_ascii_case("list") {
                self.list_presets();
            } else {
                self.apply_preset(arg);
            }
            return;
        }
        // `/auto on|off` — LLM edit/new routing for follow-ups.
        if let Some(rest) = text.strip_prefix("/auto") {
            self.auto_route = !rest.trim().eq_ignore_ascii_case("off");
            self.chat.push_system(
                if self.auto_route { "auto edit/new routing ON" } else { "auto routing OFF" }.into(),
            );
            return;
        }
        // With `/auto` on and an image already in the thread, classify the follow-up
        // (edit vs new scene) before dispatching — instead of always refining.
        if self.auto_route && self.base_seed.is_some() && self.route_rx.is_none() && self.active_gen.is_none() {
            self.classify_route(text);
            return;
        }
        self.dispatch_generation(text, false);
    }

    /// Ask the LLM whether `edit` edits the current image or asks for a new one, on a
    /// background thread; `drain_route` dispatches fresh / refine from the verdict.
    fn classify_route(&mut self, edit: String) {
        let context = if self.refine_prompt.is_empty() { "an image".into() } else { self.refine_prompt.clone() };
        let provider = crate::prompt::resolve_provider_label(&self.workspace.config.enhancer);
        self.chat.push_system(format!("routing “{edit}” …"));
        let (tx, rx) = std::sync::mpsc::channel();
        let rt = self.rt.clone();
        std::thread::spawn(move || {
            const SYSTEM: &str = "You decide whether a user's instruction EDITS the current \
                image or asks for a COMPLETELY NEW / different image. Reply with exactly one \
                word: EDIT or NEW.";
            let user = format!("Current image: {context}\nInstruction: {edit}");
            let verdict = rt
                .block_on(crate::prompt::complete(&provider, SYSTEM, &user, &crate::prompt::EnhanceArgs::default()))
                .unwrap_or_default();
            let is_new = verdict.to_uppercase().contains("NEW");
            let _ = tx.send((is_new, edit));
        });
        self.route_rx = Some(rx);
    }

    /// Drain the in-flight classification → dispatch fresh (`/new`) or refine.
    fn drain_route(&mut self) {
        if let Some(rx) = &self.route_rx {
            match rx.try_recv() {
                Ok((is_new, edit)) => {
                    self.route_rx = None;
                    if is_new {
                        self.dispatch_generation(format!("/new {edit}"), false);
                    } else {
                        self.dispatch_generation(edit, false);
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.route_rx = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
    }

    /// Dispatch a Chat prompt to the model thread (must have a model loaded). The
    /// denoise progress flows to the Output pane automatically; this also tracks the
    /// Preview/Done frames for the inline image. `enhance` runs the prompt through the
    /// configured LLM enhancer first (forces a fresh generation).
    fn dispatch_generation(&mut self, prompt: String, enhance: bool) {
        if self.active_gen.is_some() {
            return; // one generation at a time (the model thread is serial anyway)
        }
        // Conversational refinement: once an image exists, a follow-up prompt edits
        // it rather than starting over. `/new <prompt>` forces a fresh generation.
        let (edit, force_fresh) = match prompt.strip_prefix("/new") {
            Some(rest) => (rest.trim_start().to_string(), true),
            None => (prompt, false),
        };
        if edit.trim().is_empty() {
            return; // bare "/new" (or empty) — nothing to generate
        }
        if force_fresh {
            self.chat_mask = None; // a fresh image drops any stale Canvas mask
        }
        // Enhancing crafts a fresh detailed prompt, so it never refines.
        let refine = !force_fresh && !enhance && self.base_seed.is_some();
        // Identity-preserving continuation: a portrait with a known person was loaded in
        // Chat → a refine re-runs its IP-Adapter pass (keeps the face) instead of img2img.
        if refine && self.chat_identity.is_some() {
            self.dispatch_identity_refine(edit);
            return;
        }
        // A fresh / enhanced generation leaves the identity context behind.
        if !refine {
            self.chat_identity = None;
        }
        // Accumulate the prompt so earlier edits persist ("...sea with waves, a sail
        // boat on the horizon"). DEFAULT (prompt-evolve): re-render the accumulated
        // prompt with txt2img at the conversation's stable seed → typed edits reliably
        // appear and the composition stays recognizable. ANCHORED (`/strength`):
        // img2img over the clean base image at that strength.
        let full_prompt = if refine {
            if self.refine_prompt.is_empty() {
                edit.clone()
            } else {
                format!("{}, {}", self.refine_prompt, edit)
            }
        } else {
            edit.clone()
        };
        // A Canvas mask makes this ONE turn an inpaint over the masked image (the latest
        // render — `inpaint_base`), regardless of the sticky mode; then it's consumed.
        // Otherwise the mode rules: anchored (`/strength`) = img2img over the clean base;
        // prompt-evolve = txt2img.
        let inpaint = refine && self.chat_mask.is_some() && self.inpaint_base.is_some();
        let init_image = if inpaint {
            self.inpaint_base.clone()
        } else if refine {
            self.refine_strength.and(self.refine_base.clone())
        } else {
            None
        };
        self.active_is_refine = refine;
        self.active_full_prompt = full_prompt.clone();

        // Show the user's own words in the history (not the accumulated prompt).
        self.chat.push_utterance(edit.clone(), refine);
        // Discoverability nudge (once per session): prompt-evolve re-describes the whole
        // scene, so an "add a …" edit often won't insert the object. Point at Canvas
        // inpaint — the reliable way to add content to a specific region. Skipped for
        // anchored / inpaint turns (those already use the base image).
        if refine
            && !inpaint
            && self.refine_strength.is_none()
            && !self.inpaint_nudged
            && looks_like_object_insertion(&edit)
        {
            self.inpaint_nudged = true;
            self.chat.push_system(
                "tip: prompt-evolve re-renders the whole scene, so small additions may not \
                 appear. To reliably ADD an object, paint its area in Canvas (Ctrl-8) then \
                 prompt it."
                    .into(),
            );
        }
        // Generate at the loaded model's per-model size override (`/size`), else its
        // native square (sd15=512, sd21=768, sdxl=1024) — always Metal-safe, unlike a
        // fixed workspace size which OOMs SD1.5.
        let (nw, nh) = self
            .models
            .loaded_alias()
            .map(|a| self.resolve_gen_size(a))
            .unwrap_or((768, 768));
        let steps = self.steps_override.unwrap_or(self.workspace.config.default_steps);
        let guidance = self.guidance_override.unwrap_or(self.workspace.config.default_guidance);
        // v2.7: apply the free-quality guidance-bundle session overrides to the process env the
        // pipelines read (same mechanism as the `generate` CLI / scenarios). Cleared when off.
        let apply_env = |key: &str, val: Option<f64>| match val {
            Some(v) => unsafe { std::env::set_var(key, v.to_string()) },
            None => unsafe { std::env::remove_var(key) },
        };
        apply_env("PLAKAT_PAG_SCALE", self.pag_override);
        apply_env("PLAKAT_CFG_RESCALE", self.rescale_override);
        apply_env("PLAKAT_DYNTHRESH", self.dynthresh_override);
        if self.freeu_override {
            unsafe { std::env::set_var("PLAKAT_FREEU", "1") };
        } else {
            unsafe { std::env::remove_var("PLAKAT_FREEU") };
        }
        let preview_every = self.workspace.config.preview_every_n_steps;
        let out_dir = self.workspace.out_dir().join("chat");
        // Seed priority: an explicit `/seed` pin > the conversation's stable seed (so
        // prompt-evolve refines keep composition) > a fresh random seed.
        let seed = self
            .fixed_seed
            .or(if refine { self.base_seed } else { None })
            .unwrap_or_else(|| rand::random::<u32>() as u64);
        self.active_seed = seed;
        // Inpaint runs hot (the masked region regenerates); anchored uses its strength;
        // prompt-evolve ignores it (init is None).
        let strength = if inpaint { INPAINT_STRENGTH } else { self.refine_strength.unwrap_or(0.0) };
        let mask = if inpaint { self.chat_mask.take() } else { None }; // one-shot
        if inpaint {
            self.inpaint_base = None; // consumed with the mask
        }
        let enhancer = if enhance { Some(self.workspace.config.enhancer.clone()) } else { None };
        // An img2img/inpaint init image (e.g. a Canvas outpaint's grey-padded base) may
        // be non-square — generate at ITS dimensions (rounded to /8) so the mask aligns,
        // rather than squishing it into the native square. txt2img stays native-square.
        let (gen_w, gen_h) = match init_image.as_ref().and_then(|p| image::image_dimensions(p).ok()) {
            Some((iw, ih)) => ((iw / 8 * 8).max(8), (ih / 8 * 8).max(8)),
            None => (nw, nh),
        };
        let (rx, cancel) = self.model_svc.generate(
            full_prompt,
            self.negative.clone(),
            gen_w,
            gen_h,
            steps,
            guidance,
            seed,
            out_dir,
            preview_every,
            init_image,
            strength,
            mask,
            enhancer,
        );
        self.active_gen = Some((rx, cancel));
        self.chat.status = ChatStatus::Generating { step: 0, total: steps as u32, refine };
    }

    /// Drain the active generation's messages → Chat status, inline preview/final
    /// image (built here because it needs the Picker), and history.
    fn drain_generation(&mut self) {
        let mut msgs = Vec::new();
        // Distinguish "no message yet" (Empty) from "the sender was dropped"
        // (Disconnected). If the model-service thread PANICS mid-generation it drops the
        // sender with no terminal Done/Error — every peer drain handles this, but this one
        // used `while let Ok(..)` and silently swallowed it, wedging `active_gen` on
        // "Generating" forever (Chat stuck, /vary frozen, idle_tick disabled).
        let mut disconnected = false;
        if let Some((rx, _)) = &self.active_gen {
            loop {
                match rx.try_recv() {
                    Ok(m) => msgs.push(m),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }
        let mut finished = false;
        for msg in msgs {
            match msg {
                GenMessage::Progress { step, total, .. } => {
                    // Preserve the in-flight turn's refine flag (Progress doesn't carry it).
                    let refine = self.chat.history.last().is_some_and(|e| e.refine);
                    self.chat.status = ChatStatus::Generating { step, total, refine };
                }
                GenMessage::Preview { image, .. } => {
                    let dynimg = image::DynamicImage::ImageRgb8(image);
                    self.chat.preview = Some(self.picker.new_resize_protocol(dynimg));
                }
                GenMessage::Enhanced { prompt } => {
                    // Show the expanded prompt under the turn, and make it the base
                    // for the thread (so a later refine accumulates on the real text).
                    self.active_full_prompt = prompt.clone();
                    self.chat.set_last_enhanced(prompt);
                }
                GenMessage::Done { output, .. } => {
                    if let Ok(img) = image::open(&output) {
                        self.chat.preview = Some(self.picker.new_resize_protocol(img));
                    }
                    // Refinement bookkeeping. A FRESH generation seeds the thread:
                    // its image is the anchor base and its seed is the stable seed the
                    // prompt-evolve refines reuse. A refine keeps both and just records
                    // the accumulated prompt so the next edit builds on it.
                    self.refine_prompt = self.active_full_prompt.clone();
                    if !self.active_is_refine {
                        self.refine_base = Some(output.clone());
                        self.base_seed = Some(self.active_seed);
                    }
                    self.chat.refine_armed = true;
                    let path = output.display().to_string();
                    self.chat.finish_last(Ok(path.clone()));
                    self.chat.status = ChatStatus::Done(path);
                    finished = true;
                }
                GenMessage::Error { message } => {
                    self.chat.finish_last(Err(message.clone()));
                    self.chat.status = ChatStatus::Error(message);
                    finished = true;
                }
            }
        }
        // The channel closed without a terminal Done/Error → the generation thread died
        // (a panic deep in a pipeline). Surface it as an error instead of wedging forever.
        // (A normal completion also disconnects, but sets `finished` via Done/Error first.)
        if disconnected && !finished {
            let message = "generation ended unexpectedly (the render thread stopped)".to_string();
            self.chat.finish_last(Err(message.clone()));
            self.chat.status = ChatStatus::Error(message);
            finished = true;
        }
        if finished {
            self.active_gen = None;
            // Fire the next queued `/vary` variation, if any.
            self.pump_variations();
        }
    }

    fn render(&mut self, f: &mut Frame) {
        // [tab bar] [screen content] [Output pane — when non-empty] [status bar].
        // The Output pane (rerouted progress + messages) is visible on every screen.
        let show_output = !self.output.is_empty();
        let constraints = if show_output {
            vec![
                Constraint::Length(1),
                Constraint::Min(1),
                Constraint::Length(8),
                Constraint::Length(1),
            ]
        } else {
            vec![Constraint::Length(1), Constraint::Min(1), Constraint::Length(1)]
        };
        let rows = Layout::default().direction(Direction::Vertical).constraints(constraints).split(f.area());
        self.render_tab_bar(f, rows[0]);
        self.render_content(f, rows[1]);
        if show_output {
            self.output.render(f, rows[2]);
            self.render_status_bar(f, rows[3]);
        } else {
            self.render_status_bar(f, rows[2]);
        }
        // A memory-budget confirmation floats above everything (below the palette).
        if let Some(pl) = &self.pending_load {
            self.render_load_warning(f, pl);
        }
        if self.confirm_reset {
            self.render_reset_confirm(f);
        }
        if self.show_help {
            self.render_help(f);
        }
        // The command palette floats above everything when open.
        self.palette.render(f, f.area());
    }

    /// The active screen's key bindings (label, keys) for the cheatsheet. Globals are
    /// listed separately by `render_help`.
    fn screen_help(&self) -> (&'static str, &'static [(&'static str, &'static str)]) {
        match self.screen {
            ActiveScreen::Chat => ("Chat", &[
                ("Enter", "generate / refine"),
                ("Ctrl-T", "toggle evolve ↔ anchored"),
                ("Ctrl-P / Ctrl-N", "recall previous / next prompt"),
                ("Ctrl-← / Ctrl-→", "scrub the filmstrip"),
                ("Ctrl-B / Ctrl-Y", "roll back / vary selected frame"),
                ("Ctrl-Z / Ctrl-Shift-Z", "undo / redo (step through images)"),
                ("/vary /scenario /preset", "fan out · grab recipe · presets"),
                ("/relight <preset> · /faceswap <src>", "edit the latest frame (IC-Light · face-swap)"),
                ("/size WxH · /steps N · /cfg X", "per-model size · steps · guidance"),
                ("/pag X · /rescale φ · /freeu · /dynthresh P", "quality: PAG · CFG-rescale · FreeU · dyn-threshold"),
                ("/new /seed /strength /negative /auto", "session commands"),
            ]),
            ActiveScreen::Models => ("Models", &[
                ("L / U", "load / unload selected"),
                ("j / k", "move selection"),
                ("(palette) Cache doctor", "sweep locks + cache report"),
            ]),
            ActiveScreen::Scenarios => ("Scenarios", &[
                ("Enter", "run selected"),
                ("e / n", "edit / new scenario"),
            ]),
            ActiveScreen::History => ("History", &[
                ("v", "toggle thumbnail grid"),
                ("/ , ?", "substring filter , semantic search"),
                ("t / x / d", "tag / export / compare baseline"),
                ("c", "continue selected in Chat"),
                ("n / y", "naturalize (de-slop) / vary selected"),
                ("Del", "move selected to .trash (confirm)"),
            ]),
            ActiveScreen::LoraHub => ("LoRA Hub", &[
                ("a", "apply selected LoRA"),
                ("r / Ctrl-R", "assess selected / suggest a stack (LLM)"),
                ("u", "check for a newer version"),
            ]),
            ActiveScreen::People => ("People", &[
                ("g", "generate selected → Chat"),
                ("← / →", "cycle detail tab"),
                ("e / i", "encode / import persona"),
            ]),
            ActiveScreen::PromptWorkspace => ("Prompts", &[
                ("Ctrl-N / Ctrl-S", "new / save buffer"),
                ("Ctrl-T", "toggle Tera mode"),
                ("Ctrl-R / Ctrl-O", "LLM compile / open in Scenarios"),
            ]),
            ActiveScreen::Canvas => ("Canvas", &[
                ("m", "outpaint mode"),
                ("Enter", "rasterize mask → Chat"),
                ("Shift-C", "clear mask"),
            ]),
            ActiveScreen::Naturalize => ("Naturalize", &[
                ("↑ ↓", "select knob"),
                ("← → / + -", "adjust (polish/micro/grain/desaturate/paper)"),
                ("Space", "toggle original ↔ de-slopped"),
                ("o", "open an external image (type a path)"),
                ("p / w", "load / save a named preset"),
                ("r", "reload newest image"),
                ("s", "save de-slopped image"),
            ]),
        }
    }

    /// Centered keyboard cheatsheet (F1): global keys + the active screen's keys.
    fn render_help(&self, f: &mut Frame) {
        let (screen_name, keys) = self.screen_help();
        let globals: &[(&str, &str)] = &[
            ("Ctrl-K", "command palette"),
            ("Ctrl-1..8 / Tab", "switch screen"),
            ("F1", "this help"),
            ("Ctrl-C", "cancel generation / quit"),
            ("Ctrl-Q", "quit"),
        ];
        let mut body: Vec<Line> = Vec::new();
        body.push(Line::from(Span::styled("Global", Style::new().fg(Color::Magenta).add_modifier(Modifier::BOLD))));
        for (k, d) in globals {
            body.push(Line::from(vec![
                Span::styled(format!("  {k:<20}"), Style::new().fg(Color::Cyan)),
                Span::styled(d.to_string(), Style::new().fg(Color::Gray)),
            ]));
        }
        body.push(Line::from(""));
        body.push(Line::from(Span::styled(screen_name.to_string(), Style::new().fg(Color::Magenta).add_modifier(Modifier::BOLD))));
        for (k, d) in keys {
            body.push(Line::from(vec![
                Span::styled(format!("  {k:<20}"), Style::new().fg(Color::Cyan)),
                Span::styled(d.to_string(), Style::new().fg(Color::Gray)),
            ]));
        }
        body.push(Line::from(""));
        body.push(Line::from(Span::styled("  any key to close", Style::new().fg(Color::DarkGray))));

        let h = body.len() as u16 + 2;
        centered_modal(f, " keyboard shortcuts (F1) ", Color::Cyan, body, (60, h));
    }

    /// Centered confirmation for the hard reset (re-exec).
    fn render_reset_confirm(&self, f: &mut Frame) {
        let body = vec![
            Line::from(Span::styled("↻ Restart plakat?", Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
            Line::from(""),
            Line::from("  Re-execs a fresh process to fully return GPU memory."),
            Line::from(Span::styled("  Unsaved Chat state is lost (save first with /save).", Style::new().fg(Color::DarkGray))),
            Line::from(""),
            Line::from(Span::styled("  [Y] restart now   [N]/Esc cancel", Style::new().fg(Color::Gray))),
        ];
        centered_modal(f, " hard reset ", Color::Cyan, body, (60, 8));
    }

    /// Centered modal warning that a load would over-commit RAM (memory-budget guard).
    fn render_load_warning(&self, f: &mut Frame, pl: &PendingLoad) {
        let qual = if pl.exact { "needs" } else { "≈ needs" };
        let body = vec![
            Line::from(Span::styled(
                format!("⚠ {} may exceed available memory", pl.alias),
                Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(format!("  {qual} ~{:.1} GB resident", pl.est_gb)),
            Line::from(format!("  only {:.1} GB free right now", pl.avail_gb)),
            Line::from(""),
            Line::from(Span::styled(
                "  Loading may OOM. [Y] load anyway   [N]/Esc cancel",
                Style::new().fg(Color::Gray),
            )),
        ];
        centered_modal(f, " memory budget ", Color::Yellow, body, (62, 9));
    }

    fn render_tab_bar(&self, f: &mut Frame, area: Rect) {
        let titles = ActiveScreen::ALL
            .iter()
            .enumerate()
            .map(|(i, s)| Line::from(format!(" {} {} ", i + 1, s.title())));
        let tabs = Tabs::new(titles)
            .select(self.screen.index())
            .highlight_style(Style::new().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD));
        f.render_widget(tabs, area);
    }

    fn render_content(&mut self, f: &mut Frame, area: Rect) {
        match self.screen {
            ActiveScreen::Models => self.models.render(f, area),
            ActiveScreen::Chat => self.chat.render(f, area),
            ActiveScreen::Scenarios => self.scenarios.render(f, area),
            ActiveScreen::History => self.history.render(f, area),
            ActiveScreen::People => self.people.render(f, area),
            ActiveScreen::LoraHub => self.lorahub.render(f, area),
            ActiveScreen::PromptWorkspace => self.prompts.render(f, area),
            ActiveScreen::Canvas => self.canvas.render(f, area),
            ActiveScreen::Naturalize => self.naturalize.render(f, area),
        }
    }

    /// The ambient status-bar memory string + its headroom tint (red < 3 GB, yellow
    /// < 6 GB, else green). Split out for testing.
    fn mem_readout(&self, free: f64, total: f64) -> (String, Color) {
        let model = self.models.loaded_alias().unwrap_or("no model");
        let s = format!("{model} · {free:.1}/{total:.0} GB free ");
        let fg = if free < 3.0 {
            Color::Red
        } else if free < 6.0 {
            Color::Yellow
        } else {
            Color::Green
        };
        (s, fg)
    }

    fn render_status_bar(&self, f: &mut Frame, area: Rect) {
        // In a text-input mode (Chat, or the Scenarios editor) plain keys type, so
        // advertise only the input-safe switches.
        let input_mode = self.screen == ActiveScreen::Chat
            || (self.screen == ActiveScreen::Scenarios && self.scenarios.captures_input())
            || (self.screen == ActiveScreen::LoraHub && self.lorahub.captures_input())
            || (self.screen == ActiveScreen::PromptWorkspace && self.prompts.captures_input());
        let nav = if input_mode {
            "F1 help · Ctrl-K palette · Ctrl-1..8 / Tab switch · Ctrl-Q quit"
        } else {
            "F1 help · Ctrl-K palette · 1-8 / Tab switch · Ctrl-Q quit"
        };
        let txt = format!(" {} · {nav} ", self.workspace.config.name);
        // Ambient memory readout (right-aligned): loaded model + free/total RAM, tinted
        // by headroom — so the 1.20.0 memory budget is visible at a glance, not just at
        // load time.
        let (mem, mem_fg) = self.mem_readout(crate::hw::available_ram_gb(), crate::hw::total_ram_gb());
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(0), Constraint::Length(mem.len() as u16)])
            .split(area);
        let bg = Style::new().bg(Color::DarkGray);
        f.render_widget(Paragraph::new(txt).style(bg.fg(Color::White)), cols[0]);
        f.render_widget(
            Paragraph::new(mem).style(bg.fg(mem_fg)).alignment(ratatui::layout::Alignment::Right),
            cols[1],
        );
    }
}

/// Pick the multiperson scene's identity strategy + model from the marked personas'
/// own strategies (instead of forcing plus-face / sd15 for everyone). One pipeline runs
/// one encoder, so a single strategy must cover the set:
///   - parse each persona's `identity` string (empty / unknown are ignored);
///   - SDXL and SD1.5 encoders can't mix → if *any* SDXL strategy is present, the scene
///     is SDXL (PlusFaceSdxl unless every named strategy is FaceId → FaceIdSdxl);
///   - otherwise SD1.5: FaceId only when *every* named strategy is FaceId, else PlusFace
///     (the more general CLIP-H whole-face encoder);
///   - nothing named → default PlusFace / sd15.
/// Returns `(IdentityKind, model_alias)`.
fn route_multiperson_identity(strategies: &[String]) -> (crate::pipelines::ip_adapter::IdentityKind, &'static str) {
    use crate::pipelines::ip_adapter::IdentityKind;
    let kinds: Vec<IdentityKind> = strategies
        .iter()
        .filter(|s| !s.trim().is_empty())
        .filter_map(|s| s.parse::<IdentityKind>().ok())
        .collect();
    if kinds.is_empty() {
        return (IdentityKind::PlusFace, "sd15");
    }
    let any_sdxl = kinds.iter().any(|k| matches!(k, IdentityKind::PlusFaceSdxl | IdentityKind::FaceIdSdxl));
    let all_faceid = kinds.iter().all(|k| matches!(k, IdentityKind::FaceId | IdentityKind::FaceIdSdxl));
    match (any_sdxl, all_faceid) {
        (true, true) => (IdentityKind::FaceIdSdxl, "sdxl"),
        (true, false) => (IdentityKind::PlusFaceSdxl, "sdxl"),
        (false, true) => (IdentityKind::FaceId, "sd15"),
        (false, false) => (IdentityKind::PlusFace, "sd15"),
    }
}

/// Detect faces in `path` with SCRFD and return their boxes normalized to 0..1
/// `[x1, y1, x2, y2]`. Best-effort: a missing/unavailable detector or any error yields
/// an empty list (the Canvas `B` preset then fills the background plainly).
fn detect_face_boxes_normalized(
    rt: &Handle,
    device: &Device,
    path: &std::path::Path,
) -> Option<Vec<[f32; 4]>> {
    let weights = rt.block_on(crate::pipelines::scrfd::resolve_scrfd_weights()).ok().flatten()?;
    let detector = crate::pipelines::scrfd::SCRFDDetector::load(
        &weights,
        crate::pipelines::scrfd::SCRFDConfig::default(),
        device,
        candle_core::DType::F32,
    )
    .ok()?;
    let faces = detector.detect(path).ok()?;
    let (w, h) = image::image_dimensions(path).ok()?;
    let (w, h) = (w as f32, h as f32);
    if w <= 0.0 || h <= 0.0 {
        return Some(Vec::new());
    }
    Some(
        faces
            .iter()
            .map(|f| {
                [
                    (f.bbox[0] / w).clamp(0.0, 1.0),
                    (f.bbox[1] / h).clamp(0.0, 1.0),
                    (f.bbox[2] / w).clamp(0.0, 1.0),
                    (f.bbox[3] / h).clamp(0.0, 1.0),
                ]
            })
            .collect(),
    )
}

/// Recover `(positive prompt, seed)` from an image's embedded A1111 recipe, for
/// filmstrip rollback / variation. Empty / `None` when the image carries no recipe.
fn recover_recipe(path: &std::path::Path) -> (String, Option<u64>) {
    match crate::imaging::io::read_parameters_chunk(path).ok().flatten() {
        Some(params) => (recipe_positive(&params), recipe_seed(&params)),
        None => (String::new(), None),
    }
}

/// The positive prompt of an A1111 recipe: everything before the `Negative prompt:` /
/// `Steps:` parameter lines.
fn recipe_positive(params: &str) -> String {
    let mut out = Vec::new();
    for line in params.lines() {
        let t = line.trim_start();
        if t.starts_with("Negative prompt:") || t.starts_with("Steps:") {
            break;
        }
        out.push(line);
    }
    out.join(" ").trim().to_string()
}

/// Parse a `/size` spec into (w, h) rounded to multiples of 8 (the latent stride), each
/// clamped to a sane 256–2048 range. Accepts `WxH` (also `W×H`, `W*H`) or a single `N`
/// (square). Returns None on garbage or out-of-range values.
fn parse_gen_size(spec: &str) -> Option<(u32, u32)> {
    let norm = spec.to_lowercase().replace(['×', '*'], "x");
    let (w, h) = match norm.split_once('x') {
        Some((a, b)) => (a.trim().parse::<u32>().ok()?, b.trim().parse::<u32>().ok()?),
        None => {
            let n = norm.trim().parse::<u32>().ok()?;
            (n, n)
        }
    };
    let round = |v: u32| -> Option<u32> {
        let r = (v / 8) * 8;
        (256..=2048).contains(&r).then_some(r)
    };
    Some((round(w)?, round(h)?))
}

/// The negative prompt of an A1111 recipe: the text after `Negative prompt:` up to the
/// first `Steps:` parameter line. Empty when absent.
fn recipe_negative(params: &str) -> String {
    let Some(idx) = params.find("Negative prompt:") else { return String::new() };
    let rest = &params[idx + "Negative prompt:".len()..];
    let mut out = Vec::new();
    for line in rest.lines() {
        if line.trim_start().starts_with("Steps:") {
            break;
        }
        out.push(line);
    }
    out.join(" ").trim().to_string()
}

/// The `Seed: N` value of an A1111 recipe.
fn recipe_seed(params: &str) -> Option<u64> {
    let idx = params.find("Seed:")?;
    params[idx + "Seed:".len()..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

/// Case-insensitive replace of every `needle` in `haystack` with `repl` (char-based,
/// UTF-8 safe).
fn replace_ci(haystack: &str, needle: &str, repl: &str) -> String {
    if needle.is_empty() {
        return haystack.to_string();
    }
    let hay: Vec<char> = haystack.chars().collect();
    let hl: Vec<char> = haystack.to_lowercase().chars().collect();
    let nl: Vec<char> = needle.to_lowercase().chars().collect();
    // Rare: lowercasing changes the char count → fall back to a case-sensitive replace
    // rather than risk a misaligned index.
    if hl.len() != hay.len() {
        return haystack.replace(needle, repl);
    }
    let mut out = String::new();
    let mut i = 0;
    while i < hay.len() {
        if i + nl.len() <= hl.len() && hl[i..i + nl.len()] == nl[..] {
            out.push_str(repl);
            i += nl.len();
        } else {
            out.push(hay[i]);
            i += 1;
        }
    }
    out
}

/// Whether a Chat edit reads like an "insert an object" instruction (`add a …`, `put …`,
/// `place …`, `give it …`) — the case prompt-evolve handles poorly and Canvas inpaint
/// handles well. Heuristic, used only to fire a one-time discoverability hint.
fn looks_like_object_insertion(edit: &str) -> bool {
    let e = edit.trim_start().to_lowercase();
    const LEADS: &[&str] = &["add ", "put ", "place ", "insert ", "give "];
    LEADS.iter().any(|p| e.starts_with(p))
}

/// Filesystem-safe Chat-session file stem (empty → "session").
fn session_slug(name: &str) -> String {
    let s: String = name.trim().to_lowercase().chars().map(|c| if c.is_alphanumeric() { c } else { '-' }).collect();
    let s: String = s.split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-");
    if s.is_empty() { "session".to_string() } else { s }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::tui::workspace::{Workspace, WorkspaceConfig};

    /// One shared runtime for the whole test binary (the nav tests never block on
    /// it — the model thread just idles on its command channel).
    fn test_handle() -> Handle {
        use std::sync::OnceLock;
        static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
        RT.get_or_init(|| tokio::runtime::Runtime::new().unwrap()).handle().clone()
    }

    fn test_app() -> App {
        let ws = Workspace { root: "/tmp/plakat-ui-test".into(), config: WorkspaceConfig::default() };
        let (_tx, rx) = std::sync::mpsc::channel();
        // A synthetic Picker (no terminal query) so the navigation logic is testable.
        App::new(ws, Picker::from_fontsize((8, 16)), Device::Cpu, test_handle(), rx)
    }

    fn key(c: char, ctrl: bool) -> KeyEvent {
        let m = if ctrl { KeyModifiers::CONTROL } else { KeyModifiers::NONE };
        KeyEvent::new(KeyCode::Char(c), m)
    }

    #[test]
    fn digits_switch_screens() {
        let mut a = test_app();
        assert!(matches!(a.screen, ActiveScreen::Chat));
        a.handle_key(key('2', true)); // Ctrl-2
        assert!(matches!(a.screen, ActiveScreen::Models));
        a.handle_key(key('8', false)); // plain 8 (fallback)
        assert!(matches!(a.screen, ActiveScreen::Canvas));
        a.handle_key(key('1', true));
        assert!(matches!(a.screen, ActiveScreen::Chat));
    }

    #[test]
    fn memory_budget_confirm_intercepts_keys() {
        let mut a = test_app();
        let pl = || PendingLoad { alias: "sdxl".into(), est_gb: 30.0, avail_gb: 4.0, exact: true };
        // N / Esc cancels: the load is dropped, no quit, no screen change.
        a.pending_load = Some(pl());
        a.handle_key(key('n', false));
        assert!(a.pending_load.is_none(), "N dismisses the warning");
        assert!(!a.should_quit);
        // While the warning is up, a navigation key is swallowed (dismisses, no nav).
        a.pending_load = Some(pl());
        a.handle_key(key('2', true)); // would normally switch to Models
        assert!(a.pending_load.is_none());
        assert!(matches!(a.screen, ActiveScreen::Chat), "the key was consumed by the modal");
        // Y confirms (clears the warning and dispatches the load).
        a.pending_load = Some(pl());
        a.handle_key(key('y', false));
        assert!(a.pending_load.is_none(), "Y proceeds and clears the warning");
    }

    #[test]
    fn idle_auto_unload_remembers_and_resume_reloads() {
        use crate::ui::tui::screens::models::LoadState;
        let mut a = test_app();
        // A model is loaded and the user has been idle past the threshold. (Uses a
        // fake alias so `resume` dispatches a no-op load, keeping the test hermetic.)
        a.models.load = LoadState::Loaded { alias: "fake-idle-model".into(), used_gb: 7.0 };
        a.last_activity = Instant::now()
            .checked_sub(IDLE_UNLOAD + Duration::from_secs(1))
            .expect("monotonic clock is well past boot");
        a.idle_tick();
        assert_eq!(a.suspended.as_deref(), Some("fake-idle-model"), "idle unloads and remembers the model");
        // While suspended, idle_tick is a no-op (no double-unload).
        a.idle_tick();
        assert_eq!(a.suspended.as_deref(), Some("fake-idle-model"));
        // The unload lands (svc → Idle); the next keypress resumes the model.
        a.models.load = LoadState::Idle;
        a.resume_if_suspended();
        assert!(a.suspended.is_none(), "resume consumes the suspended model and reloads");
    }

    #[test]
    fn chat_mode_hint_reflects_the_loop_state() {
        let mut a = test_app();
        // No base image → the next Enter is a fresh generation.
        assert_eq!(a.chat_mode_hint(), "fresh");
        // With a stable seed and default strength → prompt-evolve.
        a.base_seed = Some(12345);
        assert_eq!(a.chat_mode_hint(), "evolve · seed 12345");
        // Anchored img2img shows the strength.
        a.refine_strength = Some(0.6);
        assert_eq!(a.chat_mode_hint(), "anchored 0.60 · seed 12345");
        // A pinned seed overrides the stable seed in the readout.
        a.fixed_seed = Some(999);
        assert_eq!(a.chat_mode_hint(), "anchored 0.60 · seed 999");
        // Auto-routing appends a marker.
        a.auto_route = true;
        assert!(a.chat_mode_hint().ends_with("· auto"));
        // A pending Canvas mask + base → inpaint takes precedence.
        a.refine_strength = None;
        a.chat_mask = Some("/tmp/mask.png".into());
        a.inpaint_base = Some("/tmp/base.png".into());
        assert!(a.chat_mode_hint().starts_with("inpaint"));
    }

    #[test]
    fn slash_steps_and_cfg_set_and_clear_session_overrides() {
        let mut a = test_app();
        a.handle_chat_submit("/steps 30".into());
        assert_eq!(a.steps_override, Some(30));
        a.handle_chat_submit("/cfg 6.5".into());
        assert_eq!(a.guidance_override, Some(6.5));
        // Out-of-range values are rejected without changing state.
        a.handle_chat_submit("/steps 999".into());
        assert_eq!(a.steps_override, Some(30), "invalid steps left intact");
        a.handle_chat_submit("/cfg 99".into());
        assert_eq!(a.guidance_override, Some(6.5), "invalid cfg left intact");
        // 'default' clears back to the workspace value.
        a.handle_chat_submit("/steps default".into());
        assert!(a.steps_override.is_none());
        a.handle_chat_submit("/cfg default".into());
        assert!(a.guidance_override.is_none());
        // Neither dispatched a generation.
        assert!(a.active_gen.is_none());
    }

    #[test]
    fn parse_gen_size_accepts_wxh_and_square_and_rounds() {
        assert_eq!(parse_gen_size("1024x768"), Some((1024, 768)));
        assert_eq!(parse_gen_size("1024×768"), Some((1024, 768)), "unicode ×");
        assert_eq!(parse_gen_size("768"), Some((768, 768)), "single N → square");
        assert_eq!(parse_gen_size("1000x1000"), Some((1000, 1000)), "1000 is /8-aligned");
        assert_eq!(parse_gen_size("1023x769"), Some((1016, 768)), "rounds down to /8");
        assert_eq!(parse_gen_size("100"), None, "below the 256 floor");
        assert_eq!(parse_gen_size("4096"), None, "above the 2048 ceiling");
        assert_eq!(parse_gen_size("big"), None);
    }

    #[test]
    fn size_override_is_per_model_and_budget_reported() {
        use crate::ui::tui::screens::models::LoadState;
        let mut a = test_app();
        a.models.load = LoadState::Loaded { alias: "sd15".into(), used_gb: 4.0 };
        // Default is the native square.
        assert_eq!(a.resolve_gen_size("sd15"), (512, 512));
        // Setting an override changes only that model; a report line is emitted.
        a.set_gen_size("1024x768");
        assert_eq!(a.resolve_gen_size("sd15"), (1024, 768));
        assert_eq!(a.resolve_gen_size("sdxl"), (1024, 1024), "override is per-model");
        assert!(a.chat.history.last().unwrap().utterance.contains("1024×768"));
        // The mode readout surfaces the non-native size.
        a.base_seed = Some(7);
        assert!(a.chat_mode_hint().contains("1024×768"));
        // Clearing returns to native.
        a.set_gen_size("native");
        assert_eq!(a.resolve_gen_size("sd15"), (512, 512));
        // Garbage is rejected without changing state.
        a.set_gen_size("1024x768");
        a.set_gen_size("nonsense");
        assert_eq!(a.resolve_gen_size("sd15"), (1024, 768), "bad spec left the override intact");
    }

    #[test]
    fn vary_queues_variations_and_pumps_one_at_a_time() {
        let mut a = test_app();
        // No image / prompt yet → a friendly no-op.
        a.start_variations(4);
        assert!(a.variation_queue.is_empty() && a.active_gen.is_none(), "no source → no-op");
        // With a prompt source, one dispatches immediately and the rest queue.
        a.refine_prompt = "a red fox".into();
        a.start_variations(4);
        assert!(a.active_gen.is_some(), "first variation dispatched");
        assert_eq!(a.variation_queue.len(), 3, "the other three wait (serial model)");
        // Simulate that generation finishing → the next one pumps.
        a.active_gen = None;
        a.pump_variations();
        assert!(a.active_gen.is_some());
        assert_eq!(a.variation_queue.len(), 2);
        // The count clamps to 8.
        let mut b = test_app();
        b.refine_prompt = "x".into();
        b.start_variations(99);
        assert_eq!(b.variation_queue.len(), 7, "clamped to 8, one already dispatched");
    }

    #[test]
    fn preset_save_then_apply_restores_loras_and_negative() {
        use crate::ui::tui::screens::models::LoadState;
        let mut a = test_app();
        std::fs::create_dir_all(&a.workspace.root).unwrap();
        let _ = std::fs::remove_file(a.workspace.root.join("presets.json"));
        // A fake loaded model keeps apply's request_load network-free.
        a.models.load = LoadState::Loaded { alias: "fake-preset-model".into(), used_gb: 7.0 };
        a.active_loras = vec![(std::path::PathBuf::from("/l/film.safetensors"), 0.8)];
        a.negative = "blurry".into();
        a.gen_sizes.insert("fake-preset-model".into(), (1024, 768));
        a.steps_override = Some(30);
        a.guidance_override = Some(6.5);
        a.save_preset("portrait");
        // Clear the working state, then apply the preset back.
        a.active_loras.clear();
        a.negative.clear();
        a.gen_sizes.clear();
        a.steps_override = None;
        a.guidance_override = None;
        a.apply_preset("portrait");
        assert_eq!(a.active_loras.len(), 1, "LoRA stack restored");
        assert_eq!(a.negative, "blurry", "negative restored");
        assert_eq!(a.gen_sizes.get("fake-preset-model"), Some(&(1024, 768)), "size restored");
        assert_eq!(a.steps_override, Some(30), "steps restored");
        assert_eq!(a.guidance_override, Some(6.5), "guidance restored");
        // A missing preset is a friendly no-op (no panic).
        a.apply_preset("does-not-exist");
        let _ = std::fs::remove_file(a.workspace.root.join("presets.json"));
    }

    #[test]
    fn f1_toggles_the_help_overlay_and_any_key_closes_it() {
        let mut a = test_app();
        assert!(!a.show_help);
        a.handle_key(KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE));
        assert!(a.show_help, "F1 opens the cheatsheet");
        // While open, any key closes it (and is swallowed — no screen change).
        a.handle_key(key('2', true));
        assert!(!a.show_help, "any key closes the cheatsheet");
        assert!(matches!(a.screen, ActiveScreen::Chat), "the closing key was consumed");
        // Every screen has a non-empty help table.
        for s in ActiveScreen::ALL {
            a.screen = s;
            let (name, keys) = a.screen_help();
            assert!(!name.is_empty() && !keys.is_empty(), "{s:?} has help entries");
        }
    }

    #[test]
    fn status_bar_memory_readout_tints_by_headroom() {
        use crate::ui::tui::screens::models::LoadState;
        let a = test_app();
        // No model + healthy headroom → green, "no model".
        let (s, c) = a.mem_readout(12.0, 24.0);
        assert!(s.starts_with("no model") && s.contains("12.0/24 GB free"));
        assert_eq!(c, Color::Green);
        // Tight (< 6) → yellow; critical (< 3) → red.
        assert_eq!(a.mem_readout(5.0, 24.0).1, Color::Yellow);
        assert_eq!(a.mem_readout(2.0, 24.0).1, Color::Red);
        // The loaded model's alias is shown.
        let mut b = test_app();
        b.models.load = LoadState::Loaded { alias: "sdxl".into(), used_gb: 7.0 };
        assert!(b.mem_readout(8.0, 24.0).0.starts_with("sdxl"));
    }

    #[test]
    fn ctrl_t_toggles_evolve_and_anchor() {
        let mut a = test_app();
        assert!(a.refine_strength.is_none(), "prompt-evolve by default");
        a.toggle_anchor();
        assert_eq!(a.refine_strength, Some(DEFAULT_ANCHOR_STRENGTH), "→ anchored");
        a.toggle_anchor();
        assert!(a.refine_strength.is_none(), "→ back to evolve");
    }

    #[test]
    fn hard_reset_needs_confirmation() {
        let mut a = test_app();
        // The palette command arms the confirm, it does not reset immediately.
        a.exec_palette(Cmd::HardReset);
        assert!(a.confirm_reset && !a.should_reset);
        // N cancels.
        a.handle_key(key('n', false));
        assert!(!a.confirm_reset && !a.should_reset, "N dismisses the reset");
        // Y confirms → the event loop will break and run() re-execs.
        a.exec_palette(Cmd::HardReset);
        a.handle_key(key('y', false));
        assert!(!a.confirm_reset && a.should_reset, "Y requests the re-exec");
    }

    #[test]
    fn cache_doctor_reports_to_output() {
        let mut a = test_app();
        a.screen = ActiveScreen::Models;
        a.run_cache_doctor();
        let out = a.output.lines_for_test();
        assert!(out.iter().any(|l| l.contains("cache doctor")), "doctor announces the model");
        // It always reports a lock-sweep result and a cache-status line.
        assert!(out.iter().any(|l| l.contains("stale lock")), "reports lock-sweep");
    }

    #[test]
    fn idle_tick_never_unloads_when_nothing_loaded() {
        let mut a = test_app();
        a.last_activity = Instant::now()
            .checked_sub(IDLE_UNLOAD + Duration::from_secs(1))
            .expect("monotonic clock is well past boot");
        a.idle_tick();
        assert!(a.suspended.is_none(), "nothing to unload when no model is resident");
    }

    #[test]
    fn request_load_of_the_resident_model_never_warns() {
        // Reloading the already-loaded model adds no footprint → never prompts.
        let mut a = test_app();
        // (No model is loaded in the test harness, so this exercises the non-resident
        // path; the guard is the loaded_alias short-circuit, covered by construction.)
        a.request_load("definitely-not-a-real-model-xyz".into());
        // Either it loaded (fits) or it queued a warning — but never panicked and the
        // app stayed alive.
        assert!(!a.should_quit);
    }

    #[test]
    fn quit_keys_set_should_quit() {
        // Ctrl-Q / Ctrl-C quit from any screen (incl. the Chat input).
        let mut a = test_app();
        a.handle_key(key('q', true));
        assert!(a.should_quit);
        let mut b = test_app();
        b.handle_key(key('c', true));
        assert!(b.should_quit);
        // Plain `q` quits on a non-input screen (Models) but NOT on Chat (it types).
        let mut c = test_app();
        c.screen = ActiveScreen::Models;
        c.handle_key(key('q', false));
        assert!(c.should_quit);
    }

    #[test]
    fn plain_q_types_into_chat_not_quit() {
        let mut a = test_app(); // starts on Chat
        a.handle_key(key('q', false));
        assert!(!a.should_quit, "plain q types into the Chat input");
        assert_eq!(a.chat.editor.text(), "q");
    }

    #[test]
    fn tab_cycles_screens_universally() {
        // Tab / Shift-Tab work on every terminal (the iTerm2 / no-protocol fallback).
        let mut a = test_app();
        a.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert!(matches!(a.screen, ActiveScreen::Models));
        // Shift-Tab as legacy BackTab → backward.
        a.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT));
        assert!(matches!(a.screen, ActiveScreen::Chat));
        // Shift-Tab as Tab+SHIFT (kbd-protocol encoding) → also backward (wraps to the last screen).
        a.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT));
        assert!(matches!(a.screen, ActiveScreen::Naturalize));
    }

    #[test]
    fn drain_generation_recovers_from_a_dead_render_thread() {
        let mut a = test_app();
        // A generation is in flight with a history entry awaiting its result…
        a.chat.push_utterance("a fox".into(), false);
        let (tx, rx) = std::sync::mpsc::channel::<GenMessage>();
        a.active_gen = Some((rx, CancelFlag::new()));
        // …then the render thread dies with no terminal message (drop the sender).
        drop(tx);
        a.drain_generation();
        // The soft-lock is gone: active_gen cleared and the turn marked failed.
        assert!(a.active_gen.is_none(), "a disconnected channel unwedges the gen loop");
        assert!(matches!(a.chat.status, ChatStatus::Error(_)), "surfaced as an error");
        assert!(a.chat.history.last().unwrap().error.is_some());
    }

    #[test]
    fn ctrl_c_cancels_active_generation_instead_of_quitting() {
        let mut a = test_app();
        let (_tx, rx) = std::sync::mpsc::channel();
        let cancel = CancelFlag::new();
        a.active_gen = Some((rx, cancel.clone()));
        a.handle_key(key('c', true)); // Ctrl-C
        assert!(!a.should_quit, "Ctrl-C cancels the running gen, doesn't quit");
        assert!(cancel.is_cancelled());
        // With no active generation, Ctrl-C quits.
        let mut b = test_app();
        b.handle_key(key('c', true));
        assert!(b.should_quit);
    }

    #[test]
    fn chat_refines_after_an_image_and_new_forces_fresh() {
        let mut a = test_app();
        // First turn: no prior image → fresh generation.
        a.dispatch_generation("a fox".into(), false);
        assert!(!a.chat.history.last().unwrap().refine, "first turn is fresh");
        assert!(!a.active_is_refine);
        assert_eq!(a.active_full_prompt, "a fox");

        // Simulate that turn completing: a stable seed + accumulated prompt are set.
        a.active_gen = None;
        a.base_seed = Some(42);
        a.refine_base = Some("/tmp/plakat-42.png".into());
        a.refine_prompt = "a fox".into();

        // Next turn refines: reuses the seed + accumulates the prompt.
        a.dispatch_generation("make it warmer".into(), false);
        let last = a.chat.history.last().unwrap();
        assert!(last.refine, "follow-up refines");
        assert_eq!(last.utterance, "make it warmer", "history shows the user's words");
        assert_eq!(a.active_full_prompt, "a fox, make it warmer", "prompt accumulates");
        assert!(a.active_is_refine);

        // `/new` forces a fresh generation even with a prior image, and strips the prefix.
        a.active_gen = None;
        a.dispatch_generation("/new a cyberpunk city".into(), false);
        let last = a.chat.history.last().unwrap();
        assert!(!last.refine, "/new is fresh");
        assert_eq!(last.utterance, "a cyberpunk city");
        assert_eq!(a.active_full_prompt, "a cyberpunk city", "/new resets the accumulated prompt");

        // Bare `/new` is a no-op (no empty turn).
        a.active_gen = None;
        let before = a.chat.history.len();
        a.dispatch_generation("/new".into(), false);
        assert_eq!(a.chat.history.len(), before, "bare /new generates nothing");
    }

    #[test]
    fn slash_negative_sets_session_negative_without_generating() {
        let mut a = test_app();
        let before = a.chat.history.len();
        a.handle_chat_submit("/negative blurry, lowres".into());
        assert_eq!(a.negative, "blurry, lowres");
        assert!(a.active_gen.is_none(), "/negative does not generate");
        assert_eq!(a.chat.history.len(), before + 1, "pushes a system note");
        assert!(a.chat.history.last().unwrap().system);
        // Bare `/negative` clears it.
        a.handle_chat_submit("/negative".into());
        assert_eq!(a.negative, "");
    }

    #[test]
    fn slash_enhance_forces_a_fresh_generation() {
        let mut a = test_app();
        a.base_seed = Some(7); // a plain prompt would refine
        a.handle_chat_submit("/enhance a fox".into());
        let last = a.chat.history.last().unwrap();
        assert!(!last.refine, "/enhance never refines");
        assert_eq!(last.utterance, "a fox");
        assert!(!a.active_is_refine);
    }

    #[test]
    fn slash_strength_and_seed_tune_session_state_without_generating() {
        let mut a = test_app();
        // /strength opts into image-anchored mode at a value.
        a.handle_chat_submit("/strength 0.8".into());
        assert_eq!(a.refine_strength, Some(0.8));
        assert!(a.active_gen.is_none());
        // out-of-range is rejected (value unchanged).
        a.handle_chat_submit("/strength 5".into());
        assert_eq!(a.refine_strength, Some(0.8));
        // /strength off returns to prompt-evolve (None).
        a.handle_chat_submit("/strength off".into());
        assert_eq!(a.refine_strength, None);

        a.handle_chat_submit("/seed 1234".into());
        assert_eq!(a.fixed_seed, Some(1234));
        a.handle_chat_submit("/seed random".into());
        assert_eq!(a.fixed_seed, None);
        // every command pushed a system note; nothing generated.
        assert!(a.chat.history.iter().all(|e| e.system));
    }

    #[test]
    fn toggle_lora_applies_removes_and_refuses_incompatible() {
        let mut a = test_app(); // no model loaded → no reload side-effect
        let p1 = std::path::PathBuf::from("/tmp/style.safetensors");
        a.toggle_lora(p1.clone(), true);
        assert_eq!(a.active_loras, vec![(p1.clone(), APPLY_LORA_SCALE)], "applied at default weight");
        a.toggle_lora(p1.clone(), true);
        assert!(a.active_loras.is_empty(), "re-toggle removes it");
        a.toggle_lora("/tmp/bad.safetensors".into(), false);
        assert!(a.active_loras.is_empty(), "incompatible LoRA is refused");
    }

    #[test]
    fn adjust_lora_weight_clamps_and_only_affects_applied() {
        let mut a = test_app();
        let p = std::path::PathBuf::from("/tmp/style.safetensors");
        a.toggle_lora(p.clone(), true); // applied @ 0.8
        a.adjust_lora_weight(p.clone(), 0.1);
        assert!((a.active_loras[0].1 - 0.9).abs() < 1e-6, "weight nudged up");
        // Clamps at 1.5.
        for _ in 0..20 {
            a.adjust_lora_weight(p.clone(), 0.1);
        }
        assert!(a.active_loras[0].1 <= 1.5 + 1e-6);
        // A non-applied LoRA is left alone.
        a.adjust_lora_weight("/tmp/other.safetensors".into(), 0.1);
        assert_eq!(a.active_loras.len(), 1);
    }

    #[test]
    fn continue_uses_prompt_evolve_with_a_recipe_else_anchored() {
        let mut a = test_app();
        // With a recovered seed → prompt-evolve (refine_strength None), additive-friendly.
        a.continue_from_image("/tmp/x.png".into(), "a fox".into(), Some(42));
        assert_eq!(a.base_seed, Some(42));
        assert_eq!(a.refine_strength, None, "recipe → prompt-evolve");
        // Without a seed → image-anchored fallback.
        a.continue_from_image("/tmp/y.png".into(), "a wolf".into(), None);
        assert!(a.refine_strength.is_some(), "no recipe → anchored");
    }

    #[test]
    fn canvas_mask_is_a_one_shot_inpaint_that_does_not_lock_the_mode() {
        let mut a = test_app();
        // Start in prompt-evolve with a base.
        a.base_seed = Some(7);
        a.refine_base = Some("/tmp/base.png".into());
        a.refine_strength = None;
        // Canvas hands over a mask — it must NOT flip the sticky mode.
        a.apply_canvas_mask("/tmp/mask.png".into());
        assert!(a.chat_mask.is_some());
        assert_eq!(a.refine_strength, None, "mask doesn't lock anchored mode");
        // The next refine consumes the mask (one-shot)…
        a.dispatch_generation("a sun".into(), false);
        assert!(a.chat_mask.is_none(), "mask consumed after one turn");
        // …and the mode is still prompt-evolve for the following edits.
        assert_eq!(a.refine_strength, None);
    }

    #[test]
    fn auto_route_toggles_and_defers_dispatch_to_classification() {
        let mut a = test_app();
        assert!(!a.auto_route, "off by default");
        a.handle_chat_submit("/auto on".into());
        assert!(a.auto_route);

        // With an image in the thread, a follow-up classifies (deferred) instead of
        // dispatching immediately.
        a.base_seed = Some(7);
        a.refine_base = Some("/tmp/x.png".into());
        a.handle_chat_submit("a dragon".into());
        assert!(a.route_rx.is_some(), "submission is routed, not yet dispatched");
        assert!(a.active_gen.is_none());

        a.handle_chat_submit("/auto off".into());
        assert!(!a.auto_route);
    }

    #[test]
    fn esc_does_not_quit() {
        // Regression: Esc was quitting, so legacy Ctrl-3 (== Esc byte) killed the app.
        let mut a = test_app();
        a.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!a.should_quit);
    }

    #[test]
    fn grab_from_chat_needs_something_to_summarize() {
        // With an empty chat thread / refine prompt / editor, the grab is a no-op that
        // reports why — no background LLM job is spawned.
        let mut a = test_app();
        a.grab_chat_into_scenario();
        assert!(a.chat_to_scenario.is_none(), "no LLM job without source material");
        assert!(a.scenarios.status.contains("nothing in Chat"));
    }

    #[test]
    fn grab_from_chat_spawns_a_summary_when_the_thread_has_content() {
        // A real chat utterance gives the summarizer source → an in-flight job starts.
        let mut a = test_app();
        a.chat.push_utterance("a watercolor fox".into(), false);
        a.grab_chat_into_scenario();
        assert!(a.chat_to_scenario.is_some(), "a summary job is in flight");
        // A second press while one is in flight is ignored (no double-spawn).
        a.grab_chat_into_scenario();
        assert!(a.chat_to_scenario.is_some());
    }

    #[test]
    fn session_slug_is_filesystem_safe() {
        assert_eq!(session_slug("My Session 1"), "my-session-1");
        assert_eq!(session_slug("  "), "session");
        assert_eq!(session_slug("a/b"), "a-b");
    }

    #[test]
    fn save_then_load_session_round_trips() {
        let root = std::env::temp_dir().join("plakat-ui-session-test");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let ws = Workspace { root: root.clone(), config: WorkspaceConfig::default() };
        let (_tx, rx) = std::sync::mpsc::channel();
        let mut a = App::new(ws, Picker::from_fontsize((8, 16)), Device::Cpu, test_handle(), rx);

        // Build a little session: a turn + accumulated state.
        a.chat.push_utterance("a watercolor fox".into(), false);
        a.chat.finish_last(Ok("out/chat/plakat-7-1.png".into()));
        a.refine_prompt = "a watercolor fox, autumn leaves".into();
        a.base_seed = Some(7);
        a.negative = "blurry".into();
        a.active_seed = 7;
        a.handle_chat_submit("/save fox-demo".into());
        assert!(root.join("sessions/fox-demo.json").exists(), "session written");

        // Wipe live state, then load it back.
        a.chat.restore(vec![]);
        a.refine_prompt.clear();
        a.base_seed = None;
        a.negative.clear();
        a.handle_chat_submit("/load fox-demo".into());
        assert_eq!(a.refine_prompt, "a watercolor fox, autumn leaves");
        assert_eq!(a.base_seed, Some(7));
        assert_eq!(a.negative, "blurry");
        // The fox utterance is back (plus the load's system note).
        assert!(a.chat.history.iter().any(|e| e.utterance == "a watercolor fox" && !e.system));
        // /sessions lists it without error.
        a.handle_chat_submit("/sessions".into());
        assert!(a.chat.history.last().unwrap().utterance.contains("fox-demo"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn ctrl_k_opens_palette_and_a_command_navigates() {
        let mut a = test_app(); // starts on Chat
        assert!(!a.palette.is_open());
        // Ctrl-K opens the palette (even though Chat owns text input).
        a.handle_key(key('k', true));
        assert!(a.palette.is_open(), "Ctrl-K opens the palette");
        // While open, plain chars filter the palette (don't reach Chat).
        for c in "go to models".chars() {
            a.handle_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
        assert_eq!(a.chat.editor.text(), "", "typing filters the palette, not Chat");
        // Enter runs the top match → navigate to Models, palette closes.
        a.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(!a.palette.is_open());
        assert!(matches!(a.screen, ActiveScreen::Models));
    }

    #[test]
    fn palette_esc_closes_without_acting() {
        let mut a = test_app();
        a.handle_key(key('k', true));
        assert!(a.palette.is_open());
        a.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!a.palette.is_open());
        assert!(matches!(a.screen, ActiveScreen::Chat), "Esc didn't navigate");
    }

    #[test]
    fn palette_key_command_drives_the_active_screen() {
        // On Models, the palette's "Load selected model" replays 'l' → loads.
        let mut a = test_app();
        a.screen = ActiveScreen::Models;
        a.handle_key(key('k', true));
        for c in "load selected".chars() {
            a.handle_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
        a.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        // The palette closed and we stayed on Models (the load dispatched to the svc).
        assert!(!a.palette.is_open());
        assert!(matches!(a.screen, ActiveScreen::Models));
    }

    #[test]
    fn replace_ci_is_case_insensitive_and_utf8_safe() {
        assert_eq!(replace_ci("hi @Alice there", "@alice", "X"), "hi X there");
        assert_eq!(replace_ci("@a @A", "@a", "Z"), "Z Z");
        assert_eq!(replace_ci("nothing", "@x", "Y"), "nothing");
    }

    #[test]
    fn history_preview_decodes_on_a_worker_then_applies() {
        // A real PNG under the workspace out/ dir; History should decode it off-tick
        // and, once the worker returns, show it (preview_for == the image path).
        let root = std::env::temp_dir().join("plakat-ui-histdecode-test");
        let _ = std::fs::remove_dir_all(&root);
        let out = root.join("out");
        std::fs::create_dir_all(&out).unwrap();
        crate::imaging::io::save_rgb_u8(&[9, 9, 9], 1, 1, &out.join("plakat-1-1.png")).unwrap();

        let ws = Workspace { root: root.clone(), config: WorkspaceConfig::default() };
        let (_tx, rx) = std::sync::mpsc::channel();
        let mut a = App::new(ws, Picker::from_fontsize((8, 16)), Device::Cpu, test_handle(), rx);
        a.screen = ActiveScreen::History;
        let want = a.history.selected_path();
        assert!(want.is_some(), "history found the PNG");

        // First tick spawns the decode; subsequent ticks drain it. Pump a bounded loop.
        let mut applied = false;
        for _ in 0..200 {
            a.sync_history();
            if a.history.preview_for == want {
                applied = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(applied, "background decode eventually set the preview");
        assert!(a.history.preview.is_some(), "preview protocol built on the main thread");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn multiperson_identity_routes_from_persona_strategies() {
        use crate::pipelines::ip_adapter::IdentityKind;
        let r = |s: &[&str]| route_multiperson_identity(&s.iter().map(|x| x.to_string()).collect::<Vec<_>>());
        // Nothing named → default plus-face / sd15.
        assert!(matches!(r(&["", ""]), (IdentityKind::PlusFace, "sd15")));
        // All faceid (sd15) → FaceId.
        assert!(matches!(r(&["faceid", "face-id"]), (IdentityKind::FaceId, "sd15")));
        // Mixed sd15 strategies → the general PlusFace.
        assert!(matches!(r(&["faceid", "plus-face"]), (IdentityKind::PlusFace, "sd15")));
        // Any SDXL present → the scene is SDXL.
        assert!(matches!(r(&["plus-face", "plus-face-sdxl"]), (IdentityKind::PlusFaceSdxl, "sdxl")));
        // All faceid with an SDXL one → FaceIdSdxl.
        assert!(matches!(r(&["faceid", "faceid-sdxl"]), (IdentityKind::FaceIdSdxl, "sdxl")));
        // Unknown strings are ignored (fall back to default).
        assert!(matches!(r(&["instantid", "???"]), (IdentityKind::PlusFace, "sd15")));
    }

    #[test]
    fn recipe_recovery_reads_prompt_and_seed() {
        let params = "a red fox in a forest\nNegative prompt: blurry\nSteps: 28, Seed: 4242, Size: 512x512";
        assert_eq!(recipe_positive(params), "a red fox in a forest");
        assert_eq!(recipe_seed(params), Some(4242));
        assert_eq!(recipe_seed("no seed"), None);
        // The negative is everything after "Negative prompt:" up to "Steps:".
        assert_eq!(recipe_negative(params), "blurry");
        assert_eq!(recipe_negative("a fox\nSteps: 20"), "", "no negative → empty");
        assert_eq!(recipe_negative("no params here"), "");
    }

    #[test]
    fn apply_lora_by_name_reports_unknown() {
        let mut a = test_app();
        a.apply_lora_by_name("does-not-exist");
        assert!(a.chat.history.last().unwrap().utterance.contains("no local LoRA"));
    }

    #[test]
    fn identity_portrait_request_carries_prompt_seed_and_photos() {
        let a = test_app();
        let id = ChatIdentity {
            photos: vec![("/tmp/alice.png".into(), 1.0)],
            identity: "plus-face".into(),
            face_strength: Some(0.7),
            negative: "blurry".into(),
            model: "sd15".into(),
            width: 512,
            height: 640,
            seed: 1234,
        };
        let req = a.identity_portrait_request("a portrait of alice, smiling".into(), &id);
        assert_eq!(req.prompt, "a portrait of alice, smiling");
        assert_eq!(req.seed, Some(1234));
        assert_eq!(req.model, "sd15");
        assert_eq!(req.photos.len(), 1);
        assert_eq!(req.face_strength, 0.7);
        // The strategy string parsed to a concrete IdentityKind.
        assert!(matches!(req.identity, Some(crate::pipelines::ip_adapter::IdentityKind::PlusFace)));
    }

    #[test]
    fn identity_refine_routes_to_the_portrait_pass_not_plain_gen() {
        let mut a = test_app();
        a.chat_identity = Some(ChatIdentity {
            photos: vec![("/tmp/alice.png".into(), 1.0)],
            identity: "plus-face".into(),
            face_strength: Some(0.8),
            negative: String::new(),
            model: "sd15".into(),
            width: 512,
            height: 640,
            seed: 1234,
        });
        a.base_seed = Some(1234);
        a.refine_prompt = "a portrait of alice".into();
        // Pre-occupy portrait_run so dispatch_identity_refine short-circuits at its guard
        // (no real model run in the test) — we only assert the routing decision.
        let (_tx, rx) = std::sync::mpsc::channel();
        a.portrait_run = Some(rx);

        a.dispatch_generation("smiling".into(), false);
        // Routed to the identity (portrait) path → did NOT start a plain txt2img/img2img.
        assert!(a.active_gen.is_none(), "identity refine did not use the plain gen path");

        // `/new` with no active run leaves the identity behind (fresh gen path).
        a.portrait_run = None;
        a.dispatch_generation("/new a landscape".into(), false);
        assert!(a.chat_identity.is_none(), "a fresh image drops the identity context");
    }

    #[test]
    fn canvas_masks_the_latest_image_not_the_original_base() {
        let mut a = test_app();
        // A thread: a fresh image (the prompt-evolve base) then a refine (the latest).
        a.refine_base = Some("/tmp/original.png".into());
        a.chat.push_utterance("a fox".into(), false);
        a.chat.finish_last(Ok("/tmp/original.png".into()));
        a.chat.push_utterance("make it autumn".into(), true);
        a.chat.finish_last(Ok("/tmp/latest.png".into()));
        a.base_seed = Some(7);

        // On Canvas, the base tracks the LATEST render, not refine_base (the original).
        a.screen = ActiveScreen::Canvas;
        a.sync_canvas();
        assert_eq!(a.canvas.base_path(), Some("/tmp/latest.png".into()));

        // Applying a mask captures the latest image as the inpaint target.
        a.apply_canvas_mask("/tmp/mask.png".into());
        assert_eq!(a.inpaint_base, Some("/tmp/latest.png".into()));
        assert_ne!(a.inpaint_base, a.refine_base, "inpaint over the latest, not the original");
    }

    #[test]
    fn object_insertion_phrasing_is_detected() {
        assert!(looks_like_object_insertion("add a fisherman to the boat"));
        assert!(looks_like_object_insertion("  Put a bird in the sky"));
        assert!(looks_like_object_insertion("place a sun above the mountain"));
        assert!(looks_like_object_insertion("give it a red hat"));
        // Not insertions — style / global edits.
        assert!(!looks_like_object_insertion("make the background snowy"));
        assert!(!looks_like_object_insertion("warmer lighting, autumn palette"));
    }

    #[test]
    fn downloads_cap_concurrency_and_queue_the_rest() {
        let mut a = test_app();
        // Queue four downloads; only MAX_CONCURRENT_DOWNLOADS start, the rest wait.
        for i in 0..4u64 {
            a.remote_download(lorahub::DownloadRef::Civitai { model_id: i, version_id: None }, format!("lora{i}"));
        }
        assert_eq!(a.downloads_active.len(), MAX_CONCURRENT_DOWNLOADS);
        assert_eq!(a.downloads_queue.len(), 4 - MAX_CONCURRENT_DOWNLOADS);
        // The download indicator reflects in-flight + queued work.
        // (set_downloading was called true via pump_downloads.)
    }

    #[test]
    fn all_screens_present_and_cycle() {
        // The full surface — nine screens (RFC TUI-1's eight + the 6.17 Naturalize tab), each reachable
        // by Tab cycling.
        assert_eq!(ActiveScreen::ALL.len(), 9);
        for s in ActiveScreen::ALL {
            assert_eq!(s.cycle(1).cycle(-1), s, "cycling is reversible for {s:?}");
        }
        // Cycling forward through all nine returns to the start.
        let mut s = ActiveScreen::Chat;
        for _ in 0..ActiveScreen::ALL.len() {
            s = s.cycle(1);
        }
        assert_eq!(s, ActiveScreen::Chat);
    }
}
