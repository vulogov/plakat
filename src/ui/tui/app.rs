//! The TUI application shell — `App`, the event loop, and global chrome (RFC
//! TUI-1 §4–§5). This Phase-1 increment is the navigable frame: a tab bar, a
//! status bar, `Ctrl-1..8` (or plain `1..8`) screen switching, and per-screen
//! placeholders. The Chat + Models screen bodies, services, and channel draining
//! land in the next increments.

use anyhow::Result;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Block, Borders, Paragraph, Tabs},
};
use candle_core::Device;
use ratatui_image::picker::Picker;
use tokio::runtime::Handle;

use std::sync::mpsc::Receiver;

use crate::cli::scenario::ScenarioEvent;
use crate::pipelines::gen_channel::{CancelFlag, GenMessage};

use super::output::OutputPane;
use super::screens::chat::{ChatAction, ChatState, ChatStatus};
use super::screens::models::ModelsState;
use super::screens::scenarios::{ScenariosAction, ScenariosState};
use super::services::model_service::ModelService;
use super::workspace::Workspace;

/// img2img strength used when the user opts INTO image-anchored refinement via
/// `/strength` without a value (or as the default for that mode).
const DEFAULT_ANCHOR_STRENGTH: f32 = 0.6;

/// The eight screens (RFC §1). Release 1 implements Chat + Models; the rest show a
/// placeholder until their cycle.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ActiveScreen {
    Chat,
    Models,
    Scenarios,
    History,
    LoraHub,
    People,
    PromptWorkspace,
    Canvas,
}

impl ActiveScreen {
    const ALL: [ActiveScreen; 8] = [
        Self::Chat,
        Self::Models,
        Self::Scenarios,
        Self::History,
        Self::LoraHub,
        Self::People,
        Self::PromptWorkspace,
        Self::Canvas,
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

    /// Whether this screen has a real body yet (Release 1: Chat + Models;
    /// Release 2: Scenarios).
    fn implemented(self) -> bool {
        matches!(self, Self::Chat | Self::Models | Self::Scenarios)
    }
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
    // Shared Output pane (messages + live progress, fed by the rerouted sink).
    pub output: OutputPane,
    progress_rx: Receiver<String>,
    // Background services.
    pub model_svc: ModelService,
    rt: Handle,
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
    // The in-flight scenario run (its terminal-result channel).
    scenario_run: Option<Receiver<Result<(), String>>>,
    // Live per-task events from the in-flight scenario run (RUNNER board).
    scenario_events: Option<Receiver<ScenarioEvent>>,
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
        Self {
            scenarios,
            chat: ChatState::new(),
            models: ModelsState::new(),
            output: OutputPane::new(),
            progress_rx,
            model_svc: ModelService::spawn(device, rt.clone()),
            rt,
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
            self.model_svc.load(default);
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
        let res = self.event_loop(&mut terminal);
        if enhanced {
            let _ = crossterm::execute!(std::io::stdout(), crossterm::event::PopKeyboardEnhancementFlags);
        }
        ratatui::restore();
        res
    }

    fn event_loop(&mut self, terminal: &mut ratatui::DefaultTerminal) -> Result<()> {
        while !self.should_quit {
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
        }
        Ok(())
    }

    fn handle_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        // ── Always-global keys (work even while a text input is focused) ──
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
            KeyCode::Char(c @ '1'..='8') if ctrl => {
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
            KeyCode::Tab => {
                self.screen = self.screen.cycle(1);
                return;
            }
            _ => {}
        }

        // ── Chat owns text input: plain chars / Enter / Backspace go to it. ──
        if self.screen == ActiveScreen::Chat {
            if let ChatAction::Submit(prompt) = self.chat.handle_key(key) {
                self.handle_chat_submit(prompt);
            }
            return;
        }

        // ── The Scenarios EDITOR / RUNNER own the keyboard (type into the buffer /
        //    capture Esc) — route everything to them while active. ──
        if self.screen == ActiveScreen::Scenarios && self.scenarios.captures_input() {
            if let ScenariosAction::Run(path) = self.scenarios.handle_key(key) {
                self.run_scenario(path);
            }
            return;
        }

        // ── Non-input screens: plain digits switch, q quits, else delegate. ──
        match key.code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char(c @ '1'..='8') => {
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
                        self.model_svc.load(alias);
                    }
                }
                KeyCode::Char('u' | 'U') => self.model_svc.unload(),
                _ => {
                    self.models.handle_key(key);
                }
            },
            ActiveScreen::Scenarios => {
                if let ScenariosAction::Run(path) = self.scenarios.handle_key(key) {
                    self.run_scenario(path);
                }
            }
            _ => {}
        }
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

        let (tx, rx) = std::sync::mpsc::channel();
        let (etx, erx) = std::sync::mpsc::channel();
        let rt = self.rt.clone();
        std::thread::spawn(move || {
            let args = crate::cli::scenario::ScenarioArgs {
                file: path,
                dry_run: false,
                resume: false,
                force: false,
                only: Vec::new(),
                limit: 0,
                json_summary: None,
            };
            let result = rt.block_on(crate::cli::scenario::run_with_events(args, Some(etx)));
            let _ = tx.send(result.map_err(|e| format!("{e:#}")));
        });
        self.scenario_run = Some(rx);
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
        }
    }

    /// Route a submitted Chat line: slash commands (`/negative`, `/enhance`, `/new`)
    /// or a plain prompt. `/negative` updates session state without generating.
    fn handle_chat_submit(&mut self, text: String) {
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
        // `/enhance <prompt>` AI-expands the prompt, then generates fresh.
        if let Some(rest) = text.strip_prefix("/enhance") {
            let p = rest.trim().to_string();
            if !p.is_empty() {
                self.dispatch_generation(p, true);
            }
            return;
        }
        self.dispatch_generation(text, false);
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
        // Enhancing crafts a fresh detailed prompt, so it never refines.
        let refine = !force_fresh && !enhance && self.base_seed.is_some();
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
        let init_image = if refine {
            self.refine_strength.and(self.refine_base.clone())
        } else {
            None
        };
        self.active_is_refine = refine;
        self.active_full_prompt = full_prompt.clone();

        // Show the user's own words in the history (not the accumulated prompt).
        self.chat.push_utterance(edit.clone(), refine);
        // Generate at the LOADED model's native square resolution (sd15=512,
        // sd21=768, sdxl=1024) — always Metal-safe, unlike a fixed workspace size
        // which OOMs SD1.5. A per-model size override is a future item.
        let n = self
            .models
            .loaded_alias()
            .map(crate::capability::native_res)
            .unwrap_or(768);
        let steps = self.workspace.config.default_steps;
        let guidance = self.workspace.config.default_guidance;
        let preview_every = self.workspace.config.preview_every_n_steps;
        let out_dir = self.workspace.out_dir().join("chat");
        // Seed priority: an explicit `/seed` pin > the conversation's stable seed (so
        // prompt-evolve refines keep composition) > a fresh random seed.
        let seed = self
            .fixed_seed
            .or(if refine { self.base_seed } else { None })
            .unwrap_or_else(|| rand::random::<u32>() as u64);
        self.active_seed = seed;
        // Anchored mode passes its strength; prompt-evolve ignores it (init is None).
        let strength = self.refine_strength.unwrap_or(0.0);
        let enhancer = if enhance { Some(self.workspace.config.enhancer.clone()) } else { None };
        let (rx, cancel) = self.model_svc.generate(
            full_prompt,
            self.negative.clone(),
            n,
            n,
            steps,
            guidance,
            seed,
            out_dir,
            preview_every,
            init_image,
            strength,
            enhancer,
        );
        self.active_gen = Some((rx, cancel));
        self.chat.status = ChatStatus::Generating { step: 0, total: steps as u32, refine };
    }

    /// Drain the active generation's messages → Chat status, inline preview/final
    /// image (built here because it needs the Picker), and history.
    fn drain_generation(&mut self) {
        let mut msgs = Vec::new();
        if let Some((rx, _)) = &self.active_gen {
            while let Ok(m) = rx.try_recv() {
                msgs.push(m);
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
        if finished {
            self.active_gen = None;
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
            other => {
                let body = format!("[{}] — coming in a later release (RFC TUI-1).", other.title());
                let block = Block::default().borders(Borders::ALL).title(other.title());
                f.render_widget(Paragraph::new(body).block(block), area);
            }
        }
    }

    fn render_status_bar(&self, f: &mut Frame, area: Rect) {
        // In a text-input mode (Chat, or the Scenarios editor) plain keys type, so
        // advertise only the input-safe switches.
        let input_mode = self.screen == ActiveScreen::Chat
            || (self.screen == ActiveScreen::Scenarios && self.scenarios.captures_input());
        let nav = if input_mode {
            "Ctrl-1..8 / Tab switch · Ctrl-Q quit"
        } else {
            "1-8 / Tab switch · Ctrl-Q quit"
        };
        let txt = format!(" {} · {nav} ", self.workspace.config.name);
        let bar = Paragraph::new(txt).style(Style::new().bg(Color::DarkGray).fg(Color::White));
        f.render_widget(bar, area);
    }
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
        // Shift-Tab as Tab+SHIFT (kbd-protocol encoding) → also backward (wraps).
        a.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT));
        assert!(matches!(a.screen, ActiveScreen::Canvas));
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
    fn esc_does_not_quit() {
        // Regression: Esc was quitting, so legacy Ctrl-3 (== Esc byte) killed the app.
        let mut a = test_app();
        a.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!a.should_quit);
    }

    #[test]
    fn release1_screens_are_implemented() {
        assert!(ActiveScreen::Chat.implemented());
        assert!(ActiveScreen::Models.implemented());
        assert!(ActiveScreen::Scenarios.implemented());
        assert!(!ActiveScreen::History.implemented());
        assert_eq!(ActiveScreen::ALL.len(), 8);
    }
}
