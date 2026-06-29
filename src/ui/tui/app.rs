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
    widgets::{Paragraph, Tabs},
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
    /// Release 2: Scenarios + History).
    fn implemented(self) -> bool {
        // All eight screens have a real body — the RFC TUI-1 surface is complete.
        true
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
    pub history: HistoryState,
    pub people: PeopleState,
    pub lorahub: LoraHubState,
    pub prompts: PromptsState,
    pub canvas: CanvasState,
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
    // In-flight remote (Civitai / HF) search + download for the LoRA Hub.
    remote_search: Option<Receiver<Result<Vec<lorahub::RemoteHit>, String>>>,
    remote_download: Option<Receiver<Result<String, String>>>,
    // In-flight LLM LoRA assessment: (item key, assessment text).
    lora_assess: Option<Receiver<(String, String)>>,
    // In-flight LLM recommend-for-context (LoRA Hub search tabs).
    lora_recommend: Option<Receiver<String>>,
    // In-flight Prompt Workspace LLM compile.
    prompt_compile: Option<Receiver<Result<String, String>>>,
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
            remote_search: None,
            remote_download: None,
            lora_assess: None,
            lora_recommend: None,
            prompt_compile: None,
            active_loras: Vec::new(),
            chat_mask: None,
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
            self.drain_route();
        }
        Ok(())
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
    }

    /// Lazily decode the selected History image into a preview (and read its recipe),
    /// but only while History is the active screen and only when the selection
    /// changed — so navigating the list never blocks the event loop on every frame.
    fn sync_history(&mut self) {
        if self.screen != ActiveScreen::History {
            return;
        }
        self.history.sync_detail();
        let sel = self.history.selected_path();
        if sel != self.history.preview_for {
            self.history.preview = sel
                .as_ref()
                .and_then(|p| image::open(p).ok())
                .map(|img| self.picker.new_resize_protocol(img));
            self.history.preview_for = sel;
        }
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

        // ── The Canvas owns the keyboard (preset letters + Space painting). ──
        if self.screen == ActiveScreen::Canvas {
            if let canvas::CanvasAction::MaskReady(path) = self.canvas.handle_key(key) {
                self.apply_canvas_mask(path);
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
                        self.load_model(alias);
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
            ActiveScreen::History => {
                if let HistoryAction::Continue { path, prompt, seed } = self.history.handle_key(key) {
                    self.continue_from_image(path, prompt, seed);
                }
            }
            ActiveScreen::People => match self.people.handle_key(key) {
                people::PeopleAction::Generate(spec) => self.quick_generate(spec),
                people::PeopleAction::GenerateMulti(specs) => self.quick_generate_multi(specs),
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
            _ => {}
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

    /// Deterministic structural compile (no LLM) of the Prompt Workspace buffer,
    /// recomputed only when the text changed. compile_to_string with no_enhance +
    /// no_negative does no network, so a `block_on` here is instant.
    fn sync_prompts(&mut self) {
        if self.screen != ActiveScreen::PromptWorkspace || self.prompts.compiling {
            return;
        }
        let src = self.prompts.editor_text();
        if self.prompts.last_compiled_src.as_deref() == Some(src.as_str()) {
            return;
        }
        let opts = self.compile_opts(&self.prompts.buffer_name(), true);
        match self.rt.block_on(crate::compile::compile_to_string(&src, &opts)) {
            Ok(hjson) => {
                self.prompts.compiled = hjson;
                self.prompts.compile_err = None;
            }
            Err(e) => self.prompts.compile_err = Some(format!("{e:#}")),
        }
        self.prompts.last_compiled_src = Some(src);
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
        }
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
        let provider = crate::prompt::resolve_provider_label(&self.workspace.config.enhancer);
        let (tx, rx) = std::sync::mpsc::channel();
        let rt = self.rt.clone();
        std::thread::spawn(move || {
            const SYSTEM: &str = "You are a concise assistant describing Stable Diffusion \
                LoRAs. Reply in ONE plain sentence, no preamble, no markdown.";
            let text = rt
                .block_on(crate::prompt::complete(&provider, SYSTEM, &prompt, &crate::prompt::EnhanceArgs::default()))
                .unwrap_or_else(|e| format!("(assessment failed: {e:#})"));
            let _ = tx.send((key, text.trim().to_string()));
        });
        self.lora_assess = Some(rx);
    }

    /// Run a LoRA search (Civitai or HF) on a background thread; results land in the Hub.
    fn remote_search(&mut self, source: lorahub::RemoteSource, query: String) {
        if self.remote_search.is_some() {
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
            let _ = tx.send(result);
        });
        self.remote_search = Some(rx);
    }

    /// Download a LoRA (Civitai → its cache; HF → copied into the workspace loras/
    /// dir) on a background thread; on success rescan LOCAL.
    fn remote_download(&mut self, dl: lorahub::DownloadRef, title: String) {
        if self.remote_download.is_some() {
            return;
        }
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
        self.remote_download = Some(rx);
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
        if let Some(rx) = &self.remote_download {
            match rx.try_recv() {
                Ok(Ok(name)) => {
                    self.lorahub.set_remote_status(format!("✓ downloaded {name} — see LOCAL"));
                    self.lorahub.rescan();
                    self.remote_download = None;
                }
                Ok(Err(e)) => {
                    self.lorahub.set_remote_status(format!("✗ download failed: {e}"));
                    self.remote_download = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.remote_download = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
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
        // A mask needs a base to inpaint over; ensure a refine is triggered.
        if self.base_seed.is_none() {
            self.base_seed = Some(rand::random::<u32>() as u64);
        }
        self.chat.refine_armed = true;
        self.chat.push_system("inpaint mask set from Canvas — type an edit for the masked region (one-shot)".into());
        self.screen = ActiveScreen::Chat;
    }

    /// Keep the Canvas base in sync with the current Chat base image (lazy preview).
    fn sync_canvas(&mut self) {
        if self.screen != ActiveScreen::Canvas {
            return;
        }
        if self.refine_base != self.canvas.base_path() {
            let dims = self.refine_base.as_ref().and_then(|p| image::open(p).ok()).map(|i| (i.width(), i.height()));
            self.canvas.set_base(self.refine_base.clone(), dims);
        }
        let sel = self.refine_base.clone();
        if sel != self.canvas.preview_for {
            self.canvas.preview = sel
                .as_ref()
                .and_then(|p| image::open(p).ok())
                .map(|img| self.picker.new_resize_protocol(img));
            self.canvas.preview_for = sel;
        }
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
        let photos: Vec<crate::pipelines::ip_adapter::WeightedPhoto> = spec
            .photos
            .iter()
            .map(|(p, w)| crate::pipelines::ip_adapter::WeightedPhoto { path: p.clone(), weight: Some(*w) })
            .collect();
        let seed = rand::random::<u32>() as u64;
        let out_dir = self.workspace.out_dir().join("people");
        let req = crate::pipelines::portrait::Request {
            prompt: prompt.clone(),
            negative: spec.negative,
            photos,
            model,
            width: w,
            height: h,
            count: 1,
            steps: 30,
            guidance: 7.0,
            seed: Some(seed),
            out_dir: out_dir.clone(),
            device: self.device.clone(),
            loras: Vec::new(),
            lora_scale: 1.0,
            scheduler: crate::pipelines::scheduler::SchedulerKind::default(),
            refine: None,
            refine_strength: 0.0,
            face_strength: spec.face_strength.unwrap_or(0.8),
            face_bbox: None,
            face_landmarks: None,
            identity,
            shared_clip_h: None,
            controls: Vec::new(),
        };

        self.chat.push_system(format!("generating portrait of {} — opens in Chat…", spec.label));
        self.portrait_prompt = prompt;
        let (tx, rx) = std::sync::mpsc::channel();
        let rt = self.rt.clone();
        std::thread::spawn(move || {
            let result = rt
                .block_on(crate::pipelines::portrait::run(req))
                .map(|_| out_dir.join(format!("plakat-portrait-{seed}.png")))
                .map_err(|e| format!("{e:#}"));
            let _ = tx.send(result);
        });
        self.portrait_run = Some(rx);
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
            model: "sd15".into(),
            identity: crate::pipelines::ip_adapter::IdentityKind::PlusFace,
            style: None,
            negative: String::new(),
            layout_provider: "none".into(),
            enhancer: None,
            width: 768,
            height: 768,
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

        self.chat.push_system(format!("generating multiperson scene: {} — opens in Chat…", labels.join(", ")));
        self.portrait_prompt = scene;
        let (tx, rx) = std::sync::mpsc::channel();
        let rt = self.rt.clone();
        std::thread::spawn(move || {
            let result = rt
                .block_on(crate::pipelines::multiperson::run(req))
                .map(|_| out_dir.join(format!("plakat-multiperson-{seed}.png")))
                .map_err(|e| format!("{e:#}"));
            let _ = tx.send(result);
        });
        self.portrait_run = Some(rx);
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
            match result {
                // Portraits carry no Chat recipe → image-anchored continuation.
                Ok(path) => self.continue_from_image(path, self.portrait_prompt.clone(), None),
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
        // A Canvas mask makes this ONE turn an inpaint (img2img over the base, masked)
        // regardless of the sticky mode — then it's consumed. Otherwise the mode rules:
        // anchored (`/strength`) = img2img over the base; prompt-evolve = txt2img.
        let inpaint = refine && self.chat_mask.is_some() && self.refine_base.is_some();
        let init_image = if inpaint {
            self.refine_base.clone()
        } else if refine {
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
        // Inpaint runs hot (the masked region regenerates); anchored uses its strength;
        // prompt-evolve ignores it (init is None).
        let strength = if inpaint { INPAINT_STRENGTH } else { self.refine_strength.unwrap_or(0.0) };
        let mask = if inpaint { self.chat_mask.take() } else { None }; // one-shot
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
            ActiveScreen::History => self.history.render(f, area),
            ActiveScreen::People => self.people.render(f, area),
            ActiveScreen::LoraHub => self.lorahub.render(f, area),
            ActiveScreen::PromptWorkspace => self.prompts.render(f, area),
            ActiveScreen::Canvas => self.canvas.render(f, area),
        }
    }

    fn render_status_bar(&self, f: &mut Frame, area: Rect) {
        // In a text-input mode (Chat, or the Scenarios editor) plain keys type, so
        // advertise only the input-safe switches.
        let input_mode = self.screen == ActiveScreen::Chat
            || (self.screen == ActiveScreen::Scenarios && self.scenarios.captures_input())
            || (self.screen == ActiveScreen::LoraHub && self.lorahub.captures_input())
            || (self.screen == ActiveScreen::PromptWorkspace && self.prompts.captures_input());
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
    fn release1_screens_are_implemented() {
        assert!(ActiveScreen::Chat.implemented());
        assert!(ActiveScreen::Models.implemented());
        assert!(ActiveScreen::Scenarios.implemented());
        assert!(ActiveScreen::History.implemented());
        assert!(ActiveScreen::People.implemented());
        // Every screen has a real body now — the RFC TUI-1 surface is complete.
        assert!(ActiveScreen::ALL.iter().all(|s| s.implemented()));
        assert_eq!(ActiveScreen::ALL.len(), 8);
    }
}
