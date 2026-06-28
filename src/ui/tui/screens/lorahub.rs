//! LoRA Hub screen (RFC TUI-1 §10). Release 3 LOCAL tab: scan the workspace + global
//! LoRA dirs for `.safetensors`, read each file's safetensors header to infer its
//! base family + rank (and a `.plakat.hjson` sidecar for trigger words / notes), and
//! flag compatibility against the currently-loaded model. The CIVITAI / HUGGINGFACE
//! search tabs and the LLM recommendation features are later releases.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use serde::Deserialize;

use crate::preset::discovery::BaseFamily;

/// One local LoRA file.
struct LoraInfo {
    path: PathBuf,
    name: String,
    family: Option<BaseFamily>,
    rank: Option<u32>,
    size: u64,
    triggers: Vec<String>,
    notes: String,
    /// A short label for which dir it came from.
    location: String,
}

/// `.plakat.hjson` sidecar fields we surface (the full schema is RFC §10).
#[derive(Deserialize, Default)]
struct Sidecar {
    #[serde(default)]
    trigger_words: Vec<String>,
    #[serde(default)]
    base_model: String,
    #[serde(default)]
    notes: String,
}

/// Which sub-tab is showing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Local,
    Civitai,
    HuggingFace,
}

/// A remote search source.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RemoteSource {
    Civitai,
    HuggingFace,
}

/// Remote-tab focus: editing the query vs browsing results.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Search,
    Results,
}

/// What the App needs to download a hit.
#[derive(Clone)]
pub enum DownloadRef {
    Civitai { model_id: u64, version_id: Option<u64> },
    Hf { repo: String },
}

/// One remote search hit (filled by the App from the Civitai / HF API).
pub struct RemoteHit {
    pub title: String,
    /// Base model (Civitai) or pipeline tag (HF), shown dim.
    pub subtitle: String,
    pub downloads: u64,
    pub dl: DownloadRef,
}

/// What the App should do after a key.
pub enum LoraHubAction {
    None,
    /// `A` — toggle this LoRA in Chat's active set (App reloads the model).
    /// `compatible` is false when its family clashes with the loaded model.
    ToggleApply { path: PathBuf, compatible: bool },
    /// Remote Enter — run a search (App does the async call).
    Search { source: RemoteSource, query: String },
    /// Remote `D`/Enter on a result — download it (App does the async call).
    Download { dl: DownloadRef, title: String },
    /// `R` — ask the LLM to assess the selected LoRA. `key` identifies the item
    /// (path for LOCAL, "hit:title" for remote); `prompt` is the user message.
    Assess { key: String, prompt: String },
}

pub struct LoraHubState {
    dirs: Vec<(PathBuf, String)>,
    loras: Vec<LoraInfo>,
    selected: usize,
    /// Family of the currently-loaded model (set by the App), for compatibility.
    loaded_family: Option<BaseFamily>,
    /// Paths currently applied to Chat (mirrored from the App each tick).
    applied: std::collections::HashSet<PathBuf>,
    // ── Remote (CIVITAI / HUGGINGFACE) tabs ──
    tab: Tab,
    phase: Phase,
    query: String,
    hits: Vec<RemoteHit>,
    hit_sel: usize,
    remote_status: String,
    /// LLM assessments keyed by item key; `assessing` is the in-flight key.
    assessments: HashMap<String, String>,
    assessing: Option<String>,
}

impl LoraHubState {
    pub fn new(dirs: Vec<(PathBuf, String)>) -> Self {
        let mut s = Self {
            dirs,
            loras: Vec::new(),
            selected: 0,
            loaded_family: None,
            applied: std::collections::HashSet::new(),
            tab: Tab::Local,
            phase: Phase::Search,
            query: String::new(),
            hits: Vec::new(),
            hit_sel: 0,
            remote_status: "type a query · Enter to search".into(),
            assessments: HashMap::new(),
            assessing: None,
        };
        s.rescan();
        s
    }

    /// The App delivers an LLM assessment for `key` (or an error string).
    pub fn set_assessment(&mut self, key: String, text: String) {
        if self.assessing.as_deref() == Some(key.as_str()) {
            self.assessing = None;
        }
        self.assessments.insert(key, text);
    }

    /// True while a remote query box is focused — the App routes all keys here.
    pub fn captures_input(&self) -> bool {
        self.tab != Tab::Local && self.phase == Phase::Search
    }

    /// The App sets the search results (and clears the searching status).
    pub fn set_remote_hits(&mut self, hits: Vec<RemoteHit>) {
        self.remote_status = if hits.is_empty() { "no results".into() } else { format!("{} results", hits.len()) };
        self.hits = hits;
        self.hit_sel = 0;
    }

    pub fn set_remote_status(&mut self, status: impl Into<String>) {
        self.remote_status = status.into();
    }

    fn source(&self) -> RemoteSource {
        match self.tab {
            Tab::HuggingFace => RemoteSource::HuggingFace,
            _ => RemoteSource::Civitai,
        }
    }

    /// The App calls this each tick with the loaded model's family so the
    /// compatibility column reflects what's actually loaded.
    pub fn set_loaded_family(&mut self, family: Option<BaseFamily>) {
        self.loaded_family = family;
    }

    /// The App calls this each tick with the paths currently applied to Chat.
    pub fn set_applied(&mut self, paths: &[PathBuf]) {
        self.applied = paths.iter().cloned().collect();
    }

    fn selected_path(&self) -> Option<PathBuf> {
        self.loras.get(self.selected).map(|l| l.path.clone())
    }

    pub fn rescan(&mut self) {
        let mut loras = Vec::new();
        for (dir, label) in &self.dirs {
            collect(dir, label, &mut loras, 0);
        }
        loras.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        self.loras = loras;
        if self.selected >= self.loras.len() {
            self.selected = self.loras.len().saturating_sub(1);
        }
    }

    fn next(&mut self) {
        if !self.loras.is_empty() {
            self.selected = (self.selected + 1) % self.loras.len();
        }
    }

    fn prev(&mut self) {
        if !self.loras.is_empty() {
            self.selected = (self.selected + self.loras.len() - 1) % self.loras.len();
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> LoraHubAction {
        match self.tab {
            Tab::Local => self.handle_local_key(key),
            _ => self.handle_remote_key(key),
        }
    }

    /// Move to an adjacent tab (LOCAL ↔ CIVITAI ↔ HUGGINGFACE), resetting the remote
    /// view to its search box.
    fn switch_tab(&mut self, delta: i32) {
        let order = [Tab::Local, Tab::Civitai, Tab::HuggingFace];
        let i = order.iter().position(|t| *t == self.tab).unwrap_or(0) as i32;
        let n = order.len() as i32;
        self.tab = order[(i + delta).rem_euclid(n) as usize];
        if self.tab != Tab::Local {
            self.phase = Phase::Search;
            self.query.clear();
            self.hits.clear();
            self.remote_status = "type a query · Enter to search".into();
        }
    }

    fn handle_local_key(&mut self, key: KeyEvent) -> LoraHubAction {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.next(),
            KeyCode::Up | KeyCode::Char('k') => self.prev(),
            KeyCode::Char('r') => self.rescan(),
            // R — LLM assessment of the selected LoRA.
            KeyCode::Char('R') => {
                if let Some(l) = self.loras.get(self.selected) {
                    let key = l.path.display().to_string();
                    let prompt = format!(
                        "Stable Diffusion LoRA file '{}', base model {}, trigger words: {}. \
                         In ONE sentence say what this LoRA is for and when to use it.",
                        l.name,
                        l.family.map(family_label).unwrap_or("unknown"),
                        if l.triggers.is_empty() { "(none)".into() } else { l.triggers.join(", ") },
                    );
                    self.assessing = Some(key.clone());
                    return LoraHubAction::Assess { key, prompt };
                }
            }
            KeyCode::Right | KeyCode::Char('l') => self.switch_tab(1),
            KeyCode::Left | KeyCode::Char('h') => self.switch_tab(-1),
            KeyCode::Char('a' | 'A') | KeyCode::Enter => {
                if let Some(l) = self.loras.get(self.selected) {
                    let compatible = self.compatible(l) != Some(false);
                    if let Some(path) = self.selected_path() {
                        return LoraHubAction::ToggleApply { path, compatible };
                    }
                }
            }
            _ => {}
        }
        LoraHubAction::None
    }

    fn handle_remote_key(&mut self, key: KeyEvent) -> LoraHubAction {
        match self.phase {
            // SEARCH: a tiny single-line query editor.
            Phase::Search => match key.code {
                KeyCode::Esc => self.tab = Tab::Local,
                KeyCode::Char(c) => self.query.push(c),
                KeyCode::Backspace => {
                    self.query.pop();
                }
                KeyCode::Enter => {
                    self.phase = Phase::Results;
                    self.remote_status = "searching…".into();
                    return LoraHubAction::Search { source: self.source(), query: self.query.clone() };
                }
                _ => {}
            },
            // RESULTS: browse + download.
            Phase::Results => match key.code {
                KeyCode::Down | KeyCode::Char('j') => {
                    if !self.hits.is_empty() {
                        self.hit_sel = (self.hit_sel + 1).min(self.hits.len() - 1);
                    }
                }
                KeyCode::Up | KeyCode::Char('k') => self.hit_sel = self.hit_sel.saturating_sub(1),
                KeyCode::Char('/') => self.phase = Phase::Search,
                // R — LLM assessment of the selected search hit.
                KeyCode::Char('R') => {
                    if let Some(h) = self.hits.get(self.hit_sel) {
                        let key = format!("hit:{}", h.title);
                        let prompt = format!(
                            "Stable Diffusion LoRA '{}' ({}, {} downloads). In ONE sentence, \
                             what is it and when would you use it?",
                            h.title,
                            if h.subtitle.is_empty() { "unknown base".into() } else { h.subtitle.clone() },
                            h.downloads,
                        );
                        self.assessing = Some(key.clone());
                        return LoraHubAction::Assess { key, prompt };
                    }
                }
                KeyCode::Left | KeyCode::Char('h') => self.switch_tab(-1),
                KeyCode::Right | KeyCode::Char('l') => self.switch_tab(1),
                KeyCode::Char('d' | 'D') | KeyCode::Enter => {
                    if let Some(h) = self.hits.get(self.hit_sel) {
                        self.remote_status = format!("downloading {}…", h.title);
                        return LoraHubAction::Download { dl: h.dl.clone(), title: h.title.clone() };
                    }
                }
                _ => {}
            },
        }
        LoraHubAction::None
    }

    /// Compatibility of a LoRA vs the loaded model: Some(true)=match, Some(false)=
    /// mismatch, None=can't tell (unknown family, or no model loaded).
    fn compatible(&self, l: &LoraInfo) -> Option<bool> {
        match (l.family, self.loaded_family) {
            (Some(a), Some(b)) => Some(a == b),
            _ => None,
        }
    }

    pub fn render(&self, f: &mut Frame, area: Rect) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(1)])
            .split(area);
        self.render_tabline(f, rows[0]);
        match self.tab {
            Tab::Local => {
                let cols = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
                    .split(rows[1]);
                self.render_list(f, cols[0]);
                self.render_detail(f, cols[1]);
            }
            _ => self.render_remote(f, rows[1]),
        }
    }

    fn render_tabline(&self, f: &mut Frame, area: Rect) {
        let tab = |name: &str, active: bool| -> Span {
            if active {
                Span::styled(format!(" {name} "), Style::new().bg(Color::Cyan).fg(Color::Black).add_modifier(Modifier::BOLD))
            } else {
                Span::styled(format!(" {name} "), Style::new().fg(Color::Gray))
            }
        };
        let hint = if self.tab == Tab::Local {
            "  →/← switch tab"
        } else {
            "  ←/→ tab · / edit query · D download"
        };
        f.render_widget(
            Paragraph::new(Line::from(vec![
                tab("LOCAL", self.tab == Tab::Local),
                Span::raw(" "),
                tab("CIVITAI", self.tab == Tab::Civitai),
                Span::raw(" "),
                tab("HUGGINGFACE", self.tab == Tab::HuggingFace),
                Span::styled(hint, Style::new().fg(Color::DarkGray)),
            ])),
            area,
        );
    }

    fn render_remote(&self, f: &mut Frame, area: Rect) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(1), Constraint::Length(3)])
            .split(area);
        self.render_assess_footer(f, rows[2]);

        // Query box (block cursor while editing).
        let editing = self.phase == Phase::Search;
        let mut spans = vec![Span::styled("search ", Style::new().fg(Color::Cyan)), Span::raw(self.query.clone())];
        if editing {
            spans.push(Span::styled(" ", Style::new().bg(Color::Cyan)));
        }
        let qcolor = if editing { Color::Cyan } else { Color::DarkGray };
        let src = match self.tab {
            Tab::HuggingFace => "HuggingFace",
            _ => "Civitai",
        };
        f.render_widget(
            Paragraph::new(Line::from(spans)).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(" {src} LoRA · {} ", self.remote_status))
                    .border_style(Style::new().fg(qcolor)),
            ),
            rows[0],
        );

        // Results.
        let block = Block::default().borders(Borders::ALL).title(" Results ");
        let inner = block.inner(rows[1]);
        f.render_widget(block, rows[1]);
        if self.hits.is_empty() {
            f.render_widget(
                Paragraph::new("No results yet — type a query above and press Enter.")
                    .style(Style::new().fg(Color::DarkGray))
                    .wrap(Wrap { trim: true }),
                inner,
            );
            return;
        }
        let mut lines: Vec<Line> = Vec::new();
        for (i, h) in self.hits.iter().enumerate() {
            // Civitai hits carry a base model → show compatibility; HF hits don't.
            let (glyph, gcolor) = match family_from_str(&h.subtitle) {
                Some(a) => match self.loaded_family {
                    Some(b) if a == b => ("✓", Color::Green),
                    Some(_) => ("✗", Color::Red),
                    None => ("·", Color::DarkGray),
                },
                None => ("·", Color::DarkGray),
            };
            let name_style = if i == self.hit_sel && self.phase == Phase::Results {
                Style::new().bg(Color::Cyan).fg(Color::Black).add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(Color::White)
            };
            lines.push(Line::from(vec![
                Span::styled(format!(" {glyph} "), Style::new().fg(gcolor).add_modifier(Modifier::BOLD)),
                Span::styled(trunc(&h.title, 32), name_style),
                Span::styled(
                    format!("  {}  ↓{}", if h.subtitle.is_empty() { "?" } else { &h.subtitle }, compact_count(h.downloads)),
                    Style::new().fg(Color::DarkGray),
                ),
            ]));
        }
        f.render_widget(Paragraph::new(lines), inner);
    }

    /// Footer showing the selected hit's LLM assessment (`R`).
    fn render_assess_footer(&self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" ✦ assess [R] ");
        let key = self.hits.get(self.hit_sel).map(|h| format!("hit:{}", h.title));
        let body = match &key {
            Some(k) if self.assessing.as_deref() == Some(k.as_str()) => {
                Span::styled("assessing…", Style::new().fg(Color::Yellow))
            }
            Some(k) => match self.assessments.get(k) {
                Some(a) => Span::styled(a.clone(), Style::new().fg(Color::Gray)),
                None => Span::styled("press R for an AI assessment", Style::new().fg(Color::DarkGray)),
            },
            None => Span::styled("no result selected", Style::new().fg(Color::DarkGray)),
        };
        f.render_widget(Paragraph::new(Line::from(body)).block(block).wrap(Wrap { trim: true }), area);
    }

    fn render_list(&self, f: &mut Frame, area: Rect) {
        let loaded = self
            .loaded_family
            .map(family_label)
            .map(|s| format!(" vs {s} "))
            .unwrap_or_else(|| " no model loaded ".into());
        let applied_n = self.applied.len();
        let block = Block::default().borders(Borders::ALL).title(format!(
            " LoRA · LOCAL ({}) ·{loaded}· [A] apply ({applied_n} on) ",
            self.loras.len()
        ));
        let inner = block.inner(area);
        f.render_widget(block, area);
        if self.loras.is_empty() {
            f.render_widget(
                Paragraph::new("No LoRAs found. Drop .safetensors into the workspace loras/ dir.")
                    .style(Style::new().fg(Color::DarkGray))
                    .wrap(Wrap { trim: true }),
                inner,
            );
            return;
        }
        let mut lines: Vec<Line> = Vec::new();
        for (i, l) in self.loras.iter().enumerate() {
            let (glyph, gcolor) = match self.compatible(l) {
                Some(true) => ("✓", Color::Green),
                Some(false) => ("✗", Color::Red),
                None => ("·", Color::DarkGray),
            };
            let name_style = if i == self.selected {
                Style::new().bg(Color::Cyan).fg(Color::Black).add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(Color::White)
            };
            // ★ = applied to Chat.
            let applied = if self.applied.contains(&l.path) {
                Span::styled("★", Style::new().fg(Color::Yellow))
            } else {
                Span::raw(" ")
            };
            lines.push(Line::from(vec![
                Span::styled(format!(" {glyph} "), Style::new().fg(gcolor).add_modifier(Modifier::BOLD)),
                applied,
                Span::styled(trunc(&l.name, 24), name_style),
                Span::styled(
                    format!("  {}", l.family.map(family_label).unwrap_or("?")),
                    Style::new().fg(Color::DarkGray),
                ),
            ]));
        }
        f.render_widget(Paragraph::new(lines), inner);
    }

    fn render_detail(&self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" Detail · [A] apply · [R] assess ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        let Some(l) = self.loras.get(self.selected) else { return };

        let kv = |k: &str, v: String| -> Line {
            Line::from(vec![
                Span::styled(format!("{k:<10}"), Style::new().fg(Color::DarkGray)),
                Span::styled(v, Style::new().fg(Color::White)),
            ])
        };
        let mut lines = vec![Line::from(Span::styled(
            l.name.clone(),
            Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ))];
        lines.push(kv("family", l.family.map(family_label).unwrap_or("unknown").to_string()));
        lines.push(kv("rank", l.rank.map(|r| r.to_string()).unwrap_or_else(|| "—".into())));
        lines.push(kv("size", format!("{} MB", l.size / 1_048_576)));
        lines.push(kv("location", l.location.clone()));
        lines.push(kv("path", l.path.display().to_string()));

        match self.compatible(l) {
            Some(true) => lines.push(Line::from(Span::styled("  ✓ compatible with the loaded model", Style::new().fg(Color::Green)))),
            Some(false) => lines.push(Line::from(Span::styled("  ✗ family mismatch with the loaded model", Style::new().fg(Color::Red)))),
            None => lines.push(Line::from(Span::styled("  · compatibility unknown (load a model)", Style::new().fg(Color::DarkGray)))),
        }

        if !l.triggers.is_empty() {
            lines.push(Line::from(""));
            lines.push(kv("triggers", l.triggers.join(", ")));
        }
        if !l.notes.is_empty() {
            lines.push(Line::from(""));
            lines.push(kv("notes", l.notes.clone()));
        }
        // LLM assessment (R).
        let key = l.path.display().to_string();
        if self.assessing.as_deref() == Some(key.as_str()) {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("  ✦ assessing…", Style::new().fg(Color::Yellow))));
        } else if let Some(a) = self.assessments.get(&key) {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("✦ AI assessment", Style::new().fg(Color::Magenta).add_modifier(Modifier::BOLD))));
            lines.push(Line::from(Span::styled(a.clone(), Style::new().fg(Color::Gray))));
        }
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
    }
}

/// Map a free-text base-model string (e.g. a sidecar's "SDXL 1.0") to a family.
fn family_from_str(s: &str) -> Option<BaseFamily> {
    let s = s.to_lowercase();
    if s.is_empty() {
        return None;
    }
    if s.contains("xl") {
        Some(BaseFamily::Sdxl)
    } else if s.contains("cascade") {
        Some(BaseFamily::StableCascade)
    } else if s.contains("pixart") {
        Some(BaseFamily::PixArt)
    } else if s.contains("flux") {
        Some(BaseFamily::Flux)
    } else if s.contains("sd3") || s.contains("sd-3") || s.contains("stable diffusion 3") {
        Some(BaseFamily::Sd3)
    } else if s.contains("2.1") || s.contains("v2") || s.contains(" 2 ") {
        Some(BaseFamily::Sd21)
    } else if s.contains("1.5") || s.contains("v1") || s.contains("1-5") {
        Some(BaseFamily::Sd15)
    } else {
        None
    }
}

fn family_label(f: BaseFamily) -> &'static str {
    match f {
        BaseFamily::Sd15 => "SD1.5",
        BaseFamily::Sd21 => "SD2.1",
        BaseFamily::Sdxl => "SDXL",
        BaseFamily::Flux => "Flux",
        BaseFamily::Sd3 => "SD3",
        BaseFamily::PixArt => "PixArt",
        BaseFamily::StableCascade => "Cascade",
    }
}

/// Compact a download count: 1500 → "1.5k", 2_000_000 → "2.0M".
fn compact_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        format!("{s:<width$}", width = n)
    } else {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
    }
}

/// Recursively collect `.safetensors` under `dir`.
fn collect(dir: &Path, label: &str, out: &mut Vec<LoraInfo>, depth: usize) {
    if depth > 5 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, label, out, depth + 1);
        } else if p.extension().and_then(|x| x.to_str()) == Some("safetensors") {
            out.push(load_lora(&p, label));
        }
    }
}

fn load_lora(path: &Path, location: &str) -> LoraInfo {
    let name = path.file_stem().and_then(|n| n.to_str()).unwrap_or("?").to_string();
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let (meta, dims) = read_st_header(path).unwrap_or_default();
    let mut family = infer_family(&meta, &dims);
    let rank = infer_rank(&meta, &dims);

    // Optional `<name>.plakat.hjson` sidecar.
    let mut triggers = Vec::new();
    let mut notes = String::new();
    let sidecar = path.with_extension("plakat.hjson");
    if let Ok(text) = std::fs::read_to_string(&sidecar) {
        if let Ok(sc) = deser_hjson::from_str::<Sidecar>(&text) {
            triggers = sc.trigger_words;
            notes = sc.notes;
            // Trust the sidecar's declared base_model when the header was inconclusive.
            if family.is_none() {
                family = family_from_str(&sc.base_model);
            }
        }
    }

    LoraInfo { path: path.to_path_buf(), name, family, rank, size, triggers, notes, location: location.to_string() }
}

/// Read a safetensors header: the `__metadata__` map + each tensor's shape. Reads
/// only the JSON header (8-byte LE length prefix), never the tensor data.
fn read_st_header(path: &Path) -> Option<(HashMap<String, String>, Vec<(String, Vec<usize>)>)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).ok()?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    if header_len == 0 || header_len > 100 * 1_048_576 {
        return None; // sanity bound
    }
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header).ok()?;
    let json: serde_json::Value = serde_json::from_slice(&header).ok()?;
    let obj = json.as_object()?;

    let mut meta = HashMap::new();
    if let Some(m) = obj.get("__metadata__").and_then(|v| v.as_object()) {
        for (k, v) in m {
            if let Some(s) = v.as_str() {
                meta.insert(k.clone(), s.to_string());
            }
        }
    }
    let mut dims = Vec::new();
    for (k, v) in obj {
        if k == "__metadata__" {
            continue;
        }
        if let Some(shape) = v.get("shape").and_then(|s| s.as_array()) {
            let shape: Vec<usize> = shape.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect();
            dims.push((k.clone(), shape));
        }
    }
    Some((meta, dims))
}

/// Infer the base family from kohya `ss_base_model_version` metadata, falling back
/// to the cross-attention context dim of a `to_k`/`to_v` LoRA-down weight
/// (768=SD1.5, 1024=SD2.1, 2048=SDXL).
fn infer_family(meta: &HashMap<String, String>, dims: &[(String, Vec<usize>)]) -> Option<BaseFamily> {
    if let Some(v) = meta.get("ss_base_model_version") {
        let v = v.to_lowercase();
        if v.contains("xl") {
            return Some(BaseFamily::Sdxl);
        }
        if v.contains("v2") || v.contains("_2.") || v.contains("768") {
            return Some(BaseFamily::Sd21);
        }
        if v.contains("v1") || v.contains("1-5") || v.contains("1.5") {
            return Some(BaseFamily::Sd15);
        }
    }
    // Dim heuristic: a cross-attn (attn2) to_k/to_v down weight is [rank, ctx_dim].
    for (name, shape) in dims {
        let n = name.to_lowercase();
        let is_ca = n.contains("attn2") && (n.contains("to_k") || n.contains("to_v"));
        let is_down = n.contains("lora_down") || n.contains("lora.down") || n.contains("lora_a");
        if is_ca && is_down && shape.len() == 2 {
            return match shape[1] {
                768 => Some(BaseFamily::Sd15),
                1024 => Some(BaseFamily::Sd21),
                2048 => Some(BaseFamily::Sdxl),
                _ => None,
            };
        }
    }
    None
}

fn infer_rank(meta: &HashMap<String, String>, dims: &[(String, Vec<usize>)]) -> Option<u32> {
    if let Some(d) = meta.get("ss_network_dim").and_then(|s| s.parse::<u32>().ok()) {
        return Some(d);
    }
    // A lora_down weight is [rank, in]; rank is the smaller leading dim.
    for (name, shape) in dims {
        let n = name.to_lowercase();
        if (n.contains("lora_down") || n.contains("lora.down") || n.contains("lora_a")) && shape.len() == 2 {
            return Some(shape[0] as u32);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal `.safetensors` (8-byte LE header len + JSON header) with the
    /// given `__metadata__` and one tensor of `shape`.
    fn write_st(path: &Path, metadata: &[(&str, &str)], tensor: &str, shape: &[usize]) {
        let meta: serde_json::Map<String, serde_json::Value> =
            metadata.iter().map(|(k, v)| (k.to_string(), serde_json::Value::String(v.to_string()))).collect();
        let header = serde_json::json!({
            "__metadata__": meta,
            tensor: { "dtype": "F16", "shape": shape, "data_offsets": [0, 0] },
        });
        let bytes = serde_json::to_vec(&header).unwrap();
        let mut out = (bytes.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&bytes);
        std::fs::write(path, out).unwrap();
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("plakat-lorahub-test-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn infers_family_from_kohya_metadata_and_dim_heuristic() {
        let d = tmp("infer");
        // Kohya metadata says SDXL.
        let xl = d.join("style-xl.safetensors");
        write_st(&xl, &[("ss_base_model_version", "sdxl_base_v1-0"), ("ss_network_dim", "32")], "x", &[32, 4]);
        // No metadata, but a cross-attn down weight reveals SD1.5 (ctx dim 768).
        let sd = d.join("char.safetensors");
        write_st(&sd, &[], "lora_unet_..._attn2_to_k.lora_down.weight", &[16, 768]);

        let s = LoraHubState::new(vec![(d.clone(), "loras".into())]);
        assert_eq!(s.loras.len(), 2);
        let by = |n: &str| s.loras.iter().find(|l| l.name == n).unwrap();
        assert_eq!(by("style-xl").family, Some(BaseFamily::Sdxl));
        assert_eq!(by("style-xl").rank, Some(32));
        assert_eq!(by("char").family, Some(BaseFamily::Sd15));
        assert_eq!(by("char").rank, Some(16));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn remote_tabs_search_and_download_state_machine() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let key = |c: KeyCode| KeyEvent::new(c, KeyModifiers::NONE);
        let ch = |c: char| KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE);

        let d = tmp("remote");
        let mut s = LoraHubState::new(vec![(d.clone(), "loras".into())]);
        assert!(!s.captures_input());

        // → enters CIVITAI (search box captures input).
        s.handle_key(key(KeyCode::Right));
        assert!(s.captures_input());
        for c in "water".chars() {
            s.handle_key(ch(c));
        }
        match s.handle_key(key(KeyCode::Enter)) {
            LoraHubAction::Search { source: RemoteSource::Civitai, query } => assert_eq!(query, "water"),
            _ => panic!("expected Civitai Search"),
        }
        assert!(!s.captures_input(), "Results phase doesn't capture input");

        s.set_remote_hits(vec![RemoteHit {
            title: "Watercolor".into(),
            subtitle: "SDXL 1.0".into(),
            downloads: 12345,
            dl: DownloadRef::Civitai { model_id: 42, version_id: Some(7) },
        }]);
        match s.handle_key(ch('D')) {
            LoraHubAction::Download { dl: DownloadRef::Civitai { model_id, version_id }, title } => {
                assert_eq!((model_id, version_id), (42, Some(7)));
                assert_eq!(title, "Watercolor");
            }
            _ => panic!("expected Civitai Download"),
        }

        // → again reaches HUGGINGFACE; its search yields an Hf source.
        s.handle_key(key(KeyCode::Right));
        assert!(s.captures_input());
        for c in "anime".chars() {
            s.handle_key(ch(c));
        }
        match s.handle_key(key(KeyCode::Enter)) {
            LoraHubAction::Search { source: RemoteSource::HuggingFace, query } => assert_eq!(query, "anime"),
            _ => panic!("expected HF Search"),
        }
        s.set_remote_hits(vec![RemoteHit {
            title: "user/anime-lora".into(),
            subtitle: "text-to-image".into(),
            downloads: 99,
            dl: DownloadRef::Hf { repo: "user/anime-lora".into() },
        }]);
        match s.handle_key(key(KeyCode::Enter)) {
            LoraHubAction::Download { dl: DownloadRef::Hf { repo }, .. } => assert_eq!(repo, "user/anime-lora"),
            _ => panic!("expected HF Download"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn r_requests_an_assessment_and_stores_the_result() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let d = tmp("assess");
        let xl = d.join("foo.safetensors");
        write_st(&xl, &[("ss_base_model_version", "sdxl")], "x", &[8, 4]);
        let mut s = LoraHubState::new(vec![(d.clone(), "loras".into())]);

        let key = match s.handle_key(KeyEvent::new(KeyCode::Char('R'), KeyModifiers::NONE)) {
            LoraHubAction::Assess { key, prompt } => {
                assert!(prompt.contains("foo"), "prompt names the LoRA: {prompt}");
                key
            }
            _ => panic!("expected Assess"),
        };
        assert_eq!(s.assessing.as_deref(), Some(key.as_str()), "marked in-flight");
        s.set_assessment(key.clone(), "A watercolour style LoRA.".into());
        assert!(s.assessing.is_none(), "cleared on result");
        assert_eq!(s.assessments.get(&key).unwrap(), "A watercolour style LoRA.");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn compact_count_formats() {
        assert_eq!(compact_count(999), "999");
        assert_eq!(compact_count(1500), "1.5k");
        assert_eq!(compact_count(2_000_000), "2.0M");
    }

    #[test]
    fn sidecar_base_model_is_a_family_fallback_when_header_is_inconclusive() {
        let d = tmp("sidecar-fallback");
        // No usable metadata / cross-attn dim → header inference is None.
        let f = d.join("mystery.safetensors");
        write_st(&f, &[], "some.lora_down.weight", &[8, 999]);
        std::fs::write(d.join("mystery.plakat.hjson"), r#"{"base_model":"SDXL 1.0"}"#).unwrap();
        let s = LoraHubState::new(vec![(d.clone(), "loras".into())]);
        assert_eq!(s.loras[0].family, Some(BaseFamily::Sdxl), "falls back to the sidecar base_model");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn compatibility_tracks_the_loaded_family_and_sidecar_loads() {
        let d = tmp("compat");
        let xl = d.join("foo.safetensors");
        write_st(&xl, &[("ss_base_model_version", "sdxl")], "x", &[8, 4]);
        std::fs::write(d.join("foo.plakat.hjson"), r#"{"trigger_words":["foo style"],"notes":"hi"}"#).unwrap();

        let mut s = LoraHubState::new(vec![(d.clone(), "loras".into())]);
        assert_eq!(s.loras[0].triggers, vec!["foo style"]);
        assert_eq!(s.loras[0].notes, "hi");
        // No model → unknown.
        assert_eq!(s.compatible(&s.loras[0]), None);
        s.set_loaded_family(Some(BaseFamily::Sdxl));
        assert_eq!(s.compatible(&s.loras[0]), Some(true));
        s.set_loaded_family(Some(BaseFamily::Sd15));
        assert_eq!(s.compatible(&s.loras[0]), Some(false));
        let _ = std::fs::remove_dir_all(&d);
    }
}
