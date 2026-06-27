//! Scenarios screen (RFC TUI-1 §8). Two modes:
//!  - SELECT — browse the workspace's scenario HJSON files (task count + model),
//!    run one (`Enter`), edit one (`e`), or start a new one (`n`).
//!  - EDITOR — a `tui-textarea` multi-line editor over the selected/new file;
//!    `Ctrl-S` saves to disk (and re-scans), `Esc` returns to SELECT.
//!
//! A run's task-by-task progress flows to the Output pane automatically (the
//! scenario runner uses the rerouted `ui::progress`). The nested RUNNER sub-tab
//! (live per-task status board) is a follow-up.

use std::path::{Path, PathBuf};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
};
use tui_textarea::TextArea;

/// Starter content for a brand-new scenario (valid HJSON — one field per line).
const STARTER_TEMPLATE: &str = "{\n  \
    model: stable-diffusion-v1-5/stable-diffusion-v1-5\n  \
    size: 512x512\n  \
    steps: 28\n  \
    guidance: 7.0\n  \
    tasks: [\n    \
        {\n      \
            name: first\n      \
            prompt: a serene mountain lake at dawn, soft light\n    \
        }\n  \
    ]\n}\n";

/// One scenario file's at-a-glance info.
pub struct ScenarioInfo {
    pub path: PathBuf,
    pub name: String,
    pub tasks: usize,
    pub model: String,
}

/// What the App should do after a key.
pub enum ScenariosAction {
    None,
    Run(PathBuf),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Select,
    Editor,
}

pub struct ScenariosState {
    pub dir: PathBuf,
    pub files: Vec<ScenarioInfo>,
    pub selected: usize,
    /// Last run / save status line.
    pub status: String,
    mode: Mode,
    editor: TextArea<'static>,
    /// Path the editor buffer will save to (existing file or a new one).
    editing_path: Option<PathBuf>,
    /// Unsaved edits in the buffer.
    dirty: bool,
}

impl ScenariosState {
    pub fn new(dir: PathBuf) -> Self {
        let mut s = Self {
            dir,
            files: Vec::new(),
            selected: 0,
            status: String::new(),
            mode: Mode::Select,
            editor: TextArea::default(),
            editing_path: None,
            dirty: false,
        };
        s.rescan();
        s
    }

    /// Whether the EDITOR is focused — the App routes ALL keys to us when true
    /// (so plain chars / Enter type into the buffer instead of switching screens).
    pub fn is_editing(&self) -> bool {
        self.mode == Mode::Editor
    }

    /// (Re)scan the scenarios dir for `*.hjson` (excluding `*.run.hjson` sidecars).
    pub fn rescan(&mut self) {
        let mut files = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            for e in rd.flatten() {
                let p = e.path();
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
                if name.ends_with(".hjson") && !name.ends_with(".run.hjson") {
                    let (tasks, model) = peek(&p);
                    files.push(ScenarioInfo {
                        name: name.trim_end_matches(".hjson").to_string(),
                        path: p,
                        tasks,
                        model,
                    });
                }
            }
        }
        files.sort_by(|a, b| a.name.cmp(&b.name));
        self.files = files;
        if self.selected >= self.files.len() {
            self.selected = self.files.len().saturating_sub(1);
        }
    }

    fn next(&mut self) {
        if !self.files.is_empty() {
            self.selected = (self.selected + 1) % self.files.len();
        }
    }

    fn prev(&mut self) {
        if !self.files.is_empty() {
            self.selected = (self.selected + self.files.len() - 1) % self.files.len();
        }
    }

    pub fn selected_path(&self) -> Option<PathBuf> {
        self.files.get(self.selected).map(|f| f.path.clone())
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> ScenariosAction {
        match self.mode {
            Mode::Select => self.handle_select_key(key),
            Mode::Editor => {
                self.handle_editor_key(key);
                ScenariosAction::None
            }
        }
    }

    fn handle_select_key(&mut self, key: KeyEvent) -> ScenariosAction {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.next(),
            KeyCode::Up | KeyCode::Char('k') => self.prev(),
            // Run the selected scenario.
            KeyCode::Enter | KeyCode::Char('r' | 'R') => {
                return self.selected_path().map(ScenariosAction::Run).unwrap_or(ScenariosAction::None);
            }
            // Edit the selected scenario in the buffer.
            KeyCode::Char('e' | 'E') => self.open_selected_in_editor(),
            // Start a brand-new scenario from a template.
            KeyCode::Char('n' | 'N') => self.new_scenario(),
            _ => {}
        }
        ScenariosAction::None
    }

    fn handle_editor_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            // Save to disk + re-scan (keeps editing).
            KeyCode::Char('s' | 'S') if ctrl => self.save(),
            // Back to the list (the buffer is kept in case you re-enter).
            KeyCode::Esc => {
                self.mode = Mode::Select;
                self.status = if self.dirty {
                    "Left editor with unsaved changes (buffer kept; Ctrl-S to save).".into()
                } else {
                    String::new()
                };
            }
            // Everything else edits the buffer.
            _ => {
                if self.editor.input(key) {
                    self.dirty = true;
                }
            }
        }
    }

    /// Load the selected file into the editor buffer.
    fn open_selected_in_editor(&mut self) {
        let Some(path) = self.selected_path() else {
            self.status = "No scenario selected to edit.".into();
            return;
        };
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                self.editor = text_area_from(&text);
                self.editing_path = Some(path);
                self.mode = Mode::Editor;
                self.dirty = false;
                self.status.clear();
            }
            Err(e) => self.status = format!("✗ Could not read file: {e}"),
        }
    }

    /// Start a new scenario from the template, in an unused `untitled[-N].hjson`.
    fn new_scenario(&mut self) {
        let mut n = 0usize;
        let path = loop {
            let name = if n == 0 { "untitled.hjson".to_string() } else { format!("untitled-{n}.hjson") };
            let p = self.dir.join(&name);
            if !p.exists() {
                break p;
            }
            n += 1;
        };
        self.editor = text_area_from(STARTER_TEMPLATE);
        self.editing_path = Some(path);
        self.mode = Mode::Editor;
        self.dirty = true; // not on disk yet
        self.status.clear();
    }

    /// Write the buffer to its path and re-scan the listing.
    fn save(&mut self) {
        let Some(path) = self.editing_path.clone() else {
            self.status = "✗ No path to save to.".into();
            return;
        };
        let body = self.editor.lines().join("\n");
        // Ensure a trailing newline (POSIX-friendly).
        let body = if body.ends_with('\n') { body } else { format!("{body}\n") };
        match std::fs::write(&path, body) {
            Ok(()) => {
                self.dirty = false;
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("scenario").to_string();
                self.status = format!("✓ Saved {name}");
                self.rescan();
                // Keep the cursor on the file we just saved.
                if let Some(i) = self.files.iter().position(|f| f.path == path) {
                    self.selected = i;
                }
            }
            Err(e) => self.status = format!("✗ Save failed: {e}"),
        }
    }

    pub fn render(&self, f: &mut Frame, area: Rect) {
        match self.mode {
            Mode::Select => self.render_select(f, area),
            Mode::Editor => self.render_editor(f, area),
        }
    }

    fn render_select(&self, f: &mut Frame, area: Rect) {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(area);
        self.render_list(f, cols[0]);
        self.render_detail(f, cols[1]);
    }

    fn render_list(&self, f: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self
            .files
            .iter()
            .map(|s| {
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{:<22}", s.name), Style::new().fg(Color::White)),
                    Span::styled(format!("{} task{}", s.tasks, if s.tasks == 1 { "" } else { "s" }), Style::new().fg(Color::DarkGray)),
                ]))
            })
            .collect();
        let items = if items.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "No scenarios. Press [n] for a new one.",
                Style::new().fg(Color::DarkGray),
            )))]
        } else {
            items
        };
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(format!(" Scenarios ({}) ", self.files.len())))
            .highlight_style(Style::new().bg(Color::Cyan).fg(Color::Black).add_modifier(Modifier::BOLD))
            .highlight_symbol("▶ ");
        let mut ls = ListState::default();
        ls.select(Some(self.selected));
        f.render_stateful_widget(list, area, &mut ls);
    }

    fn render_detail(&self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" Detail ");
        let lines = match self.files.get(self.selected) {
            Some(s) => {
                let mut v = vec![
                    Line::from(Span::styled(s.name.clone(), Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
                    Line::from(format!("Tasks:  {}", s.tasks)),
                    Line::from(format!("Model:  {}", s.model)),
                    Line::from(format!("File:   {}", s.path.display())),
                    Line::from(""),
                    Line::from(Span::styled(
                        "[Enter] Run   [e] Edit   [n] New   (progress → Output pane)",
                        Style::new().fg(Color::Gray),
                    )),
                ];
                if !self.status.is_empty() {
                    v.push(Line::from(""));
                    v.push(Line::from(Span::styled(self.status.clone(), Style::new().fg(Color::Yellow))));
                }
                v
            }
            None => vec![
                Line::from("No scenario selected."),
                Line::from(""),
                Line::from(Span::styled("[n] New scenario", Style::new().fg(Color::Gray))),
            ],
        };
        f.render_widget(Paragraph::new(lines).block(block).wrap(Wrap { trim: true }), area);
    }

    fn render_editor(&self, f: &mut Frame, area: Rect) {
        let name = self
            .editing_path
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("scenario");
        let title = format!(
            " Editing {name}{}   [Ctrl-S] save  [Esc] back ",
            if self.dirty { " ●" } else { "" }
        );
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .border_style(Style::new().fg(if self.dirty { Color::Yellow } else { Color::Cyan }));
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(&self.editor, inner);
    }
}

/// Build a `TextArea` from raw file text (split into lines).
fn text_area_from(text: &str) -> TextArea<'static> {
    let mut ta = TextArea::new(text.lines().map(str::to_string).collect());
    ta.set_cursor_line_style(Style::default()); // no underline on the active line
    ta
}

/// Peek a scenario file for `(task_count, model)`, reusing the real scenario
/// parser. Best-effort for the listing — `(0, "?")` on a malformed file.
fn peek(path: &Path) -> (usize, String) {
    crate::cli::scenario::peek(path).unwrap_or((0, "?".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;

    fn key(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    fn ch(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("plakat-scen-test-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn scans_hjson_excluding_run_sidecars() {
        let d = tmp_dir("scan");
        // Strict JSON is valid HJSON and sidesteps the quoteless-to-end-of-line trap.
        std::fs::write(d.join("a.hjson"), r#"{"model":"sdxl","tasks":[{"name":"t1"},{"name":"t2"}]}"#).unwrap();
        std::fs::write(d.join("b.hjson"), r#"{"model":"sd15","tasks":[{"name":"t1"}]}"#).unwrap();
        std::fs::write(d.join("a.run.hjson"), "{}").unwrap(); // sidecar — excluded
        let s = ScenariosState::new(d.clone());
        assert_eq!(s.files.len(), 2);
        assert_eq!(s.files[0].name, "a");
        assert_eq!(s.files[0].tasks, 2);
        assert_eq!(s.files[0].model, "sdxl");
        assert_eq!(s.files[1].name, "b");
        assert_eq!(s.files[1].tasks, 1);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn peek_parses_a_real_scenario_file() {
        // The shipped example is real HJSON (comments, quoteless values) — peek must
        // count its tasks and read its model, not fall back to (0, "?").
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/scenario.hjson");
        if !p.exists() {
            return; // example not present in this checkout — skip
        }
        let (tasks, model) = peek(&p);
        assert!(tasks > 0, "expected >0 tasks from the real scenario, got {tasks}");
        assert!(model.contains("stable-diffusion"), "model was {model:?}");
    }

    #[test]
    fn enter_runs_the_selected() {
        let d = tmp_dir("run");
        std::fs::write(d.join("x.hjson"), r#"{"model":"sdxl","tasks":[]}"#).unwrap();
        let mut s = ScenariosState::new(d.clone());
        match s.handle_key(key(KeyCode::Enter)) {
            ScenariosAction::Run(p) => assert!(p.ends_with("x.hjson")),
            _ => panic!("expected Run"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn edit_then_save_round_trips_and_rescans() {
        let d = tmp_dir("edit");
        std::fs::write(d.join("z.hjson"), r#"{"model":"sdxl","tasks":[{"name":"t1"}]}"#).unwrap();
        let mut s = ScenariosState::new(d.clone());
        assert!(!s.is_editing());

        // [e] opens the editor on the selected file.
        s.handle_key(ch('e'));
        assert!(s.is_editing());
        assert!(!s.dirty);

        // Type a character → buffer is dirty.
        s.handle_key(ch('X'));
        assert!(s.dirty);

        // Ctrl-S saves to disk, clears dirty, re-scans, and stays in the editor.
        s.handle_key(ctrl('s'));
        assert!(!s.dirty);
        assert!(s.is_editing());
        assert!(s.status.starts_with("✓ Saved"));
        let on_disk = std::fs::read_to_string(d.join("z.hjson")).unwrap();
        assert!(on_disk.contains('X'), "edit was persisted");

        // Esc returns to SELECT.
        s.handle_key(key(KeyCode::Esc));
        assert!(!s.is_editing());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn new_scenario_creates_buffer_and_saves_a_file() {
        let d = tmp_dir("new");
        let mut s = ScenariosState::new(d.clone());
        assert_eq!(s.files.len(), 0);

        // [n] opens a fresh template in the editor (dirty, not yet on disk).
        s.handle_key(ch('n'));
        assert!(s.is_editing());
        assert!(s.dirty);
        assert!(!d.join("untitled.hjson").exists(), "new file is only on disk after save");

        // Ctrl-S writes it and the rescan picks it up.
        s.handle_key(ctrl('s'));
        assert!(d.join("untitled.hjson").exists());
        assert_eq!(s.files.len(), 1);
        assert_eq!(s.files[0].name, "untitled");
        // The template parses (model + 1 task).
        assert_eq!(s.files[0].tasks, 1);
        let _ = std::fs::remove_dir_all(&d);
    }
}
