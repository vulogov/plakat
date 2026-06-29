//! Prompt Workspace screen (RFC TUI-1 §12, Release 4). A `tui-textarea` editor over
//! the prompts dir (`.txt`/`.tera`/`.hjson` buffers) with a LIVE structural-compile
//! pane (deterministic parse + inheritance + `//` split — no LLM), `Ctrl-R` for the
//! full LLM compile, `Ctrl-S` save, and `Ctrl-O` to save the compiled HJSON and open
//! it in the Scenarios EDITOR. The structural compile is driven by the App (it owns
//! the runtime); this screen renders state + handles input. Tera mode is a follow-up.

use std::path::PathBuf;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
};
use tui_textarea::TextArea;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    Files,
    Editor,
}

/// What the App should do after a key.
pub enum PromptsAction {
    None,
    /// `Ctrl-R` — run the full LLM compile over this prompt text (App does it).
    LlmCompile(String),
    /// `Ctrl-O` — save the compiled HJSON and open it in the Scenarios EDITOR.
    OpenInScenarios { name: String, hjson: String },
}

pub struct PromptsState {
    dir: PathBuf,
    files: Vec<PathBuf>,
    file_sel: usize,
    focus: Focus,
    editor: TextArea<'static>,
    editing_path: Option<PathBuf>,
    dirty: bool,
    status: String,
    // Compile pane (filled by the App).
    pub compiled: String,
    pub compile_err: Option<String>,
    pub compiling: bool,
    /// The editor text last handed to the structural compiler (App-owned debounce).
    pub last_compiled_src: Option<String>,
}

impl PromptsState {
    pub fn new(dir: PathBuf) -> Self {
        let mut s = Self {
            dir,
            files: Vec::new(),
            file_sel: 0,
            focus: Focus::Editor,
            editor: text_area_from(STARTER),
            editing_path: None,
            dirty: false,
            status: String::new(),
            compiled: String::new(),
            compile_err: None,
            compiling: false,
            last_compiled_src: None,
        };
        s.rescan();
        s
    }

    pub fn rescan(&mut self) {
        let mut files = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            for e in rd.flatten() {
                let p = e.path();
                let ok = p
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|x| matches!(x, "txt" | "tera" | "hjson"))
                    .unwrap_or(false);
                if p.is_file() && ok {
                    files.push(p);
                }
            }
        }
        files.sort();
        self.files = files;
        if self.file_sel >= self.files.len() {
            self.file_sel = self.files.len().saturating_sub(1);
        }
    }

    /// The App routes ALL keys here while the editor is focused.
    pub fn captures_input(&self) -> bool {
        self.focus == Focus::Editor
    }

    /// Current editor text (the App compiles this).
    pub fn editor_text(&self) -> String {
        self.editor.lines().join("\n")
    }

    /// The buffer name (for compiler header + the saved scenario filename).
    pub fn buffer_name(&self) -> String {
        self.editing_path
            .as_ref()
            .and_then(|p| p.file_stem())
            .and_then(|n| n.to_str())
            .unwrap_or("untitled")
            .to_string()
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> PromptsAction {
        match self.focus {
            Focus::Files => self.handle_files_key(key),
            Focus::Editor => self.handle_editor_key(key),
        }
    }

    fn handle_files_key(&mut self, key: KeyEvent) -> PromptsAction {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                if !self.files.is_empty() {
                    self.file_sel = (self.file_sel + 1).min(self.files.len() - 1);
                }
            }
            KeyCode::Up | KeyCode::Char('k') => self.file_sel = self.file_sel.saturating_sub(1),
            KeyCode::Char('r') => self.rescan(),
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                if let Some(p) = self.files.get(self.file_sel).cloned() {
                    self.load(p);
                }
                self.focus = Focus::Editor;
            }
            _ => {}
        }
        PromptsAction::None
    }

    fn handle_editor_key(&mut self, key: KeyEvent) -> PromptsAction {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Esc => self.focus = Focus::Files,
            // Ctrl-Tab cycles to the next saved buffer; Ctrl-N starts a fresh one.
            KeyCode::Tab if ctrl => self.cycle_buffer(),
            KeyCode::Char('n' | 'N') if ctrl => self.new_buffer(),
            KeyCode::Char('s' | 'S') if ctrl => self.save(),
            KeyCode::Char('r' | 'R') if ctrl => {
                self.compiling = true;
                self.compile_err = None;
                return PromptsAction::LlmCompile(self.editor_text());
            }
            KeyCode::Char('o' | 'O') if ctrl => {
                if self.compiled.is_empty() {
                    self.status = "compile something first (it shows on the right)".into();
                } else {
                    return PromptsAction::OpenInScenarios {
                        name: self.buffer_name(),
                        hjson: self.compiled.clone(),
                    };
                }
            }
            _ => {
                if self.editor.input(key) {
                    self.dirty = true;
                }
            }
        }
        PromptsAction::None
    }

    /// Cycle the editor to the next saved buffer (wraps), staying in the editor. With a
    /// dirty unsaved buffer it warns rather than silently discarding edits.
    fn cycle_buffer(&mut self) {
        self.rescan();
        if self.files.is_empty() {
            self.status = "no other buffers — Ctrl-N for a new one".into();
            return;
        }
        if self.dirty {
            self.status = "unsaved edits — Ctrl-S to save before cycling".into();
            return;
        }
        let cur = self.editing_path.as_ref().and_then(|p| self.files.iter().position(|f| f == p));
        let next = match cur {
            Some(i) => (i + 1) % self.files.len(),
            None => 0,
        };
        let path = self.files[next].clone();
        self.file_sel = next;
        self.load(path);
        self.status = format!("buffer {}/{}: {}", next + 1, self.files.len(), self.buffer_name());
    }

    /// Open a fresh, uniquely-named `prompt-N.txt` buffer (name it by saving). The
    /// buffer is dirty (not yet on disk) until `Ctrl-S`.
    fn new_buffer(&mut self) {
        let mut n = 0usize;
        let path = loop {
            let name = if n == 0 { "prompt.txt".to_string() } else { format!("prompt-{n}.txt") };
            let p = self.dir.join(&name);
            if !p.exists() {
                break p;
            }
            n += 1;
        };
        self.editor = text_area_from(STARTER);
        self.editing_path = Some(path);
        self.dirty = true;
        self.last_compiled_src = None;
        self.focus = Focus::Editor;
        self.status = format!("new buffer ‘{}’ — Ctrl-S to save (renames the file)", self.buffer_name());
    }

    fn load(&mut self, path: PathBuf) {
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                self.editor = text_area_from(&text);
                self.editing_path = Some(path);
                self.dirty = false;
                self.last_compiled_src = None; // force a recompile
                self.status.clear();
            }
            Err(e) => self.status = format!("✗ {e}"),
        }
    }

    fn save(&mut self) {
        let path = match &self.editing_path {
            Some(p) => p.clone(),
            None => {
                let p = self.dir.join("untitled.txt");
                self.editing_path = Some(p.clone());
                p
            }
        };
        let body = self.editor_text();
        let body = if body.ends_with('\n') { body } else { format!("{body}\n") };
        let _ = std::fs::create_dir_all(&self.dir);
        match std::fs::write(&path, body) {
            Ok(()) => {
                self.dirty = false;
                self.status = format!("✓ saved {}", path.file_name().and_then(|n| n.to_str()).unwrap_or("?"));
                self.rescan();
            }
            Err(e) => self.status = format!("✗ save failed: {e}"),
        }
    }

    pub fn render(&self, f: &mut Frame, area: Rect) {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(20), Constraint::Percentage(42), Constraint::Percentage(38)])
            .split(area);
        self.render_files(f, cols[0]);
        self.render_editor(f, cols[1]);
        self.render_compiled(f, cols[2]);
    }

    fn render_files(&self, f: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self
            .files
            .iter()
            .map(|p| ListItem::new(p.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string()))
            .collect();
        let items = if items.is_empty() {
            vec![ListItem::new(Span::styled("(no .txt/.tera/.hjson)", Style::new().fg(Color::DarkGray)))]
        } else {
            items
        };
        let border = if self.focus == Focus::Files { Color::Cyan } else { Color::DarkGray };
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(" Buffers ").border_style(Style::new().fg(border)))
            .highlight_style(Style::new().bg(Color::Cyan).fg(Color::Black).add_modifier(Modifier::BOLD))
            .highlight_symbol("▶ ");
        let mut ls = ListState::default();
        ls.select(Some(self.file_sel));
        f.render_stateful_widget(list, area, &mut ls);
    }

    fn render_editor(&self, f: &mut Frame, area: Rect) {
        let name = self.buffer_name();
        let title = format!(
            " {name}{}  [Ctrl-S save · Ctrl-N new · Ctrl-Tab cycle · Ctrl-R LLM · Ctrl-O→Scenarios · Esc] ",
            if self.dirty { " ●" } else { "" }
        );
        let border = if self.focus == Focus::Editor { Color::Cyan } else { Color::DarkGray };
        let block = Block::default().borders(Borders::ALL).title(title).border_style(Style::new().fg(border));
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(&self.editor, inner);
    }

    fn render_compiled(&self, f: &mut Frame, area: Rect) {
        let title = if self.compiling {
            " Compiled · LLM…".to_string()
        } else if self.compile_err.is_some() {
            " Compiled · ✗ error".to_string()
        } else {
            " Compiled (structural — Ctrl-R for LLM) ".to_string()
        };
        let block = Block::default().borders(Borders::ALL).title(title);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let lines: Vec<Line> = match &self.compile_err {
            Some(e) => e.lines().map(|l| Line::from(Span::styled(l.to_string(), Style::new().fg(Color::Red)))).collect(),
            None if self.compiled.is_empty() => {
                vec![Line::from(Span::styled("type a prompt — the structural compile shows here", Style::new().fg(Color::DarkGray)))]
            }
            None => self.compiled.lines().map(|l| Line::from(l.to_string())).collect(),
        };
        // Show the tail (most relevant after the header) if it overflows.
        let h = inner.height as usize;
        let start = lines.len().saturating_sub(h.max(1));
        f.render_widget(Paragraph::new(lines[start..].to_vec()).wrap(Wrap { trim: false }), inner);
        let _ = &self.status;
    }
}

const STARTER: &str = "// a watercolor fox in a misty forest\n// a neon city street at night, rain\n";

fn text_area_from(text: &str) -> TextArea<'static> {
    let mut ta = TextArea::new(text.lines().map(str::to_string).collect());
    ta.set_cursor_line_style(Style::default());
    ta
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }
    fn ch(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("plakat-prompts-test-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn editor_focus_and_ctrl_r_requests_llm_compile() {
        let d = tmp("focus");
        let mut s = PromptsState::new(d.clone());
        assert!(s.captures_input(), "editor focused by default");
        s.handle_key(ch('x'));
        assert!(s.editor_text().contains('x'));
        match s.handle_key(ctrl('r')) {
            PromptsAction::LlmCompile(text) => assert!(text.contains('x')),
            _ => panic!("expected LlmCompile"),
        }
        assert!(s.compiling);
        // Esc moves focus to the buffer list (no longer capturing input).
        s.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!s.captures_input());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn ctrl_o_opens_compiled_in_scenarios_when_present() {
        let d = tmp("open");
        let mut s = PromptsState::new(d.clone());
        // Nothing compiled yet → no action.
        assert!(matches!(s.handle_key(ctrl('o')), PromptsAction::None));
        s.compiled = "{ tasks: [] }".into();
        match s.handle_key(ctrl('o')) {
            PromptsAction::OpenInScenarios { hjson, .. } => assert!(hjson.contains("tasks")),
            _ => panic!("expected OpenInScenarios"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn ctrl_n_makes_a_new_unique_buffer_and_ctrl_tab_cycles() {
        let d = tmp("buffers");
        std::fs::write(d.join("a.txt"), "alpha").unwrap();
        std::fs::write(d.join("b.txt"), "beta").unwrap();
        let mut s = PromptsState::new(d.clone());

        // Ctrl-N opens a fresh, uniquely-named, dirty buffer.
        s.handle_key(ctrl('n'));
        assert_eq!(s.buffer_name(), "prompt");
        assert!(s.dirty);
        // Save it, then Ctrl-Tab cycles among the saved buffers (a, b, prompt).
        s.handle_key(ctrl('s'));
        assert!(!s.dirty);
        let first = s.buffer_name();
        s.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::CONTROL));
        assert_ne!(s.buffer_name(), first, "cycled to a different buffer");
        assert!(s.status.starts_with("buffer "));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn ctrl_tab_warns_on_unsaved_edits() {
        let d = tmp("dirty-cycle");
        std::fs::write(d.join("a.txt"), "alpha").unwrap();
        let mut s = PromptsState::new(d.clone());
        s.handle_key(ch('z')); // dirty the buffer
        assert!(s.dirty);
        s.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::CONTROL));
        assert!(s.status.contains("unsaved"), "cycling a dirty buffer warns");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn rescan_lists_prompt_buffers_only() {
        let d = tmp("scan");
        std::fs::write(d.join("a.txt"), "x").unwrap();
        std::fs::write(d.join("b.tera"), "y").unwrap();
        std::fs::write(d.join("c.hjson"), "z").unwrap();
        std::fs::write(d.join("ignore.png"), "n").unwrap();
        let s = PromptsState::new(d.clone());
        assert_eq!(s.files.len(), 3);
        let _ = std::fs::remove_dir_all(&d);
    }
}
