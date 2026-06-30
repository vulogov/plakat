//! Prompt Workspace screen (RFC TUI-1 §12, Release 4). A `tui-textarea` editor over
//! the prompts dir (`.txt`/`.tera`/`.hjson` buffers) with a LIVE structural-compile
//! pane (deterministic parse + inheritance + `//` split — no LLM), `Ctrl-R` for the
//! full LLM compile, `Ctrl-S` save, and `Ctrl-O` to save the compiled HJSON and open
//! it in the Scenarios EDITOR. The structural compile is driven by the App (it owns
//! the runtime); this screen renders state + handles input. Tera mode is a follow-up.

use std::collections::BTreeMap;
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    Files,
    Editor,
    /// Editing Tera variable values (Tera mode only).
    Vars,
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
    /// Tera mode (`Ctrl-T`): run the Tera pre-pass before the structural compile.
    pub tera_mode: bool,
    /// Tera variable values (the live variable panel), edited in `Focus::Vars`.
    tera_vars: BTreeMap<String, String>,
    /// Selected row in the variable panel.
    var_sel: usize,
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
            tera_mode: false,
            tera_vars: BTreeMap::new(),
            var_sel: 0,
        };
        s.rescan();
        s
    }

    /// The editing buffer's path (so the Tera pre-pass can resolve sibling includes).
    pub fn path(&self) -> Option<&Path> {
        self.editing_path.as_deref()
    }

    /// Variable values for the Tera pre-pass (`--var KEY=VALUE` equivalents).
    pub fn tera_var_pairs(&self) -> Vec<(String, String)> {
        self.tera_vars.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    /// Variables referenced by the current template (for the live panel).
    fn variables(&self) -> Vec<String> {
        extract_variables(&self.editor_text())
    }

    /// Toggle Tera mode; forces a recompile and seeds any newly-seen variables.
    fn toggle_tera(&mut self) {
        self.tera_mode = !self.tera_mode;
        self.last_compiled_src = None; // mode change → recompute the compiled pane
        if self.tera_mode {
            for v in self.variables() {
                self.tera_vars.entry(v).or_default();
            }
            self.status = "Tera mode ON — Ctrl-V edits variables (needs --features templates)".into();
        } else {
            self.focus = Focus::Editor;
            self.status = "Tera mode OFF".into();
        }
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

    /// The App routes ALL keys here while the editor / variable panel is focused.
    pub fn captures_input(&self) -> bool {
        matches!(self.focus, Focus::Editor | Focus::Vars)
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
            Focus::Vars => {
                self.handle_vars_key(key);
                PromptsAction::None
            }
        }
    }

    /// Variable-panel editing (Tera mode): Up/Down select, typing edits the selected
    /// variable's value, Esc/Enter return to the editor.
    fn handle_vars_key(&mut self, key: KeyEvent) {
        let vars = self.variables();
        if vars.is_empty() {
            self.focus = Focus::Editor;
            return;
        }
        match key.code {
            KeyCode::Esc | KeyCode::Enter => self.focus = Focus::Editor,
            KeyCode::Up => self.var_sel = self.var_sel.saturating_sub(1),
            KeyCode::Down => self.var_sel = (self.var_sel + 1).min(vars.len() - 1),
            KeyCode::Char(c) => {
                if let Some(name) = vars.get(self.var_sel) {
                    self.tera_vars.entry(name.clone()).or_default().push(c);
                    self.last_compiled_src = None; // re-render live
                }
            }
            KeyCode::Backspace => {
                if let Some(name) = vars.get(self.var_sel) {
                    self.tera_vars.entry(name.clone()).or_default().pop();
                    self.last_compiled_src = None;
                }
            }
            _ => {}
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
            // Ctrl-T toggles Tera mode; Ctrl-V jumps to the variable panel (Tera on).
            KeyCode::Char('t' | 'T') if ctrl => self.toggle_tera(),
            KeyCode::Char('v' | 'V') if ctrl && self.tera_mode => {
                if self.variables().is_empty() {
                    self.status = "no template variables to edit".into();
                } else {
                    self.var_sel = 0;
                    self.focus = Focus::Vars;
                }
            }
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
        // In Tera mode the right column carries the live variable panel above the
        // compiled HJSON.
        if self.tera_mode {
            let right = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
                .split(cols[2]);
            self.render_vars(f, right[0]);
            self.render_compiled(f, right[1]);
        } else {
            self.render_compiled(f, cols[2]);
        }
    }

    /// The live Tera variable panel: each referenced variable + its current value.
    fn render_vars(&self, f: &mut Frame, area: Rect) {
        let focused = self.focus == Focus::Vars;
        let border = if focused { Color::Cyan } else { Color::Magenta };
        let title = if focused { " Variables · type value · Esc " } else { " Variables · Ctrl-V " };
        let block = Block::default().borders(Borders::ALL).title(title).border_style(Style::new().fg(border));
        let inner = block.inner(area);
        f.render_widget(block, area);
        let vars = self.variables();
        if vars.is_empty() {
            f.render_widget(
                Paragraph::new("(no {{ variables }} in this template)").style(Style::new().fg(Color::DarkGray)).wrap(Wrap { trim: true }),
                inner,
            );
            return;
        }
        let mut lines: Vec<Line> = Vec::new();
        for (i, name) in vars.iter().enumerate() {
            let val = self.tera_vars.get(name).cloned().unwrap_or_default();
            let sel = focused && i == self.var_sel;
            let name_style = if sel {
                Style::new().bg(Color::Cyan).fg(Color::Black).add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(Color::White)
            };
            let shown = if val.is_empty() { "—".to_string() } else { val };
            lines.push(Line::from(vec![
                Span::styled(format!("{name} "), name_style),
                Span::styled(format!("= {shown}"), Style::new().fg(Color::Gray)),
                if sel { Span::styled("▏", Style::new().fg(Color::Cyan)) } else { Span::raw("") },
            ]));
        }
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
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
        let tera = if self.tera_mode { " · τ Tera" } else { "" };
        let title = format!(
            " {name}{}{tera}  [Ctrl-S · Ctrl-N · Ctrl-Tab · Ctrl-T tera · Ctrl-R LLM · Ctrl-O→Scen · Esc] ",
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

/// Tera keywords / builtins that look like identifiers but aren't user variables.
const TERA_KEYWORDS: &[&str] = &[
    "if", "elif", "else", "endif", "for", "endfor", "in", "and", "or", "not", "is", "set",
    "block", "endblock", "include", "import", "macro", "endmacro", "filter", "endfilter",
    "true", "false", "loop", "self", "as", "with", "raw", "endraw",
];

/// Best-effort extraction of the top-level variables a Tera template *reads* (for the
/// live variable panel). Scans `{{ … }}` expressions and `{% if/for/set … %}` tags for
/// leading identifiers, then subtracts loop variables and `set` targets (locally bound)
/// and keywords. Heuristic, not a full parser — good enough to surface what to fill in.
fn extract_variables(src: &str) -> Vec<String> {
    let mut referenced: Vec<String> = Vec::new();
    let mut defined: std::collections::HashSet<String> = std::collections::HashSet::new();
    let bytes = src.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'{' && (bytes[i + 1] == b'{' || bytes[i + 1] == b'%') {
            let stmt = bytes[i + 1] == b'%';
            let close: &[u8] = if stmt { b"%}" } else { b"}}" };
            let start = i + 2;
            let end = find_sub(&bytes[start..], close).map(|p| start + p).unwrap_or(bytes.len());
            let inner = src[start..end].trim();
            if stmt {
                scan_tag(inner, &mut referenced, &mut defined);
            } else {
                // {{ expr }} → idents in the expression are reads.
                for id in idents(inner) {
                    referenced.push(id);
                }
            }
            i = end + close.len();
        } else {
            i += 1;
        }
    }
    // Dedup (stable order), drop locally-bound names + keywords.
    let mut out = Vec::new();
    for r in referenced {
        if !defined.contains(&r) && !out.contains(&r) && !TERA_KEYWORDS.contains(&r.as_str()) {
            out.push(r);
        }
    }
    out
}

/// Process a `{% … %}` tag body: record `for`-loop vars + `set` targets as *defined*,
/// and the rest of the idents (collection / condition / value) as *referenced*.
fn scan_tag(inner: &str, referenced: &mut Vec<String>, defined: &mut std::collections::HashSet<String>) {
    let toks: Vec<&str> = inner.split_whitespace().collect();
    match toks.first().copied() {
        Some("for") => {
            // for X[, Y] in EXPR
            if let Some(in_pos) = toks.iter().position(|t| *t == "in") {
                for t in &toks[1..in_pos] {
                    for id in idents(t) {
                        defined.insert(id);
                    }
                }
                for t in &toks[in_pos + 1..] {
                    for id in idents(t) {
                        referenced.push(id);
                    }
                }
            }
        }
        Some("set") => {
            // set X = EXPR
            if let Some(eq) = toks.iter().position(|t| *t == "=") {
                for t in &toks[1..eq] {
                    for id in idents(t) {
                        defined.insert(id);
                    }
                }
                for t in &toks[eq + 1..] {
                    for id in idents(t) {
                        referenced.push(id);
                    }
                }
            }
        }
        Some("if") | Some("elif") => {
            for t in &toks[1..] {
                for id in idents(t) {
                    referenced.push(id);
                }
            }
        }
        _ => {}
    }
}

/// Root identifiers in `s` that name variables. A `.attr` access or `| filter` tail is
/// dropped (only the chain's root is a variable); numbers / string literals are skipped.
fn idents(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let mut bound = false; // the previous separator run made this ident an attr/filter
    while i < chars.len() {
        let c = chars[i];
        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            if !bound {
                out.push(chars[start..i].iter().collect());
            }
            // Scan the following separator run; `.`/`|` bind the next ident as a
            // non-variable (attribute / filter).
            bound = false;
            while i < chars.len() && !(chars[i].is_alphanumeric() || chars[i] == '_') {
                if chars[i] == '.' || chars[i] == '|' {
                    bound = true;
                }
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    out
}

/// Index of the first occurrence of `needle` in `hay`.
fn find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
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
    fn extract_variables_finds_reads_and_skips_locals() {
        let tpl = "{% set greeting = title %}\n{% for item in items %}{{ item }} {{ subject | upper }} {{ scene.name }}{% endfor %}\n{% if dramatic %}moody{% endif %}";
        let vars = extract_variables(tpl);
        // Reads: title (set rhs), items (for coll), subject, scene (root of scene.name), dramatic.
        assert!(vars.contains(&"title".to_string()));
        assert!(vars.contains(&"items".to_string()));
        assert!(vars.contains(&"subject".to_string()));
        assert!(vars.contains(&"scene".to_string()));
        assert!(vars.contains(&"dramatic".to_string()));
        // Locals are excluded: `greeting` (set target) and `item` (loop var).
        assert!(!vars.contains(&"greeting".to_string()));
        assert!(!vars.contains(&"item".to_string()));
        // Attribute + keyword excluded.
        assert!(!vars.contains(&"name".to_string()));
        assert!(!vars.contains(&"upper".to_string()));
    }

    #[test]
    fn ctrl_t_toggles_tera_and_seeds_variables() {
        let d = tmp("tera");
        let mut s = PromptsState::new(d.clone());
        s.editor = text_area_from("a {{ subject }} in {{ place }}");
        assert!(!s.tera_mode);
        s.handle_key(ctrl('t'));
        assert!(s.tera_mode);
        // Variables are seeded (empty) so the panel + render have entries.
        assert!(s.tera_vars.contains_key("subject"));
        assert!(s.tera_vars.contains_key("place"));
        // A recompile is forced.
        assert!(s.last_compiled_src.is_none());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn ctrl_v_enters_var_panel_and_typing_edits_values() {
        let d = tmp("vars");
        let mut s = PromptsState::new(d.clone());
        s.editor = text_area_from("a {{ subject }}");
        s.handle_key(ctrl('t')); // Tera on
        s.handle_key(ctrl('v')); // into the variable panel
        assert!(matches!(s.focus, Focus::Vars));
        assert!(s.captures_input(), "Vars focus captures input");
        for c in "fox".chars() {
            s.handle_key(ch(c));
        }
        assert_eq!(s.tera_vars.get("subject").map(String::as_str), Some("fox"));
        // Esc returns to the editor.
        s.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(matches!(s.focus, Focus::Editor));
        let _ = std::fs::remove_dir_all(&d);
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
