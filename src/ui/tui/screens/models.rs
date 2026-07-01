//! Models screen (RFC TUI-1 §7) — browse the model registry, see live memory.
//! This increment is the read-only view (list + detail + memory bar); load/unload
//! via the background ModelService lands in the next increment.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
};

/// One row of the model list, derived from `hf::ALIAS_TABLE`.
pub struct ModelRow {
    pub alias: String,
    pub family: String,
    pub kind: String,
    pub gated: bool,
    pub note: String,
    pub repo: String,
    /// Loadable in the TUI today (SD-family / SD3 / PixArt / Cascade). Flux is CLI-only.
    pub usable: bool,
}

/// Live load state, fed by the ModelService over its channel.
#[derive(Clone, Default)]
pub enum LoadState {
    #[default]
    Idle,
    Loading(String),
    Loaded { alias: String, used_gb: f64 },
    Error(String),
}

pub struct ModelsState {
    pub rows: Vec<ModelRow>,
    pub selected: usize,
    pub load: LoadState,
}

impl Default for ModelsState {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelsState {
    pub fn new() -> Self {
        let mut rows: Vec<ModelRow> = crate::hf::ALIAS_TABLE
            .iter()
            .map(|e| {
                let alias = e.aliases.first().copied().unwrap_or("?").to_string();
                let usable = crate::ui::tui::services::model_service::t2i_load_check(&alias).is_ok();
                ModelRow {
                    alias,
                    family: e.family.to_string(),
                    kind: e.kind.to_string(),
                    gated: e.gated,
                    note: e.note.to_string(),
                    repo: e.repo.to_string(),
                    usable,
                }
            })
            .collect();
        // Models you can actually use in the TUI float to the top (CLI-only families like
        // Flux sink); within each group, sort by name.
        rows.sort_by(|a, b| b.usable.cmp(&a.usable).then_with(|| a.alias.cmp(&b.alias)));
        Self { rows, selected: 0, load: LoadState::Idle }
    }

    /// The alias the cursor is on (what [L]/[U] act on).
    pub fn selected_alias(&self) -> Option<String> {
        self.rows.get(self.selected).map(|r| r.alias.clone())
    }

    /// Move the cursor onto a model by alias (e.g. the workspace default at startup).
    pub fn select_by_alias(&mut self, alias: &str) {
        if let Some(pos) = self.rows.iter().position(|r| r.alias == alias) {
            self.selected = pos;
        }
    }

    /// Apply a ModelService status update.
    pub fn apply(&mut self, msg: &crate::ui::tui::services::model_service::ModelMessage) {
        use crate::ui::tui::services::model_service::ModelMessage as M;
        self.load = match msg {
            M::LoadStarted(a) => LoadState::Loading(a.clone()),
            M::Loaded { alias, used_gb } => LoadState::Loaded { alias: alias.clone(), used_gb: *used_gb },
            M::Unloaded => LoadState::Idle,
            M::Error(e) => LoadState::Error(e.clone()),
        };
    }

    /// Whether a load is currently in flight (used to avoid double-dispatching a
    /// background reload while one is already running).
    pub fn is_loading(&self) -> bool {
        matches!(self.load, LoadState::Loading(_))
    }

    /// The currently-loaded model's alias (drives the generation size).
    pub fn loaded_alias(&self) -> Option<&str> {
        match &self.load {
            LoadState::Loaded { alias, .. } => Some(alias.as_str()),
            _ => None,
        }
    }

    fn next(&mut self) {
        if !self.rows.is_empty() {
            self.selected = (self.selected + 1) % self.rows.len();
        }
    }

    fn prev(&mut self) {
        if !self.rows.is_empty() {
            self.selected = (self.selected + self.rows.len() - 1) % self.rows.len();
        }
    }

    fn selected_row(&self) -> Option<&ModelRow> {
        self.rows.get(self.selected)
    }

    /// Handle a screen-local key. Returns `true` if consumed.
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                self.next();
                true
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.prev();
                true
            }
            _ => false,
        }
    }

    pub fn render(&self, f: &mut Frame, area: Rect) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(1)])
            .split(area);
        crate::ui::tui::memory::render_memory_bar(f, rows[0]);

        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(rows[1]);
        self.render_list(f, cols[0]);
        self.render_detail(f, cols[1]);
    }

    fn render_list(&self, f: &mut Frame, area: Rect) {
        // Rows are sorted usable-first, then by name (CLI-only families sink), so the
        // family is shown as a plain column rather than a group header.
        let mut items: Vec<ListItem> = Vec::new();
        for row in &self.rows {
            // Non-usable (CLI-only, e.g. Flux) rows are dimmed + tagged.
            let alias_style = if row.usable {
                Style::new().fg(Color::White)
            } else {
                Style::new().fg(Color::DarkGray)
            };
            let fam = Span::styled(format!("{:<10}", row.family), Style::new().fg(Color::DarkGray));
            let gated = if row.gated {
                Span::styled(" 🔒", Style::new().fg(Color::Yellow))
            } else {
                Span::raw("")
            };
            let tail = if self.loaded_alias() == Some(row.alias.as_str()) {
                Span::styled(" ✓ loaded", Style::new().fg(Color::Green).add_modifier(Modifier::BOLD))
            } else if !row.usable {
                Span::styled(" CLI-only", Style::new().fg(Color::DarkGray))
            } else {
                Span::raw("")
            };
            items.push(ListItem::new(Line::from(vec![
                fam,
                Span::styled(format!("{:<16}", row.alias), alias_style),
                Span::styled(row.kind.clone(), Style::new().fg(Color::DarkGray)),
                gated,
                tail,
            ])));
        }
        let usable_n = self.rows.iter().filter(|r| r.usable).count();
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(format!(" Models · {usable_n} usable / {} ", self.rows.len())))
            .highlight_style(Style::new().bg(Color::Cyan).fg(Color::Black).add_modifier(Modifier::BOLD))
            .highlight_symbol("▶ ");
        let mut ls = ListState::default();
        ls.select(Some(self.selected));
        f.render_stateful_widget(list, area, &mut ls);
    }

    fn render_detail(&self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" Detail ");
        let body = match self.selected_row() {
            Some(r) => {
                let mut lines = vec![
                    Line::from(Span::styled(r.alias.clone(), Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
                    Line::from(format!("Family:  {}", r.family)),
                    Line::from(format!("Kind:    {}", r.kind)),
                    Line::from(format!("Repo:    {}", r.repo)),
                    Line::from(format!("Gated:   {}", if r.gated { "yes (accept the licence on HF)" } else { "no" })),
                    Line::from(""),
                    Line::from(Span::styled(r.note.clone(), Style::new().fg(Color::Gray))),
                    Line::from(""),
                ];
                let status = match &self.load {
                    LoadState::Idle => Span::styled("[L] Load   [U] Unload", Style::new().fg(Color::Gray)),
                    LoadState::Loading(a) => Span::styled(
                        format!("⟳ loading {a}… (downloads on first use; UI stays responsive)"),
                        Style::new().fg(Color::Yellow),
                    ),
                    LoadState::Loaded { alias, used_gb } => Span::styled(
                        format!("✓ {alias} loaded · {used_gb:.1} GB in use   [U] Unload"),
                        Style::new().fg(Color::Green),
                    ),
                    LoadState::Error(e) => Span::styled(format!("✗ {e}"), Style::new().fg(Color::Red)),
                };
                lines.push(Line::from(status));
                lines
            }
            None => vec![Line::from("No models.")],
        };
        f.render_widget(Paragraph::new(body).block(block).wrap(Wrap { trim: true }), area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;

    fn key(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    #[test]
    fn registry_populates_rows() {
        let s = ModelsState::new();
        assert!(s.rows.len() >= 10, "alias table has many models");
        assert!(s.rows.iter().any(|r| r.alias == "sd15"));
        assert!(s.rows.iter().any(|r| r.family.contains("SD 1.5") || r.family.contains("SDXL")));
    }

    #[test]
    fn rows_sort_usable_first_then_by_name() {
        let s = ModelsState::new();
        // All usable models precede all non-usable (CLI-only) ones.
        let first_unusable = s.rows.iter().position(|r| !r.usable);
        if let Some(i) = first_unusable {
            assert!(s.rows[..i].iter().all(|r| r.usable), "usable rows float to the top");
            assert!(s.rows[i..].iter().all(|r| !r.usable), "CLI-only rows sink");
        }
        // Within the usable group, aliases are alphabetical.
        let usable: Vec<&str> = s.rows.iter().filter(|r| r.usable).map(|r| r.alias.as_str()).collect();
        let mut sorted = usable.clone();
        sorted.sort();
        assert_eq!(usable, sorted, "usable models are name-sorted");
        // Flux (CLI-only) is marked non-usable.
        assert!(s.rows.iter().any(|r| r.alias.contains("flux") && !r.usable));
        // sd15 (usable) is near the top.
        assert!(s.rows.iter().take(usable.len()).any(|r| r.alias == "sd15"));
    }

    #[test]
    fn j_k_navigate_and_wrap() {
        let mut s = ModelsState::new();
        assert_eq!(s.selected, 0);
        assert!(s.handle_key(key(KeyCode::Char('j'))));
        assert_eq!(s.selected, 1);
        assert!(s.handle_key(key(KeyCode::Char('k'))));
        assert_eq!(s.selected, 0);
        // wraps backward to the last row
        s.handle_key(key(KeyCode::Up));
        assert_eq!(s.selected, s.rows.len() - 1);
    }

    #[test]
    fn unrelated_keys_not_consumed() {
        let mut s = ModelsState::new();
        assert!(!s.handle_key(key(KeyCode::Char('x'))));
    }
}
