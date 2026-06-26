//! Models screen (RFC TUI-1 §7) — browse the model registry, see live memory.
//! This increment is the read-only view (list + detail + memory bar); load/unload
//! via the background ModelService lands in the next increment.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, ListState, Paragraph, Wrap},
};

/// One row of the model list, derived from `hf::ALIAS_TABLE`.
pub struct ModelRow {
    pub alias: String,
    pub family: String,
    pub kind: String,
    pub gated: bool,
    pub note: String,
    pub repo: String,
}

pub struct ModelsState {
    pub rows: Vec<ModelRow>,
    pub selected: usize,
}

impl Default for ModelsState {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelsState {
    pub fn new() -> Self {
        let rows = crate::hf::ALIAS_TABLE
            .iter()
            .map(|e| ModelRow {
                alias: e.aliases.first().copied().unwrap_or("?").to_string(),
                family: e.family.to_string(),
                kind: e.kind.to_string(),
                gated: e.gated,
                note: e.note.to_string(),
                repo: e.repo.to_string(),
            })
            .collect();
        Self { rows, selected: 0 }
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
        self.render_memory_bar(f, rows[0]);

        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(rows[1]);
        self.render_list(f, cols[0]);
        self.render_detail(f, cols[1]);
    }

    fn render_memory_bar(&self, f: &mut Frame, area: Rect) {
        let total = crate::hw::total_ram_gb().max(0.1);
        let used = (total - crate::hw::available_ram_gb()).clamp(0.0, total);
        let ratio = (used / total).clamp(0.0, 1.0);
        let gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(" Memory (unified) "))
            .gauge_style(Style::new().fg(Color::Cyan))
            .ratio(ratio)
            .label(format!("{used:.1} / {total:.1} GB"));
        f.render_widget(gauge, area);
    }

    fn render_list(&self, f: &mut Frame, area: Rect) {
        let mut last_family: Option<&str> = None;
        let mut items: Vec<ListItem> = Vec::new();
        // Track which list index maps to which model row (family headers are
        // interleaved as non-selectable rows would complicate selection, so we
        // instead prefix the family only when it changes, inline).
        for row in &self.rows {
            let fam_changed = last_family != Some(row.family.as_str());
            last_family = Some(row.family.as_str());
            let fam = if fam_changed {
                Span::styled(format!("{:<10}", row.family), Style::new().fg(Color::DarkGray))
            } else {
                Span::raw(format!("{:<10}", ""))
            };
            let gated = if row.gated {
                Span::styled(" 🔒", Style::new().fg(Color::Yellow))
            } else {
                Span::raw("")
            };
            items.push(ListItem::new(Line::from(vec![
                fam,
                Span::styled(format!("{:<16}", row.alias), Style::new().fg(Color::White)),
                Span::styled(row.kind.clone(), Style::new().fg(Color::DarkGray)),
                gated,
            ])));
        }
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(format!(" Models ({}) ", self.rows.len())))
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
                lines.push(Line::from(Span::styled(
                    "[L] Load  [U] Unload  — coming in the next increment",
                    Style::new().fg(Color::DarkGray),
                )));
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
