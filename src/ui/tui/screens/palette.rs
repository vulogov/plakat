//! Command palette (RFC TUI-1 §5) — a fuzzy action launcher overlaid on any screen.
//! `Ctrl-K` opens it; the App fills it with the actions available in the current
//! context (global navigation + screen-specific commands). Type to fuzzy-filter,
//! `↑/↓` to move, `Enter` to run, `Esc` to dismiss. The chosen [`Cmd`] is handed back
//! to the App, which executes it — mostly by replaying a key into the active screen,
//! so the palette stays a thin launcher over the existing handlers.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
};

/// What running a palette entry does. Kept deliberately small: navigate, quit, replay a
/// key into the active screen, or submit a Chat line.
#[derive(Clone)]
pub enum Cmd {
    /// Switch to the screen at this `ActiveScreen` index.
    Goto(usize),
    Quit,
    /// Replay a key event through the App's normal routing (reuses existing handlers).
    Key(KeyEvent),
    /// Submit a Chat line (a slash command), as if typed + Enter.
    Submit(String),
    /// Restart the process in place to fully return the GPU buffer pool.
    HardReset,
    /// Sweep stale download locks + report cache health for the selected model.
    CacheDoctor,
}

/// Outcome of a key while the palette is open.
pub enum PaletteResult {
    /// Still open (filtering / navigating).
    None,
    /// Dismissed without running anything.
    Closed,
    /// Run this command (the palette has closed itself).
    Run(Cmd),
}

struct Entry {
    label: String,
    cmd: Cmd,
}

#[derive(Default)]
pub struct PaletteState {
    open: bool,
    query: String,
    entries: Vec<Entry>,
    /// Indices into `entries` matching the query, best first.
    filtered: Vec<usize>,
    selected: usize,
}

impl PaletteState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Open the palette with the context's commands (`(label, cmd)` pairs).
    pub fn open(&mut self, commands: Vec<(String, Cmd)>) {
        self.entries = commands.into_iter().map(|(label, cmd)| Entry { label, cmd }).collect();
        self.query.clear();
        self.selected = 0;
        self.open = true;
        self.refilter();
    }

    pub fn close(&mut self) {
        self.open = false;
    }

    fn refilter(&mut self) {
        let q = self.query.trim().to_lowercase();
        let mut scored: Vec<(i32, usize)> = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| fuzzy_score(&q, &e.label.to_lowercase()).map(|s| (s, i)))
            .collect();
        // Lower score = better; tie-break by original order (stable).
        scored.sort_by(|a, b| a.0.cmp(&b.0));
        self.filtered = scored.into_iter().map(|(_, i)| i).collect();
        if self.selected >= self.filtered.len() {
            self.selected = self.filtered.len().saturating_sub(1);
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> PaletteResult {
        match key.code {
            KeyCode::Esc => {
                self.close();
                PaletteResult::Closed
            }
            KeyCode::Up => {
                self.selected = self.selected.saturating_sub(1);
                PaletteResult::None
            }
            KeyCode::Down => {
                if self.selected + 1 < self.filtered.len() {
                    self.selected += 1;
                }
                PaletteResult::None
            }
            KeyCode::Enter => match self.filtered.get(self.selected).map(|&i| self.entries[i].cmd.clone()) {
                Some(cmd) => {
                    self.close();
                    PaletteResult::Run(cmd)
                }
                None => PaletteResult::None,
            },
            KeyCode::Backspace => {
                self.query.pop();
                self.refilter();
                PaletteResult::None
            }
            KeyCode::Char(c) => {
                self.query.push(c);
                self.refilter();
                PaletteResult::None
            }
            _ => PaletteResult::None,
        }
    }

    /// Draw the palette as a centered overlay (the caller renders the screen first).
    pub fn render(&self, f: &mut Frame, full: Rect) {
        if !self.open {
            return;
        }
        let area = centered(full, 64, 18);
        f.render_widget(Clear, area);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::new().fg(Color::Magenta))
            .title(" Command palette · type to filter · ↑/↓ · Enter · Esc ");
        let inner = block.inner(area);
        f.render_widget(block, area);

        let mut lines: Vec<Line> = Vec::new();
        lines.push(Line::from(vec![
            Span::styled("> ", Style::new().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::styled(self.query.clone(), Style::new().fg(Color::White)),
            Span::styled("▏", Style::new().fg(Color::Magenta)),
        ]));
        lines.push(Line::from(""));
        let rows = inner.height.saturating_sub(2) as usize;
        // Keep the selection visible (simple window around it).
        let start = self.selected.saturating_sub(rows.saturating_sub(1));
        for (vi, &ei) in self.filtered.iter().enumerate().skip(start).take(rows) {
            let style = if vi == self.selected {
                Style::new().bg(Color::Magenta).fg(Color::Black).add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(Color::Gray)
            };
            let marker = if vi == self.selected { "▶ " } else { "  " };
            lines.push(Line::from(Span::styled(format!("{marker}{}", self.entries[ei].label), style)));
        }
        if self.filtered.is_empty() {
            lines.push(Line::from(Span::styled("  (no matching command)", Style::new().fg(Color::DarkGray))));
        }
        f.render_widget(Paragraph::new(lines), inner);
    }
}

/// Subsequence fuzzy match: every char of `query` must appear in `label` in order.
/// Returns a score (lower is better) favouring a contiguous, early match. An empty
/// query matches everything (score 0).
fn fuzzy_score(query: &str, label: &str) -> Option<i32> {
    if query.is_empty() {
        return Some(0);
    }
    let lb: Vec<char> = label.chars().collect();
    let mut qi = query.chars().peekable();
    let mut score = 0i32;
    let mut last_pos: Option<usize> = None;
    let mut next_q = qi.next();
    for (pos, &lc) in lb.iter().enumerate() {
        if let Some(qc) = next_q {
            if lc == qc {
                if let Some(prev) = last_pos {
                    score += (pos - prev) as i32; // reward adjacency (gap = 1 is ideal)
                } else {
                    score += pos as i32; // reward an early first match
                }
                last_pos = Some(pos);
                next_q = qi.next();
            }
        }
    }
    if next_q.is_none() { Some(score) } else { None }
}

/// A `w`×`h` rect centered in `full` (clamped to fit).
fn centered(full: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(full.width);
    let h = h.min(full.height);
    let x = full.x + (full.width.saturating_sub(w)) / 2;
    let y = full.y + (full.height.saturating_sub(h)) / 2;
    Rect { x, y, width: w, height: h }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;

    fn ch(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    fn sample() -> PaletteState {
        let mut p = PaletteState::new();
        p.open(vec![
            ("Go to Chat".into(), Cmd::Goto(0)),
            ("Go to Models".into(), Cmd::Goto(1)),
            ("Load selected model".into(), Cmd::Key(ch('l'))),
            ("Quit".into(), Cmd::Quit),
        ]);
        p
    }

    #[test]
    fn opens_with_all_entries_and_filters_fuzzily() {
        let mut p = sample();
        assert!(p.is_open());
        assert_eq!(p.filtered.len(), 4, "empty query → all");
        // "mdl" is a subsequence of "Models" (m-o-d-e-l-s) and "Load selected model".
        for c in "model".chars() {
            p.handle_key(ch(c));
        }
        assert!(!p.filtered.is_empty());
        let labels: Vec<&str> = p.filtered.iter().map(|&i| p.entries[i].label.as_str()).collect();
        assert!(labels.iter().any(|l| l.contains("Models")));
        // "Quit" should not survive the "model" filter.
        assert!(!labels.iter().any(|l| *l == "Quit"));
    }

    #[test]
    fn enter_runs_the_selected_command_and_closes() {
        let mut p = sample();
        // Filter to the model loader and run it.
        for c in "load".chars() {
            p.handle_key(ch(c));
        }
        match p.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)) {
            PaletteResult::Run(Cmd::Key(k)) => assert_eq!(k.code, KeyCode::Char('l')),
            _ => panic!("expected Run(Key)"),
        }
        assert!(!p.is_open(), "palette closes after running");
    }

    #[test]
    fn esc_closes_without_running() {
        let mut p = sample();
        assert!(matches!(p.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)), PaletteResult::Closed));
        assert!(!p.is_open());
    }

    #[test]
    fn fuzzy_score_requires_subsequence() {
        assert!(fuzzy_score("ldm", "load model").is_some());
        assert!(fuzzy_score("zzz", "load model").is_none());
        // Earlier / tighter matches score lower (better).
        assert!(fuzzy_score("lo", "load").unwrap() < fuzzy_score("lo", "xxlo").unwrap());
    }
}
