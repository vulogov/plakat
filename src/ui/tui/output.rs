//! The "Output" pane — a bounded, scrollable log of messages + live progress,
//! fed by the rerouted indicatif sink (`ui::progress::install_tui_sink`). Every
//! pipeline's output (model load, downloads, the denoise `pos/len` bar, scenario
//! runs) flows here, so any screen can show what's happening. A live bar updates
//! in place (consecutive frames with the same label replace, not append).

use std::collections::VecDeque;

use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Style},
    text::Line,
    widgets::{Block, Borders, List, ListItem},
};

pub struct OutputPane {
    lines: VecDeque<String>,
    max: usize,
}

impl Default for OutputPane {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputPane {
    pub fn new() -> Self {
        Self { lines: VecDeque::new(), max: 500 }
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Snapshot of the captured lines (test-only introspection).
    #[cfg(test)]
    pub fn lines_for_test(&self) -> Vec<String> {
        self.lines.iter().cloned().collect()
    }

    /// Append a captured line. If it's a new frame of the same bar/spinner as the
    /// last line (same label after stripping the leading glyph), it replaces the
    /// last line so a live bar animates in place rather than flooding the log.
    pub fn push(&mut self, line: String) {
        if let Some(last) = self.lines.back_mut() {
            if !label_key(last).is_empty() && label_key(last) == label_key(&line) {
                *last = line;
                return;
            }
        }
        self.lines.push_back(line);
        while self.lines.len() > self.max {
            self.lines.pop_front();
        }
    }

    /// Render the last `area.height` lines (a tail view) in a bordered pane.
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let rows = area.height.saturating_sub(2) as usize; // minus the border
        let start = self.lines.len().saturating_sub(rows.max(1));
        let items: Vec<ListItem> = self
            .lines
            .iter()
            .skip(start)
            .map(|l| ListItem::new(Line::from(l.as_str())))
            .collect();
        let list = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Output ")
                .border_style(Style::new().fg(Color::DarkGray)),
        );
        f.render_widget(list, area);
    }
}

/// The stable label of a progress line: strip a leading status glyph (spinner
/// frame, ✓, ✗, ⤓, etc.) + spaces, then take everything up to the first `[` (bar)
/// or ASCII digit (counter). Two frames of the same bar share this key.
fn label_key(s: &str) -> &str {
    let trimmed = s.trim_start_matches(|c: char| !c.is_ascii_alphanumeric()).trim_start();
    let end = trimmed
        .find(|c: char| c == '[' || c.is_ascii_digit())
        .unwrap_or(trimmed.len());
    trimmed[..end].trim_end()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spinner_frames_update_in_place() {
        let mut o = OutputPane::new();
        o.push("⠋ Loading SD core".into());
        o.push("⠙ Loading SD core".into());
        o.push("⠹ Loading SD core".into());
        assert_eq!(o.lines.len(), 1, "spinner frames collapse to one live line");
        assert_eq!(o.lines.back().unwrap(), "⠹ Loading SD core");
    }

    #[test]
    fn distinct_phases_append() {
        let mut o = OutputPane::new();
        o.push("⠋ Loading SD core".into());
        o.push("✓ base weights ready".into());
        o.push("⠋ Merging LoRA into UNet".into());
        assert_eq!(o.lines.len(), 3);
    }

    #[test]
    fn a_bar_advancing_updates_in_place() {
        let mut o = OutputPane::new();
        o.push("⤓ repo model.safetensors [==>   ] 30%".into());
        o.push("⤓ repo model.safetensors [====> ] 55%".into());
        assert_eq!(o.lines.len(), 1);
        assert!(o.lines.back().unwrap().contains("55%"));
    }

    #[test]
    fn bounded_to_max() {
        let mut o = OutputPane::new();
        o.max = 3;
        // Distinct labels (no shared key) so each appends rather than dedup-replacing.
        for w in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"] {
            o.push(format!("{w} ready"));
        }
        assert_eq!(o.lines.len(), 3);
        assert_eq!(o.lines.back().unwrap(), "zeta ready");
    }
}
