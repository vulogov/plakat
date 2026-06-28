//! A compact prompt editor for the Chat input (RFC TUI-1 §6): one logical line of
//! text that SOFT-WRAPS to the box width, with full cursor editing, a fixed-height
//! viewport, and vertical scrolling when the wrapped text exceeds it. tui-textarea
//! scrolls long lines horizontally rather than wrapping, which a prompt box
//! shouldn't — hence this small purpose-built editor.
//!
//! The text is a single logical line (no embedded newlines); Enter submits. Wrapping
//! is char-exact (every char belongs to exactly one visual row), so the cursor maps
//! cleanly between a char index and a (row, col) for Up/Down movement.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

/// What the caller should do after a key the editor handled.
pub enum EditorOutcome {
    /// Consumed (or ignored) — keep editing.
    Consumed,
    /// Enter pressed — submit the current text.
    Submit,
}

pub struct PromptEditor {
    chars: Vec<char>,
    /// Cursor as a char index, `0..=chars.len()`.
    cursor: usize,
    /// First visible visual row.
    scroll: usize,
    /// Last-rendered inner viewport (used by Up/Down + scroll between renders).
    viewport_width: usize,
    viewport_height: usize,
}

impl Default for PromptEditor {
    fn default() -> Self {
        Self { chars: Vec::new(), cursor: 0, scroll: 0, viewport_width: 40, viewport_height: 2 }
    }
}

impl PromptEditor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn text(&self) -> String {
        self.chars.iter().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    /// Replace the whole buffer (e.g. recalling a previous prompt); cursor to end.
    pub fn set_text(&mut self, text: &str) {
        self.chars = text.chars().collect();
        self.cursor = self.chars.len();
        self.scroll = 0;
    }

    pub fn clear(&mut self) {
        self.chars.clear();
        self.cursor = 0;
        self.scroll = 0;
    }

    /// Return the trimmed text and clear the buffer (on submit).
    pub fn take(&mut self) -> String {
        let t = self.text().trim().to_string();
        self.clear();
        t
    }

    /// Handle one key. Returns [`EditorOutcome::Submit`] on Enter; otherwise edits.
    pub fn handle_key(&mut self, key: KeyEvent) -> EditorOutcome {
        match key.code {
            KeyCode::Enter => return EditorOutcome::Submit,
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => self.insert(c),
            KeyCode::Backspace => self.backspace(),
            KeyCode::Delete => self.delete(),
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => self.cursor = (self.cursor + 1).min(self.chars.len()),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.chars.len(),
            KeyCode::Up => self.move_vertical(-1),
            KeyCode::Down => self.move_vertical(1),
            _ => {}
        }
        EditorOutcome::Consumed
    }

    fn insert(&mut self, c: char) {
        self.chars.insert(self.cursor, c);
        self.cursor += 1;
    }

    fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
        }
    }

    fn delete(&mut self) {
        if self.cursor < self.chars.len() {
            self.chars.remove(self.cursor);
        }
    }

    /// Move the cursor up (`-1`) / down (`+1`) one visual row, keeping the column.
    fn move_vertical(&mut self, delta: isize) {
        let w = self.viewport_width.max(1);
        let rows = wrap_rows(&self.chars, w);
        let (r, col) = cursor_rowcol(&self.chars, self.cursor, &rows);
        let target = r as isize + delta;
        if target < 0 {
            self.cursor = 0;
        } else if (target as usize) >= rows.len() {
            self.cursor = self.chars.len();
        } else {
            let (s, e) = rows[target as usize];
            self.cursor = s + col.min(e - s);
        }
    }

    /// Keep the cursor's row inside the viewport (called during render).
    fn ensure_visible(&mut self) {
        let rows = wrap_rows(&self.chars, self.viewport_width.max(1));
        let (r, _) = cursor_rowcol(&self.chars, self.cursor, &rows);
        let h = self.viewport_height.max(1);
        if r < self.scroll {
            self.scroll = r;
        } else if r >= self.scroll + h {
            self.scroll = r + 1 - h;
        }
    }

    /// Render the editor inside `area` with a bordered `title`. Updates the viewport
    /// dimensions (so the next Up/Down/scroll uses the true width).
    pub fn render(&mut self, f: &mut Frame, area: Rect, title: &str) {
        let block = Block::default().borders(Borders::ALL).title(title.to_string());
        let inner = block.inner(area);
        f.render_widget(block, area);
        self.viewport_width = (inner.width as usize).max(1);
        self.viewport_height = (inner.height as usize).max(1);
        self.ensure_visible();

        let rows = wrap_rows(&self.chars, self.viewport_width);
        let (cr, cc) = cursor_rowcol(&self.chars, self.cursor, &rows);
        let cursor_style = Style::new().bg(Color::Cyan).fg(Color::Black);

        let end = (self.scroll + self.viewport_height).min(rows.len());
        let mut lines: Vec<Line> = Vec::new();
        for r in self.scroll..end {
            let (s, e) = rows[r];
            let row: Vec<char> = self.chars[s..e].to_vec();
            if r == cr {
                let before: String = row.iter().take(cc).collect();
                let mut spans = vec![Span::raw(before)];
                match row.get(cc).copied() {
                    Some(ch) => {
                        spans.push(Span::styled(ch.to_string(), cursor_style));
                        spans.push(Span::raw(row.iter().skip(cc + 1).collect::<String>()));
                    }
                    None => spans.push(Span::styled(" ", cursor_style)), // cursor at row end
                }
                lines.push(Line::from(spans));
            } else {
                lines.push(Line::from(row.iter().collect::<String>()));
            }
        }
        f.render_widget(Paragraph::new(lines), inner);
    }
}

/// Char-exact soft wrap: returns `[start, end)` char ranges per visual row, covering
/// every char (spaces at a wrap point stay at the end of their row). Prefers a break
/// at the last space within the width; hard-splits an over-long word.
fn wrap_rows(chars: &[char], width: usize) -> Vec<(usize, usize)> {
    let n = chars.len();
    if width == 0 {
        return vec![(0, n)];
    }
    let mut rows = Vec::new();
    let mut start = 0;
    while start < n {
        let mut end = (start + width).min(n);
        if end < n {
            if let Some(sp) = (start..end).rev().find(|&i| chars[i] == ' ') {
                if sp > start {
                    end = sp + 1; // keep the space at the end of this row
                }
            }
        }
        rows.push((start, end));
        start = end;
    }
    if rows.is_empty() {
        rows.push((0, 0));
    }
    rows
}

/// Map a cursor char index to its `(row, col)` in the wrapped layout.
fn cursor_rowcol(_chars: &[char], cursor: usize, rows: &[(usize, usize)]) -> (usize, usize) {
    for (r, &(s, e)) in rows.iter().enumerate() {
        let last = r + 1 == rows.len();
        if cursor < e || (last && cursor <= e) {
            return (r, cursor.saturating_sub(s));
        }
    }
    let last = rows.len().saturating_sub(1);
    (last, cursor.saturating_sub(rows.get(last).map(|r| r.0).unwrap_or(0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ed(text: &str, width: usize) -> PromptEditor {
        let mut e = PromptEditor::new();
        e.set_text(text);
        e.viewport_width = width;
        e
    }

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn wrap_breaks_at_spaces_and_covers_every_char() {
        let c = chars("the quick brown fox");
        let rows = wrap_rows(&c, 10);
        // Reassembling the rows must reproduce the text exactly (no dropped chars).
        let joined: String = rows.iter().flat_map(|&(s, e)| c[s..e].iter()).collect();
        assert_eq!(joined, "the quick brown fox");
        // Each row fits the width.
        assert!(rows.iter().all(|&(s, e)| e - s <= 10));
        assert!(rows.len() >= 2);
    }

    #[test]
    fn wrap_hard_splits_an_overlong_word() {
        let c = chars("supercalifragilistic");
        let rows = wrap_rows(&c, 8);
        assert_eq!(rows[0], (0, 8));
        let joined: String = rows.iter().flat_map(|&(s, e)| c[s..e].iter()).collect();
        assert_eq!(joined, "supercalifragilistic");
    }

    #[test]
    fn cursor_maps_to_rowcol() {
        let c = chars("the quick brown fox"); // width 10 → "the quick " / "brown fox"
        let rows = wrap_rows(&c, 10);
        assert_eq!(cursor_rowcol(&c, 0, &rows), (0, 0));
        // cursor at the wrap boundary (start of row 2)
        let boundary = rows[1].0;
        assert_eq!(cursor_rowcol(&c, boundary, &rows), (1, 0));
        // cursor at the very end
        assert_eq!(cursor_rowcol(&c, c.len(), &rows).0, rows.len() - 1);
    }

    #[test]
    fn editing_inserts_and_deletes_at_cursor() {
        let mut e = ed("ct", 40);
        e.cursor = 1;
        e.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));
        assert_eq!(e.text(), "cat");
        assert_eq!(e.cursor, 2);
        e.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(e.text(), "ct");
        e.handle_key(KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE));
        assert_eq!(e.text(), "c");
    }

    #[test]
    fn down_then_up_moves_between_wrapped_rows() {
        let mut e = ed("the quick brown fox", 10); // 2 rows
        e.cursor = 2; // row 0, col 2
        e.move_vertical(1); // down → row 1, col 2
        let rows = wrap_rows(&e.chars, 10);
        assert_eq!(cursor_rowcol(&e.chars, e.cursor, &rows).0, 1);
        e.move_vertical(-1); // up → back to row 0
        let rows = wrap_rows(&e.chars, 10);
        assert_eq!(cursor_rowcol(&e.chars, e.cursor, &rows).0, 0);
    }

    #[test]
    fn enter_submits_and_take_trims_and_clears() {
        let mut e = ed("  hello  ", 40);
        assert!(matches!(e.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)), EditorOutcome::Submit));
        assert_eq!(e.take(), "hello");
        assert!(e.is_empty());
    }

    #[test]
    fn ctrl_chars_are_not_typed() {
        let mut e = ed("", 40);
        // Ctrl-P (history recall, handled by the caller) must not insert a 'p'.
        e.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL));
        assert!(e.is_empty());
    }
}
