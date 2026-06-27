//! Chat screen (RFC TUI-1 §6) — the conversational generation interface: a text
//! input, the session history, the inline image, and live progress. The first
//! prompt generates (txt2img); once an image exists, a follow-up prompt refines it
//! (img2img over the previous output) unless it starts with `/new`. The App owns
//! the dispatch + refine decision; this screen renders state + handles input keys.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

/// One turn in the session: the user's utterance + (once generated) its result.
pub struct ChatEntry {
    pub utterance: String,
    pub result: Option<String>,
    pub error: Option<String>,
    /// This turn refined the previous image (img2img) rather than generating fresh.
    pub refine: bool,
}

/// Live generation status for the status line.
#[derive(Clone, Default)]
pub enum ChatStatus {
    #[default]
    Idle,
    Generating {
        step: u32,
        total: u32,
        refine: bool,
    },
    Done(String),
    Error(String),
}

/// What the App should do after a key (so the screen stays dispatch-free).
pub enum ChatAction {
    None,
    Submit(String),
}

pub struct ChatState {
    pub input: String,
    /// Cursor position in the input, as a CHAR index (0..=char count).
    pub cursor: usize,
    pub history: Vec<ChatEntry>,
    pub status: ChatStatus,
    /// The latest preview / final image to show in the right pane (built by the
    /// App from `GenMessage` frames via the image Picker).
    pub preview: Option<ratatui_image::protocol::StatefulProtocol>,
    /// True once a previous image exists, so the next prompt refines it (img2img).
    /// Surfaced in the input hint; the App owns the actual refine decision.
    pub refine_armed: bool,
}

impl ChatState {
    pub fn new() -> Self {
        Self {
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            status: ChatStatus::Idle,
            preview: None,
            refine_armed: false,
        }
    }

    fn input_len(&self) -> usize {
        self.input.chars().count()
    }

    /// Byte offset of the char at `char_idx` (or end of string).
    fn byte_at(&self, char_idx: usize) -> usize {
        self.input.char_indices().nth(char_idx).map(|(b, _)| b).unwrap_or(self.input.len())
    }

    /// Handle a key while the Chat input is focused — a cursor-aware single-line
    /// editor (insert/delete at the cursor, ←/→ Home/End). Plain characters type
    /// into the input (the App routes global keys away first); Enter submits.
    pub fn handle_key(&mut self, key: KeyEvent) -> ChatAction {
        match key.code {
            KeyCode::Char(c) => {
                let b = self.byte_at(self.cursor);
                self.input.insert(b, c);
                self.cursor += 1;
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    let b = self.byte_at(self.cursor - 1);
                    self.input.remove(b);
                    self.cursor -= 1;
                }
            }
            KeyCode::Delete => {
                if self.cursor < self.input_len() {
                    let b = self.byte_at(self.cursor);
                    self.input.remove(b);
                }
            }
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => self.cursor = (self.cursor + 1).min(self.input_len()),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.input_len(),
            KeyCode::Enter => {
                let prompt = self.input.trim().to_string();
                if !prompt.is_empty() {
                    self.input.clear();
                    self.cursor = 0;
                    return ChatAction::Submit(prompt);
                }
            }
            _ => {}
        }
        ChatAction::None
    }

    /// Record a submitted utterance (the App calls this when dispatching it).
    pub fn push_utterance(&mut self, utterance: String, refine: bool) {
        self.history.push(ChatEntry { utterance, result: None, error: None, refine });
    }

    /// Mark the most recent entry done / failed (called when a generation finishes).
    pub fn finish_last(&mut self, result: Result<String, String>) {
        if let Some(last) = self.history.last_mut() {
            match result {
                Ok(path) => last.result = Some(path),
                Err(e) => last.error = Some(e),
            }
        }
    }

    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(3)])
            .split(area);
        // Top: chat history on the left, the generated image on the right.
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(rows[0]);
        self.render_history(f, cols[0]);
        self.render_image(f, cols[1]);
        self.render_input(f, rows[1]);
    }

    fn render_image(&mut self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" Image ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        match &mut self.preview {
            Some(protocol) => {
                // ratatui-image renders the decoded image inline via the terminal's
                // graphics protocol (Kitty/iTerm2/Sixel).
                f.render_stateful_widget(ratatui_image::StatefulImage::new(), inner, protocol);
            }
            None => {
                let body = "\n  The generated image will appear here.\n\n  Type a prompt on the left and press Enter.";
                f.render_widget(Paragraph::new(body).wrap(Wrap { trim: false }), inner);
            }
        }
    }

    fn render_history(&self, f: &mut Frame, area: Rect) {
        let inner_w = area.width.saturating_sub(2) as usize;
        let mut lines: Vec<Line> = Vec::new();
        for (i, e) in self.history.iter().enumerate() {
            // ↻ marks a refinement of the previous image; ▸ a fresh generation.
            let (glyph, gcolor) = if e.refine { ("↻", Color::Magenta) } else { ("▸", Color::Cyan) };
            wrap_entry(&mut lines, &format!("{:>2} {glyph} ", i + 1), &e.utterance, gcolor, Color::White, inner_w);
            if let Some(path) = &e.result {
                wrap_entry(&mut lines, "      → ", path, Color::Green, Color::Green, inner_w);
            }
            if let Some(err) = &e.error {
                wrap_entry(&mut lines, "      ✗ ", err, Color::Red, Color::Red, inner_w);
            }
        }
        if lines.is_empty() {
            lines.push(Line::from(Span::styled(
                "Describe an image and press Enter. (Load a model first in Models / Ctrl-2.)",
                Style::new().fg(Color::DarkGray),
            )));
        }
        // Keep the tail visible (latest messages) within the pane height.
        let rows = area.height.saturating_sub(2) as usize;
        let start = lines.len().saturating_sub(rows.max(1));
        let visible: Vec<Line> = lines.split_off(start);

        let title = match &self.status {
            ChatStatus::Generating { step, total, refine } => {
                let verb = if *refine { "refining" } else { "generating" };
                format!(" Chat  ⟳ {verb} {step}/{total} ")
            }
            ChatStatus::Error(e) => format!(" Chat  ✗ {e} "),
            _ => " Chat ".to_string(),
        };
        f.render_widget(
            Paragraph::new(visible).block(Block::default().borders(Borders::ALL).title(title)),
            area,
        );
    }

    fn render_input(&self, f: &mut Frame, area: Rect) {
        // Render the prompt with a block cursor at its char position (insert mode).
        let chars: Vec<char> = self.input.chars().collect();
        let before: String = chars.iter().take(self.cursor).collect();
        let cursor_style = Style::new().bg(Color::Cyan).fg(Color::Black);
        let mut spans = vec![
            Span::styled("> ", Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(before),
        ];
        match chars.get(self.cursor) {
            Some(c) => {
                spans.push(Span::styled(c.to_string(), cursor_style));
                spans.push(Span::raw(chars.iter().skip(self.cursor + 1).collect::<String>()));
            }
            None => spans.push(Span::styled(" ", cursor_style)), // cursor at end
        }
        // Once an image exists, the next prompt refines it; advertise that + the
        // /new escape hatch for a fresh generation.
        let title = if self.refine_armed {
            " prompt · Enter to refine · /new <prompt> = fresh "
        } else {
            " prompt · Enter to generate "
        };
        let block = Block::default().borders(Borders::ALL).title(title);
        f.render_widget(Paragraph::new(Line::from(spans)).block(block).wrap(Wrap { trim: false }), area);
    }
}

/// Word-wrap `text` to `width` columns (char-based; hard-splits a word longer than
/// `width`, e.g. a long path). Always returns at least one line.
fn wrap_to(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_len = 0usize;
    let add = |lines: &mut Vec<String>, cur: &mut String, cur_len: &mut usize, piece: &str| {
        let pl = piece.chars().count();
        if *cur_len == 0 {
            cur.push_str(piece);
            *cur_len = pl;
        } else if *cur_len + 1 + pl <= width {
            cur.push(' ');
            cur.push_str(piece);
            *cur_len += 1 + pl;
        } else {
            lines.push(std::mem::take(cur));
            cur.push_str(piece);
            *cur_len = pl;
        }
    };
    for word in text.split_whitespace() {
        if word.chars().count() <= width {
            add(&mut lines, &mut cur, &mut cur_len, word);
        } else {
            let chars: Vec<char> = word.chars().collect();
            for chunk in chars.chunks(width) {
                let piece: String = chunk.iter().collect();
                add(&mut lines, &mut cur, &mut cur_len, &piece);
            }
        }
    }
    if !cur.is_empty() || lines.is_empty() {
        lines.push(cur);
    }
    lines
}

/// Push a history entry (a `prefix` + `text`) into `lines`, wrapped to `width`.
/// The first line shows the coloured prefix; continuation lines are indented to
/// align under the text.
fn wrap_entry(
    lines: &mut Vec<Line<'static>>,
    prefix: &str,
    text: &str,
    prefix_color: Color,
    text_color: Color,
    width: usize,
) {
    let plen = prefix.chars().count();
    let avail = width.saturating_sub(plen).max(1);
    let indent = " ".repeat(plen);
    for (k, w) in wrap_to(text, avail).into_iter().enumerate() {
        if k == 0 {
            lines.push(Line::from(vec![
                Span::styled(prefix.to_string(), Style::new().fg(prefix_color)),
                Span::styled(w, Style::new().fg(text_color)),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::raw(indent.clone()),
                Span::styled(w, Style::new().fg(text_color)),
            ]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;

    fn k(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    #[test]
    fn typing_builds_the_input() {
        let mut s = ChatState::new();
        for c in "cat".chars() {
            assert!(matches!(s.handle_key(k(KeyCode::Char(c))), ChatAction::None));
        }
        assert_eq!(s.input, "cat");
        s.handle_key(k(KeyCode::Backspace));
        assert_eq!(s.input, "ca");
    }

    #[test]
    fn cursor_editing_inserts_and_deletes_mid_string() {
        let mut s = ChatState::new();
        for c in "ct".chars() {
            s.handle_key(k(KeyCode::Char(c)));
        }
        assert_eq!((s.input.as_str(), s.cursor), ("ct", 2));
        s.handle_key(k(KeyCode::Left)); // between c and t
        assert_eq!(s.cursor, 1);
        s.handle_key(k(KeyCode::Char('a'))); // insert mid-string
        assert_eq!((s.input.as_str(), s.cursor), ("cat", 2));
        s.handle_key(k(KeyCode::Home));
        assert_eq!(s.cursor, 0);
        s.handle_key(k(KeyCode::Delete)); // delete 'c' at cursor
        assert_eq!((s.input.as_str(), s.cursor), ("at", 0));
        s.handle_key(k(KeyCode::End));
        assert_eq!(s.cursor, 2);
    }

    #[test]
    fn enter_submits_and_clears() {
        let mut s = ChatState::new();
        for c in "a fox".chars() {
            s.handle_key(k(KeyCode::Char(c)));
        }
        match s.handle_key(k(KeyCode::Enter)) {
            ChatAction::Submit(p) => assert_eq!(p, "a fox"),
            _ => panic!("expected submit"),
        }
        assert!(s.input.is_empty());
    }

    #[test]
    fn empty_enter_is_noop() {
        let mut s = ChatState::new();
        assert!(matches!(s.handle_key(k(KeyCode::Enter)), ChatAction::None));
    }

    #[test]
    fn wrap_to_wraps_at_width_and_preserves_words() {
        let out = wrap_to("the quick brown fox jumps", 9);
        assert!(out.iter().all(|l| l.chars().count() <= 9), "no line exceeds width");
        assert_eq!(out.join(" "), "the quick brown fox jumps", "words preserved in order");
        assert!(out.len() >= 3);
        // a long, space-less token (e.g. a path) is hard-split, not truncated.
        let p = wrap_to("/a/very/long/path/with/no/spaces/at/all.png", 10);
        assert!(p.iter().all(|l| l.chars().count() <= 10));
        assert_eq!(p.concat(), "/a/very/long/path/with/no/spaces/at/all.png");
    }

    #[test]
    fn history_records_utterance_and_result() {
        let mut s = ChatState::new();
        s.push_utterance("a fox".into(), false);
        s.finish_last(Ok("out/chat/plakat-42.png".into()));
        assert_eq!(s.history.len(), 1);
        assert_eq!(s.history[0].result.as_deref(), Some("out/chat/plakat-42.png"));
        assert!(!s.history[0].refine);
    }

    #[test]
    fn refine_turn_is_flagged() {
        let mut s = ChatState::new();
        s.push_utterance("make it warmer".into(), true);
        assert!(s.history[0].refine, "a refinement turn records refine=true");
    }
}
