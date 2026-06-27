//! Chat screen (RFC TUI-1 §6) — the conversational generation interface. This
//! increment is the shell: a text input, the session history, and input-focus key
//! handling. Generation dispatch + progressive preview land in the next increment.

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
}

/// Live generation status for the status line.
#[derive(Clone, Default)]
pub enum ChatStatus {
    #[default]
    Idle,
    Generating {
        step: u32,
        total: u32,
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
    pub history: Vec<ChatEntry>,
    pub status: ChatStatus,
    /// The latest preview / final image to show in the right pane (built by the
    /// App from `GenMessage` frames via the image Picker).
    pub preview: Option<ratatui_image::protocol::StatefulProtocol>,
}

impl ChatState {
    pub fn new() -> Self {
        Self { input: String::new(), history: Vec::new(), status: ChatStatus::Idle, preview: None }
    }

    /// Handle a key while the Chat input is focused. Plain characters type into the
    /// input (the App routes global keys away first); Enter submits.
    pub fn handle_key(&mut self, key: KeyEvent) -> ChatAction {
        match key.code {
            KeyCode::Char(c) => {
                self.input.push(c);
                ChatAction::None
            }
            KeyCode::Backspace => {
                self.input.pop();
                ChatAction::None
            }
            KeyCode::Enter => {
                let prompt = self.input.trim().to_string();
                if prompt.is_empty() {
                    ChatAction::None
                } else {
                    self.input.clear();
                    ChatAction::Submit(prompt)
                }
            }
            _ => ChatAction::None,
        }
    }

    /// Record a submitted utterance (the App calls this when dispatching it).
    pub fn push_utterance(&mut self, utterance: String) {
        self.history.push(ChatEntry { utterance, result: None, error: None });
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
            wrap_entry(&mut lines, &format!("{:>2} ▸ ", i + 1), &e.utterance, Color::Cyan, Color::White, inner_w);
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
            ChatStatus::Generating { step, total } => format!(" Chat  ⟳ generating {step}/{total} "),
            ChatStatus::Error(e) => format!(" Chat  ✗ {e} "),
            _ => " Chat ".to_string(),
        };
        f.render_widget(
            Paragraph::new(visible).block(Block::default().borders(Borders::ALL).title(title)),
            area,
        );
    }

    fn render_input(&self, f: &mut Frame, area: Rect) {
        let line = Line::from(vec![
            Span::styled("> ", Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(self.input.as_str()),
            // A block cursor.
            Span::styled("▌", Style::new().fg(Color::Cyan)),
        ]);
        let block = Block::default().borders(Borders::ALL).title(" prompt · Enter to generate ");
        f.render_widget(Paragraph::new(line).block(block).wrap(Wrap { trim: false }), area);
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
    let mut add = |lines: &mut Vec<String>, cur: &mut String, cur_len: &mut usize, piece: &str| {
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
        s.push_utterance("a fox".into());
        s.finish_last(Ok("out/chat/plakat-42.png".into()));
        assert_eq!(s.history.len(), 1);
        assert_eq!(s.history[0].result.as_deref(), Some("out/chat/plakat-42.png"));
    }
}
