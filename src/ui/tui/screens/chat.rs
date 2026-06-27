//! Chat screen (RFC TUI-1 §6) — the conversational generation interface. This
//! increment is the shell: a text input, the session history, and input-focus key
//! handling. Generation dispatch + progressive preview land in the next increment.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
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
        let mut items: Vec<ListItem> = Vec::new();
        for (i, e) in self.history.iter().enumerate() {
            items.push(ListItem::new(Line::from(vec![
                Span::styled(format!("{:>2} ▸ ", i + 1), Style::new().fg(Color::Cyan)),
                Span::styled(e.utterance.clone(), Style::new().fg(Color::White)),
            ])));
            if let Some(path) = &e.result {
                items.push(ListItem::new(Line::from(Span::styled(
                    format!("      → {path}"),
                    Style::new().fg(Color::Green),
                ))));
            }
            if let Some(err) = &e.error {
                items.push(ListItem::new(Line::from(Span::styled(
                    format!("      ✗ {err}"),
                    Style::new().fg(Color::Red),
                ))));
            }
        }
        if items.is_empty() {
            items.push(ListItem::new(Line::from(Span::styled(
                "Describe an image and press Enter. (Load a model first in Models / Ctrl-2.)",
                Style::new().fg(Color::DarkGray),
            ))));
        }
        let title = match &self.status {
            ChatStatus::Generating { step, total } => format!(" Chat  ⟳ generating {step}/{total} "),
            ChatStatus::Error(e) => format!(" Chat  ✗ {e} "),
            _ => " Chat ".to_string(),
        };
        let list = List::new(items).block(Block::default().borders(Borders::ALL).title(title));
        f.render_widget(list, area);
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
    fn history_records_utterance_and_result() {
        let mut s = ChatState::new();
        s.push_utterance("a fox".into());
        s.finish_last(Ok("out/chat/plakat-42.png".into()));
        assert_eq!(s.history.len(), 1);
        assert_eq!(s.history[0].result.as_deref(), Some("out/chat/plakat-42.png"));
    }
}
