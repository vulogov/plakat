//! The TUI application shell — `App`, the event loop, and global chrome (RFC
//! TUI-1 §4–§5). This Phase-1 increment is the navigable frame: a tab bar, a
//! status bar, `Ctrl-1..8` (or plain `1..8`) screen switching, and per-screen
//! placeholders. The Chat + Models screen bodies, services, and channel draining
//! land in the next increments.

use anyhow::Result;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Block, Borders, Paragraph, Tabs},
};
use ratatui_image::picker::Picker;

use super::workspace::Workspace;

/// The eight screens (RFC §1). Release 1 implements Chat + Models; the rest show a
/// placeholder until their cycle.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ActiveScreen {
    Chat,
    Models,
    Scenarios,
    History,
    LoraHub,
    People,
    PromptWorkspace,
    Canvas,
}

impl ActiveScreen {
    const ALL: [ActiveScreen; 8] = [
        Self::Chat,
        Self::Models,
        Self::Scenarios,
        Self::History,
        Self::LoraHub,
        Self::People,
        Self::PromptWorkspace,
        Self::Canvas,
    ];

    fn title(self) -> &'static str {
        match self {
            Self::Chat => "Chat",
            Self::Models => "Models",
            Self::Scenarios => "Scenarios",
            Self::History => "History",
            Self::LoraHub => "LoRA Hub",
            Self::People => "People",
            Self::PromptWorkspace => "Prompts",
            Self::Canvas => "Canvas",
        }
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|s| *s == self).unwrap_or(0)
    }

    fn from_index(i: usize) -> Option<Self> {
        Self::ALL.get(i).copied()
    }

    /// Whether this screen has a real body yet (Release 1: Chat + Models).
    fn implemented(self) -> bool {
        matches!(self, Self::Chat | Self::Models)
    }
}

/// The running TUI. Holds the workspace, the image `Picker` (for inline previews,
/// used once screens render images), the active screen, and the quit flag. Services
/// (ModelService / GenQueue / LlmPool) join in the next increment.
pub struct App {
    pub workspace: Workspace,
    pub picker: Picker,
    pub screen: ActiveScreen,
    pub should_quit: bool,
}

impl App {
    pub fn new(workspace: Workspace, picker: Picker) -> Self {
        Self { workspace, picker, screen: ActiveScreen::Chat, should_quit: false }
    }

    /// Enter the alternate screen + raw mode, run the loop, and always restore the
    /// terminal on the way out (`ratatui::init` also installs a panic hook that
    /// restores, so a panic won't leave the terminal wedged).
    pub fn run(&mut self) -> Result<()> {
        let mut terminal = ratatui::init();
        let res = self.event_loop(&mut terminal);
        ratatui::restore();
        res
    }

    fn event_loop(&mut self, terminal: &mut ratatui::DefaultTerminal) -> Result<()> {
        while !self.should_quit {
            terminal.draw(|f| self.render(f))?;
            // 100 ms tick: poll input, then (later) drain the gen/llm/download
            // channels so a running generation keeps the UI live.
            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    self.handle_key(key);
                }
            }
        }
        Ok(())
    }

    fn handle_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            // Ctrl-Q always quits; plain `q` / Esc quit too while no screen owns
            // text input (the Chat input will refine this once it lands).
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Char('c') if ctrl => self.should_quit = true,
            // Ctrl-1..8 (RFC) or plain 1..8 (terminals that don't emit Ctrl-digit).
            KeyCode::Char(c @ '1'..='8') => {
                if let Some(s) = ActiveScreen::from_index((c as u8 - b'1') as usize) {
                    self.screen = s;
                }
            }
            _ => {}
        }
    }

    fn render(&self, f: &mut Frame) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(1), Constraint::Length(1)])
            .split(f.area());
        self.render_tab_bar(f, rows[0]);
        self.render_content(f, rows[1]);
        self.render_status_bar(f, rows[2]);
    }

    fn render_tab_bar(&self, f: &mut Frame, area: Rect) {
        let titles = ActiveScreen::ALL
            .iter()
            .enumerate()
            .map(|(i, s)| Line::from(format!(" {} {} ", i + 1, s.title())));
        let tabs = Tabs::new(titles)
            .select(self.screen.index())
            .highlight_style(Style::new().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD));
        f.render_widget(tabs, area);
    }

    fn render_content(&self, f: &mut Frame, area: Rect) {
        let body = if self.screen.implemented() {
            format!(
                "[{}] screen — UI body lands in the next increment.\n\nworkspace: {}\n{}",
                self.screen.title(),
                self.workspace.config.name,
                self.workspace.root.display()
            )
        } else {
            format!("[{}] — coming in a later release (RFC TUI-1).", self.screen.title())
        };
        let block = Block::default().borders(Borders::ALL).title(self.screen.title());
        f.render_widget(Paragraph::new(body).block(block), area);
    }

    fn render_status_bar(&self, f: &mut Frame, area: Rect) {
        let txt = format!(
            " {} · Ctrl-1..8 switch screens · Ctrl-Q / q quit ",
            self.workspace.config.name
        );
        let bar = Paragraph::new(txt).style(Style::new().bg(Color::DarkGray).fg(Color::White));
        f.render_widget(bar, area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::tui::workspace::{Workspace, WorkspaceConfig};

    fn test_app() -> App {
        let ws = Workspace { root: "/tmp/plakat-ui-test".into(), config: WorkspaceConfig::default() };
        // A synthetic Picker (no terminal query) so the navigation logic is testable.
        App::new(ws, Picker::from_fontsize((8, 16)))
    }

    fn key(c: char, ctrl: bool) -> KeyEvent {
        let m = if ctrl { KeyModifiers::CONTROL } else { KeyModifiers::NONE };
        KeyEvent::new(KeyCode::Char(c), m)
    }

    #[test]
    fn digits_switch_screens() {
        let mut a = test_app();
        assert!(matches!(a.screen, ActiveScreen::Chat));
        a.handle_key(key('2', true)); // Ctrl-2
        assert!(matches!(a.screen, ActiveScreen::Models));
        a.handle_key(key('8', false)); // plain 8 (fallback)
        assert!(matches!(a.screen, ActiveScreen::Canvas));
        a.handle_key(key('1', true));
        assert!(matches!(a.screen, ActiveScreen::Chat));
    }

    #[test]
    fn quit_keys_set_should_quit() {
        let mut a = test_app();
        a.handle_key(key('q', false));
        assert!(a.should_quit);
        let mut b = test_app();
        b.handle_key(key('c', true)); // Ctrl-C
        assert!(b.should_quit);
    }

    #[test]
    fn release1_screens_are_implemented() {
        assert!(ActiveScreen::Chat.implemented());
        assert!(ActiveScreen::Models.implemented());
        assert!(!ActiveScreen::Scenarios.implemented());
        assert_eq!(ActiveScreen::ALL.len(), 8);
    }
}
