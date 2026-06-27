//! Scenarios screen (RFC TUI-1 §8). First increment: the SELECT view — browse the
//! workspace's scenario HJSON files (task count + model) and run one. The scenario
//! runner's progress flows to the Output pane automatically (it uses the rerouted
//! `ui::progress`). The EDITOR + nested RUNNER sub-tabs are follow-ups.

use std::path::{Path, PathBuf};

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
};
/// One scenario file's at-a-glance info.
pub struct ScenarioInfo {
    pub path: PathBuf,
    pub name: String,
    pub tasks: usize,
    pub model: String,
}

/// What the App should do after a key.
pub enum ScenariosAction {
    None,
    Run(PathBuf),
}

pub struct ScenariosState {
    pub dir: PathBuf,
    pub files: Vec<ScenarioInfo>,
    pub selected: usize,
    /// Last run status line (set by the App while a run is in flight).
    pub status: String,
}

impl ScenariosState {
    pub fn new(dir: PathBuf) -> Self {
        let mut s = Self { dir, files: Vec::new(), selected: 0, status: String::new() };
        s.rescan();
        s
    }

    /// (Re)scan the scenarios dir for `*.hjson` (excluding `*.run.hjson` sidecars).
    pub fn rescan(&mut self) {
        let mut files = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            for e in rd.flatten() {
                let p = e.path();
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
                if name.ends_with(".hjson") && !name.ends_with(".run.hjson") {
                    let (tasks, model) = peek(&p);
                    files.push(ScenarioInfo {
                        name: name.trim_end_matches(".hjson").to_string(),
                        path: p,
                        tasks,
                        model,
                    });
                }
            }
        }
        files.sort_by(|a, b| a.name.cmp(&b.name));
        self.files = files;
        if self.selected >= self.files.len() {
            self.selected = self.files.len().saturating_sub(1);
        }
    }

    fn next(&mut self) {
        if !self.files.is_empty() {
            self.selected = (self.selected + 1) % self.files.len();
        }
    }

    fn prev(&mut self) {
        if !self.files.is_empty() {
            self.selected = (self.selected + self.files.len() - 1) % self.files.len();
        }
    }

    pub fn selected_path(&self) -> Option<PathBuf> {
        self.files.get(self.selected).map(|f| f.path.clone())
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> ScenariosAction {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                self.next();
                ScenariosAction::None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.prev();
                ScenariosAction::None
            }
            // Run the selected scenario.
            KeyCode::Enter | KeyCode::Char('r' | 'R') => {
                self.selected_path().map(ScenariosAction::Run).unwrap_or(ScenariosAction::None)
            }
            _ => ScenariosAction::None,
        }
    }

    pub fn render(&self, f: &mut Frame, area: Rect) {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(area);
        self.render_list(f, cols[0]);
        self.render_detail(f, cols[1]);
    }

    fn render_list(&self, f: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self
            .files
            .iter()
            .map(|s| {
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{:<22}", s.name), Style::new().fg(Color::White)),
                    Span::styled(format!("{} task{}", s.tasks, if s.tasks == 1 { "" } else { "s" }), Style::new().fg(Color::DarkGray)),
                ]))
            })
            .collect();
        let items = if items.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "No scenarios. Add *.hjson under the workspace scenarios/ dir.",
                Style::new().fg(Color::DarkGray),
            )))]
        } else {
            items
        };
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(format!(" Scenarios ({}) ", self.files.len())))
            .highlight_style(Style::new().bg(Color::Cyan).fg(Color::Black).add_modifier(Modifier::BOLD))
            .highlight_symbol("▶ ");
        let mut ls = ListState::default();
        ls.select(Some(self.selected));
        f.render_stateful_widget(list, area, &mut ls);
    }

    fn render_detail(&self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" Detail ");
        let lines = match self.files.get(self.selected) {
            Some(s) => {
                let mut v = vec![
                    Line::from(Span::styled(s.name.clone(), Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
                    Line::from(format!("Tasks:  {}", s.tasks)),
                    Line::from(format!("Model:  {}", s.model)),
                    Line::from(format!("File:   {}", s.path.display())),
                    Line::from(""),
                    Line::from(Span::styled("[Enter] Run  (progress shows in the Output pane below)", Style::new().fg(Color::Gray))),
                ];
                if !self.status.is_empty() {
                    v.push(Line::from(""));
                    v.push(Line::from(Span::styled(self.status.clone(), Style::new().fg(Color::Yellow))));
                }
                v
            }
            None => vec![Line::from("No scenario selected.")],
        };
        f.render_widget(Paragraph::new(lines).block(block).wrap(Wrap { trim: true }), area);
    }
}

/// Peek a scenario file for `(task_count, model)`, reusing the real scenario
/// parser. Best-effort for the listing — `(0, "?")` on a malformed file.
fn peek(path: &Path) -> (usize, String) {
    crate::cli::scenario::peek(path).unwrap_or((0, "?".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;

    fn key(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("plakat-scen-test-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn scans_hjson_excluding_run_sidecars() {
        let d = tmp_dir("scan");
        // Strict JSON is valid HJSON and sidesteps the quoteless-to-end-of-line trap.
        std::fs::write(d.join("a.hjson"), r#"{"model":"sdxl","tasks":[{"name":"t1"},{"name":"t2"}]}"#).unwrap();
        std::fs::write(d.join("b.hjson"), r#"{"model":"sd15","tasks":[{"name":"t1"}]}"#).unwrap();
        std::fs::write(d.join("a.run.hjson"), "{}").unwrap(); // sidecar — excluded
        let s = ScenariosState::new(d.clone());
        assert_eq!(s.files.len(), 2);
        assert_eq!(s.files[0].name, "a");
        assert_eq!(s.files[0].tasks, 2);
        assert_eq!(s.files[0].model, "sdxl");
        assert_eq!(s.files[1].name, "b");
        assert_eq!(s.files[1].tasks, 1);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn peek_parses_a_real_scenario_file() {
        // The shipped example is real HJSON (comments, quoteless values) — peek must
        // count its tasks and read its model, not fall back to (0, "?").
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/scenario.hjson");
        if !p.exists() {
            return; // example not present in this checkout — skip
        }
        let (tasks, model) = peek(&p);
        assert!(tasks > 0, "expected >0 tasks from the real scenario, got {tasks}");
        assert!(model.contains("stable-diffusion"), "model was {model:?}");
    }

    #[test]
    fn enter_runs_the_selected() {
        let d = tmp_dir("run");
        std::fs::write(d.join("x.hjson"), r#"{"model":"sdxl","tasks":[]}"#).unwrap();
        let mut s = ScenariosState::new(d.clone());
        match s.handle_key(key(KeyCode::Enter)) {
            ScenariosAction::Run(p) => assert!(p.ends_with("x.hjson")),
            _ => panic!("expected Run"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }
}
