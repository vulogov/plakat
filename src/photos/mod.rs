//! `plakat photos` — TUI photo & image collection manager (RFC PHOTOS-1, the 3.x flagship).
//!
//! Phase 1 scaffold: walk + classify the library ([`library`]), the per-album HJSON store
//! ([`hjson`]), and a runnable three-pane shell (status bar · tree | album-grid placeholder ·
//! command pane) with a navigable, collapsible Tree pane. Later phases (RFC §29) fill in the grid,
//! image view, editing, browse, and vision features.

pub mod hjson;
pub mod library;

use std::collections::HashSet;
use std::io::stdout;
use std::path::PathBuf;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use library::{LibraryNode, NodeKind};

/// Which pane has focus (RFC §6).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Tree,
    Album,
    Command,
}

/// A flattened, displayable tree row (depth-first pre-order over expanded nodes).
struct Row {
    path: PathBuf,
    name: String,
    kind: NodeKind,
    count: usize,
    depth: usize,
    expanded: bool,
    has_children: bool,
}

struct App {
    root: LibraryNode,
    root_dir: PathBuf,
    expanded: HashSet<PathBuf>,
    cursor: usize,
    focus: Focus,
    status: String,
}

impl App {
    fn new(root: LibraryNode, root_dir: PathBuf) -> Self {
        let mut expanded = HashSet::new();
        expanded.insert(root.path.clone()); // root open by default
        Self { root, root_dir, expanded, cursor: 0, focus: Focus::Tree, status: String::new() }
    }

    /// Depth-first flatten over expanded subtrees.
    fn rows(&self) -> Vec<Row> {
        let mut out = Vec::new();
        fn rec(node: &LibraryNode, depth: usize, exp: &HashSet<PathBuf>, out: &mut Vec<Row>) {
            let is_open = exp.contains(&node.path);
            out.push(Row {
                path: node.path.clone(),
                name: node.name.clone(),
                kind: node.kind,
                count: node.total_images(),
                depth,
                expanded: is_open,
                has_children: !node.children.is_empty(),
            });
            if is_open {
                for c in &node.children {
                    rec(c, depth + 1, exp, out);
                }
            }
        }
        rec(&self.root, 0, &self.expanded, &mut out);
        out
    }

    fn album_count(&self) -> usize {
        fn rec(n: &LibraryNode) -> usize {
            let here = if n.kind == NodeKind::Album { 1 } else { 0 };
            here + n.children.iter().map(rec).sum::<usize>()
        }
        rec(&self.root)
    }
}

/// Entry point. Walks `root_dir`, then runs the TUI until `q` / `Ctrl-b q`.
pub async fn run(root_dir: PathBuf) -> Result<()> {
    anyhow::ensure!(root_dir.is_dir(), "photo root {} is not a directory", root_dir.display());
    let root = library::walk(&root_dir)?;
    let mut app = App::new(root, root_dir);

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let res = event_loop(&mut terminal, &mut app);

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    res
}

fn event_loop(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>, app: &mut App) -> Result<()> {
    loop {
        terminal.draw(|f| draw(f, app))?;
        if let Event::Key(k) = event::read()? {
            if k.kind != KeyEventKind::Press {
                continue;
            }
            let rows = app.rows();
            match k.code {
                KeyCode::Char('q') => break,
                KeyCode::Char('j') | KeyCode::Down => {
                    app.cursor = (app.cursor + 1).min(rows.len().saturating_sub(1));
                }
                KeyCode::Char('k') | KeyCode::Up => {
                    app.cursor = app.cursor.saturating_sub(1);
                }
                KeyCode::Char('g') => app.cursor = 0,
                KeyCode::Char('G') => app.cursor = rows.len().saturating_sub(1),
                // Expand / open.
                KeyCode::Char('l') | KeyCode::Right | KeyCode::Enter => {
                    if let Some(r) = rows.get(app.cursor) {
                        if r.has_children {
                            app.expanded.insert(r.path.clone());
                        } else if r.kind == NodeKind::Album {
                            app.status = format!("album: {} ({} images) — grid pane lands in Phase 1", r.name, r.count);
                        }
                    }
                }
                // Collapse.
                KeyCode::Char('h') | KeyCode::Left => {
                    if let Some(r) = rows.get(app.cursor) {
                        app.expanded.remove(&r.path);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn draw(f: &mut Frame, app: &App) {
    let [status_bar, body, cmd_pane] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Fill(1),
        Constraint::Length(3),
    ])
    .areas(f.area());

    // Status bar.
    let status = format!(
        " plakat photos  {}   {} albums · {} images   {}",
        app.root_dir.display(),
        app.album_count(),
        app.root.total_images(),
        app.status,
    );
    f.render_widget(
        Paragraph::new(status).style(Style::default().fg(Color::Black).bg(Color::Gray)),
        status_bar,
    );

    let [tree_col, album_col] =
        Layout::horizontal([Constraint::Percentage(30), Constraint::Fill(1)]).areas(body);

    // Tree pane.
    let tree_active = app.focus == Focus::Tree;
    let rows = app.rows();
    let lines: Vec<Line> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let icon = match (r.kind, r.has_children, r.expanded) {
                (NodeKind::Album, false, _) => "│ ",
                (_, true, true) => "▼ ",
                (_, true, false) => "▶ ",
                _ => "  ",
            };
            let text = format!(
                "{}{}{}  [{}]",
                "  ".repeat(r.depth),
                icon,
                r.name,
                r.count
            );
            let mut style = Style::default();
            if i == app.cursor {
                style = style.add_modifier(Modifier::REVERSED);
            }
            Line::from(Span::styled(text, style))
        })
        .collect();
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Library ")
                .border_style(Style::default().fg(if tree_active { Color::Yellow } else { Color::DarkGray })),
        ),
        tree_col,
    );

    // Album grid placeholder (Phase 1 continues here).
    f.render_widget(
        Paragraph::new("\n  Album grid — Phase 1 (thumbnail grid) in progress.\n  Select an album in the tree.")
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Album ")
                    .border_style(Style::default().fg(Color::DarkGray)),
            ),
        album_col,
    );

    // Command pane.
    f.render_widget(
        Paragraph::new(" CMD ▶ ").block(Block::default().borders(Borders::ALL)),
        cmd_pane,
    );
}
