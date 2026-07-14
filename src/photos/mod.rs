//! `plakat photos` — TUI photo & image collection manager (RFC PHOTOS-1, the 3.x flagship).
//!
//! Phase 1: walk + classify the library ([`library`]), the per-album HJSON store ([`hjson`]), image
//! + RAW decode / EXIF / thumbnail cache ([`loader`], [`exif`]), and a three-pane shell (status bar ·
//! tree | album grid · command) with a navigable Tree and a lazily-rendered thumbnail grid. Later
//! phases (RFC §29) add the image view, editing, browse, and vision features.

pub mod exif;
pub mod hjson;
pub mod library;
pub mod loader;
pub mod watcher;

use std::collections::{HashMap, HashSet};
use std::io::stdout;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use ratatui_image::StatefulImage;

use library::{LibraryNode, NodeKind};

/// Which pane has focus (RFC §6).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Tree,
    Album,
}

/// A flattened, displayable tree row.
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
    tree_cursor: usize,
    focus: Focus,

    // Album grid.
    album_dir: Option<PathBuf>,
    album_paths: Vec<PathBuf>,
    album_cursor: usize,
    thumbs: HashMap<PathBuf, StatefulProtocol>,
    cols: usize,
    thumb_px: u32,

    picker: Picker,
    status: String,

    // Live filesystem watch (RFC §23) + debounce.
    watch: Option<watcher::Watch>,
    dirty_since: Option<Instant>,
}

impl App {
    fn new(root: LibraryNode, root_dir: PathBuf, picker: Picker, thumb_px: u32) -> Self {
        let mut expanded = HashSet::new();
        expanded.insert(root.path.clone());
        Self {
            root,
            root_dir,
            expanded,
            tree_cursor: 0,
            focus: Focus::Tree,
            album_dir: None,
            album_paths: Vec::new(),
            album_cursor: 0,
            thumbs: HashMap::new(),
            cols: 4,
            thumb_px,
            picker,
            status: String::from("↑/↓ move · →/Enter open · h collapse · Tab pane · q quit"),
            watch: None,
            dirty_since: None,
        }
    }

    /// Re-walk the library and reload the open album (after a filesystem change). Keeps the album
    /// cursor in range; new thumbnails decode lazily on the next ticks.
    fn rescan(&mut self) {
        if let Ok(root) = library::walk(&self.root_dir) {
            self.root = root;
        }
        if let Some(dir) = self.album_dir.clone() {
            if dir.is_dir() {
                let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
                    .into_iter()
                    .flatten()
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.is_file() && library::is_image(p))
                    .collect();
                paths.sort();
                self.thumbs.retain(|k, _| paths.contains(k)); // drop thumbs for removed files
                self.album_paths = paths;
                self.album_cursor = self.album_cursor.min(self.album_paths.len().saturating_sub(1));
            }
        }
    }

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
            (n.kind == NodeKind::Album) as usize + n.children.iter().map(rec).sum::<usize>()
        }
        rec(&self.root)
    }

    /// Open the album at `dir`: list its images (non-recursive), reset the grid, switch focus.
    fn open_album(&mut self, dir: PathBuf) {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && library::is_image(p))
            .collect();
        paths.sort();
        self.status = format!("{}  ·  {} images", dir.display(), paths.len());
        self.album_dir = Some(dir);
        self.album_paths = paths;
        self.album_cursor = 0;
        self.thumbs.clear();
        self.focus = Focus::Album;
    }

    /// Lazily decode a few thumbnails per tick for images near the cursor that aren't built yet.
    fn build_thumbs(&mut self, budget: usize) {
        let mut built = 0;
        // Prioritise from the cursor forward (the visible page), then wrap.
        let n = self.album_paths.len();
        for off in 0..n {
            if built >= budget {
                break;
            }
            let idx = (self.album_cursor + off) % n.max(1);
            let Some(path) = self.album_paths.get(idx).cloned() else { break };
            if self.thumbs.contains_key(&path) {
                continue;
            }
            match loader::get_or_render_thumb(&path, self.thumb_px).and_then(|c| {
                Ok(image::open(&c)?)
            }) {
                Ok(img) => {
                    self.thumbs.insert(path, self.picker.new_resize_protocol(img));
                    built += 1;
                }
                Err(_) => {
                    // Insert a 1×1 placeholder protocol so we don't retry a broken file every tick.
                    let ph = image::DynamicImage::new_rgb8(1, 1);
                    self.thumbs.insert(path, self.picker.new_resize_protocol(ph));
                }
            }
        }
    }
}

/// Entry point. Walks `root_dir`, then runs the TUI until `q`.
pub async fn run_with(root_dir: PathBuf, thumb_px: u32) -> Result<()> {
    anyhow::ensure!(root_dir.is_dir(), "photo root {} is not a directory", root_dir.display());
    let picker = Picker::from_query_stdio().map_err(|_| {
        anyhow::anyhow!(
            "no graphics-capable terminal detected. `plakat photos` needs a terminal with pixel \
             image support (Kitty, Ghostty, WezTerm, iTerm2, or Sixel)."
        )
    })?;
    let root = library::walk(&root_dir)?;
    let mut app = App::new(root, root_dir.clone(), picker, thumb_px);
    // Live watch (best-effort — the manager still works statically if it can't be started).
    app.watch = watcher::spawn(&root_dir).ok();

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    let res = event_loop(&mut terminal, &mut app);
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    res
}

/// Back-compat entry (default thumb size).
pub async fn run(root_dir: PathBuf) -> Result<()> {
    run_with(root_dir, 128).await
}

fn event_loop(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>, app: &mut App) -> Result<()> {
    loop {
        terminal.draw(|f| draw(f, app))?;
        // Poll with a timeout so thumbnails keep decoding even without input.
        if event::poll(Duration::from_millis(80))? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                if handle_key(app, k.code) {
                    break;
                }
            }
        }
        if app.focus == Focus::Album && !app.album_paths.is_empty() {
            app.build_thumbs(2);
        }
        // Filesystem watch: coalesce change signals, then rescan after a 500 ms quiet period.
        if let Some(w) = &app.watch {
            let mut changed = false;
            while w.rx.try_recv().is_ok() {
                changed = true;
            }
            if changed {
                app.dirty_since = Some(Instant::now());
            }
        }
        if let Some(t) = app.dirty_since {
            if t.elapsed() >= Duration::from_millis(500) {
                app.rescan();
                app.dirty_since = None;
            }
        }
    }
    Ok(())
}

/// Returns true to quit.
fn handle_key(app: &mut App, code: KeyCode) -> bool {
    match app.focus {
        Focus::Tree => handle_tree_key(app, code),
        Focus::Album => handle_album_key(app, code),
    }
}

fn handle_tree_key(app: &mut App, code: KeyCode) -> bool {
    let rows = app.rows();
    match code {
        KeyCode::Char('q') => return true,
        KeyCode::Char('j') | KeyCode::Down => {
            app.tree_cursor = (app.tree_cursor + 1).min(rows.len().saturating_sub(1));
        }
        KeyCode::Char('k') | KeyCode::Up => app.tree_cursor = app.tree_cursor.saturating_sub(1),
        KeyCode::Char('g') => app.tree_cursor = 0,
        KeyCode::Char('G') => app.tree_cursor = rows.len().saturating_sub(1),
        KeyCode::Char('l') | KeyCode::Right | KeyCode::Enter => {
            if let Some(r) = rows.get(app.tree_cursor) {
                if r.kind == NodeKind::Album {
                    app.open_album(r.path.clone());
                } else if r.has_children {
                    app.expanded.insert(r.path.clone());
                }
            }
        }
        KeyCode::Char('h') | KeyCode::Left => {
            if let Some(r) = rows.get(app.tree_cursor) {
                app.expanded.remove(&r.path);
            }
        }
        KeyCode::Tab => {
            if app.album_dir.is_some() {
                app.focus = Focus::Album;
            }
        }
        _ => {}
    }
    false
}

fn handle_album_key(app: &mut App, code: KeyCode) -> bool {
    let n = app.album_paths.len();
    let cols = app.cols.max(1);
    match code {
        KeyCode::Char('q') => return true,
        KeyCode::Esc | KeyCode::Tab => app.focus = Focus::Tree,
        KeyCode::Char('l') | KeyCode::Right => {
            if app.album_cursor + 1 < n {
                app.album_cursor += 1;
            }
        }
        KeyCode::Char('h') | KeyCode::Left => app.album_cursor = app.album_cursor.saturating_sub(1),
        KeyCode::Char('j') | KeyCode::Down => {
            if app.album_cursor + cols < n {
                app.album_cursor += cols;
            }
        }
        KeyCode::Char('k') | KeyCode::Up => {
            app.album_cursor = app.album_cursor.saturating_sub(cols);
        }
        KeyCode::Char('g') => app.album_cursor = 0,
        KeyCode::Char('G') => app.album_cursor = n.saturating_sub(1),
        KeyCode::Char('[') => app.cols = app.cols.saturating_sub(1).max(1),
        KeyCode::Char(']') => app.cols = (app.cols + 1).min(12),
        _ => {}
    }
    false
}

fn draw(f: &mut Frame, app: &mut App) {
    let [status_bar, body, cmd_pane] =
        Layout::vertical([Constraint::Length(1), Constraint::Fill(1), Constraint::Length(3)])
            .areas(f.area());

    let status = format!(
        " plakat photos  {}   {} albums · {} images   {}",
        app.root_dir.display(),
        app.album_count(),
        app.root.total_images(),
        app.status,
    );
    f.render_widget(Paragraph::new(status).style(Style::default().fg(Color::Black).bg(Color::Gray)), status_bar);

    let [tree_col, album_col] =
        Layout::horizontal([Constraint::Percentage(30), Constraint::Fill(1)]).areas(body);

    draw_tree(f, app, tree_col);
    draw_album(f, app, album_col);

    f.render_widget(Paragraph::new(" CMD ▶ ").block(Block::default().borders(Borders::ALL)), cmd_pane);
}

fn draw_tree(f: &mut Frame, app: &App, area: Rect) {
    let active = app.focus == Focus::Tree;
    let lines: Vec<Line> = app
        .rows()
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let icon = match (r.kind, r.has_children, r.expanded) {
                (NodeKind::Album, false, _) => "│ ",
                (_, true, true) => "▼ ",
                (_, true, false) => "▶ ",
                _ => "  ",
            };
            let text = format!("{}{}{}  [{}]", "  ".repeat(r.depth), icon, r.name, r.count);
            let mut style = Style::default();
            if i == app.tree_cursor {
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
                .border_style(Style::default().fg(if active { Color::Yellow } else { Color::DarkGray })),
        ),
        area,
    );
}

fn draw_album(f: &mut Frame, app: &mut App, area: Rect) {
    let active = app.focus == Focus::Album;
    let title = match &app.album_dir {
        Some(d) => format!(" {} ", d.file_name().and_then(|n| n.to_str()).unwrap_or("album")),
        None => " Album ".to_string(),
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(Style::default().fg(if active { Color::Yellow } else { Color::DarkGray }));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if app.album_paths.is_empty() {
        f.render_widget(
            Paragraph::new("\n  Select an album in the tree (→ / Enter).")
                .style(Style::default().fg(Color::DarkGray)),
            inner,
        );
        return;
    }

    let cols = app.cols.max(1);
    let rows = inner.height as usize / 8; // ~8 text rows per thumbnail cell
    let per_page = (cols * rows).max(1);
    let page = app.album_cursor / per_page;
    let start = page * per_page;

    let row_rects = Layout::vertical(vec![Constraint::Ratio(1, rows.max(1) as u32); rows.max(1)]).split(inner);
    for r in 0..rows.max(1) {
        let col_rects =
            Layout::horizontal(vec![Constraint::Ratio(1, cols as u32); cols]).split(row_rects[r]);
        for c in 0..cols {
            let vi = start + r * cols + c;
            let Some(path) = app.album_paths.get(vi).cloned() else { continue };
            let cell = col_rects[c];
            let selected = vi == app.album_cursor;
            let name: String = path.file_name().and_then(|n| n.to_str()).unwrap_or("").chars().collect();
            let cap: String = {
                let cs: Vec<char> = name.chars().collect();
                if cs.len() > 14 { cs[cs.len() - 14..].iter().collect() } else { name }
            };
            let cb = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::new().fg(if selected { Color::Cyan } else { Color::DarkGray }))
                .title(if selected { format!("▶{cap}") } else { cap });
            let ci = cb.inner(cell);
            f.render_widget(cb, cell);
            match app.thumbs.get_mut(&path) {
                Some(proto) => f.render_stateful_widget(StatefulImage::new(), ci, proto),
                None => f.render_widget(Paragraph::new("…").style(Style::new().fg(Color::DarkGray)), ci),
            }
        }
    }
}
