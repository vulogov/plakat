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

/// Album pane sub-mode (RFC §6): thumbnail grid, full-pane image view, or the culling loupe.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AlbumMode {
    Grid,
    Image,
    Cull,
}

/// A tree/curation action awaiting text or confirmation in the command pane (RFC §11).
enum PendingCmd {
    NewFolder { parent: PathBuf },
    NewAlbum { parent: PathBuf },
    Rename { path: PathBuf },
    Delete { path: PathBuf, is_album: bool },
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
    album_meta: hjson::AlbumMeta,
    album_paths: Vec<PathBuf>,
    /// Filtered view: indices into `album_paths` that pass `filter` (all when empty). `album_cursor`
    /// indexes into this, so navigation/rendering operate on the filtered set.
    view: Vec<usize>,
    filter: String,
    filter_active: bool,
    album_cursor: usize,
    /// Multi-selection stored as `album_paths` indices (survives filter changes).
    selected: HashSet<usize>,
    mode: AlbumMode,
    show_exif: bool,
    view_proto: Option<StatefulProtocol>,
    view_exif: Option<hjson::ExifRecord>,
    thumbs: HashMap<PathBuf, StatefulProtocol>,
    cols: usize,
    thumb_px: u32,

    // Command pane input (RFC §11).
    cmd_active: bool,
    cmd_prompt: String,
    cmd_buffer: String,
    pending: Option<PendingCmd>,

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
            album_meta: hjson::AlbumMeta::default(),
            album_paths: Vec::new(),
            view: Vec::new(),
            filter: String::new(),
            filter_active: false,
            album_cursor: 0,
            selected: HashSet::new(),
            mode: AlbumMode::Grid,
            show_exif: false,
            view_proto: None,
            view_exif: None,
            thumbs: HashMap::new(),
            cols: 4,
            thumb_px,
            cmd_active: false,
            cmd_prompt: String::new(),
            cmd_buffer: String::new(),
            pending: None,
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
                self.rebuild_view();
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
        self.album_meta = hjson::read_album(&dir).unwrap_or_default();
        self.album_dir = Some(dir);
        self.album_paths = paths;
        self.album_cursor = 0;
        self.selected.clear();
        self.filter.clear();
        self.rebuild_view();
        self.mode = AlbumMode::Grid;
        self.view_proto = None;
        self.thumbs.clear();
        self.focus = Focus::Album;
    }

    /// Filename (no path) of the album image at `album_paths` index `i`.
    fn image_name(&self, i: usize) -> Option<String> {
        self.album_paths.get(i).and_then(|p| p.file_name()).and_then(|n| n.to_str()).map(String::from)
    }

    /// The `album_paths` index at the cursor (through the filtered view).
    fn cur_idx(&self) -> Option<usize> {
        self.view.get(self.album_cursor).copied()
    }

    /// `album_paths` indices the next curation op applies to: the multi-selection if any, else the
    /// cursor image.
    fn targets(&self) -> Vec<usize> {
        if self.selected.is_empty() {
            self.cur_idx().into_iter().collect()
        } else {
            let mut v: Vec<usize> = self.selected.iter().copied().collect();
            v.sort_unstable();
            v
        }
    }

    /// Rebuild the filtered view from `filter` over `album_paths` + the curation records.
    fn rebuild_view(&mut self) {
        let f = self.filter.trim().to_string();
        self.view = (0..self.album_paths.len())
            .filter(|&i| {
                let name = self.album_paths[i].file_name().and_then(|n| n.to_str()).unwrap_or("");
                matches_filter(name, self.album_meta.images.get(name), &f)
            })
            .collect();
        if self.view.is_empty() {
            self.album_cursor = 0;
        } else {
            self.album_cursor = self.album_cursor.min(self.view.len() - 1);
        }
    }

    /// Mutate each target image's record, then persist `album.hjson`.
    fn edit_targets(&mut self, mut f: impl FnMut(&mut hjson::ImageRecord)) {
        for i in self.targets() {
            if let Some(name) = self.image_name(i) {
                f(self.album_meta.images.entry(name).or_default());
            }
        }
        self.save_album();
    }

    fn save_album(&mut self) {
        if let Some(dir) = &self.album_dir {
            if let Err(e) = hjson::write_album(dir, &self.album_meta) {
                self.status = format!("save failed: {e}");
            }
        }
    }

    /// Open the command pane for a pending action (RFC §11).
    fn prompt(&mut self, prompt: impl Into<String>, prefill: impl Into<String>, pending: PendingCmd) {
        self.cmd_prompt = prompt.into();
        self.cmd_buffer = prefill.into();
        self.pending = Some(pending);
        self.cmd_active = true;
    }

    /// Execute the pending command with `cmd_buffer` as the argument, then rescan.
    fn commit_cmd(&mut self) {
        let arg = self.cmd_buffer.trim().to_string();
        let result: Result<()> = (|| {
            match self.pending.take() {
                Some(PendingCmd::NewFolder { parent }) if !arg.is_empty() => {
                    let dir = parent.join(&arg);
                    std::fs::create_dir_all(&dir)?;
                    hjson::write_folder(&dir, &hjson::FolderMeta::default())?;
                }
                Some(PendingCmd::NewAlbum { parent }) if !arg.is_empty() => {
                    let dir = parent.join(&arg);
                    std::fs::create_dir_all(&dir)?;
                    hjson::write_album(&dir, &hjson::AlbumMeta::default())?;
                }
                Some(PendingCmd::Rename { path }) if !arg.is_empty() => {
                    if let Some(parent) = path.parent() {
                        std::fs::rename(&path, parent.join(&arg))?;
                    }
                }
                Some(PendingCmd::Delete { path, is_album }) if arg.eq_ignore_ascii_case("y") => {
                    if is_album {
                        std::fs::remove_dir_all(&path)?;
                    } else {
                        std::fs::remove_dir(&path)?; // folders: only if empty
                    }
                }
                _ => {}
            }
            Ok(())
        })();
        if let Err(e) = result {
            self.status = format!("error: {e}");
        }
        self.cmd_active = false;
        self.cmd_buffer.clear();
        self.rescan();
    }

    fn enter_image_view(&mut self) {
        self.mode = AlbumMode::Image;
        self.load_view();
    }

    /// Decode the cursor image (bounded to ~1600 px) into the full-pane view protocol + its EXIF.
    fn load_view(&mut self) {
        let path = self.cur_idx().and_then(|i| self.album_paths.get(i)).cloned();
        self.view_proto = path
            .as_ref()
            .and_then(|p| loader::thumbnail(p, 1600).ok())
            .map(|img| self.picker.new_resize_protocol(img));
        self.view_exif = path.and_then(|p| exif::read_exif(&p).ok());
    }

    /// Lazily decode a few thumbnails per tick for images near the cursor that aren't built yet.
    fn build_thumbs(&mut self, budget: usize) {
        let mut built = 0;
        // Prioritise the visible view from the cursor forward, then wrap.
        let n = self.view.len();
        for off in 0..n {
            if built >= budget {
                break;
            }
            let vi = (self.album_cursor + off) % n.max(1);
            let Some(path) = self.view.get(vi).and_then(|&pi| self.album_paths.get(pi)).cloned()
            else {
                break;
            };
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
                if handle_key(app, k) {
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
fn handle_key(app: &mut App, k: crossterm::event::KeyEvent) -> bool {
    use crossterm::event::KeyModifiers;
    if app.cmd_active {
        handle_cmd_key(app, k.code);
        return false;
    }
    if app.filter_active {
        handle_filter_key(app, k.code);
        return false;
    }
    let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
    match app.focus {
        Focus::Tree => handle_tree_key(app, k.code),
        Focus::Album if app.mode == AlbumMode::Image => handle_image_key(app, k.code),
        Focus::Album if app.mode == AlbumMode::Cull => handle_cull_key(app, k.code),
        Focus::Album => handle_grid_key(app, k.code, ctrl),
    }
}

/// Filter-bar input: type a filter expression, Enter applies, Esc clears.
fn handle_filter_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => {
            app.filter_active = false;
            app.filter.clear();
            app.rebuild_view();
        }
        KeyCode::Enter => {
            app.filter_active = false;
            app.rebuild_view();
        }
        KeyCode::Backspace => {
            app.filter.pop();
            app.rebuild_view();
        }
        KeyCode::Char(c) => {
            app.filter.push(c);
            app.rebuild_view();
        }
        _ => {}
    }
}

/// Command-pane text input (RFC §11): type the argument, Enter commits, Esc cancels.
fn handle_cmd_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => {
            app.cmd_active = false;
            app.pending = None;
            app.cmd_buffer.clear();
        }
        KeyCode::Enter => app.commit_cmd(),
        KeyCode::Backspace => {
            app.cmd_buffer.pop();
        }
        KeyCode::Char(c) => app.cmd_buffer.push(c),
        _ => {}
    }
}

fn handle_tree_key(app: &mut App, code: KeyCode) -> bool {
    let rows = app.rows();
    let cur = rows.get(app.tree_cursor).map(|r| (r.path.clone(), r.kind, r.has_children));
    match code {
        KeyCode::Char('q') => return true,
        KeyCode::Char('j') | KeyCode::Down => {
            app.tree_cursor = (app.tree_cursor + 1).min(rows.len().saturating_sub(1));
        }
        KeyCode::Char('k') | KeyCode::Up => app.tree_cursor = app.tree_cursor.saturating_sub(1),
        KeyCode::Char('g') => app.tree_cursor = 0,
        KeyCode::Char('G') => app.tree_cursor = rows.len().saturating_sub(1),
        KeyCode::Char('l') | KeyCode::Right | KeyCode::Enter => {
            if let Some((path, kind, has_children)) = cur {
                if kind == NodeKind::Album {
                    app.open_album(path);
                } else if has_children {
                    app.expanded.insert(path);
                }
            }
        }
        KeyCode::Char('h') | KeyCode::Left => {
            if let Some((path, ..)) = cur {
                app.expanded.remove(&path);
            }
        }
        KeyCode::Tab => {
            if app.album_dir.is_some() {
                app.focus = Focus::Album;
            }
        }
        // Mutations (RFC §7.4) → command pane.
        KeyCode::Char('n') => {
            if let Some((path, ..)) = cur {
                app.prompt("new folder: ", "", PendingCmd::NewFolder { parent: path });
            }
        }
        KeyCode::Char('a') => {
            if let Some((path, ..)) = cur {
                app.prompt("new album: ", "", PendingCmd::NewAlbum { parent: path });
            }
        }
        KeyCode::Char('R') => {
            if let Some((path, ..)) = cur {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
                app.prompt("rename to: ", name, PendingCmd::Rename { path });
            }
        }
        KeyCode::Char('D') => {
            if let Some((path, kind, _)) = cur {
                let n = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
                app.prompt(
                    format!("delete \"{n}\"? [y/N] "),
                    "",
                    PendingCmd::Delete { path, is_album: kind == NodeKind::Album },
                );
            }
        }
        _ => {}
    }
    false
}

fn handle_grid_key(app: &mut App, code: KeyCode, ctrl: bool) -> bool {
    let n = app.view.len();
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
        KeyCode::Char('k') | KeyCode::Up => app.album_cursor = app.album_cursor.saturating_sub(cols),
        KeyCode::Char('g') => app.album_cursor = 0,
        KeyCode::Char('G') => app.album_cursor = n.saturating_sub(1),
        KeyCode::Char('[') => app.cols = app.cols.saturating_sub(1).max(1),
        KeyCode::Char(']') => app.cols = (app.cols + 1).min(12),
        KeyCode::Enter => {
            if n > 0 {
                app.enter_image_view();
            }
        }
        // Filter bar (`/`) + culling loupe (`C`).
        KeyCode::Char('/') => app.filter_active = true,
        KeyCode::Char('C') => {
            if n > 0 {
                app.selected.clear(); // cull operates one image at a time
                app.mode = AlbumMode::Cull;
                app.load_view();
            }
        }
        // Selection (RFC §8.4) — indices are into `album_paths`, restricted to the visible view.
        KeyCode::Char(' ') => {
            if let Some(i) = app.cur_idx() {
                if !app.selected.insert(i) {
                    app.selected.remove(&i);
                }
            }
        }
        KeyCode::Char('a') if ctrl => app.selected = app.view.iter().copied().collect(),
        KeyCode::Char('d') if ctrl => app.selected.clear(),
        KeyCode::Char('i') if ctrl => {
            let visible: HashSet<usize> = app.view.iter().copied().collect();
            app.selected = visible.difference(&app.selected).copied().collect();
        }
        // Curation (RFC §8.5) — applies to selection or cursor; persisted to album.hjson.
        _ => return apply_curation(app, code),
    }
    false
}

/// Culling loupe (RFC §21.1): one image at a time, keep / reject / rate, advancing.
fn handle_cull_key(app: &mut App, code: KeyCode) -> bool {
    let advance = |app: &mut App| {
        if app.album_cursor + 1 < app.view.len() {
            app.album_cursor += 1;
            app.load_view();
        } else {
            app.mode = AlbumMode::Grid; // end of album → back to grid
        }
    };
    match code {
        KeyCode::Char('q') => return true,
        KeyCode::Esc => app.mode = AlbumMode::Grid,
        KeyCode::Right | KeyCode::Char(' ') => advance(app), // keep + advance
        KeyCode::Left => {
            if app.album_cursor > 0 {
                app.album_cursor -= 1;
                app.load_view();
            }
        }
        KeyCode::Char('x') => {
            app.edit_targets(|rec| rec.rejected = true);
            advance(app);
        }
        KeyCode::Char('f') => {
            app.edit_targets(|rec| rec.flagged = true);
            advance(app);
        }
        KeyCode::Char(d @ '1'..='5') => {
            let r = d.to_digit(10).unwrap() as u8;
            app.edit_targets(|rec| rec.rating = r);
            advance(app);
        }
        KeyCode::Char('i') => app.show_exif = !app.show_exif,
        _ => {}
    }
    false
}

fn handle_image_key(app: &mut App, code: KeyCode) -> bool {
    let n = app.view.len();
    match code {
        KeyCode::Char('q') => return true,
        KeyCode::Esc => {
            app.mode = AlbumMode::Grid;
            app.view_proto = None;
        }
        KeyCode::Char('l') | KeyCode::Right => {
            if app.album_cursor + 1 < n {
                app.album_cursor += 1;
                app.load_view();
            }
        }
        KeyCode::Char('h') | KeyCode::Left => {
            if app.album_cursor > 0 {
                app.album_cursor -= 1;
                app.load_view();
            }
        }
        KeyCode::Char('i') => app.show_exif = !app.show_exif,
        _ => return apply_curation(app, code),
    }
    false
}

/// Rating (1–5, 0 clears) / flag / reject / color-label, on the selection or cursor.
fn apply_curation(app: &mut App, code: KeyCode) -> bool {
    match code {
        KeyCode::Char(d @ '0'..='5') => {
            let r = d.to_digit(10).unwrap() as u8;
            app.edit_targets(|rec| rec.rating = r);
        }
        KeyCode::Char('f') => app.edit_targets(|rec| rec.flagged = !rec.flagged),
        KeyCode::Char('x') => app.edit_targets(|rec| rec.rejected = !rec.rejected),
        KeyCode::Char('c') => {
            const LABELS: [&str; 5] = ["red", "yellow", "green", "blue", "purple"];
            app.edit_targets(|rec| {
                let next = match rec.color_label.as_deref() {
                    None => Some(LABELS[0]),
                    Some(l) => LABELS.iter().position(|x| *x == l).and_then(|i| LABELS.get(i + 1)).copied(),
                };
                rec.color_label = next.map(String::from);
            });
        }
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

    // Command pane: active input (mutations/confirm) or a passive hint.
    let cmd = if app.cmd_active {
        format!(" {}{}_", app.cmd_prompt, app.cmd_buffer)
    } else {
        " CMD ▶ ".to_string()
    };
    let cmd_style = if app.cmd_active { Style::default().fg(Color::Yellow) } else { Style::default() };
    f.render_widget(
        Paragraph::new(cmd).style(cmd_style).block(Block::default().borders(Borders::ALL)),
        cmd_pane,
    );
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

    if app.mode == AlbumMode::Image || app.mode == AlbumMode::Cull {
        draw_image_view(f, app, inner);
        return;
    }

    // Optional filter bar at the top of the grid (shown while typing or when a filter is set).
    let grid_area = if app.filter_active || !app.filter.is_empty() {
        let [bar, grid] = Layout::vertical([Constraint::Length(1), Constraint::Fill(1)]).areas(inner);
        let cursor = if app.filter_active { "_" } else { "" };
        let txt = format!(" filter ▸ {}{}   ({} match)", app.filter, cursor, app.view.len());
        f.render_widget(Paragraph::new(txt).style(Style::new().fg(Color::Yellow)), bar);
        grid
    } else {
        inner
    };

    if app.view.is_empty() {
        f.render_widget(
            Paragraph::new("  no images match the filter").style(Style::new().fg(Color::DarkGray)),
            grid_area,
        );
        return;
    }

    let cols = app.cols.max(1);
    let rows = grid_area.height as usize / 8; // ~8 text rows per thumbnail cell
    let per_page = (cols * rows).max(1);
    let page = app.album_cursor / per_page;
    let start = page * per_page;

    let row_rects = Layout::vertical(vec![Constraint::Ratio(1, rows.max(1) as u32); rows.max(1)]).split(grid_area);
    for r in 0..rows.max(1) {
        let col_rects =
            Layout::horizontal(vec![Constraint::Ratio(1, cols as u32); cols]).split(row_rects[r]);
        for c in 0..cols {
            let vi = start + r * cols + c;
            let Some(&pi) = app.view.get(vi) else { continue };
            let Some(path) = app.album_paths.get(pi).cloned() else { continue };
            let cell = col_rects[c];
            let is_cursor = vi == app.album_cursor;
            let is_sel = app.selected.contains(&pi);
            let name: String = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
            let cap: String = {
                let cs: Vec<char> = name.chars().collect();
                if cs.len() > 14 { cs[cs.len() - 14..].iter().collect() } else { name.clone() }
            };
            let border = if is_cursor {
                Color::Cyan
            } else if is_sel {
                Color::Green
            } else {
                Color::DarkGray
            };
            let cb = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::new().fg(border))
                .title(if is_cursor { format!("▶{cap}") } else { cap });
            let ci = cb.inner(cell);
            f.render_widget(cb, cell);
            // Curation badge (computed before the mutable thumbs borrow).
            let badge = curation_badge(app.album_meta.images.get(&name));
            let [thumb_area, badge_area] =
                Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(ci);
            match app.thumbs.get_mut(&path) {
                Some(proto) => f.render_stateful_widget(StatefulImage::new(), thumb_area, proto),
                None => f.render_widget(
                    Paragraph::new("…").style(Style::new().fg(Color::DarkGray)),
                    thumb_area,
                ),
            }
            f.render_widget(Paragraph::new(badge), badge_area);
        }
    }
}

/// Filter grammar (RFC §16 subset): whitespace-separated predicates, ALL must match. Supports
/// `rating>=N` / `rating>N` / `rating=N` / `unrated`, `flag` / `-flag`, `rejected` / `-rejected`,
/// `ai` (scored), `tag:X` / `-tag:X`, and free text (filename contains).
fn matches_filter(name: &str, rec: Option<&hjson::ImageRecord>, filter: &str) -> bool {
    let f = filter.trim();
    if f.is_empty() {
        return true;
    }
    f.split_whitespace().all(|tok| tok_match(tok, name, rec))
}

fn tok_match(tok: &str, name: &str, rec: Option<&hjson::ImageRecord>) -> bool {
    let rating = rec.map(|r| r.rating).unwrap_or(0);
    let flagged = rec.map(|r| r.flagged).unwrap_or(false);
    let rejected = rec.map(|r| r.rejected).unwrap_or(false);
    let has_tag = |t: &str| rec.map(|r| r.tags.iter().any(|g| g.eq_ignore_ascii_case(t))).unwrap_or(false);
    match tok {
        "unrated" => rating == 0,
        "flag" | "flagged" => flagged,
        "-flag" => !flagged,
        "rejected" => rejected,
        "-rejected" => !rejected,
        "ai" | "scored" => rec.map(|r| r.score.is_some()).unwrap_or(false),
        _ if tok.starts_with("rating>=") => tok[8..].parse::<u8>().map(|n| rating >= n).unwrap_or(true),
        _ if tok.starts_with("rating>") => tok[7..].parse::<u8>().map(|n| rating > n).unwrap_or(true),
        _ if tok.starts_with("rating=") => tok[7..].parse::<u8>().map(|n| rating == n).unwrap_or(true),
        _ if tok.starts_with("tag:") => has_tag(&tok[4..]),
        _ if tok.starts_with("-tag:") => !has_tag(&tok[5..]),
        _ => name.to_lowercase().contains(&tok.to_lowercase()),
    }
}

fn label_color(l: &str) -> Color {
    match l {
        "red" => Color::Red,
        "yellow" => Color::Yellow,
        "green" => Color::Green,
        "blue" => Color::Blue,
        "purple" => Color::Magenta,
        _ => Color::White,
    }
}

/// Rating stars + flag / reject / color-label, from an image's curation record.
fn curation_badge(rec: Option<&hjson::ImageRecord>) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    if let Some(r) = rec {
        if r.rating > 0 {
            let stars: String = (0..5).map(|i| if i < r.rating { '★' } else { '☆' }).collect();
            spans.push(Span::styled(stars, Style::new().fg(Color::Yellow)));
        }
        if r.flagged {
            spans.push(Span::styled(" ⚑", Style::new().fg(Color::Yellow)));
        }
        if r.rejected {
            spans.push(Span::styled(" ✗", Style::new().fg(Color::Red)));
        }
        if let Some(l) = &r.color_label {
            spans.push(Span::styled(" ●", Style::new().fg(label_color(l))));
        }
    }
    Line::from(spans)
}

/// Full-pane image view (RFC §9): the image, optionally with an EXIF + curation side panel (`i`).
fn draw_image_view(f: &mut Frame, app: &mut App, area: Rect) {
    let (img_area, panel) = if app.show_exif {
        let [a, b] = Layout::horizontal([Constraint::Percentage(60), Constraint::Fill(1)]).areas(area);
        (a, Some(b))
    } else {
        (area, None)
    };

    match app.view_proto.as_mut() {
        Some(proto) => f.render_stateful_widget(StatefulImage::new(), img_area, proto),
        None => f.render_widget(Paragraph::new("  decoding…").style(Style::new().fg(Color::DarkGray)), img_area),
    }

    if let Some(panel) = panel {
        let name = app.cur_idx().and_then(|i| app.image_name(i)).unwrap_or_default();
        let mut lines: Vec<Line> = vec![Line::from(Span::styled(name.clone(), Style::new().add_modifier(Modifier::BOLD)))];
        if let Some(e) = &app.view_exif {
            let mut kv = |k: &str, v: Option<String>| {
                if let Some(v) = v {
                    lines.push(Line::from(format!("{k:<9}{v}")));
                }
            };
            kv("Date", e.date_taken.clone());
            kv("Camera", match (&e.camera_make, &e.camera_model) {
                (Some(m), Some(md)) => Some(format!("{m} {md}")),
                (_, Some(md)) => Some(md.clone()),
                _ => None,
            });
            kv("Lens", e.lens_model.clone());
            kv("Focal", e.focal_length_mm.map(|f| format!("{f:.0}mm")));
            kv("Aperture", e.aperture.clone());
            kv("Shutter", e.shutter.clone());
            kv("ISO", e.iso.map(|i| i.to_string()));
            kv("Size", match (e.width_px, e.height_px) {
                (Some(w), Some(h)) => Some(format!("{w}×{h}")),
                _ => None,
            });
            kv("GPS", match (e.gps_lat, e.gps_lon) {
                (Some(la), Some(lo)) => Some(format!("{la:.4}, {lo:.4}")),
                _ => None,
            });
        }
        if let Some(r) = app.album_meta.images.get(&name) {
            lines.push(Line::from("─ curation ─"));
            lines.push(curation_badge(Some(r)));
            if let Some(s) = r.score {
                lines.push(Line::from(format!("score    {s:.2}")));
            }
            if !r.tags.is_empty() {
                lines.push(Line::from(format!("tags     {}", r.tags.join(", "))));
            }
        }
        f.render_widget(
            Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" EXIF ")),
            panel,
        );
    }
}

#[cfg(test)]
mod filter_tests {
    use super::*;
    use crate::photos::hjson::ImageRecord;

    #[test]
    fn filter_predicates() {
        let mut rec = ImageRecord { rating: 4, flagged: true, ..Default::default() };
        rec.tags = vec!["waterfall".into()];
        let r = Some(&rec);
        assert!(matches_filter("a.jpg", r, ""));            // empty → all
        assert!(matches_filter("a.jpg", r, "rating>=4"));
        assert!(!matches_filter("a.jpg", r, "rating>=5"));
        assert!(matches_filter("a.jpg", r, "flag tag:waterfall"));  // AND
        assert!(!matches_filter("a.jpg", r, "-flag"));
        assert!(matches_filter("a.jpg", r, "-rejected"));
        assert!(matches_filter("IMG_1.jpg", r, "img"));     // free text (case-insensitive)
        assert!(!matches_filter("a.jpg", None, "rating>=1")); // no record → rating 0
        assert!(matches_filter("a.jpg", None, "unrated"));
    }
}
