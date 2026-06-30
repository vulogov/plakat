//! History screen (RFC TUI-1 §9) — browse every image under the workspace `out/`
//! dir, grouped by date, with a lazy preview + embedded recipe, and `C` to continue
//! from an image in Chat. The full thumbnail GRID, semantic search, and side-by-side
//! compare are follow-ups; this increment is the date-grouped list + preview + recipe
//! + continue-in-Chat. Preview decode is lazy (only the selected image, and only
//! while this screen is active) so the event loop stays responsive.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

/// One image on disk.
struct HistoryEntry {
    path: PathBuf,
    date_label: String, // YYYY-MM-DD
    time_label: String, // HH:MM
    mtime: SystemTime,
    /// Tags from the `<image>.tags` sidecar (one per line), for collection building.
    tags: Vec<String>,
    /// The embedded recipe text, lazily read when first needed for search.
    recipe_cache: Option<String>,
    recipe_loaded: bool,
}

impl HistoryEntry {
    fn file_name(&self) -> &str {
        self.path.file_name().and_then(|n| n.to_str()).unwrap_or("?")
    }
    /// The `<image>.tags` sidecar path.
    fn tags_path(&self) -> PathBuf {
        self.path.with_extension("png.tags")
    }

    /// The text the semantic ranker embeds: filename + tags + recipe (loaded by then).
    fn searchable_text(&self) -> String {
        let mut s = self.file_name().replace(['-', '_', '.'], " ");
        if !self.tags.is_empty() {
            s.push(' ');
            s.push_str(&self.tags.join(" "));
        }
        if let Some(r) = &self.recipe_cache {
            s.push(' ');
            s.push_str(r);
        }
        s
    }
}

/// What the App should do after a key.
pub enum HistoryAction {
    None,
    /// Continue from this image in Chat. `prompt`/`seed` come from the embedded
    /// recipe when present — with both, Chat continues in prompt-evolve mode (so
    /// additive edits work); without, it falls back to image-anchored img2img.
    Continue { path: PathBuf, prompt: String, seed: Option<u64> },
}

pub struct HistoryState {
    out_dir: PathBuf,
    entries: Vec<HistoryEntry>,
    /// Indices into `entries` that pass the current filter, in display order. All
    /// navigation / selection is relative to this view; `selected` indexes into it.
    view: Vec<usize>,
    selected: usize,
    /// Current filter query (matches filename, tags, and recipe text); empty = all.
    query: String,
    /// Typing the filter query (`/`) — the screen captures input.
    filtering: bool,
    /// Semantic ranking (TF-IDF cosine, `Tab` in the filter) vs plain substring filter.
    semantic: bool,
    /// Typing a tag (`T`) for the selected image — the screen captures input.
    tag_input: Option<String>,
    /// Baseline image for side-by-side recipe compare (`d` marks, `d` again diffs).
    compare_base: Option<PathBuf>,
    /// First visible display row (headers + entries), for scrolling.
    scroll: usize,
    // Lazily-synced detail for the selected entry.
    detail_for: Option<PathBuf>,
    recipe: Option<String>,
    dims: Option<(u32, u32)>,
    file_size: u64,
    // Preview built by the App (it owns the image Picker), like Chat.
    pub preview: Option<ratatui_image::protocol::StatefulProtocol>,
    pub preview_for: Option<PathBuf>,
    status: String,
    /// `true` → thumbnail GRID view; `false` → list + single preview (default).
    grid: bool,
    /// Thumbnail protocols by path (built by the App), with an LRU eviction order.
    thumbs: std::collections::HashMap<PathBuf, ratatui_image::protocol::StatefulProtocol>,
    thumb_lru: Vec<PathBuf>,
    /// Rows of grid cells the last render fit — so the App's sync knows the page size.
    grid_rows_cache: usize,
}

/// Columns in the thumbnail grid, and the cap on cached thumbnail protocols.
const GRID_COLS: usize = 4;
const THUMB_CACHE_CAP: usize = 40;

impl HistoryState {
    pub fn new(out_dir: PathBuf) -> Self {
        let mut s = Self {
            out_dir,
            entries: Vec::new(),
            view: Vec::new(),
            selected: 0,
            query: String::new(),
            filtering: false,
            semantic: false,
            tag_input: None,
            compare_base: None,
            scroll: 0,
            detail_for: None,
            recipe: None,
            dims: None,
            file_size: 0,
            preview: None,
            preview_for: None,
            status: String::new(),
            grid: false,
            thumbs: std::collections::HashMap::new(),
            thumb_lru: Vec::new(),
            grid_rows_cache: 4,
        };
        s.rescan();
        s
    }

    /// Walk `out/` recursively for PNGs, newest first.
    pub fn rescan(&mut self) {
        let mut found = Vec::new();
        collect_pngs(&self.out_dir, &mut found, 0);
        found.sort_by(|a: &HistoryEntry, b: &HistoryEntry| b.mtime.cmp(&a.mtime));
        for e in &mut found {
            e.tags = load_tags(&e.tags_path());
        }
        self.entries = found;
        self.detail_for = None; // force a re-sync of the detail pane
        self.rebuild_view();
    }

    /// Recompute the visible `view` from the current query. An empty query shows
    /// everything; otherwise an entry matches if the (lowercased) query is a substring
    /// of its filename, any tag, or its recipe text (recipes are read lazily + cached).
    fn rebuild_view(&mut self) {
        let q = self.query.trim().to_lowercase();
        if q.is_empty() {
            self.view = (0..self.entries.len()).collect();
        } else {
            // Ensure recipes are available for the search (one-time lazy read per entry).
            for e in &mut self.entries {
                if !e.recipe_loaded {
                    e.recipe_cache = crate::imaging::io::read_parameters_chunk(&e.path).ok().flatten();
                    e.recipe_loaded = true;
                }
            }
            if self.semantic {
                // Semantic ranking: each entry's searchable text → a TF-IDF vector,
                // ranked by cosine to the query (most-relevant first), not just filtered.
                let docs: Vec<String> = self.entries.iter().map(|e| e.searchable_text()).collect();
                self.view = crate::ui::tui::services::semantic::rank(&self.query, &docs)
                    .into_iter()
                    .map(|(i, _)| i)
                    .collect();
            } else {
                self.view = (0..self.entries.len())
                    .filter(|&i| {
                        let e = &self.entries[i];
                        e.file_name().to_lowercase().contains(&q)
                            || e.tags.iter().any(|t| t.to_lowercase().contains(&q))
                            || e.recipe_cache.as_deref().map(|r| r.to_lowercase().contains(&q)).unwrap_or(false)
                    })
                    .collect();
            }
        }
        if self.selected >= self.view.len() {
            self.selected = self.view.len().saturating_sub(1);
        }
        let mode = if self.semantic { "semantic" } else { "match" };
        self.status = if q.is_empty() {
            format!("{} image(s)", self.entries.len())
        } else {
            format!("{} / {} {mode} “{}”", self.view.len(), self.entries.len(), self.query)
        };
    }

    /// The entry currently under the cursor (via the filtered view).
    fn cur(&self) -> Option<&HistoryEntry> {
        self.view.get(self.selected).and_then(|&i| self.entries.get(i))
    }

    pub fn selected_path(&self) -> Option<PathBuf> {
        self.cur().map(|e| e.path.clone())
    }

    fn next(&mut self) {
        if !self.view.is_empty() {
            self.selected = (self.selected + 1).min(self.view.len() - 1);
        }
    }

    fn prev(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    /// Move the selection by `delta` (grid row navigation), clamped to the view.
    fn move_by(&mut self, delta: isize) {
        if self.view.is_empty() {
            return;
        }
        let max = self.view.len() as isize - 1;
        self.selected = (self.selected as isize + delta).clamp(0, max) as usize;
    }

    /// Whether the thumbnail grid view is active (the App switches its sync + the
    /// detail pane accordingly).
    pub fn is_grid(&self) -> bool {
        self.grid
    }

    /// The paths of the thumbnails the current grid PAGE needs (using the last-rendered
    /// row count) so the App only decodes what's visible. Empty in list view.
    pub fn visible_thumb_paths(&self) -> Vec<PathBuf> {
        let rows = self.grid_rows_cache;
        if !self.grid || rows == 0 {
            return Vec::new();
        }
        let per_page = rows * GRID_COLS;
        let page = self.selected / per_page;
        let start = page * per_page;
        self.view
            .iter()
            .skip(start)
            .take(per_page)
            .map(|&i| self.entries[i].path.clone())
            .collect()
    }

    /// True if a thumbnail protocol is already cached for `path`.
    pub fn has_thumb(&self, path: &Path) -> bool {
        self.thumbs.contains_key(path)
    }

    /// The App delivers a built thumbnail protocol; cache it (LRU-capped).
    pub fn set_thumb(&mut self, path: PathBuf, protocol: ratatui_image::protocol::StatefulProtocol) {
        self.thumb_lru.retain(|p| p != &path);
        self.thumb_lru.push(path.clone());
        self.thumbs.insert(path, protocol);
        while self.thumb_lru.len() > THUMB_CACHE_CAP {
            let evict = self.thumb_lru.remove(0);
            self.thumbs.remove(&evict);
        }
    }

    /// True while a text-input modal (filter query / tag entry) is open.
    pub fn captures_input(&self) -> bool {
        self.filtering || self.tag_input.is_some()
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> HistoryAction {
        // ── Filter-query input (`/`). ──
        if self.filtering {
            match key.code {
                KeyCode::Esc => {
                    self.query.clear();
                    self.filtering = false;
                    self.rebuild_view();
                }
                KeyCode::Enter => self.filtering = false,
                KeyCode::Backspace => {
                    self.query.pop();
                    self.rebuild_view();
                }
                KeyCode::Char(c) => {
                    self.query.push(c);
                    self.rebuild_view();
                }
                _ => {}
            }
            return HistoryAction::None;
        }
        // ── Tag input (`T`). ──
        if self.tag_input.is_some() {
            match key.code {
                KeyCode::Esc => self.tag_input = None,
                KeyCode::Backspace => {
                    if let Some(b) = self.tag_input.as_mut() {
                        b.pop();
                    }
                }
                KeyCode::Char(c) => {
                    if let Some(b) = self.tag_input.as_mut() {
                        b.push(c);
                    }
                }
                KeyCode::Enter => {
                    let tag = self.tag_input.take().unwrap_or_default();
                    self.add_tag(tag.trim());
                }
                _ => {}
            }
            return HistoryAction::None;
        }
        match key.code {
            // `v` — toggle between the list+preview view and the thumbnail grid.
            KeyCode::Char('v' | 'V') => self.grid = !self.grid,
            // In the grid, ←/→ move within a row and ↑/↓ move by a full row.
            KeyCode::Left | KeyCode::Char('h') if self.grid => self.prev(),
            KeyCode::Right | KeyCode::Char('l') if self.grid => self.next(),
            KeyCode::Down | KeyCode::Char('j') if self.grid => self.move_by(GRID_COLS as isize),
            KeyCode::Up | KeyCode::Char('k') if self.grid => self.move_by(-(GRID_COLS as isize)),
            KeyCode::Down | KeyCode::Char('j') => self.next(),
            KeyCode::Up | KeyCode::Char('k') => self.prev(),
            KeyCode::Char('g') => self.selected = 0,
            KeyCode::Char('G') => self.selected = self.view.len().saturating_sub(1),
            KeyCode::Char('r') => self.rescan(),
            // `/` — substring filter across filename / tags / recipe text.
            KeyCode::Char('/') => {
                self.filtering = true;
                self.semantic = false;
                self.status = "filter: type to match filename / tags / recipe · Enter keep · Esc clear".into();
            }
            // `?` — semantic search: rank by relevance (TF-IDF cosine), most-related first.
            KeyCode::Char('?') => {
                self.filtering = true;
                self.semantic = true;
                self.status = "semantic search: type a query · ranks by relevance · Enter keep · Esc clear".into();
            }
            // `T` — tag the selected image (collection building).
            KeyCode::Char('t' | 'T') => {
                if self.selected_path().is_some() {
                    self.tag_input = Some(String::new());
                    self.status = "tag: type a label · Enter add · Esc cancel".into();
                }
            }
            // `X` — export the current (filtered) set into out/export/.
            KeyCode::Char('x' | 'X') => self.export_view(),
            // `d` — mark the selected as compare baseline, or diff against an existing one.
            KeyCode::Char('d' | 'D') => self.toggle_compare(),
            // Continue from this image in Chat.
            KeyCode::Char('c' | 'C') | KeyCode::Enter => {
                if let Some(path) = self.selected_path() {
                    let prompt = self.recipe.as_deref().map(positive_prompt).unwrap_or_default();
                    let seed = self.recipe.as_deref().and_then(seed_from_params);
                    return HistoryAction::Continue { path, prompt, seed };
                }
            }
            _ => {}
        }
        HistoryAction::None
    }

    /// Append `tag` to the selected image's `.tags` sidecar (dedup, persisted).
    fn add_tag(&mut self, tag: &str) {
        if tag.is_empty() {
            return;
        }
        let Some(&i) = self.view.get(self.selected) else { return };
        let e = &mut self.entries[i];
        if e.tags.iter().any(|t| t.eq_ignore_ascii_case(tag)) {
            self.status = format!("already tagged ‘{tag}’");
            return;
        }
        e.tags.push(tag.to_string());
        let body = e.tags.join("\n") + "\n";
        match std::fs::write(e.tags_path(), body) {
            Ok(()) => self.status = format!("✓ tagged ‘{tag}’ ({} tag(s))", e.tags.len()),
            Err(err) => self.status = format!("✗ tag write failed: {err}"),
        }
    }

    /// Copy every image in the current view into `out/export/` (collection building).
    /// Filenames are kept; collisions get a `-N` suffix.
    fn export_view(&mut self) {
        if self.view.is_empty() {
            self.status = "nothing to export".into();
            return;
        }
        let dest = self.out_dir.join("export");
        if let Err(e) = std::fs::create_dir_all(&dest) {
            self.status = format!("✗ export dir failed: {e}");
            return;
        }
        let mut n = 0usize;
        for &i in &self.view {
            let src = &self.entries[i].path;
            if src.starts_with(&dest) {
                continue; // don't re-export the export dir itself
            }
            let mut target = dest.join(self.entries[i].file_name());
            let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("image").to_string();
            let mut k = 1;
            while target.exists() {
                target = dest.join(format!("{stem}-{k}.png"));
                k += 1;
            }
            if std::fs::copy(src, &target).is_ok() {
                n += 1;
            }
        }
        let scope = if self.query.trim().is_empty() { "all".to_string() } else { format!("“{}”", self.query) };
        self.status = format!("✓ exported {n} image(s) ({scope}) → out/export/");
    }

    /// Mark the selected image as the compare baseline; pressing `d` on a *different*
    /// image then shows their recipe diff in the detail pane (Esc/`d`-again clears).
    fn toggle_compare(&mut self) {
        let Some(path) = self.selected_path() else { return };
        match &self.compare_base {
            Some(b) if *b == path => {
                self.compare_base = None;
                self.status = "compare baseline cleared".into();
            }
            _ => {
                self.compare_base = Some(path);
                self.status = "compare baseline set — move to another image and press d to diff".into();
            }
        }
    }

    /// Read the selected image's recipe (A1111 `parameters` tEXt chunk) + header
    /// dims + file size, but only when the selection changed. Cheap (no full pixel
    /// decode); called by the App each tick while History is the active screen.
    pub fn sync_detail(&mut self) {
        let Some(path) = self.selected_path() else { return };
        if self.detail_for.as_ref() == Some(&path) {
            return;
        }
        self.detail_for = Some(path.clone());
        self.recipe = crate::imaging::io::read_parameters_chunk(&path).ok().flatten();
        self.dims = png_dims(&path);
        self.file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    }

    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        if self.grid {
            self.render_grid(f, area);
            return;
        }
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(area);
        self.render_list(f, cols[0]);

        // Right column: preview on top, recipe detail below.
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
            .split(cols[1]);
        self.render_preview(f, right[0]);
        self.render_detail(f, right[1]);
    }

    /// Rows of cells that fit the grid area, given a fixed cell height.
    fn grid_rows(inner_h: u16) -> usize {
        const CELL_H: u16 = 7; // ~6 image rows + a caption line
        (inner_h / CELL_H).max(1) as usize
    }

    fn render_grid(&mut self, f: &mut Frame, area: Rect) {
        let title = if self.query.trim().is_empty() {
            format!(" History grid ({}) · v list · ↑↓←→ move · Enter continue ", self.view.len())
        } else {
            format!(" History grid ({}/{}) · /{} · v list ", self.view.len(), self.entries.len(), self.query)
        };
        let block = Block::default().borders(Borders::ALL).title(title);
        let inner = block.inner(area);
        f.render_widget(block, area);
        if self.view.is_empty() {
            f.render_widget(
                Paragraph::new("No images. Esc to clear any filter.").style(Style::new().fg(Color::DarkGray)),
                inner,
            );
            return;
        }

        let rows = Self::grid_rows(inner.height);
        self.grid_rows_cache = rows;
        let per_page = rows * GRID_COLS;
        let page = self.selected / per_page;
        let start = page * per_page;

        // Split the area into `rows` × GRID_COLS cells.
        let row_rects = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Ratio(1, rows as u32); rows])
            .split(inner);
        for r in 0..rows {
            let col_rects = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(vec![Constraint::Ratio(1, GRID_COLS as u32); GRID_COLS])
                .split(row_rects[r]);
            for c in 0..GRID_COLS {
                let vi = start + r * GRID_COLS + c;
                let Some(&ei) = self.view.get(vi) else { continue };
                let cell = col_rects[c];
                let selected = vi == self.selected;
                let border = if selected { Color::Cyan } else { Color::DarkGray };
                let name = self.entries[ei].file_name().to_string();
                let cap: String = name.chars().rev().take(14).collect::<String>().chars().rev().collect();
                let cb = Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::new().fg(border))
                    .title(if selected { format!("▶{cap}") } else { cap });
                let ci = cb.inner(cell);
                f.render_widget(cb, cell);
                // The thumbnail (if the App has built it) fills the cell; else a hint.
                let path = self.entries[ei].path.clone();
                match self.thumbs.get_mut(&path) {
                    Some(proto) => f.render_stateful_widget(ratatui_image::StatefulImage::new(), ci, proto),
                    None => f.render_widget(Paragraph::new("…").style(Style::new().fg(Color::DarkGray)), ci),
                }
            }
        }
    }

    fn render_list(&mut self, f: &mut Frame, area: Rect) {
        let sigil = if self.semantic { "?" } else { "/" };
        let title = if self.filtering {
            format!(" History · {sigil}{}▏ ", self.query)
        } else if !self.query.trim().is_empty() {
            format!(" History ({}/{}) · {sigil}{} ", self.view.len(), self.entries.len(), self.query)
        } else {
            format!(" History ({}) · / filter · ? semantic ", self.entries.len())
        };
        let block = Block::default().borders(Borders::ALL).title(title);
        let outer = block.inner(area);
        f.render_widget(block, area);

        // One-line footer for status (filter help / tag prompt / export feedback).
        let footer_h = if self.tag_input.is_some() || !self.status.is_empty() { 1 } else { 0 };
        let parts = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(footer_h)])
            .split(outer);
        let inner = parts[0];
        if footer_h == 1 {
            let footer = if let Some(b) = &self.tag_input {
                Line::from(vec![
                    Span::styled("tag> ", Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    Span::styled(format!("{b}▏"), Style::new().fg(Color::White)),
                ])
            } else {
                let c = if self.status.starts_with('✗') { Color::Red } else { Color::DarkGray };
                Line::from(Span::styled(self.status.clone(), Style::new().fg(c)))
            };
            f.render_widget(Paragraph::new(footer), parts[1]);
        }

        if self.view.is_empty() {
            let msg = if self.entries.is_empty() {
                "No images under out/ yet. Generate something in Chat (Ctrl-1)."
            } else {
                "No images match the filter. Esc to clear."
            };
            f.render_widget(
                Paragraph::new(msg).style(Style::new().fg(Color::DarkGray)).wrap(Wrap { trim: true }),
                inner,
            );
            return;
        }

        // Build display rows: a date header before each new day, then its entries.
        // Track which display row the selected entry sits on for scroll + highlight.
        let mut rows: Vec<Line> = Vec::new();
        let mut sel_row = 0usize;
        let mut last_date: Option<String> = None;
        for (vi, &ei) in self.view.iter().enumerate() {
            let e = &self.entries[ei];
            if last_date.as_deref() != Some(e.date_label.as_str()) {
                rows.push(Line::from(Span::styled(
                    format!("─ {} ", e.date_label),
                    Style::new().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
                )));
                last_date = Some(e.date_label.clone());
            }
            if vi == self.selected {
                sel_row = rows.len();
            }
            let style = if vi == self.selected {
                Style::new().bg(Color::Cyan).fg(Color::Black).add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(Color::White)
            };
            let base_mark = if self.compare_base.as_ref() == Some(&e.path) {
                Span::styled("◆ ", Style::new().fg(Color::Magenta))
            } else {
                Span::raw("  ")
            };
            let mut spans = vec![
                base_mark,
                Span::styled(format!("{} ", e.time_label), Style::new().fg(Color::DarkGray)),
                Span::styled(e.file_name().to_string(), style),
            ];
            if !e.tags.is_empty() {
                spans.push(Span::styled(format!("  #{}", e.tags.join(" #")), Style::new().fg(Color::Yellow)));
            }
            rows.push(Line::from(spans));
        }

        // Scroll so the selected row stays visible.
        let h = inner.height.max(1) as usize;
        if sel_row < self.scroll {
            self.scroll = sel_row;
        } else if sel_row >= self.scroll + h {
            self.scroll = sel_row + 1 - h;
        }
        let end = (self.scroll + h).min(rows.len());
        let visible: Vec<Line> = rows[self.scroll..end].to_vec();
        f.render_widget(Paragraph::new(visible), inner);
    }

    fn render_preview(&mut self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" Preview ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        match &mut self.preview {
            Some(p) => f.render_stateful_widget(ratatui_image::StatefulImage::new(), inner, p),
            None => f.render_widget(
                Paragraph::new("\n  Select an image — j/k to move.")
                    .style(Style::new().fg(Color::DarkGray)),
                inner,
            ),
        }
    }

    fn render_detail(&self, f: &mut Frame, area: Rect) {
        // When a compare baseline is set and we're on a *different* image, the pane
        // becomes a recipe diff instead of the plain recipe.
        let cur_path = self.selected_path();
        let comparing = matches!((&self.compare_base, &cur_path), (Some(b), Some(c)) if b != c);
        let title = if comparing {
            " Compare  ·  ◆ baseline vs cursor  ·  [d] clear "
        } else {
            " Recipe  ·  [C] continue · [d] compare · [T] tag · [X] export · [/] filter "
        };
        let block = Block::default().borders(Borders::ALL).title(title);
        let inner = block.inner(area);
        f.render_widget(block, area);

        let mut lines: Vec<Line> = Vec::new();
        if comparing {
            let base = self.compare_base.as_ref().and_then(|p| crate::imaging::io::read_parameters_chunk(p).ok().flatten()).unwrap_or_default();
            let cur = self.recipe.clone().unwrap_or_default();
            lines.push(Line::from(vec![
                Span::styled("◆ ", Style::new().fg(Color::Magenta)),
                Span::styled(
                    self.compare_base.as_ref().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or("?").to_string(),
                    Style::new().fg(Color::Magenta),
                ),
                Span::styled("  vs  ", Style::new().fg(Color::DarkGray)),
                Span::styled(cur_path.as_ref().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or("?").to_string(), Style::new().fg(Color::Cyan)),
            ]));
            lines.push(Line::from(""));
            for (k, bv, cv) in diff_recipes(&base, &cur) {
                lines.push(Line::from(vec![
                    Span::styled(format!("{k:<10} "), Style::new().fg(Color::DarkGray)),
                    Span::styled(format!("◆ {bv}"), Style::new().fg(Color::Magenta)),
                ]));
                lines.push(Line::from(vec![
                    Span::raw("           "),
                    Span::styled(format!("→ {cv}"), Style::new().fg(Color::Cyan)),
                ]));
            }
            if lines.len() <= 2 {
                lines.push(Line::from(Span::styled("(recipes identical)", Style::new().fg(Color::DarkGray))));
            }
            f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
            return;
        }

        if let Some(e) = self.cur() {
            let mut meta = format!("{}  {}", e.date_label, e.time_label);
            if let Some((w, h)) = self.dims {
                meta.push_str(&format!("  ·  {w}×{h}"));
            }
            if self.file_size > 0 {
                meta.push_str(&format!("  ·  {} KB", self.file_size / 1024));
            }
            lines.push(Line::from(Span::styled(meta, Style::new().fg(Color::Gray))));
            if !e.tags.is_empty() {
                lines.push(Line::from(Span::styled(format!("#{}", e.tags.join(" #")), Style::new().fg(Color::Yellow))));
            }
            lines.push(Line::from(""));
            match &self.recipe {
                Some(r) => {
                    for raw in r.lines() {
                        lines.push(Line::from(Span::styled(raw.to_string(), Style::new().fg(Color::White))));
                    }
                }
                None => lines.push(Line::from(Span::styled(
                    "(no embedded recipe — image has no `parameters` chunk)",
                    Style::new().fg(Color::DarkGray),
                ))),
            }
        }
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
    }
}

/// Recursively collect `*.png` under `dir` (skips `.run.hjson`/json sidecars; depth
/// capped to avoid pathological trees).
fn collect_pngs(dir: &Path, out: &mut Vec<HistoryEntry>, depth: usize) {
    if depth > 6 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_pngs(&p, out, depth + 1);
        } else if p.extension().and_then(|x| x.to_str()) == Some("png") {
            // Skip our own export copies + tag sidecars in the listing.
            if p.file_name().and_then(|n| n.to_str()).map(|n| n.ends_with(".tags")).unwrap_or(false) {
                continue;
            }
            let mtime = e.metadata().and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
            let stamp = humantime::format_rfc3339_seconds(mtime).to_string();
            let date_label: String = stamp.chars().take(10).collect();
            let time_label: String = stamp.chars().skip(11).take(5).collect();
            out.push(HistoryEntry {
                path: p,
                date_label,
                time_label,
                mtime,
                tags: Vec::new(),
                recipe_cache: None,
                recipe_loaded: false,
            });
        }
    }
}

/// PNG width/height from the header only (no full pixel decode).
fn png_dims(path: &Path) -> Option<(u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let dec = png::Decoder::new(std::io::BufReader::new(file));
    let reader = dec.read_info().ok()?;
    let info = reader.info();
    Some((info.width, info.height))
}

/// Read a `<image>.tags` sidecar (one tag per line); missing file → no tags.
fn load_tags(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .map(|s| s.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect())
        .unwrap_or_default()
}

/// Diff two A1111 recipe blocks into `(key, baseline, current)` rows that differ. The
/// first line (the positive prompt) is keyed `prompt`; subsequent `Key: value, …`
/// fields are split out. A field present on only one side shows `—` on the other.
fn diff_recipes(base: &str, cur: &str) -> Vec<(String, String, String)> {
    let bf = recipe_fields(base);
    let cf = recipe_fields(cur);
    let mut keys: Vec<String> = bf.keys().chain(cf.keys()).cloned().collect();
    keys.sort();
    keys.dedup();
    // Keep "prompt" first for readability.
    keys.sort_by_key(|k| if k == "prompt" { 0 } else { 1 });
    let mut out = Vec::new();
    for k in keys {
        let b = bf.get(&k).cloned().unwrap_or_else(|| "—".into());
        let c = cf.get(&k).cloned().unwrap_or_else(|| "—".into());
        if b != c {
            out.push((k, b, c));
        }
    }
    out
}

/// Parse an A1111 recipe into `field → value`. Line 1 = the positive prompt (`prompt`);
/// the trailing `Key: value, Key: value` line(s) are split on commas.
fn recipe_fields(recipe: &str) -> std::collections::BTreeMap<String, String> {
    let mut m = std::collections::BTreeMap::new();
    let mut lines = recipe.lines();
    if let Some(first) = lines.next() {
        if !first.trim().is_empty() {
            m.insert("prompt".to_string(), first.trim().to_string());
        }
    }
    for line in lines {
        if let Some(rest) = line.strip_prefix("Negative prompt:") {
            m.insert("negative".to_string(), rest.trim().to_string());
            continue;
        }
        for field in line.split(',') {
            if let Some((k, v)) = field.split_once(':') {
                let k = k.trim();
                if !k.is_empty() {
                    m.insert(k.to_lowercase(), v.trim().to_string());
                }
            }
        }
    }
    m
}

/// Extract the `Seed: N` value from an A1111 `parameters` block.
fn seed_from_params(params: &str) -> Option<u64> {
    let idx = params.find("Seed:")?;
    let digits: String = params[idx + "Seed:".len()..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// Extract the positive prompt from an A1111 `parameters` block: everything before
/// the `Negative prompt:` / `Steps:` parameter lines.
fn positive_prompt(params: &str) -> String {
    let mut out = Vec::new();
    for line in params.lines() {
        let t = line.trim_start();
        if t.starts_with("Negative prompt:") || t.starts_with("Steps:") {
            break;
        }
        out.push(line);
    }
    out.join(" ").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_from_params_reads_the_seed() {
        assert_eq!(seed_from_params("a fox\nSteps: 28, Seed: 12345, Size: 512x512"), Some(12345));
        assert_eq!(seed_from_params("Seed: 7"), Some(7));
        assert_eq!(seed_from_params("no seed here"), None);
    }

    #[test]
    fn positive_prompt_stops_at_negative_and_params() {
        let p = "a red fox in a forest\nNegative prompt: blurry, lowres\nSteps: 28, Seed: 42";
        assert_eq!(positive_prompt(p), "a red fox in a forest");
        let p2 = "a wolf\nSteps: 20, Seed: 1";
        assert_eq!(positive_prompt(p2), "a wolf");
        // No params block → the whole thing is the prompt.
        assert_eq!(positive_prompt("just a prompt"), "just a prompt");
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("plakat-history-test-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn rescan_finds_pngs_recursively_newest_first() {
        let d = tmp("scan");
        std::fs::create_dir_all(d.join("chat")).unwrap();
        // A 1x1 PNG written by the real encoder so png_dims/recipe paths are valid.
        let one = d.join("chat").join("plakat-1-1.png");
        crate::imaging::io::save_rgb_u8(&[10, 20, 30], 1, 1, &one).unwrap();
        let two = d.join("plakat-2.png");
        crate::imaging::io::save_rgb_u8(&[40, 50, 60], 1, 1, &two).unwrap();

        let mut s = HistoryState::new(d.clone());
        assert_eq!(s.entries.len(), 2);
        // Newest first; png_dims reads 1x1.
        s.sync_detail();
        assert_eq!(s.dims, Some((1, 1)));
        // [C] yields a Continue action for the selected path.
        match s.handle_key(KeyEvent::new(KeyCode::Char('c'), crossterm::event::KeyModifiers::NONE)) {
            HistoryAction::Continue { path, .. } => assert!(path.extension().unwrap() == "png"),
            _ => panic!("expected Continue"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    fn ch(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), crossterm::event::KeyModifiers::NONE)
    }
    fn special(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, crossterm::event::KeyModifiers::NONE)
    }

    #[test]
    fn v_toggles_grid_and_grid_nav_moves_by_row() {
        let d = tmp("grid");
        for i in 0..10 {
            crate::imaging::io::save_rgb_u8(&[i, i, i], 1, 1, &d.join(format!("img{i:02}.png"))).unwrap();
        }
        let mut s = HistoryState::new(d.clone());
        assert_eq!(s.view.len(), 10);
        assert!(!s.is_grid());
        // `v` enters the grid.
        s.handle_key(ch('v'));
        assert!(s.is_grid());
        // ←/→ move by one; ↓ moves by a full row (GRID_COLS).
        s.handle_key(special(KeyCode::Right));
        assert_eq!(s.selected, 1);
        s.handle_key(special(KeyCode::Down));
        assert_eq!(s.selected, 1 + GRID_COLS);
        s.handle_key(special(KeyCode::Up));
        assert_eq!(s.selected, 1);
        // Clamped at the bottom (can't run past the view).
        for _ in 0..20 {
            s.handle_key(special(KeyCode::Down));
        }
        assert_eq!(s.selected, s.view.len() - 1);
        // `v` returns to the list.
        s.handle_key(ch('v'));
        assert!(!s.is_grid());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn visible_thumb_paths_pages_around_the_selection() {
        let d = tmp("thumbpage");
        for i in 0..30 {
            crate::imaging::io::save_rgb_u8(&[1, 2, 3], 1, 1, &d.join(format!("p{i:02}.png"))).unwrap();
        }
        let mut s = HistoryState::new(d.clone());
        // List view → no thumbnails wanted.
        assert!(s.visible_thumb_paths().is_empty());
        s.handle_key(ch('v')); // grid
        s.grid_rows_cache = 3; // 3 rows × 4 cols = 12 per page
        // Page 0 holds the first 12.
        assert_eq!(s.visible_thumb_paths().len(), 12);
        // Jump near the end → the page slides to cover the selection.
        s.handle_key(ch('G'));
        let page = s.visible_thumb_paths();
        assert!(!page.is_empty());
        assert!(page.iter().any(|p| p == &s.selected_path().unwrap()), "the selected cell is on the visible page");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn filter_narrows_the_view_by_filename() {
        let d = tmp("filter");
        crate::imaging::io::save_rgb_u8(&[1, 2, 3], 1, 1, &d.join("fox.png")).unwrap();
        crate::imaging::io::save_rgb_u8(&[4, 5, 6], 1, 1, &d.join("wolf.png")).unwrap();
        let mut s = HistoryState::new(d.clone());
        assert_eq!(s.view.len(), 2);
        // `/` then type "fox" → only the matching file remains in the view.
        s.handle_key(ch('/'));
        assert!(s.captures_input());
        for c in "fox".chars() {
            s.handle_key(ch(c));
        }
        assert_eq!(s.view.len(), 1);
        assert_eq!(s.selected_path().unwrap().file_name().unwrap(), "fox.png");
        // Enter keeps the filter (stops capturing); Esc would clear it.
        s.handle_key(special(KeyCode::Enter));
        assert!(!s.captures_input());
        assert_eq!(s.view.len(), 1);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn semantic_search_ranks_by_relevance() {
        let d = tmp("semantic");
        // Two real PNGs whose filenames carry the topic (recipes are absent here, so the
        // ranker embeds the filename text).
        crate::imaging::io::save_rgb_u8(&[1, 2, 3], 1, 1, &d.join("snowy-mountain-village.png")).unwrap();
        crate::imaging::io::save_rgb_u8(&[4, 5, 6], 1, 1, &d.join("neon-city-street-rain.png")).unwrap();
        let mut s = HistoryState::new(d.clone());
        // `?` enters semantic mode.
        s.handle_key(ch('?'));
        assert!(s.semantic && s.captures_input());
        for c in "winter mountain".chars() {
            s.handle_key(ch(c));
        }
        // The mountain image ranks (the city one shares no query terms → excluded).
        assert_eq!(s.view.len(), 1);
        assert_eq!(s.selected_path().unwrap().file_name().unwrap(), "snowy-mountain-village.png");
        // `/` switches back to plain substring filtering.
        s.handle_key(special(KeyCode::Esc));
        s.handle_key(ch('/'));
        assert!(!s.semantic);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn tag_writes_a_sidecar_and_filters_by_tag() {
        let d = tmp("tag");
        let img = d.join("a.png");
        crate::imaging::io::save_rgb_u8(&[1, 2, 3], 1, 1, &img).unwrap();
        crate::imaging::io::save_rgb_u8(&[4, 5, 6], 1, 1, &d.join("b.png")).unwrap();
        let _ = img; // (the specific file tagged is whichever the cursor lands on)
        let mut s = HistoryState::new(d.clone());
        // Tag the selected image "hero".
        let tagged = s.selected_path().unwrap();
        s.handle_key(ch('T'));
        assert!(s.captures_input());
        for c in "hero".chars() {
            s.handle_key(ch(c));
        }
        s.handle_key(special(KeyCode::Enter));
        assert!(tagged.with_extension("png.tags").exists(), "sidecar written");
        // Filtering by the tag finds exactly the tagged image.
        s.handle_key(ch('/'));
        for c in "hero".chars() {
            s.handle_key(ch(c));
        }
        assert_eq!(s.view.len(), 1);
        assert_eq!(s.selected_path().unwrap(), tagged);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn export_copies_the_filtered_set() {
        let d = tmp("export");
        crate::imaging::io::save_rgb_u8(&[1, 2, 3], 1, 1, &d.join("keep.png")).unwrap();
        crate::imaging::io::save_rgb_u8(&[4, 5, 6], 1, 1, &d.join("skip.png")).unwrap();
        let mut s = HistoryState::new(d.clone());
        // Filter to "keep", then export the view.
        s.handle_key(ch('/'));
        for c in "keep".chars() {
            s.handle_key(ch(c));
        }
        s.handle_key(special(KeyCode::Enter));
        s.handle_key(ch('X'));
        assert!(d.join("export/keep.png").exists());
        assert!(!d.join("export/skip.png").exists(), "only the filtered set is exported");
        assert!(s.status.starts_with("✓ exported 1"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn diff_recipes_reports_only_changed_fields() {
        let base = "a fox\nSteps: 28, Seed: 1, CFG scale: 7";
        let cur = "a fox\nSteps: 28, Seed: 2, CFG scale: 7";
        let d = diff_recipes(base, cur);
        // Only Seed changed.
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].0, "seed");
        assert_eq!(d[0].1, "1");
        assert_eq!(d[0].2, "2");
        // Identical recipes → no rows.
        assert!(diff_recipes(base, base).is_empty());
    }

    #[test]
    fn compare_baseline_toggles() {
        let d = tmp("compare");
        crate::imaging::io::save_rgb_u8(&[1, 2, 3], 1, 1, &d.join("a.png")).unwrap();
        let mut s = HistoryState::new(d.clone());
        assert!(s.compare_base.is_none());
        s.handle_key(ch('d')); // set baseline
        assert!(s.compare_base.is_some());
        s.handle_key(ch('d')); // same image → clears
        assert!(s.compare_base.is_none());
        let _ = std::fs::remove_dir_all(&d);
    }
}
