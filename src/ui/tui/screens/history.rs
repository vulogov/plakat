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
    selected: usize,
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
}

impl HistoryState {
    pub fn new(out_dir: PathBuf) -> Self {
        let mut s = Self {
            out_dir,
            entries: Vec::new(),
            selected: 0,
            scroll: 0,
            detail_for: None,
            recipe: None,
            dims: None,
            file_size: 0,
            preview: None,
            preview_for: None,
            status: String::new(),
        };
        s.rescan();
        s
    }

    /// Walk `out/` recursively for PNGs, newest first.
    pub fn rescan(&mut self) {
        let mut found = Vec::new();
        collect_pngs(&self.out_dir, &mut found, 0);
        found.sort_by(|a: &HistoryEntry, b: &HistoryEntry| b.mtime.cmp(&a.mtime));
        self.entries = found;
        self.selected = self.selected.min(self.entries.len().saturating_sub(1));
        self.detail_for = None; // force a re-sync of the detail pane
        self.status = format!("{} image(s)", self.entries.len());
    }

    pub fn selected_path(&self) -> Option<PathBuf> {
        self.entries.get(self.selected).map(|e| e.path.clone())
    }

    fn next(&mut self) {
        if !self.entries.is_empty() {
            self.selected = (self.selected + 1).min(self.entries.len() - 1);
        }
    }

    fn prev(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> HistoryAction {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.next(),
            KeyCode::Up | KeyCode::Char('k') => self.prev(),
            KeyCode::Char('g') => self.selected = 0,
            KeyCode::Char('G') => self.selected = self.entries.len().saturating_sub(1),
            KeyCode::Char('r') => self.rescan(),
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

    fn render_list(&mut self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(format!(" History ({}) ", self.entries.len()));
        let inner = block.inner(area);
        f.render_widget(block, area);

        if self.entries.is_empty() {
            f.render_widget(
                Paragraph::new("No images under out/ yet. Generate something in Chat (Ctrl-1).")
                    .style(Style::new().fg(Color::DarkGray))
                    .wrap(Wrap { trim: true }),
                inner,
            );
            return;
        }

        // Build display rows: a date header before each new day, then its entries.
        // Track which display row the selected entry sits on for scroll + highlight.
        let mut rows: Vec<Line> = Vec::new();
        let mut sel_row = 0usize;
        let mut last_date: Option<&str> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if last_date != Some(e.date_label.as_str()) {
                rows.push(Line::from(Span::styled(
                    format!("─ {} ", e.date_label),
                    Style::new().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
                )));
                last_date = Some(e.date_label.as_str());
            }
            if i == self.selected {
                sel_row = rows.len();
            }
            let name = e.path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
            let style = if i == self.selected {
                Style::new().bg(Color::Cyan).fg(Color::Black).add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(Color::White)
            };
            rows.push(Line::from(vec![
                Span::styled(format!(" {} ", e.time_label), Style::new().fg(Color::DarkGray)),
                Span::styled(name.to_string(), style),
            ]));
        }

        // Scroll so the selected row stays visible.
        let h = inner.height as usize;
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
        let block = Block::default().borders(Borders::ALL).title(" Recipe  ·  [C] continue in Chat ");
        let inner = block.inner(area);
        f.render_widget(block, area);

        let mut lines: Vec<Line> = Vec::new();
        if let Some(e) = self.entries.get(self.selected) {
            let mut meta = format!("{}  {}", e.date_label, e.time_label);
            if let Some((w, h)) = self.dims {
                meta.push_str(&format!("  ·  {w}×{h}"));
            }
            if self.file_size > 0 {
                meta.push_str(&format!("  ·  {} KB", self.file_size / 1024));
            }
            lines.push(Line::from(Span::styled(meta, Style::new().fg(Color::Gray))));
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
            let mtime = e.metadata().and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
            let stamp = humantime::format_rfc3339_seconds(mtime).to_string();
            let date_label: String = stamp.chars().take(10).collect();
            let time_label: String = stamp.chars().skip(11).take(5).collect();
            out.push(HistoryEntry { path: p, date_label, time_label, mtime });
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
}
