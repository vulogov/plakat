//! Canvas screen (RFC TUI-1 §13, Release 4). A coarse cell-grid mask painter over
//! the current Chat base image: move the cursor (arrows / hjkl), Space toggles a
//! cell, Shift+arrow paints while moving, and preset keys fill common regions (sky /
//! background / foreground / halves / person column). `Enter` rasterizes the grid to
//! a full-res white-on-black mask PNG (white = inpaint) and hands it to Chat as a
//! pre-populated inpaint mask. Coarse REGIONAL masking only (documented); fine masks
//! need an external editor + `--mask-path`. Outpaint mode (`M`) is a follow-up.

use std::path::PathBuf;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use image::{GrayImage, Luma};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

/// Default (coarse) grid; `g` cycles through [`GRID_DENSITIES`].
const COLS: usize = 16;
const ROWS: usize = 12;

/// Selectable mask grid densities `(cols, rows)` — coarse → fine (`g` cycles).
const GRID_DENSITIES: [(usize, usize); 3] = [(16, 12), (24, 18), (32, 24)];

/// One outpaint band = 128px (RFC §13). Bounded so the padded canvas stays Metal-sane.
const OUTPAINT_UNIT: u32 = 128;
const OUTPAINT_MAX_BANDS: u32 = 4;

/// Which edge an outpaint extends from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Edge {
    Left,
    Right,
    Top,
    Bottom,
}

impl Edge {
    fn label(self) -> &'static str {
        match self {
            Edge::Left => "left",
            Edge::Right => "right",
            Edge::Top => "top",
            Edge::Bottom => "bottom",
        }
    }
}

/// What the App should do after a key.
pub enum CanvasAction {
    None,
    /// `Enter` — a mask PNG was rasterized; hand it to Chat as an inpaint mask.
    MaskReady(PathBuf),
    /// `M`-mode `Enter` — an outpaint job: a grey-padded base + a band mask. Chat
    /// continues over the enlarged `base`, inpainting the new (white) `mask` region.
    OutpaintReady { base: PathBuf, mask: PathBuf },
}

pub struct CanvasState {
    out_dir: PathBuf,
    /// Grid dimensions (cols, rows) — runtime, toggled with `g`.
    cols: usize,
    rows: usize,
    painted: Vec<bool>, // cols*rows
    cr: usize,
    cc: usize,
    base_path: Option<PathBuf>,
    base_dims: Option<(u32, u32)>,
    /// Outpaint mode: `Some(edge)` while choosing how to extend; `band` is in 128px units.
    outpaint: Option<Edge>,
    band: u32,
    /// Detected face boxes in normalized `[x1, y1, x2, y2]` (0..1), App-fed; the
    /// face-aware `B` preset punches these out of the background fill.
    face_boxes: Vec<[f32; 4]>,
    /// The base path the `face_boxes` were computed for (so stale boxes aren't used).
    faces_for: Option<PathBuf>,
    pub preview: Option<ratatui_image::protocol::StatefulProtocol>,
    pub preview_for: Option<PathBuf>,
    status: String,
    /// The base image downsampled to one RGB per grid cell — so the mask grid SHOWS the
    /// picture, and you can see *where* you're painting. Recomputed when the base /
    /// density changes.
    cell_colors: Vec<[u8; 3]>,
    colors_for: Option<(PathBuf, usize, usize)>,
}

impl CanvasState {
    pub fn new(out_dir: PathBuf) -> Self {
        Self {
            out_dir,
            cols: COLS,
            rows: ROWS,
            painted: vec![false; COLS * ROWS],
            cr: 0,
            cc: 0,
            base_path: None,
            base_dims: None,
            outpaint: None,
            band: 1,
            face_boxes: Vec::new(),
            faces_for: None,
            preview: None,
            preview_for: None,
            status: String::new(),
            cell_colors: Vec::new(),
            colors_for: None,
        }
    }

    /// The base image the App should run face detection on, when it differs from the
    /// one we already have boxes for (`None` → nothing to detect / already current).
    pub fn faces_needed_for(&self) -> Option<PathBuf> {
        match (&self.base_path, &self.faces_for) {
            (Some(b), f) if Some(b) != f.as_ref() => Some(b.clone()),
            _ => None,
        }
    }

    /// The App delivers detected face boxes (normalized xyxy) for `base`.
    pub fn set_faces(&mut self, base: PathBuf, boxes: Vec<[f32; 4]>) {
        self.faces_for = Some(base);
        self.face_boxes = boxes;
    }

    /// The Canvas owns the keyboard (preset letters + Space painting).
    pub fn captures_input(&self) -> bool {
        true
    }

    /// The base image to mask (set by the App from the current Chat base).
    pub fn set_base(&mut self, path: Option<PathBuf>, dims: Option<(u32, u32)>) {
        if path != self.base_path {
            // A different base invalidates the cached face boxes.
            self.face_boxes.clear();
            self.faces_for = None;
        }
        self.base_path = path;
        self.base_dims = dims;
    }

    pub fn base_path(&self) -> Option<PathBuf> {
        self.base_path.clone()
    }

    fn at(&self, r: usize, c: usize) -> usize {
        r * self.cols + c
    }

    /// Cycle to the next grid density (`g`): rebuild the (cleared) mask + clamp cursor.
    fn cycle_density(&mut self) {
        let cur = GRID_DENSITIES.iter().position(|&(c, r)| c == self.cols && r == self.rows).unwrap_or(0);
        let (c, r) = GRID_DENSITIES[(cur + 1) % GRID_DENSITIES.len()];
        self.cols = c;
        self.rows = r;
        self.painted = vec![false; c * r];
        self.cr = self.cr.min(r - 1);
        self.cc = self.cc.min(c - 1);
        self.status = format!("grid {c}×{r} (mask cleared)");
    }

    fn fill_rows(&mut self, r0: usize, r1: usize) {
        for r in r0..r1.min(self.rows) {
            for c in 0..self.cols {
                let i = self.at(r, c);
                self.painted[i] = true;
            }
        }
    }

    /// Un-paint every cell overlapping a detected face box (slightly padded so the
    /// face's edges + a little margin stay in the preserved region). Returns the count.
    fn clear_face_cells(&mut self) -> usize {
        let mut cleared = 0;
        for b in &self.face_boxes.clone() {
            // Pad the box by ~one cell so jaw/hair near the boundary is kept too.
            let px = 1.0 / self.cols as f32;
            let py = 1.0 / self.rows as f32;
            let (x1, y1, x2, y2) = (b[0] - px, b[1] - py, b[2] + px, b[3] + py);
            for r in 0..self.rows {
                for c in 0..self.cols {
                    // This cell's normalized extent.
                    let (cx1, cx2) = (c as f32 / self.cols as f32, (c + 1) as f32 / self.cols as f32);
                    let (cy1, cy2) = (r as f32 / self.rows as f32, (r + 1) as f32 / self.rows as f32);
                    let overlaps = cx1 < x2 && cx2 > x1 && cy1 < y2 && cy2 > y1;
                    if overlaps {
                        let i = self.at(r, c);
                        if self.painted[i] {
                            self.painted[i] = false;
                            cleared += 1;
                        }
                    }
                }
            }
        }
        cleared
    }

    fn fill_cols(&mut self, c0: usize, c1: usize) {
        for r in 0..self.rows {
            for c in c0..c1.min(self.cols) {
                let i = self.at(r, c);
                self.painted[i] = true;
            }
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> CanvasAction {
        // ── Outpaint mode: pick an edge + band, Enter to apply, M/Esc to cancel. ──
        if let Some(edge) = self.outpaint {
            match key.code {
                KeyCode::Esc | KeyCode::Char('m' | 'M') => {
                    self.outpaint = None;
                    self.status = "outpaint cancelled".into();
                }
                KeyCode::Left | KeyCode::Char('h') => self.outpaint = Some(Edge::Left),
                KeyCode::Right | KeyCode::Char('l') => self.outpaint = Some(Edge::Right),
                KeyCode::Up | KeyCode::Char('k') => self.outpaint = Some(Edge::Top),
                KeyCode::Down | KeyCode::Char('j') => self.outpaint = Some(Edge::Bottom),
                KeyCode::Char('+' | '=') => self.band = (self.band + 1).min(OUTPAINT_MAX_BANDS),
                KeyCode::Char('-' | '_') => self.band = self.band.saturating_sub(1).max(1),
                KeyCode::Enter => return self.produce_outpaint(edge),
                _ => {}
            }
            return CanvasAction::None;
        }
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        match key.code {
            // M — enter outpaint mode (extend the canvas instead of masking inside it).
            KeyCode::Char('m' | 'M') => {
                self.outpaint = Some(Edge::Right);
                self.status = "outpaint: ←/→/↑/↓ edge · +/- band · Enter apply · M/Esc cancel".into();
            }
            KeyCode::Left | KeyCode::Char('h') => self.move_cursor(0, -1, shift),
            KeyCode::Right | KeyCode::Char('l') => self.move_cursor(0, 1, shift),
            KeyCode::Up | KeyCode::Char('k') => self.move_cursor(-1, 0, shift),
            KeyCode::Down | KeyCode::Char('j') => self.move_cursor(1, 0, shift),
            KeyCode::Char(' ') => {
                let i = self.at(self.cr, self.cc);
                self.painted[i] = !self.painted[i];
            }
            // g — cycle the grid density (coarse → fine), clearing the mask.
            KeyCode::Char('g') => self.cycle_density(),
            // Preset regions (white = inpaint).
            KeyCode::Char('S') => self.fill_rows(0, (self.rows as f32 * 0.30).round() as usize),
            // B — background: fill the upper region, then punch out any detected faces
            // so the inpaint regenerates the background while preserving the people.
            KeyCode::Char('B') => {
                self.fill_rows(0, (self.rows as f32 * 0.60).round() as usize);
                let cleared = self.clear_face_cells();
                self.status = match (self.faces_for.is_some(), cleared) {
                    (false, _) => "background (detecting faces…)".into(),
                    (true, 0) => "background (no faces detected)".into(),
                    (true, n) => format!("background · preserved {n} face cell(s)"),
                };
            }
            KeyCode::Char('F') => self.fill_rows((self.rows as f32 * 0.60).round() as usize, self.rows),
            KeyCode::Char('L') => self.fill_cols(0, self.cols / 2),
            KeyCode::Char('R') => self.fill_cols(self.cols / 2, self.cols),
            KeyCode::Char('P') => self.fill_cols(self.cols / 3, 2 * self.cols / 3), // centre person column
            KeyCode::Char('C') => {
                self.painted.iter_mut().for_each(|p| *p = false);
            }
            KeyCode::Enter => return self.rasterize(),
            _ => {}
        }
        CanvasAction::None
    }

    fn move_cursor(&mut self, dr: i32, dc: i32, paint: bool) {
        self.cr = (self.cr as i32 + dr).clamp(0, self.rows as i32 - 1) as usize;
        self.cc = (self.cc as i32 + dc).clamp(0, self.cols as i32 - 1) as usize;
        if paint {
            let i = self.at(self.cr, self.cc);
            self.painted[i] = true;
        }
    }

    /// Rasterize the grid to a full-res white-on-black mask PNG at the base image's
    /// resolution. White cells → inpaint region.
    fn rasterize(&mut self) -> CanvasAction {
        let Some((w, h)) = self.base_dims else {
            self.status = "no base image — generate one in Chat first".into();
            return CanvasAction::None;
        };
        if !self.painted.iter().any(|p| *p) {
            self.status = "nothing painted — paint a region first".into();
            return CanvasAction::None;
        }
        let mut img = GrayImage::new(w, h);
        for r in 0..self.rows {
            for c in 0..self.cols {
                if !self.painted[self.at(r, c)] {
                    continue;
                }
                let x0 = (c as u32 * w) / self.cols as u32;
                let x1 = ((c as u32 + 1) * w) / self.cols as u32;
                let y0 = (r as u32 * h) / self.rows as u32;
                let y1 = ((r as u32 + 1) * h) / self.rows as u32;
                for y in y0..y1 {
                    for x in x0..x1 {
                        img.put_pixel(x, y, Luma([255]));
                    }
                }
            }
        }
        let _ = std::fs::create_dir_all(&self.out_dir);
        let path = self.out_dir.join("canvas-mask.png");
        match img.save(&path) {
            Ok(()) => CanvasAction::MaskReady(path),
            Err(e) => {
                self.status = format!("✗ mask save failed: {e}");
                CanvasAction::None
            }
        }
    }

    /// Build an outpaint job: a grey-padded copy of the base extended by `band×128px`
    /// on `edge`, plus a binary mask (white = the new band to generate). Chat continues
    /// over the enlarged base, inpainting only the band. Mid-grey fill + a hard binary
    /// mask is the clean-outpaint recipe (no seam from the original's edge colour).
    fn produce_outpaint(&mut self, edge: Edge) -> CanvasAction {
        let Some(path) = self.base_path.clone() else {
            self.status = "no base image — generate one in Chat first".into();
            return CanvasAction::None;
        };
        let src = match image::open(&path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                self.status = format!("✗ couldn't open base: {e}");
                return CanvasAction::None;
            }
        };
        let (w, h) = (src.width(), src.height());
        let pad = self.band * OUTPAINT_UNIT;
        let (nw, nh, ox, oy) = match edge {
            Edge::Left => (w + pad, h, pad, 0),
            Edge::Right => (w + pad, h, 0, 0),
            Edge::Top => (w, h + pad, 0, pad),
            Edge::Bottom => (w, h + pad, 0, 0),
        };
        // Grey-filled enlarged base with the original pasted at its offset.
        let mut base = image::RgbImage::from_pixel(nw, nh, image::Rgb([128, 128, 128]));
        for y in 0..h {
            for x in 0..w {
                base.put_pixel(ox + x, oy + y, *src.get_pixel(x, y));
            }
        }
        // Binary mask: white over the whole canvas, black over the original region.
        let mut mask = GrayImage::from_pixel(nw, nh, Luma([255]));
        for y in 0..h {
            for x in 0..w {
                mask.put_pixel(ox + x, oy + y, Luma([0]));
            }
        }
        let _ = std::fs::create_dir_all(&self.out_dir);
        let base_path = self.out_dir.join("canvas-outpaint-base.png");
        let mask_path = self.out_dir.join("canvas-outpaint-mask.png");
        if let Err(e) = base.save(&base_path).and_then(|_| mask.save(&mask_path).map_err(Into::into)) {
            self.status = format!("✗ outpaint save failed: {e}");
            return CanvasAction::None;
        }
        self.outpaint = None;
        self.status = format!("outpaint {} +{}px → continue in Chat", edge.label(), pad);
        CanvasAction::OutpaintReady { base: base_path, mask: mask_path }
    }

    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        self.ensure_cell_colors();
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(area);
        self.render_image(f, cols[0]);
        self.render_grid(f, cols[1]);
    }

    /// Downsample the base image to one RGB per grid cell (cached) so the mask grid
    /// shows the picture — you can see *where* on the image you're painting. Recomputed
    /// only when the base path or grid density changes.
    fn ensure_cell_colors(&mut self) {
        let key = self.base_path.as_ref().map(|p| (p.clone(), self.cols, self.rows));
        if key == self.colors_for {
            return;
        }
        self.colors_for = key.clone();
        self.cell_colors = match key {
            Some((path, cols, rows)) => image::open(&path)
                .ok()
                .map(|img| {
                    // Nearest-neighbour shrink to the grid — one averaged-ish pixel/cell.
                    let small = img.resize_exact(cols as u32, rows as u32, image::imageops::FilterType::Triangle).to_rgb8();
                    (0..rows)
                        .flat_map(|r| (0..cols).map(move |c| (r, c)))
                        .map(|(r, c)| small.get_pixel(c as u32, r as u32).0)
                        .collect()
                })
                .unwrap_or_default(),
            None => Vec::new(),
        };
    }

    fn render_image(&mut self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" Base image ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        match &mut self.preview {
            Some(p) => f.render_stateful_widget(ratatui_image::StatefulImage::new(), inner, p),
            None => f.render_widget(
                Paragraph::new("\n  No base image. Generate one in Chat (Ctrl-1), then return here.")
                    .style(Style::new().fg(Color::DarkGray))
                    .wrap(Wrap { trim: true }),
                inner,
            ),
        }
    }

    fn render_grid(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(format!(" Mask {}×{} · over the image · white = inpaint ", self.cols, self.rows));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(4)])
            .split(inner);

        // The grid cells carry the base image's colour (so you see WHERE you're masking);
        // painted cells get a bright-green overlay; the cursor is a high-contrast box.
        let have_image = self.cell_colors.len() == self.cols * self.rows;
        let mut lines: Vec<Line> = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            let mut spans = Vec::with_capacity(self.cols);
            for c in 0..self.cols {
                let i = self.at(r, c);
                let painted = self.painted[i];
                let cursor = r == self.cr && c == self.cc;
                let img_bg = have_image.then(|| {
                    let [cr, cg, cb] = self.cell_colors[i];
                    Color::Rgb(cr, cg, cb)
                });
                let (glyph, style) = if cursor {
                    // High-contrast cursor regardless of the image colour underneath.
                    ("[]", Style::new().bg(Color::Cyan).fg(Color::Black).add_modifier(Modifier::BOLD))
                } else if painted {
                    // Bright-green mask overlay on the image colour.
                    let st = Style::new().fg(Color::LightGreen).add_modifier(Modifier::BOLD);
                    ("▓▓", if let Some(bg) = img_bg { st.bg(bg) } else { st })
                } else if let Some(bg) = img_bg {
                    // Show the image: two spaces tinted with the cell colour.
                    ("  ", Style::new().bg(bg))
                } else {
                    ("··", Style::new().fg(Color::DarkGray))
                };
                spans.push(Span::styled(glyph, style));
            }
            lines.push(Line::from(spans));
        }
        f.render_widget(Paragraph::new(lines), rows[0]);

        let painted_n = self.painted.iter().filter(|p| **p).count();
        let mut legend = if let Some(edge) = self.outpaint {
            vec![
                Line::from(Span::styled(
                    format!("OUTPAINT · edge: {} · band: {}×128px", edge.label(), self.band),
                    Style::new().fg(Color::Magenta).add_modifier(ratatui::style::Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    "←/→/↑/↓ pick edge · +/- band · Enter apply · M/Esc cancel",
                    Style::new().fg(Color::DarkGray),
                )),
            ]
        } else {
            vec![
                Line::from(Span::styled(
                    "move ←↑↓→/hjkl · Shift+move paint · Space toggle · g grid · C clear · M outpaint · Enter → Chat",
                    Style::new().fg(Color::Gray),
                )),
                Line::from(Span::styled(
                    "presets: S sky · B background (face-aware) · F foreground · L/R halves · P person",
                    Style::new().fg(Color::DarkGray),
                )),
            ]
        };
        legend.push(Line::from(Span::styled(
            if self.status.is_empty() { format!("{painted_n} cell(s) painted") } else { self.status.clone() },
            Style::new().fg(Color::Yellow),
        )));
        f.render_widget(Paragraph::new(legend).wrap(Wrap { trim: true }), rows[1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }
    fn shift(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::SHIFT)
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("plakat-canvas-test-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn space_toggles_and_shift_move_paints() {
        let mut s = CanvasState::new(tmp("paint"));
        s.handle_key(key(KeyCode::Char(' ')));
        assert!(s.painted[s.at(0, 0)], "Space painted the cursor cell");
        s.handle_key(key(KeyCode::Char(' ')));
        assert!(!s.painted[s.at(0, 0)], "Space toggles off");
        // Shift+Right paints while moving.
        s.handle_key(shift(KeyCode::Right));
        assert_eq!((s.cr, s.cc), (0, 1));
        assert!(s.painted[s.at(0, 1)]);
    }

    #[test]
    fn sky_preset_fills_the_top_rows() {
        let mut s = CanvasState::new(tmp("sky"));
        s.handle_key(KeyEvent::new(KeyCode::Char('S'), KeyModifiers::SHIFT));
        let top = (ROWS as f32 * 0.30).round() as usize;
        assert!(s.painted[s.at(0, 0)] && s.painted[s.at(top - 1, COLS - 1)]);
        assert!(!s.painted[s.at(ROWS - 1, 0)], "bottom rows untouched");
        // C clears.
        s.handle_key(KeyEvent::new(KeyCode::Char('C'), KeyModifiers::SHIFT));
        assert!(!s.painted.iter().any(|p| *p));
    }

    #[test]
    fn cell_colors_downsample_the_base_image_to_the_grid() {
        let d = tmp("cellcolors");
        // A 64×48 image: left half red, right half blue → the grid should reflect that.
        let mut img = image::RgbImage::new(64, 48);
        for (x, _y, p) in img.enumerate_pixels_mut() {
            *p = if x < 32 { image::Rgb([200, 0, 0]) } else { image::Rgb([0, 0, 200]) };
        }
        let base = d.join("base.png");
        img.save(&base).unwrap();
        let mut s = CanvasState::new(d.clone());
        s.set_base(Some(base.clone()), Some((64, 48)));
        s.ensure_cell_colors();
        assert_eq!(s.cell_colors.len(), s.cols * s.rows, "one colour per grid cell");
        // Top-left cell is reddish; top-right cell is bluish.
        let tl = s.cell_colors[s.at(0, 0)];
        let tr = s.cell_colors[s.at(0, s.cols - 1)];
        assert!(tl[0] > tl[2], "left cell leans red: {tl:?}");
        assert!(tr[2] > tr[0], "right cell leans blue: {tr:?}");
        // Toggling density recomputes for the new cell count.
        s.handle_key(key(KeyCode::Char('g')));
        s.ensure_cell_colors();
        assert_eq!(s.cell_colors.len(), s.cols * s.rows);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn enter_rasterizes_a_full_res_mask_png() {
        let d = tmp("raster");
        let mut s = CanvasState::new(d.clone());
        s.set_base(Some(d.join("base.png")), Some((64, 48)));
        // Nothing painted → no mask.
        assert!(matches!(s.handle_key(key(KeyCode::Enter)), CanvasAction::None));
        // Paint the left half, rasterize.
        s.handle_key(KeyEvent::new(KeyCode::Char('L'), KeyModifiers::SHIFT));
        match s.handle_key(key(KeyCode::Enter)) {
            CanvasAction::MaskReady(p) => {
                let img = image::open(&p).unwrap().to_luma8();
                assert_eq!(img.dimensions(), (64, 48), "full base resolution");
                assert_eq!(img.get_pixel(2, 24)[0], 255, "left half is white (inpaint)");
                assert_eq!(img.get_pixel(62, 24)[0], 0, "right half is black (preserve)");
            }
            _ => panic!("expected MaskReady"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn outpaint_extends_the_base_and_masks_the_new_band() {
        let d = tmp("outpaint");
        // A real 64×48 base image on disk (outpaint loads + pads it).
        let base = d.join("base.png");
        image::RgbImage::from_pixel(64, 48, image::Rgb([10, 20, 30])).save(&base).unwrap();
        let mut s = CanvasState::new(d.clone());
        s.set_base(Some(base.clone()), Some((64, 48)));

        // M enters outpaint mode (default edge = right).
        s.handle_key(key(KeyCode::Char('M')));
        assert!(s.outpaint.is_some());
        // Pick the right edge, bump the band to 2 (×128 = 256px).
        s.handle_key(key(KeyCode::Right));
        s.handle_key(key(KeyCode::Char('+')));
        assert_eq!(s.band, 2);

        match s.handle_key(key(KeyCode::Enter)) {
            CanvasAction::OutpaintReady { base: bp, mask: mp } => {
                let img = image::open(&bp).unwrap().to_rgb8();
                assert_eq!(img.dimensions(), (64 + 256, 48), "extended on the right by 2×128");
                // Original pixels preserved at the left; grey fill in the new band.
                assert_eq!(*img.get_pixel(2, 24), image::Rgb([10, 20, 30]));
                assert_eq!(*img.get_pixel(300, 24), image::Rgb([128, 128, 128]));
                let mask = image::open(&mp).unwrap().to_luma8();
                assert_eq!(mask.dimensions(), (64 + 256, 48));
                assert_eq!(mask.get_pixel(2, 24)[0], 0, "original region preserved (black)");
                assert_eq!(mask.get_pixel(300, 24)[0], 255, "new band regenerates (white)");
            }
            _ => panic!("expected OutpaintReady"),
        }
        // Mode cleared after producing.
        assert!(s.outpaint.is_none());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn face_aware_background_preserves_face_cells() {
        let d = tmp("faceaware");
        let mut s = CanvasState::new(d.clone());
        s.set_base(Some(d.join("base.png")), Some((512, 512)));
        // A face box covering the centre of the top half (normalized xyxy).
        s.set_faces(d.join("base.png"), vec![[0.40, 0.10, 0.60, 0.40]]);
        // B fills the top 60% then punches out the face cells.
        s.handle_key(KeyEvent::new(KeyCode::Char('B'), KeyModifiers::SHIFT));
        // A background cell well outside the face (top-left corner) stays painted.
        assert!(s.painted[s.at(0, 0)], "corner background is masked");
        // The cell at the face centre is cleared (preserved).
        let fr = (0.25 * ROWS as f32) as usize; // ~row inside [0.10,0.40]
        let fc = (0.50 * COLS as f32) as usize; // ~col inside [0.40,0.60]
        assert!(!s.painted[s.at(fr, fc)], "face cell is preserved (not masked)");
        assert!(s.status.contains("preserved"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn background_without_faces_fills_plainly() {
        let d = tmp("noface");
        let mut s = CanvasState::new(d.clone());
        s.set_base(Some(d.join("base.png")), Some((512, 512)));
        s.set_faces(d.join("base.png"), vec![]); // detection ran, found nothing
        s.handle_key(KeyEvent::new(KeyCode::Char('B'), KeyModifiers::SHIFT));
        let top = (ROWS as f32 * 0.60).round() as usize;
        assert!(s.painted[s.at(0, 0)] && s.painted[s.at(top - 1, COLS - 1)]);
        assert!(s.status.contains("no faces"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn g_cycles_grid_density_and_clears_the_mask() {
        let mut s = CanvasState::new(tmp("density"));
        assert_eq!((s.cols, s.rows), (16, 12));
        // Paint something, then toggle → finer grid, mask cleared.
        s.handle_key(key(KeyCode::Char(' ')));
        assert!(s.painted.iter().any(|p| *p));
        s.handle_key(key(KeyCode::Char('g')));
        assert_eq!((s.cols, s.rows), (24, 18));
        assert_eq!(s.painted.len(), 24 * 18);
        assert!(!s.painted.iter().any(|p| *p), "mask cleared on density change");
        // Cycle through the rest back to coarse.
        s.handle_key(key(KeyCode::Char('g')));
        assert_eq!((s.cols, s.rows), (32, 24));
        s.handle_key(key(KeyCode::Char('g')));
        assert_eq!((s.cols, s.rows), (16, 12));
    }

    #[test]
    fn fine_grid_rasterizes_at_full_resolution() {
        let d = tmp("fine-raster");
        let mut s = CanvasState::new(d.clone());
        s.set_base(Some(d.join("base.png")), Some((96, 96)));
        s.handle_key(key(KeyCode::Char('g'))); // 24×18
        // Paint the top-left cell, rasterize — still full base resolution.
        s.handle_key(key(KeyCode::Char(' ')));
        match s.handle_key(key(KeyCode::Enter)) {
            CanvasAction::MaskReady(p) => {
                let img = image::open(&p).unwrap().to_luma8();
                assert_eq!(img.dimensions(), (96, 96));
                assert_eq!(img.get_pixel(1, 1)[0], 255, "top-left cell white");
                assert_eq!(img.get_pixel(95, 95)[0], 0, "far corner preserved");
            }
            _ => panic!("expected MaskReady"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn faces_needed_for_tracks_the_base() {
        let d = tmp("needed");
        let mut s = CanvasState::new(d.clone());
        assert!(s.faces_needed_for().is_none(), "no base → nothing to detect");
        s.set_base(Some(d.join("a.png")), Some((64, 64)));
        assert_eq!(s.faces_needed_for(), Some(d.join("a.png")));
        s.set_faces(d.join("a.png"), vec![]);
        assert!(s.faces_needed_for().is_none(), "already detected for this base");
        // Switching base invalidates and re-requests.
        s.set_base(Some(d.join("b.png")), Some((64, 64)));
        assert_eq!(s.faces_needed_for(), Some(d.join("b.png")));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn outpaint_m_again_cancels() {
        let mut s = CanvasState::new(tmp("op-cancel"));
        s.handle_key(key(KeyCode::Char('M')));
        assert!(s.outpaint.is_some());
        s.handle_key(key(KeyCode::Char('M')));
        assert!(s.outpaint.is_none());
    }
}
