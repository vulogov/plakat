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
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

const COLS: usize = 16;
const ROWS: usize = 12;

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
    painted: Vec<bool>, // COLS*ROWS
    cr: usize,
    cc: usize,
    base_path: Option<PathBuf>,
    base_dims: Option<(u32, u32)>,
    /// Outpaint mode: `Some(edge)` while choosing how to extend; `band` is in 128px units.
    outpaint: Option<Edge>,
    band: u32,
    pub preview: Option<ratatui_image::protocol::StatefulProtocol>,
    pub preview_for: Option<PathBuf>,
    status: String,
}

impl CanvasState {
    pub fn new(out_dir: PathBuf) -> Self {
        Self {
            out_dir,
            painted: vec![false; COLS * ROWS],
            cr: 0,
            cc: 0,
            base_path: None,
            base_dims: None,
            outpaint: None,
            band: 1,
            preview: None,
            preview_for: None,
            status: String::new(),
        }
    }

    /// The Canvas owns the keyboard (preset letters + Space painting).
    pub fn captures_input(&self) -> bool {
        true
    }

    /// The base image to mask (set by the App from the current Chat base).
    pub fn set_base(&mut self, path: Option<PathBuf>, dims: Option<(u32, u32)>) {
        self.base_path = path;
        self.base_dims = dims;
    }

    pub fn base_path(&self) -> Option<PathBuf> {
        self.base_path.clone()
    }

    fn at(&self, r: usize, c: usize) -> usize {
        r * COLS + c
    }

    fn fill_rows(&mut self, r0: usize, r1: usize) {
        for r in r0..r1.min(ROWS) {
            for c in 0..COLS {
                let i = self.at(r, c);
                self.painted[i] = true;
            }
        }
    }

    fn fill_cols(&mut self, c0: usize, c1: usize) {
        for r in 0..ROWS {
            for c in c0..c1.min(COLS) {
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
            // Preset regions (white = inpaint).
            KeyCode::Char('S') => self.fill_rows(0, (ROWS as f32 * 0.30).round() as usize),
            KeyCode::Char('B') => self.fill_rows(0, (ROWS as f32 * 0.60).round() as usize),
            KeyCode::Char('F') => self.fill_rows((ROWS as f32 * 0.60).round() as usize, ROWS),
            KeyCode::Char('L') => self.fill_cols(0, COLS / 2),
            KeyCode::Char('R') => self.fill_cols(COLS / 2, COLS),
            KeyCode::Char('P') => self.fill_cols(COLS / 3, 2 * COLS / 3), // centre person column
            KeyCode::Char('C') => {
                self.painted.iter_mut().for_each(|p| *p = false);
            }
            KeyCode::Enter => return self.rasterize(),
            _ => {}
        }
        CanvasAction::None
    }

    fn move_cursor(&mut self, dr: i32, dc: i32, paint: bool) {
        self.cr = (self.cr as i32 + dr).clamp(0, ROWS as i32 - 1) as usize;
        self.cc = (self.cc as i32 + dc).clamp(0, COLS as i32 - 1) as usize;
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
        for r in 0..ROWS {
            for c in 0..COLS {
                if !self.painted[self.at(r, c)] {
                    continue;
                }
                let x0 = (c as u32 * w) / COLS as u32;
                let x1 = ((c as u32 + 1) * w) / COLS as u32;
                let y0 = (r as u32 * h) / ROWS as u32;
                let y1 = ((r as u32 + 1) * h) / ROWS as u32;
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
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(area);
        self.render_image(f, cols[0]);
        self.render_grid(f, cols[1]);
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
        let block = Block::default().borders(Borders::ALL).title(" Mask · white = inpaint ");
        let inner = block.inner(area);
        f.render_widget(block, area);

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(4)])
            .split(inner);

        let mut lines: Vec<Line> = Vec::with_capacity(ROWS);
        for r in 0..ROWS {
            let mut spans = Vec::with_capacity(COLS);
            for c in 0..COLS {
                let painted = self.painted[self.at(r, c)];
                let cursor = r == self.cr && c == self.cc;
                let style = if cursor {
                    Style::new().bg(Color::Cyan).fg(if painted { Color::White } else { Color::Black })
                } else if painted {
                    Style::new().fg(Color::Green)
                } else {
                    Style::new().fg(Color::DarkGray)
                };
                spans.push(Span::styled(if painted { "██" } else { "··" }, style));
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
                    "move ←↑↓→/hjkl · Shift+move paint · Space toggle · C clear · M outpaint · Enter → Chat",
                    Style::new().fg(Color::Gray),
                )),
                Line::from(Span::styled(
                    "presets: S sky · B background · F foreground · L/R halves · P person",
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
    fn outpaint_m_again_cancels() {
        let mut s = CanvasState::new(tmp("op-cancel"));
        s.handle_key(key(KeyCode::Char('M')));
        assert!(s.outpaint.is_some());
        s.handle_key(key(KeyCode::Char('M')));
        assert!(s.outpaint.is_none());
    }
}
