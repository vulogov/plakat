//! Interactive TUI fractal explorer (RFC FRACTALS-1, Phase 6) — `--fractal-explore`.
//!
//! Pan / zoom / retune a fractal live in the terminal and save when you like it. Renders
//! Track A (pure CPU, deterministic) to an in-memory buffer and displays it inline via
//! `ratatui-image` (Kitty / iTerm2 / Sixel). Gated on the `ui` feature (the TUI stack);
//! the deterministic render engine itself stays feature-free.

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;
use ratatui_image::picker::{Picker, ProtocolType};
use ratatui_image::protocol::StatefulProtocol;
use ratatui_image::{Resize, StatefulImage};
use std::path::PathBuf;
use std::time::Duration;

use super::spec::{Coloring, FractalKind};
use super::FractalSpec;

/// Kinds cycled with `n` / `N`. Escape families first (pan/zoom is most meaningful there),
/// then the auto-fit families (pan is a no-op, zoom still scales).
const CYCLE: &[FractalKind] = &[
    FractalKind::Mandelbrot, FractalKind::Julia, FractalKind::BurningShip, FractalKind::Tricorn,
    FractalKind::Newton, FractalKind::Phoenix, FractalKind::Ifs, FractalKind::Lsystem,
    FractalKind::Flame, FractalKind::Attractor, FractalKind::Buddhabrot,
];
const PALETTES: &[&str] =
    &["fire", "ice", "electric", "neon", "pastel", "monochrome", "midnight", "earth"];
const COLORINGS: &[Coloring] = &[
    Coloring::Smooth, Coloring::Histogram, Coloring::Distance, Coloring::OrbitTrap,
    Coloring::Angle, Coloring::Stripe,
];
/// Long-side pixel budget for the live preview (kept modest for interactivity; the saved
/// render uses the spec's full resolution).
const PREVIEW_LONG: u32 = 1000;

/// The default viewport center for a kind (so `r` resets somewhere sensible).
fn default_center(kind: FractalKind) -> [f64; 2] {
    match kind {
        FractalKind::Mandelbrot => [-0.5, 0.0],
        FractalKind::BurningShip => [-0.4, -0.5],
        _ => [0.0, 0.0],
    }
}

struct Explorer {
    spec: FractalSpec,
    out: PathBuf,
    picker: Picker,
    proto: Option<StatefulProtocol>,
    dirty: bool,
    should_quit: bool,
    status: String,
    render_ms: u128,
    saved: Option<String>,
}

/// A reduced copy of the spec for the live preview: smaller canvas, no supersample, capped
/// stochastic iteration counts — so a keypress re-renders in a beat.
fn preview_spec(spec: &FractalSpec) -> FractalSpec {
    let long = spec.width.max(spec.height).max(1) as f64;
    let scale = (PREVIEW_LONG as f64 / long).min(1.0);
    let mut pv = spec.clone();
    pv.width = ((spec.width as f64 * scale).round() as u32).max(64);
    pv.height = ((spec.height as f64 * scale).round() as u32).max(64);
    pv.supersample = 1;
    pv.buddha_samples = pv.buddha_samples.min(2_000_000);
    pv.ifs.iterations = pv.ifs.iterations.min(1_500_000);
    pv.flame.iterations = pv.flame.iterations.min(1_500_000);
    pv.attractor.iterations = pv.attractor.iterations.min(1_500_000);
    pv
}

impl Explorer {
    fn new(spec: FractalSpec, out: PathBuf, picker: Picker) -> Self {
        Explorer {
            spec,
            out,
            picker,
            proto: None,
            dirty: true,
            should_quit: false,
            status: String::new(),
            render_ms: 0,
            saved: None,
        }
    }

    /// Re-render the preview into a fresh display protocol.
    fn rerender(&mut self) {
        let pv = preview_spec(&self.spec);
        let t0 = std::time::Instant::now();
        match super::render_spec(&pv) {
            Ok(r) => {
                self.render_ms = t0.elapsed().as_millis();
                match image::RgbImage::from_raw(r.width, r.height, r.pixels) {
                    Some(img) => {
                        let dynimg = image::DynamicImage::ImageRgb8(img);
                        self.proto = Some(self.picker.new_resize_protocol(dynimg));
                        self.status.clear();
                    }
                    None => self.status = "render buffer mismatch".to_string(),
                }
            }
            Err(e) => self.status = format!("render error: {e}"),
        }
        self.dirty = false;
    }

    /// Vertical span of the current view in complex units (pan step basis).
    fn span(&self) -> f64 {
        3.0 / self.spec.zoom
    }

    fn cycle_kind(&mut self, forward: bool) {
        let cur = CYCLE.iter().position(|&k| k == self.spec.kind).unwrap_or(0);
        let n = CYCLE.len();
        let next = if forward { (cur + 1) % n } else { (cur + n - 1) % n };
        self.spec.kind = CYCLE[next];
        // Reset the viewport so the new kind is framed sensibly.
        self.spec.center = default_center(self.spec.kind);
        self.spec.zoom = 1.0;
        self.dirty = true;
    }

    fn cycle_palette(&mut self) {
        let cur = PALETTES.iter().position(|&p| p == self.spec.palette.preset).unwrap_or(0);
        self.spec.palette.preset = PALETTES[(cur + 1) % PALETTES.len()].to_string();
        self.spec.palette.stops.clear();
        self.dirty = true;
    }

    fn cycle_coloring(&mut self) {
        let cur = COLORINGS.iter().position(|&c| c == self.spec.coloring).unwrap_or(0);
        self.spec.coloring = COLORINGS[(cur + 1) % COLORINGS.len()];
        self.dirty = true;
    }

    fn save(&mut self) {
        match super::render_to_file(&self.spec, &self.out) {
            Ok(()) => self.saved = Some(format!("saved {}", self.out.display())),
            Err(e) => self.saved = Some(format!("save failed: {e}")),
        }
    }

    fn handle_key(&mut self, code: KeyCode, shift: bool) {
        let pan = self.span() * 0.15;
        match code {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Left | KeyCode::Char('h') => { self.spec.center[0] -= pan; self.dirty = true; }
            KeyCode::Right | KeyCode::Char('l') => { self.spec.center[0] += pan; self.dirty = true; }
            KeyCode::Up | KeyCode::Char('k') => { self.spec.center[1] += pan; self.dirty = true; }
            KeyCode::Down | KeyCode::Char('j') => { self.spec.center[1] -= pan; self.dirty = true; }
            KeyCode::Char('+') | KeyCode::Char('=') | KeyCode::Char('i') => {
                self.spec.zoom *= 1.4; self.dirty = true;
            }
            KeyCode::Char('-') | KeyCode::Char('_') | KeyCode::Char('o') => {
                self.spec.zoom = (self.spec.zoom / 1.4).max(1e-6); self.dirty = true;
            }
            KeyCode::Char(']') => {
                self.spec.max_iter = ((self.spec.max_iter as f64 * 1.3) as u32).min(100_000);
                self.dirty = true;
            }
            KeyCode::Char('[') => {
                self.spec.max_iter = ((self.spec.max_iter as f64 * 0.77) as u32).max(20);
                self.dirty = true;
            }
            KeyCode::Char('p') => self.cycle_palette(),
            KeyCode::Char('c') => self.cycle_coloring(),
            KeyCode::Char('n') => self.cycle_kind(!shift),
            KeyCode::Char('N') => self.cycle_kind(false),
            KeyCode::Char('r') => {
                self.spec.center = default_center(self.spec.kind);
                self.spec.zoom = 1.0;
                self.spec.max_iter = 500;
                self.dirty = true;
            }
            KeyCode::Char('s') => self.save(),
            _ => {}
        }
    }

    fn draw(&mut self, f: &mut Frame) {
        let chunks = Layout::vertical([Constraint::Min(3), Constraint::Length(4)]).split(f.area());

        // Image pane.
        let block = Block::default().borders(Borders::ALL).title(" plakat fractals — explore ");
        let inner = block.inner(chunks[0]);
        f.render_widget(block, chunks[0]);
        if let Some(proto) = &mut self.proto {
            f.render_stateful_widget(
                StatefulImage::new().resize(Resize::Scale(None)),
                inner,
                proto,
            );
        }

        // Status pane.
        let s = &self.spec;
        let mut l1 = format!(
            " {}  center [{:.5}, {:.5}]  zoom {:.3}  iter {}  {}",
            s.kind.as_str(), s.center[0], s.center[1], s.zoom, s.max_iter, s.palette.preset,
        );
        if s.kind.is_escape_time() {
            l1.push_str(&format!("  {:?}", s.coloring));
        }
        l1.push_str(&format!("  ({}ms)", self.render_ms));
        let mut lines = vec![
            Line::from(Span::styled(l1, Style::default().fg(Color::Cyan))),
            Line::from(Span::raw(
                " ←↑↓→/hjkl pan   +/- zoom   [ ] iter   p palette   c coloring   n/N kind   r reset   s save   q quit",
            )),
        ];
        let mut foot = String::new();
        if let Some(saved) = &self.saved {
            foot.push_str(saved);
        }
        if !self.status.is_empty() {
            if !foot.is_empty() { foot.push_str("  •  "); }
            foot.push_str(&self.status);
        }
        lines.push(Line::from(Span::styled(
            format!(" {foot}"),
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )));
        f.render_widget(Paragraph::new(lines), chunks[1]);
    }

    fn event_loop(&mut self, terminal: &mut ratatui::DefaultTerminal) -> Result<()> {
        while !self.should_quit {
            if self.dirty {
                self.rerender();
            }
            terminal.draw(|f| self.draw(f))?;
            if event::poll(Duration::from_millis(150))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        self.handle_key(key.code, key.modifiers.contains(KeyModifiers::SHIFT));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Launch the interactive explorer with an initial spec, saving to `out` on `s`.
pub fn run(spec: FractalSpec, out: PathBuf) -> Result<()> {
    let picker = Picker::from_query_stdio().map_err(|_| terminal_error())?;
    if picker.protocol_type() == ProtocolType::Halfblocks {
        return Err(terminal_error());
    }
    let mut terminal = ratatui::init();
    let mut app = Explorer::new(spec, out, picker);
    let res = app.event_loop(&mut terminal);
    ratatui::restore();
    res
}

fn terminal_error() -> anyhow::Error {
    anyhow::anyhow!(
        "--fractal-explore needs a graphics-capable terminal (Kitty, iTerm2, or a Sixel \
         terminal like WezTerm/foot). Your terminal reports no inline-image support."
    )
}

/// Preview-spec shaping is pure — unit-test it without a terminal.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_downsizes_and_caps() {
        let spec = FractalSpec {
            width: 4000,
            height: 3000,
            supersample: 4,
            kind: FractalKind::Flame,
            ..FractalSpec::default()
        };
        let pv = preview_spec(&spec);
        assert!(pv.width <= PREVIEW_LONG && pv.height <= PREVIEW_LONG);
        assert_eq!(pv.supersample, 1);
        assert!(pv.flame.iterations <= 1_500_000);
        // Aspect ratio preserved (4:3).
        let ar = pv.width as f64 / pv.height as f64;
        assert!((ar - 4.0 / 3.0).abs() < 0.05);
    }

    #[test]
    fn small_spec_is_not_upscaled() {
        let spec = FractalSpec { width: 200, height: 200, ..FractalSpec::default() };
        let pv = preview_spec(&spec);
        assert_eq!((pv.width, pv.height), (200, 200));
    }

    #[test]
    fn default_centers_are_finite() {
        for &k in CYCLE {
            let c = default_center(k);
            assert!(c[0].is_finite() && c[1].is_finite());
        }
    }
}
