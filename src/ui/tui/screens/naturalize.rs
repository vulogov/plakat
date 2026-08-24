//! The **Naturalize** tab (RFC QUALITY-8 P3) — interactive weight-free de-slop tuning inside `plakat ui`.
//!
//! Dial the weight-free knobs (polish / micro / grain / desaturate / paper) on the latest generated image
//! with a **live scorecard** (AI-tell + drivers) and image preview updating on each change, then save. The
//! naturalize *core* stays feature-agnostic; this is just the presentation layer (behind `ui`).

use std::path::PathBuf;

use crossterm::event::{KeyCode, KeyEvent};
use image::RgbImage;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::naturalize::{self, Analysis, Params};

/// What a key did, for the `App` to act on (rebuild preview / save).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NaturalizeAction {
    None,
    /// Params changed — re-apply + rebuild the preview.
    Reapply,
    /// Save the current processed image.
    Save,
}

/// One tunable knob: label, current value, range, step.
struct Knob {
    label: &'static str,
    max: f32,
}

const KNOBS: [Knob; 5] = [
    Knob { label: "polish", max: 1.0 },
    Knob { label: "micro", max: 1.0 },
    Knob { label: "grain", max: 1.0 },
    Knob { label: "desaturate", max: 1.0 },
    Knob { label: "paper", max: 1.5 },
];

pub struct NaturalizeState {
    pub source_path: Option<PathBuf>,
    /// The loaded original.
    pub source: Option<RgbImage>,
    /// Current knob values.
    pub params: Params,
    pub paper: f32,
    /// Selected knob (0..KNOBS.len()).
    pub selected: usize,
    /// The processed image (App builds `preview` from it via the Picker).
    pub processed: Option<RgbImage>,
    /// Scorecard of the processed image.
    pub analysis: Option<Analysis>,
    /// Terminal image protocol for the preview (built by the App).
    pub preview: Option<ratatui_image::protocol::StatefulProtocol>,
    pub status: String,
}

impl Default for NaturalizeState {
    fn default() -> Self {
        Self::new()
    }
}

impl NaturalizeState {
    pub fn new() -> Self {
        Self {
            source_path: None,
            source: None,
            params: naturalize::Preset::Photo.params(),
            paper: 0.0,
            selected: 0,
            processed: None,
            analysis: None,
            preview: None,
            status: String::new(),
        }
    }

    /// Load an image as the source (the App calls this with the latest generated frame). Resets the preview
    /// so the App re-applies + rebuilds it next tick.
    pub fn load(&mut self, path: PathBuf) {
        match image::open(&path) {
            Ok(img) => {
                self.source = Some(img.to_rgb8());
                self.source_path = Some(path);
                self.processed = None;
                self.preview = None;
                self.status.clear();
            }
            Err(e) => self.status = format!("load failed: {e}"),
        }
    }

    fn get(&self, i: usize) -> f32 {
        match i {
            0 => self.params.polish,
            1 => self.params.micro,
            2 => self.params.grain,
            3 => self.params.desaturate,
            _ => self.paper,
        }
    }
    fn set(&mut self, i: usize, v: f32) {
        let v = v.clamp(0.0, KNOBS[i].max);
        match i {
            0 => self.params.polish = v,
            1 => self.params.micro = v,
            2 => self.params.grain = v,
            3 => self.params.desaturate = v,
            _ => self.paper = v,
        }
    }

    /// Re-run the weight-free pass on the source → `processed` + `analysis`. The App rebuilds `preview`.
    pub fn apply(&mut self) {
        if let Some(src) = &self.source {
            let mut out = naturalize::apply(src, &self.params);
            if self.paper > 0.0 {
                out = naturalize::paper_texture(&out, self.paper);
            }
            self.analysis = Some(naturalize::analyze(&out));
            self.processed = Some(out);
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> NaturalizeAction {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected = (self.selected + KNOBS.len() - 1) % KNOBS.len();
                NaturalizeAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.selected = (self.selected + 1) % KNOBS.len();
                NaturalizeAction::None
            }
            KeyCode::Left | KeyCode::Char('h' | '-') => {
                let v = self.get(self.selected) - 0.05;
                self.set(self.selected, v);
                NaturalizeAction::Reapply
            }
            KeyCode::Right | KeyCode::Char('l' | '+' | '=') => {
                let v = self.get(self.selected) + 0.05;
                self.set(self.selected, v);
                NaturalizeAction::Reapply
            }
            KeyCode::Char('s' | 'S') => NaturalizeAction::Save,
            _ => NaturalizeAction::None,
        }
    }

    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
            .split(area);

        // ── left: image preview ──
        let block = Block::default().borders(Borders::ALL).title(" Preview ");
        let inner = block.inner(cols[0]);
        f.render_widget(block, cols[0]);
        match &mut self.preview {
            Some(p) => f.render_stateful_widget(ratatui_image::StatefulImage::new(), inner, p),
            None => f.render_widget(
                Paragraph::new("\n  No image. Generate one in Chat (Ctrl-1), then return here — the latest frame loads automatically.")
                    .style(Style::new().fg(Color::DarkGray))
                    .wrap(Wrap { trim: true }),
                inner,
            ),
        }

        // ── right: knobs + scorecard ──
        let rblock = Block::default().borders(Borders::ALL).title(" Naturalize — weight-free de-slop ");
        let rinner = rblock.inner(cols[1]);
        f.render_widget(rblock, cols[1]);
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(KNOBS.len() as u16 + 2), Constraint::Min(6), Constraint::Length(2)])
            .split(rinner);

        // knob sliders
        let mut lines: Vec<Line> = vec![Line::from("")];
        for (i, k) in KNOBS.iter().enumerate() {
            let v = self.get(i);
            let filled = ((v / k.max) * 12.0).round() as usize;
            let bar: String = "█".repeat(filled.min(12)) + &"░".repeat(12 - filled.min(12));
            let sel = i == self.selected;
            let style = if sel { Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::new() };
            lines.push(Line::from(vec![
                Span::styled(format!("{} {:<11}", if sel { "▶" } else { " " }, k.label), style),
                Span::styled(format!("{bar} {v:.2}"), style),
            ]));
        }
        f.render_widget(Paragraph::new(lines), rows[0]);

        // scorecard
        let mut sc: Vec<Line> = vec![Line::from(Span::styled("scorecard", Style::new().add_modifier(Modifier::BOLD)))];
        if let Some(an) = &self.analysis {
            let bar = |x: f32| { let n = ((x * 14.0).round() as usize).min(14); format!("{}{}", "█".repeat(n), "░".repeat(14 - n)) };
            let dot = if an.ai_tell > 0.5 { Span::styled("● ", Style::new().fg(Color::Red)) } else { Span::styled("● ", Style::new().fg(Color::Green)) };
            sc.push(Line::from(vec![dot, Span::raw(format!("AI-tell   {} {:.2}", bar(an.ai_tell), an.ai_tell))]));
            sc.push(Line::from(format!("  oversat  {} {:.2}", bar(an.saturation), an.saturation)));
            sc.push(Line::from(format!("  smooth   {} {:.2}", bar(an.smoothness_tell), an.smoothness_tell)));
            sc.push(Line::from(format!("  contrast {} {:.2}", bar(an.contrast), an.contrast)));
        } else {
            sc.push(Line::from(Span::styled("  (load an image)", Style::new().fg(Color::DarkGray))));
        }
        f.render_widget(Paragraph::new(sc), rows[1]);

        // footer / status
        let foot = if self.status.is_empty() {
            "↑↓ select · ←→ / +- adjust · s save".to_string()
        } else {
            self.status.clone()
        };
        f.render_widget(Paragraph::new(Line::from(Span::styled(foot, Style::new().fg(Color::DarkGray)))), rows[2]);
    }
}
