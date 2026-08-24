//! The **Naturalize** tab (RFC QUALITY-8 P3) — interactive weight-free de-slop tuning inside `plakat ui`.
//!
//! Dial the weight-free knobs (polish / micro / grain / desaturate / paper) on an image with a **live
//! scorecard** (AI-tell + drivers) and image preview updating on each change; **Space** toggles the
//! original ↔ de-slopped preview so the effect is visible; **s** saves. The naturalize *core* is
//! feature-agnostic; this is just the presentation layer (behind `ui`).

use std::path::PathBuf;

use crossterm::event::{KeyCode, KeyEvent};
use image::RgbImage;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::naturalize::{self, Analysis, Params};

/// What a key did, for the `App` to act on (save / reload — both need the App's workspace / Picker).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NaturalizeAction {
    None,
    /// Save the current processed image.
    Save,
    /// Drop the current source so the App reloads the newest image (latest Chat frame / workspace output).
    Reload,
}

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
    pub params: Params,
    pub paper: f32,
    pub selected: usize,
    /// The processed image; `None` = needs (re)compute (the App's sync runs `apply`).
    pub processed: Option<RgbImage>,
    /// Scorecard of the original and of the processed image (for a before/after readout).
    pub analysis: Option<Analysis>,
    pub orig_analysis: Option<Analysis>,
    /// Show the ORIGINAL in the preview (Space toggles) so the de-slop delta is visible.
    pub show_original: bool,
    /// The App should (re)build `preview` from the right image this frame.
    pub needs_preview: bool,
    /// Terminal image protocol for the preview (built by the App from `source`/`processed`).
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
            orig_analysis: None,
            show_original: false,
            needs_preview: false,
            preview: None,
            status: String::new(),
        }
    }

    /// Load an image as the source (App calls this with the latest frame / newest workspace image).
    pub fn load(&mut self, path: PathBuf) {
        match image::open(&path) {
            Ok(img) => {
                let rgb = img.to_rgb8();
                self.orig_analysis = Some(naturalize::analyze(&rgb));
                self.source = Some(rgb);
                self.source_path = Some(path);
                self.processed = None; // → App re-applies + rebuilds preview
                self.needs_preview = true;
                self.status.clear();
            }
            Err(e) => self.status = format!("load failed: {e}"),
        }
    }

    /// The image the preview should currently show (original when toggled, else processed).
    pub fn preview_image(&self) -> Option<&RgbImage> {
        if self.show_original { self.source.as_ref() } else { self.processed.as_ref() }
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

    fn adjust(&mut self, delta: f32) {
        let v = self.get(self.selected) + delta;
        self.set(self.selected, v);
        self.show_original = false; // adjusting shows the (new) processed result
        self.processed = None; // → recompute + rebuild preview
        self.needs_preview = true;
    }

    /// Re-run the weight-free pass on the source → `processed` + `analysis`. Called by the App's sync.
    pub fn apply(&mut self) {
        if let Some(src) = &self.source {
            let mut out = naturalize::apply(src, &self.params);
            if self.paper > 0.0 {
                out = naturalize::paper_texture(&out, self.paper);
            }
            self.analysis = Some(naturalize::analyze(&out));
            self.processed = Some(out);
            self.needs_preview = true;
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
                self.adjust(-0.05);
                NaturalizeAction::None
            }
            KeyCode::Right | KeyCode::Char('l' | '+' | '=') => {
                self.adjust(0.05);
                NaturalizeAction::None
            }
            KeyCode::Char(' ') => {
                // Toggle original ↔ de-slopped preview (only meaningful once processed).
                self.show_original = !self.show_original;
                self.needs_preview = true;
                NaturalizeAction::None
            }
            KeyCode::Char('s' | 'S') => NaturalizeAction::Save,
            KeyCode::Char('r' | 'R') => NaturalizeAction::Reload,
            _ => NaturalizeAction::None,
        }
    }

    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
            .split(area);

        // ── left: image preview ──
        let ptitle = if self.source.is_none() {
            " Preview ".to_string()
        } else if self.show_original {
            " Preview — ORIGINAL (Space: de-slopped) ".to_string()
        } else {
            " Preview — de-slopped (Space: original) ".to_string()
        };
        let block = Block::default().borders(Borders::ALL).title(ptitle);
        let inner = block.inner(cols[0]);
        f.render_widget(block, cols[0]);
        match &mut self.preview {
            Some(p) => f.render_stateful_widget(ratatui_image::StatefulImage::new(), inner, p),
            None => f.render_widget(
                Paragraph::new("\n  No image loaded.\n  Press r to load the newest image (a Chat generation or any file in the workspace out-dir),\n  or generate one in Chat (Ctrl-1).")
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
            .constraints([Constraint::Length(KNOBS.len() as u16 + 2), Constraint::Min(7), Constraint::Length(2)])
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

        // scorecard — before → after
        let bar = |x: f32| { let n = ((x * 12.0).round() as usize).min(12); format!("{}{}", "█".repeat(n), "░".repeat(12 - n)) };
        let mut sc: Vec<Line> = vec![Line::from(Span::styled("scorecard  (before → after)", Style::new().add_modifier(Modifier::BOLD)))];
        match (&self.orig_analysis, &self.analysis) {
            (Some(o), Some(a)) => {
                let dot = if a.ai_tell > 0.5 { Span::styled("● ", Style::new().fg(Color::Red)) } else { Span::styled("● ", Style::new().fg(Color::Green)) };
                let arrow = if a.ai_tell < o.ai_tell { Span::styled("↓", Style::new().fg(Color::Green)) } else { Span::styled("↑", Style::new().fg(Color::Red)) };
                sc.push(Line::from(vec![dot, Span::raw(format!("AI-tell   {} {:.2} → {:.2} ", bar(a.ai_tell), o.ai_tell, a.ai_tell)), arrow]));
                sc.push(Line::from(format!("  oversat  {} {:.2} → {:.2}", bar(a.saturation), o.saturation, a.saturation)));
                sc.push(Line::from(format!("  smooth   {} {:.2} → {:.2}", bar(a.smoothness_tell), o.smoothness_tell, a.smoothness_tell)));
                sc.push(Line::from(format!("  contrast {} {:.2} → {:.2}", bar(a.contrast), o.contrast, a.contrast)));
            }
            _ => sc.push(Line::from(Span::styled("  (load an image — press r)", Style::new().fg(Color::DarkGray)))),
        }
        f.render_widget(Paragraph::new(sc), rows[1]);

        // footer / status
        let foot = if self.status.is_empty() {
            "↑↓ select · ←→ / +- adjust · Space before/after · r reload · s save".to_string()
        } else {
            self.status.clone()
        };
        f.render_widget(Paragraph::new(Line::from(Span::styled(foot, Style::new().fg(Color::DarkGray)))), rows[2]);
    }
}
