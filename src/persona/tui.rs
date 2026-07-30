//! The composition TUI (RFC §17) — an interactive ratatui **view over the headless interview**
//! (`interview.rs`). Sections/progress header · the current question with its answer widget · a live
//! Tier-1 geometry wireframe (braille, redrawn every frame — "you *see* the eyes move", §17.5). The
//! engine is pure and lives elsewhere; this file is only the terminal shell + the widget input state.
//!
//! Gated behind the `ui` feature (ratatui/crossterm). Tier-2 diffusion preview + the place/list/tooth
//! widgets are P8c; this ships the structural/surface interview with the always-available wireframe.

use crate::persona::geometry;
use crate::persona::interview::{self, Answer, AnswerLog, Depth, Question};
use crate::persona::lexicon::Lexicon;
use crate::persona::PersonaSpec;
use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;
use std::path::PathBuf;
use std::time::Duration;

/// Per-question draft input (before the answer is committed).
struct Draft {
    scalar: f32,
    buffer: String,
    select: usize,
}

impl Draft {
    fn new(q: &Question, answers: &AnswerLog) -> Draft {
        // seed the draft from any prior answer to this question.
        let scalar = match answers.get(&q.path) {
            Some(Answer::Scalar(v)) => *v,
            _ => 0.5,
        };
        Draft { scalar, buffer: String::new(), select: 0 }
    }
}

struct Tui {
    lex: Lexicon,
    answers: AnswerLog,
    depth: Depth,
    name: String,
    current: Option<Question>,
    draft: Option<Draft>,
    status: String,
    finished: bool,
}

enum Flow {
    Continue,
    Save,
    Quit,
}

impl Tui {
    fn new(lex: Lexicon, depth: Depth, name: String) -> Tui {
        let mut t = Tui { lex, answers: AnswerLog::default(), depth, name, current: None, draft: None, status: "answer, or [u] unknown · [n] none · Enter next · Ctrl-S save · Ctrl-Q quit".into(), finished: false };
        t.advance();
        t
    }

    /// Move to the next question (or mark finished).
    fn advance(&mut self) {
        self.current = interview::next_question(&self.lex, &self.answers, self.depth);
        self.draft = self.current.as_ref().map(|q| Draft::new(q, &self.answers));
        self.finished = self.current.is_none();
    }

    /// Commit `answer` to the current question and advance.
    fn commit(&mut self, answer: Answer) {
        if let Some(q) = &self.current {
            interview::apply(&mut self.answers, &q.path, answer);
        }
        self.advance();
    }

    fn handle(&mut self, code: KeyCode, mods: KeyModifiers) -> Flow {
        if mods.contains(KeyModifiers::CONTROL) {
            match code {
                KeyCode::Char('s') => return Flow::Save,
                KeyCode::Char('q') | KeyCode::Char('c') => return Flow::Quit,
                _ => return Flow::Continue,
            }
        }
        let Some(q) = self.current.clone() else {
            // interview complete — save or quit only.
            if let KeyCode::Char('s') = code {
                return Flow::Save;
            }
            return Flow::Continue;
        };
        if self.draft.is_none() {
            return Flow::Continue;
        }
        let scalar = q.widget == "scalar";
        let has_opts = !q.options.is_empty();
        let buffer_empty = self.draft.as_ref().map(|d| d.buffer.is_empty()).unwrap_or(true);

        // draft-mutating keys take a short borrow; commit builds the answer then advances.
        match code {
            KeyCode::Esc => return Flow::Quit,
            KeyCode::Char('u') if !scalar && buffer_empty => {
                self.commit(Answer::Unknown);
                return Flow::Continue;
            }
            KeyCode::Char('n') if !scalar && buffer_empty => {
                self.commit(Answer::NoneEmpty);
                return Flow::Continue;
            }
            KeyCode::Enter => {
                let d = self.draft.as_ref().unwrap();
                let ans = if scalar {
                    Answer::Scalar(d.scalar)
                } else if has_opts {
                    Answer::Enum(q.options[d.select.min(q.options.len() - 1)].clone())
                } else if d.buffer.is_empty() {
                    Answer::Unknown
                } else {
                    match q.widget.as_str() {
                        "color" => Answer::Color(d.buffer.clone()),
                        "number" => d.buffer.parse::<f64>().map(Answer::Number).unwrap_or_else(|_| Answer::Text(d.buffer.clone())),
                        "text" => Answer::Text(d.buffer.clone()),
                        _ => Answer::Enum(d.buffer.clone()),
                    }
                };
                self.commit(ans);
                return Flow::Continue;
            }
            _ => {}
        }

        let opts_last = q.options.len().saturating_sub(1);
        if let Some(d) = self.draft.as_mut() {
            match code {
                KeyCode::Left if scalar => d.scalar = (d.scalar - 0.1).clamp(0.0, 1.0),
                KeyCode::Right if scalar => d.scalar = (d.scalar + 0.1).clamp(0.0, 1.0),
                KeyCode::Char('-') if scalar => d.scalar = (d.scalar - 0.02).clamp(0.0, 1.0),
                KeyCode::Char('+') | KeyCode::Char('=') if scalar => d.scalar = (d.scalar + 0.02).clamp(0.0, 1.0),
                KeyCode::Left if has_opts => d.select = d.select.saturating_sub(1),
                KeyCode::Right if has_opts => d.select = (d.select + 1).min(opts_last),
                KeyCode::Char(c) if !scalar && !has_opts => d.buffer.push(c),
                KeyCode::Backspace if !scalar && !has_opts => { d.buffer.pop(); }
                _ => {}
            }
        }
        Flow::Continue
    }

    /// Build the current partial spec, overriding the in-progress geometric scalar with the draft so
    /// the wireframe tracks the slider live (§17.5).
    fn preview_spec(&self) -> Option<PersonaSpec> {
        let json = serde_json::to_string(&interview::to_partial_spec(&self.answers)).ok()?;
        PersonaSpec::from_hjson(&json).ok()
    }

    fn wireframe(&self, cols: u32, rows: u32) -> Vec<String> {
        let spec = self.preview_spec();
        let mut values = spec.as_ref().map(geometry::geometry_values).unwrap_or_default();
        // live override for the scalar being edited.
        if let (Some(q), Some(d)) = (&self.current, &self.draft) {
            if q.widget == "scalar" && geometry::GEOMETRIC_ATTRS.contains(&q.path.as_str()) {
                values.insert(q.path.clone(), d.scalar);
            }
        }
        let open = spec.as_ref().map(geometry::open_mouth).unwrap_or(false);
        let d = geometry::resolve(&values, open, 0);
        crate::persona::preview::wireframe_braille(&d.landmarks, cols, rows)
    }

    fn render(&self, f: &mut Frame) {
        let area = f.area();
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(0), Constraint::Length(1)])
            .split(area);
        self.render_header(f, rows[0]);
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
            .split(rows[1]);
        self.render_question(f, cols[0]);
        self.render_preview(f, cols[1]);
        f.render_widget(Paragraph::new(Line::from(Span::styled(&self.status, Style::default().fg(Color::DarkGray)))), rows[2]);
    }

    fn render_header(&self, f: &mut Frame, area: Rect) {
        let (done, total) = interview::progress(&self.lex, &self.answers, self.depth);
        let depth = match self.depth {
            Depth::Quick => "quick",
            Depth::Standard => "standard",
            Depth::Full => "full",
        };
        let line = Line::from(vec![
            Span::styled(format!(" persona: {} ", self.name), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(format!("· {depth} · {done}/{total}"), Style::default().fg(Color::DarkGray)),
        ]);
        f.render_widget(Paragraph::new(line), area);
    }

    fn render_question(&self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" question ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        let mut lines: Vec<Line> = Vec::new();
        match (&self.current, &self.draft) {
            (Some(q), Some(d)) => {
                lines.push(Line::from(Span::styled(&q.ask, Style::default().add_modifier(Modifier::BOLD))));
                lines.push(Line::from(Span::styled(format!("[{}]", q.path), Style::default().fg(Color::DarkGray))));
                lines.push(Line::from(""));
                if q.widget == "scalar" {
                    let bar_w = (inner.width.saturating_sub(6)).max(4) as usize;
                    let fill = ((d.scalar * bar_w as f32).round() as usize).min(bar_w);
                    let bar: String = "█".repeat(fill) + &"░".repeat(bar_w - fill);
                    lines.push(Line::from(Span::styled(format!("{bar} {:.2}", d.scalar), Style::default().fg(Color::Cyan))));
                    lines.push(Line::from(Span::styled("←/→ coarse · +/- fine · Enter · [u] unknown", Style::default().fg(Color::DarkGray))));
                } else if !q.options.is_empty() {
                    let spans: Vec<Span> = q.options.iter().enumerate().flat_map(|(i, o)| {
                        let st = if i == d.select { Style::default().bg(Color::Cyan).fg(Color::Black) } else { Style::default().fg(Color::Gray) };
                        vec![Span::styled(format!(" {o} "), st), Span::raw(" ")]
                    }).collect();
                    lines.push(Line::from(spans));
                    lines.push(Line::from(Span::styled("←/→ choose · Enter · [u] unknown", Style::default().fg(Color::DarkGray))));
                } else {
                    lines.push(Line::from(vec![Span::raw("▏"), Span::styled(&d.buffer, Style::default().fg(Color::Cyan)), Span::styled("▌", Style::default().fg(Color::Cyan))]));
                    lines.push(Line::from(Span::styled("type · Enter · empty+Enter = unknown · [n] none", Style::default().fg(Color::DarkGray))));
                }
            }
            _ => {
                lines.push(Line::from(Span::styled("✓ interview complete", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))));
                lines.push(Line::from(""));
                lines.push(Line::from("Ctrl-S to save the spec, Ctrl-Q to discard."));
            }
        }
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
    }

    fn render_preview(&self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" preview (wireframe) ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        let lines: Vec<Line> = self
            .wireframe(inner.width.max(1) as u32, inner.height.max(1) as u32)
            .into_iter()
            .map(|s| Line::from(Span::styled(s, Style::default().fg(Color::Rgb(120, 200, 255)))))
            .collect();
        f.render_widget(Paragraph::new(lines), inner);
    }
}

/// Run the composition TUI, writing the spec to `out` on save (§17). Synchronous — owns the terminal;
/// call it from the (async) dispatcher exactly as `plakat ui` does.
pub fn run(out: PathBuf, depth: Depth, name: String) -> Result<()> {
    let mut tui = Tui::new(Lexicon::skeleton(), depth, name);
    let mut terminal = ratatui::init();
    let result = (|| -> Result<Option<serde_json::Value>> {
        loop {
            terminal.draw(|f| tui.render(f))?;
            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(k) = event::read()? {
                    if k.kind == event::KeyEventKind::Press {
                        match tui.handle(k.code, k.modifiers) {
                            Flow::Save => return Ok(Some(interview::to_partial_spec(&tui.answers))),
                            Flow::Quit => return Ok(None),
                            Flow::Continue => {}
                        }
                    }
                }
            }
        }
    })();
    ratatui::restore();

    match result? {
        Some(spec) => {
            let text = serde_json::to_string_pretty(&spec)?;
            std::fs::write(&out, &text).with_context(|| format!("writing {}", out.display()))?;
            let (done, total) = interview::progress(&tui.lex, &tui.answers, tui.depth);
            println!("✓  persona interview saved → {} ({done}/{total} answered)", out.display());
        }
        None => println!("· persona interview discarded (nothing written)"),
    }
    Ok(())
}
