//! Chat screen (RFC TUI-1 §6) — the conversational generation interface: a text
//! input, the session history, the inline image, and live progress. The first
//! prompt generates (txt2img); once an image exists, a follow-up prompt refines it
//! (img2img over the previous output) unless it starts with `/new`. The App owns
//! the dispatch + refine decision; this screen renders state + handles input keys.

use std::path::PathBuf;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};

use super::prompt_editor::{EditorOutcome, PromptEditor};

/// One turn in the session: the user's utterance + (once generated) its result.
pub struct ChatEntry {
    pub utterance: String,
    pub result: Option<String>,
    pub error: Option<String>,
    /// This turn refined the previous image (img2img) rather than generating fresh.
    pub refine: bool,
    /// A system note (e.g. `/negative` feedback), not a generation turn.
    pub system: bool,
    /// The AI-enhanced prompt (`/enhance`), shown under the utterance.
    pub enhanced: Option<String>,
}

/// Live generation status for the status line.
#[derive(Clone, Default)]
pub enum ChatStatus {
    #[default]
    Idle,
    Generating {
        step: u32,
        total: u32,
        refine: bool,
    },
    Done(String),
    Error(String),
}

/// What the App should do after a key (so the screen stays dispatch-free).
#[derive(Debug)]
pub enum ChatAction {
    None,
    Submit(String),
    /// A `@mention` of a LoRA was accepted — apply it (App resolves the name → path).
    ApplyLora(String),
    /// Filmstrip navigation — show this frame in the image pane (`None` = back to live).
    SelectFrame(Option<PathBuf>),
    /// Roll the session back to this frame (branch): make it the live base, recovering
    /// its prompt + seed so the next prompt refines from there.
    Rollback(PathBuf),
    /// Generate a fresh variation of this frame (its prompt at a new random seed).
    Vary(PathBuf),
    /// 6.25.0 — "soft vary" (Ctrl-Shift-Y): subtle subseed-blended variations of the current
    /// frame (same composition, nudged) instead of fresh seeds. The App owns the subseed.
    SoftVary,
    /// Toggle the refine mode between prompt-evolve and image-anchored (Ctrl-T). The
    /// App owns `refine_strength`, so it performs the flip.
    ToggleAnchor,
}

/// A `@mention` completion candidate.
#[derive(Clone, PartialEq, Eq)]
pub enum MentionKind {
    Person,
    Lora,
}

pub struct ChatState {
    /// The 2-line, soft-wrapping prompt editor (full cursor editing + scrolling).
    pub editor: PromptEditor,
    pub history: Vec<ChatEntry>,
    pub status: ChatStatus,
    /// The latest preview / final image to show in the right pane (built by the
    /// App from `GenMessage` frames via the image Picker).
    pub preview: Option<ratatui_image::protocol::StatefulProtocol>,
    /// W4 — maximize the preview: Ctrl-F gives the whole middle row to the image (hides the
    /// transcript column) for a bigger look; Ctrl-F again restores the split.
    pub preview_full: bool,
    /// True once a previous image exists, so the next prompt refines it (img2img).
    /// Surfaced in the input hint; the App owns the actual refine decision.
    pub refine_armed: bool,
    /// Prompt-history recall cursor (Ctrl-P / Ctrl-N), shell-history style.
    recall: Option<usize>,
    /// `@mention` candidates (fed by the App each tick): people + local LoRA names.
    mention_people: Vec<String>,
    mention_loras: Vec<String>,
    /// Highlighted row in the mention popup.
    mention_sel: usize,
    /// The `@`-index the user dismissed (Esc); suppress the popup there until it moves.
    mention_dismissed_at: Option<usize>,
    /// Session filmstrip: the selected frame (index into `frames()`); `None` = live latest.
    strip_sel: Option<usize>,
    /// Linear undo/redo cursor: the live position in `frames()`; `None` = the newest frame.
    /// Undo (Ctrl-Z) steps it back, redo (Ctrl-Shift-Z) forward; a new generation or a
    /// filmstrip rollback resets it.
    live_idx: Option<usize>,
    /// What the next Enter will do — mode + strength + seed, fed by the App each tick
    /// (e.g. "evolve · seed 12345", "anchored 0.60 · seed 12345", "inpaint"). Shown in
    /// the Chat pane title so the loop's current behaviour is always visible.
    mode_hint: String,
}

impl ChatState {
    pub fn new() -> Self {
        Self {
            editor: PromptEditor::new(),
            history: Vec::new(),
            status: ChatStatus::Idle,
            preview: None,
            preview_full: false,
            refine_armed: false,
            recall: None,
            mention_people: Vec::new(),
            mention_loras: Vec::new(),
            mention_sel: 0,
            mention_dismissed_at: None,
            strip_sel: None,
            live_idx: None,
            mode_hint: String::new(),
        }
    }

    /// The session filmstrip: `(turn-number, result-path)` for every generated image,
    /// oldest first.
    pub fn frames(&self) -> Vec<(usize, PathBuf)> {
        self.history
            .iter()
            .enumerate()
            .filter(|(_, e)| !e.system && e.error.is_none())
            .filter_map(|(i, e)| e.result.as_ref().map(|p| (i + 1, PathBuf::from(p))))
            .collect()
    }

    /// Path of the most recent generated image (for "back to live").
    pub fn latest_frame_path(&self) -> Option<PathBuf> {
        self.frames().pop().map(|(_, p)| p)
    }

    /// Move the filmstrip cursor toward older frames (Ctrl-Left).
    fn strip_left(&mut self) -> ChatAction {
        let frames = self.frames();
        if frames.is_empty() {
            return ChatAction::None;
        }
        let new = match self.strip_sel {
            None => frames.len() - 1, // from live → select the latest
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.strip_sel = Some(new);
        ChatAction::SelectFrame(Some(frames[new].1.clone()))
    }

    /// Move the filmstrip cursor toward newer frames; past the newest → back to live.
    fn strip_right(&mut self) -> ChatAction {
        let frames = self.frames();
        match self.strip_sel {
            Some(i) if i + 1 < frames.len() => {
                self.strip_sel = Some(i + 1);
                ChatAction::SelectFrame(Some(frames[i + 1].1.clone()))
            }
            Some(_) => {
                self.strip_sel = None;
                ChatAction::SelectFrame(None)
            }
            None => ChatAction::None,
        }
    }

    fn selected_frame(&self) -> Option<PathBuf> {
        self.strip_sel.and_then(|i| self.frames().into_iter().nth(i).map(|(_, p)| p))
    }

    /// Roll back / branch from the selected frame (Ctrl-B).
    fn rollback(&mut self) -> ChatAction {
        match self.strip_sel {
            Some(i) => {
                let frames = self.frames();
                let Some((_, p)) = frames.into_iter().nth(i) else { return ChatAction::None };
                self.live_idx = Some(i); // the branched frame is now the live position
                self.strip_sel = None; // back to live after branching
                ChatAction::Rollback(p)
            }
            None => ChatAction::None,
        }
    }

    /// Undo (Ctrl-Z): step the live position one frame earlier and roll back to it, so the
    /// next prompt continues from the previous image. No-op at the oldest frame.
    fn undo(&mut self) -> ChatAction {
        let frames = self.frames();
        if frames.is_empty() {
            return ChatAction::None;
        }
        let cur = self.live_idx.unwrap_or(frames.len() - 1);
        if cur == 0 {
            return ChatAction::None; // nothing earlier
        }
        let target = cur - 1;
        self.live_idx = Some(target);
        self.strip_sel = None;
        ChatAction::Rollback(frames[target].1.clone())
    }

    /// Redo (Ctrl-Shift-Z): step the live position one frame later and roll back to it.
    /// No-op once back at the newest frame.
    fn redo(&mut self) -> ChatAction {
        let frames = self.frames();
        let Some(cur) = self.live_idx else {
            return ChatAction::None; // already at the newest
        };
        let target = cur + 1;
        if target >= frames.len() {
            return ChatAction::None;
        }
        // Reaching the newest frame returns the cursor to the "live latest" sentinel.
        self.live_idx = if target + 1 == frames.len() { None } else { Some(target) };
        self.strip_sel = None;
        ChatAction::Rollback(frames[target].1.clone())
    }

    /// New variation of the selected frame (Ctrl-Y).
    fn vary(&mut self) -> ChatAction {
        match self.selected_frame() {
            Some(p) => {
                self.strip_sel = None;
                ChatAction::Vary(p)
            }
            None => ChatAction::None,
        }
    }

    /// The App feeds the current `@mention` candidates (people + local LoRA names).
    pub fn set_mention_candidates(&mut self, people: Vec<String>, loras: Vec<String>) {
        self.mention_people = people;
        self.mention_loras = loras;
    }

    /// The App feeds the current refine-loop mode summary here each tick (shown in the
    /// pane title when idle).
    pub fn set_mode_hint(&mut self, hint: String) {
        self.mode_hint = hint;
    }

    /// The filtered `@mention` candidates for the active token (people first, then
    /// LoRAs), capped. Empty when no token is active / it was dismissed / nothing matches.
    pub fn mention_items(&self) -> Vec<(MentionKind, String)> {
        let Some((start, partial)) = self.editor.active_mention() else { return Vec::new() };
        if self.mention_dismissed_at == Some(start) {
            return Vec::new();
        }
        let q = partial.to_lowercase();
        let pick = |names: &[String], kind: MentionKind| -> Vec<(MentionKind, String)> {
            names
                .iter()
                .filter(|n| q.is_empty() || n.to_lowercase().contains(&q))
                .map(|n| (kind.clone(), n.clone()))
                .collect()
        };
        let mut items = pick(&self.mention_people, MentionKind::Person);
        items.extend(pick(&self.mention_loras, MentionKind::Lora));
        items.truncate(8);
        items
    }

    fn mention_open(&self) -> bool {
        !self.mention_items().is_empty()
    }

    /// Accept the highlighted mention: a person becomes a readable `@name` token
    /// (expanded at submit); a LoRA is stripped from the text and applied via the App.
    fn accept_mention(&mut self) -> ChatAction {
        let Some((start, _)) = self.editor.active_mention() else { return ChatAction::None };
        let items = self.mention_items();
        let Some((kind, name)) = items.get(self.mention_sel).cloned() else { return ChatAction::None };
        match kind {
            MentionKind::Person => {
                self.editor.replace_mention(start, &format!("@{name} "));
                ChatAction::None
            }
            MentionKind::Lora => {
                self.editor.replace_mention(start, "");
                ChatAction::ApplyLora(name)
            }
        }
    }

    /// Handle a key while the Chat input is focused. Ctrl-P / Ctrl-N recall previous
    /// prompts into the editor; everything else goes to the editor; Enter submits.
    pub fn handle_key(&mut self, key: KeyEvent) -> ChatAction {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        // ── `@mention` completion popup owns a few keys while it's showing. ──
        if self.mention_open() && !ctrl {
            match key.code {
                KeyCode::Up => {
                    self.mention_sel = self.mention_sel.saturating_sub(1);
                    return ChatAction::None;
                }
                KeyCode::Down => {
                    self.mention_sel = (self.mention_sel + 1).min(self.mention_items().len().saturating_sub(1));
                    return ChatAction::None;
                }
                KeyCode::Tab | KeyCode::Enter => return self.accept_mention(),
                KeyCode::Esc => {
                    self.mention_dismissed_at = self.editor.active_mention().map(|(s, _)| s);
                    return ChatAction::None;
                }
                _ => {}
            }
        }
        match key.code {
            // ── Session filmstrip: navigate / rollback / vary. ──
            KeyCode::Left if ctrl => return self.strip_left(),
            KeyCode::Right if ctrl => return self.strip_right(),
            KeyCode::Char('b') if ctrl => return self.rollback(),
            // Ctrl-Shift-Y before Ctrl-Y so the shift combo isn't swallowed by the vary arm.
            KeyCode::Char('y') if ctrl && shift => return ChatAction::SoftVary,
            KeyCode::Char('y') if ctrl => return self.vary(),
            // Linear undo/redo over the frame history. Shift-first so Ctrl-Shift-Z (redo)
            // isn't swallowed by the Ctrl-Z (undo) arm.
            KeyCode::Char('z') if ctrl && shift => return self.redo(),
            KeyCode::Char('z') if ctrl => return self.undo(),
            KeyCode::Char('t') if ctrl => return ChatAction::ToggleAnchor,
            // Ctrl-F — maximize the preview (hide the transcript column), Ctrl-F again restores.
            KeyCode::Char('f') if ctrl => {
                self.preview_full = !self.preview_full;
                return ChatAction::None;
            }
            KeyCode::Char('p') if ctrl => {
                self.recall_prev();
                return ChatAction::None;
            }
            KeyCode::Char('n') if ctrl => {
                self.recall_next();
                return ChatAction::None;
            }
            _ => {}
        }
        let outcome = self.editor.handle_key(key);
        // Typing moves/changes the token → reset the popup selection + un-dismiss.
        if matches!(key.code, KeyCode::Char(_) | KeyCode::Backspace) {
            self.mention_sel = 0;
            self.mention_dismissed_at = None;
        }
        match outcome {
            EditorOutcome::Submit => {
                self.recall = None;
                let text = self.editor.take();
                if !text.is_empty() {
                    return ChatAction::Submit(text);
                }
            }
            EditorOutcome::Consumed => {}
        }
        ChatAction::None
    }

    /// Recall an earlier prompt into the editor (Ctrl-P): walks backward through the
    /// session's utterances. "Copy previous chat to prompt."
    fn recall_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let idx = match self.recall {
            None => self.history.len() - 1,
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.recall = Some(idx);
        let text = self.history[idx].utterance.clone();
        self.editor.set_text(&text);
    }

    /// Walk forward through recalled prompts (Ctrl-N); past the newest clears to a
    /// fresh empty input.
    fn recall_next(&mut self) {
        match self.recall {
            Some(i) if i + 1 < self.history.len() => {
                self.recall = Some(i + 1);
                let text = self.history[i + 1].utterance.clone();
                self.editor.set_text(&text);
            }
            Some(_) => {
                self.recall = None;
                self.editor.clear();
            }
            None => {}
        }
    }

    /// Record a submitted utterance (the App calls this when dispatching it).
    pub fn push_utterance(&mut self, utterance: String, refine: bool) {
        self.strip_sel = None; // a new turn returns the filmstrip to live
        self.live_idx = None; // …and the new frame becomes the live undo position
        self.history.push(ChatEntry {
            utterance,
            result: None,
            error: None,
            refine,
            system: false,
            enhanced: None,
        });
    }

    /// Replace the visible history from a saved session: each turn is
    /// `(utterance, result, refine, system)`. (Keeps `ChatEntry` construction here.)
    pub fn restore(&mut self, turns: Vec<(String, Option<String>, bool, bool)>) {
        self.history = turns
            .into_iter()
            .map(|(utterance, result, refine, system)| ChatEntry { utterance, result, error: None, refine, system, enhanced: None })
            .collect();
    }

    /// Append a system note (command feedback) to the history.
    pub fn push_system(&mut self, note: String) {
        self.history.push(ChatEntry {
            utterance: note,
            result: None,
            error: None,
            refine: false,
            system: true,
            enhanced: None,
        });
    }

    /// Record the AI-enhanced prompt on the most recent generation turn.
    pub fn set_last_enhanced(&mut self, prompt: String) {
        if let Some(last) = self.history.iter_mut().rev().find(|e| !e.system) {
            last.enhanced = Some(prompt);
        }
    }

    /// Mark the most recent entry done / failed (called when a generation finishes).
    pub fn finish_last(&mut self, result: Result<String, String>) {
        if let Some(last) = self.history.last_mut() {
            match result {
                Ok(path) => last.result = Some(path),
                Err(e) => last.error = Some(e),
            }
        }
    }

    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            // [memory gauges] [history+image] [2-row wrapping prompt editor + borders].
            .constraints([Constraint::Length(3), Constraint::Min(1), Constraint::Length(4)])
            .split(area);
        crate::ui::tui::memory::render_memory_bar(f, rows[0]);
        // Middle: chat history on the left, the generated image on the right. W4: Ctrl-F
        // maximizes the image to the whole middle row (the transcript column is hidden).
        let img_area = if self.preview_full {
            rows[1]
        } else {
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
                .split(rows[1]);
            self.render_history(f, cols[0]);
            cols[1]
        };
        // The image pane gives up its bottom rows to the session filmstrip when there
        // is more than one frame to scrub through.
        if self.frames().len() > 1 {
            let img_rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(2)])
                .split(img_area);
            self.render_image(f, img_rows[0]);
            self.render_filmstrip(f, img_rows[1]);
        } else {
            self.render_image(f, img_area);
        }
        self.render_input(f, rows[2]);
        // The @mention popup floats just above the input, over the history column.
        self.render_mention_popup(f, rows[1], rows[2]);
    }

    /// A one-line scrubber of every generated frame this session; the shown frame is
    /// highlighted (`live` when following the latest). Ctrl-←/→ navigate, Ctrl-B rolls
    /// back to the selected frame, Ctrl-Y makes a variation.
    fn render_filmstrip(&self, f: &mut Frame, area: Rect) {
        let frames = self.frames();
        let live = self.strip_sel.is_none();
        let mut spans: Vec<Span> = vec![Span::styled(
            "film ",
            Style::new().fg(Color::DarkGray),
        )];
        for (vi, (turn, _)) in frames.iter().enumerate() {
            let selected = self.strip_sel == Some(vi) || (live && vi + 1 == frames.len());
            let style = if selected {
                Style::new().bg(Color::Magenta).fg(Color::Black).add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(Color::Gray)
            };
            spans.push(Span::styled(format!(" {turn} "), style));
            spans.push(Span::raw(" "));
        }
        let hint = if live {
            "  Ctrl-←/→ scrub"
        } else {
            "  Ctrl-←/→ scrub · Ctrl-B rollback · Ctrl-Y vary"
        };
        spans.push(Span::styled(hint, Style::new().fg(Color::DarkGray)));
        f.render_widget(
            Paragraph::new(Line::from(spans)).block(Block::default().borders(Borders::TOP)),
            area,
        );
    }

    /// Draw the `@mention` completion popup anchored above the input (when active).
    fn render_mention_popup(&self, f: &mut Frame, content: Rect, input: Rect) {
        let items = self.mention_items();
        if items.is_empty() {
            return;
        }
        let h = (items.len() as u16 + 2).min(content.height.max(3));
        let w = 38.min(content.width.max(10));
        let x = content.x + 2;
        let y = input.y.saturating_sub(h);
        let area = Rect { x, y, width: w, height: h };
        f.render_widget(Clear, area);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::new().fg(Color::Magenta))
            .title(" @mention · ↑↓ · Tab/Enter · Esc ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        let mut lines: Vec<Line> = Vec::new();
        for (i, (kind, name)) in items.iter().enumerate() {
            let (glyph, gc) = match kind {
                MentionKind::Person => ("◆", Color::Cyan),
                MentionKind::Lora => ("★", Color::Yellow),
            };
            let style = if i == self.mention_sel {
                Style::new().bg(Color::Magenta).fg(Color::Black).add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(Color::Gray)
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{glyph} "), Style::new().fg(gc)),
                Span::styled(name.clone(), style),
            ]));
        }
        f.render_widget(Paragraph::new(lines), inner);
    }

    fn render_image(&mut self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" Image ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        match &mut self.preview {
            Some(protocol) => {
                // ratatui-image renders the decoded image inline via the terminal's
                // graphics protocol (Kitty/iTerm2/Sixel).
                f.render_stateful_widget(ratatui_image::StatefulImage::new(), inner, protocol);
            }
            None => {
                let body = "\n  The generated image will appear here.\n\n  Type a prompt on the left and press Enter.";
                f.render_widget(Paragraph::new(body).wrap(Wrap { trim: false }), inner);
            }
        }
    }

    fn render_history(&self, f: &mut Frame, area: Rect) {
        let inner_w = area.width.saturating_sub(2) as usize;
        let mut lines: Vec<Line> = Vec::new();
        for (i, e) in self.history.iter().enumerate() {
            if e.system {
                // A command-feedback note (e.g. /negative), dimmed, no turn number.
                wrap_entry(&mut lines, "   ⚙ ", &e.utterance, Color::DarkGray, Color::DarkGray, inner_w);
                continue;
            }
            // ↻ marks a refinement of the previous image; ▸ a fresh generation.
            let (glyph, gcolor) = if e.refine { ("↻", Color::Magenta) } else { ("▸", Color::Cyan) };
            wrap_entry(&mut lines, &format!("{:>2} {glyph} ", i + 1), &e.utterance, gcolor, Color::White, inner_w);
            if let Some(enh) = &e.enhanced {
                wrap_entry(&mut lines, "      ✨ ", enh, Color::Yellow, Color::Gray, inner_w);
            }
            if let Some(path) = &e.result {
                wrap_entry(&mut lines, "      → ", path, Color::Green, Color::Green, inner_w);
            }
            if let Some(err) = &e.error {
                wrap_entry(&mut lines, "      ✗ ", err, Color::Red, Color::Red, inner_w);
            }
        }
        if lines.is_empty() {
            lines.push(Line::from(Span::styled(
                "Describe an image and press Enter. (Load a model first in Models / Ctrl-2.)",
                Style::new().fg(Color::DarkGray),
            )));
            lines.push(Line::from(Span::styled(
                "Type to evolve · /new /enhance /negative /seed /strength <0.1-1|off> /auto <on|off>",
                Style::new().fg(Color::DarkGray),
            )));
        }
        // Keep the tail visible (latest messages) within the pane height.
        let rows = area.height.saturating_sub(2) as usize;
        let start = lines.len().saturating_sub(rows.max(1));
        let visible: Vec<Line> = lines.split_off(start);

        let title = match &self.status {
            ChatStatus::Generating { step, total, refine } => {
                let verb = if *refine { "refining" } else { "generating" };
                format!(" Chat  ⟳ {verb} {step}/{total} ")
            }
            ChatStatus::Error(e) => format!(" Chat  ✗ {e} "),
            _ if !self.mode_hint.is_empty() => format!(" Chat · {} ", self.mode_hint),
            _ => " Chat ".to_string(),
        };
        f.render_widget(
            Paragraph::new(visible).block(Block::default().borders(Borders::ALL).title(title)),
            area,
        );
    }

    fn render_input(&mut self, f: &mut Frame, area: Rect) {
        // Once an image exists, the next prompt refines it; advertise that + the
        // /new escape hatch. (The editor wraps + scrolls within the 2-row box.)
        let base = if self.refine_armed {
            " prompt · Enter refine · Ctrl-T evolve/anchor · /new fresh · Ctrl-P/N recall "
        } else {
            " prompt · Enter generate · Ctrl-P/N recall "
        };
        // 6.25.0: advertise prompt scheduling the moment the prompt uses `[a:b:f]` / `[a|b]`.
        let text = self.editor.text().to_string();
        let title: std::borrow::Cow<str> = if crate::prompt::scheduling::has_schedule(&text) {
            std::borrow::Cow::Owned(format!("{base}· ◆ scheduled "))
        } else {
            std::borrow::Cow::Borrowed(base)
        };
        self.editor.render(f, area, &title);
    }
}

/// Word-wrap `text` to `width` columns (char-based; hard-splits a word longer than
/// `width`, e.g. a long path). Always returns at least one line.
fn wrap_to(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_len = 0usize;
    let add = |lines: &mut Vec<String>, cur: &mut String, cur_len: &mut usize, piece: &str| {
        let pl = piece.chars().count();
        if *cur_len == 0 {
            cur.push_str(piece);
            *cur_len = pl;
        } else if *cur_len + 1 + pl <= width {
            cur.push(' ');
            cur.push_str(piece);
            *cur_len += 1 + pl;
        } else {
            lines.push(std::mem::take(cur));
            cur.push_str(piece);
            *cur_len = pl;
        }
    };
    for word in text.split_whitespace() {
        if word.chars().count() <= width {
            add(&mut lines, &mut cur, &mut cur_len, word);
        } else {
            let chars: Vec<char> = word.chars().collect();
            for chunk in chars.chunks(width) {
                let piece: String = chunk.iter().collect();
                add(&mut lines, &mut cur, &mut cur_len, &piece);
            }
        }
    }
    if !cur.is_empty() || lines.is_empty() {
        lines.push(cur);
    }
    lines
}

/// Push a history entry (a `prefix` + `text`) into `lines`, wrapped to `width`.
/// The first line shows the coloured prefix; continuation lines are indented to
/// align under the text.
fn wrap_entry(
    lines: &mut Vec<Line<'static>>,
    prefix: &str,
    text: &str,
    prefix_color: Color,
    text_color: Color,
    width: usize,
) {
    let plen = prefix.chars().count();
    let avail = width.saturating_sub(plen).max(1);
    let indent = " ".repeat(plen);
    for (k, w) in wrap_to(text, avail).into_iter().enumerate() {
        if k == 0 {
            lines.push(Line::from(vec![
                Span::styled(prefix.to_string(), Style::new().fg(prefix_color)),
                Span::styled(w, Style::new().fg(text_color)),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::raw(indent.clone()),
                Span::styled(w, Style::new().fg(text_color)),
            ]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;

    fn k(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    #[test]
    fn typing_builds_the_input() {
        let mut s = ChatState::new();
        for c in "cat".chars() {
            assert!(matches!(s.handle_key(k(KeyCode::Char(c))), ChatAction::None));
        }
        assert_eq!(s.editor.text(), "cat");
        s.handle_key(k(KeyCode::Backspace));
        assert_eq!(s.editor.text(), "ca");
    }

    #[test]
    fn ctrl_f_toggles_preview_maximize() {
        let mut s = ChatState::new();
        assert!(!s.preview_full);
        assert!(matches!(s.handle_key(ctrl('f')), ChatAction::None));
        assert!(s.preview_full, "Ctrl-F maximizes the preview");
        s.handle_key(ctrl('f'));
        assert!(!s.preview_full, "Ctrl-F again restores the split");
    }

    #[test]
    fn enter_submits_and_clears() {
        let mut s = ChatState::new();
        for c in "a fox".chars() {
            s.handle_key(k(KeyCode::Char(c)));
        }
        match s.handle_key(k(KeyCode::Enter)) {
            ChatAction::Submit(p) => assert_eq!(p, "a fox"),
            _ => panic!("expected submit"),
        }
        assert!(s.editor.is_empty());
    }

    fn type_str(s: &mut ChatState, text: &str) {
        for c in text.chars() {
            s.handle_key(k(KeyCode::Char(c)));
        }
    }

    #[test]
    fn at_mention_popup_filters_people_and_loras() {
        let mut s = ChatState::new();
        s.set_mention_candidates(vec!["Alice".into(), "Bob".into()], vec!["watercolor".into()]);
        // No popup until an @token is typed.
        assert!(s.mention_items().is_empty());
        type_str(&mut s, "a portrait of @al");
        let items = s.mention_items();
        assert_eq!(items.len(), 1, "‘al’ matches Alice only");
        assert_eq!(items[0].1, "Alice");
        assert!(matches!(items[0].0, MentionKind::Person));
    }

    #[test]
    fn accepting_a_person_mention_keeps_a_readable_token() {
        let mut s = ChatState::new();
        s.set_mention_candidates(vec!["Alice".into()], vec![]);
        type_str(&mut s, "@al");
        // Tab accepts → "@Alice " stays in the text (expanded later, by the App).
        assert!(matches!(s.handle_key(k(KeyCode::Tab)), ChatAction::None));
        assert_eq!(s.editor.text(), "@Alice ");
    }

    #[test]
    fn accepting_a_lora_mention_strips_the_token_and_applies() {
        let mut s = ChatState::new();
        s.set_mention_candidates(vec![], vec!["watercolor".into()]);
        type_str(&mut s, "sunset @water");
        match s.handle_key(k(KeyCode::Enter)) {
            ChatAction::ApplyLora(name) => assert_eq!(name, "watercolor"),
            _ => panic!("expected ApplyLora"),
        }
        // The token is removed from the prompt (the LoRA is applied, not described).
        assert_eq!(s.editor.text(), "sunset ");
    }

    #[test]
    fn enter_with_no_mention_open_still_submits() {
        let mut s = ChatState::new();
        s.set_mention_candidates(vec!["Alice".into()], vec![]);
        type_str(&mut s, "a fox");
        match s.handle_key(k(KeyCode::Enter)) {
            ChatAction::Submit(p) => assert_eq!(p, "a fox"),
            _ => panic!("expected submit when no popup"),
        }
    }

    #[test]
    fn esc_dismisses_the_mention_popup() {
        let mut s = ChatState::new();
        s.set_mention_candidates(vec!["Alice".into()], vec![]);
        type_str(&mut s, "@al");
        assert!(!s.mention_items().is_empty());
        s.handle_key(k(KeyCode::Esc));
        assert!(s.mention_items().is_empty(), "Esc hides the popup for this token");
        // Typing more re-opens it.
        type_str(&mut s, "i");
        assert!(!s.mention_items().is_empty());
    }

    fn with_two_frames() -> ChatState {
        let mut s = ChatState::new();
        s.push_utterance("a fox".into(), false);
        s.finish_last(Ok("/out/plakat-1-1.png".into()));
        s.push_utterance("make it autumn".into(), true);
        s.finish_last(Ok("/out/plakat-1-2.png".into()));
        s
    }

    fn ctrl_code(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::CONTROL)
    }

    #[test]
    fn filmstrip_collects_generated_frames() {
        let s = with_two_frames();
        let frames = s.frames();
        assert_eq!(frames.len(), 2);
        assert!(frames[0].1.ends_with("plakat-1-1.png"));
        assert_eq!(s.latest_frame_path().unwrap().file_name().unwrap(), "plakat-1-2.png");
    }

    #[test]
    fn ctrl_left_right_scrub_the_filmstrip() {
        let mut s = with_two_frames();
        // From live, Ctrl-Left selects the latest frame (index 1).
        match s.handle_key(ctrl_code(KeyCode::Left)) {
            ChatAction::SelectFrame(Some(p)) => assert!(p.ends_with("plakat-1-2.png")),
            _ => panic!("expected SelectFrame"),
        }
        // Ctrl-Left again → the older frame.
        match s.handle_key(ctrl_code(KeyCode::Left)) {
            ChatAction::SelectFrame(Some(p)) => assert!(p.ends_with("plakat-1-1.png")),
            _ => panic!("expected older frame"),
        }
        // Ctrl-Right → newer, then past-newest → back to live (None).
        s.handle_key(ctrl_code(KeyCode::Right));
        match s.handle_key(ctrl_code(KeyCode::Right)) {
            ChatAction::SelectFrame(None) => {}
            _ => panic!("expected back-to-live"),
        }
    }

    #[test]
    fn ctrl_z_undo_and_redo_walk_the_frame_history() {
        let mut s = ChatState::new();
        for i in 1..=3 {
            s.push_utterance(format!("turn {i}"), i > 1);
            s.finish_last(Ok(format!("/out/plakat-1-{i}.png")));
        }
        let z = |shift: bool| {
            let m = if shift { KeyModifiers::CONTROL | KeyModifiers::SHIFT } else { KeyModifiers::CONTROL };
            KeyEvent::new(KeyCode::Char('z'), m)
        };
        let rolled = |a: ChatAction| match a {
            ChatAction::Rollback(p) => p.file_name().unwrap().to_string_lossy().into_owned(),
            other => panic!("expected Rollback, got {other:?}"),
        };
        // Undo steps back: latest(3) → 2 → 1, then no-op at the oldest.
        assert_eq!(rolled(s.handle_key(z(false))), "plakat-1-2.png");
        assert_eq!(rolled(s.handle_key(z(false))), "plakat-1-1.png");
        assert!(matches!(s.handle_key(z(false)), ChatAction::None), "no undo past the oldest");
        // Redo steps forward: 1 → 2 → 3, then no-op at the newest.
        assert_eq!(rolled(s.handle_key(z(true))), "plakat-1-2.png");
        assert_eq!(rolled(s.handle_key(z(true))), "plakat-1-3.png");
        assert!(matches!(s.handle_key(z(true)), ChatAction::None), "no redo past the newest");
        // A new generation resets the cursor: undo now steps from the new newest.
        s.push_utterance("turn 4".into(), true);
        s.finish_last(Ok("/out/plakat-1-4.png".into()));
        assert_eq!(rolled(s.handle_key(z(false))), "plakat-1-3.png");
    }

    #[test]
    fn ctrl_b_rolls_back_and_ctrl_y_varies_the_selected_frame() {
        let mut s = with_two_frames();
        // Select the first frame.
        s.handle_key(ctrl_code(KeyCode::Left)); // latest
        s.handle_key(ctrl_code(KeyCode::Left)); // older (index 0)
        match s.handle_key(ctrl_code(KeyCode::Char('b'))) {
            ChatAction::Rollback(p) => assert!(p.ends_with("plakat-1-1.png")),
            _ => panic!("expected Rollback"),
        }
        // Rollback returns the strip to live.
        assert!(s.strip_sel.is_none());

        // Re-select and vary.
        s.handle_key(ctrl_code(KeyCode::Left));
        match s.handle_key(ctrl_code(KeyCode::Char('y'))) {
            ChatAction::Vary(_) => {}
            _ => panic!("expected Vary"),
        }
    }

    #[test]
    fn rollback_without_a_selection_is_a_noop() {
        let mut s = with_two_frames();
        assert!(matches!(s.handle_key(ctrl_code(KeyCode::Char('b'))), ChatAction::None));
    }

    #[test]
    fn empty_enter_is_noop() {
        let mut s = ChatState::new();
        assert!(matches!(s.handle_key(k(KeyCode::Enter)), ChatAction::None));
    }

    #[test]
    fn ctrl_p_recalls_previous_prompts_into_the_editor() {
        let mut s = ChatState::new();
        s.push_utterance("a fox".into(), false);
        s.push_utterance("a wolf".into(), false);
        // Ctrl-P walks back: newest first.
        s.handle_key(ctrl('p'));
        assert_eq!(s.editor.text(), "a wolf");
        s.handle_key(ctrl('p'));
        assert_eq!(s.editor.text(), "a fox");
        // Ctrl-N walks forward; past the newest clears.
        s.handle_key(ctrl('n'));
        assert_eq!(s.editor.text(), "a wolf");
        s.handle_key(ctrl('n'));
        assert!(s.editor.is_empty());
    }

    #[test]
    fn wrap_to_wraps_at_width_and_preserves_words() {
        let out = wrap_to("the quick brown fox jumps", 9);
        assert!(out.iter().all(|l| l.chars().count() <= 9), "no line exceeds width");
        assert_eq!(out.join(" "), "the quick brown fox jumps", "words preserved in order");
        assert!(out.len() >= 3);
        // a long, space-less token (e.g. a path) is hard-split, not truncated.
        let p = wrap_to("/a/very/long/path/with/no/spaces/at/all.png", 10);
        assert!(p.iter().all(|l| l.chars().count() <= 10));
        assert_eq!(p.concat(), "/a/very/long/path/with/no/spaces/at/all.png");
    }

    #[test]
    fn history_records_utterance_and_result() {
        let mut s = ChatState::new();
        s.push_utterance("a fox".into(), false);
        s.finish_last(Ok("out/chat/plakat-42.png".into()));
        assert_eq!(s.history.len(), 1);
        assert_eq!(s.history[0].result.as_deref(), Some("out/chat/plakat-42.png"));
        assert!(!s.history[0].refine);
    }

    #[test]
    fn refine_turn_is_flagged() {
        let mut s = ChatState::new();
        s.push_utterance("make it warmer".into(), true);
        assert!(s.history[0].refine, "a refinement turn records refine=true");
    }
}
