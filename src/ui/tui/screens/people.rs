//! People screen (RFC TUI-1 §11, Release 3a). A Person is an identity that persists
//! across generation contexts, stored under `people/<name>/person.hjson` (+ refs/,
//! encoding/, portfolio/). This first increment is the LIBRARY + DETAIL browse: scan
//! the people dir, parse each `person.hjson`, list people with a coverage summary,
//! and show the selected person's refs / strategy / consent / stats with a lazy
//! preview of the primary reference photo. Quick-generate (`G`), import (`I`),
//! re-encode (`E`), and right-to-be-forgotten (`Del`) are follow-ups.

use std::path::{Path, PathBuf};

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use serde::Deserialize;

/// One weighted reference photo.
#[derive(Deserialize, Default, Clone)]
struct PersonRef {
    #[serde(default)]
    path: String,
    #[serde(default = "one")]
    weight: f32,
    #[serde(default)]
    angle: String,
    // Reserved for the REFS sub-tab's coverage analysis (deferred).
    #[serde(default)]
    #[allow(dead_code)]
    lighting: String,
    #[serde(default)]
    #[allow(dead_code)]
    notes: String,
}

fn one() -> f32 {
    1.0
}

#[derive(Deserialize, Default, Clone)]
struct Consent {
    #[serde(default)]
    granted_by: String,
    #[serde(default)]
    date: String,
    #[serde(default)]
    permitted_uses: Vec<String>,
    #[serde(default)]
    restrictions: Vec<String>,
    // Reserved for the SETTINGS sub-tab's privacy audit (deferred).
    #[serde(default)]
    #[allow(dead_code)]
    notes: String,
}

#[derive(Deserialize, Default, Clone)]
struct Stats {
    #[serde(default)]
    appearances: u32,
    #[serde(default)]
    scenarios: u32,
    #[serde(default)]
    sessions: u32,
    #[serde(default)]
    last_used: String,
    #[serde(default)]
    consistency: f32,
}

/// The `person.hjson` schema (all fields default-tolerant so hand-written / partial
/// files parse). `dir` and `error` are filled by the loader, not deserialized.
#[derive(Deserialize, Default, Clone)]
struct Person {
    #[serde(default)]
    name: String,
    #[serde(default)]
    display_name: String,
    #[serde(default)]
    identity: String,
    #[serde(default)]
    encoding_mode: String,
    #[serde(default)]
    face_strength: Option<f32>,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    negative: String,
    #[serde(default)]
    refs: Vec<PersonRef>,
    #[serde(default)]
    consent: Option<Consent>,
    #[serde(default)]
    stats: Stats,

    #[serde(skip)]
    dir: PathBuf,
    #[serde(skip)]
    error: Option<String>,
    /// Where this identity came from: "people" (a person dir) or "scenario · <file>".
    #[serde(skip)]
    source: String,
}

impl Person {
    /// Display name, falling back to the folder name.
    fn label(&self) -> &str {
        if !self.display_name.is_empty() {
            &self.display_name
        } else if !self.name.is_empty() {
            &self.name
        } else {
            self.dir.file_name().and_then(|n| n.to_str()).unwrap_or("?")
        }
    }

    /// The primary reference (highest weight, else first), resolved to a full path.
    fn primary_ref(&self) -> Option<PathBuf> {
        self.refs
            .iter()
            .filter(|r| !r.path.is_empty())
            .max_by(|a, b| a.weight.total_cmp(&b.weight))
            .map(|r| self.dir.join(&r.path))
    }

    /// All reference photos resolved to full paths with their weights.
    fn resolved_photos(&self) -> Vec<(PathBuf, f32)> {
        self.refs
            .iter()
            .filter(|r| !r.path.is_empty())
            .map(|r| (self.dir.join(&r.path), r.weight))
            .collect()
    }

    fn quick_gen_spec(&self) -> QuickGen {
        QuickGen {
            label: self.label().to_string(),
            prompt: self.prompt.clone(),
            negative: self.negative.clone(),
            photos: self.resolved_photos(),
            identity: self.identity.clone(),
            face_strength: self.face_strength,
        }
    }
}

/// A quick-generate request the App turns into a `portrait::Request`.
pub struct QuickGen {
    pub label: String,
    pub prompt: String,
    pub negative: String,
    /// Resolved reference photos `(path, weight)` (empty → text-only portrait).
    pub photos: Vec<(PathBuf, f32)>,
    /// Identity strategy string (empty → default when photos are present).
    pub identity: String,
    pub face_strength: Option<f32>,
}

/// What the App should do after a key.
pub enum PeopleAction {
    None,
    /// `G` with one person selected → portrait. Result opens in Chat.
    Generate(QuickGen),
    /// `G` with ≥2 marked (Space) → multiperson scene. Result opens in Chat.
    GenerateMulti(Vec<QuickGen>),
}

pub struct PeopleState {
    dir: PathBuf,
    scenarios_dir: PathBuf,
    people: Vec<Person>,
    selected: usize,
    /// Multi-selected rows (Space) for a multiperson generation.
    marked: std::collections::BTreeSet<usize>,
    pub preview: Option<ratatui_image::protocol::StatefulProtocol>,
    pub preview_for: Option<PathBuf>,
}

impl PeopleState {
    pub fn new(dir: PathBuf, scenarios_dir: PathBuf) -> Self {
        let mut s = Self {
            dir,
            scenarios_dir,
            people: Vec::new(),
            selected: 0,
            marked: std::collections::BTreeSet::new(),
            preview: None,
            preview_for: None,
        };
        s.rescan();
        s
    }

    /// Aggregate identities from the people dir AND every scenario's `personas:`
    /// block. People-dir entries win a name clash (they're the canonical, editable
    /// identity); scenario-only personas are shown read-only, tagged by source.
    pub fn rescan(&mut self) {
        let mut people = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        // 1. The people dir — canonical identities.
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            for e in rd.flatten() {
                let pdir = e.path();
                let hjson = pdir.join("person.hjson");
                if pdir.is_dir() && hjson.exists() {
                    let p = load_person(&pdir, &hjson);
                    seen.insert(p.label().to_lowercase());
                    if !p.name.is_empty() {
                        seen.insert(p.name.to_lowercase());
                    }
                    people.push(p);
                }
            }
        }

        // 2. Personas defined in scenario HJSON files (read-only).
        if let Ok(rd) = std::fs::read_dir(&self.scenarios_dir) {
            for e in rd.flatten() {
                let path = e.path();
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !name.ends_with(".hjson") || name.ends_with(".run.hjson") {
                    continue;
                }
                let Ok(personas) = crate::cli::scenario::peek_personas(&path) else { continue };
                for ps in personas {
                    if ps.name.is_empty() || !seen.insert(ps.name.to_lowercase()) {
                        continue; // already have this identity
                    }
                    people.push(from_persona(ps, format!("scenario · {name}")));
                }
            }
        }

        people.sort_by(|a, b| a.label().to_lowercase().cmp(&b.label().to_lowercase()));
        self.people = people;
        self.marked.clear(); // indices change on rescan
        if self.selected >= self.people.len() {
            self.selected = self.people.len().saturating_sub(1);
        }
    }

    /// The primary-ref path of the selected person (for the App to lazily preview).
    pub fn selected_ref(&self) -> Option<PathBuf> {
        self.people.get(self.selected).and_then(Person::primary_ref)
    }

    fn next(&mut self) {
        if !self.people.is_empty() {
            self.selected = (self.selected + 1) % self.people.len();
        }
    }

    fn prev(&mut self) {
        if !self.people.is_empty() {
            self.selected = (self.selected + self.people.len() - 1) % self.people.len();
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> PeopleAction {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.next(),
            KeyCode::Up | KeyCode::Char('k') => self.prev(),
            KeyCode::Char('r') => self.rescan(),
            // Space toggles multi-select (for a multiperson generation).
            KeyCode::Char(' ') => {
                if !self.people.is_empty() && !self.marked.remove(&self.selected) {
                    self.marked.insert(self.selected);
                }
            }
            // G generates: marked-set if any (≥2 → multiperson), else the cursor.
            KeyCode::Char('g' | 'G') | KeyCode::Enter => {
                let idxs: Vec<usize> = if self.marked.is_empty() {
                    vec![self.selected]
                } else {
                    self.marked.iter().copied().collect()
                };
                let specs: Vec<QuickGen> = idxs
                    .iter()
                    .filter_map(|&i| self.people.get(i))
                    .filter(|p| p.error.is_none())
                    .map(Person::quick_gen_spec)
                    .collect();
                self.marked.clear();
                if specs.len() >= 2 {
                    return PeopleAction::GenerateMulti(specs);
                }
                if let Some(one) = specs.into_iter().next() {
                    return PeopleAction::Generate(one);
                }
            }
            _ => {}
        }
        PeopleAction::None
    }

    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(34), Constraint::Percentage(66)])
            .split(area);
        self.render_library(f, cols[0]);

        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(cols[1]);
        self.render_preview(f, right[0]);
        self.render_detail(f, right[1]);
    }

    fn render_library(&self, f: &mut Frame, area: Rect) {
        let title = if self.marked.is_empty() {
            format!(" People ({})  ·  Space mark ", self.people.len())
        } else {
            format!(" People  ·  {} marked → [G] multiperson ", self.marked.len())
        };
        let block = Block::default().borders(Borders::ALL).title(title);
        let inner = block.inner(area);
        f.render_widget(block, area);
        if self.people.is_empty() {
            f.render_widget(
                Paragraph::new("No people yet. Add people/<name>/person.hjson.")
                    .style(Style::new().fg(Color::DarkGray))
                    .wrap(Wrap { trim: true }),
                inner,
            );
            return;
        }
        let mut lines: Vec<Line> = Vec::new();
        for (i, p) in self.people.iter().enumerate() {
            let sel = i == self.selected;
            let name_style = if sel {
                Style::new().bg(Color::Cyan).fg(Color::Black).add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(Color::White)
            };
            let summary = if p.error.is_some() {
                Span::styled("  ✗ unreadable".to_string(), Style::new().fg(Color::Red))
            } else {
                // ◆ = a canonical people-dir identity; ◇ = read-only scenario persona.
                let glyph = if p.source == "people" { "◆" } else { "◇" };
                Span::styled(
                    format!("  {glyph} {} ref{}", p.refs.len(), if p.refs.len() == 1 { "" } else { "s" }),
                    Style::new().fg(Color::DarkGray),
                )
            };
            let mark = if self.marked.contains(&i) {
                Span::styled("● ", Style::new().fg(Color::Cyan))
            } else {
                Span::raw("  ")
            };
            lines.push(Line::from(vec![mark, Span::styled(format!("{:<14}", p.label()), name_style), summary]));
        }
        f.render_widget(Paragraph::new(lines), inner);
    }

    fn render_preview(&mut self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" Primary reference ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        match &mut self.preview {
            Some(p) => f.render_stateful_widget(ratatui_image::StatefulImage::new(), inner, p),
            None => f.render_widget(
                Paragraph::new("\n  (no reference photo found)").style(Style::new().fg(Color::DarkGray)),
                inner,
            ),
        }
    }

    fn render_detail(&self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" Identity  ·  [G] generate → Chat ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        let Some(p) = self.people.get(self.selected) else { return };

        let mut lines: Vec<Line> = Vec::new();
        lines.push(Line::from(Span::styled(p.label().to_string(), Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD))));
        if let Some(err) = &p.error {
            lines.push(Line::from(Span::styled(format!("✗ {err}"), Style::new().fg(Color::Red))));
            f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
            return;
        }

        let kv = |k: &str, v: String| -> Line {
            Line::from(vec![
                Span::styled(format!("{k:<10}"), Style::new().fg(Color::DarkGray)),
                Span::styled(v, Style::new().fg(Color::White)),
            ])
        };
        lines.push(kv("source", or_dash(&p.source)));
        lines.push(kv("strategy", or_dash(&p.identity)));
        lines.push(kv("encoding", or_dash(&p.encoding_mode)));
        lines.push(kv("face-str", p.face_strength.map(|v| format!("{v:.2}")).unwrap_or_else(|| "—".into())));

        // Coverage summary across refs (RFC: actionable guidance).
        let angles: Vec<String> = {
            let mut a: Vec<String> = p.refs.iter().map(|r| r.angle.to_lowercase()).filter(|s| !s.is_empty()).collect();
            a.sort();
            a.dedup();
            a
        };
        lines.push(kv("angles", if angles.is_empty() { "—".into() } else { angles.join(", ") }));
        if !p.refs.is_empty() && !angles.iter().any(|a| a.contains("right") || a.contains("profile")) {
            lines.push(Line::from(Span::styled(
                "  ⚠ no right/profile ref → right-facing scenes may be less consistent",
                Style::new().fg(Color::Yellow),
            )));
        }

        if !p.prompt.is_empty() {
            lines.push(Line::from(""));
            lines.push(kv("prompt", p.prompt.clone()));
        }
        if !p.negative.is_empty() {
            lines.push(kv("negative", p.negative.clone()));
        }

        // Consent.
        lines.push(Line::from(""));
        match &p.consent {
            Some(c) if !c.granted_by.is_empty() => {
                lines.push(kv("consent", format!("✓ {} {}", c.granted_by, c.date)));
                if !c.permitted_uses.is_empty() {
                    lines.push(kv("uses", c.permitted_uses.join(", ")));
                }
                if !c.restrictions.is_empty() {
                    lines.push(kv("limits", c.restrictions.join(", ")));
                }
            }
            _ => lines.push(Line::from(Span::styled(
                "  ⚠ no consent block recorded",
                Style::new().fg(Color::Yellow),
            ))),
        }

        // Stats.
        let st = &p.stats;
        if st.appearances > 0 || !st.last_used.is_empty() {
            lines.push(Line::from(""));
            lines.push(kv(
                "used",
                format!("{}× · {} scenario(s) · {} session(s) · last {}", st.appearances, st.scenarios, st.sessions, or_dash(&st.last_used)),
            ));
            if st.consistency > 0.0 {
                lines.push(kv("consistency", format!("{:.2}", st.consistency)));
            }
        }

        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
    }
}

fn or_dash(s: &str) -> String {
    if s.is_empty() { "—".into() } else { s.to_string() }
}

/// Load one person, tolerating a malformed `person.hjson` (kept in the list with an
/// error flag so the folder is still visible).
fn load_person(dir: &Path, hjson: &Path) -> Person {
    let fallback = |error: String| Person {
        name: dir.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string(),
        dir: dir.to_path_buf(),
        error: Some(error),
        source: "people".into(),
        ..Default::default()
    };
    let text = match std::fs::read_to_string(hjson) {
        Ok(t) => t,
        Err(e) => return fallback(format!("read: {e}")),
    };
    match deser_hjson::from_str::<Person>(&text) {
        Ok(mut p) => {
            p.dir = dir.to_path_buf();
            p.source = "people".into();
            p
        }
        Err(e) => fallback(format!("parse: {e}")),
    }
}

/// Build a read-only Person from a scenario persona. Photo paths are already absolute
/// (resolved by `peek_personas`), so `dir` stays empty — `primary_ref`'s `join` on an
/// absolute path returns it unchanged.
fn from_persona(ps: crate::cli::scenario::PersonaSummary, source: String) -> Person {
    let refs = ps
        .photos
        .iter()
        .map(|(p, w)| PersonRef {
            path: p.to_string_lossy().into_owned(),
            weight: *w,
            ..Default::default()
        })
        .collect();
    Person {
        name: ps.name,
        identity: ps.identity.unwrap_or_default(),
        face_strength: ps.face_strength,
        refs,
        source,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("plakat-people-test-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn scans_people_and_picks_primary_ref() {
        let d = tmp("scan");
        let people = d.join("people");
        let scen = d.join("scenarios");
        std::fs::create_dir_all(&scen).unwrap();
        let alice = people.join("alice");
        std::fs::create_dir_all(alice.join("refs")).unwrap();
        std::fs::write(
            alice.join("person.hjson"),
            r#"{"display_name":"Alice","identity":"plus-face","refs":[{"path":"refs/a.jpg","weight":0.4,"angle":"front"},{"path":"refs/b.jpg","weight":0.9,"angle":"left"}]}"#,
        )
        .unwrap();
        // A folder without person.hjson is ignored.
        std::fs::create_dir_all(people.join("not-a-person")).unwrap();

        let s = PeopleState::new(people.clone(), scen);
        assert_eq!(s.people.len(), 1);
        assert_eq!(s.people[0].label(), "Alice");
        assert_eq!(s.people[0].source, "people");
        assert_eq!(s.people[0].refs.len(), 2);
        // Primary ref = highest weight (b.jpg @ 0.9).
        assert_eq!(s.selected_ref(), Some(alice.join("refs/b.jpg")));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn reads_personas_from_scenario_hjson_and_dedups() {
        let d = tmp("scenario-personas");
        let people = d.join("people");
        let scen = d.join("scenarios");
        std::fs::create_dir_all(&people).unwrap();
        std::fs::create_dir_all(&scen).unwrap();
        // alice exists in the people dir (canonical).
        let alice = people.join("alice");
        std::fs::create_dir_all(&alice).unwrap();
        std::fs::write(alice.join("person.hjson"), r#"{"display_name":"Alice"}"#).unwrap();
        // A scenario defines alice (dup → skipped) and bob (new → scenario-sourced).
        std::fs::write(
            scen.join("shoot.hjson"),
            r#"{"model":"sd15","personas":[{"name":"alice","photo":"a.jpg"},{"name":"bob","photo":"b.jpg","identity":"faceid"}],"tasks":[]}"#,
        )
        .unwrap();

        let s = PeopleState::new(people, scen);
        let names: Vec<&str> = s.people.iter().map(|p| p.label()).collect();
        assert_eq!(names, vec!["Alice", "bob"], "alice once (people-dir wins) + bob from scenario");
        let bob = s.people.iter().find(|p| p.name == "bob").unwrap();
        assert!(bob.source.starts_with("scenario · shoot.hjson"));
        assert_eq!(bob.identity, "faceid");
        assert_eq!(bob.refs.len(), 1);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn g_builds_a_quickgen_spec_from_the_selected_person() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let d = tmp("quickgen");
        let people = d.join("people");
        let alice = people.join("alice");
        std::fs::create_dir_all(alice.join("refs")).unwrap();
        std::fs::write(
            alice.join("person.hjson"),
            r#"{"display_name":"Alice","identity":"plus-face","face_strength":0.7,"prompt":"smiling","refs":[{"path":"refs/a.jpg","weight":1.0}]}"#,
        )
        .unwrap();
        let mut s = PeopleState::new(people, d.join("scenarios"));
        eprintln!("DBG people={} marked={:?} selected={}", s.people.len(), s.marked, s.selected);
        match s.handle_key(KeyEvent::new(KeyCode::Char('G'), KeyModifiers::NONE)) {
            PeopleAction::Generate(g) => {
                assert_eq!(g.label, "Alice");
                assert_eq!(g.prompt, "smiling");
                assert_eq!(g.identity, "plus-face");
                assert_eq!(g.face_strength, Some(0.7));
                assert_eq!(g.photos.len(), 1);
                assert!(g.photos[0].0.ends_with("refs/a.jpg"), "ref path resolved");
            }
            _ => panic!("expected Generate"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn space_marks_two_people_and_g_dispatches_multiperson() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let d = tmp("multi");
        let people = d.join("people");
        for name in ["alice", "bob"] {
            let pd = people.join(name);
            std::fs::create_dir_all(pd.join("refs")).unwrap();
            std::fs::write(
                pd.join("person.hjson"),
                format!(r#"{{"display_name":"{name}","refs":[{{"path":"refs/x.jpg","weight":1.0}}]}}"#),
            )
            .unwrap();
        }
        let mut s = PeopleState::new(people, d.join("scenarios"));
        let sp = || KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE);
        // Mark alice (index 0), move down, mark bob (index 1).
        s.handle_key(sp());
        s.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        s.handle_key(sp());
        match s.handle_key(KeyEvent::new(KeyCode::Char('G'), KeyModifiers::NONE)) {
            PeopleAction::GenerateMulti(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0].label, "alice");
                assert_eq!(v[1].label, "bob");
            }
            _ => panic!("expected GenerateMulti"),
        }
        // Marks cleared after dispatch.
        assert!(s.marked.is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn malformed_person_is_kept_with_error_flag() {
        let d = tmp("bad");
        let people = d.join("people");
        let bob = people.join("bob");
        std::fs::create_dir_all(&bob).unwrap();
        // Invalid: a quoteless value that swallows the brace (HJSON one-line trap).
        std::fs::write(bob.join("person.hjson"), "{ identity: a, b }").unwrap();
        let s = PeopleState::new(people, d.join("scenarios"));
        assert_eq!(s.people.len(), 1);
        assert_eq!(s.people[0].label(), "bob");
        assert!(s.people[0].error.is_some());
        let _ = std::fs::remove_dir_all(&d);
    }
}
