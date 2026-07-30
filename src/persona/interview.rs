//! The headless persona interview (RFC §17.2/§17.3). **The interview engine is headless; the TUI is a
//! view over it.** Pure functions over data — no terminal, no I/O — so the flow is unit-testable, a
//! fixed answer sequence replays to a byte-stable spec, and the same interview can drive a terminal or
//! any future surface. The question graph is *generated from the lexicon* (§17.3), never hand-written
//! in Rust, so the lexicon and the interview can never drift apart.

use crate::persona::lexicon::Lexicon;
use serde_json::{Map, Value};

/// Interview depth (§17.6): quick (~structural), standard (+surface), full (every leaf).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Depth {
    Quick,
    Standard,
    Full,
}

impl Depth {
    pub fn tier(self) -> u8 {
        match self {
            Depth::Quick => 0,
            Depth::Standard => 1,
            Depth::Full => 2,
        }
    }
    pub fn parse(s: &str) -> Depth {
        match s {
            "quick" => Depth::Quick,
            "full" => Depth::Full,
            _ => Depth::Standard,
        }
    }
}

/// One generated question (§17.3).
#[derive(Debug, Clone)]
pub struct Question {
    pub path: String,
    pub ask: String,
    pub widget: String,
    pub section: String,
    /// Enum options (for `select`/`multi` widgets); empty otherwise.
    pub options: Vec<String>,
    pub tier: u8,
}

/// An answer to a question (§17.4). `Unknown` is distinct from a middle value (§6.4); `NoneEmpty`
/// asserts an empty collection.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    Scalar(f32),
    Enum(String),
    Color(String),
    Text(String),
    Number(f64),
    Unknown,
    NoneEmpty,
}

/// The ordered log of answers (§17.2). Replay is exact.
#[derive(Debug, Clone, Default)]
pub struct AnswerLog {
    pub entries: Vec<(String, Answer)>,
}

impl AnswerLog {
    pub fn get(&self, path: &str) -> Option<&Answer> {
        self.entries.iter().rev().find(|(p, _)| p == path).map(|(_, a)| a)
    }
    pub fn answered(&self, path: &str) -> bool {
        self.entries.iter().any(|(p, _)| p == path)
    }
    /// Answer as a string for condition evaluation (`Unknown` → absent).
    fn as_str(&self, path: &str) -> Option<String> {
        match self.get(path)? {
            Answer::Enum(s) | Answer::Color(s) | Answer::Text(s) => Some(s.clone()),
            Answer::Scalar(v) => Some(v.to_string()),
            Answer::Number(v) => Some(v.to_string()),
            Answer::NoneEmpty => Some("none".into()),
            Answer::Unknown => None,
        }
    }
}

/// A canonical section ordering — coarse to fine, structural groups before surface (§17.3).
const SECTION_ORDER: &[&str] = &["face", "eyes", "nose", "mouth", "skin", "hair", "facial_hair", "figure"];

fn section_rank(section: &str) -> usize {
    SECTION_ORDER.iter().position(|s| *s == section).unwrap_or(SECTION_ORDER.len())
}

/// Build the full question graph from the lexicon (§17.3), ordered depth-tier → section → order →
/// path. Deterministic (a pure function of the lexicon).
pub fn question_graph(lex: &Lexicon) -> Vec<Question> {
    let mut qs: Vec<Question> = lex
        .entries
        .iter()
        .map(|(path, e)| Question {
            path: path.clone(),
            ask: e.question_text(path),
            widget: e.widget_kind().to_string(),
            section: e.section.clone(),
            options: e.values.as_ref().map(|m| {
                let mut v: Vec<String> = m.keys().cloned().collect();
                v.sort();
                v
            }).unwrap_or_default(),
            tier: e.depth_tier(),
        })
        .collect();
    qs.sort_by(|a, b| {
        a.tier
            .cmp(&b.tier)
            .then_with(|| section_rank(&a.section).cmp(&section_rank(&b.section)))
            .then_with(|| lex.get(&a.path).and_then(|e| e.order).unwrap_or(0).cmp(&lex.get(&b.path).and_then(|e| e.order).unwrap_or(0)))
            .then_with(|| a.path.cmp(&b.path))
    });
    qs
}

/// The next unanswered question within `depth` whose `when` condition passes (§17.2). `None` when the
/// interview is complete at that depth.
pub fn next_question(lex: &Lexicon, answers: &AnswerLog, depth: Depth) -> Option<Question> {
    for q in question_graph(lex) {
        if q.tier > depth.tier() {
            continue;
        }
        if answers.answered(&q.path) {
            continue;
        }
        if let Some(cond) = lex.get(&q.path).and_then(|e| e.when.as_ref()) {
            if !eval_condition(cond, answers) {
                continue;
            }
        }
        return Some(q);
    }
    None
}

/// Record an answer (§17.2).
pub fn apply(answers: &mut AnswerLog, path: &str, answer: Answer) {
    answers.entries.push((path.to_string(), answer));
}

/// Interview progress at `depth`: (answered, total in-depth questions).
pub fn progress(lex: &Lexicon, answers: &AnswerLog, depth: Depth) -> (usize, usize) {
    let graph = question_graph(lex);
    let in_depth: Vec<&Question> = graph.iter().filter(|q| q.tier <= depth.tier()).collect();
    let answered = in_depth.iter().filter(|q| answers.answered(&q.path)).count();
    (answered, in_depth.len())
}

/// Build a partial `PersonaSpec` (as JSON, which the HJSON loader accepts) from the answer log (§17.2).
/// `Unknown` answers are omitted (unknown stays unknown, §6.4); `NoneEmpty` on a collection emits `[]`.
pub fn to_partial_spec(answers: &AnswerLog) -> Value {
    let mut root = Map::new();
    root.insert("schema".into(), Value::String("persona/1".into()));
    for (path, a) in &answers.entries {
        let val = match a {
            Answer::Scalar(v) => Value::from(*v),
            Answer::Number(v) => Value::from(*v),
            Answer::Enum(s) | Answer::Color(s) | Answer::Text(s) => Value::String(s.clone()),
            Answer::NoneEmpty => Value::Array(Vec::new()),
            Answer::Unknown => continue,
        };
        insert_path(&mut root, path, val);
    }
    Value::Object(root)
}

/// Build a spec from a flat `path → value` answer map (the `--answers` replay format, §17.12). A
/// string `"?"` value means "unknown" and is omitted; any other value (scalar/string/array/object) is
/// inserted at its dotted path. Deterministic — the same map always yields the same document.
pub fn spec_from_map(map: &Map<String, Value>) -> Value {
    let mut root = Map::new();
    root.insert("schema".into(), Value::String("persona/1".into()));
    for (path, val) in map {
        if path == "schema" {
            continue;
        }
        if val.as_str() == Some("?") {
            continue;
        }
        insert_path(&mut root, path, val.clone());
    }
    Value::Object(root)
}

/// Insert `value` at a dotted `path` into a JSON object tree; numeric segments create array indices.
fn insert_path(root: &mut Map<String, Value>, path: &str, value: Value) {
    let segs: Vec<&str> = path.split('.').collect();
    let mut node = Value::Object(std::mem::take(root));
    walk(&mut node, &segs, value);
    if let Value::Object(m) = node {
        *root = m;
    }
}

/// Walk/create the object path, setting `value` at the leaf. A numeric segment indexes an array.
fn walk(node: &mut Value, segs: &[&str], value: Value) {
    let Some(seg) = segs.first() else {
        *node = value;
        return;
    };
    if let Ok(idx) = seg.parse::<usize>() {
        if !node.is_array() {
            *node = Value::Array(Vec::new());
        }
        let arr = node.as_array_mut().unwrap();
        while arr.len() <= idx {
            arr.push(Value::Null);
        }
        walk(&mut arr[idx], &segs[1..], value);
    } else {
        if !node.is_object() {
            *node = Value::Object(Map::new());
        }
        let child = node.as_object_mut().unwrap().entry(seg.to_string()).or_insert(Value::Null);
        walk(child, &segs[1..], value);
    }
}

// --- a small total condition language (§17.3): `path OP value`, joined by `&&`/`||`, plus `path?` ---

fn eval_condition(expr: &str, answers: &AnswerLog) -> bool {
    // OR of ANDs.
    expr.split("||").any(|clause| clause.split("&&").all(|term| eval_term(term.trim(), answers)))
}

fn eval_term(term: &str, answers: &AnswerLog) -> bool {
    // presence: `path?` / `!path?`
    if let Some(p) = term.strip_suffix('?') {
        if let Some(p) = p.strip_prefix('!') {
            return !answers.answered(p.trim());
        }
        return answers.answered(p.trim());
    }
    for (op, cmp) in [("!=", 1), ("==", 0), (">=", 3), ("<=", 4), (">", 5), ("<", 6)] {
        if let Some((lhs, rhs)) = term.split_once(op) {
            let (lhs, rhs) = (lhs.trim(), rhs.trim());
            let lv = answers.as_str(lhs);
            return match cmp {
                0 => lv.as_deref() == Some(rhs),
                1 => lv.as_deref() != Some(rhs),
                _ => match (lv.and_then(|s| s.parse::<f64>().ok()), rhs.parse::<f64>().ok()) {
                    (Some(a), Some(b)) => match cmp {
                        3 => a >= b,
                        4 => a <= b,
                        5 => a > b,
                        _ => a < b,
                    },
                    _ => false,
                },
            };
        }
    }
    // an unparseable term does not gate (total → default visible).
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex() -> Lexicon {
        Lexicon::skeleton()
    }

    #[test]
    fn graph_is_ordered_structural_before_surface() {
        let g = question_graph(&lex());
        assert!(!g.is_empty());
        // the first question is a quick/structural one; tiers are non-decreasing.
        for w in g.windows(2) {
            assert!(w[0].tier <= w[1].tier, "tiers non-decreasing");
        }
        // eyes.spacing (structural) comes before eyes.color (surface).
        let pos = |p: &str| g.iter().position(|q| q.path == p).unwrap();
        assert!(pos("eyes.spacing") < pos("eyes.color"));
    }

    #[test]
    fn next_question_advances_and_respects_depth() {
        let l = lex();
        let mut ans = AnswerLog::default();
        // quick depth only surfaces tier-0 (structural) questions.
        let q0 = next_question(&l, &ans, Depth::Quick).unwrap();
        assert_eq!(q0.tier, 0);
        apply(&mut ans, &q0.path, Answer::Unknown);
        let q1 = next_question(&l, &ans, Depth::Quick).unwrap();
        assert_ne!(q1.path, q0.path, "advanced past the answered question");
        // a surface attribute is not asked at quick depth.
        assert!(!question_graph(&l).iter().filter(|q| q.tier == 0).any(|q| q.path == "eyes.color"));
    }

    #[test]
    fn replay_is_byte_stable() {
        let mut ans = AnswerLog::default();
        apply(&mut ans, "eyes.color", Answer::Color("blue".into()));
        apply(&mut ans, "eyes.spacing", Answer::Scalar(0.8));
        apply(&mut ans, "face.width", Answer::Unknown); // omitted
        apply(&mut ans, "marks", Answer::NoneEmpty);
        let spec = to_partial_spec(&ans);
        let s = serde_json::to_string(&spec).unwrap();
        // the same sequence always produces the same document.
        let mut ans2 = AnswerLog::default();
        apply(&mut ans2, "eyes.color", Answer::Color("blue".into()));
        apply(&mut ans2, "eyes.spacing", Answer::Scalar(0.8));
        apply(&mut ans2, "face.width", Answer::Unknown);
        apply(&mut ans2, "marks", Answer::NoneEmpty);
        assert_eq!(s, serde_json::to_string(&to_partial_spec(&ans2)).unwrap());
        // structure: nested, unknown omitted, none → [].
        assert_eq!(spec["eyes"]["color"], Value::String("blue".into()));
        assert!(spec["eyes"].get("spacing").is_some());
        assert!(spec.get("face").is_none(), "unknown face.width omitted");
        assert_eq!(spec["marks"], Value::Array(vec![]));
    }

    #[test]
    fn partial_spec_parses_as_a_persona_spec() {
        let mut ans = AnswerLog::default();
        apply(&mut ans, "eyes.color", Answer::Color("hazel".into()));
        apply(&mut ans, "identity.name", Answer::Text("ada".into()));
        let json = serde_json::to_string(&to_partial_spec(&ans)).unwrap();
        let spec = crate::persona::PersonaSpec::from_hjson(&json).expect("partial spec must parse");
        assert_eq!(spec.identity.and_then(|i| i.name).as_deref(), Some("ada"));
    }

    #[test]
    fn condition_language_is_total() {
        let mut ans = AnswerLog::default();
        apply(&mut ans, "identity.sex", Answer::Enum("female".into()));
        assert!(!eval_condition("identity.sex != female", &ans));
        assert!(eval_condition("identity.sex == female", &ans));
        assert!(eval_condition("identity.sex != male || flags.all?", &ans)); // female != male → true
        assert!(eval_condition("identity.sex?", &ans));
        assert!(eval_condition("!hair.color?", &ans));
        // numeric + unparseable-is-visible.
        apply(&mut ans, "identity.apparent_age", Answer::Number(40.0));
        assert!(eval_condition("identity.apparent_age >= 18", &ans));
        assert!(eval_condition("garbage expression", &ans));
    }
}
