//! smysl integration — 6.30.0 Phase 1 (FOUNDATION).
//!
//! smysl (github.com/vulogov/smysl) is an AI→AI→Human knowledge format: typed units (`@claim`,
//! `@evidence`, `@finding`, …) with explicit confidence, content-hash identity, and provenance edges
//! (`@rel`). plakat's compile chain (prose → make-composition → analyze → fix → enhance → condense) is
//! exactly such a handoff; this module is the foundation that turns a resolved prose scene into smysl
//! units so later phases can record the analyze/fix decisions + provenance + budget packing on top.
//!
//! This is a SIDECAR layer — it never replaces the scenario HJSON; prose stays the authoring surface.

use crate::compile::resolver::ResolvedScene;
use smysl_core::surface::{write_surface, WriteContext};
use smysl_core::{
    canonical_uid, hash_bytes, DropReason, KernelType, Label, PackInfo, RelKind, Record, Relation,
    Status, Uid, UnitCoreBuilder,
};
use std::collections::BTreeMap;

/// Turn a component name into a valid smysl label body: `prefix/kebab` (lowercase, non-alphanumerics
/// collapsed to `-`), e.g. `engine_1` → `c/engine-1`.
fn label_for(prefix: &str, raw: &str) -> String {
    let mut out = String::new();
    let mut last_dash = false;
    for ch in raw.trim().chars() {
        if ch.is_alphanumeric() {
            out.extend(ch.to_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    let body = out.trim_matches('-');
    format!("{prefix}/{}", if body.is_empty() { "x" } else { body })
}

/// Build a `@claim` unit from a description, register its `label → uid`, and return the uid (so relations
/// and later grounds can reference it). Authored scene content ⇒ `Speculative` in Phase 1: the author
/// asserts these elements with no backing evidence yet. Phase 2 GROUNDS them — a `--fix` claim is derived
/// from an `--analyze` finding — which is where the confidence model comes alive (`Derived`/`Inferred`
/// require grounds; `Measured`/`Cited` require a source — none of which a bare authored clause has).
fn add_claim(
    records: &mut Vec<Record>,
    labels: &mut BTreeMap<Label, Uid>,
    label: &str,
    gist: &str,
    body: &str,
) -> anyhow::Result<Uid> {
    let core = UnitCoreBuilder::new(KernelType::Claim, gist, Status::Speculative)
        .body(body)
        .build()
        .map_err(|e| anyhow::anyhow!("smysl unit `{label}`: {e:?}"))?;
    let uid = canonical_uid(&core);
    let lbl = Label::new(label).map_err(|e| anyhow::anyhow!("smysl label `{label}`: {e:?}"))?;
    labels.insert(lbl, uid);
    records.push(Record::Unit(core));
    Ok(uid)
}

/// Map a `relate:` verb to a smysl relation kind. plakat's relations are PHYSICAL (behind / towing /
/// holding / sitting-on), not smysl's discourse kinds, so they ride the extension escape hatch
/// `x.plakat/<verb>` (preserved verbatim, treated as `elaborates` for closure). Falls back to
/// `elaborates` if the verb can't form a valid extension id.
fn rel_kind(verb: &str) -> RelKind {
    let v = verb.trim().to_lowercase().replace(' ', "-");
    RelKind::parse(&format!("x.plakat/{v}")).unwrap_or(RelKind::Elaborates)
}

/// 6.30.0: resolved scene → smysl records + their labels. Each foreground / object / background
/// component becomes a `@claim`; each relate between two named components becomes a `@rel`; the
/// free-text scene description becomes one prose claim. Shared by [`scene_to_smysl`] (serialize) and
/// [`trace_report`] (query) so both see the identical unit set.
pub fn scene_records(scene: &ResolvedScene) -> anyhow::Result<(Vec<Record>, BTreeMap<Label, Uid>)> {
    let mut records: Vec<Record> = Vec::new();
    let mut labels: BTreeMap<Label, Uid> = BTreeMap::new();
    let mut uid_by_name: std::collections::HashMap<String, Uid> = std::collections::HashMap::new();

    // Every asserted scene element (heroes, structural objects, background people) is a claim.
    for (name, desc) in scene
        .foreground
        .iter()
        .chain(scene.objects.iter())
        .chain(scene.background.iter())
    {
        let label = label_for("c", name);
        let gist = desc.split([',', '.', ';']).next().unwrap_or(desc).trim();
        let uid = add_claim(&mut records, &mut labels, &label, gist, desc.trim())?;
        uid_by_name.insert(name.clone(), uid);
    }
    // The free-text scene sentence(s) → the overall-scene claim.
    if !scene.free_text.trim().is_empty() {
        add_claim(&mut records, &mut labels, "c/scene", "the scene", scene.free_text.trim())?;
    }
    // Relations. A component may be defined (global or task-level) yet only referenced from a
    // `relate:` — so it never lands in foreground/objects. Register any such endpoint as its own claim
    // (the RelationSpec carries its description) before drawing the edge, so components AND their
    // relations always appear.
    for rel in &scene.relations {
        for (name, desc) in [(&rel.a, &rel.a_desc), (&rel.b, &rel.b_desc)] {
            if !uid_by_name.contains_key(name) {
                let label = label_for("c", name);
                let gist = desc.split([',', '.', ';']).next().unwrap_or(desc).trim();
                let uid = add_claim(&mut records, &mut labels, &label, gist, desc.trim())?;
                uid_by_name.insert(name.clone(), uid);
            }
        }
        if let (Some(&from), Some(&to)) = (uid_by_name.get(&rel.a), uid_by_name.get(&rel.b)) {
            records.push(Record::Relation(Relation::new(rel_kind(&rel.verb), from, to)));
        }
    }
    Ok((records, labels))
}

/// Serialize a pre-built record set to readable surface text (shared by the single-scene
/// [`scene_to_smysl`] and the multi-scene `--smysl` sidecar writer in `compile`).
pub fn records_to_surface(records: &[Record], labels: &BTreeMap<Label, Uid>) -> String {
    write_surface(None, records, &WriteContext::from_labels(labels))
}

/// Merge two smysl surface documents into one (content-hash union, `base` labels win). Either side may be
/// empty. Used to fold the compile's budget-pack corpus into the scene-claims sidecar under `--smysl`.
pub fn merge_surface(base: &str, extra: &str) -> String {
    let parse = |s: &str| {
        smysl_core::surface::parse_surface(s)
            .map(|p| (p.records, p.labels))
            .unwrap_or_else(|_| (Vec::new(), BTreeMap::new()))
    };
    let (mut recs, mut labels) = parse(base);
    let (er, el) = parse(extra);
    merge_records(&mut recs, &mut labels, er, el);
    records_to_surface(&recs, &labels)
}

/// Fold `extra` records/labels INTO `records`/`labels`, deduping (smysl units are content-addressed,
/// so union by uid; relations by `(kind, from, to)`). Existing labels win, so already-bound names stay
/// stable. Used by `--smysl` to FINALIZE the scene claims onto an existing fix-corpus without clobbering.
pub fn merge_records(
    records: &mut Vec<Record>,
    labels: &mut BTreeMap<Label, Uid>,
    extra: Vec<Record>,
    extra_labels: BTreeMap<Label, Uid>,
) {
    let mut have_unit: std::collections::HashSet<Uid> = records
        .iter()
        .filter_map(|r| match r {
            Record::Unit(c) => Some(canonical_uid(c)),
            _ => None,
        })
        .collect();
    let mut have_rel: std::collections::HashSet<(String, Uid, Uid)> = records
        .iter()
        .filter_map(|r| match r {
            Record::Relation(rel) => Some((rel.kind.as_str().to_string(), rel.from, rel.to)),
            _ => None,
        })
        .collect();
    for r in extra {
        match &r {
            Record::Unit(c) => {
                if have_unit.insert(canonical_uid(c)) {
                    records.push(r);
                }
            }
            Record::Relation(rel) => {
                if have_rel.insert((rel.kind.as_str().to_string(), rel.from, rel.to)) {
                    records.push(r);
                }
            }
            _ => records.push(r),
        }
    }
    for (l, u) in extra_labels {
        labels.entry(l).or_insert(u);
    }
}

/// 6.30.0 Phase 1: resolved scene → smysl surface text. Thin serializer over [`scene_records`] —
/// the readable surface syntax the analyze/fix provenance corpus (Phase 2+) accumulates onto.
pub fn scene_to_smysl(scene: &ResolvedScene) -> anyhow::Result<String> {
    let (records, labels) = scene_records(scene)?;
    Ok(records_to_surface(&records, &labels))
}

/// Clip a phrase for a gist line (single line, ≤50 chars).
fn clip(s: &str) -> String {
    let s = s.trim().replace('\n', " ");
    if s.chars().count() > 50 {
        format!("{}…", s.chars().take(49).collect::<String>())
    } else {
        s
    }
}

/// A short, label-safe, CONTENT-ADDRESSED tag for a unit: the first 10 base32 chars of its uid. Same
/// content ⇒ same tag (so repeated `--fix` runs DEDUP an identical finding), different content ⇒ different
/// tag (so distinct findings never collide on a positional `-0`). This is what lets the corpus accumulate
/// deterministically across polish iterations instead of clobbering.
fn uid_tag(uid: &Uid) -> String {
    uid.to_string()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .skip(1) // drop the leading `b` of the `b3:` prefix so the tag isn't always `b…`
        .take(10)
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// 6.30.0 Phase 2: build the `--analyze`/`--fix` decision records. Each APPLIED fix → a `@claim` GROUNDED
/// in the `@finding` it addressed (the risk); un-auto-fixable structural items → standalone `@finding`s.
/// Labels are content-addressed (see [`uid_tag`]) so the record set MERGES cleanly across repeated runs.
///
/// `applied` = `(old, new, why)` for each edit that landed; `manual` = the structural notes.
pub fn fixes_to_records(
    applied: &[(String, String, String)],
    manual: &[String],
) -> anyhow::Result<(Vec<Record>, BTreeMap<Label, Uid>)> {
    let mut records: Vec<Record> = Vec::new();
    let mut labels: BTreeMap<Label, Uid> = BTreeMap::new();

    for (i, (old, new, why)) in applied.iter().enumerate() {
        // The risk the critic found → a finding (a prediction ⇒ Speculative at this layer).
        let finding = UnitCoreBuilder::new(KernelType::Finding, why.trim(), Status::Speculative)
            .build()
            .map_err(|e| anyhow::anyhow!("smysl finding {i}: {e:?}"))?;
        let f_uid = canonical_uid(&finding);
        labels.insert(Label::new(&format!("f/risk-{}", uid_tag(&f_uid)))?, f_uid);
        records.push(Record::Unit(finding));

        // The applied fix → a claim DERIVED from (grounded in) that finding: the provenance edge.
        let gist = format!("replace \u{201C}{}\u{201D} with \u{201C}{}\u{201D}", clip(old), clip(new));
        let fix = UnitCoreBuilder::new(KernelType::Claim, gist, Status::Derived)
            .body(format!("was: {}\nnow: {}", old.trim(), new.trim()))
            .grounds([f_uid])
            .build()
            .map_err(|e| anyhow::anyhow!("smysl fix {i}: {e:?}"))?;
        let fix_uid = canonical_uid(&fix);
        labels.insert(Label::new(&format!("c/fix-{}", uid_tag(&fix_uid)))?, fix_uid);
        records.push(Record::Unit(fix));
    }
    for (i, m) in manual.iter().enumerate() {
        let f = UnitCoreBuilder::new(KernelType::Finding, m.trim(), Status::Speculative)
            .build()
            .map_err(|e| anyhow::anyhow!("smysl manual {i}: {e:?}"))?;
        let uid = canonical_uid(&f);
        labels.insert(Label::new(&format!("f/manual-{}", uid_tag(&uid)))?, uid);
        records.push(Record::Unit(f));
    }
    Ok((records, labels))
}

/// Serialize [`fixes_to_records`] to surface text (a single snapshot; the accumulating corpus writer in
/// `apply_fixes` merges these records onto the existing corpus instead).
pub fn fixes_to_smysl(applied: &[(String, String, String)], manual: &[String]) -> anyhow::Result<String> {
    let (records, labels) = fixes_to_records(applied, manual)?;
    Ok(records_to_surface(&records, &labels))
}

// ---------------------------------------------------------------------------
// smysl-optimize Phase B — the corpus as a TABU LIST for the --fix loop.
// The corpus already REMEMBERS every fix applied in prior passes; these read it
// back so the next pass can refuse to re-apply or REVERSE a change it already
// tried — the deterministic half of "consult changes, reduce the death march".
// ---------------------------------------------------------------------------

/// Normalize a phrase for tabu comparison: trimmed, lowercased, whitespace-collapsed.
fn norm_phrase(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// Read every fix ALREADY recorded in a corpus as `(old, new)` pairs — the moves prior `--fix` passes made.
/// Parses the surface, finds the `c/fix-*` claims, and pulls `was:`/`now:` out of each body. Used to build
/// the tabu set the next pass filters against.
pub fn prior_fixes(corpus_text: &str) -> Vec<(String, String)> {
    let Ok(p) = smysl_core::surface::parse_surface(corpus_text) else {
        return Vec::new();
    };
    let uid_label: std::collections::HashMap<Uid, &str> =
        p.labels.iter().map(|(l, u)| (*u, l.as_str())).collect();
    let mut out = Vec::new();
    for r in &p.records {
        let Record::Unit(c) = r else { continue };
        // Only fix claims carry a was/now move.
        let is_fix = uid_label.get(&canonical_uid(c)).map(|l| l.starts_with("c/fix-")).unwrap_or(false);
        if !is_fix {
            continue;
        }
        if let Some(body) = &c.body {
            let (mut was, mut now) = (None, None);
            for line in body.lines() {
                if let Some(v) = line.trim().strip_prefix("was:") {
                    was = Some(v.trim().to_string());
                } else if let Some(v) = line.trim().strip_prefix("now:") {
                    now = Some(v.trim().to_string());
                }
            }
            if let (Some(w), Some(n)) = (was, now) {
                out.push((w, n));
            }
        }
    }
    out
}

/// Decide whether a proposed edit `old → new` is TABU against the prior fixes — i.e. a move the loop already
/// made and must not repeat. Returns the reason to skip, or `None` if the edit is fresh. Two deterministic
/// cases (the death-march killers): an EXACT duplicate of a prior fix, and a REVERSAL of one (proposing to
/// turn a prior fix's result back toward what it replaced — the oscillation that never converges).
pub fn tabu_reason(old: &str, new: &str, prior: &[(String, String)]) -> Option<&'static str> {
    let (no, nn) = (norm_phrase(old), norm_phrase(new));
    if no == nn {
        return None;
    }
    for (po, pn) in prior {
        let (npo, npn) = (norm_phrase(po), norm_phrase(pn));
        if npo == no && npn == nn {
            return Some("already applied in a prior pass");
        }
        if npn == no && npo == nn {
            return Some("reverses a prior fix (oscillation)");
        }
    }
    None
}

/// A short, human-readable list of the moves already made — fed to the fixer as a "do NOT re-propose these"
/// hint (the LLM-side half of the tabu; [`tabu_reason`] is the deterministic safety net). Empty string when
/// there is no prior history.
pub fn tabu_hint(prior: &[(String, String)]) -> String {
    if prior.is_empty() {
        return String::new();
    }
    let lines: Vec<String> = prior
        .iter()
        .map(|(o, n)| format!("- \u{201C}{}\u{201D} \u{2192} \u{201C}{}\u{201D}", clip(o), clip(n)))
        .collect();
    format!(
        "ALREADY APPLIED in prior passes — do NOT re-propose these edits, and do NOT reverse them:\n{}",
        lines.join("\n")
    )
}

/// Read the corpus's prior findings, split into RESOLVED (a `c/fix-*` claim grounds in them) and OPEN (no
/// fix yet). Feeds the CRITIC so `--analyze` stops re-flagging risks earlier passes already fixed — the
/// consult side of the death-march guard, at the critic level (the fixer already has [`tabu_hint`]).
pub fn resolved_open_findings(corpus_text: &str) -> (Vec<String>, Vec<String>) {
    let Ok(p) = smysl_core::surface::parse_surface(corpus_text) else {
        return (Vec::new(), Vec::new());
    };
    let uid_label: std::collections::HashMap<Uid, &str> =
        p.labels.iter().map(|(l, u)| (*u, l.as_str())).collect();
    let mut findings: Vec<(Uid, String)> = Vec::new();
    let mut resolved_uids: std::collections::HashSet<Uid> = std::collections::HashSet::new();
    for r in &p.records {
        let Record::Unit(c) = r else { continue };
        let uid = canonical_uid(c);
        match uid_label.get(&uid).copied().unwrap_or("") {
            l if l.starts_with("f/") => findings.push((uid, c.gist.clone())), // f/risk-* or f/manual-*
            l if l.starts_with("c/fix-") => resolved_uids.extend(c.grounds.iter().copied()),
            _ => {}
        }
    }
    let (mut resolved, mut open) = (Vec::new(), Vec::new());
    for (uid, gist) in findings {
        if resolved_uids.contains(&uid) {
            resolved.push(gist);
        } else {
            open.push(gist);
        }
    }
    (resolved, open)
}

/// A critic-facing hint from prior findings: don't re-flag what earlier passes already FIXED (unless it
/// clearly recurs); the STILL-OPEN ones remain fair game. Empty when there is no prior history.
pub fn findings_hint(resolved: &[String], open: &[String]) -> String {
    let mut s = String::new();
    if !resolved.is_empty() {
        s.push_str(
            "PRIOR RISKS ALREADY ADDRESSED in earlier passes — do NOT flag these again unless they \
             CLEARLY RECUR in the current prose:\n",
        );
        for g in resolved {
            s.push_str(&format!("- {}\n", clip(g)));
        }
    }
    if !open.is_empty() {
        if !s.is_empty() {
            s.push('\n');
        }
        s.push_str("PRIOR RISKS STILL OPEN — flag them only if still present:\n");
        for g in open {
            s.push_str(&format!("- {}\n", clip(g)));
        }
    }
    s.trim_end().to_string()
}

/// 6.30.0 Phase 2 `--trace`: over a record set (scene claims/relations plus any loaded fix-corpus
/// findings), report every unit whose gist/body mentions `phrase`, its confidence, and WHY it is
/// there — its `grounds` chain (a fix ← the finding it fixed) and the relations that touch it.
/// Answers "why is this phrase in the final prompt?" across the authored + decision layers. Pure /
/// weight-free; the caller assembles the records (scene + on-disk corpus).
pub fn trace_report(records: &[Record], labels: &BTreeMap<Label, Uid>, phrase: &str) -> String {
    // Reverse maps: uid → its label (for readable references) and uid → the unit (to read gist/body).
    let uid_label: std::collections::HashMap<Uid, &str> =
        labels.iter().map(|(l, u)| (*u, l.as_str())).collect();
    let units: std::collections::HashMap<Uid, &smysl_core::UnitCore> = records
        .iter()
        .filter_map(|r| match r {
            Record::Unit(c) => Some((canonical_uid(c), c)),
            _ => None,
        })
        .collect();
    let name = |u: &Uid| -> String { uid_label.get(u).map(|s| s.to_string()).unwrap_or_else(|| u.to_string()) };

    let needle = phrase.trim().to_lowercase();
    let hit = |c: &smysl_core::UnitCore| -> bool {
        c.gist.to_lowercase().contains(&needle)
            || c.body.as_deref().map(|b| b.to_lowercase().contains(&needle)).unwrap_or(false)
            || c.detail.as_deref().map(|d| d.to_lowercase().contains(&needle)).unwrap_or(false)
    };

    let mut out = String::new();
    let mut n = 0usize;
    // A unit can arrive twice (freshly-resolved scene + the same unit already in the on-disk corpus);
    // they share a content-hash, so dedup on uid keeps each mentioned unit once.
    let mut seen: std::collections::HashSet<Uid> = std::collections::HashSet::new();
    for r in records {
        let Record::Unit(c) = r else { continue };
        if !hit(c) {
            continue;
        }
        let uid = canonical_uid(c);
        if !seen.insert(uid) {
            continue;
        }
        n += 1;
        // Prefix tells the layer: c/ = authored claim / applied fix, f/ = finding (a risk),
        // o/ = observation (a budget-pack decision).
        let kind = match name(&uid).split('/').next() {
            Some("f") => "finding",
            Some("o") => "observation",
            _ => "claim",
        };
        out.push_str(&format!("• {} [{kind}, {:?}]\n    {}\n", name(&uid), c.status, clip(&c.gist)));
        // WHY it is here: the grounds chain (each ground unit's label + gist).
        for g in &c.grounds {
            let because = units.get(g).map(|u| clip(&u.gist)).unwrap_or_default();
            out.push_str(&format!("    ← grounded in {}  ({})\n", name(g), because));
        }
        // Relations that touch this unit (physical placement / towing / holding …). Dedup the lines —
        // the same edge can arrive from both the fresh scene and the folded on-disk corpus.
        let mut rel_lines: Vec<String> = Vec::new();
        for rr in records {
            if let Record::Relation(rel) = rr {
                let line = if rel.from == uid {
                    format!("    ─{}→ {}", rel.kind.as_str(), name(&rel.to))
                } else if rel.to == uid {
                    format!("    {} ─{}→ (this)", name(&rel.from), rel.kind.as_str())
                } else {
                    continue;
                };
                if !rel_lines.contains(&line) {
                    rel_lines.push(line);
                }
            }
        }
        for line in rel_lines {
            out.push_str(&line);
            out.push('\n');
        }
    }
    if n == 0 {
        format!("no smysl unit mentions “{}”.", phrase.trim())
    } else {
        format!("{n} unit(s) mention “{}”:\n\n{out}", phrase.trim())
    }
}

// ---------------------------------------------------------------------------
// 6.30.0 Phase 3 — model-free budget packing (smysl `pack` at the prompt seam)
// ---------------------------------------------------------------------------

/// Generic prompt-craft boilerplate — quality boosters that carry no scene content, so trimming them to
/// fit a budget loses nothing. NOT scene attributes (those are derived from the prose, never hardcoded);
/// these are the universal filler every prompt guide warns about. Compared against a normalized span.
const FILLER_SPANS: &[&str] = &[
    "masterpiece", "best quality", "high quality", "highly detailed", "very detailed",
    "extremely detailed", "intricate details", "intricate detail", "ultra detailed",
    "ultra-detailed", "hyperdetailed", "8k", "4k", "uhd", "hd", "sharp focus",
    "trending on artstation", "artstation", "award winning", "award-winning", "stunning",
    "beautiful", "gorgeous", "professional", "cinematic lighting", "dramatic lighting",
];

/// Outcome of [`pack_prompt`]: the packed prompt, the smysl `PackInfo` (auditable drop record), and the
/// human-readable text of each dropped span (aligned with `info.dropped`, which only carries uids).
pub struct PackResult {
    pub text: String,
    pub info: PackInfo,
    pub dropped_spans: Vec<(String, DropReason)>,
}

impl PackResult {
    /// True when the pack fit the budget by dropping ONLY generic filler (nothing essential lost) — the
    /// condition under which the model-free result is preferred over an LLM reword.
    pub fn fit_on_filler_alone(&self, budget: usize) -> bool {
        self.info.used as usize <= budget
            && self.dropped_spans.iter().all(|(_, r)| *r == DropReason::LowValue)
    }
}

/// Strip a span down to the bare phrase for filler/weight detection: drop a leading `(`/`[`, a trailing
/// `:weight)` / `)` / `]`, and lowercase. `"(intricate details:1.2)"` → `"intricate details"`.
fn normalize_span(span: &str) -> (String, Option<f32>) {
    let s = span.trim();
    let inner = s.trim_start_matches(['(', '[']).trim_end_matches([')', ']']);
    // A trailing `:number` is an attention weight, not part of the phrase.
    if let Some(idx) = inner.rfind(':') {
        if let Ok(w) = inner[idx + 1..].trim().parse::<f32>() {
            return (inner[..idx].trim().to_lowercase(), Some(w));
        }
    }
    (inner.trim().to_lowercase(), None)
}

/// Split a prompt into top-level comma spans, keeping any `(… , …:w)` weight group intact (commas inside
/// parens/brackets are not separators).
fn split_spans(prompt: &str) -> Vec<String> {
    let mut spans = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in prompt.chars() {
        match ch {
            '(' | '[' => {
                depth += 1;
                cur.push(ch);
            }
            ')' | ']' => {
                depth -= 1;
                cur.push(ch);
            }
            ',' if depth <= 0 => {
                if !cur.trim().is_empty() {
                    spans.push(cur.trim().to_string());
                }
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        spans.push(cur.trim().to_string());
    }
    spans
}

/// 6.30.0 Phase 3: a deterministic, MODEL-FREE prompt packer. Splits the prompt into top-level spans,
/// scores each by salience (attention weight, then front-load position, minus a generic-filler penalty),
/// and greedily keeps the highest-salience spans that fit `budget` tokens — NEVER dropping the leading
/// subject span. Emits a smysl [`PackInfo`] recording exactly what was dropped and why (filler →
/// `LowValue`, essential-for-space → `Budget`), so a budget trim is auditable, reproducible, and
/// model-free — versus an LLM condense that silently rewords. `token_est` is the caller's estimator so the
/// budget matches the rest of the pipeline (recorded in `PackInfo.estimator`).
pub fn pack_prompt(prompt: &str, budget: usize, token_est: &dyn Fn(&str) -> usize) -> PackResult {
    let spans = split_spans(prompt);
    let n = spans.len().max(1);
    // Score each span. The first span is the subject → mandatory (never dropped).
    struct Scored {
        idx: usize,
        text: String,
        salience: f32,
        filler: bool,
    }
    let mut scored: Vec<Scored> = spans
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let (bare, weight) = normalize_span(text);
            let filler = FILLER_SPANS.contains(&bare.as_str());
            // weighted span → its weight; else 1.0. Front-load bonus for earlier spans. Filler penalty.
            let mut salience = weight.unwrap_or(1.0);
            salience += 0.4 * (1.0 - i as f32 / n as f32);
            if filler {
                salience -= 1.0;
            }
            if i == 0 {
                salience = f32::INFINITY; // subject: keep at all costs
            }
            Scored { idx: i, text: text.clone(), salience, filler }
        })
        .collect();

    // Admission order: highest salience first, ties by original position (stable, reproducible).
    let mut order: Vec<usize> = (0..scored.len()).collect();
    order.sort_by(|&a, &b| {
        scored[b].salience
            .partial_cmp(&scored[a].salience)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(scored[a].idx.cmp(&scored[b].idx))
    });

    // Greedily admit spans (in salience order) while the reassembled prompt stays within budget.
    let mut keep = vec![false; scored.len()];
    let join_kept = |keep: &[bool], scored: &[Scored]| -> String {
        scored
            .iter()
            .filter(|s| keep[s.idx])
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    for &oi in &order {
        keep[oi] = true;
        if token_est(&join_kept(&keep, &scored)) > budget && scored[oi].idx != 0 {
            keep[oi] = false; // adding this span overflows — drop it (unless it's the subject)
        }
    }

    let text = join_kept(&keep, &scored);
    let used = token_est(&text) as u64;
    let mut info = PackInfo::new(budget as u64, used, "plakat/estimate_tokens v1");
    let mut dropped_spans = Vec::new();
    for s in &mut scored {
        if !keep[s.idx] {
            let reason = if s.filler { DropReason::LowValue } else { DropReason::Budget };
            let uid = Uid::from_bytes(hash_bytes(s.text.as_bytes()));
            info.dropped.push((uid, reason));
            dropped_spans.push((std::mem::take(&mut s.text), reason));
        }
    }
    PackResult { text, info, dropped_spans }
}

/// One scene's budget-pack decision, carried up from the compile so it can be recorded in the corpus —
/// this is the "trace BUDGETS" half of the smysl vision (the fix records are the "trace CHANGES" half).
#[derive(Clone)]
pub struct ScenePack {
    pub scene: String,
    pub budget: u64,
    pub used: u64,
    pub family: String,
    pub dropped: Vec<(String, DropReason)>,
}

/// Build corpus records for a set of scene budget-pack decisions: each scene that dropped anything → one
/// `@observation` recording the model budget, tokens used, and exactly what was cut (+reason). Labels are
/// content-addressed so these merge/dedup across runs like the fix records.
pub fn packs_to_records(packs: &[ScenePack]) -> anyhow::Result<(Vec<Record>, BTreeMap<Label, Uid>)> {
    let mut records: Vec<Record> = Vec::new();
    let mut labels: BTreeMap<Label, Uid> = BTreeMap::new();
    for p in packs {
        if p.dropped.is_empty() {
            continue;
        }
        let gist = format!(
            "budget pack ‘{}’: dropped {} span(s) to fit {} in {} tokens",
            p.scene,
            p.dropped.len(),
            p.family,
            p.budget
        );
        let body = format!(
            "used {}/{} tokens; dropped:\n{}",
            p.used,
            p.budget,
            p.dropped.iter().map(|(t, r)| format!("- {} [{}]", clip(t), r.as_str())).collect::<Vec<_>>().join("\n")
        );
        let unit = UnitCoreBuilder::new(KernelType::Observation, gist, Status::Speculative)
            .body(body)
            .build()
            .map_err(|e| anyhow::anyhow!("smysl pack observation: {e:?}"))?;
        let uid = canonical_uid(&unit);
        labels.insert(Label::new(&format!("o/pack-{}", uid_tag(&uid)))?, uid);
        records.push(Record::Unit(unit));
    }
    Ok((records, labels))
}

// ---------------------------------------------------------------------------
// smysl-recompile — the prose → emitted-prompt transformation of `compile`.
// A plain `compile` REWRITES the prompt (translate/compose/enhance/negative/fit);
// these record that transformation so `--trace` covers enhancer-introduced
// phrases and drift across recompilations is visible + versioned.
// ---------------------------------------------------------------------------

/// For each scene `(name, prose, emitted_prompt)`, build the compile transformation as two claims: the
/// authored `c/prose-…` (Speculative) and the `c/prompt-…` emitted prompt DERIVED FROM it (grounded in the
/// prose). Content-addressed, so an identical recompile dedups; a reworded emitted prompt is a new claim.
pub fn recompile_records(
    scenes: &[(String, String, String)],
) -> anyhow::Result<(Vec<Record>, BTreeMap<Label, Uid>)> {
    let mut records: Vec<Record> = Vec::new();
    let mut labels: BTreeMap<Label, Uid> = BTreeMap::new();
    for (name, prose, emitted) in scenes {
        if emitted.trim().is_empty() {
            continue;
        }
        let prose_txt = if prose.trim().is_empty() { name.trim() } else { prose.trim() };
        // The authored prose the emitted prompt derives from.
        let prose_core = UnitCoreBuilder::new(KernelType::Claim, format!("prose \u{00B7} {name}"), Status::Speculative)
            .body(prose_txt)
            .build()
            .map_err(|e| anyhow::anyhow!("smysl prose `{name}`: {e:?}"))?;
        let prose_uid = canonical_uid(&prose_core);
        let plabel = Label::new(&format!("c/prose-{}", uid_tag(&prose_uid)))?;
        if labels.insert(plabel, prose_uid).is_none() {
            records.push(Record::Unit(prose_core));
        }
        // The emitted prompt — DERIVED FROM the prose (the compile transformation, provenance edge).
        let prompt_core = UnitCoreBuilder::new(KernelType::Claim, format!("emitted prompt \u{00B7} {name}"), Status::Derived)
            .body(emitted.trim())
            .grounds([prose_uid])
            .build()
            .map_err(|e| anyhow::anyhow!("smysl emitted `{name}`: {e:?}"))?;
        let prompt_uid = canonical_uid(&prompt_core);
        labels.insert(Label::new(&format!("c/prompt-{}", uid_tag(&prompt_uid)))?, prompt_uid);
        records.push(Record::Unit(prompt_core));
    }
    Ok((records, labels))
}

/// The `(uid, grounds)` of every `c/prompt-*` emitted-prompt claim in a corpus.
fn prompt_claims(text: &str) -> Vec<(Uid, Vec<Uid>)> {
    let Ok(p) = smysl_core::surface::parse_surface(text) else {
        return Vec::new();
    };
    let uid_label: std::collections::HashMap<Uid, &str> =
        p.labels.iter().map(|(l, u)| (*u, l.as_str())).collect();
    p.records
        .iter()
        .filter_map(|r| match r {
            Record::Unit(c) => {
                let uid = canonical_uid(c);
                if uid_label.get(&uid).map(|l| l.starts_with("c/prompt-")).unwrap_or(false) {
                    Some((uid, c.grounds.iter().copied().collect()))
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect()
}

/// `Supersedes` relations linking each NEW emitted-prompt claim to the PRIOR one(s) for the same prose — the
/// recompilation drift history. Two prompt claims are "the same scene" when they ground in a shared prose uid.
pub fn supersedes_edges(new_text: &str, prior_text: &str) -> Vec<Record> {
    let news = prompt_claims(new_text);
    let priors = prompt_claims(prior_text);
    let mut edges = Vec::new();
    for (n_uid, n_g) in &news {
        for (p_uid, p_g) in &priors {
            if p_uid != n_uid && n_g.iter().any(|g| p_g.contains(g)) {
                edges.push(Record::Relation(Relation::new(RelKind::Supersedes, *n_uid, *p_uid)));
            }
        }
    }
    edges
}

/// Fold extra relation records into a serialized corpus, reusing its labels (the relations' endpoints are
/// already labeled units in `doc_text`). Dedups by `(kind, from, to)`.
pub fn merge_relations(doc_text: &str, rels: Vec<Record>) -> String {
    let Ok(p) = smysl_core::surface::parse_surface(doc_text) else {
        return doc_text.to_string();
    };
    let (mut recs, labels) = (p.records, p.labels);
    let mut have: std::collections::HashSet<(String, Uid, Uid)> = recs
        .iter()
        .filter_map(|r| match r {
            Record::Relation(rl) => Some((rl.kind.as_str().to_string(), rl.from, rl.to)),
            _ => None,
        })
        .collect();
    for r in rels {
        if let Record::Relation(rl) = &r {
            if have.insert((rl.kind.as_str().to_string(), rl.from, rl.to)) {
                recs.push(r);
            }
        }
    }
    records_to_surface(&recs, &labels)
}

/// Format the packer's drop record as a one-line user-facing audit ("dropped 3: 8k, masterpiece [filler];
/// distant hills [budget]") — so a budget trim is visible, never silent.
pub fn pack_drop_summary(dropped: &[(String, DropReason)]) -> String {
    if dropped.is_empty() {
        return "nothing dropped".to_string();
    }
    let list = dropped
        .iter()
        .map(|(t, r)| format!("{} [{}]", clip(t), r.as_str()))
        .collect::<Vec<_>>()
        .join("; ");
    format!("dropped {}: {list}", dropped.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_to_smysl_emits_claims_and_relations() {
        let scene = ResolvedScene {
            foreground: vec![("engine".into(), "a Victorian steam traction engine".into())],
            objects: vec![("cart".into(), "a four-wheeled trailer of books".into())],
            relations: vec![crate::compile::resolver::RelationSpec {
                a: "cart".into(),
                a_desc: "a four-wheeled trailer of books".into(),
                verb: "behind".into(),
                b: "engine".into(),
                b_desc: "a Victorian steam traction engine".into(),
            }],
            free_text: "A steam engine towing a trailer on a cobbled street.".into(),
            ..Default::default()
        };
        let out = scene_to_smysl(&scene).expect("builds");
        assert!(out.contains("@claim"), "emits claim records:\n{out}");
        assert!(out.contains("c/engine") && out.contains("c/cart"), "labels the components");
        assert!(out.contains("engine"), "carries the engine gist");
        assert!(out.contains("@rel"), "emits the relation");
    }

    #[test]
    fn fixes_to_smysl_grounds_each_fix_in_its_finding() {
        let applied = vec![(
            "старинный паровой локомобиль".to_string(),
            "старинный паровой тягач".to_string(),
            "rare/invented object name — model will hallucinate".to_string(),
        )];
        let manual = vec!["split the over-stuffed scene into two images".to_string()];
        let out = fixes_to_smysl(&applied, &manual).expect("builds");
        assert!(out.contains("@finding f/risk-"), "the risk is a finding:\n{out}");
        assert!(out.contains("@claim c/fix-"), "the fix is a claim");
        // The fix's `grounds:` must reference the SAME content-addressed finding label.
        let f_label = out
            .lines()
            .find_map(|l| l.trim().strip_prefix("@finding "))
            .and_then(|s| s.split_whitespace().next())
            .expect("a finding label");
        assert!(out.contains(&format!("grounds: [{f_label}]")), "fix GROUNDED in its finding:\n{out}");
        assert!(out.contains("@finding f/manual-"), "structural item is an open finding");
    }

    #[test]
    fn fixes_corpus_accumulates_across_runs_without_collision() {
        // Run 1: one fix. Run 2: a DIFFERENT fix. Merging run 2 onto run 1 must keep BOTH (distinct
        // content-addressed labels), and re-merging run 1 must add nothing (dedup).
        let run1 = vec![("локомобиль".to_string(), "тягач".to_string(), "rare name".to_string())];
        let run2 = vec![("bundles".to_string(), "stacked books".to_string(), "cargo fusion".to_string())];
        let (mut recs, mut labels) = fixes_to_records(&run1, &[]).unwrap();
        let (r2, l2) = fixes_to_records(&run2, &[]).unwrap();
        merge_records(&mut recs, &mut labels, r2, l2);
        let out = records_to_surface(&recs, &labels);
        assert!(out.contains("тягач") && out.contains("stacked books"), "both runs' fixes present:\n{out}");
        let n_before = recs.len();
        let (r1b, l1b) = fixes_to_records(&run1, &[]).unwrap();
        merge_records(&mut recs, &mut labels, r1b, l1b);
        assert_eq!(recs.len(), n_before, "re-merging run 1 dedups (content-addressed), adds nothing");
    }

    #[test]
    fn tabu_recovers_prior_fixes_and_blocks_repeats() {
        // A corpus with one applied fix (in Russian, to prove language-agnostic normalization).
        let applied = vec![("локомобиль".to_string(), "тягач".to_string(), "rare name".to_string())];
        let corpus = fixes_to_smysl(&applied, &[]).unwrap();
        let prior = prior_fixes(&corpus);
        assert_eq!(prior, vec![("локомобиль".to_string(), "тягач".to_string())], "recovers the was→now move");

        // Exact duplicate of a prior fix → tabu.
        assert_eq!(tabu_reason("локомобиль", "тягач", &prior), Some("already applied in a prior pass"));
        // The death-march move — reversing a prior fix — → tabu.
        assert_eq!(tabu_reason("тягач", "локомобиль", &prior), Some("reverses a prior fix (oscillation)"));
        // Normalization: case + surrounding whitespace still caught.
        assert!(tabu_reason("  Тягач ", "локомобиль", &prior).is_some());
        // A genuinely fresh edit is allowed through.
        assert_eq!(tabu_reason("bundles", "stacked books", &prior), None);
        assert_eq!(tabu_reason("x", "x", &prior), None, "a no-op is not tabu, just ignored");
    }

    #[test]
    fn tabu_hint_lists_moves_or_is_empty() {
        assert!(tabu_hint(&[]).is_empty(), "no history → no hint");
        let h = tabu_hint(&[("a rare name".into(), "a common one".into())]);
        assert!(h.contains("do NOT re-propose") && h.contains("rare name") && h.contains("common one"));
    }

    #[test]
    fn recompile_records_derives_emitted_from_prose() {
        let scenes = vec![(
            "lane".to_string(),
            "a foggy lane".to_string(),
            "a foggy lane, cinematic, volumetric light".to_string(),
        )];
        let (r, l) = recompile_records(&scenes).unwrap();
        let out = records_to_surface(&r, &l);
        assert!(out.contains("c/prose-"), "authored prose claim:\n{out}");
        assert!(out.contains("c/prompt-"), "emitted-prompt claim");
        assert!(out.contains("grounds:"), "the emitted prompt is DERIVED FROM the prose (grounds edge)");
        assert!(out.contains("volumetric light"), "carries the emitted prompt body");
        // Empty emitted → nothing recorded.
        assert!(recompile_records(&[("x".into(), "x".into(), "".into())]).unwrap().0.is_empty());
    }

    #[test]
    fn supersedes_links_recompile_drift() {
        // Same prose, two DIFFERENT emitted prompts — the enhancer reworded across recompiles.
        let prose = "a foggy lane".to_string();
        let (r1, l1) =
            recompile_records(&[("lane".into(), prose.clone(), "a foggy lane, v1".into())]).unwrap();
        let (r2, l2) =
            recompile_records(&[("lane".into(), prose.clone(), "a foggy lane, v2".into())]).unwrap();
        let (prior, newer) = (records_to_surface(&r1, &l1), records_to_surface(&r2, &l2));
        let edges = supersedes_edges(&newer, &prior);
        assert_eq!(edges.len(), 1, "the v2 emitted prompt SUPERSEDES v1 (same prose):\n{newer}");
        assert!(matches!(&edges[0], Record::Relation(rl) if rl.kind == RelKind::Supersedes));
        // An identical recompile supersedes nothing (same content-hash).
        assert!(supersedes_edges(&prior, &prior).is_empty(), "nothing supersedes itself");
    }

    #[test]
    fn resolved_vs_open_findings_split() {
        // Corpus: one applied fix (RESOLVES its risk finding) + one manual finding (OPEN, no fix).
        let applied = vec![("locomobile".to_string(), "traction engine".to_string(), "rare name".to_string())];
        let manual = vec!["split the over-stuffed scene".to_string()];
        let corpus = fixes_to_smysl(&applied, &manual).unwrap();
        let (resolved, open) = resolved_open_findings(&corpus);
        assert_eq!(resolved, vec!["rare name".to_string()], "the risk a fix grounds in is resolved");
        assert_eq!(open, vec!["split the over-stuffed scene".to_string()], "the manual finding stays open");
        let hint = findings_hint(&resolved, &open);
        assert!(hint.contains("ALREADY ADDRESSED") && hint.contains("rare name"));
        assert!(hint.contains("STILL OPEN") && hint.contains("over-stuffed"));
        assert!(findings_hint(&[], &[]).is_empty(), "no history → no hint");
    }

    #[test]
    fn trace_report_finds_phrase_and_shows_grounds() {
        // A fix corpus: a finding + a fix grounded in it. Tracing the changed word explains the WHY.
        let applied = vec![(
            "steam locomobile".to_string(),
            "steam traction engine".to_string(),
            "rare/invented object name — model will hallucinate".to_string(),
        )];
        let text = fixes_to_smysl(&applied, &[]).expect("builds");
        let parsed = smysl_core::surface::parse_surface(&text).expect("re-reads");
        let out = trace_report(&parsed.records, &parsed.labels, "traction engine");
        assert!(out.contains("c/fix-"), "locates the fix claim that carries the phrase:\n{out}");
        assert!(out.contains("← grounded in f/risk-"), "shows WHY — the risk it fixed:\n{out}");
        // A phrase in no unit → an honest empty answer.
        assert!(trace_report(&parsed.records, &parsed.labels, "unicorn").contains("no smysl unit"));
    }

    #[test]
    fn scene_records_registers_relate_only_components() {
        // engine/trailer are referenced ONLY via a relation (not in foreground/objects) — they must
        // still surface as claims, and the edge between them must be drawn.
        let scene = ResolvedScene {
            relations: vec![crate::compile::resolver::RelationSpec {
                a: "trailer".into(),
                a_desc: "a four-wheeled trailer of books".into(),
                verb: "behind".into(),
                b: "engine".into(),
                b_desc: "a steam traction engine".into(),
            }],
            ..Default::default()
        };
        let out = scene_to_smysl(&scene).expect("builds");
        assert!(out.contains("c/engine") && out.contains("c/trailer"), "relate-only components claim:\n{out}");
        assert!(out.contains("@rel"), "edge is drawn between them");
    }

    #[test]
    fn merge_records_unions_without_duplicating() {
        // Merging a corpus into a superset that already contains it adds nothing (content-hash dedup).
        let (mut recs, mut labels) = scene_records(&ResolvedScene {
            foreground: vec![("hero".into(), "a lone figure".into())],
            ..Default::default()
        })
        .unwrap();
        let before = recs.len();
        let (extra, extra_l) = scene_records(&ResolvedScene {
            foreground: vec![("hero".into(), "a lone figure".into())],
            ..Default::default()
        })
        .unwrap();
        merge_records(&mut recs, &mut labels, extra, extra_l);
        assert_eq!(recs.len(), before, "identical unit is deduped, not doubled");
    }

    // A stand-in token estimator for the packer tests: ~1 token per 4 chars (same shape as plakat's).
    fn est(s: &str) -> usize {
        s.len().div_ceil(4)
    }

    #[test]
    fn pack_prompt_trims_filler_to_fit_and_keeps_subject_and_weights() {
        let prompt =
            "a Victorian steam traction engine, (brass chimney:1.3), masterpiece, 8k, highly detailed, trending on artstation";
        let full = est(prompt);
        let budget = full - 6; // force a trim
        let r = pack_prompt(prompt, budget, &est);
        assert!(r.info.used as usize <= budget, "packed within budget");
        assert!(r.text.contains("traction engine"), "subject kept: {}", r.text);
        assert!(r.text.contains("(brass chimney:1.3)"), "weighted span kept: {}", r.text);
        assert!(r.fit_on_filler_alone(budget), "only filler dropped → model-free path taken");
        assert!(r.dropped_spans.iter().all(|(t, _)| {
            let l = t.to_lowercase();
            l.contains("8k") || l.contains("masterpiece") || l.contains("detailed") || l.contains("artstation")
        }), "dropped set is filler: {:?}", r.dropped_spans);
    }

    #[test]
    fn packs_to_records_records_budget_decision() {
        let packs = vec![ScenePack {
            scene: "lighthouse".into(),
            budget: 77,
            used: 75,
            family: "SD15".into(),
            dropped: vec![
                ("dramatic lighting".into(), DropReason::LowValue),
                ("intricate details".into(), DropReason::LowValue),
            ],
        }];
        let (r, l) = packs_to_records(&packs).expect("builds");
        let out = records_to_surface(&r, &l);
        assert!(out.contains("@observation o/pack-"), "budget pack is an observation:\n{out}");
        assert!(out.contains("77") && out.contains("SD15"), "records the model budget");
        assert!(out.contains("dramatic lighting"), "records what was cut");
        // A scene that dropped nothing produces no record.
        let empty = ScenePack { scene: "x".into(), budget: 77, used: 40, family: "SD15".into(), dropped: vec![] };
        assert!(packs_to_records(&[empty]).unwrap().0.is_empty(), "no drop → no observation");
    }

    #[test]
    fn pack_prompt_flags_essential_drop_for_llm_fallback() {
        // No filler — every span is content. Forcing a tight budget must drop ESSENTIALS (reason Budget),
        // so `fit_on_filler_alone` is false and the caller falls back to the LLM reword.
        let prompt = "a red barn, a tall oak tree, a winding river, a stone bridge, a flock of geese";
        let r = pack_prompt(prompt, est("a red barn, a tall oak tree"), &est);
        assert!(r.text.starts_with("a red barn"), "subject preserved");
        assert!(!r.fit_on_filler_alone(est("a red barn, a tall oak tree")), "essential drops → not filler-only");
        assert!(r.dropped_spans.iter().any(|(_, reason)| *reason == DropReason::Budget), "budget-reason drops recorded");
    }
}
