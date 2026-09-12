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
    canonical_uid, KernelType, Label, RelKind, Record, Relation, Status, Uid, UnitCoreBuilder,
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

/// 6.30.0 Phase 2: record the `--analyze`/`--fix` decisions as a smysl document. Each APPLIED fix becomes a
/// `@claim` GROUNDED in the `@finding` it addressed (the risk), so the corpus carries "this edit exists
/// because of this risk" — the provenance the loop accumulates. Un-auto-fixable structural items become
/// standalone `@finding`s (open risks for the author). Returns the surface text.
///
/// `applied` = `(old, new, why)` for each edit that landed; `manual` = the structural notes.
pub fn fixes_to_smysl(applied: &[(String, String, String)], manual: &[String]) -> anyhow::Result<String> {
    let mut records: Vec<Record> = Vec::new();
    let mut labels: BTreeMap<Label, Uid> = BTreeMap::new();

    for (i, (old, new, why)) in applied.iter().enumerate() {
        // The risk the critic found → a finding (a prediction ⇒ Speculative at this layer).
        let finding = UnitCoreBuilder::new(KernelType::Finding, why.trim(), Status::Speculative)
            .build()
            .map_err(|e| anyhow::anyhow!("smysl finding {i}: {e:?}"))?;
        let f_uid = canonical_uid(&finding);
        labels.insert(Label::new(&format!("f/risk-{i}"))?, f_uid);
        records.push(Record::Unit(finding));

        // The applied fix → a claim DERIVED from (grounded in) that finding: the provenance edge.
        let gist = format!("replace \u{201C}{}\u{201D} with \u{201C}{}\u{201D}", clip(old), clip(new));
        let fix = UnitCoreBuilder::new(KernelType::Claim, gist, Status::Derived)
            .body(format!("was: {}\nnow: {}", old.trim(), new.trim()))
            .grounds([f_uid])
            .build()
            .map_err(|e| anyhow::anyhow!("smysl fix {i}: {e:?}"))?;
        labels.insert(Label::new(&format!("c/fix-{i}"))?, canonical_uid(&fix));
        records.push(Record::Unit(fix));
    }
    for (i, m) in manual.iter().enumerate() {
        let f = UnitCoreBuilder::new(KernelType::Finding, m.trim(), Status::Speculative)
            .build()
            .map_err(|e| anyhow::anyhow!("smysl manual {i}: {e:?}"))?;
        labels.insert(Label::new(&format!("f/manual-{i}"))?, canonical_uid(&f));
        records.push(Record::Unit(f));
    }
    Ok(write_surface(None, &records, &WriteContext::from_labels(&labels)))
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
        // Prefix tells the layer: c/ = authored claim / applied fix, f/ = finding (a risk).
        let kind = match name(&uid).split('/').next() {
            Some("f") => "finding",
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
        assert!(out.contains("@finding f/risk-0"), "the risk is a finding:\n{out}");
        assert!(out.contains("@claim c/fix-0"), "the fix is a claim");
        assert!(out.contains("grounds: [f/risk-0]"), "the fix is GROUNDED in its finding (provenance edge)");
        assert!(out.contains("@finding f/manual-0"), "structural item is an open finding");
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
        assert!(out.contains("c/fix-0"), "locates the fix claim that carries the phrase:\n{out}");
        assert!(out.contains("grounded in f/risk-0"), "shows WHY — the risk it fixed:\n{out}");
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
}
