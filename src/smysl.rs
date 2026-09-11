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

/// 6.30.0 Phase 1: resolved scene → smysl surface text. Each foreground / object / background component
/// becomes a `@claim`; each relate between two named components becomes a `@rel`; the free-text scene
/// description becomes one prose claim. Written in the readable surface syntax — the foundation the
/// analyze/fix provenance corpus (Phase 2+) accumulates onto.
pub fn scene_to_smysl(scene: &ResolvedScene) -> anyhow::Result<String> {
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
    // Relations between named components (attach-to-props like a basket aren't registered → skipped).
    for rel in &scene.relations {
        if let (Some(&from), Some(&to)) = (uid_by_name.get(&rel.a), uid_by_name.get(&rel.b)) {
            records.push(Record::Relation(Relation::new(rel_kind(&rel.verb), from, to)));
        }
    }

    let ctx = WriteContext::from_labels(&labels);
    Ok(write_surface(None, &records, &ctx))
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
}
