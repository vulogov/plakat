//! Read a compiled `scenario` HJSON back for `--decompile` (E-C1: scenario →
//! prompts.txt, the inverse of compile) and `--diff` (compare a fresh compile
//! against an existing scenario). A minimal deserialize view — just the fields
//! compile emits.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize, Default)]
struct ScnTask {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    negative: Option<String>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    count: Option<u32>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    steps: Option<usize>,
    // C3 (FACESWAP-4): preserve spec-task kind on decompile so the round-trip keeps the task type.
    #[serde(rename = "type", default)]
    task_type: Option<String>,
    #[serde(default)]
    faceswap: Option<FaceswapBlock>,
    // D4 (6.22.0): capture the spec-task blocks the compiler authors, for a lossless round-trip.
    #[serde(default)]
    texture: Option<TextureBlock>,
    #[serde(default)]
    product: Option<SpecFileBlock>,
    #[serde(default)]
    comic: Option<SpecFileBlock>,
    #[serde(default)]
    fractal: Option<FractalBlock>,
}

/// The `faceswap: { … }` sub-block of a task (for a lossless faceswap round-trip).
#[derive(Deserialize, Default)]
struct FaceswapBlock {
    #[serde(default)]
    scene: Option<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    face: Option<usize>,
}

/// A `{ spec_file: "…" }` block (product / comic when authored via a spec file).
#[derive(Deserialize, Default)]
struct SpecFileBlock {
    #[serde(default)]
    spec_file: Option<String>,
}

/// The compiler-authored `texture: { spec: { material, from_image, seamless } }` fields.
#[derive(Deserialize, Default)]
struct TextureBlock {
    #[serde(default)]
    spec: Option<TextureSpecInner>,
}
#[derive(Deserialize, Default)]
struct TextureSpecInner {
    #[serde(default)]
    from_image: Option<String>,
    #[serde(default)]
    seamless: Option<SeamlessInner>,
}
#[derive(Deserialize, Default)]
struct SeamlessInner {
    #[serde(default)]
    mode: Option<String>,
}

/// The compiler-authored `fractal: { spec, kind, palette }` fields.
#[derive(Deserialize, Default)]
struct FractalBlock {
    #[serde(default)]
    spec: Option<String>,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    palette: Option<String>,
}

#[derive(Deserialize, Default)]
struct Scn {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    steps: Option<usize>,
    #[serde(default)]
    guidance: Option<f64>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    negative: Option<String>,
    #[serde(default)]
    loras: Vec<String>,
    #[serde(default)]
    tasks: Vec<ScnTask>,
}

fn parse(hjson: &str) -> Result<Scn> {
    deser_hjson::from_str(hjson).context("parsing scenario HJSON")
}

/// E-C1: scenario HJSON → a `prompts.txt` string. Header/footer can't be
/// recovered (they were folded into the prompt), so each task's prompt becomes
/// the block's free text — ready to re-edit and re-compile.
pub fn decompile(hjson: &str) -> Result<String> {
    let s = parse(hjson)?;
    let mut o = String::new();
    o.push_str("# Decompiled from a scenario by `plakat compile --decompile`.\n");
    o.push_str("# The prompt is now free text; re-add header:/footer:/style: as you like.\n");

    // Global block (commands only).
    let mut any_global = false;
    if let Some(m) = &s.model {
        o.push_str(&format!("model: {m}\n"));
        any_global = true;
    }
    if let Some(v) = &s.size {
        o.push_str(&format!("size: {v}\n"));
        any_global = true;
    }
    if let Some(v) = s.steps {
        o.push_str(&format!("steps: {v}\n"));
        any_global = true;
    }
    if let Some(v) = s.guidance {
        o.push_str(&format!("guidance: {v}\n"));
        any_global = true;
    }
    if let Some(v) = s.seed {
        o.push_str(&format!("seed: {v}\n"));
        any_global = true;
    }
    for l in &s.loras {
        o.push_str(&format!("lora: {l}\n"));
        any_global = true;
    }
    if let Some(v) = &s.negative {
        o.push_str(&format!("negative: {v}\n"));
        any_global = true;
    }
    let _ = any_global;

    // One block per task.
    for (i, t) in s.tasks.iter().enumerate() {
        o.push('\n');
        if let Some(n) = &t.name {
            o.push_str(&format!("name: {n}\n"));
        }
        if let Some(p) = &t.prompt {
            o.push_str(p.trim());
            o.push('\n');
        } else if t.task_type.is_none() {
            o.push_str(&format!("scene {}\n", i + 1));
        }
        // C3/D4: preserve the task type + spec directives so a spec-task round-trips (no bogus prompt).
        if let Some(tt) = &t.task_type {
            o.push_str(&format!("type: {tt}\n"));
            if let Some(fb) = &t.faceswap {
                if let Some(v) = &fb.scene {
                    o.push_str(&format!("faceswap-scene: {v}\n"));
                }
                if let Some(v) = &fb.source {
                    o.push_str(&format!("faceswap-source: {v}\n"));
                }
                if let Some(v) = fb.face {
                    o.push_str(&format!("faceswap-face: {v}\n"));
                }
            }
            if let Some(tx) = t.texture.as_ref().and_then(|b| b.spec.as_ref()) {
                if let Some(v) = &tx.from_image {
                    o.push_str(&format!("texture-from: {v}\n"));
                }
                if let Some(v) = tx.seamless.as_ref().and_then(|s| s.mode.as_ref()) {
                    o.push_str(&format!("texture-seamless: {v}\n"));
                }
            }
            if let Some(v) = t.product.as_ref().and_then(|b| b.spec_file.as_ref()) {
                o.push_str(&format!("product-spec-file: {v}\n"));
            }
            if let Some(v) = t.comic.as_ref().and_then(|b| b.spec_file.as_ref()) {
                o.push_str(&format!("comic-spec-file: {v}\n"));
            }
            if let Some(fr) = &t.fractal {
                if let Some(v) = &fr.spec {
                    o.push_str(&format!("fractal-spec: {v}\n"));
                }
                if let Some(v) = &fr.kind {
                    o.push_str(&format!("fractal-kind: {v}\n"));
                }
                if let Some(v) = &fr.palette {
                    o.push_str(&format!("fractal-palette: {v}\n"));
                }
            }
        }
        if let Some(v) = &t.negative {
            o.push_str(&format!("negative: {v}\n"));
        }
        if let Some(v) = t.seed {
            o.push_str(&format!("seed: {v}\n"));
        }
        if let Some(v) = t.count {
            o.push_str(&format!("count: {v}\n"));
        }
        if let Some(v) = &t.size {
            o.push_str(&format!("size: {v}\n"));
        }
        if let Some(v) = t.steps {
            o.push_str(&format!("steps: {v}\n"));
        }
    }
    Ok(o)
}

fn task_map(s: &Scn) -> BTreeMap<String, (String, String)> {
    s.tasks
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let name = t.name.clone().unwrap_or_else(|| format!("task_{i}"));
            (name, (t.prompt.clone().unwrap_or_default(), t.negative.clone().unwrap_or_default()))
        })
        .collect()
}

/// Compare a freshly-compiled scenario against an existing one: which task names
/// are new (`+`), changed (`~`, prompt or negative differs), or removed (`-`).
pub fn diff(new_hjson: &str, existing_hjson: &str) -> Result<String> {
    let old = task_map(&parse(existing_hjson)?);
    let new = task_map(&parse(new_hjson)?);
    let mut out = String::new();
    for (k, v) in &new {
        match old.get(k) {
            None => out.push_str(&format!("+ {k}  (new)\n")),
            Some(ov) if ov != v => out.push_str(&format!("~ {k}  (changed)\n")),
            Some(_) => {}
        }
    }
    for k in old.keys() {
        if !new.contains_key(k) {
            out.push_str(&format!("- {k}  (removed)\n"));
        }
    }
    if out.is_empty() {
        out.push_str("(no changes)\n");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCN: &str = r#"{ model: "sdxl", steps: 30, negative: "blurry"
        scene: [ { name: "plain", prompt: "" } ]
        weather: [ { name: "any", prompt: "" } ]
        tasks: [ { name: "a", prompt: "a tundra", negative: "blurry", seed: 7 } ] }"#;

    #[test]
    fn decompile_roundtrips_global_and_task() {
        let txt = decompile(SCN).unwrap();
        assert!(txt.contains("model: sdxl"));
        assert!(txt.contains("steps: 30"));
        assert!(txt.contains("name: a"));
        assert!(txt.contains("a tundra"), "prompt becomes free text");
        assert!(txt.contains("seed: 7"));
        // The decompiled text must itself parse as a prompts.txt.
        let doc = crate::compile::parser::parse(&txt).unwrap();
        assert_eq!(doc.scenes.len(), 1);
    }

    #[test]
    fn diff_reports_add_change_remove() {
        let new = r#"{ tasks: [ { name: "a", prompt: "x2" }, { name: "c", prompt: "new" } ] }"#;
        let old = r#"{ tasks: [ { name: "a", prompt: "x1" }, { name: "b", prompt: "gone" } ] }"#;
        let d = diff(new, old).unwrap();
        assert!(d.contains("~ a"), "{d}");
        assert!(d.contains("+ c"), "{d}");
        assert!(d.contains("- b"), "{d}");
    }
}
