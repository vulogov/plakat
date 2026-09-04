//! COMPILE-2: the Tera template pre-pass (`templates` feature). Renders a
//! `.tera`/`.j2`/… input to a `prompts.txt` string, then the existing parser
//! takes over unchanged. Built-in `plakat.*` vars + `--var`/`--vars`/`--vars-env`
//! context, custom filters/functions, and `{% include %}`/`{% import %}` of
//! sibling template files.

use anyhow::{Context as _, Result, anyhow, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tera::{Context, Tera, Value};

use super::TemplateOpts;

const TEMPLATE_EXTS: &[&str] = &["tera", "j2", "jinja", "jinja2"];

/// Render `input` (the template) into a `prompts.txt` string.
pub fn render(input: &str, input_path: Option<&Path>, opts: &TemplateOpts) -> Result<String> {
    let mut tera = Tera::default();
    register_filters(&mut tera);
    register_functions(&mut tera, input_path);

    // Load sibling template files so {% include %}/{% import %} resolve by name.
    if let Some(dir) = input_path.and_then(|p| p.parent()) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let p = e.path();
                let is_tpl = p
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|s| TEMPLATE_EXTS.contains(&s.to_ascii_lowercase().as_str()))
                    .unwrap_or(false);
                if is_tpl && Some(p.as_path()) != input_path {
                    if let (Some(name), Ok(content)) =
                        (p.file_name().and_then(|n| n.to_str()), std::fs::read_to_string(&p))
                    {
                        let _ = tera.add_raw_template(name, &content);
                    }
                }
            }
        }
    }

    let main = input_path.and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or("<input>");
    tera.add_raw_template(main, input)
        .map_err(|e| anyhow!("template parse failed ({main}):\n  {}", fmt_err(&e)))?;

    let ctx = build_context(opts, input_path)?;
    tera.render(main, &ctx)
        .map_err(|e| anyhow!("template render failed ({main}):\n  {}", fmt_err(&e)))
}

/// Flatten a Tera error + its `source` chain (parse errors carry the line there).
fn fmt_err(e: &tera::Error) -> String {
    let mut s = e.to_string();
    let mut src = std::error::Error::source(e);
    while let Some(e) = src {
        s.push_str(&format!("\n  caused by: {e}"));
        src = e.source();
    }
    s
}

/// Slugify to a scene id: lowercase, non-alphanumeric runs → one underscore.
fn slug(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut us = false;
    for c in s.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
            us = false;
        } else if !us {
            out.push('_');
            us = true;
        }
    }
    out.trim_matches('_').to_string()
}

fn build_context(opts: &TemplateOpts, input_path: Option<&Path>) -> Result<Context> {
    let mut ctx = Context::new();

    // 1) built-in plakat.* (lowest precedence).
    let mut plakat = serde_json::Map::new();
    plakat.insert("version".into(), Value::String(env!("CARGO_PKG_VERSION").into()));
    if let Some(p) = input_path {
        plakat.insert("input_path".into(), Value::String(p.display().to_string()));
        if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
            plakat.insert("input_stem".into(), Value::String(stem.into()));
        }
    }
    ctx.insert("plakat", &Value::Object(plakat));

    // 2) --vars files (JSON or TOML), later files win.
    for f in &opts.vars_files {
        let text = std::fs::read_to_string(f).with_context(|| format!("reading --vars {}", f.display()))?;
        let val: Value = if f.extension().and_then(|e| e.to_str()).map(|s| s.eq_ignore_ascii_case("toml")) == Some(true) {
            let t: toml::Value = toml::from_str(&text).with_context(|| format!("parsing TOML --vars {}", f.display()))?;
            // toml::Value → serde_json::Value (datetimes serialize as strings — OQ-TEMPLATE-3).
            serde_json::to_value(t).with_context(|| format!("converting TOML --vars {}", f.display()))?
        } else {
            serde_json::from_str(&text).with_context(|| format!("parsing JSON --vars {}", f.display()))?
        };
        match val {
            Value::Object(map) => {
                for (k, v) in map {
                    ctx.insert(&k, &v);
                }
            }
            _ => bail!("--vars {} must be a JSON/TOML object at the top level", f.display()),
        }
    }

    // 3) --vars-env PREFIX (strip prefix, lowercase key).
    for prefix in &opts.env_prefixes {
        for (k, v) in std::env::vars() {
            if let Some(rest) = k.strip_prefix(prefix) {
                if !rest.is_empty() {
                    ctx.insert(rest.to_ascii_lowercase(), &Value::String(v));
                }
            }
        }
    }

    // 4) --var KEY=VALUE (highest precedence).
    for (k, v) in &opts.vars {
        ctx.insert(k, &Value::String(v.clone()));
    }
    Ok(ctx)
}

fn register_filters(tera: &mut Tera) {
    tera.register_filter("scene_name", |v: &Value, _: &HashMap<String, Value>| {
        let s = v.as_str().ok_or_else(|| tera::Error::msg("scene_name: expected a string"))?;
        Ok(Value::String(slug(s)))
    });
    tera.register_filter("prompt_join", |v: &Value, _: &HashMap<String, Value>| {
        let arr = v.as_array().ok_or_else(|| tera::Error::msg("prompt_join: expected an array"))?;
        let parts: Vec<String> = arr
            .iter()
            .map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string()))
            .collect();
        Ok(Value::String(parts.join(", ")))
    });
    tera.register_filter("prompt_clean", |v: &Value, _: &HashMap<String, Value>| {
        let s = v.as_str().ok_or_else(|| tera::Error::msg("prompt_clean: expected a string"))?;
        Ok(Value::String(super::assembler::clean(s)))
    });
    tera.register_filter("zero_pad", |v: &Value, args: &HashMap<String, Value>| {
        let n = args.get("n").and_then(|x| x.as_u64()).unwrap_or(2) as usize;
        let num = v.as_i64().ok_or_else(|| tera::Error::msg("zero_pad: expected an integer"))?;
        Ok(Value::String(format!("{num:0width$}", width = n)))
    });
    tera.register_filter("sentence_case", |v: &Value, _: &HashMap<String, Value>| {
        let s = v.as_str().ok_or_else(|| tera::Error::msg("sentence_case: expected a string"))?;
        let mut c = s.chars();
        let out = match c.next() {
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            None => String::new(),
        };
        Ok(Value::String(out))
    });
}

fn register_functions(tera: &mut Tera, input_path: Option<&Path>) {
    let base: Option<PathBuf> = input_path.and_then(|p| p.parent()).map(|p| p.to_path_buf());
    tera.register_function("include_raw", move |args: &HashMap<String, Value>| {
        let rel = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| tera::Error::msg("include_raw(path=\"…\") requires a path"))?;
        let full = match &base {
            Some(b) => b.join(rel),
            None => PathBuf::from(rel),
        };
        let content = std::fs::read_to_string(&full)
            .map_err(|e| tera::Error::msg(format!("include_raw: reading {}: {e}", full.display())))?;
        Ok(Value::String(content))
    });
    tera.register_function("scene_separator", |args: &HashMap<String, Value>| {
        let title = args.get("title").and_then(|v| v.as_str()).unwrap_or("");
        let dashes = "─".repeat(40usize.saturating_sub(title.chars().count() + 2).max(4));
        Ok(Value::String(format!("# ── {title} {dashes}")))
    });
    tera.register_function("model_family", |args: &HashMap<String, Value>| {
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let fam = match super::classify_model(name) {
            super::ModelFamily::Sd15 => "sd15",
            super::ModelFamily::Sdxl => "sdxl",
            super::ModelFamily::Sd3 => "sd3",
            super::ModelFamily::Cascade => "cascade",
            super::ModelFamily::Flux => "flux",
            super::ModelFamily::Unknown => "unknown",
        };
        Ok(Value::String(fam.into()))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(vars: &[(&str, &str)]) -> TemplateOpts {
        TemplateOpts {
            vars: vars.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn renders_loop_with_trim_markers() {
        let t = "model: {{ model }}\n{%- for c in chars %}\n\nname: {{ c | scene_name }}\n{{ c }} stands ready.\n{%- endfor %}\n";
        let mut o = opts(&[("model", "sdxl")]);
        o.vars_files = vec![];
        // chars via a --var won't be an array; use a vars file instead.
        let dir = std::env::temp_dir();
        let vf = dir.join("plakat-tpl-test-vars.json");
        std::fs::write(&vf, r#"{ "chars": ["Lady Mireth", "Bob"] }"#).unwrap();
        o.vars_files = vec![vf.clone()];
        let out = render(t, None, &o).unwrap();
        std::fs::remove_file(&vf).ok();
        // No spurious blank line splitting the block; scene_name slugifies.
        assert!(out.contains("name: lady_mireth"), "{out}");
        assert!(out.contains("Lady Mireth stands ready."), "{out}");
        assert!(out.contains("name: bob"), "{out}");
        // Parses as a valid prompts.txt afterwards.
        let doc = super::super::parser::parse(&out).unwrap();
        assert_eq!(doc.scenes.len(), 2);
    }

    #[test]
    fn filters_and_functions() {
        let t = "{{ \"Hello World\" | scene_name }}|{{ [\"a\",\"b\"] | prompt_join }}|{{ 3 | zero_pad(n=2) }}|{{ model_family(name=\"flux-dev\") }}";
        let out = render(t, None, &opts(&[])).unwrap();
        assert_eq!(out, "hello_world|a, b|03|flux");
    }

    #[test]
    fn var_precedence_over_builtin_and_files() {
        let out = render("{{ model }}", None, &opts(&[("model", "from-var")])).unwrap();
        assert_eq!(out.trim(), "from-var");
    }

    #[test]
    fn template_error_is_reported() {
        let err = render("{{ unclosed ", None, &opts(&[])).unwrap_err();
        assert!(err.to_string().contains("template parse failed"), "{err}");
    }
}
