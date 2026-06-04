//! `plakat gallery DIR` — build a Markdown gallery index from a
//! directory of plakat-generated PNGs.
//!
//! v0.43 (proof-corpus cycle): plakat's generated PNGs are
//! self-documenting — each carries its full recipe in an embedded
//! `parameters` tEXt chunk (and a JSON sidecar). This command reads
//! that metadata back and emits a browsable gallery: a thumbnail grid
//! plus per-image prompt + settings. It dogfoods the metadata round-trip
//! (proving the "self-documenting output" claim) and retires the
//! hand-maintained gallery README.
//!
//! ```bash
//! plakat gallery corpus/            # → corpus/README.md
//! plakat gallery gallery/ --title "plakat gallery" --cols 3
//! ```

use anyhow::{Context, Result};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};

use crate::imaging::metadata::GenerationMetadata;

#[derive(clap::Args, Debug)]
pub struct GalleryArgs {
    /// Directory of plakat-generated PNGs to index.
    pub dir: PathBuf,

    /// Output Markdown file. Defaults to `<dir>/README.md`.
    #[arg(long, value_name = "OUT")]
    pub out: Option<PathBuf>,

    /// Gallery title (the H1 heading). Defaults to "plakat gallery".
    #[arg(long, value_name = "TEXT")]
    pub title: Option<String>,

    /// Thumbnails per row in the grid.
    #[arg(long, default_value_t = 3, value_name = "N")]
    pub cols: usize,

    /// Recurse into subdirectories (paths in the index stay relative
    /// to the output file).
    #[arg(long, default_value_t = false)]
    pub recursive: bool,
}

pub async fn run(args: GalleryArgs) -> Result<()> {
    anyhow::ensure!(
        args.dir.is_dir(),
        "gallery: {} is not a directory",
        args.dir.display()
    );
    let cols = args.cols.max(1);

    let mut pngs = collect_pngs(&args.dir, args.recursive)
        .with_context(|| format!("scanning {}", args.dir.display()))?;
    anyhow::ensure!(
        !pngs.is_empty(),
        "gallery: no .png files found in {}",
        args.dir.display()
    );
    pngs.sort_by(|a, b| natural_cmp(&file_name(a), &file_name(b)));

    let out = args
        .out
        .clone()
        .unwrap_or_else(|| args.dir.join("README.md"));
    let title = args.title.clone().unwrap_or_else(|| "plakat gallery".to_string());

    let entries: Vec<(PathBuf, Option<GenerationMetadata>)> =
        pngs.iter().map(|p| (p.clone(), read_meta(p))).collect();

    let md = render_markdown(&title, &out, &entries, cols);
    std::fs::write(&out, md).with_context(|| format!("writing {}", out.display()))?;

    let n_meta = entries.iter().filter(|(_, m)| m.is_some()).count();
    println!(
        "✓ gallery: indexed {} image(s) ({} with embedded metadata) → {}",
        entries.len(),
        n_meta,
        out.display()
    );
    Ok(())
}

/// Read a PNG's generation metadata: JSON sidecar first (full,
/// structured), then the embedded A1111 `parameters` chunk (best-effort
/// — works on Civitai / A1111 outputs too). `None` if neither is present.
fn read_meta(png: &Path) -> Option<GenerationMetadata> {
    let sidecar = png.with_extension("json");
    if let Ok(s) = std::fs::read_to_string(&sidecar) {
        if let Ok(m) = serde_json::from_str::<GenerationMetadata>(&s) {
            return Some(m);
        }
    }
    match crate::imaging::io::read_parameters_chunk(png) {
        Ok(Some(text)) => crate::cli::clone::parse_a1111(&text),
        _ => None,
    }
}

fn collect_pngs(dir: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            if recursive {
                out.extend(collect_pngs(&path, true)?);
            }
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("png"))
        {
            out.push(path);
        }
    }
    Ok(out)
}

fn render_markdown(
    title: &str,
    out: &Path,
    entries: &[(PathBuf, Option<GenerationMetadata>)],
    cols: usize,
) -> String {
    let base = out.parent().unwrap_or_else(|| Path::new("."));
    let mut s = String::new();
    s.push_str(&format!("# {title}\n\n"));
    s.push_str(&format!(
        "{} images generated with **plakat**. Each is self-documenting — its \
         generation parameters are embedded in the PNG. This index was built by \
         `plakat gallery`.\n\n",
        entries.len()
    ));

    // Thumbnail grid.
    let width_pct = 100 / cols;
    s.push_str("<table>\n");
    for row in entries.chunks(cols) {
        s.push_str("  <tr>\n");
        for (p, m) in row {
            let rel = rel_path(p, base);
            let alt = m
                .as_ref()
                .map(|m| escape_attr(&m.prompt))
                .unwrap_or_default();
            s.push_str(&format!(
                "    <td width=\"{width_pct}%\"><img src=\"{rel}\" alt=\"{alt}\"></td>\n"
            ));
        }
        s.push_str("  </tr>\n");
    }
    s.push_str("</table>\n\n");

    // Per-image details.
    s.push_str("## Images\n\n");
    for (p, m) in entries {
        let rel = rel_path(p, base);
        let name = file_name(p);
        s.push_str(&format!("### {name}\n\n"));
        s.push_str(&format!("![{name}]({rel})\n\n"));
        match m {
            Some(m) => {
                if !m.prompt.trim().is_empty() {
                    s.push_str(&format!("> {}\n\n", m.prompt.replace('\n', "\n> ")));
                }
                if !m.negative.is_empty() {
                    s.push_str(&format!("Negative: `{}`\n\n", m.negative));
                }
                s.push_str(&format!("{}\n\n", settings_line(m)));
            }
            None => s.push_str("_(no embedded metadata)_\n\n"),
        }
    }

    s.push_str("---\n\n_Generated by `plakat gallery`._\n");
    s
}

/// One-line settings summary from a metadata record.
fn settings_line(m: &GenerationMetadata) -> String {
    let mut parts: Vec<String> = vec![format!("`{}`", m.model)];
    if let Some(mode) = m.mode.as_deref() {
        if !mode.is_empty() && mode != "t2i" {
            parts.push(mode.to_string());
        }
    }
    if let Some(st) = m.strength {
        parts.push(format!("strength {}", fmt_f(st as f64)));
    }
    if let Some(c) = control_summary(m) {
        parts.push(c);
    }
    if let Some(l) = lora_summary(m) {
        parts.push(l);
    }
    if let Some(look) = &m.look {
        parts.push(format!("look: {look}"));
    }
    if let Some(genre) = &m.genre {
        parts.push(format!("genre: {genre}"));
    }
    parts.push(format!("{}×{}", m.width, m.height));
    parts.push(format!("{} steps", m.steps));
    parts.push(format!("CFG {}", fmt_f(m.guidance)));
    parts.push(format!("seed {}", m.seed));
    parts.join(" · ")
}

fn control_summary(m: &GenerationMetadata) -> Option<String> {
    if let Some(stack) = &m.control_stack {
        if !stack.is_empty() {
            let kinds: Vec<&str> = stack.iter().map(|c| c.kind.as_str()).collect();
            return Some(format!("ControlNet: {}", kinds.join(", ")));
        }
    }
    if !m.controls.is_empty() {
        return Some(format!("ControlNet: {}", m.controls.join(", ")));
    }
    None
}

fn lora_summary(m: &GenerationMetadata) -> Option<String> {
    if let Some(stack) = &m.lora_stack {
        if !stack.is_empty() {
            let l: Vec<String> = stack
                .iter()
                .map(|e| format!("{}:{}", e.display, fmt_f(e.scale as f64)))
                .collect();
            return Some(format!("LoRA: {}", l.join(", ")));
        }
    }
    if !m.loras.is_empty() {
        return Some(format!("LoRA: {}", m.loras.join(", ")));
    }
    None
}

/// Format an f64 without trailing `.0` (`4.0` → `4`, `3.5` → `3.5`).
fn fmt_f(x: f64) -> String {
    format!("{x}")
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Path of `p` relative to `base` (the output file's directory),
/// using forward slashes. Falls back to the file name when `p` isn't
/// under `base`.
fn rel_path(p: &Path, base: &Path) -> String {
    let rel = p.strip_prefix(base).unwrap_or_else(|_| Path::new("")).to_path_buf();
    let rel = if rel.as_os_str().is_empty() {
        Path::new(&file_name(p)).to_path_buf()
    } else {
        rel
    };
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn escape_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Natural (human) ordering: digit runs compare numerically, so
/// `2.png` sorts before `10.png`. Non-digit runs compare bytewise,
/// case-insensitively.
fn natural_cmp(a: &str, b: &str) -> Ordering {
    let (mut ai, mut bi) = (a.bytes().peekable(), b.bytes().peekable());
    loop {
        match (ai.peek().copied(), bi.peek().copied()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(ca), Some(cb)) => {
                if ca.is_ascii_digit() && cb.is_ascii_digit() {
                    let na = take_number(&mut ai);
                    let nb = take_number(&mut bi);
                    match na.cmp(&nb) {
                        Ordering::Equal => {}
                        ord => return ord,
                    }
                } else {
                    let la = ca.to_ascii_lowercase();
                    let lb = cb.to_ascii_lowercase();
                    match la.cmp(&lb) {
                        Ordering::Equal => {}
                        ord => return ord,
                    }
                    ai.next();
                    bi.next();
                }
            }
        }
    }
}

fn take_number(it: &mut std::iter::Peekable<std::str::Bytes<'_>>) -> u64 {
    let mut n: u64 = 0;
    while let Some(&c) = it.peek() {
        if c.is_ascii_digit() {
            n = n.saturating_mul(10).saturating_add((c - b'0') as u64);
            it.next();
        } else {
            break;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> GenerationMetadata {
        GenerationMetadata::new("a fox", "stable-cascade", 42, 20, 4.0, "default", 1024, 1024)
    }

    #[test]
    fn natural_cmp_orders_numbers_humanly() {
        let mut v = vec![
            "10.png".to_string(),
            "2.png".to_string(),
            "1.png".to_string(),
            "9.png".to_string(),
        ];
        v.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(v, vec!["1.png", "2.png", "9.png", "10.png"]);
    }

    #[test]
    fn natural_cmp_mixes_text_and_numbers() {
        let mut v = vec![
            "img_10.png".to_string(),
            "img_2.png".to_string(),
            "alpha.png".to_string(),
        ];
        v.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(v, vec!["alpha.png", "img_2.png", "img_10.png"]);
    }

    #[test]
    fn settings_line_basic() {
        let line = settings_line(&meta());
        assert!(line.contains("`stable-cascade`"));
        assert!(line.contains("1024×1024"));
        assert!(line.contains("20 steps"));
        assert!(line.contains("CFG 4"));
        assert!(line.contains("seed 42"));
        // No CN / LoRA / img2img on a plain t2i record.
        assert!(!line.contains("ControlNet"));
        assert!(!line.contains("LoRA"));
    }

    #[test]
    fn settings_line_surfaces_controlnet_and_lora() {
        let mut m = meta();
        m.controls = vec!["canny".to_string()];
        m.loras = vec!["anime:1".to_string()];
        m.mode = Some("img2img".to_string());
        m.strength = Some(0.6);
        let line = settings_line(&m);
        assert!(line.contains("ControlNet: canny"));
        assert!(line.contains("LoRA: anime:1"));
        assert!(line.contains("img2img"));
        assert!(line.contains("strength 0.6"));
    }

    #[test]
    fn rel_path_forward_slashes_and_strips_base() {
        let base = Path::new("/tmp/corpus");
        assert_eq!(rel_path(Path::new("/tmp/corpus/a.png"), base), "a.png");
        assert_eq!(
            rel_path(Path::new("/tmp/corpus/cn/b.png"), base),
            "cn/b.png"
        );
    }

    #[test]
    fn escape_attr_handles_quotes_and_angles() {
        assert_eq!(escape_attr(r#"a "b" <c>"#), "a &quot;b&quot; &lt;c&gt;");
    }
}
