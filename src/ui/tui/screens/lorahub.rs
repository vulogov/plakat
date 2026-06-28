//! LoRA Hub screen (RFC TUI-1 §10). Release 3 LOCAL tab: scan the workspace + global
//! LoRA dirs for `.safetensors`, read each file's safetensors header to infer its
//! base family + rank (and a `.plakat.hjson` sidecar for trigger words / notes), and
//! flag compatibility against the currently-loaded model. The CIVITAI / HUGGINGFACE
//! search tabs and the LLM recommendation features are later releases.

use std::collections::HashMap;
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

use crate::preset::discovery::BaseFamily;

/// One local LoRA file.
struct LoraInfo {
    path: PathBuf,
    name: String,
    family: Option<BaseFamily>,
    rank: Option<u32>,
    size: u64,
    triggers: Vec<String>,
    notes: String,
    /// A short label for which dir it came from.
    location: String,
}

/// `.plakat.hjson` sidecar fields we surface (the full schema is RFC §10).
#[derive(Deserialize, Default)]
struct Sidecar {
    #[serde(default)]
    trigger_words: Vec<String>,
    #[serde(default)]
    base_model: String,
    #[serde(default)]
    notes: String,
}

pub struct LoraHubState {
    dirs: Vec<(PathBuf, String)>,
    loras: Vec<LoraInfo>,
    selected: usize,
    /// Family of the currently-loaded model (set by the App), for compatibility.
    loaded_family: Option<BaseFamily>,
}

impl LoraHubState {
    pub fn new(dirs: Vec<(PathBuf, String)>) -> Self {
        let mut s = Self { dirs, loras: Vec::new(), selected: 0, loaded_family: None };
        s.rescan();
        s
    }

    /// The App calls this each tick with the loaded model's family so the
    /// compatibility column reflects what's actually loaded.
    pub fn set_loaded_family(&mut self, family: Option<BaseFamily>) {
        self.loaded_family = family;
    }

    pub fn rescan(&mut self) {
        let mut loras = Vec::new();
        for (dir, label) in &self.dirs {
            collect(dir, label, &mut loras, 0);
        }
        loras.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        self.loras = loras;
        if self.selected >= self.loras.len() {
            self.selected = self.loras.len().saturating_sub(1);
        }
    }

    fn next(&mut self) {
        if !self.loras.is_empty() {
            self.selected = (self.selected + 1) % self.loras.len();
        }
    }

    fn prev(&mut self) {
        if !self.loras.is_empty() {
            self.selected = (self.selected + self.loras.len() - 1) % self.loras.len();
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.next(),
            KeyCode::Up | KeyCode::Char('k') => self.prev(),
            KeyCode::Char('r') => self.rescan(),
            _ => {}
        }
    }

    /// Compatibility of a LoRA vs the loaded model: Some(true)=match, Some(false)=
    /// mismatch, None=can't tell (unknown family, or no model loaded).
    fn compatible(&self, l: &LoraInfo) -> Option<bool> {
        match (l.family, self.loaded_family) {
            (Some(a), Some(b)) => Some(a == b),
            _ => None,
        }
    }

    pub fn render(&self, f: &mut Frame, area: Rect) {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
            .split(area);
        self.render_list(f, cols[0]);
        self.render_detail(f, cols[1]);
    }

    fn render_list(&self, f: &mut Frame, area: Rect) {
        let loaded = self
            .loaded_family
            .map(family_label)
            .map(|s| format!(" vs {s} "))
            .unwrap_or_else(|| " no model loaded ".into());
        let block = Block::default()
            .borders(Borders::ALL)
            .title(format!(" LoRA · LOCAL ({}) ·{loaded}", self.loras.len()));
        let inner = block.inner(area);
        f.render_widget(block, area);
        if self.loras.is_empty() {
            f.render_widget(
                Paragraph::new("No LoRAs found. Drop .safetensors into the workspace loras/ dir.")
                    .style(Style::new().fg(Color::DarkGray))
                    .wrap(Wrap { trim: true }),
                inner,
            );
            return;
        }
        let mut lines: Vec<Line> = Vec::new();
        for (i, l) in self.loras.iter().enumerate() {
            let (glyph, gcolor) = match self.compatible(l) {
                Some(true) => ("✓", Color::Green),
                Some(false) => ("✗", Color::Red),
                None => ("·", Color::DarkGray),
            };
            let name_style = if i == self.selected {
                Style::new().bg(Color::Cyan).fg(Color::Black).add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(Color::White)
            };
            lines.push(Line::from(vec![
                Span::styled(format!(" {glyph} "), Style::new().fg(gcolor).add_modifier(Modifier::BOLD)),
                Span::styled(trunc(&l.name, 26), name_style),
                Span::styled(
                    format!("  {}", l.family.map(family_label).unwrap_or("?")),
                    Style::new().fg(Color::DarkGray),
                ),
            ]));
        }
        f.render_widget(Paragraph::new(lines), inner);
    }

    fn render_detail(&self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" Detail ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        let Some(l) = self.loras.get(self.selected) else { return };

        let kv = |k: &str, v: String| -> Line {
            Line::from(vec![
                Span::styled(format!("{k:<10}"), Style::new().fg(Color::DarkGray)),
                Span::styled(v, Style::new().fg(Color::White)),
            ])
        };
        let mut lines = vec![Line::from(Span::styled(
            l.name.clone(),
            Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ))];
        lines.push(kv("family", l.family.map(family_label).unwrap_or("unknown").to_string()));
        lines.push(kv("rank", l.rank.map(|r| r.to_string()).unwrap_or_else(|| "—".into())));
        lines.push(kv("size", format!("{} MB", l.size / 1_048_576)));
        lines.push(kv("location", l.location.clone()));
        lines.push(kv("path", l.path.display().to_string()));

        match self.compatible(l) {
            Some(true) => lines.push(Line::from(Span::styled("  ✓ compatible with the loaded model", Style::new().fg(Color::Green)))),
            Some(false) => lines.push(Line::from(Span::styled("  ✗ family mismatch with the loaded model", Style::new().fg(Color::Red)))),
            None => lines.push(Line::from(Span::styled("  · compatibility unknown (load a model)", Style::new().fg(Color::DarkGray)))),
        }

        if !l.triggers.is_empty() {
            lines.push(Line::from(""));
            lines.push(kv("triggers", l.triggers.join(", ")));
        }
        if !l.notes.is_empty() {
            lines.push(Line::from(""));
            lines.push(kv("notes", l.notes.clone()));
        }
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
    }
}

/// Map a free-text base-model string (e.g. a sidecar's "SDXL 1.0") to a family.
fn family_from_str(s: &str) -> Option<BaseFamily> {
    let s = s.to_lowercase();
    if s.is_empty() {
        return None;
    }
    if s.contains("xl") {
        Some(BaseFamily::Sdxl)
    } else if s.contains("cascade") {
        Some(BaseFamily::StableCascade)
    } else if s.contains("pixart") {
        Some(BaseFamily::PixArt)
    } else if s.contains("flux") {
        Some(BaseFamily::Flux)
    } else if s.contains("sd3") || s.contains("sd-3") || s.contains("stable diffusion 3") {
        Some(BaseFamily::Sd3)
    } else if s.contains("2.1") || s.contains("v2") || s.contains(" 2 ") {
        Some(BaseFamily::Sd21)
    } else if s.contains("1.5") || s.contains("v1") || s.contains("1-5") {
        Some(BaseFamily::Sd15)
    } else {
        None
    }
}

fn family_label(f: BaseFamily) -> &'static str {
    match f {
        BaseFamily::Sd15 => "SD1.5",
        BaseFamily::Sd21 => "SD2.1",
        BaseFamily::Sdxl => "SDXL",
        BaseFamily::Flux => "Flux",
        BaseFamily::Sd3 => "SD3",
        BaseFamily::PixArt => "PixArt",
        BaseFamily::StableCascade => "Cascade",
    }
}

fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        format!("{s:<width$}", width = n)
    } else {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
    }
}

/// Recursively collect `.safetensors` under `dir`.
fn collect(dir: &Path, label: &str, out: &mut Vec<LoraInfo>, depth: usize) {
    if depth > 5 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, label, out, depth + 1);
        } else if p.extension().and_then(|x| x.to_str()) == Some("safetensors") {
            out.push(load_lora(&p, label));
        }
    }
}

fn load_lora(path: &Path, location: &str) -> LoraInfo {
    let name = path.file_stem().and_then(|n| n.to_str()).unwrap_or("?").to_string();
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let (meta, dims) = read_st_header(path).unwrap_or_default();
    let mut family = infer_family(&meta, &dims);
    let rank = infer_rank(&meta, &dims);

    // Optional `<name>.plakat.hjson` sidecar.
    let mut triggers = Vec::new();
    let mut notes = String::new();
    let sidecar = path.with_extension("plakat.hjson");
    if let Ok(text) = std::fs::read_to_string(&sidecar) {
        if let Ok(sc) = deser_hjson::from_str::<Sidecar>(&text) {
            triggers = sc.trigger_words;
            notes = sc.notes;
            // Trust the sidecar's declared base_model when the header was inconclusive.
            if family.is_none() {
                family = family_from_str(&sc.base_model);
            }
        }
    }

    LoraInfo { path: path.to_path_buf(), name, family, rank, size, triggers, notes, location: location.to_string() }
}

/// Read a safetensors header: the `__metadata__` map + each tensor's shape. Reads
/// only the JSON header (8-byte LE length prefix), never the tensor data.
fn read_st_header(path: &Path) -> Option<(HashMap<String, String>, Vec<(String, Vec<usize>)>)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).ok()?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    if header_len == 0 || header_len > 100 * 1_048_576 {
        return None; // sanity bound
    }
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header).ok()?;
    let json: serde_json::Value = serde_json::from_slice(&header).ok()?;
    let obj = json.as_object()?;

    let mut meta = HashMap::new();
    if let Some(m) = obj.get("__metadata__").and_then(|v| v.as_object()) {
        for (k, v) in m {
            if let Some(s) = v.as_str() {
                meta.insert(k.clone(), s.to_string());
            }
        }
    }
    let mut dims = Vec::new();
    for (k, v) in obj {
        if k == "__metadata__" {
            continue;
        }
        if let Some(shape) = v.get("shape").and_then(|s| s.as_array()) {
            let shape: Vec<usize> = shape.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect();
            dims.push((k.clone(), shape));
        }
    }
    Some((meta, dims))
}

/// Infer the base family from kohya `ss_base_model_version` metadata, falling back
/// to the cross-attention context dim of a `to_k`/`to_v` LoRA-down weight
/// (768=SD1.5, 1024=SD2.1, 2048=SDXL).
fn infer_family(meta: &HashMap<String, String>, dims: &[(String, Vec<usize>)]) -> Option<BaseFamily> {
    if let Some(v) = meta.get("ss_base_model_version") {
        let v = v.to_lowercase();
        if v.contains("xl") {
            return Some(BaseFamily::Sdxl);
        }
        if v.contains("v2") || v.contains("_2.") || v.contains("768") {
            return Some(BaseFamily::Sd21);
        }
        if v.contains("v1") || v.contains("1-5") || v.contains("1.5") {
            return Some(BaseFamily::Sd15);
        }
    }
    // Dim heuristic: a cross-attn (attn2) to_k/to_v down weight is [rank, ctx_dim].
    for (name, shape) in dims {
        let n = name.to_lowercase();
        let is_ca = n.contains("attn2") && (n.contains("to_k") || n.contains("to_v"));
        let is_down = n.contains("lora_down") || n.contains("lora.down") || n.contains("lora_a");
        if is_ca && is_down && shape.len() == 2 {
            return match shape[1] {
                768 => Some(BaseFamily::Sd15),
                1024 => Some(BaseFamily::Sd21),
                2048 => Some(BaseFamily::Sdxl),
                _ => None,
            };
        }
    }
    None
}

fn infer_rank(meta: &HashMap<String, String>, dims: &[(String, Vec<usize>)]) -> Option<u32> {
    if let Some(d) = meta.get("ss_network_dim").and_then(|s| s.parse::<u32>().ok()) {
        return Some(d);
    }
    // A lora_down weight is [rank, in]; rank is the smaller leading dim.
    for (name, shape) in dims {
        let n = name.to_lowercase();
        if (n.contains("lora_down") || n.contains("lora.down") || n.contains("lora_a")) && shape.len() == 2 {
            return Some(shape[0] as u32);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal `.safetensors` (8-byte LE header len + JSON header) with the
    /// given `__metadata__` and one tensor of `shape`.
    fn write_st(path: &Path, metadata: &[(&str, &str)], tensor: &str, shape: &[usize]) {
        let meta: serde_json::Map<String, serde_json::Value> =
            metadata.iter().map(|(k, v)| (k.to_string(), serde_json::Value::String(v.to_string()))).collect();
        let header = serde_json::json!({
            "__metadata__": meta,
            tensor: { "dtype": "F16", "shape": shape, "data_offsets": [0, 0] },
        });
        let bytes = serde_json::to_vec(&header).unwrap();
        let mut out = (bytes.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&bytes);
        std::fs::write(path, out).unwrap();
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("plakat-lorahub-test-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn infers_family_from_kohya_metadata_and_dim_heuristic() {
        let d = tmp("infer");
        // Kohya metadata says SDXL.
        let xl = d.join("style-xl.safetensors");
        write_st(&xl, &[("ss_base_model_version", "sdxl_base_v1-0"), ("ss_network_dim", "32")], "x", &[32, 4]);
        // No metadata, but a cross-attn down weight reveals SD1.5 (ctx dim 768).
        let sd = d.join("char.safetensors");
        write_st(&sd, &[], "lora_unet_..._attn2_to_k.lora_down.weight", &[16, 768]);

        let s = LoraHubState::new(vec![(d.clone(), "loras".into())]);
        assert_eq!(s.loras.len(), 2);
        let by = |n: &str| s.loras.iter().find(|l| l.name == n).unwrap();
        assert_eq!(by("style-xl").family, Some(BaseFamily::Sdxl));
        assert_eq!(by("style-xl").rank, Some(32));
        assert_eq!(by("char").family, Some(BaseFamily::Sd15));
        assert_eq!(by("char").rank, Some(16));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn sidecar_base_model_is_a_family_fallback_when_header_is_inconclusive() {
        let d = tmp("sidecar-fallback");
        // No usable metadata / cross-attn dim → header inference is None.
        let f = d.join("mystery.safetensors");
        write_st(&f, &[], "some.lora_down.weight", &[8, 999]);
        std::fs::write(d.join("mystery.plakat.hjson"), r#"{"base_model":"SDXL 1.0"}"#).unwrap();
        let s = LoraHubState::new(vec![(d.clone(), "loras".into())]);
        assert_eq!(s.loras[0].family, Some(BaseFamily::Sdxl), "falls back to the sidecar base_model");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn compatibility_tracks_the_loaded_family_and_sidecar_loads() {
        let d = tmp("compat");
        let xl = d.join("foo.safetensors");
        write_st(&xl, &[("ss_base_model_version", "sdxl")], "x", &[8, 4]);
        std::fs::write(d.join("foo.plakat.hjson"), r#"{"trigger_words":["foo style"],"notes":"hi"}"#).unwrap();

        let mut s = LoraHubState::new(vec![(d.clone(), "loras".into())]);
        assert_eq!(s.loras[0].triggers, vec!["foo style"]);
        assert_eq!(s.loras[0].notes, "hi");
        // No model → unknown.
        assert_eq!(s.compatible(&s.loras[0]), None);
        s.set_loaded_family(Some(BaseFamily::Sdxl));
        assert_eq!(s.compatible(&s.loras[0]), Some(true));
        s.set_loaded_family(Some(BaseFamily::Sd15));
        assert_eq!(s.compatible(&s.loras[0]), Some(false));
        let _ = std::fs::remove_dir_all(&d);
    }
}
