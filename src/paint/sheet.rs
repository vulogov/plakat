//! The OUTCOME sheet (RFC PAINT-3, P1): one page in the format of the user's sample — the master
//! composition, four micro-analysis lens crops, the layer hierarchy, the palette, the brushwork facts —
//! every panel a fact of the run. Typeset through Typst (as the bookart suite is), compiled to PNG/PDF.
//!
//! The four insets are chosen by RULE, not taste (§3.2): the focal plane (the faces' centroid, else the
//! busiest place on the subject), the thickest paint off the faces, the most textured background
//! passage, and the thinnest light. Each caption carries two measured numbers so a reader can check it.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::{imageops, RgbImage};

use crate::paint::canvas::Canvas;
use crate::paint::painter::{PaintParams, PassStat};
use crate::paint::score::StrokeScore;

/// A chosen inset: where, what it shows, and the two numbers behind the label.
#[derive(Clone, Debug)]
pub struct Inset {
    pub cx: u32,
    pub cy: u32,
    pub title: String,
    pub subtitle: String,
    pub numbers: String,
}

/// Classic pigment masstones, for naming an `image` palette's pigments by their nearest (sRGB).
const CLASSIC: &[(&str, [u8; 3])] = &[
    ("titanium white", [247, 245, 241]),
    ("ivory black", [28, 26, 26]),
    ("lamp black", [20, 20, 24]),
    ("payne's grey", [64, 68, 82]),
    ("yellow ochre", [204, 153, 51]),
    ("raw sienna", [196, 136, 72]),
    ("burnt sienna", [150, 70, 40]),
    ("raw umber", [92, 72, 50]),
    ("burnt umber", [74, 46, 32]),
    ("cadmium yellow", [250, 210, 40]),
    ("cadmium orange", [240, 130, 30]),
    ("cadmium red", [200, 40, 30]),
    ("alizarin crimson", [140, 24, 40]),
    ("venetian red", [160, 60, 50]),
    ("rose madder", [200, 90, 110]),
    ("ultramarine blue", [18, 40, 130]),
    ("cobalt blue", [30, 80, 170]),
    ("cerulean blue", [50, 130, 190]),
    ("prussian blue", [16, 30, 60]),
    ("phthalo green", [20, 90, 70]),
    ("viridian", [40, 130, 100]),
    ("sap green", [90, 120, 40]),
    ("emerald green", [40, 160, 90]),
    ("dioxazine purple", [70, 30, 100]),
    ("naples yellow", [240, 220, 150]),
    ("flesh tint", [230, 180, 150]),
    ("terre verte", [110, 130, 90]),
    ("indigo", [30, 40, 70]),
];

/// The nearest classic pigment name to an sRGB masstone, with its distance (0 = exact).
pub fn classic_name(rgb: [u8; 3]) -> (&'static str, f32) {
    let lab = crate::paint::color::srgb_to_lab(rgb);
    let mut best = ("", f32::MAX);
    for (name, m) in CLASSIC {
        let d = crate::paint::color::delta_e76(lab, crate::paint::color::srgb_to_lab(*m));
        if d < best.1 {
            best = (name, d);
        }
    }
    best
}

/// Mean of a field under a mask threshold.
fn mean_where(field: &[f32], pick: impl Fn(usize) -> bool) -> Option<f32> {
    let (mut s, mut n) = (0f64, 0usize);
    for (i, v) in field.iter().enumerate() {
        if pick(i) {
            s += *v as f64;
            n += 1;
        }
    }
    (n > 0).then(|| (s / n as f64) as f32)
}

/// A box-blurred copy of a field (radius r), for finding REGIONS rather than pixels.
fn smooth(field: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let pass = |line: &[f32], out: &mut [f32]| {
        let len = line.len();
        let mut prefix = vec![0f32; len + 1];
        for i in 0..len {
            prefix[i + 1] = prefix[i] + line[i];
        }
        for i in 0..len {
            let a = i.saturating_sub(r);
            let b = (i + r).min(len - 1);
            out[i] = (prefix[b + 1] - prefix[a]) / (b - a + 1) as f32;
        }
    };
    let mut t = vec![0f32; w * h];
    for y in 0..h {
        pass(&field[y * w..(y + 1) * w], &mut t[y * w..(y + 1) * w]);
    }
    let mut out = vec![0f32; w * h];
    let (mut col, mut colo) = (vec![0f32; h], vec![0f32; h]);
    for x in 0..w {
        for y in 0..h {
            col[y] = t[y * w + x];
        }
        pass(&col, &mut colo);
        for y in 0..h {
            out[y * w + x] = colo[y];
        }
    }
    out
}

/// The arg-max of a smoothed field under a predicate, kept at least `apart` px from `taken`.
fn peak(field: &[f32], w: usize, h: usize, pick: impl Fn(usize) -> bool, taken: &[(u32, u32)], apart: f32) -> Option<(u32, u32)> {
    let mut best: Option<(usize, f32)> = None;
    for (i, v) in field.iter().enumerate() {
        if !pick(i) {
            continue;
        }
        let (x, y) = ((i % w) as f32, (i / w) as f32);
        if taken.iter().any(|&(tx, ty)| ((x - tx as f32).powi(2) + (y - ty as f32).powi(2)).sqrt() < apart) {
            continue;
        }
        if best.map_or(true, |(_, bv)| *v > bv) {
            best = Some((i, *v));
        }
    }
    let _ = h;
    best.map(|(i, _)| ((i % w) as u32, (i / w) as u32))
}

/// Choose the four insets (§3.2). `luma` is the source's value, `busy` the painter's texture measure.
pub fn choose_insets(params: &PaintParams, canvas: &Canvas, luma: &[f32], busy: &[f32], w: u32, h: u32, is_oil: bool) -> Vec<Inset> {
    let (wu, hu) = (w as usize, h as usize);
    let px = wu * hu;
    let peak_h = canvas.height.iter().copied().fold(0f32, f32::max).max(1e-6);
    let hp: Vec<f32> = canvas.height.iter().map(|v| v / peak_h * 100.0).collect();
    let r = (w.min(h) as usize / 32).max(4);
    let hs = smooth(&hp, wu, hu, r);
    let bs = smooth(busy, wu, hu, r);
    let ls = smooth(luma, wu, hu, r);
    let face = params.face_mask.as_deref();
    let subj = params.subject_mask.as_deref();
    let on_face = |i: usize| face.map_or(false, |m| m.get(i).copied().unwrap_or(0.0) > 0.35);
    let on_subj = |i: usize| subj.map_or(false, |m| m.get(i).copied().unwrap_or(0.0) > 0.5);
    let apart = w.min(h) as f32 * 0.25;
    let mut taken: Vec<(u32, u32)> = Vec::new();
    let mut out = Vec::new();
    let whole = hp.iter().sum::<f32>() / px.max(1) as f32;
    let fine_r = (params.min_brush * 0.75).round().max(2.0);

    // 1. The focal plane.
    let focal = if let Some(m) = face {
        let (mut sx, mut sy, mut n) = (0f64, 0f64, 0usize);
        for (i, v) in m.iter().enumerate() {
            if *v > 0.35 {
                sx += (i % wu) as f64;
                sy += (i / wu) as f64;
                n += 1;
            }
        }
        (n > 0).then(|| ((sx / n as f64) as u32, (sy / n as f64) as u32))
    } else {
        None
    }
    .or_else(|| peak(&bs, wu, hu, |i| on_subj(i), &taken, 0.0));
    if let Some((cx, cy)) = focal {
        let hf = mean_where(&hp, |i| on_face(i)).unwrap_or(whole);
        let hb = mean_where(&hp, |i| !on_face(i)).unwrap_or(whole);
        let smooth_face = hf < hb;
        out.push(Inset {
            cx,
            cy,
            title: if face.is_some() { "The faces".into() } else { "The focal plane".into() },
            subtitle: if is_oil {
                if smooth_face { "sfumato — thin, blended paint, the fine passes on the face alone".into() } else { "the fine passes on the face".into() }
            } else {
                "the fine brushes reserved for the face".into()
            },
            numbers: format!("height {hf:.0}% of peak vs {hb:.0}% elsewhere · {}", if params.face_mask.is_some() { "face tier on" } else { "no face found" }),
        });
        taken.push((cx, cy));
    }

    // 2. The thickest paint off the faces.
    if let Some((cx, cy)) = peak(&hs, wu, hu, |i| !on_face(i), &taken, apart) {
        let i = cy as usize * wu + cx as usize;
        let here = hs[i];
        let b = bs[i];
        out.push(Inset {
            cx,
            cy,
            title: "The thickest paint".into(),
            subtitle: if is_oil {
                if params.brush.ridges > 0.0 { "impasto — bristle ridges, plowed edges, cast micro-shadows".into() } else { "impasto — the paint built up".into() }
            } else {
                "the heaviest wash".into()
            },
            numbers: format!("height {here:.0}% of peak ({:.1}× the sheet's mean) · texture {:.0}%", here / whole.max(1e-3), b * 100.0),
        });
        taken.push((cx, cy));
    }

    // 3. The most textured background passage.
    if let Some((cx, cy)) = peak(&bs, wu, hu, |i| !on_subj(i) && !on_face(i), &taken, apart) {
        let i = cy as usize * wu + cx as usize;
        let (b, here) = (bs[i], hs[i]);
        out.push(Inset {
            cx,
            cy,
            title: "The textured ground".into(),
            subtitle: if params.brush.skip > 0.0 { "dry brush — the bristles skip the paper's fibres".into() } else if is_oil { "broken marks — short strokes over the tooth".into() } else { "textured marks".into() },
            numbers: format!("texture {:.0}% at the finest brush ({fine_r:.0} px) · height {here:.0}% of peak", b * 100.0),
        });
        taken.push((cx, cy));
    }

    // 4. The thinnest light: among the brightest fifth of the source, the least paint.
    let mut sorted: Vec<f32> = ls.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let hi = sorted[sorted.len() * 4 / 5];
    // The brightest, thinnest place: value counts twice as much as thinness.
    let inv: Vec<f32> = hs.iter().zip(&ls).map(|(v, l)| (100.0 - v) + 200.0 * l).collect();
    if let Some((cx, cy)) = peak(&inv, wu, hu, |i| ls[i] >= hi && !on_face(i), &taken, apart) {
        let i = cy as usize * wu + cx as usize;
        out.push(Inset {
            cx,
            cy,
            title: "The lights".into(),
            subtitle: if is_oil {
                if params.weave > 0.0 { "a thin glaze over the ground — the linen shows through".into() } else { "a thin glaze over the ground".into() }
            } else {
                "reserved paper under a glaze".into()
            },
            numbers: format!("height {:.0}% of peak · value {:.0}% (top fifth of the source)", hs[i], ls[i] * 100.0),
        });
        taken.push((cx, cy));
    }
    out
}

/// A swatch for the palette panel.
#[derive(Clone, Debug)]
pub struct Swatch {
    pub name: String,
    pub classic: String,
    pub hex: String,
    pub share: f32,
}

/// The palette ranked by use (sum of mix weights over the painted strokes), named by the nearest classic.
pub fn palette_by_use(score: &StrokeScore, n: usize) -> Vec<Swatch> {
    let mut usage: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
    for r in score.strokes.iter().filter(|r| !r.wipe) {
        for (name, w) in &r.mix {
            *usage.entry(name.as_str()).or_insert(0.0) += *w as f64;
        }
    }
    let total: f64 = usage.values().sum::<f64>().max(1e-9);
    let mut ranked: Vec<(&str, f64)> = usage.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
        .into_iter()
        .take(n)
        .map(|(name, w)| {
            let mass = score.header.pigments.iter().find(|(pn, _)| pn == name).map(|(_, m)| *m).unwrap_or([128, 128, 128]);
            let (classic, _) = classic_name(mass);
            let display = if name.starts_with("img-") { classic.to_string() } else { name.replace('-', " ") };
            Swatch { name: display, classic: classic.to_string(), hex: format!("#{:02x}{:02x}{:02x}", mass[0], mass[1], mass[2]), share: (w / total * 100.0) as f32 }
        })
        .collect()
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('#', "\\#").replace('@', "\\@").replace('*', "\\*").replace('_', "\\_")
}

/// What the sheet is built from.
pub struct SheetInputs<'a> {
    pub title: &'a str,
    pub medium: &'a str,
    pub master: &'a RgbImage,
    pub insets: &'a [Inset],
    /// The per-pass images (path, stage label, radius, strokes) in painting order.
    pub passes: Vec<(PathBuf, String, f32, usize)>,
    pub palette: &'a [Swatch],
    pub stats: &'a [PassStat],
    pub score: &'a StrokeScore,
    pub facts: Vec<(String, String)>,
}

/// Build the sheet: write the work files into `<out stem>_sheet/`, the Typst source, and compile it to
/// `out` (`.png` or `.pdf` by extension). Returns the path written.
pub fn build(inputs: &SheetInputs, out: &Path) -> Result<PathBuf> {
    let stem = out.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "sheet".into());
    let dir = out.with_file_name(format!("{stem}_sheet"));
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    // Master, downsized for the page.
    let (mw, mh) = (inputs.master.width(), inputs.master.height());
    let side = 1400u32;
    let master = if mw.max(mh) > side { imageops::resize(inputs.master, mw * side / mw.max(mh), mh * side / mw.max(mh), imageops::FilterType::Triangle) } else { inputs.master.clone() };
    master.save(dir.join("master.png"))?;
    // Insets, 1:1 crops.
    let crop = (mw.min(mh) / 5).clamp(240, 720);
    for (i, ins) in inputs.insets.iter().enumerate() {
        let x0 = ins.cx.saturating_sub(crop / 2).min(mw.saturating_sub(crop));
        let y0 = ins.cy.saturating_sub(crop / 2).min(mh.saturating_sub(crop));
        imageops::crop_imm(inputs.master, x0, y0, crop, crop).to_image().save(dir.join(format!("inset_{}.png", i + 1)))?;
    }
    // Pass layers, small.
    let mut layer_files: Vec<(String, String)> = Vec::new();
    for (i, (path, label, radius, strokes)) in inputs.passes.iter().enumerate() {
        if let Ok(img) = image::open(path) {
            let img = img.to_rgb8();
            let s = 420u32;
            let small = imageops::resize(&img, img.width() * s / img.width().max(img.height()), img.height() * s / img.width().max(img.height()), imageops::FilterType::Triangle);
            let f = format!("layer_{}.png", i + 1);
            small.save(dir.join(&f))?;
            layer_files.push((f, format!("{label} · {radius:.0} px · {strokes} strokes")));
        }
    }
    // ---- Typst ----
    let mut t = String::new();
    t.push_str("#set page(width: 320mm, height: auto, margin: 10mm, fill: rgb(\"#f3eee3\"))\n");
    t.push_str("#set text(font: (\"Libertinus Serif\", \"Georgia\", \"Times New Roman\"), size: 9pt, fill: rgb(\"#2a2522\"))\n");
    t.push_str("#let panel(title, body) = block(width: 100%, inset: 0pt)[#block(width: 100%, fill: rgb(\"#e4dccb\"), inset: 5pt)[#align(center)[#text(weight: \"bold\", size: 10.5pt)[#upper(title)]]]#block(width: 100%, inset: (x: 5pt, y: 6pt), stroke: 0.6pt + rgb(\"#9a8f7a\"))[#body]]\n");
    t.push_str("#let lens(file, title, sub, nums) = align(center)[#box(clip: true, radius: 50%, width: 64mm, height: 64mm, stroke: 2.2pt + rgb(\"#3a3330\"))[#image(file, width: 64mm, height: 64mm, fit: \"cover\")]#v(2mm)#text(weight: \"bold\", size: 9.5pt)[#upper(title)]#linebreak()#text(size: 8.5pt)[#sub]#linebreak()#text(size: 7.5pt, fill: rgb(\"#5a5047\"))[#nums]]\n");
    t.push_str(&format!("#align(center)[#text(size: 20pt, weight: \"bold\")[OUTCOME ANALYSIS: “{}” — {} PAINTING]]\n#v(2mm)#line(length: 100%, stroke: 0.8pt + rgb(\"#6a6055\"))\n#v(3mm)\n", esc(&inputs.title.to_uppercase()), esc(&inputs.medium.to_uppercase())));
    // Top: master | 2×2 insets
    t.push_str("#grid(columns: (46%, 54%), column-gutter: 6mm, [\n");
    t.push_str("  #align(center)[#text(size: 10pt, weight: \"bold\")[MASTER COMPOSITION]]#v(2mm)\n");
    t.push_str("  #box(stroke: 3pt + rgb(\"#5b4a2f\"), inset: 3pt, fill: white)[#image(\"master.png\", width: 100%)]\n");
    t.push_str("], [\n  #grid(columns: (1fr, 1fr), column-gutter: 4mm, row-gutter: 5mm,\n");
    for (i, ins) in inputs.insets.iter().enumerate() {
        t.push_str(&format!("    lens(\"inset_{}.png\", \"MICRO-ANALYSIS {}: {}\", \"{}\", \"{}\"),\n", i + 1, i + 1, esc(&ins.title), esc(&ins.subtitle), esc(&ins.numbers)));
    }
    t.push_str("  )\n])\n#v(5mm)\n");
    // Bottom: layers | palette | brushwork facts
    t.push_str("#grid(columns: (1fr, 1fr, 1fr), column-gutter: 5mm,\n");
    // layers
    t.push_str("  panel(\"Layer hierarchy\")[\n");
    if layer_files.is_empty() {
        t.push_str("    #text(size: 8pt)[(the passes were not dumped — run with `--dump-passes`)]\n");
    } else {
        let n = layer_files.len();
        let step = 11; // mm between layers; each shows its top band, the last pass in full
        let thumb = 34; // mm wide
        t.push_str(&format!("    #box(width: 100%, height: {}mm)[\n", step * (n - 1) + thumb + 6));
        for (i, (f, label)) in layer_files.iter().enumerate() {
            let k = n - 1 - i; // the first pass at the bottom of the stack
            t.push_str(&format!("      #place(top + left, dx: {}mm, dy: {}mm)[#box(stroke: 0.5pt + rgb(\"#6a6055\"), fill: white)[#image(\"{}\", width: {}mm)]]\n", 2 * i, step * k, f, thumb));
            t.push_str(&format!("      #place(top + left, dx: {}mm, dy: {}mm)[#text(size: 7pt)[{}]]\n", thumb + 2 * n + 3, step * k + 1, esc(label)));
        }
        t.push_str("    ]\n");
        t.push_str("    #text(size: 7.5pt, fill: rgb(\"#5a5047\"))[Bottom to top: the block-in, the restatements, the finest passes — the canvas as each pass left it.]\n");
    }
    t.push_str("  ],\n");
    // palette
    t.push_str("  panel(\"Palette analysis\")[\n    #grid(columns: (1fr, 1fr, 1fr), row-gutter: 3mm,\n");
    for sw in inputs.palette.iter().take(9) {
        t.push_str(&format!("      align(center)[#circle(radius: 8mm, fill: rgb(\"{}\"), stroke: 1pt + rgb(\"#3a3330\"))#v(1mm)#text(size: 7.5pt, weight: \"bold\")[#upper(\"{}\")]#linebreak()#text(size: 7pt)[{:.0}% of the paint]#linebreak()#text(size: 6pt, fill: rgb(\"#7a6f62\"))[{}]],\n", sw.hex, esc(&sw.name), sw.share, esc(&sw.hex)));
    }
    t.push_str("    )\n    #v(1mm)#text(size: 7pt, fill: rgb(\"#5a5047\"))[Ranked by use over every stroke; named by the nearest classic pigment to the measured masstone.]\n  ],\n");
    // brushwork facts
    t.push_str("  panel(\"Brushwork dynamics\")[\n    #table(columns: (auto, auto, auto, auto), stroke: 0.4pt + rgb(\"#9a8f7a\"), inset: 3pt, align: left,\n      [*pass*], [*brush*], [*marks*], [*dry*],\n");
    for s in inputs.stats.iter().filter(|s| s.strokes > 0) {
        let recs: Vec<&crate::paint::score::StrokeRecord> = inputs.score.strokes.iter().filter(|r| r.stage == s.stage && !r.wipe).collect();
        let dry = if recs.is_empty() { 0 } else { recs.iter().filter(|r| r.wet < 0.62).count() * 100 / recs.len() };
        t.push_str(&format!("      [{}], [{:.0} px], [{}], [{}%],\n", esc(&s.stage), s.radius, s.strokes, dry));
    }
    t.push_str("    )\n    #v(2mm)\n");
    for (k, v) in &inputs.facts {
        t.push_str(&format!("    #text(size: 7.5pt)[*{}:* {}]#linebreak()\n", esc(k), esc(v)));
    }
    t.push_str("  ],\n)\n");
    t.push_str("#v(3mm)#align(center)[#text(size: 7pt, fill: rgb(\"#5a5047\"))[Every panel is a fact of this run — the plan, the stroke score and the finished canvas. Nothing on this sheet is invented.]]\n");
    // The Typst source is kept — in the work directory with the images it references, and a copy beside
    // the output so it is easy to find and edit (`typst compile <stem>_sheet/sheet.typ`).
    let typ = dir.join("sheet.typ");
    std::fs::write(&typ, &t)?;
    let _ = std::fs::write(out.with_extension("typ"), &t);
    let is_pdf = out.extension().map_or(false, |e| e.eq_ignore_ascii_case("pdf"));
    let mut cmd = std::process::Command::new("typst");
    cmd.arg("compile").arg(&typ).arg(out);
    if !is_pdf {
        cmd.arg("--ppi").arg("150");
    }
    let o = cmd.output().context("running `typst` (install it: brew install typst)")?;
    if !o.status.success() {
        anyhow::bail!("typst failed: {}", String::from_utf8_lossy(&o.stderr));
    }
    Ok(out.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classic_names_are_sensible() {
        assert_eq!(classic_name([250, 250, 248]).0, "titanium white");
        assert_eq!(classic_name([20, 40, 125]).0, "ultramarine blue");
        assert_eq!(classic_name([150, 72, 42]).0, "burnt sienna");
    }

    #[test]
    fn the_insets_are_four_distinct_places() {
        use crate::paint::painter::{paint_from_image, PaintParams};
        use crate::paint::palette;
        let (w, h) = (160u32, 120u32);
        let img = RgbImage::from_fn(w, h, |x, y| {
            // A bright patch top-left, a busy checker bottom-right, mid elsewhere.
            if x < 50 && y < 40 { image::Rgb([240, 235, 225]) } else if x > 100 && y > 70 { if (x / 3 + y / 3) % 2 == 0 { image::Rgb([30, 30, 40]) } else { image::Rgb([200, 190, 170]) } } else { image::Rgb([110, 90, 100]) }
        });
        let mut p = PaintParams::new(palette::EARTH, 2000);
        p.brush_sizes = vec![16.0, 8.0, 4.0];
        p.min_brush = 3.0;
        let mut subj = vec![0f32; (w * h) as usize];
        for y in 30..90u32 { for x in 50..100u32 { subj[(y * w + x) as usize] = 1.0; } }
        p.subject_mask = Some(subj);
        let r = paint_from_image(&img, &p);
        let luma = crate::paint::painter::luma_map(&img);
        let fine = crate::paint::painter::local_range(&luma, w as usize, h as usize, 2);
        let busy: Vec<f32> = fine.iter().map(|f| ((f - 0.08) / 0.12).clamp(0.0, 1.0)).collect();
        let ins = choose_insets(&p, &r.canvas, &luma, &busy, w, h, true);
        assert!(ins.len() >= 3, "{ins:?}");
        for a in 0..ins.len() { for b in (a + 1)..ins.len() {
            let d = ((ins[a].cx as f32 - ins[b].cx as f32).powi(2) + (ins[a].cy as f32 - ins[b].cy as f32).powi(2)).sqrt();
            assert!(d >= 30.0, "insets {a} and {b} too close: {d}");
        } }
        assert!(ins.iter().any(|i| i.title == "The lights"));
        let sw = palette_by_use(&r.score, 6);
        assert!(!sw.is_empty() && sw[0].share > 0.0);
    }
}
