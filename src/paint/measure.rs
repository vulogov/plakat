//! THE MEASUREMENT OF A PAINTED SHEET AGAINST ITS SOURCE (7.20.0, `plakat paint compare`, the critic of
//! `--make-plan-iterate`).
//!
//! The review cycles judged every set by the same handful of numbers, taken by hand with ImageMagick: the mean
//! luma and its spread, the local σ of a crop, the chroma spread, an RMSE. This module is those numbers as a
//! function — the SAME facts the director reads from the source (`director::SceneFacts`: the key, the calm and
//! the busy fields by the local σ, the points of light on a dark ground, the chroma) read from the sheet too,
//! and the two compared: what the painting KEPT of the picture. Every number is a share or a ratio, so sheets
//! of different media and sizes compare on one table. Nothing here paints; the painter is untouched.
//!
//! The comparison works on copies of both pictures reduced to the source's aspect at a long side of 1024 or
//! less, so a sheet painted at 2k and its 1024 source meet at one scale (the director's own).

use image::{imageops, RgbImage};

use super::director::local_std;

/// The long side the pictures are reduced to before measuring.
pub const WORK_SIDE: u32 = 1024;
/// Local σ under this: a calm field (the director's line).
pub const CALM: f32 = 0.02;
/// Local σ over this: a busy passage (the director's line).
pub const BUSY: f32 = 0.08;
/// A source-calm pixel counts as KEPT calm on the sheet under this local σ — a painted field carries the brush's
/// own grain, so the sheet is allowed twice the source's calm line.
pub const CALM_KEPT: f32 = 0.04;
/// A source-busy pixel counts as KEPT busy on the sheet over this local σ.
pub const BUSY_KEPT: f32 = 0.05;

/// The facts of one picture (the source or a sheet), every one a share, a mean or a density.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Measures {
    pub width: u32,
    pub height: u32,
    /// Mean luma (0.299 R + 0.587 G + 0.114 B on sRGB, in [0,1]).
    pub luma_mean: f32,
    /// The luma's σ over the sheet — the value range.
    pub luma_stddev: f32,
    /// The share under 0.12 of luma.
    pub dark_share: f32,
    /// The share over 0.88.
    pub light_share: f32,
    /// The share whose local σ (a window of a 128th of the long side) is under `CALM`.
    pub calm_share: f32,
    /// The share whose local σ is over `BUSY`.
    pub busy_share: f32,
    /// The mean local σ — the picture's local contrast as one number.
    pub sigma_mean: f32,
    /// Mean chroma (the sRGB max−min).
    pub chroma_mean: f32,
    /// The 90th percentile of the chroma — how far the colour goes at its strongest.
    pub chroma_p90: f32,
    /// Points of light on a dark ground per megapixel (the director's detector).
    pub points_per_mpx: f32,
    /// The share of bare white — luma over 0.94 and chroma under 0.04: the paper on a watercolour, the snow on
    /// a photograph.
    pub bare_share: f32,
    /// The mean Sobel magnitude of the luma — the edge energy.
    pub edge_mean: f32,
}

/// A sheet against its source.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Comparison {
    pub source: Measures,
    pub sheet: Measures,
    /// RMSE of the luma between the two at the working size.
    pub rmse: f32,
    /// The intersection of the two 16-bin luma histograms (1 = the same tonal distribution).
    pub tone_overlap: f32,
    /// The sheet's mean luma less the source's: positive, the sheet is lighter.
    pub key_shift: f32,
    /// The sheet's value spread over the source's: under 1 the values compressed.
    pub range_ratio: f32,
    /// The sheet's mean local σ over the source's INSIDE the focus (the mask given, or the central half).
    pub contrast_focus: f32,
    /// The same outside it.
    pub contrast_periphery: f32,
    /// Of the source's calm pixels, the share still calm on the sheet (local σ under `CALM_KEPT`).
    pub calm_kept: f32,
    /// Of the source's busy pixels, the share still busy on the sheet (over `BUSY_KEPT`).
    pub busy_kept: f32,
    /// Of the source's strong edges (Sobel over its 90th percentile), the share the sheet carries at half the
    /// strength within a pixel.
    pub edges_kept: f32,
    /// The sheet's points of light over the source's, when the source has at least five.
    pub points_kept: Option<f32>,
    /// The sheet's mean chroma over the source's, when the source has colour (mean chroma over 0.01).
    pub chroma_ratio: Option<f32>,
    /// One number for the table: the mean of the tone overlap, the edges kept, the calm kept, the focus
    /// contrast capped at 1, one less the key shift over 0.2, and the chroma ratio capped at 1 — a summary of the
    /// kept shares, not a judgement of the painting.
    pub closeness: f32,
}

fn luma_of(img: &RgbImage) -> Vec<f32> {
    img.pixels().map(|p| (0.299 * p.0[0] as f32 + 0.587 * p.0[1] as f32 + 0.114 * p.0[2] as f32) / 255.0).collect()
}

fn chroma_of(img: &RgbImage) -> Vec<f32> {
    img.pixels()
        .map(|p| {
            let [r, g, b] = p.0;
            (r.max(g).max(b) - r.min(g).min(b)) as f32 / 255.0
        })
        .collect()
}

/// The local-σ window radius for a picture of this size (the director's: a 128th of the long side, 2..32).
pub fn sigma_radius(w: usize, h: usize) -> usize {
    ((w.max(h) as f32 / 256.0).round() as usize).clamp(2, 32)
}

/// The Sobel magnitude of a field (edge-clamped).
pub fn sobel(v: &[f32], w: usize, h: usize) -> Vec<f32> {
    let at = |x: i64, y: i64| v[(y.clamp(0, h as i64 - 1) as usize) * w + x.clamp(0, w as i64 - 1) as usize];
    let mut out = vec![0.0; w * h];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let gx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
            let gy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
            out[y as usize * w + x as usize] = (gx * gx + gy * gy).sqrt() / 8.0;
        }
    }
    out
}

fn percentile(v: &[f32], q: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s[((s.len() - 1) as f32 * q).round() as usize]
}

/// The working size for a source of `w`×`h`: its aspect at a long side of `WORK_SIDE` or less.
pub fn work_size(w: u32, h: u32) -> (u32, u32) {
    let long = w.max(h);
    if long <= WORK_SIDE {
        return (w, h);
    }
    let k = WORK_SIDE as f32 / long as f32;
    (((w as f32 * k).round() as u32).max(1), ((h as f32 * k).round() as u32).max(1))
}

fn fit(img: &RgbImage, (w, h): (u32, u32)) -> RgbImage {
    if img.width() == w && img.height() == h {
        img.clone()
    } else {
        imageops::resize(img, w, h, imageops::FilterType::Triangle)
    }
}

/// Measure one picture at its own size (reduce it first with `fit` to compare across sizes).
pub fn measure(img: &RgbImage) -> Measures {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let n = (w * h).max(1) as f32;
    let luma = luma_of(img);
    let chroma = chroma_of(img);
    let mean = luma.iter().sum::<f32>() / n;
    let sd = (luma.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / n).sqrt();
    let sig = if w >= 4 && h >= 4 { local_std(&luma, w, h, sigma_radius(w, h)) } else { vec![0.0; w * h] };
    let edges = sobel(&luma, w, h);
    let points = if w >= 8 && h >= 8 {
        let r = ((w.min(h) as f32) / 512.0).round().clamp(2.0, 6.0) as i32;
        let lin = super::painter::luma_map(img);
        super::painter::point_lights(&lin, w as u32, h as u32, r, 0.02).iter().filter(|p| p.ground < 0.25).count()
    } else {
        0
    };
    Measures {
        width: img.width(),
        height: img.height(),
        luma_mean: mean,
        luma_stddev: sd,
        dark_share: luma.iter().filter(|l| **l < 0.12).count() as f32 / n,
        light_share: luma.iter().filter(|l| **l > 0.88).count() as f32 / n,
        calm_share: sig.iter().filter(|v| **v < CALM).count() as f32 / n,
        busy_share: sig.iter().filter(|v| **v > BUSY).count() as f32 / n,
        sigma_mean: sig.iter().sum::<f32>() / n,
        chroma_mean: chroma.iter().sum::<f32>() / n,
        chroma_p90: percentile(&chroma, 0.9),
        points_per_mpx: points as f32 / (n / 1.0e6),
        bare_share: luma.iter().zip(&chroma).filter(|(l, c)| **l > 0.94 && **c < 0.04).count() as f32 / n,
        edge_mean: edges.iter().sum::<f32>() / n,
    }
}

/// A sheet against its source. `focus`, when given, is a 0..1 mask of any size (the face, the subject, the
/// director's attention) — resized to the working size; without it the focus is the central half of the sheet.
pub fn compare(source: &RgbImage, sheet: &RgbImage, focus: Option<(&[f32], u32, u32)>) -> Comparison {
    let size = work_size(source.width(), source.height());
    let (w, h) = (size.0 as usize, size.1 as usize);
    let n = (w * h).max(1) as f32;
    let a = fit(source, size);
    let b = fit(sheet, size);
    let (ma, mb) = (measure(&a), measure(&b));
    let (la, lb) = (luma_of(&a), luma_of(&b));
    let rmse = (la.iter().zip(&lb).map(|(x, y)| (x - y).powi(2)).sum::<f32>() / n).sqrt();
    // The tonal distributions.
    let hist = |l: &[f32]| {
        let mut hh = [0.0f32; 16];
        for v in l {
            hh[((v * 16.0) as usize).min(15)] += 1.0 / n;
        }
        hh
    };
    let (ha, hb) = (hist(&la), hist(&lb));
    let tone_overlap = ha.iter().zip(&hb).map(|(x, y)| x.min(*y)).sum::<f32>();
    // The local σ fields and the focus.
    let r = sigma_radius(w, h);
    let (sa, sb) = (local_std(&la, w, h, r), local_std(&lb, w, h, r));
    let in_focus: Vec<bool> = match focus {
        Some((m, mw, mh)) if mw > 0 && mh > 0 && m.len() == (mw * mh) as usize => (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let (sx, sy) = ((x * mw as usize / w).min(mw as usize - 1), (y * mh as usize / h).min(mh as usize - 1));
                m[sy * mw as usize + sx] > 0.5
            })
            .collect(),
        _ => (0..w * h).map(|i| { let (x, y) = (i % w, i / w); x * 4 >= w && x * 4 < 3 * w && y * 4 >= h && y * 4 < 3 * h }).collect(),
    };
    let ratio_over = |sel: &dyn Fn(usize) -> bool| {
        let (mut na, mut nb, mut c) = (0.0f32, 0.0f32, 0usize);
        for i in 0..w * h {
            if sel(i) {
                na += sa[i];
                nb += sb[i];
                c += 1;
            }
        }
        if c == 0 || na < 1e-6 { 1.0 } else { nb / na }
    };
    let contrast_focus = ratio_over(&|i| in_focus[i]);
    let contrast_periphery = ratio_over(&|i| !in_focus[i]);
    let share = |src: &dyn Fn(usize) -> bool, kept: &dyn Fn(usize) -> bool| {
        let (mut c, mut k) = (0usize, 0usize);
        for i in 0..w * h {
            if src(i) {
                c += 1;
                if kept(i) {
                    k += 1;
                }
            }
        }
        if c == 0 { 1.0 } else { k as f32 / c as f32 }
    };
    let calm_kept = share(&|i| sa[i] < CALM, &|i| sb[i] < CALM_KEPT);
    let busy_kept = share(&|i| sa[i] > BUSY, &|i| sb[i] > BUSY_KEPT);
    // The edges: the source's strong ones, found on the sheet within a pixel at half the strength.
    let (ea, eb) = (sobel(&la, w, h), sobel(&lb, w, h));
    let strong = percentile(&ea, 0.9).max(1e-4);
    let eb_max = |i: usize| {
        let (x, y) = ((i % w) as i64, (i / w) as i64);
        let mut m = 0.0f32;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (nx, ny) = (x + dx, y + dy);
                if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 {
                    m = m.max(eb[ny as usize * w + nx as usize]);
                }
            }
        }
        m
    };
    let edges_kept = share(&|i| ea[i] >= strong, &|i| eb_max(i) >= 0.5 * ea[i]);
    let src_points = ma.points_per_mpx * (n / 1.0e6);
    let points_kept = if src_points >= 5.0 { Some((mb.points_per_mpx / ma.points_per_mpx.max(1e-6)).min(4.0)) } else { None };
    let chroma_ratio = if ma.chroma_mean > 0.01 { Some(mb.chroma_mean / ma.chroma_mean) } else { None };
    let key_shift = mb.luma_mean - ma.luma_mean;
    let range_ratio = if ma.luma_stddev > 1e-4 { mb.luma_stddev / ma.luma_stddev } else { 1.0 };
    let closeness = {
        let parts = [tone_overlap, edges_kept, calm_kept, contrast_focus.min(1.0), 1.0 - (key_shift.abs() / 0.2).min(1.0), chroma_ratio.unwrap_or(1.0).min(1.0)];
        parts.iter().sum::<f32>() / parts.len() as f32
    };
    Comparison { source: ma, sheet: mb, rmse, tone_overlap, key_shift, range_ratio, contrast_focus, contrast_periphery, calm_kept, busy_kept, edges_kept, points_kept, chroma_ratio, closeness }
}

fn f2(v: f32) -> String {
    format!("{v:.2}")
}
fn pct(v: f32) -> String {
    format!("{:.0}%", v * 100.0)
}

/// The comparison as the lines of the plan's verdict (each a measurement with its reading).
pub fn readings(c: &Comparison) -> Vec<String> {
    let (s, t) = (&c.source, &c.sheet);
    let mut o = vec![
        format!("key: the sheet's mean luma {:.2} against the source's {:.2} ({}{:.2}; the spread {:.3} against {:.3}, ratio {:.2})", t.luma_mean, s.luma_mean, if c.key_shift >= 0.0 { "+" } else { "" }, c.key_shift, t.luma_stddev, s.luma_stddev, c.range_ratio),
        format!("tone: the histograms overlap {} ; RMSE of the luma {:.3}", pct(c.tone_overlap), c.rmse),
        format!("local contrast kept: {:.2} in the focus, {:.2} on the periphery (the sheet's mean local σ over the source's)", c.contrast_focus, c.contrast_periphery),
        format!("the calm fields ({} of the source) kept calm {}; the busy passages ({}) kept busy {}", pct(s.calm_share), pct(c.calm_kept), pct(s.busy_share), pct(c.busy_kept)),
        format!("edges: {} of the source's strong edges carried by the sheet", pct(c.edges_kept)),
    ];
    match c.points_kept {
        Some(k) => o.push(format!("points of light on a dark ground: {:.0} per Mpx on the source, {:.0} on the sheet ({:.2} kept)", s.points_per_mpx, t.points_per_mpx, k)),
        None => o.push(format!("points of light on a dark ground: too few on the source to count ({:.0} per Mpx)", s.points_per_mpx)),
    }
    match c.chroma_ratio {
        Some(k) => o.push(format!("colour: mean chroma {:.3} against {:.3} (ratio {:.2}); the strong colour (p90) {:.2} against {:.2}", t.chroma_mean, s.chroma_mean, k, t.chroma_p90, s.chroma_p90)),
        None => o.push("colour: a monochrome source".into()),
    }
    o.push(format!("bare white: {} of the sheet against {} of the source", pct(t.bare_share), pct(s.bare_share)));
    o.push(format!("closeness {:.2} (the mean of the kept shares — a summary, not a judgement)", c.closeness));
    o
}

/// The table: the source's measures in the first column, one column per sheet.
pub fn table(rows: &[(String, Comparison)]) -> String {
    let Some(first) = rows.first() else { return String::new() };
    let mut o = String::new();
    o.push_str("| measure | source |");
    for (name, _) in rows {
        o.push_str(&format!(" {name} |"));
    }
    o.push('\n');
    o.push_str("|---|---|");
    for _ in rows {
        o.push_str("---|");
    }
    o.push('\n');
    let mut line = |label: &str, src: String, each: &dyn Fn(&Comparison) -> String| {
        o.push_str(&format!("| {label} | {src} |"));
        for (_, c) in rows {
            o.push_str(&format!(" {} |", each(c)));
        }
        o.push('\n');
    };
    let s = &first.1.source;
    line("size", format!("{}×{}", s.width, s.height), &|c| format!("{}×{}", c.sheet.width, c.sheet.height));
    line("mean luma", f2(s.luma_mean), &|c| f2(c.sheet.luma_mean));
    line("luma σ (the value spread)", format!("{:.3}", s.luma_stddev), &|c| format!("{:.3}", c.sheet.luma_stddev));
    line("darks under 0.12", pct(s.dark_share), &|c| pct(c.sheet.dark_share));
    line("lights over 0.88", pct(s.light_share), &|c| pct(c.sheet.light_share));
    line("calm fields (local σ < 0.02)", pct(s.calm_share), &|c| pct(c.sheet.calm_share));
    line("busy passages (local σ > 0.08)", pct(s.busy_share), &|c| pct(c.sheet.busy_share));
    line("mean local σ", format!("{:.4}", s.sigma_mean), &|c| format!("{:.4}", c.sheet.sigma_mean));
    line("mean chroma", format!("{:.3}", s.chroma_mean), &|c| format!("{:.3}", c.sheet.chroma_mean));
    line("chroma p90", f2(s.chroma_p90), &|c| f2(c.sheet.chroma_p90));
    line("points of light / Mpx", format!("{:.0}", s.points_per_mpx), &|c| format!("{:.0}", c.sheet.points_per_mpx));
    line("bare white", pct(s.bare_share), &|c| pct(c.sheet.bare_share));
    line("edge energy", format!("{:.4}", s.edge_mean), &|c| format!("{:.4}", c.sheet.edge_mean));
    line("RMSE (luma)", "—".into(), &|c| format!("{:.3}", c.rmse));
    line("tone overlap", "—".into(), &|c| pct(c.tone_overlap));
    line("key shift", "—".into(), &|c| format!("{:+.2}", c.key_shift));
    line("value range ratio", "—".into(), &|c| f2(c.range_ratio));
    line("contrast kept, focus", "—".into(), &|c| f2(c.contrast_focus));
    line("contrast kept, periphery", "—".into(), &|c| f2(c.contrast_periphery));
    line("calm kept", "—".into(), &|c| pct(c.calm_kept));
    line("busy kept", "—".into(), &|c| pct(c.busy_kept));
    line("edges kept", "—".into(), &|c| pct(c.edges_kept));
    line("points kept", "—".into(), &|c| c.points_kept.map(f2).unwrap_or_else(|| "—".into()));
    line("chroma ratio", "—".into(), &|c| c.chroma_ratio.map(f2).unwrap_or_else(|| "—".into()));
    line("closeness", "—".into(), &|c| f2(c.closeness));
    o
}

// ── the contact sheet ──────────────────────────────────────────────────────────────────────────────────────────

/// A 5×7 bitmap glyph: seven rows of five cells, `#` set. The capitals, the digits and a little punctuation —
/// enough for a label under a thumbnail; a letter outside it is drawn as a space.
const GLYPHS: &[(char, &str)] = &[
    ('A', ".###.#...##...#######...##...##...#"),
    ('B', "####.#...##...#####.#...##...#####."),
    ('C', ".#####....#....#....#....#.....####"),
    ('D', "####.#...##...##...##...##...#####."),
    ('E', "######....#....####.#....#....#####"),
    ('F', "######....#....####.#....#....#...."),
    ('G', ".#####....#....#.####...##...#.####"),
    ('H', "#...##...##...#######...##...##...#"),
    ('I', "#####..#....#....#....#....#..#####"),
    ('J', "..###...#....#....#....#.#..#..##.."),
    ('K', "#...##..#.#.#..##...#.#..#..#.#...#"),
    ('L', "#....#....#....#....#....#....#####"),
    ('M', "#...###.###.#.##.#.##...##...##...#"),
    ('N', "#...###..##.#.##..###...##...##...#"),
    ('O', ".###.#...##...##...##...##...#.###."),
    ('P', "####.#...##...#####.#....#....#...."),
    ('Q', ".###.#...##...##...##.#.##..#..##.#"),
    ('R', "####.#...##...#####.#.#..#..#.#...#"),
    ('S', ".#####....#.....###.....#....#####."),
    ('T', "#####..#....#....#....#....#....#.."),
    ('U', "#...##...##...##...##...##...#.###."),
    ('V', "#...##...##...##...##...#.#.#...#.."),
    ('W', "#...##...##...##.#.##.#.###.###...#"),
    ('X', "#...##...#.#.#...#...#.#.#...##...#"),
    ('Y', "#...##...#.#.#...#....#....#....#.."),
    ('Z', "#####....#...#...#...#...#....#####"),
    ('0', ".###.#...##..###.#.###..##...#.###."),
    ('1', "..#...##....#....#....#....#...###."),
    ('2', ".###.#...#....#...#...#...#...#####"),
    ('3', "#####...#...#.....#.....##...#.###."),
    ('4', "...#...##..#.#.#..#.#####...#....#."),
    ('5', "######....####.....#....##...#.###."),
    ('6', "..###.#...#....####.#...##...#.###."),
    ('7', "#####....#...#...#...#....#....#..."),
    ('8', ".###.#...##...#.###.#...##...#.###."),
    ('9', ".###.#...##...#.####....#...#.###.."),
    ('.', "..........................##...##.."),
    ('-', "...............#####..............."),
    (':', "......##...##.........##...##......"),
    ('%', "##..###..#...#...#...#...#..###..##"),
    ('_', "..............................#####"),
    ('/', "....#...#....#...#...#...#....#...."),
    (' ', "..................................."),
];

/// Draw `text` in the bitmap font at (`x`, `y`), each cell `scale` pixels, in `colour`.
pub fn draw_text(img: &mut RgbImage, x: u32, y: u32, text: &str, scale: u32, colour: [u8; 3]) {
    let (w, h) = (img.width(), img.height());
    for (i, ch) in text.chars().enumerate() {
        let ch = ch.to_ascii_uppercase();
        let rows = GLYPHS.iter().find(|(c, _)| *c == ch).or_else(|| GLYPHS.iter().find(|(c, _)| *c == ' ')).map(|(_, g)| *g).unwrap_or("");
        for (k, cell) in rows.chars().enumerate() {
            if cell != '#' {
                continue;
            }
            let (cx, cy) = ((k % 5) as u32, (k / 5) as u32);
            for dy in 0..scale {
                for dx in 0..scale {
                    let px = x + (i as u32 * 6 + cx) * scale + dx;
                    let py = y + cy * scale + dy;
                    if px < w && py < h {
                        img.put_pixel(px, py, image::Rgb(colour));
                    }
                }
            }
        }
    }
}

/// The width in pixels of `text` drawn at `scale`.
pub fn text_width(text: &str, scale: u32) -> u32 {
    (text.chars().count() as u32 * 6).saturating_sub(1) * scale
}

/// The source and the sheets in a row, each scaled to `thumb_h` high, a label strip over each.
pub fn contact_sheet(source: &RgbImage, sheets: &[(String, RgbImage)], thumb_h: u32) -> RgbImage {
    let gutter = 8u32;
    let strip = 24u32;
    let scale = 2u32;
    let mut items: Vec<(String, RgbImage)> = vec![("SOURCE".into(), source.clone())];
    items.extend(sheets.iter().cloned());
    let thumbs: Vec<(String, RgbImage)> = items
        .into_iter()
        .map(|(name, img)| {
            let tw = ((img.width() as f32 * thumb_h as f32 / img.height().max(1) as f32).round() as u32).max(1);
            (name, imageops::resize(&img, tw, thumb_h, imageops::FilterType::Triangle))
        })
        .collect();
    let total_w: u32 = thumbs.iter().map(|(_, t)| t.width() + gutter).sum::<u32>() + gutter;
    let mut out = RgbImage::from_pixel(total_w.max(1), thumb_h + strip + 2 * gutter, image::Rgb([24, 24, 24]));
    let mut x = gutter;
    for (name, t) in &thumbs {
        let label: String = name.chars().take(((t.width() / (6 * scale)) as usize).max(1)).collect();
        draw_text(&mut out, x, gutter + (strip - 7 * scale) / 2, &label, scale, [230, 230, 230]);
        imageops::replace(&mut out, t, x as i64, (gutter + strip) as i64);
        x += t.width() + gutter;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient_with_checker(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            let base = 40.0 + 160.0 * x as f32 / w as f32;
            // A busy box in the centre, a calm ramp elsewhere.
            let busy = x > w / 3 && x < 2 * w / 3 && y > h / 3 && y < 2 * h / 3 && (x / 3 + y / 3) % 2 == 0;
            let v = if busy { 255.0 - base } else { base } as u8;
            image::Rgb([v, v, (v as f32 * 0.8) as u8])
        })
    }

    #[test]
    fn a_sheet_identical_to_its_source_keeps_everything() {
        let img = gradient_with_checker(256, 192);
        let c = compare(&img, &img, None);
        assert!(c.rmse < 1e-6 && (c.tone_overlap - 1.0).abs() < 1e-4, "rmse {} overlap {}", c.rmse, c.tone_overlap);
        assert!((c.contrast_focus - 1.0).abs() < 1e-4 && (c.contrast_periphery - 1.0).abs() < 1e-4);
        assert!((c.calm_kept - 1.0).abs() < 1e-6 && (c.busy_kept - 1.0).abs() < 1e-6 && (c.edges_kept - 1.0).abs() < 1e-6);
        assert!(c.key_shift.abs() < 1e-6 && (c.range_ratio - 1.0).abs() < 1e-4 && c.closeness > 0.99);
        assert_eq!(c.chroma_ratio.map(|r| (r - 1.0).abs() < 1e-4), Some(true));
    }

    #[test]
    fn a_blurred_lighter_copy_loses_its_edges_and_its_focus_and_shifts_its_key() {
        let img = gradient_with_checker(256, 192);
        let blurred = imageops::blur(&img, 4.0);
        let lighter = RgbImage::from_fn(256, 192, |x, y| {
            let p = blurred.get_pixel(x, y).0;
            image::Rgb([p[0].saturating_add(40), p[1].saturating_add(40), p[2].saturating_add(40)])
        });
        let c = compare(&img, &lighter, None);
        assert!(c.edges_kept < 0.5, "the blur loses the strong edges: {}", c.edges_kept);
        assert!(c.contrast_focus < 0.5, "the busy box blurred flat: {}", c.contrast_focus);
        assert!(c.busy_kept < 0.5, "the busy pixels no longer busy: {}", c.busy_kept);
        assert!(c.key_shift > 0.1, "lighter by 40/255: {}", c.key_shift);
        assert!(c.closeness < 0.8, "closeness {}", c.closeness);
        // A larger sheet of the same picture meets the source at the working size.
        let big = imageops::resize(&img, 512, 384, imageops::FilterType::Triangle);
        let c2 = compare(&img, &big, None);
        assert_eq!((c2.sheet.width, c2.sheet.height), (256, 192));
        assert!(c2.rmse < 0.03 && c2.edges_kept > 0.8, "rmse {} edges {}", c2.rmse, c2.edges_kept);
    }

    #[test]
    fn the_focus_mask_selects_the_region_it_names() {
        let img = gradient_with_checker(256, 192);
        // Flatten the central box only: the focus loses its contrast, the periphery keeps it.
        let flat = RgbImage::from_fn(256, 192, |x, y| if x > 256 / 3 && x < 512 / 3 && y > 64 && y < 128 { image::Rgb([128, 128, 100]) } else { *img.get_pixel(x, y) });
        let mask: Vec<f32> = (0..64 * 48).map(|i| { let (x, y) = (i % 64, i / 64); if x > 21 && x < 43 && y > 16 && y < 32 { 1.0 } else { 0.0 } }).collect();
        let c = compare(&img, &flat, Some((&mask, 64, 48)));
        assert!(c.contrast_focus < 0.3, "focus flattened: {}", c.contrast_focus);
        assert!(c.contrast_periphery > 0.8, "periphery untouched but for the window's bleed: {}", c.contrast_periphery);
    }

    #[test]
    fn the_table_and_the_sheet_render() {
        let img = gradient_with_checker(128, 96);
        let c = compare(&img, &img, None);
        let t = table(&[("A".into(), c.clone()), ("B".into(), c)]);
        assert!(t.starts_with("| measure | source | A | B |"), "{t}");
        assert_eq!(t.lines().count(), 2 + 25, "{t}");
        let sheet = contact_sheet(&img, &[("OIL-DIRECT V9".into(), img.clone())], 64);
        assert_eq!(sheet.height(), 64 + 24 + 16);
        // Some label pixels were drawn (the strip is not all background).
        let lit = sheet.pixels().filter(|p| p.0[0] > 200).count();
        assert!(lit > 50, "the label drawn: {lit}");
        for (ch, g) in GLYPHS {
            assert_eq!(g.len(), 35, "glyph {ch:?} is 5×7");
        }
    }
}
