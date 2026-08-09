//! G0.1 (ROADMAP COMIC-1) — the one novel weight-free algorithm: **speech-balloon placement + lettering**.
//! Given a panel, a subject/face exclusion mask, and dialogue lines with speaker points, it must:
//!   (a) word-wrap + FIT each string into a rounded balloon (largest font that fits a bounded box),
//!   (b) PLACE balloons in open space — off the mask, non-overlapping siblings, biased to the reading
//!       corner (top, L→R),
//!   (c) draw a TAIL from each balloon toward its speaker.
//! De-risks the algorithm before Track P2. Text metrics use a monospace approximation here (real
//! `ab_glyph` shaping — the proven `bookart::glyph` path — lands in P2); the placement/wrap/fit/tail
//! geometry is what G0 proves. Pure/weight-free:
//!
//!   cargo run --release --example comic_balloon_probe
//!
//! Exit: all balloons placed, zero overlap, every balloon clear of the mask, text fits, tails point the
//! right way — measured on a busy synthetic panel (a face mask + 4 speakers).

use image::{Rgb, RgbImage};

// --- monospace-ish text metrics (P2 swaps in ab_glyph) ---
fn char_w(px: f32) -> f32 {
    0.55 * px
}
fn line_h(px: f32) -> f32 {
    1.3 * px
}

/// Greedy word-wrap `text` at `font_px` into lines no wider than `max_w`. Returns (lines, widest line px).
fn wrap(text: &str, font_px: f32, max_w: f32) -> (Vec<String>, f32) {
    let cw = char_w(font_px);
    let mut lines = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        let trial = if cur.is_empty() { word.to_string() } else { format!("{cur} {word}") };
        if trial.chars().count() as f32 * cw <= max_w || cur.is_empty() {
            cur = trial;
        } else {
            lines.push(std::mem::take(&mut cur));
            cur = word.to_string();
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    let widest = lines.iter().map(|l| l.chars().count() as f32 * cw).fold(0.0, f32::max);
    (lines, widest)
}

struct Fit {
    font_px: f32,
    lines: Vec<String>,
    w: f32,
    h: f32,
}

/// Largest font whose wrapped text fits inside `max_w × max_h`; returns the fitted text box (no padding).
fn fit_text(text: &str, max_w: f32, max_h: f32) -> Fit {
    let mut best = Fit { font_px: 10.0, lines: vec![text.to_string()], w: max_w, h: line_h(10.0) };
    let mut px = 10.0f32;
    while px <= 40.0 {
        let (lines, widest) = wrap(text, px, max_w);
        let h = lines.len() as f32 * line_h(px);
        if widest <= max_w && h <= max_h {
            best = Fit { font_px: px, lines, w: widest, h };
        } else {
            break;
        }
        px += 1.0;
    }
    best
}

#[derive(Clone, Copy)]
struct Rect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}
impl Rect {
    fn overlaps(&self, o: &Rect) -> bool {
        self.x < o.x + o.w && self.x + self.w > o.x && self.y < o.y + o.h && self.y + self.h > o.y
    }
    fn cx(&self) -> f32 {
        self.x + self.w / 2.0
    }
    fn cy(&self) -> f32 {
        self.y + self.h / 2.0
    }
}

struct Balloon {
    text: String,
    speaker: (f32, f32),
}
struct Placed {
    rect: Rect,
    fit: Fit,
    tail_to: (f32, f32),
}

const PAD: f32 = 10.0;

/// Place `balloons` (in reading order) on a `pw × ph` panel, avoiding `mask` and each other, biased to the
/// top reading corner and toward each speaker.
fn place(pw: f32, ph: f32, mask: &Rect, balloons: &[Balloon]) -> Vec<Placed> {
    let mut placed: Vec<Placed> = Vec::new();
    for b in balloons {
        // Try progressively NARROWER balloons until one fits an open gap (what a letterer does when the
        // page is tight): a narrower box wraps to more/taller lines but squeezes into side gutters.
        let mut chosen: Option<(Rect, Fit)> = None;
        for &wf in &[0.42f32, 0.32, 0.24, 0.18] {
            let fit = fit_text(&b.text, pw * wf, ph * 0.55);
            let (bw, bh) = (fit.w + 2.0 * PAD, fit.h + 2.0 * PAD);
            let mut best: Option<(Rect, f32)> = None;
            let steps = 28;
            for iy in 0..steps {
                for ix in 0..steps {
                    let x = 6.0 + ix as f32 / (steps - 1) as f32 * (pw - bw - 12.0);
                    let y = 6.0 + iy as f32 / (steps - 1) as f32 * (ph - bh - 12.0);
                    let r = Rect { x, y, w: bw, h: bh };
                    if r.x < 0.0 || r.y < 0.0 || r.x + r.w > pw || r.y + r.h > ph {
                        continue;
                    }
                    if r.overlaps(mask) || placed.iter().any(|p| r.overlaps(&p.rect)) {
                        continue;
                    }
                    // score: prefer top (reading), near the speaker in x, slight L→R.
                    let dx = (r.cx() - b.speaker.0).abs();
                    let score = r.y + dx * 0.5 + r.x * 0.15;
                    if best.map(|(_, s)| score < s).unwrap_or(true) {
                        best = Some((r, score));
                    }
                }
            }
            if let Some((rect, _)) = best {
                chosen = Some((rect, fit));
                break; // widest box that fit
            }
        }
        if let Some((rect, fit)) = chosen {
            placed.push(Placed { rect, fit, tail_to: b.speaker });
        }
    }
    placed
}

// --- drawing (visual sanity) ---
fn fill_rect(img: &mut RgbImage, r: &Rect, c: Rgb<u8>) {
    for y in (r.y.max(0.0) as u32)..((r.y + r.h).min(img.height() as f32) as u32) {
        for x in (r.x.max(0.0) as u32)..((r.x + r.w).min(img.width() as f32) as u32) {
            img.put_pixel(x, y, c);
        }
    }
}
fn stroke_rect(img: &mut RgbImage, r: &Rect, c: Rgb<u8>) {
    for x in (r.x as i32)..(r.x + r.w) as i32 {
        for t in 0..2 {
            put(img, x, r.y as i32 + t, c);
            put(img, x, (r.y + r.h) as i32 - t, c);
        }
    }
    for y in (r.y as i32)..(r.y + r.h) as i32 {
        for t in 0..2 {
            put(img, r.x as i32 + t, y, c);
            put(img, (r.x + r.w) as i32 - t, y, c);
        }
    }
}
fn line(img: &mut RgbImage, a: (f32, f32), b: (f32, f32), c: Rgb<u8>) {
    let steps = ((a.0 - b.0).abs().max((a.1 - b.1).abs())) as i32 + 1;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        put(img, (a.0 + (b.0 - a.0) * t) as i32, (a.1 + (b.1 - a.1) * t) as i32, c);
    }
}
fn put(img: &mut RgbImage, x: i32, y: i32, c: Rgb<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, c);
    }
}

fn main() {
    let (pw, ph) = (800.0f32, 600.0f32);
    // a busy panel: the characters/faces occupy a central-lower band (the exclusion mask).
    let mask = Rect { x: 180.0, y: 300.0, w: 440.0, h: 260.0 };
    let balloons = vec![
        Balloon { text: "Did you hear that? I think something is following us down the alley.".into(), speaker: (260.0, 320.0) },
        Balloon { text: "SCANNING… TARGET ACQUIRED.".into(), speaker: (560.0, 340.0) },
        Balloon { text: "Run!".into(), speaker: (300.0, 360.0) },
        Balloon { text: "There's nowhere left to run, human.".into(), speaker: (540.0, 330.0) },
    ];

    let placed = place(pw, ph, &mask, &balloons);

    // measurements
    let all_placed = placed.len() == balloons.len();
    let mut overlaps = 0;
    for i in 0..placed.len() {
        for j in i + 1..placed.len() {
            if placed[i].rect.overlaps(&placed[j].rect) {
                overlaps += 1;
            }
        }
    }
    let off_mask = placed.iter().all(|p| !p.rect.overlaps(&mask));
    let text_fits = placed.iter().all(|p| p.fit.w <= p.rect.w - 2.0 * PAD + 0.5 && p.fit.h <= p.rect.h - 2.0 * PAD + 0.5);
    // tails point the right way: the speaker is outside the balloon (a real tail direction).
    let tails_ok = placed.iter().all(|p| {
        let inside = p.tail_to.0 >= p.rect.x && p.tail_to.0 <= p.rect.x + p.rect.w && p.tail_to.1 >= p.rect.y && p.tail_to.1 <= p.rect.y + p.rect.h;
        !inside
    });

    // render for eyeballing
    let mut img = RgbImage::from_pixel(pw as u32, ph as u32, Rgb([40, 42, 48]));
    fill_rect(&mut img, &mask, Rgb([70, 60, 60])); // the subject/face band
    for p in &placed {
        fill_rect(&mut img, &p.rect, Rgb([245, 245, 245]));
        stroke_rect(&mut img, &p.rect, Rgb([20, 20, 20]));
        // text lines as light-gray bars (metric preview)
        for (i, l) in p.fit.lines.iter().enumerate() {
            let ly = p.rect.y + PAD + i as f32 * line_h(p.fit.font_px);
            fill_rect(&mut img, &Rect { x: p.rect.x + PAD, y: ly + 2.0, w: l.chars().count() as f32 * char_w(p.fit.font_px), h: p.fit.font_px * 0.8 }, Rgb([90, 90, 90]));
        }
        // tail from the balloon edge toward the speaker
        let from = (p.rect.cx().clamp(p.rect.x, p.rect.x + p.rect.w), if p.tail_to.1 > p.rect.cy() { p.rect.y + p.rect.h } else { p.rect.y });
        line(&mut img, from, p.tail_to, Rgb([20, 20, 20]));
        put(&mut img, p.tail_to.0 as i32, p.tail_to.1 as i32, Rgb([255, 80, 80]));
    }
    let _ = img.save("/tmp/comic_balloon_probe.png");

    println!("G0.1 — balloon placement + lettering");
    println!("panel {pw}x{ph}, mask {}x{} at ({},{}), {} balloons", mask.w, mask.h, mask.x, mask.y, balloons.len());
    for (i, p) in placed.iter().enumerate() {
        println!("  balloon {i}: {}px, {} line(s), rect {:.0}x{:.0} at ({:.0},{:.0}) → speaker ({:.0},{:.0})", p.fit.font_px, p.fit.lines.len(), p.rect.w, p.rect.h, p.rect.x, p.rect.y, p.tail_to.0, p.tail_to.1);
    }
    println!("\n  all placed:  {all_placed} ({}/{})", placed.len(), balloons.len());
    println!("  overlaps:    {overlaps}");
    println!("  off mask:    {off_mask}");
    println!("  text fits:   {text_fits}");
    println!("  tails ok:    {tails_ok}");
    println!("  (saved /tmp/comic_balloon_probe.png)");
    let pass = all_placed && overlaps == 0 && off_mask && text_fits && tails_ok;
    println!("\n{}", if pass { "PASS — the placement/wrap/fit/tail algorithm holds on a busy panel → P2 uses it." } else { "FAIL — revisit before P2." });
}
