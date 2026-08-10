//! Speech-balloon placement + lettering (RFC COMIC-1 §3) — the one novel weight-free algorithm, proven in
//! G0.1 (`examples/comic_balloon_probe.rs`). Given a panel size, per-panel dialogue, and an optional
//! subject-exclusion mask, it (a) word-wraps + fits each line into the largest legible box that holds it,
//! (b) places boxes in open space — off the mask, non-overlapping, biased to the reading corner and any
//! `at` hint, (c) draws the balloon (speech/thought/shout/caption) + a tail toward the speaker. No GPU.
//!
//! Lettering rides the always-compiled 5×7 bitmap face (`crate::map::labels`) — all-caps, which is the
//! classic comic hand-lettered look; `--features shaped-labels` + a supplied font overrides it for
//! non-Latin scripts, exactly as map labels do.

use crate::map::labels;
use image::{Rgb, RgbImage};

// ---- vocabulary ----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Speech,
    Thought,
    Shout,
    Caption,
}
impl Kind {
    pub fn parse(s: Option<&str>) -> Kind {
        match s.map(|x| x.to_ascii_lowercase()).as_deref() {
            Some("thought") => Kind::Thought,
            Some("shout") => Kind::Shout,
            Some("caption") => Kind::Caption,
            _ => Kind::Speech,
        }
    }
    fn has_tail(self) -> bool {
        !matches!(self, Kind::Caption)
    }
}

/// A placement hint. `Auto` lets the placer choose; the rest bias the search origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    Auto,
    TopLeft,
    Top,
    TopRight,
    Left,
    Center,
    Right,
    BottomLeft,
    Bottom,
    BottomRight,
}
impl Anchor {
    pub fn parse(s: Option<&str>) -> Anchor {
        match s.map(|x| x.trim().to_ascii_lowercase().replace([' ', '_'], "-")).as_deref() {
            Some("top-left") => Anchor::TopLeft,
            Some("top") => Anchor::Top,
            Some("top-right") => Anchor::TopRight,
            Some("left") => Anchor::Left,
            Some("center") | Some("centre") => Anchor::Center,
            Some("right") => Anchor::Right,
            Some("bottom-left") => Anchor::BottomLeft,
            Some("bottom") => Anchor::Bottom,
            Some("bottom-right") => Anchor::BottomRight,
            _ => Anchor::Auto,
        }
    }
    /// Preferred centre (fractional 0..1 of the panel) for this anchor, or `None` for `Auto`.
    fn pref(self) -> Option<(f32, f32)> {
        let (fx, fy) = match self {
            Anchor::Auto => return None,
            Anchor::TopLeft => (0.22, 0.16),
            Anchor::Top => (0.5, 0.14),
            Anchor::TopRight => (0.78, 0.16),
            Anchor::Left => (0.2, 0.5),
            Anchor::Center => (0.5, 0.5),
            Anchor::Right => (0.8, 0.5),
            Anchor::BottomLeft => (0.22, 0.84),
            Anchor::Bottom => (0.5, 0.86),
            Anchor::BottomRight => (0.78, 0.84),
        };
        Some((fx, fy))
    }
    /// The preferred x (px) for this anchor on a panel of width `pw`, or `None` for `Auto`.
    pub fn pref_x(self, pw: f32) -> Option<f32> {
        self.pref().map(|(fx, _)| fx * pw)
    }
}

/// One line of dialogue destined for a balloon.
#[derive(Debug, Clone)]
pub struct Line {
    pub text: String,
    pub kind: Kind,
    pub anchor: Anchor,
    /// Where the tail points (panel-local px). `None` → a spread default (P3 supplies real face centroids).
    pub speaker: Option<(f32, f32)>,
}

#[derive(Debug, Clone, Copy)]
pub struct Rectf {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}
impl Rectf {
    fn overlaps(&self, o: &Rectf) -> bool {
        self.x < o.x + o.w && self.x + self.w > o.x && self.y < o.y + o.h && self.y + self.h > o.y
    }
    fn cx(&self) -> f32 {
        self.x + self.w / 2.0
    }
    fn cy(&self) -> f32 {
        self.y + self.h / 2.0
    }
}

/// A placed, fitted balloon ready to draw (panel-local coordinates).
#[derive(Debug, Clone)]
pub struct Placed {
    pub rect: Rectf,
    pub lines: Vec<String>,
    pub scale: u32,
    pub kind: Kind,
    pub tail_to: Option<(f32, f32)>,
}

// ---- fit: largest legible box that holds the text ----

/// Greedy word-wrap at bitmap `scale` into lines no wider than `max_w`. Returns (lines, widest line px).
fn wrap(text: &str, scale: u32, max_w: f32) -> (Vec<String>, f32) {
    let mut lines = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        let trial = if cur.is_empty() { word.to_string() } else { format!("{cur} {word}") };
        if (labels::text_width(&trial, scale) as f32) <= max_w || cur.is_empty() {
            cur = trial;
        } else {
            lines.push(std::mem::take(&mut cur));
            cur = word.to_string();
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    let widest = lines.iter().map(|l| labels::text_width(l, scale) as f32).fold(0.0, f32::max);
    (lines, widest)
}

struct Fit {
    scale: u32,
    lines: Vec<String>,
    w: f32,
    h: f32,
}

/// Largest bitmap `scale` (≤ `max_scale`) whose wrapped text fits `max_w × max_h`. `None` if even scale 1
/// overflows (caller then tries a wider box).
fn fit_text(text: &str, max_w: f32, max_h: f32, max_scale: u32) -> Option<Fit> {
    let mut best: Option<Fit> = None;
    for scale in 1..=max_scale.max(1) {
        let (lines, widest) = wrap(text, scale, max_w);
        let lh = labels::line_advance(scale) as f32;
        let h = lines.len() as f32 * lh;
        if widest <= max_w && h <= max_h {
            best = Some(Fit { scale, lines, w: widest, h });
        } else {
            break; // larger scales only get worse
        }
    }
    best
}

// ---- placement ----

const PAD: f32 = 8.0;

fn width_fractions(kind: Kind) -> &'static [f32] {
    match kind {
        // Captions span wide bands; balloons squeeze into gutters via progressively narrower boxes.
        Kind::Caption => &[0.9, 0.72, 0.55],
        _ => &[0.5, 0.4, 0.3, 0.22],
    }
}

/// Place `lines` (reading order) on a `pw × ph` panel, avoiding every `mask` (e.g. detected faces) and
/// each other, biased to each line's anchor (or the top reading corner) and toward its speaker. Lines that
/// cannot be placed are dropped (never overlapped) — the caller can report the count.
pub fn place(pw: f32, ph: f32, masks: &[Rectf], lines: &[Line]) -> Vec<Placed> {
    let max_scale = ((ph / 180.0) as u32).clamp(2, 9);
    let n = lines.len().max(1);
    let mut placed: Vec<Placed> = Vec::new();
    for (i, ln) in lines.iter().enumerate() {
        // default tail target: spread speakers across the lower band (P3 overrides with face centroids).
        let speaker = ln.speaker.unwrap_or((pw * (i as f32 + 1.0) / (n as f32 + 1.0), ph * 0.82));
        let mut chosen: Option<(Rectf, Fit)> = None;
        for &wf in width_fractions(ln.kind) {
            let Some(fit) = fit_text(&ln.text, pw * wf - 2.0 * PAD, ph * 0.6, max_scale) else { continue };
            let (bw, bh) = (fit.w + 2.0 * PAD, fit.h + 2.0 * PAD);
            if bw > pw || bh > ph {
                continue;
            }
            let mut best: Option<(Rectf, f32)> = None;
            let steps = 28;
            for iy in 0..steps {
                for ix in 0..steps {
                    let x = 4.0 + ix as f32 / (steps - 1) as f32 * (pw - bw - 8.0);
                    let y = 4.0 + iy as f32 / (steps - 1) as f32 * (ph - bh - 8.0);
                    let r = Rectf { x, y, w: bw, h: bh };
                    if r.x < 0.0 || r.y < 0.0 || r.x + r.w > pw || r.y + r.h > ph {
                        continue;
                    }
                    if masks.iter().any(|m| r.overlaps(m)) || placed.iter().any(|p| r.overlaps(&p.rect)) {
                        continue;
                    }
                    // score: bias to anchor (or top reading corner), then toward the speaker in x.
                    let score = match ln.anchor.pref() {
                        Some((fx, fy)) => (r.cx() - fx * pw).abs() + (r.cy() - fy * ph).abs(),
                        None => r.y + (r.cx() - speaker.0).abs() * 0.4 + r.x * 0.12,
                    };
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
            let tail_to = ln.kind.has_tail().then_some(speaker);
            placed.push(Placed { rect, lines: fit.lines, scale: fit.scale, kind: ln.kind, tail_to });
        }
    }
    placed
}

// ---- drawing ----

const INK: Rgb<u8> = Rgb([15, 15, 15]);
const FILL: Rgb<u8> = Rgb([255, 255, 255]);
const CAPTION_FILL: Rgb<u8> = Rgb([250, 244, 220]);

fn put(img: &mut RgbImage, x: i32, y: i32, c: Rgb<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, c);
    }
}

/// Signed test: is `(px,py)` inside the rounded rect `r` with corner radius `rad`?
fn in_round(px: f32, py: f32, r: &Rectf, rad: f32) -> bool {
    if px < r.x || py < r.y || px > r.x + r.w || py > r.y + r.h {
        return false;
    }
    let rad = rad.min(r.w / 2.0).min(r.h / 2.0);
    // clamp the point to the inner rectangle; distance to that clamp ≤ rad → inside the corner arc.
    let cx = px.clamp(r.x + rad, r.x + r.w - rad);
    let cy = py.clamp(r.y + rad, r.y + r.h - rad);
    let (dx, dy) = (px - cx, py - cy);
    dx * dx + dy * dy <= rad * rad
}

fn fill_round(img: &mut RgbImage, r: &Rectf, rad: f32, ox: i32, oy: i32, color: Rgb<u8>) {
    let (x0, y0) = (r.x.floor() as i32, r.y.floor() as i32);
    let (x1, y1) = ((r.x + r.w).ceil() as i32, (r.y + r.h).ceil() as i32);
    for y in y0..=y1 {
        for x in x0..=x1 {
            if in_round(x as f32 + 0.5, y as f32 + 0.5, r, rad) {
                put(img, x + ox, y + oy, color);
            }
        }
    }
}

fn stroke_round(img: &mut RgbImage, r: &Rectf, rad: f32, th: f32, ox: i32, oy: i32, color: Rgb<u8>) {
    let inner = Rectf { x: r.x + th, y: r.y + th, w: (r.w - 2.0 * th).max(0.0), h: (r.h - 2.0 * th).max(0.0) };
    let irad = (rad - th).max(0.0);
    let (x0, y0) = (r.x.floor() as i32, r.y.floor() as i32);
    let (x1, y1) = ((r.x + r.w).ceil() as i32, (r.y + r.h).ceil() as i32);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            if in_round(fx, fy, r, rad) && !in_round(fx, fy, &inner, irad) {
                put(img, x + ox, y + oy, color);
            }
        }
    }
}

/// Fill a convex polygon (scanline) at page offset. Used for tails + the shout burst.
fn fill_poly(img: &mut RgbImage, pts: &[(f32, f32)], ox: i32, oy: i32, color: Rgb<u8>) {
    let (mut ymin, mut ymax) = (f32::MAX, f32::MIN);
    for &(_, y) in pts {
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }
    let (y0, y1) = (ymin.floor() as i32, ymax.ceil() as i32);
    for y in y0..=y1 {
        let yc = y as f32 + 0.5;
        let mut xs: Vec<f32> = Vec::new();
        for i in 0..pts.len() {
            let (ax, ay) = pts[i];
            let (bx, by) = pts[(i + 1) % pts.len()];
            if (ay <= yc && by > yc) || (by <= yc && ay > yc) {
                let t = (yc - ay) / (by - ay);
                xs.push(ax + t * (bx - ax));
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut k = 0;
        while k + 1 < xs.len() {
            for x in (xs[k].floor() as i32)..=(xs[k + 1].ceil() as i32) {
                put(img, x + ox, y + oy, color);
            }
            k += 2;
        }
    }
}

fn stroke_poly(img: &mut RgbImage, pts: &[(f32, f32)], th: f32, ox: i32, oy: i32, color: Rgb<u8>) {
    for i in 0..pts.len() {
        line(img, pts[i], pts[(i + 1) % pts.len()], th, ox, oy, color);
    }
}

fn line(img: &mut RgbImage, a: (f32, f32), b: (f32, f32), th: f32, ox: i32, oy: i32, color: Rgb<u8>) {
    let steps = ((a.0 - b.0).abs().max((a.1 - b.1).abs())).ceil() as i32 + 1;
    let r = (th / 2.0).ceil() as i32;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let (cx, cy) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
        for dy in -r..=r {
            for dx in -r..=r {
                put(img, cx as i32 + dx + ox, cy as i32 + dy + oy, color);
            }
        }
    }
}

fn disc(img: &mut RgbImage, cx: f32, cy: f32, rad: f32, ox: i32, oy: i32, color: Rgb<u8>) {
    let r = rad.ceil() as i32;
    for dy in -r..=r {
        for dx in -r..=r {
            if (dx * dx + dy * dy) as f32 <= rad * rad {
                put(img, cx as i32 + dx + ox, cy as i32 + dy + oy, color);
            }
        }
    }
}

/// The tail base edge point + two flanking base points, for a given target, on rect `r`.
fn tail_base(r: &Rectf, target: (f32, f32)) -> ((f32, f32), (f32, f32)) {
    let (dx, dy) = (target.0 - r.cx(), target.1 - r.cy());
    let base = (r.w.min(r.h) * 0.28).max(6.0);
    if dx.abs() > dy.abs() {
        // left / right edge
        let ex = if dx > 0.0 { r.x + r.w } else { r.x };
        let ey = r.cy().clamp(r.y + base, r.y + r.h - base);
        ((ex, ey - base / 2.0), (ex, ey + base / 2.0))
    } else {
        let ey = if dy > 0.0 { r.y + r.h } else { r.y };
        let ex = r.cx().clamp(r.x + base, r.x + r.w - base);
        ((ex - base / 2.0, ey), (ex + base / 2.0, ey))
    }
}

/// Draw one placed balloon onto `page` at panel offset `(ox, oy)`.
pub fn draw(page: &mut RgbImage, p: &Placed, ox: i32, oy: i32) {
    let th = (p.scale as f32).max(2.0);
    let r = p.rect;
    let (fill, rad) = match p.kind {
        Kind::Caption => (CAPTION_FILL, 2.0 * p.scale as f32),
        Kind::Thought => (FILL, (r.h * 0.5).min(r.w * 0.5)),
        _ => (FILL, 6.0 * p.scale as f32),
    };

    match p.kind {
        Kind::Shout => {
            // spiky burst: alternate outer/inner radius around the box centre.
            let (cx, cy) = (r.cx(), r.cy());
            let (rx, ry) = (r.w * 0.72, r.h * 0.72);
            let spikes = 14;
            let mut pts = Vec::with_capacity(spikes * 2);
            for k in 0..spikes * 2 {
                let ang = k as f32 / (spikes * 2) as f32 * std::f32::consts::TAU;
                let f = if k % 2 == 0 { 1.0 } else { 0.72 };
                pts.push((cx + ang.cos() * rx * f, cy + ang.sin() * ry * f));
            }
            fill_poly(page, &pts, ox, oy, fill);
            stroke_poly(page, &pts, th, ox, oy, INK);
        }
        _ => {
            // tail first (under the body outline) for speech/thought.
            if let Some(t) = p.tail_to {
                match p.kind {
                    Kind::Thought => {
                        // a trail of shrinking bubbles from the balloon edge toward the speaker.
                        let (b0, b1) = tail_base(&r, t);
                        let start = ((b0.0 + b1.0) / 2.0, (b0.1 + b1.1) / 2.0);
                        for (j, f) in [0.35f32, 0.62, 0.85].iter().enumerate() {
                            let cx = start.0 + (t.0 - start.0) * f;
                            let cy = start.1 + (t.1 - start.1) * f;
                            let rad = (r.h * 0.14) * (1.0 - j as f32 * 0.28);
                            disc(page, cx, cy, rad + th, ox, oy, INK);
                            disc(page, cx, cy, rad, ox, oy, fill);
                        }
                    }
                    _ => {
                        let (b0, b1) = tail_base(&r, t);
                        let apex = (r.cx() + (t.0 - r.cx()) * 0.9, r.cy() + (t.1 - r.cy()) * 0.9);
                        fill_poly(page, &[b0, b1, apex], ox, oy, fill);
                        // outline the two tail flanks (the base sits under the body).
                        line(page, b0, apex, th, ox, oy, INK);
                        line(page, b1, apex, th, ox, oy, INK);
                    }
                }
            }
            fill_round(page, &r, rad, ox, oy, fill);
            stroke_round(page, &r, rad, th, ox, oy, INK);
        }
    }

    // lettering: centred lines, all-caps bitmap face (or shaped font if active).
    let lh = labels::line_advance(p.scale) as f32;
    let total_h = p.lines.len() as f32 * lh;
    let mut ty = r.cy() - total_h / 2.0;
    for l in &p.lines {
        let tw = labels::text_width(l, p.scale) as f32;
        let tx = r.cx() - tw / 2.0;
        labels::draw_text(page, ox + tx as i32, oy + ty as i32, l, p.scale, [INK.0[0], INK.0[1], INK.0[2]]);
        ty += lh;
    }
}

/// Collect a panel's dialogue (caption + balloons) into placement inputs, in reading order.
pub fn lines_for_panel(panel: &crate::comic::spec::Panel) -> Vec<Line> {
    let mut out = Vec::new();
    if let Some(cap) = panel.caption.as_deref().filter(|s| !s.trim().is_empty()) {
        out.push(Line { text: cap.to_string(), kind: Kind::Caption, anchor: Anchor::Top, speaker: None });
    }
    for b in &panel.balloons {
        let Some(say) = b.say.as_deref().filter(|s| !s.trim().is_empty()) else { continue };
        out.push(Line {
            text: say.to_string(),
            kind: Kind::parse(b.kind.as_deref()),
            anchor: Anchor::parse(b.at.as_deref()),
            speaker: None,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn places_and_fits_without_overlap() {
        let lines = vec![
            Line { text: "Did you hear that down the alley?".into(), kind: Kind::Speech, anchor: Anchor::Auto, speaker: Some((160.0, 520.0)) },
            Line { text: "SCANNING. TARGET ACQUIRED.".into(), kind: Kind::Shout, anchor: Anchor::TopRight, speaker: Some((640.0, 520.0)) },
            Line { text: "Run!".into(), kind: Kind::Speech, anchor: Anchor::Auto, speaker: Some((300.0, 540.0)) },
        ];
        let mask = Rectf { x: 180.0, y: 320.0, w: 440.0, h: 260.0 };
        let placed = place(800.0, 600.0, &[mask], &lines);
        assert_eq!(placed.len(), 3, "all placed");
        for (i, p) in placed.iter().enumerate() {
            assert!(!p.lines.is_empty() && p.scale >= 1);
            // clear of the mask
            assert!(!p.rect.overlaps(&mask), "balloon {i} off mask");
            // clear of siblings
            for q in &placed[i + 1..] {
                assert!(!p.rect.overlaps(&q.rect), "balloon {i} no sibling overlap");
            }
            // fits inside the panel
            assert!(p.rect.x >= 0.0 && p.rect.y >= 0.0 && p.rect.x + p.rect.w <= 800.0 && p.rect.y + p.rect.h <= 600.0);
        }
        // the shout has no straight tail target stored the same way — but speech ones do.
        assert!(placed[0].tail_to.is_some());
    }

    #[test]
    fn draw_puts_ink_and_letters() {
        let placed = place(400.0, 300.0, &[], &[Line { text: "HELLO".into(), kind: Kind::Speech, anchor: Anchor::Top, speaker: Some((200.0, 260.0)) }]);
        assert_eq!(placed.len(), 1);
        let mut page = RgbImage::from_pixel(400, 300, Rgb([120, 120, 120]));
        draw(&mut page, &placed[0], 0, 0);
        // white balloon fill landed somewhere, and black ink (outline/letters) too.
        let mut white = 0;
        let mut black = 0;
        for px in page.pixels() {
            if px.0 == [255, 255, 255] {
                white += 1;
            }
            if px.0[0] < 40 && px.0[1] < 40 && px.0[2] < 40 {
                black += 1;
            }
        }
        assert!(white > 200, "balloon body drawn ({white}px)");
        assert!(black > 40, "outline + letters drawn ({black}px)");
    }
}
