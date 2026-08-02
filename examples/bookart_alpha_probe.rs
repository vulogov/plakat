//! G0.5 gating probe (RFC BOOKART-1 §13 / ROADMAP_BOOKART_1 G0.5): tune the
//! **luminance→alpha** transparency curve — the headline of `plakat bookart`.
//!
//! B/W book ornament must be transparent so it sits on any page. The clean model:
//! ink darkness *is* opacity — `alpha = f(1 − luminance)`. But the exact curve
//! matters: too flat (linear) and thin anti-aliased grey lines go faint + a subtle
//! grey haze tints the page; too steep and fine lines drop out. This probe builds a
//! synthetic B/W ornament test card (gradient wedge, fine lines of increasing width,
//! cross-hatch, stipple, solid mass, mid-grey lines), applies several
//! `(white_cut, gamma)` curves, and composites each over a warm page and a mid-grey
//! so the trade-off is visible. Column 0 shows the ink over a checkerboard (raw
//! transparency).
//!
//! run:  cargo run --release --example bookart_alpha_probe -- <out.png>
//! curve: cov = clamp((1−L − white_cut)/(1 − white_cut)); alpha = cov^gamma

use image::{GrayImage, Luma, Rgb, RgbImage};

const S: u32 = 256;

/// Synthetic B/W ornament test card: white paper (255) with black/grey marks (0..).
fn test_card() -> GrayImage {
    let mut g = GrayImage::from_pixel(S, S, Luma([255]));
    let put = |g: &mut GrayImage, x: i32, y: i32, v: u8| {
        if x >= 0 && y >= 0 && (x as u32) < S && (y as u32) < S {
            g.put_pixel(x as u32, y as u32, Luma([v]));
        }
    };
    // (a) top strip: horizontal grey gradient wedge, L 255→0
    for x in 0..S {
        let l = 255 - (x * 255 / (S - 1)) as u8;
        for y in 0..28 {
            g.put_pixel(x, y, Luma([l]));
        }
    }
    // (b) fine black lines, widths 1..4 px, spaced (upper-left)
    let mut y = 40i32;
    for w in 1..=4 {
        for dy in 0..w {
            for x in 8..120 {
                put(&mut g, x, y + dy, 0);
            }
        }
        y += w + 10;
    }
    // (c) mid-grey lines (test anti-aliased / grey-mark preservation), widths 1..3
    for w in 1..=3 {
        for dy in 0..w {
            for x in 8..120 {
                put(&mut g, x, y + dy, 128);
            }
        }
        y += w + 10;
    }
    // (d) cross-hatch region (upper-right): 45° + 135° black lines
    for i in 0..40 {
        let off = i * 6;
        for t in 0..60 {
            put(&mut g, 140 + t, 40 + off - t, 0); //  ╲
            put(&mut g, 140 + t, 40 - 60 + off + t, 0); //  ╱
        }
    }
    // (e) stipple region (lower-left): deterministic dots of varying density
    for j in 0..48 {
        for i in 0..48 {
            // density rises to the right via a simple hash gate
            let h = ((i * 73 + j * 149) % 100) as u32;
            if h < 12 + (i as u32) {
                put(&mut g, 12 + i * 2, 150 + j * 2, 0);
            }
        }
    }
    // (f) solid black mass (lower-right)
    for yy in 150..230 {
        for xx in 150..240 {
            put(&mut g, xx, yy, 0);
        }
    }
    g
}

/// The candidate curve.
fn alpha_from_luma(l: u8, white_cut: f32, gamma: f32) -> u8 {
    let cov = (255.0 - l as f32) / 255.0; // ink coverage 0..1
    let cov = ((cov - white_cut) / (1.0 - white_cut)).clamp(0.0, 1.0);
    (cov.powf(gamma) * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Composite black ink at coverage `a` over background `bg`.
fn over(bg: Rgb<u8>, a: u8) -> Rgb<u8> {
    let af = a as f32 / 255.0; // ink is black → out = bg*(1-a)
    Rgb([
        (bg[0] as f32 * (1.0 - af)).round() as u8,
        (bg[1] as f32 * (1.0 - af)).round() as u8,
        (bg[2] as f32 * (1.0 - af)).round() as u8,
    ])
}

fn checker(x: u32, y: u32) -> Rgb<u8> {
    if ((x / 16) + (y / 16)) % 2 == 0 { Rgb([210, 210, 210]) } else { Rgb([245, 245, 245]) }
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "bookart_alpha_grid.png".into());
    let card = test_card();

    // (white_cut, gamma, label)
    let curves = [
        (0.00f32, 1.0f32, "linear  (1-L)"),
        (0.00, 0.6, "gamma 0.6 boost"),
        (0.05, 1.0, "white-cut .05, lin"),
        (0.08, 0.7, "white-cut .08, g .70"),
    ];
    let cream = Rgb([232u8, 220, 192]);
    let gray = Rgb([128u8, 128, 128]);

    let cols = 3u32; // [ink/checker, over cream, over gray]
    let rows = curves.len() as u32;
    let mut grid = RgbImage::from_pixel(cols * S, rows * S, Rgb([255, 255, 255]));

    for (r, &(wc, gm, label)) in curves.iter().enumerate() {
        let oy = r as u32 * S;
        // stats: page-tint (mean alpha over the near-white gradient columns) and
        // thin-line retention (alpha of the 1px black line vs 1px grey line).
        let mut page_tint = 0u64;
        let mut n_tint = 0u64;
        for x in 0..S {
            for y in 0..28 {
                let l = card.get_pixel(x, y).0[0];
                if l > 235 {
                    page_tint += alpha_from_luma(l, wc, gm) as u64;
                    n_tint += 1;
                }
            }
        }
        let black1 = alpha_from_luma(0, wc, gm);
        let grey1 = alpha_from_luma(128, wc, gm);
        let mean_tint = if n_tint > 0 { page_tint as f32 / n_tint as f32 } else { 0.0 };
        println!(
            "row {r}: {label:22}  page-haze α(near-white)={mean_tint:5.1}  black-ink α={black1:3}  mid-grey-line α={grey1:3}"
        );

        for x in 0..S {
            for y in 0..S {
                let a = alpha_from_luma(card.get_pixel(x, y).0[0], wc, gm);
                grid.put_pixel(0 * S + x, oy + y, over(checker(x, y), a));
                grid.put_pixel(1 * S + x, oy + y, over(cream, a));
                grid.put_pixel(2 * S + x, oy + y, over(gray, a));
            }
        }
    }
    grid.save(&out).expect("save grid");
    println!("\nrows = curves (top→bottom): {:?}", curves.iter().map(|c| c.2).collect::<Vec<_>>());
    println!("cols = [ink over checkerboard | over cream page | over mid-grey]");
    println!("wrote {out}");
}
