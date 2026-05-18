//! v3: smart zone overrides derived from the generated image itself.
//!
//! Two cheap signals:
//!
//! * **Depth** (Depth-Anything-V2-small) drives the four vertical
//!   bands (`sky` / `far_plan` / `middle_plan` / `close_plan`). Per-
//!   row mean depth is bucketed by quantile (q25 / q50 / q75); each
//!   band's `[y0, y1]` extent is the row range labelled with that
//!   band. For typical landscape scenes (sky at top, ground at
//!   bottom) the bands come out ordered top → bottom; for weird
//!   scenes a band might collapse to a thin slice, which is fine
//!   — it just means the model's depth estimate disagrees with the
//!   rigid grid's assumption.
//! * **Luminance** drives the three horizontal bands
//!   (`left` / `center` / `right`). We compute per-column vertical
//!   variance ("how busy is this column?"), find the variance-
//!   weighted column centroid, and centre a 1/3-width window on it.
//!   `left` and `right` fill the remainder symmetrically. No extra
//!   ML model — luminance is a 1-pass image read.
//!
//! Both signals fail soft: when no rows match a depth band (e.g. all
//! rows clamp to the same quantile due to a flat depth field), the
//! band's grid default is kept; same for the horizontal split when
//! the variance signal is degenerate.
//!
//! The output is a [`ZoneOverrides`] — drop-in for the v1/v2
//! compositor. v3 doesn't change anything downstream.

use anyhow::{Context, Result};
use std::path::Path;

use crate::artefacts::ZoneOverrides;
use crate::pipelines::depth::DepthPipeline;

/// Width (normalized) of the horizontal centre band after variance
/// centring. Same as the rigid grid (1/3) — we only shift the band,
/// we don't widen / shrink it.
const CENTER_WIDTH_FRAC: f32 = 1.0 / 3.0;

/// Derive smart zone overrides from a generated image. The depth
/// pipeline is borrowed (caller manages its lifetime to amortize
/// the model load across many images).
pub fn smart_zones_from_image(
    image_path: &Path,
    width: u32,
    height: u32,
    depth: &DepthPipeline,
) -> Result<ZoneOverrides> {
    let depth_map = depth
        .depth_map(image_path, width, height)
        .with_context(|| format!("computing depth on {}", image_path.display()))?;
    let vertical = vertical_bands_from_depth(&depth_map, width as usize, height as usize);
    let horizontal = horizontal_bands_from_image(image_path)
        .with_context(|| format!("computing horizontal split on {}", image_path.display()))?;
    Ok(ZoneOverrides {
        sky: vertical.sky,
        far_plan: vertical.far_plan,
        middle_plan: vertical.middle_plan,
        close_plan: vertical.close_plan,
        left: horizontal.left,
        center: horizontal.center,
        right: horizontal.right,
    })
}

struct VerticalBands {
    sky: Option<[f32; 2]>,
    far_plan: Option<[f32; 2]>,
    middle_plan: Option<[f32; 2]>,
    close_plan: Option<[f32; 2]>,
}

struct HorizontalBands {
    left: Option<[f32; 2]>,
    center: Option<[f32; 2]>,
    right: Option<[f32; 2]>,
}

/// Compute the four vertical depth bands. Larger depth = closer (the
/// Depth-Anything-V2 convention after min-max normalisation).
/// So *sky* = the lowest-depth quartile of rows.
fn vertical_bands_from_depth(depth_map: &[f32], w: usize, h: usize) -> VerticalBands {
    if w == 0 || h == 0 || depth_map.len() != w * h {
        return VerticalBands {
            sky: None,
            far_plan: None,
            middle_plan: None,
            close_plan: None,
        };
    }

    // Per-row mean depth.
    let mut row_depth = vec![0f32; h];
    for y in 0..h {
        let row = &depth_map[y * w..y * w + w];
        let s: f32 = row.iter().copied().sum();
        row_depth[y] = s / w as f32;
    }

    // Quantile thresholds. Sort a clone of row_depth to read off
    // q25 / q50 / q75 (linear interpolation between neighbours).
    let mut sorted = row_depth.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q25 = quantile(&sorted, 0.25);
    let q50 = quantile(&sorted, 0.50);
    let q75 = quantile(&sorted, 0.75);

    // Degenerate: all rows the same depth → fall back to grid for every band.
    if q75 - q25 < 1e-6 {
        return VerticalBands {
            sky: None,
            far_plan: None,
            middle_plan: None,
            close_plan: None,
        };
    }

    // Label each row by its band.
    let mut sky_rows = (h, 0usize); // (min, max) — track extent
    let mut far_rows = (h, 0usize);
    let mut mid_rows = (h, 0usize);
    let mut close_rows = (h, 0usize);
    let mut sky_n = 0;
    let mut far_n = 0;
    let mut mid_n = 0;
    let mut close_n = 0;
    for (y, &d) in row_depth.iter().enumerate() {
        let bucket = if d < q25 {
            0
        } else if d < q50 {
            1
        } else if d < q75 {
            2
        } else {
            3
        };
        match bucket {
            0 => {
                sky_rows = update_extent(sky_rows, y);
                sky_n += 1;
            }
            1 => {
                far_rows = update_extent(far_rows, y);
                far_n += 1;
            }
            2 => {
                mid_rows = update_extent(mid_rows, y);
                mid_n += 1;
            }
            _ => {
                close_rows = update_extent(close_rows, y);
                close_n += 1;
            }
        }
    }

    let to_band = |extent: (usize, usize), count: usize| -> Option<[f32; 2]> {
        if count == 0 {
            return None;
        }
        let y0 = extent.0 as f32 / h as f32;
        let y1 = (extent.1 as f32 + 1.0) / h as f32;
        Some([y0.clamp(0.0, 1.0), y1.clamp(0.0, 1.0)])
    };

    VerticalBands {
        sky: to_band(sky_rows, sky_n),
        far_plan: to_band(far_rows, far_n),
        middle_plan: to_band(mid_rows, mid_n),
        close_plan: to_band(close_rows, close_n),
    }
}

fn update_extent(curr: (usize, usize), y: usize) -> (usize, usize) {
    let (mn, mx) = curr;
    (mn.min(y), mx.max(y))
}

/// Linear quantile of a sorted slice. `q` in `[0, 1]`.
fn quantile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let w = pos - lo as f32;
    sorted[lo] * (1.0 - w) + sorted[hi] * w
}

/// Compute the centre-of-content column via per-column vertical
/// variance of luminance, then derive left/center/right bands.
fn horizontal_bands_from_image(path: &Path) -> Result<HorizontalBands> {
    let img = image::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        return Ok(HorizontalBands {
            left: None,
            center: None,
            right: None,
        });
    }
    let raw = img.as_raw();

    // Per-column luminance mean + variance, single pass.
    let mut sum = vec![0f64; w];
    let mut sum_sq = vec![0f64; w];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let r = raw[i] as f64;
            let g = raw[i + 1] as f64;
            let b = raw[i + 2] as f64;
            let y_lum = 0.299 * r + 0.587 * g + 0.114 * b;
            sum[x] += y_lum;
            sum_sq[x] += y_lum * y_lum;
        }
    }
    let inv_h = 1.0 / h as f64;
    let mut var = vec![0f64; w];
    for x in 0..w {
        let mean = sum[x] * inv_h;
        var[x] = (sum_sq[x] * inv_h - mean * mean).max(0.0);
    }

    let total_var: f64 = var.iter().sum();
    if total_var < 1e-6 {
        // Flat image — keep default grid for horizontals.
        return Ok(HorizontalBands {
            left: None,
            center: None,
            right: None,
        });
    }
    let centroid_px: f64 = var
        .iter()
        .enumerate()
        .map(|(x, &v)| x as f64 * v)
        .sum::<f64>()
        / total_var;
    let centroid_frac = (centroid_px / w as f64) as f32;

    let half = CENTER_WIDTH_FRAC * 0.5;
    let mut c0 = (centroid_frac - half).max(0.0);
    let mut c1 = (centroid_frac + half).min(1.0);
    // Maintain centre-band width if clipped at one edge.
    if (c1 - c0) < CENTER_WIDTH_FRAC - 1e-3 {
        if c0 <= 0.0 {
            c1 = CENTER_WIDTH_FRAC.min(1.0);
        } else if c1 >= 1.0 {
            c0 = (1.0 - CENTER_WIDTH_FRAC).max(0.0);
        }
    }

    let left = if c0 > 0.0 { Some([0.0, c0]) } else { None };
    let right = if c1 < 1.0 { Some([c1, 1.0]) } else { None };
    let center = Some([c0, c1]);

    Ok(HorizontalBands {
        left,
        center,
        right,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_depth_top_to_bottom(w: usize, h: usize) -> Vec<f32> {
        // Smooth gradient: row 0 = 0.0 (sky), row h-1 = 1.0 (close).
        let mut v = vec![0f32; w * h];
        for y in 0..h {
            let val = if h == 1 { 0.0 } else { y as f32 / (h - 1) as f32 };
            for x in 0..w {
                v[y * w + x] = val;
            }
        }
        v
    }

    #[test]
    fn vertical_bands_partition_a_monotonic_scene() {
        let w = 32;
        let h = 64;
        let depth = synthetic_depth_top_to_bottom(w, h);
        let bands = vertical_bands_from_depth(&depth, w, h);

        let sky = bands.sky.expect("sky band");
        let far = bands.far_plan.expect("far band");
        let mid = bands.middle_plan.expect("middle band");
        let close = bands.close_plan.expect("close band");

        // Sky (lowest depth) should land near the top of the image.
        assert!(sky[0] < 0.05, "sky y0 should be near 0, got {sky:?}");
        assert!(sky[1] < 0.30, "sky y1 should be in the top quarter-ish, got {sky:?}");

        // close (highest depth) near the bottom.
        assert!(close[1] > 0.95, "close y1 should be near 1, got {close:?}");
        assert!(close[0] > 0.70, "close y0 should be near 0.75, got {close:?}");

        // Bands should be ordered top → bottom (sky ≤ far ≤ mid ≤ close on y0).
        assert!(sky[0] <= far[0] && far[0] <= mid[0] && mid[0] <= close[0]);
    }

    #[test]
    fn vertical_bands_collapse_returns_none_on_flat_depth() {
        let depth = vec![0.5f32; 16 * 16];
        let bands = vertical_bands_from_depth(&depth, 16, 16);
        assert!(bands.sky.is_none());
        assert!(bands.far_plan.is_none());
        assert!(bands.middle_plan.is_none());
        assert!(bands.close_plan.is_none());
    }

    #[test]
    fn quantile_interpolates_linearly() {
        let v = vec![0.0, 1.0, 2.0, 3.0];
        assert!((quantile(&v, 0.0) - 0.0).abs() < 1e-6);
        assert!((quantile(&v, 1.0) - 3.0).abs() < 1e-6);
        assert!((quantile(&v, 0.5) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn horizontal_bands_centre_on_busy_column() {
        use image::{Rgb, RgbImage};
        // Image with a small white square in the right third only.
        // Columns inside the square have vertical variance (white + black
        // pixels); other columns are all black (zero variance). The
        // variance centroid should land in the right third.
        let mut img = RgbImage::from_pixel(96, 32, Rgb([0u8, 0, 0]));
        for y in 10..20 {
            for x in 70..80 {
                img.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
        let tmp = std::env::temp_dir().join("plakat_smart_zones_test.png");
        img.save(&tmp).unwrap();
        let bands = horizontal_bands_from_image(&tmp).unwrap();
        let center = bands.center.expect("center band");
        // Centre should be shifted toward the right (>0.5).
        let mid = (center[0] + center[1]) * 0.5;
        assert!(mid > 0.5, "centroid expected right of mid, got {mid}");
    }
}
