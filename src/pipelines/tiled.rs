//! Helpers for tiled hi-res generation (MultiDiffusion).
//!
//! Strategy: split a hi-res latent canvas into overlapping windows,
//! denoise each window with the regular UNet at every timestep, then
//! blend the per-window noise predictions back into a full-size noise
//! prediction via a 2D Hann (cosine) window. A single scheduler then
//! drives the full-size latent through `scheduler.step`.
//!
//! The model never sees the full canvas — it operates entirely within
//! each tile's native receptive field — so SDXL's effective working
//! resolution stays at the tile size (1024² by default). This is what
//! makes 4K+ generation tractable on consumer GPUs.
//!
//! Reference:
//!   MultiDiffusion (Bar-Tal et al., 2023)
//!   <https://multidiffusion.github.io/>

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// User-facing tile-mode config. Pixel units throughout; the pipeline
/// divides by the VAE downsample factor (8) internally to get latent
/// units.
#[derive(Debug, Clone, Copy)]
pub struct TiledConfig {
    /// Tile size in pixels (square). Default 1024 — SDXL's native
    /// working resolution. Larger tiles preserve more global
    /// coherence per pass but trade off the MultiDiffusion benefit of
    /// staying within the model's trained scale.
    pub tile_size: u32,
    /// Stride between consecutive tile origins in pixels. Smaller
    /// stride = more overlap = smoother seams = more tiles + more
    /// compute. Default 768 (256 px overlap, ~25 % of a 1024 tile).
    pub stride: u32,
}

impl Default for TiledConfig {
    fn default() -> Self {
        Self {
            tile_size: 1024,
            stride: 768,
        }
    }
}

/// One tile location in latent coordinates. `x` and `y` are the
/// top-left corner of the window into the full latent canvas; `size`
/// is the side length (square tiles).
#[derive(Debug, Clone, Copy)]
pub struct TilePos {
    pub y: usize,
    pub x: usize,
    pub size: usize,
}

/// Walk a [0, canvas_size) range in steps of `stride`, emitting tile
/// origins so that:
///   * the first tile starts at 0,
///   * every covered pixel sits inside at least one tile,
///   * the last tile is aligned flush to the right edge (so we never
///     "overshoot" the canvas; the final stride may be shorter than
///     the configured one).
fn tile_positions_1d(canvas_size: usize, tile_size: usize, stride: usize) -> Vec<usize> {
    if canvas_size <= tile_size {
        return vec![0];
    }
    let mut positions = Vec::new();
    let mut pos = 0usize;
    loop {
        positions.push(pos);
        if pos + tile_size >= canvas_size {
            // The current tile already covers the right edge.
            break;
        }
        pos += stride;
        if pos + tile_size > canvas_size {
            // The next stride would overshoot — snap the final tile
            // to the right edge so we don't leave a sliver uncovered.
            positions.push(canvas_size - tile_size);
            break;
        }
    }
    positions
}

/// Produce the grid of latent-space tile positions for the full
/// canvas. `latent_h` / `latent_w` are the latent dims; `tile_latent`
/// and `stride_latent` are the user-config values pre-divided by the
/// VAE downsample factor.
pub fn tile_positions(
    latent_h: usize,
    latent_w: usize,
    tile_latent: usize,
    stride_latent: usize,
) -> Vec<TilePos> {
    let ys = tile_positions_1d(latent_h, tile_latent, stride_latent);
    let xs = tile_positions_1d(latent_w, tile_latent, stride_latent);
    let mut out = Vec::with_capacity(ys.len() * xs.len());
    for &y in &ys {
        for &x in &xs {
            out.push(TilePos {
                y,
                x,
                size: tile_latent,
            });
        }
    }
    out
}

/// Build a `(1, 1, n, n)` 2D Hann window (raised cosine), with values
/// in `(0, 1]`. The window is `1.0` at the centre and tapers toward
/// (small positive) values at the edges. A small epsilon keeps the
/// per-pixel weight strictly positive so the divide at the end of
/// each step doesn't NaN even when only one tile covers a pixel.
pub fn hann_window_2d(n: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let mut row: Vec<f32> = Vec::with_capacity(n);
    let pi = std::f32::consts::PI;
    let denom = (n as f32 - 1.0).max(1.0);
    for i in 0..n {
        // Standard Hann: 0.5 * (1 - cos(2π i / (N-1))). Clamped from
        // below at a small epsilon so an overlap-of-1 still divides
        // cleanly.
        let raw = 0.5 * (1.0 - (2.0 * pi * (i as f32) / denom).cos());
        row.push(raw.max(1e-3));
    }
    // Outer product → 2D window.
    let row_t = Tensor::from_vec(row, (n, 1), device)?;
    let col_t = row_t.reshape((1, n))?;
    let win = row_t.broadcast_mul(&col_t)?;
    // Shape it for broadcasting against (B, C, H, W).
    win.reshape((1, 1, n, n))?.to_dtype(dtype).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_positions_cover_full_range() {
        // 128 canvas, 64 tile, 32 stride → positions 0, 32, 64 covers
        // [0, 192) which is more than 128 — last must snap to 64.
        let p = tile_positions_1d(128, 64, 32);
        // Final tile must end exactly at canvas: pos + 64 = 128 → 64.
        assert_eq!(*p.last().unwrap(), 64);
        // First tile starts at 0.
        assert_eq!(p[0], 0);
        // Every pixel is covered by at least one tile.
        for i in 0..128 {
            assert!(
                p.iter().any(|&start| start <= i && i < start + 64),
                "pixel {i} not covered by any tile in {p:?}"
            );
        }
    }

    #[test]
    fn tile_positions_canvas_equals_tile() {
        // Canvas exactly tile-sized → single tile at 0.
        assert_eq!(tile_positions_1d(64, 64, 32), vec![0]);
    }

    #[test]
    fn tile_positions_canvas_smaller_than_tile() {
        // Pathological but well-defined: clamp to single tile at 0,
        // caller has to handle the size mismatch.
        assert_eq!(tile_positions_1d(32, 64, 32), vec![0]);
    }

    #[test]
    fn tile_positions_2d_product() {
        let pos = tile_positions(128, 128, 64, 32);
        // Each dim emits 3 positions (0, 32, 64), grid is 3×3.
        assert_eq!(pos.len(), 9);
        // Bottom-right tile snaps to (64, 64).
        let br = pos.last().unwrap();
        assert_eq!((br.y, br.x), (64, 64));
    }

    #[test]
    fn hann_window_positive_and_centered() {
        let w = hann_window_2d(8, &Device::Cpu, DType::F32).unwrap();
        let v: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v.len(), 64);
        for &x in &v {
            assert!(x > 0.0, "window must be strictly positive (got {x})");
        }
        // Centre is the max.
        let centre = v[3 * 8 + 3];
        for &x in &v {
            assert!(centre >= x - 1e-5);
        }
    }
}
