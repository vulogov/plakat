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

/// Tile a VAE decode pass over an arbitrary-size latent. For each
/// latent tile (`latent_tile_size × latent_tile_size`), `decode_fn`
/// is called to produce a pixel tile of size
/// `(latent_tile_size * scale) × (latent_tile_size * scale)`. The
/// pixel tiles are blended back together with a 2D Hann window —
/// same blending math the MultiDiffusion noise-prediction tiling
/// uses, just applied at pixel resolution.
///
/// Use this when the whole-canvas decode would OOM (typically 4K+
/// outputs on tight GPUs).
///
/// `latent` is shape `(B, C, H, W)`. The returned tensor has the
/// same batch + channel dims; spatial dims are scaled by `scale`
/// (8 for both SDXL VAE and Flux AE).
pub fn tile_decode_2d<F>(
    latent: &Tensor,
    latent_tile_size: usize,
    latent_stride: usize,
    scale: usize,
    mut decode_fn: F,
) -> Result<Tensor>
where
    F: FnMut(&Tensor) -> Result<Tensor>,
{
    let (b, c_latent, latent_h, latent_w) = latent.dims4()?;
    if latent_h <= latent_tile_size && latent_w <= latent_tile_size {
        // Canvas fits in one tile — just decode whole-cloth.
        return decode_fn(latent);
    }
    let positions = tile_positions(latent_h, latent_w, latent_tile_size, latent_stride);
    let _ = c_latent;

    let pixel_tile = latent_tile_size * scale;
    let pixel_h = latent_h * scale;
    let pixel_w = latent_w * scale;

    let device = latent.device();
    let dtype = latent.dtype();
    let pixel_window = hann_window_2d(pixel_tile, device, dtype)?; // (1, 1, t, t)

    // We don't know the output channel count without running one
    // decode. Allocate accumulators after the first tile.
    let mut acc: Option<Tensor> = None;
    let mut weights: Option<Tensor> = None;

    for TilePos { y, x, size } in positions {
        // Narrow latent tile and decode.
        let tile_latent = latent.narrow(2, y, size)?.narrow(3, x, size)?;
        let tile_pixels = decode_fn(&tile_latent)?;
        let (tile_b, tile_c, tile_ph, tile_pw) = tile_pixels.dims4()?;
        if tile_ph != pixel_tile || tile_pw != pixel_tile {
            anyhow::bail!(
                "tile_decode_2d: decode produced {tile_ph}×{tile_pw} pixels, \
                 expected {pixel_tile}×{pixel_tile} (scale={scale})"
            );
        }
        if tile_b != b {
            anyhow::bail!(
                "tile_decode_2d: decode produced batch {tile_b}, expected {b}"
            );
        }

        // Lazy-allocate accumulators on the first tile (now that we
        // know the output channel count).
        let acc_t = match acc.take() {
            Some(t) => t,
            None => Tensor::zeros((b, tile_c, pixel_h, pixel_w), dtype, device)?,
        };
        let weights_t = match weights.take() {
            Some(t) => t,
            None => Tensor::zeros((1, 1, pixel_h, pixel_w), dtype, device)?,
        };

        // Pixel positions for this tile.
        let py = y * scale;
        let px = x * scale;
        let pixel_size = size * scale;

        // Weighted pixel contribution: tile_pixels * window.
        let weighted = tile_pixels.broadcast_mul(&pixel_window)?;
        let acc_region = acc_t.narrow(2, py, pixel_size)?.narrow(3, px, pixel_size)?;
        let acc_updated = (acc_region + &weighted)?;
        let acc_t = acc_t.slice_assign(
            &[0..b, 0..tile_c, py..py + pixel_size, px..px + pixel_size],
            &acc_updated,
        )?;

        let weights_region = weights_t
            .narrow(2, py, pixel_size)?
            .narrow(3, px, pixel_size)?;
        let weights_updated = weights_region.broadcast_add(&pixel_window)?;
        let weights_t = weights_t.slice_assign(
            &[0..1, 0..1, py..py + pixel_size, px..px + pixel_size],
            &weights_updated,
        )?;

        // Finish this tile's GPU work before the next so Metal reclaims the
        // decode's transient buffers (see the denoise-loop note in t2i.rs —
        // unbounded queued buffers OOM Metal). VAE decode at pixel resolution
        // is the memory-tightest op in the tiled pipeline.
        acc_t.device().synchronize()?;
        acc = Some(acc_t);
        weights = Some(weights_t);
    }

    let acc = acc.expect("tile_decode_2d ran at least one tile");
    let weights = weights.expect("tile_decode_2d weights set on first tile");
    Ok(acc.broadcast_div(&weights)?)
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

// ---- Regional prompting (MultiDiffusion with per-region prompts) ----

/// One prompted region: a bbox in `[x0, y0, x1, y1]` canvas fractions (`[0,1]`)
/// and the prompt that applies there. The base `plakat.generate` prompt covers
/// everything else (and provides global coherence).
#[derive(Debug, Clone)]
pub struct RegionSpec {
    pub bbox: [f32; 4],
    pub prompt: String,
    /// 6.25.0 P3 — per-region **strength**: scales this region's contribution in the
    /// per-pixel blend, so a higher-weight region dominates where masks overlap. `1.0` =
    /// neutral (the pre-6.25 behaviour). Parsed from a `w=`/`weight=` modifier.
    pub weight: f32,
    /// 6.25.0 P3 — per-region **feather**: soft-edge ramp width as a canvas fraction. Larger
    /// = softer blend into neighbours. Defaults to [`DEFAULT_REGION_FEATHER`]. `f=`/`feather=`.
    pub feather: f32,
}

/// Default soft-edge ramp for a region mask (canvas fraction) — the pre-6.25 constant.
pub const DEFAULT_REGION_FEATHER: f32 = 0.05;

impl RegionSpec {
    /// Parse `"x0,y0,x1,y1[,w=W][,feather=F]:prompt"` (coords are `[0,1]` canvas fractions).
    /// The section before the first `:` is comma-separated: the first four numbers are the
    /// bbox, any trailing `key=value` tokens are per-region modifiers (`w`/`weight`,
    /// `f`/`feather`). Putting modifiers in the coord section keeps them clear of colons the
    /// prompt itself may contain (e.g. `(word:1.2)` weights).
    pub fn parse(s: &str) -> Result<Self> {
        let (coords, prompt) = s
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("region {s:?}: expected \"x0,y0,x1,y1:prompt\""))?;
        let mut nums: Vec<f32> = Vec::new();
        let mut weight = 1.0f32;
        let mut feather = DEFAULT_REGION_FEATHER;
        for tok in coords.split(',') {
            let tok = tok.trim();
            if tok.is_empty() {
                continue;
            }
            if let Some((k, v)) = tok.split_once('=') {
                let val: f32 = v.trim().parse().map_err(|_| {
                    anyhow::anyhow!("region {s:?}: modifier {tok:?} needs a number")
                })?;
                match k.trim().to_lowercase().as_str() {
                    "w" | "weight" => weight = val.max(0.0),
                    "f" | "feather" => feather = val.clamp(0.0, 0.5),
                    other => anyhow::bail!("region {s:?}: unknown modifier {other:?} (use w= / feather=)"),
                }
            } else {
                nums.push(tok.parse::<f32>().map_err(|_| {
                    anyhow::anyhow!("region {s:?}: coords must be 4 numbers in [0,1]")
                })?);
            }
        }
        if nums.len() != 4 {
            anyhow::bail!("region {s:?}: expected 4 coords x0,y0,x1,y1, got {}", nums.len());
        }
        let (x0, y0, x1, y1) = (
            nums[0].min(nums[2]),
            nums[1].min(nums[3]),
            nums[0].max(nums[2]),
            nums[1].max(nums[3]),
        );
        let prompt = prompt.trim();
        if prompt.is_empty() {
            anyhow::bail!("region {s:?}: empty prompt");
        }
        Ok(Self {
            bbox: [
                x0.clamp(0.0, 1.0),
                y0.clamp(0.0, 1.0),
                x1.clamp(0.0, 1.0),
                y1.clamp(0.0, 1.0),
            ],
            prompt: prompt.to_string(),
            weight,
            feather,
        })
    }
}

/// Build a `(1, 1, lh, lw)` latent-space mask that is `1.0` inside the bbox and
/// `0.0` outside (a latent pixel is "inside" if its center falls in the bbox).
pub fn region_mask(
    bbox: [f32; 4],
    lh: usize,
    lw: usize,
    device: &Device,
    dtype: DType,
    feather: f32,
) -> Result<Tensor> {
    let [x0, y0, x1, y1] = bbox;
    // Soft edge: the mask ramps 0→1 across ~`feather` of the canvas, centered on
    // the bbox boundary (signed distance to the nearest edge, corner-aware), so
    // neighbouring regions blend instead of seaming. Hard masks leave visible
    // joins between regions. 6.25.0 P3 makes `feather` per-region (was a constant).
    let feather = feather.max(1e-3); // avoid divide-by-zero → hard mask limit
    let mut data = vec![0f32; lh * lw];
    for i in 0..lh {
        let cy = (i as f32 + 0.5) / lh as f32;
        let sy = (cy - y0).min(y1 - cy); // +inside, −outside (y)
        for j in 0..lw {
            let cx = (j as f32 + 0.5) / lw as f32;
            let sx = (cx - x0).min(x1 - cx); // +inside, −outside (x)
            let s = sx.min(sy); // signed distance to the nearest box edge
            data[i * lw + j] = (s / feather + 0.5).clamp(0.0, 1.0);
        }
    }
    Ok(Tensor::from_vec(data, (1, 1, lh, lw), device)?.to_dtype(dtype)?)
}

/// Reject regional prompting combined with options it can't compose with —
/// both would otherwise be silently dropped. Called from the CLI and scenario
/// dispatch when `regions` is non-empty.
pub fn check_regional_combo(tiled: bool, control: bool) -> Result<()> {
    if tiled {
        anyhow::bail!(
            "--region (regional prompting) doesn't compose with --tiled — both are \
             MultiDiffusion passes over the same canvas; use one or the other"
        );
    }
    if control {
        anyhow::bail!(
            "--region (regional prompting) doesn't compose with ControlNet (--control*) — \
             the regional path doesn't apply CN residuals; use one or the other"
        );
    }
    Ok(())
}

#[cfg(test)]
mod regional_tests {
    use super::*;

    #[test]
    fn parses_region_spec() {
        let r = RegionSpec::parse("0,0,0.5,1:a dense forest").unwrap();
        assert_eq!(r.bbox, [0.0, 0.0, 0.5, 1.0]);
        assert_eq!(r.prompt, "a dense forest");
        // coords normalize (min/max) + clamp.
        assert_eq!(RegionSpec::parse("0.5,0,1,1:x").unwrap().bbox, [0.5, 0.0, 1.0, 1.0]);
        assert!(RegionSpec::parse("0,0,1:bad").is_err());
        assert!(RegionSpec::parse("no colon").is_err());
        assert!(RegionSpec::parse("0,0,1,1:").is_err());
    }

    #[test]
    fn region_parses_weight_and_feather_modifiers() {
        // 6.25.0 P3: w= / feather= in the coord section; defaults otherwise.
        let plain = RegionSpec::parse("0,0,0.5,1:a wolf").unwrap();
        assert_eq!(plain.weight, 1.0);
        assert_eq!(plain.feather, DEFAULT_REGION_FEATHER);

        let r = RegionSpec::parse("0,0,0.5,1,w=1.3,feather=0.1:a wolf (fur:1.2)").unwrap();
        assert_eq!(r.bbox, [0.0, 0.0, 0.5, 1.0]);
        assert!((r.weight - 1.3).abs() < 1e-6);
        assert!((r.feather - 0.1).abs() < 1e-6);
        // The prompt keeps its own colon-bearing weight syntax intact.
        assert_eq!(r.prompt, "a wolf (fur:1.2)");

        // Short aliases + feather clamp (≤0.5).
        let s = RegionSpec::parse("0,0,1,1,f=0.9:x").unwrap();
        assert_eq!(s.feather, 0.5);
        // Unknown modifier errors.
        assert!(RegionSpec::parse("0,0,1,1,zzz=2:x").is_err());
    }

    #[test]
    fn bigger_feather_widens_the_soft_band() {
        let narrow = region_mask([0.0, 0.0, 0.5, 1.0], 1, 40, &Device::Cpu, DType::F32, 0.05).unwrap();
        let wide = region_mask([0.0, 0.0, 0.5, 1.0], 1, 40, &Device::Cpu, DType::F32, 0.2).unwrap();
        let soft = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().filter(|&&x| x > 0.05 && x < 0.95).count();
        assert!(soft(&wide) > soft(&narrow), "larger feather → more partial-value pixels");
    }

    #[test]
    fn region_mask_left_half() {
        let m = region_mask([0.0, 0.0, 0.5, 1.0], 4, 4, &Device::Cpu, DType::F32, DEFAULT_REGION_FEATHER).unwrap();
        let v: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
        // 4×4: columns 0,1 (centers 0.125,0.375 < 0.5) inside; cols 2,3 outside.
        for row in 0..4 {
            assert_eq!(v[row * 4], 1.0);
            assert_eq!(v[row * 4 + 1], 1.0);
            assert_eq!(v[row * 4 + 2], 0.0);
            assert_eq!(v[row * 4 + 3], 0.0);
        }
    }

    #[test]
    fn region_mask_feathers_at_edge() {
        // 40-wide row, box [0,0,0.5,1]: columns straddling x=0.5 are PARTIAL
        // (soft falloff), not a hard 0/1 cut.
        let m = region_mask([0.0, 0.0, 0.5, 1.0], 1, 40, &Device::Cpu, DType::F32, DEFAULT_REGION_FEATHER).unwrap();
        let v: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
        let soft = v.iter().filter(|&&x| x > 0.05 && x < 0.95).count();
        assert!(soft >= 1, "expected soft (partial) values near the edge, got {soft}");
        // Well inside the box (away from any edge, incl. the canvas edge at x=0) ≈ 1.
        assert!((v[5] - 1.0).abs() < 1e-4, "far inside ≈ 1, got {}", v[5]);
        assert!(v[39] < 1e-4, "far outside ≈ 0");
    }

    #[test]
    fn regional_combo_guard() {
        assert!(check_regional_combo(false, false).is_ok());
        assert!(check_regional_combo(true, false).is_err()); // + tiled
        assert!(check_regional_combo(false, true).is_err()); // + control
    }
}
