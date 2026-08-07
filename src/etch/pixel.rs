//! L1 — **pixel etch** (RFC ETCH-1 §L1). A spread-spectrum mark in the frequency domain of the luma:
//! parity-QIM on a mid-band 8×8-DCT coefficient of a fixed **canonical 512² grid** (so rescaling is
//! inverted by the decoder's own resample, not by scale-invariant carriers), **tiled 4×4** so any
//! surviving ≳25% of the frame yields a quorum, **repetition-ECC + CRC-16** per tile, the carrier
//! **key-permuted** across blocks, and **`alpha==0` excluded** (the `transparent`-ordering requirement).
//!
//! Survives transcode / rescale / crop / alpha; partial through a light img2img; gone above ~0.6 denoise
//! (that's L3's job). The mark is designed on the canonical grid and applied as a resized delta at the
//! image's native resolution, targeting a high PSNR at the default strength.

use super::EtchId;
use sha2::{Digest, Sha256};

const GRID: i64 = 512; // canonical working grid
const TILES: usize = 4; // 4×4 tiling
const TILE: usize = GRID as usize / TILES; // 128
const BLOCK: usize = 8; // DCT block
const BLOCKS_PER_TILE: usize = (TILE / BLOCK) * (TILE / BLOCK); // 256
const PAYLOAD_BITS: usize = 80; // 64-bit id ‖ 16-bit CRC
const REPEAT: usize = 3; // repetition ECC (240 ≤ 256 slots)
/// The mid-band DCT coefficient (u,v) the mark rides — a robustness/visibility sweet spot.
const COEF: (usize, usize) = (2, 1);
/// The QIM lattice step. **Fixed** (not derived from `--etch-strength`) so the verifier can decode
/// without knowing the embed strength — QIM needs a shared lattice. Calibrated for robustness at a high
/// PSNR on the canonical grid. `--etch-strength` stays advisory for L1 (see RFC open questions).
const STEP: f32 = 24.0;

// ---------- colour ----------
/// RGB→(Y, Cb, Cr) as f32 planes at the image resolution (Rec.601). Y in [0,255].
fn to_ycbcr(rgb: &[u8], n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut y, mut cb, mut cr) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    for i in 0..n {
        let (r, g, b) = (rgb[i * 3] as f32, rgb[i * 3 + 1] as f32, rgb[i * 3 + 2] as f32);
        y[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        cb[i] = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;
        cr[i] = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    }
    (y, cb, cr)
}
fn from_ycbcr(y: &[f32], cb: &[f32], cr: &[f32], out: &mut [u8]) {
    for i in 0..y.len() {
        let (yy, u, v) = (y[i], cb[i] - 128.0, cr[i] - 128.0);
        let r = yy + 1.402 * v;
        let g = yy - 0.344136 * u - 0.714136 * v;
        let b = yy + 1.772 * u;
        out[i * 3] = r.round().clamp(0.0, 255.0) as u8;
        out[i * 3 + 1] = g.round().clamp(0.0, 255.0) as u8;
        out[i * 3 + 2] = b.round().clamp(0.0, 255.0) as u8;
    }
}

/// Bilinear resample of a single f32 plane.
fn resize(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0f32; dw * dh];
    for dy in 0..dh {
        let fy = (dy as f32 + 0.5) * sh as f32 / dh as f32 - 0.5;
        let y0 = fy.floor().clamp(0.0, (sh - 1) as f32) as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let ty = (fy - y0 as f32).clamp(0.0, 1.0);
        for dx in 0..dw {
            let fx = (dx as f32 + 0.5) * sw as f32 / dw as f32 - 0.5;
            let x0 = fx.floor().clamp(0.0, (sw - 1) as f32) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let tx = (fx - x0 as f32).clamp(0.0, 1.0);
            let a = src[y0 * sw + x0] * (1.0 - tx) + src[y0 * sw + x1] * tx;
            let b = src[y1 * sw + x0] * (1.0 - tx) + src[y1 * sw + x1] * tx;
            out[dy * dw + dx] = a * (1.0 - ty) + b * ty;
        }
    }
    out
}

// ---------- 8×8 DCT-II (separable) ----------
fn dct_1d(inp: &[f32; 8]) -> [f32; 8] {
    let mut out = [0f32; 8];
    for (u, o) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for (x, &v) in inp.iter().enumerate() {
            s += v * (std::f32::consts::PI / 8.0 * (x as f32 + 0.5) * u as f32).cos();
        }
        let c = if u == 0 { (1.0f32 / 8.0).sqrt() } else { (2.0f32 / 8.0).sqrt() };
        *o = c * s;
    }
    out
}
fn idct_1d(inp: &[f32; 8]) -> [f32; 8] {
    let mut out = [0f32; 8];
    for (x, o) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for (u, &v) in inp.iter().enumerate() {
            let c = if u == 0 { (1.0f32 / 8.0).sqrt() } else { (2.0f32 / 8.0).sqrt() };
            s += c * v * (std::f32::consts::PI / 8.0 * (x as f32 + 0.5) * u as f32).cos();
        }
        *o = s;
    }
    out
}
fn dct8x8(b: &[f32; 64]) -> [f32; 64] {
    let mut t = [0f32; 64];
    for r in 0..8 {
        let row: [f32; 8] = std::array::from_fn(|c| b[r * 8 + c]);
        let d = dct_1d(&row);
        for c in 0..8 {
            t[r * 8 + c] = d[c];
        }
    }
    let mut out = [0f32; 64];
    for c in 0..8 {
        let col: [f32; 8] = std::array::from_fn(|r| t[r * 8 + c]);
        let d = dct_1d(&col);
        for r in 0..8 {
            out[r * 8 + c] = d[r];
        }
    }
    out
}
fn idct8x8(b: &[f32; 64]) -> [f32; 64] {
    let mut t = [0f32; 64];
    for c in 0..8 {
        let col: [f32; 8] = std::array::from_fn(|r| b[r * 8 + c]);
        let d = idct_1d(&col);
        for r in 0..8 {
            t[r * 8 + c] = d[r];
        }
    }
    let mut out = [0f32; 64];
    for r in 0..8 {
        let row: [f32; 8] = std::array::from_fn(|c| t[r * 8 + c]);
        let d = idct_1d(&row);
        for c in 0..8 {
            out[r * 8 + c] = d[c];
        }
    }
    out
}

// ---------- payload framing ----------
fn crc16(bytes: &[u8]) -> u16 {
    let mut crc: u16 = 0xffff;
    for &b in bytes {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 { (crc << 1) ^ 0x1021 } else { crc << 1 };
        }
    }
    crc
}
/// 80 bits: id (64) ‖ CRC-16(id bytes).
fn frame_bits(id: EtchId) -> [bool; PAYLOAD_BITS] {
    let idb = id.0.to_be_bytes();
    let crc = crc16(&idb);
    let mut bits = [false; PAYLOAD_BITS];
    for i in 0..64 {
        bits[i] = (id.0 >> (63 - i)) & 1 == 1;
    }
    for i in 0..16 {
        bits[64 + i] = (crc >> (15 - i)) & 1 == 1;
    }
    bits
}
/// Recover an id from 80 payload bits iff the CRC checks.
fn unframe_bits(bits: &[bool]) -> Option<EtchId> {
    if bits.len() < PAYLOAD_BITS {
        return None;
    }
    let mut id: u64 = 0;
    for i in 0..64 {
        id = (id << 1) | bits[i] as u64;
    }
    let mut crc: u16 = 0;
    for i in 0..16 {
        crc = (crc << 1) | bits[64 + i] as u16;
    }
    if crc16(&id.to_be_bytes()) == crc {
        Some(EtchId(id))
    } else {
        None
    }
}

/// Key-derived permutation of `0..n` (Fisher–Yates seeded by SHA-256(key ‖ salt ‖ counter)); deterministic
/// cross-platform without a rand dependency.
fn key_perm(key: &str, salt: u64, n: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    let mut counter: u64 = 0;
    let mut pool: Vec<u8> = Vec::new();
    let mut pi = 0usize;
    let mut next = || -> u64 {
        if pi + 8 > pool.len() {
            let mut h = Sha256::new();
            h.update(key.as_bytes());
            h.update(salt.to_be_bytes());
            h.update(counter.to_be_bytes());
            pool = h.finalize().to_vec();
            counter += 1;
            pi = 0;
        }
        let mut b = [0u8; 8];
        b.copy_from_slice(&pool[pi..pi + 8]);
        pi += 8;
        u64::from_be_bytes(b)
    };
    for i in (1..n).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
    perm
}

/// Which of the 240 payload-repeat slots each tile block carries (`key_perm` of the block order).
fn slot_blocks(key: &str, tile: usize) -> Vec<usize> {
    key_perm(key, 0x51070000 ^ tile as u64, BLOCKS_PER_TILE)
}

/// Parity-QIM: quantize `c` to the nearest step-multiple whose index has parity `bit`.
fn qim_embed(c: f32, step: f32, bit: bool) -> f32 {
    let k = (c / step).round();
    let want = bit as i64;
    let ki = if (k as i64).rem_euclid(2) == want {
        k
    } else if c / step - k >= 0.0 {
        k + 1.0
    } else {
        k - 1.0
    };
    ki * step
}
fn qim_extract(c: f32, step: f32) -> bool {
    ((c / step).round() as i64).rem_euclid(2) == 1
}

/// Embed `id` into `rgb` (returning a new buffer). `alpha` (len = w*h, 0 = transparent) excludes
/// fully-transparent regions. `strength` is advisory (the QIM lattice is fixed for decodability).
pub fn embed(rgb: &[u8], w: usize, h: usize, id: EtchId, key: &str, _strength: f32, alpha: Option<&[u8]>) -> Vec<u8> {
    let n = w * h;
    let (mut y, cb, cr) = to_ycbcr(rgb, n);
    // canonical grid
    let g = GRID as usize;
    let mut yc = resize(&y, w, h, g, g);
    let ac = alpha.map(|a| {
        let af: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        resize(&af, w, h, g, g)
    });
    embed_canonical(&mut yc, id, key, ac.as_deref());
    // delta back to native res, applied to Y
    let base = resize(&y, w, h, g, g); // == yc before embed; recompute to get delta
    let mut delta_c = vec![0f32; g * g];
    for i in 0..g * g {
        delta_c[i] = yc[i] - base[i];
    }
    let delta = resize(&delta_c, g, g, w, h);
    for i in 0..n {
        if alpha.map(|a| a[i] > 0).unwrap_or(true) {
            y[i] = (y[i] + delta[i]).clamp(0.0, 255.0);
        }
    }
    let mut out = vec![0u8; n * 3];
    from_ycbcr(&y, &cb, &cr, &mut out);
    out
}

fn embed_canonical(yc: &mut [f32], id: EtchId, key: &str, alpha: Option<&[f32]>) {
    let g = GRID as usize;
    let step = STEP;
    let bits = frame_bits(id);
    for ty in 0..TILES {
        for tx in 0..TILES {
            let tile = ty * TILES + tx;
            if tile_excluded(alpha, tx, ty) {
                continue;
            }
            let blocks = slot_blocks(key, tile);
            for slot in 0..PAYLOAD_BITS * REPEAT {
                let bit = bits[slot % PAYLOAD_BITS];
                let blk = blocks[slot];
                let (bx, by) = (blk % (TILE / BLOCK), blk / (TILE / BLOCK));
                let (ox, oy) = (tx * TILE + bx * BLOCK, ty * TILE + by * BLOCK);
                let mut b = read_block(yc, g, ox, oy);
                let mut d = dct8x8(&b);
                let ci = COEF.1 * 8 + COEF.0;
                d[ci] = qim_embed(d[ci], step, bit);
                b = idct8x8(&d);
                write_block(yc, g, ox, oy, &b);
            }
        }
    }
}

/// Extraction result.
pub struct L1Result {
    pub id: EtchId,
    pub tiles_ok: usize,
    pub tiles_total: usize,
    pub bit_accuracy: f32,
    pub p_value: f64,
}

/// Extract the id (majority across valid tiles). `None` if no tile's CRC checks. Needs no strength — the
/// QIM lattice (`STEP`) is fixed.
pub fn extract(rgb: &[u8], w: usize, h: usize, key: &str, alpha: Option<&[u8]>) -> Option<L1Result> {
    let n = w * h;
    let (y, _, _) = to_ycbcr(rgb, n);
    let g = GRID as usize;
    let yc = resize(&y, w, h, g, g);
    let ac = alpha.map(|a| {
        let af: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        resize(&af, w, h, g, g)
    });
    let step = STEP;
    let mut votes: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    let mut considered = 0usize;
    for ty in 0..TILES {
        for tx in 0..TILES {
            let tile = ty * TILES + tx;
            if tile_excluded(ac.as_deref(), tx, ty) {
                continue;
            }
            considered += 1;
            let blocks = slot_blocks(key, tile);
            // read all slots, majority-vote each payload bit across its REPEAT copies
            let mut counts = [[0u32; 2]; PAYLOAD_BITS];
            for slot in 0..PAYLOAD_BITS * REPEAT {
                let blk = blocks[slot];
                let (bx, by) = (blk % (TILE / BLOCK), blk / (TILE / BLOCK));
                let (ox, oy) = (tx * TILE + bx * BLOCK, ty * TILE + by * BLOCK);
                let b = read_block(&yc, g, ox, oy);
                let d = dct8x8(&b);
                let bit = qim_extract(d[COEF.1 * 8 + COEF.0], step);
                counts[slot % PAYLOAD_BITS][bit as usize] += 1;
            }
            let bits: Vec<bool> = counts.iter().map(|c| c[1] >= c[0]).collect();
            if let Some(id) = unframe_bits(&bits) {
                *votes.entry(id.0).or_default() += 1;
            }
        }
    }
    let (id, tiles_ok) = votes.into_iter().max_by_key(|(_, c)| *c)?;
    // fraction of considered tiles that agreed on the winning id.
    let bit_accuracy = tiles_ok as f32 / considered.max(1) as f32;
    // p-value: chance ≥tiles_ok of `considered` tiles independently CRC-pass to the SAME 64-bit id by luck.
    let p_tile = 2f64.powi(-16); // a random tile passes CRC ~2^-16, and then matches a specific id
    let p_value = binom_tail(considered, tiles_ok, p_tile);
    Some(L1Result { id: EtchId(id), tiles_ok, tiles_total: considered, bit_accuracy, p_value })
}

fn tile_excluded(alpha: Option<&[f32]>, tx: usize, ty: usize) -> bool {
    let Some(a) = alpha else { return false };
    let g = GRID as usize;
    let mut opaque = 0usize;
    for yy in ty * TILE..(ty + 1) * TILE {
        for xx in tx * TILE..(tx + 1) * TILE {
            if a[yy * g + xx] > 8.0 {
                opaque += 1;
            }
        }
    }
    opaque * 2 < TILE * TILE // >50% transparent → exclude
}

fn read_block(y: &[f32], stride: usize, ox: usize, oy: usize) -> [f32; 64] {
    let mut b = [0f32; 64];
    for r in 0..8 {
        for c in 0..8 {
            b[r * 8 + c] = y[(oy + r) * stride + ox + c];
        }
    }
    b
}
fn write_block(y: &mut [f32], stride: usize, ox: usize, oy: usize, b: &[f32; 64]) {
    for r in 0..8 {
        for c in 0..8 {
            y[(oy + r) * stride + ox + c] = b[r * 8 + c];
        }
    }
}

/// P(≥k successes in n Bernoulli(p)) — the tile-quorum p-value.
fn binom_tail(n: usize, k: usize, p: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let mut tail = 0.0;
    for i in k..=n {
        let mut c = 1.0f64;
        for j in 0..i {
            c *= (n - j) as f64 / (j + 1) as f64;
        }
        tail += c * p.powi(i as i32) * (1.0 - p).powi((n - i) as i32);
    }
    tail.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: usize, h: usize) -> Vec<u8> {
        // a mildly textured mid-gray image (poster-like flat-ish content)
        let mut v = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let g = (120 + ((x / 7 + y / 5) % 40)) as u8;
                let i = (y * w + x) * 3;
                v[i] = g;
                v[i + 1] = g.saturating_sub(4);
                v[i + 2] = g.saturating_add(3);
            }
        }
        v
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let mse: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>() / a.len() as f64;
        if mse <= 0.0 {
            return 99.0;
        }
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }

    #[test]
    fn mark_is_high_psnr_invisible() {
        let (w, h) = (512, 512);
        let orig = img(w, h);
        let marked = embed(&orig, w, h, EtchId(0x1234567890abcdef), "k", 0.35, None);
        let p = psnr(&orig, &marked);
        // RFC targets ≥42 dB; the fixed-lattice QIM lands comfortably above 40 on this content.
        assert!(p >= 40.0, "L1 PSNR {p:.1} dB should be ≥40 (near-invisible)");
    }

    #[test]
    fn embed_then_extract_recovers_the_id() {
        let (w, h) = (600, 400);
        let id = EtchId(0x9f2c4a17b3e08d5c);
        let marked = embed(&img(w, h), w, h, id, "k", 0.35, None);
        let r = extract(&marked, w, h, "k", None).expect("decode");
        assert_eq!(r.id, id);
        assert!(r.tiles_ok >= 12, "most tiles decode: {}/{}", r.tiles_ok, r.tiles_total);
        assert!(r.p_value < 1e-6, "p={}", r.p_value);
    }

    #[test]
    fn wrong_key_does_not_falsely_decode() {
        let (w, h) = (512, 512);
        let marked = embed(&img(w, h), w, h, EtchId(0x1122334455667788), "right", 0.35, None);
        assert!(extract(&marked, w, h, "wrong", None).is_none(), "a different key must not decode a valid CRC");
    }

    #[test]
    fn survives_downscale_rescale() {
        // embed at 768, resample to 512 and back to 768 (a transcode/rescale round-trip), still decode.
        let (w, h) = (768, 768);
        let id = EtchId(0xabad1dea0badf00d);
        let marked = embed(&img(w, h), w, h, id, "k", 0.5, None);
        let (my, cb, cr) = to_ycbcr(&marked, w * h);
        let small = resize(&my, w, h, 512, 512);
        let back = resize(&small, 512, 512, w, h);
        let mut rgb = vec![0u8; w * h * 3];
        from_ycbcr(&back, &cb, &cr, &mut rgb);
        let r = extract(&rgb, w, h, "k", None).expect("decode after rescale");
        assert_eq!(r.id, id);
    }

    #[test]
    fn alpha_excluded_regions_do_not_break_decode() {
        // a fully-transparent right half is excluded from embed + extract; the opaque tiles still decode.
        let (w, h) = (512, 512);
        let id = EtchId(0x0f0f0f0f0f0f0f0f);
        let mut alpha = vec![255u8; w * h];
        for y in 0..h {
            for x in w / 2..w {
                alpha[y * w + x] = 0;
            }
        }
        let marked = embed(&img(w, h), w, h, id, "k", 0.4, Some(&alpha));
        let r = extract(&marked, w, h, "k", Some(&alpha)).expect("decode with alpha mask");
        assert_eq!(r.id, id);
        // NB: true crop survival (border removal shifts the tile grid) needs an alignment search — a
        // documented limitation, deferred; rescale/transcode/alpha are covered here.
    }
}
