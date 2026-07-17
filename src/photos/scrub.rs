//! File-level management ops (RFC PHOTOS-1 Phase 3, management extension): **strip metadata** and
//! **convert / resize**. Unlike the replayable pixel edits ([`super::edit`]) these act on the file
//! itself, so they are not part of the undo/redo edit log:
//! - `strip_metadata` rewrites the file in place, removing EXIF/XMP/IPTC/GPS. For JPEG and PNG this
//!   is **lossless** (metadata segments/chunks are spliced out; the pixel stream is untouched);
//!   other formats fall back to a decode + re-encode.
//! - `convert` writes a **new** file (the source is never modified) in a chosen format, optionally
//!   capping the longest side or targeting a JPEG file size.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

fn ext_lower(path: &Path) -> String {
    path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase()
}

/// Write `bytes` to `path` atomically (temp file in the same dir → rename).
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension(format!("{}.plakat_tmp", ext_lower(path)));
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

/// Remove APP1..APP15 (EXIF/XMP/IPTC…) and COM segments from a JPEG, keeping pixels + JFIF (APP0).
/// Returns `None` if the bytes aren't a JPEG we can walk.
fn strip_jpeg(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 2 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }
    let mut out = vec![0xFF, 0xD8];
    let mut i = 2;
    while i + 1 < data.len() {
        if data[i] != 0xFF {
            return None; // not aligned on a marker → bail rather than corrupt
        }
        let marker = data[i + 1];
        if marker == 0xFF {
            i += 1; // fill byte
            continue;
        }
        if marker == 0xDA {
            // Start of scan: the entropy-coded stream runs to the end; copy verbatim.
            out.extend_from_slice(&data[i..]);
            return Some(out);
        }
        if (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
            out.extend_from_slice(&data[i..i + 2]); // standalone marker, no length
            i += 2;
            continue;
        }
        if i + 4 > data.len() {
            return None;
        }
        let len = ((data[i + 2] as usize) << 8) | data[i + 3] as usize; // includes the 2 length bytes
        let seg_end = i + 2 + len;
        if len < 2 || seg_end > data.len() {
            return None;
        }
        // Drop APP1..APP15 (EXIF/XMP/IPTC) and COM; keep everything structural (APP0/JFIF, DQT, …).
        if !matches!(marker, 0xE1..=0xEF | 0xFE) {
            out.extend_from_slice(&data[i..seg_end]);
        }
        i = seg_end;
    }
    Some(out)
}

/// Remove metadata chunks (eXIf/tEXt/iTXt/zTXt/tIME) from a PNG, keeping pixels + colour chunks.
fn strip_png(data: &[u8]) -> Option<Vec<u8>> {
    const SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
    if data.len() < 8 || data[..8] != SIG {
        return None;
    }
    let mut out = Vec::with_capacity(data.len());
    out.extend_from_slice(&data[..8]);
    let mut i = 8;
    while i + 12 <= data.len() {
        let len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        let typ = &data[i + 4..i + 8];
        let chunk_end = i + 12 + len; // len(4) + type(4) + data + crc(4)
        if chunk_end > data.len() {
            return None;
        }
        let is_iend = typ == b"IEND";
        if !matches!(typ, b"eXIf" | b"tEXt" | b"iTXt" | b"zTXt" | b"tIME") {
            out.extend_from_slice(&data[i..chunk_end]);
        }
        i = chunk_end;
        if is_iend {
            break;
        }
    }
    Some(out)
}

/// Strip EXIF/XMP/IPTC/GPS metadata from `path`, in place. JPEG/PNG are rewritten losslessly; other
/// formats fall back to a decode + re-encode (pixels re-compressed). Returns `true` if lossless.
pub fn strip_metadata(path: &Path) -> Result<bool> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let stripped = match ext_lower(path).as_str() {
        "jpg" | "jpeg" => strip_jpeg(&data),
        "png" => strip_png(&data),
        _ => None,
    };
    if let Some(bytes) = stripped {
        atomic_write(path, &bytes)?;
        Ok(true)
    } else {
        let img = super::loader::load(path)?;
        let tmp = path.with_extension(format!("{}.plakat_tmp", ext_lower(path)));
        img.save(&tmp).with_context(|| format!("re-encoding {}", path.display()))?;
        std::fs::rename(&tmp, path)?;
        Ok(false)
    }
}

// ---- GPS-only redaction (keep the rest of the EXIF) ---------------------------------------------

fn rd_u16(b: &[u8], le: bool, o: usize) -> u16 {
    if le { u16::from_le_bytes([b[o], b[o + 1]]) } else { u16::from_be_bytes([b[o], b[o + 1]]) }
}
fn rd_u32(b: &[u8], le: bool, o: usize) -> u32 {
    let v = [b[o], b[o + 1], b[o + 2], b[o + 3]];
    if le { u32::from_le_bytes(v) } else { u32::from_be_bytes(v) }
}

/// Bytes per TIFF field type (0 for types we don't size).
fn type_size(t: u16) -> usize {
    match t {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 => 4,
        5 | 10 | 12 => 8,
        _ => 0,
    }
}

/// Given an EXIF TIFF block, find the GPS IFD (via IFD0 tag 0x8825), zero every GPS value (inline or
/// at its data offset), and set the GPS IFD entry count to 0. Returns whether a GPS IFD was found.
/// Bounds-checked throughout: any malformation returns `false` (nothing written) rather than corrupt.
fn zero_gps_in_tiff(tiff: &mut [u8]) -> bool {
    if tiff.len() < 8 {
        return false;
    }
    let le = match &tiff[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return false,
    };
    let ifd0 = rd_u32(tiff, le, 4) as usize;
    if ifd0 + 2 > tiff.len() {
        return false;
    }
    let n0 = rd_u16(tiff, le, ifd0) as usize;
    let mut gps_ifd = None;
    for i in 0..n0 {
        let e = ifd0 + 2 + i * 12;
        if e + 12 > tiff.len() {
            break;
        }
        if rd_u16(tiff, le, e) == 0x8825 {
            gps_ifd = Some(rd_u32(tiff, le, e + 8) as usize);
        }
    }
    let Some(g) = gps_ifd else { return false };
    if g + 2 > tiff.len() {
        return false;
    }
    let ng = rd_u16(tiff, le, g) as usize;
    // Collect the byte ranges holding GPS values (immutable reads first), then zero them.
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for i in 0..ng {
        let e = g + 2 + i * 12;
        if e + 12 > tiff.len() {
            break;
        }
        let sz = type_size(rd_u16(tiff, le, e + 2)) * rd_u32(tiff, le, e + 4) as usize;
        if sz == 0 {
            continue;
        }
        if sz <= 4 {
            ranges.push((e + 8, sz));
        } else {
            let off = rd_u32(tiff, le, e + 8) as usize;
            if off + sz <= tiff.len() {
                ranges.push((off, sz));
            }
        }
    }
    for (s, l) in ranges {
        for b in &mut tiff[s..s + l] {
            *b = 0;
        }
    }
    // Empty the GPS IFD (count = 0) so readers see no GPS tags.
    tiff[g] = 0;
    tiff[g + 1] = 0;
    true
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

/// Redact GPS from a JPEG's APP1 EXIF in place. Returns whether GPS was found.
fn redact_gps_jpeg(data: &mut [u8]) -> bool {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return false;
    }
    let mut i = 2;
    while i + 4 <= data.len() {
        if data[i] != 0xFF {
            return false;
        }
        let marker = data[i + 1];
        if marker == 0xDA || marker == 0xD9 {
            break; // scan / end
        }
        if (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
            i += 2;
            continue;
        }
        let len = ((data[i + 2] as usize) << 8) | data[i + 3] as usize;
        let seg_end = i + 2 + len;
        if len < 2 || seg_end > data.len() {
            return false;
        }
        // APP1 whose payload starts with "Exif\0\0" → the TIFF block follows.
        if marker == 0xE1 && seg_end >= i + 4 + 6 && &data[i + 4..i + 4 + 6] == b"Exif\0\0" {
            return zero_gps_in_tiff(&mut data[i + 10..seg_end]);
        }
        i = seg_end;
    }
    false
}

/// Redact GPS from a PNG's eXIf chunk in place (recomputing its CRC). Returns whether GPS was found.
fn redact_gps_png(data: &mut [u8]) -> bool {
    const SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
    if data.len() < 8 || data[..8] != SIG {
        return false;
    }
    let mut i = 8;
    while i + 12 <= data.len() {
        let len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        let chunk_end = i + 12 + len;
        if chunk_end > data.len() {
            return false;
        }
        if &data[i + 4..i + 8] == b"eXIf" {
            if zero_gps_in_tiff(&mut data[i + 8..i + 8 + len]) {
                let crc = crc32(&data[i + 4..i + 8 + len]); // over type + data
                data[i + 8 + len..chunk_end].copy_from_slice(&crc.to_be_bytes());
                return true;
            }
            return false;
        }
        if &data[i + 4..i + 8] == b"IEND" {
            break;
        }
        i = chunk_end;
    }
    false
}

/// Remove only the GPS location from `path`'s EXIF, keeping the rest (camera, date, …), in place.
/// JPEG (APP1) and PNG (eXIf) only. Returns whether GPS was found and zeroed.
pub fn redact_gps(path: &Path) -> Result<bool> {
    let mut data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let redacted = match ext_lower(path).as_str() {
        "jpg" | "jpeg" => redact_gps_jpeg(&mut data),
        "png" => redact_gps_png(&mut data),
        other => anyhow::bail!("GPS-only redact supports JPEG/PNG (got {other}); use strip metadata"),
    };
    if redacted {
        atomic_write(path, &data)?;
    }
    Ok(redacted)
}

/// A size target for [`convert`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvertSize {
    /// Keep the original dimensions.
    Keep,
    /// Cap the longest side to this many pixels (downscale only).
    MaxPx(u32),
    /// Target a JPEG file size in kilobytes (quality search; JPEG output only).
    MaxKb(u32),
}

/// Canonicalise a requested format to a supported extension.
fn normalize_fmt(fmt: &str) -> Result<&'static str> {
    match fmt.trim().to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => Ok("jpg"),
        "png" => Ok("png"),
        "webp" => Ok("webp"),
        other => anyhow::bail!("unsupported format '{other}' (use jpg, png, or webp)"),
    }
}

/// `<album>/<stem>.<ext>`, suffixing `-2`, `-3`, … so an existing file (incl. the source) is never
/// overwritten.
fn dest_path(album: &Path, src: &Path, ext: &str) -> PathBuf {
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let cand = album.join(format!("{stem}.{ext}"));
    if !cand.exists() {
        return cand;
    }
    for i in 2..10_000 {
        let c = album.join(format!("{stem}-{i}.{ext}"));
        if !c.exists() {
            return c;
        }
    }
    album.join(format!("{stem}-dup.{ext}"))
}

fn encode_jpeg(img: &image::DynamicImage, quality: u8) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality);
    enc.encode_image(&image::DynamicImage::ImageRgb8(img.to_rgb8()))?;
    Ok(buf)
}

/// Encode JPEG at the highest quality whose size is ≤ `target` bytes (binary search on quality);
/// falls back to the lowest quality if even that overshoots.
fn encode_jpeg_target(img: &image::DynamicImage, target: usize) -> Result<Vec<u8>> {
    let mut result = encode_jpeg(img, 20)?; // smallest; best-effort fallback
    let (mut lo, mut hi) = (20u8, 95u8);
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let enc = encode_jpeg(img, mid)?;
        if enc.len() <= target {
            result = enc;
            lo = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            hi = mid - 1;
        }
    }
    Ok(result)
}

/// Convert `src` to `fmt`, optionally resized/size-targeted, writing a **new** file into `album`.
/// Returns the output filename. The source is never modified.
pub fn convert(src: &Path, album: &Path, fmt: &str, size: ConvertSize) -> Result<String> {
    let ext = normalize_fmt(fmt)?;
    let mut img = super::loader::load(src)?;
    if let ConvertSize::MaxPx(px) = size {
        if img.width().max(img.height()) > px {
            img = img.resize(px, px, image::imageops::FilterType::Lanczos3);
        }
    }
    let dest = dest_path(album, src, ext);
    match size {
        ConvertSize::MaxKb(kb) if ext == "jpg" => {
            let bytes = encode_jpeg_target(&img, kb as usize * 1024)?;
            std::fs::write(&dest, bytes).with_context(|| format!("writing {}", dest.display()))?;
        }
        _ => img.save(&dest).with_context(|| format!("writing {}", dest.display()))?,
    }
    Ok(dest.file_name().unwrap_or_default().to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Rgb};

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("plakat-scrub-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn strip_jpeg_removes_app1_keeps_pixels() {
        let dir = tmpdir("jpg");
        let p = dir.join("shot.jpg");
        // Build a JPEG, then splice a fake EXIF APP1 segment right after SOI.
        let mut base = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut base, 90)
            .encode_image(&DynamicImage::ImageRgb8(ImageBuffer::from_pixel(16, 16, Rgb([90u8, 120, 200]))))
            .unwrap();
        let payload = b"Exif\0\0secret!";
        let seg_len = (payload.len() + 2) as u16; // length field includes its own 2 bytes
        let mut app1 = vec![0xFF, 0xE1, (seg_len >> 8) as u8, (seg_len & 0xFF) as u8];
        app1.extend_from_slice(payload);
        let mut withmeta = base[..2].to_vec();
        withmeta.extend_from_slice(&app1);
        withmeta.extend_from_slice(&base[2..]);
        std::fs::write(&p, &withmeta).unwrap();

        assert!(strip_metadata(&p).unwrap(), "jpeg strip is lossless");
        let out = std::fs::read(&p).unwrap();
        assert!(!out.windows(6).any(|w| w == b"Exif\0\0"), "EXIF marker gone");
        // Still a valid, same-size JPEG.
        let img = image::load_from_memory(&out).unwrap();
        assert_eq!((img.width(), img.height()), (16, 16));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn zero_gps_wipes_the_gps_ifd_only() {
        // Hand-built little-endian TIFF: IFD0 has one entry (GPS pointer 0x8825) → a GPS IFD with one
        // RATIONAL×3 entry (GPSLatitude) whose 24 data bytes are nonzero.
        let mut t: Vec<u8> = Vec::new();
        t.extend_from_slice(b"II"); // little-endian
        t.extend_from_slice(&42u16.to_le_bytes());
        t.extend_from_slice(&8u32.to_le_bytes()); // IFD0 at 8
        // IFD0 @8: count 1, entry, next=0  → 18 bytes → ends at 26.
        t.extend_from_slice(&1u16.to_le_bytes());
        t.extend_from_slice(&0x8825u16.to_le_bytes()); // GPS IFD pointer
        t.extend_from_slice(&4u16.to_le_bytes()); // LONG
        t.extend_from_slice(&1u32.to_le_bytes()); // count 1
        t.extend_from_slice(&26u32.to_le_bytes()); // → GPS IFD at 26
        t.extend_from_slice(&0u32.to_le_bytes()); // next IFD
        // GPS IFD @26: count 1, entry (GPSLatitude RATIONAL×3 @44), next=0 → 18 bytes → ends at 44.
        t.extend_from_slice(&1u16.to_le_bytes());
        t.extend_from_slice(&0x0002u16.to_le_bytes()); // GPSLatitude
        t.extend_from_slice(&5u16.to_le_bytes()); // RATIONAL (8 bytes each)
        t.extend_from_slice(&3u32.to_le_bytes()); // ×3 → 24 bytes
        t.extend_from_slice(&44u32.to_le_bytes()); // data @44
        t.extend_from_slice(&0u32.to_le_bytes()); // next IFD
        let data_at = t.len();
        t.extend_from_slice(&[0xAB; 24]); // the GPS coordinate bytes

        assert!(zero_gps_in_tiff(&mut t));
        assert_eq!(&t[data_at..data_at + 24], &[0u8; 24], "GPS coordinate bytes zeroed");
        assert_eq!(rd_u16(&t, true, 26), 0, "GPS IFD emptied (count = 0)");
        // IFD0's own entry count is untouched (the rest of the EXIF survives).
        assert_eq!(rd_u16(&t, true, 8), 1, "IFD0 intact");
    }

    #[test]
    fn convert_png_to_jpg_resized_makes_new_file() {
        let dir = tmpdir("conv");
        let src = dir.join("pic.png");
        DynamicImage::ImageRgb8(ImageBuffer::from_pixel(400, 300, Rgb([10u8, 20, 30]))).save(&src).unwrap();
        let name = convert(&src, &dir, "jpg", ConvertSize::MaxPx(100)).unwrap();
        assert_eq!(name, "pic.jpg");
        let out = image::open(dir.join(&name)).unwrap();
        assert_eq!(out.width().max(out.height()), 100, "longest side capped");
        assert!(src.exists(), "source untouched");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn convert_jpg_target_kb_stays_under_budget() {
        let dir = tmpdir("kb");
        let src = dir.join("big.png");
        // A noisy image so JPEG has something to spend bits on.
        let buf = ImageBuffer::from_fn(256, 256, |x, y| Rgb([(x ^ y) as u8, (x * 3) as u8, (y * 5) as u8]));
        DynamicImage::ImageRgb8(buf).save(&src).unwrap();
        let name = convert(&src, &dir, "jpg", ConvertSize::MaxKb(8)).unwrap();
        let bytes = std::fs::metadata(dir.join(&name)).unwrap().len();
        assert!(bytes <= 8 * 1024, "got {bytes} bytes, target 8 KB");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
