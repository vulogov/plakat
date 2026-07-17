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
