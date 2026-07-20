//! Binary-EXIF **write-back** (RFC PHOTOS 3.9). The metadata editor ([`super::mod`] `Ctrl-B d`) keeps
//! title / author / copyright / capture-date / geotag in the album record (`album.hjson`),
//! non-destructively. This module writes that same metadata **into the image file's own EXIF**, so it
//! travels with the file to other tools.
//!
//! Like [`super::scrub`] it is hand-rolled and dependency-free: it builds a little-endian TIFF/EXIF
//! block from scratch and splices it into a JPEG (as an `APP1 "Exif\0\0"` segment) or a PNG (as an
//! `eXIf` chunk), replacing any existing EXIF. Only the fields the user set are written; the pixel
//! stream is never touched. Round-trips through the [`super::exif`] reader (`kamadak-exif`).
//!
//! Tag map: title → `ImageDescription` (0x010E), author → `Artist` (0x013B), copyright →
//! `Copyright` (0x8298), date → `DateTime` (0x0132, IFD0) + `DateTimeOriginal` (0x9003, Exif IFD),
//! geotag → a GPS IFD (0x8825) with lat/lon rationals + hemisphere refs.

use std::path::Path;

use anyhow::{Context, Result};

/// The metadata to embed. Every field is optional; `None` means "don't write this tag".
#[derive(Default, Clone)]
pub struct MetaFields {
    pub title: Option<String>,
    pub author: Option<String>,
    pub copyright: Option<String>,
    /// Capture date, any of ISO-8601 (`2024-07-14T12:00:00`) or EXIF (`2024:07:14 12:00:00`) form.
    pub date: Option<String>,
    /// `(lat, lon)` in signed decimal degrees.
    pub gps: Option<(f64, f64)>,
    /// Keyword tags → EXIF `XPKeywords` (0x9C9E, UTF-16LE, `;`-separated) so other tools see them.
    pub keywords: Vec<String>,
}

impl MetaFields {
    /// True when there is at least one field to write.
    pub fn is_empty(&self) -> bool {
        self.title.is_none()
            && self.author.is_none()
            && self.copyright.is_none()
            && self.date.is_none()
            && self.gps.is_none()
            && self.keywords.is_empty()
    }
}

fn ext_lower(path: &Path) -> String {
    path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase()
}

/// Normalize a date string to the EXIF `YYYY:MM:DD HH:MM:SS` form (19 chars). A date without a time
/// gets `00:00:00`. Returns `None` if it can't find a year/month/day.
fn to_exif_datetime(s: &str) -> Option<String> {
    let s = s.trim().replace('T', " ");
    let (date, time) = s.split_once(' ').unwrap_or((s.as_str(), "00:00:00"));
    let dp: Vec<&str> = date.split(['-', ':', '/']).filter(|p| !p.is_empty()).collect();
    if dp.len() < 3 {
        return None;
    }
    let tp: Vec<&str> = time.split(':').collect();
    let hh = tp.first().copied().unwrap_or("0");
    let mm = tp.get(1).copied().unwrap_or("0");
    let ss = tp.get(2).copied().unwrap_or("0");
    // Reject non-numeric components rather than emitting a malformed timestamp.
    if [dp[0], dp[1], dp[2], hh, mm, ss].iter().any(|p| p.parse::<u32>().is_err()) {
        return None;
    }
    Some(format!(
        "{:0>4}:{:0>2}:{:0>2} {:0>2}:{:0>2}:{:0>2}",
        dp[0], dp[1], dp[2], hh, mm, ss
    ))
}

/// A decimal degree → three unsigned rationals (deg/1, min/1, sec×10000/10000).
fn dms_rationals(deg: f64) -> [(u32, u32); 3] {
    let a = deg.abs();
    let d = a.floor();
    let m = ((a - d) * 60.0).floor();
    let s = (((a - d) * 60.0) - m) * 60.0;
    [(d as u32, 1), (m as u32, 1), ((s * 10_000.0).round() as u32, 10_000)]
}

// ---- TIFF/EXIF builder --------------------------------------------------------------------------

/// One IFD entry. `Ext` bytes are placed out-of-line and the entry stores their offset; `Inline`
/// already holds the 4-byte value field.
enum Val {
    Inline([u8; 4]),
    Ext(Vec<u8>),
}

struct Field {
    tag: u16,
    typ: u16,
    count: u32,
    val: Val,
}

fn ascii_field(tag: u16, s: &str) -> Field {
    // TIFF ASCII count includes the trailing NUL.
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    let count = bytes.len() as u32;
    let val = if bytes.len() <= 4 {
        let mut b = [0u8; 4];
        b[..bytes.len()].copy_from_slice(&bytes);
        Val::Inline(b)
    } else {
        Val::Ext(bytes)
    };
    Field { tag, typ: 2, count, val }
}

fn rational3_field(tag: u16, rs: [(u32, u32); 3]) -> Field {
    let mut bytes = Vec::with_capacity(24);
    for (n, d) in rs {
        bytes.extend_from_slice(&n.to_le_bytes());
        bytes.extend_from_slice(&d.to_le_bytes());
    }
    Field { tag, typ: 5, count: 3, val: Val::Ext(bytes) } // 24 bytes → always out-of-line
}

fn long_field(tag: u16, v: u32) -> Field {
    Field { tag, typ: 4, count: 1, val: Val::Inline(v.to_le_bytes()) }
}

/// `XPKeywords`-style BYTE field: the string as UTF-16LE bytes + a UTF-16 NUL terminator.
fn xp_field(tag: u16, s: &str) -> Field {
    let mut bytes = Vec::with_capacity(s.len() * 2 + 2);
    for u in s.encode_utf16() {
        bytes.extend_from_slice(&u.to_le_bytes());
    }
    bytes.extend_from_slice(&[0, 0]); // UTF-16 terminator
    let count = bytes.len() as u32;
    let val = if bytes.len() <= 4 {
        let mut b = [0u8; 4];
        b[..bytes.len()].copy_from_slice(&bytes);
        Val::Inline(b)
    } else {
        Val::Ext(bytes)
    };
    Field { tag, typ: 1, count, val } // BYTE
}

/// Serialize one little-endian IFD (its 2+n·12+4 structure followed by its out-of-line data) sitting
/// at absolute offset `ifd_off`. Ext values get offsets relative to `ifd_off`.
fn serialize_ifd(fields: &[Field], ifd_off: u32) -> Vec<u8> {
    let n = fields.len() as u32;
    let struct_len = 2 + n * 12 + 4;
    let mut ifd = Vec::new();
    let mut ext = Vec::new();
    ifd.extend_from_slice(&(n as u16).to_le_bytes());
    for f in fields {
        ifd.extend_from_slice(&f.tag.to_le_bytes());
        ifd.extend_from_slice(&f.typ.to_le_bytes());
        ifd.extend_from_slice(&f.count.to_le_bytes());
        match &f.val {
            Val::Inline(b) => ifd.extend_from_slice(b),
            Val::Ext(bytes) => {
                let at = ifd_off + struct_len + ext.len() as u32;
                ifd.extend_from_slice(&at.to_le_bytes());
                ext.extend_from_slice(bytes);
                if ext.len() % 2 == 1 {
                    ext.push(0); // keep the next value word-aligned
                }
            }
        }
    }
    ifd.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
    ifd.extend_from_slice(&ext);
    ifd
}

/// Build a complete little-endian TIFF/EXIF block for `f`. Returns `None` when nothing to write.
fn build_tiff(f: &MetaFields) -> Option<Vec<u8>> {
    let exif_datetime = f.date.as_deref().and_then(to_exif_datetime);

    // IFD0 fields, in ascending tag order (a TIFF requirement).
    let mut ifd0: Vec<Field> = Vec::new();
    if let Some(t) = &f.title {
        if !t.is_empty() {
            ifd0.push(ascii_field(0x010E, t)); // ImageDescription
        }
    }
    if let Some(dt) = &exif_datetime {
        ifd0.push(ascii_field(0x0132, dt)); // DateTime (IFD0)
    }
    if let Some(a) = &f.author {
        if !a.is_empty() {
            ifd0.push(ascii_field(0x013B, a)); // Artist
        }
    }
    if let Some(c) = &f.copyright {
        if !c.is_empty() {
            ifd0.push(ascii_field(0x8298, c)); // Copyright
        }
    }
    let has_exif = exif_datetime.is_some();
    let has_gps = f.gps.is_some();
    // Pointer placeholders (patched with real offsets below); keep them last (highest tags).
    if has_exif {
        ifd0.push(long_field(0x8769, 0)); // ExifIFDPointer
    }
    if has_gps {
        ifd0.push(long_field(0x8825, 0)); // GPSInfoIFDPointer
    }
    // XPKeywords (0x9C9E) is the highest tag → must come after the pointers to keep IFD0 ascending.
    let kw: Vec<&str> = f.keywords.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if !kw.is_empty() {
        ifd0.push(xp_field(0x9C9E, &kw.join(";")));
    }
    if ifd0.is_empty() {
        return None;
    }

    // Lay out: header(8) | IFD0 | Exif IFD | GPS IFD. IFD0 length is independent of the (inline)
    // pointer values, so serialize once to measure, place the sub-IFDs, then patch the pointers.
    let ifd0_len = serialize_ifd(&ifd0, 8).len() as u32;
    let exif_off = 8 + ifd0_len;

    let mut exif_bytes = Vec::new();
    if let Some(dt) = &exif_datetime {
        exif_bytes = serialize_ifd(&[ascii_field(0x9003, dt)], exif_off); // DateTimeOriginal
    }
    let gps_off = exif_off + exif_bytes.len() as u32;

    let mut gps_bytes = Vec::new();
    if let Some((lat, lon)) = f.gps {
        let lat_ref = if lat >= 0.0 { "N" } else { "S" };
        let lon_ref = if lon >= 0.0 { "E" } else { "W" };
        let gps = vec![
            Field { tag: 0x0000, typ: 1, count: 4, val: Val::Inline([2, 3, 0, 0]) }, // GPSVersionID
            ascii_field(0x0001, lat_ref),
            rational3_field(0x0002, dms_rationals(lat)),
            ascii_field(0x0003, lon_ref),
            rational3_field(0x0004, dms_rationals(lon)),
        ];
        gps_bytes = serialize_ifd(&gps, gps_off);
    }

    // Patch the pointer values now that offsets are known.
    for fld in ifd0.iter_mut() {
        match fld.tag {
            0x8769 => fld.val = Val::Inline(exif_off.to_le_bytes()),
            0x8825 => fld.val = Val::Inline(gps_off.to_le_bytes()),
            _ => {}
        }
    }
    let ifd0_bytes = serialize_ifd(&ifd0, 8);
    debug_assert_eq!(ifd0_bytes.len() as u32, ifd0_len);

    let mut tiff = Vec::with_capacity(8 + ifd0_bytes.len() + exif_bytes.len() + gps_bytes.len());
    tiff.extend_from_slice(b"II"); // little-endian
    tiff.extend_from_slice(&42u16.to_le_bytes());
    tiff.extend_from_slice(&8u32.to_le_bytes()); // IFD0 at offset 8
    tiff.extend_from_slice(&ifd0_bytes);
    tiff.extend_from_slice(&exif_bytes);
    tiff.extend_from_slice(&gps_bytes);
    Some(tiff)
}

// ---- Container splicing -------------------------------------------------------------------------

/// Replace/insert the EXIF `APP1` in a JPEG with one carrying `tiff`. Drops any existing
/// `"Exif\0\0"` APP1 (keeps XMP/ICC and everything structural). Returns `None` if not a walkable JPEG
/// or the block would overflow a JPEG segment (64 KB).
fn embed_jpeg(data: &[u8], tiff: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 2 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }
    let payload_len = 6 + tiff.len(); // "Exif\0\0" + TIFF
    if payload_len + 2 > 0xFFFF {
        return None; // too big for a single APP1 segment
    }
    let mut app1 = vec![0xFF, 0xE1];
    app1.extend_from_slice(&((payload_len + 2) as u16).to_be_bytes()); // length includes its 2 bytes
    app1.extend_from_slice(b"Exif\0\0");
    app1.extend_from_slice(tiff);

    let mut out = vec![0xFF, 0xD8];
    let mut i = 2;
    let mut inserted = false;
    // Insert our APP1 right after a leading APP0 (JFIF) if present, else immediately after SOI.
    while i + 1 < data.len() {
        if data[i] != 0xFF {
            return None;
        }
        let marker = data[i + 1];
        if marker == 0xFF {
            i += 1;
            continue;
        }
        if marker == 0xD9 {
            // EOI reached before a scan (odd, but copy it and stop).
            if !inserted {
                out.extend_from_slice(&app1);
            }
            out.extend_from_slice(&data[i..]);
            return Some(out);
        }
        if marker == 0xDA {
            // Start of scan: insert (if not already) before the entropy stream, then copy to the end.
            if !inserted {
                out.extend_from_slice(&app1);
            }
            out.extend_from_slice(&data[i..]);
            return Some(out);
        }
        if (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
            out.extend_from_slice(&data[i..i + 2]);
            i += 2;
            continue;
        }
        if i + 4 > data.len() {
            return None;
        }
        let len = ((data[i + 2] as usize) << 8) | data[i + 3] as usize;
        let seg_end = i + 2 + len;
        if len < 2 || seg_end > data.len() {
            return None;
        }
        let is_exif_app1 = marker == 0xE1
            && seg_end >= i + 4 + 6
            && &data[i + 4..i + 4 + 6] == b"Exif\0\0";
        if marker == 0xE0 {
            // Keep the APP0/JFIF, then drop our new APP1 in right after it.
            out.extend_from_slice(&data[i..seg_end]);
            out.extend_from_slice(&app1);
            inserted = true;
        } else if is_exif_app1 {
            // Drop the stale EXIF; if we hadn't inserted yet (no APP0), insert here.
            if !inserted {
                out.extend_from_slice(&app1);
                inserted = true;
            }
        } else {
            if !inserted {
                out.extend_from_slice(&app1);
                inserted = true;
            }
            out.extend_from_slice(&data[i..seg_end]);
        }
        i = seg_end;
    }
    if !inserted {
        out.extend_from_slice(&app1);
    }
    Some(out)
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

/// Replace/insert a PNG `eXIf` chunk carrying `tiff` (raw TIFF, no `Exif\0\0` prefix per the PNG
/// spec). Drops any existing `eXIf`; inserts right after `IHDR`. Returns `None` if not a PNG.
fn embed_png(data: &[u8], tiff: &[u8]) -> Option<Vec<u8>> {
    const SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
    if data.len() < 8 || data[..8] != SIG {
        return None;
    }
    let mut chunk = Vec::with_capacity(12 + tiff.len());
    chunk.extend_from_slice(&(tiff.len() as u32).to_be_bytes());
    chunk.extend_from_slice(b"eXIf");
    chunk.extend_from_slice(tiff);
    let crc = crc32(&chunk[4..]); // over type + data
    chunk.extend_from_slice(&crc.to_be_bytes());

    let mut out = Vec::with_capacity(data.len() + chunk.len());
    out.extend_from_slice(&data[..8]);
    let mut i = 8;
    while i + 12 <= data.len() {
        let len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        let typ = &data[i + 4..i + 8];
        let chunk_end = i + 12 + len;
        if chunk_end > data.len() {
            return None;
        }
        if typ == b"eXIf" {
            i = chunk_end; // drop the stale one
            continue;
        }
        out.extend_from_slice(&data[i..chunk_end]);
        if typ == b"IHDR" {
            out.extend_from_slice(&chunk); // eXIf must precede IDAT; after IHDR is always valid
        }
        i = chunk_end;
        if typ == b"IEND" {
            break;
        }
    }
    Some(out)
}

/// Write `fields` into `path`'s binary EXIF, in place (atomic temp → rename). JPEG and PNG only.
/// Returns `Ok(false)` when there was nothing to write; errors on an unsupported format or I/O
/// failure. Never touches the pixel stream.
pub fn write_metadata(path: &Path, fields: &MetaFields) -> Result<bool> {
    if fields.is_empty() {
        return Ok(false);
    }
    let Some(tiff) = build_tiff(fields) else { return Ok(false) };
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let out = match ext_lower(path).as_str() {
        "jpg" | "jpeg" => embed_jpeg(&data, &tiff),
        "png" => embed_png(&data, &tiff),
        other => anyhow::bail!("EXIF write-back supports JPEG/PNG (got {other})"),
    };
    let Some(bytes) = out else {
        anyhow::bail!("{} is not a well-formed JPEG/PNG", path.display());
    };
    let tmp = path.with_extension(format!("{}.plakat_tmp", ext_lower(path)));
    std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Rgb};

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("plakat-exifw-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn datetime_normalizes() {
        assert_eq!(to_exif_datetime("2024-07-14T12:34:56").unwrap(), "2024:07:14 12:34:56");
        assert_eq!(to_exif_datetime("2024:07:14 12:34:56").unwrap(), "2024:07:14 12:34:56");
        assert_eq!(to_exif_datetime("2024-7-4").unwrap(), "2024:07:04 00:00:00");
        assert!(to_exif_datetime("not a date").is_none());
    }

    fn fields() -> MetaFields {
        MetaFields {
            title: Some("Sunset over the bay".into()),
            author: Some("Jane Roe".into()),
            copyright: Some("© 2024 Jane Roe".into()),
            date: Some("2024-07-14T12:34:56".into()),
            gps: Some((37.7749, -122.4194)),
            keywords: vec!["sunset".into(), "ocean".into()],
        }
    }

    /// Full round-trip through the project's own reader + a direct kamadak read of the string tags.
    fn assert_roundtrips(path: &std::path::Path) {
        // The project reader recovers date + GPS.
        let rec = super::super::exif::read_exif(path).unwrap();
        assert_eq!(rec.date_taken.as_deref(), Some("2024-07-14T12:34:56"));
        let lat = rec.gps_lat.expect("lat");
        let lon = rec.gps_lon.expect("lon");
        assert!((lat - 37.7749).abs() < 1e-3, "lat {lat}");
        assert!((lon - (-122.4194)).abs() < 1e-3, "lon {lon}");

        // kamadak directly for the string tags.
        let file = std::fs::File::open(path).unwrap();
        let mut buf = std::io::BufReader::new(file);
        let exif = exif::Reader::new().read_from_container(&mut buf).unwrap();
        let get = |tag| {
            exif.get_field(tag, exif::In::PRIMARY).map(|f| f.display_value().to_string())
        };
        assert!(get(exif::Tag::ImageDescription).unwrap().contains("Sunset over the bay"));
        assert!(get(exif::Tag::Artist).unwrap().contains("Jane Roe"));
        assert!(get(exif::Tag::Copyright).unwrap().contains("Jane Roe"));

        // XPKeywords is a UTF-16LE BYTE array; verify our ";"-joined keywords are embedded verbatim.
        let raw = std::fs::read(path).unwrap();
        let utf16: Vec<u8> = "sunset;ocean".encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
        assert!(raw.windows(utf16.len()).any(|w| w == utf16), "XPKeywords bytes present");
    }

    #[test]
    fn writes_and_reads_back_jpeg() {
        let dir = tmpdir("jpg");
        let p = dir.join("shot.jpg");
        DynamicImage::ImageRgb8(ImageBuffer::from_pixel(32, 24, Rgb([180u8, 90, 40])))
            .save(&p)
            .unwrap();
        assert!(write_metadata(&p, &fields()).unwrap());
        // Still a valid, same-size JPEG.
        let img = image::open(&p).unwrap();
        assert_eq!((img.width(), img.height()), (32, 24));
        assert_roundtrips(&p);

        // Re-writing (drops the stale EXIF and inserts fresh) still round-trips.
        assert!(write_metadata(&p, &fields()).unwrap());
        assert_roundtrips(&p);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn writes_and_reads_back_png() {
        let dir = tmpdir("png");
        let p = dir.join("shot.png");
        DynamicImage::ImageRgb8(ImageBuffer::from_pixel(20, 20, Rgb([10u8, 200, 120])))
            .save(&p)
            .unwrap();
        assert!(write_metadata(&p, &fields()).unwrap());
        let img = image::open(&p).unwrap();
        assert_eq!((img.width(), img.height()), (20, 20));
        assert_roundtrips(&p);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn nothing_to_write_is_a_noop() {
        let dir = tmpdir("noop");
        let p = dir.join("x.png");
        DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4, 4, Rgb([0u8, 0, 0]))).save(&p).unwrap();
        let before = std::fs::read(&p).unwrap();
        assert!(!write_metadata(&p, &MetaFields::default()).unwrap());
        assert_eq!(std::fs::read(&p).unwrap(), before, "file untouched");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn partial_fields_only_writes_those() {
        let dir = tmpdir("partial");
        let p = dir.join("t.jpg");
        DynamicImage::ImageRgb8(ImageBuffer::from_pixel(8, 8, Rgb([50u8, 50, 50]))).save(&p).unwrap();
        let f = MetaFields { author: Some("Solo".into()), ..Default::default() };
        assert!(write_metadata(&p, &f).unwrap());
        let rec = super::super::exif::read_exif(&p).unwrap();
        assert!(rec.gps_lat.is_none() && rec.date_taken.is_none());
        let img = image::open(&p).unwrap();
        assert_eq!((img.width(), img.height()), (8, 8));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
