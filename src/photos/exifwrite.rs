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

/// The metadata tags this writer sets, as IFD0 fields (ascending tag order, with pointer placeholders
/// for the Exif/GPS sub-IFDs) plus the Exif and GPS sub-IFD field lists. `None` when nothing to write.
fn metadata_ifd(f: &MetaFields) -> Option<(Vec<Field>, Vec<Field>, Vec<Field>)> {
    let exif_datetime = f.date.as_deref().and_then(to_exif_datetime);

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
    if exif_datetime.is_some() {
        ifd0.push(long_field(0x8769, 0)); // ExifIFDPointer (patched at layout)
    }
    if f.gps.is_some() {
        ifd0.push(long_field(0x8825, 0)); // GPSInfoIFDPointer (patched at layout)
    }
    // XPKeywords (0x9C9E) is the highest tag → after the pointers to keep IFD0 ascending.
    let kw: Vec<&str> = f.keywords.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if !kw.is_empty() {
        ifd0.push(xp_field(0x9C9E, &kw.join(";")));
    }
    if ifd0.is_empty() {
        return None;
    }

    let exif = match &exif_datetime {
        Some(dt) => vec![ascii_field(0x9003, dt)], // DateTimeOriginal
        None => Vec::new(),
    };
    let gps = match f.gps {
        Some((lat, lon)) => vec![
            Field { tag: 0x0000, typ: 1, count: 4, val: Val::Inline([2, 3, 0, 0]) }, // GPSVersionID
            ascii_field(0x0001, if lat >= 0.0 { "N" } else { "S" }),
            rational3_field(0x0002, dms_rationals(lat)),
            ascii_field(0x0003, if lon >= 0.0 { "E" } else { "W" }),
            rational3_field(0x0004, dms_rationals(lon)),
        ],
        None => Vec::new(),
    };
    Some((ifd0, exif, gps))
}

/// A merged-IFD entry: an original 12-byte entry copied verbatim, or a new field we compute.
enum Ent {
    Raw([u8; 12]),
    New(Field),
}

fn ent_tag(e: &Ent) -> u16 {
    match e {
        Ent::Raw(b) => u16::from_le_bytes([b[0], b[1]]),
        Ent::New(f) => f.tag,
    }
}

/// Serialize one little-endian IFD (main directory) at absolute offset `base`: emit `ents` (sorted by
/// tag), the `next_ifd` link, this IFD's out-of-line data, then the Exif + GPS sub-IFDs. `Raw` entries
/// keep their bytes (their offsets already point into the untouched file); `New` fields get absolute
/// offsets; the Exif/GPS pointer fields are patched to the sub-IFD offsets. Returns the block bytes.
fn layout_ifd(base: u32, mut ents: Vec<Ent>, exif: Vec<Field>, gps: Vec<Field>, next_ifd: u32) -> Vec<u8> {
    let count = ents.len() as u32;
    let struct_len = 2 + count * 12 + 4;
    let ext_len: u32 = ents
        .iter()
        .map(|e| match e {
            Ent::New(Field { val: Val::Ext(b), .. }) => b.len() as u32 + (b.len() as u32 & 1),
            _ => 0,
        })
        .sum();
    let exif_off = base + struct_len + ext_len;
    let exif_bytes = if exif.is_empty() { Vec::new() } else { serialize_ifd(&exif, exif_off) };
    let gps_off = exif_off + exif_bytes.len() as u32;
    let gps_bytes = if gps.is_empty() { Vec::new() } else { serialize_ifd(&gps, gps_off) };

    for e in ents.iter_mut() {
        if let Ent::New(f) = e {
            match f.tag {
                0x8769 => f.val = Val::Inline(exif_off.to_le_bytes()),
                0x8825 => f.val = Val::Inline(gps_off.to_le_bytes()),
                _ => {}
            }
        }
    }

    let mut ifd = Vec::new();
    let mut ext = Vec::new();
    ifd.extend_from_slice(&(count as u16).to_le_bytes());
    for e in &ents {
        match e {
            Ent::Raw(b) => ifd.extend_from_slice(b),
            Ent::New(f) => {
                ifd.extend_from_slice(&f.tag.to_le_bytes());
                ifd.extend_from_slice(&f.typ.to_le_bytes());
                ifd.extend_from_slice(&f.count.to_le_bytes());
                match &f.val {
                    Val::Inline(b) => ifd.extend_from_slice(b),
                    Val::Ext(bytes) => {
                        let at = base + struct_len + ext.len() as u32;
                        ifd.extend_from_slice(&at.to_le_bytes());
                        ext.extend_from_slice(bytes);
                        if ext.len() % 2 == 1 {
                            ext.push(0);
                        }
                    }
                }
            }
        }
    }
    ifd.extend_from_slice(&next_ifd.to_le_bytes());
    ifd.extend_from_slice(&ext);
    ifd.extend_from_slice(&exif_bytes);
    ifd.extend_from_slice(&gps_bytes);
    ifd
}

/// Build a complete little-endian standalone TIFF/EXIF block for `f` (used as the EXIF payload spliced
/// into JPEG/PNG/WebP). Returns `None` when nothing to write.
fn build_tiff(f: &MetaFields) -> Option<Vec<u8>> {
    let (ifd0, exif, gps) = metadata_ifd(f)?;
    let ents: Vec<Ent> = ifd0.into_iter().map(Ent::New).collect(); // already ascending
    let block = layout_ifd(8, ents, exif, gps, 0);
    let mut tiff = Vec::with_capacity(8 + block.len());
    tiff.extend_from_slice(b"II");
    tiff.extend_from_slice(&42u16.to_le_bytes());
    tiff.extend_from_slice(&8u32.to_le_bytes()); // IFD0 at offset 8
    tiff.extend_from_slice(&block);
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

/// Replace/insert the EXIF in a WebP (RIFF container). A simple `RIFF/WEBP/VP8|VP8L` file is upgraded
/// to the extended `VP8X` form (with the EXIF flag set + the canvas size) and an `EXIF` chunk is
/// appended; an already-extended file has its `VP8X` EXIF flag set and any stale `EXIF` chunk
/// replaced. Returns `None` if not a walkable WebP.
fn embed_webp(data: &[u8], tiff: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return None;
    }
    // Walk the chunks after the 12-byte RIFF/WEBP header.
    let mut chunks: Vec<(&[u8], &[u8])> = Vec::new();
    let mut i = 12;
    while i + 8 <= data.len() {
        let fourcc = &data[i..i + 4];
        let size = u32::from_le_bytes([data[i + 4], data[i + 5], data[i + 6], data[i + 7]]) as usize;
        let start = i + 8;
        let end = start + size;
        if end > data.len() {
            break;
        }
        chunks.push((fourcc, &data[start..end]));
        i = end + (size & 1); // chunks are padded to an even length
    }
    if chunks.is_empty() {
        return None;
    }
    // Canvas dimensions (for a fresh VP8X).
    let img = image::load_from_memory(data).ok()?;
    let (cw, ch) = (img.width(), img.height());
    if cw == 0 || ch == 0 || cw > (1 << 24) || ch > (1 << 24) {
        return None;
    }

    let has_vp8x = chunks.iter().any(|(f, _)| *f == b"VP8X");
    let has_alpha = chunks.iter().any(|(f, _)| *f == b"ALPH");
    let mut out_chunks: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    if has_vp8x {
        for (f, p) in &chunks {
            if *f == b"EXIF" {
                continue; // drop the stale EXIF
            }
            if *f == b"VP8X" {
                let mut vp = p.to_vec();
                if !vp.is_empty() {
                    vp[0] |= 0x08; // set the EXIF flag
                }
                out_chunks.push((f.to_vec(), vp));
            } else {
                out_chunks.push((f.to_vec(), p.to_vec()));
            }
        }
    } else {
        // Upgrade to extended: a VP8X (EXIF flag, optional alpha, canvas size) must come first.
        let flags = 0x08u8 | if has_alpha { 0x10 } else { 0 };
        let mut vp = vec![flags, 0, 0, 0];
        vp.extend_from_slice(&(cw - 1).to_le_bytes()[0..3]);
        vp.extend_from_slice(&(ch - 1).to_le_bytes()[0..3]);
        out_chunks.push((b"VP8X".to_vec(), vp));
        for (f, p) in &chunks {
            out_chunks.push((f.to_vec(), p.to_vec()));
        }
    }
    out_chunks.push((b"EXIF".to_vec(), tiff.to_vec()));

    // Reassemble: RIFF + size + WEBP + padded chunks.
    let mut body = b"WEBP".to_vec();
    for (f, p) in &out_chunks {
        body.extend_from_slice(f);
        body.extend_from_slice(&(p.len() as u32).to_le_bytes());
        body.extend_from_slice(p);
        if p.len() & 1 == 1 {
            body.push(0);
        }
    }
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    Some(out)
}

/// Write `fields` into a **little-endian** TIFF by appending a merged IFD0 (the original entries — kept
/// verbatim, their offsets still valid since nothing moves — plus our tags + Exif/GPS sub-IFDs) at the
/// end and repointing the header to it. The image strips stay put. Returns `None` for a big-endian
/// (`MM`) or malformed TIFF. Note: a pre-existing Exif/GPS sub-IFD is superseded (its extra sub-tags,
/// e.g. exposure, are not carried into the new one) when we write a date/geotag.
fn embed_tiff(data: &[u8], fields: &MetaFields) -> Option<Vec<u8>> {
    if data.len() < 8 || &data[0..2] != b"II" || u16::from_le_bytes([data[2], data[3]]) != 42 {
        return None; // little-endian TIFF only
    }
    let ifd0_off = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    if ifd0_off < 8 || ifd0_off + 2 > data.len() {
        return None;
    }
    let n = u16::from_le_bytes([data[ifd0_off], data[ifd0_off + 1]]) as usize;
    let entries_off = ifd0_off + 2;
    let next_off = entries_off + n * 12;
    if next_off + 4 > data.len() {
        return None;
    }
    let orig_next = u32::from_le_bytes([data[next_off], data[next_off + 1], data[next_off + 2], data[next_off + 3]]);

    let (new_ifd0, exif, gps) = metadata_ifd(fields)?;
    let overrides: Vec<u16> = new_ifd0.iter().map(|f| f.tag).collect();

    // Merge: original entries (dropping the tags we're overriding) + our new fields, sorted by tag.
    let mut ents: Vec<Ent> = Vec::new();
    for k in 0..n {
        let e = entries_off + k * 12;
        let mut raw = [0u8; 12];
        raw.copy_from_slice(&data[e..e + 12]);
        if !overrides.contains(&u16::from_le_bytes([raw[0], raw[1]])) {
            ents.push(Ent::Raw(raw));
        }
    }
    ents.extend(new_ifd0.into_iter().map(Ent::New));
    ents.sort_by_key(ent_tag);

    // Append at an even offset (TIFF offsets are word-aligned); repoint the header at the new IFD0.
    let mut out = data.to_vec();
    if out.len() % 2 == 1 {
        out.push(0);
    }
    let base = out.len() as u32;
    out.extend_from_slice(&layout_ifd(base, ents, exif, gps, orig_next));
    out[4..8].copy_from_slice(&base.to_le_bytes());
    Some(out)
}

/// Write `fields` into `path`'s binary EXIF, in place (atomic temp → rename). JPEG, PNG, WebP and
/// (little-endian) TIFF. Returns `Ok(false)` when there was nothing to write; errors on an unsupported
/// format or I/O failure. Never touches the pixel stream.
pub fn write_metadata(path: &Path, fields: &MetaFields) -> Result<bool> {
    if fields.is_empty() {
        return Ok(false);
    }
    let Some(tiff) = build_tiff(fields) else { return Ok(false) };
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let out = match ext_lower(path).as_str() {
        "jpg" | "jpeg" => embed_jpeg(&data, &tiff),
        "png" => embed_png(&data, &tiff),
        "webp" => embed_webp(&data, &tiff),
        "tif" | "tiff" => embed_tiff(&data, fields),
        other => anyhow::bail!("EXIF write-back supports JPEG/PNG/WebP/TIFF (got {other})"),
    };
    let Some(bytes) = out else {
        anyhow::bail!("{} is not a well-formed little-endian JPEG/PNG/WebP/TIFF", path.display());
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
    fn writes_and_reads_back_webp() {
        let dir = tmpdir("webp");
        let p = dir.join("shot.webp");
        // Encode a WebP via the image crate (lossless); skip gracefully if unsupported in this build.
        if DynamicImage::ImageRgb8(ImageBuffer::from_pixel(24, 18, Rgb([60u8, 140, 200])))
            .save(&p)
            .is_err()
        {
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        assert!(write_metadata(&p, &fields()).unwrap());
        // Still a valid, same-size WebP.
        let img = image::open(&p).unwrap();
        assert_eq!((img.width(), img.height()), (24, 18));
        // The project reader recovers date + GPS from the WebP EXIF chunk.
        let rec = super::super::exif::read_exif(&p).unwrap();
        assert_eq!(rec.date_taken.as_deref(), Some("2024-07-14T12:34:56"));
        assert!((rec.gps_lat.unwrap() - 37.7749).abs() < 1e-3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn writes_and_reads_back_tiff() {
        let dir = tmpdir("tiff");
        let p = dir.join("scan.tiff");
        if DynamicImage::ImageRgb8(ImageBuffer::from_pixel(30, 20, Rgb([120u8, 60, 30])))
            .save(&p)
            .is_err()
        {
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        assert!(write_metadata(&p, &fields()).unwrap());
        // The image still decodes at the same size (strips untouched).
        let img = image::open(&p).unwrap();
        assert_eq!((img.width(), img.height()), (30, 20));
        // The appended IFD is read back by the project reader (date + GPS).
        let rec = super::super::exif::read_exif(&p).unwrap();
        assert_eq!(rec.date_taken.as_deref(), Some("2024-07-14T12:34:56"));
        assert!((rec.gps_lat.unwrap() - 37.7749).abs() < 1e-3, "lat {:?}", rec.gps_lat);

        // Direct kamadak read of the string tags in the appended IFD0.
        let file = std::fs::File::open(&p).unwrap();
        let mut buf = std::io::BufReader::new(file);
        let exif = exif::Reader::new().read_from_container(&mut buf).unwrap();
        let desc = exif.get_field(exif::Tag::ImageDescription, exif::In::PRIMARY).map(|f| f.display_value().to_string());
        assert!(desc.unwrap().contains("Sunset over the bay"), "title in appended IFD");

        // Big-endian TIFF is declined rather than corrupted.
        let mut be = std::fs::read(&p).unwrap();
        be[0] = b'M';
        be[1] = b'M';
        assert!(embed_tiff(&be, &fields()).is_none(), "big-endian declined");
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
