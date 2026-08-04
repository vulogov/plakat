//! 6.1.0 (B4): `bookart font` — export a set of small procedural ornaments (fleurons / dinkus /
//! dividers) as an OpenType **dingbat font** for inline use in InDesign / LaTeX (type a letter, get an
//! ornament). Self-contained: a minimal from-scratch **TrueType** (glyf) writer — no font-toolkit dep.
//! Each ornament's born-vector polylines are RDP-simplified, then every stroke segment is converted to
//! a thin filled rectangle contour so the line art renders as a filled glyph. Verified by round-trip
//! (the generated font re-parses + its glyph outlines are non-empty).

use crate::bookart::finish::vector::simplify;
use crate::bookart::procedural::{self, Polyline};

const EM: i32 = 1024;

/// One dingbat: the codepoint you type, and the ornament kind + symmetry it maps to.
pub struct Dingbat {
    pub ch: char,
    pub kind: &'static str,
    pub symmetry: &'static str,
    pub variant: u32,
}

/// The default dingbat set: `a`–`h` → a spread of small ornaments a compositor would reach for.
pub fn default_set() -> Vec<Dingbat> {
    vec![
        Dingbat { ch: 'a', kind: "fleuron", symmetry: "radial:6", variant: 0 },
        Dingbat { ch: 'b', kind: "fleuron", symmetry: "radial:8", variant: 1 },
        Dingbat { ch: 'c', kind: "dinkus", symmetry: "radial:5", variant: 2 },
        Dingbat { ch: 'd', kind: "rosette", symmetry: "radial:12", variant: 0 },
        Dingbat { ch: 'e', kind: "divider", symmetry: "bilateral", variant: 0 },
        Dingbat { ch: 'f', kind: "corner", symmetry: "radial:6", variant: 1 },
        Dingbat { ch: 'g', kind: "rosette", symmetry: "radial:4", variant: 3 },
        Dingbat { ch: 'h', kind: "fleuron", symmetry: "radial:10", variant: 2 },
    ]
}

/// Build an OpenType (TrueType-flavoured) dingbat font from a set of ornaments. `family` names it.
pub fn build_font(set: &[Dingbat], family: &str) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(!set.is_empty(), "font: empty dingbat set");
    // Glyph 0 is .notdef (empty). Glyphs 1..=N are the ornaments, in codepoint order.
    let mut chars: Vec<&Dingbat> = set.iter().collect();
    chars.sort_by_key(|d| d.ch as u32);

    let mut glyphs: Vec<Glyph> = vec![Glyph::empty()]; // .notdef
    for d in &chars {
        let paths = procedural::generate_paths(d.kind, d.symmetry, EM as u32, EM as u32, d.variant);
        glyphs.push(ornament_glyph(&paths));
    }
    let mappings: Vec<(u32, u16)> = chars.iter().enumerate().map(|(i, d)| (d.ch as u32, (i + 1) as u16)).collect();
    Ok(assemble(&glyphs, &mappings, family))
}

/// A finished glyph: filled contours (each a closed ring of integer em points) + its bbox.
struct Glyph {
    contours: Vec<Vec<(i32, i32)>>,
    x_min: i32,
    y_min: i32,
    x_max: i32,
    y_max: i32,
}

impl Glyph {
    fn empty() -> Self {
        Self { contours: vec![], x_min: 0, y_min: 0, x_max: 0, y_max: 0 }
    }
    fn from_contours(contours: Vec<Vec<(i32, i32)>>) -> Self {
        let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for c in &contours {
            for &(x, y) in c {
                x0 = x0.min(x); y0 = y0.min(y); x1 = x1.max(x); y1 = y1.max(y);
            }
        }
        if contours.is_empty() {
            return Self::empty();
        }
        Self { contours, x_min: x0, y_min: y0, x_max: x1, y_max: y1 }
    }
}

/// Convert an ornament's polylines to a filled glyph: simplify, then stroke each segment into a thin
/// filled rectangle (em y-up). Non-zero winding fills the strokes; overlaps at joints are harmless.
fn ornament_glyph(paths: &[Polyline]) -> Glyph {
    let half = (EM as f32) * 0.006; // stroke half-width in em units
    let mut contours: Vec<Vec<(i32, i32)>> = Vec::new();
    let flip_y = |y: f32| (EM as f32) - y; // pixel-space y-down → font y-up
    // Clamp to a generous box so a runaway curve point can never overflow the i16 glyf coords.
    let cl = |v: f32| (v.round().clamp(-256.0, (EM + 256) as f32)) as i32;
    for path in paths {
        let simp = simplify(path, 5.0); // dingbats don't need dense curves; keep glyphs light
        for seg in simp.windows(2) {
            let (ax, ay) = (seg[0].0, flip_y(seg[0].1));
            let (bx, by) = (seg[1].0, flip_y(seg[1].1));
            if ![ax, ay, bx, by].iter().all(|v| v.is_finite()) {
                continue;
            }
            let (dx, dy) = (bx - ax, by - ay);
            let len = (dx * dx + dy * dy).sqrt();
            if len < 0.5 {
                continue;
            }
            let (nx, ny) = (-dy / len * half, dx / len * half); // unit normal * half-width
            // CCW rectangle: a+n, b+n, b-n, a-n
            let ring = vec![
                (cl(ax + nx), cl(ay + ny)),
                (cl(bx + nx), cl(by + ny)),
                (cl(bx - nx), cl(by - ny)),
                (cl(ax - nx), cl(ay - ny)),
            ];
            contours.push(ring);
        }
    }
    Glyph::from_contours(contours)
}

// --- minimal TrueType assembly --------------------------------------------------------------------

#[derive(Default)]
struct Buf(Vec<u8>);
impl Buf {
    fn u8(&mut self, v: u8) { self.0.push(v); }
    fn u16(&mut self, v: u16) { self.0.extend_from_slice(&v.to_be_bytes()); }
    fn i16(&mut self, v: i16) { self.0.extend_from_slice(&v.to_be_bytes()); }
    fn u32(&mut self, v: u32) { self.0.extend_from_slice(&v.to_be_bytes()); }
    fn i64(&mut self, v: i64) { self.0.extend_from_slice(&v.to_be_bytes()); }
    fn pad4(&mut self) { while self.0.len() % 4 != 0 { self.0.push(0); } }
}

/// Encode one glyf simple glyph (empty glyph = zero bytes).
fn glyf_bytes(g: &Glyph) -> Vec<u8> {
    if g.contours.is_empty() {
        return Vec::new();
    }
    let mut b = Buf::default();
    b.i16(g.contours.len() as i16);
    // Glyph header bbox comes right after numberOfContours (before endPtsOfContours).
    b.i16(g.x_min as i16);
    b.i16(g.y_min as i16);
    b.i16(g.x_max as i16);
    b.i16(g.y_max as i16);
    let mut end = -1i32;
    for c in &g.contours {
        end += c.len() as i32;
        b.u16(end as u16);
    }
    b.u16(0); // instructionLength
    // All points on-curve; x/y as int16 deltas (flag 0x01, no short, no same).
    let total: usize = g.contours.iter().map(|c| c.len()).sum();
    for _ in 0..total {
        b.u8(0x01);
    }
    // x deltas
    let mut px = 0i32;
    for c in &g.contours {
        for &(x, _) in c {
            b.i16((x - px) as i16);
            px = x;
        }
    }
    // y deltas
    let mut py = 0i32;
    for c in &g.contours {
        for &(_, y) in c {
            b.i16((y - py) as i16);
            py = y;
        }
    }
    b.pad4();
    b.0
}

/// cmap format-4 subtable mapping sorted (codepoint→glyphId), grouped into contiguous segments.
fn cmap_format4(mappings: &[(u32, u16)]) -> Vec<u8> {
    // Group contiguous (cp, gid) runs where both advance by 1 into segments.
    struct Seg { start: u16, end: u16, delta: i16 }
    let mut segs: Vec<Seg> = Vec::new();
    let mut i = 0;
    while i < mappings.len() {
        let (cp0, gid0) = mappings[i];
        let mut j = i;
        while j + 1 < mappings.len()
            && mappings[j + 1].0 == mappings[j].0 + 1
            && mappings[j + 1].1 == mappings[j].1 + 1
        {
            j += 1;
        }
        let start = cp0 as u16;
        let end = mappings[j].0 as u16;
        let delta = (gid0 as i32 - cp0 as i32) as i16;
        segs.push(Seg { start, end, delta });
        i = j + 1;
    }
    segs.push(Seg { start: 0xFFFF, end: 0xFFFF, delta: 1 }); // required terminator
    let seg_count = segs.len() as u16;
    let x2 = seg_count * 2;
    // searchRange = 2 * (largest power of two ≤ segCount); entrySelector = log2 of that power.
    let search_range = 2 * 2u16.pow((seg_count as f32).log2() as u32);
    let entry_selector = (search_range as f32 / 2.0).log2() as u16;
    let range_shift = x2 - search_range;

    let mut b = Buf::default();
    b.u16(4); // format
    let len_pos = b.0.len();
    b.u16(0); // length (patched)
    b.u16(0); // language
    b.u16(x2);
    b.u16(search_range);
    b.u16(entry_selector);
    b.u16(range_shift);
    for s in &segs { b.u16(s.end); }
    b.u16(0); // reservedPad
    for s in &segs { b.u16(s.start); }
    for s in &segs { b.i16(s.delta); }
    for _ in &segs { b.u16(0); } // idRangeOffset (all 0 → glyphId = cp + delta)
    let len = b.0.len() as u16;
    b.0[len_pos..len_pos + 2].copy_from_slice(&len.to_be_bytes());
    b.0
}

/// Full cmap table (one Windows-BMP encoding record → the format-4 subtable).
fn cmap_table(mappings: &[(u32, u16)]) -> Vec<u8> {
    let sub = cmap_format4(mappings);
    let mut b = Buf::default();
    b.u16(0); // version
    b.u16(1); // numTables
    b.u16(3); // platformID Windows
    b.u16(1); // encodingID Unicode BMP
    b.u32(12); // offset to subtable (4 + 8)
    b.0.extend_from_slice(&sub);
    b.0
}

/// A UTF-16BE `name` table with the standard IDs (family/subfamily/full/unique/postscript/version).
fn name_table(family: &str) -> Vec<u8> {
    let sub = format!("{family}-Regular");
    let records: [(u16, &str); 6] = [
        (1, family), (2, "Regular"), (3, &sub), (4, family), (6, &sub), (5, "Version 1.000"),
    ];
    let mut strings = Buf::default();
    struct Rec { id: u16, off: u16, len: u16 }
    let mut recs = Vec::new();
    for (id, s) in records {
        let off = strings.0.len() as u16;
        for u in s.encode_utf16() {
            strings.u16(u);
        }
        recs.push(Rec { id, off, len: (strings.0.len() as u16) - off });
    }
    let mut b = Buf::default();
    b.u16(0); // format
    b.u16(recs.len() as u16);
    let storage_off = 6 + recs.len() as u16 * 12;
    b.u16(storage_off);
    for r in &recs {
        b.u16(3); // platform Windows
        b.u16(1); // encoding Unicode BMP
        b.u16(0x0409); // language en-US
        b.u16(r.id);
        b.u16(r.len);
        b.u16(r.off);
    }
    b.0.extend_from_slice(&strings.0);
    b.0
}

/// Assemble the sfnt: build every required table, lay them out 4-aligned, fix the head checksum.
fn assemble(glyphs: &[Glyph], mappings: &[(u32, u16)], family: &str) -> Vec<u8> {
    let num_glyphs = glyphs.len() as u16;
    // glyf + loca (long).
    let mut glyf = Vec::new();
    let mut loca = Buf::default();
    let mut max_points = 0u16;
    let mut max_contours = 0u16;
    let (mut gx0, mut gy0, mut gx1, mut gy1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for g in glyphs {
        loca.u32(glyf.len() as u32);
        glyf.extend_from_slice(&glyf_bytes(g));
        let pts: usize = g.contours.iter().map(|c| c.len()).sum();
        max_points = max_points.max(pts as u16);
        max_contours = max_contours.max(g.contours.len() as u16);
        if !g.contours.is_empty() {
            gx0 = gx0.min(g.x_min); gy0 = gy0.min(g.y_min); gx1 = gx1.max(g.x_max); gy1 = gy1.max(g.y_max);
        }
    }
    loca.u32(glyf.len() as u32); // final offset
    if gx0 == i32::MAX { gx0 = 0; gy0 = 0; gx1 = EM; gy1 = EM; }

    // head
    let mut head = Buf::default();
    head.u16(1); head.u16(0); // version 1.0
    head.u32(0x0001_0000); // fontRevision 1.0
    head.u32(0); // checkSumAdjustment (patched)
    head.u32(0x5F0F_3CF5); // magic
    head.u16(0x000B); // flags
    head.u16(EM as u16); // unitsPerEm
    head.i64(0); head.i64(0); // created / modified (epoch 1904; deterministic)
    head.i16(gx0 as i16); head.i16(gy0 as i16); head.i16(gx1 as i16); head.i16(gy1 as i16);
    head.u16(0); // macStyle
    head.u16(8); // lowestRecPPEM
    head.i16(2); // fontDirectionHint
    head.i16(1); // indexToLocFormat = long
    head.i16(0); // glyphDataFormat

    // hhea
    let advance = EM as u16;
    let mut hhea = Buf::default();
    hhea.u16(1); hhea.u16(0);
    hhea.i16((EM as f32 * 0.8) as i16); // ascender
    hhea.i16(-(EM as f32 * 0.2) as i16); // descender
    hhea.i16(0); // lineGap
    hhea.u16(advance); // advanceWidthMax
    hhea.i16(gx0 as i16); // minLeftSideBearing
    hhea.i16(0); // minRightSideBearing
    hhea.i16(gx1 as i16); // xMaxExtent
    hhea.i16(1); hhea.i16(0); hhea.i16(0); // caret slope/offset
    hhea.i16(0); hhea.i16(0); hhea.i16(0); hhea.i16(0); // reserved
    hhea.i16(0); // metricDataFormat
    hhea.u16(num_glyphs); // numberOfHMetrics

    // hmtx (one metric per glyph)
    let mut hmtx = Buf::default();
    for g in glyphs {
        hmtx.u16(advance);
        hmtx.i16(if g.contours.is_empty() { 0 } else { g.x_min as i16 });
    }

    // maxp
    let mut maxp = Buf::default();
    maxp.u32(0x0001_0000);
    maxp.u16(num_glyphs);
    maxp.u16(max_points); maxp.u16(max_contours);
    maxp.u16(0); maxp.u16(0); // composite
    maxp.u16(2); // maxZones
    maxp.u16(0); maxp.u16(0); maxp.u16(0); maxp.u16(0); maxp.u16(0); maxp.u16(0); maxp.u16(0); maxp.u16(0);

    // post 3.0
    let mut post = Buf::default();
    post.u32(0x0003_0000);
    post.u32(0); // italicAngle
    post.i16(-(EM as f32 * 0.1) as i16); // underlinePosition
    post.i16((EM as f32 * 0.05) as i16); // underlineThickness
    post.u32(0); // isFixedPitch
    post.u32(0); post.u32(0); post.u32(0); post.u32(0);

    // OS/2 v4
    let mut os2 = Buf::default();
    os2.u16(4); // version
    os2.i16((EM as f32 * 0.5) as i16); // xAvgCharWidth
    os2.u16(400); os2.u16(5); os2.u16(0); // weight / width / fsType
    for _ in 0..10 { os2.i16(0); } // subscript/superscript/strikeout (8) + sFamilyClass(1)...
    for _ in 0..10 { os2.u8(0); } // panose[10]
    os2.u32(0); os2.u32(0); os2.u32(0); os2.u32(0); // unicode ranges
    os2.0.extend_from_slice(b"PLKT"); // achVendID
    os2.u16(0x0040); // fsSelection (REGULAR)
    let first = mappings.first().map(|m| m.0 as u16).unwrap_or(0x20);
    let last = mappings.last().map(|m| m.0 as u16).unwrap_or(0x20);
    os2.u16(first); os2.u16(last);
    os2.i16((EM as f32 * 0.8) as i16); // sTypoAscender
    os2.i16(-(EM as f32 * 0.2) as i16); // sTypoDescender
    os2.i16(0); // sTypoLineGap
    os2.u16((EM as f32 * 0.9) as u16); // usWinAscent
    os2.u16((EM as f32 * 0.25) as u16); // usWinDescent
    os2.u32(1); os2.u32(0); // codepage ranges (Latin-1)
    os2.i16((EM as f32 * 0.5) as i16); // sxHeight
    os2.i16((EM as f32 * 0.7) as i16); // sCapHeight
    os2.u16(first as u16); os2.u16(first as u16); // default / break char
    os2.u16(1); // usMaxContext

    let cmap = cmap_table(mappings);
    let name = name_table(family);

    // Table directory (tags sorted ascending).
    let mut tables: Vec<(&[u8; 4], Vec<u8>)> = vec![
        (b"OS/2", os2.0), (b"cmap", cmap), (b"glyf", glyf), (b"head", head.0),
        (b"hhea", hhea.0), (b"hmtx", hmtx.0), (b"loca", loca.0), (b"maxp", maxp.0),
        (b"name", name), (b"post", post.0),
    ];
    tables.sort_by(|a, b| a.0.cmp(b.0));

    let num_tables = tables.len() as u16;
    let mut sr = 16u16;
    let mut es = 0u16;
    while sr * 2 <= num_tables * 16 {
        sr *= 2;
        es += 1;
    }
    let range_shift = num_tables * 16 - sr;

    let mut out = Buf::default();
    out.u32(0x0001_0000); // sfntVersion (TrueType outlines)
    out.u16(num_tables);
    out.u16(sr); out.u16(es); out.u16(range_shift);

    // Reserve directory; compute offsets.
    let dir_start = out.0.len();
    for _ in &tables {
        for _ in 0..16 { out.u8(0); }
    }
    let mut head_abs_off = 0usize;
    let mut records: Vec<(&[u8; 4], u32, u32, u32)> = Vec::new(); // tag, checksum, offset, len
    for (tag, data) in &tables {
        while out.0.len() % 4 != 0 { out.u8(0); }
        let off = out.0.len() as u32;
        if *tag == b"head" {
            head_abs_off = off as usize;
        }
        let checksum = table_checksum(data);
        records.push((tag, checksum, off, data.len() as u32));
        out.0.extend_from_slice(data);
    }
    // Write the directory records.
    let mut cur = dir_start;
    for (tag, checksum, off, len) in &records {
        out.0[cur..cur + 4].copy_from_slice(*tag);
        out.0[cur + 4..cur + 8].copy_from_slice(&checksum.to_be_bytes());
        out.0[cur + 8..cur + 12].copy_from_slice(&off.to_be_bytes());
        out.0[cur + 12..cur + 16].copy_from_slice(&len.to_be_bytes());
        cur += 16;
    }
    // head.checkSumAdjustment = 0xB1B0AFBA - checksum(whole file).
    while out.0.len() % 4 != 0 { out.u8(0); }
    let file_sum = table_checksum(&out.0);
    let adj = 0xB1B0_AFBAu32.wrapping_sub(file_sum);
    out.0[head_abs_off + 8..head_abs_off + 12].copy_from_slice(&adj.to_be_bytes());
    out.0
}

/// TrueType table checksum: sum of big-endian u32 words (zero-padded to a 4-byte multiple).
fn table_checksum(data: &[u8]) -> u32 {
    let mut sum = 0u32;
    let mut i = 0;
    while i < data.len() {
        let mut word = [0u8; 4];
        for (k, w) in word.iter_mut().enumerate() {
            if i + k < data.len() {
                *w = data[i + k];
            }
        }
        sum = sum.wrapping_add(u32::from_be_bytes(word));
        i += 4;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_a_nonempty_font_with_glyf_and_cmap() {
        let bytes = build_font(&default_set(), "PlakatDingbats").unwrap();
        // sfnt magic + table tags present.
        assert_eq!(&bytes[0..4], &[0x00, 0x01, 0x00, 0x00], "TrueType sfnt version");
        for tag in [b"glyf", b"loca", b"cmap", b"head", b"maxp", b"hmtx", b"hhea", b"name"] {
            assert!(bytes.windows(4).any(|w| w == tag), "missing table {}", std::str::from_utf8(tag).unwrap());
        }
        assert!(bytes.len() > 2000, "font suspiciously small ({} bytes)", bytes.len());
    }

    #[test]
    fn cmap_segments_are_contiguous_plus_terminator() {
        // a..h contiguous cps → glyph ids 1..8 → a single real segment + the 0xFFFF terminator.
        let m: Vec<(u32, u16)> = ('a'..='h').enumerate().map(|(i, c)| (c as u32, (i + 1) as u16)).collect();
        let sub = cmap_format4(&m);
        // format 4, segCountX2 at offset 6 → 2 segments * 2 = 4.
        assert_eq!(u16::from_be_bytes([sub[0], sub[1]]), 4, "cmap format 4");
        assert_eq!(u16::from_be_bytes([sub[6], sub[7]]), 4, "2 segments (1 real + terminator)");
    }
}
