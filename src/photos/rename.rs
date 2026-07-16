//! Batch rename planning (RFC PHOTOS-1 Phase 5). Pure: turns a pattern + a list of files into the
//! new filenames, so it's fully testable without touching disk. The actual on-disk rename + record
//! migration lives in the parent module (album-local, two-phase to avoid intra-set collisions).
//!
//! Pattern grammar: a run of `#` is replaced by the 1-based sequence number, zero-padded to the run
//! length (`trip_###` → `trip_001`, `trip_002`, …). With no `#`, `-N` is appended (padded to the
//! digit-width of the count). The original file's extension is always preserved.

use std::path::PathBuf;

/// First run of `#` in `pattern` as `(start, len)`, if any.
fn hash_run(pattern: &str) -> Option<(usize, usize)> {
    let bytes = pattern.as_bytes();
    let start = bytes.iter().position(|&b| b == b'#')?;
    let len = bytes[start..].iter().take_while(|&&b| b == b'#').count();
    Some((start, len))
}

/// Map each file to its new filename (extension preserved). 1-based numbering in input order.
pub fn plan(files: &[PathBuf], pattern: &str) -> Vec<(PathBuf, String)> {
    let width_default = files.len().to_string().len().max(1);
    let run = hash_run(pattern);
    files
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let n = i + 1;
            let ext = f.extension().and_then(|e| e.to_str()).unwrap_or("");
            let stem = match run {
                Some((start, len)) => {
                    let num = format!("{n:0len$}");
                    format!("{}{}{}", &pattern[..start], num, &pattern[start + len..])
                }
                None => format!("{pattern}-{n:0width_default$}"),
            };
            let name = if ext.is_empty() { stem } else { format!("{stem}.{ext}") };
            (f.clone(), name)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files() -> Vec<PathBuf> {
        vec![PathBuf::from("a.jpg"), PathBuf::from("b.png"), PathBuf::from("c.JPG")]
    }

    #[test]
    fn hash_run_padded_and_ext_preserved() {
        let out = plan(&files(), "trip_##");
        let names: Vec<&str> = out.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, ["trip_01.jpg", "trip_02.png", "trip_03.JPG"]);
    }

    #[test]
    fn no_hash_appends_padded_number() {
        // 3 files → width 1.
        let out = plan(&files(), "photo");
        assert_eq!(out[0].1, "photo-1.jpg");
        assert_eq!(out[2].1, "photo-3.JPG");
    }

    #[test]
    fn embedded_hash_run_keeps_prefix_and_suffix() {
        let out = plan(&[PathBuf::from("x.webp")], "IMG_###_final");
        assert_eq!(out[0].1, "IMG_001_final.webp");
    }
}
