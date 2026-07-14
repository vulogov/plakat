//! EXIF extraction (RFC PHOTOS-1 §17) via `kamadak-exif`. Reads JPEG / TIFF / PNG / WebP / HEIF /
//! AVIF and TIFF-based RAW (CR2/NEF/ARW/DNG/ORF/RAF). Populated once per image on first scan.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use exif::{In, Reader, Tag, Value};

use super::hjson::ExifRecord;

fn ascii(exif: &exif::Exif, tag: Tag) -> Option<String> {
    match &exif.get_field(tag, In::PRIMARY)?.value {
        Value::Ascii(v) if !v.is_empty() => {
            Some(String::from_utf8_lossy(&v[0]).trim().to_string()).filter(|s| !s.is_empty())
        }
        _ => None,
    }
}

fn urational_f64(exif: &exif::Exif, tag: Tag) -> Option<f64> {
    match &exif.get_field(tag, In::PRIMARY)?.value {
        Value::Rational(v) if !v.is_empty() => Some(v[0].to_f64()),
        _ => None,
    }
}

fn u32_val(exif: &exif::Exif, tag: Tag) -> Option<u32> {
    exif.get_field(tag, In::PRIMARY)?.value.get_uint(0)
}

/// GPS DMS (three rationals) + hemisphere ref → signed decimal degrees.
fn gps_coord(exif: &exif::Exif, coord: Tag, refr: Tag, neg_ref: &str) -> Option<f64> {
    let dms = match &exif.get_field(coord, In::PRIMARY)?.value {
        Value::Rational(v) if v.len() == 3 => v,
        _ => return None,
    };
    let deg = dms[0].to_f64() + dms[1].to_f64() / 60.0 + dms[2].to_f64() / 3600.0;
    let sign = ascii(exif, refr)
        .map(|r| if r.eq_ignore_ascii_case(neg_ref) { -1.0 } else { 1.0 })
        .unwrap_or(1.0);
    Some(deg * sign)
}

/// Read EXIF for `path` into an [`ExifRecord`]. Returns a default (all-`None`) record when the file
/// carries no EXIF — never an error for that case; only I/O errors propagate.
pub fn read_exif(path: &Path) -> anyhow::Result<ExifRecord> {
    let file = File::open(path)?;
    let mut buf = BufReader::new(file);
    // `continue_on_error` tolerates partial EXIF; a hard error means no decodable EXIF → empty record.
    match Reader::new().continue_on_error(true).read_from_container(&mut buf) {
        Ok(exif) => Ok(build(&exif)),
        Err(_) => Ok(ExifRecord::default()),
    }
}

fn build(exif: &exif::Exif) -> ExifRecord {
    let date_taken = ascii(exif, Tag::DateTimeOriginal).and_then(|s| {
        // "YYYY:MM:DD HH:MM:SS" → ISO-8601.
        let s = s.replacen(':', "-", 2).replacen(' ', "T", 1);
        Some(s).filter(|s| s.len() >= 19)
    });

    let aperture = urational_f64(exif, Tag::FNumber).map(|f| format!("f/{f:.1}"));
    let shutter = match &exif.get_field(Tag::ExposureTime, In::PRIMARY).map(|f| &f.value) {
        Some(Value::Rational(v)) if !v.is_empty() => {
            let r = v[0];
            Some(if r.denom > r.num {
                format!("1/{}", (r.denom as f64 / r.num.max(1) as f64).round() as u64)
            } else {
                format!("{:.0}s", r.to_f64())
            })
        }
        _ => None,
    };

    ExifRecord {
        date_taken,
        camera_make: ascii(exif, Tag::Make),
        camera_model: ascii(exif, Tag::Model),
        lens_model: ascii(exif, Tag::LensModel),
        focal_length_mm: urational_f64(exif, Tag::FocalLength),
        aperture,
        shutter,
        iso: u32_val(exif, Tag::PhotographicSensitivity).or_else(|| u32_val(exif, Tag::ISOSpeed)),
        gps_lat: gps_coord(exif, Tag::GPSLatitude, Tag::GPSLatitudeRef, "S"),
        gps_lon: gps_coord(exif, Tag::GPSLongitude, Tag::GPSLongitudeRef, "W"),
        width_px: u32_val(exif, Tag::PixelXDimension).or_else(|| u32_val(exif, Tag::ImageWidth)),
        height_px: u32_val(exif, Tag::PixelYDimension).or_else(|| u32_val(exif, Tag::ImageLength)),
        orientation: u32_val(exif, Tag::Orientation).map(|o| o as u16),
    }
}
