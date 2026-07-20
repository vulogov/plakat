//! Static web-gallery export (RFC PHOTOS 3.9 — "present & share").
//!
//! Turns a selection / album into a **portable, fully-offline** folder you can open locally or drop
//! on any static host: `index.html` (self-contained — inline CSS + JS, no CDN, no network) plus a
//! `thumbs/` grid and a `full/` set of (optionally down-sized) images. The lightbox is keyboard-first
//! (←/→/Esc, click to advance). Create-only, like [`super::export`] / [`super::portfolio`] — the
//! album copies stay put.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Options for [`export`]. `thumb_px` bounds the grid thumbnail's longer side; `full_px`, when set,
/// bounds each full image's longer side (otherwise the source is copied verbatim).
pub struct Options<'a> {
    pub title: &'a str,
    pub thumb_px: u32,
    pub full_px: Option<u32>,
}

impl Default for Options<'_> {
    fn default() -> Self {
        Options { title: "Gallery", thumb_px: 400, full_px: None }
    }
}

/// One image to publish, with the curation / shot metadata to embed in the page.
#[derive(Default, Clone)]
pub struct Photo {
    pub path: PathBuf,
    /// Display caption (title / caption); defaults to the file stem when empty.
    pub title: Option<String>,
    /// 0–5 stars (0 hides the rating).
    pub rating: u8,
    pub tags: Vec<String>,
    /// A short capture-date string.
    pub date: Option<String>,
    /// A pre-formatted EXIF summary line (camera · lens · exposure · ISO).
    pub exif: Option<String>,
}

/// One image's place in the gallery: its full/thumbnail relative paths + the embedded metadata.
struct Item {
    full: String,
    thumb: String,
    caption: String,
    rating: u8,
    tags: Vec<String>,
    date: String,
    exif: String,
}

/// Build a static gallery of `photos` under `dest` (created if missing): `dest/full/NNN.ext`,
/// `dest/thumbs/NNN.jpg`, and `dest/index.html` — with captions, ratings, tags and an EXIF summary
/// embedded in each lightbox entry. Returns the number of images included. Best-effort per file (a
/// source that won't decode is skipped with a note).
pub fn export(photos: &[Photo], dest: &Path, opts: &Options) -> Result<usize> {
    let full_dir = dest.join("full");
    let thumb_dir = dest.join("thumbs");
    std::fs::create_dir_all(&full_dir).with_context(|| format!("creating {}", full_dir.display()))?;
    std::fs::create_dir_all(&thumb_dir).with_context(|| format!("creating {}", thumb_dir.display()))?;

    let mut items: Vec<Item> = Vec::new();
    for photo in photos {
        match one(&photo.path, &full_dir, &thumb_dir, items.len(), opts) {
            Ok((full, thumb)) => {
                let caption = photo
                    .title
                    .clone()
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or_else(|| photo.path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string());
                items.push(Item {
                    full,
                    thumb,
                    caption,
                    rating: photo.rating.min(5),
                    tags: photo.tags.clone(),
                    date: photo.date.clone().unwrap_or_default(),
                    exif: photo.exif.clone().unwrap_or_default(),
                });
            }
            Err(e) => {
                crate::ui::progress::println(&format!("  gallery skipped {}: {e:#}", photo.path.display()))
            }
        }
    }

    let html = render_html(opts.title, &items);
    let index = dest.join("index.html");
    std::fs::write(&index, html).with_context(|| format!("writing {}", index.display()))?;
    Ok(items.len())
}

/// Emit `full/NNN.ext` (down-sized to `full_px` if set) and `thumbs/NNN.jpg` for one source; returns
/// their album-relative paths.
fn one(src: &Path, full_dir: &Path, thumb_dir: &Path, idx: usize, opts: &Options) -> Result<(String, String)> {
    let ext = src
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .filter(|e| matches!(e.as_str(), "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp"))
        .unwrap_or_else(|| "png".into());
    let full_name = format!("{idx:04}.{ext}");
    let thumb_name = format!("{idx:04}.jpg");
    let full_out = full_dir.join(&full_name);
    let thumb_out = thumb_dir.join(&thumb_name);

    let img = image::open(src).with_context(|| format!("reading {}", src.display()))?;

    // Full image: copy verbatim when small enough / no cap, else down-size and re-encode.
    match opts.full_px {
        Some(px) if img.width().max(img.height()) > px => {
            img.resize(px, px, image::imageops::FilterType::Lanczos3)
                .save(&full_out)
                .with_context(|| format!("writing {}", full_out.display()))?;
        }
        _ => {
            std::fs::copy(src, &full_out).with_context(|| format!("copying {}", src.display()))?;
        }
    }

    // Thumbnail: always a bounded JPEG for a light, fast grid.
    let thumb = img.resize(opts.thumb_px, opts.thumb_px, image::imageops::FilterType::Lanczos3);
    thumb.to_rgb8().save(&thumb_out).with_context(|| format!("writing {}", thumb_out.display()))?;

    Ok((format!("full/{full_name}"), format!("thumbs/{thumb_name}")))
}

/// Rating as filled/empty stars (`★★★☆☆`), empty when unrated.
fn stars(rating: u8) -> String {
    let r = rating.min(5) as usize;
    if r == 0 {
        String::new()
    } else {
        format!("{}{}", "★".repeat(r), "☆".repeat(5 - r))
    }
}

/// Escape a string for safe inclusion in HTML text / attribute contexts.
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Assemble the self-contained `index.html`: inline CSS (responsive dark grid) + a keyboard lightbox
/// that shows each image's caption, rating, EXIF summary and tags — all embedded as data attributes.
fn render_html(title: &str, items: &[Item]) -> String {
    let t = esc(title);
    let mut cells = String::new();
    for it in items.iter() {
        let star_overlay = if it.rating > 0 {
            format!("<span class=\"stars\">{}</span>", esc(&stars(it.rating)))
        } else {
            String::new()
        };
        cells.push_str(&format!(
            "<a class=\"cell\" href=\"{full}\" title=\"{cap}\" \
             data-rating=\"{rating}\" data-date=\"{date}\" data-exif=\"{exif}\" data-tags=\"{tags}\">\
             <img loading=\"lazy\" src=\"{thumb}\" alt=\"{cap}\">{star_overlay}</a>\n",
            full = esc(&it.full),
            thumb = esc(&it.thumb),
            cap = esc(&it.caption),
            rating = it.rating,
            date = esc(&it.date),
            exif = esc(&it.exif),
            tags = esc(&it.tags.join(", ")),
        ));
    }
    // The lightbox reads caption/rating/exif/tags straight off the clicked <a>, so the data model is
    // the DOM — no image list duplicated into JS.
    format!(
        r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{t}</title>
<style>
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #14161a; color: #e7e9ee;
  font: 15px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
header {{ padding: 22px 24px 6px; }}
header h1 {{ margin: 0; font-size: 22px; font-weight: 600; }}
header .sub {{ color: #8b93a3; font-size: 13px; margin-top: 4px; }}
.grid {{ display: grid; gap: 6px; padding: 16px 24px 40px;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); }}
.cell {{ position: relative; display: block; overflow: hidden; border-radius: 6px; background: #1d2027;
  aspect-ratio: 1; cursor: zoom-in; }}
.cell img {{ width: 100%; height: 100%; object-fit: cover; display: block;
  transition: transform .18s ease; }}
.cell:hover img {{ transform: scale(1.05); }}
.cell .stars {{ position: absolute; left: 6px; bottom: 6px; font-size: 12px; color: #ffd24a;
  text-shadow: 0 1px 3px rgba(0,0,0,.8); pointer-events: none; }}
#lb {{ position: fixed; inset: 0; background: rgba(6,7,9,.94); display: none;
  flex-direction: column; align-items: center; justify-content: center; cursor: zoom-out; }}
#lb.open {{ display: flex; }}
#lb img {{ max-width: 94vw; max-height: 80vh; object-fit: contain; border-radius: 4px;
  box-shadow: 0 10px 60px rgba(0,0,0,.6); }}
#lb .meta {{ margin-top: 12px; max-width: 90vw; text-align: center; pointer-events: none; }}
#lb .meta .cap {{ font-size: 16px; font-weight: 600; color: #eef1f6; }}
#lb .meta .stars {{ color: #ffd24a; font-size: 14px; margin-left: 8px; }}
#lb .meta .exif {{ color: #9aa3b2; font-size: 12px; margin-top: 3px; font-variant-numeric: tabular-nums; }}
#lb .meta .tags {{ margin-top: 5px; }}
#lb .meta .tag {{ display: inline-block; background: #262a33; color: #c7cdda; border-radius: 10px;
  padding: 1px 9px; font-size: 11px; margin: 2px; }}
#lb .nav {{ position: fixed; top: 0; bottom: 0; width: 22vw; cursor: pointer; }}
#lb .prev {{ left: 0; }} #lb .next {{ right: 0; }}
.empty {{ padding: 40px 24px; color: #8b93a3; }}
</style>
</head>
<body>
<header><h1>{t}</h1><div class="sub">{n} image{plural} · plakat</div></header>
{body}
<div id="lb"><img alt="">
  <div class="meta"><div class="cap"></div><div class="exif"></div><div class="tags"></div></div>
  <div class="nav prev"></div><div class="nav next"></div></div>
<script>
(function() {{
  var cells = Array.prototype.slice.call(document.querySelectorAll('.cell'));
  var lb = document.getElementById('lb'), img = lb.querySelector('img');
  var elCap = lb.querySelector('.cap'), elExif = lb.querySelector('.exif'), elTags = lb.querySelector('.tags');
  var cur = -1;
  function starStr(n) {{ n = n|0; return n > 0 ? '★'.repeat(n) + '☆'.repeat(5 - n) : ''; }}
  function show(i) {{
    if (i < 0 || i >= cells.length) return;
    cur = i;
    var c = cells[i];
    img.src = c.getAttribute('href');
    var cap = c.getAttribute('title') || '';
    var st = starStr(c.getAttribute('data-rating'));
    elCap.innerHTML = '';
    elCap.appendChild(document.createTextNode(cap));
    if (st) {{ var s = document.createElement('span'); s.className = 'stars'; s.textContent = st; elCap.appendChild(s); }}
    var exif = c.getAttribute('data-exif') || '', date = c.getAttribute('data-date') || '';
    elExif.textContent = [date, exif].filter(Boolean).join('  ·  ');
    elTags.innerHTML = '';
    (c.getAttribute('data-tags') || '').split(',').map(function(s) {{ return s.trim(); }}).filter(Boolean)
      .forEach(function(tag) {{ var e = document.createElement('span'); e.className = 'tag'; e.textContent = tag; elTags.appendChild(e); }});
    lb.classList.add('open');
  }}
  function close() {{ lb.classList.remove('open'); img.src = ''; cur = -1; }}
  cells.forEach(function(c, i) {{
    c.addEventListener('click', function(e) {{ e.preventDefault(); show(i); }});
  }});
  lb.querySelector('.prev').addEventListener('click', function(e) {{ e.stopPropagation(); show(cur - 1); }});
  lb.querySelector('.next').addEventListener('click', function(e) {{ e.stopPropagation(); show(cur + 1); }});
  img.addEventListener('click', function(e) {{ e.stopPropagation(); show(cur + 1); }});
  lb.addEventListener('click', close);
  document.addEventListener('keydown', function(e) {{
    if (!lb.classList.contains('open')) return;
    if (e.key === 'Escape') close();
    else if (e.key === 'ArrowLeft') show(cur - 1);
    else if (e.key === 'ArrowRight' || e.key === ' ') {{ e.preventDefault(); show(cur + 1); }}
  }});
}})();
</script>
</body>
</html>
"##,
        t = t,
        n = items.len(),
        plural = if items.len() == 1 { "" } else { "s" },
        body = if items.is_empty() {
            "<div class=\"empty\">No images.</div>".to_string()
        } else {
            format!("<div class=\"grid\">\n{cells}</div>")
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Rgb};

    #[test]
    fn builds_gallery_folder_and_index() {
        let base = std::env::temp_dir().join(format!("plakat-webgallery-{}", std::process::id()));
        let src = base.join("src");
        let dst = base.join("out");
        std::fs::create_dir_all(&src).unwrap();
        let photos: Vec<Photo> = (0..3)
            .map(|i| {
                let p = src.join(format!("shot{i}.png"));
                DynamicImage::ImageRgb8(ImageBuffer::from_pixel(300, 200, Rgb([i * 60, 20, 20])))
                    .save(&p)
                    .unwrap();
                Photo {
                    path: p,
                    title: Some(format!("Shot {i}")),
                    rating: (i as u8 % 6),
                    tags: vec!["trip".into(), "test".into()],
                    date: Some("2024-07-14".into()),
                    exif: Some("Canon EOS R5 · 50mm · f/2.8".into()),
                }
            })
            .collect();

        let opts = Options { title: "My <Trip>", thumb_px: 120, full_px: Some(150) };
        let n = export(&photos, &dst, &opts).unwrap();
        assert_eq!(n, 3);

        // Structure: an index, three thumbs, three full images.
        assert!(dst.join("index.html").exists());
        assert!(dst.join("thumbs/0000.jpg").exists());
        assert!(dst.join("full/0002.png").exists());

        // The full image was down-sized to ≤150.
        let full = image::open(dst.join("full/0000.png")).unwrap();
        assert!(full.width().max(full.height()) <= 150, "full sized to {}x{}", full.width(), full.height());

        // The HTML is self-contained (no external URLs) and escapes the title.
        let html = std::fs::read_to_string(dst.join("index.html")).unwrap();
        assert!(html.contains("My &lt;Trip&gt;"), "title escaped");
        assert!(!html.contains("http://") && !html.contains("https://"), "no network refs");
        assert!(html.contains("thumbs/0000.jpg") && html.contains("full/0000.png"));
        assert_eq!(html.matches("class=\"cell\"").count(), 3);

        // Metadata is embedded: caption, rating, EXIF summary, tags.
        assert!(html.contains("Shot 2"), "caption present");
        assert!(html.contains("data-rating=\"2\""), "rating embedded");
        assert!(html.contains("Canon EOS R5 · 50mm · f/2.8"), "EXIF summary embedded");
        assert!(html.contains("data-tags=\"trip, test\""), "tags embedded");
        assert!(html.contains("class=\"stars\""), "grid star overlay for rated images");

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn empty_selection_still_writes_a_page() {
        let base = std::env::temp_dir().join(format!("plakat-webgallery-empty-{}", std::process::id()));
        let n = export(&[], &base, &Options::default()).unwrap();
        assert_eq!(n, 0);
        let html = std::fs::read_to_string(base.join("index.html")).unwrap();
        assert!(html.contains("No images."));
        let _ = std::fs::remove_dir_all(&base);
    }
}
