//! Page compositor (RFC COMIC-1 §4) — composite panel images into the bordered page canvas in reading
//! order + the `panels.json` sidecar. Weight-free: with **supplied** panel images this produces a finished
//! page with no GPU (P3 adds generated scene art on the same seam).

use super::layout::{PanelRect, Plan};
use image::{DynamicImage, GenericImageView, Rgb, RgbImage};

/// Cover-crop-fit `src` into a `w×h` cell (scale to fill, centre-crop the overflow).
fn cover_fit(src: &DynamicImage, w: u32, h: u32) -> RgbImage {
    let (sw, sh) = src.dimensions();
    if sw == 0 || sh == 0 || w == 0 || h == 0 {
        return RgbImage::from_pixel(w.max(1), h.max(1), Rgb([200, 200, 200]));
    }
    let scale = (w as f32 / sw as f32).max(h as f32 / sh as f32);
    let (rw, rh) = ((sw as f32 * scale).ceil() as u32, (sh as f32 * scale).ceil() as u32);
    let resized = image::imageops::resize(&src.to_rgb8(), rw.max(w), rh.max(h), image::imageops::FilterType::Lanczos3);
    let (ox, oy) = ((resized.width().saturating_sub(w)) / 2, (resized.height().saturating_sub(h)) / 2);
    image::imageops::crop_imm(&resized, ox, oy, w, h).to_image()
}

fn draw_border(page: &mut RgbImage, r: &PanelRect, bw: u32) {
    if bw == 0 {
        return;
    }
    let (x0, y0, x1, y1) = (r.x, r.y, r.x + r.w, r.y + r.h);
    for t in 0..bw {
        for x in x0..x1.min(page.width()) {
            if y0 + t < page.height() {
                page.put_pixel(x, y0 + t, Rgb([20, 20, 20]));
            }
            if y1 >= t + 1 && y1 - t - 1 < page.height() {
                page.put_pixel(x, y1 - t - 1, Rgb([20, 20, 20]));
            }
        }
        for y in y0..y1.min(page.height()) {
            if x0 + t < page.width() {
                page.put_pixel(x0 + t, y, Rgb([20, 20, 20]));
            }
            if x1 >= t + 1 && x1 - t - 1 < page.width() {
                page.put_pixel(x1 - t - 1, y, Rgb([20, 20, 20]));
            }
        }
    }
}

/// Composite `panel_images[i]` (aligned with `plan.panels`, `None` = empty placeholder) into the page.
pub fn compose(plan: &Plan, panel_images: &[Option<DynamicImage>]) -> RgbImage {
    let mut page = RgbImage::from_pixel(plan.w, plan.h, Rgb([plan.bg.0, plan.bg.1, plan.bg.2]));
    for r in &plan.panels {
        // fill the panel: supplied image (cover-fit) or a light placeholder.
        let cell = match panel_images.get(r.panel).and_then(|o| o.as_ref()) {
            Some(img) => cover_fit(img, r.w, r.h),
            None => RgbImage::from_pixel(r.w.max(1), r.h.max(1), Rgb([230, 230, 232])),
        };
        for (px, py, p) in cell.enumerate_pixels() {
            let (x, y) = (r.x + px, r.y + py);
            if x < page.width() && y < page.height() {
                page.put_pixel(x, y, *p);
            }
        }
        draw_border(&mut page, r, plan.border);
    }
    page
}

/// Letter a composited `page` in place: for every panel, place + draw its caption/balloons within the
/// panel's page rect (interior only — inset by the border so ink never lands on the panel frame). The
/// balloon algorithm is [`super::balloon`]; face-aware exclusion masks arrive in P3. Returns how many
/// dialogue lines were placed vs requested.
pub fn letter(page: &mut RgbImage, plan: &Plan, spec: &super::ComicSpec) -> (usize, usize) {
    let (mut placed, mut requested) = (0usize, 0usize);
    let bw = plan.border as f32;
    for r in &plan.panels {
        let Some(panel) = spec.panels.get(r.panel) else { continue };
        let lines = super::balloon::lines_for_panel(panel);
        requested += lines.len();
        // inset the drawable area by the border so balloons sit inside the frame.
        let (iw, ih) = ((r.w as f32 - 2.0 * bw).max(1.0), (r.h as f32 - 2.0 * bw).max(1.0));
        let laid = super::balloon::place(iw, ih, None, &lines);
        placed += laid.len();
        let (ox, oy) = ((r.x as f32 + bw) as i32, (r.y as f32 + bw) as i32);
        for b in &laid {
            super::balloon::draw(page, b, ox, oy);
        }
    }
    (placed, requested)
}

/// The `panels.json` UV sidecar: each panel's page rect + reading index (an engine / DCC / re-lettering
/// pass maps to panels by this).
pub fn panels_json(plan: &Plan) -> String {
    serde_json::to_string_pretty(&serde_json::json!({
        "page": { "w": plan.w, "h": plan.h, "dpi": plan.dpi },
        "reading": plan.reading,
        "gutter": plan.gutter,
        "border": plan.border,
        "panels": plan.panels,
    }))
    .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comic::spec::ComicSpec;

    #[test]
    fn composes_a_page_from_supplied_panels() {
        let spec = ComicSpec::from_hjson(r#"{ page: { size: "square", dpi: 100, gutter: 8, border: 3 }, layout: { rows: [[1,1]] }, panels: [{},{}] }"#).unwrap();
        let plan = crate::comic::layout::resolve(&spec);
        // two distinct solid-colour panels.
        let red = DynamicImage::ImageRgb8(RgbImage::from_pixel(50, 50, Rgb([200, 30, 30])));
        let blue = DynamicImage::ImageRgb8(RgbImage::from_pixel(50, 50, Rgb([30, 30, 200])));
        let page = compose(&plan, &[Some(red), Some(blue)]);
        assert_eq!(page.dimensions(), (plan.w, plan.h));
        // sample inside each panel → the right colour landed there.
        let p0 = &plan.panels[0];
        let c0 = page.get_pixel(p0.x + p0.w / 2, p0.y + p0.h / 2).0;
        assert!(c0[0] > 150 && c0[2] < 80, "panel 0 is reddish: {c0:?}");
        // sidecar has 2 panels + page dims.
        let js: serde_json::Value = serde_json::from_str(&panels_json(&plan)).unwrap();
        assert_eq!(js["panels"].as_array().unwrap().len(), 2);
        assert_eq!(js["page"]["w"], plan.w);
    }
}
