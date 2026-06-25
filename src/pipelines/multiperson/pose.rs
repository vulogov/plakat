//! Synthetic OpenPose skeletons for multiperson composition.
//!
//! The binding problem: a plain text-to-image model won't reliably put "person A
//! on the left, person B on the right" where we asked, so face-swap/inpaint lands
//! the wrong identity on the wrong figure. The fix is an **OpenPose ControlNet**:
//! we draw one stick-figure skeleton per persona, positioned in that persona's
//! placement region, and feed the rendered skeleton map as the control hint. The
//! model then puts a figure exactly where each skeleton is → persona↔figure
//! binding holds.
//!
//! We render the lllyasviel-convention coloured skeleton directly from keypoints
//! (no detector), reusing `openpose_post`'s `LIMB_SEQ` + `LIMB_COLORS`.

use image::RgbImage;

use crate::pipelines::openpose_post::{LIMB_COLORS, LIMB_SEQ, NUM_KEYPOINTS};

use super::placement::Facing;

/// Canonical upper-body skeleton in a unit box (x→[0,1] left→right, y→[0,1]
/// top→bottom), front-facing. Legs (knees/ankles) are omitted (`NaN`) so the
/// pose pins head+torso position without forcing standing vs seated — the model
/// fills the lower body from the scene context (e.g. seated at a table). Index
/// order matches `openpose_post` (0 nose, 1 neck, 2/5 shoulders, 3/6 elbows,
/// 4/7 wrists, 8/11 hips, 9/10/12/13 legs, 14/15 eyes, 16/17 ears).
const NAN: f32 = f32::NAN;
const SKELETON: [[f32; 2]; NUM_KEYPOINTS] = [
    [0.50, 0.12], // 0 nose
    [0.50, 0.26], // 1 neck
    [0.37, 0.29], // 2 R shoulder
    [0.32, 0.46], // 3 R elbow
    [0.43, 0.58], // 4 R wrist (toward centre/table)
    [0.63, 0.29], // 5 L shoulder
    [0.68, 0.46], // 6 L elbow
    [0.57, 0.58], // 7 L wrist
    [0.43, 0.63], // 8 R hip
    [NAN, NAN],   // 9 R knee
    [NAN, NAN],   // 10 R ankle
    [0.57, 0.63], // 11 L hip
    [NAN, NAN],   // 12 L knee
    [NAN, NAN],   // 13 L ankle
    [0.46, 0.10], // 14 R eye
    [0.54, 0.10], // 15 L eye
    [0.41, 0.12], // 16 R ear
    [0.59, 0.12], // 17 L ear
];

/// Keypoints for one persona, placed into `bbox` (normalised `[x0,y0,x1,y1]`) at
/// pixel scale, adjusted for `facing` and `scale` (figure height; a child < 1.0
/// occupies the lower part of the region so they render shorter, with a slightly
/// larger head). `NaN` marks an absent keypoint.
fn place(
    bbox: &[f32; 4],
    facing: Facing,
    scale: f32,
    width: u32,
    height: u32,
) -> [[f32; 2]; NUM_KEYPOINTS] {
    let (w, h) = (width as f32, height as f32);
    let mut kp = SKELETON;

    // Shorter figures occupy the BOTTOM `scale` fraction of the region (feet stay
    // put, head drops); their head is proportionally a touch larger.
    if scale < 0.999 {
        let head_grow = 1.0 + (1.0 - scale) * 0.4;
        for (i, p) in kp.iter_mut().enumerate() {
            if p[1].is_finite() {
                p[1] = 1.0 - (1.0 - p[1]) * scale; // compress toward the bottom
                if matches!(i, 0 | 14 | 15 | 16 | 17) {
                    // enlarge the head cluster around the nose
                    p[0] = 0.5 + (p[0] - 0.5) * head_grow;
                }
            }
        }
    }

    match facing {
        Facing::Front => {}
        Facing::Side => {
            // Narrow toward a 3/4 profile and drop the far-side eye/ear.
            for p in kp.iter_mut() {
                if p[0].is_finite() {
                    p[0] = 0.5 + (p[0] - 0.5) * 0.5;
                }
            }
            kp[15] = [NAN, NAN]; // L eye
            kp[17] = [NAN, NAN]; // L ear
        }
        Facing::Back => {
            // Facing away: no face keypoints, just head/shoulders from behind.
            for i in [0usize, 14, 15, 16, 17] {
                kp[i] = [NAN, NAN];
            }
        }
    }

    let mut out = [[NAN, NAN]; NUM_KEYPOINTS];
    for (i, p) in kp.iter().enumerate() {
        if p[0].is_finite() && p[1].is_finite() {
            out[i] = [
                (bbox[0] + p[0] * (bbox[2] - bbox[0])) * w,
                (bbox[1] + p[1] * (bbox[3] - bbox[1])) * h,
            ];
        }
    }
    out
}

/// Render one OpenPose skeleton map for the given `(bbox, facing)` regions onto a
/// black `width × height` RGB image — the ControlNet-OpenPose conditioning.
pub fn render_pose_map(
    regions: &[([f32; 4], Facing, f32)],
    width: u32,
    height: u32,
) -> RgbImage {
    let mut img = RgbImage::new(width, height);
    for (bbox, facing, scale) in regions {
        let kp = place(bbox, *facing, *scale, width, height);
        // Limbs (coloured sticks).
        for (li, [a, b]) in LIMB_SEQ.iter().enumerate() {
            let (pa, pb) = (kp[*a], kp[*b]);
            if pa[0].is_finite() && pb[0].is_finite() {
                draw_thick_line(&mut img, pa, pb, LIMB_COLORS[li], 4);
            }
        }
        // Joints (coloured discs) — colour by the limb that ends there, else white.
        for (i, p) in kp.iter().enumerate() {
            if p[0].is_finite() {
                let c = LIMB_SEQ
                    .iter()
                    .position(|[_, b]| *b == i)
                    .map(|li| LIMB_COLORS[li])
                    .unwrap_or([255, 255, 255]);
                draw_disc(&mut img, *p, 4, c);
            }
        }
    }
    img
}

fn draw_thick_line(img: &mut RgbImage, a: [f32; 2], b: [f32; 2], color: [u8; 3], thick: i32) {
    let steps = ((a[0] - b[0]).hypot(a[1] - b[1]).ceil() as i32).max(1);
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let x = a[0] + (b[0] - a[0]) * t;
        let y = a[1] + (b[1] - a[1]) * t;
        draw_disc(img, [x, y], thick / 2, color);
    }
}

fn draw_disc(img: &mut RgbImage, c: [f32; 2], r: i32, color: [u8; 3]) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let (cx, cy) = (c[0].round() as i32, c[1].round() as i32);
    let r = r.max(1);
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let (x, y) = (cx + dx, cy + dy);
                if x >= 0 && x < w && y >= 0 && y < h {
                    img.put_pixel(x as u32, y as u32, image::Rgb(color));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pose_map_draws_into_each_region() {
        let regions = [
            ([0.0, 0.2, 0.33, 1.0], Facing::Front, 1.0),
            ([0.66, 0.2, 1.0, 1.0], Facing::Front, 1.0),
        ];
        let img = render_pose_map(&regions, 256, 192);
        // Non-black pixels exist in the left third and the right third.
        let nonblack = |x0: u32, x1: u32| {
            (x0..x1).any(|x| (0..192).any(|y| img.get_pixel(x, y).0 != [0, 0, 0]))
        };
        assert!(nonblack(0, 85), "left skeleton drawn");
        assert!(nonblack(170, 256), "right skeleton drawn");
        // Centre is empty (no persona there).
        assert!(!nonblack(110, 150), "centre is empty");
    }

    #[test]
    fn child_scale_skeleton_is_shorter() {
        let bbox = [0.3, 0.0, 0.7, 1.0];
        let top_y = |scale: f32| -> u32 {
            let img = render_pose_map(&[(bbox, Facing::Front, scale)], 128, 256);
            (0..256)
                .find(|&y| (0..128).any(|x| img.get_pixel(x, y).0 != [0, 0, 0]))
                .unwrap()
        };
        // A 0.6-scale figure's topmost drawn pixel sits lower than an adult's.
        assert!(top_y(0.6) > top_y(1.0) + 20, "child skeleton starts lower");
    }
}
