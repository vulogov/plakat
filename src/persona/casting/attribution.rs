//! Multiperson face→figure attribution (RFC §14.2). When several personas share a frame, each
//! persona's face must swap only into *its* figure and its details composite only against *that*
//! figure's landmarks. Figure A's scar composited onto figure B is a catastrophic, very visible
//! failure — so the compositor **refuses** to place a detail when the assignment confidence is below
//! threshold, reports the refusal, and leaves the mark absent rather than wrong.
//!
//! This is the pure assignment half — detected face boxes + per-persona figure regions → a one-to-one
//! assignment with a confidence, refusing low-confidence matches. Deterministic + testable.

/// Minimum overlap confidence to attribute a face to a figure (§14.2). Below this the assignment is
/// refused (`figure = None`) and the persona's details are left absent, not mis-placed.
pub const ATTRIBUTION_CONFIDENCE_MIN: f32 = 0.35;

/// One face's attribution: which figure (persona region) it was assigned to, and the confidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Assignment {
    pub face: usize,
    /// The assigned figure index, or `None` when no figure cleared the confidence threshold.
    pub figure: Option<usize>,
    pub confidence: f32,
}

/// Intersection-over-union of two `[x0,y0,x1,y1]` boxes.
pub fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let ix0 = a[0].max(b[0]);
    let iy0 = a[1].max(b[1]);
    let ix1 = a[2].min(b[2]);
    let iy1 = a[3].min(b[3]);
    let iw = (ix1 - ix0).max(0.0);
    let ih = (iy1 - iy0).max(0.0);
    let inter = iw * ih;
    let area = |x: [f32; 4]| ((x[2] - x[0]).max(0.0)) * ((x[3] - x[1]).max(0.0));
    let uni = area(a) + area(b) - inter;
    if uni <= 1e-6 {
        0.0
    } else {
        inter / uni
    }
}

/// Fraction of `inner` contained inside `outer` — a face is usually *inside* its figure box, so
/// containment discriminates better than IoU (a small face has low IoU with a whole-body box).
pub fn containment(inner: [f32; 4], outer: [f32; 4]) -> f32 {
    let ix0 = inner[0].max(outer[0]);
    let iy0 = inner[1].max(outer[1]);
    let ix1 = inner[2].min(outer[2]);
    let iy1 = inner[3].min(outer[3]);
    let inter = (ix1 - ix0).max(0.0) * (iy1 - iy0).max(0.0);
    let inner_area = ((inner[2] - inner[0]).max(0.0)) * ((inner[3] - inner[1]).max(0.0));
    if inner_area <= 1e-6 {
        0.0
    } else {
        inter / inner_area
    }
}

/// The assignment score of a face to a figure — containment dominates (faces sit inside figures), with
/// a light IoU term to break ties toward the tightest-matching figure.
fn score(face: [f32; 4], figure: [f32; 4]) -> f32 {
    0.85 * containment(face, figure) + 0.15 * iou(face, figure)
}

/// Assign each detected `face` box to the best figure region, greedily and **one-to-one** (a figure
/// takes at most one face), refusing matches below `min_conf` (§14.2). Returns one `Assignment` per
/// face, in face order.
pub fn assign(faces: &[[f32; 4]], figures: &[[f32; 4]], min_conf: f32) -> Vec<Assignment> {
    // all (face, figure, score) triples, best first.
    let mut pairs: Vec<(usize, usize, f32)> = Vec::new();
    for (fi, &face) in faces.iter().enumerate() {
        for (gi, &fig) in figures.iter().enumerate() {
            pairs.push((fi, gi, score(face, fig)));
        }
    }
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut face_conf = vec![0.0f32; faces.len()];
    let mut face_fig: Vec<Option<usize>> = vec![None; faces.len()];
    let mut figure_taken = vec![false; figures.len()];
    let mut face_done = vec![false; faces.len()];
    for (fi, gi, sc) in pairs {
        if face_done[fi] || figure_taken[gi] {
            continue;
        }
        face_conf[fi] = sc;
        face_done[fi] = true;
        if sc >= min_conf {
            face_fig[fi] = Some(gi);
            figure_taken[gi] = true;
        }
        // below threshold: record the confidence but leave the figure free + the face unassigned.
    }
    (0..faces.len())
        .map(|fi| Assignment { face: fi, figure: face_fig[fi], confidence: face_conf[fi] })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_and_containment() {
        assert!((iou([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]) - 1.0).abs() < 1e-4);
        assert_eq!(iou([0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]), 0.0);
        // a face fully inside a figure → containment 1.
        assert!((containment([40.0, 20.0, 60.0, 40.0], [0.0, 0.0, 100.0, 200.0]) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn two_personas_get_their_own_figures() {
        // two figures side by side; each has a face near its top.
        let figures = [[0.0, 0.0, 100.0, 300.0], [100.0, 0.0, 200.0, 300.0]];
        let faces = [[30.0, 20.0, 70.0, 70.0], [130.0, 20.0, 170.0, 70.0]];
        let a = assign(&faces, &figures, ATTRIBUTION_CONFIDENCE_MIN);
        assert_eq!(a[0].figure, Some(0));
        assert_eq!(a[1].figure, Some(1));
        // one-to-one: the two faces went to different figures.
        assert_ne!(a[0].figure, a[1].figure);
    }

    #[test]
    fn low_confidence_face_is_refused_not_misattributed() {
        // a stray face overlapping nothing → refused (None), never forced onto a figure.
        let figures = [[0.0, 0.0, 100.0, 300.0]];
        let faces = [[400.0, 400.0, 440.0, 440.0]];
        let a = assign(&faces, &figures, ATTRIBUTION_CONFIDENCE_MIN);
        assert_eq!(a[0].figure, None, "a face outside every figure must not be attributed");
        assert!(a[0].confidence < ATTRIBUTION_CONFIDENCE_MIN);
    }

    #[test]
    fn a_figure_takes_at_most_one_face() {
        // two faces both inside the same single figure → only one is attributed.
        let figures = [[0.0, 0.0, 200.0, 300.0]];
        let faces = [[30.0, 20.0, 70.0, 70.0], [120.0, 20.0, 160.0, 70.0]];
        let a = assign(&faces, &figures, ATTRIBUTION_CONFIDENCE_MIN);
        let assigned = a.iter().filter(|x| x.figure == Some(0)).count();
        assert_eq!(assigned, 1, "a figure cannot own two faces");
    }
}
