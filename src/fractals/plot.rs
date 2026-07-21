//! Shared plotting geometry for the line-drawing families (IFS, L-system).
//!
//! Both compute geometry in an arbitrary model space (the attractor's own coordinates),
//! then fit it — aspect-preserving, centered, with a margin — into the pixel canvas.
//! `+y` points up in model space (screen `y` is flipped). RFC FRACTALS-1, Phase 3.

/// Axis-aligned bounding box in model space.
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl Bounds {
    pub fn empty() -> Self {
        Bounds { min_x: f64::INFINITY, min_y: f64::INFINITY, max_x: f64::NEG_INFINITY, max_y: f64::NEG_INFINITY }
    }

    pub fn include(&mut self, x: f64, y: f64) {
        if x < self.min_x { self.min_x = x; }
        if y < self.min_y { self.min_y = y; }
        if x > self.max_x { self.max_x = x; }
        if y > self.max_y { self.max_y = y; }
    }

    pub fn is_valid(&self) -> bool {
        self.min_x.is_finite() && self.max_x >= self.min_x && self.max_y >= self.min_y
    }

    pub fn width(&self) -> f64 {
        (self.max_x - self.min_x).max(1e-12)
    }

    pub fn height(&self) -> f64 {
        (self.max_y - self.min_y).max(1e-12)
    }
}

/// An aspect-preserving model→pixel transform: model space is scaled by a single factor
/// (so the drawing isn't distorted), centered, and inset by `margin`. An extra `zoom`
/// multiplier scales further about the center.
#[derive(Debug, Clone, Copy)]
pub struct Fit {
    scale: f64,
    cx: f64,
    cy: f64,
    w: f64,
    h: f64,
}

impl Fit {
    pub fn new(b: &Bounds, width: u32, height: u32, margin: f64, zoom: f64) -> Self {
        let sx = (width as f64 * margin) / b.width();
        let sy = (height as f64 * margin) / b.height();
        let scale = sx.min(sy) * zoom.max(1e-6);
        Fit {
            scale,
            cx: (b.min_x + b.max_x) * 0.5,
            cy: (b.min_y + b.max_y) * 0.5,
            w: width as f64,
            h: height as f64,
        }
    }

    /// Map a model point to a floating-point pixel coordinate (`+y` up → screen flip).
    #[inline]
    pub fn map_f(&self, x: f64, y: f64) -> (f64, f64) {
        let px = (x - self.cx) * self.scale + self.w * 0.5;
        let py = self.h * 0.5 - (y - self.cy) * self.scale;
        (px, py)
    }

    /// Map to an integer pixel inside the canvas, or `None` if outside.
    #[inline]
    pub fn map_px(&self, x: f64, y: f64) -> Option<(u32, u32)> {
        let (px, py) = self.map_f(x, y);
        let (px, py) = (px.round(), py.round());
        if px >= 0.0 && px < self.w && py >= 0.0 && py < self.h {
            Some((px as u32, py as u32))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_accumulate() {
        let mut b = Bounds::empty();
        assert!(!b.is_valid());
        b.include(-1.0, 2.0);
        b.include(3.0, -4.0);
        assert!(b.is_valid());
        assert_eq!((b.min_x, b.max_x, b.min_y, b.max_y), (-1.0, 3.0, -4.0, 2.0));
        assert_eq!(b.width(), 4.0);
        assert_eq!(b.height(), 6.0);
    }

    #[test]
    fn fit_centers_and_flips_y() {
        let mut b = Bounds::empty();
        b.include(0.0, 0.0);
        b.include(10.0, 10.0);
        let fit = Fit::new(&b, 100, 100, 1.0, 1.0);
        // Model center → canvas center.
        let (cx, cy) = fit.map_f(5.0, 5.0);
        assert!((cx - 50.0).abs() < 1e-9 && (cy - 50.0).abs() < 1e-9);
        // Top of model (max y) maps to a smaller pixel-y than the bottom (y flip).
        let (_, top) = fit.map_f(5.0, 10.0);
        let (_, bot) = fit.map_f(5.0, 0.0);
        assert!(top < bot);
        // With an inset margin (as in real use) the extreme corners land inside the canvas.
        let inset = Fit::new(&b, 100, 100, 0.9, 1.0);
        assert!(inset.map_px(0.0, 0.0).is_some());
        assert!(inset.map_px(10.0, 10.0).is_some());
    }
}
