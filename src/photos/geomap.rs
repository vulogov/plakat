//! Geo map view: plot geotagged photos on a world map and browse by location. Fully **offline** —
//! the coastline backdrop is ratatui's built-in vector world map (no tiles, no network); photos are
//! plotted from their `exif.gps_lat/lon`. Pan/zoom adjusts the view window; a grid-bin clusters dense
//! areas into a count so a shoot doesn't overplot. (Reverse-geocoding place *names* is the one part
//! that needs a downloaded+cached gazetteer — kept separate.)

/// The visible map window: a centre (lon, lat) and a longitudinal half-span in degrees (the zoom).
#[derive(Clone, Copy)]
pub struct GeoView {
    pub lon_c: f64,
    pub lat_c: f64,
    pub span: f64,
}

impl GeoView {
    /// The whole world (centred a little north so most land shows).
    pub fn world() -> Self {
        GeoView { lon_c: 0.0, lat_c: 20.0, span: 180.0 }
    }

    /// Fit the window to `points` (lon, lat) with a margin; falls back to the world if empty.
    pub fn fit(points: &[(f64, f64)]) -> Self {
        if points.is_empty() {
            return Self::world();
        }
        let (mut lo_x, mut hi_x, mut lo_y, mut hi_y) = (180.0f64, -180.0f64, 90.0f64, -90.0f64);
        for &(x, y) in points {
            lo_x = lo_x.min(x);
            hi_x = hi_x.max(x);
            lo_y = lo_y.min(y);
            hi_y = hi_y.max(y);
        }
        let lon_c = (lo_x + hi_x) / 2.0;
        let lat_c = (lo_y + hi_y) / 2.0;
        // Half-span covering the spread (+40 % margin), clamped to sane zoom limits.
        let span = (((hi_x - lo_x).max(hi_y - lo_y) / 2.0) * 1.4).clamp(2.0, 180.0);
        GeoView { lon_c, lat_c, span }
    }

    /// `[lon_min, lon_max]` for the Canvas x-axis.
    pub fn x_bounds(&self) -> [f64; 2] {
        [self.lon_c - self.span, self.lon_c + self.span]
    }

    /// `[lat_min, lat_max]` for the Canvas y-axis; `aspect` = pane height/width in cells, so the map
    /// isn't stretched.
    pub fn y_bounds(&self, aspect: f64) -> [f64; 2] {
        let vspan = (self.span * aspect).clamp(1.0, 90.0);
        [(self.lat_c - vspan).max(-90.0), (self.lat_c + vspan).min(90.0)]
    }

    pub fn pan(&mut self, dlon: f64, dlat: f64) {
        self.lon_c = (self.lon_c + dlon * self.span).clamp(-180.0, 180.0);
        self.lat_c = (self.lat_c + dlat * self.span).clamp(-85.0, 85.0);
    }

    pub fn zoom(&mut self, factor: f64) {
        self.span = (self.span * factor).clamp(1.0, 180.0);
    }
}

/// Indices of `items` within `radius` degrees of `(lon, lat)` — for selecting a cluster.
pub fn near(items: &[(f64, f64)], lon: f64, lat: f64, radius: f64) -> Vec<usize> {
    items
        .iter()
        .enumerate()
        .filter(|(_, coord)| {
            let (x, y) = **coord;
            (x - lon).powi(2) + (y - lat).powi(2) <= radius * radius
        })
        .map(|(i, _)| i)
        .collect()
}

/// Grid-bin `points` into `cols`×`rows` cells over the view bounds; returns `(lon, lat, count)` at
/// each non-empty cell's centroid — so dense shoots read as one labelled marker, not a blob.
pub fn cluster(points: &[(f64, f64)], view: &GeoView, aspect: f64, cols: usize, rows: usize) -> Vec<(f64, f64, usize)> {
    let [x0, x1] = view.x_bounds();
    let [y0, y1] = view.y_bounds(aspect);
    let (cols, rows) = (cols.max(1), rows.max(1));
    let mut acc = vec![(0.0f64, 0.0f64, 0usize); cols * rows];
    for &(x, y) in points {
        if x < x0 || x > x1 || y < y0 || y > y1 {
            continue;
        }
        let cx = (((x - x0) / (x1 - x0) * cols as f64) as usize).min(cols - 1);
        let cy = (((y - y0) / (y1 - y0) * rows as f64) as usize).min(rows - 1);
        let cell = &mut acc[cy * cols + cx];
        cell.0 += x;
        cell.1 += y;
        cell.2 += 1;
    }
    acc.into_iter()
        .filter(|&(_, _, n)| n > 0)
        .map(|(sx, sy, n)| (sx / n as f64, sy / n as f64, n))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_centres_and_spans_the_points() {
        let v = GeoView::fit(&[(-122.4, 37.7), (139.7, 35.7)]);
        assert!((v.lon_c - 8.65).abs() < 1.0, "centre lon between SF and Tokyo");
        assert!(v.span > 90.0, "spans the Pacific");
        // A single point still yields a valid, tight-ish window.
        let one = GeoView::fit(&[(0.0, 51.5)]);
        assert_eq!((one.lon_c, one.lat_c), (0.0, 51.5));
        assert!(one.span >= 2.0);
    }

    #[test]
    fn cluster_bins_and_near_selects() {
        let pts = vec![(0.0, 0.0), (0.1, 0.1), (50.0, 40.0)];
        let v = GeoView::world();
        let cells = cluster(&pts, &v, 0.5, 36, 18);
        // The two near-(0,0) points land in one cell (count 2); the far one in another.
        assert!(cells.iter().any(|&(_, _, n)| n == 2));
        assert_eq!(cells.iter().map(|&(_, _, n)| n).sum::<usize>(), 3);
        assert_eq!(near(&pts, 0.05, 0.05, 1.0), vec![0, 1]);
    }
}
