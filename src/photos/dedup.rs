//! Near-duplicate detection (RFC PHOTOS-1 Phase 5) via a 64-bit perceptual hash (dHash).
//!
//! dHash resizes to 9×8 grayscale and records, per row, whether each pixel is brighter than its right
//! neighbour — 64 bits robust to scaling, mild colour/exposure shifts, and re-compression. Two images
//! are "near-duplicate" when their hashes differ in ≤ `threshold` bits (Hamming distance). Grouping is
//! a simple greedy transitive clustering, fine for album-sized sets.

use std::path::PathBuf;

use image::DynamicImage;

/// 64-bit difference hash of `img` (see module docs). Pure function of the pixels.
pub fn dhash(img: &DynamicImage) -> u64 {
    let small = img.resize_exact(9, 8, image::imageops::FilterType::Triangle).to_luma8();
    let mut hash = 0u64;
    let mut bit = 0u32;
    for y in 0..8u32 {
        for x in 0..8u32 {
            let left = small.get_pixel(x, y)[0];
            let right = small.get_pixel(x + 1, y)[0];
            if left < right {
                hash |= 1 << bit;
            }
            bit += 1;
        }
    }
    hash
}

/// Hamming distance (differing bits) between two hashes.
pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Group paths whose hashes are within `threshold` bits of each other (greedy transitive clustering).
/// Only groups of ≥ 2 are returned; each input path appears in at most one group.
pub fn find_duplicates(hashes: &[(PathBuf, u64)], threshold: u32) -> Vec<Vec<PathBuf>> {
    let n = hashes.len();
    let mut used = vec![false; n];
    let mut groups = Vec::new();
    for i in 0..n {
        if used[i] {
            continue;
        }
        let mut group = vec![i];
        used[i] = true;
        // Transitively absorb any not-yet-grouped image close to *any* current member.
        let mut k = 0;
        while k < group.len() {
            let a = group[k];
            for j in 0..n {
                if !used[j] && hamming(hashes[a].1, hashes[j].1) <= threshold {
                    used[j] = true;
                    group.push(j);
                }
            }
            k += 1;
        }
        if group.len() >= 2 {
            groups.push(group.into_iter().map(|idx| hashes[idx].0.clone()).collect());
        }
    }
    groups
}

/// A near-duplicate group member's quality signals, for keeper selection (6.26.0).
/// Ordered so a tuple comparison ranks by **rating**, then **sharpness** (crispest wins a
/// rating tie — the fix vs the pre-6.26 rating→aesthetic heuristic), then **aesthetic** score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeeperKey {
    pub rating: u8,
    pub sharpness: f32,
    pub aesthetic: f64,
}

/// The index of the frame to KEEP within a near-dup group: highest rating, then sharpest, then
/// best aesthetic. Ties resolve to the lowest index (order-stable). `None` for an empty slice.
pub fn pick_keeper(keys: &[KeeperKey]) -> Option<usize> {
    if keys.is_empty() {
        return None;
    }
    let mut best = 0usize;
    for i in 1..keys.len() {
        if keeper_cmp(&keys[i], &keys[best]) == std::cmp::Ordering::Greater {
            best = i;
        }
    }
    Some(best)
}

/// Order two keeper keys: rating, then sharpness, then aesthetic (all higher = better).
fn keeper_cmp(a: &KeeperKey, b: &KeeperKey) -> std::cmp::Ordering {
    a.rating
        .cmp(&b.rating)
        .then(a.sharpness.partial_cmp(&b.sharpness).unwrap_or(std::cmp::Ordering::Equal))
        .then(a.aesthetic.partial_cmp(&b.aesthetic).unwrap_or(std::cmp::Ordering::Equal))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn gradient(shift: u8) -> DynamicImage {
        DynamicImage::ImageRgb8(ImageBuffer::from_fn(64, 64, |x, _| {
            let v = ((x as u8).wrapping_add(shift)) as u8;
            Rgb([v, v, v])
        }))
    }

    #[test]
    fn identical_images_hash_equal() {
        assert_eq!(dhash(&gradient(0)), dhash(&gradient(0)));
        assert_eq!(hamming(0xABCD, 0xABCD), 0);
    }

    #[test]
    fn distinct_images_differ() {
        // A horizontal gradient vs a flat image differ a lot.
        let flat = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(64, 64, Rgb([128, 128, 128])));
        assert!(hamming(dhash(&gradient(0)), dhash(&flat)) > 8);
    }

    #[test]
    fn keeper_prefers_rating_then_sharpness_then_aesthetic() {
        let k = |rating, sharpness, aesthetic| KeeperKey { rating, sharpness, aesthetic };
        // Highest rating wins outright.
        assert_eq!(pick_keeper(&[k(3, 1.0, 9.0), k(5, 0.1, 0.0), k(4, 9.9, 9.9)]), Some(1));
        // On a rating tie, the SHARPEST wins (the 6.26 fix — index 1 is crisper).
        assert_eq!(pick_keeper(&[k(5, 10.0, 9.0), k(5, 90.0, 1.0), k(5, 50.0, 5.0)]), Some(1));
        // Rating + sharpness tie → best aesthetic.
        assert_eq!(pick_keeper(&[k(5, 20.0, 3.0), k(5, 20.0, 8.0)]), Some(1));
        // Empty → None; order-stable on a full tie (lowest index).
        assert_eq!(pick_keeper(&[]), None);
        assert_eq!(pick_keeper(&[k(5, 1.0, 1.0), k(5, 1.0, 1.0)]), Some(0));
    }

    #[test]
    fn groups_near_duplicates_only() {
        let a = dhash(&gradient(0));
        let b = a ^ 0b11; // 2 bits away → a near-dup of a
        let c = 0xFFFF_0000_FFFF_0000; // far from both
        let hashes = vec![
            (PathBuf::from("a.png"), a),
            (PathBuf::from("b.png"), b),
            (PathBuf::from("c.png"), c),
        ];
        let groups = find_duplicates(&hashes, 5);
        assert_eq!(groups.len(), 1, "one dup group");
        assert_eq!(groups[0].len(), 2);
        assert!(groups[0].contains(&PathBuf::from("a.png")));
        assert!(groups[0].contains(&PathBuf::from("b.png")));
    }
}
