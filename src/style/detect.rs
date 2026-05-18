//! Cosine-match a query embedding against a [`StyleCatalog`].

use anyhow::{bail, Result};
use candle_core::Tensor;

use super::catalog::{Aggregation, DetectionResult, StyleCatalog, StyleMatch};

/// Detect the closest style to `image_embedding` in `catalog`.
///
/// `image_embedding`: shape `(embed_dim,)`, f32, **L2-normalized**.
/// Produced by [`super::encode::encode_reference_photo`].
///
/// `top_k`: how many ranked matches to return for display/debug. The
/// `picked` / `ambiguous` decision applies the catalog's
/// `min_confidence` + `margin_over_runner_up` policy and is
/// independent of K.
pub fn detect_style(
    catalog: &StyleCatalog,
    image_embedding: &Tensor,
    top_k: usize,
) -> Result<DetectionResult> {
    let d = image_embedding.dims1()?;
    if d != catalog.embed_dim {
        bail!(
            "embedding dim {} doesn't match catalog embed_dim {}",
            d,
            catalog.embed_dim
        );
    }

    // (1, D) query; we transpose each exemplar matrix to (D, N) and matmul.
    // Both sides are unit-normalized, so cosine = dot product.
    let q = image_embedding.unsqueeze(0)?;

    let mut scored: Vec<StyleMatch> = Vec::with_capacity(catalog.order.len());

    for style_id in &catalog.order {
        let style = &catalog.styles[style_id];

        let sims = q.matmul(&style.exemplars.t()?)?.squeeze(0)?;
        let score = aggregate(&sims, catalog.policy.aggregation)?;

        scored.push(StyleMatch {
            style_id: style.id.clone(),
            display_name: style.display_name.clone(),
            score,
        });
    }

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let policy = &catalog.policy;
    let top1 = scored.first().map(|m| m.score);
    let top2 = scored.get(1).map(|m| m.score);

    let picked = match top1 {
        Some(s) if s >= policy.min_confidence => Some(scored[0].style_id.clone()),
        _ => None,
    };

    let ambiguous = matches!(
        (top1, top2),
        (Some(a), Some(b)) if (a - b) < policy.margin_over_runner_up
    );

    scored.truncate(top_k);

    Ok(DetectionResult {
        top: scored,
        picked,
        ambiguous,
    })
}

/// Collapse per-exemplar similarities into a single per-style score.
///
/// With N=1 every policy returns the single value, so single-exemplar
/// styles aren't a special case.
fn aggregate(sims: &Tensor, policy: Aggregation) -> Result<f32> {
    let v: Vec<f32> = sims.to_vec1()?;
    let n = v.len();

    Ok(match policy {
        Aggregation::Max => v.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        Aggregation::Mean => v.iter().sum::<f32>() / n as f32,
        Aggregation::Top3Mean => {
            let mut s = v;
            s.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let k = s.len().min(3);
            s[..k].iter().sum::<f32>() / k as f32
        }
    })
}
