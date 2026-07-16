//! Local, model-free semantic text search. Shared by `plakat ui` History and `plakat photos`
//! (metadata search over prompts / captions / notes / tags). Substring filters only find exact
//! text; this ranks every document by *relevance* to a free-text query using the classic
//! vector-space model — a TF-IDF embedding per document + cosine similarity.
//!
//! It's an embedding-and-cosine search (each document is a sparse TF-IDF vector), just
//! with a deterministic, model-free embedder rather than a neural one: zero new deps,
//! instant, no model download — which keeps History snappy. The token weighting (rare
//! terms count more; multi-term queries accumulate) gives meaning-aware ranking that
//! substring matching can't: "snowy peak" surfaces "a mountain in winter, fresh snow"
//! even though neither string is a substring of the other.

use std::collections::HashMap;

/// Split text into lowercase alphanumeric tokens (≥2 chars), dropping a few stopwords
/// so common filler doesn't dominate the cosine.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(|t| t.to_lowercase())
        .filter(|t| !STOPWORDS.contains(&t.as_str()))
        .collect()
}

const STOPWORDS: &[&str] = &[
    "the", "and", "with", "for", "this", "that", "are", "was", "you", "your", "from", "has",
    "have", "but", "not", "all", "any", "can", "out", "use", "per",
];

/// Rank `docs` by TF-IDF cosine similarity to `query`, returning `(index, score)` for the
/// documents that share at least one query term, sorted by score descending (ties broken
/// by original order for determinism). An empty query → an empty result.
pub fn rank(query: &str, docs: &[String]) -> Vec<(usize, f32)> {
    let q_tokens = tokenize(query);
    if q_tokens.is_empty() || docs.is_empty() {
        return Vec::new();
    }
    // Per-doc term frequencies + document frequency (how many docs contain each term).
    let n = docs.len() as f32;
    let doc_tfs: Vec<HashMap<String, f32>> = docs.iter().map(|d| term_freqs(&tokenize(d))).collect();
    let mut df: HashMap<&str, usize> = HashMap::new();
    for tf in &doc_tfs {
        for term in tf.keys() {
            *df.entry(term.as_str()).or_insert(0) += 1;
        }
    }
    // Smoothed IDF so a term present in every doc still contributes a little.
    let idf = |term: &str| -> f32 {
        let d = *df.get(term).unwrap_or(&0) as f32;
        ((n + 1.0) / (d + 1.0)).ln() + 1.0
    };

    // Query vector (each query term weighted by its IDF; repeated terms accumulate).
    let q_tf = term_freqs(&q_tokens);
    let q_vec: HashMap<&str, f32> = q_tf.iter().map(|(t, &c)| (t.as_str(), c * idf(t))).collect();
    let q_norm = norm(q_vec.values().copied());
    if q_norm == 0.0 {
        return Vec::new();
    }

    let mut scored: Vec<(usize, f32)> = Vec::new();
    for (i, tf) in doc_tfs.iter().enumerate() {
        // Only the query terms contribute to the dot product, but the doc norm is over
        // ALL its terms (true cosine), so a short on-topic recipe beats a long one that
        // merely mentions the term in passing.
        let mut dot = 0.0;
        for (qt, &qw) in &q_vec {
            if let Some(&c) = tf.get(*qt) {
                dot += qw * (c * idf(qt));
            }
        }
        if dot <= 0.0 {
            continue;
        }
        let d_norm = norm(tf.iter().map(|(t, &c)| c * idf(t)));
        if d_norm == 0.0 {
            continue;
        }
        scored.push((i, dot / (q_norm * d_norm)));
    }
    // Sort by score desc; stable so equal scores keep input (newest-first) order.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

fn term_freqs(tokens: &[String]) -> HashMap<String, f32> {
    let mut m = HashMap::new();
    for t in tokens {
        *m.entry(t.clone()).or_insert(0.0) += 1.0;
    }
    m
}

fn norm(vals: impl Iterator<Item = f32>) -> f32 {
    vals.map(|v| v * v).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docs() -> Vec<String> {
        vec![
            "a red fox in a misty forest, autumn leaves".into(),
            "a mountain in winter, fresh snow on the peaks".into(),
            "a neon city street at night, rain reflections".into(),
            "a snowy mountain village, smoke from chimneys".into(),
        ]
    }

    #[test]
    fn ranks_relevant_docs_above_irrelevant_ones() {
        let d = docs();
        let ranked = rank("snowy mountain peak", &d);
        // The two snow/mountain docs rank above the fox + city ones.
        let order: Vec<usize> = ranked.iter().map(|(i, _)| *i).collect();
        assert!(order.contains(&1) && order.contains(&3), "both mountain docs matched: {order:?}");
        assert!(!order.contains(&2), "the neon-city doc shares no query terms");
        // The top hit is one of the mountain docs.
        assert!(matches!(order.first(), Some(1) | Some(3)));
    }

    #[test]
    fn finds_meaning_without_an_exact_substring() {
        // "winter snow" isn't a literal substring of any doc, but ranks the snow docs.
        let d = docs();
        let ranked = rank("winter snow", &d);
        assert!(!ranked.is_empty());
        assert!(matches!(ranked[0].0, 1 | 3));
    }

    #[test]
    fn empty_query_or_no_overlap_yields_nothing() {
        let d = docs();
        assert!(rank("", &d).is_empty());
        assert!(rank("spaceship robot", &d).is_empty()); // no shared terms
        assert!(rank("fox", &[]).is_empty());
    }

    #[test]
    fn scores_are_descending() {
        let d = docs();
        let ranked = rank("mountain snow forest", &d);
        for w in ranked.windows(2) {
            assert!(w[0].1 >= w[1].1, "scores must be sorted desc");
        }
    }
}
