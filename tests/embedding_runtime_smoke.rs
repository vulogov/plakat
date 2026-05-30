//! v0.30 phase 0: integration test for the Textual Inversion runtime
//! injection seam.
//!
//! The full end-to-end path (download SD weights, run inference) is
//! too heavy for the test suite, so we instead synthesize a tiny
//! CLIP-L-shaped safetensors (1-layer encoder, embed_dim=8, vocab=4)
//! and verify the seam between:
//!
//! 1. `pipelines::embedding::merge_embeddings_into_te_weights` —
//!    appends TI rows to `text_model.embeddings.token_embedding.weight`
//!    and writes a new safetensors.
//! 2. `pipelines::vendored_clip::build_clip_transformer` with a
//!    `Config::with_vocab(new_vocab_size)` override — loads the
//!    extended safetensors and exposes a working forward pass on
//!    inputs that reference the new vocab IDs.
//!
//! This pins the contract: the merger's output format and the
//! vendored CLIP loader stay in lockstep.

use candle_core::{DType, Device, Tensor};
use plakat::pipelines::embedding::{
    EmbeddingRegistration, ResolvedEmbedding, merge_embeddings_into_te_weights,
};
use plakat::pipelines::vendored_clip::{Config, build_clip_transformer};
use std::collections::HashMap;

/// Build a tiny synthetic CLIP-L safetensors with every tensor key
/// the vendored loader expects. 1 encoder layer, embed_dim=8,
/// vocab=4, max_pos=4. Deterministic content so test failures are
/// reproducible.
fn write_tiny_clip_safetensors(path: &std::path::Path, vocab_size: usize) {
    let dev = Device::Cpu;
    let mut t = HashMap::<String, Tensor>::new();
    let embed_dim = 8usize;
    let intermediate = 12usize;
    let max_pos = 4usize;

    // Embeddings.
    t.insert(
        "text_model.embeddings.token_embedding.weight".into(),
        Tensor::ones((vocab_size, embed_dim), DType::F32, &dev).unwrap(),
    );
    t.insert(
        "text_model.embeddings.position_embedding.weight".into(),
        Tensor::ones((max_pos, embed_dim), DType::F32, &dev).unwrap(),
    );

    // One encoder layer.
    for proj in ["k_proj", "v_proj", "q_proj", "out_proj"] {
        t.insert(
            format!("text_model.encoder.layers.0.self_attn.{proj}.weight"),
            Tensor::ones((embed_dim, embed_dim), DType::F32, &dev).unwrap(),
        );
        t.insert(
            format!("text_model.encoder.layers.0.self_attn.{proj}.bias"),
            Tensor::zeros((embed_dim,), DType::F32, &dev).unwrap(),
        );
    }
    for ln in ["layer_norm1", "layer_norm2"] {
        t.insert(
            format!("text_model.encoder.layers.0.{ln}.weight"),
            Tensor::ones((embed_dim,), DType::F32, &dev).unwrap(),
        );
        t.insert(
            format!("text_model.encoder.layers.0.{ln}.bias"),
            Tensor::zeros((embed_dim,), DType::F32, &dev).unwrap(),
        );
    }
    // MLP.
    t.insert(
        "text_model.encoder.layers.0.mlp.fc1.weight".into(),
        Tensor::ones((intermediate, embed_dim), DType::F32, &dev).unwrap(),
    );
    t.insert(
        "text_model.encoder.layers.0.mlp.fc1.bias".into(),
        Tensor::zeros((intermediate,), DType::F32, &dev).unwrap(),
    );
    t.insert(
        "text_model.encoder.layers.0.mlp.fc2.weight".into(),
        Tensor::ones((embed_dim, intermediate), DType::F32, &dev).unwrap(),
    );
    t.insert(
        "text_model.encoder.layers.0.mlp.fc2.bias".into(),
        Tensor::zeros((embed_dim,), DType::F32, &dev).unwrap(),
    );

    // Final layer norm.
    t.insert(
        "text_model.final_layer_norm.weight".into(),
        Tensor::ones((embed_dim,), DType::F32, &dev).unwrap(),
    );
    t.insert(
        "text_model.final_layer_norm.bias".into(),
        Tensor::zeros((embed_dim,), DType::F32, &dev).unwrap(),
    );

    candle_core::safetensors::save(&t, path).unwrap();
}

/// Tiny Config matching the synthetic safetensors above.
fn tiny_config(vocab_size: usize) -> Config {
    Config::v1_5()
        // Override the constants to match the synthetic tensors.
        .with_vocab(vocab_size)
        // Need to rebuild manually since we want non-default dims.
        // The `with_vocab` only touches vocab; we patch the rest by
        // hand because the helper preserves SD 1.5 dims otherwise.
}

#[test]
fn no_embedding_vendored_clip_loads_synthetic_safetensors() {
    // Sanity: the vendored CLIP can load a safetensors that uses
    // exact diffusers-style key naming. This is the baseline that
    // the embedding path extends.
    let tmp = tempfile::NamedTempFile::new().unwrap();
    write_tiny_clip_safetensors(tmp.path(), 4);

    let mut cfg = tiny_config(4);
    // Override the SD 1.5 defaults to match the synthetic tensors.
    cfg.embed_dim = 8;
    cfg.intermediate_size = 12;
    cfg.max_position_embeddings = 4;
    cfg.num_hidden_layers = 1;
    cfg.num_attention_heads = 2;

    let dev = Device::Cpu;
    let enc = build_clip_transformer(&cfg, tmp.path(), &dev, DType::F32).unwrap();

    // Forward pass on valid token IDs (0..4) succeeds.
    let ids = Tensor::new(&[[0u32, 1, 2, 3]], &dev).unwrap();
    let out = candle_nn::Module::forward(&enc, &ids).unwrap();
    let dims = out.dims();
    assert_eq!(dims, &[1, 4, 8]);
}

#[test]
fn merged_ti_safetensors_loads_with_extended_vocab() {
    // The full TI runtime seam: synthesize a base safetensors, merge a
    // TI on top, load with the vendored CLIP at the extended vocab.
    let base = tempfile::NamedTempFile::new().unwrap();
    write_tiny_clip_safetensors(base.path(), 4);

    let dev = Device::Cpu;

    // Synthetic TI: 2 vectors of embed_dim=8, scaled by 0.5.
    let ti_vecs = Tensor::full(0.25f32, (2, 8), &dev).unwrap();
    let ti = ResolvedEmbedding {
        trigger: "tiny-style".into(),
        vectors: ti_vecs,
        scale: 0.5,
    };

    let merged = tempfile::NamedTempFile::new().unwrap();
    let report = merge_embeddings_into_te_weights(
        base.path(),
        merged.path(),
        std::slice::from_ref(&ti),
        8, // expected_embed_dim
        &dev,
    )
    .unwrap();

    assert_eq!(report.new_vocab_size, 6); // 4 base + 2 TI rows
    assert_eq!(report.registered.len(), 1);
    let reg: &EmbeddingRegistration = &report.registered[0];
    assert_eq!(reg.trigger, "tiny-style");
    assert_eq!(reg.base_token_id, 4);
    assert_eq!(reg.num_tokens, 2);

    // Build CLIP with the extended-vocab Config. This is the key
    // path that v0.16 phase 9 couldn't take.
    let mut cfg = Config::v1_5().with_vocab(report.new_vocab_size);
    cfg.embed_dim = 8;
    cfg.intermediate_size = 12;
    cfg.max_position_embeddings = 4;
    cfg.num_hidden_layers = 1;
    cfg.num_attention_heads = 2;

    let enc = build_clip_transformer(&cfg, merged.path(), &dev, DType::F32).unwrap();

    // Forward pass that REFERENCES the new TI token IDs (4, 5). If
    // the merge truncated or the loader silently dropped the extra
    // rows, this would panic with an out-of-range embedding lookup.
    let ids = Tensor::new(&[[0u32, 4, 5, 1]], &dev).unwrap();
    let out = candle_nn::Module::forward(&enc, &ids).unwrap();
    let dims = out.dims();
    assert_eq!(dims, &[1, 4, 8]);
}

#[test]
fn multi_embedding_merge_chains_correctly() {
    // Two embeddings stack: each appends its rows, registered with
    // sequentially advancing base_token_ids.
    let base = tempfile::NamedTempFile::new().unwrap();
    write_tiny_clip_safetensors(base.path(), 4);

    let dev = Device::Cpu;
    let ti_a = ResolvedEmbedding {
        trigger: "alpha".into(),
        vectors: Tensor::ones((1, 8), DType::F32, &dev).unwrap(),
        scale: 1.0,
    };
    let ti_b = ResolvedEmbedding {
        trigger: "beta".into(),
        vectors: Tensor::ones((3, 8), DType::F32, &dev).unwrap(),
        scale: 1.0,
    };

    let merged = tempfile::NamedTempFile::new().unwrap();
    let report = merge_embeddings_into_te_weights(
        base.path(),
        merged.path(),
        &[ti_a, ti_b],
        8,
        &dev,
    )
    .unwrap();

    assert_eq!(report.new_vocab_size, 8); // 4 + 1 + 3
    assert_eq!(report.registered[0].base_token_id, 4);
    assert_eq!(report.registered[0].num_tokens, 1);
    assert_eq!(report.registered[1].base_token_id, 5);
    assert_eq!(report.registered[1].num_tokens, 3);

    // The CLIP loader accepts the doubly-extended file.
    let mut cfg = Config::v1_5().with_vocab(report.new_vocab_size);
    cfg.embed_dim = 8;
    cfg.intermediate_size = 12;
    cfg.max_position_embeddings = 4;
    cfg.num_hidden_layers = 1;
    cfg.num_attention_heads = 2;

    let _ = build_clip_transformer(&cfg, merged.path(), &dev, DType::F32).unwrap();
}
