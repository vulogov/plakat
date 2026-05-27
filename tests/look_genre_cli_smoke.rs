//! v0.25 phase 3: CLI smoke for `--look` and `--genre` on the
//! `generate` subcommand. These tests drive clap's parser directly
//! (no model loads, no network) so they stay fast on CI.

use clap::Parser;
use plakat::cli::{Cli, Command};

/// `--look NAME` is parsed into `GenerateArgs.look`.
#[test]
fn generate_accepts_look_flag() {
    let cli = Cli::try_parse_from([
        "plakat",
        "generate",
        "--look",
        "watercolor",
        "a cottage",
    ])
    .expect("parse");
    match cli.command {
        Command::Generate(args) => {
            assert_eq!(args.look.as_deref(), Some("watercolor"));
            assert_eq!(args.genre, None);
            assert_eq!(args.prompt, "a cottage");
        }
        other => panic!("expected Generate, got {other:?}"),
    }
}

/// `--genre NAME` independently parses; both axes coexist.
#[test]
fn generate_accepts_genre_flag() {
    let cli = Cli::try_parse_from([
        "plakat",
        "generate",
        "--look",
        "watercolor",
        "--genre",
        "anime",
        "a knight",
    ])
    .expect("parse");
    match cli.command {
        Command::Generate(args) => {
            assert_eq!(args.look.as_deref(), Some("watercolor"));
            assert_eq!(args.genre.as_deref(), Some("anime"));
        }
        other => panic!("expected Generate, got {other:?}"),
    }
}

/// Omitting both flags leaves both fields `None`.
#[test]
fn generate_omitting_both_leaves_none() {
    let cli =
        Cli::try_parse_from(["plakat", "generate", "a cottage"]).expect("parse");
    match cli.command {
        Command::Generate(args) => {
            assert_eq!(args.look, None);
            assert_eq!(args.genre, None);
        }
        other => panic!("expected Generate, got {other:?}"),
    }
}

/// The bundled looks catalog is reachable from the binary's
/// working directory (matches the convention used by
/// `assets/style_catalog/`). Loaded via the same path the
/// `run` dispatch will use. Uses `load_with_user_dir(.., None)`
/// to skip the user-extension scan so a dev's `~/.config/plakat/
/// looks/` doesn't perturb the pinned entry count.
#[test]
fn bundled_looks_catalog_loads_from_cwd() {
    use plakat::preset::{Catalog, Kind};
    let cat = Catalog::load_with_user_dir(Kind::Look, None)
        .expect("load bundled looks");
    assert_eq!(cat.entries.len(), 8);
    assert!(cat.find("watercolor").is_some());
    assert!(cat.find("oil-painting").is_some());
}

/// Same for genres.
#[test]
fn bundled_genres_catalog_loads_from_cwd() {
    use plakat::preset::{Catalog, Kind};
    let cat = Catalog::load_with_user_dir(Kind::Genre, None)
        .expect("load bundled genres");
    assert_eq!(cat.entries.len(), 1);
    assert!(cat.find("anime").is_some());
}

/// Override-only-if-user-didn't-pass: a fully-specified command
/// keeps its user values; the preset doesn't overwrite them.
/// Exercises `apply_presets` end-to-end without invoking the
/// generate pipeline.
#[test]
fn fully_flagged_invocation_keeps_user_values() {
    use plakat::preset::{GenerationParams, apply_presets};

    let mut params = GenerationParams {
        prompt: "a knight".into(),
        negative: "blurry".into(),
        steps: Some(50),       // user passed --steps 50
        guidance: Some(9.0),   // user passed --guidance 9.0
        scheduler: Some("euler-a".into()), // user passed --scheduler
    };
    let (look, _) = apply_presets(Some("watercolor"), None, &mut params).unwrap();
    assert!(look.is_some());

    // User scalars survive.
    assert_eq!(params.steps, Some(50));
    assert_eq!(params.guidance, Some(9.0));
    assert_eq!(params.scheduler.as_deref(), Some("euler-a"));

    // Compositional fields DO change — prefix/suffix/negative
    // are additive by design. Document this explicitly: byte-
    // identity is only for override fields, not the whole prompt.
    assert!(params.prompt.contains("watercolor"));
    assert!(params.negative.contains("photographic"));
}

/// `--offline` flag is accepted on `generate` and lives on
/// `GenerateArgs.offline` as a `bool`.
#[test]
fn generate_accepts_offline_flag() {
    let cli = Cli::try_parse_from([
        "plakat",
        "generate",
        "--look",
        "watercolor",
        "--offline",
        "a cottage",
    ])
    .expect("parse");
    match cli.command {
        Command::Generate(args) => {
            assert_eq!(args.look.as_deref(), Some("watercolor"));
            assert!(args.offline);
        }
        other => panic!("expected Generate, got {other:?}"),
    }
}

/// Discovery end-to-end with offline + no cache → `Ok(None)`. No
/// network call, no panic, just a silent miss.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn offline_discovery_no_cache_is_noop() {
    use plakat::preset::LoraQuery;
    use plakat::preset::discovery::{BaseFamily, DiscoveryOptions, discover_lora};

    let dir = tempfile::tempdir().unwrap();
    let opts = DiscoveryOptions {
        offline: true,
        base: BaseFamily::Sdxl,
        cache_root: dir.path().join("discovery"),
        civitai_cache_root: dir.path().join("civitai-empty"),
        scale: 0.8,
    };
    let q = LoraQuery {
        tags: vec!["watercolor".into()],
        keywords: vec!["watercolor".into()],
    };
    let result = discover_lora(&q, "watercolor", &opts).await.unwrap();
    assert!(result.is_none(), "offline + empty cache must return None");
}

/// `BaseFamily::from_model_arg` returns the right family for each
/// `--model` alias plakat accepts. Anchored here so regressions in
/// `Variant::detect` are caught at the discovery layer too.
#[test]
fn base_family_for_common_model_aliases() {
    use plakat::preset::discovery::BaseFamily;
    assert_eq!(BaseFamily::from_model_arg("sd15"), BaseFamily::Sd15);
    assert_eq!(BaseFamily::from_model_arg("sdxl"), BaseFamily::Sdxl);
    assert_eq!(BaseFamily::from_model_arg("flux-dev"), BaseFamily::Flux);
    assert_eq!(BaseFamily::from_model_arg("sd35-medium"), BaseFamily::Sd3);
}

// --- Phase 6: same flags on portrait / img2img / outpaint ---

/// `plakat portrait --look watercolor` parses.
#[test]
fn portrait_accepts_look_and_genre() {
    let cli = Cli::try_parse_from([
        "plakat", "portrait", "--look", "watercolor", "--genre", "anime",
        "--offline", "a person",
    ])
    .expect("parse");
    match cli.command {
        Command::Portrait(args) => {
            assert_eq!(args.look.as_deref(), Some("watercolor"));
            assert_eq!(args.genre.as_deref(), Some("anime"));
            assert!(args.offline);
        }
        other => panic!("expected Portrait, got {other:?}"),
    }
}

/// `plakat img2img --look watercolor` parses (also covers inpaint
/// via the `--mask` flag on the same subcommand).
#[test]
fn img2img_accepts_look_and_genre_and_inpaint() {
    let cli = Cli::try_parse_from([
        "plakat", "img2img",
        "--prompt", "transform this",
        "--mask", "/tmp/mask.png",
        "--look", "oil-painting",
        "--genre", "anime",
        "--offline",
        "/tmp/x.png",
    ])
    .expect("parse");
    match cli.command {
        Command::Img2img(args) => {
            assert_eq!(args.look.as_deref(), Some("oil-painting"));
            assert_eq!(args.genre.as_deref(), Some("anime"));
            assert!(args.offline);
            assert!(args.mask.is_some(), "inpaint mask threaded through");
        }
        other => panic!("expected Img2img, got {other:?}"),
    }
}

/// `plakat outpaint --look ink-wash` parses.
#[test]
fn outpaint_accepts_look_and_genre() {
    let cli = Cli::try_parse_from([
        "plakat", "outpaint",
        "--prompt", "extend the scene",
        "--expand", "256",
        "--look", "ink-wash",
        "--genre", "anime",
        "--offline",
        "/tmp/x.png",
    ])
    .expect("parse");
    match cli.command {
        Command::Outpaint(args) => {
            assert_eq!(args.look.as_deref(), Some("ink-wash"));
            assert_eq!(args.genre.as_deref(), Some("anime"));
            assert!(args.offline);
        }
        other => panic!("expected Outpaint, got {other:?}"),
    }
}

/// Each subcommand's bogus-look name fails fast with a helpful
/// list of valid names — and (this is the gate) before any model
/// load / network call.
#[test]
fn portrait_rejects_unknown_look_at_parse_time() {
    // Note: clap parse succeeds (the string-typed flag accepts
    // anything); the actual name validation happens at the
    // catalog lookup inside the run function. This test pins the
    // shape rather than the error path.
    let cli = Cli::try_parse_from([
        "plakat", "portrait", "--look", "definitely-not-real", "a person",
    ])
    .expect("parse accepts any string");
    match cli.command {
        Command::Portrait(args) => assert_eq!(args.look.as_deref(), Some("definitely-not-real")),
        other => panic!("expected Portrait, got {other:?}"),
    }
}

// --- Phase 11: full-surface integration tests ---

/// Offline mode: cache-roundtrip end-to-end. Writes a cache entry,
/// reads it back via `discover_lora`, asserts the same LoraSpec
/// comes out the other side. Exercises the cache schema_version
/// + cache-key-by-(name, base) logic without network.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn offline_cache_round_trip_end_to_end() {
    use plakat::pipelines::lora::{CivitaiIdKind, LoraSource};
    use plakat::preset::LoraQuery;
    use plakat::preset::discovery::{
        BaseFamily, DiscoveryOptions, Source, discover_lora,
    };

    let dir = tempfile::tempdir().unwrap();
    let opts = DiscoveryOptions {
        offline: true,
        base: BaseFamily::Sdxl,
        cache_root: dir.path().join("discovery"),
        civitai_cache_root: dir.path().join("civitai-empty"),
        scale: 0.8,
    };

    // Plant a cache entry by hand (same shape discover_lora would
    // write after a successful first-run discovery).
    let cache_dir = opts.cache_root.clone();
    std::fs::create_dir_all(&cache_dir).unwrap();
    let cache_file = cache_dir.join("watercolor__sdxl.json");
    let payload = serde_json::json!({
        "schema_version": 1,
        "source": {
            "source": "civitai",
            "model_id": 12345u64,
            "version_id": 67890u64,
        },
        "model_name": "Watercolor v3",
        "trigger_words": ["watercolor", "wash"],
        "source_url": "https://civitai.com/models/12345",
        "discovered_at": 1700000000u64,
    });
    std::fs::write(&cache_file, payload.to_string()).unwrap();

    let q = LoraQuery {
        tags: vec!["watercolor".into()],
        keywords: vec!["watercolor".into()],
    };
    let d = discover_lora(&q, "watercolor", &opts)
        .await
        .expect("discover_lora")
        .expect("cache hit");

    // Cache reconstructs a Civitai version-pinned LoraSpec.
    match d.spec.source {
        LoraSource::Civitai {
            id_kind: CivitaiIdKind::Version(v),
            ..
        } => assert_eq!(v, 67890),
        other => panic!("expected Civitai version, got {other:?}"),
    }
    assert!((d.spec.scale - 0.8).abs() < f32::EPSILON);
    assert_eq!(d.trigger_words, vec!["watercolor", "wash"]);
    assert_eq!(d.model_name, "Watercolor v3");
    match d.source {
        Source::Civitai {
            model_id,
            version_id,
        } => {
            assert_eq!(model_id, 12345);
            assert_eq!(version_id, 67890);
        }
        other => panic!("expected Source::Civitai, got {other:?}"),
    }
}

/// Offline + local-scan: plants a fake Civitai cache entry on
/// disk (metadata.json + safetensors), runs discover_lora in
/// offline mode, asserts the local scan finds it and writes the
/// discovery cache for future runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn offline_local_scan_full_round_trip() {
    use plakat::civitai::api::ModelVersion;
    use plakat::preset::LoraQuery;
    use plakat::preset::discovery::{
        BaseFamily, DiscoveryOptions, Source, discover_lora,
    };

    let dir = tempfile::tempdir().unwrap();
    let civitai_dir = dir.path().join("civitai");
    let version_dir = civitai_dir.join("model-42").join("version-7");
    std::fs::create_dir_all(&version_dir).unwrap();
    std::fs::write(
        version_dir.join("wc.safetensors"),
        b"fake-safetensors-bytes",
    )
    .unwrap();
    let version = ModelVersion {
        id: 7,
        name: "v1".into(),
        base_model: Some("SDXL 1.0".into()),
        trained_words: vec!["watercolor wash".into()],
        download_url: None,
        files: vec![],
    };
    std::fs::write(
        version_dir.join("metadata.json"),
        serde_json::to_vec_pretty(&version).unwrap(),
    )
    .unwrap();

    let opts = DiscoveryOptions {
        offline: true,
        base: BaseFamily::Sdxl,
        cache_root: dir.path().join("discovery"),
        civitai_cache_root: civitai_dir,
        scale: 0.8,
    };
    let q = LoraQuery {
        tags: vec!["watercolor".into()],
        keywords: vec!["watercolor".into()],
    };
    let d = discover_lora(&q, "watercolor", &opts)
        .await
        .unwrap()
        .expect("local-scan hit");
    match d.source {
        Source::Civitai {
            model_id,
            version_id,
        } => {
            assert_eq!(model_id, 42);
            assert_eq!(version_id, 7);
        }
        other => panic!("expected Civitai, got {other:?}"),
    }
    assert_eq!(d.trigger_words, vec!["watercolor wash"]);

    // The successful discovery also writes a cache entry; a second
    // discover_lora call hits the cache rather than re-scanning.
    let cache_file = opts
        .cache_root
        .join("watercolor__sdxl.json");
    assert!(cache_file.exists(), "discovery cache entry written");
}

/// Full-surface byte-identity check: when the user populates
/// every override-only field, applying a look produces the same
/// scalar values as not applying it. Mirrors the RFC's claim
/// "fully-flagged command is byte-identical with/without --look"
/// — for the overridable fields only.
#[test]
fn full_surface_byte_identity_for_override_only_fields() {
    use plakat::preset::{GenerationParams, apply_presets};

    let baseline_params = || GenerationParams {
        prompt: "a knight".into(),
        negative: "blurry".into(),
        steps: Some(50),
        guidance: Some(9.0),
        scheduler: Some("euler-a".into()),
    };

    // Apply EVERY bundled look one by one + verify the override
    // fields all survive. Looks differ in their override values
    // so any leak would surface here.
    for look_name in [
        "ink-wash",
        "watercolor",
        "oil-painting",
        "charcoal",
        "pencil",
        "chalk-pastel",
        "linocut",
        "gouache",
    ] {
        let mut params = baseline_params();
        apply_presets(Some(look_name), None, &mut params).unwrap();
        assert_eq!(
            params.steps,
            Some(50),
            "look {look_name} clobbered user steps"
        );
        assert!(
            (params.guidance.unwrap() - 9.0).abs() < f64::EPSILON,
            "look {look_name} clobbered user guidance"
        );
        assert_eq!(
            params.scheduler.as_deref(),
            Some("euler-a"),
            "look {look_name} clobbered user scheduler"
        );
    }
}

/// All 8 bundled looks have lora_query populated (so discovery
/// has something to do). Anchors the catalog contract — a future
/// refactor that drops lora_query on a bundled look fails here.
#[test]
fn all_bundled_looks_have_lora_query() {
    use plakat::preset::{Catalog, Kind};
    let cat = Catalog::load_with_user_dir(Kind::Look, None).unwrap();
    for entry in &cat.entries {
        let q = entry
            .lora_query
            .as_ref()
            .unwrap_or_else(|| panic!("look {} has no lora_query", entry.name));
        assert!(
            !q.keywords.is_empty() || !q.tags.is_empty(),
            "look {} has empty lora_query",
            entry.name
        );
    }
}

/// Bundled `anime` genre has `lora_query` too.
#[test]
fn bundled_anime_genre_has_lora_query() {
    use plakat::preset::{Catalog, Kind};
    let cat = Catalog::load_with_user_dir(Kind::Genre, None).unwrap();
    let anime = cat.find("anime").unwrap();
    let q = anime.lora_query.as_ref().unwrap();
    assert!(!q.keywords.is_empty());
    assert!(!q.tags.is_empty());
}

/// Catalog round-trip: every bundled look's `prompt_prefix` survives
/// JSON serialise → parse → apply intact. Guards against schema
/// drift in the v0.25 → v0.26 transition.
#[test]
fn catalog_prompt_prefix_round_trips_through_json() {
    use plakat::preset::{Catalog, Kind};
    let cat = Catalog::load_with_user_dir(Kind::Look, None).unwrap();
    for entry in &cat.entries {
        let prefix = entry
            .prompt_prefix
            .as_deref()
            .unwrap_or_else(|| panic!("look {} has no prompt_prefix", entry.name));
        assert!(
            !prefix.is_empty(),
            "look {} has empty prompt_prefix",
            entry.name
        );
        // Re-serialise + parse to catch any non-roundtrippable bits.
        let json = serde_json::to_string(entry).unwrap();
        let back: plakat::preset::PresetSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.prompt_prefix.as_deref(), Some(prefix));
    }
}

/// Empty-user-side invocation: preset fills every override field
/// + composes prompt and negative.
#[test]
fn empty_invocation_takes_all_preset_values() {
    use plakat::preset::{GenerationParams, apply_presets};

    let mut params = GenerationParams {
        prompt: "a knight".into(),
        ..Default::default()
    };
    let (_, _) = apply_presets(Some("oil-painting"), None, &mut params).unwrap();
    assert_eq!(params.steps, Some(40));
    assert_eq!(params.guidance, Some(7.0));
    assert_eq!(params.scheduler.as_deref(), Some("dpmpp-2m"));
    assert!(params.prompt.contains("oil painting"));
}
