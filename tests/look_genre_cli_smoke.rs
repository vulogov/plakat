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
/// `run` dispatch will use.
#[test]
fn bundled_looks_catalog_loads_from_cwd() {
    use plakat::preset::{Catalog, Kind};
    let cat = Catalog::load_default(Kind::Look).expect("load bundled looks");
    assert_eq!(cat.entries.len(), 8);
    assert!(cat.find("watercolor").is_some());
    assert!(cat.find("oil-painting").is_some());
}

/// Same for genres.
#[test]
fn bundled_genres_catalog_loads_from_cwd() {
    use plakat::preset::{Catalog, Kind};
    let cat = Catalog::load_default(Kind::Genre).expect("load bundled genres");
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
        cache_root: dir.path().to_path_buf(),
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
