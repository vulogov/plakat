//! v0.27 phase 7: CLI smoke tests for `plakat animate --animatediff`.
//!
//! Drives clap's parser directly — no model loads, no network. Pins
//! the v0.27 flag surface (AnimateDiff + ControlNet + sliding window)
//! so a future refactor that drifts flag names / defaults gets
//! caught by `cargo test` rather than at user-machine smoke time.

use clap::Parser;
use plakat::cli::{Cli, Command};

/// `--animatediff` flag parses on `animate`. Default is off.
#[test]
fn animate_animatediff_flag_parses() {
    let cli = Cli::try_parse_from([
        "plakat",
        "animate",
        "--from",
        "test",
        "--animatediff",
        "--frames",
        "16",
    ])
    .expect("parse");
    match cli.command {
        Command::Animate(args) => {
            assert!(args.animatediff);
            assert_eq!(args.frames, 16);
        }
        other => panic!("expected Animate, got {other:?}"),
    }
}

/// `--animatediff` defaults to false when omitted.
#[test]
fn animate_animatediff_defaults_off() {
    let cli = Cli::try_parse_from([
        "plakat",
        "animate",
        "--from",
        "a",
        "--to",
        "b",
    ])
    .expect("parse");
    match cli.command {
        Command::Animate(args) => assert!(!args.animatediff),
        other => panic!("expected Animate, got {other:?}"),
    }
}

/// `--motion-lora SPEC` is repeatable; v0.26 added the flag, v0.27
/// keeps the same parsing surface (LoraSpec grammar).
#[test]
fn animate_motion_lora_is_repeatable() {
    let cli = Cli::try_parse_from([
        "plakat",
        "animate",
        "--from",
        "test",
        "--animatediff",
        "--motion-lora",
        "hf:guoyww/animatediff-motion-lora-zoom-in:0.7",
        "--motion-lora",
        "civitai:67890:0.5",
    ])
    .expect("parse");
    match cli.command {
        Command::Animate(args) => assert_eq!(args.motion_loras.len(), 2),
        other => panic!("expected Animate, got {other:?}"),
    }
}

/// v0.27 phase 5/6: `--window-size` + `--window-overlap` parse with
/// sensible defaults.
#[test]
fn animate_long_form_flag_defaults() {
    let cli = Cli::try_parse_from([
        "plakat",
        "animate",
        "--from",
        "x",
        "--animatediff",
    ])
    .expect("parse");
    match cli.command {
        Command::Animate(args) => {
            // V3 native window + 25 % overlap = community defaults.
            assert_eq!(args.window_size, 16);
            assert_eq!(args.window_overlap, 4);
        }
        other => panic!("expected Animate, got {other:?}"),
    }
}

/// v0.27 phase 5/6: explicit `--window-size` / `--window-overlap`
/// override the defaults.
#[test]
fn animate_long_form_flags_override_defaults() {
    let cli = Cli::try_parse_from([
        "plakat",
        "animate",
        "--from",
        "x",
        "--animatediff",
        "--frames",
        "64",
        "--window-size",
        "8",
        "--window-overlap",
        "2",
    ])
    .expect("parse");
    match cli.command {
        Command::Animate(args) => {
            assert_eq!(args.frames, 64);
            assert_eq!(args.window_size, 8);
            assert_eq!(args.window_overlap, 2);
        }
        other => panic!("expected Animate, got {other:?}"),
    }
}

/// v0.27 phase 3/4: `--control KIND` + `--control-image PATH` +
/// `--control-strength F` parse.
#[test]
fn animate_controlnet_flags_parse() {
    let cli = Cli::try_parse_from([
        "plakat",
        "animate",
        "--from",
        "x",
        "--animatediff",
        "--control",
        "depth",
        "--control-image",
        "/tmp/depth.png",
        "--control-strength",
        "0.8",
    ])
    .expect("parse");
    match cli.command {
        Command::Animate(args) => {
            assert_eq!(args.control.as_deref(), Some("depth"));
            assert_eq!(
                args.control_image.as_deref().and_then(|p| p.to_str()),
                Some("/tmp/depth.png"),
            );
            assert!((args.control_strength - 0.8).abs() < f32::EPSILON);
        }
        other => panic!("expected Animate, got {other:?}"),
    }
}

/// `--control-from PATH` is mutually exclusive with `--control-image`
/// at runtime, but both flags parse independently (the exclusivity
/// fires inside `run_animatediff`, not in clap).
#[test]
fn animate_controlnet_from_flag_parses() {
    let cli = Cli::try_parse_from([
        "plakat",
        "animate",
        "--from",
        "x",
        "--animatediff",
        "--control",
        "canny",
        "--control-from",
        "/tmp/source.jpg",
    ])
    .expect("parse");
    match cli.command {
        Command::Animate(args) => {
            assert_eq!(args.control.as_deref(), Some("canny"));
            assert_eq!(
                args.control_from.as_deref().and_then(|p| p.to_str()),
                Some("/tmp/source.jpg"),
            );
            assert_eq!(args.control_image, None);
        }
        other => panic!("expected Animate, got {other:?}"),
    }
}

/// `--format` parses every variant the v0.26 video module supports.
#[test]
fn animate_format_flag_parses_all_variants() {
    for variant in ["frames", "gif", "mp4", "webm", "all"] {
        let cli = Cli::try_parse_from([
            "plakat",
            "animate",
            "--from",
            "x",
            "--animatediff",
            "--format",
            variant,
        ])
        .unwrap_or_else(|_| panic!("--format {variant} should parse"));
        match cli.command {
            Command::Animate(args) => {
                // Round-trip through Display to verify.
                assert_eq!(args.format.to_string(), variant);
            }
            other => panic!("expected Animate, got {other:?}"),
        }
    }
}

/// `--motion-lora-scale F` parses + defaults to 1.0.
#[test]
fn animate_motion_lora_scale_default_and_override() {
    let default = Cli::try_parse_from([
        "plakat",
        "animate",
        "--from",
        "x",
        "--animatediff",
    ])
    .expect("parse default");
    let overridden = Cli::try_parse_from([
        "plakat",
        "animate",
        "--from",
        "x",
        "--animatediff",
        "--motion-lora-scale",
        "0.5",
    ])
    .expect("parse override");
    match (default.command, overridden.command) {
        (Command::Animate(d), Command::Animate(o)) => {
            assert!((d.motion_lora_scale - 1.0).abs() < f32::EPSILON);
            assert!((o.motion_lora_scale - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected Animate pair, got {other:?}"),
    }
}
