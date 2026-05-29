//! v0.28 phase 3: CLI smoke tests for `plakat motion-adapter`.
//!
//! Drives clap's parser directly — no network, no model loads.
//! Pins the subcommand surface (`info` + `list`) so a future
//! refactor that drifts flag names or sub-action layout gets
//! caught here.

use clap::Parser;
use plakat::cli::{
    motion_adapter::{InfoArgs, MotionAdapterArgs, MotionAdapterCmd},
    Cli, Command,
};

/// `plakat motion-adapter list` parses with no extra args.
#[test]
fn motion_adapter_list_parses() {
    let cli =
        Cli::try_parse_from(["plakat", "motion-adapter", "list"]).expect("parse");
    match cli.command {
        Command::MotionAdapter(MotionAdapterArgs { cmd: MotionAdapterCmd::List }) => {}
        other => panic!("expected motion-adapter list, got {other:?}"),
    }
}

/// `plakat motion-adapter info REPO` parses + carries the repo
/// string verbatim through to the arg struct.
#[test]
fn motion_adapter_info_carries_repo_arg() {
    let cli = Cli::try_parse_from([
        "plakat",
        "motion-adapter",
        "info",
        "guoyww/animatediff-motion-adapter-v1-5-3",
    ])
    .expect("parse");
    match cli.command {
        Command::MotionAdapter(MotionAdapterArgs {
            cmd: MotionAdapterCmd::Info(InfoArgs { repo }),
        }) => {
            assert_eq!(repo, "guoyww/animatediff-motion-adapter-v1-5-3");
        }
        other => panic!("expected motion-adapter info, got {other:?}"),
    }
}

/// `plakat motion-adapter info` with no REPO bails at clap level.
#[test]
fn motion_adapter_info_without_repo_bails() {
    let err = Cli::try_parse_from(["plakat", "motion-adapter", "info"]).unwrap_err();
    let s = err.to_string();
    assert!(
        s.contains("required") || s.contains("<REPO>") || s.contains("REPO"),
        "expected missing-arg error, got: {s}"
    );
}

/// `plakat motion-adapter` without a sub-action bails at clap level
/// rather than silently doing nothing.
#[test]
fn motion_adapter_without_subaction_bails() {
    let err = Cli::try_parse_from(["plakat", "motion-adapter"]).unwrap_err();
    let s = err.to_string();
    assert!(
        s.contains("required") || s.contains("<OP>") || s.contains("subcommand"),
        "expected missing-subcommand error, got: {s}"
    );
}
