//! v0.29 phase 4: integration tests for animate-in-scenarios.
//!
//! Two layers: (1) HJSON parse + validation via the deser_hjson path
//! used at runtime, and (2) `plakat scenario --dry-run` via the
//! release binary so the end-to-end CLI surface is pinned.
//!
//! No network, no model loads — every test stays under 100 ms.
//!
//! The CLI binary tests are best-effort: they assume `cargo build
//! --release` has produced `target/release/plakat`. CI environments
//! that don't pre-build are expected to drive these via `cargo run
//! --release`.

use std::process::Command;

/// Locate the plakat release binary. Falls back to the debug build
/// if the release one isn't present (local dev convenience).
fn plakat_bin() -> std::path::PathBuf {
    let release = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/release/plakat");
    if release.exists() {
        release
    } else {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/debug/plakat")
    }
}

fn run_plakat(args: &[&str], hjson: &str) -> (i32, String) {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    std::fs::write(tmp.path(), hjson).expect("write hjson");
    let mut cli_args: Vec<String> = vec!["scenario".to_string()];
    cli_args.push(tmp.path().to_string_lossy().to_string());
    for a in args {
        cli_args.push((*a).to_string());
    }
    let output = Command::new(plakat_bin())
        .args(&cli_args)
        .output()
        .unwrap_or_else(|e| {
            panic!("running plakat scenario: {e}")
        });
    let combined = String::from_utf8_lossy(&output.stdout).to_string()
        + &String::from_utf8_lossy(&output.stderr);
    let code = output.status.code().unwrap_or(-1);
    (code, combined)
}

/// v0.29 phase 3: a minimal all-animate scenario runs `--dry-run`
/// to completion. The output should list the planned animate task
/// with frame count + format inline.
#[test]
fn animate_scenario_dry_run_shows_plan() {
    let hjson = r#"{
        model: sd15
        type: animatediff
        frames: 16
        lcm: true
        format: gif
        out: /tmp/scenario-animate-smoke
        scene: [
            {
                name: dawn
                prompt: "at dawn"
            }
        ]
        weather: [
            {
                name: mist
                prompt: "misty"
            }
        ]
        tasks: [
            {
                name: cottage
                scene: dawn
                weather: mist
                prompt: "a watercolor cottage"
            }
        ]
    }"#;
    let (code, out) = run_plakat(&["--dry-run"], hjson);
    assert_eq!(code, 0, "expected exit 0, got {code}; output:\n{out}");
    assert!(
        out.contains("[1/1] cottage animate"),
        "expected animate task in output:\n{out}"
    );
    assert!(
        out.contains("dry-run"),
        "expected dry-run marker:\n{out}"
    );
    assert!(
        out.contains("format=gif"),
        "expected gif format:\n{out}"
    );
}

/// v0.29 phase 2: an HJSON scenario with `format: avif` (unknown)
/// surfaces the supported-formats hint with the task name.
#[test]
fn animate_scenario_bad_format_bails_with_task_name() {
    let hjson = r#"{
        model: sd15
        type: animatediff
        format: avif
        scene: [
            {
                name: dawn
                prompt: "at dawn"
            }
        ]
        weather: [
            {
                name: mist
                prompt: "misty"
            }
        ]
        tasks: [
            {
                name: badtask
                scene: dawn
                weather: mist
                prompt: "x"
            }
        ]
    }"#;
    let (code, out) = run_plakat(&[], hjson);
    assert_ne!(code, 0, "expected non-zero exit on bad format");
    assert!(
        out.contains("badtask"),
        "expected task name in error:\n{out}"
    );
    assert!(
        out.contains("avif"),
        "expected format value in error:\n{out}"
    );
}

/// v0.29 phase 2: window-size > 32 bails at parse time before any
/// pipeline load.
#[test]
fn animate_scenario_oversize_window_bails() {
    let hjson = r#"{
        model: sd15
        type: animatediff
        window-size: 99
        scene: [
            {
                name: dawn
                prompt: "x"
            }
        ]
        weather: [
            {
                name: mist
                prompt: "y"
            }
        ]
        tasks: [
            {
                name: bigwin
                scene: dawn
                weather: mist
                prompt: "z"
            }
        ]
    }"#;
    let (code, out) = run_plakat(&[], hjson);
    assert_ne!(code, 0, "expected non-zero exit on oversize window");
    assert!(
        out.contains("motion_max_seq_length") || out.contains("1..="),
        "expected max_seq_length hint:\n{out}"
    );
    assert!(
        out.contains("bigwin"),
        "expected task name in error:\n{out}"
    );
}

/// v0.29 phase 3: per-task `type: generate` override in an all-
/// animate scenario routes that one task through the generate path.
/// Without an enhancer set (and DEEPSEEK_API_KEY unset), the
/// generate task would bail at the enhancer check — confirming the
/// mixed-kind classification works.
#[test]
fn animate_scenario_per_task_generate_override_triggers_enhancer_check() {
    let hjson = r#"{
        model: sd15
        type: animatediff
        scene: [
            {
                name: dawn
                prompt: "x"
            }
        ]
        weather: [
            {
                name: mist
                prompt: "y"
            }
        ]
        tasks: [
            {
                name: anim
                scene: dawn
                weather: mist
                prompt: "z"
            }
            {
                name: gen
                scene: dawn
                weather: mist
                prompt: "z"
                type: generate
            }
        ]
    }"#;
    let (code, out) = run_plakat(&[], hjson);
    assert_ne!(code, 0, "expected non-zero exit for mixed-kind without enhancer");
    assert!(
        out.contains("enhancer"),
        "expected enhancer requirement in error:\n{out}"
    );
}
