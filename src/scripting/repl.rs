//! v0.21 phase 7: `plakat run --repl` interactive line editor.
//!
//! Read-eval-loop against a **persistent** `Bund` instance so
//! variables, named lambdas, and stack state survive across
//! lines. Pattern lifted from blackInkhaven's CLI subcommand, but
//! we're not embedding bundcore's stdlib (RFC decision #2) — so
//! the REPL only sees the `plakat.*` words plus the bare
//! multistackvm primitives.
//!
//! Line edit + history powered by `rustyline`. History file
//! lives next to plakat's config:
//!
//!   Linux:   ~/.config/plakat/repl_history
//!   macOS:   ~/Library/Application Support/ai.plakat.plakat/repl_history
//!   Windows: %APPDATA%\plakat\plakat\config\repl_history
//!
//! Meta-commands (start with `.` — Forth convention):
//!
//!   .q  / .quit  — exit
//!   .s  / .stack — print the workbench stack
//!   .help        — list `plakat.*` words + meta-commands
//!
//! Everything else is fed verbatim to `bund.eval`. After a
//! successful eval that left a value on top of the workbench,
//! the REPL prints it (pull → display → push back; Value is
//! Clone so no state lost).

use anyhow::Result;
use bundcore::bundcore::Bund;
use rustyline::error::ReadlineError;
use rustyline::history::FileHistory;
use rustyline::{Config, Editor};
use std::path::PathBuf;

use super::build_plakat_bund;

const PROMPT: &str = "plakat> ";
const META_HELP: &str = "\
Meta-commands:
  .q | .quit     exit the REPL
  .s | .stack    print the workbench stack
  .help          this message
plakat.* words: plakat.load, plakat.generate, plakat.img2img,
                plakat.portrait, plakat.upscale, plakat.save,
                plakat.config.set, plakat.echo
Stack-based syntax (Bund / Forth-like). Examples:
  \"sd15\"           plakat.load
  \"a fox\"          plakat.generate
                    \"fox.png\"  plakat.save
  40   \"steps\"     plakat.config.set";

/// History file path under plakat's config dir. `None` if
/// `directories::ProjectDirs` can't resolve a config dir on this
/// platform — the REPL still runs, just without persistent
/// history.
fn history_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("ai", "plakat", "plakat")
        .map(|d| d.config_dir().join("repl_history"))
}

/// v0.21 phase 7: entry point invoked by `cli::run::run` when
/// `--repl` is set.
///
/// Built directly on `rustyline`'s blocking API; that's fine
/// because (a) the host words themselves do the
/// `block_in_place + block_on` async bridge internally (see
/// `plakat.generate`), and (b) the REPL is interactive — we
/// **want** to block the calling task while waiting for input.
pub fn run() -> Result<()> {
    let mut bund = build_plakat_bund()?;
    let cfg = Config::builder().auto_add_history(true).build();
    let mut rl: Editor<(), FileHistory> = Editor::with_config(cfg)?;
    let history = history_path();
    if let Some(path) = history.as_ref() {
        // Best-effort: a missing file on first run is normal. Any
        // other error gets swallowed (logged at debug) — the REPL
        // is more useful without history than not at all.
        if path.exists() {
            let _ = rl.load_history(path);
        }
    }

    println!("plakat REPL (v0.21). Type .help for commands, .q to exit.");
    loop {
        match rl.readline(PROMPT) {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                match handle_line(&mut bund, trimmed) {
                    LineOutcome::Quit => break,
                    LineOutcome::Continue => {}
                }
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl-C: don't exit; clear the partial line.
                continue;
            }
            Err(ReadlineError::Eof) => {
                // Ctrl-D: exit cleanly.
                break;
            }
            Err(e) => {
                eprintln!("REPL read error: {e}");
                break;
            }
        }
    }

    if let Some(path) = history.as_ref() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = rl.save_history(path);
    }
    Ok(())
}

pub(crate) enum LineOutcome {
    Continue,
    Quit,
}

/// Process one line. Pure function over `(bund, line)` — used by
/// tests to drive eval state changes without running the
/// interactive loop.
///
/// On a successful eval, prints the new top-of-stack (if any) so
/// the user gets immediate feedback. The print uses a non-
/// consuming peek: pull the value (for inspection), display it,
/// then push it back. `rust_dynamic::Value` is `Clone` + this
/// pattern matches Forth REPLs.
pub(crate) fn handle_line(bund: &mut Bund, line: &str) -> LineOutcome {
    match line {
        ".q" | ".quit" => return LineOutcome::Quit,
        ".help" => {
            println!("{META_HELP}");
            return LineOutcome::Continue;
        }
        ".s" | ".stack" => {
            let formatted = format_workbench(bund);
            if formatted.is_empty() {
                println!("(empty)");
            } else {
                println!("{formatted}");
            }
            return LineOutcome::Continue;
        }
        _ => {}
    }

    match bund.eval(line) {
        Ok(_) => {
            // Show the top of stack if there is one — Forth REPL
            // convention. Non-destructive: pull, display, push back.
            if let Some(top) = bund.vm.stack.pull() {
                let display = format_value_for_repl(&top);
                println!("=> {display}");
                bund.vm.stack.push(top);
            }
        }
        Err(e) => {
            // Don't bail the REPL on a script error — print +
            // continue so the user can fix the line and retry.
            eprintln!("error: {e}");
        }
    }
    LineOutcome::Continue
}

/// Format the entire workbench stack for the `.s` meta command.
/// Non-destructive: pulls every value, formats it, then pushes
/// the values back in their original order. Returns the multi-
/// line `[ N ] <value>` listing (bottom-up — top of stack is the
/// last line, matching Forth REPL convention).
pub(crate) fn format_workbench(bund: &mut Bund) -> String {
    let mut tmp: Vec<rust_dynamic::value::Value> = Vec::new();
    while let Some(v) = bund.vm.stack.pull() {
        tmp.push(v);
    }
    // tmp now holds values top-first. Iterate that order for the
    // listing (top → bottom going down the lines is unusual;
    // Forth shows bottom-up, so reverse) — and push them back in
    // reverse order so the original top ends up on top again.
    let mut lines: Vec<String> = Vec::with_capacity(tmp.len());
    for (i, v) in tmp.iter().rev().enumerate() {
        lines.push(format!("  [{}] {}", i, format_value_for_repl(v)));
    }
    // Restore: tmp is top-first, so we want to push bottom-first.
    for v in tmp.into_iter().rev() {
        bund.vm.stack.push(v);
    }
    lines.join("\n")
}

/// Format a `rust_dynamic::Value` for the REPL's `=>` echo.
/// rust_dynamic's `Debug` impl is noisy (includes `id`, `stamp`,
/// `dt` tag, etc); strip to just the user-facing payload.
pub(crate) fn format_value_for_repl(v: &rust_dynamic::value::Value) -> String {
    use rust_dynamic::types;
    match v.dt {
        types::INTEGER => v
            .cast_int()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "<int?>".to_string()),
        types::FLOAT => v
            .cast_float()
            .map(|f| format!("{f:.6}"))
            .unwrap_or_else(|_| "<float?>".to_string()),
        types::STRING => v
            .cast_string()
            .map(|s| format!("{s:?}"))
            .unwrap_or_else(|_| "<string?>".to_string()),
        types::BOOL => v
            .cast_bool()
            .map(|b| b.to_string())
            .unwrap_or_else(|_| "<bool?>".to_string()),
        types::NONE => "(none)".to_string(),
        _ => format!("<value dt={}>", v.dt),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scripting::ctx::ScriptCtx;
    use candle_core::Device;

    /// Drive the REPL line-by-line against a persistent Bund.
    /// Each line is eval'd; state survives across calls. This is
    /// the load-bearing test for phase 7 — the whole point of the
    /// REPL is that `1 2` then `+` then `.s` shows `3`, not a
    /// fresh empty stack each line.
    ///
    /// Runs on a multi-thread tokio runtime because some host words
    /// (none used here, but kept consistent) need the bridge.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn handle_line_preserves_state_across_calls() {
        // Reuse the same singleton-context gate as the rest of
        // scripting::tests so we don't double-init.
        crate::scripting::tests_with_singleton_ctx(|| {
            let mut bund = build_plakat_bund().unwrap();
            // Push three ints across two lines; verify the stack
            // accumulates.
            assert!(matches!(
                handle_line(&mut bund, "1 2 +"),
                LineOutcome::Continue
            ));
            // Stack should contain `3` now.
            let len = bund.vm.stack.current_stack_len();
            assert_eq!(len, 1, "expected 1 value on stack, got {len}");
            // Add another op against the persisted state.
            assert!(matches!(
                handle_line(&mut bund, "10 +"),
                LineOutcome::Continue
            ));
            // Top should now be 13 = 3 + 10.
            let top = bund.vm.stack.pull().unwrap();
            assert_eq!(top.cast_int().unwrap(), 13);
        });
        let _ = ScriptCtx::init(Device::Cpu, std::env::temp_dir()); // suppress unused-import
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn handle_line_quit_returns_quit_outcome() {
        crate::scripting::tests_with_singleton_ctx(|| {
            let mut bund = build_plakat_bund().unwrap();
            assert!(matches!(handle_line(&mut bund, ".q"), LineOutcome::Quit));
            assert!(matches!(handle_line(&mut bund, ".quit"), LineOutcome::Quit));
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn handle_line_error_does_not_propagate() {
        crate::scripting::tests_with_singleton_ctx(|| {
            let mut bund = build_plakat_bund().unwrap();
            // Bogus word → parser/exec error. Should print
            // (silenced in the test runner) but return Continue.
            let outcome = handle_line(&mut bund, "definitely-not-a-word");
            assert!(matches!(outcome, LineOutcome::Continue));
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn format_workbench_round_trips_and_restores_order() {
        crate::scripting::tests_with_singleton_ctx(|| {
            let mut bund = build_plakat_bund().unwrap();
            handle_line(&mut bund, "1 2 3");
            let s = format_workbench(&mut bund);
            // Forth `.s` convention is bottom-up listing: index 0 is
            // the BOTTOM of the stack (the first value pushed, `1`).
            // The top of the stack (last pushed, `3`) gets the
            // highest index. Reading the output top-down, you see
            // the stack from bottom to top — the order a Forth
            // programmer expects.
            assert!(s.contains("[0] 1"), "got:\n{s}");
            assert!(s.contains("[1] 2"), "got:\n{s}");
            assert!(s.contains("[2] 3"), "got:\n{s}");
            // Restore contract: format_workbench is non-destructive.
            // After it returns, popping the stack should yield the
            // values in the original top-down order: 3, 2, 1.
            assert_eq!(
                bund.vm.stack.pull().unwrap().cast_int().unwrap(),
                3
            );
            assert_eq!(
                bund.vm.stack.pull().unwrap().cast_int().unwrap(),
                2
            );
            assert_eq!(
                bund.vm.stack.pull().unwrap().cast_int().unwrap(),
                1
            );
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn format_workbench_empty_stack_returns_empty_string() {
        crate::scripting::tests_with_singleton_ctx(|| {
            let mut bund = build_plakat_bund().unwrap();
            // Drain anything prior tests left.
            while bund.vm.stack.pull().is_some() {}
            assert_eq!(format_workbench(&mut bund), "");
        });
    }

    #[test]
    fn format_value_for_repl_int() {
        let v = rust_dynamic::value::Value::from_int(42);
        assert_eq!(format_value_for_repl(&v), "42");
    }

    #[test]
    fn format_value_for_repl_string_is_quoted() {
        let v = rust_dynamic::value::Value::from_string("hi");
        assert_eq!(format_value_for_repl(&v), "\"hi\"");
    }

    #[test]
    fn format_value_for_repl_float() {
        let v = rust_dynamic::value::Value::from_float(3.5);
        let s = format_value_for_repl(&v);
        // Six fractional digits per the format string.
        assert!(s.starts_with("3.500000"), "got {s}");
    }
}
