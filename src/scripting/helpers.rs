//! v0.21: stack / value helpers for plakat host words.
//!
//! Pattern lifted from blackInkhaven's `src/scripting/stdlib/helpers.rs`:
//! every host word is a `fn(&mut VM) -> Result<&mut VM, easy_error::Error>`,
//! pops its args off the workbench (top-most popped first), validates
//! shapes, calls into plakat's `anyhow`-flavoured world, and pushes
//! its result back. The `anyhow inside / easy_error at boundary`
//! pattern keeps the rest of plakat insulated from bundcore's error
//! type — only this module + the words/ leaves see `easy_error`.

use anyhow::{Result, anyhow};
use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

/// The host-function return type bundcore expects. Aliased here so
/// every word file references it through one name.
pub type BundResult<'a> = std::result::Result<&'a mut VM, easy_error::Error>;

/// Adapter: `anyhow::Error` → `easy_error::Error`. Preserves the
/// display message; loses the backtrace, but bundcore can't surface
/// one to script-land anyway.
pub fn to_bund_err(e: anyhow::Error) -> easy_error::Error {
    easy_error::err_msg(format!("{e:#}"))
}

/// Pop the top of the workbench (or main stack if workbench is
/// empty). Mirrors blackInkhaven's `pull` helper.
///
/// `tag` is included in error messages so the user sees *which*
/// host word's pop failed, not just "stack underflow."
pub fn pull(vm: &mut VM, tag: &str) -> Result<Value> {
    vm.stack
        .pull()
        .ok_or_else(|| anyhow!("{tag}: stack underflow (no value to pop)"))
}

/// Push a value onto the workbench. Stays an infallible helper for
/// symmetry with `pull` — bundcore's `push` is itself infallible.
pub fn push(vm: &mut VM, value: Value) {
    vm.stack.push(value);
}

/// Bail unless the workbench has at least `n` values to pop. Run
/// before the first `pull` in any multi-arg word so a script that
/// invoked the word with the wrong arity fails up front rather
/// than after a partial pop.
pub fn require_depth(vm: &mut VM, n: usize, tag: &str) -> Result<()> {
    let have = vm.stack.current_stack_len();
    if have < n {
        return Err(anyhow!(
            "{tag}: needs {n} arg(s) on the stack, found {have}"
        ));
    }
    Ok(())
}

/// Coerce a `Value` to a `String`. Accepts string values; rejects
/// everything else with a typed error. `field` is the human-name
/// of the arg (e.g. "prompt", "path") for diagnostics.
pub fn value_to_string(v: Value, field: &str, tag: &str) -> Result<String> {
    v.cast_string()
        .map_err(|e| anyhow!("{tag}: arg {field:?} must be a string ({e})"))
}

/// Coerce a `Value` to an `i64`. Accepts integer values; rejects
/// everything else with a typed error.
pub fn value_to_int(v: Value, field: &str, tag: &str) -> Result<i64> {
    v.cast_int()
        .map_err(|e| anyhow!("{tag}: arg {field:?} must be an integer ({e})"))
}

/// Coerce a `Value` to an `f64`. Accepts float OR int values
/// (int → lossless cast to float) so scripts can pass `7` where
/// `7.0` is expected without the syntax friction.
pub fn value_to_float(v: Value, field: &str, tag: &str) -> Result<f64> {
    // Try float first; fall back to int. rust_dynamic's cast_float
    // doesn't auto-promote integers, so the two-step gives us the
    // user-friendly behaviour.
    if let Ok(f) = v.cast_float() {
        return Ok(f);
    }
    if let Ok(i) = v.cast_int() {
        return Ok(i as f64);
    }
    Err(anyhow!(
        "{tag}: arg {field:?} must be a number (float or integer)"
    ))
}

/// Resolve a stack arg that is either a **string path** or an **integer image handle** into a
/// filesystem `PathBuf` (plus a tempfile guard that must outlive the path when the arg was a
/// handle — the handle's image is materialised to disk so pipelines can read it). Shared by the
/// image-consuming words (stylize/img2img/relight/transparent/segment/…) so they all accept the
/// same two shapes. `tag` names the calling word for error messages.
pub fn resolve_image_arg(
    v: Value,
    field: &str,
    tag: &str,
) -> Result<(Option<tempfile::NamedTempFile>, std::path::PathBuf)> {
    use rust_dynamic::types;
    match v.dt {
        types::STRING => {
            let s = value_to_string(v, field, tag)?;
            if s.is_empty() {
                anyhow::bail!("{tag}: {field} path can't be empty");
            }
            Ok((None, std::path::PathBuf::from(s)))
        }
        types::INTEGER => {
            let handle = v.cast_int().unwrap_or(0);
            let img = crate::scripting::ctx::with_ctx_mut(|ctx| ctx.image_at(handle).cloned())??;
            let tmp = tempfile::Builder::new()
                .prefix(&format!("plakat-script-{field}-"))
                .suffix(".png")
                .tempfile()
                .map_err(|e| anyhow!("{tag}: tempfile for {field} handle {handle}: {e}"))?;
            img.save(tmp.path())
                .map_err(|e| anyhow!("{tag}: writing {field} handle {handle} to tempfile: {e}"))?;
            let path = tmp.path().to_path_buf();
            Ok((Some(tmp), path))
        }
        _ => anyhow::bail!(
            "{tag}: {field} must be a string path or an integer handle (got dt = {})",
            v.dt
        ),
    }
}

/// Collect the image files (png/jpg/jpeg/webp) directly inside `dir`, sorted by name. Used by
/// the training words (`plakat.style.train` / `plakat.embedding.train`) which take a folder of
/// training images. `tag` names the calling word for error messages.
pub fn collect_images_in_dir(dir: &str, tag: &str) -> Result<Vec<std::path::PathBuf>> {
    let rd = std::fs::read_dir(dir)
        .map_err(|e| anyhow!("{tag}: reading images dir {dir:?}: {e}"))?;
    let mut paths: Vec<std::path::PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|x| x.to_str())
                .map(|x| matches!(x.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "webp"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        anyhow::bail!("{tag}: no image files (png/jpg/jpeg/webp) in {dir:?}");
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_multistackvm::multistackvm::VM;

    #[test]
    fn to_bund_err_round_trips_message() {
        let err = to_bund_err(anyhow!("something specific"));
        let s = format!("{err}");
        assert!(s.contains("something specific"), "got {s}");
    }

    #[test]
    fn require_depth_bails_on_empty_stack() {
        let mut vm = VM::new();
        let err = require_depth(&mut vm, 1, "test.word").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("test.word"), "got {msg}");
        assert!(msg.contains("found 0"), "got {msg}");
    }

    #[test]
    fn require_depth_passes_with_enough_values() {
        let mut vm = VM::new();
        vm.stack.push(Value::from_string("a".to_string()));
        vm.stack.push(Value::from_string("b".to_string()));
        require_depth(&mut vm, 2, "test.word").unwrap();
    }

    #[test]
    fn pull_returns_top_of_workbench() {
        let mut vm = VM::new();
        vm.stack.push(Value::from_string("hello".to_string()));
        let got = pull(&mut vm, "test.word").unwrap();
        assert_eq!(got.cast_string().unwrap(), "hello");
    }

    #[test]
    fn pull_on_empty_stack_bails_with_tag() {
        let mut vm = VM::new();
        let err = pull(&mut vm, "test.word").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("test.word"), "got {msg}");
        assert!(msg.contains("underflow"), "got {msg}");
    }

    #[test]
    fn value_to_string_passes_through_strings() {
        let v = Value::from_string("forty-two".to_string());
        let got = value_to_string(v, "field", "test.word").unwrap();
        assert_eq!(got, "forty-two");
    }

    #[test]
    fn value_to_int_passes_through_ints() {
        let v = Value::from_int(42);
        let got = value_to_int(v, "field", "test.word").unwrap();
        assert_eq!(got, 42);
    }

    #[test]
    fn value_to_int_bails_on_string_with_tag() {
        let v = Value::from_string("not-a-number".to_string());
        let err = value_to_int(v, "handle", "test.word").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("test.word"), "got {msg}");
        assert!(msg.contains("handle"), "got {msg}");
        assert!(msg.contains("integer"), "got {msg}");
    }
}
