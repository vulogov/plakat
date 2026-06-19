//! Compiled when the `templates` feature is OFF. Any attempt to use a template
//! input returns a clear "recompile" error — the parser path is unaffected.

use anyhow::{Result, bail};
use std::path::Path;

use super::TemplateOpts;

/// Feature not compiled — error pointing at the flag.
pub fn render(_input: &str, _input_path: Option<&Path>, _opts: &TemplateOpts) -> Result<String> {
    bail!(
        "plakat was compiled without the `templates` feature, so `.tera` / `--template` \
         inputs aren't supported. Recompile with:  cargo build --features templates"
    );
}
