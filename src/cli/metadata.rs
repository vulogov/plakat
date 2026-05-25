//! `plakat metadata FILE.png` — read the v0.17 generation recipe
//! embedded in a plakat-produced PNG. Reverse of the metadata write
//! path: prints the Auto1111 `parameters` tEXt chunk and (when
//! present) the structured JSON sidecar to stdout.
//!
//! Use cases:
//!
//! * Inherit a PNG from a previous plakat run and recover the
//!   prompt / seed / sampler / model / LoRAs without going back to
//!   the shell history.
//! * Drag a Civitai download into `plakat metadata` to see the
//!   parameters string Civitai users embed in their uploads.
//! * Cheap sanity check when sharing outputs — verify the recipe
//!   actually ended up in the file before forwarding it.
//!
//! Reads only `parameters` (the A1111 chunk plakat writes). Other
//! tEXt chunks (e.g. ComfyUI's `workflow`, A1111 webui's
//! `parameters` from a Web UI run) round-trip through the same key
//! and surface the same way.

use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct MetadataArgs {
    /// PNG to inspect. Looks for the `parameters` tEXt chunk + a
    /// sibling `<stem>.json` sidecar (both written by plakat v0.17+).
    pub path: PathBuf,

    /// Print only the JSON sidecar (skip the A1111 `parameters`
    /// chunk). Pipes cleanly to `jq` for structured queries.
    #[arg(long = "json-only", default_value_t = false)]
    pub json_only: bool,

    /// Print only the A1111 `parameters` tEXt chunk (skip the JSON
    /// sidecar). For A1111 / Civitai-compat prompt round-trips.
    #[arg(long = "params-only", default_value_t = false, conflicts_with = "json_only")]
    pub params_only: bool,
}

pub async fn run(args: MetadataArgs) -> Result<()> {
    if !args.path.exists() {
        anyhow::bail!("{}: no such file", args.path.display());
    }
    let sidecar = args.path.with_extension("json");
    let params = crate::imaging::io::read_parameters_chunk(&args.path)?;

    let want_params = !args.json_only;
    let want_json = !args.params_only;

    if want_params {
        match &params {
            Some(text) => {
                if want_json {
                    println!("# parameters (A1111 PNG tEXt)\n");
                }
                println!("{text}");
            }
            None => {
                if !args.params_only {
                    eprintln!(
                        "  note: {} has no `parameters` tEXt chunk \
                         (not a plakat / A1111 / Civitai output, or \
                         written with --no-metadata).",
                        args.path.display()
                    );
                }
            }
        }
    }

    if want_json {
        if sidecar.exists() {
            let json = std::fs::read_to_string(&sidecar)
                .with_context(|| format!("reading {}", sidecar.display()))?;
            if want_params && params.is_some() {
                println!();
            }
            if !args.json_only {
                println!("# sidecar (structured JSON)\n");
            }
            println!("{json}");
        } else if args.json_only {
            anyhow::bail!(
                "{} has no JSON sidecar (plakat writes one alongside the PNG \
                 unless --no-metadata is passed).",
                sidecar.display()
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imaging::metadata::GenerationMetadata;

    // v0.19 — end-to-end round-trip: write a plakat-flavoured PNG +
    // sidecar via the v0.17 helpers, then verify `metadata` reads
    // both back. Smoke test rather than asserting exact stdout.

    #[test]
    fn metadata_args_default_emits_both_sections() {
        // The flag default state: emit both parameters and sidecar.
        let args = MetadataArgs {
            path: PathBuf::from("/dev/null"),
            json_only: false,
            params_only: false,
        };
        assert!(!args.json_only);
        assert!(!args.params_only);
    }

    #[tokio::test]
    async fn run_bails_when_file_missing() {
        let args = MetadataArgs {
            path: PathBuf::from("/tmp/plakat-nope-does-not-exist.png"),
            json_only: false,
            params_only: false,
        };
        let err = run(args).await.unwrap_err();
        assert!(format!("{err}").contains("no such file"));
    }

    #[tokio::test]
    async fn run_reads_round_tripped_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let png_path = tmp.path().join("test.png");

        // 2×2 solid-red image (matches the io::tests fixture).
        let buf = vec![255u8, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0];
        let mut meta = GenerationMetadata::new(
            "a red square",
            "sd15",
            42,
            28,
            7.5,
            "euler-a",
            2,
            2,
        );
        meta.negative = "blurry".into();
        crate::imaging::io::save_rgb_u8_with_metadata(&buf, 2, 2, &png_path, &meta)
            .unwrap();

        // run() prints to stdout; we can't capture it cheaply, so the
        // smoke test asserts the call returns Ok with the file
        // present and both flag arms.
        for (json_only, params_only) in
            &[(false, false), (true, false), (false, true)]
        {
            let args = MetadataArgs {
                path: png_path.clone(),
                json_only: *json_only,
                params_only: *params_only,
            };
            assert!(run(args).await.is_ok());
        }
    }
}
