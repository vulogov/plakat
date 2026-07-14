use std::path::PathBuf;

use anyhow::Result;
use clap::Args as ClapArgs;

/// `plakat photos [ROOT_DIR]` — TUI photo & image collection manager (RFC PHOTOS-1).
/// Browse → curate → edit → generate over an image library. Needs a graphics-capable terminal.
#[derive(ClapArgs, Debug)]
pub struct PhotosArgs {
    /// Library root. Defaults to `$PLAKAT_PHOTOS_ROOT`, then `~/Pictures`.
    #[arg(value_name = "ROOT_DIR")]
    pub root: Option<PathBuf>,

    /// Thumbnail size in pixels.
    #[arg(long = "thumb-size", default_value_t = 128)]
    pub thumb_size: u32,

    /// Thumbnail decode workers.
    #[arg(long = "thumb-workers", default_value_t = 4)]
    pub thumb_workers: usize,

    /// Disable the filesystem watcher (static snapshot).
    #[arg(long = "no-watch", default_value_t = false)]
    pub no_watch: bool,
}

/// Resolve the library root: explicit arg > `$PLAKAT_PHOTOS_ROOT` > `~/Pictures`.
fn resolve_root(arg: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = arg {
        return Ok(p);
    }
    if let Ok(env) = std::env::var("PLAKAT_PHOTOS_ROOT") {
        return Ok(PathBuf::from(env));
    }
    let home = std::env::var("HOME").map(PathBuf::from).map_err(|_| {
        anyhow::anyhow!("no library root given and $HOME is unset (pass a ROOT_DIR)")
    })?;
    Ok(home.join("Pictures"))
}

pub async fn run(args: PhotosArgs) -> Result<()> {
    let root = resolve_root(args.root)?;
    crate::photos::run_with(root, args.thumb_size).await
}
