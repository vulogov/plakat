use anyhow::Result;
use clap::Parser;

use plakat::{cli, ui};

fn main() -> Result<()> {
    // Friendly crash reports for end users: in RELEASE builds, an unexpected
    // panic prints a calm message and writes a full report to a temp file
    // instead of dumping a raw Rust backtrace. DEBUG builds keep the standard
    // backtrace for development (human-panic is `debug_assertions`-gated); in a
    // release build, `RUST_BACKTRACE=1` also restores the default backtrace.
    human_panic::setup_panic!(
        human_panic::Metadata::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
            .authors(env!("CARGO_PKG_AUTHORS"))
            .homepage(env!("CARGO_PKG_HOMEPAGE"))
            .support(
                "Please report this by opening an issue at \
                 https://github.com/vulogov/plakat/issues and attaching the \
                 report file listed above."
            )
    );

    let cli = cli::Cli::parse();
    ui::logging::init(cli.verbose);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(cli::dispatch(cli))
}
