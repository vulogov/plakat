use anyhow::Result;
use clap::Parser;

mod cli;
mod config;
mod device;
mod hf;
mod imaging;
mod pipelines;
mod prompt;
mod ui;

fn main() -> Result<()> {
    let cli = cli::Cli::parse();
    ui::logging::init(cli.verbose);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(cli::dispatch(cli))
}
