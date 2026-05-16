use tracing_subscriber::{EnvFilter, fmt, prelude::*};

pub fn init(verbosity: u8) {
    let default = match verbosity {
        0 => "plakat=info,warn",
        1 => "plakat=debug,info",
        _ => "plakat=trace,debug",
    };
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default));
    let layer = fmt::layer()
        .with_target(false)
        .without_time()
        .with_ansi(true);
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(layer)
        .try_init();
}
