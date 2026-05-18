# _plakat.sh — autodiscovery helper for tutorial scripts.
#
# Source this from each script to set `$PLAKAT` to the resolved
# plakat binary. Search order:
#   1. `plakat` on $PATH (installed via `cargo install plakat`).
#   2. ./target/release/plakat (built via `cargo build --release`).
#   3. ./target/debug/plakat   (built via `cargo build`).
#
# Repo root is computed relative to this file, so the helper works
# regardless of the caller's cwd.

_zones_repo_root() {
    # Walk up from this script's directory to the workspace root.
    # Layout: <repo>/examples/tutorials/ZONES/scripts/_plakat.sh
    local self_dir
    self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    (cd "$self_dir/../../../.." && pwd)
}

if command -v plakat >/dev/null 2>&1; then
    PLAKAT=plakat
else
    _root="$(_zones_repo_root)"
    if [[ -x "$_root/target/release/plakat" ]]; then
        PLAKAT="$_root/target/release/plakat"
    elif [[ -x "$_root/target/debug/plakat" ]]; then
        PLAKAT="$_root/target/debug/plakat"
    else
        echo "error: plakat not found on PATH or in $_root/target/{release,debug}/" >&2
        echo "       build it first:  cargo build --release" >&2
        echo "       or install it:   cargo install plakat" >&2
        exit 1
    fi
    unset _root
fi

export PLAKAT
