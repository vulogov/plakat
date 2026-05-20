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
        echo "       build it first (Apple Silicon, GPU acceleration):" >&2
        echo "         cargo build --release --features metal" >&2
        echo "       on NVIDIA / Linux:" >&2
        echo "         cargo build --release --features cuda" >&2
        echo "       or install it globally:" >&2
        echo "         cargo install plakat --features metal" >&2
        exit 1
    fi
    unset _root
fi

# Sanity-check that the binary was built with a GPU backend. A
# CPU-only build silently falls back to F32 inference and chews
# through 12-16 GB of RAM for an SD 1.5 generation — OOM-ing even
# on 24 GB Macs. The check looks for the device-selection log
# strings that are `#[cfg(feature = "metal")]` / `"cuda"` gated:
# their presence in the binary proves the corresponding backend
# was compiled in.
if [[ "$PLAKAT" != "plakat" ]] && command -v strings >/dev/null 2>&1; then
    _bin_strings=$(strings "$PLAKAT" 2>/dev/null || true)
    case "$(uname -s)" in
        Darwin)
            if ! grep -q 'Using Metal device' <<<"$_bin_strings"; then
                echo "warn: $PLAKAT was built without Metal support." >&2
                echo "      SD generation will run on CPU in F32 and may OOM even on" >&2
                echo "      24 GB Macs. Rebuild with: cargo build --release --features metal" >&2
                echo
            fi
            ;;
        Linux)
            if ! grep -q 'Using CUDA device' <<<"$_bin_strings"; then
                echo "warn: $PLAKAT was built without CUDA support." >&2
                echo "      Rebuild with: cargo build --release --features cuda" >&2
                echo
            fi
            ;;
    esac
    unset _bin_strings
fi

export PLAKAT
