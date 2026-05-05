#!/bin/bash
# Clear generated Reid MCMC result directories and collected plots/summaries.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEFAULT_TARGET="$ROOT/results/Megamaser/reid_mcmc"

TARGET="$DEFAULT_TARGET"
CONFIRM=false

usage() {
    cat <<EOF
Usage: bash $0 [options]

Clear generated Reid MCMC outputs. By default this is a dry run.

Options:
  --target DIR   Directory to clear (default: $DEFAULT_TARGET)
  --yes          Actually remove files/directories
  -h, --help

This removes direct children of the target directory, except hidden files.
The target directory itself is kept.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --yes) CONFIRM=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -d "$TARGET" ]]; then
    echo "No Reid result directory found: $TARGET"
    exit 0
fi

root_results="$ROOT/results"
target_real="$(readlink -f "$TARGET")"
root_results_real="$(readlink -f "$root_results")"
case "$target_real" in
    "$root_results_real"/*) ;;
    *)
        echo "[ERROR] Refusing to clear outside $root_results_real: $target_real" >&2
        exit 1
        ;;
esac

mapfile -d '' items < <(
    find "$target_real" -mindepth 1 -maxdepth 1 \
        ! -name '.*' \
        -print0 | sort -z
)

if [[ "${#items[@]}" -eq 0 ]]; then
    echo "No generated Reid outputs to clear in: $target_real"
    exit 0
fi

if $CONFIRM; then
    echo "Removing generated Reid outputs from: $target_real"
else
    echo "Dry run. Would remove generated Reid outputs from: $target_real"
    echo "Pass --yes to actually remove them."
fi

for item in "${items[@]}"; do
    echo "  $item"
done

if $CONFIRM; then
    rm -rf -- "${items[@]}"
fi
