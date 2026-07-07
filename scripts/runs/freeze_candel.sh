#!/bin/bash -l
#
# Freeze the candel/ package and scripts/runs/main.py into a per-cluster
# install root (CANDEL_FROZEN_ROOT), so subsequent submissions run against
# a stable snapshot instead of the evolving source tree.
#
# Paths and cluster identity come from local_config.toml via _submit_lib.sh.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

src_dir="$CANDEL_ROOT/candel"
main_script="$CANDEL_ROOT/scripts/runs/main.py"
frozen_dir="$CANDEL_FROZEN_ROOT"

echo "[INFO] Freezing candel package + main.py"
echo "[INFO] Cluster: $CANDEL_CLUSTER"
echo "[INFO] From:    $src_dir"
echo "[INFO] To:      $frozen_dir"

rm -rf "$frozen_dir"
mkdir -p "$frozen_dir"

rsync -a --exclude '__pycache__' --exclude '*.pyc' "$src_dir" "$frozen_dir"
cp "$main_script" "$frozen_dir/main.py"

echo "[INFO] Frozen structure:"
if command -v tree >/dev/null 2>&1; then
    tree -L 2 "$frozen_dir"
else
    echo "[INFO] (Skipping tree output: 'tree' not found)"
    ls -l "$frozen_dir"
fi
