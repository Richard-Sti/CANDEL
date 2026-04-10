#!/bin/bash
set -euo pipefail

SRC_USER="rstiskalek"
SRC_HOST="glamdring.physics.ox.ac.uk"
SSH_KEY="$HOME/.ssh/glamdring"
DEST_DIR="$HOME/Downloads"

usage() {
    echo "Usage: $0 <remote_path> [local_dest_dir]"
    echo "  e.g. $0 /mnt/users/rstiskalek/CANDEL/results/CH0/plot.png"
    echo "  e.g. $0 /mnt/users/rstiskalek/CANDEL/results/CH0/plot.png ~/Desktop"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

REMOTE_PATH="$1"
[[ $# -ge 2 ]] && DEST_DIR="$2"

mkdir -p "$DEST_DIR"

echo "[INFO] Fetching ${REMOTE_PATH} -> ${DEST_DIR}/"
rsync -avh --progress -e "ssh -i $SSH_KEY" \
    "${SRC_USER}@${SRC_HOST}:${REMOTE_PATH}" \
    "$DEST_DIR/" | tee "$TMPDIR/sfg_out.txt"

# Count transferred files from rsync --itemize-changes style output.
# Keep only lines that look like relative file paths (no spaces at start,
# no summary lines, no directories).
FILES=$(awk '/^[^ ]/ && !/\/$/ && !/^(sending|receiving|sent |total |Transfer|created|$)/' \
    "$TMPDIR/sfg_out.txt" || true)
NFILES=$(printf '%s\n' "$FILES" | grep -c . || true)

echo "[INFO] ${NFILES} file(s) transferred."
if [[ $NFILES -gt 0 && $NFILES -le 100 ]]; then
    echo "$FILES" | while read -r f; do echo "${DEST_DIR}/${f}"; done
fi
rm -f "$TMPDIR/sfg_out.txt"
