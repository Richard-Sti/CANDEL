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
    "$SRC_USER@$SRC_HOST:$REMOTE_PATH" \
    "$DEST_DIR/"

echo "[INFO] Saved to ${DEST_DIR}/$(basename "$REMOTE_PATH")"
