#!/bin/bash
set -euo pipefail

# ---- local source ----
SRC_BASE="$HOME/Projects/CANDEL"

# ---- glamdring destination ----
DEST_USER="rstiskalek"
DEST_HOST="glamdring.physics.ox.ac.uk"
DEST_PATH="/mnt/users/rstiskalek/CANDEL"
SSH_KEY="$HOME/.ssh/glamdring"

usage() {
    echo "Usage: $0 [results|data]"
    exit 1
}

# ---- parse argument ----
if [[ $# -ne 1 ]]; then
    usage
fi

case "$1" in
    results)
        echo "[INFO] Syncing 'results' -> glamdring:${DEST_PATH}"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_BASE/results" \
          "$DEST_USER@$DEST_HOST:$DEST_PATH/"
        ;;
    data)
        echo "[INFO] Syncing 'data' -> glamdring:${DEST_PATH}"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_BASE/data" \
          "$DEST_USER@$DEST_HOST:$DEST_PATH/"
        ;;
    *)
        usage
        ;;
esac

echo "[INFO] Sync complete."