#!/bin/bash
set -euo pipefail

# ---- local destination ----
DEST_BASE="$HOME/code/CANDEL"

# ---- glamdring source ----
SRC_USER="yasin"
SRC_HOST="glamdring.physics.ox.ac.uk"
SRC_PATH="/mnt/users/yasin/code/CANDEL"
SSH_KEY="$HOME/.ssh/id_ed25519"

usage() {
    echo "Usage: $0 [results|data]"
    exit 1
}

# ---- parse argument ----
if [[ $# -ne 1 ]]; then
    usage
fi

echo "[INFO] Ensuring local destination exists: ${DEST_BASE}"
mkdir -p "$DEST_BASE"

case "$1" in
    results)
        echo "[INFO] Pulling 'results' from glamdring -> ${DEST_BASE}"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_USER@$SRC_HOST:$SRC_PATH/results" \
          "$DEST_BASE/"
        ;;
    data)
        echo "[INFO] Pulling 'data' from glamdring -> ${DEST_BASE}"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_USER@$SRC_HOST:$SRC_PATH/data" \
          "$DEST_BASE/"
        ;;
    *)
        usage
        ;;
esac

echo "[INFO] Sync complete."