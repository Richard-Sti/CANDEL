#!/bin/bash
set -euo pipefail

# ---- local source ----
SRC_BASE="$HOME/code/CANDEL"

# ---- glamdring destination ----
DEST_USER="yasin"
DEST_HOST="glamdring.physics.ox.ac.uk"
DEST_PATH="/mnt/users/yasin/code/CANDEL"
SSH_KEY="$HOME/.ssh/id_ed25519"

usage() {
    echo "Usage: $0 [results|data|scripts|all]"
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
    scripts)
        echo "[INFO] Syncing 'scripts' -> glamdring:${DEST_PATH}"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_BASE/scripts" \
          "$DEST_USER@$DEST_HOST:$DEST_PATH/"
        ;;
    all)
        echo "[INFO] Syncing 'data', 'scripts', and 'configs' -> glamdring:${DEST_PATH}"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_BASE/data" \
          "$DEST_USER@$DEST_HOST:$DEST_PATH/"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_BASE/scripts" \
          "$DEST_USER@$DEST_HOST:$DEST_PATH/"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_BASE/configs" \
          "$DEST_USER@$DEST_HOST:$DEST_PATH/"
        ;;
    *)
        usage
        ;;
esac

echo "[INFO] Sync complete."