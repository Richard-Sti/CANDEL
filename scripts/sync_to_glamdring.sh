#!/bin/bash
set -euo pipefail

# ---- local source ----
SRC_BASE="$HOME/projects/CANDEL"

# ---- glamdring destination ----
DEST_USER="yasin"
DEST_HOST="glamdring.physics.ox.ac.uk"
DEST_PATH="/mnt/users/yasin/projects/CANDEL"
SSH_KEY="$HOME/.ssh/id_ed25519"

usage() {
    echo "Usage: $0 [results|data|scripts|all|results/<subfolder>|data/<subfolder>|scripts/<subfolder>]"
    exit 1
}

# ---- parse argument ----
if [[ $# -ne 1 ]]; then
    usage
fi

SYNC_TARGET="$1"

case "$SYNC_TARGET" in
    results|data|scripts)
        echo "[INFO] Syncing '${SYNC_TARGET}' -> glamdring:${DEST_PATH}"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_BASE/$SYNC_TARGET" \
          "$DEST_USER@$DEST_HOST:$DEST_PATH/"
        ;;
    results/*|data/*|scripts/*)
        ROOT_DIR="${SYNC_TARGET%%/*}"
        SRC_DIR="$SRC_BASE/$SYNC_TARGET"
        DEST_DIR="$DEST_PATH/$ROOT_DIR/"
        echo "[INFO] Syncing '${SYNC_TARGET}' -> glamdring:${DEST_DIR}"
        rsync -avh --progress -e "ssh -i $SSH_KEY" \
          "$SRC_DIR" \
          "$DEST_USER@$DEST_HOST:$DEST_DIR"
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
