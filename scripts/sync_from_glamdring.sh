#!/bin/bash
set -euo pipefail

# ---- local destination ----
DEST_BASE="$HOME/code/CANDEL"

# ---- glamdring source ----
SRC_USER="yasin"
SRC_HOST="glamdring.physics.ox.ac.uk"
SRC_PATH="/mnt/users/yasin/projects/CANDEL"
SSH_PASS="$(security find-generic-password -a yasin -s glamdring -w)"

usage() {
    echo "Usage: $0 [results|data|results/<subfolder>|data/<subfolder>]"
    exit 1
}

# ---- parse argument ----
if [[ $# -ne 1 ]]; then
    usage
fi

echo "[INFO] Ensuring local destination exists: ${DEST_BASE}"
mkdir -p "$DEST_BASE"

SYNC_TARGET="$1"

case "$SYNC_TARGET" in
    results|data|results/*|data/*)
        ROOT_DIR="${SYNC_TARGET%%/*}"
        SRC_DIR="$SRC_PATH/$SYNC_TARGET"
        if [[ "$SYNC_TARGET" == "$ROOT_DIR" ]]; then
            DEST_DIR="$DEST_BASE/"
        else
            DEST_DIR="$DEST_BASE/$ROOT_DIR/"
        fi
        echo "[INFO] Pulling '${SYNC_TARGET}' from glamdring -> ${DEST_DIR}"
        mkdir -p "$DEST_DIR"
        rsync -avh --progress -e "sshpass -p '$SSH_PASS' ssh" \
          "$SRC_USER@$SRC_HOST:$SRC_DIR" \
          "$DEST_DIR"
        ;;
    *)
        usage
        ;;
esac

echo "[INFO] Sync complete."
