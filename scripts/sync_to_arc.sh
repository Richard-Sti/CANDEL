#!/bin/bash
set -euo pipefail

# ---- local source ----
SRC_BASE="$HOME/Projects/CANDEL"

# ---- ARC destination ----
DEST_ALIAS="arc-htc"              # alias from ~/.ssh/config
DEST_HOME="/home/phys1997/CANDEL"
DEST_DATA="/data/phys-galsim/phys1997/CANDEL"

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
        echo "[INFO] Ensuring destination directory exists: $DEST_HOME"
        ssh "$DEST_ALIAS" "mkdir -p '$DEST_HOME'"
        echo "[INFO] Syncing 'results' to ${DEST_ALIAS}:${DEST_HOME}"
        rsync -avh --progress \
          -e "ssh" \
          "$SRC_BASE/results" \
          "$DEST_ALIAS:$DEST_HOME/"
        ;;
    data)
        echo "[INFO] Ensuring destination directory exists: $DEST_DATA"
        ssh "$DEST_ALIAS" "mkdir -p '$DEST_DATA'"
        echo "[INFO] Syncing 'data' to ${DEST_ALIAS}:${DEST_DATA}"
        rsync -avh --progress \
          -e "ssh" \
          "$SRC_BASE/data" \
          "$DEST_ALIAS:$DEST_DATA/"
        ;;
    *)
        usage
        ;;
esac

echo "[INFO] Sync complete."