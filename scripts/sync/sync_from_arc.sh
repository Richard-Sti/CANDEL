#!/bin/bash
set -euo pipefail

# ---- local destination ----
DEST_BASE="$HOME/Projects/CANDEL"   # /Users/rstiskalek/Projects/CANDEL

# ---- ARC source ----
SRC_ALIAS="arc-htc"                  # defined in ~/.ssh/config
SRC_HOME="/home/phys1997/CANDEL"
SRC_DATA="/data/phys-galsim/phys1997/CANDEL"

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
        echo "[INFO] Pulling 'results' from ${SRC_ALIAS}:${SRC_HOME} -> ${DEST_BASE}"
        rsync -avh --progress -e "ssh" \
          "${SRC_ALIAS}:${SRC_HOME}/results" \
          "${DEST_BASE}/"
        ;;
    data)
        echo "[INFO] Pulling 'data' from ${SRC_ALIAS}:${SRC_DATA} -> ${DEST_BASE}"
        rsync -avh --progress -e "ssh" \
          "${SRC_ALIAS}:${SRC_DATA}/data" \
          "${DEST_BASE}/"
        ;;
    *)
        usage
        ;;
esac

echo "[INFO] Sync complete."