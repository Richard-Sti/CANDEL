#!/bin/bash
set -euo pipefail

# ---- local source ----
SRC_BASE="$HOME/Projects/CANDEL"

# ---- ARC destination ----
DEST_ALIAS="arc-htc"              # alias from ~/.ssh/config
DEST_HOME="/home/phys1997/CANDEL"
DEST_DATA="/data/phys-galsim/phys1997/CANDEL"

usage() {
    echo "Usage: $0 [results|data] [--delete]"
    echo "  --delete : show and remove files on remote not present locally"
    exit 1
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
fi

TARGET=$1
DELETE_FLAG=${2:-}

do_rsync() {
    local src=$1
    local dest=$2
    local desc=$3

    ssh "$DEST_ALIAS" "mkdir -p '$dest'"

    if [[ "$DELETE_FLAG" == "--delete" ]]; then
        echo "[INFO] Showing files that would be deleted from ${DEST_ALIAS}:${dest}:"
        rsync -avh --dry-run --delete --itemize-changes -e "ssh" "$src/" "$DEST_ALIAS:$dest/"
        echo
        read -p "Proceed with deleting the above files? [y/N] " answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            echo "[INFO] Syncing with deletions enabled"
            rsync -avh --progress --delete -e "ssh" "$src/" "$DEST_ALIAS:$dest/"
        else
            echo "[INFO] Aborted by user."
            exit 0
        fi
    else
        echo "[INFO] Syncing without deletions"
        rsync -avh --progress -e "ssh" "$src/" "$DEST_ALIAS:$dest/"
    fi
}

case "$TARGET" in
    results)
        do_rsync "$SRC_BASE/results" "$DEST_HOME/results" "results"
        ;;
    data)
        do_rsync "$SRC_BASE/data" "$DEST_DATA/data" "data"
        ;;
    *)
        usage
        ;;
esac

echo "[INFO] Sync complete."








# #!/bin/bash
# set -euo pipefail

# # ---- local source ----
# SRC_BASE="$HOME/Projects/CANDEL"

# # ---- ARC destination ----
# DEST_ALIAS="arc-htc"              # alias from ~/.ssh/config
# DEST_HOME="/home/phys1997/CANDEL"
# DEST_DATA="/data/phys-galsim/phys1997/CANDEL"

# usage() {
#     echo "Usage: $0 [results|data]"
#     exit 1
# }

# # ---- parse argument ----
# if [[ $# -ne 1 ]]; then
#     usage
# fi

# case "$1" in
#     results)
#         echo "[INFO] Ensuring destination directory exists: $DEST_HOME"
#         ssh "$DEST_ALIAS" "mkdir -p '$DEST_HOME'"
#         echo "[INFO] Syncing 'results' to ${DEST_ALIAS}:${DEST_HOME}"
#         rsync -avh --progress \
#           -e "ssh" \
#           "$SRC_BASE/results" \
#           "$DEST_ALIAS:$DEST_HOME/"
#         ;;
#     data)
#         echo "[INFO] Ensuring destination directory exists: $DEST_DATA"
#         ssh "$DEST_ALIAS" "mkdir -p '$DEST_DATA'"
#         echo "[INFO] Syncing 'data' to ${DEST_ALIAS}:${DEST_DATA}"
#         rsync -avh --progress \
#           -e "ssh" \
#           "$SRC_BASE/data" \
#           "$DEST_ALIAS:$DEST_DATA/"
#         ;;
#     *)
#         usage
#         ;;
# esac

# echo "[INFO] Sync complete."