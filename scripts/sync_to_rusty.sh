#!/bin/bash

set -e  # Exit on error

SRC_BASE="/Users/rstiskalek/Projects/CANDEL"
DEST_USER="rstiskalek"
DEST_HOST="gateway.flatironinstitute.org"
DEST_PORT=61022
DEST_PATH="/mnt/home/rstiskalek/ceph/CANDEL"

echo "[INFO] Syncing 'results' and 'data' with progress..."
rsync -avh --progress -e "ssh -p $DEST_PORT" \
  "$SRC_BASE/results" "$SRC_BASE/data" \
  "$DEST_USER@$DEST_HOST:$DEST_PATH/"

echo "[INFO] Sync complete."

