#!/bin/bash
set -e

frozen_dir="/mnt/home/${USER}/frozen_candel/current"
src_dir="/mnt/home/${USER}/CANDEL/candel"

echo "[INFO] Freezing package to: $frozen_dir"
rm -rf "$frozen_dir"
mkdir -p "$frozen_dir"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$src_dir/" "$frozen_dir/"
echo "$frozen_dir"