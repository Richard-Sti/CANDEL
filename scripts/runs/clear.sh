#!/bin/bash

set -e

log_dir="logs"

echo "[INFO] Removing SLURM job array output files from '$log_dir/'..."
rm -v "$log_dir"/logs-*.out "$log_dir"/logs-*.err \
       "$log_dir"/python-*.out "$log_dir"/python-*.err 2>/dev/null || true