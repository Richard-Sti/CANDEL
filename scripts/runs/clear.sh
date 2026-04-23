#!/bin/bash
#
# Remove cluster job-output files left in scripts/runs/ by submit.sh
# (arc: logs-<jobid>.{out,err}; glamdring: python-<jobid>.{out,err}).

set -e

log_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[INFO] Removing cluster job-output files from '$log_dir/'..."
rm -vf "$log_dir"/logs-*.out "$log_dir"/logs-*.err \
       "$log_dir"/python-*.out "$log_dir"/python-*.err
