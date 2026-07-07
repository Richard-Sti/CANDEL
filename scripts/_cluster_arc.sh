#!/bin/bash -l
# Arc HTC cluster profile: env needed on a compute node to run a CANDEL job.
# Sourced inside the sbatch job script. Modules come from local_config.toml
# via CANDEL_MODULES_ACTIVE (space-separated).

if command -v module >/dev/null 2>&1; then
    module --force purge
    for m in ${CANDEL_MODULES_ACTIVE:-}; do
        module add "$m"
    done
fi

export XLA_FLAGS="--xla_hlo_profile=false --xla_dump_to=/tmp/nowhere"
