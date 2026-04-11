#!/bin/bash -l
# Submit NSS (nested sampling) for all five megamaser galaxies to gpulong.
ROOT="/mnt/users/rstiskalek/CANDEL"
PYTHON="$ROOT/venv_gpu_candel/bin/python"

for GAL in CGCG074-064 NGC5765b NGC6264 NGC6323 UGC3789; do
    echo "Submitting $GAL..."
    addqueue -q gpulong -s -m 16 --gpus 1 \
        $PYTHON -u $ROOT/scripts/megamaser/run_maser_disk.py $GAL --sampler nss
done
