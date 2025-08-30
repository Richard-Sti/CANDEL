#!/bin/bash

# ---- command line arguments ----
if [ $# -lt 1 ]; then
    echo "Usage: $0 <catalogue>"
    exit 1
fi
catalogue="$1"
# --------------------------------

# ---- user variables ----
# reconstruction="Carrick2015"
reconstruction="manticore_2MPP_MULTIBIN_N256_DES_V2"
config="../runs/config.toml"
queue="berg"
ncpu=3
memory=64
env="/mnt/users/rstiskalek/CANDEL/venv_candel/bin/python"
# ------------------------

pythoncm="$env compute_los.py --catalogue $catalogue --reconstruction $reconstruction --config $config"
cm="addqueue -q $queue -n $ncpu -m $memory $pythoncm"

echo "Submitting:"
echo $cm
eval $cm
