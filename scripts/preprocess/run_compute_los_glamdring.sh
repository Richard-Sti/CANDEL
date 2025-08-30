#!/bin/bash

# ---- user variables ----
# reconstruction="Carrick2015"
# reconstruction="manticore_2MPP_MULTIBIN_N256_DES_V2"
reconstruction="CLONES"
config="../runs/config.toml"
queue="cmb"
ncpu=1
memory=64
env="/mnt/users/rstiskalek/CANDEL/venv_candel/bin/python"
# ------------------------

# ---- command line arguments / defaults ----
if [ $# -lt 1 ]; then
    catalogues=("CF4" "2MTF" "SFI" "Foundation" "LOSS")

    echo "No catalogue specified."
    echo "Jobs will be submitted with:"
    echo "  Reconstruction: $reconstruction"
    echo "  Queue: $queue"
    echo "  CPUs: $ncpu"
    echo "  Memory: ${memory} GB"
    echo "  Config: $config"
    echo
    echo "Default catalogue list:"
    for c in "${catalogues[@]}"; do
        echo "  - $c"
    done
    echo
    read -p "Submit jobs for ALL of these catalogues? [y/N]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborting."
        exit 1
    fi
else
    catalogues=("$@")
fi
# ------------------------------------------

for catalogue in "${catalogues[@]}"; do
    pythoncm="$env compute_los.py --catalogue $catalogue --reconstruction $reconstruction --config $config"
    cm="addqueue -q $queue -n $ncpu -m $memory $pythoncm"

    echo "Submitting catalogue: $catalogue"
    echo "$cm"
    eval "$cm"
    echo
done
