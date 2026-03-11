#!/bin/bash

# ---- user variables ----
config="../runs/config.toml"
queue="berg"
ncpu=10
memory=32
env="/mnt/users/rstiskalek/CANDEL/venv_candel/bin/python"
# ------------------------

reconstructions=("manticore_2MPP_MULTIBIN_N256_DES_V2")
# reconstructions=("Carrick2015")

echo "Reconstructions to process:"
for r in "${reconstructions[@]}"; do
    echo "  - $r"
done
echo

echo "How would you like to run?"
echo "  1) Submit to queue (addqueue -q $queue)"
echo "  2) Run on login node"
echo "  3) Abort"
read -p "Select [1/2/3]: " choice

case $choice in
    1)
        for reconstruction in "${reconstructions[@]}"; do
            # Carrick2015 has only 1 realisation, no need for multiple CPUs
            if [ "$reconstruction" == "Carrick2015" ]; then
                ncpu_use=1
            else
                ncpu_use=$ncpu
            fi
            cm="addqueue -q $queue -n $ncpu_use -m $memory $env compute_vrad_GW170817.py --reconstruction $reconstruction --config $config"
            echo "Submitting: $reconstruction"
            echo "  $cm"
            eval "$cm"
            echo
        done
        ;;
    2)
        for reconstruction in "${reconstructions[@]}"; do
            echo "Running: $reconstruction"
            $env compute_vrad_GW170817.py --reconstruction $reconstruction --config $config
            echo
        done
        ;;
    *)
        echo "Aborting."
        exit 0
        ;;
esac
