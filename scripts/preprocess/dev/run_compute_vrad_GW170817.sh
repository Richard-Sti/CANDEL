#!/bin/bash -l

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# ---- user variables ----
config="$ROOT/scripts/runs/configs/config.toml"
queue="berg"
ncpu=10
memory=32
python_exec="$ROOT/venv_candel/bin/python"
script="$ROOT/scripts/preprocess/dev/compute_vrad_GW170817.py"
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
            cmd=(addqueue -q "$queue" -n "$ncpu_use" -m "$memory"
                 "$python_exec" "$script"
                 --reconstruction "$reconstruction" --config "$config")
            echo "Submitting: $reconstruction"
            printf '  %q' "${cmd[@]}"
            echo
            "${cmd[@]}"
            echo
        done
        ;;
    2)
        for reconstruction in "${reconstructions[@]}"; do
            echo "Running: $reconstruction"
            "$python_exec" "$script" \
                --reconstruction "$reconstruction" --config "$config"
            echo
        done
        ;;
    *)
        echo "Aborting."
        exit 0
        ;;
esac
