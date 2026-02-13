#!/bin/bash
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# OpenMPI TCP transport workaround for glamdring
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self

NMOCKS=${1:-1}
RELATION=${2:-YT}

source ~/code/CANDEL/venv_candel/bin/activate

if [ "$RELATION" = "LTYT" ]; then
    CONFIG=~/code/CANDEL/scripts/cluster_runs/mock_cluster_LTYT_dipole.toml
    OUTDIR=results/mock_dipole_LTYT
else
    CONFIG=~/code/CANDEL/scripts/cluster_runs/mock_cluster_dipole.toml
    OUTDIR=results/mock_dipole
fi

exec python ~/code/CANDEL/scripts/cluster_runs/run_mocks.py \
    --n_mocks_total $NMOCKS --output_dir $OUTDIR --nclusters 275 \
    --num_samples 250 --dipole_only --which_relation $RELATION \
    --config_dipole $CONFIG
