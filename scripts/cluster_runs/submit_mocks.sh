#!/bin/bash
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# OpenMPI TCP transport workaround for glamdring
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self

NMOCKS=${1:-1}
RELATION=LTYT

source ~/code/CANDEL/venv_candel/bin/activate

CONFIG=~/code/CANDEL/scripts/cluster_runs/mock_cluster_LTYT_dipH0_nodensity2.toml
OUTDIR=results/mock_cluster_LTYT_dipH0

exec python ~/code/CANDEL/scripts/cluster_runs/run_mocks.py \
    --n_mocks_total $NMOCKS --output_dir $OUTDIR --nclusters 275 \
    --num_samples 1000 --dipole_only --which_relation $RELATION \
    --config_dipole $CONFIG
