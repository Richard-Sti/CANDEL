#!/bin/bash
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# OpenMPI TCP transport workaround for glamdring
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self

NMOCKS=${1:-1}

source ~/code/CANDEL/venv_candel/bin/activate

exec python ~/code/CANDEL/scripts/cluster_runs/run_mocks.py \
    --n_mocks_total $NMOCKS --output_dir results/prior_sampled_mocks_Rdist --nclusters 275 --num_samples 2000 --dipole_only
