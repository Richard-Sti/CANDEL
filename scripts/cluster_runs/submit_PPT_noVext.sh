#!/bin/bash -l
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self

NMOCKS=${1:-10}
SCRIPT_DIR=~/code/CANDEL/scripts/cluster_runs

source ~/code/CANDEL/venv_candel/bin/activate

exec python $SCRIPT_DIR/run_mocks.py \
    --config_dipole $SCRIPT_DIR/mock_cluster_LTYT_dipH0_nofield.toml \
    --config_truth $SCRIPT_DIR/mock_cluster_LTYT_dipH0_fixedtruth.toml \
    --output_dir results/PPT_nofield \
    --n_mocks_total $NMOCKS --nclusters 276 --dipole_only \
    --which_relation LTYT --num_samples 1000
