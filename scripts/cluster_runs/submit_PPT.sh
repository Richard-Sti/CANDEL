#!/bin/bash -l
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# OpenMPI TCP transport workaround for glamdring
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self

NMOCKS=${1:-10}
NPROCS=${2:-1}
SCRIPT_DIR=~/code/CANDEL/scripts/cluster_runs

source ~/code/CANDEL/venv_candel/bin/activate

TRUTH_CONFIG=$SCRIPT_DIR/mock_cluster_LTYT_dipH0_fixedtruth.toml

if [ "$NPROCS" -gt 1 ]; then
    RUN="mpirun -np $NPROCS python"
else
    RUN="python"
fi

# PPT 1: With Carrick2015 reconstruction
echo "=== PPT with Carrick2015 reconstruction ($NMOCKS mocks, $NPROCS procs) ==="
$RUN $SCRIPT_DIR/run_mocks.py \
    --config_dipole $SCRIPT_DIR/mock_cluster_LTYT_dipH0_nodensity2.toml \
    --config_truth $TRUTH_CONFIG \
    --output_dir results/PPT_carrick \
    --n_mocks_total $NMOCKS --nclusters 276 --dipole_only \
    --which_relation LTYT --num_samples 1000

# PPT 2: Without reconstruction (no peculiar velocities)
echo "=== PPT without reconstruction ($NMOCKS mocks, $NPROCS procs) ==="
$RUN $SCRIPT_DIR/run_mocks.py \
    --config_dipole $SCRIPT_DIR/mock_cluster_LTYT_dipH0_nofield.toml \
    --config_truth $TRUTH_CONFIG \
    --output_dir results/PPT_nofield \
    --n_mocks_total $NMOCKS --nclusters 276 --dipole_only \
    --which_relation LTYT --num_samples 1000
