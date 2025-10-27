#!/bin/bash
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# OpenMPI TCP transport workaround for glamdring
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self

NMOCKS=${1:-1}

DIPOLE_ONLY=""
if [[ "$2" == "--dipole_only" ]]; then
	DIPOLE_ONLY="--dipole_only"
fi

source ~/code/CANDEL/venv_candel/bin/activate

exec python ~/code/CANDEL/scripts/cluster_runs/run_mocks.py \
    --n_mocks_total $NMOCKS $DIPOLE_ONLY
