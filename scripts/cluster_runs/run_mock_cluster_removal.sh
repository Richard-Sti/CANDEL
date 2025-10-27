#!/bin/bash
# Example script to run mock cluster analysis with MPI
# Adjust paths and parameters as needed
#
# Usage: 
#   Local/direct: ./run_mock_cluster_removal.sh [NRANKS] [N_MOCKS_TOTAL] [--dipole-only]
#   Glamdring queue: ./run_mock_cluster_removal.sh --glamdring [NRANKS] [N_MOCKS_TOTAL] [--dipole-only]
#
#   NRANKS: Number of MPI ranks (default: 1)
#   N_MOCKS_TOTAL: Total number of mocks to generate (default: NRANKS, i.e., 1 per rank)
#   --dipole-only: Only run dipole inference (skip no-dipole and removal)
#   --glamdring: Submit to glamdring queue instead of running directly

# Check if running via glamdring queue
USE_GLAMDRING=false
if [[ "$1" == "--glamdring" ]]; then
    USE_GLAMDRING=true
    shift  # Remove --glamdring from arguments
fi

# Parse arguments
NRANKS=${1:-1}
N_MOCKS_TOTAL=""
DIPOLE_ONLY=""

# Check second argument
if [[ "$2" == "--dipole-only" ]]; then
    DIPOLE_ONLY="--dipole_only"
elif [[ -n "$2" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
    N_MOCKS_TOTAL="--n_mocks_total $2"
fi

# Check third argument
if [[ "$3" == "--dipole-only" ]]; then
    DIPOLE_ONLY="--dipole_only"
fi

# If glamdring mode, submit job and exit
if [[ "$USE_GLAMDRING" == "true" ]]; then
    echo "=========================================="
    echo "Submitting to Glamdring Queue"
    echo "=========================================="
    echo "Cores: $NRANKS"
    echo "Memory: 5 GB per core"
    echo "Queue: redwood"
    echo "=========================================="
    
    # Create a wrapper script that will be executed by the queue
    WRAPPER_SCRIPT="/tmp/run_mock_wrapper_$$.sh"
    cat > "$WRAPPER_SCRIPT" << 'EOF'
#!/bin/bash
# This wrapper is executed by addqueue and runs the actual analysis
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ~/code/CANDEL
source venv_candel/bin/activate
# OpenMPI TCP transport workaround for glamdring
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self
exec "$0.real" "$@"
EOF
    chmod +x "$WRAPPER_SCRIPT"
    # Submit to queue with TCP transport workaround for OpenMPI fabric issues
    # Set OMPI_MCA environment variables to force TCP transport
    addqueue -n $NRANKS -m 5 -q redwood /usr/bin/bash -c "cd ~/code/CANDEL && source venv_candel/bin/activate && /usr/bin/bash ~/code/CANDEL/scripts/cluster_runs/run_mock_cluster_removal.sh $NRANKS $2 $3"
    
    echo ""
    echo "Job submitted! Check status with 'qstat' or similar."
    echo "DEBUG: Exiting after addqueue submission (should not run mpiexec in submission shell)"
    exit 0
fi

# Configuration files
CONFIG_NODIPOLE="scripts/cluster_runs/mock_cluster_nodipole.toml"
CONFIG_DIPOLE="scripts/cluster_runs/mock_cluster_dipole.toml"

# Field reconstruction paths (Carrick2015)
FIELD_DENSITY="$HOME/code/CANDEL/data/fields/carrick2015_twompp_density.npy"
FIELD_VELOCITY="$HOME/code/CANDEL/data/fields/carrick2015_twompp_velocity.npy"

# Check if field data exists
if [ ! -f "$FIELD_DENSITY" ]; then
    echo "Error: Field density file does not exist: $FIELD_DENSITY"
    echo "Please update FIELD_DENSITY in this script"
    exit 1
fi

if [ ! -f "$FIELD_VELOCITY" ]; then
    echo "Error: Field velocity file does not exist: $FIELD_VELOCITY"
    echo "Please update FIELD_VELOCITY in this script"
    exit 1
fi

# Output directory
OUTPUT_DIR="results/mock_cluster_removal"

# Analysis parameters
NSAMPLES=275              # Number of clusters per mock
N_ITERATIONS=10           # Number of removal iterations
N_REMOVE_PER_ITER=1       # Clusters to remove per iteration
SEED_OFFSET=1000          # Starting seed

echo "=========================================="
echo "Mock Cluster Removal Analysis"
echo "=========================================="
echo "Mock Cluster Removal Analysis"
echo "=========================================="
echo "MPI ranks: $NRANKS"
if [ -n "$N_MOCKS_TOTAL" ]; then
    echo "Total mocks: $(echo $N_MOCKS_TOTAL | awk '{print $2}')"
else
    echo "Total mocks: $NRANKS (1 per rank)"
fi
echo "Clusters per mock: $NSAMPLES"
if [ -n "$DIPOLE_ONLY" ]; then
    echo "Mode: DIPOLE ONLY (no removal)"
else
    echo "Removal iterations: $N_ITERATIONS"
    echo "Remove per iteration: $N_REMOVE_PER_ITER"
fi
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (two levels up from scripts/cluster_runs/)
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"

echo "DEBUG: Script dir: $SCRIPT_DIR"
echo "DEBUG: Project root: $PROJECT_ROOT"

# Activate virtual environment
VENV_ACTIVATE="${PROJECT_ROOT}/venv_candel/bin/activate"
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "Error: Virtual environment activation script not found at: $VENV_ACTIVATE"
    exit 1
fi

echo "DEBUG: Activating venv from: $VENV_ACTIVATE"
source "$VENV_ACTIVATE"

# Get absolute path to Python executable
PYTHON_EXEC="${PROJECT_ROOT}/venv_candel/bin/python"

# Verify Python executable exists
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: Python executable not found at: $PYTHON_EXEC"
    exit 1
fi

# Add project root to PYTHONPATH so candel module can be found
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "Using Python: $PYTHON_EXEC"
echo "Project root: $PROJECT_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# Detect MPI implementation
MPI_VENDOR=$(mpiexec --version 2>&1 | head -1)
if [[ "$MPI_VENDOR" == *"Open MPI"* ]] || [[ "$MPI_VENDOR" == *"OpenRTE"* ]]; then
    echo "Detected OpenMPI"
    MPI_ENV_FLAGS="-x PATH -x PYTHONPATH -x VIRTUAL_ENV -x LD_LIBRARY_PATH"
elif [[ "$MPI_VENDOR" == *"Intel"* ]]; then
    echo "Detected Intel MPI"
    MPI_ENV_FLAGS="-genv PYTHONPATH $PYTHONPATH -genv VIRTUAL_ENV $VIRTUAL_ENV"
elif [[ "$MPI_VENDOR" == *"HYDRA"* ]] || [[ "$MPI_VENDOR" == *"MPICH"* ]]; then
    echo "Detected MPICH"
    MPI_ENV_FLAGS="-genv PYTHONPATH $PYTHONPATH -genv VIRTUAL_ENV $VIRTUAL_ENV"
else
    echo "Warning: Unknown MPI implementation ($MPI_VENDOR)"
    echo "Trying MPICH/Intel MPI flags (-genv)"
    MPI_ENV_FLAGS="-genv PYTHONPATH $PYTHONPATH -genv VIRTUAL_ENV $VIRTUAL_ENV"
fi
echo ""

# Add MPI tuning flags to avoid segfaults with OpenMPI 5.0.1 over fabric
# These flags force TCP transport which is slower but more stable
MPI_TUNING_FLAGS=""
if [[ "$MPI_VENDOR" == *"Open MPI"* ]]; then
    echo "Using TCP transport for stability (OpenMPI fabric issues on glamdring)"
    MPI_TUNING_FLAGS="--mca pml ob1 --mca btl tcp,self"
fi

# Run the analysis using full path to venv python
mpiexec --oversubscribe -n $NRANKS $MPI_ENV_FLAGS $MPI_TUNING_FLAGS \
    "$PYTHON_EXEC" scripts/cluster_runs/analyze_mocks_with_cluster_removal.py \
    --config_nodipole "$CONFIG_NODIPOLE" \
    --config_dipole "$CONFIG_DIPOLE" \
    --field_density "$FIELD_DENSITY" \
    --field_velocity "$FIELD_VELOCITY" \
    --nsamples $NSAMPLES \
    --n_remove_iterations $N_ITERATIONS \
    --n_remove_per_iteration $N_REMOVE_PER_ITER \
    --output_dir "$OUTPUT_DIR" \
    --seed_offset $SEED_OFFSET \
    $N_MOCKS_TOTAL \
    $DIPOLE_ONLY

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Analysis completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "=========================================="
    echo ""
    echo "To analyze results, see scripts/README_mock_cluster_removal.md"
else
    echo ""
    echo "=========================================="
    echo "Analysis failed with errors"
    echo "=========================================="
    exit 1
fi
