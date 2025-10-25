#!/bin/bash
# Example script to run mock cluster analysis with MPI
# Adjust paths and parameters as needed
#
# Usage: ./run_mock_cluster_removal.sh [NRANKS] [N_MOCKS_TOTAL] [--dipole-only]
#   NRANKS: Number of MPI ranks (default: 1)
#   N_MOCKS_TOTAL: Total number of mocks to generate (default: NRANKS, i.e., 1 per rank)
#   --dipole-only: Only run dipole inference (skip no-dipole and removal)

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

# Configuration files
CONFIG_NODIPOLE="scripts/cluster_runs/mock_cluster_nodipole.toml"
CONFIG_DIPOLE="scripts/cluster_runs/mock_cluster_dipole.toml"

# Field reconstruction paths (Carrick2015)
FIELD_DENSITY="/Users/yasin/code/CANDEL/data/fields/carrick2015_twompp_density.npy"
FIELD_VELOCITY="/Users/yasin/code/CANDEL/data/fields/carrick2015_twompp_velocity.npy"

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

# Activate virtual environment
source venv_candel/bin/activate

# Run the analysis
mpiexec -n $NRANKS python scripts/cluster_runs/analyze_mocks_with_cluster_removal.py \
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
