#!/bin/bash
#SBATCH -p gen -C rome -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=gen_mocks
#SBATCH --output=logs/generate_mocks-%j.out
#SBATCH --error=logs/generate_mocks-%j.err
#SBATCH --mail-user=rstiskalek@flatironinstitute.org
#SBATCH --mail-type=FAIL,END

set -e

echo "[INFO] Job started on $(hostname) at $(date)"
echo "[INFO] Running generate_mocks.py"

# Optional: Load modules if needed
# module load python

python generate_mocks.py

echo "[INFO] Job finished at $(date)"