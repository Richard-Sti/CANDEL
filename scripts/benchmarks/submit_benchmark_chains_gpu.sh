#!/bin/bash -l
# Submit the chain-method GPU benchmark to Glamdring.
# Usage: bash submit_benchmark_chains_gpu.sh [QUEUE]
# QUEUE defaults to cmbgpu (RTX 3090, 24 GB).

QUEUE=${1:-cmbgpu}

case "$QUEUE" in
    gpulong) GPUTYPE="rtx2080with12gb" ;;
    cmbgpu)  GPUTYPE="rtx3090with24gb" ;;
    optgpu)  GPUTYPE="rtxa6000with48gb" ;;
    *) echo "Unknown queue: $QUEUE"; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$(dirname "$SCRIPT_DIR")/runs/local_config.toml"

get_toml_key() {
    local key="$1" file="$2"
    grep -m1 "^${key}" "$file" 2>/dev/null | sed 's/.*=\s*"\?\([^"]*\)"\?.*/\1/' | tr -d ' '
}

PYTHON=$(get_toml_key "python_exec_gpu" "$CONFIG")
[[ -z "$PYTHON" ]] && PYTHON=$(get_toml_key "python_exec" "$CONFIG")
[[ -z "$PYTHON" ]] && { echo "ERROR: python_exec not set in $CONFIG"; exit 1; }

ROOT=$(get_toml_key "root_main" "$CONFIG")
[[ -z "$ROOT" ]] && { echo "ERROR: root_main not set in $CONFIG"; exit 1; }

BENCHMARK="$ROOT/scripts/benchmarks/benchmark_chains_gpu.py"
OUTPUT="$ROOT/scripts/benchmarks/results_chains_gpu.json"

CMD="$PYTHON $BENCHMARK --output $OUTPUT"
echo "Queue:   $QUEUE ($GPUTYPE)"
echo "Python:  $PYTHON"
echo "Command: $CMD"
echo

addqueue -q "$QUEUE" -s -m 16 --gpus 1 --gputype "$GPUTYPE" $CMD
