#!/bin/bash -l
# Benchmark maser likelihood evaluation on GPU.
# Times Mode 1 (potential_fn + grad) and Mode 2 (log_likelihood_fn).

QUEUE="optgpu"
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash $0 [-q QUEUE] [ARGS...]"
            echo ""
            echo "Benchmark maser likelihood evaluation on GPU."
            echo "Times the functions the production samplers call:"
            echo "  Mode 1 (NUTS): potential_fn + value_and_grad"
            echo "  Mode 2 (NSS):  log_likelihood_fn"
            echo ""
            echo "Default: all galaxies in their configured mode."
            echo ""
            echo "  -q QUEUE      GPU queue (default: optgpu)"
            echo ""
            echo "Python toggles (forwarded as-is):"
            echo "  --galaxies G [G ...]  subset to run"
            echo "      choices: CGCG074-064 NGC4258 NGC5765b NGC6264 NGC6323 UGC3789"
            echo "  --n-repeats N         timed iterations (default: 100)"
            echo "  --n-warmup N          extra warm-up calls after JIT (default: 5)"
            echo "  --spot-batch N        spot-axis chunk for Mode 2 (0=off, default: 0)"
            echo "  --f64                 use float64 (default: float32)"
            echo ""
            echo "Examples:"
            echo "  bash $0 --galaxies NGC4258"
            echo "  bash $0 --galaxies NGC5765b --n-repeats 200"
            echo "  bash $0                              # all galaxies"
            echo ""
            echo "For full Python help:"
            echo "  python scripts/megamaser/convergence/benchmark_likelihood.py --help"
            exit 0
            ;;
        -q) QUEUE="$2"; shift 2 ;;
        *) PASS_ARGS+=("$1"); shift ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=../../_submit_lib.sh
source "$ROOT_DIR/scripts/_submit_lib.sh"
if [[ "$CANDEL_CLUSTER" != "glamdring" ]]; then
    echo "[ERROR] This script is glamdring-only (machine=$CANDEL_CLUSTER)" >&2
    exit 1
fi
PYTHON="$CANDEL_PYTHON"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export JAX_PLATFORMS=cuda

echo "Submitting benchmark_likelihood -> $QUEUE"
echo "JAX: XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda"
echo "Args: ${PASS_ARGS[*]:-(defaults)}"

addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
    $PYTHON -u "$ROOT_DIR/scripts/megamaser/convergence/benchmark_likelihood.py" \
    "${PASS_ARGS[@]}"
