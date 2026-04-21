#!/bin/bash -l
# Submit a CPU+MPI mode2 run (DE MAP or ultranest posterior) for one maser
# galaxy. Reads the real spot count from the data file (optionally capped by
# `--max-spots`) and submits either one-rank-per-spot or an explicit MPI
# layout via addqueue.
#
# Usage:
#   ./submit_mode2_mpi.sh --galaxy NGC4258 --method de|ns [options]
#
# The default queue is `redwood`; memory is 5 GB per CPU (from the config).
# If `--mpi-n` is omitted, submit on `1x64` by default.

set -euo pipefail

METHOD=""
GALAXY=""
QUEUE=""
MAX_SPOTS=""
MPI_N_SPEC=""
OUT_DIR=""
RESUME=0
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $0 --galaxy GAL --method {de,ns} [options]

Required:
  --galaxy GAL           CGCG074-064 | NGC4258 | NGC5765b | NGC6264 | NGC6323 | UGC3789
  --method de|ns         DE MAP or ultranest posterior

Options:
  -q, --queue QUEUE      addqueue queue (default: redwood)
  --resume               resume from the last checkpoint; requires --out-dir
  --de-popsize N         DE population size (default 150)
  --de-maxiter N         DE max generations (default 2000)
  --de-F F               DE mutation scale factor (default 0.7)
  --de-CR CR             DE crossover probability (default 0.9)
  --checkpoint-every N   DE: checkpoint every N generations (default 25)
  --ns-min-live N        NS: minimum live points kept by ultranest (default 400)
  --ns-max-ncalls N      NS: cap on total likelihood calls (default: unlimited)
  --ns-stepsampler MODE  NS proposal: none | region-slice | slice-mixture
  --ns-nsteps N          NS step-sampler steps (default: ndim in runner)
  --max-spots N          launch only the first N spots / ranks
  --mpi-n SPEC           raw addqueue -n spec (default: 1x64), e.g. 64, 1x64, 2x64
  --seed N
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --galaxy) GALAXY="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        -q|--queue) QUEUE="$2"; shift 2 ;;
        --resume)
            RESUME=1
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --mpi-n) MPI_N_SPEC="$2"; shift 2 ;;
        --max-spots)
            MAX_SPOTS="$2"
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --de-popsize|--de-maxiter|--de-F|--de-CR|--checkpoint-every|\
        --ns-min-live|--ns-max-ncalls|--ns-nsteps|--seed|\
        --n-phi-hv-high|--n-phi-hv-low|--n-phi-sys|\
        --n-r-local|--n-r-brute)
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --ns-stepsampler)
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --out-dir)
            OUT_DIR="$2"
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "$GALAXY" || -z "$METHOD" ]]; then
    usage >&2
    exit 1
fi
if [[ "$METHOD" != "de" && "$METHOD" != "ns" ]]; then
    echo "Error: --method must be de or ns" >&2
    exit 1
fi

ROOT="/mnt/users/rstiskalek/CANDEL"
PYTHON="$ROOT/venv_candel/bin/python"
RUNNER="$ROOT/scripts/megamaser/run_mode2_mpi.py"
CONFIG="$ROOT/scripts/megamaser/config_maser.toml"
cd "$ROOT"

if [[ -z "$OUT_DIR" ]]; then
    if [[ "$RESUME" -eq 1 ]]; then
        echo "Error: --resume requires --out-dir so the exact previous run is unambiguous." >&2
        exit 1
    fi
    STAMP="$(date +%Y%m%d_%H%M%S)"
    OUT_DIR="$ROOT/results/Megamaser/${GALAXY}_mode2_mpi_${METHOD}_${STAMP}_$$"
    EXTRA_ARGS+=("--out-dir" "$OUT_DIR")
fi

# Parse the real spot count from the data file, then apply `--max-spots`
# if requested. Memory and default queue still come from [mode2_mpi].
read N_SPOTS MEMORY DEFAULT_QUEUE ACTUAL_SPOTS < <("$PYTHON" - "$CONFIG" "$GALAXY" "${MAX_SPOTS:-}" <<'PY'
import contextlib
import io
import sys
import tomli
from candel.pvdata.megamaser_data import load_megamaser_spots

with open(sys.argv[1], "rb") as f:
    cfg = tomli.load(f)
galaxy = sys.argv[2]
max_spots = sys.argv[3]
mpi_sec = cfg.get("mode2_mpi", {})
gal_sec = mpi_sec.get("galaxies", {}).get(galaxy)
if gal_sec is None:
    sys.exit(f"Error: no [mode2_mpi.galaxies.{galaxy}] block in config")
gcfg = cfg["model"]["galaxies"].get(galaxy)
if gcfg is None:
    sys.exit(f"Error: no [model.galaxies.{galaxy}] block in config")
with contextlib.redirect_stdout(io.StringIO()):
    full = load_megamaser_spots(
        cfg["io"]["maser_data"]["root"], galaxy, v_sys_obs=gcfg["v_sys_obs"])
actual_n = int(full["n_spots"])
if max_spots:
    max_spots = int(max_spots)
    if max_spots <= 0:
        sys.exit("Error: --max-spots must be a positive integer")
    launch_n = min(actual_n, max_spots)
else:
    launch_n = actual_n
print(launch_n,
      mpi_sec.get("memory_gb_per_cpu", 5),
      mpi_sec.get("queue", "redwood"),
      actual_n)
PY
)

[[ -z "$QUEUE" ]] && QUEUE="$DEFAULT_QUEUE"
if [[ -z "$MPI_N_SPEC" ]]; then
    MPI_N_SPEC="1x64"
fi

MPI_RANKS=$("$PYTHON" - "$MPI_N_SPEC" <<'PY'
import sys
spec = sys.argv[1].strip()
if "x" in spec:
    a, b = spec.split("x", 1)
    n = int(a) * int(b)
else:
    n = int(spec)
if n <= 0:
    raise SystemExit("MPI rank count must be positive")
print(n)
PY
)

# MPI transport knobs: OFI MTL is broken on glamdring; force TCP/shm. See
# memory note `maser_mpi_mode2.md`. We export these so they propagate to the
# MPI ranks that `addqueue` launches via srun.
export OMP_NUM_THREADS=1
export OMPI_MCA_mtl="^ofi"
export OMPI_MCA_btl="self,vader,tcp"
export OMPI_MCA_pml="ob1"

PYCMD=("$PYTHON" "-u" "$RUNNER" "$GALAXY" "--method" "$METHOD" "--config-path" "$CONFIG")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    PYCMD+=("${EXTRA_ARGS[@]}")
fi

printf -v PYCMD_STR '%q ' "${PYCMD[@]}"

echo "Submitting $GALAXY ($METHOD, queue=$QUEUE, n_spots=$N_SPOTS, mem=${MEMORY}GB/cpu)"
if [[ "$N_SPOTS" != "$ACTUAL_SPOTS" ]]; then
    echo "  Truncated from $ACTUAL_SPOTS spots via --max-spots $N_SPOTS"
fi
echo "  Output dir: $OUT_DIR"
echo "  MPI layout: -n $MPI_N_SPEC ($MPI_RANKS ranks, $(printf '%.2f' "$("$PYTHON" - "$N_SPOTS" "$MPI_RANKS" <<'PY'
import sys
spots = int(sys.argv[1])
ranks = int(sys.argv[2])
print(spots / ranks)
PY
)") spots/rank on average)"
echo "  Command: $PYCMD_STR"
echo "  addqueue -n $MPI_N_SPEC runs that command as $MPI_RANKS MPI ranks "\
"(srun, no explicit mpirun)."
echo

addqueue -q "$QUEUE" -n "$MPI_N_SPEC" -m "$MEMORY" \
    -c "mode2-mpi $GALAXY $METHOD" \
    "${PYCMD[@]}"
