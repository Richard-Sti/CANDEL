#!/bin/bash -l
# Submit a CPU+MPI mode2 run (DE MAP or ultranest posterior) for one maser
# galaxy. Cluster (arc or glamdring) is picked up from `machine` in
# local_config.toml via _submit_lib.sh.
#
# Reads the real spot count from the data file (optionally capped by
# --max-spots) and submits either one-rank-per-spot or an explicit MPI
# layout. Memory per CPU and default MPI layout come from [mode2_mpi] in
# config_maser.toml.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

METHOD=""
GALAXY=""
QUEUE=""
MAX_SPOTS=""
MPI_N_SPEC=""
OUT_DIR=""
RESUME=0
DRY=false
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $0 --galaxy GAL --method {de,ns} -q QUEUE [options]

Required:
  --galaxy GAL           CGCG074-064 | NGC4258 | NGC5765b | NGC6264 | NGC6323 | UGC3789
  --method de|ns         DE MAP or ultranest posterior
  -q, --queue QUEUE      queue/partition (falls back to [mode2_mpi].queue in config)

Options:
  --resume               resume from last checkpoint; requires --out-dir
  --de-popsize N         DE population size (default 150)
  --de-maxiter N         DE max generations (default 2000)
  --de-F F               DE mutation scale (default 0.7)
  --de-CR CR             DE crossover (default 0.9)
  --checkpoint-every N   DE: checkpoint every N gens (default 25)
  --ns-min-live N        NS: min live points (default 400)
  --ns-max-ncalls N      NS: cap on likelihood calls
  --ns-stepsampler MODE  NS proposal: none | region-slice | slice-mixture
  --ns-nsteps N          NS step-sampler steps
  --max-spots N          launch only first N spots
  --mpi-n SPEC           raw MPI layout (default: 1x64), e.g. 64, 1x64, 2x64
  --seed N
  --dry                  print submit command without submitting
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --galaxy) GALAXY="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        -q|--queue) QUEUE="$2"; shift 2 ;;
        --resume)
            RESUME=1; EXTRA_ARGS+=("$1"); shift ;;
        --mpi-n) MPI_N_SPEC="$2"; shift 2 ;;
        --max-spots)
            MAX_SPOTS="$2"; EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --de-popsize|--de-maxiter|--de-F|--de-CR|--checkpoint-every|\
        --ns-min-live|--ns-max-ncalls|--ns-nsteps|--seed|\
        --n-phi-hv-high|--n-phi-hv-low|--n-phi-sys|\
        --n-r-local|--n-r-brute)
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --ns-stepsampler)
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --out-dir)
            OUT_DIR="$2"; EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --dry) DRY=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "$GALAXY" || -z "$METHOD" ]]; then
    usage >&2; exit 1
fi
if [[ "$METHOD" != "de" && "$METHOD" != "ns" ]]; then
    echo "Error: --method must be de or ns" >&2; exit 1
fi

RUNNER="$ROOT/scripts/megamaser/run_mode2_mpi.py"
CONFIG="$ROOT/scripts/megamaser/config_maser.toml"
cd "$ROOT"

if [[ -z "$OUT_DIR" ]]; then
    if [[ "$RESUME" -eq 1 ]]; then
        echo "Error: --resume requires --out-dir" >&2; exit 1
    fi
    STAMP="$(date +%Y%m%d_%H%M%S)"
    OUT_DIR="$ROOT/results/Megamaser/${GALAXY}_mode2_mpi_${METHOD}_${STAMP}_$$"
    EXTRA_ARGS+=("--out-dir" "$OUT_DIR")
fi

read -r N_SPOTS MEMORY DEFAULT_QUEUE ACTUAL_SPOTS < <("$CANDEL_PYTHON" - "$CONFIG" "$GALAXY" "${MAX_SPOTS:-}" <<'PY'
import contextlib
import io
import sys
import tomli
from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.util import data_path

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
        data_path(cfg["io"]["maser_data"]["root"]),
        galaxy, v_sys_obs=gcfg["v_sys_obs"])
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
[[ -z "$MPI_N_SPEC" ]] && MPI_N_SPEC="1x64"

MPI_RANKS=$("$CANDEL_PYTHON" - "$MPI_N_SPEC" <<'PY'
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

# MPI transport knobs (glamdring: OFI MTL broken; force TCP/shm).
export OMP_NUM_THREADS=1
export OMPI_MCA_mtl="^ofi"
export OMPI_MCA_btl="self,vader,tcp"
export OMPI_MCA_pml="ob1"

PYCMD=("$CANDEL_PYTHON" "-u" "$RUNNER" "$GALAXY" "--method" "$METHOD" "--config-path" "$CONFIG")
(( ${#EXTRA_ARGS[@]} > 0 )) && PYCMD+=("${EXTRA_ARGS[@]}")

printf -v PYCMD_STR '%q ' "${PYCMD[@]}"

echo "Submitting $GALAXY ($METHOD) -> $CANDEL_CLUSTER:$QUEUE"
echo "  Spots: $N_SPOTS (of $ACTUAL_SPOTS), mem=${MEMORY}GB/cpu"
echo "  Output dir: $OUT_DIR"
echo "  MPI layout: $MPI_N_SPEC ($MPI_RANKS ranks)"
echo "  Command: $PYCMD_STR"
echo

dry_flag=()
$DRY && dry_flag=(--dry)

submit_job --queue "$QUEUE" --mem "$MEMORY" --mpi-n "$MPI_N_SPEC" \
    --name "mode2_${GALAXY}_${METHOD}" \
    "${dry_flag[@]}" -- "${PYCMD[@]}"
