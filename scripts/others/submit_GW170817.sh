#!/bin/bash -l
# Submit GW170817 PE with Bilby + JetFit afterglow constraint. Cluster (arc
# or glamdring) is picked up from `machine` in local_config.toml via
# _submit_lib.sh.
#
# Default settings match high_spin_PhenomPNRT (Abbott et al. 2019).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

queue=""
ncpu=32
memory=7
nlive=1000
maxmcmc=5000
nact=10
duration=128
fmin=23
sampling_frequency=4096
outdir="$ROOT/results/GW170817_vanilla"
extra_args="--no-jetfit --marginalize-psi --aligned-spin --marginalize-phase --marginalize-time"
local_mode=false
real_data=true
use_roq=false
duration_set=false
dry=false

usage() {
    cat <<EOF
usage: $(basename "$0") -q QUEUE [options] [-- extra_args...]

Submit GW170817 PE (Bilby + dynesty + JetFit) with an MPI pool of --ncpu ranks.

options:
  -q, --queue QUEUE       queue/partition (REQUIRED unless --local)
  -n, --ncpu NCPU         MPI ranks / npool (default: $ncpu)
  -m, --memory MEMORY     GB per job (default: $memory)
  --nlive N               dynesty live points (default: $nlive)
  --maxmcmc N             max MCMC per proposal (default: $maxmcmc)
  --nact N                autocorrelation lengths (default: $nact)
  --duration T            segment duration (default: $duration s)
  --fmin F                min frequency (default: $fmin Hz)
  --label NAME            output subdir under results/ (e.g. GW170817_vanilla)
  --outdir PATH           absolute output dir (default: $outdir)
  --no-jetfit             disable JetFit EM constraint
  --density-prior         density-weighted distance prior
  --roq                   use ROQ waveform
  --marginalize-psi       marginalise polarisation
  --n-psi N               psi quadrature points
  --aligned-spin          IMRPhenomD_NRTidal waveform
  --marginalize-phase     analytic phase marginalisation (requires aligned)
  --marginalize-time      FFT-grid time marginalisation
  --real-data             use GWOSC strain (default in submit mode)
  --local                 run locally, no submission
  --dry                   print submit command without submitting
  -h, --help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)              usage ;;
        -q|--queue)             queue="$2"; shift 2 ;;
        -n|--ncpu)              ncpu="$2"; shift 2 ;;
        -m|--memory)            memory="$2"; shift 2 ;;
        --nlive)                nlive="$2"; shift 2 ;;
        --maxmcmc)              maxmcmc="$2"; shift 2 ;;
        --nact)                 nact="$2"; shift 2 ;;
        --duration)             duration="$2"; duration_set=true; shift 2 ;;
        --fmin)                 fmin="$2"; shift 2 ;;
        --outdir)               outdir="$2"; shift 2 ;;
        --label)                outdir="$ROOT/results/$2"; shift 2 ;;
        --no-jetfit)            extra_args="$extra_args --no-jetfit"; shift ;;
        --density-prior)        extra_args="$extra_args --density-prior"; shift ;;
        --roq)                  extra_args="$extra_args --roq"; use_roq=true; shift ;;
        --marginalize-psi)      extra_args="$extra_args --marginalize-psi"; shift ;;
        --n-psi)                extra_args="$extra_args --n-psi $2"; shift 2 ;;
        --aligned-spin)         extra_args="$extra_args --aligned-spin"; shift ;;
        --marginalize-phase)    extra_args="$extra_args --marginalize-phase"; shift ;;
        --marginalize-time)     extra_args="$extra_args --marginalize-time"; shift ;;
        --real-data)            real_data=true; shift ;;
        --local)                local_mode=true; shift ;;
        --dry)                  dry=true; shift ;;
        *)                      extra_args="$extra_args $1"; shift ;;
    esac
done

if $use_roq && ! $duration_set; then duration=106.5; fi
if $use_roq && [[ "$fmin" -lt 25 ]]; then fmin=25; fi

if ! $local_mode && [[ -z "$queue" ]]; then
    echo "[ERROR] -q QUEUE is required (cluster=$CANDEL_CLUSTER)"; exit 1
fi

if $local_mode && $real_data; then
    real_data=false
    echo "[INFO] Local mode: defaulting to injection test (pass --real-data to override)"
fi
$real_data && extra_args="$extra_args --real-data"

echo "GW170817 PE submission"
echo "============================================================"
echo "  Cluster:     $CANDEL_CLUSTER"
if $local_mode; then
    echo "  Mode:        LOCAL"
else
    echo "  Mode:        SUBMIT (queue=$queue)"
fi
echo "  MPI ranks:   $ncpu"
echo "  nlive:       $nlive"
echo "  maxmcmc:     $maxmcmc"
echo "  nact:        $nact"
echo "  duration:    ${duration} s"
echo "  fmin:        ${fmin} Hz"
echo "  sampling_f:  ${sampling_frequency} Hz"
echo "  Output:      $outdir"
[[ -n "$extra_args" ]] && echo "  Extra args: $extra_args"
echo

read -rp "Proceed? [y/N]: " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborting."; exit 1
fi

if [[ ! "$outdir" = /* ]]; then
    echo "[ERROR] --outdir must be absolute" >&2; exit 1
fi
mkdir -p "$outdir"

pycmd="/usr/bin/stdbuf -oL -eL $CANDEL_PYTHON -u $ROOT/scripts/others/run_GW170817.py \
    --nlive $nlive \
    --maxmcmc $maxmcmc \
    --nact $nact \
    --duration $duration \
    --fmin $fmin \
    --sampling-frequency $sampling_frequency \
    --npool $ncpu \
    --outdir $outdir \
    $extra_args"

if $local_mode; then
    echo "Running locally..."
    eval "$pycmd"
else
    dry_flag=()
    $dry && dry_flag=(--dry)
    submit_job --queue "$queue" --mem "$memory" --mpi-n "1x$ncpu" \
        --name "GW170817" --logdir "$ROOT/scripts/others/logs" \
        "${dry_flag[@]}" -- $pycmd
fi

echo
echo "Done."
