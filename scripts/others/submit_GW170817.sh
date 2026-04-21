#!/bin/bash -l
#
# Submit GW170817 PE with Bilby + JetFit afterglow constraint.
#
# Default settings match the high_spin_PhenomPNRT analysis from
# Abbott et al. 2019 (Phys. Rev. X 9, 011001).
#
# Machine-specific settings (python_exec) are read from local_config.toml
# at the project root.

set -e

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

# Extract a TOML key from local_config.toml
get_toml_key() {
    local key="$1"
    grep -E "^${key} *= *" "$repo_root/local_config.toml" 2>/dev/null \
        | sed -E "s/^${key} *= *\"([^\"]+)\"$/\1/"
}

# ---- defaults (production: matches Abbott et al. 2019) ----
queue="cmb"
ncpu=32
memory=7
nlive=1000
maxmcmc=5000
nact=10
duration=128
fmin=23
sampling_frequency=4096
npool="$ncpu"
outdir="$repo_root/results/GW170817_vanilla"
extra_args="--no-jetfit --marginalize-psi --aligned-spin --marginalize-phase --marginalize-time"

usage() {
    cat <<EOF
usage: $(basename "$0") [-h] [-q QUEUE] [-n NCPU] [-m MEMORY] [--nlive N]
                        [--maxmcmc N] [--duration T] [--fmin F]
                        [--label NAME | --outdir PATH] [--no-jetfit]
                        [--density-prior] [--marginalize-psi] [--n-psi N]
                        [--aligned-spin] [--marginalize-phase]
                        [--marginalize-time] [--local]

Submit GW170817 PE (Bilby + dynesty + JetFit constraint) to Glamdring.

Default settings match the LIGO high_spin_PhenomPNRT analysis:
  - IMRPhenomPv2_NRTidal waveform
  - 23--2048 Hz, 128 s segment, 4096 Hz sampling
  - nlive=1000, nact=10, maxmcmc=5000 (act-walk sampling)
  - H1 + L1 + V1, sky position fixed to NGC 4993
  - JetFit afterglow EM constraint enabled by default

options:
  -h, --help              show this help message and exit
  -q, --queue QUEUE       Glamdring queue name (default: $queue)
  -n, --ncpu NCPU         Number of CPUs for npool (default: $ncpu)
  -m, --memory MEMORY     Memory per job in GB (default: $memory)
  --nlive N               Number of dynesty live points (default: $nlive)
  --maxmcmc N             Max MCMC steps per proposal (default: $maxmcmc)
  --nact N                Autocorrelation lengths for act-walk (default: $nact)
  --duration T            Segment duration in seconds (default: $duration)
  --fmin F                Minimum frequency in Hz (default: $fmin)
  --label NAME            Output subdirectory name under results/
                          (e.g. --label GW170817_vanilla)
  --outdir PATH           Output directory as absolute path
                          (default: results/GW170817_jetfit)
  --no-jetfit             Disable JetFit EM constraint
  --density-prior         Use density-weighted distance prior
  --marginalize-psi       Marginalize over polarisation angle (numerical)
  --n-psi N               Number of psi quadrature points (default: 50)
  --aligned-spin          Use aligned-spin waveform (IMRPhenomD_NRTidal)
  --marginalize-phase     Analytically marginalize over phase
                          (requires --aligned-spin)
  --marginalize-time      Marginalize over coalescence time (FFT grid)
  --real-data             Use GWOSC strain (default; omit for injection test)
  --local                 Run locally instead of submitting to queue

Extra arguments are forwarded to run_GW170817.py.

Examples:
  # Production run with JetFit (submit to queue)
  $(basename "$0")

  # Without JetFit constraint
  $(basename "$0") --no-jetfit --label GW170817_vanilla

  # Quick local test with injection
  $(basename "$0") --local --nlive 50 --maxmcmc 500 -n 4

  # Psi-marginalised + aligned spin (9D sampling)
  $(basename "$0") --marginalize-psi --aligned-spin --label GW170817_jetfit_psimarg_aligned

  # With density-weighted prior
  $(basename "$0") --density-prior --label GW170817_jetfit_density
EOF
    exit 0
}

# ---- parse arguments ----
local_mode=false
real_data=true
use_roq=false
duration_set=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)              usage ;;
        -q|--queue)             queue="$2"; shift 2 ;;
        -n|--ncpu)              ncpu="$2"; npool="$2"; shift 2 ;;
        -m|--memory)            memory="$2"; shift 2 ;;
        --nlive)                nlive="$2"; shift 2 ;;
        --maxmcmc)              maxmcmc="$2"; shift 2 ;;
        --nact)                 nact="$2"; shift 2 ;;
        --duration)             duration="$2"; duration_set=true; shift 2 ;;
        --fmin)                 fmin="$2"; shift 2 ;;
        --outdir)               outdir="$2"; shift 2 ;;
        --label)                outdir="$repo_root/results/$2"; shift 2 ;;
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
        *)                      extra_args="$extra_args $1"; shift ;;
    esac
done

# ROQ requires shorter duration (128/s must give integer * fs)
# and fmin >= scaled flow (20 * 128/duration ~ 24 Hz)
if $use_roq && ! $duration_set; then
    duration=106.5
fi
if $use_roq && [[ "$fmin" -lt 25 ]]; then
    fmin=25
fi

python_exec=$(get_toml_key "python_exec")
if [[ -z "$python_exec" ]]; then
    echo "[ERROR] Could not determine python_exec from $repo_root/local_config.toml" >&2
    exit 1
fi

# If running locally without --real-data explicitly, default to injection
if $local_mode && $real_data; then
    real_data=false
    echo "[INFO] Local mode: defaulting to injection test (pass --real-data to override)"
fi

if $real_data; then
    extra_args="$extra_args --real-data"
fi

echo "GW170817 PE submission"
echo "============================================================"
if $local_mode; then
    echo "  Mode:        LOCAL"
else
    echo "  Mode:        SUBMIT (queue=$queue)"
fi
echo "  CPUs/npool:  $ncpu"
echo "  nlive:       $nlive"
echo "  maxmcmc:     $maxmcmc"
echo "  nact:        $nact"
echo "  duration:    ${duration} s"
echo "  fmin:        ${fmin} Hz"
echo "  sampling_f:  ${sampling_frequency} Hz"
echo "  Output:      $outdir"
echo "  Python:      $python_exec"
if [[ -n "$extra_args" ]]; then
    echo "  Extra args: $extra_args"
fi
echo

read -p "Proceed? [y/N]: " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborting."
    exit 1
fi

if [[ ! "$outdir" = /* ]] || [[ "$outdir" = /results* && ! -d /results ]]; then
    echo "[ERROR] Output directory '$outdir' looks invalid (is \$REPO set?)" >&2
    exit 1
fi
mkdir -p "$outdir"

pythoncmd="/usr/bin/stdbuf -oL -eL $python_exec -u $script_dir/run_GW170817.py \
    --nlive $nlive \
    --maxmcmc $maxmcmc \
    --nact $nact \
    --duration $duration \
    --fmin $fmin \
    --sampling-frequency $sampling_frequency \
    --npool $npool \
    --outdir $outdir \
    $extra_args"

if $local_mode; then
    echo "Running locally..."
    eval "$pythoncmd"
else
    cm="addqueue -s -q $queue -n 1x$ncpu -m $memory -o python-%j.out $pythoncmd"
    echo "Submitting..."
    echo "  $cm"
    eval "$cm"
fi

echo
echo "Done."
