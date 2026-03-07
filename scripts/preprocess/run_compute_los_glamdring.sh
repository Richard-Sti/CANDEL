#!/bin/bash
#
# Submit compute_los.py jobs to the Glamdring queue system.
#
# Machine-specific settings (python_exec, root_main) are read from
# local_config.toml at the project root.

set -e

# Extract a TOML key from a config file, with fallback to local_config.toml
get_toml_key() {
    local key="$1"
    local config="$2"
    local val
    val=$(grep -E "^${key} *= *" "$config" 2>/dev/null | sed -E "s/^${key} *= *\"([^\"]+)\"$/\1/")
    if [[ -z "$val" ]]; then
        local local_config
        local_config="$(cd "$(dirname "$0")/../.." && pwd)/local_config.toml"
        val=$(grep -E "^${key} *= *" "$local_config" 2>/dev/null | sed -E "s/^${key} *= *\"([^\"]+)\"$/\1/")
    fi
    echo "$val"
}

# ---- defaults ----
config="../runs/config_EDD_TRGB.toml"
# reconstruction="Carrick2015"
reconstruction="manticore_2MPP_MULTIBIN_N256_DES_V2"
queue="cmb"
ncpu=1
memory=32
smooth_target=0
generic_filepath=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [CATALOGUE ...]

Submit compute_los.py jobs to the Glamdring queue for one or more catalogues.

If no catalogues are given, submits jobs for:
  CF4, 2MTF, SFI, Foundation, LOSS

Options:
  -c, --config PATH           TOML config file (default: $config)
  -r, --reconstruction NAME   Reconstruction name (default: $reconstruction)
  -q, --queue NAME            Queue name (default: $queue)
  -n, --ncpu N                Number of CPUs per job (default: $ncpu)
  -m, --memory GB             Memory per job in GB (default: $memory)
  -s, --smooth-target VALUE   Smoothing target (default: $smooth_target)
  -g, --generic-filepath PATH Data file for the 'generic' catalogue
  -h, --help                  Show this help message

Examples:
  $(basename "$0") CF4 2MTF
  $(basename "$0") -r manticore_2MPP_MULTIBIN_N256_DES_V2 CF4
  $(basename "$0") -g /path/to/data.txt generic
EOF
    exit 0
}

# ---- parse arguments ----
catalogues=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)             usage ;;
        -c|--config)           config="$2"; shift 2 ;;
        -r|--reconstruction)   reconstruction="$2"; shift 2 ;;
        -q|--queue)            queue="$2"; shift 2 ;;
        -n|--ncpu)             ncpu="$2"; shift 2 ;;
        -m|--memory)           memory="$2"; shift 2 ;;
        -s|--smooth-target)    smooth_target="$2"; shift 2 ;;
        -g|--generic-filepath) generic_filepath="$2"; shift 2 ;;
        -*)
            echo "[ERROR] Unknown option: $1" >&2
            echo "Run $(basename "$0") --help for usage." >&2
            exit 1 ;;
        *)  catalogues+=("$1"); shift ;;
    esac
done

# Resolve python_exec and root_main from config / local_config.toml
python_exec=$(get_toml_key "python_exec" "$config")
root_main=$(get_toml_key "root_main" "$config")

if [[ -z "$python_exec" ]]; then
    echo "[ERROR] Could not determine python_exec from config or local_config.toml" >&2
    exit 1
fi

# Default catalogues if none specified
if [[ ${#catalogues[@]} -eq 0 ]]; then
    catalogues=("CF4" "2MTF" "SFI" "Foundation" "LOSS")

    echo "No catalogues specified — will submit jobs for:"
    for c in "${catalogues[@]}"; do
        echo "  - $c"
    done
    echo
    echo "Settings:"
    echo "  Reconstruction: $reconstruction"
    echo "  Queue: $queue"
    echo "  CPUs: $ncpu"
    echo "  Memory: ${memory} GB"
    echo "  Config: $config"
    echo "  Python: $python_exec"
    echo

    read -p "Submit jobs for ALL of these? [y/N]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborting."
        exit 1
    fi
fi

# ---- submit jobs ----
for catalogue in "${catalogues[@]}"; do
    pythoncm="$python_exec compute_los.py --catalogue $catalogue --reconstruction $reconstruction --config $config --smooth_target $smooth_target"

    if [[ "$catalogue" == "generic" ]]; then
        if [[ -z "$generic_filepath" ]]; then
            echo "[ERROR] --generic-filepath is required for the 'generic' catalogue" >&2
            exit 1
        fi
        pythoncm="$pythoncm --filepath $generic_filepath"
    fi

    cm="addqueue -q $queue -n $ncpu -m $memory $pythoncm"

    echo "Submitting catalogue: $catalogue"
    echo "  $cm"
    eval "$cm"
    echo
done
