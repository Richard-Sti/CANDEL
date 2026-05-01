#!/bin/bash
# Unified job submitter for CANDEL scripts.
#
# Source this file from a submit_*.sh, then call:
#
#   submit_job --queue Q --mem GB
#              [--cpus N]                 # CPU cores (default: 1, or 4 with --gpu)
#              [--mpi-n N | AxB]          # MPI ranks; mutually exclusive with --gpu
#              [--gpu] [--gputype TYPE]   # single GPU; TYPE e.g. l40s, h100, a100
#              [--gpu-mem GB]             # min GPU VRAM; arc-only, queries sinfo
#              [--time H | D-HH:MM:SS]    # bare integer = hours; required on 'long'
#                                         # defaults: short=12h, medium=48h
#              [--name JOB]               # job name (default: candel)
#              [--logdir DIR]             # log directory (default: logs)
#              [--dry]                    # print command without submitting
#              -- <cmd...>
#
# Cluster is read from `machine` in local_config.toml:
#   machine="arc"       -> sbatch (time/partition defaults, --gputype via --gres)
#   machine="glamdring" -> addqueue (--time is currently ignored)
#
# Exposes for callers:
#   CANDEL_ROOT    absolute path to the repo root
#   CANDEL_PYTHON  python_exec from local_config.toml
#   CANDEL_CLUSTER value of `machine`

_submit_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANDEL_ROOT="$(cd "$_submit_lib_dir/.." && pwd)"
export CANDEL_ROOT
export PYTHONPATH="$CANDEL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

_toml_get() {
    local key="$1" file="$2"
    awk -v key="$key" '
        /^[[:space:]]*\[/ { exit }
        $0 ~ "^[[:space:]]*"key"[[:space:]]*=" {
            line = $0
            sub(/^[^=]*=[[:space:]]*/, "", line)
            sub(/[[:space:]]*(#.*)?$/, "", line)
            gsub(/^["'\'']|["'\'']$/, "", line)
            print line
            exit
        }
    ' "$file"
}

_local_config="$CANDEL_ROOT/local_config.toml"
if [[ ! -f "$_local_config" ]]; then
    echo "[submit_lib] local_config.toml not found: $_local_config" >&2
    return 1 2>/dev/null || exit 1
fi

CANDEL_CLUSTER="$(_toml_get machine "$_local_config")"
CANDEL_PYTHON="$(_toml_get python_exec "$_local_config")"
CANDEL_MODULES="$(_toml_get modules "$_local_config")"
CANDEL_MODULES_GPU="$(_toml_get modules_gpu "$_local_config")"
export CANDEL_CLUSTER CANDEL_PYTHON CANDEL_MODULES CANDEL_MODULES_GPU

if [[ -z "$CANDEL_CLUSTER" ]]; then
    echo "[submit_lib] 'machine' not set in $_local_config" >&2
    return 1 2>/dev/null || exit 1
fi
if [[ -z "$CANDEL_PYTHON" ]]; then
    echo "[submit_lib] 'python_exec' not set in $_local_config" >&2
    return 1 2>/dev/null || exit 1
fi

_cluster_profile="$_submit_lib_dir/_cluster_${CANDEL_CLUSTER}.sh"
if [[ ! -f "$_cluster_profile" ]]; then
    echo "[submit_lib] no cluster profile: $_cluster_profile" >&2
    return 1 2>/dev/null || exit 1
fi

# Source the profile in the current shell. On glamdring this puts the env
# needed by addqueue jobs into place (inherited via -s). On arc this still
# runs but is mostly redundant; the sbatch script re-sources it anyway.
# shellcheck disable=SC1090
source "$_cluster_profile"

launch_detached() {
    # Launch a command in a detached screen/tmux session.
    # Usage: launch_detached <session-name> <logfile> <cmd...>
    local sname="$1" logfile="$2"; shift 2

    # Build a properly quoted command string for tmux.
    local cmd_str=""
    for arg in "$@"; do
        cmd_str+="$(printf '%q ' "$arg")"
    done

    local mux=""
    if command -v screen &>/dev/null; then
        mux="screen"
        screen -dmS "$sname" -L -Logfile "$logfile" "$@"
        sleep 3
        [[ -f "$logfile" ]] && cat "$logfile"
        echo ""
        echo "[watch] screen: $sname"
        echo "[watch]   reattach: screen -r $sname"
        echo "[watch]   kill:     screen -S $sname -X quit"
    elif command -v tmux &>/dev/null; then
        mux="tmux"
        tmux new-session -d -s "$sname" \
            "bash -c '${cmd_str} 2>&1 | tee ${logfile}'"
        sleep 3
        [[ -f "$logfile" ]] && cat "$logfile"
        echo ""
        echo "[watch] tmux: $sname"
        echo "[watch]   reattach: tmux attach -t $sname"
        echo "[watch]   kill:     tmux kill-session -t $sname"
    else
        echo "[watch] Error: neither screen nor tmux found" >&2
        return 1
    fi

    # Print management hints once per shell invocation, regardless of how
    # many detached sessions we spawn.
    if [[ -z "${_LAUNCH_DETACHED_HINTED:-}" ]]; then
        case "$mux" in
            screen)
                echo "[watch]   list all: screen -ls"
                echo "[watch]   detach:   Ctrl-a d (from inside an attached session)"
                ;;
            tmux)
                echo "[watch]   list all: tmux ls"
                echo "[watch]   detach:   Ctrl-b d (from inside an attached session)"
                ;;
        esac
        export _LAUNCH_DETACHED_HINTED=1
    fi
}

submit_job() {
    local queue="" mem="" cpus="" time="" name="candel" logdir="logs"
    local gpu=0 dry=0 mpi_n="" gputype="" gpu_mem_min=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --queue)   queue="$2"; shift 2 ;;
            --mem)     mem="$2"; shift 2 ;;
            --cpus)    cpus="$2"; shift 2 ;;
            --mpi-n)   mpi_n="$2"; shift 2 ;;
            --time)    time="$2"; shift 2 ;;
            --name)    name="$2"; shift 2 ;;
            --logdir)  logdir="$2"; shift 2 ;;
            --gpu)     gpu=1; shift ;;
            --gputype) gputype="$2"; shift 2 ;;
            --gpu-mem) gpu_mem_min="$2"; shift 2 ;;
            --dry)     dry=1; shift ;;
            --)        shift; break ;;
            *)
                echo "[submit_job] unknown flag: $1" >&2
                return 2 ;;
        esac
    done

    if [[ -z "$queue" || -z "$mem" ]]; then
        echo "[submit_job] --queue and --mem are required" >&2
        return 2
    fi
    if [[ $# -eq 0 ]]; then
        echo "[submit_job] missing <cmd> after --" >&2
        return 2
    fi
    if [[ -n "$mpi_n" ]] && (( gpu )); then
        echo "[submit_job] --mpi-n and --gpu are mutually exclusive" >&2
        return 2
    fi
    if [[ -z "$cpus" ]]; then
        if (( gpu )); then cpus=4; else cpus=1; fi
    fi
    if [[ -n "${CANDEL_WATCH_ROUND:-}" && "$CANDEL_WATCH_ROUND" -gt 0 ]] 2>/dev/null; then
        name="${name}_r${CANDEL_WATCH_ROUND}"
    fi
    if [[ -n "$gputype" ]] && (( ! gpu )); then
        echo "[submit_job] --gputype given without --gpu; ignoring" >&2
        gputype=""
    fi
    if [[ -n "$gpu_mem_min" ]] && (( ! gpu )); then
        echo "[submit_job] --gpu-mem given without --gpu; ignoring" >&2
        gpu_mem_min=""
    fi
    if [[ -n "$gputype" && -n "$gpu_mem_min" ]]; then
        echo "[submit_job] --gputype and --gpu-mem are mutually exclusive" >&2
        return 2
    fi

    # Total MPI ranks, resolved from --mpi-n spec "N" or "AxB".
    local mpi_total=""
    if [[ -n "$mpi_n" ]]; then
        if [[ "$mpi_n" == *x* ]]; then
            mpi_total=$(( ${mpi_n%x*} * ${mpi_n#*x} ))
        else
            mpi_total="$mpi_n"
        fi
    fi

    local cmd_str="$*"

    case "$CANDEL_CLUSTER" in
        arc)
            # On arc logs land in the submit CWD. Merge stderr into the
            # same file so there is a single log per job.
            # --logdir is ignored here.
            local sbatch_flags=(
                -p "$queue"
                --mem="${mem}G"
                --job-name="$name"
                --chdir="$PWD"
                --output="logs-%j.out"
                --error="logs-%j.out"
                --mail-type=BEGIN,END,FAIL
                --mail-user=richard.stiskalek@physics.ox.ac.uk
            )
            if [[ -n "$mpi_n" ]]; then
                sbatch_flags+=(--ntasks="$mpi_total" --cpus-per-task=1)
            else
                sbatch_flags+=(--ntasks=1 --cpus-per-task="$cpus")
            fi
            if [[ -z "$time" ]]; then
                case "$queue" in
                    short)  time=12 ;;
                    medium) time=48 ;;
                    long)
                        echo "[submit_job] --time is required on partition 'long'" >&2
                        return 2 ;;
                esac
            fi
            # Accept bare hours (e.g. "12" or "36"); emit HH:MM:SS if <24h
            # else D-HH:MM:SS.
            if [[ "$time" =~ ^[0-9]+$ ]]; then
                if (( time < 24 )); then
                    time="$(printf '%02d:00:00' "$time")"
                else
                    time="$((time/24))-$(printf '%02d:00:00' $((time%24)))"
                fi
            fi
            sbatch_flags+=(--time="$time")
            if (( gpu )); then
                if [[ -n "$gputype" ]]; then
                    sbatch_flags+=(--gres="gpu:${gputype}:1")
                else
                    sbatch_flags+=(--gres=gpu:1)
                fi
                if [[ -n "$gpu_mem_min" ]]; then
                    local _mem_constraint
                    _mem_constraint=$(sinfo -p "$queue" -h -o "%f" \
                        | tr ',' '\n' | grep -oP 'gpu_mem:\K[0-9]+' | sort -un \
                        | awk -v min="$gpu_mem_min" '$1 >= min {printf "gpu_mem:%dGB|", $1}' \
                        | sed 's/|$//')
                    if [[ -z "$_mem_constraint" ]]; then
                        echo "[submit_job] no GPUs with >= ${gpu_mem_min}GB in partition '$queue'" >&2
                        return 2
                    fi
                    sbatch_flags+=(--constraint "$_mem_constraint")
                fi
            fi
            echo "[submit_job] arc: sbatch ${sbatch_flags[*]}"
            echo "[submit_job] cmd : $cmd_str"
            if (( dry )); then
                echo "[submit_job] (dry: not submitting)"
                return 0
            fi
            local mods="$CANDEL_MODULES"
            (( gpu )) && [[ -n "$CANDEL_MODULES_GPU" ]] && mods="$CANDEL_MODULES_GPU"
            local _sbatch_out
            _sbatch_out=$(sbatch "${sbatch_flags[@]}" <<SCRIPT
#!/bin/bash -l
export CANDEL_MODULES_ACTIVE="$mods"
export PYTHONPATH="$CANDEL_ROOT\${PYTHONPATH:+:\$PYTHONPATH}"
source "$_cluster_profile"
$cmd_str
SCRIPT
            )
            echo "$_sbatch_out"
            local _jid
            _jid=$(echo "$_sbatch_out" | grep -oP 'Submitted batch job \K[0-9]+')
            [[ -n "$_jid" ]] && echo "JOBID=$_jid"
            ;;
        glamdring)
            if [[ -n "$time" ]]; then
                echo "[submit_job] glamdring: --time is not plumbed to addqueue; ignoring ($time)" >&2
            fi
            if [[ -n "$gpu_mem_min" ]]; then
                echo "[submit_job] glamdring: --gpu-mem not supported; ignoring" >&2
            fi
            local addqueue_flags=(-s -q "$queue" -m "$mem" -c "$name")
            if (( gpu )); then
                addqueue_flags+=(--gpus 1 -n "$cpus")
                [[ -n "$gputype" ]] && addqueue_flags+=(--gputype "$gputype")
            elif [[ -n "$mpi_n" ]]; then
                addqueue_flags+=(-n "$mpi_n")
            else
                addqueue_flags+=(-n "$cpus")
            fi
            echo "[submit_job] glamdring: addqueue ${addqueue_flags[*]} $cmd_str"
            echo "[submit_job] log     : $PWD/python-<jobid>.out"
            if (( dry )); then
                echo "[submit_job] (dry: not submitting)"
                return 0
            fi
            local _aq_out
            _aq_out=$(addqueue --sbatch "${addqueue_flags[@]}" $cmd_str 2>&1)
            echo "$_aq_out"
            local _jid
            _jid=$(echo "$_aq_out" | grep -oP 'Submitted batch job \K[0-9]+')
            [[ -n "$_jid" ]] && echo "JOBID=$_jid"
            ;;
        *)
            echo "[submit_job] unknown cluster: $CANDEL_CLUSTER" >&2
            return 1 ;;
    esac
}
