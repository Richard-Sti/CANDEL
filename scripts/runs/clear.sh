#!/bin/bash
#
# Remove generated artefacts from scripts/runs/.
#   logs   — cluster job-output files (logs-*.{out,err}, logs-*.status.tsv,
#            python-*.{out,err})
#   tasks  — tasks_*.txt task lists and generated_configs/ tree produced
#            by generate_tasks.py, plus generated submission batch scripts
#   all    — both of the above
#
# At least one target is required. Supports --dry to preview.

set -e

usage() {
    cat <<EOF
usage: $(basename "$0") <logs|tasks|all> [logs|tasks|all ...] [--dry]

targets:
  logs    remove logs-*.{out,err}, logs-*.status.tsv, and python-*.{out,err}
  tasks   remove tasks_*.txt, generated_configs/, and generated batch scripts
  all     both

options:
  --dry   print what would be removed without deleting
EOF
    exit 1
}

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dry=false
targets=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        --dry)     dry=true; shift ;;
        logs|tasks|all) targets+=("$1"); shift ;;
        *) echo "[ERROR] unknown arg: $1"; usage ;;
    esac
done

(( ${#targets[@]} )) || usage

# Expand 'all' and dedupe.
expanded=()
for t in "${targets[@]}"; do
    case "$t" in
        all)  expanded+=(logs tasks) ;;
        *)    expanded+=("$t") ;;
    esac
done
do_logs=false
do_tasks=false
for t in "${expanded[@]}"; do
    case "$t" in
        logs)  do_logs=true ;;
        tasks) do_tasks=true ;;
    esac
done

rm_or_echo() {
    if $dry; then
        for p in "$@"; do [[ -e "$p" ]] && echo "[dry] would remove: $p"; done
    else
        rm -vf "$@"
    fi
}
rmdir_or_echo() {
    if $dry; then
        [[ -d "$1" ]] && echo "[dry] would remove tree: $1"
    else
        rm -rvf "$1"
    fi
}

if $do_logs; then
    echo "[INFO] Clearing job-output files from '$run_dir/'..."
    shopt -s nullglob
    # Top-level logs:
    #   arc sbatch       -> logs-<jobid>.{out,err}
    #   batched jobs     -> logs-<jobid>-batch_<range>.status.tsv
    #   glamdring legacy -> python-<jobid>.{out,err}
    # Plus any logs/ subdirectory that collects the same patterns.
    logs_files=("$run_dir"/logs-*.out "$run_dir"/logs-*.err \
                "$run_dir"/logs-*.status.tsv \
                "$run_dir"/python-*.out "$run_dir"/python-*.err \
                "$run_dir"/logs/logs-*.out "$run_dir"/logs/logs-*.err \
                "$run_dir"/logs/logs-*.status.tsv \
                "$run_dir"/logs/python-*.out "$run_dir"/logs/python-*.err)
    shopt -u nullglob
    if (( ${#logs_files[@]} )); then
        rm_or_echo "${logs_files[@]}"
    else
        echo "  (no log files)"
    fi
    if [[ -d "$run_dir/logs" ]]; then
        # Only drop the dir itself if it's now empty — don't surprise the
        # user by wiping unrelated files they happened to stash there.
        if $dry; then
            if [[ -z "$(ls -A "$run_dir/logs" 2>/dev/null)" ]]; then
                echo "[dry] would remove empty dir: $run_dir/logs"
            fi
        else
            rmdir "$run_dir/logs" 2>/dev/null || true
        fi
    fi
fi

if $do_tasks; then
    echo "[INFO] Clearing task lists + generated scripts/configs from '$run_dir/'..."
    shopt -s nullglob
    task_files=("$run_dir"/tasks_*.txt "$run_dir"/batch_*.sh)
    shopt -u nullglob
    if (( ${#task_files[@]} )); then
        rm_or_echo "${task_files[@]}"
    else
        echo "  (no task or batch files)"
    fi
    if [[ -d "$run_dir/generated_configs" ]]; then
        rmdir_or_echo "$run_dir/generated_configs"
    else
        echo "  (no generated_configs/ tree)"
    fi
    if [[ -d "$run_dir/generated_batch_scripts" ]]; then
        rmdir_or_echo "$run_dir/generated_batch_scripts"
    else
        echo "  (no generated_batch_scripts/ tree)"
    fi
fi
