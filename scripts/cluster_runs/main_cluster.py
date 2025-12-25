#!/usr/bin/env python
"""
MPI launcher that distributes entries from a tasks file (e.g. tasks_0.txt)
across ranks and runs the standard PV inference for each configuration.
Each rank processes the subset of tasks whose line index satisfies
`line_index % world_size == rank`.
"""

import argparse
import os
import sys
import time
from os.path import exists

from mpi4py import MPI


def _set_single_thread_env_if_unset():
    """
    Prevent oversubscription and huge per-rank memory spikes on HPC nodes.

    We only set these if the user/cluster hasn't already set them, so we don't
    fight site policies.
    """
    defaults = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


def _preparse_host_devices():
    default_host_devices = int(os.getenv("LOCAL_HOST_DEVICES", "1"))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--host-devices",
        type=int,
        default=default_host_devices,
        help=(
            "Number of NumPyro/JAX host devices to expose PER MPI RANK. "
            "For MPI task farming you almost always want 1."
        ),
    )
    parser.add_argument(
        "--no-thread-limits",
        action="store_true",
        help="Do not set OMP/MKL/BLAS thread env vars to 1.",
    )
    return parser


_pre_parser = _preparse_host_devices()
_pre_args, _remaining = _pre_parser.parse_known_args()

# 1) Thread limits must be set before importing numpy/jax stacks
if not _pre_args.no_thread_limits:
    _set_single_thread_env_if_unset()

# 2) Host device count must be set before importing candel/JAX/NumPyro
if _pre_args.host_devices and _pre_args.host_devices != 1:
    # Only import numpyro if we need to change this from the default.
    import numpyro  # noqa: E402

    numpyro.set_host_device_count(_pre_args.host_devices)

# Import heavy stuff AFTER environment is sane
import candel  # noqa: E402
from candel import fprint, get_nested  # noqa: E402


def insert_comment_at_top(path: str, label: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    comment = f"# Job {label} at: {timestamp}\n"
    with open(path, "r") as f:
        original = f.readlines()
    with open(path, "w") as f:
        f.write(comment)
        f.writelines(original)


def run_config(config_path: str):
    insert_comment_at_top(config_path, "started")

    config = candel.load_config(config_path)

    fname_out = get_nested(config, "io/fname_output")
    skip_if_exists = get_nested(config, "inference/skip_if_exists", False)
    if skip_if_exists and exists(fname_out):
        fprint(f"[Rank {RANK}] Output `{fname_out}` exists; skipping.")
        insert_comment_at_top(config_path, "skipped")
        return

    is_CH0 = get_nested(config, "model/is_CH0", False)
    if is_CH0:
        fprint(f"[Rank {RANK}] running `CH0` model for `{config_path}`")
        data = candel.pvdata.load_SH0ES_from_config(config_path)
        model = candel.model.SH0ESModel(config_path, data)
        candel.run_SH0ES_inference(model)
    else:
        data = candel.pvdata.load_PV_dataframes(config_path)

        model_name = config["inference"]["model"]
        data_name = config["io"]["catalogue_name"]
        fprint(
            f"[Rank {RANK}] Loading model `{model_name}` "
            f"for data `{data_name}` using `{config_path}`"
        )

        shared_param = get_nested(config, "inference/shared_params", None)
        model = candel.model.name2model(model_name, shared_param, config_path)

        if isinstance(data, list):
            if not isinstance(model, candel.model.JointPVModel):
                raise TypeError(
                    "Multiple datasets provided but model is not JointPVModel."
                )
            if len(data) != len(model.submodels):
                raise ValueError(
                    f"Datasets ({len(data)}) != submodels ({len(model.submodels)})"
                )

        candel.run_pv_inference(model, {"data": data})

    insert_comment_at_top(config_path, "finished")


def parse_tasks_file(path: str):
    tasks = []
    with open(path, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            tasks.append(parts[1])
    return tasks


def main():
    parser = argparse.ArgumentParser(
        parents=[_pre_parser],
        description="MPI launcher for CANDEL inference tasks.",
    )
    parser.add_argument(
        "--tasks-file",
        default="tasks_0.txt",
        help="Path to tasks file with `idx path/to/config.toml` per line.",
    )
    args = parser.parse_args()

    # Optional: one-time debug print to confirm env is sane
    if RANK == 0:
        fprint(
            "[Rank 0] Env summary: "
            f"python={sys.executable} "
            f"VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV')} "
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
            f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')} "
            f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')} "
            f"host_devices={args.host_devices}"
        )

    tasks = parse_tasks_file(args.tasks_file)
    if not tasks:
        if RANK == 0:
            fprint(f"No tasks found in `{args.tasks_file}`.")
        return

    local_tasks = [cfg for i, cfg in enumerate(tasks) if i % SIZE == RANK]
    if not local_tasks:
        fprint(f"[Rank {RANK}] No assigned tasks.")
        return

    fprint(f"[Rank {RANK}] running {len(local_tasks)} task(s).")
    for cfg in local_tasks:
        run_config(cfg)


comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()

if __name__ == "__main__":
    main()
