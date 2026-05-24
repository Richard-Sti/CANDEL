#!/usr/bin/env python
"""Batch mock chi2 runner for MW Cepheid photometric parallax tests (MPI).

Runs all campaign/likelihood combinations and prints a summary table.
Rank 0 is the master that distributes seeds; ranks 1..N are workers.

Usage:
    mpirun -np 28 python run_mock_simple.py --n-mocks 1000 \
        --outdir ../../../results/MWCepheids/mocks \
        --campaign C22 --likelihood gaussian
"""
import argparse
import copy
import os
import signal
import sys
import time
from datetime import datetime

import numpy as np
from mpi4py import MPI

import mock_utils
from mock_utils import (DEFAULT_CONFIGS, TRUE_VALS, likelihood_label,
                        parse_likelihood, run_one_mock)

TASKS = [
    ("C22", "gaussian"),
    ("C22", "chi2"),
    ("C27", "gaussian"),
    ("C27", "parallax_selection"),
    ("C27", "chi2"),
]

PARAMS = ["MWH", "bW", "ZW", "delta_pi"]

TAG_WORK = 1
TAG_RESULT = 2
TAG_DONE = 3


def master(comm, n_workers, n_mocks, label, outdir):
    """Rank 0: distribute seeds to workers and collect results."""
    print(f"\n{'=' * 60}")
    print(f"[INFO] {label}: {n_mocks} mocks, {n_workers} workers")

    t0 = time.time()
    seeds = list(range(n_mocks))
    results = []
    n_sent = 0
    n_done = 0
    n_skipped = 0
    status = MPI.Status()

    # Send initial work to each worker
    for rank in range(1, n_workers + 1):
        if n_sent < n_mocks:
            comm.send(seeds[n_sent], dest=rank, tag=TAG_WORK)
            n_sent += 1
        else:
            comm.send(None, dest=rank, tag=TAG_DONE)

    # Collect results and send more work
    while n_done < n_mocks:
        result = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT,
                           status=status)
        source = status.Get_source()
        n_done += 1

        elapsed = time.time() - t0
        now = datetime.now().strftime("%H:%M:%S")
        if result is None:
            n_skipped += 1
            print(f"[WARN {now}] {label}: {n_done}/{n_mocks} — seed timed "
                  f"out on rank {source} ({elapsed:.0f}s)", flush=True)
        else:
            results.append(result)
            print(f"[INFO {now}] {label}: {n_done}/{n_mocks} done "
                  f"({elapsed:.0f}s)", flush=True)

        if n_sent < n_mocks:
            comm.send(seeds[n_sent], dest=source, tag=TAG_WORK)
            n_sent += 1
        else:
            comm.send(None, dest=source, tag=TAG_DONE)

    elapsed = time.time() - t0
    print(f"[INFO] {label}: done in {elapsed:.0f}s "
          f"({elapsed / n_mocks:.1f}s/mock)")
    if n_skipped:
        print(f"[WARN] {label}: {n_skipped}/{n_mocks} mocks skipped "
              f"(timed out)")

    biases = {lab: [] for lab in TRUE_VALS}
    for b in results:
        for lab in b:
            biases[lab].append(b[lab])

    save_dict = {}
    for lab in list(biases):
        if biases[lab]:
            biases[lab] = np.array(biases[lab])
            save_dict[lab] = biases[lab]
        else:
            del biases[lab]

    save_dict["params"] = np.array(list(biases.keys()))
    save_dict["n_mocks"] = np.array(n_mocks)
    save_dict["n_skipped"] = np.array(n_skipped)

    outfile = f"{outdir}/mock_{label}.npz"
    np.savez(outfile, **save_dict)
    print(f"[INFO] Saved to {outfile}")

    return label, biases


def worker(comm, which, configs, use_gaussian, sigma_int_val, timeout):
    """Ranks 1..N: receive seeds, run mocks, send back results."""
    def _alarm_handler(signum, frame):
        raise TimeoutError

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)

    while True:
        status = MPI.Status()
        seed = comm.recv(source=0, status=status)

        if status.Get_tag() == TAG_DONE:
            break

        if timeout > 0:
            signal.alarm(timeout)
        try:
            result = run_one_mock(
                seed, which, configs, use_gaussian=use_gaussian,
                sigma_int_val=sigma_int_val)
        except TimeoutError:
            result = None
        finally:
            signal.alarm(0)

        comm.send(result, dest=0, tag=TAG_RESULT)

    signal.signal(signal.SIGALRM, old_handler)


def collect_results(outdir, labels):
    """Load saved .npz files, return (results dict, list of file paths)."""
    all_results = {}
    files = []
    for label in labels:
        fpath = f"{outdir}/mock_{label}.npz"
        try:
            data = np.load(fpath)
            biases = {p: data[p] for p in PARAMS if p in data}
            all_results[label] = biases
            files.append(fpath)
        except FileNotFoundError:
            print(f"[WARN] Missing {fpath}, skipping")
    return all_results, files


def print_table(all_results):
    header = f"{'Run':<28s}" + "".join(f"{p:>18s}" for p in PARAMS)
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    print(header)
    print("-" * len(header))
    for label, biases in all_results.items():
        row = f"{label:<28s}"
        for p in PARAMS:
            if p in biases:
                b = biases[p]
                row += f"{f'{b.mean():+.2f} +/- {b.std():.2f}':>18s}"
            else:
                row += f"{'---':>18s}"
        print(row)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_workers = size - 1

    # Rank 0 parses arguments and broadcasts config
    if rank == 0:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--n-mocks", type=int, default=10000)
        parser.add_argument("--outdir",
                            default="../../../results/MWCepheids/mocks")
        parser.add_argument("--sigma-int", type=float, default=None,
                            help="Override intrinsic scatter (default: "
                            "use mock_utils value)")
        parser.add_argument("--campaign", type=str, default=None,
                            help="Run a single campaign (e.g. C22, C27)")
        parser.add_argument("--likelihood", type=str, default=None,
                            help="Run a single likelihood (e.g. gaussian,"
                            " chi2, parallax_selection)")
        parser.add_argument("--timeout", type=int, default=900,
                            help="Per-mock wall-time limit in seconds "
                            "(default: 900 = 15 min, 0 = no limit)")
        args = parser.parse_args()

        if n_workers < 1:
            print("[ERROR] Need at least 2 MPI ranks "
                  "(1 master + 1 worker).")
            sys.exit(1)

        sigma_int_tag = None
        sigma_int_val = None
        if args.sigma_int is not None:
            sigma_int_val = args.sigma_int
            sigma_int_tag = f"{args.sigma_int:.3f}"
        si = (sigma_int_val if sigma_int_val is not None
              else mock_utils.sigma_int)
        print(f"[INFO] sigma_int = {si}")
        print(f"[INFO] timeout = {args.timeout}s"
              if args.timeout > 0 else "[INFO] timeout = none")
        print(f"[INFO] MPI size = {size} ({n_workers} workers)")

        # Single-task mode
        if args.campaign is not None and args.likelihood is not None:
            task_list = [(args.campaign, args.likelihood)]
        elif args.campaign is None and args.likelihood is None:
            task_list = TASKS
        else:
            parser.error(
                "--campaign and --likelihood must be used together")

        os.makedirs(args.outdir, exist_ok=True)

        # Build per-task configs
        tasks = []
        for campaign, likelihood in task_list:
            use_gaussian = parse_likelihood(likelihood)
            configs = copy.deepcopy(DEFAULT_CONFIGS)
            which = [campaign]
            label = f"{campaign}_{likelihood_label(use_gaussian)}"
            if sigma_int_tag is not None:
                label += f"_sint{sigma_int_tag}"
            tasks.append((which, configs, use_gaussian, label))

        config = {
            "n_mocks": args.n_mocks,
            "outdir": args.outdir,
            "sigma_int_val": sigma_int_val,
            "sigma_int_tag": sigma_int_tag,
            "tasks": tasks,
            "timeout": args.timeout,
        }
    else:
        config = None

    config = comm.bcast(config, root=0)

    n_mocks = config["n_mocks"]
    outdir = config["outdir"]
    sigma_int_val = config["sigma_int_val"]
    sigma_int_tag = config["sigma_int_tag"]
    tasks = config["tasks"]
    timeout = config["timeout"]

    single_task = len(tasks) == 1

    labels = []
    for which, configs, use_gaussian, label in tasks:
        if rank == 0:
            label, biases = master(
                comm, n_workers, n_mocks, label, outdir)
            labels.append(label)
        else:
            worker(comm, which, configs, use_gaussian,
                   sigma_int_val, timeout)

    # In single-task mode, keep the per-task .npz for collect_mock_simple.py
    if rank == 0 and not single_task:
        all_results, intermediate_files = collect_results(outdir, labels)
        print_table(all_results)

        combined = {}
        for label, biases in all_results.items():
            for p, vals in biases.items():
                combined[f"{label}/{p}"] = vals
        combined["labels"] = np.array(list(all_results.keys()))
        combined["params"] = np.array(PARAMS)
        combined["n_mocks"] = np.array(n_mocks)
        combined["sigma_int"] = np.array(
            sigma_int_val if sigma_int_val is not None
            else mock_utils.sigma_int)

        tag = f"_sint{sigma_int_tag}" if sigma_int_tag else ""
        outfile = f"{outdir}/mock_batch{tag}.npz"
        np.savez(outfile, **combined)
        print(f"\n[INFO] Combined results saved to {outfile}")

        for f in intermediate_files:
            os.remove(f)
            print(f"[INFO] Removed {f}")


if __name__ == "__main__":
    main()
