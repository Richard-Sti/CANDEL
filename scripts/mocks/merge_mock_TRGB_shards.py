#!/usr/bin/env python
"""Merge sharded TRGB mock closure outputs."""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest


def _scalar_int(value):
    arr = np.asarray(value)
    return int(arr.reshape(-1)[0])


def _plot_bias_summary(save_dict, output):
    params = [str(p) for p in save_dict["params"]
              if p in save_dict and len(np.asarray(save_dict[p])) > 0]
    if not params:
        return None

    ncols = 3
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, param in zip(axes, params):
        vals = np.asarray(save_dict[param])
        ax.hist(vals, bins="auto", alpha=0.75, edgecolor="black",
                linewidth=0.5)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(
            f"{param}: {vals.mean():+.2f} +/- {vals.std():.2f}",
            fontsize=9)
        ax.set_xlabel("standardised bias")
        ax.set_ylabel("mocks")

    for ax in axes[len(params):]:
        ax.axis("off")

    fig.tight_layout()
    plotfile = os.path.splitext(output)[0] + ".png"
    fig.savefig(plotfile, dpi=150)
    plt.close(fig)
    return plotfile


def _print_bias_table(save_dict):
    """Print mean standardised bias and KS p-value against N(0, 1)."""
    params = [str(p) for p in save_dict["params"]
              if p in save_dict and len(np.asarray(save_dict[p])) >= 2]
    if not params:
        return

    has_unc = any(f"unc_{p}" in save_dict for p in params)
    w = 92 if has_unc else 76
    print(f"\n{'=' * w}")
    print("Merged bias summary against standard normal")
    print("=" * w)
    if has_unc:
        print(f"{'param':<20s}  {'n':>5s}  {'mean +/- std':>20s}  "
              f"{'mean unc.':>12s}  {'KS p-value':>10s}")
    else:
        print(f"{'param':<20s}  {'n':>5s}  {'mean +/- std':>20s}  "
              f"{'KS p-value':>10s}")
    print("-" * w)
    for param in params:
        vals = np.asarray(save_dict[param])
        pval = kstest(vals, "norm").pvalue
        line = (f"{param:<20s}  {len(vals):>5d}  "
                f"{f'{vals.mean():+.3f} +/- {vals.std():.3f}':>20s}")
        if has_unc:
            key = f"unc_{param}"
            if key in save_dict:
                line += f"  {np.asarray(save_dict[key]).mean():>12.4g}"
            else:
                line += f"  {'':>12s}"
        line += f"  {pval:>10.3f}"
        print(line)


def _print_mock_observable_summary(save_dict):
    """Print summary of mock input observables if present."""
    if "mock_mag_obs" not in save_dict or "mock_czcmb" not in save_dict:
        return

    print("\nMock input observable summary")
    print("-----------------------------")
    for key, label, unit in [
            ("mock_mag_obs", "mag_obs", "mag"),
            ("mock_czcmb", "czcmb", "km/s"),
            ("mock_r_true", "r_true", "Mpc")]:
        if key not in save_dict:
            continue
        vals = np.asarray(save_dict[key])
        if len(vals) == 0:
            continue
        print(f"{label:<8s}: n={len(vals):d}, "
              f"mean={vals.mean():.3f} {unit}, "
              f"std={vals.std():.3f} {unit}, "
              f"range=[{vals.min():.3f}, {vals.max():.3f}] {unit}")


def _plot_mock_observables(save_dict, output):
    """Save histograms of mock magnitudes, redshifts, and true radii."""
    if "mock_mag_obs" not in save_dict or "mock_czcmb" not in save_dict:
        return None

    mag_obs = np.asarray(save_dict["mock_mag_obs"]).ravel()
    czcmb = np.asarray(save_dict["mock_czcmb"]).ravel()
    if len(mag_obs) == 0 or len(czcmb) == 0:
        return None

    has_r = "mock_r_true" in save_dict and len(save_dict["mock_r_true"]) > 0
    ncols = 3 if has_r else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.2))
    axes = np.atleast_1d(axes)

    axes[0].hist(mag_obs, bins=40)
    axes[0].set_xlabel("TRGB magnitude")
    axes[0].set_ylabel("hosts")

    axes[1].hist(czcmb, bins=40)
    axes[1].set_xlabel(r"$cz_{\rm CMB}$ [km/s]")
    axes[1].set_ylabel("hosts")

    if has_r:
        r_true = np.asarray(save_dict["mock_r_true"]).ravel()
        axes[2].hist(r_true, bins=40)
        axes[2].set_xlabel("true radius [Mpc]")
        axes[2].set_ylabel("hosts")

    fig.tight_layout()
    plotfile = os.path.splitext(output)[0] + "_mock_observables.png"
    fig.savefig(plotfile, dpi=150)
    plt.close(fig)
    return plotfile


def _delete_input_shards(files):
    """Delete merged shard files and their empty temporary shard dirs."""
    removed_files = 0
    removed_dirs = 0
    parent_dirs = []

    for fname in files:
        parent_dirs.append(os.path.dirname(os.path.abspath(fname)))
        try:
            os.remove(fname)
            removed_files += 1
        except FileNotFoundError:
            pass

    for parent in sorted(set(parent_dirs), reverse=True):
        current = parent
        while True:
            base = os.path.basename(current)
            if not (base.startswith("shard_")
                    or base.startswith("gpu_shards_")):
                break
            try:
                os.rmdir(current)
                removed_dirs += 1
            except OSError:
                break
            current = os.path.dirname(current)

    print(f"[INFO] Deleted {removed_files} merged shard files "
          f"and {removed_dirs} empty shard directories")


def merge(paths, output, delete_inputs=False):
    files = []
    for path in paths:
        matches = glob.glob(path)
        if matches:
            files.extend(matches)
        elif not glob.has_magic(path):
            files.append(path)
    files = sorted(set(files))
    if not files:
        raise ValueError("No shard files found.")

    loaded = []
    for fname in files:
        if not os.path.exists(fname):
            raise FileNotFoundError(fname)
        loaded.append(np.load(fname, allow_pickle=True))

    params = [str(p) for p in loaded[0]["params"]]
    out = {"params": loaded[0]["params"]}

    for param in params:
        arrays = [d[param] for d in loaded if param in d]
        if arrays:
            out[param] = np.concatenate(arrays)
        key = f"unc_{param}"
        arrays = [d[key] for d in loaded if key in d]
        if arrays:
            out[key] = np.concatenate(arrays)

    n_hosts = [d["n_hosts"] for d in loaded if "n_hosts" in d]
    if n_hosts:
        out["n_hosts"] = np.concatenate(n_hosts)

    for key in ["mock_mag_obs", "mock_czcmb", "mock_r_true"]:
        arrays = [d[key] for d in loaded if key in d]
        if arrays:
            out[key] = np.concatenate(arrays)

    out["n_mocks"] = np.array(sum(_scalar_int(d["n_mocks"]) for d in loaded))
    out["n_skipped"] = np.array(
        sum(_scalar_int(d["n_skipped"]) for d in loaded))

    for key in loaded[0].files:
        if key.startswith("true_"):
            out[key] = loaded[0][key]

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.savez(output, **out)
    plotfile = _plot_bias_summary(out, output)
    mock_plotfile = _plot_mock_observables(out, output)
    _print_bias_table(out)
    _print_mock_observable_summary(out)
    for d in loaded:
        d.close()
    print(f"[INFO] Merged {len(files)} shard files into {output}")
    if plotfile is not None:
        print(f"[INFO] Saved plot to {plotfile}")
    if mock_plotfile is not None:
        print(f"[INFO] Saved mock observable plot to {mock_plotfile}")
    if delete_inputs:
        _delete_input_shards(files)


def main():
    parser = argparse.ArgumentParser(
        description="Merge sharded TRGB mock closure .npz files.")
    parser.add_argument("inputs", nargs="+",
                        help="Shard .npz files or glob patterns.")
    parser.add_argument("--out", required=True,
                        help="Output .npz file.")
    parser.add_argument("--delete-inputs", action="store_true",
                        help="Delete merged shard files and empty shard "
                             "directories after a successful merge.")
    args = parser.parse_args()

    try:
        merge(args.inputs, args.out, delete_inputs=args.delete_inputs)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
