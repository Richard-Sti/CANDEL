#!/usr/bin/env python
"""Compare two gridded SPH density fields and their cross-power spectrum."""
from argparse import ArgumentParser
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import Pk_library as PKL  # noqa: E402

from borg_field_config import configured_chain_path  # noqa: E402

DEFAULT_OUTDIR = (
    Path(__file__).resolve().parents[2] / "results" / "BORG_field_checks"
)
BOXSIZE = 681.0  # Mpc / h


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        type=int,
        help=(
            "Use active BORG chain field_output_dir and reference_fields_dir "
            "mcmc_<step> paths."
        ),
    )
    parser.add_argument("--field-a", type=Path, help="Reference HDF5 field.")
    parser.add_argument("--field-b", type=Path, help="Comparison HDF5 field.")
    parser.add_argument("--label-a", default="BORG forward SPH")
    parser.add_argument("--label-b", default="February N-body SPH")
    parser.add_argument("--out-stem", help="Output filename stem.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Ignore cached downsampled arrays.")
    return parser.parse_args()


def default_paths(step):
    return (
        configured_chain_path("field_output_dir") / f"mcmc_{step}.hdf5",
        configured_chain_path("reference_fields_dir") / f"mcmc_{step}.hdf5",
    )


def dataset_name(handle):
    if "overdensity" in handle:
        return "overdensity"
    if "density" in handle:
        return "density"
    raise KeyError("Expected either a `density` or `overdensity` dataset.")


def downsample_density(dataset, target_shape):
    source_shape = dataset.shape
    factors = tuple(s // t for s, t in zip(source_shape, target_shape))
    if any(s % t != 0 for s, t in zip(source_shape, target_shape)):
        raise ValueError(f"Cannot block-average {source_shape} to {target_shape}.")

    out = np.empty(target_shape, dtype=np.float32)
    reshape_shape = (
        factors[0],
        target_shape[1],
        factors[1],
        target_shape[2],
        factors[2],
    )
    for ix in range(target_shape[0]):
        slab = dataset[factors[0] * ix:factors[0] * (ix + 1), :, :]
        block = slab.reshape(reshape_shape)
        out[ix] = block.mean(axis=(0, 2, 4), dtype=np.float64)
        if (ix + 1) % 32 == 0 or ix + 1 == target_shape[0]:
            print(f"Downsampled {ix + 1} / {target_shape[0]} x-slabs", flush=True)
    return out


def load_delta(path, target_shape=None):
    with h5py.File(path, "r") as handle:
        name = dataset_name(handle)
        dataset = handle[name]
        if name == "overdensity":
            delta = dataset[...].astype(np.float32, copy=False)
        elif target_shape is not None and dataset.shape != target_shape:
            density = downsample_density(dataset, target_shape)
            delta = density / np.mean(density, dtype=np.float64) - 1.0
        else:
            density = dataset[...].astype(np.float32, copy=False)
            delta = density / np.mean(density, dtype=np.float64) - 1.0

    return delta.astype(np.float32, copy=False)


def pearson_corr(a, b):
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask] - np.mean(x[mask])
    y = y[mask] - np.mean(y[mask])
    return float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))


def robust_limits(*arrays):
    values = np.concatenate([np.asarray(a).ravel() for a in arrays])
    return np.nanpercentile(values, [1.0, 99.0])


def compute_stats(field_a, field_b):
    diff = field_b - field_a
    return {
        "pearson_r": pearson_corr(field_a, field_b),
        "rms_difference": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))),
        "mean_abs_difference": float(np.mean(np.abs(diff.astype(np.float64)))),
        "mean_difference": float(np.mean(diff, dtype=np.float64)),
        "field_a_mean": float(np.mean(field_a, dtype=np.float64)),
        "field_a_std": float(np.std(field_a, dtype=np.float64)),
        "field_b_mean": float(np.mean(field_b, dtype=np.float64)),
        "field_b_std": float(np.std(field_b, dtype=np.float64)),
    }


def make_slice_plot(field_a, field_b, stats, args, outpng):
    diff = field_b - field_a
    ratio = (field_b + 1.0) / np.clip(field_a + 1.0, 1e-6, None)

    idx = field_a.shape[0] // 2
    slice_a = field_a[idx]
    slice_b = field_b[idx]
    slice_diff = diff[idx]
    slice_ratio = ratio[idx]
    vmin, vmax = robust_limits(slice_a, slice_b)
    dlim = float(np.nanpercentile(np.abs(slice_diff), 99.0))
    rmin, rmax = np.nanpercentile(slice_ratio, [1.0, 99.0])

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.5), constrained_layout=True)
    im = axes[0, 0].imshow(slice_a.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(args.label_a)
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)

    im = axes[0, 1].imshow(slice_b.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(args.label_b)
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = axes[1, 0].imshow(slice_diff.T, origin="lower", cmap="coolwarm", vmin=-dlim, vmax=dlim)
    axes[1, 0].set_title(f"{args.label_b} - {args.label_a}")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(slice_ratio.T, origin="lower", cmap="viridis", vmin=rmin, vmax=rmax)
    axes[1, 1].set_title(f"(1 + {args.label_b}) / (1 + {args.label_a})")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

    for ax in axes.flat:
        ax.set_xlabel("grid y")
        ax.set_ylabel("grid z")

    fig.suptitle(
        (
            f"{args.label_b} vs {args.label_a}, x mid-plane\n"
            f"Pearson r={stats['pearson_r']:.5f}, "
            f"RMS(B-A)={stats['rms_difference']:.4g}, "
            f"mean(B-A)={stats['mean_difference']:.3g}"
        ),
        fontsize=11,
    )
    fig.savefig(outpng, dpi=220)
    plt.close(fig)


def compute_power(field_a, field_b, threads):
    a = field_a - np.mean(field_a, dtype=np.float64)
    b = field_b - np.mean(field_b, dtype=np.float64)
    spectra = PKL.XPk(
        [a.astype(np.float32), b.astype(np.float32)],
        BOXSIZE,
        axis=0,
        MAS=["None", "None"],
        threads=threads,
    )
    p_a = spectra.Pk[:, 0, 0]
    p_b = spectra.Pk[:, 0, 1]
    p_cross = spectra.XPk[:, 0, 0]
    r = p_cross / np.sqrt(p_a * p_b)
    return {
        "k": spectra.k3D,
        "nmodes": spectra.Nmodes3D.astype(np.int64),
        "p_a": p_a,
        "p_b": p_b,
        "p_cross": p_cross,
        "r": r,
    }


def write_power_csv(power, outcsv):
    with outcsv.open("w") as handle:
        handle.write("k_h_per_Mpc,nmodes,p_a,p_b,p_cross,r\n")
        for i in range(len(power["k"])):
            handle.write(
                f"{power['k'][i]:.10e},{power['nmodes'][i]},"
                f"{power['p_a'][i]:.10e},{power['p_b'][i]:.10e},"
                f"{power['p_cross'][i]:.10e},{power['r'][i]:.10e}\n"
            )


def make_power_plot(power, args, outpng):
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True)
    k = power["k"]

    ax = axes[0]
    ax.loglog(k, power["p_a"], lw=1.3, label=args.label_a)
    ax.loglog(k, power["p_b"], lw=1.3, label=args.label_b)
    ax.loglog(k, np.abs(power["p_cross"]), lw=1.1, ls="--", label=r"$|P_\times|$")
    ax.set_ylabel(r"$P(k)~[(h^{-1}{\rm Mpc})^3]$")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(k, power["r"], marker="o", ms=3, lw=1.1)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.axhline(1.0, color="0.8", lw=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$k~[h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$r(k)$")

    fig.suptitle(f"{args.label_b} vs {args.label_a}")
    fig.tight_layout()
    fig.savefig(outpng, dpi=220)
    plt.close(fig)


def resolve_inputs(args):
    if args.step is not None:
        field_a, field_b = default_paths(args.step)
        if args.out_stem is None:
            args.out_stem = f"feb_nbody_vs_borg_forward_mcmc{args.step}_density"
    else:
        if args.field_a is None or args.field_b is None:
            raise ValueError("Specify either --step or both --field-a and --field-b.")
        field_a, field_b = args.field_a, args.field_b
        if args.out_stem is None:
            args.out_stem = f"{field_b.stem}_vs_{field_a.stem}_density"

    return field_a, field_b


def load_cached_fields(path):
    cached = np.load(path)
    if "field_a_delta" in cached and "field_b_delta" in cached:
        return (
            cached["field_a_delta"].astype(np.float32, copy=False),
            cached["field_b_delta"].astype(np.float32, copy=False),
        )
    if "forward_delta" in cached and "feb_delta" in cached:
        return (
            cached["forward_delta"].astype(np.float32, copy=False),
            cached["feb_delta"].astype(np.float32, copy=False),
        )
    return None


def main():
    args = parse_args()
    field_a_path, field_b_path = resolve_inputs(args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    outnpz = args.outdir / f"{args.out_stem}_stats.npz"
    outcomparison = args.outdir / f"{args.out_stem}_comparison.png"
    outpower = args.outdir / f"{args.out_stem}_pylians_power.png"
    outcsv = args.outdir / f"{args.out_stem}_pylians_power.csv"

    cached_fields = None if args.force or not outnpz.exists() else load_cached_fields(outnpz)
    if cached_fields is None:
        field_a = load_delta(field_a_path)
        field_b = load_delta(field_b_path, target_shape=field_a.shape)
    else:
        field_a, field_b = cached_fields

    if field_a.shape != field_b.shape:
        raise ValueError(f"Field shapes differ after loading: {field_a.shape} vs {field_b.shape}.")

    stats = compute_stats(field_a, field_b)
    np.savez(
        outnpz,
        field_a_delta=field_a,
        field_b_delta=field_b,
        field_a_path=str(field_a_path),
        field_b_path=str(field_b_path),
        label_a=args.label_a,
        label_b=args.label_b,
        **stats,
    )
    make_slice_plot(field_a, field_b, stats, args, outcomparison)

    power = compute_power(field_a, field_b, args.threads)
    write_power_csv(power, outcsv)
    make_power_plot(power, args, outpower)

    low = power["k"] < 0.05
    mid = (power["k"] >= 0.05) & (power["k"] < 0.2)
    print(f"Wrote {outcomparison}")
    print(f"Wrote {outpower}")
    print(f"Wrote {outcsv}")
    print(f"Wrote {outnpz}")
    print(f"Pearson density correlation: {stats['pearson_r']:.8f}")
    print(f"RMS overdensity difference: {stats['rms_difference']:.8g}")
    print(f"Mean absolute overdensity difference: {stats['mean_abs_difference']:.8g}")
    print(f"Mean overdensity difference: {stats['mean_difference']:.8g}")
    print(f"{args.label_a} mean/std: {stats['field_a_mean']:.8g} / {stats['field_a_std']:.8g}")
    print(f"{args.label_b} mean/std: {stats['field_b_mean']:.8g} / {stats['field_b_std']:.8g}")
    print(f"Mean r(k), k<0.05: {np.nanmean(power['r'][low]):.6f}")
    print(f"Mean r(k), 0.05<=k<0.2: {np.nanmean(power['r'][mid]):.6f}")


if __name__ == "__main__":
    main()
