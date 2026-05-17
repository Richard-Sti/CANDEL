#!/usr/bin/env python3
"""Compute Pylians cross-correlation for native and generated RSD fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import Pk_library as PKL  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcmc", type=Path, required=True)
    parser.add_argument("--output-pattern", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--nprocs", type=int, required=True)
    parser.add_argument("--boxsize", type=float, default=681.0)
    parser.add_argument("--axis", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--kmax", type=float, default=0.2)
    parser.add_argument("--min-mean-r", type=float, default=0.99)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    return parser.parse_args()


def output_path(output_pattern: Path, iteration: int) -> Path:
    return Path(str(output_pattern) % iteration)


def load_generated_rsd(output_pattern: Path, iteration: int, nprocs: int) -> np.ndarray:
    output = output_path(output_pattern, iteration)
    paths = [Path(f"{output}_{rank}") for rank in range(nprocs)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing split output files:\n" + "\n".join(str(path) for path in missing))

    slab_shapes = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            slab_shapes.append(handle["final_density"].shape)

    shape = (sum(item[0] for item in slab_shapes),) + slab_shapes[0][1:]
    field = np.empty(shape, dtype=np.float32)
    offset = 0
    for path in paths:
        with h5py.File(path, "r") as handle:
            slab = handle["final_density"]
            stop = offset + slab.shape[0]
            field[offset:stop] = slab[...].astype(np.float32, copy=False)
            offset = stop
    return field


def load_native_rsd(mcmc: Path) -> np.ndarray:
    with h5py.File(mcmc, "r") as handle:
        return handle["/scalars/BORG_final_density"][...].astype(np.float32, copy=False)


def compute_power(reference: np.ndarray, generated: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    reference = reference - np.mean(reference, dtype=np.float64)
    generated = generated - np.mean(generated, dtype=np.float64)
    spectra = PKL.XPk(
        [reference.astype(np.float32), generated.astype(np.float32)],
        args.boxsize,
        axis=args.axis,
        MAS=["None", "None"],
        threads=args.threads,
    )
    p_reference = spectra.Pk[:, 0, 0]
    p_generated = spectra.Pk[:, 0, 1]
    p_cross = spectra.XPk[:, 0, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = p_cross / np.sqrt(p_reference * p_generated)
    return {
        "k": spectra.k3D,
        "nmodes": spectra.Nmodes3D.astype(np.int64),
        "p_reference": p_reference,
        "p_generated": p_generated,
        "p_cross": p_cross,
        "r": r,
    }


def write_csv(path: Path, power: dict[str, np.ndarray]) -> None:
    with path.open("w") as handle:
        handle.write("k_h_per_Mpc,nmodes,p_borg_rsd,p_generated_rsd,p_cross,r\n")
        for i in range(len(power["k"])):
            handle.write(
                f"{power['k'][i]:.10e},{power['nmodes'][i]},"
                f"{power['p_reference'][i]:.10e},{power['p_generated'][i]:.10e},"
                f"{power['p_cross'][i]:.10e},{power['r'][i]:.10e}\n"
            )


def write_plot(path: Path, power: dict[str, np.ndarray], metrics: dict[str, float | int | bool]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True)
    k = power["k"]

    ax = axes[0]
    ax.loglog(k, power["p_reference"], lw=1.3, label="BORG sample RSD")
    ax.loglog(k, power["p_generated"], lw=1.3, label="generated RSD")
    ax.loglog(k, np.abs(power["p_cross"]), lw=1.1, ls="--", label=r"$|P_\times|$")
    ax.set_ylabel(r"$P(k)~[(h^{-1}{\rm Mpc})^3]$")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(k, power["r"], marker="o", ms=3, lw=1.1)
    ax.axhline(metrics["min_mean_r"], color="0.35", lw=0.9, ls=":")
    ax.axhline(1.0, color="0.8", lw=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$k~[h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$r(k)$")

    fig.suptitle(
        f"RSD cross-correlation, mean r(k<={metrics['kmax']:.3g})="
        f"{metrics['mean_r_constrained']:.6f}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def nanmean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)

    reference = load_native_rsd(args.mcmc)
    generated = load_generated_rsd(args.output_pattern, args.iteration, args.nprocs)
    if reference.shape != generated.shape:
        raise ValueError(f"Shape mismatch: BORG RSD {reference.shape} != generated RSD {generated.shape}")

    power = compute_power(reference, generated, args)
    finite = np.isfinite(power["r"])
    constrained = finite & (power["k"] > 0.0) & (power["k"] <= args.kmax)
    low = finite & (power["k"] > 0.0) & (power["k"] < 0.05)
    mid = finite & (power["k"] >= 0.05) & (power["k"] < 0.2)
    mean_r = nanmean(power["r"][constrained])
    metrics = {
        "boxsize": float(args.boxsize),
        "axis": int(args.axis),
        "threads": int(args.threads),
        "kmax": float(args.kmax),
        "min_mean_r": float(args.min_mean_r),
        "mean_r_constrained": mean_r,
        "mean_r_k_lt_0p05": nanmean(power["r"][low]),
        "mean_r_0p05_to_0p2": nanmean(power["r"][mid]),
        "min_r_constrained": float(np.nanmin(power["r"][constrained])) if np.any(constrained) else float("nan"),
        "nmodes_constrained": int(np.sum(power["nmodes"][constrained])),
        "passed": bool(np.isfinite(mean_r) and mean_r >= args.min_mean_r),
    }

    stem = f"rsd_cross_correlation_mcmc_{args.iteration}"
    csv_path = args.output_dir / f"{stem}.csv"
    plot_path = args.output_dir / f"{stem}.png"
    write_csv(csv_path, power)
    write_plot(plot_path, power, metrics)
    metrics["csv"] = str(csv_path)
    metrics["plot"] = str(plot_path)
    args.metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    status = "PASS" if metrics["passed"] else "FAIL"
    print(
        f"RSD Pylians cross-correlation: {status} "
        f"mean_r(k<={args.kmax:g})={metrics['mean_r_constrained']:.8f} "
        f"threshold={args.min_mean_r:g}",
        flush=True,
    )
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {plot_path}", flush=True)
    print(f"Wrote {args.metrics_json}", flush=True)


if __name__ == "__main__":
    main()
