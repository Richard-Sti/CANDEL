#!/usr/bin/env python3
"""Plot and compare a generated BORG density field against an MCMC field."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make slab, power-spectrum, and cross-correlation plots comparing "
            "borg_forward split final_density outputs to /scalars/BORG_final_density."
        )
    )
    parser.add_argument("mcmc", type=Path, help="Native Manticore/BORG mcmc_*.h5 file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory containing split output_<iteration>.h5_<rank> files.",
    )
    parser.add_argument("--iteration", type=int, help="Iteration number. Defaults to mcmc filename.")
    parser.add_argument("--nprocs", type=int, default=28, help="Number of split files. Default: 28.")
    parser.add_argument("--slice-index", type=int, help="Slab index along axis 2. Default: middle slab.")
    parser.add_argument("--nbins", type=int, default=50, help="Number of k bins. Default: 50.")
    parser.add_argument(
        "--out-prefix",
        type=Path,
        help="Output prefix. Default: <output-dir>/borg_forward_vs_mcmc.",
    )
    return parser.parse_args()


def iteration_from_mcmc(mcmc: Path) -> int:
    match = re.fullmatch(r"mcmc_(\d+)", mcmc.stem)
    if match is None:
        raise ValueError(f"Could not infer iteration from MCMC filename: {mcmc}")
    return int(match.group(1))


def infer_output_dir(mcmc: Path) -> Path:
    chain_dir = mcmc.parent
    if chain_dir.parent.name != "chain":
        raise ValueError(
            "Could not infer output directory. Expected "
            "<run-root>/chain/<subchain>/mcmc_*.h5; pass --output-dir."
        )
    return chain_dir.parent.parent / "forward" / chain_dir.name / mcmc.stem / "rsd"


def load_generated(output_dir: Path, iteration: int, nprocs: int) -> tuple[np.ndarray, float]:
    slabs = []
    box_size = None
    for rank in range(nprocs):
        path = output_dir / f"output_{iteration:04d}.h5_{rank}"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split output: {path}")
        with h5py.File(path, "r") as handle:
            slabs.append(handle["final_density"][...])
            if box_size is None:
                box_size = float(handle["scalars/L0"][0])

    field = np.concatenate(slabs, axis=0)
    return field, float(box_size)


def load_reference(mcmc: Path) -> np.ndarray:
    with h5py.File(mcmc, "r") as handle:
        return handle["/scalars/BORG_final_density"][...]


def binned_power(field_a: np.ndarray, field_b: np.ndarray, box_size: float, nbins: int) -> dict[str, np.ndarray]:
    if field_a.shape != field_b.shape:
        raise ValueError(f"Shape mismatch: {field_a.shape} != {field_b.shape}")

    n0, n1, n2 = field_a.shape
    volume = box_size**3
    norm = volume / (n0 * n1 * n2) ** 2

    delta_a = field_a - np.mean(field_a)
    delta_b = field_b - np.mean(field_b)
    fft_a = np.fft.rfftn(delta_a)
    fft_b = np.fft.rfftn(delta_b)

    kx = 2 * np.pi * np.fft.fftfreq(n0, d=box_size / n0)
    ky = 2 * np.pi * np.fft.fftfreq(n1, d=box_size / n1)
    kz = 2 * np.pi * np.fft.rfftfreq(n2, d=box_size / n2)
    kk = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)

    p_aa = (fft_a * np.conj(fft_a)).real * norm
    p_bb = (fft_b * np.conj(fft_b)).real * norm
    p_ab = (fft_a * np.conj(fft_b)).real * norm

    k_flat = kk.ravel()
    mask = k_flat > 0
    k_flat = k_flat[mask]
    p_aa_flat = p_aa.ravel()[mask]
    p_bb_flat = p_bb.ravel()[mask]
    p_ab_flat = p_ab.ravel()[mask]

    edges = np.linspace(k_flat.min(), k_flat.max(), nbins + 1)
    idx = np.digitize(k_flat, edges) - 1
    valid = (idx >= 0) & (idx < nbins)
    idx = idx[valid]

    counts = np.bincount(idx, minlength=nbins).astype(float)
    k_sum = np.bincount(idx, weights=k_flat[valid], minlength=nbins)
    aa_sum = np.bincount(idx, weights=p_aa_flat[valid], minlength=nbins)
    bb_sum = np.bincount(idx, weights=p_bb_flat[valid], minlength=nbins)
    ab_sum = np.bincount(idx, weights=p_ab_flat[valid], minlength=nbins)

    nonzero = counts > 0
    k = k_sum[nonzero] / counts[nonzero]
    p_ref = aa_sum[nonzero] / counts[nonzero]
    p_gen = bb_sum[nonzero] / counts[nonzero]
    p_cross = ab_sum[nonzero] / counts[nonzero]
    r = p_cross / np.sqrt(p_ref * p_gen)

    return {
        "k": k,
        "p_ref": p_ref,
        "p_gen": p_gen,
        "p_cross": p_cross,
        "r": r,
        "ratio": p_gen / p_ref,
        "counts": counts[nonzero],
    }


def save_slab_plot(ref: np.ndarray, gen: np.ndarray, out_prefix: Path, slice_index: int) -> Path:
    diff = gen - ref
    ref_slice = ref[:, :, slice_index]
    gen_slice = gen[:, :, slice_index]
    diff_slice = diff[:, :, slice_index]
    denom_floor = 1e-3 * np.percentile(np.abs(ref_slice), 99)
    rel_slice = diff_slice / np.maximum(np.abs(ref_slice), denom_floor)

    vmin, vmax = np.percentile(np.concatenate([ref_slice.ravel(), gen_slice.ravel()]), [1, 99])
    dlim = np.percentile(np.abs(diff_slice), 99.5)
    rlim = np.percentile(np.abs(rel_slice), 99.5)

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2), constrained_layout=True)
    panels = [
        (ref_slice, "MCMC /scalars/BORG_final_density", vmin, vmax, "viridis"),
        (gen_slice, "Regenerated RSD final_density", vmin, vmax, "viridis"),
        (diff_slice, "Generated - MCMC", -dlim, dlim, "coolwarm"),
        (rel_slice, "Signed relative residual", -rlim, rlim, "coolwarm"),
    ]
    for ax, (image, title, lo, hi, cmap) in zip(axes, panels):
        im = ax.imshow(image.T, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_xlabel("x cell")
        ax.set_ylabel("y cell")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    path = out_prefix.with_name(out_prefix.name + "_slab.png")
    fig.suptitle(f"z-slab index {slice_index}")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_power_plot(spectra: dict[str, np.ndarray], out_prefix: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True, constrained_layout=True)

    axes[0].loglog(spectra["k"], spectra["p_ref"], label="MCMC")
    axes[0].loglog(spectra["k"], spectra["p_gen"], "--", label="Regenerated RSD")
    axes[0].set_ylabel("P(k)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].semilogx(spectra["k"], spectra["ratio"], label="P_regen / P_mcmc")
    axes[1].semilogx(spectra["k"], spectra["r"], label="cross-correlation r(k)")
    axes[1].axhline(1.0, color="0.3", lw=1)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("ratio / r(k)")
    axes[1].set_ylim(0.999, 1.001)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    path = out_prefix.with_name(out_prefix.name + "_power_cross.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_spectra_csv(spectra: dict[str, np.ndarray], out_prefix: Path) -> Path:
    path = out_prefix.with_name(out_prefix.name + "_power_cross.csv")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k", "p_mcmc", "p_generated", "p_cross", "cross_correlation", "power_ratio", "nmodes"])
        for row in zip(
            spectra["k"],
            spectra["p_ref"],
            spectra["p_gen"],
            spectra["p_cross"],
            spectra["r"],
            spectra["ratio"],
            spectra["counts"],
        ):
            writer.writerow(row)
    return path


def main() -> None:
    args = parse_args()
    mcmc = args.mcmc.expanduser().resolve()
    iteration = args.iteration if args.iteration is not None else iteration_from_mcmc(mcmc)
    output_dir = (args.output_dir if args.output_dir is not None else infer_output_dir(mcmc)).expanduser().resolve()
    out_prefix = args.out_prefix.expanduser().resolve() if args.out_prefix is not None else output_dir / "borg_forward_vs_mcmc"

    print(f"Loading MCMC field: {mcmc}", flush=True)
    ref = load_reference(mcmc)
    print(f"Loading generated split field from: {output_dir}", flush=True)
    gen, box_size = load_generated(output_dir, iteration, args.nprocs)

    if ref.shape != gen.shape:
        raise ValueError(f"Shape mismatch: reference {ref.shape}, generated {gen.shape}")

    slice_index = args.slice_index if args.slice_index is not None else ref.shape[2] // 2
    diff = gen - ref
    print(f"Field shape: {ref.shape}", flush=True)
    print(f"Box size: {box_size}", flush=True)
    print(f"Difference max_abs={np.max(np.abs(diff)):.16e}, rms={np.sqrt(np.mean(diff * diff)):.16e}", flush=True)

    slab_path = save_slab_plot(ref, gen, out_prefix, slice_index)
    spectra = binned_power(ref, gen, box_size, args.nbins)
    power_path = save_power_plot(spectra, out_prefix)
    csv_path = save_spectra_csv(spectra, out_prefix)

    print(f"Saved slab plot: {slab_path}", flush=True)
    print(f"Saved power/cross plot: {power_path}", flush=True)
    print(f"Saved spectra table: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
