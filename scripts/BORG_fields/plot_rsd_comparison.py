#!/usr/bin/env python3
"""Plot an RSD validation density slice against the native BORG sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcmc", type=Path, required=True)
    parser.add_argument("--output-pattern", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--nprocs", type=int, required=True)
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--index", type=int, help="Default: middle slice along --axis.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def field_slice(dataset, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return np.asarray(dataset[index, :, :])
    if axis == 1:
        return np.asarray(dataset[:, index, :])
    return np.asarray(dataset[:, :, index])


def generated_slice(output_pattern: Path, iteration: int, nprocs: int, axis: int, index: int) -> np.ndarray:
    output = Path(str(output_pattern) % iteration)
    pieces = []
    offset = 0
    for rank in range(nprocs):
        with h5py.File(Path(f"{output}_{rank}"), "r") as handle:
            density = handle["final_density"]
            n0 = density.shape[0]
            if axis == 0:
                if offset <= index < offset + n0:
                    return np.asarray(density[index - offset, :, :])
                offset += n0
                continue
            if n0 == 0:
                continue
            if axis == 1:
                pieces.append(np.asarray(density[:, index, :]))
            else:
                pieces.append(np.asarray(density[:, :, index]))
            offset += n0

    if axis == 0:
        raise ValueError(f"Slice index {index} not found in split outputs")
    if not pieces:
        raise ValueError("No non-empty split outputs found")
    return np.concatenate(pieces, axis=0)


def robust_limits(image: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    if symmetric:
        limit = float(np.percentile(np.abs(finite), 99))
        return -limit, limit
    lo, hi = np.percentile(finite, [1, 99])
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def draw_panel(fig, ax, image: np.ndarray, title: str, label: str, symmetric: bool = False) -> None:
    vmin, vmax = robust_limits(image, symmetric=symmetric)
    im = ax.imshow(image.T, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("grid index")
    ax.set_ylabel("grid index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(label)


def main() -> None:
    args = parse_args()
    with h5py.File(args.mcmc.expanduser().resolve(), "r") as handle:
        ref = handle["/scalars/BORG_final_density"]
        index = args.index if args.index is not None else ref.shape[args.axis] // 2
        if index < 0 or index >= ref.shape[args.axis]:
            raise ValueError(f"Slice index {index} outside axis {args.axis} with size {ref.shape[args.axis]}")
        ref_slice = field_slice(ref, args.axis, index)

    gen_slice = generated_slice(args.output_pattern.expanduser().resolve(), args.iteration, args.nprocs, args.axis, index)
    diff = gen_slice - ref_slice

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), constrained_layout=True)
    draw_panel(fig, axes[0], gen_slice, "generated RSD density", "density")
    draw_panel(fig, axes[1], ref_slice, "reference BORG density", "density")
    draw_panel(fig, axes[2], diff, "generated - reference", "density difference", symmetric=True)
    fig.suptitle(f"RSD validation: axis {args.axis} slice {index}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(f"Saved RSD comparison plot: {output}", flush=True)


if __name__ == "__main__":
    main()
