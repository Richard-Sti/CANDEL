#!/usr/bin/env python3
"""Plot density and velocity slices from a packed Manticore field product."""

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
    parser.add_argument("product", type=Path, help="Packed manticore_fields_*.h5 product.")
    parser.add_argument(
        "--groups",
        nargs="+",
        help="Groups to plot. Default: all top-level groups with field datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Plot directory. Default: PRODUCT_PARENT/plots.",
    )
    parser.add_argument("--filename-prefix", default="", help="Prefix for output plot filenames.")
    parser.add_argument("--slice-axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--slice-index", type=int, help="Default: middle of the selected axis.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned plot paths without writing files.")
    return parser.parse_args()


def field_slice(dataset, axis: int, index: int, component: int | None = None) -> np.ndarray:
    selection: list[slice | int] = [slice(None), slice(None), slice(None)]
    selection[axis] = index
    if component is not None:
        selection.append(component)
    return np.asarray(dataset[tuple(selection)])


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


def draw_panel(
    fig,
    ax,
    image: np.ndarray,
    title: str,
    cmap: str,
    colorbar_label: str,
    symmetric: bool = False,
) -> None:
    vmin, vmax = robust_limits(image, symmetric=symmetric)
    im = ax.imshow(image.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("grid index")
    ax.set_ylabel("grid index")
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)


def mas_groups(group: h5py.Group) -> list[str]:
    return [name for name in ("sph", "cic") if name in group]


def overdensity_dataset(fields: h5py.Group) -> h5py.Dataset | None:
    for name in ("overdensity", "density_contrast"):
        if name in fields:
            return fields[name]
    return None


def save_field_plot(
    handle: h5py.File,
    group_name: str,
    mas: str,
    output_dir: Path,
    filename_prefix: str,
    axis: int,
    index: int,
) -> Path | None:
    group = handle[group_name]
    if mas not in group or "velocity" not in group[mas]:
        return None

    fields = group[mas]
    overdensity = overdensity_dataset(fields)
    if overdensity is None:
        return None

    velocity = fields["velocity"]
    components = [field_slice(velocity, axis, index, component=i) for i in range(3)]
    speed = np.sqrt(sum(component * component for component in components))
    units = velocity.attrs.get("units", "km/s")
    velocity_label = f"velocity [{units}]" if units else "velocity"
    speed_label = f"speed [{units}]" if units else "speed"

    fig, axes = plt.subplots(1, 5, figsize=(22.0, 4.4), constrained_layout=True)
    draw_panel(
        fig,
        axes[0],
        field_slice(overdensity, axis, index),
        f"{group_name} {mas.upper()} overdensity",
        "coolwarm",
        r"$\delta = \rho / \langle\rho\rangle - 1$",
        symmetric=False,
    )
    for ax, image, label in zip(axes[1:4], components, ("vx", "vy", "vz")):
        draw_panel(fig, ax, image, f"{group_name} {mas.upper()} {label}", "coolwarm", velocity_label, symmetric=True)
    draw_panel(fig, axes[4], speed, f"{group_name} {mas.upper()} |v|", "magma", speed_label, symmetric=False)
    fig.suptitle(f"{group_name}: axis {axis} slice {index}")

    path = output_dir / f"{filename_prefix}{group_name}_{mas}_field_slices.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def available_groups(handle: h5py.File) -> list[str]:
    groups = []
    for name, item in handle.items():
        if not isinstance(item, h5py.Group):
            continue
        has_mas = any(mas in item and "velocity" in item[mas] for mas in ("sph", "cic"))
        if "final_density" in item or has_mas:
            groups.append(name)
    return groups


def slice_shape(group: h5py.Group) -> tuple[int, int, int]:
    if "final_density" in group:
        return group["final_density"].shape
    for first_mas in mas_groups(group):
        fields = group[first_mas]
        dataset = overdensity_dataset(fields)
        if dataset is not None:
            return dataset.shape
        if "velocity" in fields:
            return fields["velocity"].shape[:3]
    raise ValueError("No plottable field found")


def dry_run_field_plot(
    handle: h5py.File,
    group_name: str,
    mas: str,
    output_dir: Path,
    filename_prefix: str,
    axis: int,
    index: int,
) -> Path | None:
    group = handle[group_name]
    if mas not in group or "velocity" not in group[mas]:
        return None

    fields = group[mas]
    overdensity = overdensity_dataset(fields)
    if overdensity is None:
        return None

    velocity = fields["velocity"]
    if velocity.shape[:3] != overdensity.shape:
        raise ValueError(
            f"Shape mismatch for /{group_name}/{mas}: "
            f"overdensity {overdensity.shape}, velocity {velocity.shape}"
        )
    if velocity.shape[3:] != (3,):
        raise ValueError(f"Expected /{group_name}/{mas}/velocity shape (N,N,N,3), got {velocity.shape}")

    units = velocity.attrs.get("units", "km/s")
    path = output_dir / f"{filename_prefix}{group_name}_{mas}_field_slices.png"
    print(
        f"Would plot /{group_name}/{mas}: overdensity={overdensity.shape}, "
        f"velocity={velocity.shape} [{units}], axis={axis}, slice={index} -> {path}",
        flush=True,
    )
    return path


def main() -> None:
    args = parse_args()
    product = args.product.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else product.parent / "plots"
    )
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(product, "r") as handle:
        groups = args.groups if args.groups is not None else available_groups(handle)
        if not groups:
            raise ValueError(f"No plottable groups found in {product}")

        saved = []
        for group_name in groups:
            if group_name not in handle:
                raise KeyError(f"Missing group /{group_name} in {product}")
            group = handle[group_name]
            shape = slice_shape(group)
            index = args.slice_index if args.slice_index is not None else shape[args.slice_axis] // 2
            if index < 0 or index >= shape[args.slice_axis]:
                raise ValueError(f"Slice index {index} outside axis {args.slice_axis} with size {shape[args.slice_axis]}")

            for mas in mas_groups(group):
                if args.dry_run:
                    field_path = dry_run_field_plot(
                        handle,
                        group_name,
                        mas,
                        output_dir,
                        args.filename_prefix,
                        args.slice_axis,
                        index,
                    )
                else:
                    field_path = save_field_plot(
                        handle,
                        group_name,
                        mas,
                        output_dir,
                        args.filename_prefix,
                        args.slice_axis,
                        index,
                    )
                if field_path is not None:
                    saved.append(field_path)

    for path in saved:
        if args.dry_run:
            print(f"Dry run OK: {path}", flush=True)
        else:
            print(f"Saved plot: {path}", flush=True)


if __name__ == "__main__":
    main()
