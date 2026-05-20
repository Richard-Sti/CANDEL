#!/usr/bin/env python
"""Plot local COLA-Manticore density fields near the observer."""
from argparse import ArgumentParser
import csv
from pathlib import Path
import re

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from candel.field.loader import ManticoreCOLA_FieldLoader  # noqa: E402


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
SOURCE_ROOT = (
    Path("/mnt/extraspace/rstiskalek/MANTICORE")
    / "2MPP_MULTIBIN_N256_DES_V2"
    / "forward_sph_fields"
)
OUTDIR = (
    ROOT
    / "results"
    / "TRGBH0_paper"
    / "manticore_fields_const_sigv"
    / "plots"
)
DEFAULT_FIELDS = (0, 1, 2, 24, 47)
DEFAULT_RADIUS = 10.0
FIGURE_DPI = 400
FIELD_RE = re.compile(r"mcmc_(\d+)\.hdf5$")


def parse_fields(text):
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return tuple(out)


def discover_fields(source_root):
    fields = []
    for path in Path(source_root).glob("mcmc_*.hdf5"):
        match = FIELD_RE.fullmatch(path.name)
        if match is not None:
            fields.append(int(match.group(1)))
    if not fields:
        raise FileNotFoundError(f"No `mcmc_*.hdf5` files in `{source_root}`.")
    return tuple(sorted(fields))


def local_axis(n, boxsize, observer_coord, radius):
    dx = boxsize / n
    coords = (np.arange(n, dtype=np.float32) + 0.5) * dx - observer_coord
    idx = np.flatnonzero(np.abs(coords) <= radius)
    if idx.size == 0:
        idx = np.asarray([int(np.argmin(np.abs(coords)))])
    sl = slice(int(idx[0]), int(idx[-1]) + 1)
    return coords[sl], sl, dx


def central_indices(coords):
    abs_coords = np.abs(coords)
    return np.flatnonzero(np.isclose(abs_coords, abs_coords.min()))


def load_local_field(field, source_root, radius, load_velocity=True):
    loader = ManticoreCOLA_FieldLoader(
        nsim=int(field), fpath_root=str(source_root))
    path = Path(loader.fname)
    if not path.exists():
        raise FileNotFoundError(f"Missing COLA field file `{path}`.")

    with h5py.File(path, "r") as handle:
        dset = handle["overdensity"]
        n = dset.shape[0]
        x, sx, dx = local_axis(n, loader.boxsize,
                               loader.observer_pos[0], radius)
        y, sy, _ = local_axis(n, loader.boxsize,
                              loader.observer_pos[1], radius)
        z, sz, _ = local_axis(n, loader.boxsize,
                              loader.observer_pos[2], radius)
        delta = np.asarray(dset[sx, sy, sz], dtype=np.float32)
        if load_velocity:
            velocity = np.asarray(
                handle["velocity"][sx, sy, sz, :], dtype=np.float32)
        else:
            velocity = None

    rho = np.clip(1.0 + delta, 1e-4, None)
    item = {
        "field": int(field),
        "rho": rho,
        "x": x,
        "y": y,
        "z": z,
        "dx": dx,
    }
    if velocity is not None:
        xx = x[:, None, None]
        yy = y[None, :, None]
        zz = z[None, None, :]
        r = np.sqrt(xx**2 + yy**2 + zz**2)
        r_safe = np.where(r > 0.0, r, np.inf)
        item["vrad"] = (
            velocity[..., 0] * xx / r_safe
            + velocity[..., 1] * yy / r_safe
            + velocity[..., 2] * zz / r_safe
        ).astype(np.float32)
    return item


def load_local_density(field, source_root, radius):
    return load_local_field(
        field, source_root, radius, load_velocity=False)


def slice_images(item):
    logrho = np.log10(item["rho"])
    cx = central_indices(item["x"])
    cy = central_indices(item["y"])
    cz = central_indices(item["z"])
    out = {
        "xy": logrho[:, :, cz].mean(axis=2),
        "xz": logrho[:, cy, :].mean(axis=1),
        "yz": logrho[cx, :, :].mean(axis=0),
    }
    if "vrad" in item:
        vrad = item["vrad"]
        out.update({
            "vrad_xy": vrad[:, :, cz].mean(axis=2),
            "vrad_xz": vrad[:, cy, :].mean(axis=1),
            "vrad_yz": vrad[cx, :, :].mean(axis=0),
        })
    return out


def scalar_slices(arr, x, y, z):
    cx = central_indices(x)
    cy = central_indices(y)
    cz = central_indices(z)
    return {
        "xy": arr[:, :, cz].mean(axis=2),
        "xz": arr[:, cy, :].mean(axis=1),
        "yz": arr[cx, :, :].mean(axis=0),
    }


def extent(a, b, dx):
    return (
        float(a[0] - 0.5 * dx),
        float(a[-1] + 0.5 * dx),
        float(b[0] - 0.5 * dx),
        float(b[-1] + 0.5 * dx),
    )


def save_pdf_png(fig, out_pdf):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_pdf, out_png


def plot_slices(items, radius, outdir):
    images = {item["field"]: slice_images(item) for item in items}
    rho_values = np.concatenate([
        image.reshape(-1)
        for per_field in images.values()
        for key, image in per_field.items()
        if not key.startswith("vrad_")
    ])
    vrad_values = np.concatenate([
        image.reshape(-1)
        for per_field in images.values()
        for key, image in per_field.items()
        if key.startswith("vrad_")
    ])
    vmin, vmax = np.nanpercentile(rho_values, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = -0.5, 1.0
    vabs = np.nanpercentile(np.abs(vrad_values), 98)
    if not np.isfinite(vabs) or vabs <= 0.0:
        vabs = 300.0

    cols = [
        ("xy", "x", "y", "density"),
        ("xz", "x", "z", "density"),
        ("yz", "y", "z", "density"),
        ("vrad_xy", "x", "y", "velocity"),
        ("vrad_xz", "x", "z", "velocity"),
        ("vrad_yz", "y", "z", "velocity"),
    ]
    fig, axes = plt.subplots(
        len(items), len(cols), figsize=(15.0, 2.25 * len(items)),
        sharex=False, sharey=False, constrained_layout=True)
    axes = np.atleast_2d(axes)

    rho_mappable = None
    vrad_mappable = None
    for row, item in enumerate(items):
        field = item["field"]
        for col, (name, xlabel, ylabel, quantity) in enumerate(cols):
            ax = axes[row, col]
            slice_name = name.replace("vrad_", "")
            if slice_name == "xy":
                img = images[field][name]
                xy_extent = extent(item["x"], item["y"], item["dx"])
            elif slice_name == "xz":
                img = images[field][name]
                xy_extent = extent(item["x"], item["z"], item["dx"])
            else:
                img = images[field][name]
                xy_extent = extent(item["y"], item["z"], item["dx"])

            if quantity == "density":
                rho_mappable = ax.imshow(
                    img.T, origin="lower", extent=xy_extent, cmap="magma",
                    vmin=vmin, vmax=vmax, interpolation="nearest",
                    aspect="equal")
            else:
                vrad_mappable = ax.imshow(
                    img.T, origin="lower", extent=xy_extent,
                    cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                    interpolation="nearest", aspect="equal")
            ax.add_patch(Circle(
                (0.0, 0.0), radius, fill=False, color="white",
                lw=0.8, ls="--", alpha=0.8))
            ax.axhline(0.0, color="white", lw=0.4, alpha=0.6)
            ax.axvline(0.0, color="white", lw=0.4, alpha=0.6)
            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)
            ax.set_xlabel(fr"${xlabel}$ [$h^{{-1}}$ Mpc]")
            ax.set_ylabel(fr"${ylabel}$ [$h^{{-1}}$ Mpc]")
            if row == 0:
                prefix = r"$\log_{10}(1+\delta)$" if quantity == "density" \
                    else r"$v_r$"
                ax.set_title(f"{prefix} {slice_name.upper()}")
            if col == 0:
                ax.text(
                    0.03, 0.95, f"field {field}", transform=ax.transAxes,
                    ha="left", va="top", color="white", fontsize=9,
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "black",
                        "edgecolor": "none",
                        "alpha": 0.45,
                    })

    cbar = fig.colorbar(
        rho_mappable, ax=axes[:, :3].ravel().tolist(),
        shrink=0.8, pad=0.01)
    cbar.set_label(r"$\log_{10}(1+\delta)$")
    cbar = fig.colorbar(
        vrad_mappable, ax=axes[:, 3:].ravel().tolist(),
        shrink=0.8, pad=0.01)
    cbar.set_label(r"$v_r$ [km s$^{-1}$]")

    suffix = "_".join(str(item["field"]) for item in items)
    out_pdf = outdir / (
        f"cola_density_velocity_near_origin_slices_fields_{suffix}.pdf")
    return save_pdf_png(fig, out_pdf)


def density_summary(item, radius):
    x, y, z = item["x"], item["y"], item["z"]
    rho = item["rho"]
    r = np.sqrt(
        x[:, None, None]**2 + y[None, :, None]**2 + z[None, None, :]**2)
    mask = r <= radius
    values = rho[mask]
    values5 = rho[r <= 5.0]
    return {
        "field": item["field"],
        "n_voxels_r10": int(values.size),
        "mean_rho_r10": float(np.mean(values)),
        "std_rho_r10": float(np.std(values, ddof=1)),
        "median_rho_r10": float(np.median(values)),
        "q16_rho_r10": float(np.quantile(values, 0.16)),
        "q84_rho_r10": float(np.quantile(values, 0.84)),
        "mean_rho_r5": float(np.mean(values5)) if values5.size else np.nan,
        "std_rho_r5": (
            float(np.std(values5, ddof=1)) if values5.size > 1 else np.nan),
        "median_rho_r5": (
            float(np.median(values5)) if values5.size else np.nan),
    }


def compute_density_moments(fields, source_root, radius):
    items = [
        load_local_field(field, source_root, radius, load_velocity=True)
        for field in fields
    ]
    ref = items[0]
    for item in items[1:]:
        if item["rho"].shape != ref["rho"].shape:
            raise ValueError("All local density cubes must share shape.")
        for key in ("x", "y", "z"):
            if not np.array_equal(item[key], ref[key]):
                raise ValueError("All local density cubes must share axes.")

    rho_stack = np.stack([item["rho"] for item in items]).astype(np.float32)
    vrad_stack = np.stack([item["vrad"] for item in items]).astype(np.float32)
    return {
        "fields": tuple(int(item["field"]) for item in items),
        "rho_stack": rho_stack,
        "vrad_stack": vrad_stack,
        "mean": np.mean(rho_stack, axis=0),
        "std": np.std(rho_stack, axis=0, ddof=1),
        "mean_vrad": np.mean(vrad_stack, axis=0),
        "std_vrad": np.std(vrad_stack, axis=0, ddof=1),
        "x": ref["x"],
        "y": ref["y"],
        "z": ref["z"],
        "dx": ref["dx"],
    }


def plot_radial_profiles(items, radius, outdir, density_moments=None):
    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    bins = np.linspace(0.0, radius, 6)
    centres = 0.5 * (bins[:-1] + bins[1:])

    if density_moments is not None:
        x = density_moments["x"]
        y = density_moments["y"]
        z = density_moments["z"]
        r = np.sqrt(
            x[:, None, None]**2 + y[None, :, None]**2
            + z[None, None, :]**2)
        stack = density_moments["rho_stack"]
        means = []
        stds = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (r >= lo) & (r < hi)
            values = stack[:, mask].reshape(-1)
            means.append(np.nan if values.size == 0 else np.mean(values))
            stds.append(
                np.nan if values.size < 2 else np.std(values, ddof=1))
        means = np.asarray(means)
        stds = np.asarray(stds)
        ax.fill_between(
            centres, means - stds, means + stds,
            color="0.75", alpha=0.45, lw=0.0,
            label="all-sample mean +/- std")
        ax.plot(
            centres, means, color="black", marker="s", lw=1.8,
            label=f"all {len(density_moments['fields'])} samples")

    for item in items:
        x, y, z = item["x"], item["y"], item["z"]
        rho = item["rho"]
        r = np.sqrt(
            x[:, None, None]**2 + y[None, :, None]**2
            + z[None, None, :]**2)
        means = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (r >= lo) & (r < hi)
            means.append(np.nan if not np.any(mask) else np.mean(rho[mask]))
        ax.plot(centres, means, marker="o", lw=1.4,
                label=f"field {item['field']}")

    ax.axhline(1.0, color="0.35", lw=1.0, ls="--")
    ax.set_xlabel(r"$r$ [$h^{-1}$ Mpc]")
    ax.set_ylabel(r"mean $1+\delta$")
    ax.set_xlim(0.0, radius)
    ax.legend(frameon=False, ncol=2, fontsize=8)

    suffix = "_".join(str(item["field"]) for item in items)
    out_pdf = outdir / (
        f"cola_density_near_origin_radial_profiles_fields_{suffix}.pdf")
    return save_pdf_png(fig, out_pdf)


def write_summary(items, radius, outdir):
    rows = [density_summary(item, radius) for item in items]
    suffix = "_".join(str(item["field"]) for item in items)
    out = outdir / f"cola_density_near_origin_summary_fields_{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def plot_density_moments(moments, radius, outdir):
    mean_slices = scalar_slices(
        moments["mean"], moments["x"], moments["y"], moments["z"])
    std_slices = scalar_slices(
        moments["std"], moments["x"], moments["y"], moments["z"])
    mean_vrad_slices = scalar_slices(
        moments["mean_vrad"], moments["x"], moments["y"], moments["z"])
    std_vrad_slices = scalar_slices(
        moments["std_vrad"], moments["x"], moments["y"], moments["z"])
    panels = [
        ("mean", mean_slices, r"mean $1+\delta$", "viridis", "density"),
        ("std", std_slices, r"std$(1+\delta)$", "cividis", "positive"),
        ("mean_vrad", mean_vrad_slices,
         r"mean $v_r$ [km s$^{-1}$]", "RdBu_r", "symmetric"),
        ("std_vrad", std_vrad_slices,
         r"std($v_r$) [km s$^{-1}$]", "cividis", "positive"),
    ]
    cols = [("xy", "x", "y"), ("xz", "x", "z"), ("yz", "y", "z")]
    fig, axes = plt.subplots(
        len(panels), len(cols), figsize=(8.6, 10.2),
        constrained_layout=True)

    for row, (label, images, quantity_label, cmap, scale) in enumerate(panels):
        values = np.concatenate([img.reshape(-1) for img in images.values()])
        if scale == "symmetric":
            vabs = np.nanpercentile(np.abs(values), 98)
            if not np.isfinite(vabs) or vabs <= 0.0:
                vabs = 50.0
            vmin, vmax = -vabs, vabs
        else:
            vmin, vmax = np.nanpercentile(values, [2, 98])
        if scale == "positive":
            vmin = 0.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            if scale == "symmetric":
                vmin, vmax = -50.0, 50.0
            else:
                vmin, vmax = (0.0, 1.0) if label == "std" else (0.5, 1.5)
        mappable = None
        for col, (name, xlabel, ylabel) in enumerate(cols):
            ax = axes[row, col]
            img = images[name]
            if name == "xy":
                xy_extent = extent(moments["x"], moments["y"], moments["dx"])
            elif name == "xz":
                xy_extent = extent(moments["x"], moments["z"], moments["dx"])
            else:
                xy_extent = extent(moments["y"], moments["z"], moments["dx"])
            mappable = ax.imshow(
                img.T, origin="lower", extent=xy_extent, cmap=cmap,
                vmin=vmin, vmax=vmax, interpolation="nearest",
                aspect="equal")
            ax.add_patch(Circle(
                (0.0, 0.0), radius, fill=False, color="white",
                lw=0.8, ls="--", alpha=0.8))
            ax.axhline(0.0, color="white", lw=0.4, alpha=0.6)
            ax.axvline(0.0, color="white", lw=0.4, alpha=0.6)
            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)
            ax.set_xlabel(fr"${xlabel}$ [$h^{{-1}}$ Mpc]")
            ax.set_ylabel(fr"${ylabel}$ [$h^{{-1}}$ Mpc]")
            ax.set_title(f"{quantity_label} {name.upper()}")
        cbar = fig.colorbar(
            mappable, ax=axes[row, :].ravel().tolist(),
            shrink=0.85, pad=0.01)
        cbar.set_label(quantity_label)

    n = len(moments["fields"])
    out_pdf = outdir / (
        f"cola_density_velocity_near_origin_all_sample_moments_n{n}.pdf")
    return save_pdf_png(fig, out_pdf)


def plot_mean_velocity_profile(moments, radius, outdir):
    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    bins = np.linspace(0.0, radius, 6)
    centres = 0.5 * (bins[:-1] + bins[1:])
    x, y, z = moments["x"], moments["y"], moments["z"]
    r = np.sqrt(
        x[:, None, None]**2 + y[None, :, None]**2 + z[None, None, :]**2)
    stack = moments["vrad_stack"]
    means = []
    stds = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (r >= lo) & (r < hi)
        values = stack[:, mask].reshape(-1)
        means.append(np.nan if values.size == 0 else np.mean(values))
        stds.append(np.nan if values.size < 2 else np.std(values, ddof=1))
    means = np.asarray(means)
    stds = np.asarray(stds)

    ax.fill_between(
        centres, means - stds, means + stds,
        color="0.75", alpha=0.45, lw=0.0,
        label="all-sample mean +/- std")
    ax.plot(
        centres, means, color="black", marker="s", lw=1.8,
        label=f"all {len(moments['fields'])} samples")
    ax.axhline(0.0, color="0.35", lw=1.0, ls="--")
    ax.set_xlabel(r"$r$ [$h^{-1}$ Mpc]")
    ax.set_ylabel(r"mean $v_r$ [km s$^{-1}$]")
    ax.set_xlim(0.0, radius)
    ax.legend(frameon=False, fontsize=8)

    out_pdf = outdir / (
        f"cola_radial_velocity_near_origin_all_sample_profile_n"
        f"{len(moments['fields'])}.pdf")
    return save_pdf_png(fig, out_pdf)


def write_all_sample_summary(moments, radius, outdir):
    x, y, z = moments["x"], moments["y"], moments["z"]
    r = np.sqrt(
        x[:, None, None]**2 + y[None, :, None]**2 + z[None, None, :]**2)
    rho_stack = moments["rho_stack"]
    vrad_stack = moments["vrad_stack"]
    mask10 = r <= radius
    mask5 = r <= 5.0
    values10 = rho_stack[:, mask10].reshape(-1)
    values5 = rho_stack[:, mask5].reshape(-1)
    vrad10 = vrad_stack[:, mask10].reshape(-1)
    vrad5 = vrad_stack[:, mask5].reshape(-1)
    rows = [{
        "n_samples": len(moments["fields"]),
        "first_field": min(moments["fields"]),
        "last_field": max(moments["fields"]),
        "n_voxels_per_sample_r10": int(np.sum(mask10)),
        "n_values_r10": int(values10.size),
        "mean_rho_r10": float(np.mean(values10)),
        "std_rho_r10": float(np.std(values10, ddof=1)),
        "median_rho_r10": float(np.median(values10)),
        "n_voxels_per_sample_r5": int(np.sum(mask5)),
        "n_values_r5": int(values5.size),
        "mean_rho_r5": float(np.mean(values5)),
        "std_rho_r5": float(np.std(values5, ddof=1)),
        "median_rho_r5": float(np.median(values5)),
        "mean_vrad_r10_km_s": float(np.mean(vrad10)),
        "std_vrad_r10_km_s": float(np.std(vrad10, ddof=1)),
        "median_vrad_r10_km_s": float(np.median(vrad10)),
        "mean_vrad_r5_km_s": float(np.mean(vrad5)),
        "std_vrad_r5_km_s": float(np.std(vrad5, ddof=1)),
        "median_vrad_r5_km_s": float(np.median(vrad5)),
    }]
    out = outdir / (
        f"cola_density_near_origin_all_sample_summary_n"
        f"{len(moments['fields'])}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fields", default=",".join(str(x) for x in DEFAULT_FIELDS),
        help="Comma-separated COLA field indices.")
    parser.add_argument(
        "--stats-fields", default="all",
        help=("Fields used for density mean/std. Use `all` for every "
              "mcmc_*.hdf5 file in source-root, or a comma-separated list."))
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    fields = parse_fields(args.fields)
    if args.stats_fields.strip().lower() == "all":
        stats_fields = discover_fields(args.source_root)
    else:
        stats_fields = parse_fields(args.stats_fields)

    items = [
        load_local_field(field, args.source_root, args.radius)
        for field in fields
    ]
    density_moments = compute_density_moments(
        stats_fields, args.source_root, args.radius)

    slice_pdf, slice_png = plot_slices(items, args.radius, args.output_dir)
    profile_pdf, profile_png = plot_radial_profiles(
        items, args.radius, args.output_dir,
        density_moments=density_moments)
    moments_pdf, moments_png = plot_density_moments(
        density_moments, args.radius, args.output_dir)
    velocity_profile_pdf, velocity_profile_png = plot_mean_velocity_profile(
        density_moments, args.radius, args.output_dir)
    summary = write_summary(items, args.radius, args.output_dir)
    all_sample_summary = write_all_sample_summary(
        density_moments, args.radius, args.output_dir)

    print(f"wrote {slice_pdf}")
    print(f"wrote {slice_png}")
    print(f"wrote {profile_pdf}")
    print(f"wrote {profile_png}")
    print(f"wrote {moments_pdf}")
    print(f"wrote {moments_png}")
    print(f"wrote {velocity_profile_pdf}")
    print(f"wrote {velocity_profile_png}")
    print(f"wrote {summary}")
    print(f"wrote {all_sample_summary}")


if __name__ == "__main__":
    main()
