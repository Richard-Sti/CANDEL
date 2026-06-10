#!/usr/bin/env python
# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Plot SH0ES Cepheid magnitudes against period by galaxy for CH0.

The data transformation mirrors the Cepheid part of
``candel.pvdata.catalogues.load_SH0ES_separated`` without building covariance
factorisations that are not needed for this diagnostic plot.
"""
from argparse import ArgumentParser
from math import ceil
import os
from pathlib import Path
import tempfile

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "scripts" / "runs" / "configs" / "config_CH0.toml"
DEFAULT_OUTDIR = ROOT / "plots" / "paper_CH0"

N_CEPHEIDS = 3130
N_SN_HOSTS = 37
SPEED_OF_LIGHT = 299_792.458

ANCHOR_NAMES = ("NGC 4258", "LMC", "M31")
ANCHOR_CZ_CMB = (667.0, 327.0, -582.0)
MU_N4258_ANCHOR = 29.398
MU_LMC_ANCHOR = 18.477

PLOT_STYLE = ["default"]


def _heavy_imports():
    """Defer plotting/data imports so ``--help`` stays fast."""
    global fits, np, plt, PLOT_STYLE

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "candel-matplotlib-cache"),
    )

    import matplotlib
    matplotlib.use("Agg")

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits

    try:
        import scienceplots  # noqa: F401
        PLOT_STYLE = ["science", "no-latex"]
    except ImportError:
        PLOT_STYLE = ["default"]


def _deep_merge(base, override):
    """Recursively merge TOML dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_toml_with_bases(path):
    """Load a TOML config with the repo's simple ``base`` convention."""
    path = Path(path).resolve()
    with open(path, "rb") as f:
        config = tomllib.load(f)

    base_paths = config.pop("base", None)
    if base_paths is None:
        return config
    if isinstance(base_paths, str):
        base_paths = [base_paths]

    merged = {}
    for base in base_paths:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = _deep_merge(merged, _load_toml_with_bases(base_path))
    return _deep_merge(merged, config)


def _convert_none_strings(value):
    if isinstance(value, dict):
        return {k: _convert_none_strings(v) for k, v in value.items()}
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return value


def _load_plot_config(config_path):
    """Load just enough config information for this data diagnostic."""
    config = {}
    local_config = ROOT / "local_config.toml"
    if local_config.exists():
        with open(local_config, "rb") as f:
            config = _deep_merge(config, tomllib.load(f))
    config = _deep_merge(config, _load_toml_with_bases(config_path))
    return _convert_none_strings(config)


def _nested(config, keys, default=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _resolve_data_root(config, data_root):
    if data_root is not None:
        return Path(data_root).expanduser().resolve()

    root = _nested(config, ("io", "SH0ES", "root"), "data/SH0ES")
    root = Path(root).expanduser()
    if root.is_absolute():
        return root

    root_data = Path(config.get("root_data", ROOT)).expanduser()
    if not root_data.is_absolute():
        root_data = (ROOT / root_data).resolve()
    return (root_data / root).resolve()


def _format_cut(cz_cmb_max):
    if cz_cmb_max is None:
        return "all SH0ES hosts"
    return rf"$cz_{{\rm CMB}} < {cz_cmb_max:g}\,\mathrm{{km}}\,\mathrm{{s}}^{{-1}}$"


def _resolve_drop_index(drop_observation, num_active_hosts):
    if drop_observation is None:
        return None
    if isinstance(drop_observation, str):
        value = drop_observation.strip()
        if value == "" or value.lower() == "none":
            return None
        if value.isdecimal():
            drop_observation = int(value)
        else:
            raise TypeError(
                "`drop_observation` must be 'none' or an integer active "
                "host index."
            )
    if isinstance(drop_observation, bool) or not isinstance(
        drop_observation, int
    ):
        raise TypeError(
            "`drop_observation` must be 'none' or an integer active host "
            "index."
        )
    if not (0 <= drop_observation < num_active_hosts):
        raise ValueError(
            "`drop_observation` index out of range: "
            f"{drop_observation}. Active host indices run from 0 to "
            f"{num_active_hosts - 1}."
        )
    return drop_observation


def load_sh0es_cepheids(data_root, cz_cmb_max=None, drop_observation=None):
    """Load SH0ES Cepheids with CH0 host/anchor assignment."""
    data_root = Path(data_root)
    y_path = data_root / "ally_shoes_ceph_topantheonwt6.0_112221.fits"
    l_path = data_root / "alll_shoes_ceph_topantheonwt6.0_112221.fits"
    redshift_path = data_root / "processed" / "Cepheid_anchors_redshifts.npy"

    y = np.asarray(fits.open(y_path, memmap=False)[0].data, dtype=float)
    matrix = np.asarray(
        fits.open(l_path, memmap=False)[0].data.T, dtype=float
    )

    logp_centered = matrix[:N_CEPHEIDS, -6].copy()
    oxygen = matrix[:N_CEPHEIDS, -4].copy()
    magnitude = y[:N_CEPHEIDS].copy()

    # The public SH0ES matrix has the fiducial PL slope removed. Undo that
    # exactly as in the CH0 inference loader.
    magnitude += -3.285 * logp_centered

    # Host columns are the 37 SN hosts plus NGC 4258, LMC, and M31. Column 38
    # is the Cepheid absolute-magnitude calibration and is intentionally
    # skipped.
    host_matrix = np.hstack(
        [matrix[:, :N_SN_HOSTS], matrix[:, [37, 39, 40]]]
    )[:N_CEPHEIDS]

    # Undo the anchor offsets used in the SH0ES equation matrix.
    magnitude[2150:2593] += MU_N4258_ANCHOR
    magnitude[2648:] += MU_LMC_ANCHOR

    redshifts = np.load(redshift_path, allow_pickle=True)
    host_names = [str(name) for name in redshifts["Galaxy"]]
    all_names = np.asarray(host_names + list(ANCHOR_NAMES), dtype=object)
    cz_cmb = np.concatenate(
        [redshifts["zCMB"] * SPEED_OF_LIGHT, np.asarray(ANCHOR_CZ_CMB)]
    )

    active_columns = np.ones(len(all_names), dtype=bool)
    if cz_cmb_max is not None:
        if cz_cmb_max < 1000:
            raise ValueError(
                "`cz_cmb_max` must be larger than 1000 km/s so the "
                "geometric anchors are not accidentally removed."
            )
        active_columns = cz_cmb < cz_cmb_max

    assigned = host_matrix == 1
    if not np.all(np.sum(assigned, axis=1) == 1):
        raise ValueError("Each Cepheid row must map to exactly one host.")

    host_index_all = np.argmax(assigned, axis=1)
    keep_cepheid = active_columns[host_index_all]
    host_matrix = host_matrix[keep_cepheid][:, active_columns]
    logp_centered = logp_centered[keep_cepheid]
    oxygen = oxygen[keep_cepheid]
    magnitude = magnitude[keep_cepheid]

    names = all_names[active_columns]
    active_host_columns = active_columns[:N_SN_HOSTS]
    num_active_hosts = int(np.sum(active_host_columns))

    drop_index = _resolve_drop_index(drop_observation, num_active_hosts)
    dropped_name = None
    if drop_index is not None:
        dropped_name = str(names[drop_index])
        keep_rows = host_matrix[:, drop_index] == 0
        keep_columns = np.ones(host_matrix.shape[1], dtype=bool)
        keep_columns[drop_index] = False

        host_matrix = host_matrix[keep_rows][:, keep_columns]
        logp_centered = logp_centered[keep_rows]
        oxygen = oxygen[keep_rows]
        magnitude = magnitude[keep_rows]
        names = names[keep_columns]

    galaxy_index = np.argmax(host_matrix == 1, axis=1)
    counts = np.bincount(galaxy_index, minlength=len(names))

    return {
        "names": names,
        "galaxy_index": galaxy_index,
        "counts": counts,
        "logp_centered": logp_centered,
        "oxygen": oxygen,
        "magnitude": magnitude,
        "cz_cmb_max": cz_cmb_max,
        "dropped_name": dropped_name,
    }


def save_cepheid_table(data, outdir):
    """Save the plotted Cepheid rows as a compact CSV table."""
    path = outdir / "sh0es_cepheids_by_galaxy.csv"
    names = data["names"]
    galaxy = names[data["galaxy_index"]]
    table = np.column_stack(
        [
            galaxy,
            data["logp_centered"] + 1.0,
            data["logp_centered"],
            data["magnitude"],
            data["oxygen"],
        ]
    )
    header = "galaxy,log10_period,logp_centered,magnitude,oxygen"
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%s")
    return path


def plot_by_galaxy(data, outdir, formats, centered_logp=False, ncols=6):
    names = data["names"]
    counts = data["counts"]
    galaxy_index = data["galaxy_index"]
    magnitude = data["magnitude"]
    if centered_logp:
        logp = data["logp_centered"]
        xlabel = r"$\log_{10}(P/\mathrm{day}) - 1$"
        stem = "sh0es_cepheids_mag_centered_logp_by_galaxy"
    else:
        logp = data["logp_centered"] + 1.0
        xlabel = r"$\log_{10}(P/\mathrm{day})$"
        stem = "sh0es_cepheids_mag_logp_by_galaxy"

    n_panels = len(names)
    nrows = ceil(n_panels / ncols)
    figsize = (2.05 * ncols, 1.65 * nrows)
    color_host = "#1e42b9"
    color_anchor = "#87193d"
    anchor_start = n_panels - len(ANCHOR_NAMES)

    finite = np.isfinite(logp) & np.isfinite(magnitude)
    xpad = 0.04 * np.ptp(logp[finite])
    ypad = 0.04 * np.ptp(magnitude[finite])
    xlim = (np.min(logp[finite]) - xpad, np.max(logp[finite]) + xpad)
    ylim = (np.max(magnitude[finite]) + ypad, np.min(magnitude[finite]) - ypad)

    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes).ravel()

        for i, ax in enumerate(axes):
            if i >= n_panels:
                ax.axis("off")
                continue

            mask = galaxy_index == i
            color = color_anchor if i >= anchor_start else color_host
            ax.scatter(
                logp[mask],
                magnitude[mask],
                s=9,
                alpha=0.7,
                color=color,
                edgecolors="none",
            )
            ax.set_title(f"{names[i]} ({counts[i]})", fontsize=7, pad=2)
            ax.grid(alpha=0.25, linewidth=0.4)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

        for ax in axes[(nrows - 1) * ncols:]:
            if ax.has_data():
                ax.set_xlabel(xlabel)
        for ax in axes[::ncols]:
            if ax.has_data():
                ax.set_ylabel(r"$m_W^H~[\mathrm{mag}]$")

        fig.suptitle(
            "SH0ES Cepheids by galaxy "
            f"({_format_cut(data['cz_cmb_max'])})",
            y=0.995,
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.985))

        paths = []
        for fmt in formats:
            path = outdir / f"{stem}.{fmt}"
            fig.savefig(path, dpi=350, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

    return paths


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help="CH0 config used to read the SH0ES root and redshift cut.",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="Override the SH0ES data root from the config.",
    )
    parser.add_argument(
        "--cz-cmb-max", type=float, default=None,
        help="Override the config Cepheid-host redshift cut in km/s.",
    )
    parser.add_argument(
        "--all-hosts", action="store_true",
        help="Disable the Cepheid-host redshift cut.",
    )
    parser.add_argument(
        "--drop-observation", default=None,
        help="Drop one active host by zero-based index, overriding config.",
    )
    parser.add_argument(
        "--centered-logp", action="store_true",
        help="Plot the CH0 equation-matrix covariate log10(P/day)-1.",
    )
    parser.add_argument(
        "--outdir", default=DEFAULT_OUTDIR,
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--formats", nargs="+", default=("pdf", "png"),
        choices=("pdf", "png", "jpg", "svg"),
        help="Output figure formats.",
    )
    parser.add_argument(
        "--ncols", type=int, default=6,
        help="Number of columns in the faceted plot.",
    )
    parser.add_argument(
        "--save-table", action="store_true",
        help="Also save the plotted Cepheid rows as CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _heavy_imports()

    config = _load_plot_config(args.config)
    data_root = _resolve_data_root(config, args.data_root)
    if args.all_hosts:
        cz_cmb_max = None
    elif args.cz_cmb_max is not None:
        cz_cmb_max = args.cz_cmb_max
    else:
        cz_cmb_max = _nested(
            config, ("io", "SH0ES", "cepheid_host_cz_cmb_max"), 3300.0
        )

    drop_observation = (
        args.drop_observation
        if args.drop_observation is not None
        else _nested(config, ("io", "SH0ES", "drop_observation"), None)
    )

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_sh0es_cepheids(
        data_root, cz_cmb_max=cz_cmb_max,
        drop_observation=drop_observation,
    )
    figure_paths = plot_by_galaxy(
        data, outdir, args.formats,
        centered_logp=args.centered_logp, ncols=args.ncols,
    )
    table_path = save_cepheid_table(data, outdir) if args.save_table else None

    print(
        f"Loaded {len(data['magnitude'])} Cepheids in "
        f"{len(data['names'])} galaxies/anchors from {data_root}."
    )
    if data["dropped_name"] is not None:
        print(f"Dropped active host: {data['dropped_name']}.")
    print("Saved figures:")
    for path in figure_paths:
        print(f"  {path}")
    if table_path is not None:
        print(f"Saved table:\n  {table_path}")


if __name__ == "__main__":
    main()
