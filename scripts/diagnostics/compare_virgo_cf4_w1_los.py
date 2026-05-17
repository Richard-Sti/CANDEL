#!/usr/bin/env python
"""Compare CF4 W1 LOS fields toward Virgo and random sky directions."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from h5py import File

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import candel  # noqa: E402
from candel.pvdata.volume_density import _density_unit_normalization  # noqa: E402
from candel.util import SPEED_OF_LIGHT, radec_to_galactic  # noqa: E402


RECONSTRUCTIONS = (
    ("COLA Manticore", "COLA_manticore_2MPP_MULTIBIN_N256_DES_V2", "tab:orange"),
    ("Manticore", "manticore_2MPP_MULTIBIN_N256_DES_V2", "tab:blue"),
    ("Carrick2015", "Carrick2015", "black"),
)


def none_if_string(value):
    if isinstance(value, str) and value.lower() == "none":
        return None
    return value


def _angular_sep_deg(ra1, dec1, ra2, dec2):
    ra1 = np.deg2rad(ra1)
    dec1 = np.deg2rad(dec1)
    ra2 = np.deg2rad(ra2)
    dec2 = np.deg2rad(dec2)
    cos_sep = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )
    return np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0)))


def load_cf4_w1_catalog(config):
    cfg = config["io"]["CF4_W1"]
    root = Path(cfg["root"])
    with File(root / "CF4_TFR.hdf5", "r") as f:
        grp = f["cf4"]
        zcmb = grp["Vcmb"][...] / SPEED_OF_LIGHT
        ra = grp["RA"][...] * 15.0
        dec = grp["DE"][...]
        mag = grp["w1"][...]
        mag_quality = grp["Qw"][...]
        eta = grp["lgWmxi"][...] - 2.5
        pgc = grp["pgc"][...]

    mask = eta > cfg["eta_min"]
    if cfg["best_mag_quality"]:
        mask &= mag_quality == 5
    else:
        mask &= mag > 5

    zcmb_min = none_if_string(cfg["zcmb_min"])
    zcmb_max = none_if_string(cfg["zcmb_max"])
    if zcmb_min is not None:
        mask &= zcmb > zcmb_min
    if zcmb_max is not None:
        mask &= zcmb < zcmb_max
    b_min = none_if_string(cfg["b_min"])
    if b_min is not None:
        b = radec_to_galactic(ra, dec)[1]
        mask &= np.abs(b) > b_min

    if cfg["remove_outliers"]:
        outliers = np.concatenate([
            np.genfromtxt(root / f"CF4_{band}_outliers.csv",
                          delimiter=",", names=True)
            for band in ("W1", "i")
        ])
        mask &= ~np.isin(pgc, outliers["PGC"])

    return {
        "mask": mask,
        "ra": ra[mask],
        "dec": dec[mask],
        "zcmb": zcmb[mask],
        "pgc": pgc[mask],
        "eta": eta[mask],
        "mag": mag[mask],
    }


def resolve_los_path(config, reconstruction):
    template = config["io"]["CF4_W1"]["los_file"]
    return Path(template.replace("<X>", reconstruction))


def load_los(path, mask, label):
    with File(path, "r") as f:
        ra = f["RA"][...][mask]
        dec = f["dec"][...][mask]
        r = f["r"][...].astype(np.float64)
        density = f["los_density"][:, mask, :].astype(np.float64)
        velocity = f["los_velocity"][:, mask, :].astype(np.float64)

    # Native Manticore SPH LOS density is stored in physical density units.
    # COLA Manticore and Carrick2015 are already stored as 1 + delta.
    if label == "Manticore":
        divisor, _, _ = _density_unit_normalization(path)
        density /= divisor

    return {
        "ra": ra,
        "dec": dec,
        "r": r,
        "density": density,
        "velocity": velocity,
    }


def validate_los_coordinates(los, catalog, label):
    if not np.allclose(los["ra"], catalog["ra"]):
        raise ValueError(f"RA mismatch after masking for {label}.")
    if not np.allclose(los["dec"], catalog["dec"]):
        raise ValueError(f"Dec mismatch after masking for {label}.")


def selected_from_index(cf4, idx, sep=None):
    selected = {
        "index": int(idx),
        "ra": float(cf4["ra"][idx]),
        "dec": float(cf4["dec"][idx]),
        "pgc": int(cf4["pgc"][idx]),
        "zcmb": float(cf4["zcmb"][idx]),
        "eta": float(cf4["eta"][idx]),
        "mag": float(cf4["mag"][idx]),
    }
    if sep is not None:
        selected["sep_deg"] = float(sep[idx])
    return selected


def select_lines_of_sight(args, catalog):
    sep = _angular_sep_deg(
        catalog["ra"], catalog["dec"], args.virgo_ra, args.virgo_dec)
    virgo_idx = int(np.argmin(sep))
    selected = [selected_from_index(catalog, virgo_idx, sep=sep)]
    selected[0]["label"] = "Virgo-nearest"

    available = np.setdiff1d(np.arange(len(catalog["ra"])), [virgo_idx])
    gen = np.random.default_rng(args.seed)
    random_indices = gen.choice(
        available, size=min(args.num_random_los, len(available)),
        replace=False)
    for i, idx in enumerate(random_indices, start=1):
        item = selected_from_index(catalog, int(idx))
        item["label"] = f"Random {i}"
        selected.append(item)
    return selected


def _plot_ensemble(ax, x, y, color, label=None):
    if y.shape[0] == 1:
        ax.plot(x, y[0], color=color, lw=2.0, label=label)
    else:
        q16, q50, q84 = np.percentile(y, [16, 50, 84], axis=0)
        ax.fill_between(x, q16, q84, color=color, alpha=0.20, lw=0)
        ax.plot(x, q50, color=color, lw=2.0, label=label)


def make_plot(args, selected, los_data):
    r = los_data["COLA Manticore"]["r"]
    ncols = len(selected)
    fig_width = max(12.0, 3.7 * ncols)
    fig, axes = plt.subplots(
        2, ncols, figsize=(fig_width, 7.2), sharex=True, sharey="row",
        constrained_layout=True)
    if ncols == 1:
        axes = axes[:, None]

    for col, item in enumerate(selected):
        for recon_label, _, colour in RECONSTRUCTIONS:
            data = los_data[recon_label]
            velocity = data["velocity"][:, item["index"], :]
            if recon_label == "Carrick2015":
                velocity = args.beta_carrick * velocity
            plot_label = recon_label if col == 0 else None
            _plot_ensemble(
                axes[0, col], r, velocity, colour, plot_label)

            overdensity = data["density"][:, item["index"], :] - 1.0
            _plot_ensemble(
                axes[1, col], r, overdensity, colour, plot_label)

        axes[0, col].axhline(0.0, color="0.5", lw=0.8)
        axes[1, col].axhline(0.0, color="0.5", lw=0.8)
        axes[1, col].set_yscale("symlog", linthresh=0.1)
        axes[1, col].set_xlabel(r"$r$ [$h^{-1}$ Mpc]")
        subtitle = (
            f"{item['label']}\nPGC {item['pgc']}\n"
            f"RA={item['ra']:.1f}, Dec={item['dec']:.1f}"
        )
        if "sep_deg" in item:
            subtitle += f"\nsep={item['sep_deg']:.2f} deg"
        axes[0, col].set_title(subtitle, fontsize=9)

    axes[0, 0].set_ylabel(r"$v_{\rm los}$ [km s$^{-1}$]")
    axes[1, 0].set_ylabel(r"$\delta$")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(
        "CF4 W1 LOS fields: Virgo-nearest and random sightlines",
        fontsize=13)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)


def print_selected_sightlines(selected, output):
    print("Selected CF4 W1 lines of sight:")
    for item in selected:
        suffix = ""
        if "sep_deg" in item:
            suffix = f", Virgo sep={item['sep_deg']:.6f} deg"
        print(
            f"{item['label']}: index={item['index']}, PGC {item['pgc']}, "
            f"RA={item['ra']:.6f} deg, Dec={item['dec']:.6f} deg, "
            f"zcmb={item['zcmb']:.6f}, eta={item['eta']:.3f}, "
            f"W1={item['mag']:.3f}{suffix}")
    print(f"Wrote {output.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/runs/configs/config.toml"),
        help="CANDEL config containing the CF4_W1 settings.",
    )
    parser.add_argument(
        "--virgo-ra",
        type=float,
        default=187.70593,
        help="Virgo centre RA in degrees. Default is M87.",
    )
    parser.add_argument(
        "--virgo-dec",
        type=float,
        default=12.39112,
        help="Virgo centre Dec in degrees. Default is M87.",
    )
    parser.add_argument(
        "--beta-carrick",
        type=float,
        default=0.43,
        help="Scale applied to Carrick2015 LOS velocities.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/diagnostics/virgo_cf4_w1_los_fields.png"),
    )
    parser.add_argument(
        "--num-random-los",
        type=int,
        default=5,
        help="Number of additional random CF4 W1 lines of sight to show.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for additional line-of-sight selection.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    config = candel.load_config(args.config)
    catalog = load_cf4_w1_catalog(config)

    selected = select_lines_of_sight(args, catalog)

    los_data = {}
    for label, reconstruction, _ in RECONSTRUCTIONS:
        path = resolve_los_path(config, reconstruction)
        los_data[label] = load_los(path, catalog["mask"], label)
        validate_los_coordinates(los_data[label], catalog, label)

    make_plot(args, selected, los_data)
    print_selected_sightlines(selected, args.output)


if __name__ == "__main__":
    main()
