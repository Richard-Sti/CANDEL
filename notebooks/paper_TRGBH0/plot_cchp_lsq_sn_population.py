#!/usr/bin/env python
"""Compare CCHP TRGB-calibrator SNe with the LSQ CSP sample."""
import argparse
import csv
import math
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/candel_mplconfig")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from edd_trgb_plot_data import PAPER_RC
from trgbh0_plot_style import TRGBH0_COLOURS


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
OUTDIR = ROOT / "notebooks" / "paper_TRGBH0" / "output"
OUTNAME = "cchp_lsq_sn_population_comparison"
SPEED_OF_LIGHT = 299_792.458


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--paper-figdir", type=Path, default=None,
        help="Optional directory to copy the generated figures into.")
    parser.add_argument(
        "--h", type=float, default=0.7,
        help="Dimensionless Hubble constant used for LSQ redshift distances.")
    return parser.parse_args()


def _float_or_nan(value):
    try:
        if value in ("", '""', "-1", "nan", "NaN"):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _read_tsv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _read_space_table(path):
    with path.open() as handle:
        header = handle.readline().split()
        rows = []
        for line in handle:
            values = line.split()
            if not values:
                continue
            rows.append(dict(zip(header, values)))
    return rows


def _coordinate_names(root):
    names = set()
    for fname in ("cspallcal_sncoords.csv", "csp_sncoords.csv",
                  "missing_coords_simbad.csv"):
        path = root / fname
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if math.isfinite(_float_or_nan(row["snra"])) and math.isfinite(
                        _float_or_nan(row["sndec"])):
                    names.add(row["sn"])
    return names


def _deduplicate_first(rows):
    seen = set()
    out = []
    for row in rows:
        if row["sn"] in seen:
            continue
        seen.add(row["sn"])
        out.append(row)
    return out


def _normalise_sn_name(name):
    return name[2:] if name.startswith("SN") else name


def _numeric_rows(rows):
    cols = ("zcmb", "st", "Mmax", "BV", "phys")
    out = []
    for row in rows:
        item = {key: _float_or_nan(row[key]) for key in cols}
        item["sn"] = row["sn"]
        item["name_norm"] = _normalise_sn_name(row["sn"])
        item["sample"] = row["sample"]
        item["host"] = row["host"]
        item["czcmb"] = item["zcmb"] * SPEED_OF_LIGHT
        if all(math.isfinite(item[key]) for key in ("zcmb", "st", "Mmax", "BV")):
            out.append(item)
    return out


def load_samples():
    csp_root = ROOT / "data" / "CSP"
    csp_rows = _numeric_rows(_deduplicate_first(
        _read_space_table(csp_root / "B_all_noj21.csv")))
    coord_names = _coordinate_names(csp_root)

    lsq = [
        row for row in csp_rows
        if row["sn"].startswith("LSQ")
        and row["phys"] == 0
        and row["sn"] in coord_names
    ]

    cchp_rows = _read_tsv(ROOT / "data" / "CCHP" / "cchp_processed_data.tsv")
    cchp_by_name = {row["SN"]: row for row in cchp_rows}
    csp_by_name = {row["name_norm"]: row for row in csp_rows}
    calibrators = []
    for name, cchp in cchp_by_name.items():
        csp = csp_by_name.get(name)
        if csp is None:
            continue
        mu_trgb = _float_or_nan(cchp["mu_TRGB_CCHP"])
        if not math.isfinite(mu_trgb):
            continue
        row = dict(csp)
        row["mu_trgb"] = mu_trgb
        row["galaxy"] = cchp["Galaxy"]
        calibrators.append(row)

    if not calibrators or not lsq:
        raise RuntimeError("Could not build both calibrator and LSQ samples.")

    return calibrators, lsq


def _as_array(rows, key):
    return np.asarray([row[key] for row in rows], dtype=float)


def _standardised_residuals(calibrators, lsq, h):
    from candel.cosmo.cosmography import Redshift2Distmod

    redshift2distmod = Redshift2Distmod(Om0=0.3)
    z_lsq = _as_array(lsq, "zcmb")
    mu_lsq = np.asarray(redshift2distmod(z_lsq, h=h), dtype=float)

    y_lsq = _as_array(lsq, "Mmax") - mu_lsq
    x_lsq = np.column_stack([
        np.ones(len(lsq)),
        -(_as_array(lsq, "st") - 1.0),
        _as_array(lsq, "BV"),
    ])
    coeff, *_ = np.linalg.lstsq(x_lsq, y_lsq, rcond=None)

    resid_lsq = y_lsq - x_lsq @ coeff

    y_cal = _as_array(calibrators, "Mmax") - _as_array(calibrators, "mu_trgb")
    x_cal = np.column_stack([
        np.ones(len(calibrators)),
        -(_as_array(calibrators, "st") - 1.0),
        _as_array(calibrators, "BV"),
    ])
    resid_cal = y_cal - x_cal @ coeff

    return resid_cal, resid_lsq, coeff


def _plot_hist(ax, calibrators, lsq, key, xlabel, bins, show_legend=False):
    cal = _as_array(calibrators, key)
    flow = _as_array(lsq, key)
    ax.hist(
        flow, bins=bins, histtype="step", density=True, linewidth=1.4,
        color=TRGBH0_COLOURS[1], label=rf"LSQ ($N={len(flow)}$)")
    ax.hist(
        cal, bins=bins, histtype="stepfilled", density=True, alpha=0.42,
        color=TRGBH0_COLOURS[0], label=rf"TRGB calibrators ($N={len(cal)}$)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\mathrm{Density}$")
    ks = ks_2samp(cal, flow)
    ax.text(
        0.97, 0.94, rf"$p_{{\rm KS}}={ks.pvalue:.2g}$",
        transform=ax.transAxes, ha="right", va="top")
    if show_legend:
        ax.legend(frameon=False, loc="upper left")


def make_figure(calibrators, lsq, h=0.7):
    resid_cal, resid_lsq, coeff = _standardised_residuals(calibrators, lsq, h)
    alpha, beta = coeff[1], coeff[2]

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            2, 2, figsize=(6.7, 4.9), constrained_layout=True)

        _plot_hist(
            axes[0, 0], calibrators, lsq, "czcmb",
            r"$cz_{\rm CMB}\ [\mathrm{km}\,\mathrm{s}^{-1}]$",
            np.linspace(0.0, 40_000.0, 33),
            show_legend=True)

        _plot_hist(
            axes[0, 1], calibrators, lsq, "st",
            r"$s$", np.linspace(0.45, 1.35, 31))
        _plot_hist(
            axes[1, 0], calibrators, lsq, "BV",
            r"$B-V$", np.linspace(-0.16, 0.52, 31))

        bins = np.linspace(-0.75, 0.75, 31)
        axes[1, 1].hist(
            resid_lsq, bins=bins, histtype="step", density=True,
            linewidth=1.4, color=TRGBH0_COLOURS[1],
            label=rf"LSQ ($N={len(resid_lsq)}$)")
        axes[1, 1].hist(
            resid_cal, bins=bins, histtype="stepfilled", density=True,
            alpha=0.42, color=TRGBH0_COLOURS[0],
            label=rf"TRGB calibrators ($N={len(resid_cal)}$)")
        ks = ks_2samp(resid_cal, resid_lsq)
        axes[1, 1].text(
            0.97, 0.94, rf"$p_{{\rm KS}}={ks.pvalue:.2g}$",
            transform=axes[1, 1].transAxes, ha="right", va="top")
        axes[1, 1].set_xlabel(
            r"$m_B+\alpha(s-1)-\beta(B-V)-\mu-\mathcal{M}_{\rm LSQ}$")
        axes[1, 1].set_ylabel(r"$\mathrm{Density}$")
        axes[1, 1].text(
            0.03, 0.06, rf"$\alpha={alpha:.2f},\ \beta={beta:.2f}$",
            transform=axes[1, 1].transAxes, ha="left", va="bottom")

    return fig, coeff


def save_figure(fig, paper_figdir=None):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".pdf", ".png"):
        out = OUTDIR / f"{OUTNAME}{suffix}"
        fig.savefig(out, dpi=250 if suffix == ".png" else None)
        outputs.append(out)
        if paper_figdir is not None:
            paper_figdir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out, paper_figdir / out.name)
    return outputs


def main():
    args = parse_args()
    calibrators, lsq = load_samples()
    fig, coeff = make_figure(calibrators, lsq, h=args.h)
    outputs = save_figure(fig, args.paper_figdir)
    plt.close(fig)
    alpha, beta = coeff[1], coeff[2]
    print(f"TRGB calibrator SNe with CSP observables: {len(calibrators)}")
    print(f"Unique TRGB calibrator hosts: {len({row['galaxy'] for row in calibrators})}")
    print(f"LSQ SNe: {len(lsq)}")
    print(f"LSQ-fitted alpha={alpha:.4f}, beta={beta:.4f}")
    for out in outputs:
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
