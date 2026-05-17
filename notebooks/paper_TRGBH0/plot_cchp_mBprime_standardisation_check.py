#!/usr/bin/env python
"""Compare CCHP m_Bprime values with fiducial CSP standardisation."""
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

from edd_trgb_plot_data import PAPER_RC
from trgbh0_plot_style import TRGBH0_COLOURS


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
OUTDIR = ROOT / "notebooks" / "paper_TRGBH0" / "output"
OUTNAME = "cchp_mBprime_standardisation_check"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument(
        "--paper-figdir", type=Path, default=None,
        help="Optional directory to copy the generated figures into.")
    return parser.parse_args()


def _float_or_nan(value):
    try:
        if value in ("", '""', "-1", "nan", "NaN"):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _read_cchp():
    path = ROOT / "data" / "CCHP" / "cchp_processed_data.tsv"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _read_csp_first_rows():
    path = ROOT / "data" / "CSP" / "B_all_noj21.csv"
    with path.open() as handle:
        header = handle.readline().split()
        rows = []
        for line in handle:
            values = line.split()
            if values:
                rows.append(dict(zip(header, values)))

    out = {}
    for row in rows:
        name = row["sn"][2:] if row["sn"].startswith("SN") else row["sn"]
        out.setdefault(name, row)
    return out


def load_comparison(alpha, beta):
    csp_by_name = _read_csp_first_rows()
    rows = []
    for row in _read_cchp():
        csp = csp_by_name.get(row["SN"])
        if csp is None:
            continue

        m_bprime = _float_or_nan(row["m_Bprime_CSP"])
        sigma_bprime = _float_or_nan(row["sigma_Bprime_CSP"])
        mmax = _float_or_nan(csp["Mmax"])
        st = _float_or_nan(csp["st"])
        bv = _float_or_nan(csp["BV"])
        if not all(math.isfinite(x) for x in (
                m_bprime, sigma_bprime, mmax, st, bv)):
            continue

        m_std = mmax + alpha * (st - 1.0) - beta * bv
        rows.append({
            "SN": row["SN"],
            "Galaxy": row["Galaxy"],
            "m_Bprime_CSP": m_bprime,
            "sigma_Bprime_CSP": sigma_bprime,
            "Mmax_CSP": mmax,
            "st_CSP": st,
            "BV_CSP": bv,
            "m_std_fid": m_std,
            "delta_m_Bprime_minus_fid": m_bprime - m_std,
        })
    return rows


def write_table(rows):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{OUTNAME}.tsv"
    names = [
        "SN", "Galaxy", "m_Bprime_CSP", "sigma_Bprime_CSP", "Mmax_CSP",
        "st_CSP", "BV_CSP", "m_std_fid", "delta_m_Bprime_minus_fid",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def make_figure(rows, alpha, beta):
    m_bprime = np.asarray([row["m_Bprime_CSP"] for row in rows])
    m_std = np.asarray([row["m_std_fid"] for row in rows])
    delta = np.asarray([row["delta_m_Bprime_minus_fid"] for row in rows])
    bv = np.asarray([row["BV_CSP"] for row in rows])
    st = np.asarray([row["st_CSP"] for row in rows])

    med = np.median(delta)
    scatter = 1.4826 * np.median(np.abs(delta - med))
    rms = np.sqrt(np.mean(delta**2))

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            1, 3, figsize=(7.1, 2.45), constrained_layout=True)

        ax = axes[0]
        ax.errorbar(
            m_std, m_bprime,
            yerr=[row["sigma_Bprime_CSP"] for row in rows],
            fmt="o", ms=4, color=TRGBH0_COLOURS[0], alpha=0.85)
        lo = min(np.min(m_std), np.min(m_bprime)) - 0.15
        hi = max(np.max(m_std), np.max(m_bprime)) + 0.15
        ax.plot([lo, hi], [lo, hi], color="0.35", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"$m_B+\alpha(s-1)-\beta(B-V)$")
        ax.set_ylabel(r"$m_{B'}^{\rm CCHP}$")

        ax = axes[1]
        ax.hist(delta, bins=np.linspace(-0.25, 0.25, 21),
                color=TRGBH0_COLOURS[1], alpha=0.75)
        ax.axvline(0.0, color="0.35", lw=1)
        ax.axvline(med, color=TRGBH0_COLOURS[0], lw=1.5)
        ax.set_xlabel(r"$m_{B'}^{\rm CCHP}-m_{\rm std}^{\rm fid}$")
        ax.set_ylabel(r"$\mathrm{Counts}$")
        ax.text(
            0.05, 0.95,
            rf"$\Delta_{{50}}={med:.3f}$" "\n"
            rf"$\sigma_{{\rm MAD}}={scatter:.3f}$" "\n"
            rf"$\mathrm{{RMS}}={rms:.3f}$",
            transform=ax.transAxes, ha="left", va="top")

        ax = axes[2]
        sc = ax.scatter(
            bv, delta, c=st, cmap="viridis", s=22, edgecolor="none")
        ax.axhline(0.0, color="0.35", lw=1)
        ax.set_xlabel(r"$B-V$")
        ax.set_ylabel(r"$m_{B'}^{\rm CCHP}-m_{\rm std}^{\rm fid}$")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"$s$")

        fig.suptitle(
            rf"Fiducial standardisation: $\alpha={alpha:.2f}$, "
            rf"$\beta={beta:.2f}$",
            y=1.06)
    return fig, med, scatter, rms


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
    rows = load_comparison(args.alpha, args.beta)
    table = write_table(rows)
    fig, med, scatter, rms = make_figure(rows, args.alpha, args.beta)
    outputs = save_figure(fig, args.paper_figdir)
    plt.close(fig)

    print(f"Compared {len(rows)} CCHP SNe with finite m_Bprime and CSP m,s,BV.")
    print(f"median delta = {med:.4f} mag")
    print(f"MAD scatter = {scatter:.4f} mag")
    print(f"RMS delta = {rms:.4f} mag")
    print(f"Wrote {table}")
    for out in outputs:
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
