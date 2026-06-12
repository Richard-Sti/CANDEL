#!/usr/bin/env python
"""Plot beta-free Student-t TRGBH0 H0 posterior diagnostics."""
from argparse import ArgumentParser
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import re
import shutil

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: F401,E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from trgbh0_plot_style import (  # noqa: E402
    FIGURE_DPI,
    OUTPUT_DIR,
    TRGBH0_COLOURS,
    TRGBH0_RESULTS,
    TRGBH0_TABLE_RESULTS,
    paper_style,
    save_pdf_png,
)


RESULTS = TRGBH0_RESULTS
TABLE = TRGBH0_TABLE_RESULTS
SINGLE_FIELDS = RESULTS / "single_fields"
OUTDIR = OUTPUT_DIR

FIXED_BETA_POSTERIOR = (
    TABLE
    / "EDD_TRGB_rhoSmoothR4_cz-student_t_MAS-PCS_sel-TRGB_magnitude_"
      "ManticoreLocalCOLA_main.hdf5"
)
FREE_BETA_POSTERIOR = (
    TABLE
    / "EDD_TRGB_rhoSmoothR4_cz-student_t_MAS-PCS_sel-TRGB_magnitude_"
      "ManticoreLocalCOLA_beta_free_main.hdf5"
)
SINGLE_PATTERN = (
    "EDD_TRGB_rhoSmoothR4_cz-student_t_MAS-PCS_sel-TRGB_magnitude_"
    "ManticoreLocalCOLA_beta_free_field*_single.hdf5"
)
FIELD_RE = re.compile(r"_field(\d+)_")

DEFAULT_OUTDIR = OUTDIR / "trgbh0_student_t_beta"
H0_LABEL = (
    r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"
)
LNZ_LABEL = r"$\ln \mathcal{Z}$"
SIGMA_V_LABEL = r"$\sigma_v~[\mathrm{km}\,\mathrm{s}^{-1}]$"
LNZ_CMAP = "Blues"
SIGMA_V_CMAP = "magma"
MARGINAL_NAME = "trgbh0_student_t_beta_marginal_h0.pdf"
STACKED_NAME = "trgbh0_student_t_beta_free_h0_posteriors_by_lnz.pdf"
H0_LNZ_NAME = "trgbh0_student_t_beta_free_h0_vs_lnz.pdf"
SUMMARY_NAME = "trgbh0_student_t_beta_free_single_fields.csv"
REFERENCE_BANDS = [
    ("Planck", 67.4, 0.5, TRGBH0_COLOURS[2]),
    ("SH0ES", 73.04, 1.04, TRGBH0_COLOURS[3]),
]
POSTERIOR_SPECS = [
    (r"Free $\beta$", FREE_BETA_POSTERIOR, TRGBH0_COLOURS[0], "-"),
    (r"$\beta = 1$", FIXED_BETA_POSTERIOR, TRGBH0_COLOURS[1], "--"),
]


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory for generated PDF and PNG figures.",
    )
    parser.add_argument(
        "--paper-figdir",
        type=Path,
        default=None,
        help="Optional paper figure directory to receive PDF copies.",
    )
    return parser.parse_args()


def field_index(path):
    match = FIELD_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse field index from `{path}`.")
    return int(match.group(1))


def finite_h0(path):
    with h5py.File(path, "r") as handle:
        h0 = np.asarray(handle["samples/H0"], dtype=float).reshape(-1)
    h0 = h0[np.isfinite(h0)]
    if h0.size == 0:
        raise ValueError(f"`{path}` has no finite H0 samples.")
    return h0


def finite_samples(handle, key, path):
    samples_key = f"samples/{key}"
    if samples_key not in handle:
        raise KeyError(f"`{path}` does not contain `{samples_key}`.")
    samples = np.asarray(handle[samples_key], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"`{path}` has no finite {key} samples.")
    return samples


def read_scalar(handle, key, path, default=None):
    if key not in handle:
        if default is not None:
            return default
        raise KeyError(f"`{path}` does not contain `{key}`.")
    value = float(handle[key][()])
    if not np.isfinite(value):
        if default is not None:
            return default
        raise ValueError(f"`{path}` has non-finite `{key}`: {value}.")
    return value


def read_single_field(path):
    with h5py.File(path, "r") as handle:
        h0 = finite_samples(handle, "H0", path)
        sigma_v = finite_samples(handle, "sigma_v", path)
        lnz = read_scalar(handle, "gof/lnZ_harmonic", path)
        err_lnz = read_scalar(
            handle, "gof/err_lnZ_harmonic", path, default=np.nan)
    return h0, sigma_v, lnz, err_lnz


def load_marginal_posteriors():
    rows = []
    for label, path, colour, linestyle in POSTERIOR_SPECS:
        if not path.exists():
            raise FileNotFoundError(f"Missing posterior: {path}")
        rows.append({
            "label": label,
            "samples": finite_h0(path),
            "colour": colour,
            "linestyle": linestyle,
        })
    return rows


def load_single_fields():
    paths = sorted(SINGLE_FIELDS.glob(SINGLE_PATTERN), key=field_index)
    if not paths:
        raise FileNotFoundError(
            f"No single-field posteriors matching `{SINGLE_PATTERN}`.")
    rows = []
    for path in paths:
        samples, sigma_v, lnz, err_lnz = read_single_field(path)
        rows.append({
            "field": field_index(path),
            "path": path,
            "samples": samples,
            "sigma_v_q50": float(np.percentile(sigma_v, 50.0)),
            "lnZ_harmonic": lnz,
            "err_lnZ_harmonic": err_lnz,
            **h0_summary(samples),
        })
    return rows


def h0_summary(samples):
    q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)),
        "q16": float(q16),
        "q50": float(q50),
        "q84": float(q84),
    }


def density_grid(sample_sets, qlo=0.2, qhi=99.8, num=800):
    all_samples = np.concatenate([np.asarray(samples).reshape(-1)
                                  for samples in sample_sets])
    xmin, xmax = np.percentile(all_samples, [qlo, qhi])
    pad = 0.08 * (xmax - xmin)
    return np.linspace(xmin - pad, xmax + pad, num)


def kde_on_grid(samples, x_grid, bw=1.15):
    kde = gaussian_kde(samples)
    kde.set_bandwidth(kde.factor * bw)
    return kde(x_grid)


def truncated_cmap(name, lower=0.2, upper=0.95):
    base = plt.get_cmap(name)
    colours = base(np.linspace(lower, upper, 256))
    return LinearSegmentedColormap.from_list(
        f"{name}_{lower:.2f}_{upper:.2f}", colours)


def plot_marginal(rows, out_pdf):
    x_grid = density_grid(
        [row["samples"] for row in rows], qlo=0.1, qhi=99.9)

    with paper_style():
        fig, ax = plt.subplots(figsize=(3.45, 2.45))
        for label, mean, sigma, ref_colour in REFERENCE_BANDS:
            ax.axvspan(
                mean - sigma,
                mean + sigma,
                color=ref_colour,
                alpha=0.75,
                lw=0,
                label=label,
                zorder=0,
            )
        for row in rows:
            density = kde_on_grid(row["samples"], x_grid)
            ax.plot(
                x_grid,
                density,
                color=row["colour"],
                ls=row["linestyle"],
                lw=1.35,
                label=row["label"],
                zorder=3,
            )
        ax.set_xlabel(H0_LABEL)
        ax.set_ylabel("Posterior density")
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.set_ylim(bottom=0)
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=6.4,
            handlelength=1.4,
        )
        fig.tight_layout()
        return save_pdf_png(fig, out_pdf)


def plot_stacked(rows, out_pdf):
    x_grid = density_grid([row["samples"] for row in rows])
    evidence = np.asarray([row["lnZ_harmonic"] for row in rows], dtype=float)
    medians = np.asarray([row["q50"] for row in rows], dtype=float)
    norm = Normalize(vmin=float(np.min(evidence)), vmax=float(np.max(evidence)))
    cmap = truncated_cmap(LNZ_CMAP)
    all_samples = np.concatenate([row["samples"] for row in rows])
    stacked_density = kde_on_grid(
        all_samples, x_grid)

    with paper_style(extra_rc={"legend.fontsize": 6.4}):
        fig, ax = plt.subplots(figsize=(3.45, 2.65))
        ax.hist(
            medians,
            bins=min(11, max(5, int(np.sqrt(len(medians))) + 2)),
            density=True,
            color="#b8b8b8",
            alpha=0.42,
            label=r"Field median $H_0$",
            zorder=0,
        )
        for row in rows:
            density = kde_on_grid(row["samples"], x_grid)
            colour = cmap(norm(row["lnZ_harmonic"]))
            ax.plot(
                x_grid,
                density,
                color=colour,
                lw=0.75,
                alpha=0.64,
                zorder=2,
            )

        ax.plot(
            x_grid,
            stacked_density,
            color="black",
            lw=1.3,
            label="Stacked posterior",
            zorder=4,
        )
        ax.axvline(
            np.median(medians),
            color="black",
            lw=0.9,
            ls=":",
            label=r"Median of field medians",
            zorder=5,
        )
        ax.set_xlabel(H0_LABEL)
        ax.set_ylabel("Posterior density")
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", frameon=False, fontsize=6.4)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.055)
        cbar.set_label(LNZ_LABEL)
        cbar.ax.tick_params(labelsize=6.4)
        fig.tight_layout()
        return save_pdf_png(fig, out_pdf)


def plot_h0_vs_lnz(rows, out_pdf):
    evidence = np.asarray([row["lnZ_harmonic"] for row in rows], dtype=float)
    h0 = np.asarray([row["q50"] for row in rows], dtype=float)
    h0_lo = np.asarray([row["q16"] for row in rows], dtype=float)
    h0_hi = np.asarray([row["q84"] for row in rows], dtype=float)
    xerr = np.asarray([row["err_lnZ_harmonic"] for row in rows], dtype=float)
    sigma_v = np.asarray([row["sigma_v_q50"] for row in rows], dtype=float)
    yerr = np.vstack([h0 - h0_lo, h0_hi - h0])
    finite_xerr = np.isfinite(xerr)

    norm = Normalize(vmin=float(np.min(sigma_v)), vmax=float(np.max(sigma_v)))
    cmap = truncated_cmap(SIGMA_V_CMAP)

    with paper_style():
        fig, ax = plt.subplots(figsize=(3.45, 2.65))
        for mask, maybe_xerr in (
            (finite_xerr, xerr[finite_xerr]),
            (~finite_xerr, None),
        ):
            if not np.any(mask):
                continue
            colours = cmap(norm(sigma_v[mask]))
            ax.errorbar(
                evidence[mask],
                h0[mask],
                xerr=maybe_xerr,
                yerr=yerr[:, mask],
                fmt="none",
                ecolor="0.55",
                elinewidth=0.45,
                capsize=1.2,
                alpha=0.55,
                zorder=1,
            )
            ax.scatter(
                evidence[mask],
                h0[mask],
                c=colours,
                s=17,
                edgecolor="black",
                linewidth=0.22,
                alpha=0.86,
                zorder=2,
            )
        ax.set_ylabel(H0_LABEL)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.055)
        ax.set_xlabel(LNZ_LABEL)
        cbar.set_label(SIGMA_V_LABEL)
        cbar.ax.tick_params(labelsize=7.0)
        fig.tight_layout()
        return save_pdf_png(fig, out_pdf)


def write_summary(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        handle.write(
            "field,lnZ_harmonic,err_lnZ_harmonic,sigma_v_q50,"
            "n_H0,H0_mean,H0_std,"
            "H0_q16,H0_q50,H0_q84,source\n")
        for row in rows:
            handle.write(
                f"{row['field']},{row['lnZ_harmonic']:.8g},"
                f"{row['err_lnZ_harmonic']:.8g},"
                f"{row['sigma_v_q50']:.8g},{row['samples'].size},"
                f"{row['mean']:.8g},{row['std']:.8g},"
                f"{row['q16']:.8g},{row['q50']:.8g},{row['q84']:.8g},"
                f"{row['path']}\n")
    return path


def copy_to_paper(paths, paper_figdir):
    if paper_figdir is None:
        return []
    paper_figdir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in paths:
        if path.suffix != ".pdf":
            continue
        destination = paper_figdir / path.name
        shutil.copyfile(path, destination)
        copied.append(destination)
    return copied


def main():
    args = parse_args()
    marginal_rows = load_marginal_posteriors()
    field_rows = load_single_fields()

    marginal_pdf, marginal_png = plot_marginal(
        marginal_rows, args.output_dir / MARGINAL_NAME)
    stacked_pdf, stacked_png = plot_stacked(
        field_rows, args.output_dir / STACKED_NAME)
    h0_lnz_pdf, h0_lnz_png = plot_h0_vs_lnz(
        field_rows, args.output_dir / H0_LNZ_NAME)
    summary_csv = write_summary(
        field_rows, args.output_dir / SUMMARY_NAME)
    copied = copy_to_paper([stacked_pdf, h0_lnz_pdf], args.paper_figdir)

    for path in (
        marginal_pdf, marginal_png, stacked_pdf, stacked_png,
        h0_lnz_pdf, h0_lnz_png, summary_csv,
    ):
        print(f"Wrote {path}")
    for path in copied:
        print(f"Copied {path}")
    print(f"Single-field realisations plotted: {len(field_rows)}")


if __name__ == "__main__":
    main()
