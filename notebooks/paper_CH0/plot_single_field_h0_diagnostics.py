#!/usr/bin/env python
"""Plot CH0 single-field H0 summaries and evidence diagnostics."""

from argparse import ArgumentParser
import csv
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: F401
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from scipy.stats import gaussian_kde, pearsonr, spearmanr  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "CH0_paper" / "single_fields"
DEFAULT_OUTDIR = RESULTS / "plots"
DEFAULT_PATTERN = (
    "CH0_sel-SN_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_"
    "field*_single.hdf5"
)
DEFAULT_OUTPUT_PREFIX = "ch0_single_field"
PRESETS = {
    "pcs-redshift": {
        "pattern": (
            "CH0_MAS-PCS_sel-redshift_ManticoreLocalCOLA_"
            "field*_single.hdf5"
        ),
        "output_dir": DEFAULT_OUTDIR / "pcs_redshift_selection",
        "output_prefix": "ch0_pcs_redshift_selection",
        "plot_title": "CH0 PCS redshift-selection fields",
    },
}
FIELD_RE = re.compile(r"_field(\d+)_")
FIGURE_DPI = 500
COLOURS = ["#87193d", "#1e42b9", "#d42a29", "#05dd6b", "#ee35d5"]
FIELD_COLOURS = ["#ef476f", "#473198", "#a8c256", "#5adbff", "#fe9000"]
H0_LABEL = (
    r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"
)
EVIDENCE_PANEL_SPECS = (
    {
        "key": "lnZ_laplace",
        "title": "Laplace",
        "x_label": r"Laplace $\ln Z$",
        "xerr_key": "err_lnZ_laplace",
        "colour": COLOURS[0],
    },
    {
        "key": "lnZ_harmonic",
        "title": "Harmonic",
        "x_label": r"Harmonic $\ln Z$",
        "xerr_key": "err_lnZ_harmonic",
        "colour": COLOURS[1],
    },
    {
        "key": "lnZ_bic",
        "title": "BIC",
        "x_label": r"BIC $\ln Z \simeq -\mathrm{BIC}/2$",
        "xerr_key": None,
        "colour": "#168039",
    },
)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default=None,
        help="Named plotting preset for a maintained single-field subset.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="Directory containing the single-field CH0 HDF5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PDF, PNG, and CSV outputs.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern for the single-field HDF5 files.",
    )
    parser.add_argument(
        "--field-stat",
        choices=("median", "mean"),
        default="median",
        help="Posterior summary used as the per-field H0 histogram value.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Filename prefix for PDF, PNG, and CSV outputs.",
    )
    parser.add_argument(
        "--plot-title",
        default=None,
        help="Title used in the generated plots.",
    )
    args = parser.parse_args()

    preset = PRESETS.get(args.preset, {})
    if args.output_dir is None:
        args.output_dir = preset.get("output_dir", DEFAULT_OUTDIR)
    if args.pattern == DEFAULT_PATTERN and "pattern" in preset:
        args.pattern = preset["pattern"]
    if args.output_prefix is None:
        args.output_prefix = preset.get("output_prefix", DEFAULT_OUTPUT_PREFIX)
    if args.plot_title is None:
        args.plot_title = preset.get(
            "plot_title", "CH0 single Manticore fields")
    return args


def field_index(path):
    match = FIELD_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse field index from `{path}`.")
    return int(match.group(1))


def read_scalar(handle, name, path):
    if name not in handle:
        raise KeyError(f"`{path}` does not contain `{name}`.")
    value = float(handle[name][()])
    if not np.isfinite(value):
        raise ValueError(f"`{path}` has non-finite `{name}`: {value}.")
    return value


def finite_h0(handle, path):
    samples = np.asarray(handle["samples/H0"], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"`{path}` has no finite H0 samples.")
    return samples


def summarise_h0(samples):
    q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
    return {
        "n_H0": int(samples.size),
        "H0_mean": float(np.mean(samples)),
        "H0_std": float(np.std(samples, ddof=1)),
        "H0_q16": float(q16),
        "H0_q50": float(q50),
        "H0_q84": float(q84),
    }


def load_field_rows(results_dir, pattern):
    paths = sorted(results_dir.glob(pattern), key=field_index)
    if not paths:
        raise FileNotFoundError(
            f"No HDF5 files matching `{results_dir / pattern}`.")

    rows = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            samples = finite_h0(handle, path)
            bic = read_scalar(handle, "gof/BIC", path)
            rows.append({
                "kind": "field",
                "field": field_index(path),
                "source": str(path),
                "samples": samples,
                **summarise_h0(samples),
                "lnZ_laplace": read_scalar(handle, "gof/lnZ_laplace", path),
                "err_lnZ_laplace": read_scalar(
                    handle, "gof/err_lnZ_laplace", path),
                "lnZ_harmonic": read_scalar(
                    handle, "gof/lnZ_harmonic", path),
                "err_lnZ_harmonic": read_scalar(
                    handle, "gof/err_lnZ_harmonic", path),
                "BIC": bic,
                "lnZ_bic": -0.5 * bic,
            })
    return rows


def stacked_row(field_rows):
    stacked = np.concatenate([row["samples"] for row in field_rows])
    return {
        "kind": "stacked",
        "field": "",
        "source": "",
        "samples": stacked,
        **summarise_h0(stacked),
    }


def write_h0_summary(field_rows, stack, path):
    fieldnames = [
        "kind", "field", "n_H0", "H0_mean", "H0_std",
        "H0_q16", "H0_q50", "H0_q84", "source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in [*field_rows, stack]:
            writer.writerow({name: row[name] for name in fieldnames})


def write_evidence_summary(field_rows, path):
    fieldnames = [
        "field", "n_H0", "H0_mean", "H0_std", "H0_q16", "H0_q50",
        "H0_q84", "lnZ_laplace", "err_lnZ_laplace", "lnZ_harmonic",
        "err_lnZ_harmonic", "BIC", "lnZ_bic", "source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in field_rows:
            writer.writerow({name: row[name] for name in fieldnames})


def set_paper_rc():
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.5,
    })


def kde_on_grid(samples, x_grid, bw=1.5):
    kde = gaussian_kde(samples)
    kde.set_bandwidth(kde.factor * bw)
    return kde(x_grid)


def field_cmap():
    return LinearSegmentedColormap.from_list(
        "ch0_manticore_single_field", FIELD_COLOURS)


def field_norm(field_rows):
    fields = np.asarray([row["field"] for row in field_rows], dtype=float)
    vmin = min(0.0, float(np.min(fields)))
    vmax = float(np.max(fields))
    if vmin == vmax:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def plot_h0_histogram(field_rows, stack, field_stat, plot_title, out_pdf):
    stat_key = "H0_q50" if field_stat == "median" else "H0_mean"
    field_values = np.asarray([row[stat_key] for row in field_rows])
    stacked_samples = stack["samples"]

    x_min = min(np.percentile(stacked_samples, 0.3), np.min(field_values))
    x_max = max(np.percentile(stacked_samples, 99.7), np.max(field_values))
    pad = 0.08 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 700)

    stacked_mean = stack["H0_mean"]
    stacked_std = stack["H0_std"]
    cmap = field_cmap()
    norm = field_norm(field_rows)
    stacked_label = (
        rf"Stacked posterior "
        rf"(${stacked_mean:.2f}\pm{stacked_std:.2f}$)"
    )

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(7.1, 3.1),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"wspace": 0.10},
        )
        ax_summary, ax_posteriors = axes

        ax_summary.hist(
            field_values,
            bins=min(10, max(5, int(np.sqrt(field_values.size)) + 2)),
            density=True,
            color=COLOURS[4],
            alpha=0.35,
            edgecolor="white",
            linewidth=0.6,
            label=rf"Field posterior {field_stat}s",
        )
        ax_summary.axvspan(
            stacked_mean - stacked_std,
            stacked_mean + stacked_std,
            color="black",
            alpha=0.10,
            lw=0,
        )
        ax_summary.axvline(stacked_mean, color="black", lw=0.9, ls=":")
        ax_summary.axvline(
            np.median(field_values),
            color=COLOURS[0],
            lw=0.95,
            ls="--",
            label=rf"Median field {field_stat} "
            rf"(${np.median(field_values):.2f}$)",
        )
        ax_summary.text(
            0.03,
            0.97,
            rf"$N_{{\rm fields}}={field_values.size}$" "\n"
            rf"$N_{{\rm samples}}={stacked_samples.size}$",
            transform=ax_summary.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
        )
        ax_summary.set_title("Field summaries", loc="left")
        ax_summary.set_ylabel("Density")
        ax_summary.legend(loc="upper right", frameon=False, handlelength=1.8)

        for row in field_rows:
            ax_posteriors.plot(
                x_grid,
                kde_on_grid(row["samples"], x_grid, bw=1.25),
                color=cmap(norm(row["field"])),
                alpha=0.74,
                linewidth=0.75,
            )
        ax_posteriors.plot(
            x_grid,
            kde_on_grid(stacked_samples, x_grid),
            color="black",
            lw=1.35,
            label=stacked_label,
        )
        ax_posteriors.axvspan(
            stacked_mean - stacked_std,
            stacked_mean + stacked_std,
            color="black",
            alpha=0.11,
            lw=0,
        )
        ax_posteriors.axvline(stacked_mean, color="black", lw=0.9, ls=":")
        ax_posteriors.plot(
            [],
            [],
            color=cmap(norm(np.median(
                [row["field"] for row in field_rows]))),
            alpha=0.55,
            lw=0.7,
            label="Individual field posteriors",
        )
        ax_posteriors.set_title("Individual posterior KDEs", loc="left")
        ax_posteriors.legend(loc="upper right", frameon=False,
                             handlelength=1.8)

        for ax in axes:
            ax.set_xlim(x_grid[0], x_grid[-1])
            ax.set_ylim(bottom=0)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=ax_posteriors, pad=0.015, fraction=0.055)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        fig.suptitle(plot_title)
        fig.supxlabel(H0_LABEL)

        fig.savefig(out_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
        out_png = out_pdf.with_suffix(".png")
        fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
    return out_png


def p_label(value):
    if value < 1e-3:
        return r"<10^{-3}"
    return rf"={value:.2f}"


def plot_h0_vs_evidence(field_rows, plot_title, out_pdf):
    h0 = np.asarray([row["H0_q50"] for row in field_rows], dtype=float)
    h0_lo = np.asarray([row["H0_q16"] for row in field_rows], dtype=float)
    h0_hi = np.asarray([row["H0_q84"] for row in field_rows], dtype=float)
    yerr = np.vstack([h0 - h0_lo, h0_hi - h0])

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(7.2, 2.75),
            sharey=True,
            constrained_layout=True,
        )

        for ax, spec in zip(axes, EVIDENCE_PANEL_SPECS):
            x = np.asarray([row[spec["key"]] for row in field_rows],
                           dtype=float)
            xerr = None
            if spec["xerr_key"] is not None:
                xerr = np.asarray(
                    [abs(row[spec["xerr_key"]]) for row in field_rows],
                    dtype=float,
                )

            pearson_r, pearson_p = pearsonr(x, h0)
            spearman_r, spearman_p = spearmanr(x, h0)

            ax.errorbar(
                x,
                h0,
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                ms=3.3,
                lw=0.75,
                elinewidth=0.55,
                capsize=1.4,
                color=spec["colour"],
                ecolor=spec["colour"],
                alpha=0.78,
                zorder=2,
            )

            ax.set_title(spec["title"], loc="left")
            ax.set_xlabel(spec["x_label"])
            ax.text(
                0.03,
                0.97,
                (
                    rf"$r={pearson_r:.2f}$, $p{p_label(pearson_p)}$" "\n"
                    rf"$\rho={spearman_r:.2f}$, "
                    rf"$p{p_label(spearman_p)}$"
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.4,
                bbox={
                    "boxstyle": "round,pad=0.14",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
            )

        axes[0].set_ylabel(H0_LABEL)
        fig.suptitle(f"{plot_title}; {len(field_rows)} fields")
        fig.savefig(out_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
        out_png = out_pdf.with_suffix(".png")
        fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
    return out_png


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    field_rows = load_field_rows(args.results_dir, args.pattern)
    stack = stacked_row(field_rows)

    if args.preset is None and args.output_prefix == DEFAULT_OUTPUT_PREFIX:
        hist_pdf = args.output_dir / "ch0_single_field_h0_histogram.pdf"
        h0_summary_csv = args.output_dir / "ch0_single_field_h0_summary.csv"
        evidence_pdf = args.output_dir / "ch0_single_field_h0_vs_evidence.pdf"
        evidence_csv = args.output_dir / "ch0_single_field_h0_vs_evidence.csv"
    else:
        prefix = args.output_prefix
        hist_pdf = args.output_dir / f"{prefix}_h0_stacked.pdf"
        h0_summary_csv = args.output_dir / f"{prefix}_h0_summary.csv"
        evidence_pdf = args.output_dir / f"{prefix}_h0_vs_lnz.pdf"
        evidence_csv = args.output_dir / f"{prefix}_h0_vs_lnz.csv"

    hist_png = plot_h0_histogram(
        field_rows, stack, args.field_stat, args.plot_title, hist_pdf)
    write_h0_summary(field_rows, stack, h0_summary_csv)

    evidence_png = plot_h0_vs_evidence(
        field_rows, args.plot_title, evidence_pdf)
    write_evidence_summary(field_rows, evidence_csv)

    print(f"Wrote {hist_pdf}")
    print(f"Wrote {hist_png}")
    print(f"Wrote {h0_summary_csv}")
    print(f"Wrote {evidence_pdf}")
    print(f"Wrote {evidence_png}")
    print(f"Wrote {evidence_csv}")
    print(
        "stacked: "
        f"H0={stack['H0_mean']:.3f} +- {stack['H0_std']:.3f}; "
        f"q16/q50/q84={stack['H0_q16']:.3f}/"
        f"{stack['H0_q50']:.3f}/{stack['H0_q84']:.3f}"
    )


if __name__ == "__main__":
    main()
