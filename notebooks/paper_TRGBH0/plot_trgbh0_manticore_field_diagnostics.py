#!/usr/bin/env python
"""Plot one-Manticore-field TRGBH0 nuisance-parameter diagnostics."""
from argparse import ArgumentParser
import csv
import re
import shutil
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: F401
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import gaussian_kde, pearsonr, spearmanr  # noqa: E402

from trgbh0_plot_style import trgbh0_cmap  # noqa: E402


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
RESULTS = ROOT / "results" / "TRGBH0_paper" / "manticore_fields_const_sigv"
OUTDIR = ROOT / "notebooks" / "paper_TRGBH0" / "output"

PATTERN = (
    "EDD_TRGB_sel-TRGB_magnitude_"
    "manticore_2MPP_MULTIBIN_N256_DES_V2_field*_"
    "manticore_field_const_sigv.hdf5"
)
FIELD_RE = re.compile(r"_field(\d+)_")
PARAMS = ("H0", "sigma_int", "sigma_v")

PLOT_CHOICES = (
    "sigma-v-posterior",
    "h0-sigma-v",
    "h0-sigma-int",
    "sigma-int-sigma-v",
    "all",
)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "plots",
        nargs="*",
        help=(
            "Diagnostics to plot: sigma-v-posterior, h0-sigma-v, "
            "h0-sigma-int, sigma-int-sigma-v, or all. Default: all."
        ),
    )
    return parser.parse_args()


def field_index(path):
    match = FIELD_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse field index from `{path}`.")
    return int(match.group(1))


def finite_samples(handle, name, path):
    samples = np.asarray(handle[f"samples/{name}"], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"`{path}` has no finite {name} samples.")
    return samples


def summary(samples):
    q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)),
        "q16": float(q16),
        "q50": float(q50),
        "q84": float(q84),
    }


def load_rows():
    paths = sorted(RESULTS.glob(PATTERN), key=field_index)
    if not paths:
        raise FileNotFoundError(f"No HDF5 files matching `{PATTERN}`.")

    rows = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            samples = {name: finite_samples(handle, name, path) for name in PARAMS}
        rows.append({
            "field": field_index(path),
            "source": str(path),
            **{f"n_{name}": samples[name].size for name in PARAMS},
            **{name: summary(samples[name]) for name in PARAMS},
            "samples": samples,
        })
    return rows


def write_summary(rows, params, path):
    fieldnames = ["field", *[f"n_{name}" for name in params]]
    for name in params:
        fieldnames.extend([
            f"{name}_mean",
            f"{name}_std",
            f"{name}_q16",
            f"{name}_q50",
            f"{name}_q84",
        ])
    fieldnames.append("source")

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {"field": row["field"], "source": row["source"]}
            for name in params:
                out[f"n_{name}"] = row[f"n_{name}"]
                for key in ("mean", "std", "q16", "q50", "q84"):
                    out[f"{name}_{key}"] = row[name][key]
            writer.writerow(out)


def p_label(value):
    if value < 1e-3:
        return r"<10^{-3}"
    return f"{value:.2f}"


def kde_on_grid(samples, x_grid, bw=1.2):
    kde = gaussian_kde(samples)
    kde.set_bandwidth(kde.factor * bw)
    return kde(x_grid)


def set_paper_rc():
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.4,
    })


def copy_outputs(*paths):
    for path in paths:
        shutil.copyfile(path, RESULTS / path.name)


def fields_and_missing(rows):
    fields = [row["field"] for row in rows]
    missing = [i for i in range(30) if i not in set(fields)]
    return fields, missing


def print_outputs(name, out_pdf, out_png, summary_csv, rows, corr=None):
    fields, missing = fields_and_missing(rows)
    print(f"[{name}] Wrote {out_pdf}")
    print(f"[{name}] Wrote {out_png}")
    print(f"[{name}] Wrote {summary_csv}")
    print(f"[{name}] Fields plotted: {len(fields)}; missing: {missing}")
    if corr is not None:
        pearson_r, pearson_p, spearman_r, spearman_p = corr
        print(f"[{name}] Pearson r={pearson_r:.3f}, p={pearson_p:.3g}")
        print(f"[{name}] Spearman rho={spearman_r:.3f}, p={spearman_p:.3g}")


def plot_sigma_v_posterior(rows):
    summary_csv = RESULTS / "manticore_field_sigma_v_summary.csv"
    write_summary(rows, ("sigma_v",), summary_csv)

    fields = np.asarray([row["field"] for row in rows], dtype=int)
    medians = np.asarray([row["sigma_v"]["q50"] for row in rows])
    all_samples = np.concatenate([row["samples"]["sigma_v"] for row in rows])
    x_min, x_max = np.percentile(all_samples, [0.2, 99.8])
    pad = 0.15 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 600)

    cmap = trgbh0_cmap("trgbh0_manticore_fields_sigma_v")
    norm = plt.Normalize(vmin=0, vmax=29)
    fields_list, missing = fields_and_missing(rows)

    with plt.style.context("science"):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(4.7, 3.1))
        ax.hist(
            medians,
            bins=min(9, max(4, int(np.sqrt(len(medians))) + 2)),
            density=True,
            color="#b8b8b8",
            alpha=0.45,
            label=r"Field median $\sigma_v$",
            zorder=0,
        )
        for row in rows:
            ax.plot(
                x_grid,
                kde_on_grid(row["samples"]["sigma_v"], x_grid),
                color=cmap(norm(row["field"])),
                lw=0.9,
                alpha=0.72,
                zorder=2,
            )
        ax.plot(
            x_grid,
            kde_on_grid(all_samples, x_grid),
            color="black",
            lw=1.35,
            label=r"Stacked posterior",
            zorder=4,
        )
        ax.axvline(
            np.median(medians),
            color="black",
            lw=1.0,
            ls=":",
            label=r"Median of field medians",
        )
        ax.set_xlabel(r"$\sigma_v ~ [\mathrm{km}\,\mathrm{s}^{-1}]$")
        ax.set_ylabel("Posterior density")
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.set_ylim(bottom=0)

        note = f"{len(fields_list)} fields"
        if missing:
            note += "; missing " + ", ".join(str(i) for i in missing)
        ax.set_title(note, loc="left")

        curve_proxy = Line2D(
            [0], [0], color=cmap(norm(15)), lw=1.0,
            label=r"Individual field posterior")
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(1, curve_proxy)
        labels.insert(1, curve_proxy.get_label())
        ax.legend(
            handles, labels,
            loc="upper right",
            fontsize=6.5,
            frameon=False,
            handlelength=1.7,
        )

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.055)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)

        out_pdf = OUTDIR / "trgbh0_manticore_field_sigma_v_posteriors.pdf"
        fig.tight_layout()
        fig.savefig(out_pdf, dpi=500, bbox_inches="tight")
        out_png = out_pdf.with_suffix(".png")
        fig.savefig(out_png, dpi=500, bbox_inches="tight")
        plt.close(fig)

    copy_outputs(out_pdf, out_png)
    print_outputs("sigma-v-posterior", out_pdf, out_png, summary_csv, rows)


def scatter_arrays(rows, x_name, y_name):
    x = np.asarray([row[x_name]["q50"] for row in rows])
    x_lo = np.asarray([row[x_name]["q16"] for row in rows])
    x_hi = np.asarray([row[x_name]["q84"] for row in rows])
    y = np.asarray([row[y_name]["q50"] for row in rows])
    y_lo = np.asarray([row[y_name]["q16"] for row in rows])
    y_hi = np.asarray([row[y_name]["q84"] for row in rows])
    return x, x_lo, x_hi, y, y_lo, y_hi


def plot_two_parameter_scatter(
    rows,
    name,
    x_name,
    y_name,
    x_label,
    y_label,
    cmap_name,
    out_basename,
    summary_basename,
    colour_by=None,
    colourbar_label="Manticore field",
):
    params = (x_name, y_name) if colour_by is None else (x_name, y_name, colour_by)
    summary_csv = RESULTS / f"{summary_basename}.csv"
    write_summary(rows, params, summary_csv)

    fields = np.asarray([row["field"] for row in rows], dtype=int)
    x, x_lo, x_hi, y, y_lo, y_hi = scatter_arrays(rows, x_name, y_name)
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    cmap = trgbh0_cmap(cmap_name)
    if colour_by is None:
        colour_values = fields
        norm = plt.Normalize(vmin=0, vmax=29)
    else:
        colour_values = np.asarray([row[colour_by]["mean"] for row in rows])
        norm = plt.Normalize(vmin=np.min(colour_values), vmax=np.max(colour_values))

    with plt.style.context("science"):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(4.45, 3.25))
        colours = cmap(norm(colour_values))
        for i in range(len(rows)):
            ax.errorbar(
                x[i],
                y[i],
                xerr=[[x[i] - x_lo[i]], [x_hi[i] - x[i]]],
                yerr=[[y[i] - y_lo[i]], [y_hi[i] - y[i]]],
                fmt="o",
                ms=3.9,
                lw=0.75,
                elinewidth=0.55,
                capsize=1.6,
                color=colours[i],
                ecolor=colours[i],
                alpha=0.82,
                zorder=2,
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{len(rows)} fields", loc="left")
        ax.text(
            0.03,
            0.97,
            (
                rf"$r={pearson_r:.2f}$, $p{p_label(pearson_p)}$" "\n"
                rf"$\rho={spearman_r:.2f}$, $p{p_label(spearman_p)}$"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
        )

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.055)
        cbar.set_label(colourbar_label)
        cbar.ax.tick_params(labelsize=7.0)

        out_pdf = OUTDIR / f"{out_basename}.pdf"
        fig.tight_layout()
        fig.savefig(out_pdf, dpi=500, bbox_inches="tight")
        out_png = out_pdf.with_suffix(".png")
        fig.savefig(out_png, dpi=500, bbox_inches="tight")
        plt.close(fig)

    copy_outputs(out_pdf, out_png)
    print_outputs(
        name,
        out_pdf,
        out_png,
        summary_csv,
        rows,
        corr=(pearson_r, pearson_p, spearman_r, spearman_p),
    )


def requested_plots(items):
    if not items:
        return [item for item in PLOT_CHOICES if item != "all"]
    invalid = [item for item in items if item not in PLOT_CHOICES]
    if invalid:
        raise ValueError(f"Unknown plot choice(s): {invalid}.")
    if "all" in items:
        return [item for item in PLOT_CHOICES if item != "all"]
    return items


def main():
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()

    for item in requested_plots(args.plots):
        if item == "sigma-v-posterior":
            plot_sigma_v_posterior(rows)
        elif item == "h0-sigma-v":
            plot_two_parameter_scatter(
                rows,
                name=item,
                x_name="sigma_v",
                y_name="H0",
                x_label=r"$\sigma_v ~ [\mathrm{km}\,\mathrm{s}^{-1}]$",
                y_label=(
                    r"$H_0 ~ [\mathrm{km}\,\mathrm{s}^{-1}\,"
                    r"\mathrm{Mpc}^{-1}]$"
                ),
                cmap_name="trgbh0_manticore_fields_h0_sigma_v",
                out_basename="trgbh0_manticore_field_h0_sigma_v",
                summary_basename="manticore_field_h0_sigma_v_summary",
            )
        elif item == "h0-sigma-int":
            plot_two_parameter_scatter(
                rows,
                name=item,
                x_name="sigma_int",
                y_name="H0",
                x_label=r"$\sigma_{\rm int} ~ [\mathrm{mag}]$",
                y_label=(
                    r"$H_0 ~ [\mathrm{km}\,\mathrm{s}^{-1}\,"
                    r"\mathrm{Mpc}^{-1}]$"
                ),
                cmap_name="trgbh0_manticore_fields_h0_sigma_int",
                out_basename="trgbh0_manticore_field_h0_sigma_int",
                summary_basename="manticore_field_h0_sigma_int_summary",
            )
        elif item == "sigma-int-sigma-v":
            plot_two_parameter_scatter(
                rows,
                name=item,
                x_name="sigma_int",
                y_name="sigma_v",
                x_label=r"$\sigma_{\rm int} ~ [\mathrm{mag}]$",
                y_label=r"$\sigma_v ~ [\mathrm{km}\,\mathrm{s}^{-1}]$",
                cmap_name="trgbh0_manticore_fields_sigma_int_sigma_v",
                out_basename="trgbh0_manticore_field_sigma_int_sigma_v",
                summary_basename="manticore_field_sigma_int_sigma_v_summary",
                colour_by="H0",
                colourbar_label=(
                    r"Mean $H_0 ~ [\mathrm{km}\,\mathrm{s}^{-1}\,"
                    r"\mathrm{Mpc}^{-1}]$"
                ),
            )
        else:
            raise ValueError(f"Unhandled plot `{item}`.")


if __name__ == "__main__":
    main()
