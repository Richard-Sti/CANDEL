#!/usr/bin/env python
"""Plot TRGBH0 single-field density-smoothing diagnostics."""

from argparse import ArgumentParser
from collections import defaultdict
import csv
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: F401,E402
import tomllib  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy.stats import gaussian_kde, pearsonr, spearmanr  # noqa: E402

from trgbh0_plot_style import TRGBH0_COLOURS, trgbh0_cmap  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = ROOT / "scripts" / "runs" / "tasks_TRGBH0_single_smoothed.txt"
BASELINE_TASK_FILE = ROOT / "scripts" / "runs" / "tasks_TRGBH0_single.txt"
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "output" / (
    "trgbh0_single_smoothed")
FIGURE_DPI = 500
MAX_KDE_SAMPLES = 40_000
H0_LABEL = (
    r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"
)
BIAS_PARAMS = ("alpha_low", "alpha_high", "log_rho_t", "log_rho_width")
NUISANCE_PARAMS = (
    "M_TRGB", "mu_LMC", "mu_N4258", "sigma_int", "sigma_v",
    "mag_lim_TRGB", "mag_lim_TRGB_width",
)
BIAS_LABELS = {
    "alpha_low": r"$\alpha_\mathrm{low}$",
    "alpha_high": r"$\alpha_\mathrm{high}$",
    "log_rho_t": r"$\log\rho_t$",
    "log_rho_width": r"$\log\Delta\rho$",
}
SCALE_COLOURS = {
    0.0: "#4d4d4d",
    4.0: TRGBH0_COLOURS[0],
    8.0: TRGBH0_COLOURS[1],
}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-file", type=Path, default=TASK_FILE,
        help="Task file listing TRGBH0 smoothed single-field configs.")
    parser.add_argument(
        "--baseline-task-file", type=Path, default=BASELINE_TASK_FILE,
        help=("Task file from which to add the unsmoothed PCS "
              "double-power-law density-field baseline."))
    parser.add_argument(
        "--no-unsmoothed-baseline", action="store_true",
        help="Do not add the unsmoothed density-field baseline.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTDIR,
        help="Directory for plots and summaries.")
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Skip missing HDF5 outputs instead of failing.")
    return parser.parse_args()


def repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def get_nested(mapping, keys):
    value = mapping
    for key in keys:
        value = value[key]
    return value


def get_nested_default(mapping, keys, default=None):
    value = mapping
    for key in keys:
        if key not in value:
            return default
        value = value[key]
    return value


def reconstruction_label(reconstruction, mas):
    text = str(reconstruction)
    if "ManticoreLocalCOLA" in text:
        return f"COLA/{mas}" if mas else "COLA"
    if "ManticoreLocalSWIFT" in text:
        return "SWIFT"
    return text


def family_sort_key(row):
    return (
        row["which_bias"],
        row["reconstruction_label"],
        float(row["smooth_R"]),
        row["field"],
    )


def task_specs(task_file):
    specs = []
    with repo_path(task_file).open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            task, config_path = line.split(maxsplit=1)
            config_path = repo_path(config_path)
            with config_path.open("rb") as config_handle:
                config = tomllib.load(config_handle)

            mas = get_nested_default(
                config, ("io", "reconstruction_main",
                         "ManticoreLocalCOLA", "which_MAS"), "")
            reconstruction = get_nested(
                config, ("io", "PV_main", "EDD_TRGB", "reconstruction"))
            smooth_R = float(get_nested_default(
                config, ("model", "field_3d_smoothing_scale"), 0.0))
            which_bias = get_nested(config, ("model", "which_bias"))
            recon_label = reconstruction_label(reconstruction, mas)
            specs.append({
                "task": int(task),
                "field": int(get_nested(config, ("io", "field_indices"))),
                "smooth_R": smooth_R,
                "which_bias": which_bias,
                "which_selection": get_nested_default(
                    config, ("model", "which_selection"), ""),
                "mas": mas,
                "reconstruction": str(reconstruction),
                "reconstruction_label": recon_label,
                "family": (
                    f"{which_bias} {recon_label} R={smooth_R:g}"),
                "config": str(config_path),
                "source": str(repo_path(get_nested(
                    config, ("io", "fname_output")))),
            })
    if not specs:
        raise ValueError(f"No task configs found in `{task_file}`.")
    return sorted(specs, key=family_sort_key)


def unsmoothed_density_baseline_specs(task_file):
    return [
        spec for spec in task_specs(task_file)
        if spec["which_bias"] == "double_powerlaw"
        and spec["reconstruction_label"] == "COLA/PCS"
        and spec["smooth_R"] == 0.0
    ]


def finite_samples(handle, name, path):
    samples = np.asarray(handle[f"samples/{name}"], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"`{path}` has no finite `{name}` samples.")
    return samples


def read_scalar(handle, name, path, default=None):
    if name not in handle:
        if default is not None:
            return default
        raise KeyError(f"`{path}` does not contain `{name}`.")
    value = float(handle[name][()])
    if not np.isfinite(value):
        raise ValueError(f"`{path}` has non-finite `{name}`: {value}.")
    return value


def h0_summary(samples):
    q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
    return {
        "n_H0": int(samples.size),
        "H0_mean": float(np.mean(samples)),
        "H0_std": float(np.std(samples, ddof=1)),
        "H0_q16": float(q16),
        "H0_q50": float(q50),
        "H0_q84": float(q84),
    }


def sample_summary(samples, prefix):
    q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
    return {
        f"{prefix}_mean": float(np.mean(samples)),
        f"{prefix}_std": float(np.std(samples, ddof=1)),
        f"{prefix}_q16": float(q16),
        f"{prefix}_q50": float(q50),
        f"{prefix}_q84": float(q84),
    }


def sample_summaries(handle, path, names):
    summaries = {}
    for name in names:
        dataset = f"samples/{name}"
        if dataset not in handle:
            continue
        samples = np.asarray(handle[dataset], dtype=float).reshape(-1)
        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            raise ValueError(f"`{path}` has no finite `{name}` samples.")
        summaries.update(sample_summary(samples, name))
    return summaries


def read_row(spec):
    path = Path(spec["source"])
    if not path.is_file():
        return {**spec, "status": "missing"}
    with h5py.File(path, "r") as handle:
        h0 = finite_samples(handle, "H0", path)
        bic = read_scalar(handle, "gof/BIC", path)
        return {
            **spec,
            "status": "complete",
            "source": str(path),
            "samples": h0,
            **h0_summary(h0),
            "lnZ_harmonic": read_scalar(handle, "gof/lnZ_harmonic", path),
            "err_lnZ_harmonic": read_scalar(
                handle, "gof/err_lnZ_harmonic", path),
            "lnZ_laplace": read_scalar(handle, "gof/lnZ_laplace", path),
            "err_lnZ_laplace": read_scalar(
                handle, "gof/err_lnZ_laplace", path),
            "BIC": bic,
            "lnZ_bic": -0.5 * bic,
            **sample_summaries(handle, path, BIAS_PARAMS),
            **sample_summaries(handle, path, NUISANCE_PARAMS),
        }


def load_rows(task_file, baseline_task_file, allow_missing):
    specs = [
        {**spec, "task_set": "smoothed"}
        for spec in task_specs(task_file)
    ]
    if baseline_task_file is not None:
        baseline_specs = [
            {**spec, "task_set": "unsmoothed_baseline"}
            for spec in unsmoothed_density_baseline_specs(
                baseline_task_file)
        ]
        specs = baseline_specs + specs
    rows = [read_row(spec) for spec in specs]
    missing = [row for row in rows if row["status"] != "complete"]
    if missing and not allow_missing:
        preview = "\n".join(row["source"] for row in missing[:8])
        raise FileNotFoundError(
            f"{len(missing)} outputs are missing. First missing:\n"
            f"{preview}\nPass --allow-missing to skip them.")
    completed = [row for row in rows if row["status"] == "complete"]
    if not completed:
        raise ValueError("No completed TRGBH0 single-field outputs found.")
    add_reference_deltas(completed)
    return completed, missing


def result_rows(rows):
    return [
        row for row in rows
        if row["which_bias"] == "double_powerlaw"
        and row["reconstruction_label"] == "COLA/PCS"
    ]


def grouped_by_scale(rows):
    scales = sorted({row["smooth_R"] for row in rows})
    return {
        scale: sorted([row for row in rows if row["smooth_R"] == scale],
                      key=lambda row: row["field"])
        for scale in scales
    }


def matched_fields_by_scale(grouped):
    field_sets = [
        set(row["field"] for row in group) for group in grouped.values()
    ]
    return sorted(set.intersection(*field_sets))


def available_bias_params(rows):
    return [
        name for name in BIAS_PARAMS
        if any(f"{name}_q50" in row for row in rows)
    ]


def add_reference_deltas(rows):
    reference_rows = result_rows(rows)
    if not reference_rows:
        return
    reference_scale = min(row["smooth_R"] for row in reference_rows)
    reference = {
        row["field"]: row
        for row in reference_rows if row["smooth_R"] == reference_scale
    }
    for row in rows:
        base = reference.get(row["field"])
        if base is None:
            continue
        row["reference_smooth_R"] = reference_scale
        row["reference_H0_q50"] = base["H0_q50"]
        row["reference_H0_mean"] = base["H0_mean"]
        row["reference_lnZ_harmonic"] = base["lnZ_harmonic"]
        row["delta_H0_q50_vs_reference"] = (
            row["H0_q50"] - base["H0_q50"])
        row["delta_H0_mean_vs_reference"] = (
            row["H0_mean"] - base["H0_mean"])
        row["delta_lnZ_harmonic_vs_reference"] = (
            row["lnZ_harmonic"] - base["lnZ_harmonic"])


def set_paper_rc():
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.5,
    })


def save_pdf_png(fig, out_pdf):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_pdf, out_png


def field_cmap():
    return trgbh0_cmap("trgbh0_single_smoothed_fields")


def field_norm(rows):
    fields = np.asarray([row["field"] for row in rows], dtype=float)
    vmin = min(0.0, float(np.min(fields)))
    vmax = float(np.max(fields))
    if vmin == vmax:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def scale_label(scale):
    if float(scale) == 0.0:
        return "unsmoothed"
    return rf"$R={scale:g}\,h^{{-1}}\mathrm{{Mpc}}$"


def scale_text(scale):
    if float(scale) == 0.0:
        return "none"
    return f"{scale:g}"


def scale_colour(scale):
    return SCALE_COLOURS.get(float(scale), "0.45")


def reference_text(scale):
    if float(scale) == 0.0:
        return "unsmoothed double-power-law COLA/PCS density field"
    return f"double-power-law COLA/PCS density smoothing R={scale:g} Mpc/h"


def stacked_samples(rows):
    return np.concatenate([row["samples"] for row in rows])


def kde_on_grid(samples, x_grid, bw=1.15):
    if samples.size > MAX_KDE_SAMPLES:
        indices = np.linspace(0, samples.size - 1, MAX_KDE_SAMPLES,
                              dtype=int)
        samples = samples[indices]
    kde = gaussian_kde(samples)
    kde.set_bandwidth(kde.factor * bw)
    return kde(x_grid)


def plot_matched_field_impact(rows, out_pdf):
    grouped = grouped_by_scale(rows)
    scales = np.asarray(sorted(grouped), dtype=float)
    positions = np.arange(len(scales), dtype=float)
    fields = matched_fields_by_scale(grouped)
    cmap = field_cmap()
    norm = field_norm(rows)
    ref = float(scales[0])

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 1, figsize=(4.6, 4.35), sharex=True,
            constrained_layout=True, height_ratios=(1.15, 1.0))
        ax_h0, ax_delta = axes

        for field in fields:
            field_rows = [
                next(row for row in grouped[scale]
                     if row["field"] == field)
                for scale in scales
            ]
            colour = cmap(norm(field))
            h0 = np.asarray([row["H0_q50"] for row in field_rows])
            delta = np.asarray([
                row["delta_H0_q50_vs_reference"] for row in field_rows])
            ax_h0.plot(positions, h0, color=colour, lw=0.65, alpha=0.48)
            ax_h0.scatter(
                positions, h0, color=colour, s=11, alpha=0.80, zorder=3)
            ax_delta.plot(
                positions, delta, color=colour, lw=0.65, alpha=0.48)
            ax_delta.scatter(
                positions, delta, color=colour, s=11, alpha=0.80, zorder=3)

        mean_h0 = []
        std_h0 = []
        mean_delta = []
        std_delta = []
        for scale in scales:
            group = grouped[scale]
            h0 = np.asarray([row["H0_q50"] for row in group], dtype=float)
            delta = np.asarray([
                row["delta_H0_q50_vs_reference"] for row in group],
                dtype=float)
            mean_h0.append(np.mean(h0))
            std_h0.append(np.std(h0, ddof=1))
            mean_delta.append(np.mean(delta))
            std_delta.append(np.std(delta, ddof=1))

        ax_h0.errorbar(
            positions, mean_h0, yerr=std_h0, color="black", marker="o",
            ms=4.1, lw=1.25, capsize=2.4,
            label="field mean\n(error bar: field-to-field std)",
            zorder=5)
        ax_delta.errorbar(
            positions, mean_delta, yerr=std_delta, color="black",
            marker="o", ms=4.1, lw=1.25, capsize=2.4, zorder=5)
        ax_delta.axhline(0.0, color="0.35", lw=0.75, ls="--")
        ax_h0.set_ylabel(H0_LABEL)
        ax_delta.set_ylabel(rf"$\Delta H_0$ vs {scale_label(ref)}")
        ax_delta.set_xlabel(
            r"Density-field smoothing scale [$h^{-1}\mathrm{Mpc}$]")
        ax_h0.set_title("Matched Manticore realisations", loc="left")
        ax_h0.legend(loc="upper right", frameon=False, handlelength=1.7)
        for ax in axes:
            ax.set_xticks(positions)
            ax.set_xticklabels([scale_text(scale) for scale in scales])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, pad=0.015, fraction=0.055)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_h0_distributions(rows, out_pdf):
    grouped = grouped_by_scale(rows)
    scales = np.asarray(sorted(grouped), dtype=float)
    all_h0 = np.concatenate([
        np.asarray([row["H0_q50"] for row in grouped[scale]], dtype=float)
        for scale in scales
    ])
    all_samples = np.concatenate([
        stacked_samples(grouped[scale]) for scale in scales])
    x_min = min(np.percentile(all_samples, 0.3), np.min(all_h0))
    x_max = max(np.percentile(all_samples, 99.7), np.max(all_h0))
    pad = 0.08 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 800)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 2, figsize=(7.1, 3.05), constrained_layout=True)
        ax_violin, ax_kde = axes

        values = [
            np.asarray([row["H0_q50"] for row in grouped[scale]],
                       dtype=float)
            for scale in scales
        ]
        positions = np.arange(len(scales))
        parts = ax_violin.violinplot(
            values, positions=positions, widths=0.78,
            showmeans=False, showextrema=False, showmedians=False)
        for body, scale in zip(parts["bodies"], scales):
            body.set_facecolor(scale_colour(scale))
            body.set_edgecolor("none")
            body.set_alpha(0.38)
        for i, (scale, vals) in enumerate(zip(scales, values)):
            colour = scale_colour(scale)
            jitter = np.linspace(-0.18, 0.18, len(vals))
            ax_violin.scatter(
                np.full(len(vals), i) + jitter, vals, s=7,
                color=colour, alpha=0.40, edgecolor="none")
            q16, q50, q84 = np.percentile(vals, [16.0, 50.0, 84.0])
            ax_violin.errorbar(
                i, q50, yerr=[[q50 - q16], [q84 - q50]],
                fmt="o", color="black", ms=3.7, capsize=2.4, zorder=5)
        ax_violin.set_xticks(positions)
        ax_violin.set_xticklabels([scale_text(scale) for scale in scales])
        ax_violin.set_xlabel(
            r"Density-field smoothing scale [$h^{-1}\mathrm{Mpc}$]")
        ax_violin.set_ylabel(H0_LABEL)
        ax_violin.set_title("Field-median distribution", loc="left")

        for scale in scales:
            group = grouped[scale]
            samples = stacked_samples(group)
            summary = h0_summary(samples)
            colour = scale_colour(scale)
            ax_kde.plot(
                x_grid, kde_on_grid(samples, x_grid), color=colour, lw=1.2,
                label=(
                    rf"{scale_label(scale)} "
                    rf"(${summary['H0_mean']:.2f}\pm"
                    rf"{summary['H0_std']:.2f}$)"
                ))
            ax_kde.axvline(
                summary["H0_mean"], color=colour, lw=0.65, alpha=0.45)
        ax_kde.set_xlabel(H0_LABEL)
        ax_kde.set_ylabel("Density")
        ax_kde.set_title("Stacked posteriors", loc="left")
        ax_kde.set_ylim(bottom=0)
        ax_kde.legend(loc="upper right", frameon=False, handlelength=1.8)
        return save_pdf_png(fig, out_pdf)


def p_label(value):
    if value < 1e-3:
        return r"<10^{-3}"
    return rf"={value:.2f}"


def correlation_summary(x, h0):
    if x.size < 3 or np.std(x) == 0.0 or np.std(h0) == 0.0:
        return r"$r=\mathrm{n/a}$" "\n" r"$\rho=\mathrm{n/a}$"
    pearson_r, pearson_p = pearsonr(x, h0)
    spearman_r, spearman_p = spearmanr(x, h0)
    return (
        rf"$r={pearson_r:.2f}$, $p{p_label(pearson_p)}$" "\n"
        rf"$\rho={spearman_r:.2f}$, $p{p_label(spearman_p)}$"
    )


def plot_h0_vs_harmonic_lnz(rows, out_pdf):
    grouped = grouped_by_scale(rows)
    scales = sorted(grouped)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, len(scales), figsize=(2.35 * len(scales), 2.55),
            sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, scale in zip(axes, scales):
            group = grouped[scale]
            x = np.asarray([row["lnZ_harmonic"] for row in group],
                           dtype=float)
            xerr = np.asarray([
                abs(row["err_lnZ_harmonic"]) for row in group], dtype=float)
            h0 = np.asarray([row["H0_q50"] for row in group], dtype=float)
            h0_lo = np.asarray([row["H0_q16"] for row in group], dtype=float)
            h0_hi = np.asarray([row["H0_q84"] for row in group], dtype=float)
            yerr = np.vstack([h0 - h0_lo, h0_hi - h0])
            colour = scale_colour(scale)
            ax.errorbar(
                x, h0, xerr=xerr, yerr=yerr, fmt="o", ms=3.1,
                color=colour, ecolor=colour, elinewidth=0.45,
                capsize=1.0, alpha=0.72, zorder=2)
            ax.set_title(scale_label(scale), loc="left")
            ax.set_xlabel(r"harmonic $\ln Z$")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.text(
                0.03, 0.97, correlation_summary(x, h0),
                transform=ax.transAxes, ha="left", va="top", fontsize=6.3,
                bbox={
                    "boxstyle": "round,pad=0.14",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                })
            if ax is axes[0]:
                ax.set_ylabel(H0_LABEL)
            else:
                ax.tick_params(labelleft=False)
        return save_pdf_png(fig, out_pdf)


def plot_lnz_distributions(rows, out_pdf):
    grouped = grouped_by_scale(rows)
    scales = np.asarray(sorted(grouped), dtype=float)
    positions = np.arange(len(scales), dtype=float)
    fields = matched_fields_by_scale(grouped)
    cmap = field_cmap()
    norm = field_norm(rows)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 1, figsize=(4.9, 4.15), sharex=True,
            constrained_layout=True, height_ratios=(1.0, 1.0))
        ax_lnz, ax_delta = axes

        values = [
            np.asarray([row["lnZ_harmonic"] for row in grouped[scale]],
                       dtype=float)
            for scale in scales
        ]
        parts = ax_lnz.violinplot(
            values, positions=positions, widths=0.72, showmeans=False,
            showextrema=False, showmedians=False)
        for body, scale in zip(parts["bodies"], scales):
            body.set_facecolor(scale_colour(scale))
            body.set_edgecolor("none")
            body.set_alpha(0.34)
        for i, (scale, vals) in enumerate(zip(scales, values)):
            colour = scale_colour(scale)
            jitter = np.linspace(-0.17, 0.17, len(vals))
            ax_lnz.scatter(
                np.full(len(vals), positions[i]) + jitter, vals,
                s=6.5, color=colour, alpha=0.38, edgecolor="none")
            ax_lnz.errorbar(
                positions[i], np.mean(vals), yerr=np.std(vals, ddof=1),
                fmt="o", color="black", ms=3.5, capsize=2.2, zorder=5,
                label=("field mean\n(error bar: field-to-field std)"
                       if i == 0 else None))

        for field in fields:
            field_rows = [
                next(row for row in grouped[scale]
                     if row["field"] == field)
                for scale in scales
            ]
            delta = np.asarray([
                row["delta_lnZ_harmonic_vs_reference"]
                for row in field_rows
            ], dtype=float)
            colour = cmap(norm(field))
            ax_delta.plot(
                positions, delta, color=colour, lw=0.6, alpha=0.40)
            ax_delta.scatter(
                positions, delta, color=colour, s=8.0, alpha=0.68,
                edgecolor="none", zorder=3)

        mean_delta = []
        std_delta = []
        for scale in scales:
            vals = np.asarray([
                row["delta_lnZ_harmonic_vs_reference"]
                for row in grouped[scale]
            ], dtype=float)
            mean_delta.append(np.mean(vals))
            std_delta.append(np.std(vals, ddof=1))
        ax_delta.errorbar(
            positions, mean_delta, yerr=std_delta, color="black",
            marker="o", ms=3.7, lw=1.15, capsize=2.2, zorder=5)
        ax_delta.axhline(0.0, color="0.35", lw=0.75, ls="--")

        ax_lnz.set_ylabel(r"harmonic $\ln Z$")
        ax_lnz.set_title("Evidence distribution", loc="left")
        ax_lnz.legend(loc="best", frameon=False, handlelength=1.6)
        ax_delta.set_ylabel(r"$\Delta$ harmonic $\ln Z$")
        ax_delta.set_xlabel(
            r"Density-field smoothing scale [$h^{-1}\mathrm{Mpc}$]")
        ax_delta.set_xticks(positions)
        ax_delta.set_xticklabels([scale_text(scale) for scale in scales])
        return save_pdf_png(fig, out_pdf)


def plot_bias_param_matched_fields(rows, out_pdf):
    params = available_bias_params(rows)
    if not params:
        return []
    grouped = grouped_by_scale(rows)
    scales = np.asarray(sorted(grouped), dtype=float)
    positions = np.arange(len(scales), dtype=float)
    fields = matched_fields_by_scale(grouped)
    cmap = field_cmap()
    norm = field_norm(rows)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            len(params), 1, figsize=(4.9, 1.75 * len(params)),
            sharex=True, constrained_layout=True)
        axes = np.atleast_1d(axes)

        for ax, param in zip(axes, params):
            for field in fields:
                field_rows = [
                    next(row for row in grouped[scale]
                         if row["field"] == field)
                    for scale in scales
                ]
                values = np.asarray([row[f"{param}_q50"]
                                     for row in field_rows], dtype=float)
                colour = cmap(norm(field))
                ax.plot(positions, values, color=colour, lw=0.6, alpha=0.40)
                ax.scatter(
                    positions, values, color=colour, s=8.0, alpha=0.68,
                    edgecolor="none", zorder=3)

            means = []
            stds = []
            for scale in scales:
                vals = np.asarray([
                    row[f"{param}_q50"] for row in grouped[scale]
                ], dtype=float)
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=1))
            ax.errorbar(
                positions, means, yerr=stds, color="black", marker="o",
                ms=3.7, lw=1.15, capsize=2.2,
                label="field mean\n(error bar: field-to-field std)",
                zorder=5)
            ax.set_ylabel(BIAS_LABELS.get(param, param))
            ax.set_title(BIAS_LABELS.get(param, param), loc="left")

        axes[0].legend(loc="best", frameon=False, handlelength=1.6)
        axes[-1].set_xticks(positions)
        axes[-1].set_xticklabels([scale_text(scale) for scale in scales])
        axes[-1].set_xlabel(
            r"Density-field smoothing scale [$h^{-1}\mathrm{Mpc}$]")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, pad=0.015, fraction=0.040)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_bias_param_distributions(rows, out_pdf):
    params = available_bias_params(rows)
    if not params:
        return []
    grouped = grouped_by_scale(rows)
    scales = sorted(grouped)
    positions = np.arange(len(scales), dtype=float)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            len(params), 1, figsize=(4.9, 1.65 * len(params)),
            sharex=True, constrained_layout=True)
        axes = np.atleast_1d(axes)

        for ax, param in zip(axes, params):
            values = [
                np.asarray([row[f"{param}_q50"] for row in grouped[scale]],
                           dtype=float)
                for scale in scales
            ]
            parts = ax.violinplot(
                values, positions=positions, widths=0.72,
                showmeans=False, showextrema=False, showmedians=False)
            for body, scale in zip(parts["bodies"], scales):
                body.set_facecolor(scale_colour(scale))
                body.set_edgecolor("none")
                body.set_alpha(0.35)
            for i, (scale, vals) in enumerate(zip(scales, values)):
                colour = scale_colour(scale)
                jitter = np.linspace(-0.17, 0.17, len(vals))
                ax.scatter(
                    np.full(len(vals), positions[i]) + jitter, vals,
                    s=6.5, color=colour, alpha=0.35, edgecolor="none")
                q16, q50, q84 = np.percentile(vals, [16.0, 50.0, 84.0])
                ax.errorbar(
                    positions[i], q50, yerr=[[q50 - q16], [q84 - q50]],
                    fmt="o", color="black", ms=3.4, capsize=2.1, zorder=5)
            ax.set_ylabel(BIAS_LABELS.get(param, param))
            ax.set_title(BIAS_LABELS.get(param, param), loc="left")

        axes[-1].set_xticks(positions)
        axes[-1].set_xticklabels([scale_text(scale) for scale in scales])
        axes[-1].set_xlabel(
            r"Density-field smoothing scale [$h^{-1}\mathrm{Mpc}$]")
        return save_pdf_png(fig, out_pdf)


def write_rows_csv(rows, path):
    fieldnames = [
        "status", "task_set", "task", "field", "smooth_R",
        "reference_smooth_R", "which_bias", "which_selection", "mas",
        "reconstruction",
        "reconstruction_label", "family", "n_H0", "H0_mean", "H0_std",
        "H0_q16", "H0_q50", "H0_q84", "reference_H0_q50",
        "delta_H0_q50_vs_reference", "delta_H0_mean_vs_reference",
        "lnZ_harmonic", "err_lnZ_harmonic", "reference_lnZ_harmonic",
        "delta_lnZ_harmonic_vs_reference", "lnZ_laplace",
        "err_lnZ_laplace", "BIC", "lnZ_bic", "source", "config",
    ]
    for name in (*BIAS_PARAMS, *NUISANCE_PARAMS):
        fieldnames.extend([
            f"{name}_mean", f"{name}_std", f"{name}_q16",
            f"{name}_q50", f"{name}_q84",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_summary(rows, missing, path):
    selected = result_rows(rows)
    grouped = grouped_by_scale(selected)
    ref = min(grouped)
    smoothed = [row for row in rows if row.get("task_set") == "smoothed"]
    baseline = [
        row for row in rows
        if row.get("task_set") == "unsmoothed_baseline"
    ]
    lines = [
        "# TRGBH0 Single-Field Smoothing Diagnostics",
        "",
        f"Complete outputs: {len(rows)}.",
        f"Smoothed task outputs: {len(smoothed)}.",
        f"Matched unsmoothed baseline outputs: {len(baseline)}.",
        f"Missing outputs: {len(missing)}.",
        f"Result set: {len(selected)} double-power-law COLA/PCS runs.",
        f"Reference model for matched shifts: {reference_text(ref)}.",
        "",
        "| density smoothing R [Mpc/h] | fields | mean field H0 | "
        "std field H0 | median field H0 | stacked H0 mean | "
        "stacked H0 std | mean delta H0 | std delta H0 | "
        "median harmonic lnZ | best harmonic lnZ field |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |",
    ]
    for scale, group in grouped.items():
        h0 = np.asarray([row["H0_q50"] for row in group], dtype=float)
        deltas = np.asarray([
            row["delta_H0_q50_vs_reference"] for row in group], dtype=float)
        samples = stacked_samples(group)
        summary = h0_summary(samples)
        best = max(group, key=lambda row: row["lnZ_harmonic"])
        lnz = np.asarray([row["lnZ_harmonic"] for row in group], dtype=float)
        lines.append(
            f"| {scale_text(scale)} | {len(group)} | {np.mean(h0):.3f} | "
            f"{np.std(h0, ddof=1):.3f} | {np.median(h0):.3f} | "
            f"{summary['H0_mean']:.3f} | {summary['H0_std']:.3f} | "
            f"{np.mean(deltas):+.3f} | {np.std(deltas, ddof=1):.3f} | "
            f"{np.median(lnz):.3f} | {best['field']} |"
        )

    shifted = [
        row for row in selected
        if row["smooth_R"] != ref and "delta_H0_q50_vs_reference" in row
    ]
    if shifted:
        largest = sorted(
            shifted, key=lambda row: abs(row[
                "delta_H0_q50_vs_reference"]), reverse=True)[:8]
        lines.extend([
            "",
            "## Largest matched H0 shifts",
            "",
            "| field | R [Mpc/h] | delta H0 | H0 | reference H0 | "
            "delta harmonic lnZ |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in largest:
            lines.append(
                f"| {row['field']} | {scale_text(row['smooth_R'])} | "
                f"{row['delta_H0_q50_vs_reference']:+.3f} | "
                f"{row['H0_q50']:.3f} | {row['reference_H0_q50']:.3f} | "
                f"{row['delta_lnZ_harmonic_vs_reference']:+.3f} |"
            )

    if missing:
        lines.extend(["", "## Missing", ""])
        for row in missing:
            lines.append(f"- `{row['source']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_task_file = (
        None if args.no_unsmoothed_baseline else args.baseline_task_file)
    rows, missing = load_rows(
        args.task_file, baseline_task_file, args.allow_missing)
    selected = result_rows(rows)

    csv_path = args.output_dir / "trgbh0_single_smoothed_summary.csv"
    txt_path = args.output_dir / "trgbh0_single_smoothed_summary.txt"
    impact_pdf = (
        args.output_dir / "trgbh0_single_smoothed_h0_matched_fields.pdf")
    h0_dist_pdf = (
        args.output_dir / "trgbh0_single_smoothed_h0_distributions.pdf")
    h0_lnz_pdf = (
        args.output_dir / "trgbh0_single_smoothed_h0_vs_harmonic_lnz.pdf")
    lnz_dist_pdf = (
        args.output_dir / "trgbh0_single_smoothed_lnz_distributions.pdf")
    bias_matched_pdf = (
        args.output_dir /
        "trgbh0_single_smoothed_bias_params_matched_fields.pdf")
    bias_dist_pdf = (
        args.output_dir /
        "trgbh0_single_smoothed_bias_params_distributions.pdf")

    write_rows_csv(rows, csv_path)
    write_summary(rows, missing, txt_path)
    written = [
        csv_path,
        txt_path,
        *plot_matched_field_impact(selected, impact_pdf),
        *plot_h0_distributions(selected, h0_dist_pdf),
        *plot_h0_vs_harmonic_lnz(selected, h0_lnz_pdf),
        *plot_lnz_distributions(selected, lnz_dist_pdf),
        *plot_bias_param_matched_fields(selected, bias_matched_pdf),
        *plot_bias_param_distributions(selected, bias_dist_pdf),
    ]

    for path in written:
        print(f"Wrote {path}")
    print(f"Complete outputs: {len(rows)}; missing: {len(missing)}.")


if __name__ == "__main__":
    main()
