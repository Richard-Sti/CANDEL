#!/usr/bin/env python
"""Plot CH0 single-field smoothing diagnostics."""

from argparse import ArgumentParser
from collections import defaultdict
import csv
from math import ceil
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: F401,E402
import tomllib  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy.stats import gaussian_kde, pearsonr, spearmanr  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = ROOT / "scripts" / "runs" / "tasks_CH0_single_smoothed.txt"
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "ch0_single_smoothed_plots"
FIGURE_DPI = 500
MAX_KDE_SAMPLES = 40_000
H0_LABEL = (
    r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"
)
FIELD_COLOURS = ["#ef476f", "#473198", "#a8c256", "#5adbff", "#fe9000"]
SCALE_COLOURS = {
    4.0: "#87193d",
    8.0: "#1e42b9",
    16.0: "#168039",
    32.0: "#fe9000",
}
VARIANT_COLOURS = {
    "uniform_cola_cic": "#7d4e93",
    "uniform_swift": "#008b8b",
    "other": "0.35",
}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-file", type=Path, default=TASK_FILE,
        help="Task file listing CH0 smoothed single-field configs.")
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


def family_label(which_bias, smooth_R, recon_label):
    if which_bias == "double_powerlaw":
        return f"double_powerlaw {recon_label} R={smooth_R:g}"
    if which_bias == "uniform":
        return f"uniform {recon_label}"
    return f"{which_bias} {recon_label} R={smooth_R:g}"


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
            which_bias = get_nested(config, ("model", "which_bias"))
            smooth_R = float(get_nested(
                config, ("model", "field_3d_smoothing_scale")))
            mas = get_nested_default(
                config, ("io", "reconstruction_main",
                         "ManticoreLocalCOLA", "which_MAS"), "")
            reconstruction = get_nested(
                config, ("io", "SH0ES", "reconstruction"))
            recon_label = reconstruction_label(reconstruction, mas)
            specs.append({
                "task": int(task),
                "field": int(get_nested(config, ("io", "field_indices"))),
                "smooth_R": smooth_R,
                "which_bias": which_bias,
                "mas": mas,
                "reconstruction": str(reconstruction),
                "reconstruction_label": recon_label,
                "family": family_label(which_bias, smooth_R, recon_label),
                "config": str(config_path),
                "source": str(repo_path(get_nested(
                    config, ("io", "fname_output")))),
            })
    if not specs:
        raise ValueError(f"No task configs found in `{task_file}`.")
    return sorted(specs, key=lambda spec: (
        family_sort_key(spec), spec["field"], spec["task"]))


def finite_samples(handle, name, path):
    samples = np.asarray(handle[f"samples/{name}"], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"`{path}` has no finite `{name}` samples.")
    return samples


def read_scalar(handle, name, path):
    if name not in handle:
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
        }


def load_rows(task_file, allow_missing):
    rows = [read_row(spec) for spec in task_specs(task_file)]
    missing = [row for row in rows if row["status"] != "complete"]
    if missing and not allow_missing:
        preview = "\n".join(row["source"] for row in missing[:8])
        raise FileNotFoundError(
            f"{len(missing)} outputs are missing. First missing:\n"
            f"{preview}\nPass --allow-missing to skip them.")
    completed = [row for row in rows if row["status"] == "complete"]
    if not completed:
        raise ValueError("No completed smoothed single-field outputs found.")
    add_reference_deltas(completed)
    return completed, missing


def family_sort_key(row):
    if row["which_bias"] == "double_powerlaw":
        return (0, row["reconstruction_label"], row["smooth_R"])
    if row["which_bias"] == "uniform":
        recon_rank = 0 if row["reconstruction_label"] == "COLA/CIC" else 1
        return (1, recon_rank, row["reconstruction_label"])
    return (2, row["which_bias"], row["reconstruction_label"],
            row["smooth_R"])


def smoothed_result_rows(rows):
    return [
        row for row in rows
        if row["which_bias"] == "double_powerlaw"
        and row["reconstruction_label"] == "COLA/CIC"
    ]


def category_groups(rows):
    grouped = defaultdict(list)
    first = {}
    for row in rows:
        grouped[row["family"]].append(row)
        first.setdefault(row["family"], row)
    return [
        (first[family], sorted(group, key=lambda row: row["field"]))
        for family, group in sorted(
            grouped.items(), key=lambda item: family_sort_key(first[item[0]]))
    ]


def grouped_by_scale(rows):
    scales = sorted({row["smooth_R"] for row in rows})
    return {
        scale: sorted([row for row in rows if row["smooth_R"] == scale],
                      key=lambda row: row["field"])
        for scale in scales
    }


def add_reference_deltas(rows):
    reference_rows = smoothed_result_rows(rows)
    if not reference_rows:
        return
    reference_scale = min(row["smooth_R"] for row in reference_rows)
    reference = {
        (row["field"], row["reconstruction_label"]): row
        for row in reference_rows if row["smooth_R"] == reference_scale
    }
    for row in rows:
        base = reference.get((row["field"], row["reconstruction_label"]))
        if base is None:
            continue
        row["reference_smooth_R"] = reference_scale
        row["reference_which_bias"] = base["which_bias"]
        row["reference_reconstruction_label"] = base[
            "reconstruction_label"]
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
    return LinearSegmentedColormap.from_list(
        "ch0_smoothed_single_fields", FIELD_COLOURS)


def field_norm(rows):
    fields = np.asarray([row["field"] for row in rows], dtype=float)
    vmin = min(0.0, float(np.min(fields)))
    vmax = float(np.max(fields))
    if vmin == vmax:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def scale_label(scale):
    return rf"$R={scale:g}\,h^{{-1}}\mathrm{{Mpc}}$"


def plot_category_label(row):
    if row["which_bias"] == "double_powerlaw":
        return scale_label(row["smooth_R"])
    if row["which_bias"] == "uniform":
        return f"uniform {row['reconstruction_label']}"
    return row["family"]


def tick_category_label(row):
    if row["which_bias"] == "double_powerlaw":
        return f"R={row['smooth_R']:g}"
    if row["which_bias"] == "uniform":
        return "uniform\n" + row["reconstruction_label"]
    return row["family"]


def row_colour(row):
    if row["which_bias"] == "double_powerlaw":
        return SCALE_COLOURS.get(float(row["smooth_R"]), "0.5")
    if row["which_bias"] == "uniform":
        if row["reconstruction_label"] == "COLA/CIC":
            return VARIANT_COLOURS["uniform_cola_cic"]
        if row["reconstruction_label"] == "SWIFT":
            return VARIANT_COLOURS["uniform_swift"]
    return VARIANT_COLOURS["other"]


def stacked_samples(rows):
    return np.concatenate([row["samples"] for row in rows])


def kde_on_grid(samples, x_grid, bw=1.15):
    if samples.size > MAX_KDE_SAMPLES:
        indices = np.linspace(
            0, samples.size - 1, MAX_KDE_SAMPLES, dtype=int)
        samples = samples[indices]
    kde = gaussian_kde(samples)
    kde.set_bandwidth(kde.factor * bw)
    return kde(x_grid)


def matched_fields_by_scale(grouped):
    field_sets = [set(row["field"] for row in group)
                  for group in grouped.values()]
    common = set.intersection(*field_sets)
    return sorted(common)


def plot_matched_field_impact(rows, out_pdf):
    grouped = grouped_by_scale(rows)
    scales = np.asarray(sorted(grouped), dtype=float)
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
                next(row for row in grouped[scale] if row["field"] == field)
                for scale in scales
            ]
            colour = cmap(norm(field))
            h0 = np.asarray([row["H0_q50"] for row in field_rows])
            delta = np.asarray([
                row["delta_H0_q50_vs_reference"] for row in field_rows])
            ax_h0.plot(scales, h0, color=colour, lw=0.65, alpha=0.48)
            ax_h0.scatter(
                scales, h0, color=colour, s=11, alpha=0.80, zorder=3)
            ax_delta.plot(scales, delta, color=colour, lw=0.65, alpha=0.48)
            ax_delta.scatter(
                scales, delta, color=colour, s=11, alpha=0.80, zorder=3)

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
            scales, mean_h0, yerr=std_h0, color="black", marker="o",
            ms=4.1, lw=1.25, capsize=2.4, label="field mean +- std",
            zorder=5)
        ax_delta.errorbar(
            scales, mean_delta, yerr=std_delta, color="black", marker="o",
            ms=4.1, lw=1.25, capsize=2.4, label="field mean +- std",
            zorder=5)
        ax_delta.axhline(0.0, color="0.35", lw=0.75, ls="--")
        ax_h0.set_ylabel(H0_LABEL)
        ax_delta.set_ylabel(rf"$\Delta H_0$ vs {scale_label(ref)}")
        ax_delta.set_xlabel(r"Field smoothing scale [$h^{-1}\mathrm{Mpc}$]")
        ax_h0.set_title("Matched Manticore realisations", loc="left")
        ax_h0.legend(loc="upper right", frameon=False, handlelength=1.7)
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xticks(scales)
            ax.set_xticklabels([f"{scale:g}" for scale in scales])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, pad=0.015, fraction=0.055)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_broad_shift(rows, out_pdf):
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
        parts = ax_violin.violinplot(
            values, positions=np.arange(len(scales)), widths=0.78,
            showmeans=False, showextrema=False, showmedians=False)
        for body, scale in zip(parts["bodies"], scales):
            body.set_facecolor(SCALE_COLOURS.get(float(scale), "0.5"))
            body.set_edgecolor("none")
            body.set_alpha(0.38)
        for i, (scale, vals) in enumerate(zip(scales, values)):
            colour = SCALE_COLOURS.get(float(scale), "0.5")
            jitter = np.linspace(-0.18, 0.18, len(vals))
            ax_violin.scatter(
                np.full(len(vals), i) + jitter, vals, s=7,
                color=colour, alpha=0.40, edgecolor="none")
            q16, q50, q84 = np.percentile(vals, [16.0, 50.0, 84.0])
            ax_violin.errorbar(
                i, q50, yerr=[[q50 - q16], [q84 - q50]],
                fmt="o", color="black", ms=3.7, capsize=2.4, zorder=5)
        ax_violin.set_xticks(np.arange(len(scales)))
        ax_violin.set_xticklabels([f"{scale:g}" for scale in scales])
        ax_violin.set_xlabel(
            r"Field smoothing scale [$h^{-1}\mathrm{Mpc}$]")
        ax_violin.set_ylabel(H0_LABEL)
        ax_violin.set_title("Field-median distribution", loc="left")

        for scale in scales:
            group = grouped[scale]
            samples = stacked_samples(group)
            summary = h0_summary(samples)
            colour = SCALE_COLOURS.get(float(scale), "0.5")
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
            1, len(scales), figsize=(9.2, 2.55), sharey=True,
            constrained_layout=True)
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
            colour = SCALE_COLOURS.get(float(scale), "0.3")
            ax.errorbar(
                x, h0, xerr=xerr, yerr=yerr, fmt="o", ms=3.1,
                color=colour, ecolor=colour, elinewidth=0.45,
                capsize=1.0, alpha=0.72, zorder=2)
            ax.set_title(scale_label(scale), loc="left")
            ax.set_xlabel(r"harmonic $\ln Z$")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.text(
                0.03, 0.97,
                correlation_summary(x, h0),
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


def plot_all_variant_distributions(rows, out_pdf):
    categories = category_groups(rows)
    all_h0 = np.concatenate([
        np.asarray([row["H0_q50"] for row in group], dtype=float)
        for _, group in categories
    ])
    all_samples = np.concatenate([stacked_samples(group)
                                  for _, group in categories])
    x_min = min(np.percentile(all_samples, 0.3), np.min(all_h0))
    x_max = max(np.percentile(all_samples, 99.7), np.max(all_h0))
    pad = 0.08 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 800)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 2, figsize=(7.6, 3.25), constrained_layout=True)
        ax_violin, ax_kde = axes

        values = [
            np.asarray([row["H0_q50"] for row in group], dtype=float)
            for _, group in categories
        ]
        positions = np.arange(len(categories))
        parts = ax_violin.violinplot(
            values, positions=positions, widths=0.76,
            showmeans=False, showextrema=False, showmedians=False)
        for body, (template, _) in zip(parts["bodies"], categories):
            body.set_facecolor(row_colour(template))
            body.set_edgecolor("none")
            body.set_alpha(0.35)
        for i, ((template, _), vals) in enumerate(zip(categories, values)):
            colour = row_colour(template)
            jitter = np.linspace(-0.18, 0.18, len(vals))
            ax_violin.scatter(
                np.full(len(vals), i) + jitter, vals, s=6.5,
                color=colour, alpha=0.38, edgecolor="none")
            q16, q50, q84 = np.percentile(vals, [16.0, 50.0, 84.0])
            ax_violin.errorbar(
                i, q50, yerr=[[q50 - q16], [q84 - q50]],
                fmt="o", color="black", ms=3.5, capsize=2.4, zorder=5)
        ax_violin.set_xticks(positions)
        ax_violin.set_xticklabels(
            [tick_category_label(template) for template, _ in categories])
        ax_violin.set_ylabel(H0_LABEL)
        ax_violin.set_title("Field-median distribution", loc="left")

        for template, group in categories:
            samples = stacked_samples(group)
            summary = h0_summary(samples)
            colour = row_colour(template)
            ax_kde.plot(
                x_grid, kde_on_grid(samples, x_grid), color=colour, lw=1.15,
                label=(
                    rf"{plot_category_label(template)} "
                    rf"(${summary['H0_mean']:.2f}\pm"
                    rf"{summary['H0_std']:.2f}$)"
                ))
            ax_kde.axvline(
                summary["H0_mean"], color=colour, lw=0.60, alpha=0.42)
        ax_kde.set_xlabel(H0_LABEL)
        ax_kde.set_ylabel("Density")
        ax_kde.set_title("Stacked posteriors", loc="left")
        ax_kde.set_ylim(bottom=0)
        ax_kde.legend(loc="upper right", frameon=False, handlelength=1.7)
        return save_pdf_png(fig, out_pdf)


def plot_all_h0_vs_harmonic_lnz(rows, out_pdf):
    categories = category_groups(rows)
    ncols = min(3, len(categories))
    nrows = ceil(len(categories) / ncols)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.2 * ncols, 2.45 * nrows),
            sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes).reshape(-1)
        for ax, (template, group) in zip(axes, categories):
            x = np.asarray([row["lnZ_harmonic"] for row in group],
                           dtype=float)
            xerr = np.asarray([
                abs(row["err_lnZ_harmonic"]) for row in group], dtype=float)
            h0 = np.asarray([row["H0_q50"] for row in group], dtype=float)
            h0_lo = np.asarray([row["H0_q16"] for row in group], dtype=float)
            h0_hi = np.asarray([row["H0_q84"] for row in group], dtype=float)
            yerr = np.vstack([h0 - h0_lo, h0_hi - h0])
            colour = row_colour(template)
            ax.errorbar(
                x, h0, xerr=xerr, yerr=yerr, fmt="o", ms=3.0,
                color=colour, ecolor=colour, elinewidth=0.45,
                capsize=1.0, alpha=0.72, zorder=2)
            ax.set_title(plot_category_label(template), loc="left")
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
        for i, ax in enumerate(axes):
            if i % ncols == 0:
                ax.set_ylabel(H0_LABEL)
            else:
                ax.tick_params(labelleft=False)
        for ax in axes[len(categories):]:
            ax.set_visible(False)
        return save_pdf_png(fig, out_pdf)


def plot_uniform_control_shift(rows, out_pdf):
    smoothed = smoothed_result_rows(rows)
    if not smoothed:
        return []
    reference_scale = min(row["smooth_R"] for row in smoothed)
    reference = {
        row["field"]: row for row in smoothed
        if row["smooth_R"] == reference_scale
    }
    uniform_cola = sorted([
        row for row in rows
        if row["which_bias"] == "uniform"
        and row["reconstruction_label"] == "COLA/CIC"
        and row["field"] in reference
    ], key=lambda row: row["field"])
    if not uniform_cola:
        return []

    fields = np.asarray([row["field"] for row in uniform_cola], dtype=float)
    baseline = np.asarray([
        reference[row["field"]]["H0_q50"] for row in uniform_cola],
        dtype=float)
    uniform = np.asarray([row["H0_q50"] for row in uniform_cola],
                         dtype=float)
    delta = uniform - baseline
    cmap = field_cmap()
    norm = field_norm(rows)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 2, figsize=(7.0, 3.05), constrained_layout=True)
        ax_pair, ax_delta = axes
        colours = cmap(norm(fields))
        ax_pair.scatter(
            baseline, uniform, c=colours, s=13, alpha=0.76,
            edgecolor="none")
        lim_min = min(np.min(baseline), np.min(uniform))
        lim_max = max(np.max(baseline), np.max(uniform))
        pad = 0.08 * (lim_max - lim_min)
        limits = (lim_min - pad, lim_max + pad)
        ax_pair.plot(limits, limits, color="0.35", lw=0.75, ls="--")
        ax_pair.set_xlim(limits)
        ax_pair.set_ylim(limits)
        ax_pair.set_xlabel(rf"{scale_label(reference_scale)} "
                           r"double-power-law " + H0_LABEL)
        ax_pair.set_ylabel("uniform COLA/CIC " + H0_LABEL)
        ax_pair.set_title("Matched realisations", loc="left")

        ax_delta.axhline(0.0, color="0.35", lw=0.75, ls="--")
        mean_delta = float(np.mean(delta))
        std_delta = float(np.std(delta, ddof=1))
        ax_delta.axhline(mean_delta, color="black", lw=1.0)
        ax_delta.fill_between(
            [np.min(fields), np.max(fields)],
            mean_delta - std_delta, mean_delta + std_delta,
            color="black", alpha=0.08, lw=0)
        ax_delta.scatter(
            fields, delta, c=colours, s=12, alpha=0.78, edgecolor="none")
        ax_delta.set_xlabel("Manticore field")
        ax_delta.set_ylabel(rf"$\Delta H_0$ vs {scale_label(reference_scale)}")
        ax_delta.set_title(
            rf"$\langle\Delta H_0\rangle={mean_delta:+.2f}$, "
            rf"$\sigma={std_delta:.2f}$", loc="left")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, pad=0.015, fraction=0.050)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def write_rows_csv(rows, path):
    fieldnames = [
        "status", "task", "field", "smooth_R", "reference_smooth_R",
        "which_bias", "mas", "reconstruction", "reconstruction_label",
        "family", "reference_which_bias", "reference_reconstruction_label",
        "n_H0", "H0_mean", "H0_std", "H0_q16", "H0_q50",
        "H0_q84", "reference_H0_q50", "delta_H0_q50_vs_reference",
        "delta_H0_mean_vs_reference", "lnZ_harmonic",
        "err_lnZ_harmonic", "reference_lnZ_harmonic",
        "delta_lnZ_harmonic_vs_reference", "lnZ_laplace",
        "err_lnZ_laplace", "BIC", "lnZ_bic", "source", "config",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_summary(rows, missing, path):
    smoothed = smoothed_result_rows(rows)
    grouped = grouped_by_scale(smoothed)
    ref = min(grouped)
    uniform_groups = [
        (template, group) for template, group in category_groups(rows)
        if template["which_bias"] == "uniform"
    ]
    lines = [
        "# CH0 Single-Field Smoothing Diagnostics",
        "",
        f"Complete outputs: {len(rows)}.",
        f"Missing outputs: {len(missing)}.",
        "Result families: "
        f"{len(smoothed)} double-power-law COLA/CIC smoothing runs; "
        f"{sum(len(group) for _, group in uniform_groups)} "
        "uniform-bias/density-field control runs.",
        "Reference model for matched shifts: "
        f"double-power-law COLA/CIC R={ref:g} Mpc/h.",
        "",
        "## Double-power-law COLA/CIC smoothing sequence",
        "",
        "| smoothing R [Mpc/h] | fields | mean field H0 | std field H0 | "
        "median field H0 | stacked H0 mean | stacked H0 std | "
        "mean delta H0 | std delta H0 | median harmonic lnZ | "
        "best harmonic lnZ field |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |",
    ]
    for scale, group in grouped.items():
        h0 = np.asarray([row["H0_q50"] for row in group], dtype=float)
        deltas = np.asarray([
            row["delta_H0_q50_vs_reference"] for row in group],
            dtype=float)
        samples = stacked_samples(group)
        summary = h0_summary(samples)
        best = max(group, key=lambda row: row["lnZ_harmonic"])
        lnz = np.asarray([row["lnZ_harmonic"] for row in group], dtype=float)
        lines.append(
            f"| {scale:g} | {len(group)} | {np.mean(h0):.3f} | "
            f"{np.std(h0, ddof=1):.3f} | {np.median(h0):.3f} | "
            f"{summary['H0_mean']:.3f} | {summary['H0_std']:.3f} | "
            f"{np.mean(deltas):+.3f} | {np.std(deltas, ddof=1):.3f} | "
            f"{np.median(lnz):.3f} | {best['field']} |"
        )
    if uniform_groups:
        lines.extend([
            "",
            "## Uniform-bias/density-field controls",
            "",
            "| control | fields | mean field H0 | std field H0 | "
            "median field H0 | stacked H0 mean | stacked H0 std | "
            "mean delta H0 | std delta H0 | median harmonic lnZ | "
            "best harmonic lnZ field |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: |",
        ])
        for template, group in uniform_groups:
            h0 = np.asarray([row["H0_q50"] for row in group], dtype=float)
            samples = stacked_samples(group)
            summary = h0_summary(samples)
            best = max(group, key=lambda row: row["lnZ_harmonic"])
            lnz = np.asarray([row["lnZ_harmonic"] for row in group],
                             dtype=float)
            deltas = np.asarray([
                row["delta_H0_q50_vs_reference"] for row in group
                if "delta_H0_q50_vs_reference" in row
            ], dtype=float)
            if deltas.size:
                mean_delta = f"{np.mean(deltas):+.3f}"
                std_delta = f"{np.std(deltas, ddof=1):.3f}"
            else:
                mean_delta = "--"
                std_delta = "--"
            lines.append(
                f"| {template['reconstruction_label']} | {len(group)} | "
                f"{np.mean(h0):.3f} | {np.std(h0, ddof=1):.3f} | "
                f"{np.median(h0):.3f} | {summary['H0_mean']:.3f} | "
                f"{summary['H0_std']:.3f} | {mean_delta} | {std_delta} | "
                f"{np.median(lnz):.3f} | {best['field']} |"
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
    rows, missing = load_rows(args.task_file, args.allow_missing)
    smoothed = smoothed_result_rows(rows)

    csv_path = args.output_dir / "ch0_single_smoothed_summary.csv"
    txt_path = args.output_dir / "ch0_single_smoothed_summary.txt"
    impact_pdf = args.output_dir / "ch0_single_smoothed_h0_matched_fields.pdf"
    broad_pdf = args.output_dir / "ch0_single_smoothed_h0_distributions.pdf"
    evidence_pdf = args.output_dir / "ch0_single_smoothed_h0_vs_harmonic_lnz.pdf"
    all_broad_pdf = (
        args.output_dir / "ch0_single_smoothed_all_h0_distributions.pdf")
    all_evidence_pdf = (
        args.output_dir / "ch0_single_smoothed_all_h0_vs_harmonic_lnz.pdf")
    uniform_pdf = (
        args.output_dir / "ch0_single_smoothed_uniform_control_shift.pdf")

    write_rows_csv(rows, csv_path)
    write_summary(rows, missing, txt_path)
    written = [
        csv_path,
        txt_path,
        *plot_matched_field_impact(smoothed, impact_pdf),
        *plot_broad_shift(smoothed, broad_pdf),
        *plot_h0_vs_harmonic_lnz(smoothed, evidence_pdf),
        *plot_all_variant_distributions(rows, all_broad_pdf),
        *plot_all_h0_vs_harmonic_lnz(rows, all_evidence_pdf),
        *plot_uniform_control_shift(rows, uniform_pdf),
    ]

    for path in written:
        print(f"Wrote {path}")
    print(f"Complete outputs: {len(rows)}; missing: {len(missing)}.")


if __name__ == "__main__":
    main()
