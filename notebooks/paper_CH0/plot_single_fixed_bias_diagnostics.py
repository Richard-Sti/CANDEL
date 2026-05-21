#!/usr/bin/env python
"""Plot CH0 fixed-bias single-field diagnostics."""

from argparse import ArgumentParser
import csv
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: F401,E402
import tomllib  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from scipy.stats import gaussian_kde, pearsonr, spearmanr  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = ROOT / "scripts" / "runs" / "tasks_CH0_single_fixed_bias.txt"
RESULTS = ROOT / "results" / "CH0_paper" / "single_fields_fixed_bias"
DEFAULT_OUTDIR = RESULTS / "plots"
FIELD_RE = re.compile(r"_field(\d+)_")
FIGURE_DPI = 500
H0_LABEL = (
    r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"
)
FIELD_COLOURS = ["#ef476f", "#473198", "#a8c256", "#5adbff", "#fe9000"]
DOUBLE_POWERLAW_BIAS_PARAMS = (
    "alpha_low", "alpha_high", "log_rho_t", "log_rho_width")


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-file", type=Path, default=TASK_FILE,
        help="Task file listing fixed-bias generated configs.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTDIR,
        help="Directory for PDF, PNG, and CSV outputs.")
    parser.add_argument(
        "--field-stat", choices=("median", "mean"), default="median",
        help="Per-field H0 statistic used in the summary histogram.")
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Skip missing HDF5 outputs instead of failing.")
    return parser.parse_args()


def repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def task_config_paths(task_file):
    task_file = repo_path(task_file)
    configs = []
    with task_file.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Could not parse task line: {line!r}")
            configs.append(repo_path(parts[1]))
    if not configs:
        raise ValueError(f"No configs found in `{task_file}`.")
    return configs


def get_nested(mapping, keys):
    value = mapping
    for key in keys:
        value = value[key]
    return value


def field_index_from_name(path):
    match = FIELD_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse field index from `{path}`.")
    return int(match.group(1))


def delta_prior_value(config, name):
    prior = get_nested(config, ("model", "priors", name))
    if prior.get("dist") != "delta":
        raise ValueError(
            f"Expected fixed-bias prior `{name}` to be a delta prior.")
    return float(prior["value"])


def output_from_config(config_path):
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    output = repo_path(get_nested(config, ("io", "fname_output")))
    field = int(get_nested(config, ("io", "field_indices")))
    mas = get_nested(
        config, ("io", "reconstruction_main", "ManticoreLocalCOLA",
                 "which_MAS"))
    which_bias = get_nested(config, ("model", "which_bias"))
    bias_params = {
        name: delta_prior_value(config, name)
        for name in DOUBLE_POWERLAW_BIAS_PARAMS
    }
    return {
        "config": str(config_path),
        "path": output,
        "field": field,
        "mas": mas,
        "which_bias": which_bias,
        **bias_params,
    }


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


def selection_integral_total(log_s, n_hosts):
    if log_s.ndim == 1:
        return -float(n_hosts) * log_s
    if log_s.ndim == 2:
        return -np.sum(log_s, axis=1)
    raise ValueError(f"Unexpected log-selection-integral shape {log_s.shape}.")


def read_field_row(spec):
    path = spec["path"]
    with h5py.File(path, "r") as handle:
        h0 = finite_samples(handle, "H0", path)
        ll_total = np.asarray(
            handle["auxiliary/log_likelihood_total"], dtype=float)
        observed = np.asarray(
            handle["auxiliary/log_observed_selection_per_galaxy"],
            dtype=float)
        log_s = np.asarray(
            handle["auxiliary/log_selection_integral"], dtype=float)
        if observed.ndim != 2:
            raise ValueError(
                f"Unexpected observed-selection shape {observed.shape} in "
                f"`{path}`.")

        n_hosts = observed.shape[1]
        observed_total = np.sum(observed, axis=1)
        selection_norm = selection_integral_total(log_s, n_hosts)
        raw_total = ll_total - observed_total - selection_norm
        bic = read_scalar(handle, "gof/BIC", path)

        return {
            "field": int(spec["field"]),
            "mas": spec["mas"],
            "which_bias": spec["which_bias"],
            "alpha_low": spec["alpha_low"],
            "alpha_high": spec["alpha_high"],
            "log_rho_t": spec["log_rho_t"],
            "log_rho_width": spec["log_rho_width"],
            "source": str(path),
            "config": spec["config"],
            "samples": h0,
            "n_hosts": int(n_hosts),
            **h0_summary(h0),
            "lnZ_harmonic": read_scalar(handle, "gof/lnZ_harmonic", path),
            "err_lnZ_harmonic": read_scalar(
                handle, "gof/err_lnZ_harmonic", path),
            "lnZ_laplace": read_scalar(handle, "gof/lnZ_laplace", path),
            "err_lnZ_laplace": read_scalar(
                handle, "gof/err_lnZ_laplace", path),
            "BIC": bic,
            "lnZ_bic": -0.5 * bic,
            "raw_total_mean": float(np.mean(raw_total)),
            "raw_total_std": float(np.std(raw_total, ddof=1)),
            "observed_selection_total_mean": float(np.mean(observed_total)),
            "selection_normalisation_mean": float(np.mean(selection_norm)),
            "selection_normalisation_std": float(
                np.std(selection_norm, ddof=1)),
            "log_likelihood_total_mean": float(np.mean(ll_total)),
            "log_likelihood_total_std": float(np.std(ll_total, ddof=1)),
        }


def load_rows(task_file, allow_missing):
    specs = [output_from_config(path) for path in task_config_paths(task_file)]
    specs = sorted(specs, key=lambda spec: spec["field"])
    rows = []
    missing = []
    for spec in specs:
        if not spec["path"].is_file():
            missing.append(spec)
            continue
        rows.append(read_field_row(spec))

    if missing and not allow_missing:
        preview = "\n".join(str(spec["path"]) for spec in missing[:8])
        raise FileNotFoundError(
            f"{len(missing)} HDF5 outputs are missing. First missing:\n"
            f"{preview}\nPass --allow-missing to skip them.")
    if not rows:
        raise ValueError("No usable fixed-bias HDF5 outputs were found.")
    return rows, missing


def stacked_row(rows):
    samples = np.concatenate([row["samples"] for row in rows])
    return {
        "field": "stacked",
        "mas": rows[0]["mas"],
        "source": "",
        "config": "",
        **h0_summary(samples),
        "samples": samples,
    }


def set_paper_rc():
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.5,
    })


def field_cmap():
    return LinearSegmentedColormap.from_list(
        "ch0_fixed_bias_manticore_fields", FIELD_COLOURS)


def field_norm(rows):
    fields = np.asarray([row["field"] for row in rows], dtype=float)
    vmin = min(0.0, float(np.min(fields)))
    vmax = float(np.max(fields))
    if vmin == vmax:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def save_pdf_png(fig, out_pdf):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_pdf, out_png


def p_label(value):
    if value < 1e-3:
        return r"<10^{-3}"
    return rf"={value:.2f}"


def plot_h0_vs_evidence(rows, out_pdf):
    fields = np.asarray([row["field"] for row in rows], dtype=float)
    h0 = np.asarray([row["H0_q50"] for row in rows], dtype=float)
    h0_lo = np.asarray([row["H0_q16"] for row in rows], dtype=float)
    h0_hi = np.asarray([row["H0_q84"] for row in rows], dtype=float)
    lnz = np.asarray([row["lnZ_harmonic"] for row in rows], dtype=float)
    lnz_err = np.asarray(
        [abs(row["err_lnZ_harmonic"]) for row in rows], dtype=float)
    yerr = np.vstack([h0 - h0_lo, h0_hi - h0])
    pearson_r, pearson_p = pearsonr(lnz, h0)
    spearman_r, spearman_p = spearmanr(lnz, h0)

    with plt.style.context("science"):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(3.55, 3.0), constrained_layout=True)
        sc = ax.scatter(
            lnz, h0, c=fields, cmap=field_cmap(), norm=field_norm(rows),
            s=30, edgecolor="0.15", linewidth=0.3, zorder=3)
        ax.errorbar(
            lnz, h0, xerr=lnz_err, yerr=yerr, fmt="none", ecolor="0.45",
            elinewidth=0.45, capsize=1.2, alpha=0.65, zorder=2)
        ax.set_xlabel(r"harmonic $\ln Z$")
        ax.set_ylabel(H0_LABEL)
        ax.set_title("Fixed-bias single fields", loc="left")
        ax.text(
            0.03, 0.97,
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
        cbar = fig.colorbar(sc, ax=ax, pad=0.018, fraction=0.065)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_selection_vs_raw(rows, out_pdf):
    fields = np.asarray([row["field"] for row in rows], dtype=float)
    raw = np.asarray([row["raw_total_mean"] for row in rows], dtype=float)
    selection = np.asarray([
        row["selection_normalisation_mean"] for row in rows], dtype=float)
    total = np.asarray([
        row["log_likelihood_total_mean"] for row in rows], dtype=float)

    with plt.style.context("science"):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(3.65, 3.05), constrained_layout=True)
        sc = ax.scatter(
            raw, selection, c=fields, cmap=field_cmap(), norm=field_norm(rows),
            s=32, edgecolor="0.15", linewidth=0.3, alpha=0.9, zorder=3)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xgrid = np.linspace(xlim[0], xlim[1], 200)
        for level in np.percentile(total, (20.0, 50.0, 80.0)):
            ax.plot(
                xgrid, level - xgrid, color="0.55", lw=0.65, ls="--",
                alpha=0.55, zorder=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.text(
            0.03, 0.04, "dashed: constant total likelihood",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.6, color="0.35")
        ax.set_xlabel("raw total likelihood")
        ax.set_ylabel(r"selection normalisation, $-\log S$")
        ax.set_title("Likelihood decomposition", loc="left")
        cbar = fig.colorbar(sc, ax=ax, pad=0.018, fraction=0.065)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def double_powerlaw_log_bias(log_rho, alpha_low, alpha_high, log_rho_t,
                             log_rho_width):
    log_x = log_rho - log_rho_t
    z = log_x / log_rho_width
    return (
        alpha_low * log_x
        + (alpha_high - alpha_low) * log_rho_width * np.logaddexp(0.0, z)
    )


def plot_galaxy_bias(rows, out_pdf):
    params = {
        name: rows[0][name]
        for name in DOUBLE_POWERLAW_BIAS_PARAMS
    }
    which_bias = rows[0]["which_bias"]
    for row in rows[1:]:
        if row["which_bias"] != which_bias:
            raise ValueError("Cannot plot one galaxy-bias curve for mixed "
                             "bias models.")
        for name, value in params.items():
            if not np.isclose(row[name], value, rtol=0.0, atol=1e-12):
                raise ValueError("Cannot plot one galaxy-bias curve for mixed "
                                 "fixed-bias parameters.")
    if which_bias != "double_powerlaw":
        raise ValueError(
            f"Galaxy-bias plot only supports double_powerlaw, got "
            f"`{which_bias}`.")

    log_rho = np.linspace(-3.0, 3.0, 800)
    rho = np.exp(log_rho)
    log_bias = double_powerlaw_log_bias(log_rho, **params)
    log_bias_at_mean = double_powerlaw_log_bias(np.asarray([0.0]), **params)[0]
    rel_bias = np.exp(log_bias - log_bias_at_mean)
    transition_rho = np.exp(params["log_rho_t"])

    with plt.style.context("science"):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(3.55, 3.0), constrained_layout=True)
        ax.plot(rho, rel_bias, color="#473198", lw=1.4)
        ax.axvline(
            transition_rho, color="0.35", lw=0.8, ls="--",
            label=rf"$\rho_t/\bar{{\rho}}={transition_rho:.2f}$")
        ax.axhline(1.0, color="0.55", lw=0.65, ls=":", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\rho/\bar{\rho}$")
        ax.set_ylabel(r"$b_{\rm gal}(\rho) / b_{\rm gal}(\bar{\rho})$")
        ax.set_title("Fixed galaxy bias", loc="left")
        ax.text(
            0.04, 0.96,
            (
                rf"$\alpha_\mathrm{{low}}={params['alpha_low']:.3f}$" "\n"
                rf"$\alpha_\mathrm{{high}}={params['alpha_high']:.3f}$" "\n"
                rf"$\log\rho_t={params['log_rho_t']:.3f}$" "\n"
                rf"$w={params['log_rho_width']:.3f}$"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
            },
        )
        ax.legend(loc="lower right", frameon=False, handlelength=1.7)
        return save_pdf_png(fig, out_pdf)


def kde_on_grid(samples, x_grid, bw=1.35):
    kde = gaussian_kde(samples)
    kde.set_bandwidth(kde.factor * bw)
    return kde(x_grid)


def plot_h0_histogram(rows, stack, field_stat, out_pdf):
    stat_key = "H0_q50" if field_stat == "median" else "H0_mean"
    field_values = np.asarray([row[stat_key] for row in rows], dtype=float)
    stacked_samples = stack["samples"]
    x_min = min(np.percentile(stacked_samples, 0.3), np.min(field_values))
    x_max = max(np.percentile(stacked_samples, 99.7), np.max(field_values))
    pad = 0.08 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 700)
    cmap = field_cmap()
    norm = field_norm(rows)

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 2, figsize=(7.1, 3.1), sharex=True, constrained_layout=True,
            gridspec_kw={"wspace": 0.10})
        ax_hist, ax_kde = axes
        ax_hist.hist(
            field_values,
            bins=min(12, max(6, int(np.sqrt(field_values.size)) + 2)),
            density=True,
            color="#ee35d5",
            alpha=0.35,
            edgecolor="white",
            linewidth=0.6,
            label=rf"Field posterior {field_stat}s",
        )
        ax_hist.axvline(
            np.median(field_values), color="#87193d", lw=0.95, ls="--",
            label=rf"Median field {field_stat} "
            rf"(${np.median(field_values):.2f}$)")
        ax_hist.axvline(
            stack["H0_mean"], color="black", lw=0.9, ls=":",
            label=rf"stacked mean (${stack['H0_mean']:.2f}$)")
        ax_hist.axvspan(
            stack["H0_mean"] - stack["H0_std"],
            stack["H0_mean"] + stack["H0_std"],
            color="black", alpha=0.10, lw=0)
        ax_hist.set_title("Field summaries", loc="left")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(loc="upper right", frameon=False, handlelength=1.8)

        for row in rows:
            ax_kde.plot(
                x_grid, kde_on_grid(row["samples"], x_grid),
                color=cmap(norm(row["field"])), alpha=0.72, linewidth=0.72)
        ax_kde.plot(
            x_grid, kde_on_grid(stacked_samples, x_grid),
            color="black", lw=1.35,
            label=rf"Stacked posterior "
            rf"(${stack['H0_mean']:.2f}\pm{stack['H0_std']:.2f}$)")
        ax_kde.axvline(stack["H0_mean"], color="black", lw=0.9, ls=":")
        ax_kde.axvspan(
            stack["H0_mean"] - stack["H0_std"],
            stack["H0_mean"] + stack["H0_std"],
            color="black", alpha=0.11, lw=0)
        ax_kde.plot(
            [], [], color=cmap(norm(np.median(
                [row["field"] for row in rows]))),
            alpha=0.55, lw=0.8, label="Individual field posteriors")
        ax_kde.set_title("Manticore realisations", loc="left")
        ax_kde.legend(loc="upper right", frameon=False, handlelength=1.8)

        for ax in axes:
            ax.set_xlim(x_grid[0], x_grid[-1])
            ax.set_ylim(bottom=0)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_kde, pad=0.015, fraction=0.055)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        fig.supxlabel(H0_LABEL)
        return save_pdf_png(fig, out_pdf)


def write_csv(rows, stack, path):
    fieldnames = [
        "field", "mas", "which_bias", "alpha_low", "alpha_high",
        "log_rho_t", "log_rho_width", "n_H0", "H0_mean", "H0_std",
        "H0_q16", "H0_q50", "H0_q84", "lnZ_harmonic",
        "err_lnZ_harmonic", "lnZ_laplace", "err_lnZ_laplace", "BIC",
        "lnZ_bic", "raw_total_mean", "raw_total_std",
        "selection_normalisation_mean", "selection_normalisation_std",
        "observed_selection_total_mean", "log_likelihood_total_mean",
        "log_likelihood_total_std", "n_hosts", "source", "config",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
        writer.writerow({name: stack.get(name, "") for name in fieldnames})


def write_summary(rows, missing, stack, path):
    best = max(rows, key=lambda row: row["lnZ_harmonic"])
    h0_values = np.asarray([row["H0_q50"] for row in rows], dtype=float)
    lines = [
        "# CH0 Fixed-Bias Single-Field Diagnostics",
        "",
        f"Usable fields: {len(rows)}.",
        f"Missing outputs: {len(missing)}.",
        f"MAS: {rows[0]['mas']}.",
        f"Galaxy bias: {rows[0]['which_bias']}.",
        "",
        "| quantity | value |",
        "| --- | ---: |",
        f"| best harmonic lnZ field | {best['field']} |",
        f"| best harmonic lnZ | {best['lnZ_harmonic']:.3f} |",
        f"| best-field H0 mean | {best['H0_mean']:.3f} |",
        f"| stacked H0 mean | {stack['H0_mean']:.3f} |",
        f"| stacked H0 std | {stack['H0_std']:.3f} |",
        f"| median field H0 median | {np.median(h0_values):.3f} |",
        f"| alpha_low | {rows[0]['alpha_low']:.3f} |",
        f"| alpha_high | {rows[0]['alpha_high']:.3f} |",
        f"| log_rho_t | {rows[0]['log_rho_t']:.3f} |",
        f"| log_rho_width | {rows[0]['log_rho_width']:.3f} |",
    ]
    if missing:
        lines.extend(["", "## Missing", ""])
        for spec in missing:
            lines.append(f"- field {spec['field']}: `{spec['path']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, missing = load_rows(args.task_file, args.allow_missing)
    stack = stacked_row(rows)

    csv_path = args.output_dir / "ch0_single_fixed_bias_summary.csv"
    summary_path = args.output_dir / "ch0_single_fixed_bias_summary.txt"
    h0_evidence_pdf = (
        args.output_dir / "ch0_single_fixed_bias_h0_vs_evidence.pdf")
    selection_raw_pdf = (
        args.output_dir / "ch0_single_fixed_bias_selection_vs_raw.pdf")
    h0_hist_pdf = args.output_dir / "ch0_single_fixed_bias_h0_histogram.pdf"
    galaxy_bias_pdf = args.output_dir / "ch0_single_fixed_bias_galaxy_bias.pdf"

    write_csv(rows, stack, csv_path)
    write_summary(rows, missing, stack, summary_path)
    written = [
        csv_path,
        summary_path,
        *plot_h0_vs_evidence(rows, h0_evidence_pdf),
        *plot_selection_vs_raw(rows, selection_raw_pdf),
        *plot_h0_histogram(rows, stack, args.field_stat, h0_hist_pdf),
        *plot_galaxy_bias(rows, galaxy_bias_pdf),
    ]

    for path in written:
        print(f"Wrote {path}")
    print(
        "stacked: "
        f"H0={stack['H0_mean']:.3f} +- {stack['H0_std']:.3f}; "
        f"q16/q50/q84={stack['H0_q16']:.3f}/"
        f"{stack['H0_q50']:.3f}/{stack['H0_q84']:.3f}"
    )


if __name__ == "__main__":
    main()
