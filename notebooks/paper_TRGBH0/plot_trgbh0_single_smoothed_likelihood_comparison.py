#!/usr/bin/env python
"""Compare Gaussian and Student-t TRGBH0 smoothed single-field runs."""

from argparse import ArgumentParser
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
from scipy.stats import gaussian_kde  # noqa: E402

from trgbh0_plot_style import TRGBH0_COLOURS, trgbh0_cmap  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = ROOT / "scripts" / "runs" / "tasks_TRGBH0_single_smoothed.txt"
DEFAULT_OUTDIR = (
    Path(__file__).resolve().parent / "output"
    / "trgbh0_single_smoothed_likelihood_comparison_R4"
)
FIGURE_DPI = 500
MAX_KDE_SAMPLES = 40_000
LIKELIHOOD_ORDER = ("gaussian", "student_t")
LIKELIHOOD_LABELS = {
    "gaussian": "Gaussian",
    "student_t": r"Student-$t$",
}
LIKELIHOOD_COLOURS = {
    "gaussian": TRGBH0_COLOURS[0],
    "student_t": TRGBH0_COLOURS[1],
}
H0_LABEL = (
    r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"
)
PARAMETERS = (
    "H0",
    "M_TRGB",
    "mu_LMC",
    "mu_N4258",
    "sigma_int",
    "sigma_v",
    "nu_cz",
    "mag_lim_TRGB",
    "mag_lim_TRGB_width",
    "alpha_low",
    "alpha_high",
    "log_rho_t",
    "log_rho_width",
)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-file", type=Path, default=TASK_FILE,
        help="Task file listing TRGBH0 smoothed single-field configs.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTDIR,
        help="Directory for plots and summaries.")
    parser.add_argument(
        "--smooth-R", type=float, default=4.0,
        help="Density-field smoothing scale in Mpc/h to compare.")
    parser.add_argument(
        "--mas", default="PCS",
        help="COLA mass-assignment scheme to compare.")
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Skip missing HDF5 outputs instead of failing.")
    return parser.parse_args()


def repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def get_nested(mapping, keys, default=None):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def task_specs(task_file, smooth_R, mas):
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

            reconstruction = get_nested(
                config, ("io", "PV_main", "EDD_TRGB", "reconstruction"))
            if reconstruction != "ManticoreLocalCOLA":
                continue
            config_mas = get_nested(
                config,
                ("io", "reconstruction_main", "ManticoreLocalCOLA",
                 "which_MAS"),
            )
            if config_mas != mas:
                continue
            config_smooth_R = float(get_nested(
                config, ("model", "field_3d_smoothing_scale"), 0.0))
            if not np.isclose(config_smooth_R, smooth_R):
                continue
            likelihood = get_nested(
                config, ("model", "cz_likelihood"), "gaussian")
            if likelihood not in LIKELIHOOD_ORDER:
                continue
            specs.append({
                "task": int(task),
                "field": int(get_nested(config, ("io", "field_indices"))),
                "smooth_R": config_smooth_R,
                "mas": config_mas,
                "cz_likelihood": likelihood,
                "which_bias": get_nested(config, ("model", "which_bias")),
                "which_selection": get_nested(
                    config, ("model", "which_selection"), ""),
                "config": str(config_path),
                "source": str(repo_path(get_nested(
                    config, ("io", "fname_output")))),
            })
    if not specs:
        raise ValueError(
            f"No R={smooth_R:g} {mas} likelihood-comparison specs found.")
    return specs


def finite_samples(handle, name, path):
    dataset = f"samples/{name}"
    if dataset not in handle:
        return None
    samples = np.asarray(handle[dataset], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"`{path}` has no finite `{name}` samples.")
    return samples


def read_scalar(handle, name, default=np.nan):
    if name not in handle:
        return default
    value = float(handle[name][()])
    if not np.isfinite(value):
        return default
    return value


def sample_summary(samples, prefix):
    q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
    return {
        f"{prefix}_mean": float(np.mean(samples)),
        f"{prefix}_std": float(np.std(samples, ddof=1)),
        f"{prefix}_q16": float(q16),
        f"{prefix}_q50": float(q50),
        f"{prefix}_q84": float(q84),
    }


def read_row(spec):
    path = Path(spec["source"])
    if not path.is_file():
        return {**spec, "status": "missing"}
    with h5py.File(path, "r") as handle:
        h0 = finite_samples(handle, "H0", path)
        if h0 is None:
            raise ValueError(f"`{path}` does not contain `samples/H0`.")
        row = {
            **spec,
            "status": "complete",
            "samples_H0": h0,
            "lnZ_harmonic": read_scalar(handle, "gof/lnZ_harmonic"),
            "err_lnZ_harmonic": read_scalar(
                handle, "gof/err_lnZ_harmonic"),
            "lnZ_laplace": read_scalar(handle, "gof/lnZ_laplace"),
            "err_lnZ_laplace": read_scalar(handle, "gof/err_lnZ_laplace"),
            "BIC": read_scalar(handle, "gof/BIC"),
            "AIC": read_scalar(handle, "gof/AIC"),
        }
        for parameter in PARAMETERS:
            samples = finite_samples(handle, parameter, path)
            if samples is not None:
                row.update(sample_summary(samples, parameter))
        return row


def load_rows(task_file, smooth_R, mas, allow_missing):
    specs = task_specs(task_file, smooth_R, mas)
    seen = set()
    duplicate_keys = []
    for spec in specs:
        key = (spec["field"], spec["cz_likelihood"])
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)
    if duplicate_keys:
        raise ValueError(f"Duplicate field/likelihood specs: {duplicate_keys}")

    rows = [read_row(spec) for spec in specs]
    missing = [row for row in rows if row["status"] != "complete"]
    if missing and not allow_missing:
        preview = "\n".join(row["source"] for row in missing[:8])
        raise FileNotFoundError(
            f"{len(missing)} outputs are missing. First missing:\n{preview}")

    by_field = {}
    for row in rows:
        if row["status"] == "complete":
            by_field.setdefault(row["field"], {})[
                row["cz_likelihood"]] = row
    matched = {
        field: values for field, values in sorted(by_field.items())
        if all(likelihood in values for likelihood in LIKELIHOOD_ORDER)
    }
    if not matched:
        raise ValueError("No fields have both Gaussian and Student-t outputs.")
    return rows, matched, missing


def set_paper_rc():
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
    })


def save_pdf_png(fig, out_pdf):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_pdf, out_png


def smooth_token(smooth_R):
    return f"R{smooth_R:g}".replace(".", "p")


def smooth_label(smooth_R):
    return rf"$R={smooth_R:g}\,h^{{-1}}\mathrm{{Mpc}}$"


def paired_rows(matched):
    return [
        (field, matched[field]["gaussian"], matched[field]["student_t"])
        for field in sorted(matched)
    ]


def paired_values(matched, key):
    pairs = paired_rows(matched)
    fields = np.asarray([field for field, _, _ in pairs], dtype=float)
    gaussian = np.asarray([row_g[key] for _, row_g, _ in pairs], dtype=float)
    student = np.asarray([row_t[key] for _, _, row_t in pairs], dtype=float)
    return fields, gaussian, student


def field_norm(matched):
    fields = np.asarray(sorted(matched), dtype=float)
    return Normalize(vmin=min(0.0, float(np.min(fields))),
                     vmax=float(np.max(fields)))


def plot_matched_fields(matched, smooth_R, out_pdf):
    fields = sorted(matched)
    cmap = trgbh0_cmap("trgbh0_likelihood_comparison_fields")
    norm = field_norm(matched)
    positions = np.arange(len(LIKELIHOOD_ORDER), dtype=float)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 1, figsize=(5.0, 4.55), sharex=True,
            constrained_layout=True)
        for ax, key, ylabel in (
            (axes[0], "H0_q50", H0_LABEL),
            (axes[1], "lnZ_harmonic", r"harmonic $\ln Z$"),
        ):
            values_by_field = []
            for field in fields:
                values = np.asarray([
                    matched[field][likelihood][key]
                    for likelihood in LIKELIHOOD_ORDER
                ], dtype=float)
                values_by_field.append(values)
                if not np.all(np.isfinite(values)):
                    continue
                colour = cmap(norm(field))
                ax.plot(positions, values, color=colour, lw=0.7, alpha=0.42)
                ax.scatter(positions, values, color=colour, s=9, alpha=0.72)
            values_by_field = np.asarray(values_by_field, dtype=float)
            means = np.nanmean(values_by_field, axis=0)
            stds = np.nanstd(values_by_field, axis=0, ddof=1)
            ax.errorbar(
                positions, means, yerr=stds, color="black", marker="o",
                lw=1.25, ms=4.0, capsize=2.4,
                label="field mean\n(error bar: field-to-field std)",
                zorder=5)
            ax.set_ylabel(ylabel)
        axes[0].set_title(
            rf"Matched fields; {smooth_label(smooth_R)}", loc="left")
        axes[0].legend(loc="best", frameon=False, handlelength=1.7)
        axes[1].set_xticks(
            positions,
            [LIKELIHOOD_LABELS[likelihood]
             for likelihood in LIKELIHOOD_ORDER],
        )
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, pad=0.015, fraction=0.05)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_deltas(matched, out_pdf):
    fields, h0_g, h0_t = paired_values(matched, "H0_q50")
    _, lnz_g, lnz_t = paired_values(matched, "lnZ_harmonic")
    delta_h0 = h0_t - h0_g
    delta_lnz = lnz_t - lnz_g
    cmap = trgbh0_cmap("trgbh0_likelihood_comparison_deltas")
    norm = field_norm(matched)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 3, figsize=(8.2, 2.7), constrained_layout=True)
        specs = (
            (axes[0], delta_h0, r"$\Delta H_0$ Student-$t$ $-$ Gaussian"),
            (axes[1], delta_lnz, r"$\Delta$ harmonic $\ln Z$"),
        )
        for ax, delta, ylabel in specs:
            finite = np.isfinite(delta)
            ax.axhline(0.0, color="0.35", lw=0.75, ls=":")
            ax.scatter(
                fields[finite], delta[finite], c=fields[finite], cmap=cmap,
                norm=norm, s=18, edgecolor="0.15", linewidth=0.18,
                alpha=0.82)
            median = float(np.nanmedian(delta))
            q16, q84 = np.nanpercentile(delta, [16.0, 84.0])
            ax.axhline(median, color="black", lw=0.9)
            ax.axhspan(q16, q84, color="black", alpha=0.10, lw=0)
            ax.set_xlabel("Manticore field")
            ax.set_ylabel(ylabel)

        finite = np.isfinite(delta_h0) & np.isfinite(delta_lnz)
        sc = axes[2].scatter(
            delta_lnz[finite], delta_h0[finite], c=fields[finite],
            cmap=cmap, norm=norm, s=20, edgecolor="0.15", linewidth=0.18,
            alpha=0.84)
        axes[2].axhline(0.0, color="0.35", lw=0.75, ls=":")
        axes[2].axvline(0.0, color="0.35", lw=0.75, ls=":")
        axes[2].set_xlabel(r"$\Delta$ harmonic $\ln Z$")
        axes[2].set_ylabel(r"$\Delta H_0$")
        axes[0].set_title("Field-by-field likelihood shift", loc="left")
        cbar = fig.colorbar(sc, ax=axes, pad=0.012, fraction=0.035)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def stacked_samples(rows):
    samples = np.concatenate([
        row["samples_H0"] for row in rows if row["status"] == "complete"
    ])
    if samples.size > MAX_KDE_SAMPLES:
        indices = np.linspace(0, samples.size - 1, MAX_KDE_SAMPLES,
                              dtype=int)
        samples = samples[indices]
    return samples


def kde_on_grid(samples, x_grid, bw=1.15):
    kde = gaussian_kde(samples)
    kde.set_bandwidth(kde.factor * bw)
    return kde(x_grid)


def plot_h0_distributions(matched, out_pdf):
    rows_by_likelihood = {
        likelihood: [matched[field][likelihood] for field in sorted(matched)]
        for likelihood in LIKELIHOOD_ORDER
    }
    all_samples = np.concatenate([
        stacked_samples(rows) for rows in rows_by_likelihood.values()
    ])
    x_min, x_max = np.percentile(all_samples, [0.3, 99.7])
    pad = 0.08 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 800)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 2, figsize=(7.1, 3.05), constrained_layout=True)
        ax_violin, ax_kde = axes
        values = [
            np.asarray([row["H0_q50"] for row in rows], dtype=float)
            for rows in rows_by_likelihood.values()
        ]
        positions = np.arange(len(LIKELIHOOD_ORDER), dtype=float)
        parts = ax_violin.violinplot(
            values, positions=positions, widths=0.72, showmeans=False,
            showextrema=False, showmedians=False)
        for body, likelihood in zip(parts["bodies"], LIKELIHOOD_ORDER):
            body.set_facecolor(LIKELIHOOD_COLOURS[likelihood])
            body.set_edgecolor("none")
            body.set_alpha(0.34)
        for index, (likelihood, vals) in enumerate(
                zip(LIKELIHOOD_ORDER, values)):
            colour = LIKELIHOOD_COLOURS[likelihood]
            jitter = np.linspace(-0.16, 0.16, len(vals))
            ax_violin.scatter(
                np.full(len(vals), index) + jitter, vals, s=7,
                color=colour, alpha=0.38, edgecolor="none")
            q16, q50, q84 = np.percentile(vals, [16.0, 50.0, 84.0])
            ax_violin.errorbar(
                index, q50, yerr=[[q50 - q16], [q84 - q50]],
                fmt="o", color="black", ms=3.7, capsize=2.4, zorder=5)
        ax_violin.set_xticks(
            positions,
            [LIKELIHOOD_LABELS[likelihood]
             for likelihood in LIKELIHOOD_ORDER],
        )
        ax_violin.set_ylabel(H0_LABEL)
        ax_violin.set_title("Field-median distribution", loc="left")

        for likelihood, rows in rows_by_likelihood.items():
            samples = stacked_samples(rows)
            summary = sample_summary(samples, "H0")
            colour = LIKELIHOOD_COLOURS[likelihood]
            ax_kde.plot(
                x_grid, kde_on_grid(samples, x_grid), color=colour, lw=1.2,
                label=(
                    rf"{LIKELIHOOD_LABELS[likelihood]} "
                    rf"(${summary['H0_mean']:.2f}\pm"
                    rf"{summary['H0_std']:.2f}$)"
                ))
            ax_kde.axvline(summary["H0_mean"], color=colour, lw=0.7,
                           alpha=0.45)
        ax_kde.set_xlabel(H0_LABEL)
        ax_kde.set_ylabel("Density")
        ax_kde.set_title("Stacked posteriors", loc="left")
        ax_kde.set_ylim(bottom=0)
        ax_kde.legend(loc="upper right", frameon=False, handlelength=1.8)
        return save_pdf_png(fig, out_pdf)


def csv_fieldnames(matched):
    fields = [
        "field", "smooth_R", "mas",
        "delta_H0_q50_student_t_minus_gaussian",
        "delta_H0_mean_student_t_minus_gaussian",
        "delta_lnZ_harmonic_student_t_minus_gaussian",
        "delta_lnZ_laplace_student_t_minus_gaussian",
    ]
    for likelihood in LIKELIHOOD_ORDER:
        prefix = f"{likelihood}"
        fields.extend([
            f"{prefix}_task", f"{prefix}_status", f"{prefix}_source",
            f"{prefix}_config", f"{prefix}_lnZ_harmonic",
            f"{prefix}_err_lnZ_harmonic", f"{prefix}_lnZ_laplace",
            f"{prefix}_err_lnZ_laplace", f"{prefix}_BIC",
            f"{prefix}_AIC",
        ])
        for parameter in PARAMETERS:
            for suffix in ("mean", "std", "q16", "q50", "q84"):
                key = f"{parameter}_{suffix}"
                if any(
                    key in matched[field][likelihood]
                    for field in matched
                ):
                    fields.append(f"{prefix}_{key}")
    return fields


def write_rows_csv(matched, path):
    fieldnames = csv_fieldnames(matched)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for field, gaussian, student in paired_rows(matched):
            row = {
                "field": field,
                "smooth_R": gaussian["smooth_R"],
                "mas": gaussian["mas"],
                "delta_H0_q50_student_t_minus_gaussian": (
                    student["H0_q50"] - gaussian["H0_q50"]),
                "delta_H0_mean_student_t_minus_gaussian": (
                    student["H0_mean"] - gaussian["H0_mean"]),
                "delta_lnZ_harmonic_student_t_minus_gaussian": (
                    student["lnZ_harmonic"] - gaussian["lnZ_harmonic"]),
                "delta_lnZ_laplace_student_t_minus_gaussian": (
                    student["lnZ_laplace"] - gaussian["lnZ_laplace"]),
            }
            for likelihood, source in (
                    ("gaussian", gaussian), ("student_t", student)):
                row.update({
                    f"{likelihood}_task": source["task"],
                    f"{likelihood}_status": source["status"],
                    f"{likelihood}_source": source["source"],
                    f"{likelihood}_config": source["config"],
                    f"{likelihood}_lnZ_harmonic": source["lnZ_harmonic"],
                    f"{likelihood}_err_lnZ_harmonic": (
                        source["err_lnZ_harmonic"]),
                    f"{likelihood}_lnZ_laplace": source["lnZ_laplace"],
                    f"{likelihood}_err_lnZ_laplace": (
                        source["err_lnZ_laplace"]),
                    f"{likelihood}_BIC": source["BIC"],
                    f"{likelihood}_AIC": source["AIC"],
                })
                for parameter in PARAMETERS:
                    for suffix in ("mean", "std", "q16", "q50", "q84"):
                        key = f"{parameter}_{suffix}"
                        row[f"{likelihood}_{key}"] = source.get(key, "")
            writer.writerow({
                name: row.get(name, "") for name in fieldnames
            })


def summary_line(values):
    q16, q50, q84 = np.nanpercentile(values, [16.0, 50.0, 84.0])
    return (
        f"{np.nanmean(values):.3f} | {np.nanstd(values, ddof=1):.3f} | "
        f"{q50:.3f} | {q16:.3f} | {q84:.3f}"
    )


def write_summary(matched, missing, smooth_R, path):
    fields, h0_g, h0_t = paired_values(matched, "H0_q50")
    _, lnz_g, lnz_t = paired_values(matched, "lnZ_harmonic")
    delta_h0 = h0_t - h0_g
    delta_lnz = lnz_t - lnz_g
    finite_lnz = np.isfinite(delta_lnz)
    best = fields[finite_lnz][int(np.nanargmax(delta_lnz[finite_lnz]))]
    worst = fields[finite_lnz][int(np.nanargmin(delta_lnz[finite_lnz]))]
    lines = [
        f"# TRGBH0 {smooth_token(smooth_R)} Likelihood Comparison",
        "",
        "Matched model: double-power-law COLA/PCS, "
        f"{smooth_label(smooth_R)} density smoothing.",
        f"Matched fields: {len(matched)}.",
        f"Missing outputs: {len(missing)}.",
        "",
        "| quantity | mean | std | median | q16 | q84 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Gaussian field H0 | {summary_line(h0_g)} |",
        f"| Student-t field H0 | {summary_line(h0_t)} |",
        f"| Delta H0 | {summary_line(delta_h0)} |",
        f"| Delta harmonic lnZ | {summary_line(delta_lnz)} |",
        "",
        f"Largest positive delta harmonic lnZ field: {int(best)}.",
        f"Largest negative delta harmonic lnZ field: {int(worst)}.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, matched, missing = load_rows(
        args.task_file, args.smooth_R, args.mas, args.allow_missing)

    token = smooth_token(args.smooth_R)
    csv_path = (
        args.output_dir
        / f"trgbh0_single_smoothed_{token}_likelihood_comparison.csv"
    )
    txt_path = (
        args.output_dir
        / f"trgbh0_single_smoothed_{token}_likelihood_comparison_summary.txt"
    )
    matched_pdf = (
        args.output_dir
        / f"trgbh0_single_smoothed_{token}_likelihood_matched_fields.pdf"
    )
    deltas_pdf = (
        args.output_dir
        / f"trgbh0_single_smoothed_{token}_likelihood_deltas.pdf"
    )
    h0_dist_pdf = (
        args.output_dir
        / f"trgbh0_single_smoothed_{token}_likelihood_h0_distributions.pdf"
    )

    write_rows_csv(matched, csv_path)
    write_summary(matched, missing, args.smooth_R, txt_path)
    written = [
        csv_path,
        txt_path,
        *plot_matched_fields(matched, args.smooth_R, matched_pdf),
        *plot_deltas(matched, deltas_pdf),
        *plot_h0_distributions(matched, h0_dist_pdf),
    ]
    for path in written:
        print(f"Wrote {path}")
    print(f"Complete outputs: {sum(row['status'] == 'complete' for row in rows)}.")
    print(f"Matched fields: {len(matched)}; missing: {len(missing)}.")


if __name__ == "__main__":
    main()
