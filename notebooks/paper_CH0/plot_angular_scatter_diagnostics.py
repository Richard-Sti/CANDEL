#!/usr/bin/env python
"""Plot CH0 angular-position-scatter diagnostics."""

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
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = ROOT / "scripts" / "runs" / "tasks_CH0_angular_scatter.txt"
RESULTS = ROOT / "results" / "CH0_paper"
DEFAULT_OUTDIR = RESULTS / "angular_scatter" / "plots"
BASELINE_DIR = RESULTS / "single_fields"
BASELINE_TEMPLATE = (
    "CH0_MAS-CIC_sel-SN_magnitude_ManticoreLocalCOLA_field"
    "{field:02d}_single.hdf5"
)
FIGURE_DPI = 500
H0_LABEL = (
    r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"
)
FIELD_COLOURS = ["#ef476f", "#473198", "#a8c256", "#5adbff", "#fe9000"]
COLOUR_H0 = "#473198"
SCATTER_COLOURS = {
    0.0: "black",
    2.0: "#168039",
    4.0: "#473198",
    8.0: "#d42a29",
    16.0: "#fe9000",
}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-file", type=Path, default=TASK_FILE,
        help="Task file containing CH0 angular-scatter configs.")
    parser.add_argument(
        "--baseline-dir", type=Path, default=BASELINE_DIR,
        help="Directory containing no-scatter COLA CIC field outputs.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTDIR,
        help="Directory for plots and summaries.")
    parser.add_argument(
        "--require-complete", action="store_true",
        help="Fail if any angular-scatter or baseline output is missing.")
    return parser.parse_args()


def repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def get_nested(mapping, keys, default=None):
    value = mapping
    for key in keys:
        if key not in value:
            return default
        value = value[key]
    return value


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
            specs.append({
                "task": int(task),
                "field": int(get_nested(config, ("io", "field_indices"))),
                "scatter_deg": float(get_nested(
                    config, ("io", "angular_position_scatter_deg"))),
                "scatter_seed": int(get_nested(
                    config, ("io", "angular_position_scatter_seed"))),
                "mas": get_nested(
                    config, ("io", "reconstruction_main",
                             "ManticoreLocalCOLA", "which_MAS")),
                "config": str(config_path),
                "source": str(repo_path(get_nested(
                    config, ("io", "fname_output")))),
            })
    if not specs:
        raise ValueError(f"No task configs found in `{task_file}`.")
    return sorted(specs, key=lambda spec: (
        spec["field"], spec["scatter_deg"], spec["task"]))


def finite_samples(handle, name, path):
    samples = np.asarray(handle[f"samples/{name}"], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"`{path}` has no finite `{name}` samples.")
    return samples


def read_scalar(handle, name, path, default=np.nan):
    if name not in handle:
        return default
    value = float(handle[name][()])
    if not np.isfinite(value):
        return default
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


def read_hdf5_row(spec):
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


def baseline_spec(field, baseline_dir):
    path = repo_path(baseline_dir) / BASELINE_TEMPLATE.format(field=field)
    return {
        "task": "",
        "field": int(field),
        "scatter_deg": 0.0,
        "scatter_seed": "",
        "mas": "CIC",
        "config": "",
        "source": str(path),
    }


def load_rows(task_file, baseline_dir, require_complete):
    rows = [read_hdf5_row(spec) for spec in task_specs(task_file)]
    fields = sorted({row["field"] for row in rows})
    baselines = [
        read_hdf5_row(baseline_spec(field, baseline_dir))
        for field in fields
    ]
    missing = [
        row for row in [*rows, *baselines] if row["status"] != "complete"]
    if missing and require_complete:
        preview = "\n".join(row["source"] for row in missing[:8])
        raise FileNotFoundError(
            f"{len(missing)} outputs are missing. First missing:\n{preview}")
    completed_rows = [row for row in rows if row["status"] == "complete"]
    completed_baselines = [
        row for row in baselines if row["status"] == "complete"]
    add_baseline_deltas(completed_rows, completed_baselines)
    return completed_rows, completed_baselines, missing


def add_baseline_deltas(rows, baselines):
    by_field = {row["field"]: row for row in baselines}
    for row in rows:
        base = by_field.get(row["field"])
        if base is None:
            continue
        row["baseline_H0_q50"] = base["H0_q50"]
        row["baseline_H0_mean"] = base["H0_mean"]
        row["baseline_lnZ_harmonic"] = base["lnZ_harmonic"]
        row["delta_H0_q50_vs_baseline"] = row["H0_q50"] - base["H0_q50"]
        row["delta_H0_mean_vs_baseline"] = row["H0_mean"] - base["H0_mean"]
        row["delta_lnZ_harmonic_vs_baseline"] = (
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
        "ch0_angular_scatter_fields", FIELD_COLOURS)


def field_norm(rows):
    fields = np.asarray([row["field"] for row in rows], dtype=float)
    vmin = min(0.0, float(np.min(fields)))
    vmax = float(np.max(fields))
    if vmin == vmax:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def rows_by_scatter(rows):
    values = sorted({row["scatter_deg"] for row in rows})
    return {
        value: sorted(
            [row for row in rows if row["scatter_deg"] == value],
            key=lambda row: row["field"])
        for value in values
    }


def stacked_samples(rows):
    if not rows:
        return np.asarray([], dtype=float)
    return np.concatenate([row["samples"] for row in rows])


def kde_on_grid(samples, x_grid, bw=1.15):
    kde = gaussian_kde(samples)
    kde.set_bandwidth(kde.factor * bw)
    return kde(x_grid)


def scatter_label(value):
    if value == 0.0:
        return "no scatter"
    return rf"${value:g}^\circ$"


def plot_joint_h0(rows, baselines, out_pdf):
    grouped = {0.0: baselines, **rows_by_scatter(rows)}
    all_samples = np.concatenate([
        stacked_samples(group) for group in grouped.values() if group])
    x_min, x_max = np.percentile(all_samples, [0.3, 99.7])
    pad = 0.08 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 800)

    with plt.style.context("science"):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(3.75, 3.05), constrained_layout=True)
        for scatter, group in grouped.items():
            samples = stacked_samples(group)
            if samples.size == 0:
                continue
            summary = h0_summary(samples)
            colour = SCATTER_COLOURS.get(scatter)
            ax.plot(
                x_grid, kde_on_grid(samples, x_grid), color=colour, lw=1.2,
                label=(
                    rf"{scatter_label(scatter)} "
                    rf"(${summary['H0_mean']:.2f}\pm"
                    rf"{summary['H0_std']:.2f}$)"
                ))
            ax.axvline(summary["H0_mean"], color=colour, lw=0.65, alpha=0.45)
        ax.set_xlabel(H0_LABEL)
        ax.set_ylabel("Density")
        ax.set_title("Stacked field posteriors", loc="left")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", frameon=False, handlelength=1.8)
        return save_pdf_png(fig, out_pdf)


def plot_field_shift_lines(rows, out_pdf):
    grouped = rows_by_scatter(rows)
    scatters = np.asarray(sorted(grouped), dtype=float)
    fields = sorted({row["field"] for row in rows})
    cmap = field_cmap()
    norm = field_norm(rows)

    with plt.style.context("science"):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(4.2, 3.25), constrained_layout=True)
        for field in fields:
            field_rows = [
                next((row for row in grouped[scatter]
                      if row["field"] == field), None)
                for scatter in scatters
            ]
            field_rows = [row for row in field_rows if row is not None]
            if len(field_rows) < 2:
                continue
            x = np.asarray([row["scatter_deg"] for row in field_rows])
            y = np.asarray([
                row["delta_H0_q50_vs_baseline"] for row in field_rows])
            ax.plot(
                x, y, color=cmap(norm(field)), lw=0.75, alpha=0.55)
            ax.scatter(
                x, y, color=cmap(norm(field)), s=13, alpha=0.85, zorder=3)

        means = []
        stds = []
        for scatter in scatters:
            vals = np.asarray([
                row["delta_H0_q50_vs_baseline"]
                for row in grouped[scatter]
                if "delta_H0_q50_vs_baseline" in row
            ], dtype=float)
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1))
        ax.errorbar(
            scatters, means, yerr=stds, color="black", marker="o", ms=4.0,
            lw=1.25, capsize=2.2,
            label=r"mean $H_0$ shift across fields",
            zorder=5)
        ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
        ax.set_xlabel("Angular scatter [deg]")
        ax.set_ylabel(
            r"$H_0(\mathrm{scatter}) - H_0(\mathrm{no\ scatter})$")
        ax.set_title("Matched realisations", loc="left")
        ax.text(
            0.02, 0.04,
            "thin lines: individual matched fields\n"
            "error bars: field-to-field standard deviation",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.5, color="0.25")
        ax.legend(
            loc="upper left", frameon=True, facecolor="white",
            edgecolor="none", framealpha=0.85, handlelength=1.8)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.018, fraction=0.065)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_shift_summary(rows, out_pdf):
    grouped = rows_by_scatter(rows)
    scatters = np.asarray(sorted(grouped), dtype=float)
    mean_shift = []
    med_shift = []
    std_shift = []
    mean_abs_shift = []
    for scatter in scatters:
        vals = np.asarray([
            row["delta_H0_q50_vs_baseline"] for row in grouped[scatter]
        ], dtype=float)
        mean_shift.append(np.mean(vals))
        med_shift.append(np.median(vals))
        std_shift.append(np.std(vals, ddof=1))
        mean_abs_shift.append(np.mean(np.abs(vals)))

    with plt.style.context("science"):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(4.85, 3.05), constrained_layout=True)
        ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
        ax.errorbar(
            scatters, mean_shift, yerr=std_shift, color=COLOUR_H0,
            marker="o", capsize=2.2, lw=1.1)
        ax.plot(
            scatters, med_shift, color="#87193d", marker="s", lw=1.0)
        ax.plot(
            scatters, mean_abs_shift, color="#fe9000", marker="^", lw=1.0)
        ax.set_xlim(1.3, 23.5)
        ax.set_xlabel("Angular scatter [deg]")
        ax.set_ylabel(
            r"$H_0(\mathrm{scatter}) - H_0(\mathrm{no\ scatter})$")
        ax.set_title("Average angular-scatter shift", loc="left")
        ax.text(
            0.02, 0.04,
            r"shift = $H_0(\mathrm{scatter}) - H_0(\mathrm{no\ scatter})$"
            "\nerror bars = field-to-field std; absolute shift ignores sign",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.2, color="0.25")
        label_x = 16.35
        ax.text(
            label_x, mean_shift[-1],
            "mean signed shift",
            color=COLOUR_H0, fontsize=6.5, va="center", ha="left")
        ax.text(
            label_x, med_shift[-1],
            "median signed shift",
            color="#87193d", fontsize=6.5, va="center", ha="left")
        ax.text(
            label_x, mean_abs_shift[-1],
            "mean absolute shift",
            color="#fe9000", fontsize=6.5, va="center", ha="left")
        return save_pdf_png(fig, out_pdf)


def plot_h0_vs_lnz(rows, baselines, out_pdf):
    all_rows = [*baselines, *rows]
    cmap = field_cmap()
    norm = field_norm(all_rows)
    scatters = [0.0, *sorted({row["scatter_deg"] for row in rows})]
    groups = {0.0: baselines, **rows_by_scatter(rows)}

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, len(scatters), figsize=(9.0, 2.35), sharey=True,
            constrained_layout=True)
        for ax, scatter in zip(axes, scatters):
            group = groups[scatter]
            fields = np.asarray([row["field"] for row in group], dtype=float)
            x = np.asarray([row["lnZ_harmonic"] for row in group])
            xerr = np.asarray([
                abs(row.get("err_lnZ_harmonic", np.nan)) for row in group])
            h0 = np.asarray([row["H0_q50"] for row in group])
            h0_lo = np.asarray([row["H0_q16"] for row in group])
            h0_hi = np.asarray([row["H0_q84"] for row in group])
            yerr = np.vstack([h0 - h0_lo, h0_hi - h0])
            ax.errorbar(
                x, h0, xerr=xerr, yerr=yerr, fmt="none", ecolor="0.55",
                elinewidth=0.45, capsize=1.1, alpha=0.60, zorder=1)
            sc = ax.scatter(
                x, h0, c=fields, cmap=cmap, norm=norm, s=23,
                edgecolor="0.15", linewidth=0.25, zorder=3)
            ax.set_title(scatter_label(scatter), loc="left")
            ax.set_xlabel(r"harmonic $\ln Z$")
            if ax is axes[0]:
                ax.set_ylabel(H0_LABEL)
        cbar = fig.colorbar(sc, ax=axes, pad=0.012, fraction=0.035)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def write_rows_csv(rows, baselines, path):
    fieldnames = [
        "status", "task", "field", "scatter_deg", "scatter_seed", "mas",
        "n_H0", "H0_mean", "H0_std", "H0_q16", "H0_q50", "H0_q84",
        "baseline_H0_q50", "delta_H0_q50_vs_baseline",
        "delta_H0_mean_vs_baseline", "lnZ_harmonic", "err_lnZ_harmonic",
        "baseline_lnZ_harmonic", "delta_lnZ_harmonic_vs_baseline",
        "lnZ_laplace", "err_lnZ_laplace", "BIC", "lnZ_bic",
        "source", "config",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in [*baselines, *rows]:
            out = {name: row.get(name, "") for name in fieldnames}
            writer.writerow(out)


def write_summary(rows, baselines, missing, path):
    grouped = {0.0: baselines, **rows_by_scatter(rows)}
    lines = [
        "# CH0 Angular-Scatter Diagnostics",
        "",
        f"Complete angular-scatter outputs: {len(rows)}.",
        f"Complete no-scatter baselines: {len(baselines)}.",
        f"Missing outputs: {len(missing)}.",
        "",
        "| scatter [deg] | fields | stacked H0 mean | stacked H0 std | "
        "mean signed delta H0 | field-to-field std delta H0 | "
        "mean absolute delta H0 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scatter, group in grouped.items():
        samples = stacked_samples(group)
        summary = h0_summary(samples)
        if scatter == 0.0:
            deltas = np.zeros(len(group), dtype=float)
        else:
            deltas = np.asarray([
                row["delta_H0_q50_vs_baseline"] for row in group],
                dtype=float)
        lines.append(
            f"| {scatter:g} | {len(group)} | {summary['H0_mean']:.3f} | "
            f"{summary['H0_std']:.3f} | {np.mean(deltas):+.3f} | "
            f"{np.std(deltas, ddof=1) if deltas.size > 1 else 0.0:.3f} | "
            f"{np.mean(np.abs(deltas)):.3f} |"
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
    rows, baselines, missing = load_rows(
        args.task_file, args.baseline_dir, args.require_complete)

    csv_path = args.output_dir / "angular_scatter_summary.csv"
    txt_path = args.output_dir / "angular_scatter_summary.txt"
    joint_pdf = args.output_dir / "angular_scatter_joint_h0.pdf"
    shift_lines_pdf = args.output_dir / "angular_scatter_h0_shift_fields.pdf"
    shift_summary_pdf = args.output_dir / "angular_scatter_h0_shift_summary.pdf"
    h0_lnz_pdf = args.output_dir / "angular_scatter_h0_vs_lnz.pdf"

    write_rows_csv(rows, baselines, csv_path)
    write_summary(rows, baselines, missing, txt_path)
    written = [
        csv_path,
        txt_path,
        *plot_joint_h0(rows, baselines, joint_pdf),
        *plot_field_shift_lines(rows, shift_lines_pdf),
        *plot_shift_summary(rows, shift_summary_pdf),
        *plot_h0_vs_lnz(rows, baselines, h0_lnz_pdf),
    ]

    for path in written:
        print(f"Wrote {path}")
    print(
        f"Complete angular-scatter outputs: {len(rows)}; "
        f"baselines: {len(baselines)}; missing: {len(missing)}."
    )


if __name__ == "__main__":
    main()
