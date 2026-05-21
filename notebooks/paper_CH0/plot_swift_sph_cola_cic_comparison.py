#!/usr/bin/env python
"""Compare CH0 single-field SWIFT SPH and COLA CIC inferences."""

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


ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = ROOT / "scripts" / "runs" / "tasks_CH0_single.txt"
DEFAULT_OUTDIR = (
    ROOT / "results" / "CH0_paper" / "single_fields" / "plots"
    / "swift_sph_cola_cic"
)
FIGURE_DPI = 500
HIGHLIGHT_FIELD = 21
H0_LABEL = (
    r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"
)
FIELD_COLOURS = ["#ef476f", "#473198", "#a8c256", "#5adbff", "#fe9000"]
RUN_LABELS = {
    "swift_sph": "SWIFT SPH",
    "cola_cic": "COLA CIC",
}
RUN_COLOURS = {
    "swift_sph": "#473198",
    "cola_cic": "#168039",
}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-file", type=Path, default=TASK_FILE,
        help="Task file containing the CH0 single-field configs.")
    parser.add_argument(
        "--task-min", type=int, default=0,
        help="First task index to include.")
    parser.add_argument(
        "--task-max", type=int, default=109,
        help="Last task index to include.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTDIR,
        help="Directory for plots and summaries.")
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


def task_config_paths(task_file, task_min, task_max):
    paths = []
    with repo_path(task_file).open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            task_index, config = line.split(maxsplit=1)
            task_index = int(task_index)
            if task_min <= task_index <= task_max:
                paths.append((task_index, repo_path(config)))
    if not paths:
        raise ValueError(
            f"No tasks in index range [{task_min}, {task_max}].")
    return paths


def classify_config(config):
    reconstruction = get_nested(config, ("io", "SH0ES", "reconstruction"))
    if reconstruction == "ManticoreLocalSWIFT":
        return "swift_sph"
    if reconstruction == "ManticoreLocalCOLA":
        mas = get_nested(
            config, ("io", "reconstruction_main", "ManticoreLocalCOLA",
                     "which_MAS"))
        if mas == "CIC":
            return "cola_cic"
    return None


def output_spec(task_index, config_path):
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    run = classify_config(config)
    if run is None:
        return None
    return {
        "task": task_index,
        "run": run,
        "field": int(get_nested(config, ("io", "field_indices"))),
        "config": str(config_path),
        "source": str(repo_path(get_nested(config, ("io", "fname_output")))),
    }


def finite_h0(handle, path):
    samples = np.asarray(handle["samples/H0"], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"`{path}` has no finite H0 samples.")
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
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        samples = finite_h0(handle, path)
        bic = read_scalar(handle, "gof/BIC", path)
        return {
            **spec,
            **h0_summary(samples),
            "lnZ_harmonic": read_scalar(handle, "gof/lnZ_harmonic", path),
            "err_lnZ_harmonic": read_scalar(
                handle, "gof/err_lnZ_harmonic", path),
            "lnZ_laplace": read_scalar(handle, "gof/lnZ_laplace", path),
            "err_lnZ_laplace": read_scalar(
                handle, "gof/err_lnZ_laplace", path),
            "BIC": bic,
            "lnZ_bic": -0.5 * bic,
        }


def load_rows(task_file, task_min, task_max):
    specs = []
    for task_index, config_path in task_config_paths(
            task_file, task_min, task_max):
        spec = output_spec(task_index, config_path)
        if spec is not None:
            specs.append(spec)
    rows = [read_row(spec) for spec in specs]
    rows = sorted(rows, key=lambda row: (row["run"], row["field"]))
    counts = {run: sum(row["run"] == run for row in rows)
              for run in RUN_LABELS}
    if counts["swift_sph"] == 0 or counts["cola_cic"] == 0:
        raise ValueError(f"Expected both runs, got counts {counts}.")
    return rows


def split_rows(rows):
    return {
        run: sorted([row for row in rows if row["run"] == run],
                    key=lambda row: row["field"])
        for run in RUN_LABELS
    }


def common_rows(rows_by_run):
    swift = {row["field"]: row for row in rows_by_run["swift_sph"]}
    cola = {row["field"]: row for row in rows_by_run["cola_cic"]}
    fields = sorted(set(swift) & set(cola))
    return [(field, swift[field], cola[field]) for field in fields]


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
        "swift_sph_cola_cic_fields", FIELD_COLOURS)


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


def add_best_label(ax, rows, key):
    best = max(rows, key=lambda row: row[key])
    ax.scatter(
        best[key], best["H0_q50"], marker="*", s=72,
        color="black", edgecolor="white", linewidth=0.4, zorder=5)
    ax.annotate(
        f"{best['field']}", (best[key], best["H0_q50"]),
        xytext=(4, 4), textcoords="offset points", fontsize=6.7)


def highlight_point(ax, x, y, label=HIGHLIGHT_FIELD):
    ax.scatter(
        x, y, marker="D", s=58, facecolor="none", edgecolor="#d42a29",
        linewidth=1.1, zorder=6)
    ax.annotate(
        f"{label}", (x, y), xytext=(5, -9), textcoords="offset points",
        fontsize=6.8, color="#d42a29", zorder=7)


def row_for_field(rows, field):
    for row in rows:
        if row["field"] == field:
            return row
    return None


def plot_h0_vs_lnz(rows_by_run, out_pdf):
    all_rows = rows_by_run["swift_sph"] + rows_by_run["cola_cic"]
    cmap = field_cmap()
    norm = field_norm(all_rows)
    specs = (
        ("lnZ_harmonic", "err_lnZ_harmonic", r"harmonic $\ln Z$"),
        ("lnZ_laplace", "err_lnZ_laplace", r"Laplace $\ln Z$"),
    )

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 2, figsize=(7.1, 5.4), sharey="row",
            constrained_layout=True)
        for col, run in enumerate(("swift_sph", "cola_cic")):
            rows = rows_by_run[run]
            fields = np.asarray([row["field"] for row in rows], dtype=float)
            h0 = np.asarray([row["H0_q50"] for row in rows], dtype=float)
            h0_lo = np.asarray([row["H0_q16"] for row in rows], dtype=float)
            h0_hi = np.asarray([row["H0_q84"] for row in rows], dtype=float)
            yerr = np.vstack([h0 - h0_lo, h0_hi - h0])
            for row_idx, (key, err_key, xlabel) in enumerate(specs):
                ax = axes[row_idx, col]
                x = np.asarray([row[key] for row in rows], dtype=float)
                xerr = np.asarray(
                    [abs(row[err_key]) for row in rows], dtype=float)
                ax.errorbar(
                    x, h0, xerr=xerr, yerr=yerr, fmt="none",
                    ecolor="0.55", elinewidth=0.45, capsize=1.2, alpha=0.65,
                    zorder=1)
                sc = ax.scatter(
                    x, h0, c=fields, cmap=cmap, norm=norm, s=26,
                    edgecolor="0.15", linewidth=0.25, zorder=3)
                add_best_label(ax, rows, key)
                if run == "cola_cic":
                    row = row_for_field(rows, HIGHLIGHT_FIELD)
                    if row is not None:
                        highlight_point(ax, row[key], row["H0_q50"])
                ax.set_title(
                    f"{RUN_LABELS[run]}: {len(rows)} fields", loc="left")
                ax.set_xlabel(xlabel)
                if col == 0:
                    ax.set_ylabel(H0_LABEL)
        cbar = fig.colorbar(sc, ax=axes, pad=0.012, fraction=0.035)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def identity_limits(x, y, pad_frac=0.05):
    low = min(float(np.min(x)), float(np.min(y)))
    high = max(float(np.max(x)), float(np.max(y)))
    pad = pad_frac * (high - low)
    if pad == 0:
        pad = 1.0
    return low - pad, high + pad


def plot_common_value_comparison(common, out_pdf):
    fields = np.asarray([field for field, _, _ in common], dtype=float)
    swift_lnz = np.asarray([
        swift["lnZ_harmonic"] for _, swift, _ in common], dtype=float)
    cola_lnz = np.asarray([
        cola["lnZ_harmonic"] for _, _, cola in common], dtype=float)
    swift_h0 = np.asarray([swift["H0_q50"] for _, swift, _ in common])
    cola_h0 = np.asarray([cola["H0_q50"] for _, _, cola in common])
    cmap = field_cmap()
    norm = field_norm([swift for _, swift, _ in common])

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 2, figsize=(7.1, 3.15), constrained_layout=True)
        for ax, x, y, label in (
                (axes[0], swift_lnz, cola_lnz, r"harmonic $\ln Z$"),
                (axes[1], swift_h0, cola_h0, H0_LABEL)):
            limits = identity_limits(x, y)
            ax.plot(limits, limits, color="0.45", lw=0.8, ls="--", zorder=1)
            sc = ax.scatter(
                x, y, c=fields, cmap=cmap, norm=norm, s=30,
                edgecolor="0.15", linewidth=0.25, zorder=3)
            if HIGHLIGHT_FIELD in fields:
                idx = int(np.where(fields == HIGHLIGHT_FIELD)[0][0])
                highlight_point(ax, x[idx], y[idx])
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_xlabel(f"SWIFT SPH {label}")
            ax.set_ylabel(f"COLA CIC {label}")
            ax.set_title(f"{len(common)} common fields", loc="left")
        cbar = fig.colorbar(sc, ax=axes, pad=0.014, fraction=0.045)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_common_deltas(common, out_pdf):
    fields = np.asarray([field for field, _, _ in common], dtype=int)
    delta_lnz = np.asarray([
        cola["lnZ_harmonic"] - swift["lnZ_harmonic"]
        for _, swift, cola in common
    ], dtype=float)
    delta_h0 = np.asarray([
        cola["H0_q50"] - swift["H0_q50"]
        for _, swift, cola in common
    ], dtype=float)

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 1, figsize=(7.1, 4.4), sharex=True,
            constrained_layout=True)
        for ax, y, ylabel, colour in (
                (axes[0], delta_lnz,
                 r"$\Delta \ln Z_{\rm harm}$ (COLA CIC - SWIFT SPH)",
                 RUN_COLOURS["cola_cic"]),
                (axes[1], delta_h0,
                 r"$\Delta H_0$ (COLA CIC - SWIFT SPH)", "#473198")):
            ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
            ax.plot(fields, y, color=colour, lw=0.75, alpha=0.65)
            ax.scatter(fields, y, color=colour, s=18, zorder=3)
            if HIGHLIGHT_FIELD in fields:
                idx = int(np.where(fields == HIGHLIGHT_FIELD)[0][0])
                highlight_point(ax, fields[idx], y[idx])
            ax.set_ylabel(ylabel)
            ax.text(
                0.01, 0.97,
                rf"mean $={np.mean(y):+.3f}$, "
                rf"median $={np.median(y):+.3f}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.7,
                bbox={
                    "boxstyle": "round,pad=0.14",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.86,
                })
        axes[1].set_xlabel("Manticore field")
        return save_pdf_png(fig, out_pdf)


def write_rows_csv(rows, path):
    fieldnames = [
        "task", "run", "field", "n_H0", "H0_mean", "H0_std", "H0_q16",
        "H0_q50", "H0_q84", "lnZ_harmonic", "err_lnZ_harmonic",
        "lnZ_laplace", "err_lnZ_laplace", "BIC", "lnZ_bic", "source",
        "config",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_common_csv(common, path):
    fieldnames = [
        "field", "swift_H0_q50", "cola_H0_q50", "delta_H0_q50_cola_swift",
        "swift_lnZ_harmonic", "cola_lnZ_harmonic",
        "delta_lnZ_harmonic_cola_swift", "swift_lnZ_laplace",
        "cola_lnZ_laplace", "delta_lnZ_laplace_cola_swift",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for field, swift, cola in common:
            writer.writerow({
                "field": field,
                "swift_H0_q50": swift["H0_q50"],
                "cola_H0_q50": cola["H0_q50"],
                "delta_H0_q50_cola_swift": (
                    cola["H0_q50"] - swift["H0_q50"]),
                "swift_lnZ_harmonic": swift["lnZ_harmonic"],
                "cola_lnZ_harmonic": cola["lnZ_harmonic"],
                "delta_lnZ_harmonic_cola_swift": (
                    cola["lnZ_harmonic"] - swift["lnZ_harmonic"]),
                "swift_lnZ_laplace": swift["lnZ_laplace"],
                "cola_lnZ_laplace": cola["lnZ_laplace"],
                "delta_lnZ_laplace_cola_swift": (
                    cola["lnZ_laplace"] - swift["lnZ_laplace"]),
            })


def write_summary(rows_by_run, common, path):
    lines = [
        "# SWIFT SPH and COLA CIC Single-Field Comparison",
        "",
    ]
    for run in ("swift_sph", "cola_cic"):
        rows = rows_by_run[run]
        best_h = max(rows, key=lambda row: row["lnZ_harmonic"])
        best_l = max(rows, key=lambda row: row["lnZ_laplace"])
        h0 = np.asarray([row["H0_q50"] for row in rows], dtype=float)
        lines.extend([
            f"## {RUN_LABELS[run]}",
            "",
            f"Fields: {len(rows)}.",
            "",
            "| quantity | value |",
            "| --- | ---: |",
            f"| best harmonic lnZ field | {best_h['field']} |",
            f"| best harmonic lnZ | {best_h['lnZ_harmonic']:.3f} |",
            f"| best-field H0 median | {best_h['H0_q50']:.3f} |",
            f"| best Laplace lnZ field | {best_l['field']} |",
            f"| best Laplace lnZ | {best_l['lnZ_laplace']:.3f} |",
            f"| median field H0 median | {np.median(h0):.3f} |",
            "",
        ])

    delta_lnz = np.asarray([
        cola["lnZ_harmonic"] - swift["lnZ_harmonic"]
        for _, swift, cola in common
    ], dtype=float)
    delta_h0 = np.asarray([
        cola["H0_q50"] - swift["H0_q50"]
        for _, swift, cola in common
    ], dtype=float)
    lines.extend([
        "## Common Fields",
        "",
        f"Common fields: {len(common)}.",
        "",
        "| quantity | value |",
        "| --- | ---: |",
        f"| mean delta harmonic lnZ, COLA-SWIFT | {np.mean(delta_lnz):+.3f} |",
        f"| median delta harmonic lnZ, COLA-SWIFT | "
        f"{np.median(delta_lnz):+.3f} |",
        f"| std delta harmonic lnZ, COLA-SWIFT | "
        f"{np.std(delta_lnz, ddof=1):.3f} |",
        f"| mean delta H0 median, COLA-SWIFT | {np.mean(delta_h0):+.3f} |",
        f"| median delta H0 median, COLA-SWIFT | "
        f"{np.median(delta_h0):+.3f} |",
        f"| std delta H0 median, COLA-SWIFT | "
        f"{np.std(delta_h0, ddof=1):.3f} |",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    rows = load_rows(args.task_file, args.task_min, args.task_max)
    rows_by_run = split_rows(rows)
    common = common_rows(rows_by_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows_csv = args.output_dir / "swift_sph_cola_cic_single_field_summary.csv"
    common_csv = (
        args.output_dir / "swift_sph_cola_cic_common_field_summary.csv")
    summary_txt = args.output_dir / "swift_sph_cola_cic_summary.txt"
    h0_lnz_pdf = args.output_dir / "swift_sph_cola_cic_h0_vs_lnz.pdf"
    common_pdf = (
        args.output_dir / "swift_sph_cola_cic_common_value_comparison.pdf")
    deltas_pdf = args.output_dir / "swift_sph_cola_cic_common_deltas.pdf"

    write_rows_csv(rows, rows_csv)
    write_common_csv(common, common_csv)
    write_summary(rows_by_run, common, summary_txt)
    written = [
        rows_csv,
        common_csv,
        summary_txt,
        *plot_h0_vs_lnz(rows_by_run, h0_lnz_pdf),
        *plot_common_value_comparison(common, common_pdf),
        *plot_common_deltas(common, deltas_pdf),
    ]

    for path in written:
        print(f"Wrote {path}")
    print(
        f"Loaded {len(rows_by_run['swift_sph'])} SWIFT SPH fields, "
        f"{len(rows_by_run['cola_cic'])} COLA CIC fields, "
        f"and {len(common)} common fields."
    )


if __name__ == "__main__":
    main()
