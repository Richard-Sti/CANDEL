#!/usr/bin/env python
"""Compare COLA CIC, PCS, and SPH CH0 single-field inferences."""

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
    / "cola_mas_comparison"
)
FIGURE_DPI = 500
MAS_ORDER = ("CIC", "PCS", "SPH")
MAS_LABELS = {"CIC": "CIC", "PCS": "PCS", "SPH": "SPH"}
MAS_COLOURS = {"CIC": "#473198", "PCS": "#168039", "SPH": "#fe9000"}
FIELD_COLOURS = ["#ef476f", "#473198", "#a8c256", "#5adbff", "#fe9000"]
H0_LABEL = (
    r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"
)
LNZ_LABEL = r"harmonic $\ln Z$"


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-file", type=Path, default=TASK_FILE,
        help="Task file containing the CH0 single-field configs.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTDIR,
        help="Directory for plots and summaries.")
    parser.add_argument(
        "--require-complete", action="store_true",
        help="Fail if any of the 80 matched COLA MAS triplets is missing.")
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


def task_config_paths(task_file):
    paths = []
    with repo_path(task_file).open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            task_index, config = line.split(maxsplit=1)
            paths.append((int(task_index), repo_path(config)))
    return paths


def output_spec(task_index, config_path):
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    reconstruction = get_nested(config, ("io", "SH0ES", "reconstruction"))
    if reconstruction != "ManticoreLocalCOLA":
        return None
    mas = get_nested(
        config, ("io", "reconstruction_main", "ManticoreLocalCOLA",
                 "which_MAS"))
    if mas not in MAS_ORDER:
        return None
    return {
        "task": task_index,
        "mas": mas,
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


def load_rows(task_file):
    specs = []
    for task_index, config_path in task_config_paths(task_file):
        spec = output_spec(task_index, config_path)
        if spec is not None:
            specs.append(spec)
    rows = [read_row(spec) for spec in specs]
    return sorted(rows, key=lambda row: (row["field"], row["mas"]))


def matched_triplets(rows, require_complete=False):
    by_field = {}
    for row in rows:
        by_field.setdefault(row["field"], {})[row["mas"]] = row
    matched = {
        field: {mas: by_field[field][mas] for mas in MAS_ORDER}
        for field in sorted(by_field)
        if all(mas in by_field[field] for mas in MAS_ORDER)
    }
    missing = {
        field: [mas for mas in MAS_ORDER if mas not in rows_by_mas]
        for field, rows_by_mas in sorted(by_field.items())
        if any(mas not in rows_by_mas for mas in MAS_ORDER)
    }
    if require_complete:
        expected_fields = set(range(80))
        missing_fields = sorted(expected_fields - set(matched))
        if missing or missing_fields:
            raise FileNotFoundError(
                f"Incomplete COLA MAS comparison. Missing MAS={missing}; "
                f"missing matched fields={missing_fields}.")
    return matched, missing


def rows_by_mas(matched):
    return {
        mas: [matched[field][mas] for field in sorted(matched)]
        for mas in MAS_ORDER
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
        "cola_mas_comparison_fields", FIELD_COLOURS)


def field_norm(fields):
    fields = np.asarray(fields, dtype=float)
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


def identity_limits(*arrays, pad_frac=0.05):
    values = np.concatenate([np.asarray(array, dtype=float) for array in arrays])
    low = float(np.min(values))
    high = float(np.max(values))
    pad = pad_frac * (high - low)
    if pad == 0:
        pad = 1.0
    return low - pad, high + pad


def plot_h0_vs_lnz(by_mas, out_pdf):
    fields = [row["field"] for rows in by_mas.values() for row in rows]
    cmap = field_cmap()
    norm = field_norm(fields)
    specs = (
        ("lnZ_harmonic", "err_lnZ_harmonic", LNZ_LABEL),
        ("lnZ_laplace", "err_lnZ_laplace", r"Laplace $\ln Z$"),
    )

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 3, figsize=(8.4, 5.35), sharey="row",
            constrained_layout=True)
        for col, mas in enumerate(MAS_ORDER):
            rows = by_mas[mas]
            field_values = np.asarray([row["field"] for row in rows])
            h0 = np.asarray([row["H0_q50"] for row in rows])
            h0_lo = np.asarray([row["H0_q16"] for row in rows])
            h0_hi = np.asarray([row["H0_q84"] for row in rows])
            yerr = np.vstack([h0 - h0_lo, h0_hi - h0])
            for row_idx, (key, err_key, xlabel) in enumerate(specs):
                ax = axes[row_idx, col]
                x = np.asarray([row[key] for row in rows])
                xerr = np.asarray([abs(row[err_key]) for row in rows])
                ax.errorbar(
                    x, h0, xerr=xerr, yerr=yerr, fmt="none",
                    ecolor="0.55", elinewidth=0.45, capsize=1.1, alpha=0.55,
                    zorder=1)
                sc = ax.scatter(
                    x, h0, c=field_values, cmap=cmap, norm=norm, s=24,
                    edgecolor="0.15", linewidth=0.25, zorder=3)
                best = max(rows, key=lambda row: row[key])
                ax.scatter(
                    best[key], best["H0_q50"], marker="*", s=72,
                    color="black", edgecolor="white", linewidth=0.4, zorder=5)
                ax.annotate(
                    f"{best['field']}", (best[key], best["H0_q50"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=6.7)
                ax.set_title(f"{MAS_LABELS[mas]}: {len(rows)} fields",
                             loc="left")
                ax.set_xlabel(xlabel)
                if col == 0:
                    ax.set_ylabel(H0_LABEL)
        cbar = fig.colorbar(sc, ax=axes, pad=0.012, fraction=0.035)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_matched_realisations(matched, out_pdf):
    fields = sorted(matched)
    xpos = np.arange(len(MAS_ORDER))
    cmap = field_cmap()
    norm = field_norm(fields)

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 1, figsize=(6.25, 5.0), sharex=True,
            constrained_layout=True)
        for ax, key, ylabel in (
                (axes[0], "H0_q50", H0_LABEL),
                (axes[1], "lnZ_harmonic", LNZ_LABEL)):
            mas_values = []
            for field in fields:
                values = np.asarray([matched[field][mas][key]
                                     for mas in MAS_ORDER])
                mas_values.append(values)
                ax.plot(
                    xpos, values, color=cmap(norm(field)), lw=0.65,
                    alpha=0.38, zorder=1)
                ax.scatter(
                    xpos, values, color=cmap(norm(field)), s=9,
                    alpha=0.64, zorder=2)
            mas_values = np.asarray(mas_values)
            means = np.mean(mas_values, axis=0)
            stds = np.std(mas_values, axis=0, ddof=1)
            ax.errorbar(
                xpos, means, yerr=stds, color="black", marker="o",
                lw=1.25, ms=4.0, capsize=2.4,
                label="mean across fields\n(error bar: field-to-field std)",
                zorder=5)
            ax.set_ylabel(ylabel)
            ax.set_title(
                "Matched realisations" if key == "H0_q50" else "",
                loc="left")
            ax.legend(
                loc="best", frameon=True, facecolor="white",
                edgecolor="none", framealpha=0.86, handlelength=1.7)
        axes[1].set_xticks(xpos, [MAS_LABELS[mas] for mas in MAS_ORDER])
        axes[1].set_xlabel("Mass-assignment scheme")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, pad=0.015, fraction=0.045)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_bulk_distributions(by_mas, out_pdf):
    h0_values = [[row["H0_q50"] for row in by_mas[mas]]
                 for mas in MAS_ORDER]
    lnz_values = [[row["lnZ_harmonic"] for row in by_mas[mas]]
                  for mas in MAS_ORDER]

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 2, figsize=(7.2, 3.15), constrained_layout=True)
        for ax, values, ylabel in (
                (axes[0], h0_values, H0_LABEL),
                (axes[1], lnz_values, LNZ_LABEL)):
            parts = ax.violinplot(
                values, positions=np.arange(len(MAS_ORDER)), widths=0.72,
                showextrema=False, showmeans=False, showmedians=False)
            for mas, body in zip(MAS_ORDER, parts["bodies"]):
                body.set_facecolor(MAS_COLOURS[mas])
                body.set_edgecolor("none")
                body.set_alpha(0.28)
            for idx, (mas, vals) in enumerate(zip(MAS_ORDER, values)):
                vals = np.asarray(vals)
                jitter = np.linspace(-0.13, 0.13, vals.size)
                order = np.argsort(vals)
                ax.scatter(
                    idx + jitter[order], vals[order], s=9,
                    color=MAS_COLOURS[mas], alpha=0.58, linewidth=0,
                    zorder=3)
                ax.errorbar(
                    idx, np.mean(vals), yerr=np.std(vals, ddof=1),
                    color="black", marker="o", ms=4, capsize=2.4,
                    zorder=4)
                ax.scatter(
                    idx, np.median(vals), color="white", edgecolor="black",
                    marker="s", s=18, linewidth=0.55, zorder=5)
            ax.set_xticks(
                np.arange(len(MAS_ORDER)), [MAS_LABELS[mas]
                                            for mas in MAS_ORDER])
            ax.set_ylabel(ylabel)
        axes[0].set_title("Bulk field distribution", loc="left")
        axes[1].text(
            0.02, 0.04,
            "black circles: mean; bars: field-to-field std\n"
            "white squares: median",
            transform=axes[1].transAxes, ha="left", va="bottom",
            fontsize=6.5, color="0.25")
        return save_pdf_png(fig, out_pdf)


def plot_pairwise_comparison(matched, out_pdf):
    fields = sorted(matched)
    cmap = field_cmap()
    norm = field_norm(fields)
    pairs = (("CIC", "PCS"), ("CIC", "SPH"), ("PCS", "SPH"))
    specs = (("H0_q50", H0_LABEL), ("lnZ_harmonic", LNZ_LABEL))

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 3, figsize=(8.4, 5.2), constrained_layout=True)
        for row_idx, (key, label) in enumerate(specs):
            for col, (mas_x, mas_y) in enumerate(pairs):
                ax = axes[row_idx, col]
                x = np.asarray([matched[field][mas_x][key]
                                for field in fields])
                y = np.asarray([matched[field][mas_y][key]
                                for field in fields])
                limits = identity_limits(x, y)
                ax.plot(limits, limits, color="0.45", lw=0.8, ls="--")
                sc = ax.scatter(
                    x, y, c=fields, cmap=cmap, norm=norm, s=25,
                    edgecolor="0.15", linewidth=0.25, zorder=3)
                ax.set_xlim(limits)
                ax.set_ylim(limits)
                ax.set_xlabel(f"{MAS_LABELS[mas_x]} {label}")
                ax.set_ylabel(f"{MAS_LABELS[mas_y]} {label}")
                delta = y - x
                ax.text(
                    0.03, 0.97,
                    rf"$\langle\Delta\rangle={np.mean(delta):+.3f}$"
                    "\n"
                    rf"$\sigma_\Delta={np.std(delta, ddof=1):.3f}$",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.4,
                    bbox={
                        "boxstyle": "round,pad=0.13",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.86,
                    })
        cbar = fig.colorbar(sc, ax=axes, pad=0.012, fraction=0.035)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def write_rows_csv(rows, path):
    fieldnames = [
        "task", "mas", "field", "n_H0", "H0_mean", "H0_std", "H0_q16",
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


def write_matched_csv(matched, path):
    fieldnames = [
        "field",
        "CIC_H0_q50", "PCS_H0_q50", "SPH_H0_q50",
        "PCS_minus_CIC_H0_q50", "SPH_minus_CIC_H0_q50",
        "SPH_minus_PCS_H0_q50",
        "CIC_lnZ_harmonic", "PCS_lnZ_harmonic", "SPH_lnZ_harmonic",
        "PCS_minus_CIC_lnZ_harmonic",
        "SPH_minus_CIC_lnZ_harmonic",
        "SPH_minus_PCS_lnZ_harmonic",
        "CIC_lnZ_laplace", "PCS_lnZ_laplace", "SPH_lnZ_laplace",
        "PCS_minus_CIC_lnZ_laplace",
        "SPH_minus_CIC_lnZ_laplace",
        "SPH_minus_PCS_lnZ_laplace",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for field in sorted(matched):
            rows = matched[field]
            writer.writerow({
                "field": field,
                "CIC_H0_q50": rows["CIC"]["H0_q50"],
                "PCS_H0_q50": rows["PCS"]["H0_q50"],
                "SPH_H0_q50": rows["SPH"]["H0_q50"],
                "PCS_minus_CIC_H0_q50": (
                    rows["PCS"]["H0_q50"] - rows["CIC"]["H0_q50"]),
                "SPH_minus_CIC_H0_q50": (
                    rows["SPH"]["H0_q50"] - rows["CIC"]["H0_q50"]),
                "SPH_minus_PCS_H0_q50": (
                    rows["SPH"]["H0_q50"] - rows["PCS"]["H0_q50"]),
                "CIC_lnZ_harmonic": rows["CIC"]["lnZ_harmonic"],
                "PCS_lnZ_harmonic": rows["PCS"]["lnZ_harmonic"],
                "SPH_lnZ_harmonic": rows["SPH"]["lnZ_harmonic"],
                "PCS_minus_CIC_lnZ_harmonic": (
                    rows["PCS"]["lnZ_harmonic"]
                    - rows["CIC"]["lnZ_harmonic"]),
                "SPH_minus_CIC_lnZ_harmonic": (
                    rows["SPH"]["lnZ_harmonic"]
                    - rows["CIC"]["lnZ_harmonic"]),
                "SPH_minus_PCS_lnZ_harmonic": (
                    rows["SPH"]["lnZ_harmonic"]
                    - rows["PCS"]["lnZ_harmonic"]),
                "CIC_lnZ_laplace": rows["CIC"]["lnZ_laplace"],
                "PCS_lnZ_laplace": rows["PCS"]["lnZ_laplace"],
                "SPH_lnZ_laplace": rows["SPH"]["lnZ_laplace"],
                "PCS_minus_CIC_lnZ_laplace": (
                    rows["PCS"]["lnZ_laplace"]
                    - rows["CIC"]["lnZ_laplace"]),
                "SPH_minus_CIC_lnZ_laplace": (
                    rows["SPH"]["lnZ_laplace"]
                    - rows["CIC"]["lnZ_laplace"]),
                "SPH_minus_PCS_lnZ_laplace": (
                    rows["SPH"]["lnZ_laplace"]
                    - rows["PCS"]["lnZ_laplace"]),
            })


def write_summary(by_mas, matched, missing, path):
    lines = [
        "# COLA MAS Single-Field Comparison",
        "",
        f"Matched fields: {len(matched)}.",
        f"Missing partial fields: {len(missing)}.",
        "",
        "## Bulk Behaviour",
        "",
        "| MAS | fields | mean H0 | std H0 | median H0 | "
        "mean harmonic lnZ | std harmonic lnZ | best lnZ field |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mas in MAS_ORDER:
        rows = by_mas[mas]
        h0 = np.asarray([row["H0_q50"] for row in rows])
        lnz = np.asarray([row["lnZ_harmonic"] for row in rows])
        best = max(rows, key=lambda row: row["lnZ_harmonic"])
        lines.append(
            f"| {MAS_LABELS[mas]} | {len(rows)} | {np.mean(h0):.3f} | "
            f"{np.std(h0, ddof=1):.3f} | {np.median(h0):.3f} | "
            f"{np.mean(lnz):.3f} | {np.std(lnz, ddof=1):.3f} | "
            f"{best['field']} |")

    lines.extend(["", "## Matched Differences", ""])
    lines.extend([
        "| difference | mean dH0 | std dH0 | median dH0 | "
        "mean dlnZ | std dlnZ | median dlnZ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for mas_a, mas_b in (("CIC", "PCS"), ("CIC", "SPH"), ("PCS", "SPH")):
        dh0 = np.asarray([
            matched[field][mas_b]["H0_q50"]
            - matched[field][mas_a]["H0_q50"]
            for field in sorted(matched)
        ])
        dlnz = np.asarray([
            matched[field][mas_b]["lnZ_harmonic"]
            - matched[field][mas_a]["lnZ_harmonic"]
            for field in sorted(matched)
        ])
        lines.append(
            f"| {MAS_LABELS[mas_b]} - {MAS_LABELS[mas_a]} | "
            f"{np.mean(dh0):+.3f} | {np.std(dh0, ddof=1):.3f} | "
            f"{np.median(dh0):+.3f} | {np.mean(dlnz):+.3f} | "
            f"{np.std(dlnz, ddof=1):.3f} | {np.median(dlnz):+.3f} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    rows = load_rows(args.task_file)
    matched, missing = matched_triplets(
        rows, require_complete=args.require_complete)
    by_mas = rows_by_mas(matched)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows_csv = args.output_dir / "cola_mas_single_field_summary.csv"
    matched_csv = args.output_dir / "cola_mas_matched_field_summary.csv"
    summary_txt = args.output_dir / "cola_mas_comparison_summary.txt"
    h0_lnz_pdf = args.output_dir / "cola_mas_h0_vs_lnz.pdf"
    matched_pdf = args.output_dir / "cola_mas_matched_realisations.pdf"
    bulk_pdf = args.output_dir / "cola_mas_bulk_distributions.pdf"
    pairwise_pdf = args.output_dir / "cola_mas_pairwise_comparison.pdf"

    write_rows_csv(rows, rows_csv)
    write_matched_csv(matched, matched_csv)
    write_summary(by_mas, matched, missing, summary_txt)
    written = [
        rows_csv,
        matched_csv,
        summary_txt,
        *plot_h0_vs_lnz(by_mas, h0_lnz_pdf),
        *plot_matched_realisations(matched, matched_pdf),
        *plot_bulk_distributions(by_mas, bulk_pdf),
        *plot_pairwise_comparison(matched, pairwise_pdf),
    ]

    for path in written:
        print(f"Wrote {path}")
    print(
        f"Loaded {len(rows)} COLA MAS outputs and "
        f"{len(matched)} matched CIC/PCS/SPH fields.")


if __name__ == "__main__":
    main()
