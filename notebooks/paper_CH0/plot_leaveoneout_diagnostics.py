#!/usr/bin/env python
"""Plot CH0 leave-one-out diagnostics."""

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


ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = ROOT / "scripts" / "runs" / "tasks_CH0_leaveoneout.txt"
RESULTS = ROOT / "results" / "CH0_paper"
DEFAULT_OUTDIR = RESULTS / "leaveoneout" / "plots"
DEFAULT_REFERENCE = (
    RESULTS / "single_fields"
    / "CH0_sel-SN_magnitude_ManticoreLocalSWIFT_field21_single.hdf5"
)
FIGURE_DPI = 500
H0_LABEL = (
    r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"
)
COLOUR_H0 = "#473198"
COLOUR_LNZ = "#168039"


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-file", type=Path, default=TASK_FILE,
        help="Task file containing the CH0 leave-one-out configs.")
    parser.add_argument(
        "--reference", type=Path, default=DEFAULT_REFERENCE,
        help="Full-sample field-21 HDF5 output used as the reference.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTDIR,
        help="Directory for plots and summaries.")
    parser.add_argument(
        "--require-complete", action="store_true",
        help="Fail if any leave-one-out output is missing.")
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


def decode_names(values):
    names = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode()
        names.append(str(value))
    return names


def display_host(name):
    if name.startswith("mu_"):
        return name[3:]
    return name


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
                "drop_index": int(get_nested(
                    config, ("io", "SH0ES", "drop_observation"))),
                "field": int(get_nested(config, ("io", "field_indices"))),
                "config": str(config_path),
                "source": str(repo_path(get_nested(
                    config, ("io", "fname_output")))),
            })
    if not specs:
        raise ValueError(f"No task configs found in `{task_file}`.")
    return sorted(specs, key=lambda spec: spec["drop_index"])


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


def read_reference(path):
    path = repo_path(path)
    with h5py.File(path, "r") as handle:
        h0 = finite_samples(handle, "H0", path)
        host_names = decode_names(handle["auxiliary/host_names"][()])
        bic = read_scalar(handle, "gof/BIC", path)
        return {
            "source": str(path),
            "host_names": host_names,
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


def dropped_host_from_output(reference_names, output_names, drop_index):
    expected = reference_names[drop_index]
    missing = sorted(set(reference_names) - set(output_names))
    if len(missing) == 1:
        return missing[0]
    return expected


def missing_row(spec, reference_names):
    drop_index = spec["drop_index"]
    host = (
        reference_names[drop_index]
        if 0 <= drop_index < len(reference_names) else ""
    )
    return {
        **spec,
        "status": "missing",
        "dropped_host": host,
        "dropped_host_label": display_host(host),
    }


def read_leaveoneout_row(spec, reference):
    path = Path(spec["source"])
    if not path.is_file():
        return missing_row(spec, reference["host_names"])

    with h5py.File(path, "r") as handle:
        h0 = finite_samples(handle, "H0", path)
        output_names = decode_names(handle["auxiliary/host_names"][()])
        host = dropped_host_from_output(
            reference["host_names"], output_names, spec["drop_index"])
        bic = read_scalar(handle, "gof/BIC", path)
        row = {
            **spec,
            "status": "complete",
            "dropped_host": host,
            "dropped_host_label": display_host(host),
            "n_hosts": len(output_names),
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
        row["delta_H0_q50_vs_full"] = row["H0_q50"] - reference["H0_q50"]
        row["delta_H0_mean_vs_full"] = row["H0_mean"] - reference["H0_mean"]
        row["delta_lnZ_harmonic_vs_completed_median"] = np.nan
        row["delta_lnZ_laplace_vs_completed_median"] = np.nan
        return row


def load_rows(task_file, reference, require_complete):
    rows = [
        read_leaveoneout_row(spec, reference)
        for spec in task_specs(task_file)
    ]
    missing = [row for row in rows if row["status"] != "complete"]
    if missing and require_complete:
        preview = "\n".join(row["source"] for row in missing[:8])
        raise FileNotFoundError(
            f"{len(missing)} leave-one-out outputs are missing. "
            f"First missing:\n{preview}")

    completed = [row for row in rows if row["status"] == "complete"]
    if completed:
        med_h = float(np.median([
            row["lnZ_harmonic"] for row in completed]))
        med_l = float(np.median([
            row["lnZ_laplace"] for row in completed]))
        for row in completed:
            row["delta_lnZ_harmonic_vs_completed_median"] = (
                row["lnZ_harmonic"] - med_h)
            row["delta_lnZ_laplace_vs_completed_median"] = (
                row["lnZ_laplace"] - med_l)
    return rows


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


def completed_rows(rows):
    return [row for row in rows if row["status"] == "complete"]


def label_ticks(ax, rows):
    labels = [row["dropped_host_label"] for row in rows]
    ax.set_xticks([row["drop_index"] for row in rows])
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.set_xlim(-0.8, max(row["drop_index"] for row in rows) + 1.6)


def mark_missing(ax, rows, y):
    missing = [row for row in rows if row["status"] != "complete"]
    if not missing:
        return
    x = [row["drop_index"] for row in missing]
    ax.scatter(
        x, np.full(len(x), y), marker="x", s=22, color="#d42a29",
        linewidth=0.9, label="missing", zorder=4)


def annotate_extremes(ax, rows, key, n=3, colour="black"):
    finite = [row for row in completed_rows(rows) if np.isfinite(row[key])]
    if not finite:
        return
    finite = sorted(finite, key=lambda row: abs(row[key]), reverse=True)[:n]
    for row in finite:
        ax.annotate(
            row["dropped_host_label"],
            (row["drop_index"], row[key]),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=6.3,
            color=colour,
        )


def padded_limits(values, lower_frac=0.12, upper_frac=0.22):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    if span == 0.0:
        span = max(1.0, abs(high))
    return low - lower_frac * span, high + upper_frac * span


def plot_h0_influence(rows, reference, out_pdf):
    complete = completed_rows(rows)
    x = np.asarray([row["drop_index"] for row in complete], dtype=float)
    h0 = np.asarray([row["H0_q50"] for row in complete], dtype=float)
    h0_lo = np.asarray([row["H0_q16"] for row in complete], dtype=float)
    h0_hi = np.asarray([row["H0_q84"] for row in complete], dtype=float)
    delta = np.asarray([
        row["delta_H0_q50_vs_full"] for row in complete], dtype=float)
    yerr = np.vstack([h0 - h0_lo, h0_hi - h0])

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 1, figsize=(7.2, 5.0), sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1.2, 1.0]})
        ax_h0, ax_delta = axes
        ax_h0.axhspan(
            reference["H0_q16"], reference["H0_q84"],
            color="black", alpha=0.10, lw=0, label="full 16-84")
        ax_h0.axhline(
            reference["H0_q50"], color="black", lw=0.9, ls=":",
            label=rf"full field 21 $H_0={reference['H0_q50']:.2f}$")
        ax_h0.errorbar(
            x, h0, yerr=yerr, fmt="o", color=COLOUR_H0,
            ecolor="0.55", elinewidth=0.55, capsize=1.3, ms=3.2,
            alpha=0.86)
        mark_missing(ax_h0, rows, reference["H0_q16"] - 0.25)
        ax_h0.set_ylabel(H0_LABEL)
        ax_h0.set_title(
            f"Leave-one-out H0; {len(complete)}/{len(rows)} complete",
            loc="left")
        ax_h0.legend(loc="upper right", frameon=False, handlelength=1.8)

        ax_delta.axhline(0.0, color="0.35", lw=0.8, ls="--")
        ax_delta.plot(x, delta, color=COLOUR_H0, lw=0.75, alpha=0.70)
        ax_delta.scatter(x, delta, color=COLOUR_H0, s=18, zorder=3)
        mark_missing(ax_delta, rows, 0.0)
        ax_delta.set_ylim(*padded_limits(delta))
        annotate_extremes(
            ax_delta, rows, "delta_H0_q50_vs_full", colour=COLOUR_H0)
        ax_delta.set_ylabel(r"$\Delta H_0$ vs full")
        ax_delta.set_xlabel("Dropped Cepheid host")
        label_ticks(ax_delta, rows)
        return save_pdf_png(fig, out_pdf)


def plot_evidence_influence(rows, out_pdf):
    complete = completed_rows(rows)
    x = np.asarray([row["drop_index"] for row in complete], dtype=float)
    lnz_h = np.asarray([row["lnZ_harmonic"] for row in complete], dtype=float)
    err_h = np.asarray([
        abs(row["err_lnZ_harmonic"]) for row in complete], dtype=float)
    lnz_l = np.asarray([row["lnZ_laplace"] for row in complete], dtype=float)
    err_l = np.asarray([
        abs(row["err_lnZ_laplace"]) for row in complete], dtype=float)
    med_h = float(np.median(lnz_h)) if lnz_h.size else np.nan
    med_l = float(np.median(lnz_l)) if lnz_l.size else np.nan

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 1, figsize=(7.2, 5.0), sharex=True,
            constrained_layout=True)
        for ax, y, yerr, med, ylabel, colour in (
                (axes[0], lnz_h, err_h, med_h, r"harmonic $\ln Z$",
                 COLOUR_LNZ),
                (axes[1], lnz_l, err_l, med_l, r"Laplace $\ln Z$",
                 "#87193d")):
            ax.axhline(med, color="0.35", lw=0.8, ls="--",
                       label="completed median")
            ax.errorbar(
                x, y, yerr=yerr, fmt="o", color=colour, ecolor="0.55",
                elinewidth=0.55, capsize=1.3, ms=3.2, alpha=0.86)
            mark_missing(ax, rows, med)
            ax.set_ylabel(ylabel)
            ax.legend(loc="upper right", frameon=False, handlelength=1.8)
        axes[0].set_title(
            f"Leave-one-out evidence; {len(complete)}/{len(rows)} complete",
            loc="left")
        axes[1].set_xlabel("Dropped Cepheid host")
        label_ticks(axes[1], rows)
        return save_pdf_png(fig, out_pdf)


def plot_influence_scatter(rows, out_pdf):
    complete = completed_rows(rows)
    if not complete:
        raise ValueError("No complete leave-one-out rows to plot.")
    x = np.asarray([
        row["delta_lnZ_harmonic_vs_completed_median"]
        for row in complete
    ], dtype=float)
    y = np.asarray([
        row["delta_H0_q50_vs_full"] for row in complete], dtype=float)
    colour = np.asarray([row["drop_index"] for row in complete], dtype=float)

    with plt.style.context("science"):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(3.65, 3.05), constrained_layout=True)
        ax.axhline(0.0, color="0.45", lw=0.75, ls="--")
        ax.axvline(0.0, color="0.45", lw=0.75, ls="--")
        sc = ax.scatter(
            x, y, c=colour, cmap="viridis", norm=Normalize(
                vmin=0.0, vmax=max(1.0, float(np.max(colour)))),
            s=28, edgecolor="0.15", linewidth=0.25, zorder=3)
        ax.set_ylim(*padded_limits(y, upper_frac=0.30))
        annotate_extremes(
            ax, complete, "delta_H0_q50_vs_full", n=3, colour="black")
        ax.set_xlabel(r"$\Delta \ln Z_{\rm harm}$ vs completed median")
        ax.set_ylabel(r"$\Delta H_0$ vs full")
        ax.set_title("Leave-one-out influence", loc="left")
        cbar = fig.colorbar(sc, ax=ax, pad=0.018, fraction=0.065)
        cbar.set_label("drop index")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def write_csv(rows, path):
    fieldnames = [
        "task", "drop_index", "dropped_host", "dropped_host_label",
        "status", "field", "n_hosts", "n_H0", "H0_mean", "H0_std",
        "H0_q16", "H0_q50", "H0_q84", "delta_H0_q50_vs_full",
        "delta_H0_mean_vs_full", "lnZ_harmonic", "err_lnZ_harmonic",
        "delta_lnZ_harmonic_vs_completed_median", "lnZ_laplace",
        "err_lnZ_laplace", "delta_lnZ_laplace_vs_completed_median",
        "BIC", "lnZ_bic", "source", "config",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_summary(rows, reference, path):
    complete = completed_rows(rows)
    missing = [row for row in rows if row["status"] != "complete"]
    lines = [
        "# CH0 Leave-One-Out Diagnostics",
        "",
        f"Reference: `{reference['source']}`.",
        f"Complete outputs: {len(complete)} / {len(rows)}.",
        f"Missing outputs: {len(missing)}.",
        "",
        "| quantity | value |",
        "| --- | ---: |",
        f"| full field 21 H0 median | {reference['H0_q50']:.3f} |",
        f"| full field 21 H0 mean | {reference['H0_mean']:.3f} |",
        f"| full field 21 H0 std | {reference['H0_std']:.3f} |",
        f"| full field 21 harmonic lnZ | {reference['lnZ_harmonic']:.3f} |",
    ]
    if complete:
        largest = max(
            complete, key=lambda row: abs(row["delta_H0_q50_vs_full"]))
        best_lnz = max(complete, key=lambda row: row["lnZ_harmonic"])
        worst_lnz = min(complete, key=lambda row: row["lnZ_harmonic"])
        lines.extend([
            f"| largest absolute H0 shift host | "
            f"{largest['dropped_host_label']} |",
            f"| largest absolute H0 shift | "
            f"{largest['delta_H0_q50_vs_full']:+.3f} |",
            f"| best leave-one-out harmonic lnZ host | "
            f"{best_lnz['dropped_host_label']} |",
            f"| best leave-one-out harmonic lnZ | "
            f"{best_lnz['lnZ_harmonic']:.3f} |",
            f"| worst leave-one-out harmonic lnZ host | "
            f"{worst_lnz['dropped_host_label']} |",
            f"| worst leave-one-out harmonic lnZ | "
            f"{worst_lnz['lnZ_harmonic']:.3f} |",
        ])
    if missing:
        lines.extend(["", "## Missing", ""])
        for row in missing:
            lines.append(
                f"- drop {row['drop_index']:02d} "
                f"({row['dropped_host_label']}): `{row['source']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = read_reference(args.reference)
    rows = load_rows(args.task_file, reference, args.require_complete)

    csv_path = args.output_dir / "leaveoneout_summary.csv"
    txt_path = args.output_dir / "leaveoneout_summary.txt"
    h0_pdf = args.output_dir / "leaveoneout_h0_influence.pdf"
    evidence_pdf = args.output_dir / "leaveoneout_evidence_influence.pdf"
    scatter_pdf = args.output_dir / "leaveoneout_influence_scatter.pdf"

    write_csv(rows, csv_path)
    write_summary(rows, reference, txt_path)
    written = [
        csv_path,
        txt_path,
        *plot_h0_influence(rows, reference, h0_pdf),
        *plot_evidence_influence(rows, evidence_pdf),
    ]
    if completed_rows(rows):
        written.extend(plot_influence_scatter(rows, scatter_pdf))

    for path in written:
        print(f"Wrote {path}")
    print(
        f"Complete outputs: {len(completed_rows(rows))} / {len(rows)}; "
        f"missing: {len(rows) - len(completed_rows(rows))}."
    )


if __name__ == "__main__":
    main()
