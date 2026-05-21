#!/usr/bin/env python
"""Analyse total-likelihood drivers of CH0 single-field posteriors."""
from argparse import ArgumentParser
import csv
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from candel.plotting.selection_diagnostics import (  # noqa: E402
    plot_raw_selection_evidence,
)


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
RESULTS = ROOT / "results" / "CH0_paper" / "single_fields"
DEFAULT_OUTDIR = RESULTS / "plots"
PATTERN = (
    "CH0_sel-SN_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2"
    "_field*_single.hdf5"
)
FIELD_RE = re.compile(r"_field(\d+)_")
REQUIRED = (
    "auxiliary/log_likelihood_total",
    "auxiliary/log_observed_selection_per_galaxy",
    "auxiliary/log_selection_integral",
    "gof/lnZ_harmonic",
)
FIGURE_DPI = 450


COMPONENTS = (
    (
        "raw_total_mean",
        "delta_raw_total_vs_median_field",
        "raw total likelihood",
    ),
    (
        "observed_selection_total_mean",
        "delta_observed_selection_total_vs_median_field",
        "observed selection",
    ),
    (
        "minus_log_selection_integral_total_mean",
        "delta_minus_log_selection_integral_total_vs_median_field",
        r"selection normalisation, $-\log S$",
    ),
    (
        "log_likelihood_total_mean",
        "delta_log_likelihood_total_vs_median_field",
        "total likelihood",
    ),
)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS,
        help="Directory containing CH0 single-field HDF5 outputs.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTDIR,
        help="Directory for plots and summaries.")
    parser.add_argument(
        "--pattern", default=PATTERN,
        help="Glob pattern for the single-field HDF5 files.")
    return parser.parse_args()


def field_index(path):
    match = FIELD_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse field index from `{path}`.")
    return int(match.group(1))


def read_scalar(handle, name, default=np.nan):
    if name not in handle:
        return default
    return float(handle[name][()])


def finite_samples(handle, name):
    samples = np.asarray(handle[f"samples/{name}"], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return np.asarray([np.nan])
    return samples


def h0_summary(handle):
    h0 = finite_samples(handle, "H0")
    q16, q50, q84 = np.nanpercentile(h0, [16.0, 50.0, 84.0])
    return {
        "H0_mean": float(np.nanmean(h0)),
        "H0_std": float(np.nanstd(h0, ddof=1)),
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


def read_rows(results_dir, pattern):
    paths = sorted(results_dir.glob(pattern), key=field_index)
    if not paths:
        raise FileNotFoundError(
            f"No files matching `{results_dir / pattern}`.")

    rows = []
    skipped = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            missing = [name for name in REQUIRED if name not in handle]
            if missing:
                skipped.append((path, missing))
                continue

            ll_total = np.asarray(
                handle["auxiliary/log_likelihood_total"], dtype=float)
            obs = np.asarray(
                handle["auxiliary/log_observed_selection_per_galaxy"],
                dtype=float)
            log_s = np.asarray(
                handle["auxiliary/log_selection_integral"], dtype=float)
            if obs.ndim != 2:
                raise ValueError(
                    f"Unexpected observed-selection shape {obs.shape} in "
                    f"`{path}`.")
            n_hosts = obs.shape[1]
            observed_total = np.sum(obs, axis=1)
            minus_log_s_total = selection_integral_total(log_s, n_hosts)
            raw_total = ll_total - observed_total - minus_log_s_total

            rows.append({
                "field": field_index(path),
                "source": str(path),
                "n_samples": int(ll_total.size),
                "n_hosts": int(n_hosts),
                "lnZ_harmonic": read_scalar(handle, "gof/lnZ_harmonic"),
                "err_lnZ_harmonic": read_scalar(
                    handle, "gof/err_lnZ_harmonic"),
                "lnZ_laplace": read_scalar(handle, "gof/lnZ_laplace"),
                "err_lnZ_laplace": read_scalar(
                    handle, "gof/err_lnZ_laplace"),
                "BIC": read_scalar(handle, "gof/BIC"),
                "AIC": read_scalar(handle, "gof/AIC"),
                "log_density_mean": (
                    float(np.mean(handle["log_density"]))
                    if "log_density" in handle else np.nan),
                "raw_total_mean": float(np.mean(raw_total)),
                "raw_total_std": float(np.std(raw_total, ddof=1)),
                "observed_selection_total_mean": float(
                    np.mean(observed_total)),
                "observed_selection_total_std": float(
                    np.std(observed_total, ddof=1)),
                "minus_log_selection_integral_total_mean": float(
                    np.mean(minus_log_s_total)),
                "minus_log_selection_integral_total_std": float(
                    np.std(minus_log_s_total, ddof=1)),
                "minus_log_selection_integral_per_host_mean": float(
                    np.mean(minus_log_s_total) / n_hosts),
                "log_likelihood_total_mean": float(np.mean(ll_total)),
                "log_likelihood_total_std": float(np.std(ll_total, ddof=1)),
                **h0_summary(handle),
            })

    if not rows:
        raise ValueError("No usable files had the required auxiliary datasets.")
    add_reference_deltas(rows)
    return rows, skipped


def add_reference_deltas(rows):
    for key, delta_key, _ in COMPONENTS:
        ref = float(np.median([row[key] for row in rows]))
        for row in rows:
            row[f"{key}_median_field_reference"] = ref
            row[delta_key] = float(row[key] - ref)


def best_by(rows, key, maximise=True):
    finite = [row for row in rows if np.isfinite(row[key])]
    if not finite:
        raise ValueError(f"No finite `{key}` values.")
    return (max if maximise else min)(finite, key=lambda row: row[key])


def write_csv(rows, path):
    skip = set()
    fieldnames = [key for key in rows[0] if key not in skip]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_pdf_png(fig, out_pdf):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_pdf, out_png


def plot_component_decomposition(rows, out_pdf):
    best = best_by(rows, "lnZ_harmonic")
    best_s = best_by(rows, "minus_log_selection_integral_total_mean")
    labels = [label for _, _, label in COMPONENTS]
    deltas_best = [best[delta_key] for _, delta_key, _ in COMPONENTS]
    deltas_sel = [best_s[delta_key] for _, delta_key, _ in COMPONENTS]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    ax.bar(
        x - width / 2, deltas_best, width,
        label=f"best lnZ field {best['field']}", color="#3b76af")
    ax.bar(
        x + width / 2, deltas_sel, width,
        label=f"max -logS field {best_s['field']}", color="#c9584a")
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Delta log contribution vs median field")
    ax.set_title("CH0 total-likelihood decomposition", loc="left")
    ax.legend(frameon=False)
    return save_pdf_png(fig, out_pdf)


def plot_raw_vs_selection(rows, out_pdf):
    fields = np.asarray([row["field"] for row in rows], dtype=float)
    raw = np.asarray([row["raw_total_mean"] for row in rows], dtype=float)
    selection = np.asarray([
        row["minus_log_selection_integral_total_mean"]
        for row in rows
    ], dtype=float)
    lnz = np.asarray([row["lnZ_harmonic"] for row in rows], dtype=float)
    total = np.asarray([
        row["log_likelihood_total_mean"] for row in rows], dtype=float)
    fig, _ = plot_raw_selection_evidence(
        raw, selection, lnz, fields, total_likelihood=total)

    return save_pdf_png(fig, out_pdf)


def plot_driver_scatter(rows, out_pdf):
    lnz = np.asarray([row["lnZ_harmonic"] for row in rows], dtype=float)
    fields = np.asarray([row["field"] for row in rows], dtype=int)
    best = best_by(rows, "lnZ_harmonic")
    best_s = best_by(rows, "minus_log_selection_integral_total_mean")

    specs = [
        ("raw_total_mean", "raw total likelihood"),
        ("observed_selection_total_mean", "observed selection"),
        ("minus_log_selection_integral_total_mean", r"selection $-\log S$"),
        ("log_likelihood_total_mean", "total likelihood"),
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    for ax, (key, xlabel) in zip(axes.ravel(), specs):
        x = np.asarray([row[key] for row in rows], dtype=float)
        ax.scatter(x, lnz, s=24, color="#3b76af", alpha=0.8)
        for row, marker, colour in (
                (best, "o", "black"), (best_s, "s", "#c9584a")):
            ax.scatter(
                row[key], row["lnZ_harmonic"], s=46, marker=marker,
                color=colour, edgecolor="white", linewidth=0.5, zorder=4)
            ax.annotate(
                f"{row['field']}", (row[key], row["lnZ_harmonic"]),
                xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("harmonic lnZ")
        ax.set_title(corr_label(x, lnz), loc="left")
    return save_pdf_png(fig, out_pdf)


def write_summary(rows, skipped, path):
    best = best_by(rows, "lnZ_harmonic")
    best_laplace = best_by(rows, "lnZ_laplace")
    best_bic = best_by(rows, "BIC", maximise=False)
    best_aic = best_by(rows, "AIC", maximise=False)
    best_s = best_by(rows, "minus_log_selection_integral_total_mean")

    lines = [
        "# CH0 Manticore Evidence-Driver Study",
        "",
        f"Usable fields: {len(rows)}.",
        f"Skipped files without auxiliary likelihood tracking: {len(skipped)}.",
        "",
        "## Best Fields",
        "",
        "| proxy | best field | value |",
        "| --- | ---: | ---: |",
        f"| harmonic lnZ | {best['field']} | {best['lnZ_harmonic']:.3f} |",
        f"| Laplace lnZ | {best_laplace['field']} | "
        f"{best_laplace['lnZ_laplace']:.3f} |",
        f"| BIC | {best_bic['field']} | {best_bic['BIC']:.3f} |",
        f"| AIC | {best_aic['field']} | {best_aic['AIC']:.3f} |",
        "",
        "## Best Harmonic-Evidence Field",
        "",
        f"Field {best['field']} has "
        f"`-logS = {best['minus_log_selection_integral_per_host_mean']:.4f}` "
        "per host.",
        "The observed-selection term is constant across the single-field "
        "runs for this SN-magnitude-selection setup.",
        "",
        "| contribution | total delta vs median field |",
        "| --- | ---: |",
    ]
    for _, delta_key, label in COMPONENTS:
        lines.append(f"| {label} | {best[delta_key]:+.3f} |")

    lines.extend([
        "",
        "## Most Favourable Selection-Normalisation Field",
        "",
        f"Field {best_s['field']} has "
        f"`-logS = {best_s['minus_log_selection_integral_per_host_mean']:.4f}` "
        "per host.",
        "",
        "| contribution | total delta vs median field |",
        "| --- | ---: |",
    ])
    for _, delta_key, label in COMPONENTS:
        lines.append(f"| {label} | {best_s[delta_key]:+.3f} |")

    if skipped:
        lines.extend(["", "## Skipped Files", ""])
        for path_skipped, missing in skipped:
            lines.append(f"- `{path_skipped}`: missing {', '.join(missing)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    rows, skipped = read_rows(args.results_dir, args.pattern)

    csv_path = args.output_dir / "ch0_manticore_evidence_driver_summary.csv"
    txt_path = args.output_dir / "ch0_manticore_evidence_driver_summary.txt"
    raw_selection_pdf = (
        args.output_dir / "ch0_manticore_raw_likelihood_vs_selection.pdf")

    write_csv(rows, csv_path)
    write_summary(rows, skipped, txt_path)
    raw_selection_pdf, raw_selection_png = plot_raw_vs_selection(
        rows, raw_selection_pdf)

    print(f"wrote {csv_path}")
    print(f"wrote {txt_path}")
    print(f"wrote {raw_selection_pdf}")
    print(f"wrote {raw_selection_png}")


if __name__ == "__main__":
    main()
