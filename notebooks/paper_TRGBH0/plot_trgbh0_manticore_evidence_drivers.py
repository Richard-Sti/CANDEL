#!/usr/bin/env python
"""Analyse which galaxies drive TRGBH0 single-field evidence differences."""
from argparse import ArgumentParser
import csv
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: F401
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from trgbh0_plot_style import TRGBH0_COLOURS  # noqa: E402


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
RESULTS = ROOT / "results" / "TRGBH0_paper" / "manticore_fields_const_sigv"
DEFAULT_OUTDIR = RESULTS / "plots"
FIELD_RE = re.compile(r"_field(\d+)_")
PATTERNS = {
    "all": "EDD_TRGB_sel-TRGB_magnitude_*manticore_2MPP_MULTIBIN_N256_DES_V2_field*_manticore_field_const_sigv.hdf5",
    "cola": "EDD_TRGB_sel-TRGB_magnitude_COLA_manticore_2MPP_MULTIBIN_N256_DES_V2_field*_manticore_field_const_sigv.hdf5",
    "non-cola": "EDD_TRGB_sel-TRGB_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_field*_manticore_field_const_sigv.hdf5",
}
REQUIRED_AUX = (
    "auxiliary/log_likelihood_per_galaxy",
    "auxiliary/log_observed_selection_per_galaxy",
    "auxiliary/log_selection_integral",
    "auxiliary/log_likelihood_per_galaxy_with_selection",
    "auxiliary/host_names",
    "gof/lnZ_harmonic",
)
FIGURE_DPI = 500


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="Directory containing TRGBH0 single-field HDF5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory for plot outputs.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="Directory for CSV and text summaries. Default: --output-dir.",
    )
    parser.add_argument(
        "--field-set",
        choices=sorted(PATTERNS),
        default="cola",
        help="Which Manticore field set to analyse.",
    )
    parser.add_argument(
        "--top-galaxies",
        type=int,
        default=20,
        help="Number of top galaxies to list and plot for the best field.",
    )
    parser.add_argument(
        "--heatmap-galaxies",
        type=int,
        default=40,
        help="Number of galaxies to include in the likelihood-difference heatmap.",
    )
    return parser.parse_args()


def field_index(path):
    match = FIELD_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse field index from `{path}`.")
    return int(match.group(1))


def field_set_label(path):
    return "COLA" if "_COLA_manticore_" in path.name else "non-COLA"


def decode_names(raw):
    names = []
    for item in raw:
        if isinstance(item, bytes):
            names.append(item.decode("utf-8"))
        else:
            names.append(str(item))
    return np.asarray(names)


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


def load_rows(results_dir, field_set):
    paths = sorted(results_dir.glob(PATTERNS[field_set]), key=field_index)
    if not paths:
        raise FileNotFoundError(
            f"No files matching `{results_dir / PATTERNS[field_set]}`.")

    rows = []
    skipped = []
    reference_names = None
    for path in paths:
        with h5py.File(path, "r") as handle:
            missing = [name for name in REQUIRED_AUX if name not in handle]
            if missing:
                skipped.append((path, missing))
                continue

            names = decode_names(handle["auxiliary/host_names"][()])
            ll = np.asarray(
                handle["auxiliary/log_likelihood_per_galaxy"], dtype=float)
            obs_sel = np.asarray(
                handle["auxiliary/log_observed_selection_per_galaxy"],
                dtype=float)
            log_s = np.asarray(
                handle["auxiliary/log_selection_integral"], dtype=float)
            full = np.asarray(
                handle["auxiliary/log_likelihood_per_galaxy_with_selection"],
                dtype=float)

            if reference_names is None:
                reference_names = names
            elif not np.array_equal(names, reference_names):
                raise ValueError(
                    f"Host-name order in `{path}` differs from the first "
                    "usable file. Align before comparing per-galaxy arrays.")

            n_gal = ll.shape[1]
            ll_gal_mean = np.mean(ll, axis=0)
            obs_gal_mean = np.mean(obs_sel, axis=0)
            full_gal_mean = np.mean(full, axis=0)
            log_s_mean = float(np.mean(log_s))

            row = {
                "field_set": field_set_label(path),
                "field": field_index(path),
                "source": str(path),
                "n_samples": int(ll.shape[0]),
                "n_galaxies": int(n_gal),
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
                    if "log_density" in handle else np.nan
                ),
                "log_density_max": (
                    float(np.max(handle["log_density"]))
                    if "log_density" in handle else np.nan
                ),
                "galaxy_ll_mean": ll_gal_mean,
                "galaxy_observed_selection_mean": obs_gal_mean,
                "galaxy_full_mean": full_gal_mean,
                "log_selection_integral_mean": log_s_mean,
                "ll_total_mean": float(np.sum(ll_gal_mean)),
                "ll_mean_per_galaxy": float(np.mean(ll_gal_mean)),
                "observed_selection_total_mean": float(np.sum(obs_gal_mean)),
                "observed_selection_mean_per_galaxy": float(np.mean(obs_gal_mean)),
                "minus_log_selection_integral_mean": -log_s_mean,
                "minus_log_selection_integral_total_mean": float(-n_gal * log_s_mean),
                "full_total_mean": float(np.sum(full_gal_mean)),
                "full_mean_per_galaxy": float(np.mean(full_gal_mean)),
                **h0_summary(handle),
            }
            rows.append(row)

    if not rows:
        raise ValueError("No usable files had the required auxiliary datasets.")

    add_reference_delta_metrics(rows)
    return rows, skipped, reference_names


def add_reference_delta_metrics(rows):
    ll_matrix = np.vstack([row["galaxy_ll_mean"] for row in rows])
    obs_matrix = np.vstack([
        row["galaxy_observed_selection_mean"] for row in rows])
    full_matrix = np.vstack([row["galaxy_full_mean"] for row in rows])
    minus_log_s = np.asarray([
        row["minus_log_selection_integral_mean"] for row in rows])

    ref_ll = np.median(ll_matrix, axis=0)
    ref_obs = np.median(obs_matrix, axis=0)
    ref_full = np.median(full_matrix, axis=0)
    ref_minus_log_s = float(np.median(minus_log_s))

    for row in rows:
        delta_ll = row["galaxy_ll_mean"] - ref_ll
        delta_obs = row["galaxy_observed_selection_mean"] - ref_obs
        delta_full = row["galaxy_full_mean"] - ref_full
        pos = np.sort(delta_ll[delta_ll > 0.0])[::-1]
        pos_sum = float(np.sum(pos))
        row["delta_ll_total_vs_median_field"] = float(np.sum(delta_ll))
        row["delta_observed_selection_total_vs_median_field"] = float(
            np.sum(delta_obs))
        row["delta_minus_log_selection_integral_total_vs_median_field"] = (
            float(row["n_galaxies"] * (
                row["minus_log_selection_integral_mean"]
                - ref_minus_log_s)))
        row["delta_full_total_vs_median_field"] = float(np.sum(delta_full))
        row["delta_ll_mean_per_galaxy_vs_median_field"] = float(
            np.mean(delta_ll))
        row["delta_full_mean_per_galaxy_vs_median_field"] = float(
            np.mean(delta_full))
        row["delta_ll_max_galaxy_vs_median_field"] = float(np.max(delta_ll))
        row["delta_ll_min_galaxy_vs_median_field"] = float(np.min(delta_ll))
        row["delta_ll_positive_sum_vs_median_field"] = pos_sum
        row["delta_ll_top5_positive_share"] = (
            float(np.sum(pos[:5]) / pos_sum) if pos_sum > 0 else np.nan)
        row["delta_ll_top10_positive_share"] = (
            float(np.sum(pos[:10]) / pos_sum) if pos_sum > 0 else np.nan)
        row["delta_ll_ngal_gt_0p1"] = int(np.sum(delta_ll > 0.1))
        row["delta_ll_ngal_gt_0p25"] = int(np.sum(delta_ll > 0.25))
        row["delta_ll_ngal_gt_0p5"] = int(np.sum(delta_ll > 0.5))


def write_field_summary(rows, path):
    skip_fields = {
        "source", "galaxy_ll_mean", "galaxy_observed_selection_mean",
        "galaxy_full_mean",
    }
    fieldnames = [
        key for key in rows[0].keys()
        if key not in skip_fields
    ] + ["source"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def best_row(rows):
    finite = [row for row in rows if np.isfinite(row["lnZ_harmonic"])]
    if not finite:
        raise ValueError("No finite harmonic lnZ values.")
    return max(finite, key=lambda row: row["lnZ_harmonic"])


def best_galaxy_rows(rows, host_names, best, top_n):
    ll_matrix = np.vstack([row["galaxy_ll_mean"] for row in rows])
    obs_matrix = np.vstack([
        row["galaxy_observed_selection_mean"] for row in rows])
    full_matrix = np.vstack([row["galaxy_full_mean"] for row in rows])
    ref_ll = np.median(ll_matrix, axis=0)
    ref_obs = np.median(obs_matrix, axis=0)
    ref_full = np.median(full_matrix, axis=0)

    delta_ll = best["galaxy_ll_mean"] - ref_ll
    delta_obs = best["galaxy_observed_selection_mean"] - ref_obs
    delta_full = best["galaxy_full_mean"] - ref_full
    order = np.argsort(delta_ll)[::-1]
    out = []
    for rank, idx in enumerate(order, start=1):
        out.append({
            "rank": rank,
            "host_name": host_names[idx],
            "delta_log_likelihood": float(delta_ll[idx]),
            "delta_log_observed_selection": float(delta_obs[idx]),
            "delta_log_likelihood_with_selection": float(delta_full[idx]),
            "best_log_likelihood": float(best["galaxy_ll_mean"][idx]),
            "median_field_log_likelihood": float(ref_ll[idx]),
            "best_log_observed_selection": float(
                best["galaxy_observed_selection_mean"][idx]),
            "median_field_log_observed_selection": float(ref_obs[idx]),
        })
    return out[:top_n], out


def write_best_galaxies(rows, host_names, best, path):
    fieldnames = [
        "rank", "host_name", "delta_log_likelihood",
        "delta_log_observed_selection",
        "delta_log_likelihood_with_selection", "best_log_likelihood",
        "median_field_log_likelihood", "best_log_observed_selection",
        "median_field_log_observed_selection",
    ]
    _, all_rows = best_galaxy_rows(rows, host_names, best, len(host_names))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def set_paper_rc():
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.4,
    })


def p_label(value):
    if not np.isfinite(value):
        return r"=\mathrm{nan}"
    if value < 1e-3:
        return r"<10^{-3}"
    return rf"={value:.2f}"


def correlation_text(x, y):
    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) < 3:
        return "insufficient finite points"
    r, p = pearsonr(x[finite], y[finite])
    rho, p_s = spearmanr(x[finite], y[finite])
    return (
        rf"$r={r:.2f}$, $p{p_label(p)}$" "\n"
        rf"$\rho={rho:.2f}$, $p{p_label(p_s)}$"
    )


def annotate_best(ax, x, y, fields, best_field):
    idx = np.where(fields == best_field)[0]
    if idx.size:
        ax.annotate(
            f"field {best_field}",
            (x[idx[0]], y[idx[0]]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.2,
            color="0.1",
        )


def save_pdf_png(fig, out_pdf):
    fig.savefig(out_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_driver_scatter(rows, best, out_pdf):
    lnz = np.asarray([row["lnZ_harmonic"] for row in rows], dtype=float)
    fields = np.asarray([row["field"] for row in rows], dtype=int)
    specs = [
        (
            "ll_mean_per_galaxy",
            r"$\langle \log \mathcal{L}_{m,cz}\rangle_{\rm gal}$",
            "Magnitude/redshift likelihood",
        ),
        (
            "observed_selection_mean_per_galaxy",
            r"$\langle \log p({\rm observed})\rangle_{\rm gal}$",
            "Observed-selection probability",
        ),
        (
            "minus_log_selection_integral_mean",
            r"$-\log S$",
            "Selection integral",
        ),
        (
            "delta_ll_top10_positive_share",
            "Top-10 positive likelihood share",
            "Outlier concentration",
        ),
    ]

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 2, figsize=(7.2, 5.2), constrained_layout=True)
        for ax, (key, xlabel, title) in zip(axes.ravel(), specs):
            x = np.asarray([row[key] for row in rows], dtype=float)
            ax.scatter(
                x, lnz, s=18, color=TRGBH0_COLOURS[1],
                alpha=0.78, edgecolor="none")
            annotate_best(ax, x, lnz, fields, best["field"])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"Harmonic $\ln Z$")
            ax.set_title(title, loc="left")
            ax.text(
                0.04,
                0.96,
                correlation_text(x, lnz),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.4,
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
            )
    return save_pdf_png(fig, out_pdf)


def plot_best_galaxy_deltas(rows, host_names, best, top_n, out_pdf):
    top_rows, all_rows = best_galaxy_rows(rows, host_names, best, top_n)
    delta = np.asarray([row["delta_log_likelihood"] for row in all_rows])
    order = np.arange(delta.size)
    top = top_rows[::-1]

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 2, figsize=(7.2, 3.1), constrained_layout=True,
            gridspec_kw={"width_ratios": [1.25, 1.0]})
        ax_rank, ax_bar = axes

        ax_rank.plot(order + 1, delta, color=TRGBH0_COLOURS[1], lw=0.85)
        ax_rank.axhline(0.0, color="0.3", lw=0.7, ls=":")
        ax_rank.set_xlabel("Galaxy rank in best-field improvement")
        ax_rank.set_ylabel(
            r"$\Delta\langle\log \mathcal{L}_{m,cz}\rangle$")
        ax_rank.set_title(
            f"Best field {best['field']} minus median field", loc="left")
        for row in top_rows[:6]:
            rank = int(row["rank"])
            ax_rank.annotate(
                row["host_name"],
                (rank, row["delta_log_likelihood"]),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=5.4,
                alpha=0.85,
            )

        labels = [row["host_name"] for row in top]
        values = np.asarray([row["delta_log_likelihood"] for row in top])
        ypos = np.arange(len(top))
        ax_bar.barh(ypos, values, color=TRGBH0_COLOURS[0], alpha=0.78)
        ax_bar.set_yticks(ypos)
        ax_bar.set_yticklabels(labels, fontsize=5.7)
        ax_bar.axvline(0.0, color="0.3", lw=0.7)
        ax_bar.set_xlabel(
            r"$\Delta\langle\log \mathcal{L}_{m,cz}\rangle$")
        ax_bar.set_title(f"Top {top_n} positive galaxies", loc="left")
    return save_pdf_png(fig, out_pdf)


def plot_delta_heatmap(rows, host_names, best, n_galaxies, out_pdf):
    rows_ordered = sorted(rows, key=lambda row: row["lnZ_harmonic"])
    ll_matrix = np.vstack([row["galaxy_ll_mean"] for row in rows_ordered])
    ref = np.median(np.vstack([row["galaxy_ll_mean"] for row in rows]), axis=0)
    delta = ll_matrix - ref[None, :]

    best_delta = best["galaxy_ll_mean"] - ref
    gal_idx = np.argsort(np.abs(best_delta))[::-1][:n_galaxies]
    delta = delta[:, gal_idx]

    vmax = np.nanpercentile(np.abs(delta), 98.0)
    vmax = max(vmax, 1e-6)

    with plt.style.context("science"):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(7.2, 5.3), constrained_layout=True)
        im = ax.imshow(
            delta,
            aspect="auto",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xlabel("Galaxy")
        ax.set_ylabel("Field, sorted by harmonic evidence")
        ax.set_title(
            r"$\Delta\langle\log \mathcal{L}_{m,cz}\rangle$ "
            "relative to median field",
            loc="left",
        )
        ax.set_xticks(np.arange(len(gal_idx)))
        ax.set_xticklabels(host_names[gal_idx], rotation=90, fontsize=4.7)
        ax.set_yticks(np.arange(len(rows_ordered)))
        ax.set_yticklabels(
            [str(row["field"]) for row in rows_ordered], fontsize=5.3)
        cbar = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.04)
        cbar.set_label(
            r"$\Delta\langle\log \mathcal{L}_{m,cz}\rangle$")
    return save_pdf_png(fig, out_pdf)


def plot_best_field_components(rows, best, out_pdf):
    ll_matrix = np.vstack([row["galaxy_ll_mean"] for row in rows])
    obs_matrix = np.vstack([
        row["galaxy_observed_selection_mean"] for row in rows])
    full_matrix = np.vstack([row["galaxy_full_mean"] for row in rows])
    ref_ll = np.median(ll_matrix, axis=0)
    ref_obs = np.median(obs_matrix, axis=0)
    ref_full = np.median(full_matrix, axis=0)
    ref_minus_log_s = float(np.median([
        row["minus_log_selection_integral_mean"] for row in rows]))

    delta_ll = best["galaxy_ll_mean"] - ref_ll
    delta_obs = best["galaxy_observed_selection_mean"] - ref_obs
    delta_full = best["galaxy_full_mean"] - ref_full
    delta_minus_log_s = (
        best["minus_log_selection_integral_mean"] - ref_minus_log_s)
    n_gal = best["n_galaxies"]
    totals = [
        np.sum(delta_ll),
        np.sum(delta_obs),
        n_gal * delta_minus_log_s,
        np.sum(delta_full),
    ]
    labels = [
        r"$\log\mathcal{L}_{m,cz}$",
        r"$\log p_{\rm obs}$",
        r"$-\log S$",
        "total",
    ]

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 2, figsize=(7.2, 3.1), constrained_layout=True)
        ax_bar, ax_hist = axes

        colors = [
            TRGBH0_COLOURS[1],
            TRGBH0_COLOURS[2],
            TRGBH0_COLOURS[3],
            TRGBH0_COLOURS[0],
        ]
        xpos = np.arange(len(totals))
        ax_bar.bar(xpos, totals, color=colors, alpha=0.82)
        ax_bar.axhline(0.0, color="0.3", lw=0.75)
        ax_bar.set_xticks(xpos)
        ax_bar.set_xticklabels(labels, rotation=25, ha="right")
        ax_bar.set_ylabel("Total delta relative to median field")
        ax_bar.set_title(f"Best field {best['field']} decomposition", loc="left")

        hist_sets = [
            (delta_ll, r"$\log\mathcal{L}_{m,cz}$", TRGBH0_COLOURS[1]),
            (delta_obs, r"$\log p_{\rm obs}$", TRGBH0_COLOURS[2]),
            (
                np.full_like(delta_ll, delta_minus_log_s),
                r"$-\log S$",
                TRGBH0_COLOURS[3],
            ),
            (delta_full, "total", TRGBH0_COLOURS[0]),
        ]
        all_values = np.concatenate([item[0] for item in hist_sets])
        lo, hi = np.nanpercentile(all_values, [0.5, 99.5])
        lo = min(lo, -0.5)
        hi = max(hi, 0.8)
        bins = np.linspace(lo, hi, 60)
        for values, label, color in hist_sets:
            ax_hist.hist(
                values,
                bins=bins,
                histtype="step",
                lw=1.0,
                color=color,
                label=label,
            )
        ax_hist.axvline(0.0, color="0.3", lw=0.75, ls=":")
        ax_hist.set_xlabel("Per-galaxy delta relative to median field")
        ax_hist.set_ylabel("Number of galaxies")
        ax_hist.set_title("Per-galaxy delta distributions", loc="left")
        ax_hist.legend(frameon=False, loc="upper right")
    return save_pdf_png(fig, out_pdf)


def write_text_summary(rows, skipped, best, path):
    sorted_rows = sorted(
        [row for row in rows if np.isfinite(row["lnZ_harmonic"])],
        key=lambda row: row["lnZ_harmonic"],
        reverse=True,
    )
    median_lnz = float(np.median([row["lnZ_harmonic"] for row in sorted_rows]))
    next_best = sorted_rows[1] if len(sorted_rows) > 1 else None
    best_laplace = max(rows, key=lambda row: row["lnZ_laplace"])
    best_bic = min(rows, key=lambda row: row["BIC"])
    best_aic = min(rows, key=lambda row: row["AIC"])

    lines = [
        f"Usable fields: {len(rows)}",
        f"Skipped files without required diagnostics: {len(skipped)}",
        "",
        (
            f"Best harmonic evidence: field {best['field']} "
            f"({best['field_set']}), lnZ={best['lnZ_harmonic']:.3f}"
        ),
        f"Median harmonic lnZ: {median_lnz:.3f}",
        (
            f"Delta lnZ(best - median): "
            f"{best['lnZ_harmonic'] - median_lnz:.3f}"
        ),
    ]
    if next_best is not None:
        lines.append(
            f"Delta lnZ(best - next best field {next_best['field']}): "
            f"{best['lnZ_harmonic'] - next_best['lnZ_harmonic']:.3f}")
    lines.extend([
        "",
        "Best fields by evidence proxy:",
        (
            f"  harmonic: field {best['field']:02d}, "
            f"lnZ={best['lnZ_harmonic']:.3f}"
        ),
        (
            f"  Laplace: field {best_laplace['field']:02d}, "
            f"lnZ={best_laplace['lnZ_laplace']:.3f}"
        ),
        (
            f"  BIC: field {best_bic['field']:02d}, "
            f"BIC={best_bic['BIC']:.3f}"
        ),
        (
            f"  AIC: field {best_aic['field']:02d}, "
            f"AIC={best_aic['AIC']:.3f}"
        ),
        "",
        "Best-field decomposition relative to the median field:",
        (
            "  total magnitude/redshift likelihood delta = "
            f"{best['delta_ll_total_vs_median_field']:.3f} "
            f"({best['delta_ll_mean_per_galaxy_vs_median_field']:.4f} per galaxy)"
        ),
        (
            "  total observed-selection delta = "
            f"{best['delta_observed_selection_total_vs_median_field']:.3f}"
        ),
        (
            "  total selection-integral delta = "
            f"{best['delta_minus_log_selection_integral_total_vs_median_field']:.3f}"
        ),
        (
            "  total with-selection host-likelihood delta = "
            f"{best['delta_full_total_vs_median_field']:.3f}"
        ),
        "",
        "Best-field outlier concentration:",
        (
            "  top 5 galaxies explain "
            f"{100 * best['delta_ll_top5_positive_share']:.1f}% "
            "of positive magnitude/redshift likelihood improvements"
        ),
        (
            "  top 10 galaxies explain "
            f"{100 * best['delta_ll_top10_positive_share']:.1f}% "
            "of positive magnitude/redshift likelihood improvements"
        ),
        (
            "  number of galaxies with delta logL > 0.25: "
            f"{best['delta_ll_ngal_gt_0p25']}"
        ),
        "",
        "Field-level correlations with harmonic lnZ:",
    ])
    lnz = np.asarray([row["lnZ_harmonic"] for row in rows], dtype=float)
    for key, label in [
        ("ll_mean_per_galaxy", "mean magnitude/redshift logL per galaxy"),
        (
            "observed_selection_mean_per_galaxy",
            "mean observed-selection logp per galaxy",
        ),
        ("minus_log_selection_integral_mean", "selection-integral term"),
        (
            "full_mean_per_galaxy",
            "with-selection host log-likelihood per galaxy",
        ),
        ("delta_ll_top10_positive_share", "top-10 positive likelihood share"),
    ]:
        x = np.asarray([row[key] for row in rows], dtype=float)
        finite = np.isfinite(lnz) & np.isfinite(x)
        if np.sum(finite) < 3:
            continue
        r, _ = pearsonr(lnz[finite], x[finite])
        rho, _ = spearmanr(lnz[finite], x[finite])
        lines.append(f"  {label}: r={r:.3f}, rho={rho:.3f}")
    lines.extend([
        "",
        "Top harmonic-evidence fields:",
    ])
    for row in sorted_rows[:8]:
        lines.append(
            f"  field {row['field']:02d}: lnZ={row['lnZ_harmonic']:.3f}, "
            f"mean logL/gal={row['ll_mean_per_galaxy']:.4f}, "
            f"-logS={row['minus_log_selection_integral_mean']:.4f}")
    path.write_text("\n".join(lines) + "\n")


def run_analysis(
    results_dir=RESULTS,
    output_dir=DEFAULT_OUTDIR,
    summary_dir=None,
    field_set="cola",
    top_galaxies=20,
    heatmap_galaxies=40,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir if summary_dir is None else summary_dir
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows, skipped, host_names = load_rows(results_dir, field_set)
    best = best_row(rows)

    suffix = f"_{field_set.replace('-', '_')}"
    field_csv = summary_dir / (
        f"trgbh0_manticore_evidence_driver_fields{suffix}.csv")
    galaxy_csv = summary_dir / (
        f"trgbh0_manticore_evidence_driver_best_galaxies{suffix}.csv")
    summary_txt = summary_dir / (
        f"trgbh0_manticore_evidence_driver_summary{suffix}.txt")
    write_field_summary(rows, field_csv)
    write_best_galaxies(rows, host_names, best, galaxy_csv)
    write_text_summary(rows, skipped, best, summary_txt)

    scatter_pdf = (
        output_dir / f"trgbh0_manticore_evidence_driver_metrics{suffix}.pdf")
    scatter_png = plot_driver_scatter(
        rows, best, scatter_pdf)
    deltas_pdf = (
        output_dir / f"trgbh0_manticore_best_field_galaxy_deltas{suffix}.pdf")
    deltas_png = plot_best_galaxy_deltas(
        rows, host_names, best, top_galaxies, deltas_pdf)
    heatmap_pdf = (
        output_dir / f"trgbh0_manticore_galaxy_likelihood_heatmap{suffix}.pdf")
    heatmap_png = plot_delta_heatmap(
        rows, host_names, best, heatmap_galaxies, heatmap_pdf)
    components_pdf = (
        output_dir / f"trgbh0_manticore_best_field_components{suffix}.pdf")
    components_png = plot_best_field_components(
        rows, best, components_pdf)

    print(f"Usable fields: {len(rows)}")
    print(f"Skipped files without required diagnostics: {len(skipped)}")
    print(
        f"Best field: {best['field']} ({best['field_set']}), "
        f"lnZ_harmonic={best['lnZ_harmonic']:.3f}"
    )
    print(
        "Best vs median-field decomposition: "
        f"d_logL={best['delta_ll_total_vs_median_field']:.3f}, "
        f"d_obs_sel={best['delta_observed_selection_total_vs_median_field']:.3f}, "
        f"d_minus_logS={best['delta_minus_log_selection_integral_total_vs_median_field']:.3f}, "
        f"d_full={best['delta_full_total_vs_median_field']:.3f}"
    )
    print(
        "Outlier concentration: "
        f"top5={100 * best['delta_ll_top5_positive_share']:.1f}%, "
        f"top10={100 * best['delta_ll_top10_positive_share']:.1f}%"
    )
    print(f"Wrote {field_csv}")
    print(f"Wrote {galaxy_csv}")
    print(f"Wrote {summary_txt}")
    print(f"Wrote {scatter_png}")
    print(f"Wrote {deltas_png}")
    print(f"Wrote {heatmap_png}")
    print(f"Wrote {components_png}")
    return {
        "field_csv": field_csv,
        "galaxy_csv": galaxy_csv,
        "summary_txt": summary_txt,
        "scatter_pdf": scatter_pdf,
        "scatter_png": scatter_png,
        "deltas_pdf": deltas_pdf,
        "deltas_png": deltas_png,
        "heatmap_pdf": heatmap_pdf,
        "heatmap_png": heatmap_png,
        "components_pdf": components_pdf,
        "components_png": components_png,
    }


def main():
    args = parse_args()
    run_analysis(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        summary_dir=args.summary_dir,
        field_set=args.field_set,
        top_galaxies=args.top_galaxies,
        heatmap_galaxies=args.heatmap_galaxies,
    )


if __name__ == "__main__":
    main()
