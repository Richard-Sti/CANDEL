#!/usr/bin/env python
"""Diagnostics for the TRGBH0 single-field smoothed model-variant grid.

Discovers the run sets directly from a results directory (one HDF5 per
Manticore field) instead of a task file, groups by model variant, and makes
the usual single-field diagnostics: field-median H0 distributions, stacked
posteriors, evidence distributions, H0-vs-lnZ, bias-parameter distributions,
and matched-field traces across variants.
"""

import csv
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import h5py
import matplotlib

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: F401,E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy.stats import gaussian_kde, pearsonr, spearmanr  # noqa: E402
from trgbh0_plot_style import (OUTPUT_DIR, ROOT, save_pdf_png,  # noqa: E402
                               set_paper_rc, trgbh0_cmap)

DEFAULT_RESULTS_DIR = (
    ROOT / "results" / "TRGBH0_paper" / "single_fields_smoothed")
DEFAULT_OUTDIR = OUTPUT_DIR / "trgbh0_single_smoothed_sets"
FILE_GLOB = "*_single_smoothed.hdf5"
MAX_KDE_SAMPLES = 40_000
H0_LABEL = r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"
LNZ_LABEL = r"$\ln \mathcal{Z}$"
BIAS_PARAMS = ("alpha_low", "alpha_high", "log_rho_t", "log_rho_width")
NUISANCE_PARAMS = ("M_TRGB", "sigma_int", "sigma_v", "mag_lim_TRGB",
                   "mag_lim_TRGB_width", "beta", "nu_cz")
BIAS_LABELS = {
    "alpha_low": r"$\alpha_\mathrm{low}$",
    "alpha_high": r"$\alpha_\mathrm{high}$",
    "log_rho_t": r"$\log\rho_t$",
    "log_rho_width": r"$\log\Delta\rho$",
}
# Palette over the 8 variants (categorical).
SET_CMAP = "tab10"


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Directory holding the copied single-field HDF5 outputs.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTDIR,
        help="Directory for plots and summaries.")
    return parser.parse_args()


def parse_run(path):
    """Decompose a run filename into its model-variant features."""
    name = path.name
    stem = re.sub(r"_field\d+_single_smoothed\.hdf5$", "", name)
    field = int(re.search(r"_field(\d+)_", name).group(1))
    smooth_R = float(re.search(r"rhoSmoothR(\d+)", stem).group(1))
    likelihood = "student_t" if "cz-student_t" in stem else "gaussian"
    sky = "skyhp_nside1_k48" in stem
    vmono = "_Vmono_" in stem
    beta_free = stem.endswith("_beta_free")

    like_short = {"gaussian": "Gauss", "student_t": "Stud"}[likelihood]
    tokens = [f"R{smooth_R:g}", like_short]
    if vmono:
        tokens.append("Vmono")
    if sky:
        tokens.append("sky")
    if beta_free:
        tokens.append("beta")
    return {
        "set_key": stem,
        "field": field,
        "smooth_R": smooth_R,
        "likelihood": likelihood,
        "sky": sky,
        "vmono": vmono,
        "beta_free": beta_free,
        "label": " ".join(tokens),
        "sort_key": (smooth_R, likelihood == "student_t", vmono, sky,
                     beta_free),
        "source": str(path),
    }


def finite_samples(handle, name, path):
    samples = np.asarray(handle[f"samples/{name}"], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"`{path}` has no finite `{name}` samples.")
    return samples


def read_scalar(handle, name, default=np.nan):
    if name not in handle:
        return default
    value = float(handle[name][()])
    return value if np.isfinite(value) else default


def median_summary(samples, prefix):
    q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
    return {f"{prefix}_mean": float(np.mean(samples)),
            f"{prefix}_std": float(np.std(samples, ddof=1)),
            f"{prefix}_q16": float(q16),
            f"{prefix}_q50": float(q50),
            f"{prefix}_q84": float(q84)}


def read_row(spec):
    path = Path(spec["source"])
    with h5py.File(path, "r") as handle:
        h0 = finite_samples(handle, "H0", path)
        row = {**spec, "samples": h0, **median_summary(h0, "H0"),
               "n_H0": int(h0.size),
               "lnZ_harmonic": read_scalar(handle, "gof/lnZ_harmonic"),
               "err_lnZ_harmonic": read_scalar(handle, "gof/err_lnZ_harmonic"),
               "BIC": read_scalar(handle, "gof/BIC")}
        for name in (*BIAS_PARAMS, *NUISANCE_PARAMS):
            if f"samples/{name}" in handle:
                row.update(median_summary(
                    finite_samples(handle, name, path), name))
    return row


def load_rows(results_dir):
    files = sorted(results_dir.glob(FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No `{FILE_GLOB}` files in `{results_dir}`.")
    rows = [read_row(parse_run(path)) for path in files]
    return sorted(rows, key=lambda r: (r["sort_key"], r["field"]))


def grouped_by_set(rows):
    """Ordered {label: [rows]} keyed by model variant."""
    order = []
    groups = {}
    for row in rows:
        label = row["label"]
        if label not in groups:
            groups[label] = []
            order.append(label)
        groups[label].append(row)
    return {label: groups[label] for label in order}


def set_colours(labels):
    cmap = matplotlib.colormaps[SET_CMAP]
    return {label: cmap(i % 10) for i, label in enumerate(labels)}


def field_cmap():
    return trgbh0_cmap("trgbh0_single_smoothed_sets_fields")


def field_norm(rows):
    fields = np.asarray([row["field"] for row in rows], dtype=float)
    return Normalize(vmin=min(0.0, float(np.min(fields))),
                     vmax=float(np.max(fields)))


def stacked_samples(rows):
    return np.concatenate([row["samples"] for row in rows])


def kde_on_grid(samples, x_grid, bw=1.15):
    if samples.size > MAX_KDE_SAMPLES:
        idx = np.linspace(0, samples.size - 1, MAX_KDE_SAMPLES, dtype=int)
        samples = samples[idx]
    kde = gaussian_kde(samples)
    kde.set_bandwidth(kde.factor * bw)
    return kde(x_grid)


def style_xticks(ax, labels):
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35.0, ha="right")


def h0_summary(samples):
    return {"mean": float(np.mean(samples)),
            "std": float(np.std(samples, ddof=1))}


def plot_h0_distributions(groups, colours, out_pdf):
    labels = list(groups)
    positions = np.arange(len(labels))
    field_medians = [
        np.asarray([row["H0_q50"] for row in groups[label]], dtype=float)
        for label in labels]
    all_h0 = np.concatenate(field_medians)
    all_samples = np.concatenate(
        [stacked_samples(groups[label]) for label in labels])
    x_min = min(np.percentile(all_samples, 0.3), np.min(all_h0))
    x_max = max(np.percentile(all_samples, 99.7), np.max(all_h0))
    pad = 0.08 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 800)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, (ax_v, ax_k) = plt.subplots(
            1, 2, figsize=(8.4, 3.4), constrained_layout=True)

        parts = ax_v.violinplot(field_medians, positions=positions,
                                widths=0.8, showextrema=False)
        for body, label in zip(parts["bodies"], labels):
            body.set_facecolor(colours[label])
            body.set_edgecolor("none")
            body.set_alpha(0.38)
        for i, (label, vals) in enumerate(zip(labels, field_medians)):
            jitter = np.linspace(-0.18, 0.18, len(vals))
            ax_v.scatter(i + jitter, vals, s=6, color=colours[label],
                         alpha=0.40, edgecolor="none")
            q16, q50, q84 = np.percentile(vals, [16.0, 50.0, 84.0])
            ax_v.errorbar(i, q50, yerr=[[q50 - q16], [q84 - q50]], fmt="o",
                          color="black", ms=3.6, capsize=2.4, zorder=5)
        style_xticks(ax_v, labels)
        ax_v.set_ylabel(H0_LABEL)
        ax_v.set_title("Field-median distribution", loc="left")

        for label in labels:
            samples = stacked_samples(groups[label])
            s = h0_summary(samples)
            ax_k.plot(x_grid, kde_on_grid(samples, x_grid),
                      color=colours[label], lw=1.2,
                      label=(rf"{label} (${s['mean']:.2f}\pm"
                             rf"{s['std']:.2f}$)"))
            ax_k.axvline(s["mean"], color=colours[label], lw=0.6, alpha=0.4)
        ax_k.set_xlabel(H0_LABEL)
        ax_k.set_ylabel("Density")
        ax_k.set_ylim(bottom=0)
        ax_k.set_title("Stacked posteriors", loc="left")
        ax_k.legend(loc="upper right", frameon=False, fontsize=5.6,
                    handlelength=1.4)
        return save_pdf_png(fig, out_pdf)


def plot_lnz_distributions(groups, colours, out_pdf):
    labels = list(groups)
    positions = np.arange(len(labels))
    values = [
        np.asarray([row["lnZ_harmonic"] for row in groups[label]],
                   dtype=float)
        for label in labels]

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(6.4, 3.3), constrained_layout=True)
        parts = ax.violinplot(values, positions=positions, widths=0.78,
                              showextrema=False)
        for body, label in zip(parts["bodies"], labels):
            body.set_facecolor(colours[label])
            body.set_edgecolor("none")
            body.set_alpha(0.36)
        for i, (label, vals) in enumerate(zip(labels, values)):
            jitter = np.linspace(-0.17, 0.17, len(vals))
            ax.scatter(i + jitter, vals, s=6, color=colours[label],
                       alpha=0.38, edgecolor="none")
            ax.errorbar(i, np.mean(vals), yerr=np.std(vals, ddof=1),
                        fmt="o", color="black", ms=3.5, capsize=2.2, zorder=5)
        style_xticks(ax, labels)
        ax.set_ylabel(LNZ_LABEL)
        ax.set_title("Harmonic evidence distribution", loc="left")
        return save_pdf_png(fig, out_pdf)


def p_label(value):
    return r"<10^{-3}" if value < 1e-3 else rf"={value:.2f}"


def correlation_summary(x, y):
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return r"$r=\mathrm{n/a}$"
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return (rf"$r={pr:.2f}$, $p{p_label(pp)}$" "\n"
            rf"$\rho={sr:.2f}$, $p{p_label(sp)}$")


def plot_h0_vs_lnz(groups, colours, out_pdf):
    labels = list(groups)
    ncol = 4
    nrow = int(np.ceil(len(labels) / ncol))
    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(2.2 * ncol, 2.3 * nrow),
            sharex=False, sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, label in zip(axes, labels):
            group = groups[label]
            x = np.asarray([row["lnZ_harmonic"] for row in group])
            h0 = np.asarray([row["H0_q50"] for row in group])
            lo = np.asarray([row["H0_q16"] for row in group])
            hi = np.asarray([row["H0_q84"] for row in group])
            ok = np.isfinite(x) & np.isfinite(h0)
            ax.errorbar(x[ok], h0[ok],
                        yerr=np.vstack([h0[ok] - lo[ok], hi[ok] - h0[ok]]),
                        fmt="o", ms=2.8, color=colours[label],
                        ecolor=colours[label], elinewidth=0.4, capsize=0.8,
                        alpha=0.72)
            ax.set_title(label, loc="left", fontsize=6.6)
            ax.set_xlabel(LNZ_LABEL)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.text(0.03, 0.97, correlation_summary(x[ok], h0[ok]),
                    transform=ax.transAxes, ha="left", va="top", fontsize=5.6,
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.82})
        for ax in axes[:len(labels)]:
            if ax in axes[::ncol]:
                ax.set_ylabel(H0_LABEL)
        for ax in axes[len(labels):]:
            ax.set_visible(False)
        return save_pdf_png(fig, out_pdf)


def plot_param_distributions(groups, colours, params, out_pdf):
    avail = [p for p in params
             if any(f"{p}_q50" in row for rows in groups.values()
                    for row in rows)]
    if not avail:
        return []
    labels = list(groups)
    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            len(avail), 1, figsize=(6.6, 1.6 * len(avail)),
            sharex=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, param in zip(axes, avail):
            values = [
                np.asarray([row.get(f"{param}_q50", np.nan)
                            for row in groups[label]], dtype=float)
                for label in labels]
            # Only draw violins where the parameter actually varies; fixed
            # or variant-absent params would give a singular KDE.
            vparts = [(i, v[np.isfinite(v)]) for i, v in enumerate(values)]
            vparts = [(i, v) for i, v in vparts
                      if v.size > 1 and np.std(v) > 0]
            if vparts:
                parts = ax.violinplot(
                    [v for _, v in vparts],
                    positions=[i for i, _ in vparts], widths=0.78,
                    showextrema=False)
                for (i, _), body in zip(vparts, parts["bodies"]):
                    body.set_facecolor(colours[labels[i]])
                    body.set_edgecolor("none")
                    body.set_alpha(0.35)
            for i, (label, vals) in enumerate(zip(labels, values)):
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                jitter = np.linspace(-0.17, 0.17, len(vals))
                ax.scatter(i + jitter, vals, s=5.5, color=colours[label],
                           alpha=0.35, edgecolor="none")
                q16, q50, q84 = np.percentile(vals, [16.0, 50.0, 84.0])
                ax.errorbar(i, q50, yerr=[[q50 - q16], [q84 - q50]], fmt="o",
                            color="black", ms=3.2, capsize=2.0, zorder=5)
            ax.set_ylabel(BIAS_LABELS.get(param, param))
        style_xticks(axes[-1], labels)
        return save_pdf_png(fig, out_pdf)


def plot_matched_field_traces(groups, rows, colours, out_pdf):
    labels = list(groups)
    positions = np.arange(len(labels))
    fields = sorted(set.intersection(
        *[{row["field"] for row in groups[label]} for label in labels]))
    cmap = field_cmap()
    norm = field_norm(rows)
    by_field = {label: {row["field"]: row for row in groups[label]}
                for label in labels}

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(6.6, 3.4), constrained_layout=True)
        for field in fields:
            h0 = np.asarray([by_field[label][field]["H0_q50"]
                             for label in labels])
            colour = cmap(norm(field))
            ax.plot(positions, h0, color=colour, lw=0.55, alpha=0.40)
            ax.scatter(positions, h0, color=colour, s=8, alpha=0.62,
                       edgecolor="none", zorder=3)
        means = [np.mean([by_field[label][f]["H0_q50"] for f in fields])
                 for label in labels]
        stds = [np.std([by_field[label][f]["H0_q50"] for f in fields],
                       ddof=1) for label in labels]
        ax.errorbar(positions, means, yerr=stds, color="black", marker="o",
                    ms=4.0, lw=1.2, capsize=2.4, zorder=5,
                    label="field mean\n(error bar: field-to-field std)")
        style_xticks(ax, labels)
        ax.set_ylabel(H0_LABEL)
        ax.set_title(
            f"Matched Manticore fields ({len(fields)} fields)", loc="left")
        ax.legend(loc="best", frameon=False, handlelength=1.6)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.045)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


# Paired evidence comparisons (A vs B): each isolates one model axis. lnZ is
# differenced per matched field, so the large field-to-field evidence scatter
# cancels and only the model preference survives.
LNZ_PAIRS = (
    ("R4 Stud sky", "R4 Gauss sky", "Student-t vs Gaussian cz"),
    ("R4 Gauss Vmono sky", "R4 Gauss sky", "Vmono (Gaussian)"),
    ("R4 Stud Vmono sky", "R4 Stud sky", "Vmono (Student-t)"),
    ("R4 Gauss sky", "R4 Gauss", "sky exposure (Gaussian)"),
    ("R4 Gauss Vmono sky", "R4 Gauss Vmono", "sky exposure (Gauss Vmono)"),
    ("R4 Stud sky beta", "R4 Stud sky", "free beta (Student-t)"),
    ("R8 Gauss sky", "R4 Gauss sky", "R8 vs R4 smoothing (Gaussian)"),
    ("R8 Stud sky", "R4 Stud sky", "R8 vs R4 smoothing (Student-t)"),
)


def matched_fields(groups):
    return sorted(set.intersection(
        *[{row["field"] for row in rows} for rows in groups.values()]))


def by_field_lnz(groups):
    return {label: {row["field"]: row["lnZ_harmonic"] for row in rows}
            for label, rows in groups.items()}


def kr_verdict(lnb):
    """Kass & Raftery scale on a natural-log Bayes factor."""
    a = abs(lnb)
    if a < 1.0:
        return "inconclusive"
    if a < 3.0:
        return "positive"
    if a < 5.0:
        return "strong"
    return "decisive"


def paired_lnz_delta(lnz_map, fields, a, b):
    if a not in lnz_map or b not in lnz_map:
        return None
    d = np.asarray([lnz_map[a][f] - lnz_map[b][f] for f in fields],
                   dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    return {"mean": float(np.mean(d)), "std": float(np.std(d, ddof=1)),
            "se": float(np.std(d, ddof=1) / np.sqrt(d.size)), "n": int(d.size)}


def plot_lnz_matched_deltas(groups, rows, colours, out_pdf):
    labels = list(groups)
    fields = matched_fields(groups)
    lnz_map = by_field_lnz(groups)
    mean_lnz = {label: np.mean([lnz_map[label][f] for f in fields])
                for label in labels}
    ref = max(labels, key=lambda label: mean_lnz[label])
    order = sorted(labels, key=lambda label: mean_lnz[label])
    positions = np.arange(len(order))
    cmap = field_cmap()
    norm = field_norm(rows)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(6.8, 3.6), constrained_layout=True)
        for field in fields:
            d = np.asarray([lnz_map[label][field] - lnz_map[ref][field]
                            for label in order])
            colour = cmap(norm(field))
            ax.plot(positions, d, color=colour, lw=0.55, alpha=0.38)
            ax.scatter(positions, d, color=colour, s=8, alpha=0.6,
                       edgecolor="none", zorder=3)
        means = [np.mean([lnz_map[label][f] - lnz_map[ref][f]
                          for f in fields]) for label in order]
        stds = [np.std([lnz_map[label][f] - lnz_map[ref][f] for f in fields],
                       ddof=1) for label in order]
        ax.errorbar(positions, means, yerr=stds, color="black", marker="o",
                    ms=4.0, lw=1.2, capsize=2.4, zorder=5,
                    label="field mean\n(error bar: field-to-field std)")
        ax.axhline(0.0, color="0.35", lw=0.75, ls="--")
        style_xticks(ax, order)
        ax.set_ylabel(rf"$\Delta \ln \mathcal{{Z}}$ vs {ref}")
        ax.set_title(f"Matched-field evidence ({len(fields)} fields)",
                     loc="left")
        ax.legend(loc="lower right", frameon=False, handlelength=1.6)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.045)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def write_rows_csv(rows, path):
    fieldnames = ["set_key", "label", "field", "smooth_R", "likelihood",
                  "sky", "vmono", "beta_free", "n_H0", "H0_mean", "H0_std",
                  "H0_q16", "H0_q50", "H0_q84", "lnZ_harmonic",
                  "err_lnZ_harmonic", "BIC"]
    for name in (*BIAS_PARAMS, *NUISANCE_PARAMS):
        fieldnames.extend([f"{name}_q16", f"{name}_q50", f"{name}_q84"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames,
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_summary(groups, path):
    lines = ["# TRGBH0 Single-Field Smoothed Model-Variant Diagnostics", "",
             f"Variants: {len(groups)}.", "",
             "| variant | fields | mean field H0 | std field H0 | "
             "median field H0 | stacked H0 mean | stacked H0 std | "
             "median lnZ |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, group in groups.items():
        h0 = np.asarray([row["H0_q50"] for row in group], dtype=float)
        stacked = stacked_samples(group)
        lnz = np.asarray([row["lnZ_harmonic"] for row in group], dtype=float)
        lines.append(
            f"| {label} | {len(group)} | {np.mean(h0):.3f} | "
            f"{np.std(h0, ddof=1):.3f} | {np.median(h0):.3f} | "
            f"{np.mean(stacked):.3f} | {np.std(stacked, ddof=1):.3f} | "
            f"{np.nanmedian(lnz):.3f} |")

    # Matched-field evidence: rank by mean harmonic lnZ over shared fields,
    # report mean ln-Bayes-factor vs the best variant (paired per field).
    fields = matched_fields(groups)
    lnz_map = by_field_lnz(groups)
    mean_lnz = {label: float(np.mean([lnz_map[label][f] for f in fields]))
                for label in groups}
    best = max(mean_lnz, key=mean_lnz.get)
    lines += [
        "", f"## Matched-field evidence ({len(fields)} shared fields)", "",
        f"Mean ln-Bayes factor vs best variant (`{best}`), paired per field.",
        "",
        "| variant | mean lnZ | mean dlnZ vs best | SE | verdict vs best |",
        "| --- | ---: | ---: | ---: | --- |"]
    for label in sorted(groups, key=lambda x: mean_lnz[x], reverse=True):
        d = paired_lnz_delta(lnz_map, fields, label, best)
        verdict = "reference" if label == best else (
            f"{kr_verdict(d['mean'])} for best")
        lines.append(
            f"| {label} | {mean_lnz[label]:.2f} | {d['mean']:+.2f} | "
            f"{d['se']:.2f} | {verdict} |")

    lines += ["", "## Paired axis comparisons", "",
              "Mean dlnZ = lnZ(A) - lnZ(B), paired over shared fields "
              "(positive favours A).", "",
              "| axis | A | B | mean dlnZ | SE | verdict |",
              "| --- | --- | --- | ---: | ---: | --- |"]
    for a, b, axis in LNZ_PAIRS:
        d = paired_lnz_delta(lnz_map, fields, a, b)
        if d is None:
            continue
        favour = a if d["mean"] > 0 else b
        verdict = kr_verdict(d["mean"])
        tag = "inconclusive" if verdict == "inconclusive" else (
            f"{verdict} for {favour}")
        lines.append(
            f"| {axis} | {a} | {b} | {d['mean']:+.2f} | {d['se']:.2f} | "
            f"{tag} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.results_dir)
    groups = grouped_by_set(rows)
    colours = set_colours(list(groups))

    out = args.output_dir
    csv_path = out / "trgbh0_single_smoothed_sets_summary.csv"
    txt_path = out / "trgbh0_single_smoothed_sets_summary.txt"
    write_rows_csv(rows, csv_path)
    write_summary(groups, txt_path)
    written = [
        csv_path, txt_path,
        *plot_h0_distributions(
            groups, colours, out / "trgbh0_single_smoothed_sets_h0.pdf"),
        *plot_lnz_distributions(
            groups, colours, out / "trgbh0_single_smoothed_sets_lnz.pdf"),
        *plot_lnz_matched_deltas(
            groups, rows, colours,
            out / "trgbh0_single_smoothed_sets_lnz_matched.pdf"),
        *plot_h0_vs_lnz(
            groups, colours,
            out / "trgbh0_single_smoothed_sets_h0_vs_lnz.pdf"),
        *plot_param_distributions(
            groups, colours, BIAS_PARAMS,
            out / "trgbh0_single_smoothed_sets_bias_params.pdf"),
        *plot_param_distributions(
            groups, colours, NUISANCE_PARAMS,
            out / "trgbh0_single_smoothed_sets_nuisance_params.pdf"),
        *plot_matched_field_traces(
            groups, rows, colours,
            out / "trgbh0_single_smoothed_sets_matched_fields.pdf"),
    ]
    for path in written:
        print(f"Wrote {path}")
    print(f"Variants: {len(groups)}; fields: {len(rows)}.")


if __name__ == "__main__":
    main()
