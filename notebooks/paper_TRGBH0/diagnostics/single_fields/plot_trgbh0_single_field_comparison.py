#!/usr/bin/env python
"""Compare TRGBH0 single-field COLA MAS runs."""

from argparse import ArgumentParser
from dataclasses import dataclass
import csv
from itertools import combinations
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: F401,E402
import tomllib  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from trgbh0_plot_style import (  # noqa: E402
    FIGURE_DPI,
    ROOT,
    save_pdf_png,
    set_paper_rc,
    trgbh0_cmap,
)


TASK_FILE = ROOT / "scripts" / "runs" / "tasks_TRGBH0_single.txt"
DEFAULT_OUTDIR = (
    ROOT / "results" / "TRGBH0_paper" / "single_fields"
    / "plots" / "single_field_comparison"
)
MAX_KDE_SAMPLES = 40_000
MAS_ORDER = ("CIC", "PCS", "SPH")
MAS_LABELS = {
    "CIC": "CIC",
    "PCS": "PCS",
    "SPH": "SPH",
}
MAS_COLOURS = {
    "CIC": "#473198",
    "PCS": "#168039",
    "SPH": "#fe9000",
}
LIKELIHOOD_CHOICES = ("gaussian", "student_t", "all")
BETA_PRIOR_CHOICES = ("delta", "uniform", "all")
PARAMETERS = (
    "M_TRGB",
    "alpha_low",
    "alpha_high",
    "alpha_high_frac",
    "alpha_high_skipZ",
    "log_rho_t",
    "log_rho_width",
    "mag_lim_TRGB",
    "mag_lim_TRGB_width",
    "sigma_v",
    "beta",
    "Vext_mag",
)
H0_LABEL = r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"
LNZ_LABEL = r"$\ln \mathcal{Z}$"


@dataclass(frozen=True)
class Row:
    task: int
    field: int
    mas: str
    reconstruction: str
    config: str
    source: str
    cz_likelihood: str
    smooth_R: float
    beta_prior_dist: str
    values: dict[str, float | int | str]


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-file",
        type=Path,
        default=TASK_FILE,
        help="Task file containing the TRGBH0 single-field configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory for plots and summary CSV files.",
    )
    parser.add_argument(
        "--cz-likelihood",
        choices=LIKELIHOOD_CHOICES,
        default="gaussian",
        help="Redshift likelihood to plot from the mixed task file.",
    )
    parser.add_argument(
        "--smooth-R",
        type=float,
        default=None,
        help="Only plot runs with this 3D density-field smoothing scale.",
    )
    parser.add_argument(
        "--beta-prior-dist",
        choices=BETA_PRIOR_CHOICES,
        default="all",
        help="Only plot runs with this beta prior distribution.",
    )
    parser.add_argument(
        "--task-min",
        type=int,
        default=None,
        help="Only consider tasks with index greater than or equal to this.",
    )
    parser.add_argument(
        "--task-max",
        type=int,
        default=None,
        help="Only consider tasks with index less than or equal to this.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any field is missing a plotted MAS variant.",
    )
    parser.add_argument(
        "--fail-on-unusable",
        action="store_true",
        help="Fail if an output exists but cannot be read into the summary.",
    )
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


def active_reconstruction(config):
    return (
        get_nested(config, ("io", "PV_main", "EDD_TRGB", "reconstruction"))
        or get_nested(config, ("io", "CCHP", "reconstruction"))
    )


def task_config_paths(task_file):
    paths = []
    with repo_path(task_file).open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            task_index, config = line.split(maxsplit=1)
            paths.append((int(task_index), repo_path(config)))
    if not paths:
        raise ValueError(f"No task configs found in `{task_file}`.")
    return paths


def output_spec(
        task_index, config_path, cz_likelihood, smooth_R, beta_prior_dist,
        task_min, task_max):
    if task_min is not None and task_index < task_min:
        return None
    if task_max is not None and task_index > task_max:
        return None

    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    reconstruction = active_reconstruction(config)
    if reconstruction != "ManticoreLocalCOLA":
        return None

    mas = get_nested(
        config,
        ("io", "reconstruction_main", "ManticoreLocalCOLA", "which_MAS"),
    )
    if mas not in MAS_ORDER:
        return None

    config_likelihood = get_nested(
        config, ("model", "cz_likelihood"), default="gaussian")
    if cz_likelihood != "all" and config_likelihood != cz_likelihood:
        return None

    config_smooth_R = float(get_nested(
        config, ("model", "field_3d_smoothing_scale"), default=0.0))
    if smooth_R is not None and not np.isclose(config_smooth_R, smooth_R):
        return None

    config_beta_prior_dist = get_nested(
        config, ("model", "priors", "beta", "dist"), default="unknown")
    if (
            beta_prior_dist != "all"
            and config_beta_prior_dist != beta_prior_dist):
        return None

    return {
        "task": task_index,
        "field": int(get_nested(config, ("io", "field_indices"))),
        "mas": mas,
        "reconstruction": reconstruction,
        "config": str(config_path),
        "source": str(repo_path(get_nested(config, ("io", "fname_output")))),
        "cz_likelihood": config_likelihood,
        "smooth_R": config_smooth_R,
        "beta_prior_dist": config_beta_prior_dist,
    }


def finite_samples(handle, name, path):
    dataset = f"samples/{name}"
    if dataset not in handle:
        return None
    samples = np.asarray(handle[dataset], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"`{path}` has no finite samples for `{name}`.")
    return samples


def read_scalar(handle, name, default=np.nan):
    if name not in handle:
        return float(default)
    value = float(handle[name][()])
    if not np.isfinite(value):
        return float(default)
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
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as handle:
        h0 = finite_samples(handle, "H0", path)
        if h0 is None:
            raise ValueError(f"`{path}` does not contain `samples/H0`.")

        values = {
            "n_H0": int(h0.size),
            **sample_summary(h0, "H0"),
            "lnZ_harmonic": read_scalar(handle, "gof/lnZ_harmonic"),
            "err_lnZ_harmonic": read_scalar(
                handle, "gof/err_lnZ_harmonic"),
            "BIC": read_scalar(handle, "gof/BIC"),
            "AIC": read_scalar(handle, "gof/AIC"),
        }
        if np.isfinite(values["BIC"]):
            values["lnZ_bic"] = -0.5 * values["BIC"]
        else:
            values["lnZ_bic"] = np.nan

        for parameter in PARAMETERS:
            samples = finite_samples(handle, parameter, path)
            if samples is not None:
                values.update(sample_summary(samples, parameter))

        alpha_low = finite_samples(handle, "alpha_low", path)
        alpha_high = finite_samples(handle, "alpha_high", path)
        if alpha_high is None:
            alpha_high = finite_samples(handle, "alpha_high_skipZ", path)
        if alpha_low is not None and alpha_high is not None:
            frac = alpha_high / alpha_low
            frac = frac[np.isfinite(frac)]
            if frac.size:
                values.update(sample_summary(frac, "alpha_high_over_low"))

    return Row(**spec, values=values)


def load_rows(
        task_file, cz_likelihood, smooth_R=None, beta_prior_dist="all",
        task_min=None, task_max=None, fail_on_unusable=False):
    specs = []
    for task_index, config_path in task_config_paths(task_file):
        spec = output_spec(
            task_index,
            config_path,
            cz_likelihood,
            smooth_R,
            beta_prior_dist,
            task_min,
            task_max,
        )
        if spec is not None:
            specs.append(spec)
    rows = []
    unusable = []
    for spec in specs:
        try:
            rows.append(read_row(spec))
        except (OSError, KeyError, ValueError) as exc:
            unusable.append({
                "task": spec["task"],
                "field": spec["field"],
                "mas": spec["mas"],
                "source": spec["source"],
                "error": f"{type(exc).__name__}: {exc}",
            })
    if unusable and fail_on_unusable:
        details = "; ".join(
            f"task {item['task']} {item['mas']} field {item['field']}: "
            f"{item['error']}"
            for item in unusable
        )
        raise ValueError(f"Unusable output files: {details}")
    if not rows:
        raise ValueError("No usable TRGBH0 single-field outputs found.")
    order = {mas: i for i, mas in enumerate(MAS_ORDER)}
    return sorted(rows, key=lambda row: (row.field, order[row.mas])), unusable


def row_value(row, key):
    return row.values[key]


def rows_for_mas(rows, mas):
    return [row for row in rows if row.mas == mas]


def active_mas_order(rows):
    return tuple(mas for mas in MAS_ORDER if rows_for_mas(rows, mas))


def matched_fields(rows, mas_order, require_complete=False):
    by_field = {}
    for row in rows:
        by_field.setdefault(row.field, {})[row.mas] = row

    matched = {
        field: {mas: by_field[field][mas] for mas in mas_order}
        for field in sorted(by_field)
        if all(mas in by_field[field] for mas in mas_order)
    }
    missing = {
        field: [mas for mas in mas_order if mas not in rows_by_mas]
        for field, rows_by_mas in sorted(by_field.items())
        if any(mas not in rows_by_mas for mas in mas_order)
    }

    if require_complete:
        expected_fields = set(range(80))
        missing_fields = sorted(expected_fields - set(matched))
        if missing or missing_fields:
            raise FileNotFoundError(
                "Incomplete TRGBH0 single-field comparison. "
                f"Missing variants={missing}; missing fields={missing_fields}."
            )
    return matched, missing


def field_norm(rows):
    fields = np.asarray([row.field for row in rows], dtype=float)
    vmin = min(0.0, float(np.min(fields)))
    vmax = float(np.max(fields))
    if vmin == vmax:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def evidence_norm(values):
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def kde_on_grid(samples, x_grid, bw=1.2):
    if samples.size > MAX_KDE_SAMPLES:
        indices = np.linspace(0, samples.size - 1, MAX_KDE_SAMPLES,
                              dtype=int)
        samples = samples[indices]
    kde = gaussian_kde(samples)
    kde.set_bandwidth(kde.factor * bw)
    return kde(x_grid)


def write_summary_csv(rows, out_csv):
    base_fields = [
        "task",
        "field",
        "mas",
        "cz_likelihood",
        "smooth_R",
        "beta_prior_dist",
        "reconstruction",
        "n_H0",
        "H0_mean",
        "H0_std",
        "H0_q16",
        "H0_q50",
        "H0_q84",
        "lnZ_harmonic",
        "err_lnZ_harmonic",
        "BIC",
        "AIC",
        "lnZ_bic",
    ]
    parameter_fields = []
    for parameter in (*PARAMETERS, "alpha_high_over_low"):
        for suffix in ("mean", "std", "q16", "q50", "q84"):
            key = f"{parameter}_{suffix}"
            if any(key in row.values for row in rows):
                parameter_fields.append(key)
    fieldnames = base_fields + parameter_fields + ["source", "config"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            value_fields = [
                key for key in base_fields
                if key not in (
                    "task", "field", "mas", "cz_likelihood",
                    "smooth_R", "beta_prior_dist",
                    "reconstruction")
            ] + parameter_fields
            writer.writerow({
                "task": row.task,
                "field": row.field,
                "mas": row.mas,
                "cz_likelihood": row.cz_likelihood,
                "smooth_R": row.smooth_R,
                "beta_prior_dist": row.beta_prior_dist,
                "reconstruction": row.reconstruction,
                **{key: row.values.get(key, "") for key in value_fields},
                "source": row.source,
                "config": row.config,
            })


def delta_rows(matched, mas_order):
    preferred_order = (("PCS", "CIC"), ("SPH", "CIC"), ("SPH", "PCS"))
    comparisons = [
        pair for pair in preferred_order
        if pair[0] in mas_order and pair[1] in mas_order
    ]
    comparisons.extend(
        pair for pair in combinations(mas_order, 2)
        if pair not in comparisons and pair[::-1] not in comparisons
    )
    rows = []
    for field in sorted(matched):
        for left, right in comparisons:
            left_row = matched[field][left]
            right_row = matched[field][right]
            rows.append({
                "field": field,
                "comparison": f"{left}-{right}",
                "delta_H0_q50": (
                    row_value(left_row, "H0_q50")
                    - row_value(right_row, "H0_q50")
                ),
                "delta_lnZ_harmonic": (
                    row_value(left_row, "lnZ_harmonic")
                    - row_value(right_row, "lnZ_harmonic")
                ),
                "H0_q50_left": row_value(left_row, "H0_q50"),
                "H0_q50_right": row_value(right_row, "H0_q50"),
                "lnZ_harmonic_left": row_value(left_row, "lnZ_harmonic"),
                "lnZ_harmonic_right": row_value(right_row, "lnZ_harmonic"),
            })
    return rows


def write_delta_csv(rows, out_csv):
    fieldnames = [
        "field",
        "comparison",
        "delta_H0_q50",
        "delta_lnZ_harmonic",
        "H0_q50_left",
        "H0_q50_right",
        "lnZ_harmonic_left",
        "lnZ_harmonic_right",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_unusable_csv(rows, out_csv):
    fieldnames = ["task", "field", "mas", "source", "error"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_exclusion_reason(row):
    if not np.isfinite(row_value(row, "H0_q50")):
        return "non-finite H0_q50"
    if not np.isfinite(row_value(row, "lnZ_harmonic")):
        return "non-finite harmonic lnZ"
    if not np.isfinite(row_value(row, "err_lnZ_harmonic")):
        return "non-finite harmonic lnZ error"
    return ""


def split_plot_rows(rows):
    plot_rows = []
    excluded = []
    for row in rows:
        reason = plot_exclusion_reason(row)
        if reason:
            excluded.append({
                "task": row.task,
                "field": row.field,
                "mas": row.mas,
                "reason": reason,
                "H0_q50": row.values.get("H0_q50", ""),
                "lnZ_harmonic": row.values.get("lnZ_harmonic", ""),
                "err_lnZ_harmonic": row.values.get("err_lnZ_harmonic", ""),
                "source": row.source,
                "config": row.config,
            })
        else:
            plot_rows.append(row)
    return plot_rows, excluded


def write_plot_exclusions_csv(rows, out_csv):
    fieldnames = [
        "task",
        "field",
        "mas",
        "reason",
        "H0_q50",
        "lnZ_harmonic",
        "err_lnZ_harmonic",
        "source",
        "config",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_best_label(ax, rows, x_key, y_key="H0_q50"):
    finite_rows = [
        row for row in rows
        if np.isfinite(row_value(row, x_key)) and np.isfinite(row_value(row, y_key))
    ]
    if not finite_rows:
        return
    best = max(finite_rows, key=lambda row: row_value(row, x_key))
    ax.scatter(
        row_value(best, x_key),
        row_value(best, y_key),
        marker="*",
        s=76,
        color="black",
        edgecolor="white",
        linewidth=0.45,
        zorder=5,
    )
    ax.annotate(
        f"{best.field}",
        (row_value(best, x_key), row_value(best, y_key)),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=6.8,
    )


def plot_h0_vs_harmonic_lnz(rows, mas_order, out_pdf):
    cmap = trgbh0_cmap("trgbh0_single_harmonic_field_index")
    norm = field_norm(rows)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            1,
            len(mas_order),
            figsize=(8.2, 2.9),
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        for ax, mas in zip(axes, mas_order):
            mas_rows = rows_for_mas(rows, mas)
            x = np.asarray([
                row_value(row, "lnZ_harmonic") for row in mas_rows
            ])
            xerr = np.asarray([
                abs(row.values.get("err_lnZ_harmonic", np.nan))
                for row in mas_rows
            ])
            h0 = np.asarray([row_value(row, "H0_q50") for row in mas_rows])
            h0_lo = np.asarray([row_value(row, "H0_q16") for row in mas_rows])
            h0_hi = np.asarray([row_value(row, "H0_q84") for row in mas_rows])
            fields = np.asarray([row.field for row in mas_rows], dtype=float)
            finite = (
                np.isfinite(x)
                & np.isfinite(xerr)
                & np.isfinite(h0)
                & np.isfinite(h0_lo)
                & np.isfinite(h0_hi)
            )
            x = x[finite]
            xerr = xerr[finite]
            h0 = h0[finite]
            h0_lo = h0_lo[finite]
            h0_hi = h0_hi[finite]
            fields = fields[finite]
            yerr = np.vstack([h0 - h0_lo, h0_hi - h0])

            ax.errorbar(
                x,
                h0,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor="0.55",
                elinewidth=0.45,
                capsize=1.1,
                alpha=0.58,
                zorder=1,
            )
            sc = ax.scatter(
                x,
                h0,
                c=fields,
                cmap=cmap,
                norm=norm,
                s=25,
                edgecolor="0.15",
                linewidth=0.25,
                zorder=3,
            )
            add_best_label(ax, mas_rows, "lnZ_harmonic")
            ax.set_title(
                f"{MAS_LABELS[mas]}: {len(mas_rows)} fields",
                loc="left",
            )
            ax.set_xlabel(LNZ_LABEL)
            if ax is axes[0]:
                ax.set_ylabel(H0_LABEL)

        cbar = fig.colorbar(sc, ax=axes, pad=0.012, fraction=0.04)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_h0_histograms(rows, mas_order, out_pdf):
    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            1,
            len(mas_order),
            figsize=(4.7 * len(mas_order), 3.1),
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        scalar_mappable = None
        for ax, mas in zip(axes, mas_order):
            mas_rows = rows_for_mas(rows, mas)
            rows_with_samples = []
            for row in mas_rows:
                with h5py.File(row.source, "r") as handle:
                    samples = finite_samples(handle, "H0", row.source)
                if samples is not None:
                    rows_with_samples.append((row, samples))
            if not rows_with_samples:
                continue

            evidence = np.asarray([
                row_value(row, "lnZ_harmonic")
                for row, _ in rows_with_samples
            ])
            h0_samples = [samples for _, samples in rows_with_samples]
            stacked_h0 = np.concatenate(h0_samples)
            field_medians = np.asarray([
                row_value(row, "H0_q50") for row, _ in rows_with_samples
            ])
            x_min, x_max = np.percentile(stacked_h0, [0.2, 99.8])
            pad = 0.15 * (x_max - x_min)
            x_grid = np.linspace(x_min - pad, x_max + pad, 600)
            cmap = trgbh0_cmap(f"trgbh0_single_h0_{mas}")
            norm = evidence_norm(evidence)

            ax.hist(
                field_medians,
                bins=min(9, max(4, int(np.sqrt(len(field_medians))) + 2)),
                density=True,
                color="#b8b8b8",
                alpha=0.45,
                label=r"Field median $H_0$",
                zorder=0,
            )

            for lnz_harmonic, samples in zip(evidence, h0_samples):
                ax.plot(
                    x_grid,
                    kde_on_grid(samples, x_grid),
                    color=cmap(norm(lnz_harmonic)),
                    lw=0.9,
                    alpha=0.72,
                    zorder=2,
                )

            ax.plot(
                x_grid,
                kde_on_grid(stacked_h0, x_grid),
                color="black",
                lw=1.35,
                label=r"Stacked posterior",
                zorder=4,
            )
            ax.axvline(
                np.median(field_medians), color="black", lw=1.0, ls=":",
                label=r"Median of field medians")
            ax.set_title(
                f"{MAS_LABELS[mas]}: {len(rows_with_samples)} fields",
                loc="left",
            )
            ax.set_xlabel(H0_LABEL)
            ax.set_xlim(x_grid[0], x_grid[-1])
            ax.set_ylim(bottom=0)

            curve_proxy = Line2D(
                [0], [0], color=cmap(norm(np.median(evidence))), lw=1.0,
                label=r"Individual field posterior")
            handles, labels = ax.get_legend_handles_labels()
            handles.insert(1, curve_proxy)
            labels.insert(1, curve_proxy.get_label())
            ax.legend(
                handles, labels,
                loc="upper right",
                fontsize=6.5,
                frameon=False,
                handlelength=1.7,
            )
            scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        axes[0].set_ylabel("Posterior density")
        if scalar_mappable is not None:
            scalar_mappable.set_array([])
            cbar = fig.colorbar(
                scalar_mappable, ax=axes, pad=0.015, fraction=0.055)
            cbar.set_label(LNZ_LABEL)
            cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_matched_fields(matched, mas_order, out_pdf):
    fields = sorted(matched)
    xpos = np.arange(len(mas_order))
    cmap = trgbh0_cmap("trgbh0_single_matched_fields")
    norm = Normalize(vmin=0, vmax=max(fields) if fields else 1)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(6.1, 4.9),
            sharex=True,
            constrained_layout=True,
        )
        for ax, key, ylabel in (
            (axes[0], "H0_q50", H0_LABEL),
            (axes[1], "lnZ_harmonic", LNZ_LABEL),
        ):
            values_by_field = []
            for field in fields:
                values = np.asarray([
                    row_value(matched[field][mas], key) for mas in mas_order
                ])
                values_by_field.append(values)
                colour = cmap(norm(field))
                ax.plot(xpos, values, color=colour, lw=0.7, alpha=0.38)
                ax.scatter(xpos, values, color=colour, s=9, alpha=0.68)
            values_by_field = np.asarray(values_by_field)
            means = np.mean(values_by_field, axis=0)
            stds = np.std(values_by_field, axis=0, ddof=1)
            ax.errorbar(
                xpos,
                means,
                yerr=stds,
                color="black",
                marker="o",
                lw=1.25,
                ms=4.0,
                capsize=2.4,
                label="mean across fields\n(error bar: field-to-field std)",
                zorder=5,
            )
            ax.set_ylabel(ylabel)
            if key == "H0_q50":
                ax.set_title("Matched realisations", loc="left")
                ax.legend(
                    loc="best",
                    frameon=True,
                    facecolor="white",
                    edgecolor="none",
                    framealpha=0.85,
                    handlelength=1.7,
                )
        axes[1].set_xticks(xpos, [MAS_LABELS[mas] for mas in mas_order])
        axes[1].set_xlabel("Mass-assignment scheme")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, pad=0.012, fraction=0.04)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_mas_deltas(delta_data, out_pdf):
    comparisons = tuple(dict.fromkeys(
        item["comparison"] for item in delta_data))
    if not comparisons:
        return []
    colours = {
        "PCS-CIC": "#168039",
        "SPH-CIC": "#fe9000",
        "SPH-PCS": "#a34400",
    }

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(7.2, 3.25),
            constrained_layout=True,
        )
        for ax, key, ylabel in (
            (axes[0], "delta_H0_q50", r"$\Delta H_0$"),
            (axes[1], "delta_lnZ_harmonic", r"$\Delta \ln \mathcal{Z}$"),
        ):
            positions = np.arange(len(comparisons), dtype=float)
            for idx, comparison in enumerate(comparisons):
                values = np.asarray([
                    item[key] for item in delta_data
                    if item["comparison"] == comparison
                ])
                field_values = np.asarray([
                    item["field"] for item in delta_data
                    if item["comparison"] == comparison
                ])
                finite = np.isfinite(values)
                values = values[finite]
                field_values = field_values[finite]
                if values.size == 0:
                    continue
                jitter = (
                    (field_values - np.mean(field_values))
                    / max(np.ptp(field_values), 1.0)
                ) * 0.18
                ax.scatter(
                    positions[idx] + jitter,
                    values,
                    s=10,
                    color=colours.get(comparison, "0.45"),
                    alpha=0.45,
                    linewidth=0.0,
                )
                median = float(np.median(values))
                q16, q84 = np.percentile(values, [16.0, 84.0])
                ax.errorbar(
                    positions[idx],
                    median,
                    yerr=np.array([[median - q16], [q84 - median]]),
                    color="black",
                    marker="o",
                    ms=4.0,
                    capsize=2.4,
                    zorder=5,
                )
            ax.axhline(0.0, color="0.35", lw=0.7, ls=":")
            ax.set_xticks(positions, comparisons)
            ax.set_ylabel(ylabel)
        axes[0].set_title("Matched-field MAS shifts", loc="left")
        return save_pdf_png(fig, out_pdf)


def plot_parameter_diagnostics(rows, mas_order, out_pdf):
    x_specs = (
        ("alpha_low_q50", r"$\alpha_{\rm low}$"),
        ("alpha_high_q50", r"$\alpha_{\rm high}$"),
        ("log_rho_width_q50", r"$\log\rho$ width"),
        ("mag_lim_TRGB_q50", r"$m_{\rm lim}^{\rm TRGB}$"),
    )

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(7.3, 5.0),
            constrained_layout=True,
        )
        axes = axes.ravel()
        for ax, (x_key, xlabel) in zip(axes, x_specs):
            for mas in mas_order:
                mas_rows = [
                    row for row in rows_for_mas(rows, mas)
                    if x_key in row.values
                ]
                if not mas_rows:
                    continue
                x = np.asarray([row_value(row, x_key) for row in mas_rows])
                y = np.asarray([row_value(row, "H0_q50") for row in mas_rows])
                ax.scatter(
                    x,
                    y,
                    s=18,
                    color=MAS_COLOURS[mas],
                    alpha=0.62,
                    edgecolor="0.15",
                    linewidth=0.18,
                    label=MAS_LABELS[mas],
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(H0_LABEL)
        handles = [
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor=MAS_COLOURS[mas],
                markeredgecolor="0.15",
                markeredgewidth=0.2,
                markersize=4.5,
                label=MAS_LABELS[mas],
            )
            for mas in mas_order
        ]
        if handles:
            axes[0].legend(handles=handles, loc="best", frameon=False)
        axes[0].set_title("H0 against bias and selection parameters", loc="left")
        return save_pdf_png(fig, out_pdf)


def plot_beta_diagnostics(rows, out_pdf):
    beta_rows = [
        row for row in rows
        if all(
            np.isfinite(row.values.get(key, np.nan))
            for key in (
                "beta_q16", "beta_q50", "beta_q84",
                "H0_q16", "H0_q50", "H0_q84", "lnZ_harmonic")
        )
    ]
    if not beta_rows:
        return []

    beta_mid = np.asarray([row_value(row, "beta_q50") for row in beta_rows])
    beta_std = np.asarray([
        row.values.get("beta_std", np.nan) for row in beta_rows])
    if (
            np.nanmax(np.abs(beta_std)) <= 1.0e-10
            and np.ptp(beta_mid) <= 1.0e-10):
        return []

    fields = np.asarray([row.field for row in beta_rows], dtype=float)
    beta_lo = np.asarray([row_value(row, "beta_q16") for row in beta_rows])
    beta_hi = np.asarray([row_value(row, "beta_q84") for row in beta_rows])
    beta_err = np.vstack([beta_mid - beta_lo, beta_hi - beta_mid])
    h0_mid = np.asarray([row_value(row, "H0_q50") for row in beta_rows])
    h0_lo = np.asarray([row_value(row, "H0_q16") for row in beta_rows])
    h0_hi = np.asarray([row_value(row, "H0_q84") for row in beta_rows])
    h0_err = np.vstack([h0_mid - h0_lo, h0_hi - h0_mid])
    lnz = np.asarray([row_value(row, "lnZ_harmonic") for row in beta_rows])
    lnz_err = np.asarray([
        abs(row.values.get("err_lnZ_harmonic", np.nan))
        for row in beta_rows
    ])
    finite_lnz_err = np.isfinite(lnz_err)

    cmap = trgbh0_cmap("trgbh0_single_beta_fields")
    norm = field_norm(beta_rows)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 2, figsize=(7.1, 5.1), constrained_layout=True)
        axes = axes.ravel()

        axes[0].errorbar(
            beta_mid,
            h0_mid,
            xerr=beta_err,
            yerr=h0_err,
            fmt="none",
            ecolor="0.55",
            elinewidth=0.45,
            capsize=1.1,
            alpha=0.58,
            zorder=1,
        )
        sc = axes[0].scatter(
            beta_mid,
            h0_mid,
            c=fields,
            cmap=cmap,
            norm=norm,
            s=23,
            edgecolor="0.15",
            linewidth=0.25,
            zorder=3,
        )
        axes[0].axvline(1.0, color="0.35", lw=0.7, ls=":")
        axes[0].set_xlabel(r"$\beta$")
        axes[0].set_ylabel(H0_LABEL)
        axes[0].set_title(r"Free-$\beta$ single fields", loc="left")

        axes[1].errorbar(
            beta_mid,
            lnz,
            xerr=beta_err,
            yerr=lnz_err if np.all(finite_lnz_err) else None,
            fmt="none",
            ecolor="0.55",
            elinewidth=0.45,
            capsize=1.1,
            alpha=0.58,
            zorder=1,
        )
        axes[1].scatter(
            beta_mid,
            lnz,
            c=fields,
            cmap=cmap,
            norm=norm,
            s=23,
            edgecolor="0.15",
            linewidth=0.25,
            zorder=3,
        )
        axes[1].axvline(1.0, color="0.35", lw=0.7, ls=":")
        axes[1].set_xlabel(r"$\beta$")
        axes[1].set_ylabel(LNZ_LABEL)

        axes[2].errorbar(
            fields,
            beta_mid,
            yerr=beta_err,
            fmt="none",
            ecolor="0.55",
            elinewidth=0.45,
            capsize=1.1,
            alpha=0.58,
            zorder=1,
        )
        axes[2].scatter(
            fields,
            beta_mid,
            c=fields,
            cmap=cmap,
            norm=norm,
            s=23,
            edgecolor="0.15",
            linewidth=0.25,
            zorder=3,
        )
        axes[2].axhline(1.0, color="0.35", lw=0.7, ls=":")
        axes[2].set_xlabel("Manticore field")
        axes[2].set_ylabel(r"$\beta$")
        axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))

        axes[3].hist(
            beta_mid,
            bins=min(12, max(5, int(np.sqrt(beta_mid.size)) + 2)),
            color="#b8b8b8",
            edgecolor="0.25",
            linewidth=0.45,
        )
        axes[3].axvline(1.0, color="0.35", lw=0.7, ls=":")
        axes[3].axvline(
            np.median(beta_mid),
            color="black",
            lw=1.0,
            label="field-median median",
        )
        axes[3].set_xlabel(r"$\beta$")
        axes[3].set_ylabel("Number of fields")
        axes[3].legend(frameon=False, handlelength=1.7)

        cbar = fig.colorbar(sc, ax=axes, pad=0.012, fraction=0.04)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def print_summary(rows, mas_order, matched, missing, unusable, plot_exclusions):
    print(f"Loaded {len(rows)} outputs.")
    if unusable:
        print(f"Unusable outputs: {len(unusable)}.")
        for item in unusable:
            print(
                f"  task {item['task']} {item['mas']} field {item['field']}: "
                f"{item['error']}"
            )
    if plot_exclusions:
        print(f"Excluded from plots: {len(plot_exclusions)}.")
        for item in plot_exclusions:
            print(
                f"  task {item['task']} {item['mas']} field {item['field']}: "
                f"{item['reason']} "
                f"(H0={item['H0_q50']:.3f}, "
                f"lnZ={item['lnZ_harmonic']})"
            )
    for mas in mas_order:
        mas_rows = rows_for_mas(rows, mas)
        h0 = np.asarray([row_value(row, "H0_q50") for row in mas_rows])
        lnz = np.asarray([row_value(row, "lnZ_harmonic") for row in mas_rows])
        finite_lnz = np.isfinite(lnz)
        if np.any(finite_lnz):
            best = [
                row for row, is_finite in zip(mas_rows, finite_lnz)
                if is_finite
            ][int(np.argmax(lnz[finite_lnz]))]
            best_summary = (
                f"best lnZ field={best.field} "
                f"(lnZ={row_value(best, 'lnZ_harmonic'):.2f}, "
                f"H0={row_value(best, 'H0_q50'):.3f})"
            )
        else:
            best_summary = "best lnZ field=none"
        print(
            f"{mas}: n={len(mas_rows)}, "
            f"median field H0={np.median(h0):.3f}, "
            f"field-to-field std={np.std(h0, ddof=1):.3f}, "
            f"{best_summary}"
        )

    print(f"Matched fields: {len(matched)}.")
    if missing:
        print(f"Missing MAS entries: {missing}")

    preferred_order = (("PCS", "CIC"), ("SPH", "CIC"), ("SPH", "PCS"))
    comparisons = [
        pair for pair in preferred_order
        if pair[0] in mas_order and pair[1] in mas_order
    ]
    comparisons.extend(
        pair for pair in combinations(mas_order, 2)
        if pair not in comparisons and pair[::-1] not in comparisons
    )
    for left, right in comparisons:
        comparison = f"{left}-{right}"
        deltas = []
        lnz_deltas = []
        for field in sorted(matched):
            deltas.append(
                row_value(matched[field][left], "H0_q50")
                - row_value(matched[field][right], "H0_q50")
            )
            lnz_deltas.append(
                row_value(matched[field][left], "lnZ_harmonic")
                - row_value(matched[field][right], "lnZ_harmonic")
            )
        deltas = np.asarray(deltas)
        lnz_deltas = np.asarray(lnz_deltas)
        finite_lnz_deltas = lnz_deltas[np.isfinite(lnz_deltas)]
        median_lnz_delta = (
            float(np.median(finite_lnz_deltas))
            if finite_lnz_deltas.size else np.nan
        )
        print(
            f"{comparison}: median dH0={np.median(deltas):+.3f}, "
            f"std dH0={np.std(deltas, ddof=1):.3f}, "
            f"median dlnZ={median_lnz_delta:+.2f}"
        )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variant = args.output_dir.name

    rows, unusable = load_rows(
        args.task_file,
        args.cz_likelihood,
        smooth_R=args.smooth_R,
        beta_prior_dist=args.beta_prior_dist,
        task_min=args.task_min,
        task_max=args.task_max,
        fail_on_unusable=args.fail_on_unusable,
    )
    plot_rows, plot_exclusions = split_plot_rows(rows)
    mas_order = active_mas_order(plot_rows)
    matched, missing = matched_fields(
        plot_rows, mas_order, args.require_complete)
    deltas = delta_rows(matched, mas_order)

    summary_csv = args.output_dir / "trgbh0_single_field_summary.csv"
    delta_csv = args.output_dir / "trgbh0_single_matched_deltas.csv"
    unusable_csv = args.output_dir / "trgbh0_single_unusable_outputs.csv"
    exclusions_csv = args.output_dir / "trgbh0_single_plot_exclusions.csv"
    write_summary_csv(rows, summary_csv)
    write_delta_csv(deltas, delta_csv)
    write_unusable_csv(unusable, unusable_csv)
    write_plot_exclusions_csv(plot_exclusions, exclusions_csv)

    plot_h0_vs_harmonic_lnz(
        plot_rows,
        mas_order,
        args.output_dir / f"trgbh0_single_{variant}_h0_vs_harmonic_lnz.pdf",
    )
    plot_h0_histograms(
        plot_rows,
        mas_order,
        args.output_dir / f"trgbh0_single_{variant}_h0_posteriors.pdf",
    )
    if len(mas_order) >= 2 and matched:
        plot_matched_fields(
            matched,
            mas_order,
            args.output_dir / f"trgbh0_single_{variant}_matched_fields.pdf",
        )
    plot_mas_deltas(
        deltas,
        args.output_dir / f"trgbh0_single_{variant}_mas_deltas.pdf",
    )
    plot_parameter_diagnostics(
        plot_rows,
        mas_order,
        args.output_dir / f"trgbh0_single_{variant}_h0_vs_parameters.pdf",
    )
    plot_beta_diagnostics(
        plot_rows,
        args.output_dir / f"trgbh0_single_{variant}_beta_diagnostics.pdf",
    )

    print_summary(
        plot_rows, mas_order, matched, missing, unusable, plot_exclusions)
    print(f"Summary CSV: {summary_csv}")
    print(f"Delta CSV: {delta_csv}")
    print(f"Unusable CSV: {unusable_csv}")
    print(f"Plot exclusions CSV: {exclusions_csv}")
    print(f"Plot directory: {args.output_dir}")


if __name__ == "__main__":
    main()
