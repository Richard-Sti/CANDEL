#!/usr/bin/env python
"""Compare CH0 single-field reconstruction variants."""

from argparse import ArgumentParser
from dataclasses import dataclass
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
PLOT_ROOT = ROOT / "results" / "CH0_paper" / "single_fields" / "plots"
FIGURE_DPI = 500
HIGHLIGHT_FIELD = 21
H0_LABEL = (
    r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"
)
LNZ_LABEL = r"harmonic $\ln Z$"
FIELD_COLOURS = ["#ef476f", "#473198", "#a8c256", "#5adbff", "#fe9000"]


@dataclass(frozen=True)
class ComparisonMode:
    name: str
    title: str
    default_outdir: Path
    output_prefix: str
    variant_order: tuple[str, ...]
    variant_labels: dict[str, str]
    variant_colours: dict[str, str]
    cmap_name: str
    style: tuple[str, ...]
    default_task_min: int | None = None
    default_task_max: int | None = None


MODES = {
    "cola-mas": ComparisonMode(
        name="cola-mas",
        title="COLA MAS Single-Field Comparison",
        default_outdir=PLOT_ROOT / "cola_mas_comparison",
        output_prefix="cola_mas",
        variant_order=("CIC", "PCS", "SPH"),
        variant_labels={"CIC": "CIC", "PCS": "PCS", "SPH": "SPH"},
        variant_colours={
            "CIC": "#473198",
            "PCS": "#168039",
            "SPH": "#fe9000",
        },
        cmap_name="cola_mas_comparison_fields",
        style=("science",),
    ),
    "swift-cola-sph": ComparisonMode(
        name="swift-cola-sph",
        title="SWIFT/SPH and COLA/SPH Single-Field Comparison",
        default_outdir=PLOT_ROOT / "swift_cola_sph_comparison",
        output_prefix="swift_cola_sph",
        variant_order=("SWIFT_SPH", "COLA_SPH"),
        variant_labels={
            "SWIFT_SPH": "SWIFT SPH",
            "COLA_SPH": "COLA SPH",
        },
        variant_colours={
            "SWIFT_SPH": "#473198",
            "COLA_SPH": "#fe9000",
        },
        cmap_name="swift_cola_sph_fields",
        style=("science", "no-latex"),
    ),
    "swift-sph-cola-cic": ComparisonMode(
        name="swift-sph-cola-cic",
        title="SWIFT SPH and COLA CIC Single-Field Comparison",
        default_outdir=PLOT_ROOT / "swift_sph_cola_cic",
        output_prefix="swift_sph_cola_cic",
        variant_order=("swift_sph", "cola_cic"),
        variant_labels={
            "swift_sph": "SWIFT SPH",
            "cola_cic": "COLA CIC",
        },
        variant_colours={
            "swift_sph": "#473198",
            "cola_cic": "#168039",
        },
        cmap_name="swift_sph_cola_cic_fields",
        style=("science",),
        default_task_min=0,
        default_task_max=109,
    ),
}


def parse_args(default_mode=None):
    parser = ArgumentParser(description=__doc__)
    if default_mode is None:
        parser.add_argument("mode", choices=tuple(MODES),
                            help="Comparison preset to run.")
    parser.add_argument(
        "--task-file", type=Path, default=TASK_FILE,
        help="Task file containing the CH0 single-field configs.")
    parser.add_argument(
        "--task-min", type=int, default=None,
        help="First task index to include. Defaults depend on the mode.")
    parser.add_argument(
        "--task-max", type=int, default=None,
        help="Last task index to include. Defaults depend on the mode.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for plots and summaries.")
    parser.add_argument(
        "--require-complete", action="store_true",
        help="Fail if a mode-specific matched set is incomplete.")
    args = parser.parse_args()
    if default_mode is not None:
        args.mode = default_mode
    mode = MODES[args.mode]
    if args.output_dir is None:
        args.output_dir = mode.default_outdir
    if args.task_min is None:
        args.task_min = mode.default_task_min
    if args.task_max is None:
        args.task_max = mode.default_task_max
    return args


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


def task_config_paths(task_file, task_min=None, task_max=None):
    paths = []
    with repo_path(task_file).open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            task_index, config = line.split(maxsplit=1)
            task_index = int(task_index)
            if task_min is not None and task_index < task_min:
                continue
            if task_max is not None and task_index > task_max:
                continue
            paths.append((task_index, repo_path(config)))
    if not paths:
        if task_min is None and task_max is None:
            raise ValueError(f"No task configs found in `{task_file}`.")
        raise ValueError(
            f"No tasks in index range [{task_min}, {task_max}].")
    return paths


def classify_config(config, mode):
    reconstruction = get_nested(config, ("io", "SH0ES", "reconstruction"))
    which_bias = get_nested(config, ("model", "which_bias"))
    if reconstruction == "ManticoreLocalCOLA":
        mas = get_nested(
            config, ("io", "reconstruction_main", "ManticoreLocalCOLA",
                     "which_MAS"))
    elif reconstruction == "ManticoreLocalSWIFT":
        mas = "SPH"
    else:
        return None

    if mode.name == "cola-mas":
        if reconstruction != "ManticoreLocalCOLA" or mas not in mode.variant_order:
            return None
        variant = mas
    elif mode.name == "swift-cola-sph":
        if which_bias != "double_powerlaw":
            return None
        if reconstruction == "ManticoreLocalSWIFT":
            variant = "SWIFT_SPH"
        elif reconstruction == "ManticoreLocalCOLA" and mas == "SPH":
            variant = "COLA_SPH"
        else:
            return None
    elif mode.name == "swift-sph-cola-cic":
        if reconstruction == "ManticoreLocalSWIFT":
            variant = "swift_sph"
        elif reconstruction == "ManticoreLocalCOLA" and mas == "CIC":
            variant = "cola_cic"
        else:
            return None
    else:
        raise ValueError(f"Unknown comparison mode `{mode.name}`.")

    return {
        "variant": variant,
        "run": variant,
        "reconstruction": reconstruction,
        "mas": mas,
    }


def output_spec(task_index, config_path, mode):
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    classification = classify_config(config, mode)
    if classification is None:
        return None
    return {
        "task": task_index,
        **classification,
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


def load_rows(task_file, task_min, task_max, mode):
    specs = []
    for task_index, config_path in task_config_paths(
            task_file, task_min, task_max):
        spec = output_spec(task_index, config_path, mode)
        if spec is not None:
            specs.append(spec)
    rows = [read_row(spec) for spec in specs]
    if not rows:
        raise ValueError(f"No usable outputs found for mode `{mode.name}`.")
    order = {variant: i for i, variant in enumerate(mode.variant_order)}
    if mode.name == "swift-sph-cola-cic":
        return sorted(rows, key=lambda row: (
            order[row["variant"]], row["field"]))
    return sorted(rows, key=lambda row: (
        row["field"], order[row["variant"]]))


def matched_fields(rows, mode, require_complete=False):
    by_field = {}
    for row in rows:
        by_field.setdefault(row["field"], {})[row["variant"]] = row
    matched = {
        field: {variant: by_field[field][variant]
                for variant in mode.variant_order}
        for field in sorted(by_field)
        if all(variant in by_field[field] for variant in mode.variant_order)
    }

    if mode.name == "swift-cola-sph":
        reference_fields = {
            row["field"] for row in rows if row["variant"] == "SWIFT_SPH"
        }
        missing = {
            field: [variant for variant in mode.variant_order
                    if variant not in by_field[field]]
            for field in sorted(reference_fields)
            if any(variant not in by_field[field]
                   for variant in mode.variant_order)
        }
    else:
        missing = {
            field: [variant for variant in mode.variant_order
                    if variant not in rows_by_variant]
            for field, rows_by_variant in sorted(by_field.items())
            if any(variant not in rows_by_variant
                   for variant in mode.variant_order)
        }

    if require_complete:
        if mode.name == "cola-mas":
            expected_fields = set(range(80))
            missing_fields = sorted(expected_fields - set(matched))
            if missing or missing_fields:
                raise FileNotFoundError(
                    f"Incomplete COLA MAS comparison. Missing variants="
                    f"{missing}; missing matched fields={missing_fields}.")
        elif missing:
            raise FileNotFoundError(
                f"Incomplete {mode.title}. Missing variants={missing}.")
    return matched, missing


def rows_by_variant(matched, mode):
    return {
        variant: [matched[field][variant] for field in sorted(matched)]
        for variant in mode.variant_order
    }


def split_rows(rows, mode):
    return {
        variant: sorted([row for row in rows if row["variant"] == variant],
                        key=lambda row: row["field"])
        for variant in mode.variant_order
    }


def common_rows(rows_by_run, mode):
    left_key, right_key = mode.variant_order
    left = {row["field"]: row for row in rows_by_run[left_key]}
    right = {row["field"]: row for row in rows_by_run[right_key]}
    fields = sorted(set(left) & set(right))
    return [(field, left[field], right[field]) for field in fields]


def set_paper_rc():
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.5,
    })


def field_cmap(mode):
    return LinearSegmentedColormap.from_list(mode.cmap_name, FIELD_COLOURS)


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


def plot_h0_vs_lnz(by_variant, mode, out_pdf):
    all_rows = [row for rows in by_variant.values() for row in rows]
    fields = [row["field"] for row in all_rows]
    cmap = field_cmap(mode)
    norm = field_norm(fields)
    if mode.name == "swift-cola-sph":
        evidence_specs = (("lnZ_harmonic", "err_lnZ_harmonic", LNZ_LABEL),)
        figsize = (6.4, 3.0)
    elif mode.name == "swift-sph-cola-cic":
        evidence_specs = (
            ("lnZ_harmonic", "err_lnZ_harmonic", LNZ_LABEL),
            ("lnZ_laplace", "err_lnZ_laplace", r"Laplace $\ln Z$"),
        )
        figsize = (7.1, 5.4)
    else:
        evidence_specs = (
            ("lnZ_harmonic", "err_lnZ_harmonic", LNZ_LABEL),
            ("lnZ_laplace", "err_lnZ_laplace", r"Laplace $\ln Z$"),
        )
        figsize = (8.4, 5.35)

    nrows = len(evidence_specs)
    ncols = len(mode.variant_order)
    with plt.style.context(list(mode.style)):
        set_paper_rc()
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, sharey="row",
            constrained_layout=True)
        axes = np.asarray(axes).reshape(nrows, ncols)
        for col, variant in enumerate(mode.variant_order):
            rows = by_variant[variant]
            field_values = np.asarray([row["field"] for row in rows])
            h0 = np.asarray([row["H0_q50"] for row in rows])
            h0_lo = np.asarray([row["H0_q16"] for row in rows])
            h0_hi = np.asarray([row["H0_q84"] for row in rows])
            yerr = np.vstack([h0 - h0_lo, h0_hi - h0])
            for row_idx, (key, err_key, xlabel) in enumerate(evidence_specs):
                ax = axes[row_idx, col]
                x = np.asarray([row[key] for row in rows])
                xerr = np.asarray([abs(row[err_key]) for row in rows])
                ax.errorbar(
                    x, h0, xerr=xerr, yerr=yerr, fmt="none",
                    ecolor="0.55", elinewidth=0.45, capsize=1.1, alpha=0.6,
                    zorder=1)
                sc = ax.scatter(
                    x, h0, c=field_values, cmap=cmap, norm=norm, s=25,
                    edgecolor="0.15", linewidth=0.25, zorder=3)
                add_best_label(ax, rows, key)
                if mode.name == "swift-sph-cola-cic" and variant == "cola_cic":
                    row = row_for_field(rows, HIGHLIGHT_FIELD)
                    if row is not None:
                        highlight_point(ax, row[key], row["H0_q50"])
                ax.set_title(
                    f"{mode.variant_labels[variant]}: {len(rows)} fields",
                    loc="left")
                ax.set_xlabel(xlabel)
                if col == 0:
                    ax.set_ylabel(H0_LABEL)
        cbar = fig.colorbar(
            sc, ax=axes, pad=0.012,
            fraction=0.045 if mode.name == "swift-cola-sph" else 0.035)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_matched_realisations(matched, mode, out_pdf):
    fields = sorted(matched)
    xpos = np.arange(len(mode.variant_order))
    cmap = field_cmap(mode)
    norm = field_norm(fields)
    is_swift_cola = mode.name == "swift-cola-sph"
    figsize = (4.4, 4.75) if is_swift_cola else (6.25, 5.0)
    xlabel = (
        "Manticore reconstruction" if is_swift_cola
        else "Mass-assignment scheme")

    with plt.style.context(list(mode.style)):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 1, figsize=figsize, sharex=True, constrained_layout=True)
        for ax, key, ylabel in (
                (axes[0], "H0_q50", H0_LABEL),
                (axes[1], "lnZ_harmonic", LNZ_LABEL)):
            variant_values = []
            for field in fields:
                values = np.asarray([
                    matched[field][variant][key]
                    for variant in mode.variant_order
                ])
                variant_values.append(values)
                colour = cmap(norm(field))
                ax.plot(
                    xpos, values, color=colour, lw=0.7,
                    alpha=0.45 if is_swift_cola else 0.38, zorder=1)
                ax.scatter(
                    xpos, values, color=colour, s=10 if is_swift_cola else 9,
                    alpha=0.72 if is_swift_cola else 0.64, zorder=2)
            variant_values = np.asarray(variant_values)
            means = np.mean(variant_values, axis=0)
            stds = np.std(variant_values, axis=0, ddof=1)
            ax.errorbar(
                xpos, means, yerr=stds, color="black", marker="o",
                lw=1.25, ms=4.0, capsize=2.4,
                label=("field mean\n(error bar: field-to-field std)"
                       if is_swift_cola else
                       "mean across fields\n(error bar: field-to-field std)"),
                zorder=5)
            ax.set_ylabel(ylabel)
            if key == "H0_q50":
                ax.set_title("Matched realisations", loc="left")
                ax.legend(
                    loc="best",
                    frameon=not is_swift_cola,
                    facecolor="white",
                    edgecolor="none",
                    framealpha=0.86,
                    handlelength=1.7)
        axes[1].set_xticks(
            xpos, [mode.variant_labels[variant]
                   for variant in mode.variant_order])
        axes[1].set_xlabel(xlabel)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=axes, pad=0.015,
            fraction=0.055 if is_swift_cola else 0.045)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_bulk_distributions(by_variant, mode, out_pdf):
    h0_values = [[row["H0_q50"] for row in by_variant[variant]]
                 for variant in mode.variant_order]
    lnz_values = [[row["lnZ_harmonic"] for row in by_variant[variant]]
                  for variant in mode.variant_order]
    is_swift_cola = mode.name == "swift-cola-sph"
    figsize = (6.8, 3.1) if is_swift_cola else (7.2, 3.15)

    with plt.style.context(list(mode.style)):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 2, figsize=figsize, constrained_layout=True)
        for ax, values, ylabel in (
                (axes[0], h0_values, H0_LABEL),
                (axes[1], lnz_values, LNZ_LABEL)):
            parts = ax.violinplot(
                values, positions=np.arange(len(mode.variant_order)),
                widths=0.72, showextrema=False, showmeans=False,
                showmedians=False)
            for variant, body in zip(mode.variant_order, parts["bodies"]):
                body.set_facecolor(mode.variant_colours[variant])
                body.set_edgecolor("none")
                body.set_alpha(0.30 if is_swift_cola else 0.28)
            for idx, (variant, vals) in enumerate(
                    zip(mode.variant_order, values)):
                vals = np.asarray(vals)
                jitter = np.linspace(-0.13, 0.13, vals.size)
                order = np.argsort(vals)
                ax.scatter(
                    idx + jitter[order], vals[order],
                    s=10 if is_swift_cola else 9,
                    color=mode.variant_colours[variant], alpha=0.58,
                    linewidth=0, zorder=3)
                ax.errorbar(
                    idx, np.mean(vals), yerr=np.std(vals, ddof=1),
                    color="black", marker="o", ms=4, capsize=2.4,
                    zorder=4)
                ax.scatter(
                    idx, np.median(vals), color="white", edgecolor="black",
                    marker="s", s=18, linewidth=0.55, zorder=5)
            ax.set_xticks(
                np.arange(len(mode.variant_order)),
                [mode.variant_labels[variant]
                 for variant in mode.variant_order])
            ax.set_ylabel(ylabel)
        axes[0].set_title(
            "Matched-field distribution" if is_swift_cola
            else "Bulk field distribution",
            loc="left")
        axes[1].text(
            0.02, 0.04,
            "black circles: mean; bars: field-to-field std\n"
            "white squares: median",
            transform=axes[1].transAxes, ha="left", va="bottom",
            fontsize=6.5, color="0.25")
        return save_pdf_png(fig, out_pdf)


def plot_mas_pairwise_comparison(matched, mode, out_pdf):
    fields = sorted(matched)
    cmap = field_cmap(mode)
    norm = field_norm(fields)
    pairs = (("CIC", "PCS"), ("CIC", "SPH"), ("PCS", "SPH"))
    specs = (("H0_q50", H0_LABEL), ("lnZ_harmonic", LNZ_LABEL))

    with plt.style.context(list(mode.style)):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 3, figsize=(8.4, 5.2), constrained_layout=True)
        for row_idx, (key, label) in enumerate(specs):
            for col, (variant_x, variant_y) in enumerate(pairs):
                ax = axes[row_idx, col]
                x = np.asarray([matched[field][variant_x][key]
                                for field in fields])
                y = np.asarray([matched[field][variant_y][key]
                                for field in fields])
                limits = identity_limits(x, y)
                ax.plot(limits, limits, color="0.45", lw=0.8, ls="--")
                sc = ax.scatter(
                    x, y, c=fields, cmap=cmap, norm=norm, s=25,
                    edgecolor="0.15", linewidth=0.25, zorder=3)
                ax.set_xlim(limits)
                ax.set_ylim(limits)
                ax.set_xlabel(f"{mode.variant_labels[variant_x]} {label}")
                ax.set_ylabel(f"{mode.variant_labels[variant_y]} {label}")
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


def plot_swift_cola_sph_pairwise_comparison(matched, mode, out_pdf):
    fields = sorted(matched)
    cmap = field_cmap(mode)
    norm = field_norm(fields)

    with plt.style.context(list(mode.style)):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 2, figsize=(6.8, 3.15), constrained_layout=True)
        for ax, key, label in (
                (axes[0], "H0_q50", H0_LABEL),
                (axes[1], "lnZ_harmonic", LNZ_LABEL)):
            swift = np.asarray([
                matched[field]["SWIFT_SPH"][key] for field in fields])
            cola = np.asarray([
                matched[field]["COLA_SPH"][key] for field in fields])
            limits = identity_limits(swift, cola)
            ax.plot(limits, limits, color="0.45", lw=0.8, ls="--")
            sc = ax.scatter(
                swift, cola, c=fields, cmap=cmap, norm=norm, s=28,
                edgecolor="0.15", linewidth=0.25, zorder=3)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_xlabel(f"SWIFT SPH {label}")
            ax.set_ylabel(f"COLA SPH {label}")
            delta = cola - swift
            ax.text(
                0.03, 0.97,
                rf"$\langle\Delta\rangle={np.mean(delta):+.3f}$"
                "\n"
                rf"$\sigma_\Delta={np.std(delta, ddof=1):.3f}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
                bbox={
                    "boxstyle": "round,pad=0.13",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.86,
                })
        cbar = fig.colorbar(sc, ax=axes, pad=0.012, fraction=0.045)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_common_value_comparison(common, mode, out_pdf):
    left_key, right_key = mode.variant_order
    fields = np.asarray([field for field, _, _ in common], dtype=float)
    left_lnz = np.asarray([left["lnZ_harmonic"] for _, left, _ in common],
                          dtype=float)
    right_lnz = np.asarray([right["lnZ_harmonic"] for _, _, right in common],
                           dtype=float)
    left_h0 = np.asarray([left["H0_q50"] for _, left, _ in common])
    right_h0 = np.asarray([right["H0_q50"] for _, _, right in common])
    cmap = field_cmap(mode)
    norm = field_norm(fields)

    with plt.style.context(list(mode.style)):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 2, figsize=(7.1, 3.15), constrained_layout=True)
        for ax, x, y, label in (
                (axes[0], left_lnz, right_lnz, LNZ_LABEL),
                (axes[1], left_h0, right_h0, H0_LABEL)):
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
            ax.set_xlabel(f"{mode.variant_labels[left_key]} {label}")
            ax.set_ylabel(f"{mode.variant_labels[right_key]} {label}")
            ax.set_title(f"{len(common)} common fields", loc="left")
        cbar = fig.colorbar(sc, ax=axes, pad=0.014, fraction=0.045)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_common_deltas(common, mode, out_pdf):
    left_key, right_key = mode.variant_order
    fields = np.asarray([field for field, _, _ in common], dtype=int)
    delta_lnz = np.asarray([
        right["lnZ_harmonic"] - left["lnZ_harmonic"]
        for _, left, right in common
    ], dtype=float)
    delta_h0 = np.asarray([
        right["H0_q50"] - left["H0_q50"]
        for _, left, right in common
    ], dtype=float)
    delta_label = (
        f"{mode.variant_labels[right_key]} - {mode.variant_labels[left_key]}")

    with plt.style.context(list(mode.style)):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 1, figsize=(7.1, 4.4), sharex=True,
            constrained_layout=True)
        for ax, y, ylabel, colour in (
                (axes[0], delta_lnz,
                 rf"$\Delta \ln Z_{{\rm harm}}$ ({delta_label})",
                 mode.variant_colours[right_key]),
                (axes[1], delta_h0,
                 rf"$\Delta H_0$ ({delta_label})", "#473198")):
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


def write_rows_csv(rows, mode, path):
    if mode.name == "cola-mas":
        fieldnames = [
            "task", "mas", "field", "n_H0", "H0_mean", "H0_std",
            "H0_q16", "H0_q50", "H0_q84", "lnZ_harmonic",
            "err_lnZ_harmonic", "lnZ_laplace", "err_lnZ_laplace", "BIC",
            "lnZ_bic", "source", "config",
        ]
    elif mode.name == "swift-cola-sph":
        fieldnames = [
            "task", "variant", "reconstruction", "mas", "field", "n_H0",
            "H0_mean", "H0_std", "H0_q16", "H0_q50", "H0_q84",
            "lnZ_harmonic", "err_lnZ_harmonic", "lnZ_laplace",
            "err_lnZ_laplace", "BIC", "lnZ_bic", "source", "config",
        ]
    else:
        fieldnames = [
            "task", "run", "field", "n_H0", "H0_mean", "H0_std",
            "H0_q16", "H0_q50", "H0_q84", "lnZ_harmonic",
            "err_lnZ_harmonic", "lnZ_laplace", "err_lnZ_laplace", "BIC",
            "lnZ_bic", "source", "config",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_matched_csv(matched, mode, path):
    if mode.name == "cola-mas":
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
    else:
        fieldnames = [
            "field",
            "SWIFT_H0_q50", "COLA_H0_q50", "COLA_minus_SWIFT_H0_q50",
            "SWIFT_lnZ_harmonic", "COLA_lnZ_harmonic",
            "COLA_minus_SWIFT_lnZ_harmonic",
            "SWIFT_lnZ_laplace", "COLA_lnZ_laplace",
            "COLA_minus_SWIFT_lnZ_laplace",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for field in sorted(matched):
            rows = matched[field]
            if mode.name == "cola-mas":
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
            else:
                swift = rows["SWIFT_SPH"]
                cola = rows["COLA_SPH"]
                writer.writerow({
                    "field": field,
                    "SWIFT_H0_q50": swift["H0_q50"],
                    "COLA_H0_q50": cola["H0_q50"],
                    "COLA_minus_SWIFT_H0_q50": (
                        cola["H0_q50"] - swift["H0_q50"]),
                    "SWIFT_lnZ_harmonic": swift["lnZ_harmonic"],
                    "COLA_lnZ_harmonic": cola["lnZ_harmonic"],
                    "COLA_minus_SWIFT_lnZ_harmonic": (
                        cola["lnZ_harmonic"] - swift["lnZ_harmonic"]),
                    "SWIFT_lnZ_laplace": swift["lnZ_laplace"],
                    "COLA_lnZ_laplace": cola["lnZ_laplace"],
                    "COLA_minus_SWIFT_lnZ_laplace": (
                        cola["lnZ_laplace"] - swift["lnZ_laplace"]),
                })


def write_common_csv(common, path):
    fieldnames = [
        "field", "swift_H0_q50", "cola_H0_q50", "delta_H0_q50_cola_swift",
        "swift_lnZ_harmonic", "cola_lnZ_harmonic",
        "delta_lnZ_harmonic_cola_swift", "swift_lnZ_laplace",
        "cola_lnZ_laplace", "delta_lnZ_laplace_cola_swift",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n")
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


def write_matched_summary(by_variant, matched, missing, mode, path):
    lines = [
        f"# {mode.title}",
        "",
    ]
    if mode.name == "swift-cola-sph":
        lines.extend([
            "Source task file: `scripts/runs/tasks_CH0_single.txt`.",
            "The native ManticoreLocalSWIFT outputs are treated as SWIFT SPH.",
        ])
    lines.extend([
        f"Matched fields: {len(matched)}.",
        f"Missing partial fields: {len(missing)}.",
        "",
        "## Bulk Behaviour",
        "",
        "| reconstruction | fields | mean H0 | std H0 | median H0 | "
        "mean harmonic lnZ | std harmonic lnZ | best lnZ field |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    if mode.name == "cola-mas":
        lines[-2] = (
            "| MAS | fields | mean H0 | std H0 | median H0 | "
            "mean harmonic lnZ | std harmonic lnZ | best lnZ field |")
    for variant in mode.variant_order:
        rows = by_variant[variant]
        h0 = np.asarray([row["H0_q50"] for row in rows])
        lnz = np.asarray([row["lnZ_harmonic"] for row in rows])
        best = max(rows, key=lambda row: row["lnZ_harmonic"])
        lines.append(
            f"| {mode.variant_labels[variant]} | {len(rows)} | "
            f"{np.mean(h0):.3f} | {np.std(h0, ddof=1):.3f} | "
            f"{np.median(h0):.3f} | {np.mean(lnz):.3f} | "
            f"{np.std(lnz, ddof=1):.3f} | {best['field']} |")

    lines.extend(["", "## Matched Differences", ""])
    lines.extend([
        "| difference | mean dH0 | std dH0 | median dH0 | "
        "mean dlnZ | std dlnZ | median dlnZ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    if mode.name == "cola-mas":
        pairs = (("CIC", "PCS"), ("CIC", "SPH"), ("PCS", "SPH"))
    else:
        pairs = (("SWIFT_SPH", "COLA_SPH"),)
    for variant_a, variant_b in pairs:
        dh0 = np.asarray([
            matched[field][variant_b]["H0_q50"]
            - matched[field][variant_a]["H0_q50"]
            for field in sorted(matched)
        ])
        dlnz = np.asarray([
            matched[field][variant_b]["lnZ_harmonic"]
            - matched[field][variant_a]["lnZ_harmonic"]
            for field in sorted(matched)
        ])
        lines.append(
            f"| {mode.variant_labels[variant_b]} - "
            f"{mode.variant_labels[variant_a]} | "
            f"{np.mean(dh0):+.3f} | {np.std(dh0, ddof=1):.3f} | "
            f"{np.median(dh0):+.3f} | {np.mean(dlnz):+.3f} | "
            f"{np.std(dlnz, ddof=1):.3f} | {np.median(dlnz):+.3f} |")

    if missing:
        lines.extend(["", "## Missing Partial Fields", ""])
        for field, variants in missing.items():
            labels = ", ".join(mode.variant_labels[variant]
                               for variant in variants)
            lines.append(f"- field {field}: {labels}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_common_summary(rows_by_run, common, mode, path):
    left_key, right_key = mode.variant_order
    lines = [f"# {mode.title}", ""]
    for variant in mode.variant_order:
        rows = rows_by_run[variant]
        best_h = max(rows, key=lambda row: row["lnZ_harmonic"])
        best_l = max(rows, key=lambda row: row["lnZ_laplace"])
        h0 = np.asarray([row["H0_q50"] for row in rows], dtype=float)
        lines.extend([
            f"## {mode.variant_labels[variant]}",
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
        right["lnZ_harmonic"] - left["lnZ_harmonic"]
        for _, left, right in common
    ], dtype=float)
    delta_h0 = np.asarray([
        right["H0_q50"] - left["H0_q50"]
        for _, left, right in common
    ], dtype=float)
    delta_label = (
        f"{mode.variant_labels[right_key]}-{mode.variant_labels[left_key]}")
    lines.extend([
        "## Common Fields",
        "",
        f"Common fields: {len(common)}.",
        "",
        "| quantity | value |",
        "| --- | ---: |",
        f"| mean delta harmonic lnZ, {delta_label} | "
        f"{np.mean(delta_lnz):+.3f} |",
        f"| median delta harmonic lnZ, {delta_label} | "
        f"{np.median(delta_lnz):+.3f} |",
        f"| std delta harmonic lnZ, {delta_label} | "
        f"{np.std(delta_lnz, ddof=1):.3f} |",
        f"| mean delta H0 median, {delta_label} | "
        f"{np.mean(delta_h0):+.3f} |",
        f"| median delta H0 median, {delta_label} | "
        f"{np.median(delta_h0):+.3f} |",
        f"| std delta H0 median, {delta_label} | "
        f"{np.std(delta_h0, ddof=1):.3f} |",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_matched_mode(args, mode):
    rows = load_rows(args.task_file, args.task_min, args.task_max, mode)
    matched, missing = matched_fields(
        rows, mode, require_complete=args.require_complete)
    by_variant = rows_by_variant(matched, mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prefix = mode.output_prefix
    rows_csv = args.output_dir / f"{prefix}_single_field_summary.csv"
    matched_csv = args.output_dir / f"{prefix}_matched_field_summary.csv"
    summary_txt = args.output_dir / (
        f"{prefix}_comparison_summary.txt"
        if mode.name != "cola-mas" else "cola_mas_comparison_summary.txt")
    h0_lnz_pdf = args.output_dir / f"{prefix}_h0_vs_lnz.pdf"
    matched_pdf = args.output_dir / f"{prefix}_matched_realisations.pdf"
    bulk_pdf = args.output_dir / f"{prefix}_bulk_distributions.pdf"
    pairwise_pdf = args.output_dir / f"{prefix}_pairwise_comparison.pdf"

    write_rows_csv(rows, mode, rows_csv)
    write_matched_csv(matched, mode, matched_csv)
    write_matched_summary(by_variant, matched, missing, mode, summary_txt)
    pairwise_plot = (
        plot_mas_pairwise_comparison
        if mode.name == "cola-mas"
        else plot_swift_cola_sph_pairwise_comparison)
    written = [
        rows_csv,
        matched_csv,
        summary_txt,
        *plot_h0_vs_lnz(by_variant, mode, h0_lnz_pdf),
        *plot_matched_realisations(matched, mode, matched_pdf),
        *plot_bulk_distributions(by_variant, mode, bulk_pdf),
        *pairwise_plot(matched, mode, pairwise_pdf),
    ]
    for path in written:
        print(f"Wrote {path}")
    print(
        f"Loaded {len(rows)} {mode.title.lower()} outputs and "
        f"{len(matched)} matched fields.")


def run_common_mode(args, mode):
    rows = load_rows(args.task_file, args.task_min, args.task_max, mode)
    rows_by_run = split_rows(rows, mode)
    common = common_rows(rows_by_run, mode)
    if args.require_complete:
        left_key, right_key = mode.variant_order
        left_fields = {row["field"] for row in rows_by_run[left_key]}
        common_fields = {field for field, _, _ in common}
        missing = sorted(left_fields - common_fields)
        if missing:
            raise FileNotFoundError(
                f"Incomplete {mode.title}. Missing "
                f"{mode.variant_labels[right_key]} fields for {missing}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prefix = mode.output_prefix
    rows_csv = args.output_dir / f"{prefix}_single_field_summary.csv"
    common_csv = args.output_dir / f"{prefix}_common_field_summary.csv"
    summary_txt = args.output_dir / f"{prefix}_summary.txt"
    h0_lnz_pdf = args.output_dir / f"{prefix}_h0_vs_lnz.pdf"
    common_pdf = args.output_dir / f"{prefix}_common_value_comparison.pdf"
    deltas_pdf = args.output_dir / f"{prefix}_common_deltas.pdf"

    write_rows_csv(rows, mode, rows_csv)
    write_common_csv(common, common_csv)
    write_common_summary(rows_by_run, common, mode, summary_txt)
    written = [
        rows_csv,
        common_csv,
        summary_txt,
        *plot_h0_vs_lnz(rows_by_run, mode, h0_lnz_pdf),
        *plot_common_value_comparison(common, mode, common_pdf),
        *plot_common_deltas(common, mode, deltas_pdf),
    ]
    for path in written:
        print(f"Wrote {path}")
    left_key, right_key = mode.variant_order
    print(
        f"Loaded {len(rows_by_run[left_key])} {mode.variant_labels[left_key]} "
        f"fields, {len(rows_by_run[right_key])} "
        f"{mode.variant_labels[right_key]} fields, and {len(common)} "
        "common fields.")


def run(args):
    mode = MODES[args.mode]
    if mode.name == "swift-sph-cola-cic":
        run_common_mode(args, mode)
    else:
        run_matched_mode(args, mode)


def main_for_mode(mode_name):
    args = parse_args(default_mode=mode_name)
    run(args)


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
