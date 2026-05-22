#!/usr/bin/env python
"""Compare CH0 single-field sampled-bias and fixed-bias runs."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: F401,E402
import tomllib  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
TASK_SAMPLED = ROOT / "scripts" / "runs" / "tasks_CH0_single.txt"
TASK_FIXED = ROOT / "scripts" / "runs" / "tasks_CH0_single_fixed_bias.txt"
DEFAULT_OUTDIR = (
    Path(__file__).resolve().parent
    / "ch0_single_fixed_bias_comparison_plots")
FIGURE_DPI = 500
H0_LABEL = (
    r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$")
BIAS_PARAMS = ("alpha_low", "alpha_high", "log_rho_t", "log_rho_width")
BIAS_LABELS = {
    "alpha_low": r"$\alpha_\mathrm{low}$",
    "alpha_high": r"$\alpha_\mathrm{high}$",
    "log_rho_t": r"$\log\rho_t$",
    "log_rho_width": r"$\Delta\log\rho$",
}
FAMILY_LABELS = {
    "cola_cic": "COLA/CIC",
    "swift": "SWIFT/SPH",
}
FAMILY_COLOURS = {
    "cola_cic": "#1e42b9",
    "swift": "#87193d",
}
MODE_MARKERS = {
    "sampled": "o",
    "fixed": "s",
}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sampled-task-file", type=Path, default=TASK_SAMPLED,
        help="Task file with sampled-bias single-field configs.")
    parser.add_argument(
        "--fixed-task-file", type=Path, default=TASK_FIXED,
        help="Task file with fixed-bias single-field configs.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTDIR,
        help="Directory for plots and summary tables.")
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
        if key not in value:
            return default
        value = value[key]
    return value


def reconstruction_family(config):
    reconstruction = get_nested(config, ("io", "SH0ES", "reconstruction"))
    if reconstruction == "ManticoreLocalSWIFT":
        return "swift"
    if reconstruction == "ManticoreLocalCOLA":
        mas = get_nested(
            config,
            ("io", "reconstruction_main", "ManticoreLocalCOLA", "which_MAS"),
            "")
        if mas == "CIC":
            return "cola_cic"
    return None


def task_specs(task_file, mode):
    specs = []
    with repo_path(task_file).open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            task, config_rel = line.split(maxsplit=1)
            config_path = repo_path(config_rel)
            with config_path.open("rb") as config_handle:
                config = tomllib.load(config_handle)
            family = reconstruction_family(config)
            if family is None:
                continue
            specs.append({
                "mode": mode,
                "task": int(task),
                "family": family,
                "field": int(get_nested(config, ("io", "field_indices"))),
                "config": str(config_path),
                "source": str(repo_path(get_nested(
                    config, ("io", "fname_output")))),
                "which_bias": get_nested(config, ("model", "which_bias")),
                **fixed_bias_values_from_config(config),
            })
    return specs


def fixed_bias_values_from_config(config):
    values = {}
    for name in BIAS_PARAMS:
        prior = get_nested(config, ("model", "priors", name), {})
        if prior.get("dist") == "delta":
            values[f"fixed_{name}"] = float(prior["value"])
        else:
            values[f"fixed_{name}"] = np.nan
    return values


def finite_samples(handle, name, source):
    dataset = f"samples/{name}"
    if dataset not in handle:
        return None
    samples = np.asarray(handle[dataset], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"`{source}` has no finite `{name}` samples.")
    return samples


def read_scalar(handle, name, source):
    if name not in handle:
        raise KeyError(f"`{source}` does not contain `{name}`.")
    value = float(handle[name][()])
    if not np.isfinite(value):
        raise ValueError(f"`{source}` has non-finite `{name}`: {value}.")
    return value


def h0_summary(samples):
    q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
    return {
        "H0_mean": float(np.mean(samples)),
        "H0_std": float(np.std(samples, ddof=1)),
        "H0_q16": float(q16),
        "H0_q50": float(q50),
        "H0_q84": float(q84),
    }


def read_result(spec):
    source = Path(spec["source"])
    with h5py.File(source, "r") as handle:
        h0 = finite_samples(handle, "H0", source)
        row = {
            **spec,
            "n_H0": int(h0.size),
            **h0_summary(h0),
            "lnZ_harmonic": read_scalar(handle, "gof/lnZ_harmonic", source),
            "err_lnZ_harmonic": read_scalar(
                handle, "gof/err_lnZ_harmonic", source),
            "lnZ_laplace": read_scalar(handle, "gof/lnZ_laplace", source),
            "err_lnZ_laplace": read_scalar(
                handle, "gof/err_lnZ_laplace", source),
            "BIC": read_scalar(handle, "gof/BIC", source),
        }
        for name in BIAS_PARAMS:
            samples = finite_samples(handle, name, source)
            if samples is not None:
                q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
                row[f"{name}_q16"] = float(q16)
                row[f"{name}_q50"] = float(q50)
                row[f"{name}_q84"] = float(q84)
        return row


def load_results(sampled_task_file, fixed_task_file, allow_missing):
    specs = [
        *task_specs(sampled_task_file, "sampled"),
        *task_specs(fixed_task_file, "fixed"),
    ]
    rows = []
    missing = []
    for spec in specs:
        if not Path(spec["source"]).is_file():
            missing.append(spec)
            continue
        rows.append(read_result(spec))
    if missing and not allow_missing:
        preview = "\n".join(spec["source"] for spec in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} result files are missing. First missing:\n"
            f"{preview}")
    return rows, missing


def pair_rows(rows):
    by_key = {}
    for row in rows:
        key = (row["family"], row["field"], row["mode"])
        by_key[key] = row
    pairs = []
    families = sorted({row["family"] for row in rows})
    for family in families:
        fields = sorted({
            row["field"] for row in rows
            if row["family"] == family
        })
        for field in fields:
            sampled = by_key.get((family, field, "sampled"))
            fixed = by_key.get((family, field, "fixed"))
            if sampled is None or fixed is None:
                continue
            pairs.append({
                "family": family,
                "field": field,
                "sampled_task": sampled["task"],
                "fixed_task": fixed["task"],
                "sampled_H0_q50": sampled["H0_q50"],
                "fixed_H0_q50": fixed["H0_q50"],
                "delta_H0_q50": fixed["H0_q50"] - sampled["H0_q50"],
                "sampled_H0_mean": sampled["H0_mean"],
                "fixed_H0_mean": fixed["H0_mean"],
                "delta_H0_mean": fixed["H0_mean"] - sampled["H0_mean"],
                "sampled_lnZ_harmonic": sampled["lnZ_harmonic"],
                "fixed_lnZ_harmonic": fixed["lnZ_harmonic"],
                "delta_lnZ_harmonic": (
                    fixed["lnZ_harmonic"] - sampled["lnZ_harmonic"]),
                "sampled_BIC": sampled["BIC"],
                "fixed_BIC": fixed["BIC"],
                "delta_BIC": fixed["BIC"] - sampled["BIC"],
            })
    return pairs


def set_paper_rc():
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.8,
        "text.usetex": False,
    })


def save_pdf_png(fig, out_pdf):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return [out_pdf, out_png]


def rows_for(rows, family, mode):
    return sorted(
        [row for row in rows
         if row["family"] == family and row["mode"] == mode],
        key=lambda row: row["field"])


def plot_h0_vs_lnz(rows, out_pdf):
    families = ["cola_cic", "swift"]
    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 2, figsize=(7.15, 3.15), sharey=True,
            constrained_layout=True)
        for ax, family in zip(axes, families):
            sampled = rows_for(rows, family, "sampled")
            fixed = rows_for(rows, family, "fixed")
            sampled_by_field = {row["field"]: row for row in sampled}
            fixed_by_field = {row["field"]: row for row in fixed}
            for field in sorted(set(sampled_by_field) & set(fixed_by_field)):
                srow = sampled_by_field[field]
                frow = fixed_by_field[field]
                ax.plot(
                    [srow["lnZ_harmonic"], frow["lnZ_harmonic"]],
                    [srow["H0_q50"], frow["H0_q50"]],
                    color="0.75", lw=0.45, alpha=0.55, zorder=1)

            for mode, alpha, face in (
                    ("sampled", 0.85, "none"),
                    ("fixed", 0.95, FAMILY_COLOURS[family])):
                data = rows_for(rows, family, mode)
                if not data:
                    continue
                h0 = np.asarray([row["H0_q50"] for row in data])
                h0_lo = np.asarray([row["H0_q16"] for row in data])
                h0_hi = np.asarray([row["H0_q84"] for row in data])
                lnz = np.asarray([row["lnZ_harmonic"] for row in data])
                lnz_err = np.asarray([
                    abs(row["err_lnZ_harmonic"]) for row in data])
                ax.errorbar(
                    lnz, h0, xerr=lnz_err,
                    yerr=np.vstack([h0 - h0_lo, h0_hi - h0]),
                    fmt="none", color="0.50", lw=0.35, capsize=1.0,
                    alpha=0.42, zorder=2)
                ax.scatter(
                    lnz, h0, s=20, marker=MODE_MARKERS[mode],
                    facecolor=face, edgecolor=FAMILY_COLOURS[family],
                    lw=0.65, alpha=alpha, label=mode, zorder=3)
            ax.set_title(FAMILY_LABELS[family], loc="left")
            ax.set_xlabel(r"harmonic $\ln Z$")
            ax.legend(frameon=False, loc="best", handletextpad=0.3)
        axes[0].set_ylabel(H0_LABEL)
        return save_pdf_png(fig, out_pdf)


def plot_matched_h0(rows, pairs, out_pdf):
    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 2, figsize=(7.15, 6.1), constrained_layout=True)
        for family in ("cola_cic", "swift"):
            colour = FAMILY_COLOURS[family]
            label = FAMILY_LABELS[family]
            fpairs = [pair for pair in pairs if pair["family"] == family]
            if not fpairs:
                continue
            x = np.asarray([pair["sampled_H0_q50"] for pair in fpairs])
            y = np.asarray([pair["fixed_H0_q50"] for pair in fpairs])
            d_h0 = np.asarray([pair["delta_H0_q50"] for pair in fpairs])
            d_lnz = np.asarray([
                pair["delta_lnZ_harmonic"] for pair in fpairs])
            fields = np.asarray([pair["field"] for pair in fpairs])

            axes[0, 0].scatter(
                x, y, s=21, color=colour, alpha=0.78,
                edgecolor="white", lw=0.35, label=label)
            axes[0, 1].scatter(
                fields, d_h0, s=19, color=colour, alpha=0.78,
                edgecolor="white", lw=0.35, label=label)
            axes[1, 0].hist(
                d_h0, bins=min(18, max(7, int(np.sqrt(len(d_h0))) + 2)),
                histtype="step", density=True, color=colour, lw=1.1,
                label=(rf"{label}: med={np.median(d_h0):+.2f}"))
            axes[1, 1].scatter(
                fields, d_lnz, s=19, color=colour, alpha=0.78,
                edgecolor="white", lw=0.35, label=label)

        lims = [
            min(axes[0, 0].get_xlim()[0], axes[0, 0].get_ylim()[0]),
            max(axes[0, 0].get_xlim()[1], axes[0, 0].get_ylim()[1]),
        ]
        axes[0, 0].plot(lims, lims, color="0.35", lw=0.8, ls="--")
        axes[0, 0].set_xlim(lims)
        axes[0, 0].set_ylim(lims)
        axes[0, 0].set_xlabel(r"sampled-bias median $H_0$")
        axes[0, 0].set_ylabel(r"fixed-bias median $H_0$")
        axes[0, 0].set_title("Matched realisations", loc="left")
        axes[0, 0].legend(frameon=False, loc="best")

        axes[0, 1].axhline(0.0, color="0.35", lw=0.8, ls="--")
        axes[0, 1].set_xlabel("Manticore field")
        axes[0, 1].set_ylabel(r"$\Delta H_0$ (fixed - sampled)")
        axes[0, 1].set_title("Per-field H0 shift", loc="left")
        axes[0, 1].legend(frameon=False, loc="best")

        axes[1, 0].axvline(0.0, color="0.35", lw=0.8, ls="--")
        axes[1, 0].set_xlabel(r"$\Delta H_0$ (fixed - sampled)")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_title("Shift distribution", loc="left")
        axes[1, 0].legend(frameon=False, loc="best")

        axes[1, 1].axhline(0.0, color="0.35", lw=0.8, ls="--")
        axes[1, 1].set_xlabel("Manticore field")
        axes[1, 1].set_ylabel(r"$\Delta \ln Z$ (fixed - sampled)")
        axes[1, 1].set_title("Evidence shift", loc="left")
        axes[1, 1].legend(frameon=False, loc="best")
        return save_pdf_png(fig, out_pdf)


def plot_bias_parameter_context(rows, out_pdf):
    sampled_rows = [row for row in rows if row["mode"] == "sampled"]
    fixed_rows = [row for row in rows if row["mode"] == "fixed"]
    fixed_values = {}
    for family in ("cola_cic", "swift"):
        family_fixed = [row for row in fixed_rows if row["family"] == family]
        fixed_values[family] = {
            name: np.nanmedian([
                row[f"fixed_{name}"] for row in family_fixed])
            for name in BIAS_PARAMS
        }

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 2, figsize=(7.15, 5.4), constrained_layout=True)
        for ax, name in zip(axes.ravel(), BIAS_PARAMS):
            for family in ("cola_cic", "swift"):
                data = [
                    row[f"{name}_q50"] for row in sampled_rows
                    if row["family"] == family and f"{name}_q50" in row
                ]
                if not data:
                    continue
                data = np.asarray(data, dtype=float)
                ax.hist(
                    data, bins=min(18, max(7, int(np.sqrt(len(data))) + 2)),
                    histtype="step", density=True,
                    color=FAMILY_COLOURS[family], lw=1.05,
                    label=f"{FAMILY_LABELS[family]} sampled")
                ax.axvline(
                    fixed_values[family][name],
                    color=FAMILY_COLOURS[family], lw=0.95, ls="--",
                    label=f"{FAMILY_LABELS[family]} fixed")
            ax.set_xlabel(BIAS_LABELS[name])
            ax.set_ylabel("Density")
            ax.set_title(name, loc="left")
            ax.legend(frameon=False, loc="best")
        return save_pdf_png(fig, out_pdf)


def write_rows_csv(rows, path):
    fieldnames = [
        "mode", "family", "field", "task", "which_bias", "H0_mean",
        "H0_std", "H0_q16", "H0_q50", "H0_q84", "lnZ_harmonic",
        "err_lnZ_harmonic", "lnZ_laplace", "err_lnZ_laplace", "BIC",
        *[f"{name}_q50" for name in BIAS_PARAMS],
        *[f"fixed_{name}" for name in BIAS_PARAMS],
        "source", "config",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["family"], r["mode"],
                                               r["field"])):
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_pairs_csv(pairs, path):
    fieldnames = [
        "family", "field", "sampled_task", "fixed_task",
        "sampled_H0_q50", "fixed_H0_q50", "delta_H0_q50",
        "sampled_H0_mean", "fixed_H0_mean", "delta_H0_mean",
        "sampled_lnZ_harmonic", "fixed_lnZ_harmonic",
        "delta_lnZ_harmonic", "sampled_BIC", "fixed_BIC", "delta_BIC",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in sorted(pairs, key=lambda r: (r["family"], r["field"])):
            writer.writerow(pair)


def summary_stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "q16": float(np.percentile(values, 16.0)),
        "q84": float(np.percentile(values, 84.0)),
    }


def write_summary(rows, pairs, missing, path):
    lines = [
        "# CH0 Fixed-Bias Versus Sampled-Bias Comparison",
        "",
        f"Loaded result rows: {len(rows)}.",
        f"Matched realisations: {len(pairs)}.",
        f"Missing outputs: {len(missing)}.",
        "",
    ]

    row_counts = Counter((row["family"], row["mode"]) for row in rows)
    lines.extend(["## Result Counts", ""])
    for family in ("cola_cic", "swift"):
        lines.append(
            f"- {FAMILY_LABELS[family]}: "
            f"{row_counts[(family, 'sampled')]} sampled, "
            f"{row_counts[(family, 'fixed')]} fixed.")
    lines.append("")

    lines.extend(["## Paired Shifts", ""])
    lines.append(
        "| family | N | median dH0 | 16-84 dH0 | median dlnZ | 16-84 dlnZ |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for family in ("cola_cic", "swift"):
        fpairs = [pair for pair in pairs if pair["family"] == family]
        if not fpairs:
            continue
        d_h0 = summary_stats([pair["delta_H0_q50"] for pair in fpairs])
        d_lnz = summary_stats([
            pair["delta_lnZ_harmonic"] for pair in fpairs])
        lines.append(
            f"| {FAMILY_LABELS[family]} | {len(fpairs)} | "
            f"{d_h0['median']:+.3f} | "
            f"{d_h0['q16']:+.3f}, {d_h0['q84']:+.3f} | "
            f"{d_lnz['median']:+.3f} | "
            f"{d_lnz['q16']:+.3f}, {d_lnz['q84']:+.3f} |")

    lines.extend(["", "## H0 Distributions", ""])
    lines.append("| family | mode | N | median field H0 | 16-84 field H0 |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for family in ("cola_cic", "swift"):
        for mode in ("sampled", "fixed"):
            values = [
                row["H0_q50"] for row in rows
                if row["family"] == family and row["mode"] == mode
            ]
            if not values:
                continue
            stats = summary_stats(values)
            lines.append(
                f"| {FAMILY_LABELS[family]} | {mode} | {len(values)} | "
                f"{stats['median']:.3f} | "
                f"{stats['q16']:.3f}, {stats['q84']:.3f} |")

    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, missing = load_results(
        args.sampled_task_file, args.fixed_task_file, args.allow_missing)
    pairs = pair_rows(rows)

    rows_csv = args.output_dir / "ch0_single_fixed_bias_comparison_rows.csv"
    pairs_csv = args.output_dir / "ch0_single_fixed_bias_comparison_pairs.csv"
    summary_txt = args.output_dir / "ch0_single_fixed_bias_comparison_summary.txt"
    write_rows_csv(rows, rows_csv)
    write_pairs_csv(pairs, pairs_csv)
    write_summary(rows, pairs, missing, summary_txt)

    written = [
        rows_csv,
        pairs_csv,
        summary_txt,
        *plot_h0_vs_lnz(
            rows, args.output_dir
            / "ch0_single_fixed_bias_comparison_h0_vs_lnz.pdf"),
        *plot_matched_h0(
            rows, pairs, args.output_dir
            / "ch0_single_fixed_bias_comparison_matched_fields.pdf"),
        *plot_bias_parameter_context(
            rows, args.output_dir
            / "ch0_single_fixed_bias_comparison_bias_parameters.pdf"),
    ]

    for path in written:
        print(f"Wrote {path}")

    print(f"Loaded {len(rows)} rows; matched {len(pairs)} realisations.")


if __name__ == "__main__":
    main()
