#!/usr/bin/env python
"""Compare ManticoreLocalSWIFT and ManticoreLocalCOLA/SPH CH0 runs."""

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
    / "swift_cola_sph_comparison"
)
FIGURE_DPI = 500
VARIANT_ORDER = ("SWIFT_SPH", "COLA_SPH")
VARIANT_LABELS = {
    "SWIFT_SPH": "SWIFT SPH",
    "COLA_SPH": "COLA SPH",
}
VARIANT_COLOURS = {
    "SWIFT_SPH": "#473198",
    "COLA_SPH": "#fe9000",
}
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
        help="Fail unless every SWIFT field has a matching COLA/SPH result.")
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
    which_bias = get_nested(config, ("model", "which_bias"))
    if which_bias != "double_powerlaw":
        return None
    if reconstruction == "ManticoreLocalSWIFT":
        variant = "SWIFT_SPH"
        mas = "SPH"
    elif reconstruction == "ManticoreLocalCOLA":
        mas = get_nested(
            config, ("io", "reconstruction_main", "ManticoreLocalCOLA",
                     "which_MAS"))
        if mas != "SPH":
            return None
        variant = "COLA_SPH"
    else:
        return None
    return {
        "task": task_index,
        "variant": variant,
        "reconstruction": reconstruction,
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
    return sorted(rows, key=lambda row: (row["field"], row["variant"]))


def matched_fields(rows, require_complete=False):
    by_field = {}
    for row in rows:
        by_field.setdefault(row["field"], {})[row["variant"]] = row
    matched = {
        field: {variant: by_field[field][variant]
                for variant in VARIANT_ORDER}
        for field in sorted(by_field)
        if all(variant in by_field[field] for variant in VARIANT_ORDER)
    }
    swift_fields = {
        row["field"] for row in rows if row["variant"] == "SWIFT_SPH"
    }
    missing = {
        field: [variant for variant in VARIANT_ORDER
                if variant not in by_field[field]]
        for field in sorted(swift_fields)
        if any(variant not in by_field[field] for variant in VARIANT_ORDER)
    }
    if require_complete:
        missing_swift = sorted(swift_fields - set(matched))
        if missing or missing_swift:
            raise FileNotFoundError(
                f"Incomplete SWIFT/COLA-SPH comparison. Missing={missing}; "
                f"missing SWIFT matched fields={missing_swift}.")
    return matched, missing


def rows_by_variant(matched):
    return {
        variant: [matched[field][variant] for field in sorted(matched)]
        for variant in VARIANT_ORDER
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
        "swift_cola_sph_fields", FIELD_COLOURS)


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


def plot_h0_vs_lnz(by_variant, out_pdf):
    fields = [row["field"] for rows in by_variant.values() for row in rows]
    cmap = field_cmap()
    norm = field_norm(fields)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 2, figsize=(6.4, 3.0), sharey=True,
            constrained_layout=True)
        for ax, variant in zip(axes, VARIANT_ORDER):
            rows = by_variant[variant]
            field_values = np.asarray([row["field"] for row in rows])
            h0 = np.asarray([row["H0_q50"] for row in rows])
            h0_lo = np.asarray([row["H0_q16"] for row in rows])
            h0_hi = np.asarray([row["H0_q84"] for row in rows])
            yerr = np.vstack([h0 - h0_lo, h0_hi - h0])
            x = np.asarray([row["lnZ_harmonic"] for row in rows])
            xerr = np.asarray([
                abs(row["err_lnZ_harmonic"]) for row in rows])
            ax.errorbar(
                x, h0, xerr=xerr, yerr=yerr, fmt="none",
                ecolor="0.55", elinewidth=0.45, capsize=1.1, alpha=0.55,
                zorder=1)
            sc = ax.scatter(
                x, h0, c=field_values, cmap=cmap, norm=norm, s=25,
                edgecolor="0.15", linewidth=0.25, zorder=3)
            best = max(rows, key=lambda row: row["lnZ_harmonic"])
            ax.scatter(
                best["lnZ_harmonic"], best["H0_q50"], marker="*", s=72,
                color="black", edgecolor="white", linewidth=0.4, zorder=5)
            ax.annotate(
                f"{best['field']}",
                (best["lnZ_harmonic"], best["H0_q50"]),
                xytext=(4, 4), textcoords="offset points", fontsize=6.7)
            ax.set_title(
                f"{VARIANT_LABELS[variant]}: {len(rows)} fields", loc="left")
            ax.set_xlabel(LNZ_LABEL)
        axes[0].set_ylabel(H0_LABEL)
        cbar = fig.colorbar(sc, ax=axes, pad=0.012, fraction=0.045)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_matched_realisations(matched, out_pdf):
    fields = sorted(matched)
    xpos = np.arange(len(VARIANT_ORDER))
    cmap = field_cmap()
    norm = field_norm(fields)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            2, 1, figsize=(4.4, 4.75), sharex=True,
            constrained_layout=True)
        for ax, key, ylabel in (
                (axes[0], "H0_q50", H0_LABEL),
                (axes[1], "lnZ_harmonic", LNZ_LABEL)):
            values_all = []
            for field in fields:
                values = np.asarray([
                    matched[field][variant][key]
                    for variant in VARIANT_ORDER
                ])
                values_all.append(values)
                colour = cmap(norm(field))
                ax.plot(xpos, values, color=colour, lw=0.7, alpha=0.45)
                ax.scatter(
                    xpos, values, color=colour, s=10, alpha=0.72, zorder=3)
            values_all = np.asarray(values_all)
            means = np.mean(values_all, axis=0)
            stds = np.std(values_all, axis=0, ddof=1)
            ax.errorbar(
                xpos, means, yerr=stds, color="black", marker="o",
                lw=1.25, ms=4.0, capsize=2.4,
                label="field mean\n(error bar: field-to-field std)",
                zorder=5)
            ax.set_ylabel(ylabel)
            if key == "H0_q50":
                ax.set_title("Matched realisations", loc="left")
                ax.legend(loc="best", frameon=False, handlelength=1.7)
        axes[1].set_xticks(
            xpos, [VARIANT_LABELS[variant] for variant in VARIANT_ORDER])
        axes[1].set_xlabel("Manticore reconstruction")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, pad=0.015, fraction=0.055)
        cbar.set_label("Manticore field")
        cbar.ax.tick_params(labelsize=7.0)
        return save_pdf_png(fig, out_pdf)


def plot_pairwise_comparison(matched, out_pdf):
    fields = sorted(matched)
    cmap = field_cmap()
    norm = field_norm(fields)

    with plt.style.context(["science", "no-latex"]):
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


def plot_bulk_distributions(by_variant, out_pdf):
    h0_values = [[row["H0_q50"] for row in by_variant[variant]]
                 for variant in VARIANT_ORDER]
    lnz_values = [[row["lnZ_harmonic"] for row in by_variant[variant]]
                  for variant in VARIANT_ORDER]

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 2, figsize=(6.8, 3.1), constrained_layout=True)
        for ax, values, ylabel in (
                (axes[0], h0_values, H0_LABEL),
                (axes[1], lnz_values, LNZ_LABEL)):
            parts = ax.violinplot(
                values, positions=np.arange(len(VARIANT_ORDER)), widths=0.72,
                showextrema=False, showmeans=False, showmedians=False)
            for variant, body in zip(VARIANT_ORDER, parts["bodies"]):
                body.set_facecolor(VARIANT_COLOURS[variant])
                body.set_edgecolor("none")
                body.set_alpha(0.30)
            for idx, (variant, vals) in enumerate(
                    zip(VARIANT_ORDER, values)):
                vals = np.asarray(vals)
                jitter = np.linspace(-0.13, 0.13, vals.size)
                order = np.argsort(vals)
                ax.scatter(
                    idx + jitter[order], vals[order], s=10,
                    color=VARIANT_COLOURS[variant], alpha=0.58,
                    linewidth=0, zorder=3)
                ax.errorbar(
                    idx, np.mean(vals), yerr=np.std(vals, ddof=1),
                    color="black", marker="o", ms=4, capsize=2.4,
                    zorder=4)
                ax.scatter(
                    idx, np.median(vals), color="white", edgecolor="black",
                    marker="s", s=18, linewidth=0.55, zorder=5)
            ax.set_xticks(
                np.arange(len(VARIANT_ORDER)),
                [VARIANT_LABELS[variant] for variant in VARIANT_ORDER])
            ax.set_ylabel(ylabel)
        axes[0].set_title("Matched-field distribution", loc="left")
        axes[1].text(
            0.02, 0.04,
            "black circles: mean; bars: field-to-field std\n"
            "white squares: median",
            transform=axes[1].transAxes, ha="left", va="bottom",
            fontsize=6.5, color="0.25")
        return save_pdf_png(fig, out_pdf)


def write_rows_csv(rows, path):
    fieldnames = [
        "task", "variant", "reconstruction", "mas", "field", "n_H0",
        "H0_mean", "H0_std", "H0_q16", "H0_q50", "H0_q84",
        "lnZ_harmonic", "err_lnZ_harmonic", "lnZ_laplace",
        "err_lnZ_laplace", "BIC", "lnZ_bic", "source", "config",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_matched_csv(matched, path):
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
            swift = matched[field]["SWIFT_SPH"]
            cola = matched[field]["COLA_SPH"]
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


def write_summary(by_variant, matched, missing, path):
    lines = [
        "# SWIFT/SPH and COLA/SPH Single-Field Comparison",
        "",
        "Source task file: `scripts/runs/tasks_CH0_single.txt`.",
        "The native ManticoreLocalSWIFT outputs are treated as SWIFT SPH.",
        f"Matched fields: {len(matched)}.",
        f"Missing partial fields: {len(missing)}.",
        "",
        "## Bulk Behaviour",
        "",
        "| reconstruction | fields | mean H0 | std H0 | median H0 | "
        "mean harmonic lnZ | std harmonic lnZ | best lnZ field |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANT_ORDER:
        rows = by_variant[variant]
        h0 = np.asarray([row["H0_q50"] for row in rows])
        lnz = np.asarray([row["lnZ_harmonic"] for row in rows])
        best = max(rows, key=lambda row: row["lnZ_harmonic"])
        lines.append(
            f"| {VARIANT_LABELS[variant]} | {len(rows)} | "
            f"{np.mean(h0):.3f} | {np.std(h0, ddof=1):.3f} | "
            f"{np.median(h0):.3f} | {np.mean(lnz):.3f} | "
            f"{np.std(lnz, ddof=1):.3f} | {best['field']} |")

    dh0 = np.asarray([
        matched[field]["COLA_SPH"]["H0_q50"]
        - matched[field]["SWIFT_SPH"]["H0_q50"]
        for field in sorted(matched)
    ])
    dlnz = np.asarray([
        matched[field]["COLA_SPH"]["lnZ_harmonic"]
        - matched[field]["SWIFT_SPH"]["lnZ_harmonic"]
        for field in sorted(matched)
    ])
    lines.extend([
        "",
        "## Matched Differences",
        "",
        "| difference | mean dH0 | std dH0 | median dH0 | "
        "mean dlnZ | std dlnZ | median dlnZ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| COLA SPH - SWIFT SPH | {np.mean(dh0):+.3f} | "
        f"{np.std(dh0, ddof=1):.3f} | {np.median(dh0):+.3f} | "
        f"{np.mean(dlnz):+.3f} | {np.std(dlnz, ddof=1):.3f} | "
        f"{np.median(dlnz):+.3f} |",
    ])
    if missing:
        lines.extend(["", "## Missing Partial Fields", ""])
        for field, variants in missing.items():
            labels = ", ".join(VARIANT_LABELS[variant]
                               for variant in variants)
            lines.append(f"- field {field}: {labels}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    rows = load_rows(args.task_file)
    matched, missing = matched_fields(
        rows, require_complete=args.require_complete)
    by_variant = rows_by_variant(matched)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows_csv = args.output_dir / "swift_cola_sph_single_field_summary.csv"
    matched_csv = args.output_dir / "swift_cola_sph_matched_field_summary.csv"
    summary_txt = args.output_dir / "swift_cola_sph_comparison_summary.txt"
    h0_lnz_pdf = args.output_dir / "swift_cola_sph_h0_vs_lnz.pdf"
    matched_pdf = args.output_dir / "swift_cola_sph_matched_realisations.pdf"
    pairwise_pdf = args.output_dir / "swift_cola_sph_pairwise_comparison.pdf"
    bulk_pdf = args.output_dir / "swift_cola_sph_bulk_distributions.pdf"

    write_rows_csv(rows, rows_csv)
    write_matched_csv(matched, matched_csv)
    write_summary(by_variant, matched, missing, summary_txt)
    written = [
        rows_csv,
        matched_csv,
        summary_txt,
        *plot_h0_vs_lnz(by_variant, h0_lnz_pdf),
        *plot_matched_realisations(matched, matched_pdf),
        *plot_pairwise_comparison(matched, pairwise_pdf),
        *plot_bulk_distributions(by_variant, bulk_pdf),
    ]

    for path in written:
        print(f"Wrote {path}")
    print(
        f"Loaded {len(rows)} SWIFT/COLA-SPH outputs and "
        f"{len(matched)} matched fields.")


if __name__ == "__main__":
    main()
