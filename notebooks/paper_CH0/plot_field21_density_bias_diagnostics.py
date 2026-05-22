#!/usr/bin/env python
"""Diagnose why SWIFT field 21 has a low galaxy-density integral."""

from argparse import ArgumentParser
import csv
import os
from pathlib import Path

import matplotlib
import numpy as np
from scipy.special import log_ndtr, logsumexp

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import candel  # noqa: E402
from candel.pvdata.volume_density import _load_volume_data_for_H0  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
INTEGRAL_DIR = (
    Path(__file__).resolve().parent
    / "ch0_single_selection_integral_plots")
ROWS_CSV = INTEGRAL_DIR / "ch0_single_selection_integral_rows.csv"
DEFAULT_OUTDIR = INTEGRAL_DIR
FIGURE_DPI = 500

BIAS_PARAMS = ("alpha_low", "alpha_high", "log_rho_t", "log_rho_width")
SWIFT_CONFIG = (
    ROOT / "scripts" / "runs" / "generated_configs" / "CH0_single"
    / "CH0_sel-SN_magnitude_ManticoreLocalSWIFT_field21_single.toml")


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows-csv", type=Path, default=ROWS_CSV,
        help="Per-run integral table from plot_single_selection_integral_diagnostics.py.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTDIR,
        help="Directory for plots and tables.")
    return parser.parse_args()


def repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def read_rows(path):
    rows = []
    with repo_path(path).open() as handle:
        for row in csv.DictReader(handle):
            converted = {}
            for key, value in row.items():
                if value == "":
                    converted[key] = value
                    continue
                try:
                    number = float(value)
                except ValueError:
                    converted[key] = value
                else:
                    converted[key] = (
                        int(number)
                        if key in {"field", "task", "n_hosts", "n_samples"}
                        else number)
            rows.append(converted)
    return rows


def bias_params_from_row(row):
    return [row[f"{name}_q50"] for name in BIAS_PARAMS]


def log_galaxy_bias(log_rho, params):
    alpha_low, alpha_high, log_rho_t, log_rho_width = params
    log_x = log_rho - log_rho_t
    z = log_x / log_rho_width
    return (
        alpha_low * log_x
        + ((alpha_high - alpha_low)
           * log_rho_width
           * np.logaddexp(0.0, z))
    )


def load_swift_volume(rows):
    fields = np.asarray(
        [row["field"] for row in rows], dtype=np.int32)
    config = candel.load_config(str(SWIFT_CONFIG), replace_los_prior=False)
    field_kwargs = config["io"]["reconstruction_main"]["ManticoreLocalSWIFT"]
    volume = _load_volume_data_for_H0(
        "ManticoreLocalSWIFT",
        field_kwargs,
        fields,
        "double_powerlaw",
        config["model"].get("Om", 0.3),
        subcube_radius=60.0,
        voxel_subsample_fraction=1.0,
        load_velocity=False,
        geometry="sphere",
        cache_dir=str(ROOT / "data" / "field_cache"),
        cache_enabled=True,
        field_smoothing_scale=None,
        velocity_field_smoothing_scale=None,
    )
    return fields, volume


def reference_selection_params(all_rows):
    h0_ref = float(np.median([row["H0_q50"] for row in all_rows]))
    mb_ref = float(np.median([row["M_B_q50"] for row in all_rows]))
    return h0_ref, mb_ref, 0.13486865, 14.0, 0.15


def binned_contributions(log_rho, log_weight, log_p_sel, params, bins):
    log_n = log_galaxy_bias(log_rho, params)
    out = {"rho": [], "ng": [], "sel": [], "volume": []}
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (log_rho >= left) & (log_rho < right)
        if not np.any(mask):
            for value in out.values():
                value.append(0.0)
            continue
        out["rho"].append(float(np.exp(logsumexp(
            log_rho[mask] + log_weight[mask]))))
        out["ng"].append(float(np.exp(logsumexp(
            log_n[mask] + log_weight[mask]))))
        out["sel"].append(float(np.exp(logsumexp(
            log_p_sel[mask] + log_n[mask] + log_weight[mask]))))
        out["volume"].append(float(np.exp(logsumexp(log_weight[mask]))))
    return {key: np.asarray(value) for key, value in out.items()}


def field21_diagnostics(rows):
    swift_sampled = [
        row for row in rows
        if row["family"] == "swift" and row["mode"] == "sampled"]
    field21 = next(row for row in swift_sampled if row["field"] == 21)
    field21_ranks = {}
    for key in (
            "log_density_integral",
            "log_expected_galaxy_integral",
            "log_selected_galaxy_integral"):
        ordered = sorted(swift_sampled, key=lambda row: row[key])
        field21_ranks[f"{key}_rank_low"] = (
            [row["field"] for row in ordered].index(21) + 1)
    fields, volume = load_swift_volume(swift_sampled)
    index21 = int(np.where(fields == 21)[0][0])

    h0_ref, mb_ref, e_mag, mag_lim, mag_width = (
        reference_selection_params(rows))
    h = h0_ref / 100.0
    log_rho_fields = np.asarray(volume["density_3d_fields"], dtype=float)
    mu = np.asarray(volume["mu_at_h1_3d"], dtype=float) - 5 * np.log10(h)
    log_weight = float(volume["log_dV_3d"]) - 3 * np.log(h)
    if "log_volume_weight_3d" in volume:
        log_weight = (
            log_weight
            + np.asarray(volume["log_volume_weight_3d"], dtype=float))
    log_p_sel = log_ndtr(
        (mag_lim - (mu + mb_ref)) / np.sqrt(e_mag**2 + mag_width**2))

    finite = log_rho_fields[np.isfinite(log_rho_fields)]
    lo, hi = np.quantile(finite, [0.001, 0.999])
    bins = np.linspace(lo, hi, 45)
    centers = 0.5 * (bins[:-1] + bins[1:])

    all_binned = []
    for index, row in enumerate(swift_sampled):
        all_binned.append(binned_contributions(
            log_rho_fields[index],
            log_weight,
            log_p_sel,
            bias_params_from_row(row),
            bins))
    field21_binned = all_binned[index21]
    median_binned = {
        key: np.median(
            np.stack([binned[key] for binned in all_binned]), axis=0)
        for key in ("rho", "ng", "sel", "volume")
    }

    field21_params = bias_params_from_row(field21)
    median_params = [
        float(np.median([row[f"{name}_q50"] for row in swift_sampled]))
        for name in BIAS_PARAMS]
    fixed_params = [1.542, 0.286, -0.027, 0.954]

    log_rho21 = log_rho_fields[index21]
    counterfactuals = []
    for name, params in [
            ("field21 sampled bias", field21_params),
            ("SWIFT sampled median bias", median_params),
            ("fixed bias", fixed_params)]:
        log_n = log_galaxy_bias(log_rho21, params)
        counterfactuals.append({
            "bias": name,
            "log_ng": float(logsumexp(log_n + log_weight)),
            "log_selected_ng": float(logsumexp(
                log_p_sel + log_n + log_weight)),
        })

    broad_edges = np.asarray(
        [-np.inf, -1.0, -0.5, 0.0, 0.3187784105539322, 0.75, np.inf])
    broad_rows = []
    broad_field21 = binned_contributions(
        log_rho21, log_weight, log_p_sel, field21_params, broad_edges)
    broad_all = [
        binned_contributions(
            log_rho_fields[index], log_weight, log_p_sel,
            bias_params_from_row(row), broad_edges)
        for index, row in enumerate(swift_sampled)]
    broad_median = {
        key: np.median(
            np.stack([binned[key] for binned in broad_all]), axis=0)
        for key in ("rho", "ng", "sel", "volume")
    }
    for i, (left, right) in enumerate(zip(broad_edges[:-1], broad_edges[1:])):
        item = {
            "log_rho_left": left,
            "log_rho_right": right,
            "rho_left": 0.0 if not np.isfinite(left) else float(np.exp(left)),
            "rho_right": np.inf if not np.isfinite(right) else float(np.exp(right)),
        }
        for key in ("rho", "ng", "sel"):
            f21_value = broad_field21[key][i]
            median_value = broad_median[key][i]
            item[f"field21_{key}"] = f21_value
            item[f"median_{key}"] = median_value
            item[f"{key}_ratio"] = (
                f21_value / median_value if median_value > 0 else np.nan)
            item[f"{key}_deficit"] = median_value - f21_value
        broad_rows.append(item)

    fine_rows = []
    for i, (left, right, center) in enumerate(zip(
            bins[:-1], bins[1:], centers)):
        item = {
            "log_rho_left": float(left),
            "log_rho_right": float(right),
            "log_rho_center": float(center),
            "rho_center": float(np.exp(center)),
        }
        for key in ("rho", "ng", "sel", "volume"):
            f21_value = field21_binned[key][i]
            median_value = median_binned[key][i]
            item[f"field21_{key}"] = f21_value
            item[f"median_{key}"] = median_value
            item[f"{key}_ratio"] = (
                f21_value / median_value if median_value > 0 else np.nan)
            item[f"{key}_deficit"] = median_value - f21_value
        fine_rows.append(item)

    return {
        "field21": field21,
        "field21_ranks": field21_ranks,
        "field21_params": field21_params,
        "median_params": median_params,
        "fixed_params": fixed_params,
        "centers": centers,
        "field21_binned": field21_binned,
        "median_binned": median_binned,
        "fine_rows": fine_rows,
        "broad_rows": broad_rows,
        "counterfactuals": counterfactuals,
        "all_log_rho": log_rho_fields.reshape(-1),
    }


def write_csv(path, rows):
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def savefig(fig, outdir, stem):
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(diag, outdir):
    centers = diag["centers"]
    f21 = diag["field21_binned"]
    med = diag["median_binned"]
    x = np.linspace(-2.0, 2.2, 300)

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4))
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.09, top=0.95,
        wspace=0.28, hspace=0.34)

    ax = axes[0, 0]
    ax.hist(
        diag["all_log_rho"], bins=80, density=True,
        color="0.85", edgecolor="none", label="voxel PDF")
    ax2 = ax.twinx()
    ax2.plot(
        x, log_galaxy_bias(x, diag["field21_params"]),
        color="#87193d", lw=2.0, label="field 21")
    ax2.plot(
        x, log_galaxy_bias(x, diag["median_params"]),
        color="black", lw=1.5, ls="--", label="SWIFT median")
    ax2.plot(
        x, log_galaxy_bias(x, diag["fixed_params"]),
        color="#1e42b9", lw=1.3, ls=":", label="fixed bias")
    ax.axvline(diag["field21_params"][2], color="#87193d", lw=1.0, ls=":")
    ax.axvline(diag["median_params"][2], color="black", lw=1.0, ls=":")
    ax.set_xlabel(r"$\log \rho_{\rm DM}$")
    ax.set_ylabel("voxel PDF")
    ax2.set_ylabel(r"$\log n_g(\rho)$")
    ax.set_title("Sampled bias suppresses common densities")
    ax.set_xlim(-2.0, 2.2)
    ax2.legend(frameon=False, loc="lower right", fontsize=8)

    ax = axes[0, 1]
    for key, label, color in [
            ("rho", r"$\int \rho_{\rm DM}{\rm d}V$", "0.35"),
            ("ng", r"$\int n_g{\rm d}V$", "#87193d"),
            ("sel", r"$\int P_{\rm sel} n_g{\rm d}V$", "#d9981e")]:
        ratio = np.divide(
            f21[key], med[key],
            out=np.full_like(f21[key], np.nan),
            where=med[key] > 0)
        ax.plot(centers, ratio, color=color, lw=1.4, label=label)
    ax.axhline(1.0, color="0.3", lw=0.8, ls=":")
    ax.axvspan(-5.0, 0.0, color="0.9", zorder=-1)
    ax.set_xlabel(r"$\log \rho_{\rm DM}$")
    ax.set_ylabel("field 21 / SWIFT median")
    ax.set_title("Suppression is not in raw density")
    ax.set_xlim(-2.0, 2.2)
    ax.set_ylim(0.25, 1.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1, 0]
    ng_deficit = med["ng"] - f21["ng"]
    sel_deficit = med["sel"] - f21["sel"]
    ax.bar(
        centers, ng_deficit / np.sum(ng_deficit), width=np.diff(centers)[0],
        color="#87193d", alpha=0.62, label=r"$\int n_g{\rm d}V$")
    ax.step(
        centers, sel_deficit / np.sum(sel_deficit),
        where="mid", color="#d9981e", lw=1.5,
        label=r"$\int P_{\rm sel} n_g{\rm d}V$")
    ax.axvspan(-5.0, 0.0, color="0.9", zorder=-1)
    ax.set_xlabel(r"$\log \rho_{\rm DM}$")
    ax.set_ylabel("fraction of total deficit")
    ax.set_title("Deficit is from underdense and near-mean voxels")
    ax.set_xlim(-2.0, 2.2)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for broad in diag["broad_rows"]:
        left = broad["log_rho_left"]
        right = broad["log_rho_right"]
        label_left = r"$-\infty$" if not np.isfinite(left) else f"{left:.1f}"
        label_right = r"$\infty$" if not np.isfinite(right) else f"{right:.1f}"
        ax.bar(
            f"{label_left}\n{label_right}",
            broad["ng_deficit"],
            color="#87193d", alpha=0.72)
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_xlabel(r"$\log \rho_{\rm DM}$ bin")
    ax.set_ylabel(r"median $-$ field 21 contribution")
    ax.set_title(r"Broad-bin deficit in $\int n_g{\rm d}V$")

    savefig(fig, outdir, "ch0_single_field21_density_bias_diagnostics")


def write_summary(path, diag):
    f21 = diag["field21"]
    ranks = diag["field21_ranks"]
    field21_params = diag["field21_params"]
    median_params = diag["median_params"]
    broad = diag["broad_rows"]
    total_ng_deficit = sum(row["ng_deficit"] for row in broad)
    total_sel_deficit = sum(row["sel_deficit"] for row in broad)
    rho_lt1 = [
        row for row in broad
        if row["log_rho_right"] <= 0.0]
    rho_lt_transition = [
        row for row in broad
        if row["log_rho_right"] <= field21_params[2]]

    lines = [
        "# SWIFT Field 21 Density-Bias Diagnostic",
        "",
        "Field 21 is not extreme in the raw density integral.",
        f"Its sampled-bias `log int n_g dV` rank is "
        f"{ranks['log_expected_galaxy_integral_rank_low']:.0f}/30 low, "
        f"while `log int rho_DM dV` rank is "
        f"{ranks['log_density_integral_rank_low']:.0f}/30 low.",
        "",
        "## Bias Parameters",
        "",
        "| parameter | field 21 | SWIFT median |",
        "| --- | ---: | ---: |",
    ]
    for name, value21, median in zip(BIAS_PARAMS, field21_params, median_params):
        lines.append(f"| {name} | {value21:.3f} | {median:.3f} |")
    lines.extend([
        "",
        "The field-21 transition density is "
        f"`log_rho_t = {field21_params[2]:.3f}` "
        f"(`rho_t = {np.exp(field21_params[2]):.3f}`), "
        "whereas the SWIFT sampled median transition is "
        f"`log_rho_t = {median_params[2]:.3f}` "
        f"(`rho_t = {np.exp(median_params[2]):.3f}`).",
        "",
        "## Counterfactuals on the Field-21 Density Field",
        "",
        "| bias curve | log int n_g dV | log int P_sel n_g dV |",
        "| --- | ---: | ---: |",
    ])
    for item in diag["counterfactuals"]:
        lines.append(
            f"| {item['bias']} | {item['log_ng']:.3f} | "
            f"{item['log_selected_ng']:.3f} |")
    lines.extend([
        "",
        "## Deficit by Density Range",
        "",
        "| log rho range | rho range | ng deficit fraction | "
        "selected-ng deficit fraction | raw-density ratio | ng ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in broad:
        left = "-inf" if not np.isfinite(row["log_rho_left"]) else f"{row['log_rho_left']:.3f}"
        right = "inf" if not np.isfinite(row["log_rho_right"]) else f"{row['log_rho_right']:.3f}"
        rho_left = "0" if row["rho_left"] == 0 else f"{row['rho_left']:.2f}"
        rho_right = "inf" if not np.isfinite(row["rho_right"]) else f"{row['rho_right']:.2f}"
        lines.append(
            f"| {left} to {right} | {rho_left} to {rho_right} | "
            f"{row['ng_deficit'] / total_ng_deficit:.3f} | "
            f"{row['sel_deficit'] / total_sel_deficit:.3f} | "
            f"{row['rho_ratio']:.3f} | {row['ng_ratio']:.3f} |")

    lines.extend([
        "",
        "Summary: voxels with `rho < 1` account for "
        f"{sum(row['ng_deficit'] for row in rho_lt1) / total_ng_deficit:.1%} "
        "of the expected-galaxy deficit and "
        f"{sum(row['sel_deficit'] for row in rho_lt1) / total_sel_deficit:.1%} "
        "of the selected-galaxy deficit.",
        "Voxels below the field-21 bias transition "
        f"(`rho < {np.exp(field21_params[2]):.2f}`) account for "
        f"{sum(row['ng_deficit'] for row in rho_lt_transition) / total_ng_deficit:.1%} "
        "of the expected-galaxy deficit.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.rows_csv)
    diag = field21_diagnostics(rows)
    write_csv(
        outdir / "ch0_single_field21_density_bias_bins.csv",
        diag["fine_rows"])
    write_csv(
        outdir / "ch0_single_field21_density_bias_broad_bins.csv",
        diag["broad_rows"])
    write_summary(
        outdir / "ch0_single_field21_density_bias_summary.txt", diag)
    plot_diagnostics(diag, outdir)
    print(f"wrote field-21 density-bias diagnostics to {outdir}")


if __name__ == "__main__":
    main()
