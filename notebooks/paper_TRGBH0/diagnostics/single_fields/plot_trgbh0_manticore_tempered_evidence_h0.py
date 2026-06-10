#!/usr/bin/env python
"""Plot evidence-weighted H0 marginalisation for COLA Manticore fields."""
from argparse import ArgumentParser
import csv
import re
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
import scienceplots  # noqa: F401
from matplotlib.colors import Normalize  # noqa: E402

from trgbh0_plot_style import (  # noqa: E402
    FIGURE_DPI,
    ROOT,
    save_pdf_png,
    set_paper_rc,
    trgbh0_cmap,
)


RESULTS = ROOT / "results" / "TRGBH0_paper" / "manticore_fields_const_sigv"
DEFAULT_OUTDIR = RESULTS / "plots"
PATTERNS = {
    "gaussian": (
        "EDD_TRGB_sel-TRGB_magnitude_"
        "COLA_manticore_2MPP_MULTIBIN_N256_DES_V2_field*_"
        "manticore_field_const_sigv.hdf5"
    ),
    "student-t": (
        "EDD_TRGB_cz-student_t_sel-TRGB_magnitude_"
        "COLA_manticore_2MPP_MULTIBIN_N256_DES_V2_field*_"
        "manticore_field_const_sigv.hdf5"
    ),
}
LIKELIHOOD_LABELS = {
    "gaussian": "Gaussian redshift likelihood",
    "student-t": "Student-t redshift likelihood",
}
FIELD_RE = re.compile(r"_field(\d+)_")
def likelihood_suffix(likelihood):
    if likelihood == "gaussian":
        return ""
    return f"_{likelihood.replace('-', '_')}"


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--likelihood",
        choices=sorted(PATTERNS),
        default="gaussian",
        help="Redshift likelihood result set to plot.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="Directory containing the COLA Manticore HDF5 results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory for the PDF and PNG outputs.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="Directory for CSV summaries. Default: --output-dir.",
    )
    parser.add_argument(
        "--num-beta",
        type=int,
        default=201,
        help="Number of beta values between 0 and 1.",
    )
    parser.add_argument(
        "--num-bootstrap",
        type=int,
        default=20000,
        help="Number of Bayesian-bootstrap draws over field weights.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=53721,
        help="Random seed for the Bayesian bootstrap.",
    )
    parser.add_argument(
        "--field-subset-fraction",
        type=float,
        default=0.8,
        help=(
            "Fraction of fields to draw without replacement for the "
            "field-subset model-average robustness test."
        ),
    )
    parser.add_argument(
        "--target-neff",
        default="5,10,15",
        help=(
            "Comma-separated effective-field-count targets for "
            "regularised evidence tempering."
        ),
    )
    return parser.parse_args()


def parse_float_list(raw):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("At least one target N_eff value is required.")
    return tuple(values)


def field_index(path):
    match = FIELD_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse field index from `{path}`.")
    return int(match.group(1))


def finite_samples(handle, name, path):
    samples = np.asarray(handle[f"samples/{name}"], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"`{path}` has no finite {name} samples.")
    return samples


def load_field_rows(results_dir, likelihood):
    pattern = PATTERNS[likelihood]
    paths = sorted(results_dir.glob(pattern), key=field_index)
    if not paths:
        raise FileNotFoundError(
            f"No HDF5 files matching `{results_dir / pattern}`.")

    rows = []
    missing_nu = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            if likelihood == "student-t" and "samples/nu_cz" not in handle:
                missing_nu.append(path.name)
                continue
            h0 = finite_samples(handle, "H0", path)
            n_nu_cz = (
                int(finite_samples(handle, "nu_cz", path).size)
                if "samples/nu_cz" in handle else 0
            )
            lnz = float(handle["gof/lnZ_harmonic"][()])
            rows.append({
                "field": field_index(path),
                "source": str(path),
                "n_H0": int(h0.size),
                "n_nu_cz": n_nu_cz,
                "H0_samples": h0,
                "H0_mean": float(np.mean(h0)),
                "H0_second_moment": float(np.mean(h0 ** 2)),
                "lnZ_harmonic": lnz,
            })

    if missing_nu:
        raise ValueError(
            "Missing `samples/nu_cz` in: " + ", ".join(missing_nu))
    if not rows:
        raise ValueError("No usable COLA Manticore field rows.")
    return rows


def tempered_weights(lnz, beta):
    finite = np.isfinite(lnz)
    if beta == 0:
        weights = np.full(lnz.size, 1.0 / lnz.size)
        return weights
    if not np.any(finite):
        raise ValueError("No finite harmonic lnZ values for beta > 0.")

    weights = np.zeros_like(lnz, dtype=float)
    finite_lnz = lnz[finite]
    logw = beta * (finite_lnz - np.max(finite_lnz))
    weights[finite] = np.exp(logw)
    weights /= np.sum(weights)
    return weights


def row_arrays(rows):
    means = np.asarray([row["H0_mean"] for row in rows], dtype=float)
    second_moments = np.asarray(
        [row["H0_second_moment"] for row in rows], dtype=float)
    lnz = np.asarray([row["lnZ_harmonic"] for row in rows], dtype=float)
    return means, second_moments, lnz


def weighted_h0_summary(rows, weights):
    means, second_moments, lnz = row_arrays(rows)
    mean = float(np.sum(weights * means))
    second_moment = float(np.sum(weights * second_moments))
    std = float(np.sqrt(max(second_moment - mean ** 2, 0.0)))
    return {
        "H0_mean": mean,
        "H0_std": std,
        "n_eff": float(1.0 / np.sum(weights ** 2)),
        "max_weight": float(np.max(weights)),
        "dominant_field": int(rows[int(np.argmax(weights))]["field"]),
        "n_finite_lnz": int(np.sum(np.isfinite(lnz))),
        "n_fields": int(lnz.size),
    }


def mixture_summary(rows, betas):
    _, _, lnz = row_arrays(rows)

    summaries = []
    for beta in betas:
        weights = tempered_weights(lnz, float(beta))
        summaries.append({"beta": float(beta),
                          **weighted_h0_summary(rows, weights)})
    return summaries


def effective_sample_size(weights):
    return float(1.0 / np.sum(weights ** 2))


def solve_beta_for_neff(lnz, target_neff):
    finite = np.isfinite(lnz)
    n_finite = int(np.sum(finite))
    if n_finite == 0:
        raise ValueError("No finite harmonic lnZ values.")
    if not 1.0 <= target_neff <= n_finite:
        raise ValueError(
            f"Target N_eff={target_neff} is outside [1, {n_finite}].")

    neff_min = effective_sample_size(tempered_weights(lnz, 1.0))
    if target_neff < neff_min - 1e-8:
        raise ValueError(
            f"Target N_eff={target_neff} is below the beta=1 value "
            f"{neff_min:.6f}.")
    if np.isclose(target_neff, n_finite, rtol=0.0, atol=1e-8):
        return 0.0
    if np.isclose(target_neff, neff_min, rtol=0.0, atol=1e-8):
        return 1.0

    lo, hi = 0.0, 1.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        neff = effective_sample_size(tempered_weights(lnz, mid))
        if neff > target_neff:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def target_label(target_neff):
    if np.isclose(target_neff, round(target_neff), rtol=0.0, atol=1e-8):
        return f"K{int(round(target_neff))}"
    return f"K{target_neff:.2f}".replace(".", "p")


def target_neff_summaries(rows, targets):
    _, _, lnz = row_arrays(rows)
    summaries = []
    for target in targets:
        beta = solve_beta_for_neff(lnz, target)
        weights = tempered_weights(lnz, beta)
        summaries.append({
            "target_neff": float(target),
            "beta": float(beta),
            **weighted_h0_summary(rows, weights),
            "_weights": weights,
        })
    return summaries


def bayesian_bootstrap(rows, num_draws, seed):
    arrays = bayesian_bootstrap_arrays(
        rows, num_draws, np.random.default_rng(seed))
    return [
        {
            "draw": int(i),
            "H0_mean": float(arrays["H0_mean"][i]),
            "H0_std": float(arrays["H0_std"][i]),
            "n_eff": float(arrays["n_eff"][i]),
            "max_weight": float(arrays["max_weight"][i]),
            "dominant_field": int(arrays["dominant_field"][i]),
        }
        for i in range(num_draws)
    ]


def bayesian_bootstrap_arrays(rows, num_draws, rng):
    means, second_moments, lnz = row_arrays(rows)
    finite = np.isfinite(lnz)
    if not np.any(finite):
        raise ValueError("No finite harmonic lnZ values for bootstrap.")

    exp_lnz = np.zeros_like(lnz, dtype=float)
    exp_lnz[finite] = np.exp(lnz[finite] - np.max(lnz[finite]))
    field_prior = rng.exponential(scale=1.0, size=(num_draws, lnz.size))
    field_prior[:, ~finite] = 0.0
    weights = field_prior * exp_lnz[None, :]
    weights /= np.sum(weights, axis=1)[:, None]

    h0_mean = weights @ means
    h0_second_moment = weights @ second_moments
    h0_std = np.sqrt(np.maximum(h0_second_moment - h0_mean ** 2, 0.0))
    n_eff = 1.0 / np.sum(weights ** 2, axis=1)
    max_weight = np.max(weights, axis=1)
    dominant_idx = np.argmax(weights, axis=1)

    return {
        "H0_mean": h0_mean,
        "H0_std": h0_std,
        "n_eff": n_eff,
        "max_weight": max_weight,
        "dominant_field": np.asarray(
            [rows[int(i)]["field"] for i in dominant_idx], dtype=int),
    }


def progressive_top_drop(rows, num_draws, seed):
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            np.isfinite(row["lnZ_harmonic"]),
            row["lnZ_harmonic"],
        ),
        reverse=True,
    )
    rng = np.random.default_rng(seed)
    summaries = []

    for n_removed in range(len(sorted_rows)):
        remaining = sorted_rows[n_removed:]
        _, _, lnz = row_arrays(remaining)
        weights = tempered_weights(lnz, 1.0)
        full = weighted_h0_summary(remaining, weights)
        boot = bayesian_bootstrap_arrays(remaining, num_draws, rng)
        h0_q16, h0_q50, h0_q84 = h0_interval(boot["H0_mean"])
        neff_q16, neff_q50, neff_q84 = h0_interval(boot["n_eff"])

        best = remaining[int(np.argmax(lnz))]
        next_best_field = ""
        delta_lnz_best_next = np.nan
        if len(remaining) > 1:
            order = np.argsort(lnz)[::-1]
            next_best = remaining[int(order[1])]
            next_best_field = next_best["field"]
            delta_lnz_best_next = (
                best["lnZ_harmonic"] - next_best["lnZ_harmonic"])

        summaries.append({
            "n_removed": n_removed,
            "n_fields": len(remaining),
            "last_removed_field": (
                "" if n_removed == 0 else sorted_rows[n_removed - 1]["field"]
            ),
            "removed_fields": " ".join(
                str(row["field"]) for row in sorted_rows[:n_removed]),
            "best_remaining_field": best["field"],
            "next_best_remaining_field": next_best_field,
            "best_remaining_lnz_harmonic": best["lnZ_harmonic"],
            "delta_lnz_best_next": delta_lnz_best_next,
            **full,
            "bootstrap_H0_q16": float(h0_q16),
            "bootstrap_H0_q50": float(h0_q50),
            "bootstrap_H0_q84": float(h0_q84),
            "bootstrap_n_eff_q16": float(neff_q16),
            "bootstrap_n_eff_q50": float(neff_q50),
            "bootstrap_n_eff_q84": float(neff_q84),
        })
    return summaries


def field_subset_bma(rows, fraction, num_draws, seed):
    if not 0.0 < fraction <= 1.0:
        raise ValueError("`--field-subset-fraction` must be in (0, 1].")

    means, second_moments, lnz = row_arrays(rows)
    n_fields = len(rows)
    n_select = max(1, int(round(fraction * n_fields)))
    n_select = min(n_select, n_fields)
    rng = np.random.default_rng(seed)
    top_idx = int(np.nanargmax(lnz))
    top_field = rows[top_idx]["field"]

    summaries = []
    for draw in range(num_draws):
        selected = np.sort(rng.choice(n_fields, size=n_select, replace=False))
        selected_finite = selected[np.isfinite(lnz[selected])]
        if selected_finite.size == 0:
            raise ValueError("A field subset has no finite harmonic lnZ.")

        weights = np.zeros(n_fields, dtype=float)
        logw = lnz[selected_finite] - np.max(lnz[selected_finite])
        weights[selected_finite] = np.exp(logw)
        weights /= np.sum(weights)

        mean = float(np.sum(weights * means))
        second_moment = float(np.sum(weights * second_moments))
        std = float(np.sqrt(max(second_moment - mean ** 2, 0.0)))
        dominant_idx = int(np.argmax(weights))
        summaries.append({
            "draw": draw,
            "subset_fraction": fraction,
            "n_fields_selected": n_select,
            "H0_mean": mean,
            "H0_std": std,
            "n_eff": float(1.0 / np.sum(weights ** 2)),
            "max_weight": float(np.max(weights)),
            "dominant_field": rows[dominant_idx]["field"],
            "dominant_lnz_harmonic": rows[dominant_idx]["lnZ_harmonic"],
            "top_evidence_field": top_field,
            "top_evidence_field_included": bool(top_idx in selected),
            "selected_fields": " ".join(
                str(rows[int(i)]["field"]) for i in selected),
        })
    return summaries


def field_subset_stack_weights(rows, fraction, num_draws, seed):
    if not 0.0 < fraction <= 1.0:
        raise ValueError("`--field-subset-fraction` must be in (0, 1].")

    _, _, lnz = row_arrays(rows)
    n_fields = len(rows)
    n_select = max(1, int(round(fraction * n_fields)))
    n_select = min(n_select, n_fields)
    rng = np.random.default_rng(seed)
    weight_sum = np.zeros(n_fields, dtype=float)
    selection_count = np.zeros(n_fields, dtype=int)

    for _ in range(num_draws):
        selected = np.sort(rng.choice(n_fields, size=n_select, replace=False))
        selection_count[selected] += 1
        selected_finite = selected[np.isfinite(lnz[selected])]
        if selected_finite.size == 0:
            raise ValueError("A field subset has no finite harmonic lnZ.")

        weights = np.zeros(n_fields, dtype=float)
        logw = lnz[selected_finite] - np.max(lnz[selected_finite])
        weights[selected_finite] = np.exp(logw)
        weights /= np.sum(weights)
        weight_sum += weights

    stack_weights = weight_sum / num_draws
    return [
        {
            "field": row["field"],
            "lnZ_harmonic": row["lnZ_harmonic"],
            "H0_mean": row["H0_mean"],
            "stack_weight": float(stack_weights[i]),
            "selection_frequency": float(selection_count[i] / num_draws),
            "n_draws": num_draws,
            "subset_fraction": fraction,
            "n_fields_selected": n_select,
        }
        for i, row in enumerate(rows)
    ]


def leave_one_out(rows):
    _, _, lnz = row_arrays(rows)
    full_weights = tempered_weights(lnz, 1.0)
    full = weighted_h0_summary(rows, full_weights)

    summaries = []
    for i, row in enumerate(rows):
        keep = np.ones(lnz.size, dtype=bool)
        keep[i] = False
        finite_keep = keep & np.isfinite(lnz)
        if not np.any(finite_keep):
            raise ValueError("Leave-one-out removed all finite evidence rows.")

        weights = np.zeros_like(lnz, dtype=float)
        shifted = lnz[finite_keep] - np.max(lnz[finite_keep])
        weights[finite_keep] = np.exp(shifted)
        weights /= np.sum(weights)
        loo = weighted_h0_summary(rows, weights)
        summaries.append({
            "omitted_field": row["field"],
            "omitted_lnz_harmonic": row["lnZ_harmonic"],
            "omitted_full_weight": float(full_weights[i]),
            **loo,
            "delta_H0_mean": loo["H0_mean"] - full["H0_mean"],
            "delta_H0_std": loo["H0_std"] - full["H0_std"],
        })
    return full, summaries


def write_tempered_summary(path, summaries):
    fieldnames = [
        "beta", "H0_mean", "H0_std", "n_eff", "max_weight",
        "dominant_field", "n_finite_lnz", "n_fields",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_target_neff_summary(path, summaries):
    fieldnames = [
        "target_neff", "beta", "H0_mean", "H0_std", "H0_q16", "H0_q50",
        "H0_q84", "n_eff", "max_weight", "dominant_field",
        "n_finite_lnz", "n_fields",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                name: row[name]
                for name in fieldnames
            })


def write_target_neff_weights(path, summaries, rows):
    fieldnames = [
        "target_neff", "beta", "field", "lnZ_harmonic", "H0_mean",
        "weight",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            weights = summary["_weights"]
            for row, weight in zip(rows, weights):
                writer.writerow({
                    "target_neff": summary["target_neff"],
                    "beta": summary["beta"],
                    "field": row["field"],
                    "lnZ_harmonic": row["lnZ_harmonic"],
                    "H0_mean": row["H0_mean"],
                    "weight": float(weight),
                })


def write_bootstrap_summary(path, summaries):
    fieldnames = [
        "draw", "H0_mean", "H0_std", "n_eff", "max_weight",
        "dominant_field",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_leave_one_out_summary(path, summaries):
    fieldnames = [
        "omitted_field", "omitted_lnz_harmonic", "omitted_full_weight",
        "H0_mean", "H0_std", "n_eff", "max_weight", "dominant_field",
        "n_finite_lnz", "n_fields", "delta_H0_mean", "delta_H0_std",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_progressive_top_drop_summary(path, summaries):
    fieldnames = [
        "n_removed", "n_fields", "last_removed_field", "removed_fields",
        "best_remaining_field", "next_best_remaining_field",
        "best_remaining_lnz_harmonic", "delta_lnz_best_next",
        "H0_mean", "H0_std", "n_eff", "max_weight", "dominant_field",
        "n_finite_lnz", "bootstrap_H0_q16", "bootstrap_H0_q50",
        "bootstrap_H0_q84", "bootstrap_n_eff_q16",
        "bootstrap_n_eff_q50", "bootstrap_n_eff_q84",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_field_subset_summary(path, summaries):
    fieldnames = [
        "draw", "subset_fraction", "n_fields_selected", "H0_mean", "H0_std",
        "n_eff", "max_weight", "dominant_field", "dominant_lnz_harmonic",
        "top_evidence_field", "top_evidence_field_included",
        "selected_fields",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_field_subset_stack_weights(path, summaries):
    fieldnames = [
        "field", "lnZ_harmonic", "H0_mean", "stack_weight",
        "selection_frequency", "n_draws", "subset_fraction",
        "n_fields_selected",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_density_summary(path, x_grid, densities):
    fieldnames = ["H0", *densities]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, h0 in enumerate(x_grid):
            writer.writerow({
                "H0": float(h0),
                **{name: float(density[i])
                   for name, density in densities.items()},
            })


def save_tempered_plot(fig, out_pdf):
    return save_pdf_png(fig, out_pdf)[1]


def plot_tempered_h0(summaries, rows, likelihood, out_pdf):
    beta = np.asarray([row["beta"] for row in summaries])
    mean = np.asarray([row["H0_mean"] for row in summaries])
    std = np.asarray([row["H0_std"] for row in summaries])
    n_eff = np.asarray([row["n_eff"] for row in summaries])

    lnz = np.asarray([row["lnZ_harmonic"] for row in rows], dtype=float)
    bad_fields = [
        row["field"] for row in rows
        if not np.isfinite(row["lnZ_harmonic"])
    ]
    finite_lnz = lnz[np.isfinite(lnz)]
    delta_lnz = float(np.max(finite_lnz) - np.median(finite_lnz))

    cmap = trgbh0_cmap("trgbh0_tempered_evidence_h0")
    line_colour = cmap(0.22)
    band_colour = cmap(0.55)
    neff_colour = cmap(0.82)

    with plt.style.context("science"):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(4.7, 3.25))
        ax.fill_between(
            beta,
            mean - std,
            mean + std,
            color=band_colour,
            alpha=0.24,
            lw=0,
            label=r"$\mu_{H_0} \pm \sigma_{H_0}$",
        )
        ax.plot(
            beta,
            mean,
            color=line_colour,
            lw=1.45,
            label=r"Tempered-evidence mean",
        )
        ax.set_xlabel(r"Evidence tempering exponent $\beta$")
        ax.set_ylabel(
            r"$H_0 ~ [\mathrm{km}\,\mathrm{s}^{-1}\,"
            r"\mathrm{Mpc}^{-1}]$"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(
            np.min(mean - std) - 0.06 * np.ptp(mean + std),
            np.max(mean + std) + 0.06 * np.ptp(mean + std),
        )

        ax_neff = ax.twinx()
        ax_neff.plot(
            beta,
            n_eff,
            color=neff_colour,
            lw=0.9,
            ls=":",
            label=r"$N_{\rm eff}$",
        )
        ax_neff.set_ylabel(r"Effective number of fields")
        ax_neff.set_ylim(0, max(50.0, np.max(n_eff)) * 1.05)
        ax_neff.tick_params(axis="y", labelsize=7.0)

        text = (
            LIKELIHOOD_LABELS[likelihood] + "\n"
            rf"$N_{{\rm fields}}={len(rows)}$" "\n"
            rf"$\Delta \ln \mathcal{{Z}}_{{\rm max-med}}={delta_lnz:.1f}$" "\n"
            rf"$N_{{\rm eff}}(\beta=1)={n_eff[-1]:.1f}$"
        )
        if bad_fields:
            text += "\n" + "non-finite " + r"$\ln \mathcal{Z}$" + (
                ": " + ", ".join(str(field) for field in bad_fields)
            )
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.7,
        )

        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax_neff.get_legend_handles_labels()
        ax.legend(
            handles + handles2,
            labels + labels2,
            loc="lower left",
            frameon=False,
            handlelength=1.8,
        )

        fig.tight_layout()
    return save_tempered_plot(fig, out_pdf)


def h0_interval(values):
    return np.percentile(values, [16.0, 50.0, 84.0])


def fraction_suffix(fraction):
    percent = 100.0 * fraction
    if np.isclose(percent, round(percent), rtol=0.0, atol=1e-8):
        return f"{int(round(percent)):02d}pct"
    return f"{percent:.1f}pct".replace(".", "p")


def h0_density_grid(rows, n_bins=240):
    samples = np.concatenate([row["H0_samples"] for row in rows])
    lo, hi = np.percentile(samples, [0.1, 99.9])
    pad = 0.08 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, n_bins + 1)


def smooth_density(density, sigma_bins=1.15):
    radius = max(1, int(np.ceil(4.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(density, kernel, mode="same")


def mixture_h0_density(rows, field_weights, bin_edges):
    bin_width = bin_edges[1] - bin_edges[0]
    density = np.zeros(bin_edges.size - 1, dtype=float)
    for row, weight in zip(rows, field_weights):
        counts, _ = np.histogram(row["H0_samples"], bins=bin_edges)
        density += weight * counts / row["H0_samples"].size / bin_width
    density = smooth_density(density)
    density /= np.sum(density) * bin_width
    return 0.5 * (bin_edges[:-1] + bin_edges[1:]), density


def density_interval(x_grid, density):
    dx = x_grid[1] - x_grid[0]
    cdf = np.cumsum(density) * dx
    cdf /= cdf[-1]
    return np.interp([0.16, 0.5, 0.84], cdf, x_grid)


def plot_resampling_h0(bootstrap_rows, loo_rows, full, rows, likelihood, out_pdf):
    boot_h0 = np.asarray([row["H0_mean"] for row in bootstrap_rows],
                         dtype=float)
    q16, q50, q84 = h0_interval(boot_h0)
    bootstrap_collapsed = np.ptp(boot_h0) < 1e-4
    if bootstrap_collapsed:
        boot_interval_text = r"$68\%_{\rm boot}$ width $<10^{-4}$"
    else:
        boot_interval_text = (
            rf"$68\%_{{\rm boot}}=[{q16:.3f}, {q84:.3f}]$"
        )
    omitted_fields = np.asarray([row["omitted_field"] for row in loo_rows])
    delta_h0 = np.asarray([row["delta_H0_mean"] for row in loo_rows],
                          dtype=float)
    omitted_lnz = np.asarray(
        [row["omitted_lnz_harmonic"] for row in loo_rows], dtype=float)
    omitted_weight = np.asarray(
        [row["omitted_full_weight"] for row in loo_rows], dtype=float)
    most_influential = loo_rows[int(np.argmax(np.abs(delta_h0)))]

    cmap = trgbh0_cmap("trgbh0_manticore_resampling_h0")
    hist_colour = cmap(0.25)
    interval_colour = cmap(0.55)
    full_colour = cmap(0.85)
    norm = Normalize(vmin=float(np.min(omitted_lnz)),
                     vmax=float(np.max(omitted_lnz)))
    marker_size = 18.0 + 110.0 * omitted_weight / np.max(omitted_weight)

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.05))

        ax = axes[0]
        if bootstrap_collapsed:
            ax.axvspan(q50 - 5e-4, q50 + 5e-4,
                       color=interval_colour, alpha=0.22, lw=0)
            ax.text(
                0.5,
                0.55,
                "bootstrap collapsed\nonto dominant field",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6.8,
            )
            ax.set_xlim(q50 - 0.08, q50 + 0.08)
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([])
        else:
            ax.hist(
                boot_h0,
                bins=42,
                density=True,
                color=hist_colour,
                alpha=0.42,
                histtype="stepfilled",
                lw=0,
            )
            ax.axvspan(q16, q84, color=interval_colour, alpha=0.18, lw=0)
            x_pad = 0.08 * np.ptp(boot_h0)
            ax.set_xlim(np.min(boot_h0) - x_pad, np.max(boot_h0) + x_pad)
        ax.axvline(q50, color=interval_colour, lw=1.1, ls="--",
                   label=r"Bootstrap median")
        ax.axvline(full["H0_mean"], color=full_colour, lw=1.2,
                   label=r"Full evidence weighting")
        ax.set_xlabel(
            r"$\langle H_0\rangle ~ [\mathrm{km}\,\mathrm{s}^{-1}\,"
            r"\mathrm{Mpc}^{-1}]$"
        )
        ax.set_ylabel(r"Density" if not bootstrap_collapsed else "")
        ax.legend(loc="lower right", frameon=False, handlelength=1.6)
        ax.text(
            0.03,
            0.05,
            LIKELIHOOD_LABELS[likelihood] + "\n"
            rf"$N_{{\rm fields}}={len(rows)}$" "\n"
            rf"$H_0={full['H0_mean']:.3f}\pm{full['H0_std']:.3f}$" "\n"
            f"{boot_interval_text}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.6,
        )

        ax = axes[1]
        sc = ax.scatter(
            omitted_fields,
            delta_h0,
            c=omitted_lnz,
            cmap=cmap,
            norm=norm,
            s=marker_size,
            edgecolor="k",
            linewidth=0.25,
            alpha=0.9,
        )
        ax.axhline(0.0, color="0.35", lw=0.8, ls=":")
        ax.set_xlabel(r"Omitted field")
        ax.set_ylabel(
            r"$\Delta \langle H_0\rangle_{\rm LOO}$"
            r" $[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"
        )
        ax.text(
            0.03,
            0.05,
            rf"largest shift: field {most_influential['omitted_field']}" "\n"
            rf"$\Delta H_0={most_influential['delta_H0_mean']:.3f}$" "\n"
            rf"weight={most_influential['omitted_full_weight']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.6,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.048, pad=0.02)
        cbar.set_label(r"Omitted-field $\ln \mathcal{Z}$")
        cbar.ax.tick_params(labelsize=6.5)

        fig.tight_layout()
    return save_tempered_plot(fig, out_pdf)


def plot_progressive_top_drop(summaries, likelihood, out_pdf):
    n_removed = np.asarray([row["n_removed"] for row in summaries], dtype=int)
    n_fields = np.asarray([row["n_fields"] for row in summaries], dtype=int)
    h0_mean = np.asarray([row["H0_mean"] for row in summaries], dtype=float)
    h0_q16 = np.asarray(
        [row["bootstrap_H0_q16"] for row in summaries], dtype=float)
    h0_q84 = np.asarray(
        [row["bootstrap_H0_q84"] for row in summaries], dtype=float)
    n_eff = np.asarray([row["n_eff"] for row in summaries], dtype=float)
    neff_q16 = np.asarray(
        [row["bootstrap_n_eff_q16"] for row in summaries], dtype=float)
    neff_q84 = np.asarray(
        [row["bootstrap_n_eff_q84"] for row in summaries], dtype=float)

    cmap = trgbh0_cmap("trgbh0_manticore_progressive_top_drop_h0")
    line_colour = cmap(0.25)
    band_colour = cmap(0.55)
    neff_colour = cmap(0.82)
    reference_colour = "0.45"
    max_neff = summaries[int(np.argmax(n_eff))]

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.15), sharex=True)

        ax = axes[0]
        ax.fill_between(
            n_removed,
            h0_q16,
            h0_q84,
            color=band_colour,
            alpha=0.22,
            lw=0,
            label=r"Bayesian-bootstrap $68\%$",
        )
        ax.plot(
            n_removed,
            h0_mean,
            color=line_colour,
            lw=1.25,
            marker="o",
            ms=2.7,
            label=r"Evidence-weighted mean",
        )
        ax.set_ylabel(
            r"$\langle H_0\rangle ~ [\mathrm{km}\,\mathrm{s}^{-1}\,"
            r"\mathrm{Mpc}^{-1}]$"
        )
        ax.legend(loc="best", frameon=False, handlelength=1.8)
        ax.text(
            0.03,
            0.05,
            LIKELIHOOD_LABELS[likelihood] + "\n"
            rf"$N_{{\rm init}}={summaries[0]['n_fields']}$" "\n"
            rf"first removed: field {summaries[1]['last_removed_field']}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.6,
        )

        ax = axes[1]
        ax.fill_between(
            n_removed,
            neff_q16,
            neff_q84,
            color=band_colour,
            alpha=0.2,
            lw=0,
            label=r"Bootstrap $68\%$",
        )
        ax.plot(
            n_removed,
            n_eff,
            color=neff_colour,
            lw=1.25,
            marker="o",
            ms=2.7,
            label=r"Evidence $N_{\rm eff}$",
        )
        ax.plot(
            n_removed,
            n_fields,
            color=reference_colour,
            lw=0.85,
            ls=":",
            label=r"Remaining fields",
        )
        ax.set_ylabel(r"Effective number of fields")
        ax.legend(loc="best", frameon=False, handlelength=1.8)
        ax.text(
            0.03,
            0.05,
            rf"max $N_{{\rm eff}}={max_neff['n_eff']:.2f}$" "\n"
            rf"after removing {max_neff['n_removed']} fields" "\n"
            rf"best remaining: {max_neff['best_remaining_field']}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.6,
        )

        for ax in axes:
            ax.set_xlabel(r"Number of highest-evidence fields removed")
            ax.set_xlim(-0.5, np.max(n_removed) + 0.5)

        fig.tight_layout()
    return save_tempered_plot(fig, out_pdf)


def plot_field_subset_bma(summaries, likelihood, out_pdf):
    h0 = np.asarray([row["H0_mean"] for row in summaries], dtype=float)
    n_eff = np.asarray([row["n_eff"] for row in summaries], dtype=float)
    dominant = np.asarray([row["dominant_field"] for row in summaries],
                          dtype=int)
    top_included = np.asarray(
        [row["top_evidence_field_included"] for row in summaries],
        dtype=bool,
    )
    fraction = float(summaries[0]["subset_fraction"])
    n_selected = int(summaries[0]["n_fields_selected"])
    top_field = int(summaries[0]["top_evidence_field"])
    top_included_fraction = float(np.mean(top_included))
    h0_q16, h0_q50, h0_q84 = h0_interval(h0)
    neff_q16, neff_q50, neff_q84 = h0_interval(n_eff)

    h0_pad = max(0.1, 0.06 * np.ptp(h0))
    h0_bins = np.linspace(np.min(h0) - h0_pad, np.max(h0) + h0_pad, 42)
    neff_pad = max(0.05, 0.06 * np.ptp(n_eff))
    neff_bins = np.linspace(
        np.min(n_eff) - neff_pad, np.max(n_eff) + neff_pad, 36)

    fields, counts = np.unique(dominant, return_counts=True)
    order = np.argsort(counts)[::-1]
    fields = fields[order][:10]
    counts = counts[order][:10]
    frequencies = counts / len(summaries)

    cmap = trgbh0_cmap("trgbh0_manticore_field_subset_bma_h0")
    included_colour = cmap(0.25)
    excluded_colour = cmap(0.62)
    median_colour = cmap(0.86)

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.85))

        ax = axes[0]
        ax.hist(
            h0[top_included],
            bins=h0_bins,
            density=True,
            color=included_colour,
            alpha=0.38,
            label=rf"field {top_field} included",
        )
        if np.any(~top_included):
            ax.hist(
                h0[~top_included],
                bins=h0_bins,
                density=True,
                color=excluded_colour,
                alpha=0.42,
                label=rf"field {top_field} excluded",
            )
        ax.axvline(h0_q50, color=median_colour, lw=1.1, ls="--")
        ax.set_xlabel(
            r"$\langle H_0\rangle ~ [\mathrm{km}\,\mathrm{s}^{-1}\,"
            r"\mathrm{Mpc}^{-1}]$"
        )
        ax.set_ylabel(r"Density")
        ax.legend(loc="upper left", frameon=False, handlelength=1.5)
        ax.text(
            0.03,
            0.05,
            LIKELIHOOD_LABELS[likelihood] + "\n"
            rf"$f_{{\rm subset}}={fraction:.2f}$; "
            rf"$N_{{\rm selected}}={n_selected}$" "\n"
            rf"$H_0=[{h0_q16:.2f}, {h0_q84:.2f}]_{{68\%}}$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.4,
        )

        ax = axes[1]
        ax.hist(
            n_eff[top_included],
            bins=neff_bins,
            density=True,
            color=included_colour,
            alpha=0.38,
        )
        if np.any(~top_included):
            ax.hist(
                n_eff[~top_included],
                bins=neff_bins,
                density=True,
                color=excluded_colour,
                alpha=0.42,
            )
        ax.axvline(neff_q50, color=median_colour, lw=1.1, ls="--")
        ax.set_xlabel(r"Effective number of fields")
        ax.set_ylabel(r"Density")
        ax.text(
            0.03,
            0.95,
            rf"$N_{{\rm eff}}=[{neff_q16:.2f}, {neff_q84:.2f}]_{{68\%}}$"
            "\n"
            rf"$P({top_field}\in S)={top_included_fraction:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.4,
        )

        ax = axes[2]
        colours = [included_colour if field == top_field else excluded_colour
                   for field in fields]
        ax.bar(np.arange(fields.size), frequencies, color=colours, alpha=0.72)
        ax.set_xticks(np.arange(fields.size))
        ax.set_xticklabels([str(field) for field in fields])
        ax.set_xlabel(r"Dominant field")
        ax.set_ylabel(r"Draw fraction")
        ax.set_ylim(0.0, min(1.0, np.max(frequencies) * 1.22))
        ax.text(
            0.97,
            0.95,
            rf"$N_{{\rm draws}}={len(summaries)}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.4,
        )

        fig.tight_layout()
    return save_tempered_plot(fig, out_pdf)


def plot_field_subset_stacked_h0(stack_rows, rows, likelihood, out_pdf,
                                 density_csv):
    stack_weights = np.asarray(
        [row["stack_weight"] for row in stack_rows], dtype=float)
    stack_weights /= np.sum(stack_weights)
    _, _, lnz = row_arrays(rows)
    full_weights = tempered_weights(lnz, 1.0)
    equal_weights = np.full(len(rows), 1.0 / len(rows))

    bin_edges = h0_density_grid(rows)
    x_grid, stack_density = mixture_h0_density(
        rows, stack_weights, bin_edges)
    _, full_density = mixture_h0_density(rows, full_weights, bin_edges)
    _, equal_density = mixture_h0_density(rows, equal_weights, bin_edges)
    write_density_summary(density_csv, x_grid, {
        "field_subset_stack": stack_density,
        "full_evidence_bma": full_density,
        "equal_field_mixture": equal_density,
    })

    stack_q16, stack_q50, stack_q84 = density_interval(
        x_grid, stack_density)
    full_q16, full_q50, full_q84 = density_interval(x_grid, full_density)
    top = sorted(stack_rows, key=lambda row: row["stack_weight"],
                 reverse=True)[:10]

    fraction = float(stack_rows[0]["subset_fraction"])
    n_selected = int(stack_rows[0]["n_fields_selected"])
    n_draws = int(stack_rows[0]["n_draws"])
    cmap = trgbh0_cmap("trgbh0_manticore_field_subset_stacked_h0")
    stack_colour = cmap(0.25)
    full_colour = cmap(0.62)
    equal_colour = "0.5"

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.05))

        ax = axes[0]
        ax.plot(
            x_grid,
            stack_density,
            color=stack_colour,
            lw=1.35,
            label=rf"{100.0 * fraction:.0f}\% subset stack",
        )
        ax.fill_between(
            x_grid,
            0.0,
            stack_density,
            color=stack_colour,
            alpha=0.16,
            lw=0,
        )
        ax.plot(
            x_grid,
            full_density,
            color=full_colour,
            lw=1.0,
            ls="--",
            label=r"Full evidence BMA",
        )
        ax.plot(
            x_grid,
            equal_density,
            color=equal_colour,
            lw=0.85,
            ls=":",
            label=r"Equal-field mixture",
        )
        ax.axvline(stack_q50, color=stack_colour, lw=0.95, ls=":")
        ax.axvline(full_q50, color=full_colour, lw=0.95, ls=":")
        ax.set_xlabel(
            r"$H_0 ~ [\mathrm{km}\,\mathrm{s}^{-1}\,"
            r"\mathrm{Mpc}^{-1}]$"
        )
        ax.set_ylabel(r"Posterior density")
        ax.legend(loc="upper right", frameon=False, handlelength=1.8)
        ax.text(
            0.03,
            0.05,
            LIKELIHOOD_LABELS[likelihood] + "\n"
            rf"$N_{{\rm selected}}={n_selected}$; "
            rf"$N_{{\rm draws}}={n_draws}$" "\n"
            rf"stack $H_0={stack_q50:.2f}"
            rf"_{{-{stack_q50 - stack_q16:.2f}}}"
            rf"^{{+{stack_q84 - stack_q50:.2f}}}$" "\n"
            rf"full BMA $H_0={full_q50:.2f}"
            rf"_{{-{full_q50 - full_q16:.2f}}}"
            rf"^{{+{full_q84 - full_q50:.2f}}}$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.3,
        )

        ax = axes[1]
        fields = [row["field"] for row in top]
        weights = [row["stack_weight"] for row in top]
        ax.bar(np.arange(len(top)), weights, color=stack_colour, alpha=0.72)
        ax.set_xticks(np.arange(len(top)))
        ax.set_xticklabels([str(field) for field in fields])
        ax.set_xlabel(r"Field")
        ax.set_ylabel(r"Mean stacked field weight")
        ax.set_ylim(0.0, min(1.0, max(weights) * 1.22))
        ax.text(
            0.97,
            0.95,
            rf"$\sum_i \bar w_i^2$ gives "
            rf"$N_{{\rm eff}}={1.0 / np.sum(stack_weights ** 2):.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.3,
        )

        fig.tight_layout()
    return save_tempered_plot(fig, out_pdf)


def plot_target_neff_h0(summaries, rows, likelihood, out_pdf, density_csv):
    _, _, lnz = row_arrays(rows)
    full_weights = tempered_weights(lnz, 1.0)
    equal_weights = np.full(len(rows), 1.0 / len(rows))
    bin_edges = h0_density_grid(rows)

    x_grid, full_density = mixture_h0_density(rows, full_weights, bin_edges)
    _, equal_density = mixture_h0_density(rows, equal_weights, bin_edges)
    densities = {
        "full_evidence_bma": full_density,
        "equal_field_mixture": equal_density,
    }
    for summary in summaries:
        label = target_label(summary["target_neff"])
        _, density = mixture_h0_density(rows, summary["_weights"], bin_edges)
        densities[label] = density
        q16, q50, q84 = density_interval(x_grid, density)
        summary["H0_q16"] = float(q16)
        summary["H0_q50"] = float(q50)
        summary["H0_q84"] = float(q84)
    write_density_summary(density_csv, x_grid, densities)

    target = np.asarray([row["target_neff"] for row in summaries],
                        dtype=float)
    h0_q16 = np.asarray([row["H0_q16"] for row in summaries], dtype=float)
    h0_q50 = np.asarray([row["H0_q50"] for row in summaries], dtype=float)
    h0_q84 = np.asarray([row["H0_q84"] for row in summaries], dtype=float)
    beta = np.asarray([row["beta"] for row in summaries], dtype=float)

    cmap = trgbh0_cmap("trgbh0_manticore_target_neff_h0")
    colours = [cmap(x) for x in np.linspace(0.18, 0.82, len(summaries))]
    full_colour = cmap(0.95)
    equal_colour = "0.5"

    with plt.style.context("science"):
        set_paper_rc()
        fig, axes = plt.subplots(1, 2, figsize=(6.55, 3.05))

        ax = axes[0]
        for colour, summary in zip(colours, summaries):
            label = target_label(summary["target_neff"])
            ax.plot(
                x_grid,
                densities[label],
                color=colour,
                lw=1.25,
                label=rf"$N_{{\rm eff}}={summary['target_neff']:g}$",
            )
        ax.plot(
            x_grid,
            full_density,
            color=full_colour,
            lw=1.0,
            ls="--",
            label=r"$\eta=1$",
        )
        ax.plot(
            x_grid,
            equal_density,
            color=equal_colour,
            lw=0.85,
            ls=":",
            label=r"$\eta=0$",
        )
        ax.set_xlabel(
            r"$H_0 ~ [\mathrm{km}\,\mathrm{s}^{-1}\,"
            r"\mathrm{Mpc}^{-1}]$"
        )
        ax.set_ylabel(r"Posterior density")
        ax.legend(loc="upper right", frameon=False, handlelength=1.6)
        ax.text(
            0.03,
            0.05,
            LIKELIHOOD_LABELS[likelihood] + "\n"
            r"$w_i(\eta)\propto\exp(\eta\ln Z_i)$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.4,
        )

        ax = axes[1]
        yerr = np.vstack([h0_q50 - h0_q16, h0_q84 - h0_q50])
        ax.errorbar(
            target,
            h0_q50,
            yerr=yerr,
            fmt="o-",
            color=colours[-1],
            ms=3.0,
            lw=1.0,
            capsize=2.0,
            label=r"$H_0$ posterior",
        )
        ax.set_xlabel(r"Target effective number of fields")
        ax.set_ylabel(
            r"$H_0 ~ [\mathrm{km}\,\mathrm{s}^{-1}\,"
            r"\mathrm{Mpc}^{-1}]$"
        )
        ax_beta = ax.twinx()
        ax_beta.plot(
            target,
            beta,
            color=colours[0],
            lw=0.95,
            ls=":",
            marker="s",
            ms=2.6,
            label=r"$\eta$",
        )
        ax_beta.set_ylabel(r"Evidence tempering exponent $\eta$")
        ax_beta.tick_params(axis="y", labelsize=7.0)
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax_beta.get_legend_handles_labels()
        ax.legend(
            handles + handles2,
            labels + labels2,
            loc="best",
            frameon=False,
            handlelength=1.8,
        )

        fig.tight_layout()
    return save_tempered_plot(fig, out_pdf)


def run_analysis(
    likelihood="gaussian",
    results_dir=RESULTS,
    output_dir=DEFAULT_OUTDIR,
    summary_dir=None,
    num_beta=201,
    num_bootstrap=20000,
    seed=53721,
    field_subset_fraction=0.8,
    target_neff="5,10,15",
):
    if num_beta < 2:
        raise ValueError("`--num-beta` must be at least 2.")
    if num_bootstrap < 1:
        raise ValueError("`--num-bootstrap` must be positive.")
    if isinstance(target_neff, str):
        target_neff_values = parse_float_list(target_neff)
    else:
        target_neff_values = tuple(float(value) for value in target_neff)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir if summary_dir is None else summary_dir
    summary_dir.mkdir(parents=True, exist_ok=True)
    suffix = likelihood_suffix(likelihood)

    rows = load_field_rows(results_dir, likelihood)
    betas = np.linspace(0.0, 1.0, num_beta)
    summaries = mixture_summary(rows, betas)

    out_pdf = output_dir / f"trgbh0_manticore_tempered_evidence_h0{suffix}.pdf"
    out_png = plot_tempered_h0(summaries, rows, likelihood, out_pdf)
    summary_csv = (
        summary_dir / f"manticore_tempered_evidence_h0_summary{suffix}.csv"
    )
    write_tempered_summary(summary_csv, summaries)

    target_rows = target_neff_summaries(rows, target_neff_values)
    target_pdf = output_dir / f"trgbh0_manticore_target_neff_h0{suffix}.pdf"
    target_density_csv = (
        summary_dir / f"manticore_target_neff_h0_density{suffix}.csv"
    )
    target_png = plot_target_neff_h0(
        target_rows, rows, likelihood, target_pdf, target_density_csv)
    target_summary_csv = (
        summary_dir / f"manticore_target_neff_h0_summary{suffix}.csv"
    )
    target_weights_csv = (
        summary_dir / f"manticore_target_neff_h0_weights{suffix}.csv"
    )
    write_target_neff_summary(target_summary_csv, target_rows)
    write_target_neff_weights(target_weights_csv, target_rows, rows)

    bootstrap_rows = bayesian_bootstrap(rows, num_bootstrap, seed)
    full_summary, loo_rows = leave_one_out(rows)
    resampling_pdf = (
        output_dir / f"trgbh0_manticore_evidence_resampling_h0{suffix}.pdf"
    )
    resampling_png = plot_resampling_h0(
        bootstrap_rows, loo_rows, full_summary, rows, likelihood,
        resampling_pdf,
    )
    bootstrap_csv = (
        summary_dir / f"manticore_evidence_bootstrap_h0_draws{suffix}.csv"
    )
    loo_csv = summary_dir / f"manticore_evidence_leave_one_out_h0{suffix}.csv"
    write_bootstrap_summary(bootstrap_csv, bootstrap_rows)
    write_leave_one_out_summary(loo_csv, loo_rows)

    progressive_rows = progressive_top_drop(
        rows, num_bootstrap, seed + 7919)
    progressive_pdf = (
        output_dir / f"trgbh0_manticore_progressive_top_drop_h0{suffix}.pdf"
    )
    progressive_png = plot_progressive_top_drop(
        progressive_rows, likelihood, progressive_pdf)
    progressive_csv = (
        summary_dir
        / f"manticore_progressive_top_drop_h0_summary{suffix}.csv"
    )
    write_progressive_top_drop_summary(progressive_csv, progressive_rows)

    subset_suffix = fraction_suffix(field_subset_fraction)
    subset_file_suffix = (
        f"{suffix}_{subset_suffix}" if suffix else f"_{subset_suffix}"
    )
    field_subset_rows = field_subset_bma(
        rows, field_subset_fraction, num_bootstrap,
        seed + 15485863,
    )
    field_subset_pdf = (
        output_dir
        / f"trgbh0_manticore_field_subset_bma_h0{subset_file_suffix}.pdf"
    )
    field_subset_png = plot_field_subset_bma(
        field_subset_rows, likelihood, field_subset_pdf)
    field_subset_csv = (
        summary_dir
        / f"manticore_field_subset_bma_h0_draws{subset_file_suffix}.csv"
    )
    write_field_subset_summary(field_subset_csv, field_subset_rows)

    stack_rows = field_subset_stack_weights(
        rows, field_subset_fraction, num_bootstrap,
        seed + 32452843,
    )
    stack_pdf = (
        output_dir
        / f"trgbh0_manticore_field_subset_stacked_h0{subset_file_suffix}.pdf"
    )
    stack_density_csv = (
        summary_dir
        / f"manticore_field_subset_stacked_h0_density{subset_file_suffix}.csv"
    )
    stack_png = plot_field_subset_stacked_h0(
        stack_rows, rows, likelihood, stack_pdf, stack_density_csv)
    stack_weights_csv = (
        summary_dir
        / f"manticore_field_subset_stacked_h0_weights{subset_file_suffix}.csv"
    )
    write_field_subset_stack_weights(stack_weights_csv, stack_rows)

    boot_h0 = np.asarray([row["H0_mean"] for row in bootstrap_rows],
                         dtype=float)
    boot_q16, boot_q50, boot_q84 = h0_interval(boot_h0)
    loo_delta = np.asarray([row["delta_H0_mean"] for row in loo_rows],
                           dtype=float)
    most_influential = loo_rows[int(np.argmax(np.abs(loo_delta)))]
    max_progressive_neff = max(
        progressive_rows, key=lambda row: row["n_eff"])
    subset_h0 = np.asarray([row["H0_mean"] for row in field_subset_rows],
                           dtype=float)
    subset_neff = np.asarray([row["n_eff"] for row in field_subset_rows],
                             dtype=float)
    subset_h0_q16, subset_h0_q50, subset_h0_q84 = h0_interval(subset_h0)
    _, subset_neff_q50, _ = h0_interval(subset_neff)
    subset_top_included = np.asarray([
        row["top_evidence_field_included"] for row in field_subset_rows],
        dtype=bool,
    )
    subset_fields, subset_counts = np.unique(
        [row["dominant_field"] for row in field_subset_rows],
        return_counts=True,
    )
    subset_dominant_field = subset_fields[int(np.argmax(subset_counts))]
    subset_dominant_fraction = float(
        np.max(subset_counts) / len(field_subset_rows))
    stack_weights = np.asarray([row["stack_weight"] for row in stack_rows],
                               dtype=float)
    stack_neff = 1.0 / np.sum(stack_weights ** 2)
    stack_top = max(stack_rows, key=lambda row: row["stack_weight"])

    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {target_pdf}")
    print(f"Wrote {target_png}")
    print(f"Wrote {target_density_csv}")
    print(f"Wrote {target_summary_csv}")
    print(f"Wrote {target_weights_csv}")
    print(f"Wrote {resampling_pdf}")
    print(f"Wrote {resampling_png}")
    print(f"Wrote {bootstrap_csv}")
    print(f"Wrote {loo_csv}")
    print(f"Wrote {progressive_pdf}")
    print(f"Wrote {progressive_png}")
    print(f"Wrote {progressive_csv}")
    print(f"Wrote {field_subset_pdf}")
    print(f"Wrote {field_subset_png}")
    print(f"Wrote {field_subset_csv}")
    print(f"Wrote {stack_pdf}")
    print(f"Wrote {stack_png}")
    print(f"Wrote {stack_density_csv}")
    print(f"Wrote {stack_weights_csv}")
    print(
        "beta=0: "
        f"H0={summaries[0]['H0_mean']:.3f} +- "
        f"{summaries[0]['H0_std']:.3f}, "
        f"N_eff={summaries[0]['n_eff']:.1f}"
    )
    print(
        "beta=1: "
        f"H0={summaries[-1]['H0_mean']:.3f} +- "
        f"{summaries[-1]['H0_std']:.3f}, "
        f"N_eff={summaries[-1]['n_eff']:.1f}, "
        f"dominant field={summaries[-1]['dominant_field']}"
    )
    for row in target_rows:
        print(
            "target N_eff: "
            f"K={row['target_neff']:g}, eta={row['beta']:.4f}, "
            f"H0={row['H0_q50']:.3f} "
            f"-{row['H0_q50'] - row['H0_q16']:.3f} "
            f"+{row['H0_q84'] - row['H0_q50']:.3f}, "
            f"max weight={row['max_weight']:.3f}, "
            f"dominant field={row['dominant_field']}"
        )
    print(
        "Bayesian bootstrap: "
        f"median H0={boot_q50:.3f}, 68%=[{boot_q16:.3f}, {boot_q84:.3f}]"
    )
    print(
        "leave-one-out: "
        f"largest |delta H0|={most_influential['delta_H0_mean']:.3f} "
        f"from omitting field {most_influential['omitted_field']} "
        f"(full weight={most_influential['omitted_full_weight']:.3f})"
    )
    print(
        "progressive top-drop: "
        f"max N_eff={max_progressive_neff['n_eff']:.2f} "
        f"after removing {max_progressive_neff['n_removed']} fields; "
        f"best remaining field={max_progressive_neff['best_remaining_field']}"
    )
    print(
        "field-subset BMA: "
        f"H0 median={subset_h0_q50:.3f}, "
        f"68%=[{subset_h0_q16:.3f}, {subset_h0_q84:.3f}], "
        f"median N_eff={subset_neff_q50:.2f}, "
        f"top field included={np.mean(subset_top_included):.3f}, "
        f"most common dominant field={subset_dominant_field} "
        f"({subset_dominant_fraction:.3f})"
    )
    print(
        "field-subset stacked posterior: "
        f"N_eff={stack_neff:.2f}, "
        f"largest mean stack weight={stack_top['stack_weight']:.3f} "
        f"(field {stack_top['field']})"
    )
    return {
        "tempered_pdf": out_pdf,
        "tempered_png": out_png,
        "tempered_summary_csv": summary_csv,
        "target_pdf": target_pdf,
        "target_png": target_png,
        "target_density_csv": target_density_csv,
        "target_summary_csv": target_summary_csv,
        "target_weights_csv": target_weights_csv,
        "resampling_pdf": resampling_pdf,
        "resampling_png": resampling_png,
        "bootstrap_csv": bootstrap_csv,
        "loo_csv": loo_csv,
        "progressive_pdf": progressive_pdf,
        "progressive_png": progressive_png,
        "progressive_csv": progressive_csv,
        "field_subset_pdf": field_subset_pdf,
        "field_subset_png": field_subset_png,
        "field_subset_csv": field_subset_csv,
        "stack_pdf": stack_pdf,
        "stack_png": stack_png,
        "stack_density_csv": stack_density_csv,
        "stack_weights_csv": stack_weights_csv,
    }


def main():
    args = parse_args()
    run_analysis(
        likelihood=args.likelihood,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        summary_dir=args.summary_dir,
        num_beta=args.num_beta,
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
        field_subset_fraction=args.field_subset_fraction,
        target_neff=args.target_neff,
    )


if __name__ == "__main__":
    main()
