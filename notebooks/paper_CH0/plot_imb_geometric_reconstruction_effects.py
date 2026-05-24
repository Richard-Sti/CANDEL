#!/usr/bin/env python
"""Plot CH0 imb-geometric reconstruction-effect summaries."""

from argparse import ArgumentParser
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = (
    ROOT / "results" / "CH0_paper" / "imb_geometric"
    / "plots" / "reconstruction_effects")
FIGURE_DPI = 450
CPLR_PARAMS = ("M_W", "b_W", "Z_W")
PARAM_LABELS = {
    "M_W": r"$M_W$",
    "b_W": r"$b_W$",
    "Z_W": r"$Z_W$",
}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-dir", type=Path, default=SUMMARY_DIR,
        help="Directory written by summarise_imb_geometric_reconstruction_effects.py.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for plots. Defaults to --summary-dir.")
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key):
    value = row.get(key, "")
    if value == "":
        return np.nan
    return float(value)


def save_pdf_png(fig, out_pdf):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_pdf, out_png


def plot_cplr_shifts(parameter_rows, outdir):
    fig, ax = plt.subplots(figsize=(6.8, 3.2), constrained_layout=True)
    rng = np.random.default_rng(143)

    for index, parameter in enumerate(CPLR_PARAMS):
        rows = [row for row in parameter_rows
                if row["parameter"] == parameter]
        m_rows = [row for row in rows if row["family"] == "manticore"]
        c_rows = [row for row in rows if row["family"] == "carrick"]

        m_deltas = np.asarray(
            [as_float(row, "delta_mean") for row in m_rows], dtype=float)
        m_deltas = m_deltas[np.isfinite(m_deltas)]
        if m_deltas.size:
            jitter = rng.normal(0.0, 0.035, size=m_deltas.size)
            ax.scatter(
                np.full(m_deltas.size, index) + jitter, m_deltas,
                s=16, alpha=0.35, color="#4a7ebb", linewidths=0,
                label="Manticore fields" if index == 0 else None)
            q16, q50, q84 = np.percentile(m_deltas, [16.0, 50.0, 84.0])
            ax.errorbar(
                index + 0.18, q50, yerr=[[q50 - q16], [q84 - q50]],
                fmt="o", color="#194f90", ms=5, capsize=3,
                label="Manticore central 68%" if index == 0 else None)

        if c_rows:
            ax.scatter(
                index - 0.18, as_float(c_rows[0], "delta_mean"),
                marker="D", s=38, color="#c44e52", zorder=5,
                label="Carrick2015" if index == 0 else None)

    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_xticks(range(len(CPLR_PARAMS)))
    ax.set_xticklabels([PARAM_LABELS[param] for param in CPLR_PARAMS])
    ax.set_ylabel("Posterior-mean shift vs no reconstruction")
    ax.set_title("CPLR calibration shifts", loc="left")
    ax.legend(frameon=False, fontsize=8)
    return save_pdf_png(
        fig, outdir / "imb_geometric_cplr_parameter_shifts.pdf")


def strip_mu(name):
    return name[3:] if name.startswith("mu_") else name


def distmod_to_mpc(mu):
    return 10.0 ** ((mu - 25.0) / 5.0)


def plot_host_distance_shifts(distance_shift_rows, outdir):
    rows = sorted(
        distance_shift_rows,
        key=lambda row: distmod_to_mpc(as_float(row, "reference_mean")))
    y = np.arange(len(rows))
    m_q16 = np.asarray(
        [as_float(row, "manticore_delta_mean_q16") for row in rows])
    m_q50 = np.asarray(
        [as_float(row, "manticore_delta_mean_q50") for row in rows])
    m_q84 = np.asarray(
        [as_float(row, "manticore_delta_mean_q84") for row in rows])
    carrick = np.asarray(
        [as_float(row, "carrick_delta_mean") for row in rows])
    distances = np.asarray(
        [distmod_to_mpc(as_float(row, "reference_mean")) for row in rows])
    labels = [strip_mu(row["host_name"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.4, 7.8), constrained_layout=True)
    ax.errorbar(
        m_q50, y, xerr=[m_q50 - m_q16, m_q84 - m_q50],
        fmt="o", color="#194f90", ecolor="#8fb4dc", elinewidth=1.2,
        capsize=2, ms=4, label="Manticore median and central 68%")
    ax.scatter(
        carrick, y, marker="D", s=24, color="#c44e52",
        label="Carrick2015")
    ax.axvline(0.0, color="0.25", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta \mu$ vs no reconstruction [mag]")
    ax.set_ylabel("Host, nearest to farthest")
    ax.set_title("Cepheid-host distance shifts", loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax_dist = ax.twinx()
    ax_dist.set_ylim(ax.get_ylim())
    ax_dist.set_yticks(y)
    ax_dist.set_yticklabels([f"{distance:.1f}" for distance in distances],
                            fontsize=6.5)
    ax_dist.set_ylabel("No-reconstruction distance [Mpc]")

    return save_pdf_png(
        fig, outdir / "imb_geometric_host_distance_shifts.pdf")


def plot_distance_coherence(coherence_rows, outdir):
    m_rows = [row for row in coherence_rows if row["family"] == "manticore"]
    c_rows = [row for row in coherence_rows if row["family"] == "carrick"]
    if not m_rows:
        return ()

    mean_shift = np.asarray([as_float(row, "mean_delta_mu") for row in m_rows])
    frac_positive = np.asarray(
        [as_float(row, "frac_positive_hosts") for row in m_rows])

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    axes[0].hist(
        mean_shift, bins=18, color="#4a7ebb", alpha=0.85,
        edgecolor="white")
    axes[0].axvline(0.0, color="0.25", lw=0.8)
    if c_rows:
        axes[0].axvline(
            as_float(c_rows[0], "mean_delta_mu"), color="#c44e52",
            lw=1.8, label="Carrick2015")
    axes[0].set_xlabel(r"Mean host $\Delta\mu$ [mag]")
    axes[0].set_ylabel("Manticore fields")
    axes[0].set_title("Mean shift per field", loc="left")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].hist(
        frac_positive, bins=np.linspace(0.0, 1.0, 16), color="#4a7ebb",
        alpha=0.85, edgecolor="white")
    axes[1].axvline(0.5, color="0.25", lw=0.8)
    if c_rows:
        axes[1].axvline(
            as_float(c_rows[0], "frac_positive_hosts"), color="#c44e52",
            lw=1.8)
    axes[1].set_xlabel("Fraction of hosts shifted high")
    axes[1].set_title("Sign coherence", loc="left")
    return save_pdf_png(
        fig, outdir / "imb_geometric_distance_coherence.pdf")


def finite_metric(rows, metric):
    values = np.asarray([as_float(row, metric) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def plot_evidence_or_note(evidence_rows, outdir):
    metrics = [
        ("lnZ_harmonic", r"harmonic $\ln Z$"),
        ("lnZ_laplace", r"Laplace $\ln Z$"),
        ("BIC", "BIC"),
        ("AIC", "AIC"),
    ]
    available = [
        (metric, label) for metric, label in metrics
        if finite_metric(evidence_rows, metric).size]

    if not available:
        path = outdir / "imb_geometric_evidence_unavailable.txt"
        path.write_text(
            "No evidence plot was made because none of the "
            "imb-geometric HDF5 outputs contains finite /gof evidence "
            "datasets. The task configs set inference.compute_evidence=false "
            "and did not save log_density.\n")
        return (path,)

    fig, axes = plt.subplots(
        1, len(available), figsize=(3.2 * len(available), 3.0),
        constrained_layout=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (metric, label) in zip(axes, available):
        for xpos, family, colour, marker in (
                (0, "no_reconstruction", "#404040", "s"),
                (1, "carrick", "#c44e52", "D"),
                (2, "manticore", "#4a7ebb", "o")):
            rows = [row for row in evidence_rows if row["family"] == family]
            values = finite_metric(rows, metric)
            if not values.size:
                continue
            jitter = np.linspace(-0.08, 0.08, values.size)
            if values.size == 1:
                jitter = np.zeros(1)
            ax.scatter(
                np.full(values.size, xpos) + jitter, values,
                s=28, color=colour, marker=marker, alpha=0.8)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["none", "Carrick", "Manticore"], rotation=20)
        ax.set_ylabel(label)
        ax.set_title(metric, loc="left")

    return save_pdf_png(
        fig, outdir / "imb_geometric_evidence_comparison.pdf")


def main():
    args = parse_args()
    summary_dir = args.summary_dir
    outdir = args.output_dir or summary_dir
    outdir.mkdir(parents=True, exist_ok=True)

    parameter_rows = read_rows(
        summary_dir / "imb_geometric_parameter_summary.csv")
    distance_shift_rows = read_rows(
        summary_dir / "imb_geometric_host_distance_shift_summary.csv")
    coherence_rows = read_rows(
        summary_dir / "imb_geometric_distance_coherence_summary.csv")
    evidence_rows = read_rows(
        summary_dir / "imb_geometric_evidence_summary.csv")

    outputs = []
    outputs.extend(plot_cplr_shifts(parameter_rows, outdir))
    outputs.extend(plot_host_distance_shifts(distance_shift_rows, outdir))
    outputs.extend(plot_distance_coherence(coherence_rows, outdir))
    outputs.extend(plot_evidence_or_note(evidence_rows, outdir))

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
