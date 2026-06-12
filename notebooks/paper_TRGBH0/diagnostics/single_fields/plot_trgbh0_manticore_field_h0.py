#!/usr/bin/env python
"""Plot H0 posteriors across one-Manticore-field TRGBH0 runs."""
from argparse import ArgumentParser
import csv
import re
import shutil
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
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from trgbh0_plot_style import (  # noqa: E402
    FIGURE_DPI,
    ROOT,
    paper_style,
    save_pdf_png,
    trgbh0_cmap,
)


RESULTS = (
    ROOT / "results" / "TRGBH0_paper" / "manticore_fields_const_sigv"
)
OUTDIR = RESULTS / "plots"

FIELD_SET_SPECS = {
    "cola": {
        "pattern": (
            "EDD_TRGB_sel-TRGB_magnitude_"
            "COLA_manticore_2MPP_MULTIBIN_N256_DES_V2_field*_"
            "manticore_field_const_sigv.hdf5"
        ),
        "expected": 50,
        "label": "COLA",
        "suffix": "",
    },
    "non-cola": {
        "pattern": (
            "EDD_TRGB_sel-TRGB_magnitude_"
            "manticore_2MPP_MULTIBIN_N256_DES_V2_field*_"
            "manticore_field_const_sigv.hdf5"
        ),
        "expected": 30,
        "label": "non-COLA",
        "suffix": "_non_cola",
    },
}
FIELD_RE = re.compile(r"_field(\d+)_")


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field-set",
        choices=sorted(FIELD_SET_SPECS),
        default="cola",
        help="Manticore field set to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTDIR,
        help="Directory for PDF and PNG plot outputs.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=RESULTS,
        help="Directory for CSV summaries.",
    )
    parser.add_argument(
        "--no-copy-to-results",
        action="store_true",
        help="Do not copy figure outputs to the results root.",
    )
    return parser.parse_args()


def field_index(path):
    match = FIELD_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse field index from `{path}`.")
    return int(match.group(1))


def read_h0_and_evidence(path):
    with h5py.File(path, "r") as handle:
        h0 = np.asarray(handle["samples/H0"], dtype=float).reshape(-1)
        lnz_harmonic = float(handle["gof/lnZ_harmonic"][()])
    h0 = h0[np.isfinite(h0)]
    if h0.size == 0:
        raise ValueError(f"`{path}` has no finite H0 samples.")
    if not np.isfinite(lnz_harmonic):
        raise ValueError(f"`{path}` has non-finite gof/lnZ_harmonic.")
    return h0, lnz_harmonic


def load_samples(pattern):
    paths = sorted(RESULTS.glob(pattern), key=field_index)
    if not paths:
        raise FileNotFoundError(f"No HDF5 files matching `{pattern}`.")
    rows = []
    for path in paths:
        samples, lnz_harmonic = read_h0_and_evidence(path)
        rows.append((field_index(path), path, samples, lnz_harmonic))
    return rows


def h0_summary(samples):
    q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)),
        "q16": float(q16),
        "q50": float(q50),
        "q84": float(q84),
    }


def write_summary(rows, path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "field", "lnZ_harmonic", "n_samples", "mean", "std",
                "q16", "q50", "q84", "source",
            ],
        )
        writer.writeheader()
        for field, source, samples, lnz_harmonic in rows:
            summary = h0_summary(samples)
            writer.writerow({
                "field": field,
                "lnZ_harmonic": lnz_harmonic,
                "n_samples": samples.size,
                **summary,
                "source": str(source),
            })


def kde_on_grid(samples, x_grid, bw=1.2):
    kde = gaussian_kde(samples)
    kde.set_bandwidth(kde.factor * bw)
    return kde(x_grid)


def evidence_norm(values):
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    return plt.Normalize(vmin=vmin, vmax=vmax)


def make_plot(rows, out_pdf, field_spec):
    fields = np.asarray([field for field, _, _, _ in rows], dtype=int)
    evidence = np.asarray([lnz for _, _, _, lnz in rows], dtype=float)
    summaries = [h0_summary(samples) for _, _, samples, _ in rows]
    medians = np.asarray([item["q50"] for item in summaries])
    all_samples = np.concatenate([samples for _, _, samples, _ in rows])
    x_min, x_max = np.percentile(all_samples, [0.2, 99.8])
    pad = 0.15 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 600)

    missing = [
        i for i in range(field_spec["expected"]) if i not in set(fields)]
    cmap = trgbh0_cmap("trgbh0_manticore_lnz_harmonic")
    norm = evidence_norm(evidence)

    with paper_style(styles=("science",), extra_rc={"legend.fontsize": 6.4}):
        fig, ax = plt.subplots(figsize=(4.7, 3.1))
        ax.hist(
            medians,
            bins=min(9, max(4, int(np.sqrt(len(medians))) + 2)),
            density=True,
            color="#b8b8b8",
            alpha=0.45,
            label=r"Field median $H_0$",
            zorder=0,
        )

        for __, __, samples, lnz_harmonic in rows:
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
            kde_on_grid(all_samples, x_grid),
            color="black",
            lw=1.35,
            label=r"Stacked posterior",
            zorder=4,
        )
        ax.axvline(
            np.median(medians), color="black", lw=1.0, ls=":",
            label=r"Median of field medians")
        ax.set_xlabel(
            r"$H_0 ~ [\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$")
        ax.set_ylabel("Posterior density")
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.set_ylim(bottom=0)

        note = f"{field_spec['label']}; {len(rows)} fields"
        if missing:
            note += "; missing " + ", ".join(str(i) for i in missing)
        ax.set_title(note, loc="left")

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

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.055)
        cbar.set_label(r"$\ln \mathcal{Z}$")
        cbar.ax.tick_params(labelsize=7.0)

        fig.tight_layout()
    return save_pdf_png(fig, out_pdf, dpi=FIGURE_DPI)[1]


def main():
    args = parse_args()
    field_spec = FIELD_SET_SPECS[args.field_set]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    rows = load_samples(field_spec["pattern"])
    summary_csv = (
        args.summary_dir
        / f"manticore_field_h0_summary{field_spec['suffix']}.csv"
    )
    write_summary(rows, summary_csv)

    out_pdf = (
        args.output_dir
        / f"trgbh0_manticore_field_h0_posteriors{field_spec['suffix']}.pdf"
    )
    out_png = make_plot(rows, out_pdf, field_spec)
    if not args.no_copy_to_results:
        for path in (out_pdf, out_png):
            destination = RESULTS / path.name
            if path.resolve() != destination.resolve():
                shutil.copyfile(path, destination)

    fields = [field for field, _, _, _ in rows]
    missing = [
        i for i in range(field_spec["expected"]) if i not in set(fields)]
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")
    print(f"Wrote {summary_csv}")
    print(f"Fields plotted: {len(fields)}; missing: {missing}")


if __name__ == "__main__":
    main()
