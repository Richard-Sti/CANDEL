#!/usr/bin/env python
"""Create TRGBH0 result plots and a LaTeX table from posterior samples."""
from pathlib import Path
import shutil

import h5py
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
RESULTS = ROOT / "results" / "TRGBH0_main" / "table"
OUTDIR = ROOT / "notebooks" / "paper_TRGBH0"
FIGDIR = Path("/mnt/users/rstiskalek/Papers/TRGBH0/Figures")
PAPERDIR = Path("/mnt/users/rstiskalek/Papers/TRGBH0")


RUNS = [
    (
        "EDD_TRGB_sel-TRGB_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_main",
        "Individual, \\Manticore",
        "reference individual",
    ),
    (
        "EDD_TRGB_grouped_sel-TRGB_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_main",
        "Grouped, \\Manticore",
        "grouped reference",
    ),
    (
        "EDD_TRGB_sel-TRGB_magnitude_Carrick2015_main",
        "Individual, \\citetalias{Carrick_2015}",
        "C15 control",
    ),
    (
        "EDD_TRGB_grouped_sel-TRGB_magnitude_Carrick2015_main",
        "Grouped, \\citetalias{Carrick_2015}",
        "grouped C15",
    ),
    (
        "EDD_TRGB_sel-TRGB_magnitude_Vext_main",
        "Individual, bulk flow",
        "bulk-flow control",
    ),
    (
        "EDD_TRGB_grouped_sel-TRGB_magnitude_Vext_main",
        "Grouped, bulk flow",
        "grouped bulk-flow",
    ),
    (
        "EDD_TRGB_noVext_sel-TRGB_magnitude_main",
        "Individual, no bulk flow",
        "no-flow stress test",
    ),
    (
        "EDD_TRGB_sel-TRGB_magnitude_Carrick2015_beta_free_main",
        "Individual, \\citetalias{Carrick_2015} free $\\beta$",
        "weak-convergence stress test",
    ),
]


def load_samples(stem):
    with h5py.File(RESULTS / f"{stem}.hdf5", "r") as handle:
        return np.asarray(handle["samples/H0"])


def summarise(samples):
    q05, q16, q50, q84, q95 = np.percentile(samples, [5, 16, 50, 84, 95])
    return {
        "median": q50,
        "lo": q50 - q16,
        "hi": q84 - q50,
        "q05": q05,
        "q95": q95,
        "std": np.std(samples, ddof=1),
    }


def make_table(rows):
    lines = [
        "\\begin{table*}",
        "\\centering",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Sample & Peculiar-velocity model & $H_0$ median and 68 per cent CI & 5--95 per cent interval \\\\",
        " &  & $[\\kmsecMpc]$ & $[\\kmsecMpc]$ \\\\",
        "\\midrule",
    ]
    for label, note, stats in rows:
        h0 = (
            f"${stats['median']:.2f}^{{+{stats['hi']:.2f}}}"
            f"_{{-{stats['lo']:.2f}}}$"
        )
        interval = f"$[{stats['q05']:.2f},\\,{stats['q95']:.2f}]$"
        sample, model = label.split(", ", 1)
        if "weak-convergence" in note:
            model = model + r"\,$^\dagger$"
        lines.append(f"{sample} & {model} & {h0} & {interval} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Posterior constraints on $H_0$ for the TRGB-only variants.",
        "All runs use the EDD TRGB F814W magnitudes, the LMC and NGC\\,4258 TRGB anchors, a TRGB-magnitude selection function, and a uniform-in-volume baseline distance prior.",
        "Runs using reconstructed density fields additionally include inhomogeneous Malmquist weighting.",
        "The individual-galaxy \\Manticore\\ run uses the reconstructed velocity field at each galaxy position, while the grouped run replaces individual redshifts with Cosmicflows group velocities.",
        "The~\\citetalias{Carrick_2015}, bulk-flow, no-flow, and free-$\\beta$ cases test the sensitivity to the peculiar-velocity model.",
        "The $^\\dagger$ entry is a weakly converged diagnostic run and is not used as a quantitative robustness result.}",
        "\\label{tab:trgb_h0_variants}",
        "\\end{table*}",
        "",
    ]
    table_path = OUTDIR / "TRGBH0_variants_table.tex"
    table_path.write_text("\n".join(lines))
    shutil.copyfile(table_path, PAPERDIR / table_path.name)
    return table_path


def make_plot(rows):
    fig, (ax_main, ax_diag) = plt.subplots(
        2, 1, figsize=(7.2, 5.2), height_ratios=[1.2, 1.0],
        sharex=False,
    )

    def display_label(label):
        return (label
                .replace("\\Manticore", "Manticore-Local")
                .replace("\\citetalias{Carrick_2015}", "C15"))

    def plot_rows(ax, these_rows, colours, xlim, label_refs=False):
        y = np.arange(len(these_rows))[::-1]
        med = np.array([row[2]["median"] for row in these_rows])
        lo = np.array([row[2]["lo"] for row in these_rows])
        hi = np.array([row[2]["hi"] for row in these_rows])

        ax.axvspan(67.4 - 0.5, 67.4 + 0.5, color="#dddddd", zorder=0)
        ax.axvline(67.4, color="#808080", lw=1.2, ls="--", zorder=1)
        ax.axvspan(73.0 - 1.0, 73.0 + 1.0, color="#f3d6c7", zorder=0)
        ax.axvline(73.0, color="#b66a4c", lw=1.2, ls="--", zorder=1)
        ax.axvspan(71.5 - 1.8, 71.5 + 1.8, color="#d7e3f5", zorder=0)
        ax.axvline(71.5, color="#3b5b92", lw=1.2, ls=":", zorder=1)

        for i, (label, note, stats) in enumerate(these_rows):
            diagnostic = "stress" in note or "weak-convergence" in note
            marker = "s" if diagnostic else "o"
            fill = "white" if "weak-convergence" in note else colours[i]
            ax.errorbar(
                med[i],
                y[i],
                xerr=[[lo[i]], [hi[i]]],
                fmt=marker,
                ms=5,
                lw=1.6,
                capsize=3,
                color=colours[i],
                mfc=fill,
                zorder=3,
            )

        if label_refs:
            label_y = len(these_rows) - 0.05
            ax.text(67.4, label_y, r"Planck", ha="center", va="top", fontsize=8)
            ax.text(71.5, label_y - 0.3, r"Anand+22", ha="center", va="top", fontsize=8)
            ax.text(73.0, label_y, r"SH0ES", ha="center", va="top", fontsize=8)

        ax.set_yticks(y)
        ax.set_yticklabels([display_label(row[0]) for row in these_rows])
        ax.set_ylim(-0.6, len(these_rows) - 0.05)
        ax.set_xlim(*xlim)
        ax.grid(axis="x", color="#d9d9d9", lw=0.6)
        ax.tick_params(axis="y", length=0)

    plot_rows(
        ax_main, rows[:4],
        ["#3b5b92", "#577dbb", "#6f6f6f", "#9a9a9a"],
        (63, 75), label_refs=True,
    )
    plot_rows(
        ax_diag, rows[4:],
        ["#b05a4a", "#cc7d6d", "#8b3f3f", "#7a5a86"],
        (56, 88), label_refs=False,
    )
    ax_main.set_title("Reconstructed velocity-field variants", fontsize=9)
    ax_diag.set_title("Diagnostic stress tests", fontsize=9)
    ax_diag.set_xlabel(r"$H_0\ [\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$")
    fig.tight_layout()

    out = OUTDIR / "H0_TRGB_variants.pdf"
    fig.savefig(out)
    shutil.copyfile(out, FIGDIR / out.name)
    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem, label, note in RUNS:
        rows.append((label, note, summarise(load_samples(stem))))

    table_path = make_table(rows)
    figure_path = make_plot(rows)
    print(f"Wrote {table_path}")
    print(f"Wrote {figure_path}")
    print(f"Copied {figure_path.name} to {FIGDIR}")


if __name__ == "__main__":
    main()
