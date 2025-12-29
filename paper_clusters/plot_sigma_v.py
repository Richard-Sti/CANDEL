"""Sigma_v 1D distribution plot."""
from config import setup_style, COLS, RESULTS_ROOT, RESULTS_FOLDER, get_figure_path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from getdist import plots, MCSamples

from candel import read_samples
try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib


def main():
    setup_style()

    # Define file patterns for each row (dipole model) and column (relation)
    # Row 0: dipVext, Row 1: dipH0, Row 2: dipA
    rows = [
        {
            "label": r"$\mathbf{\mathrm{dip}\,V_\mathrm{ext}}$",
            "relations": [
                {
                    "name": "LT",
                    "files": [
                        f"{RESULTS_FOLDER}/Vext_LT_noMNR_dipVext.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_LT_noMNR_dipVext.hdf5",
                        f"{RESULTS_FOLDER}/manticore_LT_noMNR_dipVext.hdf5",
                    ],
                },
                {
                    "name": "YT",
                    "files": [
                        f"{RESULTS_FOLDER}/Vext_YT_noMNR_dipVext_hasY.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_YT_noMNR_dipVext_hasY.hdf5",
                        f"{RESULTS_FOLDER}/manticore_YT_noMNR_dipVext_hasY.hdf5",
                    ],
                },
                {
                    "name": "LTYT",
                    "files": [
                        f"{RESULTS_FOLDER}/Vext_LTYT_noMNR_dipVext_hasY.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_LTYT_noMNR_dipVext_hasY.hdf5",
                        f"{RESULTS_FOLDER}/manticore_LTYT_noMNR_dipVext_hasY.hdf5",
                    ],
                },
            ],
        },
        {
            "label": r"$\mathbf{\mathrm{dip}\,H_0}$",
            "relations": [
                {
                    "name": "LT",
                    "files": [
                        f"{RESULTS_FOLDER}/Vext_LT_noMNR_dipH0.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_LT_noMNR_dipH0.hdf5",
                        f"{RESULTS_FOLDER}/manticore_LT_noMNR_dipH0.hdf5",
                    ],
                },
                {
                    "name": "YT",
                    "files": [
                        f"{RESULTS_FOLDER}/Vext_YT_noMNR_dipH0_hasY.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_YT_noMNR_dipH0_hasY.hdf5",
                        f"{RESULTS_FOLDER}/manticore_YT_noMNR_dipH0_hasY.hdf5",
                    ],
                },
                {
                    "name": "LTYT",
                    "files": [
                        f"{RESULTS_FOLDER}/Vext_LTYT_noMNR_dipH0_hasY.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_LTYT_noMNR_dipH0_hasY.hdf5",
                        f"{RESULTS_FOLDER}/manticore_LTYT_noMNR_dipH0_hasY.hdf5",
                    ],
                },
            ],
        },
        {
            "label": r"$\mathbf{\mathrm{dip}\,A}$",
            "relations": [
                {
                    "name": "LT",
                    "files": [
                        f"{RESULTS_FOLDER}/Vext_LT_noMNR_dipA.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_LT_noMNR_dipA.hdf5",
                        f"{RESULTS_FOLDER}/manticore_LT_noMNR_dipA.hdf5",
                    ],
                },
                {
                    "name": "YT",
                    "files": [
                        f"{RESULTS_FOLDER}/Vext_YT_noMNR_dipA_hasY.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_YT_noMNR_dipA_hasY.hdf5",
                        f"{RESULTS_FOLDER}/manticore_YT_noMNR_dipA_hasY.hdf5",
                    ],
                },
                {
                    "name": "LTYT",
                    "files": [
                        f"{RESULTS_FOLDER}/Vext_LTYT_noMNR_dipA_hasY.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_LTYT_noMNR_dipA_hasY.hdf5",
                        f"{RESULTS_FOLDER}/manticore_LTYT_noMNR_dipA_hasY.hdf5",
                    ],
                },
            ],
        },
    ]
    labels = ["No reconstruction", "Carrick2015", "Manticore"]
    cols = [COLS[0], COLS[1], COLS[2]]

    def read_sigma_v_lower_bound(fname):
        toml_path = RESULTS_ROOT / fname
        toml_path = toml_path.with_suffix(".toml")
        try:
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
            return float(data["model"]["priors"]["sigma_v"]["low"])
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            return 0.0

    def load_sigma_samples(fnames):
        sigma_samples = []
        lower_bounds = []
        smooth_scale_1d = 0.3
        for f in fnames:
            arr = read_samples(str(RESULTS_ROOT), f, keys="sigma_v")
            samples = np.asarray(arr).reshape(-1, 1)
            max_val = float(np.max(samples))
            lower = read_sigma_v_lower_bound(f)
            lower_bounds.append(lower)
            sigma_samples.append(MCSamples(
                samples=samples,
                names=["sigma_v"],
                labels=[r"\sigma_v\,(\mathrm{km\,s^{-1}})"],
                ranges={"sigma_v": [lower, max_val]},
                settings={"smooth_scale_1D": smooth_scale_1d},
            ))
        return sigma_samples, lower_bounds

    settings = plots.GetDistPlotSettings()
    settings.alpha_filled_add = -0.25
    g = plots.get_subplot_plotter(settings=settings)
    g.make_figure(nx=3, ny=3, sharex=True, sharey=True)

    for row_idx, row in enumerate(rows):
        for col_idx, relation in enumerate(row["relations"]):
            ax = g.get_axes((row_idx, col_idx))
            sigma_samples, lower_bounds = load_sigma_samples(relation["files"])
            g.plot_1d(
                sigma_samples,
                param="sigma_v",
                colors=cols,
                legend_labels=labels,
                ax=ax,
            )
            ax.set_xlim(left=max(lower_bounds))
            # Relation name in top right
            ax.text(
                0.98,
                0.95,
                relation["name"],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
            )
            # Only add x-label on bottom row
            if row_idx == 2:
                ax.set_xlabel(r"$\sigma_v\,(\mathrm{km\,s^{-1}})$")
            else:
                ax.set_xlabel("")

        # Add per-row y-axis label and a bold row label further left
        first_ax = g.subplots[row_idx][0]
        first_ax.set_ylabel("Density", fontsize=settings.axes_fontsize)
        first_ax.text(
            -0.28,
            0.5,
            row["label"],
            transform=first_ax.transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=settings.axes_fontsize + 2,
        )

    legend_handles = [
        Line2D([0], [0], color=col, lw=1.5) for col in cols
    ]
    g.fig.legend(
        handles=legend_handles,
        labels=labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        fontsize=9,
    )
    g.fig.subplots_adjust(left=0.16, right=0.98, bottom=0.10, top=0.92, wspace=0.18, hspace=0.15)
    g.fig.savefig(str(get_figure_path("sigma_v.pdf")), bbox_inches="tight")
    plt.close('all')


if __name__ == "__main__":
    main()
