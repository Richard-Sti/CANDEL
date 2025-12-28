"""Sigma_v 1D distribution plot."""
from config import setup_style, COLS, RESULTS_ROOT, RESULTS_FOLDER, get_figure_path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from getdist import plots, MCSamples

from candel import read_samples


def main():
    setup_style()

    relations = [
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
    ]
    labels = ["No reconstruction", "Carrick2015", "Manticore"]
    cols = [COLS[0], COLS[1], COLS[2]]

    def load_sigma_samples(fnames):
        sigma_samples = []
        for f in fnames:
            arr = read_samples(str(RESULTS_ROOT), f, keys="sigma_v")
            sigma_samples.append(MCSamples(
                samples=np.asarray(arr).reshape(-1, 1),
                names=["sigma_v"],
                labels=[r"\sigma_v\,(\mathrm{km\,s^{-1}})"],
            ))
        return sigma_samples

    settings = plots.GetDistPlotSettings()
    settings.alpha_filled_add = -0.25
    g = plots.get_subplot_plotter(settings=settings)
    g.make_figure(nx=3, ny=1, sharey=True)

    for i, relation in enumerate(relations):
        ax = g.get_axes((0, i))
        sigma_samples = load_sigma_samples(relation["files"])
        g.plot_1d(
            sigma_samples,
            param="sigma_v",
            colors=cols,
            legend_labels=labels,
            ax=ax,
        )
        ax.text(
            0.98,
            0.95,
            relation["name"],
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )
        ax.set_xlabel(r"$\sigma_v\,(\mathrm{km\,s^{-1}})$")

    g.subplots[0][0].set_ylabel(
        r"$P(\sigma_v)$",
        fontsize=settings.axes_fontsize,
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
        bbox_to_anchor=(0.5, 1.08),
        fontsize=9,
    )

    g.fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.86, wspace=0.18)
    g.fig.savefig(str(get_figure_path("sigma_v.pdf")), bbox_inches="tight")
    plt.close('all')


if __name__ == "__main__":
    main()
