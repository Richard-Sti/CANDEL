"""Sigma_v 1D distribution plot."""
from config import setup_style, COLS, RESULTS_ROOT, RESULTS_FOLDER, get_figure_path

import matplotlib.pyplot as plt
import numpy as np
from getdist import plots, MCSamples

from candel import read_samples


def main():
    setup_style()

    fnames = [
        f"{RESULTS_FOLDER}/Vext_LTYT_noMNR_nodipA_dipVext_hasY.hdf5",
        f"{RESULTS_FOLDER}/carrick2015_LTYT_noMNR_nodipA_dipVext_hasY.hdf5",
        f"{RESULTS_FOLDER}/manticore_LTYT_noMNR_nodipA_dipVext_hasY.hdf5",
    ]
    labels = ["No reconstruction", "Carrick2015", "Manticore"]
    cols = [COLS[0], COLS[1], COLS[2]]

    sigma_samples = []
    for f in fnames:
        arr = read_samples(str(RESULTS_ROOT), f, keys="sigma_v")
        sigma_samples.append(MCSamples(
            samples=np.asarray(arr).reshape(-1, 1),
            names=["sigma_v"],
            labels=[r"\sigma_v"],
        ))

    settings = plots.GetDistPlotSettings()
    settings.alpha_filled_add = -0.25
    g = plots.get_subplot_plotter(settings=settings)
    g.plot_1d(
        sigma_samples,
        param="sigma_v",
        colors=cols,
        legend_labels=labels,
    )
    g.add_legend(
        legend_labels=labels,
        legend_loc='center right',
        fontsize=5,
        frameon=False,
    )

    ax = g.subplots[0][0]
    ax.set_ylabel(r"$P(\sigma_v)$", fontsize=settings.axes_fontsize)

    g.export(str(get_figure_path("sigma_v.pdf")))
    plt.close('all')


if __name__ == "__main__":
    main()
