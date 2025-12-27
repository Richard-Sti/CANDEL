"""Migkas comparison corner plot."""
from config import setup_style, COLS, RESULTS_ROOT, get_figure_path

import matplotlib.pyplot as plt
from candel import plot_corner_from_hdf5


def main():
    setup_style()

    fnames = [
        str(RESULTS_ROOT / "Migkas_comparison/Migkas_Clusters_noMNR_linear_LT.hdf5"),
        str(RESULTS_ROOT / "maxgrid/Vext_LT_noMNR_dipVext_sigv100.hdf5"),
        str(RESULTS_ROOT / "rgrid1000/Vext_LT_noMNR_dipVext.hdf5"),
        str(RESULTS_ROOT / "rgrid1000/Carrick2015_LT_noMNR_dipVext.hdf5"),
    ]

    cols = [COLS[1], COLS[0], COLS[2], COLS[3]]

    labels = [
        "Migkas model",
        "Fiducial w/ no reconstruction, $\\sigma_v=100$ km/s",
        "Fiducial w/ no reconstruction",
        "Fiducial w/ Carrick2015 reconstruction",
    ]

    keys = ['Vext_mag', 'Vext_ell', 'Vext_b', 'sigma_v']
    points = {("Vext_ell", "Vext_b"): (264., 48.)}
    contour_args = [
        {"zorder": 3, "filled": False, "lw": 2.0},
        {"zorder": 2},
        {"zorder": 1},
        {"zorder": 1},
    ]

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=True,
        points=points,
        keys=keys,
        contour_args=contour_args,
        filename=str(get_figure_path("migkas_comparison.pdf")),
        legend_fontsize=40,
    )
    plt.close('all')


if __name__ == "__main__":
    main()
