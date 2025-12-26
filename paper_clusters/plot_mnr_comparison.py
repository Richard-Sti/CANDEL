"""MNR vs noMNR comparison corner plot."""
from config import setup_style, COLS, get_results_path, get_figure_path

import matplotlib.pyplot as plt
from candel import plot_corner_from_hdf5


def main():
    setup_style()

    fnames = [
        get_results_path("Carrick2015_Clusters_MNR_LTYT_dipVext_hasY.hdf5"),
        get_results_path("Carrick2015_Clusters_noMNR_LTYT_dipVext_hasY.hdf5"),
    ]
    fnames = [str(f) for f in fnames]

    cols = [COLS[0], COLS[1], "k"]
    labels = ["MNR", "No MNR"]

    keys = ['A_CL', 'B_CL', 'Vext_mag', 'Vext_ell', 'Vext_b', 'sigma_v']
    points = {("Vext_ell", "Vext_b"): (264., 48.)}

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=False,
        points=points,
        keys=keys,
        filename=str(get_figure_path("mnr_comparison.pdf")),
    )
    plt.close('all')


if __name__ == "__main__":
    main()
