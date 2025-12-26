"""Migkas comparison corner plot."""
from config import setup_style, COLS, RESULTS_ROOT, get_figure_path

import matplotlib.pyplot as plt
from candel import plot_corner_from_hdf5


def main():
    setup_style()

    fnames = [
        "Migkas_comparison/Migkas_Clusters_noMNR_linear_LT.hdf5",
        "Migkas_comparison/Vext_Clusters_noMNR_linear_LT.hdf5",
        "Migkas_comparison/precomputed_los_Carrick2015_Clusters_noMNR_linear_LT.hdf5",
    ]
    fnames = [str(RESULTS_ROOT / f) for f in fnames]

    cols = [COLS[0], COLS[1], "k"]

    labels = [
        "Migkas model",
        "Fiducial w/ no reconstruction",
        "Fiducial w/ Carrick2015 reconstruction",
    ]

    keys = ['B_CL', 'Vext_mag', 'Vext_ell', 'Vext_b', 'sigma_v']
    points = {("Vext_ell", "Vext_b"): (264., 48.)}

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=False,
        points=points,
        keys=keys,
        filename=str(get_figure_path("migkas_comparison.pdf")),
    )
    plt.close('all')


if __name__ == "__main__":
    main()
