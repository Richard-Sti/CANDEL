"""Whole posterior corner plot."""
from config import setup_style, COLS, get_results_path, get_figure_path

import matplotlib.pyplot as plt
from candel import plot_corner_from_hdf5


def main():
    setup_style()

    fnames = [
        get_results_path("manticore_LTYT_noMNR_nodipA_hasY.hdf5"),
        get_results_path("manticore_LTYT_noMNR_nodipA_dipVext_hasY.hdf5"),
        get_results_path("manticore_LTYT_noMNR_dipA_hasY.hdf5"),
    ]
    fnames = [str(f) for f in fnames]

    keys = [
        'A_CL', 'B_CL', 'A2_CL', 'B2_CL', 'sigma', 'sigma2', 'rho12', 'sigma_v',
        'R_dist_emp', 'n_dist_emp', 'p_dist_emp',
        'Vext_mag', 'Vext_ell', 'Vext_b',
        'dH_over_H_dipole', 'zeropoint_dipole_ell', 'zeropoint_dipole_b',
    ]

    cols = [COLS[0], COLS[1], "k"]

    labels = [
        "Migkas model",
        "Fiducial w/ no reconstruction",
        "Fiducial w/ Carrick2015 reconstruction",
    ]

    points = {("Vext_ell", "Vext_b"): (264., 48.)}

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=False,
        points=points,
        keys=keys,
        filename=str(get_figure_path("main_posterior.pdf")),
        legend_fontsize=40,
    )
    plt.close('all')


if __name__ == "__main__":
    main()
