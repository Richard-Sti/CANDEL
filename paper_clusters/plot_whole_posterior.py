"""Whole posterior corner plot."""
from config import (
    setup_style, COLS, get_results_path, get_figure_path, INCLUDE_MANTICORE,
)

import matplotlib.pyplot as plt
from candel import plot_corner_from_hdf5


def main():
    setup_style()

    # Use Manticore if enabled, otherwise use Carrick2015
    recon = "manticore" if INCLUDE_MANTICORE else "Carrick2015"

    fnames = [
        get_results_path(f"{recon}_LTYT_noMNR_hasY.hdf5"),
        get_results_path(f"{recon}_LTYT_noMNR_dipVext_hasY.hdf5"),
        get_results_path(f"{recon}_LTYT_noMNR_dipH0_hasY.hdf5"),
    ]
    fnames = [str(f) for f in fnames]

    keys = [
        'A_CL', 'B_CL', 'A2_CL', 'B2_CL', 'sigma', 'sigma2', 'rho12', 'sigma_v',
        'R_dist_emp', 'n_dist_emp', 'p_dist_emp',
        'Vext_mag', 'Vext_ell', 'Vext_b',
        'dH_over_H_dipole', 'zeropoint_dipole_ell', 'zeropoint_dipole_b',
    ]

    cols = [COLS[0], COLS[1], COLS[2]]

    labels = [
        "No dipole",
        "Vext dipole",
        "H0 dipole",
    ]

    points = {("Vext_ell", "Vext_b"): (264., 48.)}

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=True,
        points=points,
        keys=keys,
        filename=str(get_figure_path("main_posterior.pdf")),
        legend_fontsize=40,
    )
    plt.close('all')


if __name__ == "__main__":
    main()
