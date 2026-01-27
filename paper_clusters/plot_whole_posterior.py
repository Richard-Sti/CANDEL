"""Whole posterior corner plot."""
from config import (
    setup_style, COLS, get_results_path, get_figure_path,
)

import matplotlib.pyplot as plt
from candel import plot_corner_from_hdf5


def main():
    setup_style()

    # Always use Carrick2015 for the main posterior plot
    recon = "Carrick2015"

    fnames = [
        get_results_path(f"{recon}_LTYT_noMNR_hasY.hdf5"),
        get_results_path(f"{recon}_LTYT_noMNR_dipVext_hasY.hdf5"),
        get_results_path(f"{recon}_LTYT_noMNR_dipH0_hasY.hdf5"),
    ]
    fnames = [str(f) for f in fnames]

    keys = [
        'A_LT', 'B_LT', 'sigma_LT',
        'A_YT', 'B_YT', 'sigma_YT',
        'Clusters_hasY/rho12', 'sigma_v', 'beta',
        'R_dist_emp', 'n_dist_emp', 'p_dist_emp',
        'Vext_mag', 'Vext_ell', 'Vext_b',
        'H0_dipole_mag', 'H0_dipole_ell', 'H0_dipole_b',
    ]

    cols = [COLS[0], COLS[1], COLS[2]]

    labels = [
        "No dipole",
        r"$V_{\rm ext}$ dipole",
        r"$H_0$ dipole",
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
