"""Scaling relation comparison: LT vs YT vs LTYT corner plot."""
from config import setup_style, COLS, get_results_path, get_figure_path

import matplotlib.pyplot as plt
from candel import plot_corner_from_hdf5


def plot_dipa_scaling_comparison():
    """DipA scaling relation comparison: LT vs YT vs LTYT."""
    fnames = [
        get_results_path("Carrick2015_LT_noMNR_dipA.hdf5"),
        get_results_path("Carrick2015_YT_noMNR_dipA_hasY.hdf5"),
        get_results_path("Carrick2015_LTYT_noMNR_dipA_hasY.hdf5"),
    ]
    fnames = [str(f) for f in fnames]

    keys = ['dH_over_H_dipole', 'zeropoint_dipole_ell', 'zeropoint_dipole_b']
    cols = [COLS[0], COLS[1], COLS[2]]
    labels = ["LT", "YT", "LTYT"]

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=True,
        keys=keys,
        filename=str(get_figure_path("LT_YT_LTYT_dipA.pdf")),
        legend_fontsize=40,
        apply_ell_offset=True,
        ell_zero=-90.0,
    )
    plt.close('all')


def plot_diph0_scaling_comparison():
    """DipH0 scaling relation comparison: LT vs YT vs LTYT."""
    fnames = [
        get_results_path("Carrick2015_LT_noMNR_dipH0.hdf5"),
        get_results_path("Carrick2015_YT_noMNR_dipH0_hasY.hdf5"),
        get_results_path("Carrick2015_LTYT_noMNR_dipH0_hasY.hdf5"),
    ]
    fnames = [str(f) for f in fnames]

    keys = ['dH_over_H_dipole', 'zeropoint_dipole_ell', 'zeropoint_dipole_b']
    cols = [COLS[0], COLS[1], COLS[2]]
    labels = ["LT", "YT", "LTYT"]

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=True,
        keys=keys,
        filename=str(get_figure_path("LT_YT_LTYT_dipH0.pdf")),
        legend_fontsize=40,
        apply_ell_offset=True,
        ell_zero=-90.0,
    )
    plt.close('all')


def main():
    setup_style()
    plot_dipa_scaling_comparison()
    plot_diph0_scaling_comparison()


if __name__ == "__main__":
    main()
