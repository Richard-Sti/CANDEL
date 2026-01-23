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

    keys = ['zeropoint_dipole_mag', 'zeropoint_dipole_ell', 'zeropoint_dipole_b']
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

    keys = ['H0_dipole_mag', 'H0_dipole_ell', 'H0_dipole_b']
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


def plot_dipvext_scaling_comparison():
    """DipVext scaling relation comparison: LT vs YT vs LTYT."""
    fnames = [
        get_results_path("Carrick2015_LT_noMNR_dipVext.hdf5"),
        get_results_path("Carrick2015_YT_noMNR_dipVext_hasY.hdf5"),
        get_results_path("Carrick2015_LTYT_noMNR_dipVext_hasY.hdf5"),
    ]
    fnames = [str(f) for f in fnames]

    keys = ['Vext_mag', 'Vext_ell', 'Vext_b']
    cols = [COLS[0], COLS[1], COLS[2]]
    labels = ["LT", "YT", "LTYT"]

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=True,
        keys=keys,
        filename=str(get_figure_path("LT_YT_LTYT_dipVext.pdf")),
        legend_fontsize=40,
        apply_ell_offset=True,
        ell_zero=180.0,
    )
    plt.close('all')


def plot_dipvext_scaling_comparison_all():
    """DipVext scaling relation comparison: LT vs YT vs LTYT - all common parameters."""
    fnames = [
        get_results_path("Carrick2015_LT_noMNR_dipVext.hdf5"),
        get_results_path("Carrick2015_YT_noMNR_dipVext_hasY.hdf5"),
        get_results_path("Carrick2015_LTYT_noMNR_dipVext_hasY.hdf5"),
    ]
    fnames = [str(f) for f in fnames]

    # Common parameters across LT, YT, and LTYT
    keys = [
        'beta', 'sigma_v',
        'R_dist_emp', 'n_dist_emp', 'p_dist_emp',
        'Vext_mag', 'Vext_ell', 'Vext_b',
    ]
    cols = [COLS[0], COLS[1], COLS[2]]
    labels = ["LT", "YT", "LTYT"]

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=True,
        keys=keys,
        filename=str(get_figure_path("LT_YT_LTYT_dipVext_all.pdf")),
        legend_fontsize=40,
        apply_ell_offset=True,
        ell_zero=180.0,
    )
    plt.close('all')


def main():
    setup_style()
    plot_dipa_scaling_comparison()
    plot_diph0_scaling_comparison()
    plot_dipvext_scaling_comparison()
    plot_dipvext_scaling_comparison_all()


if __name__ == "__main__":
    main()
