"""Dipoles for different reconstructions: Vext and H0 corner plots."""
from config import setup_style, COLS, get_results_path, get_figure_path

import matplotlib.pyplot as plt
from candel import plot_corner_from_hdf5


def plot_vext_dipole():
    """Figure (a): Vext dipole for different reconstructions."""
    fnames = [
        get_results_path("Vext_LTYT_noMNR_dipVext_hasY.hdf5"),
        get_results_path("carrick2015_LTYT_noMNR_dipVext_hasY.hdf5"),
        get_results_path("manticore_LTYT_noMNR_dipVext_hasY.hdf5"),
    ]
    fnames = [str(f) for f in fnames]

    keys = ['Vext_mag', 'Vext_ell', 'Vext_b']
    cols = [COLS[0], COLS[1], COLS[2]]
    labels = ["No reconstruction", "Carrick2015", "Manticore"]
    points = {("Vext_ell", "Vext_b"): (264., 48.)}

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=True,
        points=points,
        keys=keys,
        filename=str(get_figure_path("reconstruction_vext.pdf")),
        legend_fontsize=40,
        apply_ell_offset=True,
        ell_zero=180.0,
    )
    plt.close('all')


def plot_h0_dipole():
    """Figure (b): H0 dipole for different reconstructions."""
    fnames = [
        get_results_path("Vext_LTYT_noMNR_dipH0_hasY.hdf5"),
        get_results_path("carrick2015_LTYT_noMNR_dipH0_hasY.hdf5"),
        get_results_path("manticore_LTYT_noMNR_dipH0_hasY.hdf5"),
    ]
    fnames = [str(f) for f in fnames]

    keys = ['dH_over_H_dipole', 'zeropoint_dipole_ell', 'zeropoint_dipole_b']
    cols = [COLS[0], COLS[1], COLS[2]]
    labels = ["No reconstruction", "Carrick2015", "Manticore"]
    points = {("Vext_ell", "Vext_b"): (264., 48.)}

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=True,
        points=points,
        keys=keys,
        filename=str(get_figure_path("reconstruction_H0.pdf")),
        legend_fontsize=40,
        ell_zero=-90.0,
        apply_ell_offset=True,
    )
    plt.close('all')


def plot_zeropoint_dipole():
    """Figure (c): Zeropoint dipole for different reconstructions."""
    fnames = [
        get_results_path("Vext_LTYT_noMNR_dipA_hasY.hdf5"),
        get_results_path("carrick2015_LTYT_noMNR_dipA_hasY.hdf5"),
        get_results_path("manticore_LTYT_noMNR_dipA_hasY.hdf5"),
    ]
    fnames = [str(f) for f in fnames]

    keys = ['zeropoint_dipole_mag', 'zeropoint_dipole_ell', 'zeropoint_dipole_b']
    cols = [COLS[0], COLS[1], COLS[2]]
    labels = ["No reconstruction", "Carrick2015", "Manticore"]

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=True,
        keys=keys,
        filename=str(get_figure_path("reconstruction_zeropoint.pdf")),
        legend_fontsize=40,
        ell_zero=-90.0,
        apply_ell_offset=True,
    )
    plt.close('all')


def main():
    setup_style()
    plot_vext_dipole()
    plot_h0_dipole()
    plot_zeropoint_dipole()


if __name__ == "__main__":
    main()
