"""Redshift distribution plot: histogram + Mollweide sky map."""
from config import (
    setup_style, COLS, DATA_CONFIG_PATH, get_figure_path
)

import matplotlib.pyplot as plt
import numpy as np

import candel
from candel.util import radec_to_galactic
from candel.cosmography import Redshift2Distance


def main():
    setup_style()

    # Load data
    data = candel.pvdata.load_PV_dataframes(str(DATA_CONFIG_PATH))
    r2d = Redshift2Distance()

    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='mollweide')

    # Get data from PVDataFrame
    zcmb = data.data['zcmb']
    Y = data.data['Y']
    RA = data.data['RA']
    DEC = data.data['dec']

    # Convert RA, DEC to galactic coordinates
    ell, b = radec_to_galactic(RA, DEC)

    # Split by Y_SZ availability
    has_Y = Y > 0
    no_Y = Y < 0

    # Left panel: Stacked histogram
    bins = np.linspace(0, 0.45, 25)
    counts_no_Y, bin_edges = np.histogram(zcmb[no_Y], bins=bins)
    counts_has_Y, _ = np.histogram(zcmb[has_Y], bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]

    ax1.bar(bin_centers, counts_has_Y, width=width, alpha=0.7,
            label=f'With $Y_{{SZ}}$ (N={np.sum(has_Y)})',
            color=COLS[0], edgecolor='black', linewidth=0.5)
    ax1.bar(bin_centers, counts_no_Y, width=width, alpha=0.7,
            bottom=counts_has_Y,
            label=f'Without $Y_{{SZ}}$ (N={np.sum(no_Y)})',
            color=COLS[1], edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Redshift $z_{CMB}$', fontsize=12)
    ax1.set_ylabel('Number of clusters', fontsize=12)
    ax1.set_xlim(left=0.0)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Secondary x-axis for comoving distance
    ax1_top = ax1.twiny()

    def z_to_dist(z):
        return r2d(z, h=1.0)

    def dist_to_z(d):
        z_test = np.linspace(1e-5, 0.5, 1000)
        d_test = r2d(z_test, h=1.0)
        return np.interp(d, d_test, z_test)

    ax1_top.set_xlim(ax1.get_xlim())
    d_min, d_max = z_to_dist(np.array([1e-5, 0.5]))
    d_ticks = np.linspace(0, int(d_max / 200) * 200, 6)
    z_ticks = [dist_to_z(d) for d in d_ticks]

    ax1_top.set_xticks(z_ticks)
    ax1_top.set_xticklabels([f'{int(d)}' for d in d_ticks])
    ax1_top.set_xlabel('Comoving distance [Mpc/h]', fontsize=12)

    # Right panel: Mollweide projection
    ell_centered = np.where(ell > 180, ell - 360, ell)
    ell_rad = np.deg2rad(ell_centered)
    b_rad = np.deg2rad(b)

    ax2.scatter(ell_rad[has_Y], b_rad[has_Y], s=20, alpha=0.7,
                color=COLS[0], edgecolors='black', linewidths=0.3,
                label=f'With $Y_{{SZ}}$ (N={np.sum(has_Y)})')
    ax2.scatter(ell_rad[no_Y], b_rad[no_Y], s=20, alpha=0.7,
                color=COLS[1], edgecolors='black', linewidths=0.3,
                label=f'Without $Y_{{SZ}}$ (N={np.sum(no_Y)})')

    ax2.set_xlabel('$\\ell$ [deg]', fontsize=12)
    ax2.set_ylabel('$b$ [deg]', fontsize=12)

    # Custom tick labels for 0-360 degrees
    tick_locs = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    tick_labels = [(loc + 180) % 360 for loc in tick_locs]
    ax2.set_xticks(np.deg2rad(tick_locs))
    ax2.set_xticklabels([f'{int(label)}°' for label in tick_labels])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(get_figure_path('redshift.pdf'), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
