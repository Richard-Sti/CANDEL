"""Redshift distribution plot: histogram + Mollweide sky map."""
from config import (
    setup_style, COLS, DATA_CONFIG_PATH, get_figure_path
)

import matplotlib.pyplot as plt
import numpy as np

import candel
from candel.util import radec_to_galactic
from candel.cosmography import Redshift2Distance
from astropy.coordinates import SkyCoord, Supergalactic, Galactic
import astropy.units as u

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

    # Left panel: Overlapping histograms
    bins = np.linspace(0, 0.45, 25)
    ax1.hist(
        zcmb,
        bins=bins,
        alpha=0.45,
        label=f'All clusters (N={len(zcmb)})',
        color=COLS[1],
        edgecolor='black',
        linewidth=0.5,
    )
    ax1.hist(
        zcmb[has_Y],
        bins=bins,
        alpha=0.7,
        label=f'With $Y_{{SZ}}$ (N={np.sum(has_Y)})',
        color=COLS[0],
        edgecolor='black',
        linewidth=0.5,
    )

    ax1.set_xlabel(r'Redshift $z_{\rm CMB}$')
    ax1.set_ylabel('Number of clusters')
    ax1.set_xlim(left=0.0)
    ax1.legend()
    ax1.grid(False)

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
    ax1_top.set_xlabel(r'Comoving distance [$h^{-1}$ Mpc]')

    # Right panel: Mollweide projection
    ell_centered = ((ell + 180) % 360) - 180   # degrees in [-180, 180)
    ell_plot = -np.deg2rad(ell_centered)       # NEGATIVE makes l increase to the left
    b_rad = np.deg2rad(b)

    ax2.scatter(ell_plot[has_Y], b_rad[has_Y], s=20, alpha=0.7,
                color=COLS[0], edgecolors='black', linewidths=0.3)
    ax2.scatter(ell_plot[no_Y], b_rad[no_Y], s=20, alpha=0.7,
                color=COLS[1], edgecolors='black', linewidths=0.3)

    ax2.set_xlabel(r'$\ell$ [deg]')
    ax2.set_ylabel(r'$b$ [deg]')

    # Ticks: positions are x (already flipped), labels should be Galactic longitude
    tick_locs_deg = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    tick_locs_rad = np.deg2rad(tick_locs_deg)
    tick_labels = [f"{int((-loc) % 360)}°" for loc in tick_locs_deg]  # note the minus

    ax2.set_xticks(tick_locs_rad)
    ax2.set_xticklabels(tick_labels)
    ax2.grid(True, alpha=0.3)

    # --- Overplot Supergalactic plane: SGB = 0 deg ---
    sgl = np.linspace(0, 360, 1000) * u.deg
    sgb = np.zeros_like(sgl.value) * u.deg

    sg_coords = SkyCoord(sgl=sgl, sgb=sgb, frame=Supergalactic)
    gal_coords = sg_coords.transform_to(Galactic)

    # Galactic lon/lat in degrees
    l_sg = gal_coords.l.deg
    b_sg = gal_coords.b.deg

    # Match your plotting convention: l in [-180,180), and longitude increasing to the left
    l_sg_centered = ((l_sg + 180) % 360) - 180
    l_sg_plot = -np.deg2rad(l_sg_centered)
    b_sg_plot = np.deg2rad(b_sg)

    # Plot line; break across longitude wraps to avoid horizontal segments
    order = np.argsort(l_sg_plot)
    l_sorted = l_sg_plot[order]
    b_sorted = b_sg_plot[order]
    jumps = np.abs(np.diff(l_sorted)) > np.deg2rad(20)
    l_plot = np.insert(l_sorted, np.where(jumps)[0] + 1, np.nan)
    b_plot = np.insert(b_sorted, np.where(jumps)[0] + 1, np.nan)
    ax2.plot(l_plot, b_plot, ls='--', lw=1.2, color=COLS[2], alpha=0.9, zorder=5)

    # CMB dipole direction (Galactic l, b) as a black cross
    l_cmb, b_cmb = 264.0, 48.0
    l_cmb_centered = ((l_cmb + 180) % 360) - 180
    l_cmb_plot = -np.deg2rad(l_cmb_centered)
    b_cmb_plot = np.deg2rad(b_cmb)
    ax2.scatter([l_cmb_plot], [b_cmb_plot], marker='x', s=60,
                color='k', linewidths=1.5, zorder=6)

    plt.tight_layout()
    plt.savefig(get_figure_path('redshift.pdf'), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
