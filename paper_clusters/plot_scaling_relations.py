"""Scaling relation plot: L_x-T and Y_SZ-T relations."""
from config import setup_style, C_WITH_Y, C_NO_Y, CLUSTERS_DATA_PATH, get_figure_path

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM


def load_cluster_data():
    """Load cluster data with appropriate dtype."""
    dtype = [
        ('Cluster', 'U32'), ('z', 'f8'), ('Glon', 'f8'), ('Glat', 'f8'),
        ('Offset', 'f8'), ('T', 'f8'), ('Tmax', 'f8'), ('Tmin', 'f8'),
        ('Lx', 'f8'), ('eL', 'f8'), ('NHtot', 'f8'), ('Metal', 'f8'),
        ('Met_max', 'f8'), ('Met_min', 'f8'), ('Y_arcmin2', 'f8'),
        ('e_Y', 'f8'), ('Y5r500', 'f8'), ('e_Y2', 'f8'), ('Y_nr_no_ksz', 'f8'),
        ('e_Y3', 'f8'), ('Y_nr_mmf', 'f8'), ('e_Y4', 'f8'), ('Y_nr_mf', 'f8'),
        ('e_Y5', 'f8'), ('Abs2MASS', 'f8'), ('BCG_Offset', 'f8'),
        ('Catalog', 'U32'), ('Analysed_by', 'U32')
    ]
    return np.genfromtxt(str(CLUSTERS_DATA_PATH), dtype=dtype, skip_header=1)


def main():
    setup_style()

    data = load_cluster_data()

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    z = data['z']
    DA = cosmo.angular_diameter_distance(z).value  # [Mpc]

    # Compute Y D_A^2 in kpc^2
    arcmin = np.pi / (180 * 60)
    arcmin2_to_sr = arcmin ** 2
    Mpc_to_kpc = 1e3

    Y_arcmin2 = data['Y_arcmin2']
    e_Y = data['e_Y']
    YDA2 = Y_arcmin2 * arcmin2_to_sr * (DA * Mpc_to_kpc) ** 2
    hasY = np.isfinite(Y_arcmin2) & (Y_arcmin2 > 0)

    # Logged quantities + errors
    T = data['T']
    Tmax = data['Tmax']
    Tmin = data['Tmin']
    Lx = data['Lx']
    eL = data['eL']

    pos_T = np.isfinite(T) & (T > 0)
    pos_Lx = np.isfinite(Lx) & (Lx > 0)
    pos_YDA2 = np.isfinite(YDA2) & (YDA2 > 0) & hasY

    logT = np.log10(T)
    logLx = np.log10(Lx)
    logYDA2 = np.full_like(YDA2, np.nan, dtype=float)
    logYDA2[pos_YDA2] = np.log10(YDA2[pos_YDA2])

    log10e = np.log10(np.e)
    dT = (Tmax - Tmin) / 2.0
    e_logT = log10e * (dT / T)
    e_logLx = log10e * (eL / 100.0)
    e_logYDA2 = log10e * (e_Y / Y_arcmin2)

    # Masks
    mask_Lx_withY = pos_T & pos_Lx & hasY
    mask_Lx_withoutY = pos_T & pos_Lx & (~hasY)
    mask_YDA2 = pos_YDA2 & np.isfinite(logT) & np.isfinite(e_logT) & np.isfinite(e_logYDA2)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharex=True)

    # (1) Lx-T on the left
    ax = axes[0]
    ax.errorbar(
        logT[mask_Lx_withY], logLx[mask_Lx_withY],
        xerr=e_logT[mask_Lx_withY], yerr=e_logLx[mask_Lx_withY],
        fmt='o', ms=4, lw=0.8, capsize=2, mec='none', alpha=0.95,
        color=C_WITH_Y, label='with $Y$'
    )
    ax.errorbar(
        logT[mask_Lx_withoutY], logLx[mask_Lx_withoutY],
        xerr=e_logT[mask_Lx_withoutY], yerr=e_logLx[mask_Lx_withoutY],
        fmt='o', ms=4, lw=0.8, capsize=2, mec='none', alpha=0.95,
        color=C_NO_Y, label='no $Y$'
    )
    ax.set_xlabel(r'$\log_{10} T\ \mathrm{[keV]}$')
    ax.set_ylabel(r'$\log_{10} L_{\rm X}\ \mathrm{[10^{44}\ erg\,s^{-1}]}$')
    ax.grid(False)
    ax.legend(frameon=False)

    # (2) Y_SZ-T on the right
    ax = axes[1]
    ax.errorbar(
        logT[mask_YDA2], logYDA2[mask_YDA2],
        xerr=e_logT[mask_YDA2], yerr=e_logYDA2[mask_YDA2],
        fmt='o', ms=4, lw=0.8, capsize=2, mec='none', alpha=0.9,
        color=C_WITH_Y, label=r'$Y_{\mathrm{SZ}}$'
    )
    ax.set_xlabel(r'$\log_{10} T\ \mathrm{[keV]}$')
    ax.set_ylabel(r'$\log_{10} Y_{\mathrm{SZ}}\ \mathrm{[kpc^2]}$')
    ax.grid(False)
    # Legend removed from right panel per reviewer comment

    plt.tight_layout()
    plt.savefig(get_figure_path('relations.pdf'), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
