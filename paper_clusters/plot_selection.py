"""Plot the empirical distance selection function for different reconstructions."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc, gammaln

from config import (
    setup_style, COLS, get_results_path, get_figure_path, INCLUDE_MANTICORE,
    DATA_CONFIG_PATH
)
from candel import read_samples
import candel
from candel.cosmography import Redshift2Distance


# r grid for plotting (Mpc/h)
R_MIN = 0.1
R_MAX = 750
N_POINTS = 500


def log_prior_r_empirical(r, R, p, n, Rmax):
    """
    Log of the (empirical) truncated prior:
        π(r) ∝ r^p * exp(-(r/R)^n),   0 < r ≤ Rmax
    Normalized by Z = [R^(1+p) * γ(a, x)] / n with a = (1+p)/n, x = (Rmax/R)^n
    """
    a = (1.0 + p) / n
    x = (Rmax / R) ** n

    # log γ(a, x) = log Γ(a) + log P(a, x), P = regularized lower γ
    log_gamma_lower = (
        gammaln(a) + np.log(np.clip(gammainc(a, x), 1e-300, 1.0)))
    log_norm = (1.0 + p) * np.log(R) - np.log(n) + log_gamma_lower

    logpdf = p * np.log(r) - (r / R)**n - log_norm
    valid = (r > 0) & (r <= Rmax)
    return np.where(valid, logpdf, -np.inf)


def prior_r_empirical(r, R, p, n, Rmax):
    """Evaluate the empirical prior (not log)."""
    return np.exp(log_prior_r_empirical(r, R, p, n, Rmax))


def load_selection_samples(fname, n_samples=500):
    """Load R, p, n samples from an HDF5 file."""
    fpath = get_results_path(fname)
    samples = read_samples(str(fpath.parent), fpath.name,
                           keys=["R_dist_emp", "p_dist_emp", "n_dist_emp"])
    R = np.asarray(samples["R_dist_emp"]).flatten()
    p = np.asarray(samples["p_dist_emp"]).flatten()
    n = np.asarray(samples["n_dist_emp"]).flatten()

    # Subsample if needed
    if len(R) > n_samples:
        idx = np.random.choice(len(R), n_samples, replace=False)
        R, p, n = R[idx], p[idx], n[idx]

    return R, p, n


def compute_selection_band(r_grid, R_samples, p_samples, n_samples, Rmax,
                           percentiles=(16, 50, 84)):
    """Compute median and percentile bands for the selection function."""
    n_r = len(r_grid)
    n_samp = len(R_samples)

    # Evaluate prior for each sample
    priors = np.zeros((n_samp, n_r))
    for i in range(n_samp):
        priors[i] = prior_r_empirical(r_grid, R_samples[i], p_samples[i],
                                       n_samples[i], Rmax)

    # Compute percentiles
    lo, med, hi = np.percentile(priors, percentiles, axis=0)
    return lo, med, hi


def main():
    setup_style()

    # Load cluster data to get radii histogram
    data = candel.pvdata.load_PV_dataframes(str(DATA_CONFIG_PATH))
    zcmb = data.data['zcmb']
    r2d = Redshift2Distance()
    # Convert zcmb to comoving distance (assuming zcmb = zcosmo)
    cluster_radii = r2d(zcmb, h=1.0)

    # Define files for each dipole model
    dipole_configs = [
        {
            "name": "dipVext",
            "title": r"$\mathrm{dip}\,V_\mathrm{ext}$",
            "files": {
                "Carrick2015": "Carrick2015_LTYT_noMNR_dipVext_hasY.hdf5",
                "Manticore": "manticore_LTYT_noMNR_dipVext_hasY.hdf5",
                "2M++": "2mpp_zspace_galaxies_LTYT_noMNR_dipVext_hasY.hdf5",
                "No recon": "Vext_LTYT_noMNR_dipVext_hasY.hdf5",
            },
        },
        {
            "name": "dipH0",
            "title": r"$\mathrm{dip}\,H_0$",
            "files": {
                "Carrick2015": "Carrick2015_LTYT_noMNR_dipH0_hasY.hdf5",
                "Manticore": "manticore_LTYT_noMNR_dipH0_hasY.hdf5",
                "2M++": "2mpp_zspace_galaxies_LTYT_noMNR_dipH0_hasY.hdf5",
                "No recon": "Vext_LTYT_noMNR_dipH0_hasY.hdf5",
            },
        },
        {
            "name": "dipA",
            "title": r"$\mathrm{dip}\,A$",
            "files": {
                "Carrick2015": "Carrick2015_LTYT_noMNR_dipA_hasY.hdf5",
                "Manticore": "manticore_LTYT_noMNR_dipA_hasY.hdf5",
                "2M++": "2mpp_zspace_galaxies_LTYT_noMNR_dipA_hasY.hdf5",
                "No recon": "Vext_LTYT_noMNR_dipA_hasY.hdf5",
            },
        },
    ]

    # Reconstruction order and colors
    # Order: Carrick (pink), Manticore (orange), 2M++ (green), No recon (purple)
    recon_order = ["Carrick2015", "Manticore", "2M++", "No recon"]
    recon_cols = {
        "Carrick2015": COLS[3],  # pink
        "Manticore": COLS[1],    # orange
        "2M++": COLS[2],         # green
        "No recon": COLS[0],     # purple
    }
    recon_labels = {
        "Carrick2015": "Carrick2015",
        "Manticore": "Manticore",
        "2M++": r"2M++$\rho(z)$",
        "No recon": "No reconstruction",
    }

    # Filter out manticore if not included
    if not INCLUDE_MANTICORE:
        recon_order = [r for r in recon_order if r != "Manticore"]

    # r grid for plotting
    r_grid = np.linspace(R_MIN, R_MAX, N_POINTS)
    Rmax = R_MAX  # Truncation for the prior

    # Create figure with 3 vertically stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    for ax, config in zip(axes, dipole_configs):
        # Plot selection functions
        for recon in recon_order:
            fname = config["files"][recon]
            col = recon_cols[recon]
            label = recon_labels[recon]

            try:
                R_samp, p_samp, n_samp = load_selection_samples(fname)
                lo, med, hi = compute_selection_band(r_grid, R_samp, p_samp,
                                                      n_samp, Rmax)

                ax.plot(r_grid, med, color=col, lw=1.5, label=label)
                ax.fill_between(r_grid, lo, hi, color=col, alpha=0.2)
            except Exception as e:
                print(f"Warning: Could not load {fname}: {e}")
                continue

        ax.set_ylabel(r"$\pi(r)$")
        ax.set_title(config["title"], loc="right", fontsize=10)
        ax.set_xlim(R_MIN, R_MAX)
        ax.set_ylim(bottom=0)

        # Overplot histogram of cluster radii on secondary y-axis
        ax2 = ax.twinx()
        ax2.hist(cluster_radii, bins=30, range=(R_MIN, R_MAX),
                 alpha=0.3, color='gray', edgecolor='gray', linewidth=0.5)
        ax2.set_ylabel("Count", color='gray', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)

    axes[-1].set_xlabel(r"$r$ [Mpc/$h$]")
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(str(get_figure_path("selection_function.pdf")))
    plt.close(fig)
    print(f"Saved {get_figure_path('selection_function.pdf')}")


if __name__ == "__main__":
    main()
