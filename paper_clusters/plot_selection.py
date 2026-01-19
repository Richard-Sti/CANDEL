"""Plot the empirical distance selection function for different reconstructions."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc, gammaln

from config import (
    setup_style, get_results_path, get_figure_path,
    get_active_reconstructions, RECON_LABELS, RECON_COLORS,
    DATA_CONFIG_PATH,
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

    # Get active reconstructions
    recons = get_active_reconstructions()

    # Define files for each dipole model
    dipole_configs = []
    for model, title in [("dipVext", r"$\mathrm{dip}\,V_\mathrm{ext}$"),
                          ("dipH0", r"$\mathrm{dip}\,H_0$"),
                          ("dipA", r"$\mathrm{dip}\,A$")]:
        files = {}
        for recon in recons:
            suffix = "_hasY" if recon != "Vext" or model != "dipVext" else ""
            # LTYT always has _hasY
            suffix = "_hasY"
            files[recon] = f"{recon}_LTYT_noMNR_{model}{suffix}.hdf5"
        dipole_configs.append({
            "name": model,
            "title": title,
            "files": files,
        })

    # r grid for plotting
    r_grid = np.linspace(R_MIN, R_MAX, N_POINTS)
    Rmax = R_MAX  # Truncation for the prior

    # Create figure with 3 vertically stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    for ax, config in zip(axes, dipole_configs):
        # Plot selection functions
        for recon in recons:
            fname = config["files"][recon]
            col = RECON_COLORS[recon]
            label = RECON_LABELS[recon]

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
