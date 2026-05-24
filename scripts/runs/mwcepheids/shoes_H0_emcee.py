"""
SH0ES distance-ladder H0 inference using emcee.

Reproduces the R22 baseline H0 from the publicly released SH0ES equation
matrices (y, L, C). The model is y = L^T q + noise, noise ~ N(0, C), with
q[44] fixed at 0. H0 = 10^(q[46]/5).

The chi2 is projected into parameter space and centered on the least-squares
solution. Parameters are whitened (z = (q - mode) @ L_A, where A = L_A L_A^T
is the precision matrix) so the posterior is an isotropic unit Gaussian,
which emcee's stretch move samples efficiently.
"""
from pathlib import Path

import emcee
import numpy as np
from astropy.io import fits
from scipy.linalg import cho_factor, cho_solve

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "MWCepheids" / "SH0ES_H0"
N_PARAMS = 47


def load_data():
    """Project y = L^T q + N(0, C) into parameter space.

    chi2 = (y - L^T q)^T C^{-1} (y - L^T q) = q^T A q - 2 q^T b + const
    where A = L C^{-1} L^T  and  b = L C^{-1} y.

    Center on q0 (lstsq solution) to avoid catastrophic cancellation:
      chi2 = delta^T A delta + 2 delta^T r0,  delta = q - q0, r0 = A q0 - b.
    """
    Y = fits.open(
        DATA_DIR / "ally_shoes_ceph_topantheonwt6.0_112221.fits"
    )[0].data.astype(np.float64)
    L = fits.open(
        DATA_DIR / "alll_shoes_ceph_topantheonwt6.0_112221.fits"
    )[0].data.astype(np.float64)
    C = fits.open(
        DATA_DIR / "allc_shoes_ceph_topantheonwt6.0_112221.fits"
    )[0].data.astype(np.float64)
    q_lstsq, sigma_lstsq = np.loadtxt(
        DATA_DIR / "lstsq_results.txt", unpack=True)

    C_cho = cho_factor(C)
    C_inv = cho_solve(C_cho, np.eye(C.shape[0]))

    A = L @ (C_inv @ L.T)
    b = L @ (C_inv @ Y)
    r0 = A @ q_lstsq - b

    return A, r0, q_lstsq, sigma_lstsq


def log_prob_whitened(z):
    """Vectorized log-prob in whitened space: z is (n_walkers, n_samp)."""
    return -0.5 * np.sum(z**2, axis=1)


if __name__ == "__main__":
    n_burn, n_samples, n_walkers = 1000, 50_000, 128

    A, r0, q_lstsq, sigma_lstsq = load_data()

    # Drop fixed (index 44) and degenerate (sigma_lstsq == 0) parameters
    mask = np.ones(N_PARAMS, dtype=bool)
    mask[44] = False
    degen = np.where(sigma_lstsq == 0)[0]
    mask[degen] = False
    samp_idx = np.where(mask)[0]
    n_samp = len(samp_idx)

    print(f"Sampling {n_samp} / {N_PARAMS} parameters")
    print("  Fixed at 0: q[44]")
    for i in degen:
        print(f"  Degenerate (sigma_lstsq=0): q[{i}], "
              f"value = {q_lstsq[i]:.6f}")

    A_s = A[np.ix_(samp_idx, samp_idx)]
    r0_s = r0[samp_idx]
    q0_s = q_lstsq[samp_idx]

    # Complete the square: mode = q0 - A^{-1} r0, covariance = A^{-1}.
    # Whiten: z = (q - mode) @ L_A  where A = L_A @ L_A^T (Cholesky).
    # Then log p(z) = -0.5 ||z||^2, i.e. N(0, I).
    # Back-transform: q = z @ L_A^{-1} + mode.
    A_cho = cho_factor(A_s)
    post_mode = q0_s - cho_solve(A_cho, r0_s)
    L_A = np.linalg.cholesky(A_s)
    L_A_inv = np.linalg.inv(L_A)

    # Initialize walkers as standard normal (the whitened posterior)
    p0 = np.random.randn(n_walkers, n_samp)

    sampler = emcee.EnsembleSampler(
        n_walkers, n_samp, log_prob_whitened,
        vectorize=True,
    )

    print(f"Burning in ({n_burn} steps, {n_walkers} walkers)...")
    state = sampler.run_mcmc(p0, n_burn, progress=True)
    sampler.reset()

    print(f"Sampling ({n_samples} steps)...")
    sampler.run_mcmc(state, n_samples, progress=True)

    print(f"\nAcceptance fraction: {sampler.acceptance_fraction.mean():.3f}")

    # Autocorrelation thinning
    tau = sampler.get_autocorr_time(quiet=True)
    thin = max(1, int(np.max(tau) / 2))
    print(f"Autocorrelation time (max): {np.max(tau):.1f}, thinning by {thin}")

    # Transform back to physical parameters
    z_flat = sampler.get_chain(flat=True, thin=thin)
    q_flat = z_flat @ L_A_inv + post_mode
    print(f"Independent samples: ~{q_flat.shape[0]}")

    # q[46] = 5 log10(H0), q[37] = slope correction
    # (true slope = q[37] - 3.285)
    col = {j: k for k, j in enumerate(samp_idx)}
    H0 = 10 ** (q_flat[:, col[46]] / 5)
    MB = q_flat[:, col[42]]
    MWH = q_flat[:, col[38]]
    bW = q_flat[:, col[37]] - 3.285

    print(f"\n{'='*50}")
    print(f"H0 = {np.mean(H0):.2f} +/- {np.std(H0):.2f} km/s/Mpc")
    print(f"  median = {np.median(H0):.2f}")
    print(f"  16/84  = [{np.percentile(H0, 16):.2f}, "
          f"{np.percentile(H0, 84):.2f}]")
    print(f"M_B = {np.mean(MB):.4f} +/- {np.std(MB):.4f}")
    print(f"M_W^H = {np.mean(MWH):.4f} +/- {np.std(MWH):.4f}")
    print(f"b_W = {np.mean(bW):.4f} +/- {np.std(bW):.4f}")
    print(f"{'='*50}")
