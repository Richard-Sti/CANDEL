"""
SH0ES distance-ladder chi2 analysis using JAX/NumPyro.

Reproduces the R22 baseline H0 inference from the publicly released
SH0ES equation matrices (y, L, C). See:
  https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/SH0ES_Data
  Riess et al. 2022, Section 2.1, Equation 6.

The model is:
    y = L^T q + noise,   noise ~ N(0, C)
where q is a 47-parameter vector (distance moduli, PLR coefficients,
M_B, 5*log10(H0), etc.). Parameter q[44] is fixed at 0.

An initial PLR slope of -3.285 was absorbed into L, so the free slope
parameter q[37] is a small correction: true slope = q[37] - 3.285.

H0 is recovered as: H0 = 10^(q[46] / 5).
"""
import argparse
import warnings
from pathlib import Path

import corner
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import scienceplots  # noqa: F401
from astropy.io import fits
from jax.scipy.linalg import cho_factor, cho_solve
from numpyro.infer import MCMC, NUTS

warnings.filterwarnings("ignore", module="arviz")

# ── data loading ──────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "MWCepheids" / "SH0ES_H0"
RESULTS_DIR = REPO_ROOT / "results" / "MWCepheids"

FIXED_IDX = 44
FIXED_VAL = 0.0
N_PARAMS = 47
FREE_IDX = np.array([i for i in range(N_PARAMS) if i != FIXED_IDX])

# External constraint indices in the Y/C vectors
IDX_MWH_HST = 3207   # M^W_{H,HST}  (sigma = 0.082)
IDX_MWH_GAIA = 3208  # M^W_{H,Gaia} (sigma = 0.025)


def load_data(data_dir=DATA_DIR, drop_hst=False, gaia_constraint=None):
    fname_y = "ally_shoes_ceph_topantheonwt6.0_112221.fits"
    fname_l = "alll_shoes_ceph_topantheonwt6.0_112221.fits"
    fname_c = "allc_shoes_ceph_topantheonwt6.0_112221.fits"
    Y = fits.open(data_dir / fname_y)[0].data
    L = fits.open(data_dir / fname_l)[0].data
    C = fits.open(data_dir / fname_c)[0].data
    q_lstsq, sigma_lstsq = np.loadtxt(
        data_dir / "lstsq_results.txt", unpack=True
    )

    Y = Y.astype(np.float64)
    L = L.astype(np.float64)
    C = C.astype(np.float64)

    if drop_hst:
        print("Dropping HST M_W^H constraint (inflated to non-informative)")
        C[IDX_MWH_HST, IDX_MWH_HST] = 1e4

    if gaia_constraint is not None:
        mean, sigma = gaia_constraint
        print(f"Overriding Gaia M_W^H constraint: {mean:.4f} +/- {sigma:.4f}")
        Y[IDX_MWH_GAIA] = mean
        C[IDX_MWH_GAIA, IDX_MWH_GAIA] = sigma**2

    C_cho = cho_factor(jnp.array(C))
    C_inv = cho_solve(C_cho, jnp.eye(C.shape[0]))

    # Project into parameter space (float64 for the heavy linear algebra).
    # chi2 = q @ A @ q - 2 q @ b + c, but to avoid catastrophic
    # cancellation in float32 we center on q0 = lstsq solution:
    #   chi2 = delta @ A @ delta + 2 delta @ r0  (+ const dropped)
    # where delta = q - q0 is O(0.01) so float32 is safe.
    L_jax = jnp.array(L)
    Y_jax = jnp.array(Y)
    C_inv_L = C_inv @ L_jax.T
    A = L_jax @ C_inv_L                         # (N_params, N_params)
    b = L_jax @ (C_inv @ Y_jax)                  # (N_params,)
    q0 = jnp.array(q_lstsq)
    r0 = A @ q0 - b                              # (N_params,)

    return {
        "A": jnp.array(A, dtype=jnp.float32),
        "r0": jnp.array(r0, dtype=jnp.float32),
        "q0": jnp.array(q0, dtype=jnp.float32),
        "q_lstsq": q_lstsq.astype(np.float32),
        "sigma_lstsq": sigma_lstsq.astype(np.float32),
    }


# ── NumPyro model ────────────────────────────────────────────────────

def model(A, r0, q0, q_lstsq, sigma_lstsq, prior_width_ratio=10):
    mu_free = q_lstsq[FREE_IDX]
    hw_free = sigma_lstsq[FREE_IDX] * prior_width_ratio

    # Wide uniform priors on free parameters
    q_free = numpyro.sample(
        "q_free",
        dist.Uniform(
            jnp.array(mu_free - hw_free),
            jnp.array(mu_free + hw_free),
        ),
    )

    # Insert fixed parameter
    q = jnp.concatenate([
        q_free[:FIXED_IDX],
        jnp.array([FIXED_VAL]),
        q_free[FIXED_IDX:],
    ])

    # Centered chi2: delta @ A @ delta + 2 delta @ r0
    delta = q - q0
    chi2 = delta @ A @ delta + 2 * delta @ r0
    numpyro.factor("log_likelihood", -0.5 * chi2)


# ── run ───────────────────────────────────────────────────────────────

def run(n_warmup=500, n_samples=2000, n_chains=4, drop_hst=False,
        gaia_constraint=None):
    data = load_data(drop_hst=drop_hst, gaia_constraint=gaia_constraint)

    kernel = NUTS(model, dense_mass=True)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=n_chains)
    mcmc.run(
        jax.random.PRNGKey(42),
        A=data["A"],
        r0=data["r0"],
        q0=data["q0"],
        q_lstsq=data["q_lstsq"],
        sigma_lstsq=data["sigma_lstsq"],
    )
    mcmc.print_summary(exclude_deterministic=False)

    samples = mcmc.get_samples()
    q_free = samples["q_free"]

    # Parameter 46 in the full vector is index 45 in q_free
    # (since we removed index 44)
    idx_5logH0 = 46 if 46 < FIXED_IDX else 46 - 1  # = 45
    five_logH0 = q_free[:, idx_5logH0]
    H0 = 10 ** (five_logH0 / 5)

    print(f"\n{'='*50}")
    print(f"H0 = {np.mean(H0):.2f} +/- {np.std(H0):.2f} km/s/Mpc")
    print(f"  median = {np.median(H0):.2f}")
    print(f"  16/84  = [{np.percentile(H0, 16):.2f}, "
          f"{np.percentile(H0, 84):.2f}]")

    MB = q_free[:, 42]
    print(f"M_B = {np.mean(MB):.4f} +/- {np.std(MB):.4f}")

    MWH = q_free[:, 38]
    print(f"M_W^H = {np.mean(MWH):.4f} +/- {np.std(MWH):.4f}")

    slope_corr = q_free[:, 37]
    true_slope = slope_corr - 3.285
    print(f"b_W = {np.mean(true_slope):.4f} "
          f"+/- {np.std(true_slope):.4f}")

    ZW = q_free[:, 43]
    print(f"Z_W = {np.mean(ZW):.4f} +/- {np.std(ZW):.4f}")
    print(f"{'='*50}")

    # Check for large correlations among all 46 free parameters
    corr = np.corrcoef(np.array(q_free).T)
    np.fill_diagonal(corr, 0.0)
    ii, jj = np.where(np.abs(corr) > 0.5)
    mask = ii < jj
    if mask.any():
        print("\nCorrelated parameter pairs (|r| > 0.5):")
        for i, j in zip(ii[mask], jj[mask]):
            print(f"  q_free[{i}] -- q_free[{j}]:  r = {corr[i, j]:.3f}")
    else:
        print("\nNo parameter pairs with |r| > 0.5")

    # Corner plot of confidently identified physics parameters
    corner_data = np.column_stack([
        np.array(H0),
        np.array(MB),
        np.array(true_slope),
        np.array(q_free[:, 38]),   # M_W^H
        np.array(q_free[:, 43]),   # Z_W
    ])
    labels = [
        r"$H_0$", r"$M_B$", r"$b_W$", r"$M^W_H$", r"$Z_W$",
    ]

    out_dir = RESULTS_DIR / "baseline_SH0ES"
    out_dir.mkdir(parents=True, exist_ok=True)

    with plt.style.context("science"):
        fig = corner.corner(
            corner_data, labels=labels,
            show_titles=True, title_kwargs={"fontsize": 10},
            smooth=1,
        )

    fout = out_dir / "SH0ES_H0.png"
    fig.savefig(fout, dpi=200, bbox_inches="tight")
    print(f"\nCorner plot saved to {fout}")
    plt.close(fig)

    # Save chain to results
    fout_chain = out_dir / "samples.hdf5"
    with h5py.File(fout_chain, "w") as f:
        f.create_dataset("q_free", data=np.array(q_free))
        f.create_dataset("H0", data=np.array(H0))
    print(f"Chain saved to {fout_chain}")

    return mcmc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SH0ES H0 via NumPyro NUTS")
    parser.add_argument("--n-warmup", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--n-chains", type=int, default=10)
    parser.add_argument("--drop-hst", action="store_true",
                        help="Inflate HST M_W^H constraint to non-informative")
    parser.add_argument("--gaia-constraint", type=float, nargs=2,
                        metavar=("MEAN", "SIGMA"),
                        help="Override Gaia M_W^H constraint (mean sigma)")
    args = parser.parse_args()

    numpyro.set_host_device_count(args.n_chains)
    gaia = tuple(args.gaia_constraint) if args.gaia_constraint else None
    run(n_warmup=args.n_warmup, n_samples=args.n_samples,
        n_chains=args.n_chains, drop_hst=args.drop_hst,
        gaia_constraint=gaia)
