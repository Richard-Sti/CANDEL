import numpy as np
from scipy.integrate import simpson

C_LIGHT = 299792.458  # km/s


def rk_draw(n, rmin, rmax, k, rng):
    """Draw distances with prior p(r) ∝ r^k on [rmin, rmax]."""
    assert k > -1, "k must be > -1 for normalizable prior"
    u = rng.random(n)
    return (rmin**(k+1) + u * (rmax**(k+1) - rmin**(k+1)))**(1.0 / (k+1))


def simulate_catalog(n=1000, rmin=5.0, rmax=80.0, rmax_sel=None,
                     H0_true=73.0, sigma_m=0.4, sigma_vpec=300.0, seed=12345,
                     k=2, verbose=True):
    """
    Simulate a CF4-like catalog with TF μ and peculiar-velocity scatter.
    Distances are drawn from prior p(r) ∝ r^k on [rmin, rmax].
    """
    rng = np.random.default_rng(seed)

    if rmax_sel is not None:
        mu_max = 5.0 * np.log10(rmax_sel) + 25.0
        mu_max_sample = mu_max + 6 * sigma_m
        rmax_sample = 10**((mu_max_sample - 25.0) / 5.0)

        r_true = []
        mu_obs = []

        batch_size = int(0.3 * n)

        i = 0
        nsampled = 0
        while nsampled < n:
            r_true_i = rk_draw(batch_size, rmin, rmax_sample, k, rng)
            mu_true_i = 5.0 * np.log10(r_true_i) + 25.0
            mu_obs_i = rng.normal(mu_true_i, sigma_m, batch_size)
            mask = mu_obs_i < mu_max
            r_true.append(r_true_i[mask])
            mu_obs.append(mu_obs_i[mask])

            nsampled += mask.sum()
            i += 1

        r_true = np.concatenate(r_true)[:n]
        mu_obs = np.concatenate(mu_obs)[:n]
        print(f"It took {i} iterations to get {len(r_true)} objects.")

    else:
        r_true = rk_draw(n, rmin, rmax, k, rng)
        mu_true = 5.0 * np.log10(r_true) + 25.0
        mu_obs = rng.normal(mu_true, sigma_m, n)

    v_flow = H0_true * r_true
    v_obs = rng.normal(v_flow, sigma_vpec, n)  # observed Vcmb

    return v_obs, mu_obs


def log_pdf_gauss(x, mu, sigma):
    """Log of Gaussian PDF."""
    return -0.5 * ((x - mu)/sigma)**2 - 0.5 * np.log(2 * np.pi * sigma**2)


def fit_H0(Vcmb, mu_obs, sigma_mu=0.4, h0_min=50.0, h0_max=100,
           h0_step=0.025, k=-1, verbose=True, **kwargs):
    """Grid posterior for H0."""
    H0_grid = np.arange(h0_min, h0_max + 0.5 * h0_step, h0_step)
    if verbose:
        print(f"H0 grid: {H0_grid.size} points from {H0_grid[0]:.3f} "
              f"to {H0_grid[-1]:.3f} km/s/Mpc")

    # In practise since rmin > 0 and sigma_z assumed small this never
    # masks any objects.
    m = Vcmb > 0
    Vcmb, mu_obs = Vcmb[m], mu_obs[m]

    # Shape `(n_H0, n_data)`
    rhat = Vcmb[None, :] / H0_grid[:, None]
    muhat = 5 * np.log10(rhat) + 25.0

    # Compute the log-posterior on the grid
    lp = log_pdf_gauss(mu_obs[None, :], muhat, sigma_mu)
    lp -= (1 + k) * np.log(H0_grid[:, None])
    # Shape `(n_H0, )`
    lp = np.sum(lp, axis=-1)

    # Stabilize, normalize, compute mean and std
    p = np.exp(lp - lp.max())
    pnorm = simpson(p, H0_grid)
    if np.isfinite(pnorm) and pnorm > 0:
        p /= pnorm
    H0_mean = simpson(p * H0_grid, x=H0_grid)
    H0_std = np.sqrt(simpson(p * (H0_grid - H0_mean)**2, x=H0_grid))

    return H0_mean, H0_std


def fit_H0_sample_r(Vcmb, mu_obs, sigma_mu=0.4, h0_min=50.0, h0_max=100,
                    h0_step=0.025, k=-1, verbose=True, rmin=10, rmax=80,
                    sigma_vpec=300, nrstep=100, **kwargs):
    """Grid posterior for H0."""
    H0_grid = np.arange(h0_min, h0_max + 0.5 * h0_step, h0_step)
    if verbose:
        print(f"H0 grid: {H0_grid.size} points from {H0_grid[0]:.3f} "
              f"to {H0_grid[-1]:.3f} km/s/Mpc")

    # In practise since rmin > 0 and sigma_z assumed small this never
    # masks any objects.
    m = Vcmb > 0
    Vcmb, mu_obs = Vcmb[m], mu_obs[m]

    r_grid = np.linspace(rmin, rmax, nrstep)

    # Shape `(n_H0, n_data, n_r)`
    ll_cz = log_pdf_gauss(Vcmb[None, :, None] , H0_grid[:, None, None] * r_grid[None, None, :], sigma_vpec)
    ll_mu = log_pdf_gauss(mu_obs[None, :, None], 5 * np.log10(r_grid[None, None, :]) + 25.0, sigma_mu)
    ll_r = k * np.log(r_grid[None, None, :])

    lp = ll_cz + ll_mu + ll_r
    lp = np.log(simpson(np.exp(lp), r_grid, axis=-1))
    lp = np.sum(lp, axis=-1)

    # Stabilize, normalize, compute mean and std
    p = np.exp(lp - lp.max())
    pnorm = simpson(p, H0_grid)
    if np.isfinite(pnorm) and pnorm > 0:
        p /= pnorm
    H0_mean = simpson(p * H0_grid, x=H0_grid)
    H0_std = np.sqrt(simpson(p * (H0_grid - H0_mean)**2, x=H0_grid))

    return H0_mean, H0_std


def get_H0_map(Vcmb, mu_obs, sigma_mu, k):
    m = Vcmb > 0
    Vcmb, mu_obs = Vcmb[m], mu_obs[m]

    logH0 = - (1 + k) * sigma_mu**2 * np.log(10) / 25 + np.mean(5 - mu_obs / 5 + np.log10(Vcmb))
    return 10**logH0


sigma_vpec = 300
sigma_m = 0.4
k = 2
rmax = 80
rmax_sel = None
cat_kwargs = dict(n=500, rmin=10.0, rmax=rmax, H0_true=73.0,
                  sigma_m=sigma_m, sigma_vpec=sigma_vpec, k=k,
                  rmax_sel=rmax_sel)
fit_kwargs = dict(sigma_mu=sigma_m, sigma_vpec=sigma_vpec,
                  h0_min=50.0, h0_max=100.0, h0_step=0.1,
                  rmin=10, rmax=rmax, nrstep=100)

v_obs, mu_obs = simulate_catalog(**cat_kwargs, seed=44)

print(f"H0_true = {cat_kwargs['H0_true']} km/s/Mpc")
for k in [-1, 0, 1, 2]:
    # H0_mean, H0_std = fit_H0(v_obs, mu_obs, **fit_kwargs, k=k)
    H0_mean, H0_std = fit_H0_sample_r(v_obs, mu_obs, **fit_kwargs, k=k)
    H0_map = get_H0_map(v_obs, mu_obs, sigma_m, k)
    print(f"k={k:2d}  H0 = {H0_mean:.2f} ± {H0_std:.2f} km/s/Mpc "
          f"| H0_map = {H0_map:.2f} km/s/Mpc")




