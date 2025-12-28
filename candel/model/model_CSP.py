# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
CSP (Carnegie Supernova Project) forward model for LSQ SNe Ia.

Flow model with UNITY-style selection correction. No H0 inference.
"""
import jax.numpy as jnp
from jax import random
from jax.scipy.special import log_ndtr, logsumexp, ndtr
from jax.scipy.stats import norm as jax_norm
from numpyro import factor, plate, sample
from numpyro.distributions import MultivariateNormal

from ..util import get_nested, fprint, SPEED_OF_LIGHT
from ..cosmography import Distance2Distmod
from .model import (BaseModel, compute_Vext_radial, lp_galaxy_bias, predict_cz,
                    rsample, sample_galaxy_bias, sample_Vext)
from .simpson import ln_simpson


###############################################################################
#                         Selection integral                                  #
###############################################################################


def log1mexp(x):
    """Compute log(1 - exp(x)) stably for x < 0."""
    return jnp.where(
        x < -0.693, jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x)))


def log_ndtr_diff(a, b):
    """
    Compute log(Φ(a) - Φ(b)) in a numerically stable way.

    Assumes a > b. Uses log_ndtr which is stable for large negative arguments.
    When both a, b are large positive, uses Φ(a)-Φ(b) = Φ(-b)-Φ(-a).
    """
    # log_ndtr is stable for large negative args, unstable for large positive
    # When b > 5, both CDFs are ~1, so use: Φ(a)-Φ(b) = Φ(-b)-Φ(-a)
    flip = b > 5.0
    log_hi = jnp.where(flip, log_ndtr(-b), log_ndtr(a))
    log_lo = jnp.where(flip, log_ndtr(-a), log_ndtr(b))

    # log(Φ_hi - Φ_lo) = log(Φ_hi) + log(1 - Φ_lo/Φ_hi)
    #                  = log_hi + log(1 - exp(log_lo - log_hi))
    return log_hi + log1mexp(log_lo - log_hi)


class CSPSelection:
    """
    Compute p(S=1 | Lambda) for UNITY-style magnitude-limited selection.

    Selection model::

        m_sel = m_obs + alpha_sel * (s_obs - 1) - beta_sel * BV_obs
        p(S=1 | obs) = Phi((m_lim - m_sel) / sigma_sel)

    Also includes observable range selection for cz, s, and BV:
        p(cz_min < cz_obs < cz_max | r)
        p(s_min < s_obs < s_max | s_true)
        p(BV_min < BV_obs < BV_max | BV_true)

    Marginalizes over distance r ~ r^2, (s, BV) ~ N_2(mu, Sigma) with
    correlation rho_pop, and correlated measurement noise.
    """

    def __init__(self, r_grid, s_grid, BV_grid, distmod_grid,
                 distance2redshift, cz_lim, s_obs_lim, BV_obs_lim):
        self.r_grid = r_grid
        self.s_grid = s_grid
        self.BV_grid = BV_grid
        self.distmod_grid = distmod_grid
        self.distance2redshift = distance2redshift
        self.cz_min, self.cz_max = cz_lim
        self.s_obs_min, self.s_obs_max = s_obs_lim
        self.BV_obs_min, self.BV_obs_max = BV_obs_lim
        self.r_min, self.r_max = r_grid[0], r_grid[-1]

    def log_prior_r(self, r):
        """Log prior: pi(r) ~ r^2."""
        norm = (self.r_max**3 - self.r_min**3) / 3
        valid = (r >= self.r_min) & (r <= self.r_max)
        return jnp.where(valid, 2 * jnp.log(r) - jnp.log(norm), -jnp.inf)

    def _log_bvn(self, s, BV, mu_s, sigma_s, mu_BV, sigma_BV, rho):
        """Log PDF of bivariate normal for (s, BV)."""
        z_s = (s - mu_s) / sigma_s
        z_BV = (BV - mu_BV) / sigma_BV
        log_norm = (-jnp.log(2 * jnp.pi) - jnp.log(sigma_s)
                    - jnp.log(sigma_BV) - 0.5 * jnp.log(1 - rho**2))
        quad = (z_s**2 - 2 * rho * z_s * z_BV + z_BV**2) / (1 - rho**2)
        return log_norm - 0.5 * quad

    def log_prior_s_BV(self, s, BV, mu_s, sigma_s, mu_BV, sigma_BV, rho_pop,
                       use_gmm=False, mu_s2=None, sigma_s2=None, rho_pop2=None,
                       w_s=None):
        """
        Log prior for (s, BV).

        If use_gmm=False: (s, BV) ~ N_2 with correlation rho_pop.
        If use_gmm=True: mixture of two bivariate Gaussians,
            w_s * N_2(mu1, Sigma1) + (1-w_s) * N_2(mu2, Sigma2)
            where each component shares mu_BV, sigma_BV but has its own
            (mu_s, sigma_s, rho_pop).
        """
        if not use_gmm:
            return self._log_bvn(
                s, BV, mu_s, sigma_s, mu_BV, sigma_BV, rho_pop)

        # GMM: two bivariate Gaussians with different (mu_s, sigma_s, rho)
        log_p1 = self._log_bvn(
            s, BV, mu_s, sigma_s, mu_BV, sigma_BV, rho_pop)
        log_p2 = self._log_bvn(
            s, BV, mu_s2, sigma_s2, mu_BV, sigma_BV, rho_pop2)
        return logsumexp(
            jnp.stack([jnp.log(w_s) + log_p1,
                       jnp.log(1 - w_s) + log_p2], axis=0), axis=0)

    def sigma_eff(self, sigma_sel, sigma_m, alpha_sel, beta_sel,
                  sigma_s_obs, sigma_BV_obs, rho_ms, rho_mBV, rho_sBV):
        """Effective selection width with correlated noise."""
        cov_term = alpha_sel * beta_sel * rho_sBV * sigma_s_obs * sigma_BV_obs
        var_eta = (sigma_m**2
                   + alpha_sel**2 * sigma_s_obs**2
                   + beta_sel**2 * sigma_BV_obs**2
                   + 2 * alpha_sel * rho_ms * sigma_m * sigma_s_obs
                   - 2 * beta_sel * rho_mBV * sigma_m * sigma_BV_obs
                   - 2 * cov_term)
        return jnp.sqrt(sigma_sel**2 + var_eta)

    def __call__(self, M_B, alpha, beta, sigma_m,
                 mu_s, sigma_s, mu_BV, sigma_BV, rho_pop,
                 m_lim, alpha_sel, beta_sel, sigma_sel,
                 sigma_s_obs, sigma_BV_obs, rho_ms, rho_mBV, rho_sBV,
                 sigma_v, use_gmm=False, mu_s2=None, sigma_s2=None,
                 rho_pop2=None, w_s=None):
        """Compute log p(S=1 | Lambda)."""
        sig_eff = self.sigma_eff(sigma_sel, sigma_m, alpha_sel, beta_sel,
                                 sigma_s_obs, sigma_BV_obs,
                                 rho_ms, rho_mBV, rho_sBV)

        n_r, n_s, n_BV = len(self.r_grid), len(self.s_grid), len(self.BV_grid)

        s_3d = self.s_grid[None, :, None]
        BV_3d = self.BV_grid[None, None, :]

        # Priors
        lp_r = self.log_prior_r(self.r_grid[:, None, None])
        lp_s_BV = self.log_prior_s_BV(
            s_3d, BV_3d, mu_s, sigma_s, mu_BV, sigma_BV, rho_pop,
            use_gmm=use_gmm, mu_s2=mu_s2, sigma_s2=sigma_s2,
            rho_pop2=rho_pop2, w_s=w_s)

        # True magnitude and selection probability
        mu = self.distmod_grid[:, None, None]
        m_true = M_B + mu - alpha * (s_3d - 1) + beta * BV_3d
        m_sel_true = m_true + alpha_sel * (s_3d - 1) - beta_sel * BV_3d
        log_p_mag_sel = jax_norm.logcdf((m_lim - m_sel_true) / sig_eff)

        # Redshift selection: p(cz_min < cz_obs < cz_max | r)
        # cz_obs ~ N(cz_pred(r), sigma_v), where cz_pred = c * z(r)
        cz_pred = SPEED_OF_LIGHT * self.distance2redshift(self.r_grid, h=1.0)
        cz_pred = cz_pred[:, None, None]
        a_cz = (self.cz_max - cz_pred) / sigma_v
        b_cz = (self.cz_min - cz_pred) / sigma_v
        log_p_cz_sel = log_ndtr_diff(a_cz, b_cz)

        # Stretch selection: p(s_obs_min < s_obs < s_obs_max | s_true)
        # s_obs ~ N(s_true, sigma_s_obs)
        a_s = (self.s_obs_max - s_3d) / sigma_s_obs
        b_s = (self.s_obs_min - s_3d) / sigma_s_obs
        log_p_s_sel = log_ndtr_diff(a_s, b_s)

        # Color selection: p(BV_obs_min < BV_obs < BV_obs_max | BV_true)
        # BV_obs ~ N(BV_true, sigma_BV_obs)
        a_BV = (self.BV_obs_max - BV_3d) / sigma_BV_obs
        b_BV = (self.BV_obs_min - BV_3d) / sigma_BV_obs
        log_p_BV_sel = log_ndtr_diff(a_BV, b_BV)

        # Integrate
        log_integrand = (lp_r + lp_s_BV + log_p_mag_sel
                         + log_p_cz_sel + log_p_s_sel + log_p_BV_sel)

        BV_grid_3d = jnp.broadcast_to(
            self.BV_grid[None, None, :], (n_r, n_s, n_BV))
        s_grid_2d = jnp.broadcast_to(self.s_grid[None, :], (n_r, n_s))

        log_int_BV = ln_simpson(log_integrand, x=BV_grid_3d, axis=-1)
        log_int_s = ln_simpson(log_int_BV, x=s_grid_2d, axis=-1)
        log_int_r = ln_simpson(log_int_s, x=self.r_grid, axis=-1)

        return log_int_r


###############################################################################
#                         Simulation for verification                         #
###############################################################################


def simulate_csp(key, n_samples, r_min, r_max,
                 M_B, alpha, beta, sigma_m,
                 mu_s, sigma_s, mu_BV, sigma_BV, rho_pop,
                 m_lim, alpha_sel, beta_sel, sigma_sel,
                 sigma_s_obs, sigma_BV_obs, rho_ms, rho_mBV, rho_sBV,
                 h, Om=0.3):
    """
    Simulate CSP SNe: draw from population, add noise, apply selection.
    """
    keys = random.split(key, 4)
    distance2distmod = Distance2Distmod(Om0=Om)

    # Distance from r^2 prior
    u = random.uniform(keys[0], (n_samples,))
    r = (u * (r_max**3 - r_min**3) + r_min**3)**(1/3)

    # (s, BV) from correlated Gaussian
    z = random.normal(keys[1], (n_samples, 2))
    L_pop = jnp.array([[1.0, 0.0], [rho_pop, jnp.sqrt(1 - rho_pop**2)]])
    z_corr = z @ L_pop.T
    s = mu_s + sigma_s * z_corr[:, 0]
    BV = mu_BV + sigma_BV * z_corr[:, 1]

    # True magnitude
    mu = distance2distmod(r, h)
    m_true = M_B + mu - alpha * (s - 1) + beta * BV

    # Correlated measurement noise
    cov_ms = rho_ms * sigma_m * sigma_s_obs
    cov_mBV = rho_mBV * sigma_m * sigma_BV_obs
    cov_sBV = rho_sBV * sigma_s_obs * sigma_BV_obs
    Sigma_obs = jnp.array([
        [sigma_m**2, cov_ms, cov_mBV],
        [cov_ms, sigma_s_obs**2, cov_sBV],
        [cov_mBV, cov_sBV, sigma_BV_obs**2]
    ])
    L_obs = jnp.linalg.cholesky(Sigma_obs)
    eps = random.normal(keys[2], (n_samples, 3)) @ L_obs.T

    m_obs = m_true + eps[:, 0]
    s_obs = s + eps[:, 1]
    BV_obs = BV + eps[:, 2]

    # Selection
    m_sel = m_obs + alpha_sel * (s_obs - 1) - beta_sel * BV_obs
    p_sel = ndtr((m_lim - m_sel) / sigma_sel)
    selected = random.uniform(keys[3], (n_samples,)) < p_sel

    return {
        'r': r, 's': s, 'BV': BV, 'm_true': m_true,
        'm_obs': m_obs, 's_obs': s_obs, 'BV_obs': BV_obs,
        'selected': selected,
        'selection_fraction': jnp.mean(selected.astype(float)),
    }


###############################################################################
#                         Flow model for inference                            #
###############################################################################


class CSPModel(BaseModel):
    """
    CSP SNe Ia flow model with UNITY-style selection.

    Distance is marginalized at each MCMC step. No H0 inference (fixed h=1).
    """

    def __init__(self, config_path):
        super().__init__(config_path)

        # Selection grid config (may be set to "auto")
        self._r_lim = get_nested(
            self.config, "model/r_limits_selection", [5, 300])
        self._n_r = get_nested(self.config, "model/n_r_selection", 101)
        self._s_lim = get_nested(
            self.config, "model/s_limits_selection", [0.5, 1.5])
        self._n_s = get_nested(self.config, "model/n_s_selection", 51)
        self._BV_lim = get_nested(
            self.config, "model/BV_limits_selection", [-0.3, 0.5])
        self._n_BV = get_nested(self.config, "model/n_BV_selection", 51)
        self._zcmb_lim = get_nested(
            self.config, "model/zcmb_limits_selection", [0.01, 0.15])
        self._auto_padding_nsigma = get_nested(
            self.config, "model/auto_limits_nsigma", 5.0)

        self.selection = None  # Created lazily if auto limits

        # GMM for stretch distribution
        self.use_stretch_gmm = get_nested(
            self.config, "model/use_stretch_gmm", False)
        if self.use_stretch_gmm:
            fprint("CSP stretch distribution: 2-component GMM")

    def _setup_selection(self, data):
        """Setup selection grids, using data to compute auto limits."""
        if self.selection is not None:
            return

        s_lim = self._s_lim
        BV_lim = self._BV_lim
        pad = self._auto_padding_nsigma

        # Get observed data ranges
        s_obs = data["obs_vec"][:, 1]
        BV_obs = data["obs_vec"][:, 2]
        sigma_s_typical = data["median_sigma_s"]
        sigma_BV_typical = data["median_sigma_BV"]

        # Observed limits: actual min/max of observed values
        s_obs_lim = [float(jnp.min(s_obs)), float(jnp.max(s_obs))]
        BV_obs_lim = [float(jnp.min(BV_obs)), float(jnp.max(BV_obs))]

        # True (latent) limits: observed range ± padding for measurement error
        if s_lim == "auto":
            s_lim = [s_obs_lim[0] - pad * sigma_s_typical,
                     s_obs_lim[1] + pad * sigma_s_typical]
            fprint(f"CSP auto s_limits (true): [{s_lim[0]:.3f}, {s_lim[1]:.3f}]"
                   f" (obs ± {pad}σ)")

        if BV_lim == "auto":
            BV_lim = [BV_obs_lim[0] - pad * sigma_BV_typical,
                      BV_obs_lim[1] + pad * sigma_BV_typical]
            fprint(f"CSP auto BV_limits (true): [{BV_lim[0]:.3f}, "
                   f"{BV_lim[1]:.3f}] (obs ± {pad}σ)")

        # Redshift (cz) limits: observed range
        czcmb = data["czcmb"]
        cz_obs_lim = [float(jnp.min(czcmb)), float(jnp.max(czcmb))]

        if self._zcmb_lim == "auto":
            cz_lim = cz_obs_lim
            fprint(f"CSP auto cz_limits: [{cz_lim[0]:.0f}, {cz_lim[1]:.0f}] km/s")
        else:
            cz_lim = [self._zcmb_lim[0] * SPEED_OF_LIGHT,
                      self._zcmb_lim[1] * SPEED_OF_LIGHT]

        fprint(f"CSP observed limits: s=[{s_obs_lim[0]:.3f}, {s_obs_lim[1]:.3f}]"
               f", BV=[{BV_obs_lim[0]:.3f}, {BV_obs_lim[1]:.3f}]"
               f", cz=[{cz_obs_lim[0]:.0f}, {cz_obs_lim[1]:.0f}] km/s")

        r_grid = jnp.linspace(self._r_lim[0], self._r_lim[1], self._n_r)
        distmod_grid = self.distance2distmod(r_grid, h=1.0)

        self.selection = CSPSelection(
            r_grid=r_grid,
            s_grid=jnp.linspace(s_lim[0], s_lim[1], self._n_s),
            BV_grid=jnp.linspace(BV_lim[0], BV_lim[1], self._n_BV),
            distmod_grid=distmod_grid,
            distance2redshift=self.distance2redshift,
            cz_lim=cz_lim,
            s_obs_lim=s_obs_lim,
            BV_obs_lim=BV_obs_lim,
        )

        n_eval = self._n_r * self._n_s * self._n_BV
        fprint(f"CSP selection grid: r=[{self._r_lim[0]}, {self._r_lim[1]}] "
               f"({self._n_r}), s=[{s_lim[0]:.3f}, {s_lim[1]:.3f}] "
               f"({self._n_s}), BV=[{BV_lim[0]:.2f}, {BV_lim[1]:.2f}] "
               f"({self._n_BV})")
        fprint(f"CSP selection cz=[{cz_lim[0]:.0f}, {cz_lim[1]:.0f}] km/s, "
               f"total evaluations: {n_eval}")

    def _validate_data(self, data):
        """Check observed values are within selection grid bounds."""
        if data.has_precomputed_los:
            raise ValueError(
                "CSPModel selection does not support reconstruction.")

        s_obs = data["obs_vec"][:, 1]
        s_min, s_max = self.selection.s_grid[0], self.selection.s_grid[-1]
        if jnp.any(s_obs < s_min) or jnp.any(s_obs > s_max):
            s_range = (float(jnp.min(s_obs)), float(jnp.max(s_obs)))
            raise ValueError(
                f"Observed stretch values {s_range} outside selection "
                f"grid [{float(s_min)}, {float(s_max)}].")

        # Check redshift limits (czcmb is already in km/s)
        czcmb = data["czcmb"]
        cz_min, cz_max = self.selection.cz_min, self.selection.cz_max
        if jnp.any(czcmb < cz_min) or jnp.any(czcmb > cz_max):
            cz_range = (float(jnp.min(czcmb)), float(jnp.max(czcmb)))
            raise ValueError(
                f"Observed cz values {cz_range} km/s outside selection limits "
                f"[{cz_min}, {cz_max}] km/s. Update zcmb_limits_selection.")

    def validate_data(self, data):
        """Call this before running inference to check data bounds."""
        self._setup_selection(data)
        self._validate_data(data)

    def __call__(self, data, shared_params=None):
        """NumPyro model for MCMC inference."""
        if self.selection is None:
            raise RuntimeError("Call validate_data(data) before inference.")

        h = 1.0  # Fixed, no H0 inference
        nsamples = len(data)

        # --- Tripp parameters ---
        M_B = rsample("M_B", self.priors["M_B"], shared_params)
        alpha_tripp = rsample(
            "alpha_tripp", self.priors["alpha_tripp"], shared_params)
        beta_tripp = rsample(
            "beta_tripp", self.priors["beta_tripp"], shared_params)
        sigma_int = rsample(
            "sigma_int", self.priors["sigma_int"], shared_params)

        # --- Population hyperparameters (hierarchical prior on s, BV) ---
        mu_s = rsample("mu_s", self.priors["mu_s"], shared_params)
        sigma_s = rsample("sigma_s", self.priors["sigma_s"], shared_params)
        mu_BV = rsample("mu_BV", self.priors["mu_BV"], shared_params)
        sigma_BV = rsample("sigma_BV", self.priors["sigma_BV"], shared_params)
        rho_pop = rsample("rho_pop", self.priors["rho_pop"], shared_params)

        # GMM parameters for stretch (if enabled)
        if self.use_stretch_gmm:
            delta_mu_s = rsample(
                "delta_mu_s", self.priors["delta_mu_s"], shared_params)
            mu_s2 = mu_s + delta_mu_s  # Ensures mu_s2 > mu_s
            sigma_s2 = rsample(
                "sigma_s2", self.priors["sigma_s2"], shared_params)
            rho_pop2 = rsample(
                "rho_pop2", self.priors["rho_pop2"], shared_params)
            w_s = rsample("w_s", self.priors["w_s"], shared_params)
        else:
            mu_s2, sigma_s2, rho_pop2, w_s = None, None, None, None

        # Build population covariance for (s, BV)
        pop_mean = jnp.array([mu_s, mu_BV])
        pop_cov = jnp.array([
            [sigma_s**2, rho_pop * sigma_s * sigma_BV],
            [rho_pop * sigma_s * sigma_BV, sigma_BV**2]
        ])
        if self.use_stretch_gmm:
            pop_mean2 = jnp.array([mu_s2, mu_BV])
            pop_cov2 = jnp.array([
                [sigma_s2**2, rho_pop2 * sigma_s2 * sigma_BV],
                [rho_pop2 * sigma_s2 * sigma_BV, sigma_BV**2]
            ])

        # --- Selection parameters ---
        m_lim = rsample("m_lim", self.priors["m_lim"], shared_params)
        alpha_sel = rsample(
            "alpha_sel", self.priors["alpha_sel"], shared_params)
        beta_sel = rsample(
            "beta_sel", self.priors["beta_sel"], shared_params)
        sigma_sel = rsample(
            "sigma_sel", self.priors["sigma_sel"], shared_params)

        # Median measurement errors and correlations for selection integral
        sigma_m_obs = data["median_sigma_m"]
        sigma_s_obs = data["median_sigma_s"]
        sigma_BV_obs = data["median_sigma_BV"]
        rho_ms = data["median_rho_ms"]
        rho_mBV = data["median_rho_mBV"]
        rho_sBV = data["median_rho_sBV"]

        # --- Velocity field ---
        Vext = sample_Vext(
            self.priors, self.which_Vext, shared_params, self.kwargs_Vext)
        sigma_v = rsample("sigma_v", self.priors["sigma_v"], shared_params)
        beta = rsample("beta", self.priors["beta"], shared_params)

        # Galaxy bias
        bias_params = sample_galaxy_bias(
            self.priors, self.galaxy_bias, shared_params,
            Om=self.Om, beta=beta)

        # --- Selection correction ---
        # Total scatter on observed magnitude = intrinsic + measurement
        sigma_m_total = jnp.sqrt(sigma_int**2 + sigma_m_obs**2)
        log_p_sel = self.selection(
            M_B, alpha_tripp, beta_tripp, sigma_m_total,
            mu_s, sigma_s, mu_BV, sigma_BV, rho_pop,
            m_lim, alpha_sel, beta_sel, sigma_sel,
            sigma_s_obs, sigma_BV_obs, rho_ms, rho_mBV, rho_sBV,
            sigma_v, use_gmm=self.use_stretch_gmm,
            mu_s2=mu_s2, sigma_s2=sigma_s2, rho_pop2=rho_pop2, w_s=w_s)

        # Per-source selection: p(S=1 | obs_i)
        # m_sel = m_obs + alpha_sel * (s_obs - 1) - beta_sel * BV_obs
        m_sel_obs = (data["obs_vec"][:, 0]
                     + alpha_sel * (data["obs_vec"][:, 1] - 1)
                     - beta_sel * data["obs_vec"][:, 2])
        log_p_sel_i = jax_norm.logcdf((m_lim - m_sel_obs) / sigma_sel)

        # --- Per-SN likelihood with latent (s, BV) ---
        r_grid = data["r_grid"]

        # Distance prior: volume prior r^2, shape (n_field, n_sn, n_r)
        r_min, r_max = r_grid[0], r_grid[-1]
        log_norm = jnp.log((r_max**3 - r_min**3) / 3)
        lp_dist = (2 * jnp.log(r_grid) - log_norm)[None, None, :]

        if data.has_precomputed_los:
            # Add inhomogeneous Malmquist bias
            lp_dist = lp_dist + lp_galaxy_bias(
                data["los_delta_r_grid"], data["los_log_density_r_grid"],
                bias_params, self.galaxy_bias)
            # Normalize
            lp_dist = lp_dist - ln_simpson(
                lp_dist, x=r_grid[None, None, :], axis=-1)[..., None]
            # Peculiar velocity from reconstruction
            Vrad = beta * data["los_velocity_r_grid"]
        else:
            Vrad = 0.0

        # Predicted cz
        Vext_rad = compute_Vext_radial(
            data, r_grid, Vext, which_Vext=self.which_Vext, **self.kwargs_Vext)
        zcosmo = self.distance2redshift(r_grid, h=h)
        czpred = predict_cz(zcosmo[None, None, :], Vrad + Vext_rad)

        e_cz = jnp.sqrt(data["e_czcmb"]**2 + sigma_v**2)
        ll_cz = jax_norm.logpdf(
            data["czcmb"][None, :, None], czpred, e_cz[None, :, None])

        with plate("data", nsamples):
            # Sample latent (s, BV) from population prior
            pop_dist = MultivariateNormal(pop_mean, pop_cov)
            x_latent = sample("x_latent", pop_dist)
            s_true = x_latent[:, 0]
            BV_true = x_latent[:, 1]

            if self.use_stretch_gmm:
                # Compute mixture PDF explicitly and correct for proposal
                pop_dist2 = MultivariateNormal(pop_mean2, pop_cov2)
                log_p1 = pop_dist.log_prob(x_latent)
                log_p2 = pop_dist2.log_prob(x_latent)
                log_mixture = logsumexp(
                    jnp.stack([jnp.log(w_s) + log_p1,
                               jnp.log(1 - w_s) + log_p2], axis=0), axis=0)
                # sample() added log_p1, replace with log_mixture
                factor("mixture_correction", log_mixture - log_p1)

            # Measurement covariance: (n_sn, 3, 3) for (m, s, BV)
            # Add intrinsic scatter to magnitude variance
            cov_obs = data["cov"].copy()
            cov_obs = cov_obs.at[:, 0, 0].add(sigma_int**2)

            # True magnitude at each distance: (n_sn, n_r)
            mu_r = self.distance2distmod(r_grid, h=h)
            m_true = (M_B + mu_r - alpha_tripp * (s_true - 1)[:, None]
                      + beta_tripp * BV_true[:, None])

            # Log-likelihood of observed | true for each distance in r_grid
            # Mean vector at each r: (n_sn, n_r, 3)
            s_broadcast = jnp.broadcast_to(s_true[:, None], m_true.shape)
            BV_broadcast = jnp.broadcast_to(BV_true[:, None], m_true.shape)
            mean_vec = jnp.stack([m_true, s_broadcast, BV_broadcast], axis=-1)

            # Compute MVN log-pdf for each (sn, r) pair
            # obs_vec: (n_sn, 3), mean_vec: (n_sn, n_r, 3), cov: (n_sn, 3, 3)
            diff = data["obs_vec"][:, None, :] - mean_vec
            L = jnp.linalg.cholesky(cov_obs)
            z = jnp.linalg.solve(L[:, None, :, :], diff[..., None])[..., 0]
            log_det = jnp.sum(
                jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
            ll_obs_mvn = -0.5 * (
                3 * jnp.log(2 * jnp.pi) + 2 * log_det[:, None]
                + jnp.sum(z**2, axis=-1))

            # Combine with cz likelihood and distance prior
            # ll_cz: (n_field, n_sn, n_r), ll_obs_mvn: (n_sn, n_r)
            ll = ll_cz + ll_obs_mvn[None, :, :] + lp_dist
            ll = ln_simpson(ll, x=r_grid[None, None, :], axis=-1)

            # Average over field realizations
            ll = logsumexp(ll, axis=0) - jnp.log(data.num_fields)

            # Add per-SN log-likelihood + per-source selection probability
            factor("ll_obs", ll + log_p_sel_i)

        # Selection correction: -N * log p(S=1 | Lambda)
        factor("ll_selection", -nsamples * log_p_sel)
