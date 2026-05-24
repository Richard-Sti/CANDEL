# Copyright (C) 2026 Richard Stiskalek
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
"""Forward model for MW Cepheid parallax calibration."""
import jax
import jax.numpy as jnp
import numpyro
from jax.scipy.special import log_ndtr
from numpyro import distributions as dist

from ..utils import (DistanceModulusPrior, get_named_or_shared,
                     sample_prior)
from .distance_marg import log_likelihood_marg_distance
from .distributions import DiskPrior
from .model_setup import ModelSetupMixin
from .selection import (sample_selection_params, selection_correction,
                        spiral_log_factor)


def _chi2_loglike(residual, sigma):
    """Unnormalised Gaussian log-likelihood: -0.5 * sum((r/s)^2)."""
    return -0.5 * jnp.sum((residual / sigma)**2)


def _marginalise_latent(obs, mu, sigma, epsilon):
    """Analytically marginalise over a Gaussian latent variable.

    Returns effective value, posterior variance, and log marginalisation
    factor for each star (Eqs. 16-17 in the paper).
    """
    tilde_sigma_sq = 1.0 / (1.0 / epsilon**2 + 1.0 / sigma**2)
    x_star = tilde_sigma_sq * (obs / epsilon**2 + mu / sigma**2)

    marg_std = jnp.sqrt(epsilon**2 + sigma**2)
    log_factor = dist.Normal(mu, marg_std).log_prob(obs)

    return x_star, tilde_sigma_sq, log_factor


class MWCepheidModel(ModelSetupMixin):
    """Bayesian forward model for MW Cepheid P-L calibration.

    Global parameters (sampled once)::

        M_{H,1}    ~ prior       Fiducial absolute magnitude at logP = 1
        b_W        ~ prior       P-L slope (Wesenheit)
        Z_W        ~ prior       Metallicity coefficient
        delta_pi   ~ prior       Parallax zero-point offset [mas]
        sigma_int  ~ prior       Intrinsic P-L scatter [mag]
        f_pi       ~ prior       Parallax error inflation factor

    Per-population hyperparameters (sampled per campaign and anchor)::

        mu_logP, sigma_logP     Period distribution
        mu_OH, sigma_OH         Metallicity distribution

    Metallicity marginalisation (per star i in campaign c)::

        [O/H]_* = tilde_sigma^2 (OH_obs/eps^2 + mu_OH/sigma_OH^2)
        sigma_1 = sqrt(sigma_m^2 + sigma_int^2 + Z_W^2 * tilde_sigma^2)
        L_marg  = N(m; M_pred([O/H]_*) + mu, sigma_1)
                  * N(OH_obs; mu_OH, sqrt(eps^2 + sigma_OH^2))

    Per star i in campaign c::

        d_i | ell_i, b_i    ~ DiskPrior(ell, b; R_d, z_d, R_sun)
        mu_i                = 5 log10(d_i) + 10
        m^W_{H,obs,i}       ~ N(M_pred + mu_i, sigma_1)
        pi_{obs,i}          ~ N(1/d_i - delta_pi, f_pi * sigma_{pi,i})

    Truncated likelihood (analytical selection normalisation)::

        log p += sum_i log P(S_i | obs_i, theta)
               - N_c * log int Phi_K(h(d); R) p(d|Omega) dOmega dd

    Anchor galaxy a (optional)::

        mu_a       ~ Uniform(mu_min, mu_max)   + volume factor (3 ln10/5) mu
        mu_geo_a   ~ N(mu_a, sigma_mu)
        m^W_{H,a}  ~ MVN(M_pred([O/H]_*) + mu_a,
                         Sigma_a + Z_W^2 tilde_sigma^2 I)
    """

    def __init__(self, config, data):
        self._init_setup(config, data)

    def __call__(self):
        if self.model_type == "R21":
            self._model_R21()
        else:
            self._model_forward()

    def _model_forward(self):
        """Forward model with per-star distance sampling."""
        # --- Sample global P-L parameters ---
        params = self._sample_global_params()

        # --- Sample/fix selection thresholds and widths ---
        params.update(sample_selection_params(
            self.sel_c22, self.sel_c27, self.campaigns))

        # --- Per-campaign likelihood + selection correction ---
        for campaign, data in self.data.items():
            d = self._likelihood_campaign(campaign, data, params)
            selection_correction(
                campaign, d, data, params,
                self.sel_mc_data.get(campaign),
                self.sel_c22, self.sel_c27, self.apply_spiral_arms,
                marg_d=self.marginalise_distance)

        # --- Anchor galaxies ---
        for name, anchor in self.anchor_data.items():
            self._likelihood_anchor(name, anchor, params)

    def _model_R21(self):
        """Riess et al. (2021) parallax-space model.

        Fits in parallax space: computes photometric parallax from observed
        magnitude and P-L relation, then compares to observed EDR3 parallax.
        """
        params = {
            "M_H_1": sample_prior("M_H_1", self.priors["M_H_1"]),
            "b_W": sample_prior("b_W", self.priors["b_W"]),
            "Z_W": sample_prior("Z_W", self.priors["Z_W"]),
        }
        zp = sample_prior("delta_pi", self.priors["delta_pi"])

        if self.use_Q:
            c_W = sample_prior("c_W", self.priors["c_W"])

        ln10_02 = 0.2 * jnp.log(10.0)

        for campaign, data in self.data.items():
            M_pred = self._compute_M_pred(params, data)

            if self.use_Q:
                M_pred = M_pred + c_W * data.Q

            # Distance modulus from observed magnitude
            mu_obs = data.mW_H - M_pred

            # Photometric parallax
            pi_phot = jnp.power(10.0, -0.2 * (mu_obs - 10.0))

            # R21 error model (Q_err absorbed into sigma_int, like OH_err)
            sigma_mag_total = jnp.sqrt(
                data.mW_H_err**2 + self.r21_sigma_int**2)
            sigma_pi_phot = ln10_02 * pi_phot * sigma_mag_total
            sigma_pi_edr3 = self.r21_pi_err_inflation * data.pi_EDR3_err
            pi_std = jnp.sqrt(sigma_pi_phot**2 + sigma_pi_edr3**2)

            numpyro.factor(
                f"ll_mw_{campaign}",
                _chi2_loglike(data.pi_EDR3 - pi_phot + zp, pi_std))

        # --- Anchor galaxies ---
        for name, anchor in self.anchor_data.items():
            mu_anc = numpyro.sample(
                f"mu_{name}",
                dist.Uniform(anchor.mu_min, anchor.mu_max))
            numpyro.factor(
                f"vol_prior_{name}", (3 * jnp.log(10) / 5) * mu_anc)

            numpyro.sample(
                f"mu_geo_{name}",
                dist.Normal(mu_anc, anchor.e_mu_anchor),
                obs=anchor.mu_anchor)

            M_pred = self._compute_M_pred(params, anchor)
            if self.use_Q and anchor.Q is not None:
                M_pred = M_pred + c_W * anchor.Q

            mW_pred = M_pred + mu_anc
            residual = anchor.mW_H - mW_pred
            lam = anchor.eig_lam + self.r21_sigma_int**2
            v = anchor.eig_Q.T @ residual
            numpyro.factor(f"ll_{name}",
                           _chi2_loglike(v, jnp.sqrt(lam)))

    def _sample_global_params(self):
        """Sample global P-L relation parameters."""
        params = {
            "M_H_1": sample_prior("M_H_1", self.priors["M_H_1"]),
            "b_W": sample_prior("b_W", self.priors["b_W"]),
            "Z_W": sample_prior("Z_W", self.priors["Z_W"]),
            "delta_pi": sample_prior("delta_pi", self.priors["delta_pi"]),
            "f_pi": sample_prior("f_pi", self.priors["f_pi"]),
        }

        # Intrinsic scatter: shared or per-campaign
        if self.shared_scatter:
            params["sigma_int"] = sample_prior(
                "sigma_int", self.priors["sigma_int"])
        else:
            for campaign in self.campaigns:
                key = f"sigma_int_{campaign}"
                params[key] = sample_prior(key, self.priors["sigma_int"])

        # Per-population hyperparameters (period + metallicity)
        # Looks up population-specific prior (e.g. mu_logP_LMC) first,
        # falls back to the generic prior (e.g. mu_logP).
        populations = list(self.campaigns) + list(self.anchor_data.keys())
        for pop in populations:
            params[f"mu_logP_{pop}"] = sample_prior(
                f"mu_logP_{pop}",
                self.priors.get(f"mu_logP_{pop}", self.priors["mu_logP"]))
            params[f"sigma_logP_{pop}"] = sample_prior(
                f"sigma_logP_{pop}",
                self.priors.get(f"sigma_logP_{pop}",
                                self.priors["sigma_logP"]))

            # LMC metallicities are assigned (two discrete values, not
            # spectroscopic), so we use them as fixed covariates.
            if pop == "LMC":
                continue
            params[f"mu_OH_{pop}"] = sample_prior(
                f"mu_OH_{pop}",
                self.priors.get(f"mu_OH_{pop}", self.priors["mu_OH"]))
            params[f"sigma_OH_{pop}"] = sample_prior(
                f"sigma_OH_{pop}",
                self.priors.get(f"sigma_OH_{pop}", self.priors["sigma_OH"]))

        if self.use_Q:
            params["c_W"] = sample_prior("c_W", self.priors["c_W"])
            q_pops = list(self.campaigns) + [
                name for name, anc in self.anchor_data.items()
                if anc.Q is not None]
            for pop in q_pops:
                params[f"mu_Q_{pop}"] = sample_prior(
                    f"mu_Q_{pop}",
                    self.priors.get(f"mu_Q_{pop}", self.priors["mu_Q"]))
                params[f"sigma_Q_{pop}"] = sample_prior(
                    f"sigma_Q_{pop}",
                    self.priors.get(f"sigma_Q_{pop}",
                                    self.priors["sigma_Q"]))

        if self.apply_spiral_arms:
            params["spiral_arm_frac"] = sample_prior(
                "spiral_arm_frac", self.priors["spiral_arm_frac"])
            params["spiral_width"] = sample_prior(
                "spiral_width", self.priors["spiral_width"])

        if self.infer_anchor_scatter and not self.shared_scatter:
            params["sigma_int_anchor"] = sample_prior(
                "sigma_int_anchor", self.priors["sigma_int_anchor"])

        return params

    def _compute_M_pred(self, params, data):
        """Predicted absolute magnitude at raw observed [O/H] (R21 only)."""
        return (params["M_H_1"] + params["b_W"] * (data.logP - 1.0)
                + params["Z_W"] * data.OH)

    def _get_distance_prior(self, campaign, data):
        """Return the numpyro distribution for the distance prior."""
        d_min = self.d_bounds[campaign]["d_min"]
        d_max = self.d_bounds[campaign]["d_max"]

        if self.distance_prior == "disk":
            if data.ell is None or data.b is None:
                raise ValueError(
                    "Disk prior requires Galactic coordinates (ell, b).")
            return DiskPrior(data.ell, data.b, d_min, d_max,
                             R_d=self.disk_R_d, z_d=self.disk_z_d,
                             R_sun=self.disk_R_sun)
        elif self.distance_prior == "distance_modulus":
            return DistanceModulusPrior(d_min, d_max)
        else:
            return dist.Uniform(d_min, d_max)

    def _likelihood_campaign(self, campaign, data, params):
        """Evaluate per-star likelihood for a MW campaign.

        Marginalises over latent [O/H] analytically: uses effective
        metallicity [O/H]_*, inflated magnitude variance sigma_1, and
        per-star metallicity marginalisation factors (Eqs. 16-18).
        The delta-function period likelihood collapses the logP integral,
        leaving a population prior factor N(logP_obs; mu_logP, sigma_logP).

        When ``self.marginalise_distance`` is True, per-star distances are
        integrated out on a fixed grid (Simpson's rule) instead of being
        sampled as NUTS latent variables.
        """
        sigma_int = get_named_or_shared("sigma_int", campaign, params)

        OH_star, tilde_sigma_OH_sq, log_OH_factor = _marginalise_latent(
            data.OH, params[f"mu_OH_{campaign}"],
            params[f"sigma_OH_{campaign}"], self.epsilon_OH)

        M_pred = (params["M_H_1"]
                  + params["b_W"] * (data.logP - 1.0)
                  + params["Z_W"] * OH_star)

        sigma_1_sq = (data.mW_H_err**2 + sigma_int**2
                      + params["Z_W"]**2 * tilde_sigma_OH_sq)

        if self.use_Q:
            Q_star, tilde_sigma_Q_sq, log_Q_factor = _marginalise_latent(
                data.Q, params[f"mu_Q_{campaign}"],
                params[f"sigma_Q_{campaign}"], data.Q_err)
            M_pred = M_pred + params["c_W"] * Q_star
            sigma_1_sq = sigma_1_sq + params["c_W"]**2 * tilde_sigma_Q_sq

        sigma_1 = jnp.sqrt(sigma_1_sq)

        if self.marginalise_distance:
            d = self._likelihood_campaign_marg_d(
                campaign, data, params, M_pred, sigma_1)
        else:
            with numpyro.plate(f"stars_{campaign}", data.n_stars):
                d_prior = self._get_distance_prior(campaign, data)
                d = numpyro.sample(f"d_{campaign}", d_prior)
                mu = 5.0 * jnp.log10(d) + 10.0

                # Wesenheit magnitude likelihood (inflated variance)
                mW_pred = M_pred + mu
                numpyro.sample(
                    f"mW_H_obs_{campaign}", dist.Normal(mW_pred, sigma_1),
                    obs=data.mW_H)

                # Parallax likelihood
                pi_model = 1.0 / d - params["delta_pi"]
                numpyro.sample(
                    f"pi_obs_{campaign}",
                    dist.Normal(pi_model,
                                params["f_pi"] * data.pi_EDR3_err),
                    obs=data.pi_EDR3)

        # --- Shared factors (both paths) ---
        numpyro.factor(f"OH_marg_{campaign}", jnp.sum(log_OH_factor))

        if self.use_Q:
            numpyro.factor(f"Q_marg_{campaign}", jnp.sum(log_Q_factor))

        # Period population prior factor (delta-function collapse of logP)
        log_logP_factor = dist.Normal(
            params[f"mu_logP_{campaign}"],
            params[f"sigma_logP_{campaign}"]).log_prob(data.logP)
        numpyro.factor(f"logP_prior_{campaign}", jnp.sum(log_logP_factor))

        # Spiral arm correction (sampling path only; when marginalising,
        # the spiral factor is inside the distance integral).
        #
        # We add log spiral(d_i) to each star's distance log-prob.
        # The DiskPrior.log_prob already contributes
        # (disk_unnorm - log Z_disk), so the total becomes
        # (disk_unnorm + spiral - log Z_disk).  No renormalisation by
        # Z_full = int(disk * spiral) dr is needed: Z(u_i) cancels
        # with the angular prior pi(u_i) = Z(u_i) / Z_total, and
        # Z_total cancels with the selection normalisation (see
        # _likelihood_campaign_marg_d and _get_prior_and_norm).
        if not self.marginalise_distance:
            obs_dsq = self.spiral_obs_dist_sq_per_arm
            if self.apply_spiral_arms and campaign in obs_dsq:
                f_arm = params["spiral_arm_frac"]
                sigma = params["spiral_width"]

                obs_dsq_camp = obs_dsq[campaign]
                d_grid_sp = self.spiral_obs_d_grid[campaign]
                spiral_profiles = spiral_log_factor(
                    obs_dsq_camp, f_arm, sigma)

                spiral_at_d = jax.vmap(
                    lambda profile, di: jnp.interp(
                        di, d_grid_sp, profile)
                )(spiral_profiles, d)

                numpyro.factor(f"spiral_{campaign}", jnp.sum(spiral_at_d))

        return d

    def _likelihood_campaign_marg_d(self, campaign, data, params,
                                    M_pred, sigma_1):
        """Distance-marginalised likelihood for a MW campaign.

        Integrates over distance on a fixed grid using Simpson's rule.
        Spiral arm modulation (when active) is folded into the distance
        prior.

        The distance prior is intentionally left **unnormalised**: for
        star i at sky position u_i, the per-direction normalisation
        Z(u_i) = int p_unnorm(r, u_i) dr cancels exactly with the
        angular prior pi(u_i) = Z(u_i) / Z_total.  Concretely::

            log f_i = log int L_i p_unnorm dr          [unnorm radial]
                    - log Z(u_i) + log Z(u_i)          [cancel]
                    - log Z_total                       [absorbed into S]

        So we compute log int L_i p_unnorm dr directly, never dividing
        by Z(u_i).  The Z_total factor cancels with the same factor in
        the selection normalisation S_unnorm (see ``_mc_average``).
        This avoids a per-iteration ln_simpson_uniform call per star
        and is exact regardless of whether spirals are active.
        """
        d_grid = self.d_grid_marg[campaign]
        dx = self._obs_dx[campaign]
        log_disk_prior = self._obs_disk_log_prior[campaign]

        # Spiral arm modulation (unnormalised — see docstring)
        if (self.apply_spiral_arms
                and campaign in self._obs_spiral_dsq_on_marg_grid):
            f_arm = params["spiral_arm_frac"]
            sigma_sp = params["spiral_width"]
            spiral_profiles = spiral_log_factor(
                self._obs_spiral_dsq_on_marg_grid[campaign], f_arm, sigma_sp)
            log_prior = log_disk_prior + spiral_profiles
        else:
            log_prior = log_disk_prior

        # A_H extinction selection (C22 only, when active)
        AH_profiles = AH_valid = AH_max = AH_width = None
        if (campaign == "C22" and self.sel_c22.AH.apply
                and hasattr(self, 'AH_obs_profiles_marg')):
            AH_profiles = self.AH_obs_profiles_marg
            AH_valid = self.AH_obs_star_valid
            AH_max = params["AH_max_C22"]
            AH_width = self.sel_c22.AH.width

        # Distance-marginalised log-likelihood (chi2 terms only)
        log_like = log_likelihood_marg_distance(
            data.mW_H, data.pi_EDR3, data.pi_EDR3_err,
            params["delta_pi"], params["f_pi"],
            d_grid, dx, log_prior,
            M_pred, sigma_1,
            AH_profiles=AH_profiles, AH_valid=AH_valid,
            AH_max=AH_max, AH_width=AH_width)

        # Gaussian normalisation terms (d-independent, param-dependent):
        # -log(sigma_1) from the magnitude PDF, and
        # -log(f_pi * sigma_pi) from the parallax PDF.
        log_like = (log_like
                    - jnp.log(sigma_1)
                    - jnp.log(params["f_pi"] * data.pi_EDR3_err))

        numpyro.factor(f"ll_marg_d_{campaign}", jnp.sum(log_like))

        return None

    def _likelihood_anchor(self, name, anchor, params):
        """Evaluate anchor galaxy likelihood.

        Marginalises over latent [O/H] (except LMC, where assigned [O/H]
        values are used as fixed covariates) and applies a period selection
        correction for the logP > -0.3 truncation.
        """
        mu = numpyro.sample(
            f"mu_{name}",
            dist.Uniform(anchor.mu_min, anchor.mu_max))
        numpyro.factor(f"vol_prior_{name}", (3 * jnp.log(10) / 5) * mu)

        numpyro.sample(
            f"mu_geo_{name}",
            dist.Normal(mu, anchor.e_mu_anchor),
            obs=anchor.mu_anchor)

        if name == "LMC":
            # LMC metallicities are assigned (two discrete values, not
            # spectroscopic), so we use them as fixed covariates.
            OH_for_pred = anchor.OH
            alpha = 0.0
            ll_OH = 0.0
        else:
            OH_star, tilde_sigma_OH_sq, log_OH_factor = _marginalise_latent(
                anchor.OH, params[f"mu_OH_{name}"],
                params[f"sigma_OH_{name}"], self.epsilon_OH)
            OH_for_pred = OH_star
            alpha = params["Z_W"]**2 * tilde_sigma_OH_sq
            ll_OH = jnp.sum(log_OH_factor)

        M_pred = (params["M_H_1"]
                  + params["b_W"] * (anchor.logP - 1.0)
                  + params["Z_W"] * OH_for_pred)

        # Q index marginalisation (only for anchors with Q data).
        # Q_err is per-star so tilde_sigma_Q_sq is a per-star array,
        # requiring a non-uniform diagonal shift (Cholesky instead of
        # the eigendecomposition shortcut).
        ll_Q = 0.0
        diag_Q = None
        if self.use_Q and anchor.Q is not None:
            Q_star, tilde_sigma_Q_sq, log_Q_factor = _marginalise_latent(
                anchor.Q, params[f"mu_Q_{name}"],
                params[f"sigma_Q_{name}"], anchor.Q_err)
            M_pred = M_pred + params["c_W"] * Q_star
            diag_Q = params["c_W"]**2 * tilde_sigma_Q_sq
            ll_Q = jnp.sum(log_Q_factor)

        mW_pred = M_pred + mu

        if self.infer_anchor_scatter:
            if self.shared_scatter:
                alpha = alpha + params["sigma_int"]**2
            else:
                alpha = alpha + params["sigma_int_anchor"]**2

        n = anchor.n_stars
        residual = anchor.mW_H - mW_pred

        if diag_Q is not None:
            # Per-star diagonal from Q: fall back to Cholesky
            diag_shift = alpha + diag_Q
            covmat_shifted = anchor.covmat + jnp.diag(diag_shift)
            L = jnp.linalg.cholesky(covmat_shifted)
            v = jax.scipy.linalg.solve_triangular(L, residual, lower=True)
            ll = -0.5 * (n * jnp.log(2 * jnp.pi)
                         + 2 * jnp.sum(jnp.log(jnp.diag(L)))
                         + jnp.sum(v**2))
        else:
            # Scalar diagonal shift: use precomputed eigendecomposition
            lam = anchor.eig_lam + alpha
            v = anchor.eig_Q.T @ residual
            ll = -0.5 * (n * jnp.log(2 * jnp.pi)
                         + jnp.sum(jnp.log(lam))
                         + jnp.sum(v**2 / lam))

        numpyro.factor(f"ll_{name}", ll + ll_OH + ll_Q)

        # Period population prior factor
        mu_logP = params[f"mu_logP_{name}"]
        sigma_logP = params[f"sigma_logP_{name}"]
        log_logP_factor = dist.Normal(mu_logP, sigma_logP).log_prob(
            anchor.logP)
        numpyro.factor(f"logP_prior_{name}", jnp.sum(log_logP_factor))

        # Period selection: anchors are truncated at logP > logP_min.
        # All observed stars pass, so only the normalisation contributes.
        if anchor.logP_min is not None:
            log_sel_norm = log_ndtr(
                (mu_logP - anchor.logP_min) / sigma_logP)
            numpyro.factor(f"logP_sel_{name}", -n * log_sel_norm)
