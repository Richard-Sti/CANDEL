# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Minimal NumPyro model for CCHP TRGB distance calibrators to infer H0.
"""
from abc import ABC

import jax.numpy as jnp
from numpyro import factor, plate, sample
from numpyro.distributions import Normal, Uniform

from ..cosmography import (Distance2Distmod, Distance2Redshift,
                           Distmod2Distance, Distmod2Redshift,
                           LogGrad_Distmod2ComovingDistance)
from ..util import (fprint, get_nested, load_config, radec_to_cartesian,
                    replace_prior_with_delta)
from .interp import LOSInterpolator
from .model import (load_priors, lp_galaxy_bias, predict_cz, rsample,
                    sample_galaxy_bias)
from .model_SH0ES import logmeanexp
from .simpson import ln_simpson


class BaseCCHPModel(ABC):
    """
    Base class for the CCHP model, providing common functionality and
    configuration loading.
    """

    def __init__(self, config_path, data):
        config = load_config(config_path, replace_los_prior=False)

        # Set unused parameter priors to delta functions.
        config = self.replace_priors(config)

        # Unpack and set the priors.
        priors = config["model"]["priors"]
        self.priors, self.prior_dist_name = load_priors(priors)

        self.config = config
        self.set_data(data)

        # Initialize the interpolators
        self.Om = get_nested(config, "model/Om", 0.3)
        self.distmod2distance = Distmod2Distance(Om0=self.Om)
        self.distmod2redshift = Distmod2Redshift(Om0=self.Om)
        self.distance2distmod_scalar = Distance2Distmod(
            Om0=self.Om, is_scalar=True)
        self.distance2distmod = Distance2Distmod(Om0=self.Om)
        self.distance2redshift = Distance2Redshift(Om0=self.Om)
        self.log_grad_distmod2comoving_distance = LogGrad_Distmod2ComovingDistance(Om0=self.Om)  # noqa

        self.num_hosts = self.mag_obs.shape[0]

        # Distance modulus limits for sampling
        self.distmod_limits = config.get("model", {}).get(
            "distmod_limits", [25.0, 35.0])
        fprint(f"distmod_limits set to {self.distmod_limits}")
        self.distmod_limits_LMC = config["model"].get(
            "distmod_limits_LMC", self.distmod_limits)
        self.distmod_limits_N4258 = config["model"].get(
            "distmod_limits_N4258", self.distmod_limits)
        self.has_host_los = False
        self.num_fields = 1
        self.use_reconstruction = get_nested(
            config, "model/use_reconstruction", False)
        self.which_bias = get_nested(config, "model/galaxy_bias", "linear")
        if self.use_reconstruction:
            recon_name = get_nested(config, "io/which_host_los", "unspecified")
            fprint(f"Using reconstruction: {recon_name}")
        fprint(f"galaxy_bias set to {self.which_bias}")

        r0_decay_scale = get_nested(config, "io/los_r0_decay_scale", 5.0)
        fprint(f"los_r0_decay_scale set to {r0_decay_scale}")
        if "host_los_velocity" in data and "host_los_r" in data:
            self.has_host_los = True
            self.num_fields = data["host_los_velocity"].shape[0]
            if "host_los_density" in data:
                los_delta = jnp.asarray(data["host_los_density"]) - 1.0
                self.f_host_los_delta = LOSInterpolator(
                    data["host_los_r"],
                    los_delta,
                    r0_decay_scale=r0_decay_scale,
                    extrap_constant=0.0,
                )
            self.f_host_los_velocity = LOSInterpolator(
                data["host_los_r"],
                jnp.asarray(data["host_los_velocity"]),
                r0_decay_scale=r0_decay_scale,
                extrap_constant=0.0,
            )
        if self.use_reconstruction and not self.has_host_los:
            raise ValueError(
                "`use_reconstruction = True` but no host LOS data provided.")

    def replace_priors(self, config):
        """Replace priors on parameters that are not used in the model."""
        if not get_nested(config, "model/use_reconstruction", False):
            replace_prior_with_delta(config, "beta", 0.0, verbose=False)
        return config

    def set_data(self, data):
        """Convert data to JAX arrays and set as attributes."""
        self.mag_obs = jnp.asarray(data["mag_TRGB"])
        self.e_mag_TRGB = jnp.asarray(data["e_mag_TRGB"])
        self.cz_cmb = jnp.asarray(data["cz_cmb"])
        self.e_czcmb = jnp.asarray(data["e_czcmb"])

        # Convert RA/Dec to Cartesian coordinates
        fprint("Converting host RA/dec to Cartesian coordinates.")
        rhat = radec_to_cartesian(data["RA"], data["DEC"])
        n = jnp.linalg.norm(rhat, axis=1, keepdims=True)
        self.rhat_host = rhat / jnp.where(n == 0.0, 1.0, n)

        self.mu_LMC_anchor = data["mu_LMC_anchor"]
        self.e_mu_LMC_anchor = data["e_mu_LMC_anchor"]
        self.mag_LMC_TRGB = data["mag_LMC_TRGB"]
        self.e_mag_LMC_TRGB = data["e_mag_LMC_TRGB"]
        self.mu_N4258_anchor = data["mu_N4258_anchor"]
        self.e_mu_N4258_anchor = data["e_mu_N4258_anchor"]
        self.mag_N4258_TRGB = data["mag_N4258_TRGB"]
        self.e_mag_N4258_TRGB = data["e_mag_N4258_TRGB"]

    def sample_host_distmod(self):
        """
        Sample distance moduli for host galaxies with a uniform prior in
        distance modulus. The r^2 volume prior is added via factor.
        """
        dist = Uniform(*self.distmod_limits)

        with plate("hosts", self.num_hosts):
            mu_host = sample("mu_host", dist)

        # Anchors (LMC, NGC 4258) with custom limits if provided
        mu_LMC = sample("mu_LMC", Uniform(*self.distmod_limits_LMC))
        mu_N4258 = sample("mu_N4258", Uniform(*self.distmod_limits_N4258))

        # Anchor likelihoods
        sample("mu_LMC_ll",
               Normal(mu_LMC, self.e_mu_LMC_anchor),
               obs=self.mu_LMC_anchor)
        sample("mu_N4258_ll",
               Normal(mu_N4258, self.e_mu_N4258_anchor),
               obs=self.mu_N4258_anchor)

        return mu_host, mu_LMC, mu_N4258


class CCHPModel(BaseCCHPModel):
    """
    Forward model for TRGB distance moduli versus cz to infer H0.

    Assumes inputs from ``load_CCHP_from_config``:
    mag_TRGB (apparent, already mu + 4.336), e_mag_TRGB, cz_cmb, e_czcmb.
    """

    def __init__(self, config_path, data):
        super().__init__(config_path, data)

    def __call__(self, shared_params=None):
        H0 = rsample("H0", self.priors["H0"], shared_params)
        M_TRGB = rsample("M_TRGB", self.priors["M_TRGB"], shared_params)
        sigma_int = rsample("sigma_int", self.priors["sigma_int"],
                            shared_params)
        sigma_v = rsample("sigma_v", self.priors["sigma_v"], shared_params)
        beta = rsample("beta", self.priors["beta"], shared_params)
        Vext = rsample("Vext", self.priors["Vext"], shared_params)

        h = H0 / 100.0
        # Sample distance moduli uniformly
        mu_host, mu_LMC, mu_N4258 = self.sample_host_distmod()
        bias_params = sample_galaxy_bias(
            self.priors, self.which_bias, beta=beta, Om=self.Om)

        # Convert to comoving distances in Mpc
        r = self.distmod2distance(mu_host, h=h)

        # Volume prior via Jacobian term in distance modulus space.
        lp_host_dist = self.log_grad_distmod2comoving_distance(mu_host, h=h)
        if self.use_reconstruction:
            rh_host = r * h
            los_delta_host = self.f_host_los_delta(rh_host)
            lp_host_dist += lp_galaxy_bias(
                los_delta_host, jnp.log(1.0 + los_delta_host),
                bias_params, self.which_bias)

            # Normalize the biased prior over the LOS grid (fields x hosts).
            r_grid_h = self.f_host_los_delta.los_r
            r_grid = r_grid_h / h
            mu_grid = self.distance2distmod_scalar(r_grid, h=h)
            lp_grid = self.log_grad_distmod2comoving_distance(mu_grid, h=h)
            los_delta_grid = self.f_host_los_delta.interp_many_steps_per_galaxy(
                r_grid_h)
            lp_grid += lp_galaxy_bias(
                los_delta_grid, jnp.log(1.0 + los_delta_grid),
                bias_params, self.which_bias)
            lp_grid_norm = ln_simpson(
                lp_grid, x=r_grid[None, None, :], axis=-1)
            lp_host_dist = lp_host_dist - lp_grid_norm

        factor("lp_host_dist", lp_host_dist)

        # Convert distance to cosmological redshift
        z_cosmo = self.distance2redshift(r, h=h)

        # Project Vext along LOS
        Vext_rad_host = jnp.sum(Vext[None, :] * self.rhat_host, axis=1)

        # Get the predicted redshifts, (n_field, n_gal)
        if self.use_reconstruction:
            Vpec = self.f_host_los_velocity(r)
            cz_th_all = predict_cz(
                z_cosmo[None, :], Vext_rad_host[None, :] + beta * Vpec)
        else:
            cz_th_all = predict_cz(z_cosmo[None, :], Vext_rad_host[None, :])

        sigma_tot_mag = jnp.sqrt(self.e_mag_TRGB**2 + sigma_int**2)
        sigma_tot_cz = jnp.sqrt(self.e_czcmb**2 + sigma_v**2)

        # Magnitude likelihood
        with plate("hosts", self.num_hosts):
            sample("mag_TRGB_ll", Normal(mu_host + M_TRGB, sigma_tot_mag),
                   obs=self.mag_obs)

        if self.use_reconstruction and self.has_host_los:
            logp = logmeanexp(
                Normal(cz_th_all, sigma_tot_cz).log_prob(self.cz_cmb),
                axis=0)
            with plate("hosts_cz", self.num_hosts):
                factor("cz_ll", logp)
        else:
            with plate("hosts_cz", self.num_hosts):
                sample("cz_ll", Normal(cz_th_all[0], sigma_tot_cz),
                       obs=self.cz_cmb)

        # Anchor TRGB magnitudes
        sample("mag_LMC_TRGB_ll",
               Normal(M_TRGB + mu_LMC, self.e_mag_LMC_TRGB),
               obs=self.mag_LMC_TRGB)
        sample("mag_N4258_TRGB_ll",
               Normal(M_TRGB + mu_N4258, self.e_mag_N4258_TRGB),
               obs=self.mag_N4258_TRGB)
