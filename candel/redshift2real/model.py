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
Module for mapping observed redshift to cosmological redshift given some
calibrated density and velocity field.
"""
from abc import ABC, abstractmethod

from jax import numpy as jnp, random as jrandom
from jax.scipy.special import logsumexp
from numpyro import factor, plate, sample
from numpyro.diagnostics import gelman_rubin
from numpyro.distributions import Normal, Uniform
from numpyro.infer import MCMC, NUTS, init_to_median
from tqdm import tqdm

from ..cosmography import Distance2Redshift
from ..model import LOSInterpolator, ln_simpson, lp_galaxy_bias
from ..util import SPEED_OF_LIGHT, fprint, radec_to_cartesian


def log_mean_exp(logp, axis=-1):
    return logsumexp(logp, axis=axis) - jnp.log(logp.shape[axis])


class BaseRedshift2Real(ABC):
    """Base class for all models. """

    def __init__(self, RA, dec, zcmb, los_r, los_density, los_velocity,
                 which_bias, calibration_samples, Rmin=1e-7, Rmax=300,
                 num_rgrid=101, r0_decay_scale=5, Om0=0.3, verbose=True):
        self.verbose = verbose
        self.dist2redshift = Distance2Redshift(Om0=Om0)

        self.Rmin = Rmin
        self.Rmax = Rmax
        assert Rmin > 0 and Rmax > Rmin
        assert num_rgrid % 2 == 1

        self.len_input_data = len(zcmb)
        self.cz_cmb = jnp.asarray(zcmb * SPEED_OF_LIGHT)  # in km/s

        los_r = jnp.asarray(los_r)
        los_density = jnp.asarray(los_density)
        los_velocity = jnp.asarray(los_velocity)

        self.f_los_velocity = LOSInterpolator(
            los_r, los_velocity, r0_decay_scale=r0_decay_scale)

        rhat = radec_to_cartesian(RA, dec)

        self.los_grid_r = jnp.linspace(Rmin, Rmax, num_rgrid)

        self.calibration_samples = calibration_samples
        calibration_keys = list(calibration_samples.keys())
        self.num_cal = calibration_samples[calibration_keys[0]].shape[0]

        # LOS Vext, (ngal, ncalibration_samples)
        if "Vext" in calibration_samples:
            self.Vext_radial = jnp.sum(
                rhat[:, None, :] * calibration_samples["Vext"][None, :, :],
                axis=-1)
        else:
            fprint("No Vext in calibration samples.", verbose=self.verbose)
            self.Vext_radial = jnp.zeros(
                (self.len_input_data, self.num_cal))

        self.use_im = True
        self.which_bias = which_bias
        if which_bias is None:
            self.use_im = False
        elif which_bias == "linear":
            self.b1 = jnp.asarray(calibration_samples["b1"])
            self.f_los_delta = LOSInterpolator(
                los_r, los_density - 1, r0_decay_scale=r0_decay_scale)
            self.lp_norm = self.compute_linear_bias_lp_normalization(
                self.los_grid_r)
        elif which_bias == "double_powerlaw":
            self.alpha_low = jnp.asarray(calibration_samples["alpha_low"])
            self.alpha_high = jnp.asarray(calibration_samples["alpha_high"])
            self.log_rho_t = jnp.asarray(calibration_samples["log_rho_t"])
            self.f_los_log_density = LOSInterpolator(
                los_r, jnp.log(los_density), r0_decay_scale=r0_decay_scale)

            self.lp_norm = self.compute_double_powerlaw_bias_lp_normalization(
                self.los_grid_r)
        else:
            raise ValueError(f"Unknown bias model: {which_bias}")

        self.sigma_v = jnp.asarray(calibration_samples["sigma_v"])
        if "beta" in calibration_samples:
            self.beta = jnp.asarray(calibration_samples["beta"])
        else:
            fprint("Beta not in calibration samples. Setting beta=1.",
                   verbose=self.verbose)
            self.beta = jnp.ones_like(self.sigma_v)

        fprint(f"Loaded {self.len_input_data} objects and "
               f"{self.num_cal} calibration samples.", verbose=self.verbose)

        self.print_sample_stats()

    def print_sample_stats(self):
        if not self.verbose:
            return
        print("Calibration sample statistics:")
        print(f"sigma_v : {jnp.mean(self.sigma_v):.3f} +- {jnp.std(self.sigma_v):.3f}")  # noqa
        print(f"beta    : {jnp.mean(self.beta):.3f} +- {jnp.std(self.beta):.3f}")        # noqa
        if self.which_bias == "linear":
            print(f"b1      : {jnp.mean(self.b1):.3f} +- {jnp.std(self.b1):.3f}")          # noqa
        elif self.which_bias == "double_powerlaw":
            print(f"alpha_low  : {jnp.mean(self.alpha_low):.3f} +- {jnp.std(self.alpha_low):.3f}")      # noqa
            print(f"alpha_high : {jnp.mean(self.alpha_high):.3f} +- {jnp.std(self.alpha_high):.3f}")    # noqa
            print(f"log_rho_t  : {jnp.mean(self.log_rho_t):.3f} +- {jnp.std(self.log_rho_t):.3f}")      # noqa

    def compute_linear_bias_lp_normalization(self, los_grid_r):
        """
        Compute the normalization of the linear bias term in the distance
        prior.
        """
        if self.verbose:
            print("Computing `linear` bias lp normalization...")
        # Density constrast along the LOS, (nfield, ngal, nrad_steps)
        los_grid_delta = self.f_los_delta.interp_many_steps_per_galaxy(
            los_grid_r)
        bias_params = [self.b1[None, None, :, None],]

        intg = lp_galaxy_bias(
            los_grid_delta[:, :, None, :], None, bias_params,
            galaxy_bias="linear")
        return ln_simpson(intg, los_grid_r[None, None, None, :], axis=-1)

    def compute_double_powerlaw_bias_lp_normalization(self, los_grid_r):
        """
        Compute the normalization of the double powerlaw bias term in the
        distance prior.
        """
        if self.verbose:
            print("Computing `double_powerlaw` bias lp normalization...")
        los_log_density = self.f_los_log_density.interp_many_steps_per_galaxy(
            los_grid_r)
        bias_params = [self.alpha_low[None, None, :, None],
                       self.alpha_high[None, None, :, None],
                       self.log_rho_t[None, None, :, None],
                       ]
        intg = lp_galaxy_bias(
            None, los_log_density[:, :, None, :], bias_params,
            galaxy_bias="double_powerlaw")
        return ln_simpson(intg, los_grid_r[None, None, None, :], axis=-1)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Redshift2Real(BaseRedshift2Real):
    """
    A model for mapping observed redshift to cosmological redshift.
    """

    def __call__(self, ):
        """
        `p(z_cosmo | z_CMB, calibration)` for a single object.
        """
        with plate("data", self.len_input_data):
            r = sample("r", Uniform(self.Rmin, self.Rmax))

        # Homogeneous Malmquist bias term, shape (ngal,)
        lp_r = 2 * jnp.log(r)

        if self.which_bias == "linear":
            bias_params = [self.b1[None, None, :],]
            lp_r_bias = lp_galaxy_bias(
                self.f_los_delta(r)[..., None], None, bias_params,
                galaxy_bias="linear")
            lp_r = lp_r[None, :, None] + lp_r_bias - self.lp_norm
        elif self.which_bias == "double_powerlaw":
            bias_params = [
                self.alpha_low[None, None, :],
                self.alpha_high[None, None, :],
                self.log_rho_t[None, None, :],
                ]
            lp_r_bias = lp_galaxy_bias(
                None, self.f_los_log_density(r)[..., None],
                bias_params,
                galaxy_bias="double_powerlaw")
            lp_r = lp_r[None, :, None] + lp_r_bias - self.lp_norm
        else:
            lp_r = lp_r[None, :, None]

        # Cosmological redshift, shape (ngal,)
        zcosmo = self.dist2redshift(r)

        # Peculiar velocity from the reconstruction, (nfield, ngal,)
        Vpec = self.f_los_velocity(r)

        # Peculiar velocity redshift (nfield, ngal, ncalibration_samples)
        zpec = (
            self.beta[None, None, :] * Vpec[..., None]
            + self.Vext_radial[None, ...]) / SPEED_OF_LIGHT
        # Predicted redshift (nfield, ngal, ncalibration_samples)
        cz_pred = SPEED_OF_LIGHT * (
            (1 + zcosmo)[None, :, None] * (1 + zpec) - 1)

        ll = Normal(cz_pred, self.sigma_v[None, None, :],).log_prob(
            self.cz_cmb[None, :, None])
        ll += lp_r

        # Average over the calibration samples, shape (nfield, ngal)
        ll = log_mean_exp(ll, axis=-1) - jnp.log(ll.shape[-1])
        # Average over the density and velocity field samples
        factor(
            "log_density",
            log_mean_exp(ll, axis=0) - jnp.log(ll.shape[0])
            )


def run_batched_inference(model_kwargs, batch_size=50, num_warmup=500,
                          num_samples=2000, num_chains=1, rng_seed=0,
                          progress_bar=False, galaxy_idx=None):
    """
    Run batched inference for redshift2real model.

    Parameters
    ----------
    batch_size : int
    num_warmup, num_samples, num_chains : int
    rng_seed : int
    progress_bar : bool
        Show MCMC progress bar for each batch (default False)
    galaxy_idx : int, optional
        If provided, run inference only on this galaxy index
    **model_kwargs : All arguments for Redshift2Real (RA, dec, zcmb, los_r,
        los_density, los_velocity, calibration_samples, which_bias, Rmax, etc.)

    Returns
    -------
    r_samples : array, shape (ngal, num_samples * num_chains) or
        (num_samples * num_chains,) if galaxy_idx provided
    rhat_values : array, shape (ngal,) or scalar if galaxy_idx provided
    """
    RA = model_kwargs["RA"]
    dec = model_kwargs["dec"]
    zcmb = model_kwargs["zcmb"]
    los_r = model_kwargs["los_r"]
    los_density = model_kwargs["los_density"]
    los_velocity = model_kwargs["los_velocity"]

    # Extract remaining kwargs for model (excluding data arrays and verbose)
    data_keys = ["RA", "dec", "zcmb", "los_r", "los_density", "los_velocity",
                 "verbose"]
    remaining_kwargs = {
        k: v for k, v in model_kwargs.items() if k not in data_keys}

    # Handle single galaxy case
    if galaxy_idx is not None:
        fprint(f"Running inference for single galaxy at index {galaxy_idx}.")

        # Slice data for the single galaxy
        single_RA = RA[galaxy_idx:galaxy_idx+1]
        single_dec = dec[galaxy_idx:galaxy_idx+1]
        single_zcmb = zcmb[galaxy_idx:galaxy_idx+1]
        single_los_density = los_density[:, galaxy_idx:galaxy_idx+1, :]
        single_los_velocity = los_velocity[:, galaxy_idx:galaxy_idx+1, :]

        # Instantiate model for single galaxy
        model = Redshift2Real(
            RA=single_RA,
            dec=single_dec,
            zcmb=single_zcmb,
            los_r=los_r,
            los_density=single_los_density,
            los_velocity=single_los_velocity,
            verbose=True,
            **remaining_kwargs
        )

        # Run MCMC
        nuts_kernel = NUTS(
            model, init_strategy=init_to_median(num_samples=1000))
        mcmc = MCMC(
            nuts_kernel, num_warmup=num_warmup, num_samples=num_samples,
            num_chains=num_chains, progress_bar=True)

        rng_key = jrandom.PRNGKey(rng_seed)
        mcmc.run(rng_key)

        mcmc.print_summary()

        # Extract samples and rhat
        samples = mcmc.get_samples()["r"]
        r_samples = samples.T.squeeze()

        if num_chains > 1:
            samples_by_chain = mcmc.get_samples(
                group_by_chain=True)["r"][:, :, 0]
            rhat = gelman_rubin(samples_by_chain)
        else:
            rhat = 1.0

        fprint("Inference complete.")
        return r_samples, rhat

    # Batch processing for all galaxies
    ngal = len(zcmb)
    n_batches = (ngal + batch_size - 1) // batch_size

    fprint(f"Running inference for {ngal} galaxies in {n_batches} batches of "
           f"size {batch_size}.")

    r_samples = jnp.zeros((ngal, num_samples * num_chains))
    rhat_values = jnp.zeros(ngal)

    for i in tqdm(range(n_batches), desc="Processing batches",
                  disable=progress_bar):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, ngal)

        # Slice data for this batch
        batch_RA = RA[start_idx:end_idx]
        batch_dec = dec[start_idx:end_idx]
        batch_zcmb = zcmb[start_idx:end_idx]
        batch_los_density = los_density[:, start_idx:end_idx, :]
        batch_los_velocity = los_velocity[:, start_idx:end_idx, :]

        # Instantiate model for this batch
        # Only print for first batch
        model = Redshift2Real(
            RA=batch_RA,
            dec=batch_dec,
            zcmb=batch_zcmb,
            los_r=los_r,
            los_density=batch_los_density,
            los_velocity=batch_los_velocity,
            verbose=(i == 0),
            **remaining_kwargs
        )

        # Run MCMC
        nuts_kernel = NUTS(
            model, init_strategy=init_to_median(num_samples=1000))
        mcmc = MCMC(
            nuts_kernel, num_warmup=num_warmup, num_samples=num_samples,
            num_chains=num_chains, progress_bar=progress_bar)

        rng_key = jrandom.PRNGKey(rng_seed + i)
        mcmc.run(rng_key)

        # Extract samples for 'r', shape (num_samples * num_chains, batch_ngal)
        batch_samples = mcmc.get_samples()["r"]
        # Transpose to (batch_ngal, num_samples * num_chains)
        batch_samples = batch_samples.T
        r_samples = r_samples.at[start_idx:end_idx, :].set(batch_samples)

        # Compute Rhat for this batch
        if num_chains > 1:
            # Get samples grouped by chain, shape
            # (num_chains, num_samples, batch_ngal)
            batch_samples_by_chain = mcmc.get_samples(group_by_chain=True)["r"]
            batch_ngal = end_idx - start_idx
            batch_rhat = jnp.zeros(batch_ngal)
            # Compute R-hat for each galaxy in the batch
            for j in range(batch_ngal):
                # Extract samples for galaxy j, shape (num_chains, num_samples)
                galaxy_samples = batch_samples_by_chain[:, :, j]
                batch_rhat = batch_rhat.at[j].set(gelman_rubin(galaxy_samples))
        else:
            batch_rhat = jnp.ones(end_idx - start_idx)
        rhat_values = rhat_values.at[start_idx:end_idx].set(batch_rhat)

    fprint("Batched inference complete.")
    return r_samples, rhat_values
