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
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.stats import norm as jax_norm
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp as logsumexp_np
from tqdm import trange

from ..cosmo.cosmography import Distance2Redshift
from ..model import LOSInterpolator
from ..model.integration import ln_simpson
from ..model.pv_utils import lp_galaxy_bias
from ..model.utils import logmeanexp
from ..util import SPEED_OF_LIGHT, fprint, radec_to_cartesian

###############################################################################
#                          Utility functions                                  #
###############################################################################


def log_mean_exp_np(logp, axis=-1):
    """NumPy version of log_mean_exp."""
    return logsumexp_np(logp, axis=axis) - np.log(logp.shape[axis])


def ln_simpson_np(ln_y, x, axis=-1):
    """
    Numerically stable log-Simpson integration using scipy.

    Computes log(integral(exp(ln_y), x)) by shifting to avoid overflow.
    """
    ln_y = np.asarray(ln_y)
    max_ln_y = np.max(ln_y, axis=axis, keepdims=True)
    y_shifted = np.exp(ln_y - max_ln_y)
    integral = simpson(y_shifted, x=x, axis=axis)
    return np.log(integral) + np.squeeze(max_ln_y, axis=axis)


_LOG_2PI_HALF = 0.5 * np.log(2 * np.pi)


def normal_logpdf_np(x, loc, scale):
    """NumPy implementation of Normal log-pdf."""
    return -0.5 * ((x - loc) / scale)**2 - _LOG_2PI_HALF - np.log(scale)


def smoothclip_nr_np(nr, tau):
    """Smooth zero-clipping for the number density (NumPy version)."""
    return 0.5 * (nr + np.sqrt(nr**2 + tau**2))


def lp_galaxy_bias_np(delta, log_rho, bias_params, galaxy_bias):
    """NumPy version of lp_galaxy_bias."""
    if galaxy_bias == "powerlaw":
        lp = bias_params[0] * log_rho
    elif galaxy_bias == "double_powerlaw":
        alpha_low, alpha_high, log_rho_t = bias_params
        log_x = log_rho - log_rho_t
        lp = (alpha_low * log_x
              + (alpha_high - alpha_low) * np.logaddexp(0.0, log_x))
    elif "linear" in galaxy_bias or galaxy_bias == "unity":
        lp = np.log(smoothclip_nr_np(1 + bias_params[0] * delta, tau=0.1))
    elif galaxy_bias == "spline":
        knots_log1pd, amplitudes = bias_params
        cs = CubicSpline(knots_log1pd, amplitudes, bc_type='natural')
        x = np.clip(log_rho, knots_log1pd[0], knots_log1pd[-1])
        lp = cs(x)
    else:
        raise ValueError(f"Invalid galaxy bias model '{galaxy_bias}'.")
    return lp


###############################################################################
#                           Model classes                                     #
###############################################################################


class BaseRedshift2Real(ABC):
    """Base class for all models."""

    def __init__(self, RA, dec, zcmb, los_r, los_density, los_velocity,
                 which_bias, calibration_samples, Rmin=1e-7, Rmax=500,
                 num_rgrid=101, r0_decay_scale=5, Om0=0.3, verbose=True,
                 r_init=None):
        self.verbose = verbose
        self.dist2redshift = Distance2Redshift(Om0=Om0)
        self.r_init = np.asarray(r_init) if r_init is not None else None

        if self.r_init is not None:
            if np.any(self.r_init < Rmin) or np.any(self.r_init > Rmax):
                raise ValueError("r_init values must be within [Rmin, Rmax]")

        self.Rmin = Rmin
        self.Rmax = Rmax
        assert Rmin > 0 and Rmax > Rmin
        assert num_rgrid % 2 == 1

        self.len_input_data = len(zcmb)
        self.cz_cmb = np.asarray(zcmb * SPEED_OF_LIGHT)

        los_r = np.asarray(los_r)
        los_density = np.asarray(los_density)
        los_velocity = np.asarray(los_velocity)

        self.f_los_velocity = LOSInterpolator(
            los_r, los_velocity, r0_decay_scale=r0_decay_scale)

        rhat = radec_to_cartesian(RA, dec)
        self.los_grid_r = np.linspace(Rmin, Rmax, num_rgrid)

        self.calibration_samples = calibration_samples
        calibration_keys = list(calibration_samples.keys())
        self.num_cal = calibration_samples[calibration_keys[0]].shape[0]

        # LOS Vext, (ngal, ncalibration_samples)
        if "Vext" in calibration_samples:
            self.Vext_radial = np.sum(
                rhat[:, None, :] * calibration_samples["Vext"][None, :, :],
                axis=-1)
        else:
            fprint("No Vext in calibration samples.", verbose=self.verbose)
            self.Vext_radial = np.zeros(
                (self.len_input_data, self.num_cal))

        # Bias model setup
        self.which_bias = which_bias
        self._bias_interp = None
        self._bias_params = []
        self._bias_param_names = []

        fprint(f"Preparing galaxy bias model: {which_bias}",
               verbose=self.verbose)
        if which_bias is None:
            pass
        elif which_bias == "linear":
            self._bias_interp = LOSInterpolator(
                los_r, los_density - 1, r0_decay_scale=r0_decay_scale)
            self._bias_params = [np.asarray(calibration_samples["b1"])]
            self._bias_param_names = ["b1"]
        elif which_bias == "double_powerlaw":
            self._bias_interp = LOSInterpolator(
                los_r, np.log(los_density), r0_decay_scale=r0_decay_scale)
            self._bias_params = [
                np.asarray(calibration_samples["alpha_low"]),
                np.asarray(calibration_samples["alpha_high"]),
                np.asarray(calibration_samples["log_rho_t"]),
            ]
            self._bias_param_names = ["alpha_low", "alpha_high", "log_rho_t"]
        elif which_bias == "spline":
            raise NotImplementedError(
                "Spline bias is not yet supported in redshift2real. "
                "Use 'linear' or 'double_powerlaw' for redshift2real.")
        else:
            raise ValueError(f"Unknown bias model: {which_bias}")

        self.sigma_v = np.asarray(calibration_samples["sigma_v"])
        if "beta" in calibration_samples:
            self.beta = np.asarray(calibration_samples["beta"])
        else:
            fprint("Beta not in calibration samples. Setting beta=1.",
                   verbose=self.verbose)
            self.beta = np.ones_like(self.sigma_v)

        fprint(f"Loaded {self.len_input_data} objects and "
               f"{self.num_cal} calibration samples.", verbose=self.verbose)

        self.print_sample_stats()

    def print_sample_stats(self):
        if not self.verbose:
            return

        fprint(f"sigma_v : {np.mean(self.sigma_v):.3f} +- "
               f"{np.std(self.sigma_v):.3f}")
        fprint(f"beta    : {np.mean(self.beta):.3f} +- "
               f"{np.std(self.beta):.3f}")
        for name, p in zip(self._bias_param_names, self._bias_params):
            fprint(f"{name:12s}: {np.mean(p):.3f} +- {np.std(p):.3f}")

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Redshift2Real(BaseRedshift2Real):
    """
    Numerical evaluation of the posterior `p(z_cosmo | z_obs, calibration)`.

    Instead of MCMC sampling, this class evaluates the log-posterior on a grid
    of `z_cosmo` values and normalizes numerically using Simpson integration.
    Core computation is JIT-compiled with JAX.
    """

    def _compute_bias_normalization(self, los_grid_r, batch_size=10):
        """Compute bias log-prior normalization for the configured model."""
        if self.which_bias is None:
            return None

        fprint(f"Computing `{self.which_bias}` bias normalization...",
               verbose=self.verbose)
        los_grid_r = jnp.asarray(los_grid_r)

        field_all = jnp.asarray(
            self._bias_interp.interp_many(los_grid_r))
        nfield, ngal, _ = field_all.shape

        bias_params_bc = [jnp.asarray(p)[None, None, :, None]
                          for p in self._bias_params]

        lp_norm = np.zeros((nfield, ngal, self.num_cal))
        n_batches = (ngal + batch_size - 1) // batch_size

        for i in trange(n_batches, desc="  Computing bias norm",
                        disable=not self.verbose):
            start = i * batch_size
            end = min((i + 1) * batch_size, ngal)
            field = field_all[:, start:end, :]

            lp_norm[:, start:end, :] = np.asarray(
                self._bias_norm_jit(
                    self.which_bias, field[:, :, None, :],
                    bias_params_bc, los_grid_r))

        return lp_norm

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _bias_norm_jit(which_bias, field, bias_params, los_grid_r):
        """JIT-compiled bias normalization for a batch."""
        if which_bias == "linear":
            intg = lp_galaxy_bias(field, None, bias_params, "linear")
        else:
            intg = lp_galaxy_bias(None, field, bias_params,
                                  "double_powerlaw")
        # Pad x to match intg's ndim for ln_simpson
        x = los_grid_r[(None,) * (intg.ndim - 1)]
        return ln_simpson(intg, x, axis=-1)

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _process_batch_jit(which_bias, lp_r, z_grid, log_jacobian, Vpec,
                           Vext_radial, cz_cmb, beta, sigma_v,
                           bias_field, lp_norm, bias_params):
        """JIT-compiled core of _process_batch."""
        if which_bias is not None:
            if which_bias == "linear":
                lp_bias = lp_galaxy_bias(
                    bias_field[:, :, None, :], None, bias_params, "linear")
            else:
                lp_bias = lp_galaxy_bias(
                    None, bias_field[:, :, None, :], bias_params,
                    "double_powerlaw")
            lp_r_full = (lp_r[None, None, None, :]
                         + lp_bias - lp_norm[..., None])
        else:
            lp_r_full = lp_r[None, None, None, :]

        zpec = (beta[None, None, :, None] * Vpec[:, :, None, :]
                + Vext_radial[None, :, :, None]) / SPEED_OF_LIGHT

        cz_pred = SPEED_OF_LIGHT * (
            (1 + z_grid)[None, None, None, :] * (1 + zpec) - 1)

        ll = jax_norm.logpdf(cz_cmb[None, :, None, None], cz_pred,
                             sigma_v[None, None, :, None])
        ll += lp_r_full

        ll = logmeanexp(ll, axis=2)
        log_posterior_unnorm = logmeanexp(ll, axis=0)

        log_posterior_unnorm += log_jacobian[None, :]

        log_norm = ln_simpson(log_posterior_unnorm, z_grid[None, :], axis=-1)
        return log_posterior_unnorm - log_norm[:, None]

    def __call__(self, batch_size=10):
        """
        Compute the posterior PDF for each galaxy on a grid of z_cosmo.

        Parameters
        ----------
        batch_size : int
            Number of galaxies to process at once.

        Returns
        -------
        z_grid : array, shape (num_rgrid,)
            Grid of cosmological redshift values.
        log_posterior : array, shape (ngal, num_rgrid)
            Normalized log-posterior PDF at each grid point.
        """
        # Compute bias normalization
        self.lp_norm = self._compute_bias_normalization(
            self.los_grid_r, batch_size)

        r_grid = self.los_grid_r
        ngal = self.len_input_data
        nrad = len(r_grid)

        # Precompute quantities independent of galaxy
        lp_r = 2 * np.log(r_grid)
        z_grid = np.asarray(self.dist2redshift(r_grid))

        # Compute Jacobian |dr/dz| on a denser grid
        r_dense = np.linspace(r_grid[0], r_grid[-1], 2 * nrad - 1)
        z_dense = np.asarray(self.dist2redshift(r_dense))
        dz_dr_dense = np.gradient(z_dense, r_dense)
        dz_dr = np.interp(r_grid, r_dense, dz_dr_dense)
        log_jacobian = -np.log(dz_dr)  # log|dr/dz|

        # Pre-interpolate LOS fields
        fprint("Interpolating LOS velocity...", verbose=self.verbose)
        Vpec_all = np.asarray(
            self.f_los_velocity.interp_many(r_grid))

        bias_field_all = None
        if self.which_bias is not None:
            fprint(f"Interpolating LOS {self.which_bias} field...",
                   verbose=self.verbose)
            bias_field_all = np.asarray(
                self._bias_interp.interp_many(r_grid))

        log_posterior = np.zeros((ngal, nrad))
        n_batches = (ngal + batch_size - 1) // batch_size

        for i in trange(n_batches, desc="Processing batches",
                        disable=not self.verbose):
            start = i * batch_size
            end = min((i + 1) * batch_size, ngal)

            log_posterior[start:end] = self._process_batch(
                start, end, lp_r, z_grid, log_jacobian,
                Vpec_all, bias_field_all)

        return z_grid, log_posterior

    def _process_batch(self, start, end, lp_r, z_grid, log_jacobian,
                       Vpec_all, bias_field_all):
        """Process a batch of galaxies (thin wrapper around JIT core)."""
        Vpec = jnp.asarray(Vpec_all[:, start:end, :])
        Vext_radial = jnp.asarray(self.Vext_radial[start:end, :])
        cz_cmb = jnp.asarray(self.cz_cmb[start:end])

        if self.which_bias is not None:
            bias_field = jnp.asarray(bias_field_all[:, start:end, :])
            lp_norm = jnp.asarray(self.lp_norm[:, start:end, :])
            bias_params = [jnp.asarray(p)[None, None, :, None]
                           for p in self._bias_params]
        else:
            bias_field = None
            lp_norm = None
            bias_params = []

        return np.asarray(self._process_batch_jit(
            self.which_bias,
            jnp.asarray(lp_r), jnp.asarray(z_grid),
            jnp.asarray(log_jacobian),
            Vpec, Vext_radial, cz_cmb,
            jnp.asarray(self.beta), jnp.asarray(self.sigma_v),
            bias_field, lp_norm, bias_params))

    @staticmethod
    def posterior_summary(z_grid, log_posterior, ci=0.68):
        """
        Compute summary statistics from the posterior.

        Parameters
        ----------
        z_grid : array, shape (nz,)
        log_posterior : array, shape (ngal, nz)
        ci : float
            Credible interval (default 0.68 for 1-sigma).

        Returns
        -------
        dict with keys:
            - 'mean': posterior mean
            - 'median': posterior median
            - 'std': posterior standard deviation
            - 'map': maximum a posteriori estimate
            - 'ci_low', 'ci_high': credible interval bounds
        """
        posterior = np.exp(log_posterior)
        nz = len(z_grid)

        mean = simpson(posterior * z_grid[None, :], x=z_grid, axis=-1)

        var = simpson(
            posterior * (z_grid[None, :] - mean[:, None])**2,
            x=z_grid, axis=-1)
        std = np.sqrt(var)

        map_idx = np.argmax(posterior, axis=-1)
        map_val = z_grid[map_idx]

        cdf = cumulative_trapezoid(posterior, z_grid, axis=-1, initial=0)

        def find_quantiles(cdf, q):
            idx = np.argmax(cdf >= q, axis=-1)
            not_reached = np.all(cdf < q, axis=-1)
            idx[not_reached] = nz - 1
            return z_grid[idx]

        median = find_quantiles(cdf, 0.5)
        ci_low = find_quantiles(cdf, (1 - ci) / 2)
        ci_high = find_quantiles(cdf, 1 - (1 - ci) / 2)

        return {
            'mean': mean,
            'median': median,
            'std': std,
            'map': map_val,
            'ci_low': ci_low,
            'ci_high': ci_high,
        }
