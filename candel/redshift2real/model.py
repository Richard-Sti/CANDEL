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

import numpy as np
from scipy.integrate import simpson
from scipy.special import logsumexp as logsumexp_np
from tqdm import tqdm, trange

from ..cosmography import Distance2Redshift
from ..model import LOSInterpolator
from ..util import SPEED_OF_LIGHT, fprint, radec_to_cartesian


def log_mean_exp_np(logp, axis=-1):
    """NumPy version of log_mean_exp."""
    return logsumexp_np(logp, axis=axis) - np.log(logp.shape[axis])


def ln_simpson_np(ln_y, x, axis=-1):
    """
    Numerically stable log-Simpson integration using scipy.

    Computes log(integral(exp(ln_y), x)) by shifting to avoid overflow.
    """
    # Shift by max for numerical stability
    ln_y = np.asarray(ln_y)
    max_ln_y = np.max(ln_y, axis=axis, keepdims=True)
    y_shifted = np.exp(ln_y - max_ln_y)

    # Integrate the shifted values
    integral = simpson(y_shifted, x=x, axis=axis)

    # Return log of integral, adding back the shift
    return np.log(integral) + np.squeeze(max_ln_y, axis=axis)


_LOG_2PI_HALF = 0.5 * np.log(2 * np.pi)


def normal_logpdf_np(x, loc, scale):
    """NumPy implementation of Normal log-pdf."""
    return -0.5 * ((x - loc) / scale)**2 - _LOG_2PI_HALF - np.log(scale)


class BaseRedshift2Real(ABC):
    """Base class for all models. """

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
        self.cz_cmb = np.asarray(zcmb * SPEED_OF_LIGHT)  # in km/s

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

        self.use_im = True
        self.which_bias = which_bias
        fprint(f"Preparing galaxy bias model: {which_bias}",)
        if which_bias is None:
            self.use_im = False
        elif which_bias == "linear":
            self.b1 = np.asarray(calibration_samples["b1"])
            self.f_los_delta = LOSInterpolator(
                los_r, los_density - 1, r0_decay_scale=r0_decay_scale)
            self.lp_norm = self.compute_linear_bias_lp_normalization(
                self.los_grid_r)
        elif which_bias == "double_powerlaw":
            self.alpha_low = np.asarray(calibration_samples["alpha_low"])
            self.alpha_high = np.asarray(calibration_samples["alpha_high"])
            self.log_rho_t = np.asarray(calibration_samples["log_rho_t"])
            self.f_los_log_density = LOSInterpolator(
                los_r, np.log(los_density), r0_decay_scale=r0_decay_scale)

            self.lp_norm = self.compute_double_powerlaw_bias_lp_normalization(
                self.los_grid_r)
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
        # DEBUG: print shapes
        print(f"DEBUG: sigma_v={self.sigma_v.shape}, "
              f"beta={self.beta.shape}")
        print(f"DEBUG: Vext_radial={self.Vext_radial.shape}, "
              f"cz_cmb={self.cz_cmb.shape}")
        print(f"DEBUG: los_grid_r={self.los_grid_r.shape}")
        if self.which_bias == "linear":
            print(f"DEBUG: b1={self.b1.shape}, "
                  f"lp_norm={self.lp_norm.shape}")
        elif self.which_bias == "double_powerlaw":
            print(f"DEBUG: alpha_low={self.alpha_low.shape}, "
                  f"lp_norm={self.lp_norm.shape}")

        print("Calibration sample statistics:")
        sv = self.sigma_v
        print(f"sigma_v : {np.mean(sv):.3f} +- {np.std(sv):.3f}")
        print(f"beta    : {np.mean(self.beta):.3f} +- "
              f"{np.std(self.beta):.3f}")
        if self.which_bias == "linear":
            print(f"b1      : {np.mean(self.b1):.3f} +- "
                  f"{np.std(self.b1):.3f}")
        elif self.which_bias == "double_powerlaw":
            print(f"alpha_low  : {np.mean(self.alpha_low):.3f} +- "
                  f"{np.std(self.alpha_low):.3f}")
            print(f"alpha_high : {np.mean(self.alpha_high):.3f} +- "
                  f"{np.std(self.alpha_high):.3f}")
            print(f"log_rho_t  : {np.mean(self.log_rho_t):.3f} +- "
                  f"{np.std(self.log_rho_t):.3f}")

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Redshift2Real(BaseRedshift2Real):
    """
    Numerical evaluation of the posterior `p(z_cosmo | z_obs, calibration)`.

    Instead of MCMC sampling, this class evaluates the log-posterior on a grid
    of `z_cosmo` values and normalizes numerically using Simpson integration.
    Uses pure NumPy for computation (no JAX JIT overhead).
    """

    def compute_linear_bias_lp_normalization(self, los_grid_r, batch_size=10):
        """NumPy version of linear bias normalization (batched)."""
        fprint("Computing `linear` bias lp normalization (NumPy)...",
               verbose=self.verbose)
        los_grid_r = np.asarray(los_grid_r)
        ngal = self.len_input_data
        ncal = self.num_cal

        # Interpolate all at once, then batch the bias computation
        fprint("  Interpolating LOS delta for all galaxies...",
               verbose=self.verbose)
        los_grid_delta_all = np.asarray(
            self.f_los_delta.interp_many_steps_per_galaxy(los_grid_r))
        nfield = los_grid_delta_all.shape[0]

        b1 = np.asarray(self.b1)

        lp_norm = np.zeros((nfield, ngal, ncal))
        n_batches = (ngal + batch_size - 1) // batch_size

        for i in trange(n_batches, desc="  Computing bias norm",
                        disable=not self.verbose):
            start = i * batch_size
            end = min((i + 1) * batch_size, ngal)

            los_grid_delta = los_grid_delta_all[:, start:end, :]

            bias_params = [b1[None, None, :, None]]
            intg = lp_galaxy_bias_np(
                los_grid_delta[:, :, None, :], None, bias_params,
                galaxy_bias="linear")
            lp_norm[:, start:end, :] = ln_simpson_np(intg, los_grid_r, axis=-1)

        return lp_norm

    def compute_double_powerlaw_bias_lp_normalization(self, los_grid_r,
                                                      batch_size=10):
        """NumPy version of double powerlaw bias normalization (batched)."""
        fprint("Computing `double_powerlaw` bias lp normalization (NumPy)...",
               verbose=self.verbose)
        los_grid_r = np.asarray(los_grid_r)
        ngal = self.len_input_data
        ncal = self.num_cal

        # Interpolate all at once, then batch the bias computation
        fprint("Interpolating LOS log-density for all galaxies...",
               verbose=self.verbose)
        los_log_density_all = np.asarray(
            self.f_los_log_density.interp_many_steps_per_galaxy(los_grid_r))
        nfield = los_log_density_all.shape[0]

        alpha_low = np.asarray(self.alpha_low)
        alpha_high = np.asarray(self.alpha_high)
        log_rho_t = np.asarray(self.log_rho_t)

        lp_norm = np.zeros((nfield, ngal, ncal))
        n_batches = (ngal + batch_size - 1) // batch_size

        for i in trange(n_batches, desc="  Computing bias norm",
                        disable=not self.verbose):
            start = i * batch_size
            end = min((i + 1) * batch_size, ngal)

            los_log_density = los_log_density_all[:, start:end, :]

            bias_params = [
                alpha_low[None, None, :, None],
                alpha_high[None, None, :, None],
                log_rho_t[None, None, :, None],
            ]
            intg = lp_galaxy_bias_np(
                None, los_log_density[:, :, None, :], bias_params,
                galaxy_bias="double_powerlaw")
            lp_norm[:, start:end, :] = ln_simpson_np(intg, los_grid_r, axis=-1)

        return lp_norm

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
        r_grid = self.los_grid_r
        ngal = self.len_input_data
        nrad = len(r_grid)

        # Precompute quantities independent of galaxy
        lp_r = 2 * np.log(r_grid)
        z_grid = np.asarray(self.dist2redshift(r_grid))

        # Compute Jacobian |dr/dz| via |dz/dr|^{-1} on a denser grid
        r_dense = np.linspace(r_grid[0], r_grid[-1], 2 * len(r_grid) - 1)
        z_dense = np.asarray(self.dist2redshift(r_dense))
        dz_dr_dense = np.gradient(z_dense, r_dense)
        dz_dr = np.interp(r_grid, r_dense, dz_dr_dense)
        log_jacobian = -np.log(dz_dr)  # log|dr/dz|

        # Pre-interpolate LOS fields (do once, not per batch)
        fprint("Interpolating LOS velocity...", verbose=self.verbose)
        Vpec_all = np.asarray(
            self.f_los_velocity.interp_many_steps_per_galaxy(r_grid))

        los_delta_all = None
        los_log_density_all = None
        if self.which_bias == "linear":
            fprint("Interpolating LOS delta...", verbose=self.verbose)
            los_delta_all = np.asarray(
                self.f_los_delta.interp_many_steps_per_galaxy(r_grid))
        elif self.which_bias == "double_powerlaw":
            fprint("Interpolating LOS log-density...", verbose=self.verbose)
            los_log_density_all = np.asarray(
                self.f_los_log_density.interp_many_steps_per_galaxy(r_grid))

        log_posterior = np.zeros((ngal, nrad))
        n_batches = (ngal + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches), desc="Processing batches",
                      disable=not self.verbose):
            start = i * batch_size
            end = min((i + 1) * batch_size, ngal)

            log_posterior[start:end] = self._process_batch(
                start, end, lp_r, z_grid, log_jacobian,
                Vpec_all, los_delta_all, los_log_density_all)

        return z_grid, log_posterior

    def _process_batch(self, start, end, lp_r, z_grid, log_jacobian,
                       Vpec_all, los_delta_all, los_log_density_all):
        """Process a batch of galaxies using pure NumPy."""
        # Slice pre-interpolated data for this batch
        Vpec = Vpec_all[:, start:end, :]
        Vext_radial = self.Vext_radial[start:end, :]
        cz_cmb = self.cz_cmb[start:end]

        if self.which_bias == "linear":
            los_delta = los_delta_all[:, start:end, :]
            lp_norm = self.lp_norm[:, start:end, :]

            bias_params = [self.b1[None, None, :, None]]
            lp_bias = lp_galaxy_bias_np(
                los_delta[:, :, None, :], None, bias_params, "linear")
            lp_r_full = (lp_r[None, None, None, :]
                         + lp_bias - lp_norm[..., None])

        elif self.which_bias == "double_powerlaw":
            los_log_density = los_log_density_all[:, start:end, :]
            lp_norm = self.lp_norm[:, start:end, :]

            bias_params = [
                self.alpha_low[None, None, :, None],
                self.alpha_high[None, None, :, None],
                self.log_rho_t[None, None, :, None],
            ]
            lp_bias = lp_galaxy_bias_np(
                None, los_log_density[:, :, None, :], bias_params,
                "double_powerlaw")
            lp_r_full = (lp_r[None, None, None, :]
                         + lp_bias - lp_norm[..., None])
        else:
            lp_r_full = lp_r[None, None, None, :]

        # Peculiar velocity redshift (nfield, batch, ncal, nrad)
        zpec = (
            self.beta[None, None, :, None] * Vpec[:, :, None, :]
            + Vext_radial[None, :, :, None]) / SPEED_OF_LIGHT

        # Predicted cz (nfield, batch, ncal, nrad)
        cz_pred = SPEED_OF_LIGHT * (
            (1 + z_grid)[None, None, None, :] * (1 + zpec) - 1)

        # Log-likelihood (nfield, batch, ncal, nrad)
        ll = normal_logpdf_np(cz_cmb[None, :, None, None], cz_pred,
                              self.sigma_v[None, None, :, None])
        ll += lp_r_full

        # Average over calibration samples, shape (nfield, batch, nrad)
        ll = log_mean_exp_np(ll, axis=2)
        # Average over velocity field realizations, shape (batch, nrad)
        log_posterior_unnorm = log_mean_exp_np(ll, axis=0)

        # Apply Jacobian for p(z) = p(r) * |dr/dz|
        log_posterior_unnorm += log_jacobian[None, :]

        # Normalize using Simpson integration in z-space
        log_norm = ln_simpson_np(log_posterior_unnorm, z_grid, axis=-1)
        return log_posterior_unnorm - log_norm[:, None]

    def posterior_summary(self, z_grid, log_posterior, ci=0.68):
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
        ngal = posterior.shape[0]
        nz = len(z_grid)
        dz = z_grid[1] - z_grid[0]

        # Mean
        mean = np.sum(posterior * z_grid[None, :], axis=-1) * dz

        # Variance and std
        var = np.sum(posterior * (z_grid[None, :] - mean[:, None])**2,
                     axis=-1) * dz
        std = np.sqrt(var)

        # MAP
        map_idx = np.argmax(posterior, axis=-1)
        map_val = z_grid[map_idx]

        # CDF for median and CI
        cdf = np.cumsum(posterior, axis=-1) * dz

        # Vectorized quantile finding
        def find_quantiles_vectorized(cdf, q):
            idx = np.argmax(cdf >= q, axis=-1)
            # Handle case where q is never reached (return last index)
            not_reached = np.all(cdf < q, axis=-1)
            idx[not_reached] = nz - 1
            return z_grid[idx]

        median = find_quantiles_vectorized(cdf, 0.5)
        ci_low = find_quantiles_vectorized(cdf, (1 - ci) / 2)
        ci_high = find_quantiles_vectorized(cdf, 1 - (1 - ci) / 2)

        return {
            'mean': mean,
            'median': median,
            'std': std,
            'map': map_val,
            'ci_low': ci_low,
            'ci_high': ci_high,
        }


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
    else:
        raise ValueError(f"Invalid galaxy bias model '{galaxy_bias}'.")
    return lp
