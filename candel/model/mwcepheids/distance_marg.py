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
"""Distance marginalisation via Simpson integration."""
import jax.numpy as jnp
from jax.scipy.special import log_ndtr

from ..integration import ln_simpson_uniform


def log_likelihood_marg_distance(
    mW_H, pi_EDR3, pi_EDR3_err,
    delta_pi, f_pi,
    d_grid, dx, log_dist_prior,
    M_pred, sigma_1,
    AH_profiles=None, AH_valid=None,
    AH_max=None, AH_width=None,
):
    """Per-star distance-marginalised log-likelihood on a fixed grid.

    Integrates the unnormalised chi2 likelihood times the distance prior
    over d using composite Simpson's rule in log-space.  Only the
    d-dependent parts of the Gaussian PDFs (the chi2 exponents) are
    included; the d-independent normalisation terms (-log sigma) must
    be added by the caller.

    Parameters
    ----------
    mW_H : (n_stars,)
        Observed Wesenheit magnitudes.
    pi_EDR3 : (n_stars,)
        Observed parallaxes [mas].
    pi_EDR3_err : (n_stars,)
        Parallax measurement errors [mas].
    delta_pi : float
        Parallax zero-point offset [mas].
    f_pi : float
        Parallax error inflation factor.
    d_grid : (n_grid,)
        Distance grid [kpc].
    dx : float
        Uniform grid spacing.
    log_dist_prior : (n_stars, n_grid)
        Unnormalised log disk prior on grid.
    M_pred : (n_stars,)
        Predicted absolute magnitude (using effective [O/H]_*).
    sigma_1 : (n_stars,)
        Inflated magnitude uncertainty.
    AH_profiles : (n_stars, n_grid), optional
        H-band extinction profiles on the distance grid.
    AH_valid : (n_stars,), optional
        Per-star validity mask for extinction.
    AH_max : float, optional
        Extinction selection threshold.
    AH_width : float, optional
        Extinction selection smoothing width.

    Returns
    -------
    (n_stars,) array
        Log of the distance-marginalised integrand per star.
    """
    mu_grid = 5.0 * jnp.log10(d_grid) + 10.0
    inv_d_grid = 1.0 / d_grid

    # Magnitude chi2: -0.5 * ((mW - M_pred - mu) / sigma_1)^2
    mW_residual = mW_H[:, None] - M_pred[:, None] - mu_grid[None, :]
    ln_mW = -0.5 * (mW_residual / sigma_1[:, None])**2

    # Parallax chi2: -0.5 * ((pi - (1/d - delta_pi)) / (f_pi * sigma_pi))^2
    pi_model = inv_d_grid[None, :] - delta_pi
    pi_sigma = f_pi * pi_EDR3_err[:, None]
    ln_pi = -0.5 * ((pi_EDR3[:, None] - pi_model) / pi_sigma)**2

    ln_integrand = log_dist_prior + ln_mW + ln_pi

    # A_H extinction selection (C22 only, when active)
    if AH_profiles is not None and AH_max is not None:
        ln_AH = jnp.where(
            AH_valid[:, None],
            log_ndtr((AH_max - AH_profiles) / AH_width),
            0.0)
        ln_integrand = ln_integrand + ln_AH

    return ln_simpson_uniform(ln_integrand, dx, axis=-1)
