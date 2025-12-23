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
"""NumPyro models for H0 inference with explicit distance sampling."""
import jax.numpy as jnp
import numpyro
from candel.model.simpson import ln_simpson
from jax.scipy.special import log_ndtr
from numpyro import factor, plate
from numpyro.distributions import Normal, Uniform

LN10 = jnp.log(10.0)


def log_selection_probability_cz(H0, sigma_vpec, czmax_sel, rmin, rmax, k,
                                 num_integration_points=513):
    """
    Compute log p(S=1|H0) = log ∫ Φ((czmax - H0*r) / σ_vpec) p(r) dr

    where p(r) ∝ r^k on [rmin, rmax].
    """
    # Integration grid for r (odd number for Simpson's rule)
    r_grid = jnp.linspace(rmin, rmax, num_integration_points)

    # log p(r), where p(r) ∝ r^k, normalized
    kp1 = k + 1
    ln_pr = k * jnp.log(r_grid) + jnp.log(kp1) - jnp.log(rmax**kp1 - rmin**kp1)

    # Log detection probability at each r
    z = (czmax_sel - H0 * r_grid) / sigma_vpec
    ln_p_detect = log_ndtr(z)

    # Integrate in log-space
    return ln_simpson(ln_p_detect + ln_pr, r_grid)


def model_PV(cz, mag, sigma_mu, sigma_vpec, rmin, rmax, czmax_sel, k=2,
             num_integration_points=201):
    """
    NumPyro model for H0 inference with explicit distance sampling and
    selection in observed redshift.

    Samples true distances r from a power-law prior p(r) ∝ r^k and H0 from
    a uniform prior [50, 100]. Observables are:
        - mag ~ N(5 log10(r) + 25, sigma_mu)
        - cz ~ N(H0 * r, sigma_vpec)

    Selection is modelled via the inverse detection probability:
        p(Λ|d_obs) ∝ π(Λ) [p(S=1|Λ)]^(-n) ∏_i L(d_i|Λ)

    where p(S=1|H0) = ∫ Φ((czmax - H0*r) / σ_vpec) p(r) dr.

    Parameters
    ----------
    cz : array
        Observed radial velocities [km/s].
    mag : array
        Observed distance moduli [mag].
    sigma_mu : float
        Uncertainty in distance modulus [mag].
    sigma_vpec : float
        Uncertainty in peculiar velocity [km/s].
    rmin, rmax : float
        Minimum and maximum distances [Mpc/h] for the power-law prior.
    czmax_sel : float
        Hard upper threshold in observed cz [km/s] for selection.
    k : float
        Power-law index for the distance prior p(r) ∝ r^k. Must be > -1.
    num_integration_points : int
        Number of points for Simpson integration of selection probability.
    """
    n = len(cz)

    H0 = numpyro.sample("H0", Uniform(50.0, 100.0))

    # Selection correction: [p(S=1|H0)]^(-n)
    if czmax_sel is not None:
        ln_p_sel = log_selection_probability_cz(
            H0, sigma_vpec, czmax_sel, rmin, rmax, k, num_integration_points)
        factor("selection", -n * ln_p_sel)

    # Sample distances from power-law prior p(r) ∝ r^k via inverse CDF.
    # CDF: F(r) = (r^(k+1) - rmin^(k+1)) / (rmax^(k+1) - rmin^(k+1))
    # Inverse: r = (rmin^(k+1) + u * (rmax^(k+1) - rmin^(k+1)))^(1/(k+1))
    kp1 = k + 1
    with plate("sources", n):
        u = numpyro.sample("u", Uniform(0, 1))
        r = (rmin**kp1 + u * (rmax**kp1 - rmin**kp1))**(1.0 / kp1)
        numpyro.deterministic("r", r)

        # Distance modulus likelihood
        mu_true = 5.0 * jnp.log10(r) + 25.0
        numpyro.sample("mag_obs", Normal(mu_true, sigma_mu), obs=mag)

        # Velocity likelihood
        v_model = H0 * r
        numpyro.sample("cz_obs", Normal(v_model, sigma_vpec), obs=cz)
