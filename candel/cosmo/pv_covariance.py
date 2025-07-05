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
"""Covariance matrix of peculiar velocities in LCDM cosmology."""

import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import odeint, simpson
from scipy.interpolate import interp1d
from scipy.special import eval_legendre, spherical_jn
from tqdm import tqdm

from ..util import radec_to_cartesian


def get_Pk_CAMB(H0=67.4, Om0=0.3153, Ombh2=0.0224, As=2.100549e-9, ns=0.965,
                kmax=20.0, nonlinear=True):
    """Get (non-linear) matter power spectrum from CAMB in `Mpc / h` units."""
    try:
        import camb
    except ImportError:
        raise ImportError("CAMB is not installed. "
                          "Please install it to use `get_Pk_CAMB`.")
    pars = camb.CAMBparams()
    h = H0 / 100.0
    omch2 = Om0 * h**2 - Ombh2

    pars.set_cosmology(H0=H0, ombh2=Ombh2, omch2=omch2)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[0], kmax=kmax, nonlinear=nonlinear)

    results = camb.get_results(pars)
    k, __, Pk = results.get_linear_matter_power_spectrum(nonlinear=nonlinear)
    return k, Pk[0]


def compute_Fuv(u, v, cos_theta, ell_min=0, ell_max=100):
    """
    Compute:
    F(u, v, cosθ) = Σ_{l=ell_min}^{ell_max} (2l + 1) j_l'(u) j_l'(v) P_l(cosθ)
    """
    ells = np.arange(ell_min, ell_max + 1)
    P = eval_legendre(ells, cos_theta)
    j_u = spherical_jn(ells, u, derivative=True)
    j_v = spherical_jn(ells, v, derivative=True)
    return np.sum((2 * ells + 1) * j_u * j_v * P)


def compute_dD_dtau(a=1, Omega_m=0.315, Omega_L=0.685, H0=67.4):
    """
    Compute the derivative of the linear growth factor D with respect to
    conformal time τ.

    This function solves the linear growth ODE for a flat ΛCDM cosmology,
    normalizes D(a) such that D(a=1) = 1, and returns an interpolating function
    that evaluates dD/dτ(a).
    """
    def E(a):
        return np.sqrt(Omega_m / a**3 + Omega_L)

    def dE_da(a):
        return -1.5 * Omega_m / a**4 / E(a)

    def growth_ode(D, a):
        dD, dD_da = D
        d2D_da2 = -(3/a + dE_da(a) / E(a)) * dD_da + (3*Omega_m / (2*a**5*E(a)**2)) * dD  # noqa
        return [dD_da, d2D_da2]

    a_arr = np.linspace(1e-3, 1.0, 1000)
    D_init = [1e-5, 0.0]
    sol = odeint(growth_ode, D_init, a_arr)
    D_a = sol[:, 0]
    D_a /= D_a[-1]  # normalize D(a=1)=1

    dD_da = np.gradient(D_a, a_arr)
    dD_da_interp = interp1d(a_arr, dD_da, kind='cubic')

    def dD_dtau(a):
        return a * H0 * E(a) * dD_da_interp(a)

    return dD_dtau(a)


def _compute_covariance_element(i, j, rhat, r, k, Pk, dDdtau, ell_min,
                                ell_max):
    cos_theta = np.dot(rhat[i], rhat[j])

    integrand = np.zeros_like(k)
    for n in range(len(k)):
        kn = k[n]
        integrand[n] = Pk[n] * compute_Fuv(
            kn * r[i], kn * r[j], cos_theta, ell_min, ell_max)

    integrand /= 2 * np.pi**2
    integral = simpson(integrand, x=k)
    return (i, j, dDdtau**2 * integral)


def compute_covariance_matrix(r, RA, dec, k, Pk, dDdtau, ell_min=0,
                              ell_max=2000, n_jobs=1):
    """
    Compute the velocity covariance matrix for objects at positions
    (r, RA, dec).
    """
    N = len(r)
    C = np.zeros((N, N))
    rhat = radec_to_cartesian(RA, dec)

    i_idx, j_idx = np.triu_indices(N)
    pairs = list(zip(i_idx, j_idx))

    args = (rhat, r, k, Pk, dDdtau, ell_min, ell_max)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_covariance_element)(i, j, *args)
        for i, j in tqdm(pairs)
    )

    for i, j, value in results:
        C[i, j] = value
        C[j, i] = value

    return C
