# Copyright (C) 2024 Richard Stiskalek, Deaglan Bartlett
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
Conversion of the `beta` factor from linear theory to either `fsigma8` or `S8`
"""
import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from tqdm import tqdm


class Beta2Cosmology:
    """
    Converter for mapping samples of `beta`, which scales the velocities
    of linear theory reconstructions into either `fsigma8` or `S8`.

    Supports symbolic regression (SR) or Juszkiewicz-based mappings from
    non-linear to linear sigma8
    """

    def __init__(self, Om0=0.3111, h=0.6766, Ob=None, ns=0.9665,
                 mnu=0.0, w0=-1, wa=0, gamma=6/11,
                 sigma8g_prior="carrick", method="sr", beta_jusz=0.216):

        Ob = 0.02242 / h**2 if Ob is None else Ob
        self.Om0 = Om0
        self.gamma = gamma
        self.method = method
        self.beta_jusz = beta_jusz
        self.cosmo_params = dict(
            h=h, Om=Om0, Ob=Ob, ns=ns, mnu=mnu, w0=w0, wa=wa)

        self.mu_sigma8g, self.sigma_sigma8g = self._get_sigma8g_prior(
            sigma8g_prior)
        self._sr_interp = None

    def _get_sigma8g_prior(self, label):
        priors = {
            "westover": (0.98, 0.07),
            "carrick": (0.99, 0.04),
        }
        if label not in priors:
            raise ValueError(f"Unknown sigma8g prior: {label}")
        return priors[label]

    def compute_sigma8_nonlinear_from_pk(self, As):
        try:
            from symbolic_pofk import syren_new
        except ImportError as e:
            raise ImportError("symbolic_pofk.syren_new not found.") from e

        p = self.cosmo_params
        k = np.logspace(np.log10(9e-3), np.log10(9), 2500)
        Pk = syren_new.pnl_new_emulated(
            k, As, p["Om"], p["Ob"], p["h"],
            p["ns"], p["mnu"], p["w0"], p["wa"], a=1)
        kR = k * 8.0
        Wk = 3 * (np.sin(kR) - kR * np.cos(kR)) / kR**3
        integrand = k**2 * Pk * Wk**2
        return np.sqrt(simpson(integrand * k, x=np.log(k)) / (2 * np.pi**2))

    def compute_sigma8_nonlinear_from_beta(self, beta):
        sigma8g = np.random.normal(
            self.mu_sigma8g, self.sigma_sigma8g, size=len(beta))
        fsigma8_nl = beta * sigma8g
        return fsigma8_nl / self.Om0**self.gamma

    def _find_linear_sigma8(self, sigma8_nl):
        if self.method != "sr":
            raise ValueError("Only SR method is supported for this function.")

        try:
            from symbolic_pofk import linear_new
        except ImportError as e:
            raise ImportError("symbolic_pofk.linear_new not found.") from e

        p = self.cosmo_params

        def loss(As):
            return (self.compute_sigma8_nonlinear_from_pk(As) - sigma8_nl)**2

        res = minimize(loss, 2.2)
        return linear_new.As_to_sigma8(res.x[0], p["Om"], p["Ob"], p["h"],
                                       p["ns"], p["mnu"], p["w0"], p["wa"])

    def _build_sr_interp(self):
        if self._sr_interp is not None:
            return self._sr_interp

        x = np.linspace(0.6, 1.1, 100)
        y = np.array([
            self._find_linear_sigma8(s8_nl)
            for s8_nl in tqdm(x, desc="Building SR interpolator")])
        self._sr_interp = interp1d(x, y, kind="cubic", bounds_error=False,
                                   fill_value="extrapolate")
        return self._sr_interp

    def sigma8_nl_to_lin(self, sigma8_nl):
        """Converts a non-linear sigma8 to a linear sigma8."""
        if self.method == "sr":
            return self._build_sr_interp()(sigma8_nl)
        elif self.method == "juszkiewicz":
            root = np.sqrt(1 + 4 * self.beta_jusz * sigma8_nl**2)
            return np.sqrt((root - 1) / (2 * self.beta_jusz))
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def compute_fsigma8_linear(self, beta):
        """Compute `fsigma8_linear` from samples of `beta`."""
        sigma8_nl = self.compute_sigma8_nonlinear_from_beta(beta)
        sigma8_lin = self.sigma8_nl_to_lin(sigma8_nl)
        return sigma8_lin * self.Om0**self.gamma

    def compute_S8(self, beta):
        """
        Compute `S8` from samples of `beta`.
        """
        sigma8_nl = self.compute_sigma8_nonlinear_from_beta(beta)
        sigma8_lin = self.sigma8_nl_to_lin(sigma8_nl)
        return sigma8_lin * (self.Om0 / 0.3)**self.gamma
