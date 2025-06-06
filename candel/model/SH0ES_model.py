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
"""Turning the SH0ES framework into a forward model in JAX."""

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from numpyro import factor, plate, sample
from numpyro.distributions import Normal, Uniform

from ..util import fprint


def mvn_logpdf_cholesky(y, mu, L):
    """
    Log-pdf of a multivariate normal using Cholesky factor L (lower
    triangular).
    """
    z = solve_triangular(L, y - mu, lower=True)
    log_det = jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (len(y) * jnp.log(2 * jnp.pi) + 2 * log_det + jnp.dot(z, z))


class SH0ESModel:
    """
    A version of the SH0ES inference turned into a forward model.
    """

    def __init__(self, data):
        attrs_set = []
        for k, v in data.items():
            if k in ["q_names", "host_map"]:
                continue

            if isinstance(v, np.ndarray):
                v = jnp.asarray(v)

            setattr(self, k, v)
            attrs_set.append(k)

        fprint(f"set the following attributes: {', '.join(attrs_set)}")

    def __call__(self, include_SN=True):
        if include_SN:
            M_B = sample("M_B", Normal(-19.24, 0.5))
            H0 = sample("H0", Uniform(0, 100))

        M_W = sample("M_W", Normal(-6, 1))
        b_W = sample("b_W", Normal(-3, 1))
        Z_W = sample("Z_W", Normal(0, 1))

        # HST and Gaia zero-point calibration of MW Cepheids.
        sample("M_W_HST", Normal(M_W, self.e_M_HST), obs=self.M_HST)
        sample("M_W_Gaia", Normal(M_W, self.e_M_Gaia), obs=self.M_Gaia)

        # TODO: Exchange these for non-informative priors.
        with plate("hosts", self.num_hosts):
            mu_host = sample("mu_host", Normal(25, 5))

        mu_N4258 = sample("mu_N4258", Normal(25, 5))
        mu_LMC = sample("mu_LMC", Normal(25, 5))
        mu_M31 = sample("mu_M31", Normal(25, 5))

        sample("mu_N4258_ll",
               Normal(self.mu_N4258_anchor, self.e_mu_N4258_anchor),
               obs=mu_N4258)
        sample("mu_LMC_ll",
               Normal(self.mu_LMC_anchor, self.e_mu_LMC_anchor),
               obs=mu_LMC)

        dZP = sample("dZP", Normal(0, self.sigma_grnd))
        mu_host_cepheid = jnp.concatenate(
            [mu_host,
             jnp.array([mu_N4258, mu_LMC + dZP, mu_M31])]
            )

        mu_host = jnp.concatenate(
            [mu_host, jnp.array([mu_N4258, mu_LMC, mu_M31])]
            )

        # Now assign these host distances to each Cepheid.
        mu_cepheid = self.L_Cepheid_host_dist @ mu_host_cepheid
        # Predict the Cepheid magnitudes and compute their likelihood.
        mag_cepheid = mu_cepheid + (M_W + b_W * self.logP + Z_W * self.OH)
        factor(
            "ll_cepheid",
            mvn_logpdf_cholesky(self.mag_cepheid, mag_cepheid, self.L_Cepheid)
            )

        # Distances to the host that have both supernovae and Cepheids
        if include_SN:
            mu_SN_Cepheid = self.L_SN_Cepheid_dist @ mu_host
            mag_SN_Cepheid = mu_SN_Cepheid + M_B

            Y_SN_flow = jnp.ones(self.num_flow_SN) * (M_B - 5 * jnp.log10(H0))
            Y_SN = jnp.concatenate([mag_SN_Cepheid, Y_SN_flow])
            factor(
                "ll_SN",
                mvn_logpdf_cholesky(self.Y_SN, Y_SN, self.L_SN)
                )