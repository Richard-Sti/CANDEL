# Copyright (C) 2024 Richard Stiskalek
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
"""Magnitude selection function."""
from jax import numpy as jnp
from numpyro import factor, plate, sample
from numpyro.distributions import Uniform
from quadax import simpson


class MagnitudeSelection:
    """
    Toy magnitude selection according to Boubel+2024 [1].

    References
    ----------
    [1] https://www.arxiv.org/abs/2408.03660
    """

    def __init__(self):
        self.mrange = jnp.linspace(0, 25, 1000)

    def log_true_pdf(self, m, alpha, m1):
        """Unnormalized `true' PDF."""
        return alpha * (m - m1)

    def log_selection_function(self, m, m1, m2, a):
        """Logarithm of the Boubel+2024 selection function."""
        log_Fm = jnp.where(
            m <= m1,
            0,
            a * (m - m2)**2 - a * (m1 - m2)**2 - 0.6 * (m - m1))
        return jnp.clip(log_Fm, None, 0.)

    def log_observed_pdf(self, m, alpha, m1, m2, a):
        """
        Logarithm of the unnormalized observed PDF, which is the product
        of the true PDF and the selection function.
        """
        y = 10**(
            + self.log_true_pdf(self.mrange, alpha, m1)
            + self.log_selection_function(self.mrange, m1, m2, a)
            )
        norm = simpson(y, x=self.mrange)

        return (self.log_true_pdf(m, alpha, m1)
                + self.log_selection_function(m, m1, m2, a)
                - jnp.log10(norm))

    def __call__(self, mag):
        """NumPyro model, uses an informative prior on `alpha`."""
        alpha = 0.6
        m1 = sample("m1", Uniform(8, 15))
        m2 = sample("m2", Uniform(m1, 20))
        a = sample("a", Uniform(-1, 0))

        with plate("data", len(mag)):
            factor("ll", self.log_observed_pdf(mag, alpha, m1, m2, a))


def log_magnitude_selection(mag, m1, m2, a):
    """
    JAX implementation of `MagnitudeSelection` but natural logarithm,
    whereas the one in `MagnitudeSelection` is base 10.
    """
    Fm = jnp.log(10) * jnp.where(
        mag <= m1,
        0,
        a * (mag - m2)**2 - a * (m1 - m2)**2 - 0.6 * (mag - m1))
    return jnp.clip(Fm, None, 0.)
