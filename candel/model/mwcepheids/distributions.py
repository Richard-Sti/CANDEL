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
"""Custom NumPyro distributions for MW-Cepheid distance priors."""
import jax
import jax.numpy as jnp
import numpy as np
from numpyro import distributions as dist

from ..integration import ln_simpson


class DiskPrior(dist.Distribution):
    """Distance prior following Galactic disk density.

    Models the spatial distribution of Cepheids as an exponential disk:
        p(d | l, b) propto d^2 * exp(-R_GC / R_d) * exp(-|z| / z_d)

    where R_GC is the Galactocentric radius and z is the height above the
    disk.

    Note: The ``sample`` method draws uniformly over [low, high] for fast
    JAX compilation. The disk density shape is enforced via ``log_prob``
    during HMC/NUTS sampling.

    Parameters
    ----------
    ell : float or array
        Galactic longitude in degrees.
    b : float or array
        Galactic latitude in degrees.
    low, high : float
        Distance bounds in kpc.
    R_d : float
        Disk scale length in kpc (default 2.5).
    z_d : float
        Disk scale height in kpc (default 0.1).
    R_sun : float
        Sun's Galactocentric radius in kpc (default 8.122).
    """

    arg_constraints = {"low": dist.constraints.positive,
                       "high": dist.constraints.positive}

    def __init__(self, ell, b, low, high, R_d=2.5, z_d=0.1, R_sun=8.122,
                 validate_args=None):
        self.l_rad = jnp.deg2rad(jnp.atleast_1d(ell))
        self.b_rad = jnp.deg2rad(jnp.atleast_1d(b))
        self.low, self.high = low, high
        self.R_d, self.z_d, self.R_sun = R_d, z_d, R_sun

        # Precompute trig functions - shape (n_stars,)
        self._cos_b = jnp.cos(self.b_rad)
        self._sin_b = jnp.sin(self.b_rad)
        self._cos_l = jnp.cos(self.l_rad)
        self._sin_l = jnp.sin(self.l_rad)

        # Compute normalization numerically per star
        self._log_norm = self._compute_log_norm()

        batch_shape = jnp.broadcast_shapes(jnp.shape(self.l_rad),
                                           jnp.shape(self.b_rad))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def _unnorm_log_prob(self, d, grid_mode=False):
        """Unnormalized log probability.

        Parameters
        ----------
        d : array
            Distances. Shape (n_stars,) for per-star evaluation, or
            (n_grid,) for grid evaluation in grid_mode.
        grid_mode : bool
            If True, evaluate all stars on the distance grid.
            d should have shape (n_grid,), output has shape (n_stars, n_grid).
        """
        if grid_mode:
            # d is (n_grid,), trig values are (n_stars,)
            # Output shape: (n_stars, n_grid)
            cos_b = self._cos_b[:, None]
            sin_b = self._sin_b[:, None]
            cos_l = self._cos_l[:, None]
            sin_l = self._sin_l[:, None]
            d = d[None, :]  # (1, n_grid)
        else:
            # d is (n_stars,), element-wise with trig values (n_stars,)
            cos_b, sin_b = self._cos_b, self._sin_b
            cos_l, sin_l = self._cos_l, self._sin_l

        # Heliocentric Cartesian coordinates
        x = d * cos_b * cos_l  # toward GC
        y = d * cos_b * sin_l  # direction of rotation
        z = d * sin_b          # toward NGP

        # Galactocentric radius
        R_GC = jnp.sqrt((self.R_sun - x)**2 + y**2)

        # Log probability: d^2 * exp(-R_GC/R_d) * exp(-|z|/z_d)
        return 2.0 * jnp.log(d) - R_GC / self.R_d - jnp.abs(z) / self.z_d

    def _compute_log_norm(self, n_grid=1001):
        """Compute log normalization via numerical integration."""
        d_grid = jnp.linspace(self.low, self.high, n_grid)
        log_p = self._unnorm_log_prob(d_grid, grid_mode=True)
        return ln_simpson(log_p, d_grid, axis=-1)

    @dist.constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return dist.constraints.interval(self.low, self.high)

    def sample(self, key, sample_shape=()):
        """Sample uniformly over [low, high].

        Note: This is a simple uniform sample for NUTS initialization.
        The actual posterior is shaped by log_prob during HMC sampling.
        """
        u = jax.random.uniform(key, shape=sample_shape + self.batch_shape)
        return self.low + u * (self.high - self.low)

    def log_prob(self, value):
        return self._unnorm_log_prob(value) - self._log_norm


def sample_disk_sightlines(n, ell_min, ell_max, b_min, b_max,
                           d_min, d_max, R_d, z_d, R_sun, rng,
                           n_oversample_factor=10):
    """Sample (ell, b) proportional to disk column density.

    Generates an oversampled candidate pool uniform in (ell, sin b),
    computes the integrated disk density along each sightline, and
    resamples ``n`` directions proportional to that column density.
    """
    n_over = n * n_oversample_factor

    ell_cand = rng.uniform(ell_min, ell_max, size=n_over)
    sin_b_min = np.sin(np.deg2rad(b_min))
    sin_b_max = np.sin(np.deg2rad(b_max))
    sin_b = rng.uniform(sin_b_min, sin_b_max, size=n_over)
    b_cand = np.rad2deg(np.arcsin(sin_b))

    # Column density = normalization integral of disk prior along each LOS
    disk = DiskPrior(
        jnp.array(ell_cand), jnp.array(b_cand),
        low=d_min, high=d_max,
        R_d=R_d, z_d=z_d, R_sun=R_sun)
    log_Z = np.array(disk._log_norm)

    # Softmax -> probabilities
    log_Z -= log_Z.max()
    p = np.exp(log_Z)
    p /= p.sum()

    idx = rng.choice(n_over, size=n, replace=False, p=p)
    return ell_cand[idx], b_cand[idx]
