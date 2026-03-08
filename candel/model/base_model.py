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
Thin common base class for all forward models (PV and H0).
"""
from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpy as np
from numpyro import factor

from ..cosmography import (Distance2Distmod, Distance2Redshift,
                           Distmod2Distance, Distmod2Redshift,
                           LogGrad_Distmod2ComovingDistance, Redshift2Distance)
from ..util import (fprint, fsection, get_nested, load_config,
                    radec_to_cartesian, replace_prior_with_delta)
from .interp import LOSInterpolator
from .pv_utils import lp_galaxy_bias
from .simpson import ln_simpson_precomputed, simpson_log_weights
from .utils import load_priors, log_prob_integrand_sel, predict_cz


def make_adaptive_grid(r_min, r_max, delta_mu, dr_max):
    """Adaptive radial grid: log-like spacing at small r (constant step in
    distance modulus), transitioning to uniform spacing in r when the
    distance-modulus step would exceed `dr_max`.

    Parameters
    ----------
    r_min, r_max : float
        Radial range in Mpc/h.
    delta_mu : float
        Step size in distance modulus (mag).
    dr_max : float
        Maximum step size in comoving distance (Mpc/h).

    Returns
    -------
    r : 1-D numpy array
        Grid points (odd number for Simpson's rule).
    """
    ln10_over_5 = np.log(10) / 5
    r = [r_min]
    while r[-1] < r_max:
        dr = min(delta_mu * r[-1] * ln10_over_5, dr_max)
        r.append(r[-1] + dr)
    r = np.array(r)

    # Snap last point to r_max, then iteratively drop interior points
    # whose gap to r_max is too small relative to the preceding step
    # (Simpson weights require adjacent spacing ratios < 2).
    r[-1] = r_max
    while len(r) >= 3:
        last_step = r[-1] - r[-2]
        prev_step = r[-2] - r[-3]
        if last_step < 0.5 * prev_step:
            r = np.delete(r, -2)
        else:
            break

    # Ensure odd number of points for Simpson's rule
    if len(r) % 2 == 0:
        r = np.insert(r, -1, 0.5 * (r[-2] + r[-1]))

    return r


class ModelBase(ABC):
    r"""
    Common abstract base class for all forward models (PV and H0).

    Subclasses are expected to implement:
    - ``model()``: The main NumPyro model function that defines the parameters
      and likelihood.
    - Data loading and preprocessing in ``__init__``.

    Attributes
    ----------
    config : dict
        Loaded TOML configuration.
    Om : float
        Matter density parameter :math:`\Omega_m`.
    distance2distmod, distance2redshift, redshift2distance,
    distmod2distance : callable
        JAX-JITted cosmography interpolators.
    """

    def __init__(self, config_path):
        config = load_config(config_path, replace_los_prior=False)
        # SH0ES configs use "Om0", PV configs use "Om".
        self.Om = get_nested(config, "model/Om",
                             get_nested(config, "model/Om0", 0.3))
        self.distance2distmod, self.distance2redshift = \
            Distance2Distmod(Om0=self.Om), Distance2Redshift(Om0=self.Om)
        self.redshift2distance, self.distmod2distance = \
            Redshift2Distance(Om0=self.Om), Distmod2Distance(Om0=self.Om)
        self.config = config

    def _load_and_set_priors(self):
        """Load priors from config and store as attributes."""
        priors = self.config["model"]["priors"]
        self.priors, self.prior_dist_name = load_priors(priors)

    # ------------------------------------------------------------------
    #  Shared data-loading helpers
    # ------------------------------------------------------------------

    def _set_data_arrays(self, data, skip_keys=()):
        """Store data dict entries as JAX attributes.

        - Pops ``None``-valued keys.
        - Converts ``np.ndarray`` to ``jnp.ndarray``.
        - For keys starting with ``e_``, also stores the squared
          version under ``e2_``.
        - Converts RA/dec pairs to unit vectors.
        """
        keys_popped = []
        for key in list(data.keys()):
            if data[key] is None:
                keys_popped.append(key)
                del data[key]
        fprint("Popped the following keys with `None` "
               f"values from data: {', '.join(keys_popped)}")

        attrs_set = []
        for k, v in data.items():
            if k in skip_keys:
                continue

            if isinstance(v, np.ndarray):
                v = jnp.asarray(v)

            setattr(self, k, v)
            attrs_set.append(k)

            if k.startswith("e_"):
                k2 = k.replace("e_", "e2_")
                setattr(self, k2, v * v)
                attrs_set.append(k2)

        def _normalize_rows(x):
            # axis=-1 works for both (n_gal, 3) and (n_sims, n_gal, 3)
            n = jnp.linalg.norm(x, axis=-1, keepdims=True)
            return x / jnp.where(n == 0.0, 1.0, n)

        specs = [
            ("rhat_host",
             ("RA_host", "dec_host"), "host"),
            ("rhat_rand_los",
             ("rand_los_RA", "rand_los_dec"), "random LOS"),
        ]
        for attr, (ra_key, dec_key), label in specs:
            if ra_key in data and dec_key in data:
                ra, dec = data[ra_key], data[dec_key]
                fprint(f"Converting {label} RA/dec to "
                       "Cartesian coordinates.")
                if ra.ndim == 1:
                    assert dec.ndim == 1
                    rhat = radec_to_cartesian(ra, dec)
                elif ra.ndim == 2:
                    # Per-realisation random LOS: (n_sims, n_gal) →
                    # (n_sims, n_gal, 3)
                    assert attr == "rhat_rand_los" and dec.ndim == 2
                    ra_rad = np.deg2rad(ra)
                    dec_rad = np.deg2rad(dec)
                    cos_dec = np.cos(dec_rad)
                    rhat = np.stack([
                        cos_dec * np.cos(ra_rad),
                        cos_dec * np.sin(ra_rad),
                        np.sin(dec_rad),
                    ], axis=-1)
                else:
                    raise ValueError(
                        f"{ra_key} must be 1D or 2D, got {ra.ndim}D")
                setattr(self, attr, _normalize_rows(rhat))
                attrs_set.append(attr)

        fprint("set the following attributes: "
               f"{', '.join(attrs_set)}")

    def _load_los_interpolator(self, data, which="host",
                               r0_decay_scale=5.):
        """Build LOS density/velocity interpolators."""
        if which not in ("host", "rand"):
            raise ValueError(
                "`which` must be either 'host' or 'rand'.")

        los_delta = data[f"{which}_los_density"] - 1
        los_velocity = data[f"{which}_los_velocity"]
        los_r = data[f"{which}_los_r"]

        if which == "host" and "mask_host" in data:
            m = data["mask_host"]
            los_delta = los_delta[:, m, ...]
            los_velocity = los_velocity[:, m, ...]

        # Optionally subsample random LOS
        if which == "rand":
            n_avail = los_delta.shape[1]
            max_rand = get_nested(
                self.config, "model/max_rand_los", None)
            if max_rand is not None and max_rand > n_avail:
                raise ValueError(
                    f"max_rand_los={max_rand} exceeds available "
                    f"random LOS ({n_avail}).")
            if max_rand is not None and max_rand < n_avail:
                gen = np.random.default_rng(42)
                idx = gen.choice(n_avail, max_rand, replace=False)
                idx.sort()
                los_delta = los_delta[:, idx, ...]
                los_velocity = los_velocity[:, idx, ...]
                # Also subsample RA/dec in the data dict
                for k in ("rand_los_RA", "rand_los_dec"):
                    if k in data and data[k] is not None:
                        if data[k].ndim == 1:
                            data[k] = data[k][idx]
                        else:
                            # Per-realisation (n_sims, n_gal)
                            data[k] = data[k][:, idx]
                fprint(f"subsampled random LOS: {n_avail} -> {max_rand}.")

        fprint(f"loaded {which} galaxy LOS interpolators "
               f"for {los_delta.shape[1]} galaxies.")

        kwargs = {"r0_decay_scale": r0_decay_scale}

        setattr(self, f"has_{which}_los", True)
        setattr(
            self, f"f_{which}_los_delta",
            LOSInterpolator(
                los_r, los_delta,
                extrap_constant=0., **kwargs))
        setattr(
            self, f"f_{which}_los_velocity",
            LOSInterpolator(
                los_r, los_velocity,
                extrap_constant=0., **kwargs))

    # ------------------------------------------------------------------
    #  Shared data + grid setup (used by H0 models)
    # ------------------------------------------------------------------

    def _load_data(self, data):
        """Load LOS interpolators and store data arrays.

        Subclasses may override to add post-processing (e.g. Cepheid
        stats) but should call ``super()._load_data(data)`` first.
        """
        self.has_host_los = False
        self.has_rand_los = False
        self.num_rand_los = 1
        self.num_fields = 1

        r0_decay_scale = get_nested(
            self.config, "io/los_r0_decay_scale", 5)
        use_recon = get_nested(
            self.config, "model/use_reconstruction", False)
        if use_recon and get_nested(
                self.config, "io/load_host_los", use_recon):
            self._load_los_interpolator(
                data, which="host",
                r0_decay_scale=r0_decay_scale)
        if use_recon and get_nested(
                self.config, "io/load_rand_los", use_recon):
            self._load_los_interpolator(
                data, which="rand",
                r0_decay_scale=r0_decay_scale)

        self._set_data_arrays(data)

    def _setup_cosmography(self):
        """Set up cosmography interpolators."""
        self.distmod2redshift = Distmod2Redshift(Om0=self.Om)
        self.distmod2distance = Distmod2Distance(Om0=self.Om)
        self.distance2distmod_scalar = Distance2Distmod(
            Om0=self.Om, is_scalar=True)
        self.log_grad_distmod2comoving_distance = \
            LogGrad_Distmod2ComovingDistance(Om0=self.Om)
        self.distmod_limits = self.config["model"]["distmod_limits"]

    def _setup_malmquist_grid(self):
        """Set up the radial grid for Malmquist bias integration."""
        config = self.config
        r_limits = get_nested(
            config, "model/r_limits_malmquist", [0.01, 150])
        r_min, r_max = r_limits

        delta_mu = get_nested(config, "model/delta_mu", None)
        dr_max = get_nested(config, "model/dr_max", None)

        if delta_mu is not None and dr_max is not None:
            r = make_adaptive_grid(r_min, r_max, delta_mu, dr_max)
            self.r_host_range = jnp.asarray(r)
            self._num_points_malmquist = len(r)

            r_cross = dr_max / (delta_mu * np.log(10) / 5)
            dr = np.diff(r)
            n_mu = int(np.sum(r[:-1] < r_cross))
            n_r = len(r) - n_mu
            fprint(
                f"adaptive radial grid: {len(r)} points over "
                f"[{r_min}, {r_max}] Mpc.")
            fprint(
                f"  delta_mu = {delta_mu}, dr_max = {dr_max} Mpc, "
                f"crossover at r = {r_cross:.1f} Mpc.")
            fprint(
                f"  mu-regime: {n_mu} pts "
                f"(dr = {dr[0]:.4f} .. {dr[min(n_mu, len(dr)-1)]:.4f}), "
                f"r-regime: {n_r} pts (dr = {dr_max}).")
        else:
            num_pts = get_nested(
                config, "model/num_points_malmquist", 251)
            if num_pts % 2 == 0:
                raise ValueError(
                    f"num_points_malmquist must be odd for Simpson's "
                    f"rule, got {num_pts}")
            self.r_host_range = jnp.linspace(r_min, r_max, num_pts)
            self._num_points_malmquist = num_pts
            fprint(
                f"uniform radial grid: {num_pts} points over "
                f"[{r_min}, {r_max}] Mpc.")

        self.Rmax = jnp.max(self.r_host_range)
        self._simpson_log_w = simpson_log_weights(self.r_host_range)

        # Separate coarser grid for selection integrals (Malmquist).
        # The selection function is much smoother than the per-host
        # integrand, so a coarser grid suffices.
        delta_mu_sel = get_nested(config, "model/delta_mu_sel", None)
        dr_max_sel = get_nested(config, "model/dr_max_sel", None)
        if delta_mu_sel is not None and dr_max_sel is not None:
            r_sel = make_adaptive_grid(r_min, r_max, delta_mu_sel, dr_max_sel)
            self.r_sel_range = jnp.asarray(r_sel)
            self._simpson_log_w_sel = simpson_log_weights(self.r_sel_range)
            fprint(f"selection grid: {len(r_sel)} points "
                   f"(delta_mu_sel={delta_mu_sel}, dr_max_sel={dr_max_sel}).")
        else:
            self.r_sel_range = self.r_host_range
            self._simpson_log_w_sel = self._simpson_log_w

    def _setup_random_los_grid(self):
        """Set up dummy random LOS when no reconstruction is used."""
        if not self.use_reconstruction and self.apply_sel:
            which_sel = get_nested(
                self.config, "model/which_selection", None)

            if which_sel == "redshift":
                # Redshift selection needs many directions to average
                # over the Vext projection.
                num_rand = get_nested(
                    self.config, "model/num_rand_los_no_recon", 1000)
                fprint(f"setting {num_rand} isotropic random LOS "
                       "(no reconstruction, redshift selection).")
            else:
                # Magnitude selection is isotropic: 1 LOS suffices.
                num_rand = 1
                fprint("setting 1 random LOS "
                       "(no reconstruction, isotropic selection).")
            self.num_rand_los = num_rand
            self.rand_los_density = jnp.ones(
                (1, num_rand, self._num_points_malmquist))
            self.rand_los_velocity = jnp.zeros_like(
                self.rand_los_density)
            gen = np.random.default_rng(42)
            phi = gen.uniform(0, 2 * np.pi, num_rand)
            cos_theta = gen.uniform(-1, 1, num_rand)
            sin_theta = np.sqrt(1 - cos_theta**2)
            rhat = np.stack([sin_theta * np.cos(phi),
                             sin_theta * np.sin(phi),
                             cos_theta], axis=1)
            self.rhat_rand_los = jnp.asarray(rhat)
            self.rand_los_RA = None
            self.rand_los_dec = None

    def _setup_fields_and_bias(self):
        """Count field realizations and set bias clip."""
        if self.use_reconstruction:
            self.br_min_clip = get_nested(
                self.config, "model/galaxy_bias_min_clip",
                1e-5)
            self.num_fields = len(self.host_los_velocity)
            fprint(f"marginalizing over {self.num_fields} "
                   f"field realizations.")

    def _setup_grids(self):
        """Run all grid setup methods."""
        self._setup_cosmography()
        self._setup_malmquist_grid()
        self._setup_random_los_grid()
        self._setup_fields_and_bias()

    # ------------------------------------------------------------------
    #  Distance prior & selection helpers
    # ------------------------------------------------------------------

    def log_prior_distance(self, r):
        """Uniform-in-volume distance prior: p(r) ~ r^2."""
        return (2 * jnp.log(r)
                - 3 * jnp.log(self.Rmax) + jnp.log(3))

    def log_S_cz(self, lp_r, Vpec, H0, sigma_v,
                 cz_lim, cz_width):
        """Selection correction for a redshift-truncated sample."""
        zcosmo = self.distance2redshift(
            self.r_sel_range, h=H0 / 100)
        cz_r = predict_cz(zcosmo[None, None, :], Vpec)
        sigma_v = jnp.asarray(sigma_v)
        while sigma_v.ndim < cz_r.ndim:
            sigma_v = sigma_v[..., None]
        sigma_v = jnp.broadcast_to(sigma_v, cz_r.shape)
        log_prob = log_prob_integrand_sel(
            cz_r, sigma_v, cz_lim, cz_width)
        return ln_simpson_precomputed(
            lp_r + log_prob, self._simpson_log_w_sel, axis=-1)

    def log_S_mag(self, lp_r, M_abs, H0, e_mag,
                  mag_lim, mag_width, mu_grid=None):
        """Selection correction for a magnitude-limited sample."""
        if mu_grid is None:
            mu_grid = self.distance2distmod(
                self.r_sel_range, h=H0 / 100)
        mag = mu_grid + M_abs
        log_prob = log_prob_integrand_sel(
            mag[None, None, :], e_mag, mag_lim, mag_width)
        return ln_simpson_precomputed(
            lp_r + log_prob, self._simpson_log_w_sel, axis=-1)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class H0ModelBase(ModelBase):
    """Intermediate base for H0 models."""

    def __init__(self, config_path, data):
        super().__init__(config_path)
        fsection("Model")
        self._configure_physics()
        self._load_data(data)
        self._setup_grids()
        self._validate_config()
        fname_out = get_nested(self.config, "io/fname_output", None)
        if fname_out is not None:
            fprint(f"output will be saved to `{fname_out}`.")

    def _configure_physics(self):
        config = self.config
        config = self._replace_unused_priors(config)
        self.config = config
        self._load_and_set_priors()
        self._load_selection_thresholds()
        self._load_model_flags()

    def _load_model_flags(self):
        config = self.config
        self.which_selection = get_nested(
            config, "model/which_selection", None)
        fprint(f"which_selection set to {self.which_selection}")
        self.use_reconstruction = get_nested(
            config, "model/use_reconstruction", False)
        fprint(f"use_reconstruction set to {self.use_reconstruction}")
        self.which_bias = get_nested(
            config, "model/which_bias", "linear")
        if self.use_reconstruction:
            fprint(f"which_bias set to {self.which_bias}")
        self.apply_sel = self.which_selection is not None

    def _replace_bias_priors(self, config):
        """Inject delta priors for galaxy bias params if missing."""
        use_reconstruction = get_nested(
            config, "model/use_reconstruction", False)
        priors = config.setdefault(
            "model", {}).setdefault("priors", {})
        bias_defaults = {"b1": 1.0, "b2": 0.0, "alpha": 1.0,
                         "delta_b1": 0.0}
        for param, default in bias_defaults.items():
            if param not in priors:
                priors[param] = {"dist": "delta",
                                 "value": default}
            elif not use_reconstruction:
                replace_prior_with_delta(
                    config, param, default, verbose=False)
        return config

    # ------------------------------------------------------------------
    #  Reconstruction helpers
    # ------------------------------------------------------------------

    def _prepare_selection_grid(self, lp_host_dist_grid, Vext):
        """Prepare random-LOS distance prior grid and Vext projection."""
        if self.apply_sel:
            lp_rand_dist_grid = lp_host_dist_grid
            # Works for rhat_rand_los both (n_los, 3) and (n_sims, n_los, 3)
            Vext_rad_rand = jnp.sum(
                Vext[None, :] * self.rhat_rand_los, axis=-1)
        else:
            lp_rand_dist_grid = 0.
            Vext_rad_rand = 0.
        return lp_rand_dist_grid, Vext_rad_rand

    def _apply_host_reconstruction(self, lp_host_dist, lp_host_dist_grid,
                                   r_host, h, bias_params):
        """Apply galaxy bias to host LOS and normalize."""
        rh_host = r_host * h
        lp_host_dist = lp_host_dist[None, :]
        los_delta_host = self.f_host_los_delta(rh_host)

        _needs_log_rho = "linear" not in self.which_bias
        log_rho_host = (jnp.log(1 + los_delta_host)
                        if _needs_log_rho else None)
        lp_host_dist += lp_galaxy_bias(
            los_delta_host, log_rho_host, bias_params,
            self.which_bias)

        los_delta_grid = \
            self.f_host_los_delta.interp_many_steps_per_galaxy(
                self.r_host_range * h)
        log_rho_grid = (jnp.log(1 + los_delta_grid)
                        if _needs_log_rho else None)
        lp_host_dist_grid += lp_galaxy_bias(
            los_delta_grid, log_rho_grid, bias_params,
            self.which_bias)

        lp_host_dist_norm = ln_simpson_precomputed(
            lp_host_dist_grid, self._simpson_log_w, axis=-1)

        ll_reconstruction = lp_host_dist - lp_host_dist_norm
        lp_host_dist_grid -= lp_host_dist_norm[:, :, None]

        return ll_reconstruction, lp_host_dist_grid, los_delta_host, rh_host

    def _apply_rand_reconstruction(self, lp_rand_dist_grid, h, bias_params):
        """Apply galaxy bias to random LOS and normalize."""
        rand_los_delta_grid = \
            self.f_rand_los_delta.interp_many_steps_per_galaxy(
                self.r_host_range * h)

        log_rho = (jnp.log(1 + rand_los_delta_grid)
                   if "linear" not in self.which_bias else None)
        lp_rand_dist_grid += lp_galaxy_bias(
            rand_los_delta_grid, log_rho,
            bias_params, self.which_bias)
        log_Z = ln_simpson_precomputed(
            lp_rand_dist_grid, self._simpson_log_w, axis=-1)
        lp_rand_dist_grid -= log_Z[..., None]

        rand_los_Vpec_grid = \
            self.f_rand_los_velocity.interp_many_steps_per_galaxy(
                self.r_host_range * h)

        return lp_rand_dist_grid, rand_los_delta_grid, rand_los_Vpec_grid, \
            log_Z

    def _no_reconstruction_fallback(self, lp_host_dist, lp_host_dist_grid):
        """Handle the no-reconstruction branch."""
        lp_host_dist_grid = jnp.repeat(
            lp_host_dist_grid, self.num_hosts, axis=1)
        factor("lp_host_dist", lp_host_dist)
        return lp_host_dist_grid
