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

H0 selection-integral convention
--------------------------------
For H0 models with an explicit sample selection (`which_selection` is set)
and a reconstruction, the population selection normalizer is evaluated
directly on the 3D reconstruction grid. Without a reconstruction, the model
falls back to the radial volume integral with n = 1 and no reconstructed
peculiar-velocity field.

The ideal target is an integral over all space. In practice, the reconstruction
field is finite, so the model uses a fixed computational truncation specified
by `model.selection_integral_grid_radius` in Mpc/h. Its geometry is controlled
by `model.selection_integral_geometry`:

    "sphere": keep voxels with non-zero fractional volume inside R_grid and
              flatten them to 1D arrays. This drops cube corners and is
              usually cheaper.
    "cube":   keep the full cubic sub-grid with half-side R_grid. This is
              simpler geometrically and can be useful for cross-checks.

The likelihood sums over all loaded voxels in either representation. The
truncation radius is therefore a numerical accuracy choice, not a physical
model boundary: it should be increased until `log S` is stable.
Conceptually, the density outside the loaded reconstruction can be treated as
mean density (`n_i = 1`), but the practical requirement is stricter: for
posterior-relevant selection parameters, the selection probability should be
negligible before the truncation boundary so that this omitted tail does not
change the normalizer.

Reconstruction fields are stored in Mpc/h. At each sampled H0, with

    h = H0 / 100,

the model uses

    r_phys = r_grid / h,
    dV_phys = dV_grid / h^3.

The selection normalizer is then evaluated as

    S = sum_i P_sel,i n_i dV_phys,

or in log space,

    log S = logsumexp_i(log P_sel,i + log n_i) + log dV_phys.

Here `P_sel,i` is the selection probability at voxel i and `n_i` is the
density/galaxy-bias factor. Since the computational sphere is intended to
approximate infinity, the exact outer boundary is acceptable only if the
omitted tail has negligible selection weight.

Example: magnitude selection
----------------------------
For a magnitude-limited sample, each voxel is assigned a distance modulus using
the reconstruction-coordinate radius and the sampled H0,

    mu_i(H0) = mu_i(h=1) - 5 log10(h).

For an object population with absolute magnitude M and effective magnitude
scatter/error sigma_m, the apparent magnitude at voxel i is

    m_i = mu_i(H0) + M.

The smooth selection probability is then the same softened threshold used
elsewhere in the H0 models,

    P_sel,i = Phi((m_lim - m_i) / sigma_m),

with the implementation provided by `log_prob_integrand_sel`, allowing the
configured magnitude limit and width to be sampled or fixed. The magnitude
selection normalizer is therefore

    S_mag = sum_i Phi((m_lim - mu_i(H0) - M) / sigma_m) n_i dV_grid / h^3,

or

    log S_mag = logsumexp_i(log P_sel,i + log n_i)
                + log dV_grid - 3 log h.

The practical convergence check is to recompute `log S` with larger
`selection_integral_grid_radius`; if the changes are negligible, the loaded
grid is a sufficient approximation to the integral to infinity. This
convergence requirement is for posterior-relevant parameter values rather than
necessarily every extreme point in a deliberately broad prior.
"""
from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import logsumexp
from numpyro import factor

from ..cosmo.cosmography import (Distance2Distmod, Distance2Redshift,
                                 Distmod2Distance, Distmod2Redshift,
                                 LogGrad_Distmod2ComovingDistance,
                                 Redshift2Distance)
from ..util import (fprint, fsection, get_nested, load_config,
                    radec_to_cartesian, replace_prior_with_delta)
from .integration import ln_simpson_precomputed, simpson_log_weights
from .interp import LOSInterpolator
from .pv_utils import (lp_galaxy_bias, octupole_radial, quadrupole_radial,
                       rsample, sample_octupole, sample_quadrupole,
                       sigmoid_monopole_radial)
from .utils import (load_priors, log_prob_integrand_sel, logmeanexp,
                    normal_logpdf_var, predict_cz, student_t_logpdf_var)

# ICRS equatorial → Galactic Cartesian rotation matrix.
# Computed via astropy: SkyCoord(basis, frame='icrs').galactic.
_R_ICRS_TO_GAL = np.array([
    [-0.05487565771259163, -0.87343705195561590, -0.48383507361671546],
    [+0.49410943719272680, -0.44482972122329520, +0.74698218398666760],
    [-0.86766613755965760, -0.19807633727300053, +0.45598381368730160]])

# ICRS equatorial -> Supergalactic Cartesian rotation matrix.
# Computed via astropy: SkyCoord(basis, frame='icrs').supergalactic.
_R_ICRS_TO_SUPERGAL = np.array([
    [+0.37501555570303163, +0.34135887185624750, +0.86188018516831910],
    [-0.89832043772761380, -0.09572710024885137, +0.42878516000301936],
    [+0.22887490937543750, -0.93504569026490690, +0.27075049949244600]])


def make_adaptive_grid(r_min, r_max, delta_mu, dr_max):
    """Adaptive radial grid: log-like spacing at small r (constant step in
    distance modulus), transitioning to uniform spacing in r when the
    distance-modulus step would exceed `dr_max`.

    Parameters
    ----------
    r_min, r_max : float
        Radial range, in the caller's distance units.
    delta_mu : float
        Step size in distance modulus (mag).
    dr_max : float
        Maximum step size in the same units as ``r_min`` and ``r_max``.

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
        self.compute_evidence = bool(
            get_nested(config, "inference/compute_evidence", True))

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
        use_3d_selection_integral = (
            get_nested(self.config, "model/which_selection", None)
            is not None)
        if use_recon and get_nested(
                self.config, "io/load_host_los", use_recon):
            self._load_los_interpolator(
                data, which="host",
                r0_decay_scale=r0_decay_scale)
        if (use_recon and not use_3d_selection_integral and get_nested(
                self.config, "io/load_rand_los", use_recon)):
            self._load_los_interpolator(
                data, which="rand",
                r0_decay_scale=r0_decay_scale)

        self._set_data_arrays(data)

        if data.get("has_volume_density_3d", False):
            self.has_volume_density_3d = True
            self.density_3d_mode = data.get("density_3d_mode", "delta")
            self.volume_density_batch_size = data.get(
                "volume_density_batch_size", 1)
            self.coordinate_frame_3d = data.get(
                "coordinate_frame_3d", "icrs")
        else:
            self.has_volume_density_3d = False

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

        # Separate grid for selection integrals (Malmquist).
        # The selection function is smoother than the per-host integrand,
        # so a coarser/shorter grid often suffices. Uses `dr_malmquist`
        # for uniform spacing.
        r_max_sel = get_nested(
            config, "model/r_max_selection", r_max)
        dr_malmquist = get_nested(config, "model/dr_malmquist", None)
        delta_mu_sel = get_nested(config, "model/delta_mu_sel", None)
        dr_max_sel = get_nested(config, "model/dr_max_sel", None)

        if delta_mu_sel is not None and dr_max_sel is not None:
            r_sel = make_adaptive_grid(
                r_min, r_max_sel, delta_mu_sel, dr_max_sel)
            self.r_sel_range = jnp.asarray(r_sel)
            self._simpson_log_w_sel = simpson_log_weights(self.r_sel_range)
            fprint(f"selection grid: {len(r_sel)} points over "
                   f"[{r_min}, {r_max_sel}] Mpc "
                   f"(delta_mu_sel={delta_mu_sel}, dr_max_sel={dr_max_sel}).")
        elif dr_malmquist is not None or r_max_sel != r_max:
            if dr_malmquist is None:
                dr_malmquist = float(
                    (r_max - r_min) / (self._num_points_malmquist - 1))
            num_pts_sel = int(np.round(
                (r_max_sel - r_min) / dr_malmquist)) + 1
            # Simpson's rule requires an odd number >= 3
            num_pts_sel = max(num_pts_sel, 3)
            if num_pts_sel % 2 == 0:
                num_pts_sel += 1
            self.r_sel_range = jnp.linspace(r_min, r_max_sel, num_pts_sel)
            self._simpson_log_w_sel = simpson_log_weights(self.r_sel_range)
            dr_actual = float((r_max_sel - r_min) / (num_pts_sel - 1))
            fprint(f"selection grid: {num_pts_sel} points over "
                   f"[{r_min}, {r_max_sel}] Mpc "
                   f"(dr={dr_actual:.3f}).")
        else:
            self.r_sel_range = self.r_host_range
            self._simpson_log_w_sel = self._simpson_log_w

    def _setup_no_recon_direction_grid(self):
        """Set up isotropic directions for no-reconstruction redshift cuts."""
        if self.use_reconstruction or not self.apply_sel:
            return

        which_sel = get_nested(
            self.config, "model/which_selection", None)
        if which_sel == "SN_magnitude_or_redshift_Nmag":
            n_mag = get_nested(
                self.config, "model/num_hosts_selection_mag", None)
            if type(n_mag) is int and n_mag >= self.num_hosts:
                return
        if which_sel not in (
                "redshift", "SN_magnitude_redshift",
                "SN_magnitude_or_redshift_Nmag"):
            return

        num_dirs = get_nested(
            self.config, "model/num_no_recon_directions",
            get_nested(self.config, "model/num_rand_los_no_recon", 1000))
        fprint(f"setting {num_dirs} isotropic directions "
               "(no reconstruction, redshift selection).")
        self.num_no_recon_directions = num_dirs
        gen = np.random.default_rng(42)
        phi = gen.uniform(0, 2 * np.pi, num_dirs)
        cos_theta = gen.uniform(-1, 1, num_dirs)
        sin_theta = np.sqrt(1 - cos_theta**2)
        rhat = np.stack([sin_theta * np.cos(phi),
                         sin_theta * np.sin(phi),
                         cos_theta], axis=1)
        self.rhat_no_recon_directions = jnp.asarray(rhat)

    def _setup_fields_and_bias(self):
        """Count field realizations."""
        if self.use_reconstruction:
            self.num_fields = len(self.host_los_velocity)
            fprint(f"marginalizing over {self.num_fields} "
                   f"field realizations.")

    def _setup_grids(self):
        """Run all grid setup methods."""
        self._setup_cosmography()
        self._setup_malmquist_grid()
        self._setup_no_recon_direction_grid()
        self._setup_fields_and_bias()

    # ------------------------------------------------------------------
    #  Distance prior & selection helpers
    # ------------------------------------------------------------------

    def log_prior_distance(self, r):
        """Unnormalized uniform-in-volume distance prior: p(r) ~ r^2."""
        return 2 * jnp.log(r)

    def log_S_cz(self, lp_r, Vpec, H0, sigma_v,
                 cz_lim, cz_width, nu_cz=None):
        """Selection correction for a redshift-truncated sample."""
        zcosmo = self.distance2redshift(
            self.r_sel_range, h=H0 / 100)
        cz_r = predict_cz(zcosmo[None, None, :], Vpec)
        sigma_v = jnp.asarray(sigma_v)
        while sigma_v.ndim < cz_r.ndim:
            sigma_v = sigma_v[None, ...]
        sigma_v = jnp.broadcast_to(sigma_v, cz_r.shape)
        log_prob = log_prob_integrand_sel(
            cz_r, sigma_v, cz_lim, cz_width, nu_cz=nu_cz)
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
        fsection(f"Model: {type(self).__name__}")
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

    def _replace_unused_priors(self, config):
        use_reconstruction = get_nested(
            config, "model/use_reconstruction", False)
        if not use_reconstruction:
            replace_prior_with_delta(config, "beta", 0.0)
        config = self._replace_bias_priors(config)
        config = self._replace_cz_likelihood_priors(config)
        return config

    def _replace_cz_likelihood_priors(self, config):
        """Replace robust cz-likelihood priors when they are inactive."""
        if get_nested(config, "model/cz_likelihood", "gaussian") \
                != "student_t":
            replace_prior_with_delta(
                config, "nu_cz", 30.0, verbose=False)
        return config

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
        # Monopole: "none", "constant", "sigmoid", or legacy bool
        _mono_raw = get_nested(
            config, "model/which_Vext_monopole", "none")
        # Backward compat: treat True → "constant", False/None → "none"
        if _mono_raw is True:
            _mono_raw = "constant"
        elif _mono_raw is False or _mono_raw is None:
            _mono_raw = "none"
        # Also support legacy bool key
        if _mono_raw == "none" and get_nested(
                config, "model/use_Vext_monopole", False):
            _mono_raw = "constant"
        if _mono_raw not in ("none", "constant", "sigmoid"):
            raise ValueError(
                f"Invalid which_Vext_monopole: '{_mono_raw}'. "
                "Expected 'none', 'constant', or 'sigmoid'.")
        self.which_Vext_monopole = _mono_raw
        if self.which_Vext_monopole != "none":
            fprint(f"which_Vext_monopole set to {self.which_Vext_monopole}")
        self.use_Vext_quadrupole = get_nested(
            config, "model/use_Vext_quadrupole", False)
        if self.use_Vext_quadrupole:
            quad_prior = get_nested(
                config, "model/priors/Vext_quad", None)
            if quad_prior is None:
                raise ValueError(
                    "`use_Vext_quadrupole` requires "
                    "[model.priors.Vext_quad].")
            self.Vext_quad_mag_range = (quad_prior["low"],
                                        quad_prior["high"])
            fprint(f"use_Vext_quadrupole set to True "
                   f"(mag range: {self.Vext_quad_mag_range})")
        self.use_Vext_octupole = get_nested(
            config, "model/use_Vext_octupole", False)
        if self.use_Vext_octupole:
            oct_prior = get_nested(
                config, "model/priors/Vext_oct", None)
            if oct_prior is None:
                raise ValueError(
                    "`use_Vext_octupole` requires "
                    "[model.priors.Vext_oct].")
            self.Vext_oct_mag_range = (oct_prior["low"],
                                       oct_prior["high"])
            fprint(f"use_Vext_octupole set to True "
                   f"(mag range: {self.Vext_oct_mag_range})")
        self.apply_sel = self.which_selection is not None

        self.selection_integral_grid_radius = get_nested(
            config, "model/selection_integral_grid_radius", None)
        self.selection_integral_geometry = get_nested(
            config, "model/selection_integral_geometry", "sphere")
        if self.selection_integral_geometry not in ("sphere", "cube"):
            raise ValueError(
                "`model.selection_integral_geometry` must be 'sphere' "
                "or 'cube'.")
        if self.apply_sel and self.use_reconstruction:
            fprint("using 3D selection integral")

        # Robust velocity-error modelling options
        self.cz_likelihood = get_nested(
            config, "model/cz_likelihood", "gaussian")
        if self.cz_likelihood not in ("gaussian", "student_t"):
            raise ValueError(
                f"Invalid cz_likelihood: '{self.cz_likelihood}'. "
                "Expected 'gaussian' or 'student_t'.")
        if self.cz_likelihood != "gaussian":
            fprint(f"cz_likelihood set to {self.cz_likelihood}")

    def _load_selection_thresholds(self, active_map, spec):
        config = self.config
        priors = config.setdefault(
            "model", {}).setdefault("priors", {})
        which_sel = get_nested(config, "model/which_selection", None)
        active = active_map.get(which_sel, set())

        for name, default in spec.items():
            if name not in active:
                setattr(self, name, None)
                setattr(self, f"_infer_{name}", False)
                continue

            raw = get_nested(config, f"model/{name}", default)
            if raw == "infer":
                p = priors.get(name)
                if p is None:
                    raise ValueError(
                        f"`{name}` set to 'infer' but no "
                        f"prior [model.priors.{name}] found.")
                setattr(self, name, None)
                setattr(self, f"_infer_{name}", True)
                fprint(f"{name} will be inferred.")
            else:
                setattr(self, name, raw)
                setattr(self, f"_infer_{name}", False)

    def _resolve_threshold(self, name):
        if getattr(self, f"_infer_{name}"):
            return rsample(name, self.priors[name])
        return getattr(self, name)

    def _sample_nu_cz(self):
        """Sample Student-t cz degrees of freedom when enabled."""
        if self.cz_likelihood == "student_t":
            return rsample("nu_cz", self.priors["nu_cz"])
        return None

    def _cz_logpdf_fn(self, nu_cz):
        """Return the configured scalar cz log-likelihood kernel."""
        if nu_cz is None:
            return normal_logpdf_var

        def ll_cz_fn(obs, pred, var):
            return student_t_logpdf_var(obs, pred, var, nu_cz)

        return ll_cz_fn

    def _sample_external_velocity(self):
        """Sample shared external-velocity components."""
        Vext = rsample("Vext", self.priors["Vext"])
        Vext_quad = None
        if self.use_Vext_quadrupole:
            Vext_quad = sample_quadrupole(
                "Vext_quad", *self.Vext_quad_mag_range)
        Vext_oct = None
        if self.use_Vext_octupole:
            Vext_oct = sample_octupole(
                "Vext_oct", *self.Vext_oct_mag_range)

        Vext_mono = None
        if self.which_Vext_monopole == "constant":
            Vext_mono = rsample("Vext_mono", self.priors["Vext_mono"])
        elif self.which_Vext_monopole == "sigmoid":
            Vext_mono_left = rsample(
                "Vext_mono_left", self.priors["Vext_mono_left"])
            Vext_mono_rt = rsample(
                "Vext_mono_rt", self.priors["Vext_mono_rt"])
            Vext_mono_angle = rsample(
                "Vext_mono_angle", self.priors["Vext_mono_angle"])
            Vext_mono = (Vext_mono_left, Vext_mono_rt, Vext_mono_angle)

        return Vext, Vext_quad, Vext_oct, Vext_mono

    def _host_Vext_radial(self, Vext, Vext_quad=None, Vext_oct=None):
        """Project external multipoles onto host directions."""
        Vext_rad = jnp.sum(Vext[None, :] * self.rhat_host, axis=-1)
        if Vext_quad is not None:
            Q_mag, q1_hat, q2_hat = Vext_quad
            Vext_rad = Vext_rad + quadrupole_radial(
                Q_mag, q1_hat, q2_hat, self.rhat_host)
        if Vext_oct is not None:
            O_mag, o1_hat, o2_hat, o3_hat = Vext_oct
            Vext_rad = Vext_rad + octupole_radial(
                O_mag, o1_hat, o2_hat, o3_hat, self.rhat_host)
        return Vext_rad

    def _Vext_monopole_radial(self, Vext_mono, r):
        """Evaluate the shared radial monopole on distances ``r``."""
        if isinstance(Vext_mono, tuple):
            V_left, r_t, angle = Vext_mono
            k = jnp.tan(angle)
            return sigmoid_monopole_radial(V_left, r_t, k, r)
        if Vext_mono is not None:
            return jnp.broadcast_to(Vext_mono, jnp.shape(r))
        return None

    def _replace_bias_priors(self, config):
        """Inject delta priors for galaxy bias params if missing.

        Required params for the active bias model must have explicit priors;
        a missing prior raises ``ValueError``.  All other bias params get
        silent delta defaults so that they exist but don't affect sampling.
        """
        use_reconstruction = get_nested(
            config, "model/use_reconstruction", False)
        which_bias = get_nested(config, "model/which_bias", "linear")
        priors = config.setdefault(
            "model", {}).setdefault("priors", {})

        # Params required by each bias model.
        _required = {
            "unity": set(),
            "powerlaw": {"alpha"},
            "linear": {"b1"},
            "linear_from_beta": set(),
            "linear_from_beta_stochastic": {"delta_b1"},
            "double_powerlaw": {"alpha_low", "alpha_high", "log_rho_t"},
            "quadratic": {"b1", "b2"},
            "cubic": {"b1", "b2", "b3"},
        }
        required = _required.get(which_bias, set())

        if use_reconstruction:
            for param in required:
                if param not in priors:
                    raise ValueError(
                        f"Bias model '{which_bias}' requires prior "
                        f"[model.priors.{param}] but none was found.")

        bias_defaults = {"b1": 1.0, "b2": 0.0, "b3": 0.0, "alpha": 1.0,
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

    def _apply_host_reconstruction(self, lp_host_dist, r_host, h,
                                   bias_params):
        """Apply galaxy bias to host LOS (unnormalized).

        The distance prior is NOT normalized because the angular prior
        π(ℓ,b) ∝ Z_i(ℓ,b) cancels the normalization constant Z_i.
        """
        rh_host = r_host * h
        lp_host_dist = lp_host_dist[None, :]
        los_delta_host = self.f_host_los_delta(rh_host)

        _needs_log_rho = "linear" not in self.which_bias
        log_rho_host = (jnp.log(1 + los_delta_host)
                        if _needs_log_rho else None)
        lp_host_dist += lp_galaxy_bias(
            los_delta_host, log_rho_host, bias_params,
            self.which_bias)

        return lp_host_dist, los_delta_host, rh_host

    def _no_reconstruction_fallback(self, lp_host_dist):
        """Handle the no-reconstruction branch."""
        factor("lp_host_dist", lp_host_dist)

    def _validate_selection_integral(self, needs_velocity=False):
        """Validate common 3D selection-integral requirements."""
        if not self.apply_sel:
            return
        if self.use_Vext_quadrupole or self.use_Vext_octupole:
            raise ValueError(
                "Vext quadrupole/octupole are not supported with "
                "selection integrals.")
        if not self.use_reconstruction:
            return
        if self.selection_integral_grid_radius is None:
            raise ValueError(
                "3D selection integrals require explicit "
                "`model.selection_integral_grid_radius` in Mpc/h.")
        if not self.has_volume_density_3d:
            raise ValueError(
                "3D selection integrals require 3D density data.")
        n_3d = self.density_3d_fields.shape[0]
        if n_3d != self.num_fields:
            raise ValueError(
                f"Number of 3D density fields ({n_3d}) does not match LOS "
                f"field realisations ({self.num_fields}).")
        if needs_velocity and not hasattr(self, "vrad_3d_fields"):
            raise ValueError(
                "This selection integral requires 3D velocity data.")

    def _volume_log_dV_phys(self, H0):
        """Log physical voxel volume from reconstruction-coordinate volume."""
        return self.log_dV_3d - 3 * jnp.log(H0 / 100)

    def _volume_log_cell_weight_phys(self, H0):
        """Log physical voxel volume including fractional boundary weights."""
        return (self._volume_log_dV_phys(H0)
                + getattr(self, "log_volume_weight_3d", 0.0))

    def _vol_sel_galaxy_bias(self, density_3d, bias_params):
        """Apply galaxy bias to a 3D density field."""
        if self.density_3d_mode == "log_rho":
            return lp_galaxy_bias(
                0.0, density_3d, bias_params, self.which_bias)
        return lp_galaxy_bias(
            density_3d, 0.0, bias_params, self.which_bias)

    def _vol_sel_Vext_rad_3d(self, Vext, Vext_mono, h):
        """Project Vext onto each voxel direction in the field frame."""
        if self.coordinate_frame_3d == "galactic":
            Vext = jnp.asarray(_R_ICRS_TO_GAL) @ Vext
        elif self.coordinate_frame_3d == "supergalactic":
            Vext = jnp.asarray(_R_ICRS_TO_SUPERGAL) @ Vext
        elif self.coordinate_frame_3d != "icrs":
            raise ValueError(
                f"3D selection integrals do not support coordinate frame "
                f"'{self.coordinate_frame_3d}'.")
        Vext_rad = (Vext[0] * self.rhat_x_3d
                    + Vext[1] * self.rhat_y_3d
                    + Vext[2] * self.rhat_z_3d)
        if isinstance(Vext_mono, tuple):
            V_left, r_t, angle = Vext_mono
            k = jnp.tan(angle)
            # `r_t` is in Mpc (matches host loop); voxel radii are in Mpc/h.
            Vext_rad = Vext_rad + sigmoid_monopole_radial(
                V_left, r_t, k, jnp.exp(self.log_r_3d) / h)
        elif Vext_mono is not None:
            Vext_rad = Vext_rad + Vext_mono
        return Vext_rad

    def _vol_sel_sigma_v_fields(self, sigma_v):
        """Broadcast scalar or voxel-level sigma_v to all 3D fields."""
        sigma_v = jnp.asarray(sigma_v)
        target_shape = self.density_3d_fields.shape
        if sigma_v.ndim == 0:
            return jnp.broadcast_to(sigma_v, target_shape)
        if sigma_v.shape == target_shape:
            return sigma_v
        if sigma_v.shape == target_shape[:1]:
            shape = target_shape[:1] + (1,) * (len(target_shape) - 1)
            return jnp.broadcast_to(sigma_v.reshape(shape), target_shape)
        return jnp.broadcast_to(sigma_v, target_shape)

    def _no_recon_selection_Vpec(self, Vext, Vext_mono):
        """Selection-grid Vpec for no-reconstruction H0 selection."""
        Vext_rad = jnp.sum(
            Vext[None, :] * self.rhat_no_recon_directions, axis=-1)
        Vpec = Vext_rad[None, :, None]
        if isinstance(Vext_mono, tuple):
            V_left, r_t, angle = Vext_mono
            k = jnp.tan(angle)
            Vpec = Vpec + sigmoid_monopole_radial(
                V_left, r_t, k, self.r_sel_range)[None, None, :]
        elif Vext_mono is not None:
            Vpec = Vpec + Vext_mono
        return Vpec

    def _compute_volume_log_S_unity(self, bias_params, H0):
        """Volume normalizer for a hard ``P_sel = 1`` radial cut."""
        grid_radius = self.selection_integral_grid_radius
        if grid_radius is None:
            raise ValueError(
                "`model.selection_integral_grid_radius` is required for "
                "volume-limited no-selection CH0 runs.")

        h = H0 / 100
        r_low = self.distmod2distance(
            jnp.asarray([self.distmod_limits[0]]), h=h)[0]
        if not self.use_reconstruction:
            r_high = grid_radius / h
            volume = (r_high**3 - r_low**3) / 3.0
            return jnp.reshape(jnp.log(volume), (1,))

        log_cell_weight = self._volume_log_cell_weight_phys(H0)
        rh_low = r_low * h
        in_volume = (
            (self.log_r_3d >= jnp.log(rh_low))
            & (self.log_r_3d <= jnp.log(grid_radius))
        )

        def _one(density_3d):
            log_n = self._vol_sel_galaxy_bias(density_3d, bias_params)
            return logsumexp(jnp.where(in_volume, log_n + log_cell_weight,
                                       -jnp.inf))

        return lax.map(_one, self.density_3d_fields,
                       batch_size=self.volume_density_batch_size)

    def _compute_no_recon_log_S_cz(self, H0, sigma_v, Vext, Vext_mono,
                                   cz_lim, cz_width, nu_cz=None):
        """No-reconstruction redshift selection with n=1 and v_rec=0."""
        lp_r = self.log_prior_distance(self.r_sel_range)[None, None, :]
        Vpec = self._no_recon_selection_Vpec(Vext, Vext_mono)
        log_S = self.log_S_cz(
            lp_r, Vpec, H0, sigma_v, cz_lim, cz_width, nu_cz=nu_cz)
        # `log_S_cz` has already integrated over radius. The remaining axis
        # is the isotropic angular average over Vext projections.
        return logmeanexp(log_S, axis=-1).reshape(-1)

    def _compute_no_recon_log_S_mag_cz(self, M_abs, e_mag, H0, sigma_v,
                                       Vext, Vext_mono, mag_lim, mag_width,
                                       cz_lim, cz_width, nu_cz=None):
        """No-reconstruction combined magnitude and redshift selection."""
        h = H0 / 100
        lp_r = self.log_prior_distance(self.r_sel_range)[None, None, :]
        mu_grid = self.distance2distmod(self.r_sel_range, h=h)
        zcosmo = self.distance2redshift(self.r_sel_range, h=h)
        Vpec = self._no_recon_selection_Vpec(Vext, Vext_mono)
        cz_r = predict_cz(zcosmo[None, None, :], Vpec)

        sigma_v = jnp.asarray(sigma_v)
        while sigma_v.ndim < cz_r.ndim:
            sigma_v = sigma_v[None, ...]
        sigma_v = jnp.broadcast_to(sigma_v, cz_r.shape)

        log_prob = log_prob_integrand_sel(
            (mu_grid + M_abs)[None, None, :],
            e_mag, mag_lim, mag_width)
        log_prob += log_prob_integrand_sel(
            cz_r, sigma_v, cz_lim, cz_width, nu_cz=nu_cz)
        log_S = ln_simpson_precomputed(
            lp_r + log_prob, self._simpson_log_w_sel, axis=-1)
        # The Simpson rule above has already integrated over radius. The
        # remaining axis is the isotropic angular average.
        return logmeanexp(log_S, axis=-1).reshape(-1)

    def _compute_volume_log_S_mag(self, bias_params, M_abs, e_mag, H0,
                                  mag_lim, mag_width):
        """3D selection integral for a magnitude-limited sample."""
        if not self.use_reconstruction:
            lp_r = self.log_prior_distance(self.r_sel_range)[None, None, :]
            log_S = self.log_S_mag(
                lp_r, M_abs, H0, e_mag, mag_lim, mag_width)
            # `log_S_mag` has already integrated over radius.  Magnitude-only
            # selection is isotropic, so the LOS axis is a singleton here.
            return logmeanexp(log_S, axis=-1).reshape(-1)

        h = H0 / 100
        mu_3d = self.mu_at_h1_3d - 5 * jnp.log10(h)
        log_P_sel = log_prob_integrand_sel(
            mu_3d + M_abs, e_mag, mag_lim, mag_width)
        log_cell_weight = self._volume_log_cell_weight_phys(H0)

        def _one(density_3d):
            log_n = self._vol_sel_galaxy_bias(density_3d, bias_params)
            return logsumexp(log_P_sel + log_n + log_cell_weight)

        return lax.map(_one, self.density_3d_fields,
                       batch_size=self.volume_density_batch_size)

    def _compute_volume_log_S_cz(self, bias_params, H0, sigma_v, beta,
                                 Vext, Vext_mono, cz_lim, cz_width,
                                 nu_cz=None):
        """3D selection integral for a redshift-limited sample."""
        if not self.use_reconstruction:
            return self._compute_no_recon_log_S_cz(
                H0, sigma_v, Vext, Vext_mono, cz_lim, cz_width,
                nu_cz=nu_cz)

        log_cell_weight = self._volume_log_cell_weight_phys(H0)
        Vext_rad_3d = self._vol_sel_Vext_rad_3d(Vext, Vext_mono, H0 / 100)
        sigma_v = self._vol_sel_sigma_v_fields(sigma_v)

        def _one(inputs):
            density_3d, vrad_3d, sigma_v_3d = inputs
            log_n = self._vol_sel_galaxy_bias(density_3d, bias_params)
            Vpec = beta * vrad_3d + Vext_rad_3d
            cz_pred = predict_cz(self.zcosmo_3d, Vpec)
            log_P_sel = log_prob_integrand_sel(
                cz_pred, sigma_v_3d, cz_lim, cz_width, nu_cz=nu_cz)
            return logsumexp(log_P_sel + log_n + log_cell_weight)

        return lax.map(
            _one, (self.density_3d_fields, self.vrad_3d_fields, sigma_v),
            batch_size=self.volume_density_batch_size)

    def _compute_volume_log_S_mag_cz(self, bias_params, M_abs, e_mag, H0,
                                     sigma_v, beta, Vext, Vext_mono,
                                     mag_lim, mag_width,
                                     cz_lim, cz_width, nu_cz=None):
        """3D selection integral for combined magnitude and redshift cuts."""
        if not self.use_reconstruction:
            return self._compute_no_recon_log_S_mag_cz(
                M_abs, e_mag, H0, sigma_v, Vext, Vext_mono,
                mag_lim, mag_width, cz_lim, cz_width, nu_cz=nu_cz)

        h = H0 / 100
        mu_3d = self.mu_at_h1_3d - 5 * jnp.log10(h)
        log_P_mag = log_prob_integrand_sel(
            mu_3d + M_abs, e_mag, mag_lim, mag_width)
        log_cell_weight = self._volume_log_cell_weight_phys(H0)
        Vext_rad_3d = self._vol_sel_Vext_rad_3d(Vext, Vext_mono, h)
        sigma_v = self._vol_sel_sigma_v_fields(sigma_v)

        def _one(inputs):
            density_3d, vrad_3d, sigma_v_3d = inputs
            log_n = self._vol_sel_galaxy_bias(density_3d, bias_params)
            Vpec = beta * vrad_3d + Vext_rad_3d
            cz_pred = predict_cz(self.zcosmo_3d, Vpec)
            log_P_cz = log_prob_integrand_sel(
                cz_pred, sigma_v_3d, cz_lim, cz_width, nu_cz=nu_cz)
            return logsumexp(
                log_P_mag + log_P_cz + log_n + log_cell_weight)

        return lax.map(
            _one, (self.density_3d_fields, self.vrad_3d_fields, sigma_v),
            batch_size=self.volume_density_batch_size)
