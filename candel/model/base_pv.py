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
Base classes for peculiar velocity (PV) forward models.
"""
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from numpyro import deterministic, factor, handlers

from ..util import fprint, fsection, get_nested
from .base_model import ModelBase
from .integration import simpson_log_weights
from .pv_utils import (_rsample, compute_Vext_radial, lp_galaxy_bias, rsample,
                       sample_distance_prior, sample_galaxy_bias, sample_Vext,
                       sigma_v_from_density, sumzero_basis)
from .utils import (config_hash, log_prior_r_empirical, normal_logpdf_var,
                    predict_cz)


class BasePVModel(ModelBase):
    """
    Base class for Peculiar Velocity (PV) forward models.

    This class provides common infrastructure for models that involve
    distance-indicator observables and peculiar velocities derived from
    reconstructed density/velocity fields or external dipoles.

    It handles:
    - Loading PV-specific configuration (Vext models, galaxy bias).
    - Sampling of shared velocity-field parameters (beta, Vext, sigma_v).
    - Rejection sampling of the distance prior weighted by density.
    - Integration over the line-of-sight distance.
    """

    def __init__(self, config_path):
        super().__init__(config_path)
        config = self.config
        fsection("Model")

        kind = get_nested(config, "pv_model/kind", "Vext")
        kind_allowed = ["Vext", "Vext_radial"]
        if kind not in kind_allowed and not kind.startswith("precomputed_los_"):  # noqa
            raise ValueError(
                f"Invalid kind '{kind}'. Must be one of {kind_allowed} or "
                "start with 'precomputed_los_'.")

        self.track_log_density_per_sample = get_nested(
            config, "inference/track_log_density_per_sample", False)

        self.which_Vext = get_nested(config, "pv_model/which_Vext", "constant")

        priors = config["model"]["priors"]

        if self.which_Vext in ["radial", "radial_magnitude"]:
            d = priors[f"Vext_{self.which_Vext}"]
            fprint(
                f"using {self.which_Vext} with spline knots at {d['rknot']}")
            self.kwargs_Vext = {
                key: d[key] for key in ["rknot", "method"]}
        elif self.which_Vext == "per_pix":
            nside = get_nested(config, "pv_model/Vext_per_pix_nside", None)
            if nside is None:
                raise ValueError(
                    "Must specify `Vext_per_pix_nside` in config when "
                    "`which_Vext = 'per_pix'`.")
            if not (nside > 0 and ((nside & (nside - 1)) == 0)):
                raise ValueError(
                    f"Invalid nside={nside} in "
                    f"which_Vext = '{self.which_Vext}'. "
                    "Must be a positive power of 2.")
            fprint(f"using per-pixel `Vext` at nside={nside}.")
            npix = 12 * nside**2
            self.kwargs_Vext = {
                "nside": nside, "npix": npix,
                "Q": jnp.asarray(sumzero_basis(npix))}
        elif self.which_Vext == "constant":
            self.which_Vext = "constant"
            self.kwargs_Vext = {}
        else:
            raise ValueError(f"Invalid which_Vext '{self.which_Vext}'.")

        self._load_and_set_priors()
        self.marginalize_eta = get_nested(
            config, "model/marginalize_eta", True)
        if self.marginalize_eta:
            self.eta_grid_kwargs = get_nested(config, "model/eta_grid", None)

        self.galaxy_bias = get_nested(config, "pv_model/galaxy_bias", "unity")
        if self.galaxy_bias not in ["unity", "powerlaw", "linear",
                                    "linear_from_beta",
                                    "linear_from_beta_stochastic",
                                    "double_powerlaw", "quadratic",
                                    "spline"]:
            raise ValueError(
                f"Invalid galaxy bias model '{self.galaxy_bias}'.")
        self.quadratic_bias_delta0 = get_nested(
            config, "pv_model/quadratic_bias_delta0", 0.0)

        if self.galaxy_bias == "spline":
            knots = get_nested(config, "pv_model/spline_bias_knots_delta")
            if knots is None:
                raise ValueError(
                    "spline_bias_knots_delta must be set for spline bias.")
            if 0.0 not in knots:
                raise ValueError(
                    "spline_bias_knots_delta must include 0.0 (pinned knot).")
            self.spline_bias_knots_delta = sorted(knots)

        self.density_dependent_sigma_v = get_nested(
            config, "pv_model/density_dependent_sigma_v", False)
        if self.density_dependent_sigma_v:
            kind = get_nested(config, "pv_model/kind", "")
            if not kind.startswith("precomputed_los"):
                raise ValueError(
                    "density_dependent_sigma_v requires precomputed LOS data.")
            required = ["sigma_v_low", "sigma_v_high",
                        "log_sigma_v_rho_t", "sigma_v_k"]
            missing = [k for k in required if k not in self.priors]
            if missing:
                raise ValueError(
                    "Missing priors for density-dependent sigma_v: "
                    f"{', '.join(missing)}.")

        self.which_distance_prior = get_nested(
            config, "pv_model/which_distance_prior", "empirical")

        fprint(f"Om={self.Om}, Vext={self.which_Vext}, "
               f"galaxy_bias={self.galaxy_bias}, "
               f"distance_prior={self.which_distance_prior}, "
               f"density_dependent_sigma_v="
               f"{self.density_dependent_sigma_v}")

    def _sample_common_params(self, shared_params):
        kwargs_dist = sample_distance_prior(self.priors)
        h = 1.
        Vext = sample_Vext(
            self.priors, self.which_Vext, shared_params, self.kwargs_Vext)

        if self.density_dependent_sigma_v:
            sigma_v_low = rsample(
                "sigma_v_low", self.priors["sigma_v_low"], shared_params)
            sigma_v_high = rsample(
                "sigma_v_high", self.priors["sigma_v_high"], shared_params)
            log_sigma_v_rho_t = rsample(
                "log_sigma_v_rho_t", self.priors["log_sigma_v_rho_t"],
                shared_params)
            sigma_v_k = rsample(
                "sigma_v_k", self.priors["sigma_v_k"], shared_params)
            sigma_v = (sigma_v_low, sigma_v_high, log_sigma_v_rho_t,
                       sigma_v_k)
        else:
            sigma_v = rsample(
                "sigma_v", self.priors["sigma_v"], shared_params)

        beta = rsample("beta", self.priors["beta"], shared_params)
        bias_kwargs = dict(Om=self.Om, beta=beta)
        if self.galaxy_bias == "spline":
            bias_kwargs["spline_bias_knots_delta"] = \
                self.spline_bias_knots_delta
        bias_params = sample_galaxy_bias(
            self.priors, self.galaxy_bias, shared_params, **bias_kwargs)
        return kwargs_dist, h, Vext, sigma_v, beta, bias_params

    def _get_simpson_log_w(self, data, r_grid):
        """Return pre-computed Simpson log weights, or compute on the fly."""
        if hasattr(data, '_simpson_log_w') and data._simpson_log_w is not None:
            return data._simpson_log_w
        return simpson_log_weights(r_grid)

    def _setup_lp_dist_and_Vrad(self, data, r_grid, kwargs_dist, beta,
                                bias_params):
        lp_dist = log_prior_r_empirical(
            r_grid, **kwargs_dist, Rmax_grid=r_grid[-1])[None, None, :]

        if data.has_precomputed_los:
            Vrad = beta * data["los_velocity_r_grid"]
            lp_dist += lp_galaxy_bias(
                data["los_delta_r_grid"],
                data["los_log_density_r_grid"],
                bias_params, self.galaxy_bias,
                self.quadratic_bias_delta0)
            log_w_r = self._get_simpson_log_w(data, r_grid)
            lp_dist -= logsumexp(
                lp_dist + log_w_r[None, None, :], axis=-1)[..., None]
        else:
            Vrad = 0.

        return lp_dist, Vrad

    def _compute_ll_cz(self, data, r_grid, h, Vext, sigma_v, Vrad):
        Vext_rad = compute_Vext_radial(
            data, r_grid, Vext, which_Vext=self.which_Vext,
            **self.kwargs_Vext)
        czpred = predict_cz(
            self.distance2redshift(r_grid, h=h)[None, None, :],
            Vrad + Vext_rad)

        if self.density_dependent_sigma_v:
            sigma_v_low, sigma_v_high, log_rho_t, k = sigma_v
            sigma_v_grid = sigma_v_from_density(
                data["los_delta_r_grid"], sigma_v_low, sigma_v_high,
                log_rho_t, k)
            return normal_logpdf_var(
                data["czcmb"][None, :, None], czpred, sigma_v_grid**2)

        return normal_logpdf_var(
            data["czcmb"][None, :, None], czpred, sigma_v**2)

    def _marginalize_over_r(self, ll, r_grid, data=None):
        log_w_r = self._get_simpson_log_w(data, r_grid)
        return logsumexp(ll + log_w_r[None, None, :], axis=-1)

    def _average_fields_and_factor(self, ll, data,
                                   log_density_per_sample=None):
        ll = logsumexp(ll, axis=0) - jnp.log(data.num_fields)
        factor("ll_obs", ll)

        if self.track_log_density_per_sample and log_density_per_sample is not None:  # noqa
            log_density_per_sample += ll
            deterministic("log_density_per_sample", log_density_per_sample)


class JointPVModel:
    r"""
    Joint likelihood model for multiple independent PV datasets.

    Enables joint inference where certain parameters (e.g., :math:`\beta`,
    :math:`\sigma_v`, :math:`V_{\rm ext}`) are shared across different
    distance-indicator catalogues while others (e.g., TFR zero-points) remain
    catalogue-specific.

    Parameters
    ----------
    submodels : list of BasePVModel
        The individual models to be combined.
    shared_param_names : list of str
        Names of parameters from the ``[model.priors]`` section to be shared.
    """

    def __init__(self, submodels, shared_param_names):
        self.submodels = submodels
        self.shared_param_names = shared_param_names

        # Check that all submodels have the same config.
        ref_hash = config_hash(submodels[0].config)
        for i, model in enumerate(submodels[1:], start=1):
            if config_hash(model.config) != ref_hash:
                raise ValueError(f"Submodel {i} has a different config hash.")

        self.config = submodels[0].config
        self.which_Vext = submodels[0].which_Vext
        self.galaxy_bias = submodels[0].galaxy_bias
        if hasattr(submodels[0], 'spline_bias_knots_delta'):
            self.spline_bias_knots_delta = \
                submodels[0].spline_bias_knots_delta

    def _sample_shared_params(self, priors):
        shared = {}
        for name in self.shared_param_names:
            shared[name] = _rsample(name, priors[name])
        return shared

    def __call__(self, data):
        assert len(data) == len(self.submodels)
        shared_params = self._sample_shared_params(self.submodels[0].priors)

        for i, (submodel, data_i) in enumerate(zip(self.submodels, data)):
            name = data_i.name if data_i is not None else f"dataset_{i}"
            with handlers.scope(prefix=name):
                submodel(data_i, shared_params=shared_params)
