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
"""TRGB-calibrated H0 forward model for EDD TRGB distance indicators."""
import healpy as hp
import jax.numpy as jnp
import numpy as np
from jax import checkpoint, lax
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm as norm_jax
from numpyro import deterministic, factor, sample
from numpyro.distributions import Dirichlet, Normal, Uniform

from ..util import (fprint, get_nested, radec_to_galactic,
                    replace_prior_with_delta)
from .base_model import H0ModelBase
from .integration import ln_simpson_precomputed
from .pv_utils import (galaxy_bias_needs_log_rho, lp_galaxy_bias, rsample,
                       sample_galaxy_bias)
from .utils import (log_prob_integrand_window_sel, logmeanexp,
                    normal_logpdf_var, predict_cz)


def _validate_healpix_nside(nside):
    """Return a validated HEALPix nside value."""
    nside = int(nside)
    if nside <= 0 or not hp.isnsideok(nside):
        raise ValueError(
            "`model/TRGB_sky_exposure/nside` must be a positive "
            "HEALPix nside value.")
    return nside


def _sky_exposure_pixel_id_galactic_numpy(ell, b, nside=1):
    """Assign Galactic coordinates to HEALPix sky-exposure pixels."""
    nside = _validate_healpix_nside(nside)
    ell = np.nan_to_num(np.asarray(ell), nan=0.0, posinf=0.0,
                        neginf=0.0)
    b = np.nan_to_num(np.asarray(b), nan=0.0, posinf=90.0,
                      neginf=-90.0)
    ell = np.remainder(ell, 360.0)
    b = np.clip(b, -90.0, 90.0)
    theta = np.clip(0.5 * np.pi - np.deg2rad(b), 0.0, np.pi)
    phi = np.deg2rad(ell)
    return hp.ang2pix(nside, theta, phi, nest=False).astype(np.int32)


def _sky_exposure_pixel_id_numpy(ra, dec, nside=1):
    """Assign ICRS sky positions to HEALPix sky-exposure pixels."""
    ell, b = radec_to_galactic(ra, dec)
    return _sky_exposure_pixel_id_galactic_numpy(ell, b, nside=nside)


class TRGBModel(H0ModelBase):
    """
    TRGB-calibrated H0 model for EDD TRGB distance indicators with
    inhomogeneous Malmquist bias correction and peculiar velocity modeling.
    """
    _MAG_WINDOW_MIN_WIDTH = 1e-3

    # ------------------------------------------------------------------
    #  Phase 1: model physics
    # ------------------------------------------------------------------

    def _replace_unused_priors(self, config):
        config = super()._replace_unused_priors(config)
        priors = config.setdefault("model", {}).setdefault("priors", {})
        priors.setdefault("c_star", {
            "dist": "normal",
            "loc": 1.23,
            "scale": 0.1,
        })
        priors.setdefault("alpha_c", {
            "dist": "normal",
            "loc": 0.0,
            "scale": 1.0,
        })
        priors.setdefault("c_bar", {
            "dist": "uniform",
            "low": 0.0,
            "high": 3.0,
        })
        priors.setdefault("w_c", {
            "dist": "uniform",
            "low": 0.001,
            "high": 1.0,
        })
        which_sel = get_nested(config, "model/which_selection", None)
        use_TRGB_host_redshift = get_nested(
            config, "model/use_TRGB_host_redshift", True)
        if (which_sel in ("TRGB_magnitude", "TRGB_magnitude_redshift")
                and get_nested(config, "model/mag_lim_TRGB", None)
                == "infer"):
            self._constrain_mag_lim_prior(config)
        if which_sel == "SN_magnitude":
            priors.setdefault("sigma_int_SN", {
                "dist": "maxwell",
                "scale": 0.0627,
            })
        else:
            replace_prior_with_delta(
                config, "M_B", -19.0, verbose=False)
            replace_prior_with_delta(
                config, "sigma_int_SN", 0.0, verbose=False)

        if not use_TRGB_host_redshift:
            replace_prior_with_delta(config, "H0", 73.04)
            replace_prior_with_delta(config, "Vext", [0., 0., 0.])
            replace_prior_with_delta(config, "sigma_v", 100.0)

        return config

    def _constrain_mag_lim_prior(self, config):
        """Keep the inferred upper magnitude threshold above mag_min_TRGB."""
        priors = config.setdefault("model", {}).setdefault("priors", {})
        mag_min = float(get_nested(config, "model/mag_min_TRGB", 22.1))
        low_bound = mag_min + self._MAG_WINDOW_MIN_WIDTH
        spec = priors.setdefault("mag_lim_TRGB", {
            "dist": "truncated_normal",
            "mean": max(24.0, low_bound + 1.0),
            "scale": 1.0,
            "low": low_bound,
        })
        dist = spec.get("dist")
        if dist == "normal":
            loc = spec["loc"]
            scale = spec["scale"]
            spec.clear()
            spec.update({
                "dist": "truncated_normal",
                "mean": loc,
                "scale": scale,
                "low": low_bound,
            })
        elif dist == "truncated_normal":
            low = spec.get("low", None)
            if low is None or low < low_bound:
                spec["low"] = low_bound
            if (spec.get("high", None) is not None
                    and spec["high"] <= spec["low"]):
                raise ValueError(
                    "`mag_lim_TRGB` prior high must exceed mag_min_TRGB.")
        elif dist == "uniform":
            if spec["low"] < low_bound:
                spec["low"] = low_bound
            if spec["high"] <= spec["low"]:
                raise ValueError(
                    "`mag_lim_TRGB` prior high must exceed mag_min_TRGB.")
        else:
            raise ValueError(
                "Inferred TRGB magnitude-window selection requires "
                "`mag_lim_TRGB` prior to be normal, truncated_normal, "
                f"or uniform, got {dist!r}.")

    def _load_selection_thresholds(self):
        active_map = {
            "TRGB_magnitude": {"mag_lim_TRGB", "mag_lim_TRGB_width"},
            "SN_magnitude": {"mag_lim_SN", "mag_lim_SN_width"},
            "TRGB_magnitude_redshift": {
                "mag_lim_TRGB", "mag_lim_TRGB_width",
                "cz_lim_selection", "cz_lim_selection_width"},
        }
        spec = {
            "mag_lim_TRGB":           None,
            "mag_lim_TRGB_width":     None,
            "mag_lim_SN":             None,
            "mag_lim_SN_width":       None,
            "cz_lim_selection":       3300.0,
            "cz_lim_selection_width": None,
        }
        super()._load_selection_thresholds(active_map, spec)

    def _load_model_flags(self):
        super()._load_model_flags()
        self.use_TRGB_host_redshift = get_nested(
            self.config, "model/use_TRGB_host_redshift", True)
        fprint(f"use_TRGB_host_redshift set to "
               f"{self.use_TRGB_host_redshift}")
        self.use_density_dependent_sigma_v = get_nested(
            self.config, "model/use_density_dependent_sigma_v", False)
        fprint("use_density_dependent_sigma_v set to "
               f"{self.use_density_dependent_sigma_v}")
        self.save_log_likelihood_per_galaxy = bool(get_nested(
            self.config, "inference/save_log_likelihood_per_galaxy", False))
        if self.save_log_likelihood_per_galaxy:
            fprint("saving per-galaxy TRGB log likelihood contributions.")
        self.mag_min_TRGB = get_nested(
            self.config, "model/mag_min_TRGB", 22.1)
        if self.which_selection in (
                "TRGB_magnitude", "TRGB_magnitude_redshift"):
            fprint(f"mag_min_TRGB set to {self.mag_min_TRGB}")
        sky_exposure = get_nested(
            self.config, "model/TRGB_sky_exposure", {})
        self.use_TRGB_sky_exposure = bool(
            sky_exposure.get("enabled", False)
            if isinstance(sky_exposure, dict) else False)
        if self.use_TRGB_sky_exposure:
            if "nside" not in sky_exposure:
                raise ValueError(
                    "TRGB sky exposure requires "
                    "`model/TRGB_sky_exposure/nside`.")
            self.TRGB_sky_exposure_nside = _validate_healpix_nside(
                sky_exposure["nside"])
        else:
            self.TRGB_sky_exposure_nside = 1
        self.TRGB_sky_exposure_n_pix = int(
            hp.nside2npix(self.TRGB_sky_exposure_nside))
        self.TRGB_sky_exposure_kappa = float(
            sky_exposure.get("kappa", 48.0)
            if isinstance(sky_exposure, dict) else 48.0)
        if self.use_TRGB_sky_exposure:
            fprint(
                "TRGB HEALPix angular exposure enabled: "
                f"nside={self.TRGB_sky_exposure_nside}, "
                f"n_pix={self.TRGB_sky_exposure_n_pix}, "
                f"kappa={self.TRGB_sky_exposure_kappa:g}.")

    # ------------------------------------------------------------------
    #  Phase 2: data loading
    # ------------------------------------------------------------------

    def _load_data(self, data):
        # Read SN-level data before super() processes the data dict.
        self._has_sn_data = ("m_Bprime" in data
                             and data["m_Bprime"] is not None)
        if self._has_sn_data:
            required = (
                "sn_group_index", "m_Bprime",
                "e_m_Bprime", "e_m_Bprime_median")
            missing = [
                key for key in required
                if key not in data or data[key] is None
            ]
            if missing:
                raise ValueError(
                    "SN magnitude data are incomplete; missing "
                    f"{', '.join(missing)}.")
            self._sn_group_index = jnp.asarray(
                data["sn_group_index"])
            self._m_Bprime = jnp.asarray(data["m_Bprime"])
            self._e_m_Bprime = jnp.asarray(data["e_m_Bprime"])
            self._e2_m_Bprime = self._e_m_Bprime ** 2
            self._e_m_Bprime_median = float(
                data["e_m_Bprime_median"])
            n_sn = len(self._m_Bprime)
            n_hosts = len(np.unique(np.asarray(self._sn_group_index)))
            fprint(f"loaded {n_sn} SNe across {n_hosts} hosts "
                   f"for SN magnitude data.")

        super()._load_data(data)
        self.num_hosts = len(self.mag_obs)
        self._has_trgb_colour = (
            hasattr(self, "colour_dered")
            and hasattr(self, "e_colour_dered"))
        if not self._has_trgb_colour:
            fprint("TRGB colour data not found; treating tip magnitudes as "
                   "pivot-standardized.")
        if self.use_TRGB_sky_exposure:
            if not hasattr(self, "RA_host") or not hasattr(self, "dec_host"):
                raise ValueError(
                    "TRGB sky exposure requires `RA_host` and `dec_host`.")
            self._TRGB_sky_exposure_host_pix = jnp.asarray(
                _sky_exposure_pixel_id_numpy(
                    np.asarray(self.RA_host), np.asarray(self.dec_host),
                    nside=self.TRGB_sky_exposure_nside),
                dtype=jnp.int32)
            if (not hasattr(self, "galactic_ell_3d")
                    or not hasattr(self, "galactic_b_3d")):
                raise ValueError(
                    "TRGB sky exposure requires `galactic_ell_3d` and "
                    "`galactic_b_3d` for the 3D selection integral.")
            volume_pix = _sky_exposure_pixel_id_galactic_numpy(
                np.asarray(self.galactic_ell_3d),
                np.asarray(self.galactic_b_3d),
                nside=self.TRGB_sky_exposure_nside)
            self._TRGB_sky_exposure_volume_pix = jnp.asarray(
                volume_pix, dtype=jnp.int32)
            support_mask = np.ones(volume_pix.shape, dtype=bool)
            if self.selection_integral_b_min is not None:
                support_mask &= (
                    np.abs(np.asarray(self.galactic_b_3d))
                    >= self.selection_integral_b_min)
            if hasattr(self, "log_volume_weight_3d"):
                support_mask &= np.broadcast_to(
                    np.isfinite(np.asarray(self.log_volume_weight_3d)),
                    support_mask.shape)
            support_pix = np.flatnonzero(np.bincount(
                volume_pix[support_mask],
                minlength=self.TRGB_sky_exposure_n_pix))
            if len(support_pix) == 0:
                raise ValueError(
                    "TRGB sky exposure has no 3D selection-integral support "
                    "after the sky mask.")
            self.TRGB_sky_exposure_n_support = int(len(support_pix))
            self._TRGB_sky_exposure_support_pix = jnp.asarray(
                support_pix, dtype=jnp.int32)
            host_pix = np.asarray(self._TRGB_sky_exposure_host_pix)
            if np.any(~np.isin(host_pix, support_pix)):
                raise ValueError(
                    "Observed TRGB host falls in a sky-exposure pixel with "
                    "no 3D selection-integral support.")
        fprint(f"loaded {self.num_hosts} TRGB host galaxies.")

    def _set_data_arrays(self, data):
        if data.get("host_names") is not None:
            self.host_names = np.asarray(data["host_names"], dtype=str)
        skip = ("host_names", "sn_group_index", "m_Bprime",
                "e_m_Bprime", "e_m_Bprime_median")
        super()._set_data_arrays(data, skip_keys=skip)

    def _setup_no_recon_direction_grid(self):
        """Only the joint redshift selection needs no-reconstruction cuts."""
        if self.which_selection == "TRGB_magnitude_redshift":
            return super()._setup_no_recon_direction_grid()
        return

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def _validate_sn_data(self):
        if not self._has_sn_data:
            return
        group_index = np.asarray(self._sn_group_index)
        m_Bprime = np.asarray(self._m_Bprime)
        e_m_Bprime = np.asarray(self._e_m_Bprime)

        if group_index.ndim != 1:
            raise ValueError("`sn_group_index` must be one-dimensional.")
        if not np.issubdtype(group_index.dtype, np.integer):
            raise ValueError("`sn_group_index` must contain integer indices.")
        if m_Bprime.ndim != 1 or e_m_Bprime.ndim != 1:
            raise ValueError("SN magnitude arrays must be one-dimensional.")
        if not (len(group_index) == len(m_Bprime) == len(e_m_Bprime)):
            raise ValueError(
                "`sn_group_index`, `m_Bprime`, and `e_m_Bprime` must have "
                "the same length.")
        if len(group_index) == 0:
            raise ValueError("SN magnitude data must contain at least one SN.")
        if np.any(group_index < 0) or np.any(group_index >= self.num_hosts):
            raise ValueError(
                "`sn_group_index` entries must be in [0, num_hosts).")
        if not np.all(np.isfinite(m_Bprime)):
            raise ValueError("`m_Bprime` contains non-finite values.")
        if not np.all(np.isfinite(e_m_Bprime)):
            raise ValueError("`e_m_Bprime` contains non-finite values.")
        if np.any(e_m_Bprime <= 0):
            raise ValueError("`e_m_Bprime` entries must be positive.")
        if not np.isfinite(self._e_m_Bprime_median) \
                or self._e_m_Bprime_median <= 0:
            raise ValueError("`e_m_Bprime_median` must be positive.")

    def _validate_selection_width(self, name):
        """Require fixed selection widths to be present and positive."""
        if getattr(self, f"_infer_{name}", False):
            return
        value = getattr(self, name)
        if value is None:
            raise ValueError(
                f"`{name}` must be set or 'infer' for "
                f"{self.which_selection} selection.")
        try:
            value_arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"`{name}` must be numeric, got {value!r}.")
        if np.any(~np.isfinite(value_arr)) or np.any(value_arr <= 0):
            raise ValueError(f"`{name}` must be positive, got {value!r}.")

    def _validate_active_selection_widths(self):
        width_names = {
            "TRGB_magnitude": ("mag_lim_TRGB_width",),
            "SN_magnitude": ("mag_lim_SN_width",),
            "TRGB_magnitude_redshift": (
                "mag_lim_TRGB_width", "cz_lim_selection_width"),
        }.get(self.which_selection, ())
        for name in width_names:
            self._validate_selection_width(name)

    def _validate_config(self):
        if self._has_trgb_colour:
            fixed_colour_params = [
                name for name in ("c_star", "c_bar", "w_c")
                if self.prior_dist_name.get(name) == "delta"
            ]
            if fixed_colour_params:
                raise ValueError(
                    "TRGB colour-calibration parameters must be sampled; "
                    "use non-delta priors for "
                    f"{', '.join(fixed_colour_params)}.")
        if self.use_reconstruction and not self.has_host_los:
            raise ValueError(
                "`use_reconstruction` requires host LOS interpolators.")
        if (self.use_reconstruction and not self.apply_sel
                and self.selection_integral_grid_radius is None):
            raise ValueError(
                "Reconstructed no-selection TRGB runs require "
                "`model.selection_integral_grid_radius` for the finite "
                "3D distance-prior normalizer.")
        if (self.use_reconstruction and not self.apply_sel
                and not self.has_volume_density_3d):
            raise ValueError(
                "Reconstructed no-selection TRGB runs require 3D density "
                "data for the finite distance-prior normalizer.")
        if (self.use_reconstruction and not self.apply_sel
                and self.has_volume_density_3d
                and self.density_3d_fields.shape[0] != self.num_fields):
            raise ValueError(
                "Number of 3D density fields "
                f"({self.density_3d_fields.shape[0]}) does not match LOS "
                f"field realisations ({self.num_fields}).")
        if (self.save_log_likelihood_per_galaxy
                and self.use_reconstruction
                and self.num_fields != 1):
            raise ValueError(
                "Per-galaxy TRGB log likelihood saving is defined only for "
                "single-field runs.")
        if (self.save_log_likelihood_per_galaxy
                and hasattr(self, "host_names")
                and self.host_names.shape != (self.num_hosts,)):
            raise ValueError(
                "`host_names` must be one-dimensional with length matching "
                "the number of TRGB hosts when saving per-galaxy log "
                "likelihoods.")
        if self.use_density_dependent_sigma_v and not self.use_reconstruction:
            raise ValueError(
                "`use_density_dependent_sigma_v` requires "
                "`use_reconstruction` to be set to True.")
        if (self.use_density_dependent_sigma_v
                and not self.use_TRGB_host_redshift):
            raise ValueError(
                "`use_density_dependent_sigma_v` requires "
                "`use_TRGB_host_redshift` to be set to True.")
        if self.use_density_dependent_sigma_v:
            required = ["sigma_v_low", "sigma_v_high",
                        "log_sigma_v_rho_t", "sigma_v_k"]
            missing = [k for k in required if k not in self.priors]
            if missing:
                raise ValueError(
                    "Missing priors for density-dependent sigma_v: "
                    f"{', '.join(missing)}.")

        allowed_selection = [
            "TRGB_magnitude", "SN_magnitude",
            "TRGB_magnitude_redshift", None]
        if self.which_selection not in allowed_selection:
            raise ValueError(
                f"Unknown `which_selection`: {self.which_selection}. "
                f"Expected one of {allowed_selection}.")
        if self.use_TRGB_sky_exposure:
            if self.which_selection != "TRGB_magnitude":
                raise NotImplementedError(
                    "TRGB sky exposure currently supports only "
                    "`which_selection='TRGB_magnitude'`.")
            if not self.use_reconstruction:
                raise NotImplementedError(
                    "TRGB sky exposure currently requires a reconstructed "
                    "3D selection integral.")
            _validate_healpix_nside(self.TRGB_sky_exposure_nside)
            if self.TRGB_sky_exposure_kappa <= 0:
                raise ValueError(
                    "`model/TRGB_sky_exposure/kappa` must be positive.")
        self._validate_active_selection_widths()
        self._validate_sn_data()

        if self.which_selection == "SN_magnitude" \
                and not self._has_sn_data:
            raise ValueError(
                "SN_magnitude selection requires SN data "
                "(m_Bprime, e_m_Bprime) in the data dict.")

        needs_redshift_sel = (
            self.which_selection == "TRGB_magnitude_redshift")
        if needs_redshift_sel and not self.use_TRGB_host_redshift:
            raise ValueError(
                "`TRGB_magnitude_redshift` selection requires "
                "`use_TRGB_host_redshift` to be set to True.")
        self._validate_student_t_redshift_selection(needs_redshift_sel)
        self._validate_selection_integral(needs_velocity=needs_redshift_sel)

        if self.which_selection in (
                "TRGB_magnitude", "TRGB_magnitude_redshift"):
            if not np.isfinite(float(self.mag_min_TRGB)):
                raise ValueError("`mag_min_TRGB` must be finite.")
            if self.mag_lim_TRGB is None \
                    and not self._infer_mag_lim_TRGB:
                raise ValueError(
                    "`mag_lim_TRGB` must be set or 'infer' "
                    f"for {self.which_selection} selection.")
            if (self.mag_lim_TRGB is not None
                    and not self._infer_mag_lim_TRGB
                    and self.mag_lim_TRGB <= (
                        self.mag_min_TRGB + self._MAG_WINDOW_MIN_WIDTH)):
                raise ValueError(
                    "`mag_lim_TRGB` must exceed `mag_min_TRGB` for "
                    f"{self.which_selection} selection.")
        if self.which_selection == "SN_magnitude":
            if self.mag_lim_SN is None \
                    and not self._infer_mag_lim_SN:
                raise ValueError(
                    "`mag_lim_SN` must be set or 'infer' "
                    "for SN_magnitude selection.")

    def sigma_v_from_density(self, delta, sigma_v_low, sigma_v_high,
                             log_rho_t, k):
        """Map overdensity to sigma_v through a sigmoid in log density."""
        rho = jnp.clip(1.0 + delta, a_min=1e-6)
        log_rho = jnp.log(rho)
        return sigma_v_low + (sigma_v_high - sigma_v_low) / (
            1.0 + jnp.exp(-k * (log_rho - log_rho_t)))

    def _volume_sigma_v_fields(self, sigma_v_low, sigma_v_high,
                               log_rho_t, k):
        """Evaluate density-dependent sigma_v on the 3D selection grid."""
        if self.density_3d_mode == "log_rho":
            delta_3d = jnp.exp(self.density_3d_fields) - 1.0
        else:
            delta_3d = self.density_3d_fields
        return self.sigma_v_from_density(
            delta_3d, sigma_v_low, sigma_v_high, log_rho_t, k)

    def _sum_sn_terms_by_host(self, terms):
        """Sum SN-level terms into their corresponding TRGB host bins."""
        host_shape = (self.num_hosts,) + terms.shape[1:]
        host_terms = jnp.zeros(host_shape)
        return host_terms.at[self._sn_group_index].add(terms)

    def _record_per_galaxy_log_likelihood(
            self, ll_without_selection, ll_selection_observed,
            log_selection_integral=0.0):
        """Expose per-host likelihood terms as deterministic sites."""
        if not self.save_log_likelihood_per_galaxy:
            return

        with_selection = (
            ll_without_selection
            + ll_selection_observed
            - log_selection_integral
        )
        deterministic("log_likelihood_per_galaxy", ll_without_selection)
        deterministic(
            "log_observed_selection_per_galaxy",
            ll_selection_observed)
        deterministic("log_selection_integral", log_selection_integral)
        deterministic(
            "log_likelihood_per_galaxy_with_selection", with_selection)

    def __call__(self, **dynamic_attrs):
        if dynamic_attrs:
            with self._temporary_attrs(dynamic_attrs):
                return self.__call__()

        # --- Global parameters ---
        H0 = rsample("H0", self.priors["H0"])
        M_TRGB = rsample("M_TRGB", self.priors["M_TRGB"])
        sigma_int = rsample("sigma_int", self.priors["sigma_int"])

        if self.use_density_dependent_sigma_v:
            sigma_v_low = rsample("sigma_v_low", self.priors["sigma_v_low"])
            sigma_v_high = rsample(
                "sigma_v_high", self.priors["sigma_v_high"])
            log_sigma_v_rho_t = rsample(
                "log_sigma_v_rho_t", self.priors["log_sigma_v_rho_t"])
            sigma_v_k = rsample("sigma_v_k", self.priors["sigma_v_k"])
            sigma_v = (sigma_v_low, sigma_v_high,
                       log_sigma_v_rho_t, sigma_v_k)
        else:
            sigma_v = rsample("sigma_v", self.priors["sigma_v"])

        nu_cz = self._sample_nu_cz()
        Vext, Vext_quad, Vext_oct, Vext_mono = \
            self._sample_external_velocity()
        beta = rsample("beta", self.priors["beta"])
        bias_params = sample_galaxy_bias(
            self.priors, self.which_bias, beta=beta, Om=self.Om)

        h = H0 / 100

        # --- Distance moduli ---
        mu_LMC = sample("mu_LMC", Uniform(*self.distmod_limits_LMC))
        mu_N4258 = sample("mu_N4258", Uniform(*self.distmod_limits_N4258))

        # --- Geometric anchor constraints ---
        sample("mu_LMC_ll",
               Normal(mu_LMC, self.e_mu_LMC_anchor),
               obs=self.mu_LMC_anchor)
        sample("mu_N4258_ll",
               Normal(mu_N4258, self.e_mu_N4258_anchor),
               obs=self.mu_N4258_anchor)

        # --- TRGB magnitude anchor constraints ---
        factor("mag_LMC_TRGB_ll",
               normal_logpdf_var(self.mag_LMC_TRGB,
                                 M_TRGB + mu_LMC,
                                 self.e_mag_LMC_TRGB**2 + sigma_int**2))
        factor("mag_N4258_TRGB_ll",
               normal_logpdf_var(self.mag_N4258_TRGB,
                                 M_TRGB + mu_N4258,
                                 self.e_mag_N4258_TRGB**2 + sigma_int**2))

        # --- Per-host cosmographic grids (fine grid) ---
        r_grid = self.r_host_range
        lp_r = self.log_prior_distance(r_grid)
        Vext_rad_host = self._host_Vext_radial(
            Vext, Vext_quad, Vext_oct)
        Vext_mono_host_grid = self._Vext_monopole_radial(
            Vext_mono, r_grid)
        mu_grid = self.distance2distmod(r_grid, h=h)
        z_grid = self.distance2redshift(r_grid, h=h)

        log_S = None
        ll_sn_host = None
        ll_observed_selection_host = jnp.zeros(self.num_hosts)

        if self._has_trgb_colour:
            alpha_c = rsample("alpha_c", self.priors["alpha_c"])
            c_star = rsample("c_star", self.priors["c_star"])
            c_bar = rsample("c_bar", self.priors["c_bar"])
            w_c = rsample("w_c", self.priors["w_c"])

            e2_colour = self.e2_colour_dered
            colour_var = w_c**2 + e2_colour
            ll_colour_host = normal_logpdf_var(
                self.colour_dered, c_bar, colour_var)
            colour_post_mean = (
                e2_colour * c_bar + w_c**2 * self.colour_dered) / colour_var
            colour_post_var = (w_c**2 * e2_colour) / colour_var

            M_TRGB_host = M_TRGB + alpha_c * (colour_post_mean - c_star)
            e2_mag_host = (
                self.e2_mag_obs + sigma_int**2
                + alpha_c**2 * colour_post_var)
            M_TRGB_sel = M_TRGB + alpha_c * (c_bar - c_star)
            colour_sel_var = (alpha_c * w_c) ** 2
        else:
            ll_colour_host = jnp.zeros(self.num_hosts)
            M_TRGB_host = jnp.broadcast_to(M_TRGB, (self.num_hosts,))
            e2_mag_host = self.e2_mag_obs + sigma_int**2
            M_TRGB_sel = M_TRGB
            colour_sel_var = 0.0

        if self.which_selection == "TRGB_magnitude":
            mag_lim = self._resolve_threshold("mag_lim_TRGB")
            mag_width = self._resolve_threshold("mag_lim_TRGB_width")

            ll_observed_selection_host = log_prob_integrand_window_sel(
                self.mag_obs, 0.0, self.mag_min_TRGB,
                mag_lim, mag_width)

            e_eff = jnp.sqrt(
                self.e2_mag_median + sigma_int**2 + colour_sel_var)
            if self.use_TRGB_sky_exposure:
                log_S_pix = self._compute_volume_log_S_mag_window(
                    bias_params, M_TRGB_sel, e_eff, H0,
                    self.mag_min_TRGB, mag_lim, mag_width,
                    return_pixels=True)
                log_S, log_sky_ratio_host, _ = (
                    self._sample_TRGB_sky_exposure(log_S_pix))
                ll_observed_selection_host = (
                    ll_observed_selection_host + log_sky_ratio_host)
            else:
                log_S = self._compute_volume_log_S_mag_window(
                    bias_params, M_TRGB_sel, e_eff, H0,
                    self.mag_min_TRGB, mag_lim, mag_width)
            factor("ll_sel_per_object", jnp.sum(
                ll_observed_selection_host))

        elif self.which_selection == "TRGB_magnitude_redshift":
            mag_lim = self._resolve_threshold("mag_lim_TRGB")
            mag_width = self._resolve_threshold("mag_lim_TRGB_width")
            cz_lim = self._resolve_threshold("cz_lim_selection")
            cz_width = self._resolve_threshold("cz_lim_selection_width")

            ll_sel_mag = log_prob_integrand_window_sel(
                self.mag_obs, 0.0, self.mag_min_TRGB, mag_lim, mag_width)
            ll_sel_cz = norm_jax.logcdf((cz_lim - self.czcmb) / cz_width)
            ll_observed_selection_host = ll_sel_mag + ll_sel_cz
            factor("ll_sel_per_object",
                   jnp.sum(ll_sel_mag) + jnp.sum(ll_sel_cz))

            e_eff = jnp.sqrt(
                self.e2_mag_median + sigma_int**2 + colour_sel_var)
            sigma_v_sel = (self._volume_sigma_v_fields(*sigma_v)
                           if self.use_density_dependent_sigma_v else sigma_v)
            log_S = self._compute_volume_log_S_mag_window_cz(
                bias_params, M_TRGB_sel, e_eff, H0,
                sigma_v_sel, beta, Vext, Vext_mono,
                self.mag_min_TRGB, mag_lim, mag_width,
                cz_lim, cz_width, nu_cz=nu_cz)

        elif self.which_selection == "SN_magnitude":
            M_B = rsample("M_B", self.priors["M_B"])
            sigma_int_SN = rsample(
                "sigma_int_SN", self.priors["sigma_int_SN"])
            mag_lim = self._resolve_threshold("mag_lim_SN")
            mag_width = self._resolve_threshold("mag_lim_SN_width")

            # Per-SN selection probability
            ll_sel_sn = norm_jax.logcdf(
                (mag_lim - self._m_Bprime) / mag_width)
            factor("ll_sel_per_object", jnp.sum(ll_sel_sn))
            ll_observed_selection_host = self._sum_sn_terms_by_host(
                ll_sel_sn)

            log_S = self._compute_volume_log_S_mag(
                bias_params, M_B,
                jnp.sqrt(self._e_m_Bprime_median**2 + sigma_int_SN**2),
                H0, mag_lim, mag_width)

            # SN magnitude likelihood on distance grid
            # Per-SN: (n_sn, n_grid)
            ll_sn_per = normal_logpdf_var(
                self._m_Bprime[:, None],
                M_B + mu_grid[None, :],
                self._e2_m_Bprime[:, None] + sigma_int_SN**2)
            # Sum SNe per host: (n_hosts, n_grid)
            ll_sn_host = self._sum_sn_terms_by_host(ll_sn_per)

        self._call_marginalized(
            h, M_TRGB_host, e2_mag_host, ll_colour_host,
            sigma_v, beta, bias_params,
            Vext_rad_host, r_grid, lp_r, log_S,
            mu_grid=mu_grid, z_grid=z_grid,
            ll_sn_host=ll_sn_host,
            ll_observed_selection_host=ll_observed_selection_host,
            Vext_mono_host_grid=Vext_mono_host_grid,
            nu_cz=nu_cz)

    def _compute_volume_log_S_mag_window(
            self, bias_params, M_abs, e_mag, H0,
            mag_min, mag_lim, mag_width, return_pixels=False):
        """3D selection integral for the finite TRGB magnitude window."""
        if return_pixels and not self.use_reconstruction:
            raise NotImplementedError(
                "TRGB sky exposure requires a reconstructed 3D selection "
                "integral.")
        if not self.use_reconstruction:
            return super()._compute_volume_log_S_mag_window(
                bias_params, M_abs, e_mag, H0,
                mag_min, mag_lim, mag_width)

        h = H0 / 100
        mu_3d = self.mu_at_h1_3d - 5 * jnp.log10(h)
        log_P_sel = log_prob_integrand_window_sel(
            mu_3d + M_abs, e_mag, mag_min, mag_lim, mag_width)
        log_cell_weight = self._selection_3d_log_measure(H0)
        pix_3d = (
            self._selection_3d_sky_exposure_pixel_id()
            if return_pixels else None)

        def _one(density_3d):
            log_n = self._vol_sel_galaxy_bias(density_3d, bias_params)
            log_values = log_P_sel + log_n + log_cell_weight
            if return_pixels:
                return self._logsumexp_by_sky_exposure_pixel(
                    log_values, pix_3d)
            return logsumexp(log_values)

        return lax.map(checkpoint(_one), self.density_3d_fields,
                       batch_size=self.volume_density_batch_size)

    # ------------------------------------------------------------------
    #  Distance marginalization path
    # ------------------------------------------------------------------

    def _compute_no_selection_volume_log_norm(self, bias_params, H0):
        """3D normalizer for reconstructed no-selection TRGB runs."""
        h = H0 / 100
        log_r_phys_3d = self.log_r_3d - jnp.log(h)
        r_min = self.r_host_range[0]
        r_max = self.r_host_range[-1]
        in_support = (
            (self.log_r_3d <= jnp.log(self.selection_integral_grid_radius))
            & (log_r_phys_3d >= jnp.log(r_min))
            & (log_r_phys_3d <= jnp.log(r_max))
        )
        log_cell_weight = self._selection_3d_log_measure(H0)

        def _one(density_3d):
            log_n = self._vol_sel_galaxy_bias(density_3d, bias_params)
            return logsumexp(jnp.where(in_support,
                                       log_n + log_cell_weight,
                                       -jnp.inf))

        return lax.map(checkpoint(_one), self.density_3d_fields,
                       batch_size=self.volume_density_batch_size)

    def _selection_3d_sky_exposure_pixel_id(self):
        """HEALPix sky-exposure pixel id for each 3D volume cell."""
        return self._TRGB_sky_exposure_volume_pix

    def _logsumexp_by_sky_exposure_pixel(self, log_values, pix_3d):
        """Sum log selection-integral contributions in each sky pixel."""
        return jnp.stack([
            logsumexp(jnp.where(pix_3d == k, log_values, -jnp.inf))
            for k in range(self.TRGB_sky_exposure_n_pix)
        ])

    def _sample_TRGB_sky_exposure(self, log_S_pix):
        """Sample angular exposure and return host and volume log weights.

        `log_S_pix` has shape (num_fields, n_pix): one selection integral
        per sky pixel per reconstruction field realization. `theta` is a
        single survey-level angular exposure shared across realizations,
        so its baseline fraction `q` is derived from the field-marginalized
        (logsumexp over realizations) per-pixel selection, not one field.
        """
        log_S_pix_marginal = logsumexp(log_S_pix, axis=0)
        log_S_total = logsumexp(log_S_pix_marginal)
        support_pix = self._TRGB_sky_exposure_support_pix
        q = jnp.exp(log_S_pix_marginal - log_S_total)
        q = jnp.where(jnp.isfinite(q), q, 0.0)
        tiny = jnp.finfo(q.dtype).tiny
        q = q / jnp.clip(jnp.sum(q), tiny)
        active_pix = q > 0.0
        alpha = jnp.full(
            (self.TRGB_sky_exposure_n_support,),
            self.TRGB_sky_exposure_kappa
            / self.TRGB_sky_exposure_n_support)
        theta_support = sample(
            "TRGB_sky_exposure_theta",
            Dirichlet(alpha))
        theta = jnp.zeros(self.TRGB_sky_exposure_n_pix)
        theta = theta.at[support_pix].set(theta_support)
        theta = jnp.where(active_pix, theta, 0.0)
        theta = theta / jnp.clip(jnp.sum(theta), tiny)
        log_theta = jnp.log(theta)
        n_active = jnp.sum(active_pix)
        deterministic("TRGB_sky_exposure_baseline_fraction", q)
        deterministic("TRGB_sky_exposure_theta_full", theta)
        deterministic(
            "TRGB_sky_exposure_ratio",
            theta * n_active)
        return (
            logsumexp(log_S_pix + log_theta[None, :], axis=-1),
            log_theta[self._TRGB_sky_exposure_host_pix],
            log_theta,
        )

    def _call_marginalized(self, h, M_TRGB_host, e2_mag_host,
                           ll_colour_host, sigma_v, beta, bias_params,
                           Vext_rad_host, r_grid, lp_r, log_S,
                           mu_grid=None, z_grid=None,
                           ll_sn_host=None,
                           ll_observed_selection_host=None,
                           Vext_mono_host_grid=None,
                           nu_cz=None):
        if mu_grid is None:
            mu_grid = self.distance2distmod(r_grid, h=h)
        if z_grid is None:
            z_grid = self.distance2redshift(r_grid, h=h)
        if ll_observed_selection_host is None:
            ll_observed_selection_host = jnp.zeros(self.num_hosts)

        log_w = self._simpson_log_w

        ll_cz_fn = self._cz_logpdf_fn(nu_cz)

        ll_mag = normal_logpdf_var(
            self.mag_obs[:, None],
            M_TRGB_host[:, None] + mu_grid[None, :],
            e2_mag_host[:, None])

        if self.use_reconstruction:
            rh_grid = r_grid * h

            delta_grid = self.f_host_los_delta.interp_many(rh_grid)
            log_rho = (jnp.log(1 + delta_grid)
                       if galaxy_bias_needs_log_rho(self.which_bias)
                       else None)
            lp_bias = lp_galaxy_bias(
                delta_grid, log_rho,
                bias_params, self.which_bias)

            # Unnormalized log distance prior (volume + bias)
            lp_dist = lp_r[None, None, :] + lp_bias

            # Redshift likelihood on grid
            if self.use_TRGB_host_redshift:
                if self.use_density_dependent_sigma_v:
                    sigma_v_grid = self.sigma_v_from_density(
                        delta_grid, *sigma_v)
                    e2_cz = self.e2_czcmb[None, :, None] + sigma_v_grid**2
                else:
                    e2_cz = self.e2_czcmb[None, :, None] + sigma_v**2
                Vpec_grid = beta * self.f_host_los_velocity.interp_many(
                    rh_grid)
                Vpec_grid += Vext_rad_host[None, :, None]
                if Vext_mono_host_grid is not None:
                    Vpec_grid += Vext_mono_host_grid[None, None, :]
                cz_pred = predict_cz(z_grid[None, None, :], Vpec_grid)
                ll_cz = ll_cz_fn(
                    self.czcmb[None, :, None], cz_pred,
                    e2_cz)
            else:
                ll_cz = 0.0

            lp_dist_w = lp_dist + log_w

            # Unnormalized distance integral. Selected runs subtract the
            # selection integral; no-selection runs subtract the 3D prior
            # integral over the finite reconstruction volume.
            integrand = lp_dist_w + ll_mag[None, :, :] + ll_cz
            if ll_sn_host is not None:
                integrand = integrand + ll_sn_host[None, :, :]
            if not self.apply_sel:
                in_volume = rh_grid <= self.selection_integral_grid_radius
                integrand = jnp.where(in_volume[None, None, :],
                                      integrand, -jnp.inf)
            ll_host = logsumexp(integrand, axis=-1)

            if self.apply_sel:
                ll_without_selection = ll_host + ll_colour_host[None, :]
                log_selection_integral = log_S
                ll_host = (
                    ll_without_selection
                    - log_selection_integral[:, None]
                )
            else:
                log_norm = self._compute_no_selection_volume_log_norm(
                    bias_params, 100 * h)
                ll_host = (
                    ll_host - log_norm[:, None] + ll_colour_host[None, :])
                ll_without_selection = ll_host
                log_selection_integral = jnp.zeros_like(log_norm)

            if self.save_log_likelihood_per_galaxy:
                if self.num_fields != 1:
                    raise NotImplementedError(
                        "Per-galaxy TRGB log likelihood saving is defined "
                        "only for single-field runs.")
                self._record_per_galaxy_log_likelihood(
                    ll_without_selection[0],
                    ll_selection_observed=ll_observed_selection_host,
                    log_selection_integral=log_selection_integral[0])

            # Take the product over host likelihoods, then average over
            # field realizations.
            ll_host = logmeanexp(jnp.sum(ll_host, axis=1), axis=0)
        else:
            # Distance prior (volume only)
            lp_dist = lp_r[None, :]
            # Redshift likelihood on grid
            if self.use_TRGB_host_redshift:
                e2_cz = self.e2_czcmb[:, None] + sigma_v**2
                Vpec_no_recon = Vext_rad_host[:, None]
                if Vext_mono_host_grid is not None:
                    Vpec_no_recon = (
                        Vpec_no_recon + Vext_mono_host_grid[None, :])
                cz_pred = predict_cz(z_grid[None, :], Vpec_no_recon)
                ll_cz = ll_cz_fn(
                    self.czcmb[:, None], cz_pred, e2_cz)
            else:
                ll_cz = 0.0

            # Marginalize over distance. With explicit selection, keep the
            # same unnormalized r^2 measure as the selection integral so the
            # prior normalization cancels.
            integrand = lp_dist + ll_mag + ll_cz
            if ll_sn_host is not None:
                integrand = integrand + ll_sn_host
            ll_host = ln_simpson_precomputed(integrand, log_w, axis=-1)

            if self.apply_sel:
                ll_without_selection = ll_host + ll_colour_host
                log_selection_integral = log_S[0]
                ll_host = ll_without_selection - log_selection_integral
            else:
                ll_host = ll_host + ll_colour_host
                ll_without_selection = ll_host
                log_selection_integral = 0.0

            if self.save_log_likelihood_per_galaxy:
                self._record_per_galaxy_log_likelihood(
                    ll_without_selection,
                    ll_selection_observed=ll_observed_selection_host,
                    log_selection_integral=log_selection_integral)

        factor("ll_host", jnp.sum(ll_host))
