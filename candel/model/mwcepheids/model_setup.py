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
"""Model setup infrastructure (data loading, caching, grid construction)."""
import logging
import os

import jax.numpy as jnp
import numpy as np

from ...pvdata.dust import postprocess_extinction_profiles
from ..integration import ln_simpson_uniform
from ..utils import load_priors
from .distributions import DiskPrior, sample_disk_sightlines
from .selection import C22SelectionConfig, C27SelectionConfig, SelectionMCData
from .spiral import get_drimmel_arm_traces

logger = logging.getLogger(__name__)


def _mw_data_dir(config):
    root = config.get("io", {}).get("MWCepheids", {}).get("root")
    if root:
        return root
    return config.get("local", {}).get("paths", {}).get("data", ".")


def _cached_spiral_arm_count(config):
    data_dir = _mw_data_dir(config)
    mc_seed = config["model"].get("mc_seed", 42)
    cache_path = os.path.join(data_dir, f"spiral_cache_seed{mc_seed}.npz")
    if not os.path.exists(cache_path):
        return None

    with np.load(cache_path) as cached:
        for key in cached.files:
            if (key.endswith("_obs_dist_sq_per_arm")
                    or key.endswith("_mc_dist_sq_per_arm")):
                return int(cached[key].shape[0])
    return None


class ModelSetupMixin:
    """Infrastructure mixin: data loading, caching, grid construction.

    All methods are extracted verbatim from MWCepheidModel.__init__
    and its helpers. The mixin pattern keeps self.* references unchanged.
    """

    def _init_setup(self, config, data):
        self.config = config
        self.priors = load_priors(config["model"]["priors"])[0]
        model_cfg = config["model"]

        # Anchor galaxy data (geometric distance + Cepheid photometry)
        # Filter to only those listed in config (defaults to none if not set)
        enabled_anchors = model_cfg.get("anchors", [])
        self.anchor_data = {
            name: anc for name, anc in data.anchor_data.items()
            if name in enabled_anchors
        }
        if self.anchor_data:
            logger.info(f"Anchors enabled: {list(self.anchor_data.keys())}")

        # Anchor intrinsic scatter inference
        self.infer_anchor_scatter = (
            self.anchor_data
            and model_cfg.get("anchor_scatter_correction") is not None)
        if self.infer_anchor_scatter:
            logger.info(
                f"Anchor scatter correction: "
                f"{model_cfg['anchor_scatter_correction']} mag subtracted, "
                f"inferring sigma_int_anchor")

        # Fixed metallicity measurement uncertainty
        self.epsilon_OH = model_cfg.get("epsilon_OH", 0.06)

        # Set up campaigns and data
        self._setup_campaigns(data)

        # Reddening-free Q index
        self.use_Q = model_cfg.get("use_Q", False)
        if self.use_Q:
            # Filter out anchor stars with missing Q
            for name in list(self.anchor_data):
                anc = self.anchor_data[name]
                if anc.Q is None and not hasattr(anc, '_Q_valid_mask'):
                    raise ValueError(
                        f"use_Q = true but anchor {name} has no Q column.")
                self.anchor_data[name] = anc.filter_Q_valid()

            for campaign, camp_data in self.data.items():
                if camp_data.Q is None:
                    raise ValueError(
                        f"use_Q = true but {campaign} data has no Q column.")
                n_nan = int(np.sum(np.isnan(np.asarray(camp_data.Q))))
                if n_nan > 0:
                    raise ValueError(
                        f"use_Q = true but {campaign} has {n_nan} stars "
                        f"with NaN Q values. Remove these stars or set "
                        f"use_Q = false.")

            # Override per-star Q_err with population median
            if model_cfg.get("Q_err_median", False):
                for campaign, camp_data in self.data.items():
                    median_err = float(jnp.median(camp_data.Q_err))
                    camp_data.Q_err = jnp.full_like(
                        camp_data.Q_err, median_err)
                    logger.info(
                        f"Q_err median override ({campaign}): "
                        f"{median_err:.4f}")
                for name, anc in self.anchor_data.items():
                    if anc.Q is not None:
                        median_err = float(jnp.median(anc.Q_err))
                        anc.Q_err = jnp.full_like(anc.Q_err, median_err)
                        logger.info(
                            f"Q_err median override ({name}): "
                            f"{median_err:.4f}")

            # Log mean Q for each population
            for campaign, camp_data in self.data.items():
                q = np.asarray(camp_data.Q)
                logger.info(
                    f"Q mean({campaign}): {float(np.mean(q)):.6f} "
                    f"({len(q)} stars)")
            for name, anc in self.anchor_data.items():
                q = np.asarray(anc.Q)
                logger.info(
                    f"Q mean({name}): {float(np.mean(q)):.6f} "
                    f"({anc.n_stars} stars)")

        # Model settings
        self.distance_prior = model_cfg.get("distance_prior", "disk")
        self.model_type = model_cfg.get("model_type", "forward")
        self.shared_scatter = model_cfg.get("shared_scatter", True)
        self.marginalise_distance = model_cfg.get(
            "marginalise_distance", False)

        # Disk prior parameters ("volume" is disk with R_d, z_d -> inf)
        disk_cfg = model_cfg.get("disk_prior", {})
        if self.distance_prior == "volume":
            self.distance_prior = "disk"
            self.disk_R_d = 1e10
            self.disk_z_d = 1e10
        else:
            self.disk_R_d = disk_cfg.get("R_d", 2.5)
            self.disk_z_d = disk_cfg.get("z_d", 0.1)
        self.disk_R_sun = disk_cfg.get("R_sun", 8.122)

        # Spiral arm modulation
        spiral_cfg = model_cfg.get("spiral_arms", {})
        self.apply_spiral_arms = spiral_cfg.get("apply", False)
        if self.apply_spiral_arms:
            n_cached_arms = _cached_spiral_arm_count(config)
            if n_cached_arms is not None:
                self._arms_xy = None
                self._n_arms = n_cached_arms
                logger.info(
                    f"Spiral arms enabled: using {n_cached_arms} arms from "
                    "precomputed cache")
            else:
                self._arms_xy = get_drimmel_arm_traces(
                    R_sun=self.disk_R_sun,
                    use_extrapolated=spiral_cfg.get("use_extrapolated", True),
                    ds=spiral_cfg.get("ds", None))
                self._n_arms = len(self._arms_xy)

        # R21 model parameters
        r21_cfg = model_cfg.get("R21", {})
        self.r21_sigma_int = r21_cfg.get("sigma_int", 0.06)
        self.r21_pi_err_inflation = r21_cfg.get("pi_err_inflation", 1.1)

        # Campaign-specific distance bounds
        self._setup_distance_bounds(model_cfg)

        # Selection configurations
        self._setup_selection(model_cfg)
        self.sel_mc_data = {}

        if self.marginalise_distance and self.distance_prior != "disk":
            raise ValueError(
                "marginalise_distance requires distance_prior = 'disk'. "
                f"Got '{self.distance_prior}'.")

        # Spiral attributes (overwritten by _setup_spiral_cache if active)
        self.spiral_obs_dist_sq_per_arm = {}
        self.spiral_obs_d_grid = {}
        self._obs_disk_unnorm_spiral_grid = {}
        self._obs_disk_log_norm_spiral_grid = {}

        # Distance marginalisation attributes
        self._obs_disk_log_prior = {}
        self._obs_dx = {}
        self._obs_spiral_dsq_on_marg_grid = {}

        # Distance grids and MC samples for selection normalization
        any_sel = (self.sel_c22.mW.apply or self.sel_c22.AH.apply
                   or self.sel_c22.pi.apply or self.sel_c22.logP.apply
                   or self.sel_c27.pi.apply or self.sel_c27.mW.apply)
        needs_grids = any_sel or self.marginalise_distance
        if needs_grids:
            self._setup_distance_grids(model_cfg)

        if any_sel:
            n_mc = model_cfg.get("n_mc_selection", 50)
            self.mc_seed = model_cfg.get("mc_seed", 42)

            # Per-campaign MC sightline dicts (populated by caches
            # or _setup_mc_selection)
            self.ell_mc = {}
            self.b_mc = {}

            # Always try to load MC sightlines from caches
            self._load_cached_mc_sightlines(config, n_mc)

            # Load extinction grids when AH selection is active
            if self.sel_c22.AH.apply and "C22" in self.campaigns:
                self._setup_extinction_grids(config, n_mc)

            # Load spiral cache — provides MC + obs profiles
            if self.apply_spiral_arms:
                self._setup_spiral_cache(config, n_mc)

            self._setup_mc_selection(n_mc, model_cfg)

        elif self.apply_spiral_arms:
            # Spiral active but no selection — still need obs profiles
            self._setup_spiral_cache(config, n_mc=0)

        if self.marginalise_distance:
            self._setup_obs_disk_prior()

    def _setup_campaigns(self, data):
        """Organize data by campaign."""
        if hasattr(data, "campaigns"):
            self.data = data.campaigns
        elif isinstance(data, dict):
            self.data = data
        elif data.campaign is not None:
            self.data = {data.campaign: data}
        else:
            split_data = data.split_by_campaign()
            if split_data:
                self.data = split_data
            else:
                data.campaign = "all"
                self.data = {"all": data}

        self.campaigns = list(self.data.keys())

    def _setup_distance_bounds(self, model_cfg):
        """Set up campaign-specific distance bounds."""
        default_d_min, default_d_max = 0.1, 10.0
        self.d_bounds = {}
        for campaign in self.campaigns:
            camp_cfg = model_cfg.get(campaign, {})
            self.d_bounds[campaign] = {
                "d_min": camp_cfg.get("d_min", default_d_min),
                "d_max": camp_cfg.get("d_max", default_d_max),
            }

    def _setup_selection(self, model_cfg):
        """Set up selection configurations for C22 and C27."""
        c22_cfg = model_cfg.get("C22", {})
        c27_cfg = model_cfg.get("C27", {})

        self.sel_c22 = C22SelectionConfig.from_config(c22_cfg)
        self.sel_c27 = C27SelectionConfig.from_config(c27_cfg)
        self._attach_flat_selection_priors()

        # Validate: selection requires disk prior
        c22_any = (self.sel_c22.mW.apply or self.sel_c22.AH.apply
                   or self.sel_c22.pi.apply or self.sel_c22.logP.apply)
        if c22_any and self.distance_prior != "disk":
            raise ValueError(
                "C22 selection requires distance_prior = 'disk'. "
                f"Got '{self.distance_prior}'.")
        if self.sel_c27.pi.apply and self.distance_prior != "disk":
            raise ValueError(
                "C27 selection requires distance_prior = 'disk'. "
                f"Got '{self.distance_prior}'.")

    def _attach_flat_selection_priors(self):
        """Allow selection priors to live in the top-level prior file."""
        mappings = (
            (self.sel_c22.mW, "mW_max", "mW_max_C22"),
            (self.sel_c22.mW, "mW_width", "mW_width_C22"),
            (self.sel_c22.AH, "AH_max", "AH_max_C22"),
            (self.sel_c22.pi, "pi_min", "pi_min_C22"),
            (self.sel_c22.logP, "logP_min", "logP_min_C22"),
            (self.sel_c22.logP, "logP_width", "logP_width_C22"),
            (self.sel_c27.pi, "pi_min", "pi_min_C27"),
            (self.sel_c27.mW, "mW_min", "mW_min_C27"),
        )
        for selection, local_name, prior_name in mappings:
            if local_name not in selection.priors and prior_name in self.priors:
                selection.priors[local_name] = self.priors[prior_name]

    def _setup_distance_grids(self, model_cfg):
        """Set up distance grids for selection normalization and
        distance marginalisation (separate, denser grid)."""
        self.d_grid = {}

        # --- Selection grids (coarser, d_spacing) ---
        c22_needs_sel = (self.sel_c22.mW.apply or self.sel_c22.AH.apply
                         or self.sel_c22.pi.apply
                         or self.sel_c22.logP.apply)
        if "C22" in self.campaigns and c22_needs_sel:
            c22_cfg = model_cfg.get("C22", {})
            d_spacing = c22_cfg.get("d_spacing", 0.01)
            d_min, d_max = (self.d_bounds["C22"]["d_min"],
                            self.d_bounds["C22"]["d_max"])
            n_grid = int(round((d_max - d_min) / d_spacing)) + 1
            if n_grid % 2 == 0:
                n_grid += 1
            self.d_grid["C22"] = jnp.linspace(d_min, d_max, n_grid)

        c27_needs_sel = self.sel_c27.pi.apply or self.sel_c27.mW.apply
        if "C27" in self.campaigns and c27_needs_sel:
            c27_cfg = model_cfg.get("C27", {})
            d_spacing = c27_cfg.get("d_spacing", 0.01)
            d_min, d_max = (self.d_bounds["C27"]["d_min"],
                            self.d_bounds["C27"]["d_max"])
            n_grid = int(round((d_max - d_min) / d_spacing)) + 1
            if n_grid % 2 == 0:
                n_grid += 1
            self.d_grid["C27"] = jnp.linspace(d_min, d_max, n_grid)

        # --- Marginalisation grids (denser, d_spacing_marg) ---
        self.d_grid_marg = {}
        if self.marginalise_distance:
            for campaign in self.campaigns:
                camp_cfg = model_cfg.get(campaign, {})
                d_spacing_sel = camp_cfg.get("d_spacing", 0.01)
                d_spacing_marg = camp_cfg.get(
                    "d_spacing_marg", d_spacing_sel / 10)
                d_min = self.d_bounds[campaign]["d_min"]
                d_max = self.d_bounds[campaign]["d_max"]
                n_grid = int(round((d_max - d_min) / d_spacing_marg)) + 1
                if n_grid % 2 == 0:
                    n_grid += 1
                self.d_grid_marg[campaign] = jnp.linspace(
                    d_min, d_max, n_grid)
                logger.info(
                    f"Marg grid ({campaign}): n={n_grid}, "
                    f"dx={d_spacing_marg} kpc")

    def _spiral_cache_path(self, config):
        """Build cache file path for spiral profiles."""
        data_dir = _mw_data_dir(config)
        mc_seed = config["model"].get("mc_seed", 42)
        return os.path.join(data_dir, f"spiral_cache_seed{mc_seed}.npz")

    def _setup_spiral_cache(self, config, n_mc):
        """Load precomputed per-arm dist_sq profiles from cache.

        Run ``scripts/precompute_spiral.py`` first to create the cache.
        Loads MC and observed-star per-arm dist_sq profiles, validates
        coordinates and geometry parameters against the current
        config/data.

        Also precomputes disk-only quantities on the obs grids that are
        parameter-independent (needed for runtime normalization).

        Sets ``self.ell_mc``, ``self.b_mc`` from the cache when they
        haven't been set by the extinction cache, so that
        ``_setup_mc_selection`` reuses them.
        """
        cache_path = self._spiral_cache_path(config)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Spiral cache not found: {cache_path}\n"
                "Run scripts/precompute_spiral.py first.")

        logger.info(f"Loading spiral cache from {cache_path}")
        cached = np.load(cache_path)

        # Validate geometry parameters
        for param, attr in [("R_sun", "disk_R_sun"),
                            ("R_d", "disk_R_d"),
                            ("z_d", "disk_z_d")]:
            cached_val = float(cached[param])
            config_val = getattr(self, attr)
            if not np.isclose(cached_val, config_val, rtol=1e-6):
                raise ValueError(
                    f"Spiral cache {param}={cached_val} != "
                    f"config {config_val}. Re-run "
                    "scripts/precompute_spiral.py.")

        # --- MC sightlines (per campaign) ---
        if n_mc > 0:
            self._spiral_mc_cache = {}
            for campaign in self.d_grid:
                key_ell = f"{campaign}_ell_mc"
                key_b = f"{campaign}_b_mc"
                key_mc = f"{campaign}_mc_dist_sq_per_arm"
                key_dg = f"{campaign}_d_grid"

                if key_ell not in cached or key_mc not in cached:
                    continue

                n_cached_mc = len(cached[key_ell])
                if n_cached_mc < n_mc:
                    raise ValueError(
                        f"Spiral cache has {n_cached_mc} MC sightlines "
                        f"for {campaign} but need {n_mc}. Re-run "
                        "scripts/precompute_spiral.py.")

                ell_mc_cached = cached[key_ell][:n_mc]
                b_mc_cached = cached[key_b][:n_mc]

                # Validate against extinction cache coords if present
                if campaign in self.ell_mc:
                    if not (np.allclose(ell_mc_cached,
                                        np.asarray(self.ell_mc[campaign]),
                                        atol=1e-6)
                            and np.allclose(b_mc_cached,
                                            np.asarray(self.b_mc[campaign]),
                                            atol=1e-6)):
                        raise ValueError(
                            f"Spiral cache {campaign} MC coords don't "
                            "match extinction cache. Re-run "
                            "scripts/precompute_spiral.py.")
                else:
                    self.ell_mc[campaign] = jnp.array(ell_mc_cached)
                    self.b_mc[campaign] = jnp.array(b_mc_cached)

                self._spiral_mc_cache[campaign] = {
                    "dist_sq_per_arm": cached[key_mc][:, :n_mc],
                    "d_grid": cached[key_dg],
                }
                logger.info(
                    f"  {campaign} MC: {n_mc} sightlines, "
                    f"grid n={len(cached[key_dg])}")

        # --- Observed star dist_sq_per_arm + precomputed disk quantities ---
        self.spiral_obs_dist_sq_per_arm = {}
        self.spiral_obs_d_grid = {}
        self._obs_disk_unnorm_spiral_grid = {}
        self._obs_disk_log_norm_spiral_grid = {}

        for campaign, data in self.data.items():
            if data.ell is None:
                continue

            key_ell = f"{campaign}_ell_obs"
            if key_ell not in cached:
                logger.warning(
                    f"  No spiral data for {campaign} in cache")
                continue

            ell_obs = np.asarray(data.ell)
            b_obs = np.asarray(data.b)
            ell_cached_obs = cached[key_ell]
            b_cached_obs = cached[f"{campaign}_b_obs"]

            if (len(ell_cached_obs) == len(ell_obs)
                    and np.allclose(ell_cached_obs, ell_obs, atol=1e-8)
                    and np.allclose(b_cached_obs, b_obs, atol=1e-8)):
                obs_idx = np.arange(len(ell_obs))
            else:
                obs_idx = []
                for i in range(len(ell_obs)):
                    matches = np.where(
                        (np.abs(ell_cached_obs - ell_obs[i]) < 1e-8)
                        & (np.abs(b_cached_obs - b_obs[i]) < 1e-8))[0]
                    if len(matches) != 1:
                        raise ValueError(
                            f"{campaign} star {i} (ell={ell_obs[i]:.4f}, "
                            f"b={b_obs[i]:.4f}) not found uniquely in "
                            "spiral cache. Re-run "
                            "scripts/precompute_spiral.py.")
                    obs_idx.append(matches[0])
                obs_idx = np.array(obs_idx)
                logger.info(
                    f"  {campaign}: matched {len(obs_idx)}/"
                    f"{len(ell_cached_obs)} cached obs stars")

            d_grid = cached[f"{campaign}_d_grid"]
            d_grid_jnp = jnp.array(d_grid)
            self.spiral_obs_dist_sq_per_arm[campaign] = jnp.array(
                cached[f"{campaign}_obs_dist_sq_per_arm"][:, obs_idx])
            self.spiral_obs_d_grid[campaign] = d_grid_jnp

            # Precompute disk-only quantities on this grid (no A, sigma).
            # Use disk._log_norm (not ln_simpson) for exact cancellation
            # with DiskPrior.log_prob in _likelihood_campaign.
            d_min = float(d_grid[0])
            d_max = float(d_grid[-1])
            disk = DiskPrior(
                ell_obs, b_obs, d_min, d_max,
                R_d=self.disk_R_d, z_d=self.disk_z_d,
                R_sun=self.disk_R_sun)
            disk_unnorm = disk._unnorm_log_prob(d_grid_jnp, grid_mode=True)
            self._obs_disk_unnorm_spiral_grid[campaign] = disk_unnorm
            self._obs_disk_log_norm_spiral_grid[campaign] = disk._log_norm

            logger.info(
                f"  {campaign} obs: {len(ell_obs)} stars, "
                f"grid [{d_grid[0]:.2f}, {d_grid[-1]:.2f}] kpc, "
                f"n={len(d_grid)}")

    def _setup_obs_disk_prior(self):
        """Precompute unnorm disk prior for observed stars on marg grids.

        Uses the denser ``d_grid_marg`` (not the selection grid).
        Also reinterpolates spiral and AH profiles onto the
        marginalisation grid when needed.
        """
        for campaign, data in self.data.items():
            if campaign not in self.d_grid_marg:
                continue
            d_grid = self.d_grid_marg[campaign]
            disk = DiskPrior(
                data.ell, data.b, float(d_grid[0]), float(d_grid[-1]),
                R_d=self.disk_R_d, z_d=self.disk_z_d,
                R_sun=self.disk_R_sun)
            self._obs_disk_log_prior[campaign] = disk._unnorm_log_prob(
                d_grid, grid_mode=True)
            self._obs_dx[campaign] = float(d_grid[1] - d_grid[0])
            logger.info(
                f"Obs disk prior ({campaign}): {data.n_stars} stars, "
                f"marg grid n={len(d_grid)}")

            # Reinterpolate spiral dist_sq_per_arm to marg grid
            if (self.apply_spiral_arms
                    and campaign in self.spiral_obs_dist_sq_per_arm):
                sp_d = np.asarray(self.spiral_obs_d_grid[campaign])
                marg_d = np.asarray(d_grid)
                dsq = self.spiral_obs_dist_sq_per_arm[campaign]

                if (len(sp_d) == len(marg_d)
                        and np.allclose(sp_d, marg_d, atol=1e-10)):
                    self._obs_spiral_dsq_on_marg_grid[campaign] = dsq
                else:
                    dsq_np = np.asarray(dsq)
                    dsq_new = np.stack([
                        self._reinterp_grid(dsq_np[k], sp_d, marg_d)
                        for k in range(dsq_np.shape[0])])
                    self._obs_spiral_dsq_on_marg_grid[campaign] = jnp.array(
                        dsq_new)
                    logger.info(
                        f"  Reinterpolated {campaign} obs spiral profiles "
                        f"to marg grid (n={len(marg_d)})")

            # Reinterpolate AH obs profiles to marg grid (C22 only)
            if (campaign == "C22" and self.sel_c22.AH.apply
                    and hasattr(self, 'AH_obs_profiles')):
                sel_d = np.asarray(self.d_grid.get("C22", d_grid))
                marg_d = np.asarray(d_grid)
                if (len(sel_d) == len(marg_d)
                        and np.allclose(sel_d, marg_d, atol=1e-10)):
                    self.AH_obs_profiles_marg = self.AH_obs_profiles
                else:
                    AH_np = np.asarray(self.AH_obs_profiles)
                    self.AH_obs_profiles_marg = jnp.array(
                        self._reinterp_grid(AH_np, sel_d, marg_d))
                    logger.info(
                        f"  Reinterpolated C22 AH obs profiles "
                        f"to marg grid (n={len(marg_d)})")

    def _load_cached_mc_sightlines(self, config, n_mc):
        """Load MC sightline coordinates from any available cache.

        Tries the spiral cache first (per-campaign), then the extinction
        cache (C22 only).  Validates that both caches agree when they
        overlap.  Silently skips if no cache exists.
        """
        # Spiral cache first (per-campaign sightlines)
        spiral_path = self._spiral_cache_path(config)
        if self.apply_spiral_arms and os.path.exists(spiral_path):
            cached = np.load(spiral_path)
            for campaign in self.d_grid:
                if campaign in self.ell_mc:
                    continue
                key_ell = f"{campaign}_ell_mc"
                key_b = f"{campaign}_b_mc"
                if key_ell not in cached:
                    continue
                n_cached = len(cached[key_ell])
                if n_cached >= n_mc:
                    self.ell_mc[campaign] = jnp.array(
                        cached[key_ell][:n_mc])
                    self.b_mc[campaign] = jnp.array(
                        cached[key_b][:n_mc])
                    logger.info(
                        f"Loaded {campaign} MC sightlines from "
                        f"spiral cache ({n_mc})")

        # Extinction cache (C22 sightlines)
        if self.sel_c22.AH.apply and "C22" in self.d_grid:
            cache_path = self._extinction_cache_path(config)
            if os.path.exists(cache_path):
                cached = np.load(cache_path)
                n_cached = len(cached["ell_mc"])
                if n_cached >= n_mc:
                    ell_ext = cached["ell_mc"][:n_mc]
                    b_ext = cached["b_mc"][:n_mc]

                    if "C22" in self.ell_mc:
                        # Validate against spiral cache
                        if not (np.allclose(ell_ext,
                                            np.asarray(self.ell_mc["C22"]),
                                            atol=1e-6)
                                and np.allclose(b_ext,
                                                np.asarray(self.b_mc["C22"]),
                                                atol=1e-6)):
                            raise ValueError(
                                "C22 MC sightlines in extinction cache "
                                "don't match spiral cache. Re-run "
                                "precompute scripts.")
                    else:
                        self.ell_mc["C22"] = jnp.array(ell_ext)
                        self.b_mc["C22"] = jnp.array(b_ext)
                        logger.info(
                            f"Loaded C22 MC sightlines from "
                            f"extinction cache ({n_mc})")

    def _setup_mc_selection(self, n_mc, model_cfg):
        """Set up Monte Carlo sightlines for selection normalization.

        Draws per-campaign lines of sight (ell, b) proportional to disk
        column density (integrated over the campaign's distance range).
        Computes median measurement errors per campaign (used as
        representative values in the analytical selection normalisation).

        If ``self.ell_mc[campaign]`` is already set (loaded from the
        extinction or spiral cache), those coordinates are reused.
        """
        self.n_mc = n_mc
        ss = np.random.SeedSequence(self.mc_seed)
        ss_sky, _ = ss.spawn(2)  # Keep MC sky draws reproducible.

        # Per-campaign sky RNGs (sorted keys for determinism)
        mc_cfg = model_cfg.get("selection_mc", {})
        ell_min = mc_cfg.get("ell_min", 0.0)
        ell_max = mc_cfg.get("ell_max", 360.0)
        b_min = mc_cfg.get("b_min", -90.0)
        b_max = mc_cfg.get("b_max", 90.0)

        campaigns_sorted = sorted(self.d_grid.keys())
        sky_rngs = {c: np.random.default_rng(s)
                    for c, s in zip(campaigns_sorted,
                                    ss_sky.spawn(len(campaigns_sorted)))}

        for campaign in campaigns_sorted:
            if campaign not in self.ell_mc:
                d_min = self.d_bounds[campaign]["d_min"]
                d_max = self.d_bounds[campaign]["d_max"]
                ell_mc, b_mc = sample_disk_sightlines(
                    n_mc, ell_min, ell_max, b_min, b_max,
                    d_min, d_max,
                    self.disk_R_d, self.disk_z_d, self.disk_R_sun,
                    sky_rngs[campaign])
                self.ell_mc[campaign] = jnp.array(ell_mc)
                self.b_mc[campaign] = jnp.array(b_mc)

        # Median measurement errors per campaign (analytical marg. uses
        # a single representative sigma_m and sigma_pi per campaign)
        self.median_sigma_m = {}
        self.median_sigma_pi = {}
        for campaign, camp_data in self.data.items():
            mW_err = np.asarray(camp_data.mW_H_err)
            pi_err = np.asarray(camp_data.pi_EDR3_err)
            self.median_sigma_m[campaign] = float(np.median(mW_err))
            self.median_sigma_pi[campaign] = float(np.median(pi_err))

        # Precompute unnormalized disk prior on distance grids
        self.mc_log_dist_prior = {}
        self.mc_log_dist_norm = {}
        self.mu_grid = {}
        self.inv_d_grid = {}
        self.mc_spiral_dist_sq_per_arm_grid = {}
        for campaign, d_grid in self.d_grid.items():
            disk = DiskPrior(
                self.ell_mc[campaign], self.b_mc[campaign],
                float(d_grid[0]), float(d_grid[-1]),
                R_d=self.disk_R_d, z_d=self.disk_z_d,
                R_sun=self.disk_R_sun)
            mc_disk_unnorm = disk._unnorm_log_prob(
                d_grid, grid_mode=True)

            if self.apply_spiral_arms:
                mc_spiral = self._spiral_mc_cache[campaign]
                d_cached = mc_spiral["d_grid"]
                dsq = mc_spiral["dist_sq_per_arm"]
                d_grid_np = np.asarray(d_grid)
                if (len(d_cached) != len(d_grid_np)
                        or not np.allclose(
                            d_cached, d_grid_np, atol=1e-10)):
                    # Reinterpolate each arm separately
                    dsq = np.stack([
                        self._reinterp_grid(dsq[k], d_cached, d_grid_np)
                        for k in range(dsq.shape[0])])
                    logger.info(
                        f"  Reinterpolated {campaign} MC spiral "
                        f"dist_sq_per_arm to selection grid "
                        f"(n={len(d_grid_np)})")
                self.mc_spiral_dist_sq_per_arm_grid[campaign] = jnp.array(dsq)
                # Without spiral: mc_log_dist_prior = disk only
                self.mc_log_dist_prior[campaign] = mc_disk_unnorm
            else:
                self.mc_log_dist_prior[campaign] = mc_disk_unnorm

            dx = float(d_grid[1] - d_grid[0])
            self.mc_log_dist_norm[campaign] = ln_simpson_uniform(
                self.mc_log_dist_prior[campaign], dx, axis=-1)
            self.mu_grid[campaign] = 5.0 * jnp.log10(d_grid) + 10.0
            self.inv_d_grid[campaign] = 1.0 / d_grid

        # Build SelectionMCData per campaign
        for campaign, d_grid in self.d_grid.items():
            dx = float(d_grid[1] - d_grid[0])
            mc = SelectionMCData(
                d_grid=d_grid,
                dx=dx,
                mu_grid=self.mu_grid[campaign],
                inv_d_grid=self.inv_d_grid[campaign],
                median_sigma_m=self.median_sigma_m.get(campaign, 0.0),
                median_sigma_pi=self.median_sigma_pi.get(campaign, 0.0),
                mc_log_dist_prior=self.mc_log_dist_prior[campaign],
                mc_log_dist_norm=self.mc_log_dist_norm[campaign],
                n_mc=self.n_mc,
            )
            if (campaign == "C22" and self.sel_c22.AH.apply
                    and hasattr(self, 'AH_mc_grid')):
                mc.AH_mc_grid = self.AH_mc_grid
                mc.AH_mc_valid = self.AH_mc_valid
                mc.AH_obs_profiles = self.AH_obs_profiles
                mc.AH_obs_star_valid = self.AH_obs_star_valid
            if (self.apply_spiral_arms
                    and campaign in self.mc_spiral_dist_sq_per_arm_grid):
                mc.spiral_dist_sq_per_arm_grid = \
                    self.mc_spiral_dist_sq_per_arm_grid[campaign]
            self.sel_mc_data[campaign] = mc

    def _extinction_cache_path(self, config):
        """Build cache file path for extinction grids."""
        data_dir = _mw_data_dir(config)
        dust_map = self.sel_c22.dust_map
        fname = (f"AH_cache_{dust_map}"
                 f"_seed{self.mc_seed}.npz")
        return os.path.join(data_dir, fname)

    @staticmethod
    def _reinterp_grid(profiles, d_cached, d_new):
        """Reinterpolate profiles onto a new distance grid.

        Parameters
        ----------
        profiles : ndarray, shape (n_los, n_cached)
            Per-sightline profiles (e.g. extinction, dist_sq_per_arm).
        d_cached : ndarray, shape (n_cached,)
        d_new : ndarray, shape (n_new,)

        Returns
        -------
        ndarray, shape (n_los, n_new)
        """
        out = np.empty((profiles.shape[0], len(d_new)), dtype=profiles.dtype)
        for i in range(profiles.shape[0]):
            out[i] = np.interp(d_new, d_cached, profiles[i])
        return out

    def _setup_extinction_grids(self, config, n_mc):
        """Load precomputed A_H extinction grids from cache.

        Run ``scripts/precompute_extinction.py`` first to create the
        cache file.  MC sky coordinates are loaded from the cache
        (rejection-sampled for dust map coverage) and stored as
        ``self.ell_mc``, ``self.b_mc`` so that ``_setup_mc_selection``
        reuses them.  Observed-star coordinates are validated against
        the current data.  If the distance grid differs from the cache
        it is reinterpolated automatically.
        """
        cache_path = self._extinction_cache_path(config)
        d_grid_np = np.asarray(self.d_grid["C22"])
        data = self.data["C22"]

        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Extinction cache not found: {cache_path}\n"
                "Run scripts/precompute_extinction.py first.")

        logger.info(
            f"Loading extinction cache from {cache_path}")
        cached = np.load(cache_path)
        AH_mc = cached["AH_mc_grid"]
        AH_mc_valid = cached["AH_mc_valid"]
        AH_obs = cached["AH_obs_profiles"]
        AH_obs_valid = cached["AH_obs_valid"]

        # ----- Slice MC arrays if cache has more sightlines ---
        n_cached_mc = AH_mc.shape[0]
        if n_cached_mc < n_mc:
            raise ValueError(
                f"Cache has {n_cached_mc} MC sightlines but "
                f"n_mc={n_mc}. Re-run "
                "scripts/precompute_extinction.py with at "
                f"least n_mc_selection = {n_mc}.")
        AH_mc = AH_mc[:n_mc]
        AH_mc_valid = AH_mc_valid[:n_mc]

        # ----- Validate / load MC sky coords ---
        ell_ext = cached["ell_mc"][:n_mc]
        b_ext = cached["b_mc"][:n_mc]
        if "C22" in self.ell_mc:
            if not (np.allclose(ell_ext,
                                np.asarray(self.ell_mc["C22"]),
                                atol=1e-6)
                    and np.allclose(b_ext,
                                    np.asarray(self.b_mc["C22"]),
                                    atol=1e-6)):
                raise ValueError(
                    "C22 MC sightlines in extinction cache "
                    "don't match previously loaded coords. "
                    "Re-run precompute scripts.")
        else:
            self.ell_mc["C22"] = jnp.array(ell_ext)
            self.b_mc["C22"] = jnp.array(b_ext)

        # ----- Match observed star coordinates -----
        # Current data may be a subset of the cache (e.g. excluded stars).
        # Find the matching rows in the cache.
        ell_obs_np = np.asarray(data.ell)
        b_obs_np = np.asarray(data.b)
        ell_cached = cached["ell_obs"]
        b_cached = cached["b_obs"]

        if (len(ell_cached) == len(ell_obs_np)
                and np.allclose(ell_cached, ell_obs_np, atol=1e-8)
                and np.allclose(b_cached, b_obs_np, atol=1e-8)):
            obs_idx = np.arange(len(ell_obs_np))
        else:
            # Find each current star in the cache by coordinates
            obs_idx = []
            for i in range(len(ell_obs_np)):
                matches = np.where(
                    (np.abs(ell_cached - ell_obs_np[i]) < 1e-8)
                    & (np.abs(b_cached - b_obs_np[i]) < 1e-8))[0]
                if len(matches) != 1:
                    raise ValueError(
                        f"Star {i} (ell={ell_obs_np[i]:.4f}, "
                        f"b={b_obs_np[i]:.4f}) not found uniquely "
                        "in extinction cache. Re-run "
                        "scripts/precompute_extinction.py.")
                obs_idx.append(matches[0])
            obs_idx = np.array(obs_idx)
            logger.info(
                f"  Matched {len(obs_idx)}/{len(ell_cached)} "
                f"cached obs stars to current data")

        AH_obs = AH_obs[obs_idx]
        AH_obs_valid = AH_obs_valid[obs_idx]

        # ----- Reinterpolate if distance grid changed -----
        d_cached = cached.get("d_grid", None)
        grids_match = (d_cached is not None
                       and d_cached.shape == d_grid_np.shape
                       and np.allclose(d_cached, d_grid_np, atol=1e-10))
        if d_cached is not None and not grids_match:
            logger.info(
                "  Distance grid differs from cache — "
                "reinterpolating extinction profiles")
            AH_mc = self._reinterp_grid(
                AH_mc, d_cached, d_grid_np)
            AH_mc_valid = self._reinterp_grid(
                AH_mc_valid.astype(float),
                d_cached, d_grid_np) > 0.5
            AH_obs = self._reinterp_grid(
                AH_obs, d_cached, d_grid_np)
            AH_obs_valid = self._reinterp_grid(
                AH_obs_valid.astype(float),
                d_cached, d_grid_np) > 0.5

        # Post-process observed profiles:
        # forward-fill NaN, enforce monotonicity
        AH_obs_proc = postprocess_extinction_profiles(AH_obs)

        # Per-star validity: >50% grid coverage
        n_grid = len(d_grid_np)
        AH_obs_star_valid = (
            np.sum(AH_obs_valid, axis=1) > 0.5 * n_grid)

        # Replace remaining NaN with 0 for safe JAX usage
        AH_mc = np.where(np.isfinite(AH_mc), AH_mc, 0.0)

        # Store as JAX arrays
        self.AH_mc_grid = jnp.array(AH_mc)
        self.AH_mc_valid = jnp.array(AH_mc_valid)
        self.AH_obs_profiles = jnp.array(AH_obs_proc)
        self.AH_obs_star_valid = jnp.array(AH_obs_star_valid)

        # Log coverage statistics
        mc_cov = float(np.mean(AH_mc_valid))
        obs_cov = float(np.mean(AH_obs_valid))
        n_valid = int(np.sum(AH_obs_star_valid))
        logger.info(f"  MC grid coverage:  {mc_cov:.1%}")
        logger.info(f"  Obs grid coverage: {obs_cov:.1%}")
        logger.info(
            f"  Valid obs stars:   {n_valid}/{data.n_stars}")
