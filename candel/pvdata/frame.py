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
Data loading and preprocessing utilities for peculiar-velocity catalogues.

Provides dataframe-like containers, LOS interpolation helpers, covariance
assembly, and catalogue I/O wired to the project config files.
"""
import numpy as np
from jax import core as jcore
from jax import numpy as jnp

from ..model.integration import simpson_log_weights
from ..model.interp import LOSInterpolator
from ..util import (SPEED_OF_LIGHT, fprint, fsection, get_nested, load_config,
                    radec_to_cartesian)
from .catalogues import _CATALOGUE_LOADERS, load_CF4_data, load_CF4_mock
from .field_cache import (_field_cache_dir_from_config,
                          _field_cache_enabled_from_config)
from .los import _compute_r_grid, precompute_pixel_projection
from .volume_density import (_load_volume_density_3d_fields,
                             _prepare_pv_volume_density_arrays,
                             _reconstruction_omega_m,
                             _validate_voxel_subsample_fraction,
                             _validate_voxel_subsample_seed,
                             _volume_density_mode)


def load_PV_dataframes(config_path):
    """Load PVDataFrame objects from a configuration file."""
    config = load_config(config_path)

    if config["pv_model"]["kind"].startswith("precomputed_los_"):
        los_reconstruction = config["pv_model"]["kind"].replace(
            "precomputed_los_", "")
    else:
        los_reconstruction = None

    config_io = config["io"]
    config_pv_model = config["pv_model"]
    names = config_io.pop("catalogue_name")
    if isinstance(names, str):
        names = [names]

    dfs = []
    fsection("Data")
    fprint(f"loading {len(names)} PV dataframes: {names}")
    multi = len(names) > 1
    for name in names:
        if multi:
            fprint(f"--- {name} ---")
        is_mock = name.startswith("CF4_mock")
        if is_mock:
            kwargs = config_io["CF4_mock"].copy()
        else:
            kwargs = config_io[name].copy()

        try_pop_los = is_mock and los_reconstruction is None
        if los_reconstruction is not None and not is_mock:
            kwargs["los_data_path"] = kwargs.pop("los_file").replace(
                "<X>", los_reconstruction)
            fprint(
                f"loading existing LOS data from {kwargs['los_data_path']}.")

        recon_kwargs = None
        if los_reconstruction is not None:
            recon_main = config_io.get("reconstruction_main", {})
            recon_kwargs = recon_main.get(los_reconstruction, None)
        field_cache_enabled = _field_cache_enabled_from_config(
            config, config_pv_model)
        field_cache_dir = (
            _field_cache_dir_from_config(config, config_pv_model)
            if field_cache_enabled else None)
        if los_reconstruction is not None:
            if field_cache_enabled:
                fprint(f"field cache enabled: `{field_cache_dir}`.")
            else:
                fprint("field cache disabled.")

        df = PVDataFrame.from_config_dict(
            kwargs, name, try_pop_los=try_pop_los,
            config_pv_model=config_pv_model,
            reconstruction_kwargs=recon_kwargs,
            reconstruction_name=los_reconstruction,
            field_cache_dir=field_cache_dir,
            field_cache_enabled=field_cache_enabled)
        dfs.append(df)

    if len(dfs) == 1:
        return dfs[0]

    return dfs


class PVDataFrame:
    """Lightweight container for PV data."""
    add_eta_truncation = False

    def __init__(self, data, los_radial_decay_scale=5):
        # Convert numeric arrays to JAX, skip string arrays
        self.data = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.str_):
                continue
            self.data[k] = jnp.asarray(v)
        self.name = None

        if "los_velocity" in self.data:
            self.has_precomputed_los = True
            self.num_fields = self.data["los_delta"].shape[0]
            fprint(f"marginalising over {self.num_fields} field realisations.")

            kwargs = {"r0_decay_scale": los_radial_decay_scale}
            self.f_los_delta = LOSInterpolator(
                self.data["los_r"], self.data["los_delta"], **kwargs)
            self.f_los_log_density = LOSInterpolator(
                self.data["los_r"], jnp.log(self.data["los_density"]),
                **kwargs)
            self.f_los_velocity = LOSInterpolator(
                self.data["los_r"], self.data["los_velocity"], **kwargs)

            self.data["los_delta_r_grid"] = self.f_los_delta.interp_many_steps_per_galaxy(self.data["r_grid"])              # noqa
            self.data["los_velocity_r_grid"] = self.f_los_velocity.interp_many_steps_per_galaxy(self.data["r_grid"])        # noqa
            self.data["los_log_density_r_grid"] = self.f_los_log_density.interp_many_steps_per_galaxy(self.data["r_grid"])  # noqa
        else:
            self.num_fields = 1
            self.has_precomputed_los = False

        # Pre-compute Simpson log weights for the radial grid.
        if "r_grid" in self.data:
            self._simpson_log_w = simpson_log_weights(self.data["r_grid"])
            # Reused every step in the LOS integrand (`(r/R)^q` and Jacobian).
            self.data["log_r_grid"] = jnp.log(self.data["r_grid"])
            self.data["log_jac_los"] = 2.0 * self.data["log_r_grid"]
        else:
            self._simpson_log_w = None

        self.has_calibrators = bool(self.num_calibrators > 0)
        self._cache = {}
        self.has_volume_density_3d = False

    def attach_volume_density_3d(self, rho_3d, observer_pos, dx,
                                 galaxy_bias="linear", geometry="cube",
                                 radius=None, store_rhat_3d=False,
                                 coordinate_frame="icrs",
                                 voxel_subsample_fraction=1.0,
                                 voxel_subsample_seed=42):
        """Attach 3D density voxels for the volume-normalized empirical prior.

        Stores the minimal density representation needed by `galaxy_bias`,
        plus `log_r_3d` (log voxel distance from the observer; floored at
        `0.25 dx` so the central voxel is finite). The voxel log-volume
        `3 log(dx)` is stored on ``self.log_dV_3d``.

        `log_r_3d` is precomputed so the per-step `(r/R)^q` is evaluated as
        `exp(q · (log_r_3d − log R))`, avoiding ~ngrid^3 `log` ops per leapfrog
        step. The `0.25 dx` floor at the central voxel only affects a single
        cell whose `(r/R)^q` is O((dx/R)^q) ≈ 0 anyway.

        If ``store_rhat_3d`` is true, also stores voxel directions in
        ``coordinate_frame`` for missing-mass volume terms.
        """
        self.attach_volume_density_3d_fields(
            [(rho_3d, observer_pos, dx, coordinate_frame)],
            galaxy_bias=galaxy_bias, geometry=geometry, radius=radius,
            store_rhat_3d=store_rhat_3d,
            voxel_subsample_fraction=voxel_subsample_fraction,
            voxel_subsample_seed=voxel_subsample_seed)

    def attach_volume_density_3d_fields(self, fields, batch_size=1,
                                        galaxy_bias="linear",
                                        geometry="cube", radius=None,
                                        store_rhat_3d=False,
                                        voxel_subsample_fraction=1.0,
                                        voxel_subsample_seed=42):
        """Attach one 3D density field per field realisation.

        The model maps over the leading field axis with an explicit batch size,
        avoiding full-field vectorization of intermediates during the
        normalizer calculation. `geometry="sphere"` stores only voxels with
        non-zero fractional volume inside `radius` (Mpc/h) as flattened 1D
        arrays; `geometry="cube"` stores the full cube/sub-cube. If
        ``store_rhat_3d`` is true, each field item must include a coordinate
        frame or defaults to ``"icrs"``.
        """
        prepared = _prepare_pv_volume_density_arrays(
            fields, geometry=geometry, radius=radius,
            store_rhat_3d=store_rhat_3d,
            voxel_subsample_fraction=voxel_subsample_fraction,
            voxel_subsample_seed=voxel_subsample_seed)
        self.attach_prepared_volume_density_3d_fields(
            prepared, batch_size=batch_size, galaxy_bias=galaxy_bias,
            geometry=geometry, radius=radius)

    def attach_prepared_volume_density_3d_fields(
            self, prepared, batch_size=1, galaxy_bias="linear",
            geometry="cube", radius=None):
        """Attach compact cached PV volume-normalizer arrays."""
        mode = _volume_density_mode(galaxy_bias)
        rho_fields = np.asarray(prepared["rho_fields"], dtype=np.float32)
        if mode == "log_rho":
            density_fields = np.log(rho_fields).astype(np.float32)
        else:
            density_fields = (rho_fields - 1.0).astype(np.float32)

        self.data["density_3d_fields"] = jnp.asarray(density_fields)
        self.data["log_r_3d"] = jnp.asarray(prepared["log_r_3d"])
        if "log_volume_weight_3d" in prepared:
            self.data["log_volume_weight_3d"] = jnp.asarray(
                prepared["log_volume_weight_3d"])
        for label in ("rhat_x_3d", "rhat_y_3d", "rhat_z_3d"):
            if label in prepared:
                self.data[label] = jnp.asarray(prepared[label])

        self.log_dV_3d = float(prepared["log_dV_3d"])
        self.density_3d_mode = mode
        self.volume_density_batch_size = int(batch_size)
        self.density_3d_geometry = geometry
        self.density_3d_radius = radius
        coord_frame = prepared.get("coordinate_frame", "icrs")
        if not isinstance(coord_frame, str):
            coord_frame = str(np.asarray(coord_frame).item())
        self.coordinate_frame_3d = coord_frame
        self.has_volume_density_3d = True

    @classmethod
    def from_config_dict(cls, config, name, try_pop_los, config_pv_model,
                         reconstruction_kwargs=None, reconstruction_name=None,
                         field_cache_dir=None, field_cache_enabled=True):
        root = config.pop("root")
        nsamples_subsample = config.pop("nsamples_subsample", None)
        seed_subsample = config.pop("seed_subsample", 42)
        sample_dust = False

        smooth_target = config_pv_model.get("smooth_target", None)
        if smooth_target is not None:
            config["los_data_path"] = config["los_data_path"].replace(
                ".hdf5", f"_smooth_to_{smooth_target}.hdf5")

        if "CF4_mock" in name:
            index = name.split("_")[-1]
            data = load_CF4_mock(root, index)
        elif "CF4_" in name:
            data = load_CF4_data(root, **config)

            dust_model = config.get("dust_model", None)
            if dust_model is not None:
                fprint(f"using `{dust_model}` for the dust model.")
                sample_dust = True
        elif name in _CATALOGUE_LOADERS:
            data = _CATALOGUE_LOADERS[name](root, **config)
        else:
            raise ValueError(f"Unknown catalogue name: {name}")

        if try_pop_los:
            for key in list(data.keys()):
                if key.startswith("los_"):
                    fprint(f"removing `{key}` from data.")
                    data.pop(key, None)

        r_limits = config_pv_model["r_limits_malmquist"]
        dr = config_pv_model["dr_malmquist"]
        Om_model = config_pv_model.get("Om", config_pv_model.get("Om0", 0.3))
        Om = _reconstruction_omega_m(
            reconstruction_name, reconstruction_kwargs, fallback=Om_model)
        if reconstruction_name is not None and not np.isclose(Om, Om_model):
            fprint(
                f"using reconstruction Om0={Om:g} for "
                f"`{reconstruction_name}` instead of model Om0={Om_model:g}.")
        data["r_grid"] = _compute_r_grid(r_limits, dr, data, Om)

        los_decay_scale = config_pv_model.get("los_decay_scale", 5.0)
        fprint(f"setting los_decay_scale to {los_decay_scale}")

        if "los_density" in data:
            data["los_log_density"] = np.log(data["los_density"])
            data["los_delta"] = data["los_density"] - 1

        if nsamples_subsample is not None:
            if name == "PantheonPlusLane":
                raise ValueError(
                    "Subsampling for Pantheon+ Lane is not supported because "
                    "of the complicated covariance matrix.")

            frame = cls(data, los_decay_scale)
            frame = frame.subsample(
                nsamples_subsample, los_decay_scale, seed=seed_subsample)
        else:
            frame = cls(data, los_decay_scale)

        frame.sample_dust = sample_dust

        # Precompute Vext_per_pix data
        nside = config_pv_model.get("Vext_per_pix_nside", None)
        if nside is not None:
            fprint(f"precomputing Vext_per_pix data for nside = {nside}.")
            frame.C_pix = precompute_pixel_projection(frame["rhat"], nside)

        # Hyperparameters for the TFR linewidth modelling
        if "eta_min" in config or "eta_max" in config:
            if config["add_eta_selection"]:
                frame.add_eta_truncation = True
                assert len(frame["e_eta"]) == len(frame)
            else:
                frame.add_eta_truncation = False
                fprint(f"disabling eta truncation for `{name}`.")

        if "eta_min" in config:
            frame.eta_min = config["eta_min"]
            if np.any(frame["eta"] < frame.eta_min):
                raise ValueError(
                    f"eta_min = {frame.eta_min} is smaller than the minimum "
                    f"eta value of {np.min(frame['eta'])}.")
        else:
            frame.eta_min = None

        if "eta_max" in config:
            frame.eta_max = config["eta_max"]
            if np.any(frame["eta"] > frame.eta_max):
                raise ValueError(
                    f"eta_max = {frame.eta_max} is larger than the maximum "
                    f"eta value of {np.max(frame['eta'])}.")
        else:
            frame.eta_max = None

        frame.with_lane_covmat = name == "PantheonPlusLane"
        frame.name = name

        if (
                config_pv_model.get("which_distance_prior", "empirical")
                == "empirical"
                and reconstruction_name is not None):
            if reconstruction_kwargs is None:
                raise ValueError(
                    "The volume-normalized empirical distance prior requires "
                    "a precomputed reconstruction; set "
                    "`pv_model.kind = precomputed_los_<X>` and provide "
                    "`io.reconstruction_main.<X>` paths.")
            if not frame.has_precomputed_los:
                raise ValueError(
                    "The volume-normalized empirical distance prior requires "
                    "precomputed LOS data.")
            downsample = int(config_pv_model.get(
                "density_3d_downsample", 1))
            voxel_subsample_fraction = _validate_voxel_subsample_fraction(
                config_pv_model.get("density_3d_subsample_fraction", 1.0),
                "pv_model.density_3d_subsample_fraction")
            voxel_subsample_seed = _validate_voxel_subsample_seed(
                config_pv_model.get("density_3d_subsample_seed", 42),
                "pv_model.density_3d_subsample_seed")
            batch_size = int(config_pv_model.get(
                "density_3d_normalizer_batch_size", 1))
            if batch_size < 1:
                raise ValueError(
                    "`density_3d_normalizer_batch_size` must be positive, "
                    f"got {batch_size}.")
            galaxy_bias = config_pv_model.get("galaxy_bias", "unity")
            geometry = config_pv_model.get("density_3d_geometry", "cube")
            radius = config_pv_model.get("density_3d_radius", None)
            store_rhat_3d = bool(config_pv_model.get("use_Mmiss", False))
            if geometry not in ("cube", "sphere"):
                raise ValueError(
                    "`pv_model.density_3d_geometry` must be 'cube' or "
                    f"'sphere', got {geometry!r}.")
            if geometry == "sphere" and radius is None:
                raise ValueError(
                    "`pv_model.density_3d_radius` is required when "
                    "`density_3d_geometry = 'sphere'`.")
            fprint(
                f"loading {frame.num_fields} volume density cube(s) via "
                f"{reconstruction_name} "
                f"loader (downsample={downsample}, "
                f"voxel_subsample_fraction={voxel_subsample_fraction:g}, "
                f"geometry={geometry}, radius={radius} Mpc/h, "
                f"normalizer_batch_size={batch_size}, "
                f"density_mode="
                f"{_volume_density_mode(galaxy_bias)}).")
            field_indices = np.asarray(
                data.get("los_field_indices", np.arange(frame.num_fields)),
                dtype=np.int32)
            if len(field_indices) != frame.num_fields:
                raise ValueError(
                    "Number of LOS field indices does not match field "
                    f"realisations: {len(field_indices)} != "
                    f"{frame.num_fields}.")

            fields_3d = _load_volume_density_3d_fields(
                reconstruction_name, reconstruction_kwargs, field_indices,
                downsample=downsample,
                subcube_radius=radius,
                pad_subcube_boundary=(geometry == "sphere"),
                cache_dir=field_cache_dir,
                cache_enabled=field_cache_enabled,
                geometry=geometry,
                radius=radius,
                store_rhat_3d=store_rhat_3d,
                voxel_subsample_fraction=voxel_subsample_fraction,
                voxel_subsample_seed=voxel_subsample_seed)

            frame.attach_prepared_volume_density_3d_fields(
                fields_3d, batch_size=batch_size,
                galaxy_bias=galaxy_bias,
                geometry=geometry, radius=radius)

        return frame

    def subsample(self, nsamples, los_radial_decay_scale, seed=42):
        """
        Returns a new frame with randomly selected `nsamples`. Keeps all
        calibrators in the sample (if present), and updates associated
        calibration fields accordingly.
        """
        fprint(f"subsampling from {len(self)} to {nsamples} galaxies.")

        gen = np.random.default_rng(seed)
        ndata = len(self)

        if nsamples > ndata:
            raise ValueError(f"`n_samples = {nsamples}` must be less than the "
                             f"number of data points of {ndata}.")

        main_mask = np.zeros(ndata, dtype=bool)
        if self.num_calibrators > 0:
            main_mask[self.data["is_calibrator"]] = True

        indx_choice = np.where(~main_mask)[0]
        indx_choice = gen.choice(
            indx_choice, nsamples - int(self.num_calibrators), replace=False)
        main_mask[indx_choice] = True

        keys_skip = [
            "is_calibrator", "mu_cal", "C_mu_cal", "std_mu_cal", "los_r",
            "mag_covmat",
            "los_density", "los_delta", "los_velocity", "los_log_density",
            "r_grid", "los_delta_r_grid", "los_velocity_r_grid",
            "los_log_density_r_grid", "log_r_grid", "log_jac_los",
            "los_field_indices", "density_3d_fields", "log_r_3d",
            "log_volume_weight_3d", "rhat_x_3d", "rhat_y_3d",
            "rhat_z_3d"]

        subsampled = {key: self[key][main_mask]
                      for key in self.keys() if key not in keys_skip}

        for key in keys_skip:
            if key in self.data:
                if key == "los_field_indices":
                    subsampled[key] = self.data[key]
                elif key.startswith("los_") and key != "los_r":
                    subsampled[key] = self[key][:, main_mask, ...]
                elif key == "is_calibrator":
                    subsampled[key] = self[key][main_mask]
                elif key == "mag_covmat":
                    subsampled[key] = self.data[key][main_mask][:, main_mask]
                else:
                    subsampled[key] = self.data[key]

        out = PVDataFrame(subsampled, los_radial_decay_scale)
        out.sample_dust = getattr(self, "sample_dust", False)
        out.name = self.name
        if self.has_volume_density_3d:
            out.has_volume_density_3d = True
            out.log_dV_3d = self.log_dV_3d
            out.density_3d_mode = self.density_3d_mode
            out.volume_density_batch_size = self.volume_density_batch_size
            out.density_3d_geometry = getattr(
                self, "density_3d_geometry", "cube")
            out.density_3d_radius = getattr(
                self, "density_3d_radius", None)
            out.coordinate_frame_3d = getattr(
                self, "coordinate_frame_3d", "icrs")
        return out

    def __getitem__(self, key):
        if key in self._cache:
            return jnp.asarray(self._cache[key])

        stat_funcs = {
            "mean": np.mean,
            "std": np.std,
            "min": np.min,
            "max": np.max
            }

        if key.startswith("e2_") and key.replace("e2_", "e_") in self.data:
            val = self.data[key.replace("e2_", "e_")]**2
        elif key == "theta":
            val = 0.5 * np.pi - np.deg2rad(self.data["dec"])
        elif key == "phi":
            val = np.deg2rad(self.data["RA"])
        elif key == "C_pix":
            val = self.C_pix
        elif key == "czcmb":
            val = self.data["zcmb"] * SPEED_OF_LIGHT
        elif key == "rhat":
            val = radec_to_cartesian(self.data["RA"], self.data["dec"])
            val /= np.linalg.norm(val, axis=1)[:, None]
        elif "_" in key:
            stat, field = key.split("_", 1)
            if stat in stat_funcs and field in self.data:
                val = stat_funcs[stat](self.data[field])
            else:
                return self.data[key]  # Fallback
        else:
            return self.data[key]

        # If val is a tracer (or contains one), skip caching.
        is_tracer = isinstance(val, jcore.Tracer)
        if not is_tracer:
            try:
                val_np = np.asarray(val)
                self._cache[key] = val_np
                return jnp.asarray(val_np)
            except Exception:
                # Conversion failed (likely because it's a tracer inside
                # a pytree)
                pass

        # Traced value path: do NOT mutate cache; just return it.
        return val

    def keys(self):
        return list(self.data.keys()) + list(self._cache.keys())

    @property
    def num_calibrators(self):
        if "mu_cal" in self.data:
            num_cal = jnp.sum(self.data["is_calibrator"])
        else:
            num_cal = 0

        return num_cal

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __repr__(self):
        n = len(self)
        num_cal = self.num_calibrators

        if num_cal > 0:
            return f"<PVDataFrame: {n} galaxies | {num_cal} calibrators>"
        else:
            return f"<PVDataFrame: {n} galaxies>"
