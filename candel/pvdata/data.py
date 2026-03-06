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
from os.path import isabs, join

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from h5py import File
from jax import core as jcore
from jax import numpy as jnp
from jax.nn import one_hot
from scipy.linalg import cholesky

from ..model.interp import LOSInterpolator
from ..util import (SPEED_OF_LIGHT, fprint, fsection, get_nested,
                    heliocentric_to_cmb, load_config, radec_to_cartesian,
                    radec_to_galactic)
from .dust import read_dustmap

###############################################################################
#                            Helper functions                                 #
###############################################################################


def _zcmb_blat_mask(zcmb, RA, dec, zcmb_min=None, zcmb_max=None, b_min=None):
    """Build a boolean mask for redshift and galactic latitude cuts."""
    mask = np.ones(len(zcmb), dtype=bool)
    if zcmb_min is not None:
        mask &= zcmb > zcmb_min
    if zcmb_max is not None:
        mask &= zcmb < zcmb_max
    if b_min is not None:
        b = radec_to_galactic(RA, dec)[1]
        mask &= np.abs(b) > b_min
    return mask


def _filter_data(data, mask, los_data_path=None):
    """Apply boolean mask to data arrays, report counts, and load LOS."""
    n_total = len(mask)
    n_kept = int(np.sum(mask))
    fprint(f"removed {n_total - n_kept} objects, thus {n_kept} remain.")
    for k in data:
        if isinstance(data[k], np.ndarray):
            data[k] = data[k][mask]
    if los_data_path:
        data = load_los(los_data_path, data, mask=mask)
    return data


def _compute_r_grid(r_limits, dr, data, Om=0.3):
    """Compute the radial grid for Malmquist bias integration."""
    if isinstance(r_limits, str) and r_limits.startswith("auto"):
        if "_" in r_limits:
            h_auto = float(r_limits.split("_")[1])
        else:
            h_auto = 1.0

        from ..cosmography import Redshift2Distance

        if "czcmb" in data:
            cz_obs = data["czcmb"]
        elif "zcmb" in data:
            cz_obs = data["zcmb"] * SPEED_OF_LIGHT
        else:
            raise KeyError("Data must contain 'czcmb' or 'zcmb'.")

        cz_obs_lim = [float(np.min(cz_obs)), float(np.max(cz_obs))]
        cz_obs_lim[0] = max(cz_obs_lim[0], 50.0)
        redshift2distance = Redshift2Distance(Om0=Om)
        r_from_cz = redshift2distance(
            np.array(cz_obs_lim), h=h_auto, is_velocity=True)
        r_min_raw = float(r_from_cz[0])
        r_max_raw = float(r_from_cz[1])
        buffer_low = max(r_min_raw * 0.25, 15.0)
        buffer_high = max(r_max_raw * 0.25, 15.0)
        rmin = max(r_min_raw - buffer_low, 0.01)
        rmax = r_max_raw + buffer_high
        fprint(f"auto r_limits_malmquist (h={h_auto}): [{rmin:.1f}, "
               f"{rmax:.1f}] Mpc "
               f"(buffer: -{buffer_low:.1f}, +{buffer_high:.1f} Mpc)")
    else:
        rmin, rmax = r_limits
        fprint(f"setting the LOS radial grid from {rmin} to {rmax} Mpc.")

    num_points = int(round((rmax - rmin) / dr)) + 1
    # Simpson's rule requires an odd number of points.
    if num_points % 2 == 0:
        num_points += 1
    dr_eff = (rmax - rmin) / (num_points - 1) if num_points > 1 else 0.0
    fprint(f"r-grid: n_r={num_points}, dr={dr_eff:.2f} Mpc")

    return np.linspace(rmin, rmax, num_points)


def effective_rank_entropy(C):
    """
    Compute the entropy-based effective rank (Shannon effective rank) of C.

    https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf
    """
    w = np.linalg.eigvalsh(C)
    # Remove negative eigenvalues (numerical artefacts)
    w = w[w > 0]
    p = w / np.sum(w)
    p_nonzero = p[p > 0]
    return np.exp(-np.sum(p_nonzero * np.log(p_nonzero)))


def precompute_pixel_projection(rhat_data, nside, sigma_deg=None):
    """
    Precompute the pixel projection matrix for a given set of LOS vectors.
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    rhat_pix = np.stack([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)], axis=1)

    d = rhat_data @ rhat_pix.T  # radial projection factors

    if sigma_deg is None:
        # One-hot on nearest pixel (max dot == min angle)
        p_max = np.argmax(d, axis=1)
        w = one_hot(p_max, rhat_pix.shape[0], dtype=rhat_data.dtype)
    else:
        raise NotImplementedError("Gaussian smoothing is not implemented")

    return w * d


###############################################################################
#                             Data frames                                     #
###############################################################################


def load_PV_dataframes(config_path):
    """Loads PV dataframes from the given configuration file."""
    config = load_config(config_path)

    if config["pv_model"]["kind"].startswith("precomputed_los_"):
        los_reconstruction = config["pv_model"]["kind"].replace("precomputed_los_", "")  # noqa
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
    for name in names:
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

        df = PVDataFrame.from_config_dict(
            kwargs, name, try_pop_los=try_pop_los,
            config_pv_model=config_pv_model)
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

        self.has_calibrators = bool(self.num_calibrators > 0)
        self._cache = {}

    @classmethod
    def from_config_dict(cls, config, name, try_pop_los, config_pv_model):
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
        Om = config.get("model", {}).get("Om", 0.3)
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
            "los_log_density_r_grid"]

        subsampled = {key: self[key][main_mask]
                      for key in self.keys() if key not in keys_skip}

        for key in keys_skip:
            if key in self.data:
                if key.startswith("los_") and key != "los_r":
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


###############################################################################
#                            Specific loaders                                 #
###############################################################################


def load_los(los_data_path, data, mask=None, verbose=True):
    with File(los_data_path, 'r') as f:
        if mask is None:
            data["los_density"] = f['los_density'][...]
            data["los_velocity"] = f['los_velocity'][...]
            data["los_r"] = f['r'][...]
            data["los_RA"] = f["RA"][...]
            data["los_dec"] = f["dec"][...]
        else:
            data["los_density"] = f['los_density'][...][:, mask, ...]
            data["los_velocity"] = f['los_velocity'][...][:, mask, ...]
            data["los_r"] = f['r'][...]
            data["los_RA"] = f["RA"][...][mask]
            data["los_dec"] = f["dec"][...][mask]

        assert np.all(data["los_density"] > 0)
        assert np.all(np.isfinite(data["los_velocity"]))

        if "manticore" in los_data_path.lower():
            fprint("normalizing the Manticore LOS density (Om = 0.306)",
                   verbose=verbose)
            data["los_density"] /= 0.306 * 275.4  # Manticore normalization
        elif "_CB1" in los_data_path:
            data["los_density"] /= 0.307 * 275.4
            fprint("normalizing the CB1 LOS density (Om = 0.307)",
                   verbose=verbose)

            if len(data["los_density"]) == 100:
                fprint("downsampling the CB1 LOS density from 100 to 20",
                       verbose=verbose)
                data["los_density"] = data["los_density"][::5]
                data["los_velocity"] = data["los_velocity"][::5]
        elif "_CB2" in los_data_path:
            fprint("normalizing the CB2 LOS density (Om = 0.3111)",
                   verbose=verbose)
            data["los_density"] /= 0.3111 * 275.4
        elif "HAMLET_V1" in los_data_path:
            fprint("normalizing the HAMLET_V1 LOS density (Om = 0.3)",
                   verbose=verbose)
            data["los_density"] /= 0.3 * 275.4
        elif "_CF4.hdf5" in los_data_path and len(data["los_density"]) == 100:
            fprint("downsampling the CF4 LOS density from 100 to 20",
                   verbose=verbose)
            data["los_density"] = data["los_density"][::5]
            data["los_velocity"] = data["los_velocity"][::5]

    return data


def load_SH0ES_calibration(calibration_path, pgc_CF4):
    """
    Load SH0ES distance modulus samples and match to CF4 galaxies by PGC ID.
    """
    with File(calibration_path, 'r') as f:
        mu_samples = f["distmod_samples"][...]
        pgc_SH0ES = f["pgc"][...]

    i_CF4 = []
    i_SH0ES = []

    for i, pgc_i in enumerate(pgc_CF4):
        if pgc_i in pgc_SH0ES:
            match = np.where(pgc_SH0ES == pgc_i)[0]
            assert len(match) == 1
            i_CF4.append(i)
            i_SH0ES.append(match[0])

    i_CF4 = np.array(i_CF4)
    i_SH0ES = np.array(i_SH0ES)

    is_calibrator = np.zeros(len(pgc_CF4), dtype=bool)
    is_calibrator[i_CF4] = True

    mu_cal = np.mean(mu_samples[:, i_SH0ES], axis=0)
    C_mu_cal = np.cov(mu_samples[:, i_SH0ES], rowvar=False)

    return is_calibrator, mu_cal, C_mu_cal


def load_CF4_data(root, which_band, best_mag_quality=True, eta_min=-0.3,
                  zcmb_min=None, zcmb_max=None, b_min=7.5,
                  remove_outliers=True, calibration=None, los_data_path=None,
                  return_all=False, dust_model=None, exclude_W1=False,
                  **kwargs):
    """
    Load CF4 TFR data and apply optional filters and dust correction removal.
    """
    with File(join(root, "CF4_TFR.hdf5"), 'r') as f:
        grp = f["cf4"]
        zcmb = grp["Vcmb"][...] / SPEED_OF_LIGHT
        RA = grp["RA"][...] * 15  # deg
        DEC = grp["DE"][...]
        mag = grp[which_band][...]
        mag_quality = grp["Qw"][...] if which_band == "w1" else grp["Qs"][...]
        eta = grp["lgWmxi"][...] - 2.5
        e_eta = grp["elgWi"][...]
        pgc = grp["pgc"][...]

        if dust_model is not None:
            if which_band not in ["w1", "w2"]:
                raise ValueError(
                    f"Band `{which_band}` is not supported for dust "
                    f"correction removal. Only `w1` and `w2` are supported.")

            Ab_default = grp[f"A_{which_band}"][...]
            fprint(f"switching the dust model to `{dust_model}`.")

            mag += Ab_default
            if dust_model == "default":
                ebv = Ab_default / (0.186 if which_band == "w1" else 0.123)
            else:
                ebv = read_dustmap(RA, DEC, dust_model)

            if not np.all(np.isfinite(ebv)):
                raise ValueError(
                    f"Non-finite E(B-V) values for dust map `{dust_model}`.")
        else:
            ebv = np.full_like(mag, np.nan)

    fprint(f"initially loaded {len(pgc)} galaxies from CF4 TFR data.")

    data = dict(
        zcmb=zcmb,
        RA=RA,
        dec=DEC,
        mag=mag,
        e_mag=np.full_like(mag, 0.05),
        eta=eta,
        e_eta=e_eta,
        ebv=ebv,
    )

    if return_all:
        return data

    mask = eta > eta_min
    if best_mag_quality:
        mask &= mag_quality == 5
    else:
        mask &= mag > 5

    mask &= _zcmb_blat_mask(zcmb, RA, DEC, zcmb_min, zcmb_max, b_min)

    if remove_outliers:
        outliers = np.concatenate([
            np.genfromtxt(join(root, f"CF4_{b}_outliers.csv"),
                          delimiter=",", names=True)
            for b in ("W1", "i")
        ])
        mask &= ~np.isin(pgc, outliers["PGC"])

    if which_band == "i" and exclude_W1:
        with File(join(root, "CF4_TFR.hdf5"), 'r') as f:
            w1_quality = f["cf4"]["Qw"][...]
            w1_mag = f["cf4"]["w1"][...]
        fprint("excluding galaxies with W1 quality 5 or W1 mag < 5.")
        exclude = (w1_quality == 5) | (w1_mag > 5)
        mask &= ~exclude

    _filter_data(data, mask, los_data_path)
    pgc = pgc[mask]

    if calibration == "SH0ES":
        is_cal, mu, C_mu = load_SH0ES_calibration(
            join(root, "CF4_SH0ES_calibration.hdf5"), pgc)
        fprint(f"out of {len(pgc)} galaxies, {np.sum(is_cal)} are SH0ES "
               "calibrators.")
        data.update({
            "is_calibrator": is_cal,
            "mu_cal": mu,
            "C_mu_cal": C_mu,
            "std_mu_cal": np.sqrt(np.diag(C_mu)),
        })
    elif calibration:
        raise ValueError("Unknown calibration type.")

    return data


def load_CF4_mock(root, index):
    fname = join(root, f"mock_{index}.hdf5")
    with File(fname, 'r') as f:
        grp = f["mock"]
        data = {key: grp[key][...] for key in grp.keys()}
    return data


def load_2MTF(root, eta_min=-0.1, eta_max=0.2, zcmb_min=None, zcmb_max=None,
              b_min=7.5, los_data_path=None, return_all=False, **kwargs):
    """
    Load the 2MTF data from the given root directory.
    """
    with File(join(root, "PV_compilation.hdf5"), 'r') as f:
        grp = f["2MTF"]

        zcmb = grp["z_CMB"][...]
        RA = grp["RA"][...]
        DEC = grp["DEC"][...]
        mag = grp["mag"][...]
        eta = grp["eta"][...]

        e_eta = grp["e_eta"][...]
        e_mag = grp["e_mag"][...]

    fprint(f"initially loaded {len(zcmb)} galaxies from 2MTF data.")

    data = dict(
        zcmb=zcmb,
        RA=RA,
        dec=DEC,
        mag=mag,
        e_mag=e_mag,
        eta=eta,
        e_eta=e_eta,
    )

    if return_all:
        return data

    mask = (eta > eta_min) & (eta < eta_max)
    mask &= _zcmb_blat_mask(zcmb, RA, DEC, zcmb_min, zcmb_max, b_min)
    return _filter_data(data, mask, los_data_path)


def load_SFI(root, eta_min=-0.1, zcmb_min=None, zcmb_max=None,
             b_min=7.5, los_data_path=None, return_all=False, **kwargs):
    """
    Load the SFI++ data from the given root directory.
    """
    with File(join(root, "PV_compilation.hdf5"), 'r') as f:
        grp = f["SFI_gals"]

        zcmb = grp["z_CMB"][...]
        RA = grp["RA"][...]
        DEC = grp["DEC"][...]
        mag = grp["mag"][...]
        eta = grp["eta"][...]

        e_eta = grp["e_eta"][...]
        e_mag = grp["e_mag"][...]

    fprint(f"initially loaded {len(zcmb)} galaxies from SFI++ data.")

    data = dict(
        zcmb=zcmb,
        RA=RA,
        dec=DEC,
        mag=mag,
        e_mag=e_mag,
        eta=eta,
        e_eta=e_eta,
    )

    if return_all:
        return data

    mask = eta > eta_min
    mask &= _zcmb_blat_mask(zcmb, RA, DEC, zcmb_min, zcmb_max, b_min)
    return _filter_data(data, mask, los_data_path)


def _load_LOSS_Foundation(which, root, zcmb_min=None, zcmb_max=None,
                          b_min=7.5, los_data_path=None, return_all=False,
                          **kwargs):
    """
    Load the LOSS or Foundation SNe data from the given root directory.
    """
    with File(join(root, "PV_compilation.hdf5"), 'r') as f:
        grp = f[which]

        zcmb = grp["z_CMB"][...]
        RA = grp["RA"][...]
        DEC = grp["DEC"][...]
        mag = grp["mB"][...]
        c = grp["c"][...]
        x1 = grp["x1"][...]

        e_mag = grp["e_mB"][...]
        e_c = grp["e_c"][...]
        e_x1 = grp["e_x1"][...]

    fprint(f"initially loaded {len(zcmb)} galaxies from LOSS/Foundation data.")

    data = dict(
        zcmb=zcmb,
        RA=RA,
        dec=DEC,
        mag=mag,
        c=c,
        x1=x1,
        e_mag=e_mag,
        e_c=e_c,
        e_x1=e_x1
    )

    if return_all:
        return data

    mask = _zcmb_blat_mask(zcmb, RA, DEC, zcmb_min, zcmb_max, b_min)
    return _filter_data(data, mask, los_data_path)


def load_LOSS(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
              los_data_path=None, return_all=False, **kwargs):
    return _load_LOSS_Foundation(
        "LOSS", root, zcmb_min=zcmb_min, zcmb_max=zcmb_max,
        b_min=b_min, los_data_path=los_data_path, return_all=return_all,
        **kwargs)


def load_Foundation(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
                    los_data_path=None, return_all=False, **kwargs):
    return _load_LOSS_Foundation(
        "Foundation", root, zcmb_min=zcmb_min, zcmb_max=zcmb_max,
        b_min=b_min, los_data_path=los_data_path, return_all=return_all,
        **kwargs)


def load_PantheonPlus_Lane(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
                           los_data_path=None, return_all=False, **kwargs):
    if zcmb_max is not None and zcmb_max > 0.075:
        raise ValueError(f"`zcmb_max` of {zcmb_max} is too high for the "
                         "LOWZ sample which goes only up to 0.075.")
    fname = join(root, "full_ps1_input_LOWZ.csv")
    x = np.genfromtxt(fname, delimiter=",", names=True, dtype=None,
                      encoding=None)

    fprint(f"initially loaded {len(x)} galaxies from Pantheon+Lane data.")

    data = dict(
        zcmb=x["zCMB"],
        RA=x["RA"],
        dec=x["DEC"],
        mag=x["mB"],
        x1=x["x1"],
        c=x["c"],
    )

    if return_all:
        return data

    C = np.loadtxt(join(root, "PP_cov_new_LOWZ.txt"))

    mask = _zcmb_blat_mask(
        data["zcmb"], data["RA"], data["dec"], zcmb_min, zcmb_max, b_min)
    _filter_data(data, mask)

    C_idx = (3 * np.where(mask)[0][:, None] + np.arange(3)).ravel()
    data["mag_covmat"] = C[C_idx][:, C_idx]

    if los_data_path is not None:
        data = load_los(los_data_path, data, mask=mask)

    return data


def load_PantheonPlus(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
                      los_data_path=None, return_all=False,
                      removed_PV_from_covmat=True, **kwargs):
    """
    Load the Pantheon+ data from the given root directory, the covariance
    is expected to have peculiar velocity contribution removed.
    """
    if removed_PV_from_covmat:
        arr_fname = "Pantheon+SH0ES_zsel.dat"
        covmat_fname = "Pantheon+SH0ES_zsel_STAT+SYS_noPV.cov"
    else:
        arr_fname = "Pantheon+SH0ES.dat"
        covmat_fname = "Pantheon+SH0ES_STAT+SYS.cov"

    arr = np.genfromtxt(
        join(root, arr_fname), names=True, dtype=None, encoding=None)

    fprint(f"initially loaded {len(arr)} galaxies from Pantheon+ data.")

    data = {
        "zcmb": arr["zCMB"],
        "e_zcmb": arr["zCMBERR"],
        "RA": arr["RA"],
        "dec": arr["DEC"],
        "mag": arr["m_b_corr"],
    }

    if return_all:
        return data

    covmat = np.loadtxt(join(root, covmat_fname), delimiter=",")
    size = int(covmat[0])
    C = np.reshape(covmat[1:], (size, size))

    mask = _zcmb_blat_mask(
        data["zcmb"], data["RA"], data["dec"], zcmb_min, zcmb_max, b_min)
    _filter_data(data, mask)

    C = C[mask][:, mask]
    data["mag_covmat"] = C
    data["e_mag"] = np.sqrt(np.diag(C))  # Do not use in the inference!

    if los_data_path is not None:
        data = load_los(los_data_path, data, mask=mask)

    return data


def load_SH0ES(root):
    """
    Load the SH0ES data which can be used to sample distances.

    NOTE: Set the zero-width prior to a delta prior so it is not sampled.
    """
    lstsq_results_path = join(root, 'lstsq_results.txt')
    Y_fits_path = join(root, 'ally_shoes_ceph_topantheonwt6.0_112221.fits')
    L_fits_path = join(root, 'alll_shoes_ceph_topantheonwt6.0_112221.fits')
    C_fits_path = join(root, 'allc_shoes_ceph_topantheonwt6.0_112221.fits')

    Y = fits.open(Y_fits_path)[0].data
    L = fits.open(L_fits_path)[0].data
    C = fits.open(C_fits_path)[0].data

    from scipy import linalg

    C_inv_cho = linalg.cho_solve(linalg.cho_factor(C), np.identity(C.shape[0]))
    q_lstsq, sigma_lstsq = np.loadtxt(lstsq_results_path, unpack=True)
    mu_list = q_lstsq
    width_list = sigma_lstsq * 10

    ks = np.where(width_list == 0)[0]
    if len(ks) > 0:
        fprint("warning: zero width found in the priors. Setting it to 1e-5.")
        fprint(f"indices of zero width: {ks}")

    if len(ks) != 1:
        raise ValueError("At most one zero width is allowed.")

    k = ks[0]
    fprint(f"found zero-width prior at index {k}. Setting it to 0.")
    width_list[k] = 1e-5
    fixed_idx = k
    fixed_value = 0.

    mu_list = jnp.asarray(mu_list)
    width_list = jnp.asarray(width_list)
    theta_min, theta_max = mu_list - width_list / 2, mu_list + width_list / 2

    data = {
        "Y": Y,
        "L": L,
        "C_inv_cho": C_inv_cho,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "fixed_idx": fixed_idx,
        "fixed_value": fixed_value,
        "C": C,
        }

    for key in data:
        if not key.startswith("fixed_"):
            data[key] = jnp.asarray(data[key], dtype=jnp.float32)

    return data


def load_SH0ES_separated(root, cepheid_host_cz_cmb_max=None,
                         los_data_path=None, rand_los_data_path=None):
    """
    Load the separated SH0ES data, separating the Cepheid and supernovae and
    covariance matrices.

    Structure of the covariance matrix indices:
    ------------------------------------------
    - Indices < 2150: Cepheid hosts without geometric anchors.
    - Index 2150: Start of NGC 4258 Cepheid hosts.
    - Index 2593: Start of M31 Cepheid hosts.
    - Index 2648: Start of LMC Cepheid hosts.

    - Index 3207: Uncertainty on HST zeropoint (sigma_HST).
    - Index 3208: Uncertainty on Gaia zeropoint (sigma_Gaia).
    - Index 3209: Prior on metallicity coefficient Z_W.
    - Index 3210: Unused term (likely placeholder).
    - Index 3211: Ground-based photometry systematic uncertainty (sigma_grnd).
    - Index 3212: Prior on P–L relation slope b_W.
    - Index 3213: Constraint on NGC 4258 anchor offset (delta_mu_N4258).
    - Index 3214: Constraint on LMC anchor offset (delta_mu_LMC).
    """

    # Unpack the SH0ES data.
    data = load_SH0ES(root)
    Y = np.array(data['Y'], copy=True)
    C = np.array(data['C'], copy=True)
    L = np.array(data['L'].T, copy=True)

    # Cepheid data and covariance matrix.
    OH = L[:, -4][:3130]
    logP = L[:, -6][:3130]
    mag_cepheid = Y[:3130]
    C_Cepheid = C[:3130, :3130]

    # Undo the removal of a slope of -3.285
    mag_cepheid += -3.285 * logP

    # This will organise the host distances as
    # `[Host with Cepheids but no geometric anchors, NGC4258, LMC, M31]`. There
    # are 37 of the former, so in total there are 40 distances to be inferred.
    L_dist = np.hstack([L[:, :37], L[:, [37, 39, 40]]])
    L_Cepheid_host_dist = L_dist[:3130]

    # N4258 and LMC anchors.
    mu_N4258_anchor = 29.398
    e_mu_N4258_anchor = 0.032

    mu_LMC_anchor = 18.477
    e_mu_LMC_anchor = 0.0263

    # Undo the anchor offsets.
    mag_cepheid[2150:2593] += mu_N4258_anchor
    mag_cepheid[2648:] += mu_LMC_anchor

    C_SN_Cepheid = C[3130:3207, 3130:3207]
    Y_SN_Cepheid = Y[3130:3207]

    # HST and Gaia zero-points
    M_HST = Y[3207]
    e_M_HST = C[3207, 3207]**0.5

    M_Gaia = Y[3208]
    e_M_Gaia = C[3208, 3208]**0.5

    # Systematic uncertainties btw ground-based and HST photometry.
    sigma_grnd = C[3211, 3211]**0.5

    q_names = np.asanyarray(
        ['mu_M101', 'mu_M1337', 'mu_N0691', 'mu_N1015', 'mu_N0105',
         'mu_N1309', 'mu_N1365', 'mu_N1448', 'mu_N1559', 'mu_N2442',
         'mu_N2525', 'mu_N2608', 'mu_N3021', 'mu_N3147', 'mu_N3254',
         'mu_N3370', 'mu_N3447', 'mu_N3583', 'mu_N3972', 'mu_N3982',
         'mu_N4038', 'mu_N4424', 'mu_N4536', 'mu_N4639', 'mu_N4680',
         'mu_N5468', 'mu_N5584', 'mu_N5643', 'mu_N5728', 'mu_N5861',
         'mu_N5917', 'mu_N7250', 'mu_N7329', 'mu_N7541', 'mu_N7678',
         'mu_N0976', 'mu_U9391', 'Delta_mu_N4258', 'M_H1_W',
         'Delta_mu_LMC', 'mu_M31', 'b_W', 'MB0', 'Z_W', 'undefined',
         'Delta_zp', 'log10_H0'])

    L_SN_Cepheid_dist = L_dist[3130:3207]

    num_hosts = L_Cepheid_host_dist.shape[1] - 3
    num_cepheids = len(mag_cepheid)
    host_names = q_names[np.char.startswith(
        q_names.astype(str), "mu_")]
    host_names = host_names[:num_hosts]

    # Cepheid host redshifts and the PV covariance matrix.
    data_cepheid_host_redshift = np.load(
        join(root, "processed", "Cepheid_anchors_redshifts.npy"),
        allow_pickle=True)
    PV_covmat_cepheid_host = np.load(
        join(root, "processed", "PV_covmat_cepheid_hosts_fiducial.npy"),
        allow_pickle=True)

    def _load_los_or_none(path, keys):
        if path is None:
            return {k: None for k in keys}
        d = load_los(path, {})
        return {k: d[k] for k in keys}

    host_los_keys = ["los_density", "los_velocity", "los_r"]
    host_los = _load_los_or_none(los_data_path, host_los_keys)

    rand_los_keys = host_los_keys + ["los_RA", "los_dec"]
    rand_los = _load_los_or_none(rand_los_data_path, rand_los_keys)

    # Keep the brightest (lowest magnitude) SN per Cepheid host galaxy
    n_hosts = L_SN_Cepheid_dist.shape[1]
    best_mag = np.full(n_hosts, np.inf)
    best_idx = np.full(n_hosts, -1, dtype=int)

    for i, y in enumerate(Y_SN_Cepheid):
        # Assuming one-hot host assignment per SN
        j = np.where(L_SN_Cepheid_dist[i] == 1)[0][0]
        if y < best_mag[j]:    # use '>' if working in flux (higher = brighter)
            best_mag[j] = y
            best_idx[j] = i

    valid = best_idx >= 0
    unique_ks = best_idx[valid]

    mag_SN_unique_Cepheid_host = Y_SN_Cepheid[unique_ks]
    C_SN_unique_Cepheid_host = C_SN_Cepheid[np.ix_(unique_ks, unique_ks)]
    L_SN_unique_Cepheid_host_dist = L_SN_Cepheid_dist[unique_ks]

    data = {
        # Individual Cepheid data, covariance matrix and host association.
        "mag_cepheid": mag_cepheid,
        "logP": logP,
        "OH": OH,
        "C_Cepheid": C_Cepheid,
        "L_Cepheid": cholesky(C_Cepheid, lower=True),
        "L_Cepheid_host_dist": L_Cepheid_host_dist,
        "Cepheids_only": False,
        "num_cepheids": num_cepheids,
        "num_hosts": num_hosts,
        # Unique SNe in Cepheid host galaxies.
        "mag_SN_unique_Cepheid_host": mag_SN_unique_Cepheid_host,
        "C_SN_unique_Cepheid_host": C_SN_unique_Cepheid_host,
        "mean_std_mag_SN_unique_Cepheid_host": np.mean(np.sqrt(np.diag(C_SN_unique_Cepheid_host))),  # noqa
        "L_SN_unique_Cepheid_host": cholesky(C_SN_unique_Cepheid_host,
                                             lower=True),
        "L_SN_unique_Cepheid_host_dist": L_SN_unique_Cepheid_host_dist,
        # External constraints/priors.
        "mu_N4258_anchor": mu_N4258_anchor,
        "e_mu_N4258_anchor": e_mu_N4258_anchor,
        "mu_LMC_anchor": mu_LMC_anchor,
        "e_mu_LMC_anchor": e_mu_LMC_anchor,
        "M_HST": M_HST,
        "e_M_HST": e_M_HST,
        "M_Gaia": M_Gaia,
        "e_M_Gaia": e_M_Gaia,
        "sigma_grnd": sigma_grnd,
        # Cepheid host galaxy information.
        "q_names": q_names,
        "host_names": host_names,
        "czcmb_cepheid_host": data_cepheid_host_redshift["zCMB"] * SPEED_OF_LIGHT,  # noqa
        "e_czcmb_cepheid_host": data_cepheid_host_redshift["zCMBERR"],
        "RA_host": data_cepheid_host_redshift["RA"],
        "dec_host": data_cepheid_host_redshift["DEC"],
        "PV_covmat_cepheid_host": PV_covmat_cepheid_host,
        "host_los_density": host_los["los_density"],
        "host_los_velocity": host_los["los_velocity"],
        "host_los_r": host_los["los_r"],
        # Random LOS for modelling selection
        "has_rand_los": rand_los_data_path is not None,
        "num_rand_los": rand_los["los_density"].shape[1] if rand_los["los_density"] is not None else 1,  # noqa
        "rand_los_density": rand_los["los_density"],
        "rand_los_velocity": rand_los["los_velocity"],
        "rand_los_r": rand_los["los_r"],
        "rand_los_RA": rand_los["los_RA"],
        "rand_los_dec": rand_los["los_dec"]
        }

    if cepheid_host_cz_cmb_max is not None:
        if cepheid_host_cz_cmb_max < 1000:
            raise ValueError(
                f"`cz_cmb_max` must be larger than 1000 km/s, got "
                f"{cepheid_host_cz_cmb_max} km/s. Otherwise could eliminate "
                "some geometric anchors.")

        # Switch this flag so that these runs cannot be done jointly with SNe
        # since some shapes might not be correct.
        data["Cepheids_only"] = True

        cz_host = data["czcmb_cepheid_host"]
        cz_host_all = np.hstack([data["czcmb_cepheid_host"], [667, 327, -582]])
        cz_cepheid = data["L_Cepheid_host_dist"] @ cz_host_all
        cz_unique_SN_Cepheid_host = data["L_SN_unique_Cepheid_host_dist"] @ cz_host_all  # noqa

        mask_host = cz_host < cepheid_host_cz_cmb_max
        mask_host_all = cz_host_all < cepheid_host_cz_cmb_max
        mask_cepheid = cz_cepheid < cepheid_host_cz_cmb_max
        mask_cz_unique_SN_Cepheid_host = (
            cz_unique_SN_Cepheid_host < cepheid_host_cz_cmb_max)

        fprint(f"Masking Cepheids with cz_cmb > {cepheid_host_cz_cmb_max} "
               f"km/s: Keeping {np.sum(mask_host)} out of {len(mask_host)}.")

        data["OH"] = data["OH"][mask_cepheid]
        data["logP"] = data["logP"][mask_cepheid]
        data["mag_cepheid"] = data["mag_cepheid"][mask_cepheid]
        data["C_Cepheid"] = data["C_Cepheid"][mask_cepheid][:, mask_cepheid]
        data["L_Cepheid"] = cholesky(data["C_Cepheid"], lower=True)

        data["L_Cepheid_host_dist"] = data["L_Cepheid_host_dist"][mask_cepheid][:, mask_host_all]  # noqa
        data["czcmb_cepheid_host"] = data["czcmb_cepheid_host"][mask_host]
        data["e_czcmb_cepheid_host"] = data["e_czcmb_cepheid_host"][mask_host]
        data["RA_host"] = data["RA_host"][mask_host]
        data["dec_host"] = data["dec_host"][mask_host]
        data["PV_covmat_cepheid_host"] = data["PV_covmat_cepheid_host"][mask_host][:, mask_host]  # noqa

        data["L_SN_unique_Cepheid_host_dist"] = data["L_SN_unique_Cepheid_host_dist"][mask_cz_unique_SN_Cepheid_host][:, mask_host_all]  # noqa
        data["mag_SN_unique_Cepheid_host"] = data["mag_SN_unique_Cepheid_host"][mask_cz_unique_SN_Cepheid_host]  # noqa
        data["C_SN_unique_Cepheid_host"] = data["C_SN_unique_Cepheid_host"][mask_cz_unique_SN_Cepheid_host][:, mask_cz_unique_SN_Cepheid_host]  # noqa
        data["L_SN_unique_Cepheid_host"] = cholesky(data["C_SN_unique_Cepheid_host"], lower=True)  # noqa

        data["num_hosts"] = np.sum(mask_host)
        data["num_cepheids"] = np.sum(mask_cepheid)
        data["host_names"] = data["host_names"][mask_host]

        data["mask_host"] = mask_host

    data["Neff_C_SN_unique_Cepheid_host"] = effective_rank_entropy(data["C_SN_unique_Cepheid_host"]) # noqa
    data["Neff_PV_covmat_cepheid_host"] = effective_rank_entropy(data["PV_covmat_cepheid_host"])     # noqa
    data["Neff_C_Cepheid"] = effective_rank_entropy(data["C_Cepheid"])

    return data


def load_SH0ES_from_config(config_path):
    config = load_config(config_path, replace_los_prior=False)
    if not get_nested(config, "model/use_reconstruction", False):
        config["io"]["load_host_los"] = False
        config["io"]["load_rand_los"] = False
    d = config["io"]["SH0ES"]
    root = d["root"]
    cepheid_host_cz_cmb_max = d.get("cepheid_host_cz_cmb_max", None)
    which_host_los = d.get("which_host_los", None)
    if which_host_los is not None:
        if config["io"]["load_host_los"]:
            los_data_path = config["io"]["PV_main"]["SH0ES"]["los_file"].replace(  # noqa
                "<X>", which_host_los)
        else:
            los_data_path = None

        if config["io"]["load_rand_los"]:
            rand_los_data_path = config["io"]["los_file_random"].replace(
                "<X>", which_host_los)
        else:
            rand_los_data_path = None

    else:
        los_data_path = None
        rand_los_data_path = None

    return load_SH0ES_separated(
        root, cepheid_host_cz_cmb_max,
        los_data_path=los_data_path, rand_los_data_path=rand_los_data_path)


def load_CCHP_from_config(config_path, ra_dec_only=False):
    """
    Load the processed CCHP TRGB catalogue from a TSV file.

    Expects the TSV to contain at least the columns:
    SN, Galaxy, cz_cmb, e_czcmb, mu_TRGB_CCHP, sigma_TRGB_CCHP,
    m_Bprime_CSP, sigma_Bprime_CSP.

    Set io.CSP.load_CSP_matches = true in config to load matched CSP data
    (st, BV, obs_vec, cov, RA, dec, stellar masses) with CSP_ prefix.
    """
    config = load_config(config_path, replace_los_prior=False)
    if not get_nested(config, "model/use_reconstruction", False):
        config["io"]["load_host_los"] = False
        config["io"]["load_rand_los"] = False
    load_csp_match = get_nested(config, "io/CSP/load_CSP_matches", False)
    path = config["io"]["CCHP"]["path"]
    redshift_source = get_nested(
        config, "io/CCHP_redshift_source/kind", "cz_cmb")
    if not isabs(path):
        path = join(config["root_main"], path)

    data_tbl = np.genfromtxt(
        path,
        delimiter="\t",
        names=True,
        dtype=None,
        encoding="utf-8",
        missing_values=["-1", "nan", "NaN"],
        filling_values=np.nan,
    )

    # Check here about the wavelength!
    mag_trgb = data_tbl["mu_TRGB_CCHP"] - 4.049

    # Fixed anchor values (LMC and NGC 4258) for convenience.
    # LMC (Pietrzynski et al. 2019): https://arxiv.org/abs/1903.08096
    mu_LMC_anchor = 18.477
    e_mu_LMC_anchor = 0.026
    # Hoyt+2021 TRGB calibration: https://arxiv.org/abs/2106.13337
    mag_LMC_TRGB = 14.456
    e_mag_LMC_TRGB = 0.018

    # NGC 4258 distance (Reid et al. 2019)
    mu_N4258_anchor = 29.398
    e_mu_N4258_anchor = 0.032
    # Jang & Lee 2020 TRGB calibration: https://arxiv.org/abs/2008.04181
    # This is at F814W
    mag_N4258_TRGB = 25.347
    e_mag_N4258_TRGB = 0.0443

    ra = data_tbl["ra_deg"]
    dec = data_tbl["dec_deg"]

    source = redshift_source.lower()
    fprint(f"Using CCHP redshift source: {source}", verbose=True)
    if source == "cz_cmb":
        cz_cmb = data_tbl["cz_cmb"]
    elif source == "cz_cmb_ned":
        cz_cmb = data_tbl["cz_cmb_NED"]
    else:
        raise ValueError(
            "Unknown `io/CCHP_redshift_source/kind`: "
            f"{redshift_source}. Use 'cz_cmb' or 'cz_cmb_NED'.")

    e_czcmb = data_tbl["e_czcmb"]

    if ra_dec_only:
        return {
            "RA": ra,
            "DEC": dec,
            "cz_cmb": cz_cmb,
            "e_czcmb": e_czcmb,
        }

    data = {
        "SN": data_tbl["SN"],
        "Galaxy": data_tbl["Galaxy"],
        "cz_cmb": cz_cmb,
        "e_czcmb": e_czcmb,
        "mag_TRGB": mag_trgb,
        "e_mag_TRGB": data_tbl["sigma_TRGB_CCHP"],
        "m_Bprime": data_tbl["m_Bprime_CSP"],
        "sigma_Bprime": data_tbl["sigma_Bprime_CSP"],
        "RA": ra,
        "DEC": dec,
        "mu_LMC_anchor": mu_LMC_anchor,
        "e_mu_LMC_anchor": e_mu_LMC_anchor,
        "mag_LMC_TRGB": mag_LMC_TRGB,
        "e_mag_LMC_TRGB": e_mag_LMC_TRGB,
        "mu_N4258_anchor": mu_N4258_anchor,
        "e_mu_N4258_anchor": e_mu_N4258_anchor,
        "mag_N4258_TRGB": mag_N4258_TRGB,
        "e_mag_N4258_TRGB": e_mag_N4258_TRGB,
    }

    # Optionally load LOS data (host and/or random) if requested in config.
    los_data_path = None
    rand_los_data_path = None

    which_host_los = get_nested(config, "io/which_host_los", None)
    if get_nested(config, "io/load_host_los", False):
        los_file = get_nested(config, "io/los_file", None)
        if los_file is not None and which_host_los is not None:
            los_data_path = los_file.replace("<X>", which_host_los)
        else:
            los_data_path = los_file

    if get_nested(config, "io/load_rand_los", False):
        rand_file = get_nested(config, "io/los_file_random", None)
        if rand_file is not None and which_host_los is not None:
            rand_los_data_path = rand_file.replace("<X>", which_host_los)
        else:
            rand_los_data_path = rand_file

    if los_data_path is not None:
        host_los = load_los(los_data_path, {}, mask=None)
        data["host_los_density"] = host_los["los_density"]
        data["host_los_velocity"] = host_los["los_velocity"]
        data["host_los_r"] = host_los["los_r"]

    if rand_los_data_path is not None:
        rand_los = load_los(rand_los_data_path, {}, mask=None, verbose=False)
        data["rand_los_density"] = rand_los["los_density"]
        data["rand_los_velocity"] = rand_los["los_velocity"]
        data["rand_los_r"] = rand_los["los_r"]
        data["rand_los_RA"] = rand_los.get("los_RA", None)
        data["rand_los_dec"] = rand_los.get("los_dec", None)
        data["has_rand_los"] = True
        data["num_rand_los"] = data["rand_los_density"].shape[1]
    else:
        data["has_rand_los"] = False

    # Optionally load matched CSP data
    if load_csp_match:
        csp_root = get_nested(config, "io/CSP/root", None)
        if csp_root is None:
            raise ValueError("CSP root not specified in config [io.CSP.root]")
        if not isabs(csp_root):
            csp_root = join(config["root_main"], csp_root)

        csp_data = load_CSP(csp_root, return_all=True)
        cchp_idx, csp_idx = match_cchp_to_csp(data, csp_data)

        # Store matching indices
        data["CSP_match_cchp_idx"] = cchp_idx
        data["CSP_match_csp_idx"] = csp_idx

        # Add matched CSP fields with prefix
        n_cchp = len(data["SN"])
        csp_fields = ["st", "BV", "obs_vec", "cov", "RA", "dec",
                      "log_stellar_mass", "log_stellar_mass_lower",
                      "log_stellar_mass_upper"]
        for field in csp_fields:
            arr = csp_data[field]
            # Create array of NaN with CCHP shape, fill matched entries
            if arr.ndim == 1:
                out = np.full(n_cchp, np.nan)
            elif arr.ndim == 2:
                out = np.full((n_cchp, arr.shape[1]), np.nan)
            else:
                out = np.full((n_cchp,) + arr.shape[1:], np.nan)
            out[cchp_idx] = arr[csp_idx]
            data[f"CSP_{field}"] = out

    return data


def match_cchp_to_csp(cchp_data, csp_data):
    """
    Match CCHP TRGB hosts to CSP SNe by SN name.

    Handles naming convention differences: CCHP uses '2011fe' while CSP uses
    'SN2011fe'.

    Returns
    -------
    cchp_idx : ndarray
        Indices into cchp_data for matched SNe.
    csp_idx : ndarray
        Indices into csp_data for matched SNe.
    """
    cchp_names = cchp_data["SN"]
    csp_names = csp_data["sn"]

    # CSP names have "SN" prefix, strip it for matching
    csp_name_to_idx = {}
    for i, name in enumerate(csp_names):
        key = name[2:] if name.startswith("SN") else name
        csp_name_to_idx[key] = i

    cchp_idx, csp_idx = [], []
    for i, name in enumerate(cchp_names):
        if name in csp_name_to_idx:
            cchp_idx.append(i)
            csp_idx.append(csp_name_to_idx[name])

    cchp_idx = np.array(cchp_idx, dtype=int)
    csp_idx = np.array(csp_idx, dtype=int)

    fprint(f"matched {len(cchp_idx)}/{len(cchp_names)} CCHP SNe to CSP.")

    # Print unmatched CCHP SNe
    matched_set = set(cchp_idx)
    unmatched = [cchp_names[i] for i in range(len(cchp_names))
                 if i not in matched_set]
    if unmatched:
        fprint(f"unmatched CCHP SNe: {unmatched}")

    return cchp_idx, csp_idx


def load_CSP_from_config(config_path):
    """
    Load CSP SNe data from config, wrapped in PVDataFrame for inference.

    Uses config keys:
    - io.CSP.root: path to CSP data directory
    - io.CSP.which_sample: sample to select ("CSPI", "CSPII", or "LSQ")
    - model.r_limits_malmquist: radial grid limits for Malmquist bias
    - model.num_points_malmquist: number of grid points
    """
    config = load_config(config_path, replace_los_prior=False)

    csp_root = get_nested(config, "io/CSP/root", None)
    if csp_root is None:
        raise ValueError("CSP root not specified in config [io.CSP.root]")
    if not isabs(csp_root):
        csp_root = join(config["root_main"], csp_root)

    # Get optional CSP loading parameters
    which_sample = get_nested(config, "io/CSP/which_sample", None)
    sample_str = which_sample if which_sample else "all"
    fprint(f"loading CSP sample: {sample_str}")

    data = load_CSP(csp_root, which_sample=which_sample)

    # Add radial grid for selection integral
    r_limits = config["model"]["r_limits_malmquist"]
    num_points = config["model"]["num_points_malmquist"]
    Om = get_nested(config, "model/Om", 0.3)
    data["r_grid"] = _compute_r_grid(r_limits, num_points, data, Om)

    fprint(f"loaded {len(data['sn'])} CSP SNe (sample: {sample_str}).")
    # Return raw dict; JointTRGBCSPModel wraps in PVDataFrame after matching
    return data


###############################################################################
#                                 SDSS FP                                     #
###############################################################################

def arcsec_to_radian(arcsec):
    return (arcsec * u.arcsec).to(u.radian).value


def load_SDSS_FP(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
                 los_data_path=None, return_all=False, **kwargs):
    """Load the SDSS FP data from the given root directory."""
    fname = join(root, "SDSS_PV_public.dat")
    d_input = np.genfromtxt(fname, names=True, )

    rdev = d_input["deVRad_r"]
    e_rdev = d_input["deVRadErr_r"]
    boa = d_input["deVAB_r"]
    e_boa = d_input["deVABErr_r"]

    fprint(f"initially loaded {len(d_input)} galaxies from SDSS FP data.")

    theta_eff = arcsec_to_radian(rdev * np.sqrt(boa))
    e_theta_eff = theta_eff * np.sqrt(
        (e_rdev / rdev)**2 + (0.5 * e_boa / boa)**2)

    data = {
        "RA": d_input["RA"],
        "dec": d_input["Dec"],
        "zcmb": d_input["zcmb_group"],
        "theta_eff": theta_eff,
        "e_theta_eff": e_theta_eff,
        "log_theta_eff": np.log10(theta_eff),
        "e_log_theta_eff": e_theta_eff / (theta_eff * np.log(10)),
        "logI": d_input["i"],
        "e_logI": d_input["ei"],
        "logs": d_input["s"],
        "e_logs": d_input["es"],
        }

    if return_all:
        return data

    mask = _zcmb_blat_mask(
        data["zcmb"], data["RA"], data["dec"], zcmb_min, zcmb_max, b_min)
    return _filter_data(data, mask, los_data_path)


def load_6dF_FP(root, which_band=None, zcmb_min=None, zcmb_max=None, b_min=7.5,
                los_data_path=None, return_all=False, **kwargs):
    """Load the 6dF FP data from the given root directory."""
    d = np.genfromtxt(join(root, "6dF_FP.dat"))

    RA = d[:, 2] * 360 / 24
    dec = d[:, 3]
    czcmb = d[:, 4]

    data = {
        "RA": RA,
        "dec": dec,
        "zcmb": czcmb / SPEED_OF_LIGHT,
    }

    fprint(f"initially loaded {len(d)} galaxies from 6dF FP data.")

    if return_all:
        return data
    elif which_band is None:
        raise ValueError("which_band must be one of 'J', 'H', 'K'.")

    cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
    dA_zcmb = cosmo.angular_diameter_distance(czcmb / SPEED_OF_LIGHT).value  # noqa

    if which_band == "J":
        logRe = d[:, 5]
        e_logRe = d[:, 6]

        logIe = d[:, 11]
        e_logIe = d[:, 12]
    elif which_band == "H":
        logRe = d[:, 7]
        e_logRe = d[:, 8]

        logIe = d[:, 13]
        e_logIe = d[:, 14]
    elif which_band == "K":
        logRe = d[:, 9]
        e_logRe = d[:, 10]

        logIe = d[:, 15]
        e_logIe = d[:, 16]
    else:
        raise ValueError(f"which_band must be one of 'J', 'H', 'K', got "
                         f"{which_band}.")
    logVd = d[:, 17]
    e_logVd = d[:, 18]

    log_theta_eff = logRe - np.log10(dA_zcmb * 1e3)
    e_log_theta_eff = e_logRe

    data.update({
        "logI": logIe,
        "e_logI": e_logIe,
        "logs": logVd,
        "e_logs": e_logVd,
        "log_theta_eff": log_theta_eff,
        "e_log_theta_eff": e_log_theta_eff,
    })

    mask = _zcmb_blat_mask(
        data["zcmb"], data["RA"], data["dec"], zcmb_min, zcmb_max, b_min)
    return _filter_data(data, mask, los_data_path)


def load_generic(filepath, los_data_path=None, **kwargs):
    """
    Load generic catalog data from a .txt file with column names.

    Expected columns: RA, dec, Vcmb (in CMB frame).
    """
    d = np.genfromtxt(filepath, names=True)

    data = {
        "RA": d["RA"],
        "dec": d["dec"],
        "zcmb": d["Vcmb"] / SPEED_OF_LIGHT,
    }

    fprint(f"loaded {len(data['RA'])} galaxies from {filepath}.")

    if los_data_path is not None:
        data = load_los(los_data_path, data,)

    return data


def load_CSP(root, zcmb_min=None, zcmb_max=None, b_min=None, quality_min=None,
             st_min=None, st_max=None, t0_min=None, t0_max=None,
             phys_only=False, exclude_phys=True, which_sample=None,
             los_data_path=None, return_all=False, remove_duplicates=True,
             **kwargs):
    """
    Load CSP (Carnegie Supernova Project) SNe Ia data.

    Merges photometry from B_all_noj21.csv with coordinates from
    cspallcal_sncoords.csv, csp_sncoords.csv, and missing_coords_simbad.csv.

    Parameters
    ----------
    which_sample : str, optional
        Sample to select: "CSPI", "CSPII", or "LSQ".
    exclude_phys : bool
        If True, exclude physics sample (phys=0 only).
    """
    # Load main photometry file
    fname = join(root, "B_all_noj21.csv")
    d = np.genfromtxt(fname, names=True, dtype=None, encoding="utf-8")
    fprint(f"initially loaded {len(d)} SNe from CSP data.")

    # Remove duplicate SNe (same name, different calibration type)
    if remove_duplicates and not return_all:
        _, unique_idx = np.unique(d["sn"], return_index=True)
        unique_idx = np.sort(unique_idx)
        n_duplicates = len(d) - len(unique_idx)
        fprint(f"removed {n_duplicates} duplicate SNe (keeping first).")
        d = d[unique_idx]

    # Load coordinates from multiple sources
    coords_dict = {}
    coord_files = [
        "cspallcal_sncoords.csv",
        "csp_sncoords.csv",
        "missing_coords_simbad.csv",
    ]
    for fname in coord_files:
        fpath = join(root, fname)
        try:
            coords = np.genfromtxt(fpath, names=True, delimiter=",",
                                   dtype=None, encoding="utf-8")
            for row in coords:
                sn = row["sn"]
                if sn not in coords_dict:
                    ra, dec = row["snra"], row["sndec"]
                    if np.isfinite(ra) and np.isfinite(dec):
                        coords_dict[sn] = (ra, dec)
        except FileNotFoundError:
            pass

    # Match coordinates to main catalog
    RA = np.full(len(d), np.nan)
    dec_ = np.full(len(d), np.nan)

    for i, sn in enumerate(d["sn"]):
        if sn in coords_dict:
            RA[i], dec_[i] = coords_dict[sn]

    n_with_coords = np.sum(np.isfinite(RA))
    fprint(f"matched {n_with_coords}/{len(d)} SNe with coordinates.")

    # Build 3x3 covariance matrix for (peak_mag_B, st, BV)
    n = len(d)
    cov = np.zeros((n, 3, 3))
    cov[:, 0, 0] = d["eMmax"]**2       # var(peak_mag_B)
    cov[:, 1, 1] = d["est"]**2         # var(st)
    cov[:, 2, 2] = d["eBV"]**2         # var(BV)
    cov[:, 0, 1] = cov[:, 1, 0] = d["covMs"]      # cov(peak_mag_B, st)
    cov[:, 0, 2] = cov[:, 2, 0] = d["covBV_M"]    # cov(peak_mag_B, BV)
    # cov(st, BV) = 0 by assumption

    # Fix non-positive definite matrices by adding minimal diagonal
    for i in range(n):
        min_eig = np.linalg.eigvalsh(cov[i]).min()
        if min_eig <= 0:
            cov[i] += np.eye(3) * (abs(min_eig) + 1e-10)
            fprint(f"regularized non-PD covariance for {d['sn'][i]} "
                   f"(zcmb={d['zcmb'][i]:.4f}).")

    # Observation vector: (n_sn, 3) for (peak_mag_B, st, BV)
    obs_vec = np.stack([d["Mmax"], d["st"], d["BV"]], axis=-1)

    # Compute median measurement errors and correlations for selection integral
    sigma_m = np.sqrt(cov[:, 0, 0])
    sigma_s = np.sqrt(cov[:, 1, 1])
    sigma_BV = np.sqrt(cov[:, 2, 2])

    # Correlations from covariance
    rho_ms = cov[:, 0, 1] / (sigma_m * sigma_s)
    rho_mBV = cov[:, 0, 2] / (sigma_m * sigma_BV)
    rho_sBV = cov[:, 1, 2] / (sigma_s * sigma_BV)

    # Convert quality from string to float, empty strings become NaN
    quality = np.array([
        float(q) if q not in ('', '""') else np.nan for q in d["quality"]])

    # Redshift in km/s and default error (100 km/s)
    czcmb = d["zcmb"] * SPEED_OF_LIGHT
    e_czcmb = np.full(len(d), 100.0)

    data = {
        "sn": d["sn"],
        "zcmb": d["zcmb"],
        "czcmb": czcmb,
        "e_czcmb": e_czcmb,
        "zhel": d["zhel"],
        "peak_mag_B": d["Mmax"],
        "st": d["st"],
        "BV": d["BV"],
        "obs_vec": obs_vec,
        "cov": cov,
        "t0": d["t0"],
        "quality": quality,
        "phys": d["phys"],
        "sample": d["sample"],
        "RA": RA,
        "dec": dec_,
        "log_stellar_mass": d["m"],
        "log_stellar_mass_lower": d["ml"],
        "log_stellar_mass_upper": d["mu"],
        # Median values for selection integral
        "median_sigma_m": np.median(sigma_m),
        "median_sigma_s": np.median(sigma_s),
        "median_sigma_BV": np.median(sigma_BV),
        "median_rho_ms": np.median(rho_ms),
        "median_rho_mBV": np.median(rho_mBV),
        "median_rho_sBV": np.median(rho_sBV),
    }

    if return_all:
        return data

    # Filter out SNe without valid coordinates
    has_coords = np.isfinite(RA) & np.isfinite(dec_)
    n_no_coords = np.sum(~has_coords)
    if n_no_coords > 0:
        fprint(f"removing {n_no_coords} SNe without valid coordinates.")

    mask = has_coords
    mask &= _zcmb_blat_mask(
        data["zcmb"], data["RA"], data["dec"], zcmb_min, zcmb_max, b_min)

    if quality_min is not None:
        mask &= data["quality"] >= quality_min
    if phys_only:
        mask &= data["phys"] == "1"
    if exclude_phys:
        mask &= data["phys"] == "0"
    if st_min is not None:
        mask &= data["st"] >= st_min
    if st_max is not None:
        mask &= data["st"] <= st_max
    if t0_min is not None:
        mask &= data["t0"] >= t0_min
    if t0_max is not None:
        mask &= data["t0"] <= t0_max
    if which_sample is not None:
        fprint(f"selecting CSP sample: {which_sample}")
        if which_sample == "LSQ":
            mask &= np.char.startswith(data["sn"], "LSQ")
        elif which_sample in ("CSPI", "CSPII"):
            mask &= data["sample"] == which_sample
        else:
            raise ValueError(f"Unknown sample: {which_sample}. "
                             "Must be 'CSPI', 'CSPII', or 'LSQ'.")

    return _filter_data(data, mask, los_data_path)


def load_EDD_TRGB(root, zcmb_min=None, zcmb_max=None, b_min=None,
                  los_data_path=None, return_all=False,
                  return_mask=False, **kwargs):
    """Load EDD TRGB data from the pre-parsed CSV."""
    x = np.genfromtxt(join(root, "EDD_TRGB.csv"), delimiter=",",
                      names=True, dtype=None, encoding=None)
    fprint(f"initially loaded {len(x)} galaxies from EDD TRGB data.")

    RA = x["RA"].astype(np.float64)
    dec = x["dec"].astype(np.float64)

    z_helio = x["v"].astype(np.float64) / SPEED_OF_LIGHT
    e_z_helio = x["e_v"].astype(np.float64) / SPEED_OF_LIGHT
    zcmb, e_zcmb = heliocentric_to_cmb(z_helio, RA, dec, e_z_helio)

    data = dict(
        RA=RA,
        dec=dec,
        zcmb=zcmb,
        e_zcmb=e_zcmb,
        mag=x["T814"].astype(np.float64) - x["A_814"].astype(np.float64),
        e_mag=(x["T8_hi"].astype(np.float64)
               - x["T8_lo"].astype(np.float64)) / 2,
    )

    if return_all:
        return data

    # Build a combined mask over the original catalogue indices.
    n_orig = len(x)
    keep = np.ones(n_orig, dtype=bool)

    # Drop anchor and satellite galaxies (treated separately in the model).
    names = np.array([n.strip() for n in x["name"]])
    drop = np.isin(names, ["LMC", "SMC", "NGC4258", "NGC4258-DF6"])
    if np.any(drop):
        fprint(f"dropping {np.sum(drop)} anchor/satellite galaxies: "
               f"{', '.join(names[drop])}")
    keep &= ~drop

    # Drop galaxies with missing TRGB magnitudes.
    valid_mag = np.isfinite(data["mag"])
    n_bad = np.sum(keep & ~valid_mag)
    if n_bad > 0:
        fprint(f"dropping {n_bad} galaxies with missing TRGB magnitudes.")
    keep &= valid_mag

    # Apply zcmb / galactic latitude cuts on the kept subset.
    sub_mask = _zcmb_blat_mask(
        zcmb[keep], RA[keep], dec[keep], zcmb_min, zcmb_max, b_min)
    # Expand sub_mask back to the full catalogue.
    idx_kept = np.where(keep)[0]
    keep[idx_kept[~sub_mask]] = False

    # Filter data arrays.
    for k in data:
        if isinstance(data[k], np.ndarray):
            data[k] = data[k][keep]
    n_kept = int(np.sum(keep))
    fprint(f"removed {n_orig - n_kept} objects, thus {n_kept} remain.")

    if los_data_path:
        data = load_los(los_data_path, data, mask=keep)

    if return_mask:
        return data, keep
    return data


def load_EDD_TRGB_from_config(config_path):
    """Load EDD TRGB data with LOS and anchor calibration from config."""
    config = load_config(config_path, replace_los_prior=False)
    if not get_nested(config, "model/use_reconstruction", False):
        config["io"]["load_host_los"] = False
        config["io"]["load_rand_los"] = False
    d = config["io"]["PV_main"]["EDD_TRGB"]
    root = d["root"]

    zcmb_min = get_nested(config, "io/PV_main/EDD_TRGB/zcmb_min", None)
    zcmb_max = get_nested(config, "io/PV_main/EDD_TRGB/zcmb_max", None)
    b_min = get_nested(config, "io/PV_main/EDD_TRGB/b_min", None)

    data, mask = load_EDD_TRGB(root, zcmb_min=zcmb_min, zcmb_max=zcmb_max,
                                b_min=b_min, return_mask=True)

    # Rename to match model expectations
    data["RA_host"] = data.pop("RA")
    data["dec_host"] = data.pop("dec")
    data["mag_obs"] = data.pop("mag")
    data["e_mag_obs"] = data.pop("e_mag")
    data["czcmb"] = data.pop("zcmb") * SPEED_OF_LIGHT
    data["e_czcmb"] = data.pop("e_zcmb") * SPEED_OF_LIGHT

    # Median mag error for selection function smoothing
    data["e_mag_median"] = float(np.median(data["e_mag_obs"]))

    # LOS data
    which_host_los = get_nested(config, "io/PV_main/EDD_TRGB/which_host_los",
                                None)
    los_data_path = None
    rand_los_data_path = None

    if get_nested(config, "io/load_host_los", False):
        los_file = d.get("los_file", None)
        if los_file is not None and which_host_los is not None:
            los_data_path = los_file.replace("<X>", which_host_los)
        else:
            los_data_path = los_file

    if get_nested(config, "io/load_rand_los", False):
        rand_file = get_nested(config, "io/los_file_random", None)
        if rand_file is not None and which_host_los is not None:
            rand_los_data_path = rand_file.replace("<X>", which_host_los)
        else:
            rand_los_data_path = rand_file

    if los_data_path is not None:
        host_los = load_los(los_data_path, {}, mask=mask)
        data["host_los_density"] = host_los["los_density"]
        data["host_los_velocity"] = host_los["los_velocity"]
        data["host_los_r"] = host_los["los_r"]

    if rand_los_data_path is not None:
        rand_los = load_los(rand_los_data_path, {}, mask=None, verbose=False)
        data["rand_los_density"] = rand_los["los_density"]
        data["rand_los_velocity"] = rand_los["los_velocity"]
        data["rand_los_r"] = rand_los["los_r"]
        data["rand_los_RA"] = rand_los.get("los_RA", None)
        data["rand_los_dec"] = rand_los.get("los_dec", None)
        data["has_rand_los"] = True
        data["num_rand_los"] = data["rand_los_density"].shape[1]
    else:
        data["has_rand_los"] = False

    # Anchor calibration from config
    anchors = get_nested(config, "model/anchors", {})
    data["mu_LMC_anchor"] = anchors.get("mu_LMC", 18.477)
    data["e_mu_LMC_anchor"] = anchors.get("e_mu_LMC", 0.026)
    data["mag_LMC_TRGB"] = anchors.get("mag_LMC_TRGB", 14.456)
    data["e_mag_LMC_TRGB"] = anchors.get("e_mag_LMC_TRGB", 0.018)
    data["mu_N4258_anchor"] = anchors.get("mu_N4258", 29.398)
    data["e_mu_N4258_anchor"] = anchors.get("e_mu_N4258", 0.032)
    data["mag_N4258_TRGB"] = anchors.get("mag_N4258_TRGB", 25.347)
    data["e_mag_N4258_TRGB"] = anchors.get("e_mag_N4258_TRGB", 0.0443)

    return data


###############################################################################
#                            EDD 2MTF (K-band TFR)                            #
###############################################################################


def load_EDD_2MTF(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
                  eta_min=None, eta_max=None,
                  los_data_path=None, return_all=False,
                  return_mask=False, **kwargs):
    """Load 2MTF data from the EDD text file.

    The file format is pipe-delimited with 5 header lines.
    Returns K-band apparent magnitudes (Ktc) and linewidths
    (eta = log10(Wc) - 2.5).
    """
    fpath = join(root, "EDD_2MTF.txt")
    lines = open(fpath).readlines()
    header = lines[1].strip().split("|")
    rows = []
    for line in lines[5:]:
        rows.append(line.strip().split("|"))

    def col(name):
        return np.array([float(r[header.index(name)]) for r in rows])

    RA = col("RA")
    dec = col("Dec")
    Ktc = col("Ktc")
    eKtc = col("eKtc")
    Wc = col("Wc")
    eWc = col("eWc")
    czcmb = col("czcmb")

    eta = np.log10(Wc) - 2.5
    e_eta = eWc / (Wc * np.log(10))
    zcmb = czcmb / SPEED_OF_LIGHT

    fprint(f"initially loaded {len(RA)} galaxies from EDD 2MTF data.")

    data = dict(
        RA=RA,
        dec=dec,
        zcmb=zcmb,
        mag=Ktc,
        e_mag=eKtc,
        eta=eta,
        e_eta=e_eta,
    )

    if return_all:
        return data

    mask = np.ones(len(RA), dtype=bool)
    mask &= _zcmb_blat_mask(zcmb, RA, dec, zcmb_min, zcmb_max, b_min)
    if eta_min is not None:
        mask &= eta > eta_min
    if eta_max is not None:
        mask &= eta < eta_max

    for k in data:
        if isinstance(data[k], np.ndarray):
            data[k] = data[k][mask]

    n_kept = int(np.sum(mask))
    fprint(f"removed {len(RA) - n_kept} objects, thus {n_kept} remain.")

    if los_data_path:
        data = load_los(los_data_path, data, mask=mask)

    if return_mask:
        return data, mask
    return data


def load_EDD_2MTF_from_config(config_path):
    """Load EDD 2MTF data with LOS from config."""
    config = load_config(config_path, replace_los_prior=False)
    if not get_nested(config, "model/use_reconstruction", False):
        config["io"]["load_host_los"] = False
        config["io"]["load_rand_los"] = False
    d = config["io"]["PV_main"]["EDD_2MTF"]
    root = d["root"]

    zcmb_min = d.get("zcmb_min", None)
    zcmb_max = d.get("zcmb_max", None)
    b_min = d.get("b_min", 7.5)
    eta_min = d.get("eta_min", None)
    eta_max = d.get("eta_max", None)

    data, mask = load_EDD_2MTF(
        root, zcmb_min=zcmb_min, zcmb_max=zcmb_max, b_min=b_min,
        eta_min=eta_min, eta_max=eta_max,
        return_mask=True)

    # Rename to match model expectations
    data["RA_host"] = data.pop("RA")
    data["dec_host"] = data.pop("dec")
    data["czcmb"] = data.pop("zcmb") * SPEED_OF_LIGHT
    data["e_czcmb"] = np.full(len(data["czcmb"]), 10.0)  # ~10 km/s

    # Median errors for selection function
    data["e_mag_median"] = float(np.median(data["e_mag"]))
    data["e_eta_median"] = float(np.median(data["e_eta"]))

    # LOS data
    which_host_los = d.get("which_host_los", None)
    los_data_path = None
    rand_los_data_path = None

    if get_nested(config, "io/load_host_los", False):
        los_file = d.get("los_file", None)
        if los_file is not None and which_host_los is not None:
            los_data_path = los_file.replace("<X>", which_host_los)
        else:
            los_data_path = los_file

    if get_nested(config, "io/load_rand_los", False):
        rand_file = get_nested(config, "io/los_file_random", None)
        if rand_file is not None and which_host_los is not None:
            rand_los_data_path = rand_file.replace("<X>", which_host_los)
        else:
            rand_los_data_path = rand_file

    if los_data_path is not None:
        host_los = load_los(los_data_path, {}, mask=mask)
        data["host_los_density"] = host_los["los_density"]
        data["host_los_velocity"] = host_los["los_velocity"]
        data["host_los_r"] = host_los["los_r"]

    if rand_los_data_path is not None:
        rand_los = load_los(rand_los_data_path, {}, mask=None, verbose=False)
        data["rand_los_density"] = rand_los["los_density"]
        data["rand_los_velocity"] = rand_los["los_velocity"]
        data["rand_los_r"] = rand_los["los_r"]
        data["rand_los_RA"] = rand_los.get("los_RA", None)
        data["rand_los_dec"] = rand_los.get("los_dec", None)
        data["has_rand_los"] = True
        data["num_rand_los"] = data["rand_los_density"].shape[1]
    else:
        data["has_rand_los"] = False

    return data


###############################################################################
#                          Catalogue registry                                 #
###############################################################################

_CATALOGUE_LOADERS = {
    "2MTF": load_2MTF,
    "SFI": load_SFI,
    "SDSS_FP": load_SDSS_FP,
    "6dF_FP": load_6dF_FP,
    "LOSS": load_LOSS,
    "Foundation": load_Foundation,
    "PantheonPlus": load_PantheonPlus,
    "PantheonPlusLane": load_PantheonPlus_Lane,
    "CSP": load_CSP,
    "EDD_TRGB": load_EDD_TRGB,
    "EDD_2MTF": load_EDD_2MTF,
}
