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

from os.path import join

import numpy as np
from h5py import File
from jax import numpy as jnp

from ..util import (SPEED_OF_LIGHT, fprint, load_config, radec_to_cartesian,
                    radec_to_galactic)
from ..model.interp import LOSInterpolator

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
    names = config_io.pop("catalogue_name")
    if isinstance(names, str):
        names = [names]

    dfs = []
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
            kwargs, name, try_pop_los=try_pop_los, config_io=config_io)
        dfs.append(df)

    if len(dfs) == 1:
        return dfs[0]

    return dfs


class PVDataFrame:
    """Lightweight container for PV data."""
    add_eta_truncation = False
    add_mag_selection = False
    mag_selection_kwargs = None

    def __init__(self, data, los_method="linear", los_extrap=True):
        self.data = {k: jnp.asarray(v) for k, v in data.items()}

        if "los_velocity" in self.data:
            self.has_precomputed_los = True
            kwargs = {"method": los_method, "extrap": los_extrap}
            self.f_los_log_density = LOSInterpolator(
                self.data["los_r"], jnp.log(self.data["los_density"]),
                **kwargs)
            self.f_los_velocity = LOSInterpolator(
                self.data["los_r"], self.data["los_velocity"], **kwargs)
        else:
            self.has_precomputed_los = False

        self._cache = {}

    @classmethod
    def from_config_dict(cls, config, name, try_pop_los, config_io):
        root = config.pop("root")
        nsamples_subsample = config.pop("nsamples_subsample", None)
        seed_subsample = config.pop("seed_subsample", 42)
        mag_selection = config.pop("mag_selection", None)

        if "CF4_mock" in name:
            index = name.split("_")[-1]
            data = load_CF4_mock(root, index)
        elif "CF4_" in name:
            data = load_CF4_data(root, **config)
        else:
            raise ValueError(f"Unknown catalogue name: {name}")

        if try_pop_los:
            for key in list(data.keys()):
                if key.startswith("los_"):
                    fprint(f"removing `{key}` from data.")
                    data.pop(key, None)

        if "los_r" not in data:
            d = config_io["reconstruction_main"]
            fprint(f"setting the LOS radial grid from {d['rmin']} to "
                   f"{d['rmax']} with {d['num_steps']} steps.")
            data["los_r"] = np.linspace(d["rmin"], d["rmax"], d["num_steps"])

        if "los_density" in data:
            data["los_log_density"] = np.log(data["los_density"])

        if nsamples_subsample is not None:
            frame = cls(data)
            frame = frame.subsample(nsamples_subsample, seed=seed_subsample)
        else:
            frame = cls(data)

        # Keyword arguments for the magnitude hyperprior.
        if "mag" in data:
            frame.mag_dist_kwargs = {
                "xmin": frame["min_mag"] - 0.5 * frame["std_mag"],
                "xmax": frame["max_mag"] + 0.5 * frame["std_mag"],
                "mag_sample": frame["mag"],
                "e_mag_sample": frame["e_mag"],
                }

        # Magnitude selection hyperparameters.
        if mag_selection is not None:
            if config["add_mag_selection"]:
                frame.mag_selection_kwargs = mag_selection
            else:
                frame.mag_selection_kwargs = None
                fprint(f"disabling magnitude selection for `{name}`.")
        frame.add_mag_selection = frame.mag_selection_kwargs is not None

        # Hyperparameters for the TFR linewidth selection.
        if "eta_min" in config or "eta_max" in config:
            if config["add_eta_selection"]:
                frame.add_eta_truncation = True
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

        return frame

    def subsample(self, nsamples, seed=42):
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
            indx_choice, nsamples - self.num_calibrators, replace=False)
        main_mask[indx_choice] = True

        keys_skip = [
            "is_calibrator", "mu_cal", "C_mu_cal", "std_mu_cal", "los_r"]

        subsampled = {key: self[key][main_mask]
                      for key in self.keys() if key not in keys_skip}

        for key in keys_skip:
            if key in self.data:
                if key == "is_calibrator":
                    subsampled[key] = self[key][main_mask]
                else:
                    subsampled[key] = self.data[key]

        return PVDataFrame(subsampled)

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]

        stat_funcs = {
            "mean": np.mean,
            "std": np.std,
            "min": np.min,
            "max": np.max
            }

        if key.startswith("e2_") and key.replace("e2_", "e_") in self.data:
            val = self.data[key.replace("e2_", "e_")]**2
        elif key == "theta":
            val = np.deg2rad(self.data["RA"])
        elif key == "phi":
            val = 0.5 * np.pi - np.deg2rad(self.data["dec"])
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

        self._cache[key] = jnp.asarray(val)
        return val

    def keys(self):
        return list(self.data.keys()) + list(self._cache.keys())

    @property
    def num_calibrators(self):
        if "mu_cal" in self.data:
            num_cal = np.sum(self.data["is_calibrator"])
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
                  zcmb_max=None, b_min=7.5, remove_outliers=True,
                  calibration=None, los_data_path=None, return_all=False,
                  **kwargs):
    """
    Loads the `CF4_TFR.hdf5` file from `root` and extracts fields based on
    `which_band`. Applies filters using `eta_min`, `zcmb_max`, and
    `best_mag_quality`. Returns a dictionary of cleaned and filtered data
    arrays.
    """
    with File(join(root, "CF4_TFR.hdf5"), 'r') as f:
        zcmb = f["Vcmb"][...] / SPEED_OF_LIGHT
        RA = f["RA"][...] * 360 / 24
        DEC = f["DE"][...]

        if which_band == "w1":
            mag = f["w1"][...]
            mag_quality = f["Qw"][...]
        elif which_band == "i":
            mag = f["i"][...]
            mag_quality = f["Qs"][...]
        else:
            raise ValueError("which_band must be 'w1' or 'i'.")

        eta = f["lgWmxi"][...] - 2.5
        e_eta = f["elgWi"][...]

        pgc = f["pgc"][...]

    fprint(f"initially loaded {len(pgc)} galaxies from CF4 TFR data.")

    data = {
        "zcmb": zcmb,
        "RA": RA,
        "dec": DEC,
        "mag": mag,
        "e_mag": np.ones_like(mag) * 0.05,
        "eta": eta,
        "e_eta": e_eta,
    }

    if return_all:
        return data

    mask = data["eta"] > eta_min
    if best_mag_quality:
        mask &= mag_quality == 5
    if zcmb_max is not None:
        mask &= data["zcmb"] < zcmb_max
    if remove_outliers:
        outliers = np.concatenate([
            np.genfromtxt(
                join(root, f"CF4_{band}_outliers.csv"),
                delimiter=",", names=True)
            for band in ("W1", "i")
            ])
        pgc_outliers = outliers["PGC"]
        mask &= ~np.isin(pgc, pgc_outliers)
    if b_min is not None:
        b = radec_to_galactic(RA, DEC)[1]
        mask &= np.abs(b) > b_min

    fprint(f"removed {len(pgc) - np.sum(mask)} galaxies, thus "
           f"{len(pgc[mask])} remain.")

    for key in data:
        data[key] = data[key][mask]
    pgc = pgc[mask]

    if los_data_path is not None:
        with File(los_data_path, 'r') as f:
            data["los_density"] = f['los_density'][...][mask, ...]
            data["los_velocity"] = f['los_velocity'][...][mask, ...]
            data["los_r"] = f['r'][...]

            assert np.all(data["los_density"] > 0)
            assert np.all(np.isfinite(data["los_velocity"]))

    if calibration == "SH0ES":
        is_calibrator, mu_cal, C_mu_cal = load_SH0ES_calibration(
            join(root, "CF4_SH0ES_calibration.hdf5"), pgc)

        fprint(f"out of {len(pgc)} galaxies, {np.sum(is_calibrator)} are "
               "SH0ES calibrators.")

        data = {**data,
                "is_calibrator": is_calibrator,
                "mu_cal": mu_cal,
                "C_mu_cal": C_mu_cal,
                "std_mu_cal": np.diag(C_mu_cal)**0.5,
                }
    elif calibration is not None:
        raise ValueError("Unknown calibration type.")

    return data


def load_CF4_mock(root, index):
    fname = join(root, f"mock_{index}.hdf5")
    with File(fname, 'r') as f:
        grp = f["mock"]
        data = {key: grp[key][...] for key in grp.keys()}
    return data
