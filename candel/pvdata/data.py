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
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from h5py import File
from jax import numpy as jnp
from scipy.linalg import cholesky

from ..model.interp import LOSInterpolator
from ..util import (SPEED_OF_LIGHT, fprint, galactic_to_radec, load_config,
                    radec_to_cartesian, radec_to_galactic)
from .dust import read_dustmap

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
        self.name = None

        if "los_velocity" in self.data:
            self.has_precomputed_los = True
            kwargs = {"method": los_method, "extrap": los_extrap}
            self.f_los_delta = LOSInterpolator(
                self.data["los_r"], self.data["los_delta"], **kwargs)
            self.f_los_log_density = LOSInterpolator(
                self.data["los_r"], jnp.log(self.data["los_density"]),
                **kwargs)
            self.f_los_velocity = LOSInterpolator(
                self.data["los_r"], self.data["los_velocity"], **kwargs)
        else:
            self.has_precomputed_los = False

        self.has_calibrators = bool(self.num_calibrators > 0)
        self._cache = {}

    @classmethod
    def from_config_dict(cls, config, name, try_pop_los, config_io):
        root = config.pop("root")
        nsamples_subsample = config.pop("nsamples_subsample", None)
        seed_subsample = config.pop("seed_subsample", 42)
        mag_selection = config.pop("mag_selection", None)
        sample_dust = False

        if "CF4_mock" in name:
            index = name.split("_")[-1]
            data = load_CF4_mock(root, index)
        elif "CF4_" in name:
            data = load_CF4_data(root, **config)

            dust_model = config.get("dust_model", None)
            if dust_model is not None:
                fprint(f"using `{dust_model}` for the dust model.")
                sample_dust = True
        elif name == "2MTF":
            data = load_2MTF(root, **config)
        elif name == "SFI":
            data = load_SFI(root, **config)
        elif name == "SDSS_FP":
            data = load_SDSS_FP(root, **config)
        elif name == "PantheonPlus":
            data = load_PantheonPlus(root, **config)
        elif name == "Clusters":
            data = load_clusters(root, **config)
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
                   f"{d['rmax']} Mpc/h with {d['num_steps']} steps.")
            data["los_r"] = np.linspace(d["rmin"], d["rmax"], d["num_steps"])

        if "los_density" in data:
            data["los_log_density"] = np.log(data["los_density"])
            data["los_delta"] = data["los_density"] - 1

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

            frame.mag_dist_unif_kwargs = {
                "low": frame["min_mag"] - 0.5 * frame["std_mag"],
                "high": frame["max_mag"] + 0.5 * frame["std_mag"],
            }

        # Magnitude selection hyperparameters.
        if mag_selection is not None:
            if config["add_mag_selection"]:
                frame.mag_selection_kwargs = mag_selection
            else:
                frame.mag_selection_kwargs = None
                fprint(f"disabling magnitude selection for `{name}`.")
        frame.add_mag_selection = frame.mag_selection_kwargs is not None
        frame.sample_dust = sample_dust

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

        frame.name = name
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
            indx_choice, nsamples - int(self.num_calibrators), replace=False)
        main_mask[indx_choice] = True

        keys_skip = [
            "is_calibrator", "mu_cal", "C_mu_cal", "std_mu_cal", "los_r",
            "mag_covmat"]

        subsampled = {key: self[key][main_mask]
                      for key in self.keys() if key not in keys_skip}

        for key in keys_skip:
            if key in self.data:
                if key == "is_calibrator":
                    subsampled[key] = self[key][main_mask]
                elif key == "mag_covmat":
                    subsampled[key] = self.data[key][main_mask][:, main_mask]
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


def load_los(los_data_path, data, mask=None):
    with File(los_data_path, 'r') as f:
        data["los_density"] = f['los_density'][...][mask, ...]
        data["los_velocity"] = f['los_velocity'][...][mask, ...]
        data["los_r"] = f['r'][...]

        assert np.all(data["los_density"] > 0)
        assert np.all(np.isfinite(data["los_velocity"]))

    if "manticore" in los_data_path.lower():
        fprint("normalizing the Manticore LOS density.")
        data["los_density"] /= 0.3111 * 275.4  # Manticore normalization

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

            if not np.all(np.isfinite(ebv[0])):
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
    if zcmb_min is not None:
        mask &= zcmb > zcmb_min
    if zcmb_max is not None:
        mask &= zcmb < zcmb_max
    if remove_outliers:
        outliers = np.concatenate([
            np.genfromtxt(join(root, f"CF4_{b}_outliers.csv"),
                          delimiter=",", names=True)
            for b in ("W1", "i")
        ])
        mask &= ~np.isin(pgc, outliers["PGC"])
    if b_min is not None:
        b = radec_to_galactic(RA, DEC)[1]
        mask &= np.abs(b) > b_min

    if which_band == "i" and exclude_W1:
        with File(join(root, "CF4_TFR.hdf5"), 'r') as f:
            w1_quality = f["cf4"]["Qw"][...]
            w1_mag = f["cf4"]["w1"][...]
        fprint("excluding galaxies with W1 quality 5 or W1 mag < 5.")
        exclude = (w1_quality == 5) | (w1_mag > 5)
        mask &= ~exclude

    fprint(f"removed {len(pgc) - np.sum(mask)} galaxies, thus "
           f"{np.sum(mask)} remain.")

    for k in data:
        data[k] = data[k][mask]
    pgc = pgc[mask]

    if los_data_path:
        data = load_los(los_data_path, data, mask=mask)

    if calibration == "SH0ES":
        is_cal, mu, C_mu = load_SH0ES_calibration(
            join(root, "CF4_SH0ES_calibration.hdf5"), pgc)
        fprint(f"out of {len(pgc)} galaxies, {np.sum(is_cal)} are SH0ES  "
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

    fprint(f"initially loaded {len(zcmb)} galaxies from CF4 TFR data.")

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
    if zcmb_min is not None:
        mask &= zcmb > zcmb_min
    if zcmb_max is not None:
        mask &= zcmb < zcmb_max
    if b_min is not None:
        b = radec_to_galactic(RA, DEC)[1]
        mask &= np.abs(b) > b_min

    fprint(f"removed {len(zcmb) - np.sum(mask)} galaxies, thus "
           f"{np.sum(mask)} remain.")

    for k in data:
        data[k] = data[k][mask]

    if los_data_path:
        data = load_los(los_data_path, data, mask=mask)

    return data


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

    fprint(f"initially loaded {len(zcmb)} galaxies from CF4 TFR data.")

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
    if zcmb_min is not None:
        mask &= zcmb > zcmb_min
    if zcmb_max is not None:
        mask &= zcmb < zcmb_max
    if b_min is not None:
        b = radec_to_galactic(RA, DEC)[1]
        mask &= np.abs(b) > b_min

    fprint(f"removed {len(zcmb) - np.sum(mask)} galaxies, thus "
           f"{np.sum(mask)} remain.")

    for k in data:
        data[k] = data[k][mask]

    if los_data_path:
        data = load_los(los_data_path, data, mask=mask)

    return data


def load_PantheonPlus(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
                      los_data_path=None, return_all=False, **kwargs):
    """
    Load the Pantheon+ data from the given root directory, the covariance
    is expected to have peculiar velocity contribution removed.
    """
    arr = np.genfromtxt(
        join(root, "Pantheon+SH0ES_zsel.dat"), names=True, dtype=None,
        encoding=None)

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

    covmat = np.loadtxt(
        join(root, "Pantheon+SH0ES_zsel_STAT+SYS_noPV.cov"), delimiter=",")
    size = int(covmat[0])
    C = np.reshape(covmat[1:], (size, size))

    mask = np.ones(len(data["zcmb"]), dtype=bool)

    if zcmb_min is not None:
        mask &= data["zcmb"] > zcmb_min

    if zcmb_max is not None:
        mask &= data["zcmb"] < zcmb_max

    if b_min is not None:
        b = radec_to_galactic(data["RA"], data["dec"])[1]
        mask &= np.abs(b) > b_min

    fprint(f"removed {len(mask) - np.sum(mask)} galaxies, thus "
           f"{len(arr[mask])} remain.")

    for key in data:
        data[key] = data[key][mask]

    C = C[mask][:, mask]
    data["mag_covmat"] = C
    data["e_mag"] = np.sqrt(np.diag(C))  # Do not use in the inference!

    if los_data_path is not None:
        data = load_los(los_data_path, data, mask=mask)

    return data


def load_SH0ES(root):
    """
    Load the SH0ES data which can be used to sample distances.

    NOTE: Set the zero-width prior to a delta prior so it is not sampled?
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
                         replace_SN_HF_from_PP=False, los_data_path=None):
    """
    Load the separated SH0ES data, separating the Cepheid and supernovae and
    covariance matrices.

    Structure of the covariance matrix indices:
    ------------------------------------------
    - Indices < 2150: Cepheid hosts without geometric anchors.
    - Index 2150: Start of NGC 4258 Cepheid hosts.
    - Index 2593: Start of M31 Cepheid hosts.
    - Index 2648: Start of LMC Cepheid hosts.
    - Index 3130: Beginning of supernovae in Cepheid host galaxies.

    - Indices 3130–3206: Rung two supernovae (in Cepheid hosts).

    - Index 3207: Uncertainty on HST zeropoint (sigma_HST).
    - Index 3208: Uncertainty on Gaia zeropoint (sigma_Gaia).
    - Index 3209: Prior on metallicity coefficient Z_W.
    - Index 3210: Unused term (likely placeholder).
    - Index 3211: Ground-based photometry systematic uncertainty (sigma_grnd).
    - Index 3212: Prior on P–L relation slope b_W.
    - Index 3213: Constraint on NGC 4258 anchor offset (delta_mu_N4258).
    - Index 3214: Constraint on LMC anchor offset (delta_mu_LMC).

    - Indices ≥ 3215: Hubble flow supernovae (rung three).
    """
    if replace_SN_HF_from_PP:
        fprint("replacing SH0ES SN Hubble flow data with Pantheon+ data.")

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

    # SN data and covariance matrix.
    C_SN = C[3130:, 3130:]
    # Indices of the external constraints which we want to mask out.
    m_SN = ~np.isin(
        np.arange(len(C)),
        [3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214])[3130:]
    C_SN = C_SN[m_SN, :][:, m_SN]
    Y_SN = Y[3130:][m_SN]

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

    Y_SN_HF = Y[3215:]

    num_hosts = L_Cepheid_host_dist.shape[1] - 3
    num_cepheids = len(mag_cepheid)

    # Cepheid host redshifts and the PV covariance matrix.
    data_cepheid_host_redshift = np.load(
        join(root, "processed", "Cepheid_anchors_redshifts.npy"),
        allow_pickle=True)
    PV_covmat_cepheid_host = np.load(
        join(root, "processed", "PV_covmat_cepheid_hosts_fiducial.npy"),
        allow_pickle=True)

    if los_data_path is not None:
        data_host_los = {}
        data_host_los = load_los(
            los_data_path, data_host_los)
        host_los_density = data_host_los["los_density"][0]
        host_los_velocity = data_host_los["los_velocity"][0]
        host_los_r = data_host_los["los_r"]
    else:
        host_los_density = None
        host_los_velocity = None
        host_los_r = None

    # SH0ES-Antonio's approach for predicting H0 from Cepheid host redshifts,
    # the error propagation is biased when z -> 0.
    # zHD = data_cepheid_host_redshift["zHD"]
    # q0 = -0.55
    # c = SPEED_OF_LIGHT
    # Y_Cepheid_new = 5 * np.log10((c * zHD) * (
    #     1 +  0.5 * (1 - q0) * zHD)) + 25
    # v_pec = 250
    # Y_Cepheid_new_err = (5 / np.log(10)
    #                      * (1 + (1 - q0) * zHD) / (1 + 0.5 * (1 - q0) * zHD)
    #                      * v_pec / (c * zHD))

    if replace_SN_HF_from_PP:
        f = np.load("/Users/rstiskalek/Projects/CANDEL/data/SH0ES/processed/PP_SN_matched_to_SH0ES.npz")  # noqa
        pp = f["pp_matched"]
        m_HF = pp["USED_IN_SH0ES_HF"] == 1
        C_SN = f["cov"]
        Y_SN = pp["m_b_corr"]
        czcmb_SN_HF = pp["zCMB"][m_HF] * SPEED_OF_LIGHT
        e_czcmb_SN_HF = pp["zCMBERR"][m_HF] * SPEED_OF_LIGHT
        RA_SN_HF = pp["RA"][m_HF]
        dec_SN_HF = pp["DEC"][m_HF]
    else:
        czcmb_SN_HF, e_czcmb_SN_HF = None, None
        RA_SN_HF, dec_SN_HF = None, None

    # Pick one SN per Cepheid host galaxy
    mag_SN = np.zeros(40)
    unique_ks = []
    for i in range(len(Y_SN_Cepheid)):
        j = np.where(L_SN_Cepheid_dist[i] == 1)[0][0]

        if mag_SN[j] == 0:
            mag_SN[j] = Y_SN_Cepheid[i]
            unique_ks.append(i)

    unique_ks = np.asarray(unique_ks)
    mag_SN_unique_Cepheid_host = Y_SN_Cepheid[unique_ks]
    C_SN_unique_Cepheid_host = C_SN_Cepheid[unique_ks][:, unique_ks]
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
        # SNe in Cepheid host galaxies, covariance matrix and host association.
        "Y_SN_Cepheid": Y_SN_Cepheid,
        "C_SN_Cepheid": C_SN_Cepheid,
        "L_SN_Cepheid": cholesky(C_SN_Cepheid, lower=True),
        "L_SN_Cepheid_dist": L_SN_Cepheid_dist,
        # Unique SNe in Cepheid host galaxies.
        "mag_SN_unique_Cepheid_host": mag_SN_unique_Cepheid_host,
        "C_SN_unique_Cepheid_host": C_SN_unique_Cepheid_host,
        "std_mag_SN_unique_Cepheid_host": np.sqrt(
            np.diag(C_SN_unique_Cepheid_host)),
        "L_SN_unique_Cepheid_host": cholesky(C_SN_unique_Cepheid_host,
                                             lower=True),
        "L_SN_unique_Cepheid_host_dist": L_SN_unique_Cepheid_host_dist,
        # SNe in the Hubble flow and covariance matrix.
        "Y_SN_HF": Y_SN_HF,
        "num_flow_SN": len(Y_SN_HF),
        "czcmb_SN_HF": czcmb_SN_HF,
        "e_czcmb_SN_HF": e_czcmb_SN_HF,
        "RA_SN_HF": RA_SN_HF,
        "dec_SN_HF": dec_SN_HF,
        # All SNe together.
        "Y_SN": Y_SN,
        "C_SN": C_SN,
        "L_SN": cholesky(C_SN, lower=True),
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
        "czcmb_cepheid_host": data_cepheid_host_redshift["zCMB"] * SPEED_OF_LIGHT,  # noqa
        "e_czcmb_cepheid_host": data_cepheid_host_redshift["zCMBERR"],
        "RA_host": data_cepheid_host_redshift["RA"],
        "dec_host": data_cepheid_host_redshift["DEC"],
        "PV_covmat_cepheid_host": PV_covmat_cepheid_host,
        "host_los_density": host_los_density,
        "host_los_velocity": host_los_velocity,
        "host_los_r": host_los_r,
        # # SH0ES-Antonio's approach for predicting H0 from Cepheid host zs
        # "Y_Cepheid_new": Y_Cepheid_new,
        # "Y_Cepheid_new_err": Y_Cepheid_new_err
        "q_names": q_names,
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

        data["mask_host"] = mask_host

        for key, val in data.items():
            if "SN" in key and "SN_unique" not in key and isinstance(val, np.ndarray):  # noqa
                data[key] = np.full_like(val, np.nan, dtype=val.dtype)

    data["mean_logP"] = np.mean(data["logP"])
    data["mean_OH"] = np.mean(data["OH"])

    return data


def load_SH0ES_from_config(config_path):
    config = load_config(config_path, replace_los_prior=False)
    d = config["io"]["SH0ES"]
    root = d["root"]
    cepheid_host_cz_cmb_max = d.get("cepheid_host_cz_cmb_max", None)
    replace_SN_HF_from_PP = d.get("replace_SN_HF_from_PP", False)

    which_host_los = d.get("which_host_los", None)
    if which_host_los is not None:
        los_data_path = config["io"]["PV_main"]["SH0ES"]["los_file"].replace(
            "<X>", which_host_los)
    else:
        los_data_path = None

    return load_SH0ES_separated(
        root, cepheid_host_cz_cmb_max, replace_SN_HF_from_PP,
        los_data_path=los_data_path)


def load_clusters(root, zcmb_min=None, zcmb_max=None, los_data_path=None,
                  return_all=False, **kwargs):
    """
    Load the cluster scaling relation data from the given root directory.

    Y is currently not being loaded.
    """
    fname = join(root, "ClustersData.txt")

    dtype = [
        ('Cluster', 'U32'), ('z', 'f8'), ('Glon', 'f8'), ('Glat', 'f8'),
        ('Offset', 'f8'), ('T', 'f8'), ('Tmax', 'f8'), ('Tmin', 'f8'),
        ('Lx', 'f8'), ('eL', 'f8'), ('NHtot', 'f8'), ('Metal', 'f8'),
        ('Met_max', 'f8'), ('Met_min', 'f8'), ('Y_arcmin2', 'f8'),
        ('e_Y', 'f8'), ('Y5r500', 'f8'), ('e_Y2', 'f8'), ('Y_nr_no_ksz', 'f8'),
        ('e_Y3', 'f8'), ('Y_nr_mmf', 'f8'), ('e_Y4', 'f8'), ('Y_nr_mf', 'f8'),
        ('e_Y5', 'f8'), ('Abs2MASS', 'f8'), ('BCG_Offset', 'f8'),
        ('Catalog', 'U32'), ('Analysed_by', 'U32')
    ]

    data = np.genfromtxt(fname, dtype=dtype, skip_header=1)
    data = data[(data['Y_nr_no_ksz'] != -1.0)]
    fprint(f"initially loaded {len(data)} clusters.")

    z = data['z']
    T = data['T']
    Lx = data['Lx']
    eL = data['eL']
    Tmax = data['Tmax']
    Tmin = data['Tmin']
    Y_arcmin2 = data['Y_arcmin2']
    e_Y = data['e_Y']

    # The file assumes a cosmology with H0 = 70 km/s/Mpc and Omega_m = 0.3
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    DL = cosmo.luminosity_distance(z).value

    logT = np.log10(T)
    logF = np.log10(Lx / (4 * np.pi * DL**2))
    logY = np.log10(Y_arcmin2)

    e_logT = np.log10(np.e) * (Tmax - Tmin) / (2 * T)
    e_logF = np.log10(np.e) * (Lx * eL / 100) / Lx
    e_logY = np.log10(np.e) * e_Y / Y_arcmin2

    RA, dec = galactic_to_radec(data['Glon'], data['Glat'])

    # Add the factor of 4 \pi to the logF to avoid having to add it every time.
    logF += np.log10(4 * np.pi)

    data = {
        "zcmb": z,
        "RA": RA,
        "dec": dec,
        "logT": logT,
        "e_logT": e_logT,
        "logF": logF,
        "e_logF": e_logF,
        "logY": logY,
        "e_logY": e_logY,
    }

    if return_all:
        return data

    mask = np.ones(len(z), dtype=bool)

    if zcmb_min is not None:
        mask &= z > zcmb_min

    if zcmb_max is not None:
        mask &= z < zcmb_max

    fprint(f"removed {len(mask) - np.sum(mask)} clusters, thus "
           f"{len(data['RA'][mask])} remain.")

    for key in data:
        data[key] = data[key][mask]

    fprint("subtracting the mean logT from the data.")
    data["logT"] -= np.mean(data["logT"])

    fprint("subtracting the mean logY from the data.")
    data["logY"] -= np.mean(data["logY"])

    # fprint("subtracting the mean logF from the data.")
    # data["logF"] -= np.mean(data["logF"])

    if los_data_path is not None:
        data = load_los(los_data_path, data, mask=mask)

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
        "logI": d_input["i"],
        "e_logI": d_input["ei"],
        "logs": d_input["s"],
        "e_logs": d_input["es"],
        }

    if return_all:
        return data

    mask = np.ones(len(data["zcmb"]), dtype=bool)

    if zcmb_min is not None:
        mask &= data["zcmb"] > zcmb_min

    if zcmb_max is not None:
        mask &= data["zcmb"] < zcmb_max

    if b_min is not None:
        b = radec_to_galactic(data["RA"], data["dec"])[1]
        mask &= np.abs(b) > b_min

    fprint(f"removed {len(mask) - np.sum(mask)} galaxies, thus "
           f"{len(data['RA'][mask])} remain.")

    for key in data:
        data[key] = data[key][mask]

    if los_data_path is not None:
        data = load_los(los_data_path, data, mask=mask)

    return data
