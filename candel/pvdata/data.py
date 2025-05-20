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
            self.f_los_delta = LOSInterpolator(
                self.data["los_r"], self.data["los_delta"], **kwargs)
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
                   f"{d['rmax']} with {d['num_steps']} steps.")
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


def load_los(los_data_path, data, mask=None):
    with File(los_data_path, 'r') as f:
        data["los_density"] = f['los_density'][...][mask, ...]
        data["los_velocity"] = f['los_velocity'][...][mask, ...]
        data["los_r"] = f['r'][...]

        assert np.all(data["los_density"] > 0)
        assert np.all(np.isfinite(data["los_velocity"]))

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
                  return_all=False, dust_model=None, **kwargs):
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

        # Remove extinction correction if requested
        if dust_model is not None:
            if which_band not in ["w1", "w2"]:
                raise ValueError(
                    f"Band `{which_band}` is not supported for dust "
                    f"correction removal. Only `w1` and `w2` are supported.")

            Ab_default = grp[f"A_{which_band}"][...]
            fprint(f"switching the dust model to `{dust_model}`.")

            # Remove applied correction; new E(B-V) model handled externally
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

    fprint("subtracting the mean logF from the data.")
    data["logF"] -= np.mean(data["logF"])

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
