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

import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from h5py import File
from jax import numpy as jnp
from scipy import linalg
from scipy.linalg import cholesky

from ..util import (SPEED_OF_LIGHT, fprint, get_nested, get_root_data,
                    load_config)
from .dust import read_dustmap
from .los import (_compute_r_grid, _filter_data, _zcmb_blat_mask,
                  effective_rank_entropy, load_los)
from .volume_density import _load_h0_volume_data_from_config


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
                  field_indices=None, return_all=False, dust_model=None,
                  exclude_W1=False, **kwargs):
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

    _filter_data(data, mask, los_data_path, field_indices=field_indices)
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
                         los_data_path=None, rand_los_data_path=None,
                         volume_data=None, field_indices=None):
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
    # Indices of Cepheids needing ground-based dZP correction (column 45 of
    # the original L matrix is Delta_zp).
    idx_dZP = np.where(L[:3130, 45] == 1)[0]

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
        d = load_los(path, {}, field_indices=field_indices)
        return {k: d[k] for k in keys}

    host_los_keys = [
        "los_density", "los_velocity", "los_r", "los_field_indices"]
    host_los = _load_los_or_none(los_data_path, host_los_keys)

    rand_los_keys = [
        "los_density", "los_velocity", "los_r", "los_RA", "los_dec"]
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
        "idx_dZP": idx_dZP,
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
        "host_los_field_indices": host_los["los_field_indices"],
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

        # Remap idx_dZP: keep only indices that survive the mask, then
        # convert to new positions in the masked array.
        old_to_new = np.full(len(mask_cepheid), -1, dtype=int)
        old_to_new[mask_cepheid] = np.arange(mask_cepheid.sum())
        data["idx_dZP"] = old_to_new[data["idx_dZP"]]
        data["idx_dZP"] = data["idx_dZP"][data["idx_dZP"] >= 0]
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

    data["Neff_C_SN_unique_Cepheid_host"] = effective_rank_entropy(data["C_SN_unique_Cepheid_host"])  # noqa
    data["Neff_PV_covmat_cepheid_host"] = effective_rank_entropy(data["PV_covmat_cepheid_host"])     # noqa
    data["Neff_C_Cepheid"] = effective_rank_entropy(data["C_Cepheid"])

    if volume_data is not None:
        data.update(volume_data)
        data["has_volume_density_3d"] = True
    else:
        data["has_volume_density_3d"] = False

    return data


def load_SH0ES_from_config(config_path):
    config = load_config(config_path, replace_los_prior=False)
    use_recon = get_nested(config, "model/use_reconstruction", False)
    config["io"]["load_host_los"] = use_recon
    config["io"]["load_rand_los"] = False
    d = config["io"]["SH0ES"]
    root = d["root"]
    cepheid_host_cz_cmb_max = d.get("cepheid_host_cz_cmb_max", None)
    which_host_los = d.get("which_host_los", None)
    field_indices = get_nested(config, "io/field_indices", None)
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

    data = load_SH0ES_separated(
        root, cepheid_host_cz_cmb_max,
        los_data_path=los_data_path, rand_los_data_path=rand_los_data_path,
        field_indices=field_indices)

    velocity_selections = ["redshift", "SN_magnitude_redshift"]
    if get_nested(config, "model/which_selection", None) \
            == "SN_magnitude_or_redshift_Nmag":
        n_mag = get_nested(config, "model/num_hosts_selection_mag", None)
        if type(n_mag) is int and n_mag < data["num_hosts"]:
            velocity_selections.append("SN_magnitude_or_redshift_Nmag")

    volume_data = _load_h0_volume_data_from_config(
        config, los_data_path, which_host_los, "SH0ES",
        velocity_selections=tuple(velocity_selections),
        field_indices=data.get("host_los_field_indices", None))

    if volume_data is not None:
        data.update(volume_data)
        data["has_volume_density_3d"] = True

    return data


def load_CCHP_from_config(config_path, ra_dec_only=False):
    """
    Load the processed CCHP TRGB catalogue from a TSV file.

    If ``ra_dec_only`` is True, returns only ``RA``, ``DEC``, ``cz_cmb``, and
    ``e_czcmb`` arrays from the table before host grouping or LOS loading.
    Otherwise, returns the full model-ready data described below.

    Returns data in EDD-TRGB-compatible format (host-level arrays with
    ``mag_obs``, ``e_mag_obs``, ``czcmb``, ``e_czcmb``, ``RA_host``,
    ``dec_host``). For SN magnitude selection it also returns one SN per host
    (``m_Bprime``, ``e_m_Bprime``, ``sn_group_index``).

    Expects the TSV to contain at least the columns:
    SN, Galaxy, cz_cmb, e_czcmb, mu_TRGB_CCHP, sigma_TRGB_CCHP,
    m_Bprime_CSP, sigma_Bprime_CSP.
    """
    config = load_config(config_path, replace_los_prior=False)
    use_recon = get_nested(config, "model/use_reconstruction", False)
    which_sel = get_nested(config, "model/which_selection", None)
    use_sn_selection = which_sel == "SN_magnitude"
    config["io"]["load_host_los"] = use_recon
    config["io"]["load_rand_los"] = False
    path = config["io"]["CCHP"]["path"]
    redshift_source = get_nested(
        config, "io/CCHP_redshift_source/kind", "cz_cmb")
    if not isabs(path):
        path = join(get_root_data(config), path)

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
    e_mag_trgb = data_tbl["sigma_TRGB_CCHP"]

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
    m_Bprime = data_tbl["m_Bprime_CSP"]
    e_m_Bprime = data_tbl["sigma_Bprime_CSP"]
    galaxies = np.asarray(data_tbl["Galaxy"])

    if ra_dec_only:
        return {
            "RA": ra,
            "DEC": dec,
            "cz_cmb": cz_cmb,
            "e_czcmb": e_czcmb,
        }

    row_mask = np.ones(len(galaxies), dtype=bool)
    if use_sn_selection:
        row_mask = np.isfinite(m_Bprime) & np.isfinite(e_m_Bprime)
        n_masked = int(np.sum(~row_mask))
        if n_masked > 0:
            fprint(f"CCHP: masking {n_masked}/{len(row_mask)} entries "
                   "without finite SN photometry.")
    mag_trgb = mag_trgb[row_mask]
    e_mag_trgb = e_mag_trgb[row_mask]
    cz_cmb = cz_cmb[row_mask]
    e_czcmb = e_czcmb[row_mask]
    m_Bprime = m_Bprime[row_mask]
    e_m_Bprime = e_m_Bprime[row_mask]
    ra = ra[row_mask]
    dec = dec[row_mask]
    galaxies = galaxies[row_mask]

    # Group by Galaxy. For SN selection keep the most precise CSP SN per host;
    # otherwise keep one TRGB row per host without using SN data.
    galaxies_unique, inverse = np.unique(galaxies, return_inverse=True)
    n_hosts = len(galaxies_unique)
    if use_sn_selection:
        selected_idx = np.array([
            idx[np.argmin(e_m_Bprime[idx])]
            for i in range(n_hosts)
            for idx in [np.where(inverse == i)[0]]
        ])
        fprint(f"CCHP: selected {len(selected_idx)} lowest-uncertainty SNe "
               f"across {n_hosts} unique hosts from "
               f"{len(galaxies)} finite SNe.")
    else:
        selected_idx = np.array([np.where(inverse == i)[0][0]
                                 for i in range(n_hosts)])
        fprint(f"CCHP: selected {n_hosts} unique TRGB hosts from "
               f"{len(galaxies)} table rows.")

    # Anchor calibration from config (with CCHP defaults)
    anchors = get_nested(config, "model/anchors", {})

    # Build output dict with EDD-compatible host-level keys
    data = {
        # Host-level arrays (one per unique host)
        "mag_obs": mag_trgb[selected_idx],
        "e_mag_obs": e_mag_trgb[selected_idx],
        "czcmb": cz_cmb[selected_idx],
        "e_czcmb": e_czcmb[selected_idx],
        "RA_host": ra[selected_idx],
        "dec_host": dec[selected_idx],
        "e_mag_median": float(np.median(e_mag_trgb[selected_idx])),
        # Anchors
        "mu_LMC_anchor": anchors.get("mu_LMC", mu_LMC_anchor),
        "e_mu_LMC_anchor": anchors.get("e_mu_LMC", e_mu_LMC_anchor),
        "mag_LMC_TRGB": anchors.get("mag_LMC_TRGB", mag_LMC_TRGB),
        "e_mag_LMC_TRGB": anchors.get("e_mag_LMC_TRGB", e_mag_LMC_TRGB),
        "mu_N4258_anchor": anchors.get("mu_N4258", mu_N4258_anchor),
        "e_mu_N4258_anchor": anchors.get("e_mu_N4258", e_mu_N4258_anchor),
        "mag_N4258_TRGB": anchors.get("mag_N4258_TRGB", mag_N4258_TRGB),
        "e_mag_N4258_TRGB": anchors.get(
            "e_mag_N4258_TRGB", e_mag_N4258_TRGB),
    }
    if use_sn_selection:
        data.update({
            # SN-level arrays (one selected SN per host)
            "m_Bprime": m_Bprime[selected_idx],
            "e_m_Bprime": e_m_Bprime[selected_idx],
            "e_m_Bprime_median": float(np.median(e_m_Bprime[selected_idx])),
            "sn_group_index": np.arange(n_hosts, dtype=np.int32),
        })

    # Load LOS data (host and/or random)
    los_data_path = None
    rand_los_data_path = None

    which_host_los = get_nested(
        config, "io/which_host_los",
        get_nested(config, "io/CCHP/which_host_los", None))
    field_indices = get_nested(config, "io/field_indices", None)
    if get_nested(config, "io/load_host_los", False):
        los_file = get_nested(config, "io/CCHP/los_file", None)
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
        host_los = load_los(
            los_data_path, {}, mask=None, field_indices=field_indices)
        # LOS file has one entry per row in the original TSV (25 entries).
        # Apply the same row mask as above, then extract selected host rows.
        los_density = host_los["los_density"][:, row_mask]
        los_velocity = host_los["los_velocity"][:, row_mask]
        data["host_los_density"] = los_density[:, selected_idx]
        data["host_los_velocity"] = los_velocity[:, selected_idx]
        data["host_los_r"] = host_los["los_r"]
        data["host_los_field_indices"] = host_los["los_field_indices"]

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

    volume_data = _load_h0_volume_data_from_config(
        config, los_data_path, which_host_los, "CCHP",
        velocity_selections=("redshift",),
        field_indices=data.get("host_los_field_indices", None))
    if volume_data is not None:
        data.update(volume_data)
        data["has_volume_density_3d"] = True
    else:
        data["has_volume_density_3d"] = False

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
    - model.num_points_malmquist: value currently passed as the radial-grid
      spacing argument to ``_compute_r_grid``
    """
    config = load_config(config_path, replace_los_prior=False)

    csp_root = get_nested(config, "io/CSP/root", None)
    if csp_root is None:
        raise ValueError("CSP root not specified in config [io.CSP.root]")
    if not isabs(csp_root):
        csp_root = join(get_root_data(config), csp_root)

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
    root : str
        Directory containing the CSP data files.
    zcmb_min, zcmb_max : float, optional
        Inclusive CMB-frame redshift bounds.
    b_min : float, optional
        Minimum absolute Galactic latitude.
    quality_min : float, optional
        Minimum CSP quality flag.
    st_min, st_max : float, optional
        Light-curve stretch bounds.
    t0_min, t0_max : float, optional
        Time-of-maximum bounds.
    phys_only : bool
        If True, keep only physics-sample objects (phys=1).
    exclude_phys : bool
        If True, exclude physics sample (phys=0 only).
    which_sample : str, optional
        Sample to select: "CSPI", "CSPII", or "LSQ".
    los_data_path : str, optional
        Path to precomputed line-of-sight data to attach.
    return_all : bool
        If True, return all loaded rows before cuts and LOS attachment.
    remove_duplicates : bool
        If True, remove duplicate supernova entries before applying cuts.
    kwargs : dict
        Ignored extra keyword arguments, accepted for config-loader
        compatibility.
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


def _parse_edd_trgb_txt(fpath):
    """Parse an EDD TRGB text file (5 header lines, comma-delimited).

    Returns rows, header, and whether the file is the grouped format
    (extra Vcmb column at index 1).
    """
    with open(fpath) as f:
        lines = f.readlines()
    header = [c.strip() for c in lines[1].strip().split(",")]
    ncol = len(header)
    rows = []
    for line in lines[5:]:
        row = [c.strip().strip('"') for c in line.strip().split(",")]
        if len(row) == ncol:
            rows.append(row)

    has_group_vcmb = (header[1] == "Vcmb")
    return rows, header, has_group_vcmb


def _edd_col_float(rows, idx):
    """Extract a float column, returning NaN for empty/missing cells."""
    out = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        try:
            out[i] = float(row[idx])
        except (ValueError, IndexError):
            pass
    return out


def _edd_col_str(rows, idx):
    return np.array([row[idx].strip() for row in rows])


def _load_edd_trgb_core(fpath, label, zcmb_min=None, zcmb_max=None,
                        b_min=None, los_data_path=None, return_all=False,
                        return_mask=False, e_czcmb_default=20.0,
                        mag_min_TRGB=22.1):
    """Shared loader for ungrouped and grouped EDD TRGB files.

    The grouped file has an extra CF4 group Vcmb at column 1 (detected
    automatically), stored as ``czcmb_group`` in km/s.
    """
    rows, header, has_group_vcmb = _parse_edd_trgb_txt(fpath)
    n_orig = len(rows)
    fprint(f"initially loaded {n_orig} galaxies from {label} data.")

    off = 1 if has_group_vcmb else 0

    RA = _edd_col_float(rows, 7 + off)        # RAJ
    dec = _edd_col_float(rows, 8 + off)        # DeJ
    czcmb = _edd_col_float(rows, 20 + off)     # individual Vcmb
    T814 = _edd_col_float(rows, 45 + off)
    T8_lo = _edd_col_float(rows, 46 + off)
    T8_hi = _edd_col_float(rows, 47 + off)
    colour_606_814 = _edd_col_float(rows, 48 + off)
    colour_lo = _edd_col_float(rows, 49 + off)
    colour_hi = _edd_col_float(rows, 50 + off)
    A_814 = _edd_col_float(rows, 62 + off)
    M_TRGB_Anand = _edd_col_float(rows, 63 + off)
    names = _edd_col_str(rows, 35 + off)

    zcmb_arr = czcmb / SPEED_OF_LIGHT
    colour_edd = 1.23 + (M_TRGB_Anand + 4.06) / 0.20
    e_colour_edd = np.abs(colour_hi - colour_lo) / 2
    e_colour_edd = np.where(np.isfinite(e_colour_edd), e_colour_edd, 0.0)

    data = dict(
        RA=RA,
        dec=dec,
        zcmb=zcmb_arr,
        e_zcmb=np.full(n_orig, e_czcmb_default / SPEED_OF_LIGHT),
        mag=T814 - A_814,
        e_mag=(T8_hi - T8_lo) / 2,
        colour_dered=colour_edd,
        colour_606_814=colour_606_814,
        e_colour_dered=e_colour_edd,
        host_names=names,
    )

    if has_group_vcmb:
        data["czcmb_group"] = _edd_col_float(rows, 1)

    if return_all:
        return data

    keep = np.ones(n_orig, dtype=bool)

    # Drop anchor and satellite galaxies (treated separately in the model).
    drop = np.isin(names, ["LMC", "SMC", "NGC4258", "NGC4258-DF6"])
    if np.any(drop):
        fprint(f"dropping {np.sum(drop)} anchor/satellite galaxies: "
               f"{', '.join(names[drop])}")
    keep &= ~drop

    # Drop galaxies with missing TRGB magnitudes.
    bad_mag = keep & ~np.isfinite(data["mag"])
    if np.any(bad_mag):
        fprint(f"dropping {np.sum(bad_mag)} galaxies with missing TRGB "
               f"magnitudes.")
    keep &= ~bad_mag

    if mag_min_TRGB is not None:
        bright_mag = keep & (data["mag"] < mag_min_TRGB)
        if np.any(bright_mag):
            fprint(
                f"dropping {np.sum(bright_mag)} galaxies brighter than "
                f"mag_min_TRGB={mag_min_TRGB}.")
        keep &= ~bright_mag

    # Drop galaxies without the EDD/Rizzi colour-standardization term.
    bad_colour = keep & (
        ~np.isfinite(data["colour_dered"])
        | ~np.isfinite(data["e_colour_dered"])
    )
    if np.any(bad_colour):
        fprint(f"dropping {np.sum(bad_colour)} galaxies with missing "
               f"EDD/Rizzi colour-standardization term.")
    keep &= ~bad_colour

    # Drop galaxies with fill-value Vcmb (9999 = no measured velocity).
    bad_vcmb = keep & (np.abs(czcmb) >= 9999)
    if np.any(bad_vcmb):
        fprint(f"dropping {np.sum(bad_vcmb)} galaxies with fill-value Vcmb.")
    keep &= ~bad_vcmb

    # Apply zcmb / galactic latitude cuts on the kept subset.
    sub_mask = _zcmb_blat_mask(
        zcmb_arr[keep], RA[keep], dec[keep], zcmb_min, zcmb_max, b_min)
    keep[np.where(keep)[0][~sub_mask]] = False

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


def load_EDD_TRGB(root, **kwargs):
    """Load ungrouped EDD TRGB data (``EDD_TRGB.txt``)."""
    return _load_edd_trgb_core(
        join(root, "EDD_TRGB.txt"), "EDD TRGB", **kwargs)


def load_EDD_TRGB_grouped(root, **kwargs):
    """Load grouped EDD TRGB data (``EDD_TRGB_grouped.txt``).

    Includes ``czcmb_group`` from the CF4 group catalogue.
    """
    return _load_edd_trgb_core(
        join(root, "EDD_TRGB_grouped.txt"), "EDD TRGB grouped", **kwargs)


def _load_EDD_TRGB_from_config_common(config_path, config_key, loader):
    """Shared from_config logic for both EDD TRGB variants."""
    config = load_config(config_path, replace_los_prior=False)
    use_recon = get_nested(config, "model/use_reconstruction", False)
    config["io"]["load_host_los"] = use_recon
    config["io"]["load_rand_los"] = False
    d = config["io"]["PV_main"][config_key]
    root = d["root"]

    zcmb_min = get_nested(config, f"io/PV_main/{config_key}/zcmb_min", None)
    zcmb_max = get_nested(config, f"io/PV_main/{config_key}/zcmb_max", None)
    b_min = get_nested(config, f"io/PV_main/{config_key}/b_min", None)

    mag_min_TRGB = get_nested(config, "model/mag_min_TRGB", 22.1)
    data, mask = loader(root, zcmb_min=zcmb_min, zcmb_max=zcmb_max,
                        b_min=b_min, return_mask=True,
                        mag_min_TRGB=mag_min_TRGB)

    data["RA_host"] = data.pop("RA")
    data["dec_host"] = data.pop("dec")
    data["mag_obs"] = data.pop("mag")
    data["e_mag_obs"] = data.pop("e_mag")
    # For grouped data, use the group Vcmb instead of individual.
    if "czcmb_group" in data:
        data["czcmb"] = data.pop("czcmb_group")
        data.pop("zcmb")
    else:
        data["czcmb"] = data.pop("zcmb") * SPEED_OF_LIGHT
    data["e_czcmb"] = data.pop("e_zcmb") * SPEED_OF_LIGHT
    data["e_mag_median"] = float(np.median(data["e_mag_obs"]))

    which_los = get_nested(
        config, "io/which_host_los",
        get_nested(config, f"io/PV_main/{config_key}/which_host_los", None))
    field_indices = get_nested(config, "io/field_indices", None)

    def _resolve_los_path(path):
        if path is not None and which_los is not None:
            return path.replace("<X>", which_los)
        return path

    los_data_path = None
    rand_los_data_path = None
    if get_nested(config, "io/load_host_los", False):
        los_data_path = _resolve_los_path(d.get("los_file", None))
    if get_nested(config, "io/load_rand_los", False):
        rand_los_data_path = _resolve_los_path(
            get_nested(config, "io/los_file_random", None))

    fprint(f"reconstruction: {which_los or 'none'}")
    if los_data_path is not None:
        fprint(f"  host LOS path: {los_data_path}")
        host_los = load_los(
            los_data_path, {}, mask=mask, field_indices=field_indices)
        data["host_los_density"] = host_los["los_density"]
        data["host_los_velocity"] = host_los["los_velocity"]
        data["host_los_r"] = host_los["los_r"]
        data["host_los_field_indices"] = host_los["los_field_indices"]
        fprint(f"  host LOS shape: {host_los['los_density'].shape}")

    if rand_los_data_path is not None:
        fprint(f"  random LOS path: {rand_los_data_path}")
        rand_los = load_los(rand_los_data_path, {}, mask=None, verbose=False)
        data["rand_los_density"] = rand_los["los_density"]
        data["rand_los_velocity"] = rand_los["los_velocity"]
        data["rand_los_r"] = rand_los["los_r"]
        data["rand_los_RA"] = rand_los.get("los_RA", None)
        data["rand_los_dec"] = rand_los.get("los_dec", None)
        data["has_rand_los"] = True
        data["num_rand_los"] = data["rand_los_density"].shape[1]
        fprint(f"  random LOS shape: {rand_los['los_density'].shape}"
               f" ({data['num_rand_los']} LOS)")
    else:
        data["has_rand_los"] = False

    volume_data = _load_h0_volume_data_from_config(
        config, los_data_path, which_los, config_key,
        velocity_selections=("redshift",),
        field_indices=data.get("host_los_field_indices", None))

    if volume_data is not None:
        data.update(volume_data)
        data["has_volume_density_3d"] = True
    else:
        data["has_volume_density_3d"] = False

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


def load_EDD_TRGB_from_config(config_path):
    """Load ungrouped EDD TRGB data from config."""
    return _load_EDD_TRGB_from_config_common(
        config_path, "EDD_TRGB", load_EDD_TRGB)


def load_EDD_TRGB_grouped_from_config(config_path):
    """Load grouped EDD TRGB data from config."""
    return _load_EDD_TRGB_from_config_common(
        config_path, "EDD_TRGB_grouped", load_EDD_TRGB_grouped)


def load_EDD_2MTF(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
                  eta_min=None, eta_max=None,
                  los_data_path=None, return_all=False,
                  return_mask=False, **kwargs):
    """Load 2MTF data from the EDD text file.

    The file format is pipe-delimited with 5 header lines.
    Returns K-band apparent magnitudes (Ktc) and linewidths
    (eta = log10(Wc) - 2.5), with optional LOS arrays attached. If
    ``return_mask`` is True, returns ``(data, mask)``.
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
    use_recon = get_nested(config, "model/use_reconstruction", False)
    config["io"]["load_host_los"] = use_recon
    config["io"]["load_rand_los"] = use_recon
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
    "EDD_TRGB_grouped": load_EDD_TRGB_grouped,
    "EDD_2MTF": load_EDD_2MTF,
}
