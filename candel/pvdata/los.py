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
import healpy as hp
import numpy as np
from h5py import File
from jax.nn import one_hot

from ..cosmo.cosmography import Redshift2Distance
from ..util import SPEED_OF_LIGHT, file_last_edited, fprint, radec_to_galactic
from .volume_density import _density_unit_normalization


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


def _filter_data(data, mask, los_data_path=None, field_indices=None):
    """Apply boolean mask to data arrays, report counts, and load LOS."""
    defer_los = bool(getattr(
        los_data_path, "requires_filtered_coordinates", False))
    if los_data_path and not defer_los:
        los_data_path = resolve_los_cache_request(los_data_path, data)

    n_total = len(mask)
    n_kept = int(np.sum(mask))
    fprint(f"removed {n_total - n_kept} objects, thus {n_kept} remain.")
    for k in data:
        if isinstance(data[k], np.ndarray):
            data[k] = data[k][mask]
    if los_data_path:
        if defer_los:
            los_data_path = resolve_los_cache_request(los_data_path, data)
            mask = None
        data = load_los(
            los_data_path, data, mask=mask, field_indices=field_indices)
    return data


def resolve_los_cache_request(los_data_path, data, mask=None, verbose=True):
    """Resolve a deferred LOS cache request against loaded catalogue data."""
    if not hasattr(los_data_path, "ensure_from_data"):
        return los_data_path
    if mask is not None:
        for key in ("RA", "RA_host"):
            if key in data and len(data[key]) != len(mask):
                raise ValueError(
                    "Deferred LOS cache requests must be resolved before "
                    "catalogue cuts are applied.")
    return los_data_path.ensure_from_data(data, verbose=verbose)


def _compute_r_grid(r_limits, dr, data, Om=0.3):
    """Compute a radial grid for Malmquist bias integration.

    ``dr`` is the target grid spacing in Mpc. The returned grid is adjusted to
    have an odd number of points so Simpson integration can be used downstream.
    """
    if isinstance(r_limits, str) and r_limits.startswith("auto"):
        if "_" in r_limits:
            h_auto = float(r_limits.split("_")[1])
        else:
            h_auto = 1.0

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


def _select_los_field_indices(data, los_field_indices, field_indices,
                              los_data_path, verbose=True):
    """Select requested field rows from already-loaded LOS arrays."""
    if field_indices is None:
        return los_field_indices.astype(np.int32)

    requested = np.asarray(field_indices, dtype=np.int32)
    if requested.ndim == 0:
        requested = requested[None]
    if requested.ndim != 1 or len(requested) == 0:
        raise ValueError(
            "`io.field_indices` must be an int or non-empty 1D list.")

    rows = []
    for nsim in requested:
        match = np.where(los_field_indices == nsim)[0]
        if len(match) != 1:
            available = ", ".join(map(str, los_field_indices.tolist()))
            raise ValueError(
                f"Requested LOS field index {int(nsim)} is not "
                f"available in `{los_data_path}`. Available: "
                f"{available}.")
        rows.append(int(match[0]))

    data["los_density"] = data["los_density"][rows]
    data["los_velocity"] = data["los_velocity"][rows]
    los_field_indices = los_field_indices[rows]
    fprint("selected LOS field indices: "
           f"{los_field_indices.tolist()}", verbose=verbose)
    return los_field_indices.astype(np.int32)


def _merge_los_coordinate(values):
    """Return a shared coordinate vector or stack per-field coordinates."""
    first = values[0]
    if all(np.array_equal(value, first) for value in values[1:]):
        return first
    return np.stack(values)


def load_los(los_data_path, data, mask=None, verbose=True,
             field_indices=None):
    los_data_path = resolve_los_cache_request(
        los_data_path, data, mask=mask, verbose=verbose)

    if isinstance(los_data_path, (list, tuple)):
        parts = [
            load_los(path, {}, mask=mask, verbose=verbose,
                     field_indices=None)
            for path in los_data_path
        ]
        data["los_density"] = np.concatenate(
            [part["los_density"] for part in parts], axis=0)
        data["los_velocity"] = np.concatenate(
            [part["los_velocity"] for part in parts], axis=0)
        data["los_r"] = parts[0]["los_r"]
        data["los_RA"] = _merge_los_coordinate(
            [part["los_RA"] for part in parts])
        data["los_dec"] = _merge_los_coordinate(
            [part["los_dec"] for part in parts])
        los_field_indices = np.concatenate(
            [part["los_field_indices"] for part in parts])
        data["los_field_indices"] = _select_los_field_indices(
            data, los_field_indices, field_indices, los_data_path,
            verbose=verbose)
        return data

    with File(los_data_path, 'r') as f:
        if mask is None:
            data["los_density"] = f['los_density'][...].astype(np.float32)
            data["los_velocity"] = f['los_velocity'][...].astype(np.float32)
            data["los_r"] = f['r'][...]
            data["los_RA"] = f["RA"][...]
            data["los_dec"] = f["dec"][...]
        else:
            dens = f['los_density'][...][:, mask, ...]
            data["los_density"] = dens.astype(np.float32)
            vel = f['los_velocity'][...][:, mask, ...]
            data["los_velocity"] = vel.astype(np.float32)
            data["los_r"] = f['r'][...]
            # Random LOS always use mask=None, so this 1D index is safe.
            data["los_RA"] = f["RA"][...][mask]
            data["los_dec"] = f["dec"][...][mask]

        assert np.all(data["los_density"] > 0)
        assert np.all(np.isfinite(data["los_velocity"]))
        if "field_indices" in f:
            los_field_indices = f["field_indices"][:].astype(np.int32)
        else:
            los_field_indices = np.arange(
                data["los_density"].shape[0], dtype=np.int32)

        norm = _density_unit_normalization(los_data_path)
        if norm is not None:
            divisor, label, detail = norm
            fprint(f"normalizing the {label} LOS density ({detail})",
                   verbose=verbose)
            data["los_density"] /= divisor

        if "_CB1" in los_data_path:
            if len(data["los_density"]) == 100:
                fprint("downsampling the CB1 LOS density from 100 to 20",
                       verbose=verbose)
                data["los_density"] = data["los_density"][::5]
                data["los_velocity"] = data["los_velocity"][::5]
                los_field_indices = los_field_indices[::5]
        elif "_CF4.hdf5" in los_data_path and len(data["los_density"]) == 100:
            fprint("downsampling the CF4 LOS density from 100 to 20",
                   verbose=verbose)
            data["los_density"] = data["los_density"][::5]
            data["los_velocity"] = data["los_velocity"][::5]
            los_field_indices = los_field_indices[::5]

        data["los_field_indices"] = _select_los_field_indices(
            data, los_field_indices, field_indices, los_data_path,
            verbose=verbose)

    edited = file_last_edited(los_data_path)
    if edited is None:
        fprint(f"loaded LOS file from `{los_data_path}`.", verbose=verbose)
    else:
        fprint(f"loaded LOS file from `{los_data_path}` "
               f"(last edited {edited}).", verbose=verbose)

    return data
