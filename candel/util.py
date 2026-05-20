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
"""Various utility functions for candel."""

try:
    # Python 3.11+
    import tomllib  # noqa
except ModuleNotFoundError:
    # Backport for <=3.10
    import tomli as tomllib

from datetime import datetime
from os.path import abspath, exists, isabs, join
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord
from h5py import File

SPEED_OF_LIGHT = 299_792.458  # km / s


def fsection(title, width=60):
    """Print a section header."""
    rule = "─" * (width - len(title) - 3)
    print(f"── {title} {rule}")


def fprint(*args, verbose=True, **kwargs):
    """Print an indented status message."""
    if verbose:
        print("  ", *args, **kwargs)


def file_last_edited(path):
    """Return a local ISO timestamp for a file mtime, or ``None``."""
    try:
        timestamp = Path(path).stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(
        timespec="seconds")


def patch_tqdm(mininterval=5):
    """Monkey-patch tqdm to reduce output frequency for long-running jobs."""
    import tqdm
    _Orig = tqdm.tqdm

    class _Slow(_Orig):
        def __init__(self, *a, **kw):
            kw.setdefault("mininterval", mininterval)
            super().__init__(*a, **kw)

    tqdm.tqdm = _Slow


def convert_none_strings(d):
    """
    Convert all string values in a dictionary to None if they are equal to
    "none" (case insensitive). This is useful for parsing TOML files where
    "none" is used to represent None values.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            convert_none_strings(v)
        elif isinstance(v, str) and v.strip().lower() == "none":
            d[k] = None
    return d


def replace_prior_with_delta(config, param, value, verbose=True):
    """Replace the prior of `param` with a delta distribution at `value`."""
    if param not in config.get("model", {}).get("priors", {}):
        return config

    fprint(f"replacing prior of `{param}` with a delta function at {value}",
           verbose=verbose)
    priors = config.setdefault("model", {}).setdefault("priors", {})
    priors.pop(param, None)
    priors[param] = {
        "dist": "delta",
        "value": value
        }
    return config


def get_root_data(config):
    """Resolve the data root, defaulting to ``<root_main>/data``."""
    return config.get("root_data", join(config["root_main"], "data"))


def get_root_results(config):
    """Resolve the results root, defaulting to ``<root_main>/results``."""
    return config.get("root_results", join(config["root_main"], "results"))


_LOCAL_CONFIG_CACHE = None


def local_config():
    """Return the parsed local_config.toml for this repo (cached).

    Resolves to ``<repo>/local_config.toml`` relative to this file. Scripts
    that don't go through :func:`load_config` can use this + :func:`data_path`
    / :func:`results_path` to get the same root_data/root_results semantics.
    """
    global _LOCAL_CONFIG_CACHE
    if _LOCAL_CONFIG_CACHE is None:
        path = Path(__file__).resolve().parent.parent / "local_config.toml"
        with open(path, "rb") as f:
            _LOCAL_CONFIG_CACHE = tomllib.load(f)
    return _LOCAL_CONFIG_CACHE


def data_path(*parts):
    """Join `parts` under ``root_data`` from local_config.toml."""
    return join(get_root_data(local_config()), *parts)


def results_path(*parts):
    """Join `parts` under ``root_results`` from local_config.toml."""
    return join(get_root_results(local_config()), *parts)


def convert_to_absolute_paths(config):
    """Recursively convert relative paths in config to absolute paths."""
    root_data = get_root_data(config)
    root_results = get_root_results(config)

    path_keys_results = {
        "fname_output",
    }
    path_keys_data = {
        "root",
        "los_file",
        "los_file_random",
        "path_density",
        "path_velocity",
        "fpath_root",
    }

    def _recurse(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _recurse(v)
            elif isinstance(v, str):
                if k in path_keys_results and not isabs(v):
                    d[k] = abspath(join(root_results, v))
                elif k in path_keys_data and not isabs(v):
                    d[k] = abspath(join(root_data, v))

    _recurse(config)
    return config


def _selected_reconstruction_names(config):
    """Return reconstruction names selected by the loaded config."""
    names = set()
    for key_path in (
            "io/SH0ES/which_host_los",
            "io/which_host_los"):
        value = get_nested(config, key_path, None)
        if isinstance(value, str) and value.lower() != "none":
            names.add(value)

    for key_path in (
            "io/PV_main/EDD_TRGB/which_host_los",
            "io/PV_main/EDD_TRGB_grouped/which_host_los",
            "io/PV_main/EDD_2MTF/which_host_los"):
        value = get_nested(config, key_path, None)
        if isinstance(value, str) and value.lower() != "none":
            names.add(value)

    return names


def _validate_runtime_paths(config):
    """Catch machine-local reconstruction paths before expensive I/O."""
    if config.get("machine") != "arc":
        return

    bad_prefixes = (
        "/mnt/extraspace/",
        "/mnt/users/rstiskalek/",
    )
    recon_main = get_nested(config, "io/reconstruction_main", {})
    for name in _selected_reconstruction_names(config):
        section = recon_main.get(name, {})
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if isinstance(value, str) and value.startswith(bad_prefixes):
                raise ValueError(
                    f"ARC config selected `{name}` but "
                    f"`io.reconstruction_main.{name}.{key}` points to "
                    f"machine-local path `{value}`. Use a path relative to "
                    "`root_data` or an ARC-local absolute path."
                )


def _deep_merge(base, override):
    """Recursively merge `override` into `base`. Returns a new dict.

    Dicts containing a ``dist`` key (prior specifications) are treated as
    atomic values and replaced entirely rather than key-merged.
    """
    merged = base.copy()
    for k, v in override.items():
        if (k in merged and isinstance(merged[k], dict)
                and isinstance(v, dict) and "dist" not in v):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path, replace_none=True, fill_paths=True,
                replace_los_prior=True):
    """
    Load a TOML configuration file and convert "none" strings to None.

    Supports a ``base`` key (string or list of strings) pointing to base
    config files that are loaded first and deep-merged in order. Paths are
    resolved relative to the directory containing the config file.
    """
    config_dir = str(Path(config_path).resolve().parent)

    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    # Load and merge base configs if specified
    base_paths = config.pop("base", None)
    if base_paths is not None:
        if isinstance(base_paths, str):
            base_paths = [base_paths]
        merged = {}
        for bp in base_paths:
            if not isabs(bp):
                bp = join(config_dir, bp)
            with open(bp, 'rb') as f:
                merged = _deep_merge(merged, tomllib.load(f))
        config = _deep_merge(merged, config)

    # Inject defaults from local_config.toml (config values take precedence)
    project_root = Path(__file__).resolve().parent.parent
    local_config_path = project_root / "local_config.toml"
    if local_config_path.exists():
        with open(local_config_path, 'rb') as f:
            local_cfg = tomllib.load(f)
        for k, v in local_cfg.items():
            if k not in config:
                config[k] = v

    # Convert "none" strings to None
    if replace_none:
        config = convert_none_strings(config)

    # Assign delta priors if not using an underlying reconstruction.
    kind = config.get("pv_model", {}).get("kind", "")
    if replace_los_prior and not kind.startswith("precomputed_los"):
        config = replace_prior_with_delta(config, "alpha", 1.)
        config = replace_prior_with_delta(config, "beta", 0.)
        config = replace_prior_with_delta(config, "b1", 0.)
        config = replace_prior_with_delta(config, "b2", 0.)
        config = replace_prior_with_delta(config, "b3", 0.)
        config = replace_prior_with_delta(config, "delta_b1", 0.)

    # Convert relative paths to absolute paths
    if fill_paths:
        config = convert_to_absolute_paths(config)
        _validate_runtime_paths(config)

    shared_params = config.get("inference", {}).get("shared_params", None)
    if shared_params and str(shared_params).lower() != "none":
        config.setdefault("inference", {})["shared_params"] = (
            shared_params.split(","))

    return config


def get_nested(config, key_path, default=None):
    """Recursively access a nested value using a slash-separated key."""
    keys = key_path.split("/")
    current = config
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return default
        current = current[k]
    return current


###############################################################################
#                        Coordinate transformations                           #
###############################################################################

def radec_to_cartesian(ra, dec):
    """
    Convert right ascension and declination (in degrees) to unit Cartesian
    coordinates.
    """
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    cos_dec = np.cos(dec_rad)

    x = cos_dec * np.cos(ra_rad)
    y = cos_dec * np.sin(ra_rad)
    z = np.sin(dec_rad)

    return np.column_stack([x, y, z])


def cartesian_to_radec(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to right ascension and
    declination (RA, Dec), both in degrees.
    """
    d = (x**2 + y**2 + z**2)**0.5
    dec = np.arcsin(z / d)
    ra = np.arctan2(y, x)
    ra[ra < 0] += 2 * np.pi

    ra *= 180 / np.pi
    dec *= 180 / np.pi

    return d, ra, dec


def radec_to_galactic(ra, dec):
    """
    Convert right ascension and declination to galactic coordinates (all in
    degrees).
    """
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    return c.galactic.l.degree, c.galactic.b.degree


def galactic_to_radec(ell, b):
    """
    Convert galactic coordinates to right ascension and declination (all in
    degrees).
    """
    c = SkyCoord(l=ell*u.degree, b=b*u.degree, frame='galactic')
    return c.icrs.ra.degree, c.icrs.dec.degree


def galactic_to_radec_cartesian(ell, b):
    """
    Convert galactic coordinates (ell, b) in degrees to ICRS Cartesian unit
    vectors.
    """
    c = SkyCoord(l=np.atleast_1d(ell) * u.deg,
                 b=np.atleast_1d(b) * u.deg,
                 frame='galactic')
    icrs = c.icrs
    xyz = icrs.cartesian.xyz.value.T

    return xyz[0] if np.isscalar(ell) and np.isscalar(b) else xyz


def supergalactic_to_radec(sgl, sgb):
    """
    Convert supergalactic coordinates (sgl, sgb) to equatorial
    right ascension and declination (RA, Dec), all in degrees.
    """
    c = SkyCoord(sgl=sgl * u.deg, sgb=sgb * u.deg, frame="supergalactic")
    return c.icrs.ra.deg, c.icrs.dec.deg


def radec_to_supergalactic(ra, dec):
    """
    Convert right ascension and declination (in degrees) to supergalactic
    coordinates in degrees.
    """
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    return c.supergalactic.sgl.deg, c.supergalactic.sgb.deg


def radec_cartesian_to_galactic(x, y, z):
    """
    Convert ICRS Cartesian vectors (x, y, z) to Galactic coordinates (ell, b)
    in degrees, and return the vector magnitude.
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)

    r = np.sqrt(x**2 + y**2 + z**2)

    rep = CartesianRepresentation(x * u.one, y * u.one, z * u.one)
    c_icrs = SkyCoord(rep, frame='icrs')
    gal = c_icrs.galactic

    ell = gal.l.deg
    b = gal.b.deg

    if r.size == 1:
        return r[0], ell[0], b[0]
    return r, ell, b


def hms_to_degrees(hours, minutes=None, seconds=None):
    """Convert hours, minutes and seconds to degrees."""
    return hours * 15 + (minutes or 0) / 60 * 15 + (seconds or 0) / 3600 * 15


def dms_to_degrees(degrees, arcminutes=None, arcseconds=None):
    """Convert degrees, arcminutes and arcseconds to decimal degrees."""
    return degrees + (arcminutes or 0) / 60 + (arcseconds or 0) / 3600


def heliocentric_to_cmb(z_helio, RA, dec, e_z_helio=None):
    """
    Convert heliocentric redshift to CMB redshift using the Planck 2018 CMB
    dipole.
    """
    # CMB dipole Planck 2018 values
    vsun_mag = 369  # km/s
    RA_sun = 167.942
    dec_sun = -6.944

    theta_sun = np.pi / 2 - np.deg2rad(dec_sun)
    phi_sun = np.deg2rad(RA_sun)

    # Convert to theta/phi in radians
    theta = np.pi / 2 - np.deg2rad(dec)
    phi = np.deg2rad(RA)

    # Unit vector in the direction of each galaxy
    n = np.asarray([np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)]).T
    # CMB dipole unit vector
    vsun_normvect = np.asarray([np.sin(theta_sun) * np.cos(phi_sun),
                                np.sin(theta_sun) * np.sin(phi_sun),
                                np.cos(theta_sun)])

    # Project the CMB dipole onto the line of sight and normalize
    vsun_projected = vsun_mag * np.dot(n, vsun_normvect) / SPEED_OF_LIGHT

    zsun_tilde = np.sqrt((1 - vsun_projected) / (1 + vsun_projected))
    zcmb = (1 + z_helio) / zsun_tilde - 1

    # Optional linear error propagation
    if e_z_helio is not None:
        e_zcmb = np.abs(e_z_helio / zsun_tilde)
        return zcmb, e_zcmb

    return zcmb


###############################################################################
#              Reading from files and other minor utilities                   #
###############################################################################


def read_gof(fname, which, raise_error=True):
    """Read goodness-of-fit statistics from a file with samples."""
    if not exists(fname) and not raise_error:
        return np.nan

    convert = which.startswith("logZ_")
    key = which.replace("logZ_", "lnZ_") if convert else which

    with File(fname, "r") as f:
        try:
            stat = float(f[f"gof/{key}"][...])
        except KeyError as e:
            raise KeyError(
                f"`{key}` not found in the file. Available keys are: "
                f"{list(f['gof'].keys())}") from e

    return stat / np.log(10) if convert else stat


def read_samples(root, fname, keys=None):
    fname = join(root, fname)

    with File(fname, "r") as f:
        if keys is None:
            keys = list(f["samples"].keys())
        elif isinstance(keys, str):
            keys = [keys]

        samples = {key: f[f"samples/{key}"][...] for key in keys}

    if isinstance(keys, list) and len(keys) == 1:
        return samples[keys[0]]
    return samples


def get_dlog_density_stats(lpA, lpB):
    """
    Compute the mean and standard deviation of the difference in log density
    between two sets of log densities of shape `(num_samples, num_objects)`.
    """
    assert lpA.ndim == lpB.ndim == 2 and lpA.shape[-1] == lpB.shape[-1]

    mu_A = np.mean(lpA, axis=0)
    mu_B = np.mean(lpB, axis=0)

    var_A = np.var(lpA, axis=0, ddof=1)
    var_B = np.var(lpB, axis=0, ddof=1)

    mean_diff = mu_A - mu_B
    std_diff = np.sqrt(var_A + var_B)

    return mean_diff, std_diff
