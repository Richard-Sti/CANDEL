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
from os.path import abspath, basename, isabs, join
from pathlib import Path

import astropy.units as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord
from corner import corner
from getdist import MCSamples, plots
from h5py import File
from jax import vmap
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

SPEED_OF_LIGHT = 299_792.458  # km / s


def fprint(*args, verbose=True, **kwargs):
    """Prints a message with a timestamp prepended."""
    if verbose:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S%f")[:-6]
        print(f"{timestamp}", *args, **kwargs)


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


def replace_prior_with_delta(config, param, value):
    """Replace the prior of `param` with a delta distribution at `value`."""
    if param not in config.get("model", {}).get("priors", {}):
        return config

    fprint(f"replacing prior of `{param}` with a delta function.")
    priors = config.setdefault("model", {}).setdefault("priors", {})
    priors.pop(param, None)
    priors[param] = {
        "dist": "delta",
        "value": value
        }
    return config


def convert_to_absolute_paths(config):
    """Recursively convert relative paths in config to absolute paths."""
    root = config["root_main"]

    path_keys = {
        "fname_output",
        "los_file",
        "root",
        "path_density",
        "path_velocity",
    }

    def _recurse(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _recurse(v)
            elif k in path_keys and isinstance(v, str) and not isabs(v):
                d[k] = abspath(join(root, v))

    _recurse(config)
    return config


def load_config(config_path, replace_none=True, fill_paths=True,
                replace_los_prior=True):
    """
    Load a TOML configuration file and convert "none" strings to None.
    """
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    # Convert "none" strings to None
    if replace_none:
        config = convert_none_strings(config)

    # Assign delta priors if not using an underlying reconstruction.
    kind = config.get("pv_model", {}).get("kind", "")
    if replace_los_prior and not kind.startswith("precomputed_los"):
        config = replace_prior_with_delta(config, "alpha", 1.)
        config = replace_prior_with_delta(config, "beta", 0.)
        config = replace_prior_with_delta(config, "b1", 0.)

    # Convert relative paths to absolute paths
    if fill_paths:
        config = convert_to_absolute_paths(config)

    shared_params = config["inference"].get("shared_params", None)
    if shared_params is not None:
        config["inference"]["shared_params"] = shared_params.split(",")

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


###############################################################################
#                               Plotting                                      #
###############################################################################


def name2label(name):
    """
    Map internal parameter names to LaTeX labels, optionally including
    catalogue prefix.
    """
    latex_labels = {
        "a_TFR": r"$a_\mathrm{TFR}$",
        "b_TFR": r"$b_\mathrm{TFR}$",
        "c_TFR": r"$c_\mathrm{TFR}$",
        "sigma_mu": r"$\sigma_\mu$",
        "sigma_v": r"$\sigma_v$",
        "alpha": r"$\alpha$",
        "b1": r"$b_1$",
        "b2": r"$b_2$",
        "beta": r"$\beta$",
        "Vext_mag": r"$V_\mathrm{ext}$",
        "Vext_ell": r"$\ell_\mathrm{ext}$",
        "Vext_b": r"$b_\mathrm{ext}$",
        "h": r"$h$",
        "a": r"$a$",
        "m1": r"$m_1$",
        "m2": r"$m_2$",
        "a_TFR_dipole_mag": r"$a_\mathrm{TFR, dipole}$",
        "a_TFR_dipole_ell": r"$\ell_\mathrm{TFR, dipole}$",
        "a_TFR_dipole_b": r"$b_\mathrm{TFR, dipole}$",
        "M_dipole_mag": r"$M_\mathrm{dipole}$",
        "M_dipole_ell": r"$\ell_\mathrm{dipole}$",
        "M_dipole_b": r"$b_\mathrm{dipole}$",
        "eta_prior_mean": r"$\hat{\eta}$",
        "eta_prior_std": r"$w_\eta$",
        "A_CL": r"$A_{\rm CL}$",
        "B_CL": r"$B_{\rm CL}$",
        "C_CL": r"$C_{\rm CL}$",
        "a_FP": r"$a_{\rm FP}$",
        "b_FP": r"$b_{\rm FP}$",
        "c_FP": r"$c_{\rm FP}$",
        "R_dust": r"$R_{\rm W1}$",
    }

    if "/" in name:
        prefix, base = name.split("/", 1)
        base_label = latex_labels.get(base, base)
        prefix_latex = prefix.replace("_", r"\,").replace(" ", "~")
        return rf"$\mathrm{{{prefix_latex}}},\,{base_label.strip('$')}$"

    return latex_labels.get(name, name)


def name2labelgetdist(name):
    """
    Return a GetDist-compatible LaTeX label (no $...$) for a parameter,
    optionally prepending the catalogue name as plain text.

    Example:
        "CF4_W1/beta" → r"\mathrm{CF4~W1},\,\beta"
    """
    labels = {
        "a_TFR": r"a_\mathrm{TFR}",
        "b_TFR": r"b_\mathrm{TFR}",
        "c_TFR": r"c_\mathrm{TFR}",
        "sigma_mu": r"\sigma_\mu",
        "sigma_v": r"\sigma_v~\left[\mathrm{km}/\mathrm{s}\right]",
        "alpha": r"\alpha",
        "b1": r"b_1",
        "b2": r"b_2",
        "beta": r"\beta",
        "Vext_mag": r"V_\mathrm{ext}~\left[\mathrm{km}/\mathrm{s}\right]",
        "Vext_ell": r"\ell_\mathrm{ext}~\left[\mathrm{deg}\right]",
        "Vext_b":   r"b_\mathrm{ext}~\left[\mathrm{deg}\right]",
        "h": r"h",
        "a": r"a",
        "m1": r"m_1",
        "m2": r"m_2",
        "a_TFR_dipole_mag": r"a_\mathrm{TFR, dipole}",
        "a_TFR_dipole_ell": r"\ell_\mathrm{TFR, dipole}~\left[\mathrm{deg}\right]",  # noqa
        "a_TFR_dipole_b": r"b_\mathrm{TFR, dipole}~\left[\mathrm{deg}\right]",
        "M_dipole_mag": r"M_\mathrm{dipole}",
        "M_dipole_ell": r"\ell_\mathrm{dipole}~\left[\mathrm{deg}\right]",
        "M_dipole_b": r"b_\mathrm{dipole}~\left[\mathrm{deg}\right]",
        "eta_prior_mean": r"\hat{\eta}",
        "eta_prior_std": r"w_\eta",
        "A_CL": r"A_{\rm CL}",
        "B_CL": r"B_{\rm CL}",
        "C_CL": r"C_{\rm CL}",
        "a_FP": r"a_{\rm FP}",
        "b_FP": r"b_{\rm FP}",
        "c_FP": r"c_{\rm FP}",
        "R_dust": r"R_{\rm W1}",
        "mu_LMC": r"\mu_{\rm LMC}",
        "mu_M31": r"\mu_{\rm M31}",
        "mu_N4258": r"\mu_{\rm NGC4258}",
        "H0": r"H_0~\left[\mathrm{km}/\mathrm{s}/\mathrm{Mpc}\right]",
    }

    if "/" in name:
        prefix, base = name.split("/", 1)
        base_label = labels.get(base, base)
        prefix_latex = prefix.replace("_", r"\,").replace(" ", "~")
        return rf"\mathrm{{{prefix_latex}}},\,{base_label}"

    return labels.get(name, name)


def sort_params(keys):
    order = [
        "a_TFR", "b_TFR", "c_TFR",
        "alpha", "beta",
        "sigma_mu", "sigma_v",
        "Vext", "Vext_mag", "Vext_ell", "Vext_b"
    ]

    def sort_key(k):
        try:
            return (order.index(k), k)
        except ValueError:
            # Put unlisted keys at the end, alphabetically
            return (len(order), k)

    return sorted(keys, key=sort_key)


def plot_corner(samples, show_fig=True, filename=None, smooth=1, keys=None):
    """Plot a corner plot from posterior samples."""
    flat_samples = []
    labels = []

    for k, v in samples.items():
        if keys is not None and k not in keys:
            continue
        if v.ndim > 1:
            continue
        flat_samples.append(v.reshape(-1))
        labels.append(name2label(k))

    if not flat_samples:
        raise ValueError("No valid samples to plot.")

    data = np.vstack(flat_samples).T

    fig = corner(
        data,
        labels=labels,
        show_titles=True,
        smooth=smooth,
    )

    if filename is not None:
        fprint(f"saving a corner plot to {filename}")
        fig.savefig(filename, bbox_inches="tight")

    if show_fig:
        fig.show()
    else:
        plt.close(fig)


def plot_Vext_rad_corner(samples, show_fig=True, filename=None, smooth=1):
    """Plot a corner plot of Vext_rad_{mag, ell, b} samples."""
    keys = ["Vext_rad_mag", "Vext_rad_ell", "Vext_rad_b"]
    base_labels = [r"V", r"\ell", r"b"]

    arrays = []
    labels = []

    for key, base_label in zip(keys, base_labels):
        if key not in samples:
            raise ValueError(f"Missing key: {key}")

        arr = samples[key]

        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim != 2:
            raise ValueError(f"{key} must be 2D or 3D")

        ndim = arr.shape[1]
        arrays.append(arr)

        for i in range(ndim):
            labels.append(fr"${base_label}_{{{i}}}$")

    data = np.hstack(arrays)  # shape: (nsamples_total, total_dims)

    fig = corner(data, labels=labels, show_titles=True, smooth=smooth)

    if filename is not None:
        fprint(f"saving knots corner plot to {filename}")
        fig.savefig(filename, bbox_inches="tight")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def plot_corner_getdist(samples_list, labels=None, show_fig=True,
                        filename=None, keys=None, fontsize=None, filled=True,
                        points=None):
    """Plot a GetDist triangle plot for one or more posterior samples."""
    try:
        import scienceplots  # noqa
        use_scienceplots = True
    except ImportError:
        use_scienceplots = False

    if isinstance(samples_list, dict):
        samples_list = [samples_list]

    if labels is not None and len(labels) != len(samples_list):
        raise ValueError("Length of `labels` must match number of sample sets")

    # Build candidate key list (user-specified or inferred)
    if keys is not None:
        candidate_keys = keys
    else:
        candidate_keys = [
            k for k in samples_list[0] if samples_list[0][k].ndim == 1]

    # Include keys that are present and 1D in at least one sample set
    param_names = []
    for k in candidate_keys:
        for s in samples_list:
            if k in s and s[k].ndim == 1:
                param_names.append(k)
                break
            elif k in s and s[k].ndim > 1:
                fprint(f"[SKIP] {k} has shape {s[k].shape}")
                break

    if keys is None:
        param_names = sort_params(param_names)

    ranges = {}
    for k in param_names:
        if "mag" in k:
            ranges[k] = [0, None]

        if "ell" in k:
            ranges[k] = [0, 360]

        if "b" in k:
            ranges[k] = [-90, 90]

    gdsamples_list = []

    for samples in samples_list:
        present_params = []
        present_labels = []
        columns = []

        n_samples = len(next(iter(samples.values())))
        for k in param_names:
            if k in samples and samples[k].ndim == 1:
                col = samples[k].reshape(-1)
            else:
                col = np.full(n_samples, np.nan)
            if not np.all(np.isnan(col)):
                present_params.append(k)
                present_labels.append(name2labelgetdist(k))
                columns.append(col)

        data = np.vstack(columns).T
        gds = MCSamples(
            samples=data,
            names=present_params,
            labels=present_labels,
            ranges={k: ranges[k] for k in present_params if k in ranges},
            )
        gdsamples_list.append(gds)

    # Plot styling
    settings = plots.GetDistPlotSettings()
    if fontsize is not None:
        settings.lab_fontsize = fontsize
        settings.legend_fontsize = fontsize
        settings.axes_fontsize = fontsize - 1
        settings.title_limit_fontsize = fontsize - 1

    with plt.style.context("science" if use_scienceplots else "default"):
        g = plots.get_subplot_plotter(settings=settings)
        g.triangle_plot(
            gdsamples_list,
            params=param_names,
            filled=filled,
            legend_labels=labels,
            legend_loc="upper right",
        )

        if points is not None:
            plotted_pairs = set()
            for (x_param, y_param), (x_val, y_val) in points.items():
                if x_param not in param_names or y_param not in param_names:
                    continue
                ix = param_names.index(x_param)
                iy = param_names.index(y_param)
                if iy > ix and (ix, iy) not in plotted_pairs:
                    ax = g.subplots[iy, ix]
                    ax.plot(x_val, y_val, "x", color="red", markersize=10)
                    __, labels_ = ax.get_legend_handles_labels()
                    if "Reference" not in labels_:
                        ax.legend()
                    plotted_pairs.add((ix, iy))

        if filename is not None:
            fprint(f"[INFO] Saving GetDist triangle plot to: {filename}")
            g.export(filename, dpi=450)

        if show_fig:
            plt.show()
        else:
            plt.close()


def plot_corner_from_hdf5(fnames, keys=None, labels=None, fontsize=None,
                          filled=True, show_fig=True, filename=None,
                          points=None):
    """
    Plot a triangle plot from one or more HDF5 files containing posterior
    samples.
    """
    if isinstance(fnames, (str, Path)):
        fnames = [fnames]

    samples_list = []
    for fname in fnames:
        with File(fname, 'r') as f:
            grp = f["samples"]
            samples = {key: grp[key][...] for key in grp.keys()}
            samples_list.append(samples)

            full_keys = list(grp.keys())
            print(f"{basename(fname)}: {', '.join(full_keys)}")

    plot_corner_getdist(
        samples_list,
        labels=labels,
        keys=keys,
        fontsize=fontsize,
        filled=filled,
        show_fig=show_fig,
        filename=filename,
        points=points,
    )


###############################################################################
#                     Radial dependence of Vext                               #
###############################################################################


def interpolate_scalar_field(V, r, rbins, k=3, endpoints="not-a-knot"):
    V = jnp.asarray(V).reshape(-1, rbins.size)

    def spline_eval(y):
        spline = InterpolatedUnivariateSpline(
            rbins, y, k=k, endpoints=endpoints)
        return spline(r)

    return vmap(spline_eval)(V)


def interpolate_latitude_field(b_deg, r, rbins, k=3, endpoints="not-a-knot"):
    b_rad = jnp.deg2rad(b_deg).reshape(-1, rbins.size)
    sin_b = jnp.sin(b_rad)

    def spline_eval(y):
        spline = InterpolatedUnivariateSpline(
            rbins, y, k=k, endpoints=endpoints)
        return spline(r)

    sin_b_interp = vmap(spline_eval)(sin_b)
    return jnp.rad2deg(jnp.arcsin(jnp.clip(sin_b_interp, -1.0, 1.0)))


def interpolate_longitude_field(l_deg, r, rbins, k=3, endpoints="not-a-knot"):
    l_rad = jnp.deg2rad(l_deg).reshape(-1, rbins.size)
    sin_l = jnp.sin(l_rad)
    cos_l = jnp.cos(l_rad)

    def spline_eval(y):
        spline = InterpolatedUnivariateSpline(
            rbins, y, k=k, endpoints=endpoints)
        return spline(r)
    sin_l_interp = vmap(spline_eval)(sin_l)
    cos_l_interp = vmap(spline_eval)(cos_l)
    return jnp.rad2deg(jnp.arctan2(sin_l_interp, cos_l_interp)) % 360


def interpolate_all_radial_fields(model, Vmag, ell, b, r_eval_size=1000):
    rknot = jnp.asarray(model.kwargs_radial_Vext["rknot"])
    rmin, rmax = 0, jnp.max(rknot)
    k = model.kwargs_radial_Vext.get("k", 3)
    endpoints = model.kwargs_radial_Vext.get("endpoints", "not-a-knot")

    r = jnp.linspace(rmin, rmax, r_eval_size)

    Vmag_interp = interpolate_scalar_field(Vmag, r, rknot, k, endpoints)
    ell_interp = interpolate_longitude_field(ell, r, rknot, k, endpoints)
    b_interp = interpolate_latitude_field(b, r, rknot, k, endpoints)

    return r, Vmag_interp, ell_interp, b_interp


def plot_radial_profiles(samples, model, r_eval_size=1000, show_fig=True,
                         filename=None):
    """
    Plot the radial profiles of Vext_rad_{mag, ell, b} from the samples,
    including 1sigma and 2sigma percentile bands.
    """
    Vmag = samples["Vext_rad_mag"]
    ell = samples["Vext_rad_ell"]
    b = samples["Vext_rad_b"]

    r, V_interp, ell_interp, b_interp = interpolate_all_radial_fields(
        model, Vmag, ell, b, r_eval_size=r_eval_size
    )

    def get_percentiles(arr):
        arr = np.array(arr)
        p16, p50, p84 = np.percentile(arr, [16, 50, 84], axis=0)
        p025, p975 = np.percentile(arr, [2.5, 97.5], axis=0)
        return p025, p16, p50, p84, p975

    V025, V16, V50, V84, V975 = get_percentiles(V_interp)
    l025, l16, l50, l84, l975 = get_percentiles(ell_interp)
    b025, b16, b50, b84, b975 = get_percentiles(b_interp)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    c = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    components = [
        (V025, V16, V50, V84, V975, r"$V_{\rm dipole}~[\mathrm{km}/\mathrm{s}]$"),  # noqa
        (l025, l16, l50, l84, l975, r"$\ell_{\rm dipole}~[\mathrm{deg}]$"),
        (b025, b16, b50, b84, b975, r"$b_{\rm dipole}~[\mathrm{deg}]$"),
    ]

    for i, (lo2, lo1, med, hi1, hi2, ylabel) in enumerate(components):
        ax = axes[i]
        ax.fill_between(r, lo2, hi2, alpha=0.2, color=c)
        ax.fill_between(r, lo1, hi1, alpha=0.4, color=c)
        ax.plot(r, med, c=c)
        ax.set_xlabel(r"$r~[\mathrm{Mpc}/h]$")
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    if filename is not None:
        fprint(f"saving a radial profile plot to {filename}")
        fig.savefig(filename, bbox_inches="tight")

    if show_fig:
        fig.show()
    else:
        plt.close(fig)


def read_gof(fname, which):
    """Read goodness-of-fit statistics from a file with samples."""
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
