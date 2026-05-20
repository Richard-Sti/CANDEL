# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Plotting helpers for radial external-velocity diagnostics."""

from os.path import exists

import healpy as hp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from interpax import interp1d
from jax import vmap
from matplotlib.ticker import FuncFormatter
from scipy.integrate import cumulative_simpson

from candel.util import (data_path, fprint, galactic_to_radec_cartesian,
                         radec_cartesian_to_galactic)

###############################################################################
#                     Radial dependence of Vext                               #
###############################################################################


def _plot_knot_lines(ax, rknot, xmin, xmax):
    """Draw dashed vertical lines at knot positions."""
    dx = 0.01 * (xmax - xmin)
    kwargs = dict(color="black", linestyle="--", zorder=-1, alpha=0.5)
    for rk in rknot:
        if jnp.isclose(rk, xmin):
            ax.axvline(xmin + dx, **kwargs)
        elif jnp.isclose(rk, xmax):
            ax.axvline(xmax - dx, **kwargs)
        else:
            ax.axvline(rk, **kwargs)


def _interp1d_const(r, rbins, y, method="cubic"):
    """1D interpolation with constant extrapolation at boundaries."""
    r = jnp.asarray(r)
    rbins = jnp.asarray(rbins)
    x0, x1 = rbins[0], rbins[-1]
    r_clipped = jnp.clip(r, x0, x1)
    vals = interp1d(r_clipped, rbins, y, method=method)
    vals = jnp.where(r < x0, y[0], vals)
    vals = jnp.where(r > x1, y[-1], vals)
    return vals


def interpolate_scalar_field(V, r, rbins, method="cubic"):
    """Interpolate scalar field along last axis with constant extrapolation."""
    V = jnp.asarray(V)
    r = jnp.asarray(r)
    rbins = jnp.asarray(rbins)

    y2d = V.reshape(-1, rbins.size)

    def interp_row(y):
        return _interp1d_const(r, rbins, y, method)

    out = vmap(interp_row)(y2d)
    return out.reshape(*V.shape[:-1], r.shape[0])


def interpolate_cartesian_vector_field(V, r, rbins, method="cubic"):
    """Interpolate Cartesian vector components along the radial knot axis."""
    V = jnp.asarray(V)
    comps = [
        interpolate_scalar_field(V[..., i], r, rbins, method)
        for i in range(3)
    ]
    return jnp.stack(comps, axis=-1)


def interpolate_latitude_field(b_deg, r, rbins, method="cubic"):
    """Interpolate latitude via sin(b); return degrees."""
    b_rad = jnp.deg2rad(jnp.asarray(b_deg)).reshape(-1, rbins.size)
    sin_b = jnp.sin(b_rad)

    def interp_row(y):
        return _interp1d_const(r, rbins, y, method)

    sin_b_interp = vmap(interp_row)(sin_b)
    sin_b_interp = jnp.clip(sin_b_interp, -1.0, 1.0)
    return jnp.rad2deg(jnp.arcsin(sin_b_interp))


def interpolate_longitude_field(l_deg, r, rbins, method="cubic"):
    """Interpolate longitude via sin/cos; return degrees in [0, 360)."""
    l_rad = jnp.deg2rad(jnp.asarray(l_deg)).reshape(-1, rbins.size)
    sin_l = jnp.sin(l_rad)
    cos_l = jnp.cos(l_rad)

    def interp_row(y):
        return _interp1d_const(r, rbins, y, method)

    sin_l_i = vmap(interp_row)(sin_l)
    cos_l_i = vmap(interp_row)(cos_l)

    # renormalise to unit circle to avoid drift
    s = jnp.sqrt(jnp.clip(sin_l_i**2 + cos_l_i**2, 1e-20, None))
    sin_l_i = sin_l_i / s
    cos_l_i = cos_l_i / s

    return jnp.rad2deg(jnp.arctan2(sin_l_i, cos_l_i)) % 360.0


def radial_knots_to_cartesian(Vmag, ell, b):
    """Convert sampled Galactic radial Vext knots to ICRS Cartesian vectors."""
    Vmag = np.asarray(Vmag)
    ell = np.asarray(ell)
    b = np.asarray(b)
    if Vmag.shape != ell.shape or Vmag.shape != b.shape:
        raise ValueError(
            "`Vmag`, `ell`, and `b` must have matching shapes.")

    unit = galactic_to_radec_cartesian(ell.ravel(), b.ravel())
    unit = unit.reshape(*Vmag.shape, 3)
    return Vmag[..., None] * unit


def cartesian_vectors_to_galactic_profiles(V):
    """Return magnitude and direction for Cartesian vector profiles."""
    V = np.asarray(V)
    shape = V.shape[:-1]
    flat = V.reshape(-1, 3)
    mag = np.linalg.norm(flat, axis=1)
    safe = np.empty_like(flat)
    nonzero = mag > 0.0
    safe[nonzero] = flat[nonzero] / mag[nonzero, None]
    safe[~nonzero] = [1.0, 0.0, 0.0]
    _, ell, b = radec_cartesian_to_galactic(
        safe[:, 0], safe[:, 1], safe[:, 2])
    ell = np.where(nonzero, ell, np.nan)
    b = np.where(nonzero, b, np.nan)
    return mag.reshape(shape), ell.reshape(shape), b.reshape(shape)


def interpolate_all_radial_fields(model, Vmag, ell, b, r_eval_size=1000,
                                  method=None):
    rknot = jnp.asarray(model.kwargs_Vext["rknot"])
    if method is None:
        method = model.kwargs_Vext["method"]
    rmin, rmax = 0.0, jnp.max(rknot)

    r = jnp.linspace(rmin, rmax, r_eval_size)
    V_knot = radial_knots_to_cartesian(Vmag, ell, b)
    V_interp = interpolate_cartesian_vector_field(V_knot, r, rknot, method)
    Vmag_interp, ell_interp, b_interp = cartesian_vectors_to_galactic_profiles(
        V_interp)

    return r, Vmag_interp, ell_interp, b_interp


def _radial_vext_cartesian_profile(samples, model, r):
    rknot = jnp.asarray(model.kwargs_Vext["rknot"])
    method = model.kwargs_Vext["method"]

    if model.which_Vext == "radial":
        Vmag = samples["Vext_rad_mag"]
        ell = samples["Vext_rad_ell"]
        b = samples["Vext_rad_b"]
        V_knot = radial_knots_to_cartesian(Vmag, ell, b)
        return np.asarray(
            interpolate_cartesian_vector_field(V_knot, r, rknot, method))

    if model.which_Vext == "radial_magnitude":
        Vmag = interpolate_scalar_field(
            samples["Vext_radmag_mag"], r, rknot, method)
        Vmag = np.clip(np.asarray(Vmag), 0.0, None)
        rhat = galactic_to_radec_cartesian(
            samples["Vext_radmag_ell"], samples["Vext_radmag_b"])
        return Vmag[..., None] * rhat[:, None, :]

    raise ValueError(
        "`plot_Vext_radial_bulkflow` requires a radial Vext model.")


def _enclosed_average_vectors(r, V):
    r = np.asarray(r)
    V = np.asarray(V)
    integrand = r[None, :, None]**2 * V
    integral = cumulative_simpson(integrand, x=r, axis=1, initial=0)

    out = np.empty_like(V)
    out[:, 0, :] = V[:, 0, :]
    np.divide(
        3.0 * integral[:, 1:, :],
        r[None, 1:, None]**3,
        out=out[:, 1:, :],
        where=r[None, 1:, None] > 0.0)
    return out


def _load_reconstruction_bulkflow(model, r):
    kind = getattr(model, "kind", "")
    prefix = "precomputed_los_"
    if not isinstance(kind, str) or not kind.startswith(prefix):
        return None

    reconstruction_name = kind[len(prefix):]
    rel = ("fields", "field_shells",
           f"enclosed_mass_{reconstruction_name}.npz")
    candidates = [data_path(*rel), data_path("data", *rel)]
    fname = next((p for p in candidates if exists(p)), None)
    if fname is None:
        fprint(
            "[WARN] skipping reconstructed bulk-flow curve; "
            f"`{candidates[0]}` is missing.")
        return None

    with np.load(fname) as f:
        r_shell = f["distances"]
        B_shell = f["cumulative_velocity"]

    B = np.empty((B_shell.shape[0], len(r), 3), dtype=float)
    for i in range(B_shell.shape[0]):
        for j in range(3):
            B[i, :, j] = np.interp(
                r, r_shell, B_shell[i, :, j],
                left=B_shell[i, 0, j], right=B_shell[i, -1, j])
    return B


def _bulkflow_reference_band(rmax):
    rel = ("fields", "field_shells", "BulkFlowPlot.npy")
    candidates = [data_path(*rel), data_path("data", *rel)]
    fname = next((p for p in candidates if exists(p)), None)
    if fname is None:
        return None

    Rs, _mean, _std, _mode, _p05, p16, p84, _p95 = np.load(fname)
    mask = Rs <= rmax
    return Rs[mask], p16[mask], p84[mask]


def plot_Vext_radial_bulkflow(samples, model, r_eval_size=500, Rmax=125.0,
                              show_fig=True, filename=None):
    """Plot radial Vext magnitude and enclosed bulk-flow magnitude."""
    rknot = np.asarray(model.kwargs_Vext["rknot"])
    rmax = max(float(np.max(rknot)), float(Rmax))
    r = np.linspace(0.0, rmax, r_eval_size)

    V = _radial_vext_cartesian_profile(samples, model, r)
    Vmag = np.linalg.norm(V, axis=-1)
    Vbulk = _enclosed_average_vectors(r, V)

    B_recon = _load_reconstruction_bulkflow(model, r)
    if B_recon is None:
        B = Vbulk
    else:
        beta = np.asarray(samples.get("beta", np.ones(V.shape[0])))
        B = (
            beta[None, :, None, None] * B_recon[:, None, :, :]
            + Vbulk[None, :, :, :])
        B = B.reshape(-1, len(r), 3)
    Bmag = np.linalg.norm(B, axis=-1)

    def band(x):
        return np.percentile(x, [16, 50, 84], axis=0)

    V16, V50, V84 = band(Vmag)
    B16, B50, B84 = band(Bmag)

    fig, axes = plt.subplots(2, 1, figsize=(6.0, 6.0), sharex=True)
    fig.subplots_adjust(hspace=0.08)
    c_vext, c_bulk = plt.rcParams["axes.prop_cycle"].by_key()["color"][:2]

    axes[0].fill_between(r, V16, V84, alpha=0.35, color=c_vext)
    axes[0].plot(r, V50, color=c_vext)
    axes[0].set_ylabel(r"$|{\bf V}_{\rm ext}(r)|~[\mathrm{km/s}]$")
    axes[0].set_ylim(0, None)

    ref = _bulkflow_reference_band(rmax)
    if ref is not None:
        Rs, lo, hi = ref
        axes[1].fill_between(
            Rs, lo, hi, color="0.5", alpha=0.25,
            label=r"$\Lambda\mathrm{CDM}$")
    axes[1].fill_between(
        r, B16, B84, alpha=0.35, color=c_bulk,
        label="model")
    axes[1].plot(r, B50, color=c_bulk)
    axes[1].set_ylabel(r"$|{\bf B}(<R)|~[\mathrm{km/s}]$")
    axes[1].set_xlabel(r"$R~[h^{-1}\,\mathrm{Mpc}]$")
    axes[1].set_ylim(0, None)
    if ref is not None:
        axes[1].legend(frameon=False)

    for ax in axes:
        ax.set_xlim(0.0, rmax)
        _plot_knot_lines(ax, rknot, 0.0, rmax)

    if filename is not None:
        fprint(f"saving a radial Vext/bulk-flow plot to {filename}")
        fig.savefig(filename, bbox_inches="tight", dpi=450)

    if show_fig:
        fig.show()
    else:
        plt.close(fig)


def get_percentiles_circ_along_r(lon_deg_samples, qs=(2.5, 16, 50, 84, 97.5),
                                 rmin_conc=0.2, return_unwrapped=True):
    """
    Robust circular percentiles along radius with phase-tracked reference.
    """
    th = np.deg2rad(np.asarray(lon_deg_samples))  # (Ns, R)

    # Raw circular mean and concentration R \in [0,1]
    sinm = np.sin(th).mean(axis=0)
    cosm = np.cos(th).mean(axis=0)
    mu_raw = np.arctan2(sinm, cosm)              # (-pi, pi]
    Rconc = np.hypot(sinm, cosm)                 # mean resultant length

    # Phase-tracked reference: make mu continuous by following the
    # nearest branch
    ref = np.empty_like(mu_raw)
    ref[0] = mu_raw[0]
    two_pi = 2.0 * np.pi
    for j in range(1, mu_raw.size):
        mu = mu_raw[j]
        # choose branch closest to previous ref
        mu += two_pi * np.round((ref[j-1] - mu) / two_pi)
        # damp sudden jumps when concentration is low
        if Rconc[j] < rmin_conc:
            # blend toward previous ref (keeps continuity when mean is noisy)
            mu = 0.7 * ref[j-1] + 0.3 * mu
        ref[j] = mu

    # Align each sample at each radius to nearest branch of the reference
    shifts = two_pi * np.round((ref[None, :] - th) / two_pi)   # (Ns, R)
    th_aligned = th + shifts

    # Percentiles across samples at each r
    q_tracks = np.percentile(th_aligned, qs, axis=0)           # (Q, R)

    if return_unwrapped:
        return tuple(np.rad2deg(q_tracks[i]) for i in range(len(qs)))

    # Wrap back to [0, 360) using a single global shift (avoids seam wandering)
    median = np.rad2deg(q_tracks[qs.index(50) if 50 in qs else len(qs)//2])
    shift = 360.0 * np.round(median.mean() / 360.0)
    return tuple(
        (np.rad2deg(q_tracks[i]) - shift) % 360.0 for i in range(len(qs)))


def deg_wrap_360(y, pos):
    return f"{(y % 360 + 360) % 360:.0f}"


def plot_radial_profiles(samples, model, r_eval_size=1000, show_fig=True,
                         filename=None):
    """
    Plot the radial profiles of Vext_rad_{mag, ell, b} from the samples,
    including 1sigma and 2sigma percentile bands.
    """
    Vmag = samples["Vext_rad_mag"]
    ell = samples["Vext_rad_ell"]
    b = samples["Vext_rad_b"]

    rknot = model.kwargs_Vext["rknot"]

    r, V_interp, ell_interp, b_interp = interpolate_all_radial_fields(
        model, Vmag, ell, b, r_eval_size=r_eval_size
    )

    def get_percentiles(arr):
        arr = np.array(arr)
        p16, p50, p84 = np.percentile(arr, [16, 50, 84], axis=0)
        p025, p975 = np.percentile(arr, [2.5, 97.5], axis=0)
        return p025, p16, p50, p84, p975

    def get_percentiles_circ(arr_deg):
        arr_rad = np.deg2rad(arr_deg)
        # unwrap along axis 0 (samples)
        arr_unwrapped = np.unwrap(arr_rad, axis=0)
        p16, p50, p84 = np.percentile(arr_unwrapped, [16, 50, 84], axis=0)
        p025, p975 = np.percentile(arr_unwrapped, [2.5, 97.5], axis=0)
        # back to [0, 360)
        return np.rad2deg([p025, p16, p50, p84, p975]) % 360

    V025, V16, V50, V84, V975 = get_percentiles(V_interp)
    l025, l16, l50, l84, l975 = get_percentiles_circ_along_r(ell_interp)
    b025, b16, b50, b84, b975 = get_percentiles(b_interp)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    c = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    components = [
        (V025, V16, V50, V84, V975, r"$V_{\rm dipole}~[\mathrm{km}/\mathrm{s}]$"),  # noqa
        (l025, l16, l50, l84, l975, r"$\ell_{\rm dipole}~[\mathrm{deg}]$"),
        (b025, b16, b50, b84, b975, r"$b_{\rm dipole}~[\mathrm{deg}]$"),
    ]

    xmin, xmax = r[0], r[-1]
    for i, (lo2, lo1, med, hi1, hi2, ylabel) in enumerate(components):
        ax = axes[i]
        ax.fill_between(r, lo2, hi2, alpha=0.2, color=c)
        ax.fill_between(r, lo1, hi1, alpha=0.4, color=c)
        ax.plot(r, med, c=c)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(r"$r~[\mathrm{Mpc}/h]$")
        ax.set_ylabel(ylabel)
        _plot_knot_lines(ax, rknot, xmin, xmax)

    axes[0].set_ylim(0, None)
    axes[1].yaxis.set_major_formatter(FuncFormatter(deg_wrap_360))

    fig.tight_layout()
    if filename is not None:
        fprint(f"saving a radial profile plot to {filename}")
        fig.savefig(filename, bbox_inches="tight", dpi=450)

    if show_fig:
        fig.show()
    else:
        plt.close(fig)


def _add_equator_labels(lon_step=60, lat_step=30):
    for lon in np.arange(0, 360, lon_step):
        hp.projtext(lon, -2.0, rf"${lon:d}^\circ$", lonlat=True,
                    fontsize=9, ha="center", va="top")
    for lat in np.arange(-60, 61, lat_step):
        if lat == 0:
            continue
        hp.projtext(-178.0, lat, rf"${lat:+d}^\circ$", lonlat=True,
                    fontsize=9, ha="right", va="center")


def _upsample_map(map_lo, nside_plot, *, nest=False):
    nside_lo = hp.npix2nside(map_lo.size)
    if nside_plot is None or nside_plot <= nside_lo:
        return map_lo
    pix_hi = np.arange(hp.nside2npix(nside_plot))
    th, ph = hp.pix2ang(nside_plot, pix_hi, nest=nest)
    # Bilinear interpolation from the coarse map:
    map_hi = hp.get_interp_val(map_lo, th, ph, nest=nest, lonlat=False)
    return map_hi


def plot_Vext_radmag(samples, model, r_eval_size=1000, show_fig=True,
                     filename=None):
    Vmag = samples["Vext_radmag_mag"]
    rknot = model.kwargs_Vext["rknot"]
    method = model.kwargs_Vext["method"]

    r = jnp.linspace(0.0, np.max(rknot), r_eval_size)
    V = vmap(lambda y: interp1d(r, rknot, y, method=method))(Vmag)
    V = jnp.clip(V, 0, None)
    Vlow, Vmed, Vhigh = np.percentile(V, [16, 50, 84], axis=0)

    fig, ax = plt.subplots()

    ax.fill_between(r, Vlow, Vhigh, alpha=0.4)
    ax.plot(r, Vmed, color="C0")
    ax.set_xlabel(r"$r~[h^{-1}\,\mathrm{Mpc}]$")
    ax.set_ylabel(r"$V_{\mathrm{ext}}~[\mathrm{km/s}]$")
    ax.set_xlim(r[0], r[-1])
    ax.set_ylim(0, None)
    _plot_knot_lines(ax, rknot, r[0], r[-1])

    fig.tight_layout()

    if filename is not None:
        fprint(f"saving a radial Vext_mag plot to {filename}")
        fig.savefig(filename, bbox_inches="tight", dpi=450)

    if show_fig:
        fig.show()
    else:
        plt.close(fig)


def plot_Vext_moll(samples_pix, fname_out, coord_in="C", coord_out="G",
                   lon_step=60, lat_step=30, eps=1e-12, nside_plot=None,
                   remove_coord_label=True):
    """
    Plot three stacked Mollweide maps from MCMC samples (Nsamples, Npix):
      row 1: mean
      row 2: std
      row 3: mean/std
    If nside_plot > nside(map), upsample via healpy bilinear interpolation.
    """
    mean_map = np.nanmean(samples_pix, axis=0)
    std_map = np.nanstd(samples_pix, axis=0, ddof=0)
    snr_map = mean_map / (std_map + eps)

    if nside_plot is None:
        nside_plot = 4 * hp.npix2nside(mean_map.size)

    # Upsample (optional)
    mean_map = _upsample_map(mean_map, nside_plot)
    std_map = _upsample_map(std_map, nside_plot)
    snr_map = _upsample_map(snr_map, nside_plot)

    coord_arg = coord_out if coord_in == coord_out else [coord_in, coord_out]

    def _mollpanel(map_data, unit_label, sub):
        hp.mollview(map_data, nest=False, coord=coord_arg, notext=False,
                    xsize=2000, cbar=True, unit=unit_label, title="", sub=sub)
        hp.graticule(dpar=lat_step, dmer=lon_step)
        if remove_coord_label:
            ax = plt.gca()
            for t in ax.texts:
                if "Galactic" in t.get_text() or "Equatorial" in t.get_text():
                    t.set_visible(False)
        _add_equator_labels(lon_step, lat_step)

    plt.figure(figsize=(7, 10))
    _mollpanel(mean_map,
               r"Mean $V_{\mathrm{ext}}$ [$\mathrm{km\ s^{-1}}$]", 311)
    _mollpanel(std_map,
               r"Std($V_{\mathrm{ext}}$) [$\mathrm{km\ s^{-1}}$]", 312)
    _mollpanel(snr_map,
               r"$V_{\mathrm{ext}}$/Std($V_{\mathrm{ext}}$)", 313)
    plt.subplots_adjust(hspace=0.35)

    plt.savefig(fname_out, dpi=450, bbox_inches="tight")
    fprint(f"saving a Mollweide map to {fname_out}")
    plt.close()
