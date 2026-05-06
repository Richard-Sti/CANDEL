#!/usr/bin/env python
"""Diagnostic plots for the Gaussian missing-mass PV component."""
from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from candel.model.pv_utils import (  # noqa: E402
    convert_cartesian_frame,
    gaussian_missing_mass_delta,
    gaussian_missing_mass_velocity,
    missing_mass_los_delta_velocity,
    spherical_rhat,
)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _tangent_basis(rhat):
    pole = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(rhat, pole))) > 0.9:
        pole = np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(pole, rhat))
    e2 = _unit(np.cross(rhat, e1))
    return e1, e2


def _offset_rhat(center, e1, e2, dx_deg, dy_deg):
    dx = np.deg2rad(np.asarray(dx_deg, dtype=np.float64))
    dy = np.deg2rad(np.asarray(dy_deg, dtype=np.float64))
    theta = np.sqrt(dx * dx + dy * dy)
    tangent = dx[..., None] * e1 + dy[..., None] * e2
    tangent = np.divide(
        tangent, theta[..., None],
        out=np.zeros_like(tangent),
        where=theta[..., None] > 0,
    )
    rhat = np.cos(theta)[..., None] * center + np.sin(theta)[..., None] * tangent
    return rhat / np.linalg.norm(rhat, axis=-1, keepdims=True)


def _trapz(y, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def _plot_profiles(args, outdir, cluster_rhat_icrs):
    mass = 10.0**args.logM
    sigma = args.sigma
    s = np.linspace(0.0, args.profile_sigma_max * sigma, args.profile_n)
    pos = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
    los = np.tile(np.array([[1.0, 0.0, 0.0]]), (s.size, 1))
    delta = np.asarray(gaussian_missing_mass_delta(
        jnp.asarray(pos), jnp.zeros(3), mass, sigma, args.Om))
    v_rad = np.asarray(gaussian_missing_mass_velocity(
        jnp.asarray(pos), jnp.asarray(los), jnp.zeros(3),
        mass, sigma, args.Om, growth_index=args.growth_index))

    e1, _ = _tangent_basis(cluster_rhat_icrs)
    theta_sigma = np.rad2deg(np.arcsin(min(0.999, sigma / args.distance)))
    offsets = np.array([0.0, 0.5, 1.0, 2.0, 4.0]) * theta_sigma
    r_min = max(0.01, args.distance - args.los_sigma_half_width * sigma)
    r_max = args.distance + args.los_sigma_half_width * sigma
    r_grid = np.linspace(r_min, r_max, args.los_n)
    rhat = np.stack([
        _offset_rhat(cluster_rhat_icrs, e1, np.cross(cluster_rhat_icrs, e1),
                     theta, 0.0)
        for theta in offsets
    ])
    delta_los, v_los = missing_mass_los_delta_velocity(
        jnp.asarray(r_grid), jnp.asarray(rhat), args.distance,
        jnp.asarray(cluster_rhat_icrs), mass, sigma, args.Om,
        growth_index=args.growth_index)
    delta_los = np.asarray(delta_los)
    v_los = np.asarray(v_los)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(s / sigma, delta, color="tab:blue")
    ax.set_yscale("log")
    ax.set_xlabel(r"separation / $\sigma_{\rm miss}$")
    ax.set_ylabel(r"$\delta_{\rm miss}$")
    ax.set_title("Spherical density profile")

    ax = axes[0, 1]
    ax.plot(s / sigma, v_rad, color="tab:red")
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_xlabel(r"separation / $\sigma_{\rm miss}$")
    ax.set_ylabel(r"$v_{\rm radial}$ [km s$^{-1}$]")
    ax.set_title("Spherical infall velocity")

    ax = axes[1, 0]
    for i, theta in enumerate(offsets):
        impact = args.distance * np.sin(np.deg2rad(theta))
        ax.plot(r_grid, delta_los[i], label=f"{theta:.2f} deg ({impact:.1f} Mpc/h)")
    ax.set_yscale("log")
    ax.axvline(args.distance, color="0.3", lw=0.8, ls="--")
    ax.set_xlabel(r"LOS distance $r$ [Mpc/h]")
    ax.set_ylabel(r"$\delta_{\rm miss}(r)$")
    ax.set_title("LOS density profiles")
    ax.legend(title="offset", fontsize=8)

    ax = axes[1, 1]
    for i, theta in enumerate(offsets):
        ax.plot(r_grid, v_los[i], label=f"{theta:.2f} deg")
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axvline(args.distance, color="0.3", lw=0.8, ls="--")
    ax.set_xlabel(r"LOS distance $r$ [Mpc/h]")
    ax.set_ylabel(r"$v_{\rm los}(r)$ [km s$^{-1}$]")
    ax.set_title("LOS velocity projections")

    fig.suptitle(
        f"Gaussian Mmiss: log10(M/[Msun/h])={args.logM:g}, "
        f"r={args.distance:g} Mpc/h, sigma={sigma:g} Mpc/h",
        fontsize=11,
    )
    path = outdir / "missing_mass_profiles.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    return path


def _plot_projection(args, outdir, cluster_rhat_icrs):
    mass = 10.0**args.logM
    sigma = args.sigma
    e1, e2 = _tangent_basis(cluster_rhat_icrs)
    extent = args.map_extent_deg
    if extent is None:
        extent = np.rad2deg(np.arcsin(min(0.999, args.map_sigma_extent * sigma
                                          / args.distance)))

    offsets = np.linspace(-extent, extent, args.map_n)
    xx, yy = np.meshgrid(offsets, offsets)
    rhat = _offset_rhat(cluster_rhat_icrs, e1, e2, xx, yy).reshape(-1, 3)
    r_min = max(0.01, args.distance - args.los_sigma_half_width * sigma)
    r_max = args.distance + args.los_sigma_half_width * sigma
    r_grid = np.linspace(r_min, r_max, args.los_n)

    delta, v_los = missing_mass_los_delta_velocity(
        jnp.asarray(r_grid), jnp.asarray(rhat), args.distance,
        jnp.asarray(cluster_rhat_icrs), mass, sigma, args.Om,
        growth_index=args.growth_index)
    delta = np.asarray(delta)
    v_los = np.asarray(v_los)

    column = _trapz(delta, r_grid, axis=1)
    front = r_grid < args.distance
    back = r_grid >= args.distance
    column_front = _trapz(delta[:, front], r_grid[front], axis=1)
    column_back = _trapz(delta[:, back], r_grid[back], axis=1)
    v_front = np.divide(
        _trapz(delta[:, front] * v_los[:, front], r_grid[front], axis=1),
        column_front,
        out=np.full_like(column_front, np.nan),
        where=column_front > 0,
    )
    v_back = np.divide(
        _trapz(delta[:, back] * v_los[:, back], r_grid[back], axis=1),
        column_back,
        out=np.full_like(column_back, np.nan),
        where=column_back > 0,
    )

    column = column.reshape(args.map_n, args.map_n)
    v_front = v_front.reshape(args.map_n, args.map_n)
    v_back = v_back.reshape(args.map_n, args.map_n)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    im = axes[0].imshow(
        column, origin="lower", extent=(-extent, extent, -extent, extent),
        cmap="viridis")
    axes[0].set_title(r"$\int \delta_{\rm miss}\,dr$")
    fig.colorbar(im, ax=axes[0], label="Mpc/h")

    vmax = np.nanpercentile(np.abs(np.concatenate([
        v_front.ravel(), v_back.ravel()])), 99)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    im = axes[1].imshow(
        v_front, origin="lower", extent=(-extent, extent, -extent, extent),
        cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[1].set_title(r"foreground $\langle v_{\rm los}\rangle_\delta$")
    fig.colorbar(im, ax=axes[1], label=r"km s$^{-1}$")

    im = axes[2].imshow(
        v_back, origin="lower",
        extent=(-extent, extent, -extent, extent), cmap="coolwarm",
        vmin=-vmax, vmax=vmax)
    axes[2].set_title(r"background $\langle v_{\rm los}\rangle_\delta$")
    fig.colorbar(im, ax=axes[2], label=r"km s$^{-1}$")

    for ax in axes:
        ax.scatter([0.0], [0.0], c="white", edgecolor="black", s=35, zorder=3)
        ax.set_xlabel("tangent-plane offset [deg]")
        ax.set_ylabel("tangent-plane offset [deg]")

    fig.suptitle(
        f"LOS projections over [{r_min:.1f}, {r_max:.1f}] Mpc/h",
        fontsize=11,
    )
    path = outdir / "missing_mass_los_projection.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logM", type=float, default=15.0,
                        help="log10 missing mass in Msun/h.")
    parser.add_argument("--distance", type=float, default=50.0,
                        help="Missing-mass distance in Mpc/h.")
    parser.add_argument("--ell", type=float, default=0.0,
                        help="Missing-mass longitude in the selected frame.")
    parser.add_argument("--b", type=float, default=0.0,
                        help="Missing-mass latitude in the selected frame.")
    parser.add_argument("--frame", default="galactic",
                        choices=("icrs", "galactic", "supergalactic"),
                        help="Coordinate frame for ell/b.")
    parser.add_argument("--sigma", type=float, default=5.0,
                        help="Gaussian smoothing width in Mpc/h.")
    parser.add_argument("--Om", type=float, default=0.3,
                        help="Matter density parameter.")
    parser.add_argument("--growth-index", type=float, default=0.55,
                        help="Growth-rate index gamma in f=Omega_m^gamma.")
    parser.add_argument("--outdir", default="results/diagnostics/missing_mass",
                        help="Output directory for plots.")
    parser.add_argument("--profile-n", type=int, default=500)
    parser.add_argument("--profile-sigma-max", type=float, default=12.0)
    parser.add_argument("--los-n", type=int, default=700)
    parser.add_argument("--los-sigma-half-width", type=float, default=8.0)
    parser.add_argument("--map-n", type=int, default=91)
    parser.add_argument("--map-sigma-extent", type=float, default=4.0)
    parser.add_argument("--map-extent-deg", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sigma <= 0:
        raise ValueError("--sigma must be positive.")
    if args.distance <= 0:
        raise ValueError("--distance must be positive.")
    if args.map_n < 3:
        raise ValueError("--map-n must be at least 3.")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    cluster_rhat = spherical_rhat(args.ell, args.b)
    cluster_rhat_icrs = np.asarray(
        convert_cartesian_frame(cluster_rhat, args.frame, "icrs"))

    paths = [
        _plot_profiles(args, outdir, cluster_rhat_icrs),
        _plot_projection(args, outdir, cluster_rhat_icrs),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
