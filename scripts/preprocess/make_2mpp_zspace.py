"""
Build a pre-iteration 3D halo overdensity field from the 2M++ galaxy catalogue.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Tuple
import numpy as np
import healpy as hp
from h5py import File

import candel
from candel import fprint
from candel.util import galactic_to_radec, galactic_to_radec_cartesian, radec_to_cartesian

C_LIGHT = 299792.458

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for environments without JAX
    import numpy as jnp
    JAX_AVAILABLE = False
H0_MIN = 20.0


if JAX_AVAILABLE:
    def _jit(fn):
        return jax.jit(fn)

    def _jit_static(static_argnums):
        def wrapper(fn):
            return jax.jit(fn, static_argnums=static_argnums)
        return wrapper
else:
    def _jit(fn):
        return fn

    def _jit_static(static_argnums):
        def wrapper(fn):
            return fn
        return wrapper


def _as_float32(x):
    return jnp.asarray(x, dtype=jnp.float32)


def _load_groups(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "Group ID" in line and "|" in line:
            header_idx = i
            break

    if header_idx is not None:
        data = np.genfromtxt(
            path,
            delimiter="|",
            skip_header=header_idx + 2,
            usecols=range(8),
        )
        data = np.atleast_2d(data)
        names = [
            "gid",
            "l_deg",
            "b_deg",
            "K2Mpp",
            "richness",
            "vh",
            "vcmb",
            "sigma_v",
        ]
        return {name: data[:, i] for i, name in enumerate(names)}

    data = np.genfromtxt(path, names=True, dtype=None, encoding=None)
    name_map = {
        "gid": "gid",
        "groupid": "gid",
        "group_id": "gid",
        "l": "l_deg",
        "l_deg": "l_deg",
        "b": "b_deg",
        "b_deg": "b_deg",
        "k2mpp": "K2Mpp",
        "richness": "richness",
        "vh": "vh",
        "vcmb": "vcmb",
        "sigma_v": "sigma_v",
    }
    out = {}
    for name in data.dtype.names:
        key = name_map.get(name.strip().lower())
        if key is not None:
            out[key] = data[name]
    return out


def _load_galaxies(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "Designation" in line and "|" in line:
            header_idx = i
            break

    if header_idx is not None:
        data = np.genfromtxt(
            path,
            delimiter="|",
            skip_header=header_idx + 2,
            usecols=[3, 4, 7, 9],
        )
        data = np.atleast_2d(data)
        l_deg = data[:, 0]
        b_deg = data[:, 1]
        vcmb = data[:, 2]
        gid_raw = data[:, 3]
        gid = np.where(np.isfinite(gid_raw), gid_raw, -1).astype(int)
        return {
            "l_deg": l_deg,
            "b_deg": b_deg,
            "vcmb": vcmb,
            "gid": gid,
        }

    data = np.genfromtxt(path, names=True, dtype=None, encoding=None)
    if data.dtype.names is None:
        raise ValueError(f"Galaxy catalogue {path} is missing a header row.")
    name_map = {
        "gid": "gid",
        "groupid": "gid",
        "group_id": "gid",
        "groupid_2mpp": "gid",
        "l": "l_deg",
        "l_deg": "l_deg",
        "glon": "l_deg",
        "b": "b_deg",
        "b_deg": "b_deg",
        "glat": "b_deg",
        "vcmb": "vcmb",
        "v_cmb": "vcmb",
        "cz": "vcmb",
        "cz_cmb": "vcmb",
    }
    out = {}
    for name in data.dtype.names:
        key = name_map.get(name.strip().lower())
        if key is not None:
            out[key] = data[name]
    if "gid" not in out:
        out["gid"] = np.full_like(out["vcmb"], -1, dtype=int)
    missing = {key for key in ("l_deg", "b_deg", "vcmb") if key not in out}
    if missing:
        raise ValueError(f"Galaxy catalogue {path} missing columns: {sorted(missing)}")
    return out


def _sample_completeness(
    l_deg: np.ndarray,
    b_deg: np.ndarray,
    map11: np.ndarray,
    map12: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(90.0 - b_deg)
    phi = np.deg2rad(l_deg)
    nside11 = hp.get_nside(map11)
    nside12 = hp.get_nside(map12)
    pix11 = hp.ang2pix(nside11, theta, phi)
    pix12 = hp.ang2pix(nside12, theta, phi)
    c11 = map11[pix11]
    c12 = map12[pix12]
    use_12 = c12 > 0
    m_lim = np.where(use_12, 12.5, 11.5)
    comp = np.where(use_12, c12, c11)
    return m_lim, comp


@_jit_static(static_argnums=(2, 3))
def _cic_deposit(positions, weights, N, Rmax):
    dx = (2.0 * Rmax) / N
    coords = (positions + Rmax) / dx
    i0 = jnp.floor(coords).astype(jnp.int32)
    f = coords - i0
    offsets = jnp.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=jnp.int32,
    )
    idx = i0[:, None, :] + offsets[None, :, :]
    wx = jnp.where(offsets[:, 0] == 0, 1.0 - f[:, 0:1], f[:, 0:1])
    wy = jnp.where(offsets[:, 1] == 0, 1.0 - f[:, 1:2], f[:, 1:2])
    wz = jnp.where(offsets[:, 2] == 0, 1.0 - f[:, 2:3], f[:, 2:3])
    w = wx * wy * wz
    valid = (idx >= 0) & (idx < N)
    valid = jnp.all(valid, axis=2)
    w = w * valid
    idx = jnp.where(valid[:, :, None], idx, 0)
    idx_flat = (idx[:, :, 0] * N + idx[:, :, 1]) * N + idx[:, :, 2]
    idx_flat = idx_flat.reshape(-1)
    w_flat = (weights[:, None] * w).reshape(-1)
    grid_flat = jnp.zeros((N * N * N,), dtype=jnp.float32)
    if JAX_AVAILABLE:
        grid_flat = grid_flat.at[idx_flat].add(w_flat)
    else:  # pragma: no cover - numpy fallback
        np.add.at(grid_flat, idx_flat, w_flat)
    return grid_flat.reshape((N, N, N)), dx


@_jit
def _gaussian_smooth_fft(rho, sigma, box_side):
    N = rho.shape[0]
    dx = box_side / N
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=dx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=dx)
    kz = 2.0 * jnp.pi * jnp.fft.rfftfreq(N, d=dx)
    k2 = (
        kx[:, None, None] ** 2
        + ky[None, :, None] ** 2
        + kz[None, None, :] ** 2
    )
    kernel = jnp.exp(-0.5 * (sigma ** 2) * k2).astype(jnp.complex64)
    rho_k = jnp.fft.rfftn(rho)
    smoothed = jnp.fft.irfftn(rho_k * kernel, s=rho.shape)
    return smoothed.astype(jnp.float32)


@_jit_static(static_argnums=(2,))
def _radial_profile(s, weights, nbins, ds):
    idx = jnp.floor(s / ds).astype(jnp.int32)
    mask = (idx >= 0) & (idx < nbins)
    idx = jnp.where(mask, idx, 0)
    w = weights * mask
    sumw = jnp.zeros((nbins,), dtype=jnp.float32)
    if JAX_AVAILABLE:
        sumw = sumw.at[idx].add(w)
    else:  # pragma: no cover
        np.add.at(sumw, idx, w)
    edges = jnp.arange(nbins + 1, dtype=jnp.float32) * ds
    shell_vol = (4.0 / 3.0) * jnp.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    nbar = sumw / shell_vol
    return edges, nbar


def _smooth_1d_gaussian(values, dr, sigma=None):
    if sigma is None:
        sigma = 2.0 * dr
    half = int(np.ceil(3.0 * sigma / dr))
    if half < 1:
        return values
    x = jnp.arange(-half, half + 1, dtype=jnp.float32) * dr
    kernel = jnp.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / jnp.sum(kernel)
    return jnp.convolve(values, kernel, mode="same")


@_jit_static(static_argnums=(1,))
def _finite_diff_grad_3d(field, dx):
    grad = jnp.zeros(field.shape + (3,), dtype=jnp.float32)
    grad = grad.at[1:-1, :, :, 0].set((field[2:, :, :] - field[:-2, :, :]) / (2.0 * dx))
    grad = grad.at[0, :, :, 0].set((field[1, :, :] - field[0, :, :]) / dx)
    grad = grad.at[-1, :, :, 0].set((field[-1, :, :] - field[-2, :, :]) / dx)
    grad = grad.at[:, 1:-1, :, 1].set((field[:, 2:, :] - field[:, :-2, :]) / (2.0 * dx))
    grad = grad.at[:, 0, :, 1].set((field[:, 1, :] - field[:, 0, :]) / dx)
    grad = grad.at[:, -1, :, 1].set((field[:, -1, :] - field[:, -2, :]) / dx)
    grad = grad.at[:, :, 1:-1, 2].set((field[:, :, 2:] - field[:, :, :-2]) / (2.0 * dx))
    grad = grad.at[:, :, 0, 2].set((field[:, :, 1] - field[:, :, 0]) / dx)
    grad = grad.at[:, :, -1, 2].set((field[:, :, -1] - field[:, :, -2]) / dx)
    return grad


@_jit_static(static_argnums=(2,))
def _trilinear_samples(field, positions, N, Rmax):
    dx = (2.0 * Rmax) / N
    coords = (positions + Rmax) / dx
    i0 = jnp.floor(coords).astype(jnp.int32)
    f = coords - i0
    i0 = jnp.clip(i0, 0, N - 2)
    wx = jnp.stack([1.0 - f[:, 0], f[:, 0]], axis=1)
    wy = jnp.stack([1.0 - f[:, 1], f[:, 1]], axis=1)
    wz = jnp.stack([1.0 - f[:, 2], f[:, 2]], axis=1)
    out = jnp.zeros((positions.shape[0], field.shape[3]), dtype=jnp.float32)
    for dx_i in range(2):
        for dy_i in range(2):
            for dz_i in range(2):
                w = wx[:, dx_i] * wy[:, dy_i] * wz[:, dz_i]
                val = field[i0[:, 0] + dx_i, i0[:, 1] + dy_i, i0[:, 2] + dz_i]
                out = out + w[:, None] * val
    return out


def _dlnnbar_ds_1d(nbar_1d, ds, eps):
    logn = jnp.log(nbar_1d + eps)
    dln = jnp.zeros_like(logn)
    dln = dln.at[1:-1].set((logn[2:] - logn[:-2]) / (2.0 * ds))
    dln = dln.at[0].set((logn[1] - logn[0]) / ds)
    dln = dln.at[-1].set((logn[-1] - logn[-2]) / ds)
    return dln


def _trilinear_sample(field, positions, rmax, rmax_eff, chunk_size=200000):
    ngrid = field.shape[0]
    dx = (2.0 * rmax) / ngrid
    out = np.zeros((positions.shape[0],), dtype=np.float32)

    for start in range(0, positions.shape[0], chunk_size):
        end = min(start + chunk_size, positions.shape[0])
        pos = positions[start:end]
        radius = np.sqrt(np.sum(pos**2, axis=1))
        inside = radius <= rmax_eff
        if not np.any(inside):
            continue

        pos = np.clip(pos, -rmax_eff, rmax_eff)
        coords = (pos + rmax) / dx
        i0 = np.floor(coords).astype(np.int32)
        f = coords - i0
        i0 = np.clip(i0, 0, ngrid - 2)
        i1 = i0 + 1

        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        wx0, wx1 = 1.0 - fx, fx
        wy0, wy1 = 1.0 - fy, fy
        wz0, wz1 = 1.0 - fz, fz

        v000 = field[i0[:, 0], i0[:, 1], i0[:, 2]]
        v100 = field[i1[:, 0], i0[:, 1], i0[:, 2]]
        v010 = field[i0[:, 0], i1[:, 1], i0[:, 2]]
        v110 = field[i1[:, 0], i1[:, 1], i0[:, 2]]
        v001 = field[i0[:, 0], i0[:, 1], i1[:, 2]]
        v101 = field[i1[:, 0], i0[:, 1], i1[:, 2]]
        v011 = field[i0[:, 0], i1[:, 1], i1[:, 2]]
        v111 = field[i1[:, 0], i1[:, 1], i1[:, 2]]

        vals = (
            v000 * wx0 * wy0 * wz0
            + v100 * wx1 * wy0 * wz0
            + v010 * wx0 * wy1 * wz0
            + v110 * wx1 * wy1 * wz0
            + v001 * wx0 * wy0 * wz1
            + v101 * wx1 * wy0 * wz1
            + v011 * wx0 * wy1 * wz1
            + v111 * wx1 * wy1 * wz1
        )
        vals = vals.astype(np.float32)
        vals[~inside] = 0.0
        out[start:end] = vals

    return out


def _load_clusters_from_config(config):
    d = config["io"]["PV_main"]["Clusters"].copy()
    root = d.pop("root")
    data = candel.pvdata.load_clusters(root, **d)
    return data["RA"], data["dec"]


def compute_los_delta_from_field(
    delta,
    rho_sm,
    nbar_counts_3d,
    metadata,
    config_path,
    output_path,
    H0_base=100.0,
    chunk_size=200000,
    carrick_mask_path=None,
    carrick_mask_atol=1e-6,
    mask_smooth_sigma_mpc=5.0,
    mask_smooth_buffer_mpc=10.0,
):
    config = candel.load_config(config_path)
    d = config["io"]["reconstruction_main"].copy()
    r = np.linspace(d["rmin"], d["rmax"], d["num_steps"]).astype(np.float32)
    dist2redshift = candel.Distance2Redshift(Om0=0.3)
    z = np.asarray(dist2redshift(r, h=H0_base / 100.0), dtype=np.float32)
    s = z * C_LIGHT

    rmax = float(metadata["Rmax_kms"])
    dx = float(metadata["voxel_size_kms"])
    rmax_eff = rmax - dx

    RA, dec = _load_clusters_from_config(config)
    fprint(f"loaded {len(RA)} clusters from config.")

    rhat = radec_to_cartesian(RA, dec).astype(np.float32)
    n_gal = len(RA)
    n_s = len(s)
    los_delta = np.zeros((n_gal, n_s), dtype=np.float32)
    los_rho_sm = np.zeros((n_gal, n_s), dtype=np.float32)
    los_nbar = np.zeros((n_gal, n_s), dtype=np.float32)

    fprint("sampling LOS delta(s) from Carrick grid.")
    for i in range(n_gal):
        pos = (s[:, None] * rhat[i][None, :]).astype(np.float32)
        los_delta[i] = _trilinear_sample(delta, pos, rmax, rmax_eff, chunk_size)
        los_rho_sm[i] = _trilinear_sample(rho_sm, pos, rmax, rmax_eff, chunk_size)
        los_nbar[i] = _trilinear_sample(nbar_counts_3d, pos, rmax, rmax_eff, chunk_size)

    if carrick_mask_path is not None:
        with File(carrick_mask_path, "r") as f:
            r_c = f["r"][...]
            density_c = f["los_density"][...]
        if density_c.ndim == 3:
            density_c = density_c[0]
        if density_c.shape[0] != n_gal:
            raise ValueError("Carrick mask catalogue does not match cluster count.")
        mask_c = np.isclose(density_c, 1.0, rtol=0.0, atol=carrick_mask_atol).astype(np.float32)
        if r_c.shape != r.shape or not np.allclose(r_c, r):
            mask_interp = np.empty_like(los_delta, dtype=np.float32)
            for i in range(n_gal):
                mask_interp[i] = np.interp(r, r_c, mask_c[i], left=0.0, right=0.0)
            mask = mask_interp > 0.5
        else:
            mask = mask_c.astype(bool)
        los_delta = np.where(mask, 0.0, los_delta)

        if mask_smooth_sigma_mpc > 0.0:
            dr = float(r[1] - r[0])
            sigma_idx = mask_smooth_sigma_mpc / dr
            half = int(np.ceil(3.0 * sigma_idx))
            if half > 0:
                x = np.arange(-half, half + 1, dtype=np.float32)
                kernel = np.exp(-0.5 * (x / sigma_idx) ** 2)
                kernel = kernel / np.sum(kernel)
                for i in range(n_gal):
                    idx_mask = np.where(mask[i])[0]
                    if idx_mask.size == 0:
                        continue
                    start_r = r[idx_mask[0]] - mask_smooth_buffer_mpc
                    tail = r >= start_r
                    if not np.any(tail):
                        continue
                    smoothed = np.convolve(los_delta[i], kernel, mode="same")
                    los_delta[i, tail] = smoothed[tail]

    # Final LOS smoothing (applies everywhere).
    final_sigma_mpc = 2.0
    if final_sigma_mpc > 0.0:
        dr = float(r[1] - r[0])
        sigma_idx = final_sigma_mpc / dr
        half = int(np.ceil(3.0 * sigma_idx))
        if half > 0:
            x = np.arange(-half, half + 1, dtype=np.float32)
            kernel = np.exp(-0.5 * (x / sigma_idx) ** 2)
            kernel = kernel / np.sum(kernel)
            for i in range(n_gal):
                los_delta[i] = np.convolve(los_delta[i], kernel, mode="same")

    with File(output_path, "w") as f:
        f.create_dataset("RA", data=RA.astype(np.float32))
        f.create_dataset("dec", data=dec.astype(np.float32))
        f.create_dataset("r", data=r.astype(np.float32))
        f.create_dataset("s", data=s.astype(np.float32))
        f.create_dataset("los_delta", data=los_delta.astype(np.float32))
        f.create_dataset("los_rho_sm", data=los_rho_sm.astype(np.float32))
        f.create_dataset("los_nbar", data=los_nbar.astype(np.float32))
        los_shape = (1, los_delta.shape[0], los_delta.shape[1])
        f.create_dataset("los_velocity", data=np.zeros(los_shape, dtype=np.float32))
        los_density = np.maximum(los_delta + 1.0, 1e-6).astype(np.float32)
        f.create_dataset("los_density", data=los_density[None, :, :])
        f.attrs["H0_base"] = float(H0_base)
        f.attrs["Rmax_kms"] = rmax
        f.attrs["voxel_size_kms"] = dx

    fprint(f"saved LOS delta(s) to {output_path}.")


def plot_nbar_profile(metadata, output_path):
    import matplotlib.pyplot as plt

    s = metadata["nbar_s_kms"]
    nbar = metadata["nbar_profile"]
    z = s / C_LIGHT

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(z, np.log10(np.clip(nbar, 1e-12, None)), color="tab:blue", lw=1.2)
    ax.set_xlabel("z")
    ax.set_ylabel("log10(nbar)")
    ax.set_title("2mpp_zspace nbar(z)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
def dlnn_dr_on_grid(
    r_mpc,
    n_hat,
    H0_bar,
    A,
    l0_deg,
    b0_deg,
    grad_ln_ncorr,
    vmax_kms,
):
    n0 = galactic_to_radec_cartesian(l0_deg, b0_deg)
    dot = np.einsum("ij,j->i", n_hat, n0)
    H0_i = np.clip(H0_bar * (1.0 + A * dot), H0_MIN, np.inf)
    rmax = vmax_kms
    dx = (2.0 * rmax) / grad_ln_ncorr.shape[0]
    rmax_eff = rmax - dx
    s = H0_i * r_mpc
    clamped_sphere = s > rmax_eff
    s = np.minimum(s, rmax_eff)
    pos = s[:, None] * n_hat
    clamped_cube = np.any(np.abs(pos) > rmax_eff, axis=1)
    pos = np.clip(pos, -rmax_eff, rmax_eff)
    grad = _trilinear_samples(_as_float32(grad_ln_ncorr), _as_float32(pos), grad_ln_ncorr.shape[0], rmax)
    dlnn_ds = np.einsum("ij,ij->i", np.asarray(grad), n_hat)
    dlnn_dr = H0_i * dlnn_ds
    return dlnn_dr, dlnn_ds, (clamped_sphere | clamped_cube)


@_jit_static(static_argnums=(2, 3))
def _map_nbar_to_grid(nbar_s, Rmax, N, box_side, ds):
    dx = box_side / N
    coords = (-Rmax + 0.5 * dx) + jnp.arange(N, dtype=jnp.float32) * dx
    xx, yy, zz = jnp.meshgrid(coords, coords, coords, indexing="ij")
    rr = jnp.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    idx = jnp.floor(rr / ds).astype(jnp.int32)
    idx = jnp.clip(idx, 0, nbar_s.shape[0] - 1)
    nbar_3d = nbar_s[idx]
    nbar_3d = nbar_3d * dx ** 3
    nbar_3d = jnp.where(rr <= Rmax, nbar_3d, 0.0)
    return nbar_3d.astype(jnp.float32)


def build_delta_field(
    data_dir: str,
    tracer_mode: str = "galaxies",
    group_weight_mode: str = "unity",
    H0_bar: float = 100.0,
    A: float = 0.0,
    l0_deg: float = 0.0,
    b0_deg: float = 0.0,
    vmax_kms: float = 20000.0,
    N: int = 256,
    box_side: float | None = None,
    sigma_s_kms: float = 600.0,
    pad_sigma: float = 5.0,
    cmin: float = 0.5,
    nbar_ds_kms: float | None = 250.0,
    tiny: float = 1e-8,
    eps: float = 1e-6,
) -> Dict[str, object]:
    """
    Build the Carrick-style 3D overdensity field from 2M++ galaxies.
    """
    if box_side is None:
        box_side = 2.0 * vmax_kms
    Rmax = 0.5 * box_side

    group_path = os.path.join(data_dir, "twompp_groups.txt")
    if not os.path.exists(group_path):
        group_path = os.path.join(data_dir, "2m++_groups.txt")
    groups = _load_groups(group_path)
    group_gid = groups["gid"].astype(int)
    group_l = groups["l_deg"]
    group_b = groups["b_deg"]
    group_v = groups["vcmb"]

    galaxy_candidates = [
        "twompp_galaxies.txt",
        "2m++_galaxies.txt",
        "twompp_galcat.txt",
        "2m++_galcat.txt",
        "twompp.txt",
        "2m++.txt",
    ]
    galaxy_path = None
    for name in galaxy_candidates:
        candidate = os.path.join(data_dir, name)
        if os.path.exists(candidate):
            galaxy_path = candidate
            break
    if galaxy_path is None:
        raise FileNotFoundError(f"No galaxy catalogue found in {data_dir}. Tried {galaxy_candidates}.")

    galaxies = _load_galaxies(galaxy_path)
    l_deg = galaxies["l_deg"].astype(np.float32)
    b_deg = galaxies["b_deg"].astype(np.float32)
    vcmb = galaxies["vcmb"].astype(np.float32)
    gid = galaxies["gid"].astype(int)

    n_gal_total = len(vcmb)
    mask_finite = np.isfinite(l_deg) & np.isfinite(b_deg) & np.isfinite(vcmb)
    n_gal_bad = int(np.sum(~mask_finite))
    l_deg = l_deg[mask_finite]
    b_deg = b_deg[mask_finite]
    vcmb = vcmb[mask_finite]
    gid = gid[mask_finite]
    grouped = gid >= 0
    n_gal_grouped = int(np.sum(grouped))

    order = np.argsort(group_gid)
    gid_sorted = group_gid[order]
    g_l_sorted = group_l[order]
    g_b_sorted = group_b[order]
    g_v_sorted = group_v[order]
    idx = np.searchsorted(gid_sorted, gid[grouped])
    valid = idx < gid_sorted.size
    match = np.zeros_like(valid, dtype=bool)
    match[valid] = gid_sorted[idx[valid]] == gid[grouped][valid]
    gid_missing = grouped.copy()
    gid_missing[grouped] = ~match
    n_gal_gid_missing = int(np.sum(gid_missing))

    grouped_idx = np.where(grouped)[0]
    collapse_idx = grouped_idx[match]
    match_idx = idx[match]
    l_deg[collapse_idx] = g_l_sorted[match_idx]
    b_deg[collapse_idx] = g_b_sorted[match_idx]
    vcmb[collapse_idx] = g_v_sorted[match_idx]
    gid[gid_missing] = -1
    n_gal_collapsed = int(collapse_idx.size)

    map11 = hp.read_map(f"{data_dir}/incompleteness_11_5.fits", verbose=False)
    map12 = hp.read_map(f"{data_dir}/incompleteness_12_5.fits", verbose=False)

    mask_cat = vcmb > 0.0
    l_deg = l_deg[mask_cat]
    b_deg = b_deg[mask_cat]
    vcmb = vcmb[mask_cat]
    gid = gid[mask_cat]

    _, comp = _sample_completeness(l_deg, b_deg, map11, map12)
    comp_bad = ~np.isfinite(comp)
    n_comp_bad = int(np.sum(comp_bad))
    if n_comp_bad:
        comp = np.where(comp_bad, 0.0, comp)
    comp = np.minimum(comp, 1.0)
    comp_clip = np.clip(comp, cmin, 1.0)
    w_ang = 1.0 / comp_clip
    low_comp = comp < cmin

    ra_deg, dec_deg = galactic_to_radec(l_deg, b_deg)
    n_hat = radec_to_cartesian(ra_deg, dec_deg)
    W = w_ang

    if tracer_mode not in {"galaxies", "groups"}:
        raise ValueError(f"Unknown tracer_mode={tracer_mode}; expected 'galaxies' or 'groups'.")
    if tracer_mode == "groups":
        grouped = gid >= 0
        gid_grouped = gid[grouped]
        w_grouped = w_ang[grouped]
        l_grouped = l_deg[grouped]
        b_grouped = b_deg[grouped]
        v_grouped = vcmb[grouped]

        if gid_grouped.size > 0:
            order = np.argsort(gid_grouped)
            gid_sorted = gid_grouped[order]
            w_sorted = w_grouped[order]
            l_sorted = l_grouped[order]
            b_sorted = b_grouped[order]
            v_sorted = v_grouped[order]
            unique_gid, idx_start, counts = np.unique(gid_sorted, return_index=True, return_counts=True)
            if group_weight_mode == "membersum":
                w_group = np.add.reduceat(w_sorted, idx_start)
            elif group_weight_mode == "unity":
                w_group = w_sorted[idx_start]
            else:
                raise ValueError(f"Unknown group_weight_mode={group_weight_mode}")
            l_group = l_sorted[idx_start]
            b_group = b_sorted[idx_start]
            v_group = v_sorted[idx_start]
        else:
            unique_gid = np.array([], dtype=int)
            w_group = np.array([], dtype=np.float32)
            l_group = np.array([], dtype=np.float32)
            b_group = np.array([], dtype=np.float32)
            v_group = np.array([], dtype=np.float32)

        ungrouped = gid < 0
        l_ung = l_deg[ungrouped]
        b_ung = b_deg[ungrouped]
        v_ung = vcmb[ungrouped]
        w_ung = w_ang[ungrouped]

        l_deg = np.concatenate([l_group, l_ung], axis=0)
        b_deg = np.concatenate([b_group, b_ung], axis=0)
        vcmb = np.concatenate([v_group, v_ung], axis=0)
        W = np.concatenate([w_group, w_ung], axis=0)

        ra_deg, dec_deg = galactic_to_radec(l_deg, b_deg)
        n_hat = radec_to_cartesian(ra_deg, dec_deg)

        n_groups_from_galcat = int(unique_gid.size)
        n_groups_used = int(unique_gid.size)
        n_ungrouped_used = int(np.sum(ungrouped))
    else:
        n_groups_from_galcat = None
        n_groups_used = None
        n_ungrouped_used = None

    pos = vcmb[:, None] * n_hat
    positions = _as_float32(pos)
    weights = _as_float32(W)

    rho, dx = _cic_deposit(positions, weights, N, Rmax)
    pad = int(np.ceil(pad_sigma * sigma_s_kms / dx))
    pad = max(1, pad)
    rho_pad = jnp.pad(rho, ((pad, pad), (pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
    box_side_p = dx * rho_pad.shape[0]
    rho_sm_pad = _gaussian_smooth_fft(rho_pad, sigma_s_kms, box_side_p)
    rho_sm = rho_sm_pad[pad:pad + N, pad:pad + N, pad:pad + N]

    ds = dx if nbar_ds_kms is None else float(nbar_ds_kms)
    nbins = max(1, int(np.ceil(vmax_kms / ds)))
    edges, nbar = _radial_profile(_as_float32(vcmb), _as_float32(W), nbins, ds)
    nbar_sm = _smooth_1d_gaussian(nbar, ds)

    nbar_counts_3d = _map_nbar_to_grid(
        nbar_sm,
        Rmax,
        N,
        box_side,
        ds,
    )
    coords = (-Rmax + 0.5 * dx) + jnp.arange(N, dtype=jnp.float32) * dx
    xx, yy, zz = jnp.meshgrid(coords, coords, coords, indexing="ij")
    rr = jnp.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    mask_sphere = rr <= Rmax
    rho_sm = jnp.where(mask_sphere, rho_sm, 0.0)
    nbar_counts_3d = jnp.where(mask_sphere, nbar_counts_3d, 0.0)
    nbar_counts_3d = jnp.where(mask_sphere, jnp.maximum(nbar_counts_3d, tiny), 0.0)

    # IMB-related fields (disabled): keep for future use.
    # ln_ncorr = jnp.where(
    #     mask_sphere,
    #     jnp.log(rho_sm + eps) - jnp.log(nbar_counts_3d + eps),
    #     0.0,
    # )
    # grad_ln_ncorr = _finite_diff_grad_3d(ln_ncorr, dx)
    # grad_ln_ncorr = jnp.where(mask_sphere[..., None], grad_ln_ncorr, 0.0)
    # dlnnbar_ds_1d = _dlnnbar_ds_1d(nbar_sm, ds, eps)

    delta = jnp.where(mask_sphere, rho_sm / nbar_counts_3d - 1.0, 0.0)

    r_centers = 0.5 * (edges[:-1] + edges[1:])
    metadata = {
        "box_side_kms": float(box_side),
        "voxel_size_kms": float(dx),
        "Rmax_kms": float(Rmax),
        "coord_convention": "x=cos(dec)cos(ra), y=cos(dec)sin(ra), z=sin(dec)",
        "nbar_s_kms": np.asarray(r_centers),
        "nbar_profile": np.asarray(nbar_sm),
        "nbar_ds_kms": float(ds),
        "nbar_weight": "w_ang",
        "completeness_floor": float(cmin),
        "completeness_weighting": "1/clip(comp,cmin,1)",
        "grid_space": "z-space (s = vcmb)",
        "tracer_mode": tracer_mode,
        "group_weight_mode": group_weight_mode if tracer_mode == "groups" else "n/a",
        "diagnostics": {
            "n_gal_total": int(n_gal_total),
            "n_gal_after_vcut": int(np.sum(mask_cat)),
            "n_gal_bad_values": int(n_gal_bad),
            "n_gal_grouped": int(n_gal_grouped),
            "n_gal_collapsed": int(n_gal_collapsed),
            "n_gal_gid_missing": int(n_gal_gid_missing),
            "n_groups_from_galcat": int(n_groups_from_galcat) if n_groups_from_galcat is not None else -1,
            "n_groups_used": int(n_groups_used) if n_groups_used is not None else -1,
            "n_ungrouped_used": int(n_ungrouped_used) if n_ungrouped_used is not None else -1,
            "n_comp_bad": int(n_comp_bad),
            "low_comp_fraction": float(np.mean(low_comp)),
            "comp_min": float(np.min(comp)),
            "comp_median": float(np.median(comp)),
            "comp_mean": float(np.mean(comp)),
            "w_ang_min": float(np.min(w_ang)),
            "w_ang_median": float(np.median(w_ang)),
            "w_ang_mean": float(np.mean(w_ang)),
        },
        # "ln_ncorr_units": "dimensionless",
        # "grad_ln_ncorr_units": "1/(km/s)",
        # "dlnn_dr_units": "1/Mpc",
    }

    return {
        "rho_sm": np.asarray(rho_sm, dtype=np.float32),
        "nbar_counts_3d": np.asarray(nbar_counts_3d, dtype=np.float32),
        "nbar_1d": np.asarray(nbar_sm, dtype=np.float32),
        "delta": np.asarray(delta, dtype=np.float32),
        "metadata": metadata,
    }


def _summarize_delta(delta, nbar_counts_3d, tiny):
    mask = nbar_counts_3d > tiny
    if not np.any(mask):
        return "No finite voxels found for summary."
    finite = delta[mask]
    return (
        f"delta stats (finite voxels): min={finite.min():.4f}, "
        f"max={finite.max():.4f}, mean={finite.mean():.4f}"
    )


def _self_test_grad(ln_ncorr, grad_ln_ncorr, dx, rng_seed=0, ntest=100):
    rng = np.random.default_rng(rng_seed)
    N = ln_ncorr.shape[0]
    idx = rng.integers(1, N - 2, size=(ntest, 3))
    n_hat = rng.normal(size=(ntest, 3))
    n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True)
    samples = grad_ln_ncorr[idx[:, 0], idx[:, 1], idx[:, 2]]
    dlnn_ds = np.einsum("ij,ij->i", samples, n_hat)
    idx_f = idx + np.sign(n_hat).astype(int)
    idx_f = np.clip(idx_f, 0, N - 1)
    idx_b = idx - np.sign(n_hat).astype(int)
    idx_b = np.clip(idx_b, 0, N - 1)
    ln_f = ln_ncorr[idx_f[:, 0], idx_f[:, 1], idx_f[:, 2]]
    ln_b = ln_ncorr[idx_b[:, 0], idx_b[:, 1], idx_b[:, 2]]
    fd = (ln_f - ln_b) / (2.0 * dx)
    diff = dlnn_ds - fd
    return float(np.mean(diff)), float(np.std(diff))


def main():
    parser = argparse.ArgumentParser(
        description="Build Carrick-style 3D overdensity field from 2M++ galaxies."
    )
    parser.add_argument("--data-dir", default="data/2M++")
    parser.add_argument("--output", default="carrick_field.npz")
    parser.add_argument("--tracer-mode", choices=["galaxies", "groups"], default="galaxies")
    parser.add_argument("--group-weight-mode", choices=["unity", "membersum"], default="unity")
    parser.add_argument("--H0-bar", type=float, default=100.0)
    parser.add_argument("--A", type=float, default=0.0)
    parser.add_argument("--l0-deg", type=float, default=0.0)
    parser.add_argument("--b0-deg", type=float, default=0.0)
    parser.add_argument("--vmax-kms", type=float, default=20000.0)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--box-side", type=float, default=None)
    parser.add_argument("--sigma-s-kms", type=float, default=600.0)
    parser.add_argument("--pad-sigma", type=float, default=5.0)
    parser.add_argument("--cmin", type=float, default=0.5)
    parser.add_argument("--nbar-ds-kms", type=float, default=250.0)
    parser.add_argument("--tiny", type=float, default=1e-8)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--compute-los", action="store_true", default=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--los-output", type=str, default=None)
    parser.add_argument("--H0-base", type=float, default=100.0)
    parser.add_argument("--los-chunk-size", type=int, default=200000)
    parser.add_argument("--carrick-mask", type=str, default="data/Clusters/los_Clusters_Carrick2015.hdf5")
    parser.add_argument("--carrick-mask-atol", type=float, default=1e-6)
    parser.add_argument("--no-compare-plot", action="store_true", default=False)
    parser.add_argument("--plot-nclusters", type=int, default=312)
    parser.add_argument("--plot-ncols", type=int, default=6)
    parser.add_argument("--compare-only", action="store_true", default=False)
    args = parser.parse_args()
    if args.tracer_mode == "galaxies":
        mode_suffix = "galaxies"
    else:
        mode_suffix = f"groups_{args.group_weight_mode}"

    def _with_suffix(path, suffix):
        root, ext = os.path.splitext(path)
        return f"{root}_{suffix}{ext}"

    output_path = _with_suffix(args.output, mode_suffix)
    los_output = _with_suffix(args.los_output, mode_suffix) if args.los_output else None

    def _import_run_compare():
        try:
            from .compare_reconstructions import run_compare
        except ImportError:
            from scripts.preprocess.compare_reconstructions import run_compare
        return run_compare

    if args.compare_only:
        if los_output is None:
            raise ValueError("--compare-only requires --los-output.")
        run_compare = _import_run_compare()
        output_png = os.path.join(os.path.dirname(los_output), "compare_reconstructions_galaxies.png")
        run_compare(
            carrick="data/Clusters/los_Clusters_Carrick2015.hdf5",
            manticore="data/Clusters/los_Clusters_manticore.hdf5",
            zspace=los_output,
            output=output_png,
            ncols=args.plot_ncols,
            nclusters=args.plot_nclusters,
            config_path=args.config,
        )
        return

    t0 = time.time()
    result = build_delta_field(
        data_dir=args.data_dir,
        tracer_mode=args.tracer_mode,
        group_weight_mode=args.group_weight_mode,
        H0_bar=args.H0_bar,
        A=args.A,
        l0_deg=args.l0_deg,
        b0_deg=args.b0_deg,
        vmax_kms=args.vmax_kms,
        N=args.N,
        box_side=args.box_side,
        sigma_s_kms=args.sigma_s_kms,
        pad_sigma=args.pad_sigma,
        cmin=args.cmin,
        nbar_ds_kms=args.nbar_ds_kms,
        tiny=args.tiny,
        eps=args.eps,
    )
    elapsed = time.time() - t0

    np.savez_compressed(
        output_path,
        rho_sm=result["rho_sm"],
        nbar_counts_3d=result["nbar_counts_3d"],
        nbar_1d=result["nbar_1d"],
        delta=result["delta"],
        metadata=np.array(result["metadata"], dtype=object),
    )

    print(_summarize_delta(result["delta"], result["nbar_counts_3d"], args.tiny))
    # if args.self_test:
    #     mean_diff, std_diff = _self_test_grad(
    #         result["ln_ncorr"],
    #         result["grad_ln_ncorr"],
    #         result["metadata"]["voxel_size_kms"],
    #     )
    #     print(f"self-test dlnn_ds diff: mean={mean_diff:.3e}, std={std_diff:.3e}")
    if args.compute_los:
        if args.config is None or los_output is None:
            raise ValueError("--compute-los requires --config and --los-output.")
        diag = result["metadata"].get("diagnostics", {})
        if diag:
            print(
                "Diagnostics: total galaxies={n_gal_total}, "
                "after vcut={n_gal_after_vcut}, "
                "low_comp_frac={low_comp_fraction:.3f}, "
                "comp_med={comp_median:.3f}, "
                "w_ang_med={w_ang_median:.3f}".format(**diag)
            )
        compute_los_delta_from_field(
            result["delta"],
            result["rho_sm"],
            result["nbar_counts_3d"],
            result["metadata"],
            args.config,
            los_output,
            H0_base=args.H0_base,
            chunk_size=args.los_chunk_size,
            carrick_mask_path=args.carrick_mask,
            carrick_mask_atol=args.carrick_mask_atol,
        )
        los_dir = os.path.dirname(los_output)
        nbar_png = os.path.join(los_dir, f"nbar_profile_{mode_suffix}.png")
        plot_nbar_profile(result["metadata"], nbar_png)
        if not args.no_compare_plot:
            run_compare = _import_run_compare()
            output_png = os.path.join(los_dir, "compare_reconstructions_galaxies.png")
            run_compare(
                carrick="data/Clusters/los_Clusters_Carrick2015.hdf5",
                manticore="data/Clusters/los_Clusters_manticore.hdf5",
                zspace=los_output,
                output=output_png,
                ncols=args.plot_ncols,
                nclusters=args.plot_nclusters,
                config_path=args.config,
            )
    print(f"Saved output to {output_path}")
    print(f"Elapsed time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
