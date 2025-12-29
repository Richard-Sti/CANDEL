"""
Build a pre-iteration 3D halo overdensity field from the 2M++ group catalogue.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Tuple

import numpy as np
import healpy as hp

from ..util import galactic_to_radec, galactic_to_radec_cartesian, radec_to_cartesian

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
    H0_bar: float = 100.0,
    A: float = 0.0,
    l0_deg: float = 0.0,
    b0_deg: float = 0.0,
    vmax_kms: float = 20000.0,
    N: int = 256,
    box_side: float | None = None,
    sigma_s_kms: float = 250.0,
    pad_sigma: float = 5.0,
    cmin: float = 0.2,
    nbar_ds_kms: float | None = 250.0,
    tiny: float = 1e-8,
    eps: float = 1e-6,
) -> Dict[str, object]:
    """
    Build the Carrick-style 3D overdensity field from 2M++ groups.
    """
    if box_side is None:
        box_side = 2.0 * vmax_kms
    Rmax = 0.5 * box_side

    group_path = os.path.join(data_dir, "twompp_groups.txt")
    if not os.path.exists(group_path):
        group_path = os.path.join(data_dir, "2m++_groups.txt")
    groups = _load_groups(group_path)
    l_deg = groups["l_deg"]
    b_deg = groups["b_deg"]
    vcmb = groups["vcmb"]

    mask_cat = (vcmb > 0.0) & (vcmb < vmax_kms)
    l_deg = l_deg[mask_cat]
    b_deg = b_deg[mask_cat]
    vcmb = vcmb[mask_cat]

    map11 = hp.read_map(f"{data_dir}/incompleteness_11_5.fits", verbose=False)
    map12 = hp.read_map(f"{data_dir}/incompleteness_12_5.fits", verbose=False)

    _, comp = _sample_completeness(l_deg, b_deg, map11, map12)
    w_ang = 1.0 / np.clip(comp, cmin, 1.0)

    ra_deg, dec_deg = galactic_to_radec(l_deg, b_deg)
    n_hat = radec_to_cartesian(ra_deg, dec_deg)
    W = w_ang

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
    edges, nbar = _radial_profile(_as_float32(vcmb), _as_float32(w_ang), nbins, ds)
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
        "grid_space": "z-space (s = vcmb)",
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
        description="Build Carrick-style 3D overdensity field from 2M++ groups."
    )
    parser.add_argument("--data-dir", default="data/2M++")
    parser.add_argument("--output", default="carrick_field.npz")
    parser.add_argument("--H0-bar", type=float, default=100.0)
    parser.add_argument("--A", type=float, default=0.0)
    parser.add_argument("--l0-deg", type=float, default=0.0)
    parser.add_argument("--b0-deg", type=float, default=0.0)
    parser.add_argument("--vmax-kms", type=float, default=20000.0)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--box-side", type=float, default=None)
    parser.add_argument("--sigma-s-kms", type=float, default=250.0)
    parser.add_argument("--pad-sigma", type=float, default=5.0)
    parser.add_argument("--cmin", type=float, default=0.2)
    parser.add_argument("--nbar-ds-kms", type=float, default=250.0)
    parser.add_argument("--tiny", type=float, default=1e-8)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = build_delta_field(
        data_dir=args.data_dir,
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
        args.output,
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
    print(f"Saved output to {args.output}")
    print(f"Elapsed time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
