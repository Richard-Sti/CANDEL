"""Brute-force reference kernels for maser disk model convergence checks.

Two kernels:
- bruteforce_ll_mode2: full-2pi phi x log-r brute force (Mode 2 reference).
- bruteforce_ll_mode1: full-2pi phi at a given r_ang (Mode 1 reference).

Both batch over the r-axis and the spot-axis so the intermediate fits on a
12 GB GPU.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from candel.model.integration import trapz_log_weights
from candel.model.model_H0_maser import (
    LOG_2PI, PC_PER_MAS_MPC, SPEED_OF_LIGHT,
    _observables_from_precomputed, _precompute_r_quantities,
    warp_geometry,
)


_DTYPES = {"float32": jnp.float32, "float64": jnp.float64}


@partial(jax.jit, static_argnames=("n_phi", "has_accel", "ecc_on", "dt"))
def _bruteforce_rchunk(
        x_obs, y_obs, v_obs, var_x, var_y, var_v,
        a_obs, var_a, has_accel,
        x0, y0, D_A, M_BH, v_sys,
        r_ang_ref_i, r_ang_ref_Omega, r_ang_ref_periapsis,
        i0, di_dr, d2i_dr2, Omega0, dOmega_dr, d2Omega_dr2,
        ecc, periapsis0, dperiapsis_dr, ecc_on,
        r_chunk, log_w_r_chunk, n_phi, dt):
    """Partial brute-force integral over one r-chunk x full phi grid.
    Casts to `dt` (jnp.float32 or jnp.float64)."""
    f = lambda z: jnp.asarray(z, dtype=dt)
    x_obs, y_obs, v_obs = f(x_obs), f(y_obs), f(v_obs)
    var_x, var_y, var_v = f(var_x), f(var_y), f(var_v)
    a_obs, var_a = f(a_obs), f(var_a)
    x0, y0, v_sys = f(x0), f(y0), f(v_sys)
    D_A, M_BH = f(D_A), f(M_BH)
    r_ang_ref_i = f(r_ang_ref_i)
    r_ang_ref_Omega = f(r_ang_ref_Omega)
    r_ang_ref_periapsis = f(r_ang_ref_periapsis)
    i0, di_dr, d2i_dr2 = f(i0), f(di_dr), f(d2i_dr2)
    Omega0, dOmega_dr, d2Omega_dr2 = f(Omega0), f(dOmega_dr), f(d2Omega_dr2)
    ecc, periapsis0, dperiapsis_dr = f(ecc), f(periapsis0), f(dperiapsis_dr)
    r_chunk, log_w_r_chunk = f(r_chunk), f(log_w_r_chunk)

    phi_grid = jnp.linspace(0.0, 2 * jnp.pi, n_phi, dtype=dt)
    log_w_phi = trapz_log_weights(phi_grid)

    i_r, Om_r = warp_geometry(
        r_chunk, r_ang_ref_i, r_ang_ref_Omega,
        i0, di_dr, Omega0, dOmega_dr, d2i_dr2, d2Omega_dr2)
    sin_i = jnp.sin(i_r); cos_i = jnp.cos(i_r)
    sin_O = jnp.sin(Om_r); cos_O = jnp.cos(Om_r)
    v_kep, gamma, z_g, a_mag, pA, pB, pC, pD = _precompute_r_quantities(
        r_chunk, D_A, M_BH, sin_i, cos_i, sin_O, cos_O)

    sp = jnp.sin(phi_grid)[None, :]
    cp = jnp.cos(phi_grid)[None, :]
    X, Y, V, A = _observables_from_precomputed(
        sp, cp, x0, y0, v_sys,
        sin_i[:, None], r_chunk[:, None],
        v_kep[:, None], gamma[:, None], z_g[:, None], a_mag[:, None],
        pA[:, None], pB[:, None], pC[:, None], pD[:, None])

    if ecc_on:
        omega_r = (periapsis0
                   + dperiapsis_dr * (r_chunk - r_ang_ref_periapsis))
        sw = jnp.sin(omega_r)[:, None]
        cw = jnp.cos(omega_r)[:, None]
        cos_d = cp * cw + sp * sw
        ecc_fac = (cp + ecc * cw) / jnp.sqrt(1.0 + ecc * cos_d)
        v_z = v_kep[:, None] * ecc_fac * sin_i[:, None]
        beta_c2 = (v_kep[:, None] / SPEED_OF_LIGHT) ** 2
        beta_e2 = (beta_c2 * (1.0 + ecc ** 2 + 2.0 * ecc * cos_d)
                   / (1.0 + ecc * cos_d))
        gamma_e = 1.0 / jnp.sqrt(1.0 - beta_e2)
        one_plus_z_D = gamma_e * (1.0 + v_z / SPEED_OF_LIGHT)
        V = SPEED_OF_LIGHT * (
            one_plus_z_D * z_g[:, None]
            * (1.0 + v_sys / SPEED_OF_LIGHT) - 1.0)

    chi2 = ((x_obs[:, None, None] - X[None]) ** 2
            * (1.0 / var_x)[:, None, None]
            + (y_obs[:, None, None] - Y[None]) ** 2
            * (1.0 / var_y)[:, None, None]
            + (v_obs[:, None, None] - V[None]) ** 2
            * (1.0 / var_v)[:, None, None])
    if has_accel:
        chi2 = chi2 + ((a_obs[:, None, None] - A[None]) ** 2
                       * (1.0 / var_a)[:, None, None])

    lnorm = -0.5 * (3 * LOG_2PI + jnp.log(var_x) + jnp.log(var_y)
                     + jnp.log(var_v))
    if has_accel:
        lnorm = lnorm - 0.5 * (LOG_2PI + jnp.log(var_a))

    log_f = lnorm[:, None, None] - 0.5 * chi2
    log_w_2d = log_w_r_chunk[:, None] + log_w_phi[None, :]
    return logsumexp(log_f + log_w_2d[None], axis=(-2, -1))


def bruteforce_ll_mode2(model, phys_args, phys_kw, ref_cfg):
    """Total log-likelihood reference via chunked r x full-2pi phi.

    ref_cfg: dict with keys n_r, n_phi, r_chunk, spot_batch, dtype.
    Returns a python float.
    """
    dt = _DTYPES[str(ref_cfg.get("dtype", "float32"))]
    n_r = int(ref_cfg["n_r"])
    n_phi = int(ref_cfg["n_phi"])
    r_chunk_size = int(ref_cfg["r_chunk"])
    spot_batch = int(ref_cfg["spot_batch"])

    (x0, y0, D_A, M_BH, v_sys,
     r_ang_ref_i, r_ang_ref_Omega, r_ang_ref_periapsis,
     i0, di_dr, Omega0, dOmega_dr,
     sigma_x_floor2, sigma_y_floor2,
     var_v_sys, var_v_hv, sigma_a_floor2) = phys_args
    d2i_dr2 = phys_kw.get("d2i_dr2", 0.0)
    d2Omega_dr2 = phys_kw.get("d2Omega_dr2", 0.0)
    ecc = phys_kw.get("ecc", None)
    periapsis0 = phys_kw.get("periapsis0", 0.0)
    dperiapsis_dr = phys_kw.get("dperiapsis_dr", 0.0)
    ecc_on = ecc is not None
    if not ecc_on:
        ecc = 0.0

    conv = D_A * PC_PER_MAS_MPC
    r_min = model._R_phys_lo / conv
    r_max = model._R_phys_hi / conv

    log_r = jnp.linspace(jnp.log(r_min), jnp.log(r_max), n_r)
    r_grid = jnp.exp(log_r)
    log_w_r = trapz_log_weights(r_grid)

    var_x = model._all_sigma_x2 + sigma_x_floor2
    var_y = model._all_sigma_y2 + sigma_y_floor2
    sv_floor = jnp.where(model.is_highvel, var_v_hv, var_v_sys)
    var_v = model._all_sigma_v2 + sv_floor
    var_a = model._all_sigma_a2 + sigma_a_floor2
    has_accel_all = np.asarray(model._all_has_accel)

    total = 0.0
    for has_a in (True, False):
        mask = has_accel_all if has_a else ~has_accel_all
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        for s in range(0, len(idx), spot_batch):
            b = idx[s:s + spot_batch]
            partials = []
            for start in range(0, n_r, r_chunk_size):
                end = min(start + r_chunk_size, n_r)
                r_chunk = r_grid[start:end]
                log_w_r_chunk = log_w_r[start:end]
                p = _bruteforce_rchunk(
                    model._all_x[b], model._all_y[b], model._all_v[b],
                    var_x[b], var_y[b], var_v[b],
                    model._all_a[b], var_a[b], has_a,
                    x0, y0, D_A, M_BH, v_sys,
                    r_ang_ref_i, r_ang_ref_Omega, r_ang_ref_periapsis,
                    i0, di_dr, d2i_dr2, Omega0, dOmega_dr, d2Omega_dr2,
                    ecc, periapsis0, dperiapsis_dr, ecc_on,
                    r_chunk, log_w_r_chunk, n_phi, dt)
                partials.append(p)
            ll = logsumexp(jnp.stack(partials, axis=0), axis=0)
            total += float(jnp.sum(ll))
    return total


def bruteforce_ll_mode1(model, phys_args, phys_kw, r_ang, ref_cfg):
    """Per-type full-2pi phi brute force at a fixed r_ang vector.

    r_ang: shape (n_spots,) in mas.
    ref_cfg: dict with keys n_phi, spot_batch, dtype (dtype accepted for
        symmetry; this kernel runs at the model's working precision since
        the per-spot peaks are narrow).
    Returns dict with keys 'sys', 'red', 'blue', 'total'.
    """
    n_phi = int(ref_cfg["n_phi"])
    spot_batch = int(ref_cfg["spot_batch"])

    phi = jnp.linspace(0.0, 2 * jnp.pi, n_phi)
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    log_w = trapz_log_weights(phi)

    r_ang = jnp.asarray(r_ang)
    out = {}
    for key, idx in [("sys", model._idx_sys),
                     ("red", model._idx_red),
                     ("blue", model._idx_blue)]:
        n = int(idx.shape[0])
        if n == 0:
            out[key] = 0.0
            continue
        parts = []
        for s in range(0, n, spot_batch):
            b = idx[s:s + spot_batch]
            log_f = model._phi_integrand(
                r_ang[b], sin_phi, cos_phi, b, *phys_args, **phys_kw)
            parts.append(logsumexp(log_f + log_w, axis=-1))
        out[key] = float(jnp.sum(jnp.concatenate(parts)))
    out["total"] = out["sys"] + out["red"] + out["blue"]
    return out
