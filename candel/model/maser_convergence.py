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
from candel.model.model_H0_maser import (LOG_2PI, neg_half_chi2_acceleration,
                                         neg_half_chi2_position,
                                         neg_half_chi2_velocity,
                                         predict_acceleration_los,
                                         predict_position,
                                         predict_velocity_los, warp_geometry)

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
    def f(z):
        return jnp.asarray(z, dtype=dt)
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

    sp = jnp.sin(phi_grid)[None, :]
    cp = jnp.cos(phi_grid)[None, :]

    r_b = r_chunk[:, None]
    sin_i_b = jnp.sin(i_r)[:, None]
    cos_i_b = jnp.cos(i_r)[:, None]
    sin_O_b = jnp.sin(Om_r)[:, None]
    cos_O_b = jnp.cos(Om_r)[:, None]

    X, Y = predict_position(r_b, sp, cp, x0, y0,
                            sin_i_b, cos_i_b, sin_O_b, cos_O_b)
    A = predict_acceleration_los(r_b, sp, cp, D_A, M_BH, sin_i_b)
    if ecc_on:
        omega_r = (periapsis0
                   + dperiapsis_dr * (r_chunk - r_ang_ref_periapsis))
        V = predict_velocity_los(
            r_b, sp, cp, D_A, M_BH, v_sys, sin_i_b,
            ecc=ecc,
            sin_om=jnp.sin(omega_r)[:, None],
            cos_om=jnp.cos(omega_r)[:, None])
    else:
        V = predict_velocity_los(r_b, sp, cp, D_A, M_BH, v_sys, sin_i_b)

    dpad = (slice(None), None, None)
    neg_half_chi2 = neg_half_chi2_position(
        x_obs[dpad], y_obs[dpad], X[None], Y[None],
        var_x[dpad], var_y[dpad])
    neg_half_chi2 = neg_half_chi2 + neg_half_chi2_velocity(
        v_obs[dpad], V[None], var_v[dpad])
    if has_accel:
        has_a = jnp.ones_like(a_obs)
        neg_half_chi2 = neg_half_chi2 + neg_half_chi2_acceleration(
            a_obs[dpad], A[None], var_a[dpad], has_a[dpad])

    lnorm = -0.5 * (3 * LOG_2PI + jnp.log(var_x) + jnp.log(var_y) +
                    jnp.log(var_v))
    if has_accel:
        lnorm = lnorm - 0.5 * (LOG_2PI + jnp.log(var_a))

    log_f = lnorm[:, None, None] + neg_half_chi2
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

    r_min, r_max = model.r_ang_range(D_A)

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


def _production_ll_mode2(model, phys_args, phys_kw):
    """Total ll_disk under the production Mode 2 phi/r marginal."""
    D_A = phys_args[2]
    M_BH = phys_args[3]
    v_sys = phys_args[4]
    i0 = phys_args[8]
    var_v_hv = phys_args[15]
    sigma_a_floor2 = phys_args[16]
    groups = model._build_r_grids_mode2(
        D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv)
    ll = model._eval_phi_marginal(groups, phys_args, phys_kw)
    return float(jnp.sum(ll))


def _production_ll_mode1(model, phys_args, phys_kw, r_ang):
    """Per-type total ll_disk under the production Mode 1 phi marginal."""
    groups = []
    if model._n_sys > 0:
        groups.append(
            ("sys", model._idx_sys, r_ang[model._idx_sys], None, False))
    if model._n_red > 0:
        groups.append(
            ("red", model._idx_red, r_ang[model._idx_red], None, False))
    if model._n_blue > 0:
        groups.append(
            ("blue", model._idx_blue, r_ang[model._idx_blue], None, False))
    ll = model._eval_phi_marginal(groups, phys_args, phys_kw)
    return dict(
        sys=float(jnp.sum(ll[model._idx_sys])),
        red=float(jnp.sum(ll[model._idx_red])),
        blue=float(jnp.sum(ll[model._idx_blue])),
        total=float(jnp.sum(ll)))


def _select_draws(samples, n_draws, seed):
    """Pick `n_draws` random indices into the posterior. Returns (idx_list,
    list-of-per-draw-sample-dicts) where each per-draw dict has no leading
    draw axis."""
    some_key = next(iter(samples))
    n_total = int(np.asarray(samples[some_key]).shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=min(n_draws, n_total), replace=False)
    idx = np.sort(idx)
    draws = []
    for i in idx:
        d = {k: np.asarray(v)[i] for k, v in samples.items()}
        draws.append(d)
    return idx.tolist(), draws


def check_convergence(model, samples, conv_cfg):
    """Compare production vs brute-force at `n_draws` posterior samples.

    Parameters
    ----------
    model : MaserDiskModel (built at the production grids).
    samples : dict from NUTS/NSS, each value with a leading draw axis.
    conv_cfg : the [convergence] config dict.

    Returns a dict with fields:
        mode, n_draws, draw_idx, deltas, mean, std,
        ll_prod, ll_ref, per_type (mode1 only), ref_cfg, wall_seconds.
    """
    import time
    n_draws = int(conv_cfg.get("n_draws", 8))
    seed = int(conv_cfg.get("seed", 0))
    idx, draws = _select_draws(samples, n_draws, seed)

    mode = model.mode
    t0 = time.time()
    if mode == "mode2":
        ref_cfg = conv_cfg["mode2_reference"]
        ll_prod, ll_ref, deltas = [], [], []
        for d in draws:
            pa, pk, _ = model.phys_from_sample(d)
            p = _production_ll_mode2(model, pa, pk)
            r = bruteforce_ll_mode2(model, pa, pk, ref_cfg)
            ll_prod.append(p)
            ll_ref.append(r)
            deltas.append(p - r)
        result = dict(
            mode=mode, n_draws=len(draws), draw_idx=idx,
            ll_prod=ll_prod, ll_ref=ll_ref, deltas=deltas,
            mean=float(np.mean(deltas)),
            std=float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            ref_cfg=dict(ref_cfg),
        )
    elif mode == "mode1":
        ref_cfg = conv_cfg["mode1_reference"]
        if "r_ang" not in samples:
            raise KeyError(
                "Mode 1 convergence check expects 'r_ang' in samples.")
        r_ang_all = np.asarray(samples["r_ang"])
        per_type = dict(sys=[], red=[], blue=[], total=[])
        for i, d in zip(idx, draws):
            pa, pk, _ = model.phys_from_sample(d)
            r_ang = jnp.asarray(r_ang_all[i])
            prod = _production_ll_mode1(model, pa, pk, r_ang)
            ref = bruteforce_ll_mode1(model, pa, pk, r_ang, ref_cfg)
            for k in ("sys", "red", "blue", "total"):
                per_type[k].append(prod[k] - ref[k])
        deltas = per_type["total"]
        result = dict(
            mode=mode, n_draws=len(draws), draw_idx=idx,
            deltas=deltas,
            mean=float(np.mean(deltas)),
            std=float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            per_type={k: dict(
                mean=float(np.mean(v)),
                std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                values=v) for k, v in per_type.items()},
            ref_cfg=dict(ref_cfg),
        )
    else:
        raise ValueError(f"Unsupported mode for convergence check: {mode}")

    result["wall_seconds"] = time.time() - t0
    return result


def summarize(result):
    """Print a human-readable block to stdout."""
    mode = result["mode"]
    print("=" * 70)
    print(f"Convergence check - {mode}, {result['n_draws']} draws "
          f"({result['wall_seconds']:.1f}s)")
    print(f"  draw indices: {result['draw_idx']}")
    deltas = result["deltas"]
    delta_str = "  ".join(f"{d:+.3f}" for d in deltas)
    print(f"  delta per draw [nats]: {delta_str}")
    mean = result['mean']
    std = result['std']
    print(f"  mean +/- std: {mean:+.3f} +/- {std:.3f} nats")
    if mode == "mode1":
        for k in ("sys", "red", "blue"):
            m = result["per_type"][k]["mean"]
            s = result["per_type"][k]["std"]
            print(f"    {k:>4}: {m:+.3f} +/- {s:.3f} nats")
    print("=" * 70, flush=True)


# -----------------------------------------------------------------------
# Test-harness helpers (used by the sweep scripts in scripts/megamaser/).
# They build a MaserDiskModel with per-call grid overrides so the sweep
# can vary phi/r grid sizes while holding all other config constant.
# -----------------------------------------------------------------------

def build_model(galaxy, master_cfg, **overrides):
    """Build a MaserDiskModel with global [model] keys overridden.

    Any recognised [model] key may be passed (n_phi_hv_high, n_phi_hv_low,
    n_phi_sys, phi_hv_inner_deg, phi_hv_outer_deg, phi_sys_ranges_deg,
    n_r_local, n_r_brute, K_sigma, mode, ...).  Per-galaxy settings in
    the config normally override globals; for the convergence tests we
    want the GLOBAL values to win, so we temporarily strip the galaxy's
    Mode-1 phi keys from the config copy passed to the model.
    """
    import os
    import tempfile

    import tomli_w

    from candel.pvdata.megamaser_data import load_megamaser_spots

    cfg = {k: (v.copy() if isinstance(v, dict) else v)
           for k, v in master_cfg.items()}
    cfg["model"] = dict(master_cfg["model"])
    cfg["model"]["galaxies"] = {
        g: dict(blk) for g, blk in master_cfg["model"]["galaxies"].items()}
    gblk = cfg["model"]["galaxies"][galaxy]

    for key in ("n_phi_hv_high", "n_phi_hv_low", "n_phi_sys",
                "phi_hv_inner_deg", "phi_hv_outer_deg",
                "phi_sys_ranges_deg",
                "n_r_local", "n_r_brute", "K_sigma",
                "mode"):
        gblk.pop(key, None)
        for suffix in ("_mode1", "_mode2"):
            cfg["model"].pop(key + suffix, None)

    for k, v in overrides.items():
        cfg["model"][k] = v

    data = load_megamaser_spots(
        master_cfg["io"]["maser_data"]["root"], galaxy=galaxy,
        v_sys_obs=master_cfg["model"]["galaxies"][galaxy]["v_sys_obs"])
    for key in ("D_lo", "D_hi"):
        if key in master_cfg["model"]["galaxies"][galaxy]:
            data[key] = float(master_cfg["model"]["galaxies"][galaxy][key])
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(cfg, tmp)
    tmp.close()
    from candel.model.model_H0_maser import MaserDiskModel
    model = MaserDiskModel(tmp.name, data)
    os.unlink(tmp.name)
    return model


def get_default_grid(master_cfg):
    """Return the current global [model] phi/r grid defaults as a dict."""
    m = master_cfg["model"]
    return dict(
        n_hv_high=int(m["n_phi_hv_high"]),
        n_hv_low=int(m["n_phi_hv_low"]),
        n_sys=int(m["n_phi_sys"]),
        n_r_local=int(m["n_r_local"]),
        n_r_brute=int(m["n_r_brute"]),
    )
