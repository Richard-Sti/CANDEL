"""Benchmark alternative r-grid spacing strategies for the maser disk integral.

Tests: log-uniform (baseline), Gauss-Legendre in log(r), spot-resonant,
sinh-spacing, and power-law (1/sqrt(r) uniform).

For each strategy finds the minimum n_r achieving |ΔlogL| < 0.01 nats
vs the n_r=1001 log-uniform reference on three representative spots.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import tempfile
import tomli_w

jax.config.update("jax_enable_x64", True)

from candel.util import patch_tqdm, fprint, fsection
patch_tqdm()
from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.model.integration import trapz_log_weights
from candel.model.model_H0_maser import MaserDiskModel, C_v

# ── Fixed parameters (NGC5765b posterior centre) ─────────────────────────────
D_A = 121.0
M_BH = 10**7.655
x0 = -45.7e-3
y0 = -98.2e-3
v_sys = 8333.0
i0 = jnp.deg2rad(84.95)
di_dr = jnp.deg2rad(11.5)
Omega0 = jnp.deg2rad(146.85)
dOmega_dr = jnp.deg2rad(-3.52)
r_ang_ref = 0.970
sigma_x_floor2 = (11.0e-3)**2
sigma_y_floor2 = (3.5e-3)**2
var_v_sys = 2.3**2
var_v_hv = 5.7**2
sigma_a_floor2 = 0.073**2

# ── Data ─────────────────────────────────────────────────────────────────────
data = load_megamaser_spots("data/Megamaser", "NGC5765b", v_sys_obs=8327.6)
n_spots = data["n_spots"]

is_hv = np.asarray(data["is_highvel"])
is_blue = np.asarray(data.get("is_blue", np.zeros(n_spots, bool)))
is_red = is_hv & ~is_blue
accel = np.asarray(data.get("accel_measured", np.ones(n_spots, bool)))

SPOT_INDICES = {
    "sys":  int(np.where(~is_hv & accel)[0][0]),
    "red":  int(np.where(is_red  & accel)[0][0]),
    "blue": int(np.where(is_blue & accel)[0][0]),
}
fprint("Representative spots: " +
       ", ".join(f"{k}=#{v}" for k, v in SPOT_INDICES.items()))

R_MIN, R_MAX = 0.01, 3.0


def build_model(n_r=251):
    """Build MaserDiskModel with fine phi grids, variable r grid size."""
    config = {
        "inference": {"num_warmup": 1, "num_samples": 1, "num_chains": 1,
                      "chain_method": "sequential", "seed": 42,
                      "init_maxiter": 0, "max_tree_depth": 5,
                      "init_method": "median"},
        "model": {
            "which_run": "maser_disk", "Om": 0.315,
            "use_selection": False, "marginalise_r": True,
            "G_phi_half": 1001, "n_inner_sys": 501, "inner_deg_sys": 15.0, "n_wing_sys": 50, "n_r": n_r,
            "priors": {
                "H0":          {"dist": "delta", "value": 73.0},
                "sigma_pec":   {"dist": "delta", "value": 250.0},
                "D":           {"dist": "uniform", "low": 50.0, "high": 200.0},
                "log_MBH":     {"dist": "uniform", "low": 6.5,  "high": 9.0},
                "eta":         {"dist": "uniform", "low": 3.0,  "high": 8.0},
                "R_phys":      {"dist": "uniform", "low": R_MIN, "high": R_MAX},
                "x0":          {"dist": "normal",  "loc": 0.0, "scale": 150.0},
                "y0":          {"dist": "normal",  "loc": 0.0, "scale": 150.0},
                "i0":          {"dist": "uniform", "low": 60.0, "high": 110.0},
                "Omega0":      {"dist": "uniform", "low": 0.0,  "high": 360.0},
                "dOmega_dr":   {"dist": "uniform", "low": -20.0, "high": 20.0},
                "di_dr":       {"dist": "uniform", "low": -30.0, "high": 30.0},
                "dv_sys":      {"dist": "normal",  "loc": 0.0,  "scale": 25.0},
                "sigma_x_floor": {"dist": "truncated_normal", "mean": 10.0,
                                  "scale": 5.0, "low": 0.0, "high": 100.0},
                "sigma_y_floor": {"dist": "truncated_normal", "mean": 10.0,
                                  "scale": 5.0, "low": 0.0, "high": 100.0},
                "sigma_v_sys":   {"dist": "truncated_normal", "mean": 2.0,
                                  "scale": 5.0, "low": 0.0, "high": 100.0},
                "sigma_v_hv":    {"dist": "truncated_normal", "mean": 2.0,
                                  "scale": 5.0, "low": 0.0, "high": 100.0},
                "sigma_a_floor": {"dist": "truncated_normal", "mean": 0.1,
                                  "scale": 0.2, "low": 0.0, "high": 0.3},
            },
        },
        "io": {"fname_output": "/dev/null"},
    }
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(config, tmp)
    tmp.close()
    m = MaserDiskModel(tmp.name, data)
    os.unlink(tmp.name)
    return m


def evall(m, r_grid, log_w_r):
    """Evaluate per-spot log-likelihoods for the three representative spots."""
    r_ang = jnp.asarray(r_grid)[None, :].repeat(n_spots, axis=0)
    ll = m._eval_marginal_phi(
        r_ang, x0, y0, D_A, M_BH, v_sys, m._r_ang_ref,
        i0, di_dr, Omega0, dOmega_dr,
        sigma_x_floor2, sigma_y_floor2,
        var_v_sys, var_v_hv, sigma_a_floor2,
        log_w_r=jnp.asarray(log_w_r))
    ll = np.asarray(jax.block_until_ready(ll))
    return np.array([ll[SPOT_INDICES["sys"]],
                     ll[SPOT_INDICES["red"]],
                     ll[SPOT_INDICES["blue"]]])


# ── Build reference model (n_r=1001, log-uniform) ────────────────────────────
fsection("Building reference model (n_r=1001)")
m_ref = build_model(1001)
r_lo = float(m_ref._r_ang_lo)
r_hi = float(m_ref._r_ang_hi)
fprint(f"r_ang grid: [{r_lo:.5f}, {r_hi:.5f}] mas")

from candel.model.integration import trapz_log_weights
r_ref = m_ref.build_r_ang_grid(D_A)
log_w_ref = np.asarray(trapz_log_weights(jnp.asarray(r_ref)))
ll_ref = evall(m_ref, r_ref, log_w_ref)
fprint(f"Reference logL: sys={ll_ref[0]:.4f}, red={ll_ref[1]:.4f}, blue={ll_ref[2]:.4f}")

logr_lo = np.log(r_lo)
logr_hi = np.log(r_hi)

# ── Grid construction helpers ─────────────────────────────────────────────────


def log_uniform_grid(n):
    r_grid = np.logspace(np.log10(r_lo), np.log10(r_hi), n)
    log_w_r = np.asarray(trapz_log_weights(r_grid))
    return r_grid, log_w_r


def gl_logr_grid_and_weights(n):
    """Gauss-Legendre nodes/weights mapped from [-1,1] to log(r) ∈ [logr_lo, logr_hi].

    The integral is  ∫ f(r) dr  with change of variables u = log(r):
      ∫ f(r) dr = ∫ f(e^u) * e^u du

    GL quadrature approximates ∫_{logr_lo}^{logr_hi} g(u) du where g(u) = f(e^u)*e^u.
    The Jacobian e^u = r is absorbed into the quadrature weight:
      w_r[i] = w_gl[i] * (logr_hi - logr_lo) / 2 * r[i]

    log_w_r[i] = log(w_gl[i]) + log((logr_hi - logr_lo) / 2) + log(r[i])
    """
    xi, w_gl = np.polynomial.legendre.leggauss(n)
    # Map [-1,1] -> [logr_lo, logr_hi]
    log_r = 0.5 * (logr_hi - logr_lo) * xi + 0.5 * (logr_hi + logr_lo)
    r_grid = np.exp(log_r)
    half_range = 0.5 * (logr_hi - logr_lo)
    log_w_r = np.log(w_gl) + np.log(half_range) + log_r  # = log(w_gl * half_range * r)
    return r_grid, log_w_r


def sinh_grid(n, scale=1.0):
    """r = exp(sinh(t) * scale), t uniform on [t_lo, t_hi].

    Mapped so r(t_lo) = r_lo, r(t_hi) = r_hi.
    """
    t_lo = np.arcsinh(logr_lo / scale)
    t_hi = np.arcsinh(logr_hi / scale)
    t = np.linspace(t_lo, t_hi, n)
    log_r = np.sinh(t) * scale
    r_grid = np.exp(log_r)
    # dr/dt = cosh(t) * scale * r, trapz weights in t × dr/dt
    dr_dt = np.cosh(t) * scale * r_grid
    w = np.zeros(n)
    h = np.diff(t)
    w[0] = h[0] / 2 * dr_dt[0]
    w[-1] = h[-1] / 2 * dr_dt[-1]
    w[1:-1] = (h[:-1] + h[1:]) / 2 * dr_dt[1:-1]
    log_w_r = np.log(w)
    return r_grid, log_w_r


def power_law_grid(n, alpha=0.5):
    """r ∝ t^(1/alpha), t uniform — gives dr/dt ∝ r^(1-alpha).

    alpha=0.5: uniform in 1/sqrt(r) → dense at small r.
    alpha=2.0: uniform in r^0.5    → denser at large r.
    alpha=1.0: uniform in r        → linear spacing.
    """
    t_lo = r_lo**alpha
    t_hi = r_hi**alpha
    t = np.linspace(t_lo, t_hi, n)
    r_grid = t**(1.0 / alpha)
    # dr/dt = (1/alpha) * t^(1/alpha - 1) = r^(1-alpha) / alpha
    dr_dt = r_grid**(1.0 - alpha) / alpha
    w = np.zeros(n)
    h = np.diff(t)
    w[0] = h[0] / 2 * dr_dt[0]
    w[-1] = h[-1] / 2 * dr_dt[-1]
    w[1:-1] = (h[:-1] + h[1:]) / 2 * dr_dt[1:-1]
    log_w_r = np.log(w)
    return r_grid, log_w_r


def resonant_grid(n, n_dense=30, blend_factor=3.0):
    """Grid dense near resonant radii of HV spots, coarser elsewhere.

    Resonant radius: r_peak = M_BH / (D_A * (|dv|/C_v)^2)
    Uses a mixture approach: n_dense points near each peak (within
    blend_factor * local spacing), rest log-uniform.
    """
    # Compute resonant radii for HV spots
    v_obs = np.asarray(data["velocity"])
    v_sys_obs = float(data["v_sys_obs"])
    dv = np.abs(v_obs[is_hv] - v_sys_obs)
    # Avoid zero division; only take spots with measurable dv
    dv = np.where(dv > 1.0, dv, np.nan)
    r_peaks = M_BH / (D_A * (dv / C_v)**2)
    r_peaks = r_peaks[np.isfinite(r_peaks)]
    # Clip to grid range
    r_peaks = np.clip(r_peaks, r_lo * 1.01, r_hi * 0.99)
    r_peaks = np.unique(np.round(r_peaks, 5))

    # Start with coarse log-uniform backbone
    n_coarse = max(n - n_dense * len(r_peaks), 20)
    r_base = np.logspace(np.log10(r_lo), np.log10(r_hi), n_coarse)

    # Add dense patches around each resonant radius
    pieces = [r_base]
    for rp in r_peaks:
        # Width in log(r) proportional to local spacing of coarse grid
        i_near = np.searchsorted(r_base, rp)
        i_near = np.clip(i_near, 1, len(r_base) - 1)
        delta_logr = (np.log(r_base[i_near]) - np.log(r_base[i_near - 1]))
        half_w = blend_factor * delta_logr
        r_lo_p = max(rp * np.exp(-half_w), r_lo * 1.001)
        r_hi_p = min(rp * np.exp(half_w), r_hi * 0.999)
        patch = np.logspace(np.log10(r_lo_p), np.log10(r_hi_p), n_dense)
        pieces.append(patch)

    r_grid = np.unique(np.concatenate(pieces))
    log_w_r = np.asarray(trapz_log_weights(r_grid))
    return r_grid, log_w_r


# ── Sweep helper ──────────────────────────────────────────────────────────────

def sweep_strategy(label, grid_fn, ns, extra_info=""):
    """Sweep n values for a grid strategy, return minimum n for |ΔlogL|<0.01."""
    print(f"\n{'─'*70}")
    print(f"Strategy: {label}  {extra_info}")
    print(f"  {'n':>6}  {'Δsys':>10}  {'Δred':>10}  {'Δblue':>10}  "
          f"{'max|Δ|':>10}  {'pass?':>6}")
    best = None
    for n in ns:
        r_grid, log_w_r = grid_fn(n)
        ll = evall(m_ref, r_grid, log_w_r)
        delta = ll - ll_ref
        max_err = float(np.max(np.abs(delta)))
        ok = max_err < 0.01
        print(f"  {n:>6}  {delta[0]:+10.4f}  {delta[1]:+10.4f}  {delta[2]:+10.4f}  "
              f"{max_err:10.5f}  {'OK' if ok else '--':>6}")
        if ok and best is None:
            best = (n, delta.copy(), max_err, len(r_grid))
    if best is not None:
        fprint(f"  → PASSES at n={best[0]} (actual grid size {best[3]}) "
               f"max|Δ|={best[2]:.5f}")
    else:
        fprint(f"  → FAILS for all tested n")
    return best


# ── Strategy 0: log-uniform baseline ─────────────────────────────────────────
fsection("Strategy 0: log-uniform baseline (reference strategy)")
sweep_strategy("log-uniform", log_uniform_grid,
               [51, 71, 101, 151, 201, 251, 301, 401, 501])

# ── Strategy 1: Gauss-Legendre in log(r) ─────────────────────────────────────
fsection("Strategy 1: Gauss-Legendre in log(r)")
sweep_strategy("GL-logr", gl_logr_grid_and_weights,
               [21, 31, 41, 51, 61, 71, 81, 101, 121, 151])

# ── Strategy 2: Spot-resonant grid ───────────────────────────────────────────
fsection("Strategy 2: Spot-resonant grid")
# n is total target size; resonant patches are added on top of coarse backbone
sweep_strategy("resonant", resonant_grid,
               [51, 71, 101, 151, 201, 251])

# ── Strategy 3: Sinh-spacing ─────────────────────────────────────────────────
fsection("Strategy 3: sinh-spacing (scale=1)")
sweep_strategy("sinh(scale=1)", lambda n: sinh_grid(n, scale=1.0),
               [51, 71, 101, 151, 201, 251, 301])

fsection("Strategy 3b: sinh-spacing (scale=0.5)")
sweep_strategy("sinh(scale=0.5)", lambda n: sinh_grid(n, scale=0.5),
               [51, 71, 101, 151, 201, 251, 301])

# ── Strategy 4: Power-law spacing ────────────────────────────────────────────
fsection("Strategy 4a: power-law alpha=0.5 (uniform in 1/sqrt(r))")
sweep_strategy("power(alpha=0.5)", lambda n: power_law_grid(n, alpha=0.5),
               [51, 71, 101, 151, 201, 251, 301, 401, 501])

fsection("Strategy 4b: power-law alpha=0.25 (stronger small-r bias)")
sweep_strategy("power(alpha=0.25)", lambda n: power_law_grid(n, alpha=0.25),
               [51, 71, 101, 151, 201, 251, 301, 401, 501])

fsection("Strategy 4c: power-law alpha=0.75")
sweep_strategy("power(alpha=0.75)", lambda n: power_law_grid(n, alpha=0.75),
               [51, 71, 101, 151, 201, 251, 301, 401, 501])

# ── Strategy 5: GL in sqrt(r) ─────────────────────────────────────────────────
fsection("Strategy 5: Gauss-Legendre in sqrt(r)")

def gl_sqrtr_grid(n):
    """GL nodes/weights in sqrt(r) space.

    Change of variables: u = sqrt(r), r = u^2, dr = 2u du.
    log_w_r[i] = log(w_gl[i]) + log(half_range) + log(2 * u[i]) + log(u[i])
               = log(w_gl[i]) + log(half_range) + log(2) + 2*log(u[i])
    """
    u_lo = np.sqrt(r_lo)
    u_hi = np.sqrt(r_hi)
    xi, w_gl = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (u_hi - u_lo) * xi + 0.5 * (u_hi + u_lo)
    r_grid = u**2
    half_range = 0.5 * (u_hi - u_lo)
    # dr = 2u du => w_r = w_gl * half_range * 2u
    log_w_r = np.log(w_gl) + np.log(half_range) + np.log(2 * u)
    return r_grid, log_w_r

sweep_strategy("GL-sqrtr", gl_sqrtr_grid,
               [21, 31, 41, 51, 61, 71, 81, 101, 121, 151])

# ── Strategy 6: GL in r (linear space) ───────────────────────────────────────
fsection("Strategy 6: Gauss-Legendre in r (linear)")

def gl_r_grid(n):
    xi, w_gl = np.polynomial.legendre.leggauss(n)
    half_range = 0.5 * (r_hi - r_lo)
    r_grid = half_range * xi + 0.5 * (r_hi + r_lo)
    log_w_r = np.log(w_gl) + np.log(half_range) * np.ones(n)
    return r_grid, log_w_r

sweep_strategy("GL-r", gl_r_grid,
               [21, 31, 41, 51, 61, 71, 81, 101, 121, 151, 201])

# ── Strategy 7: GL with v_kep-weighted (1/r) variable ──────────────────────
fsection("Strategy 7: Gauss-Legendre in 1/sqrt(r)")

def gl_inv_sqrtr_grid(n):
    """GL in u=1/sqrt(r): r = 1/u^2, dr = -2/u^3 du.

    u_lo = 1/sqrt(r_hi), u_hi = 1/sqrt(r_lo) (reversed: r increases as u decreases).
    Swap to keep ascending r order.
    log_w_r[i] = log(w_gl[i]) + log(half_range) + log(2 / u[i]^3)
    """
    u_lo = 1.0 / np.sqrt(r_hi)
    u_hi = 1.0 / np.sqrt(r_lo)
    xi, w_gl = np.polynomial.legendre.leggauss(n)
    half_range = 0.5 * (u_hi - u_lo)
    u = 0.5 * (u_hi - u_lo) * xi + 0.5 * (u_hi + u_lo)
    r_grid = 1.0 / u**2
    log_w_r = np.log(w_gl) + np.log(half_range) + np.log(2.0 / u**3)
    # Sort by ascending r
    idx = np.argsort(r_grid)
    return r_grid[idx], log_w_r[idx]

sweep_strategy("GL-inv-sqrtr", gl_inv_sqrtr_grid,
               [21, 31, 41, 51, 61, 71, 81, 101, 121, 151])

# ── Strategy 8: Clustered near data r_ang values ─────────────────────────────
fsection("Strategy 8: Data-clustered grid (dense near observed HV r_ang values)")

def data_clustered_grid(n, sigma_rel=0.5, n_peak_pts=8):
    """Log-uniform backbone + fine patches at observed HV angular radii.

    Each HV spot contributes n_peak_pts points within ±sigma_rel in dex.
    """
    x_hv = np.asarray(data["x"])[is_hv]
    y_hv = np.asarray(data["y"])[is_hv]
    r_obs = np.sqrt(x_hv**2 + y_hv**2)
    r_obs = np.clip(r_obs, r_lo * 1.001, r_hi * 0.999)

    n_coarse = max(n - n_peak_pts * len(r_obs), 20)
    r_base = np.logspace(np.log10(r_lo), np.log10(r_hi), n_coarse)

    pieces = [r_base]
    for rp in r_obs:
        r_lo_p = max(rp * 10**(-sigma_rel), r_lo * 1.001)
        r_hi_p = min(rp * 10**(sigma_rel), r_hi * 0.999)
        patch = np.logspace(np.log10(r_lo_p), np.log10(r_hi_p), n_peak_pts)
        pieces.append(patch)

    r_grid = np.unique(np.concatenate(pieces))
    log_w_r = np.asarray(trapz_log_weights(r_grid))
    return r_grid, log_w_r

sweep_strategy("data-clustered", data_clustered_grid,
               [51, 101, 151, 201, 251])

# ── Summary ───────────────────────────────────────────────────────────────────
fsection("Done. Review results above for minimum n_r per strategy.")
