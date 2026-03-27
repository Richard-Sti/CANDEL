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
"""Megamaser disk forward model for H0 inference.

Implements the warped Keplerian disk model from Pesce et al. (2020),
arXiv:2001.04581. All JAX functions are JIT-compilable and auto-differentiable.

Per-spot (r, phi) are marginalised numerically:

    ll_k = log int L_k(r, phi) dr dphi

using Strategy D (adaptive per-spot grid, recentered every MCMC step):

1. Find peak (r*, phi*) for each spot using an analytical initial guess,
   Gauss--Newton refinement with autodiff Jacobians, and a scan-grid
   hard-argmax for robustness.

2. Build per-spot local grids centred at (r*_k, phi*_k) with fixed shape
   (Nr x Nphi). Different centres are fine under vmap; only the shape
   must be uniform.

3. Evaluate the full log-integrand on the grid and integrate with
   pre-computed Simpson weights in log-space.

All angles are in RADIANS inside physics functions.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm as jax_norm
from numpyro import factor

from ..util import SPEED_OF_LIGHT, fprint, fsection, get_nested
from .base_model import ModelBase
from .pv_utils import rsample
from .simpson import simpson_log_weights, ln_simpson_precomputed
from .utils import normal_logpdf_var


# -----------------------------------------------------------------------
# Disk physics constants
# -----------------------------------------------------------------------

C_v = 0.9420       # km/s: sqrt(GM_sun / (1 mas * 1 Mpc)) * 1e-3
C_a = 1.872e-4     # km/s/yr: GM_sun * yr / ((1 mas * 1 Mpc)^2 * 1e3)
C_g = 1.974e-11    # dimensionless: 2*GM_sun / (c^2 * 1 mas * 1 Mpc)


# -----------------------------------------------------------------------
# Disk physics functions
# -----------------------------------------------------------------------


def warp_geometry(r, i0_rad, di_dr_rad, Omega0_rad, dOmega_dr_rad):
    """Evaluate warped inclination and position angle at radius r.

    Parameters
    ----------
    r : radius in mas
    i0_rad, di_dr_rad : inclination at r=0 and warp rate, both in radians
    Omega0_rad, dOmega_dr_rad : position angle at r=0 and warp rate, radians

    Returns
    -------
    i, Omega : inclination and position angle at r, in radians
    """
    i = i0_rad + di_dr_rad * r
    Omega = Omega0_rad + dOmega_dr_rad * r
    return i, Omega


def predict_position(r, phi, x0, y0, i, Omega):
    """Predict sky-plane position of maser spots.

    All angles (phi, i, Omega) in radians. Positions in mas.

    Returns
    -------
    X, Y : predicted sky-plane coordinates in mas
    """
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    sin_O = jnp.sin(Omega)
    cos_O = jnp.cos(Omega)
    cos_i = jnp.cos(i)

    X = x0 + r * (sin_phi * sin_O - cos_phi * cos_O * cos_i)
    Y = y0 + r * (sin_phi * cos_O + cos_phi * sin_O * cos_i)
    return X, Y


def predict_velocity_los(r, phi, D, M_BH, v_sys, i, Omega):
    """Predict line-of-sight velocity of maser spots (optical convention).

    Includes Keplerian orbital velocity, relativistic Doppler, gravitational
    redshift, and systemic redshift.

    Parameters
    ----------
    r : orbital radius in mas
    phi : azimuthal angle in radians
    D : angular-diameter distance in Mpc
    M_BH : black hole mass in M_sun
    v_sys : systemic velocity in km/s
    i : inclination in radians
    Omega : position angle in radians (unused for circular orbits)

    Returns
    -------
    V_obs : observed velocity in km/s (optical convention)
    """
    v_kep = C_v * jnp.sqrt(M_BH / (r * D))

    v_z = v_kep * jnp.sin(phi) * jnp.sin(i)

    beta = v_kep / SPEED_OF_LIGHT
    gamma = 1.0 / jnp.sqrt(1.0 - beta**2)
    one_plus_z_D = gamma * (1.0 + v_z / SPEED_OF_LIGHT)

    one_plus_z_g = 1.0 / jnp.sqrt(1.0 - C_g * M_BH / (r * D))

    z_0 = v_sys / SPEED_OF_LIGHT

    V_obs = SPEED_OF_LIGHT * ((one_plus_z_D) * (one_plus_z_g) * (1.0 + z_0) - 1.0)
    return V_obs


def predict_acceleration_los(r, phi, D, M_BH, i):
    """Predict line-of-sight centripetal acceleration.

    Parameters
    ----------
    r : orbital radius in mas
    phi : azimuthal angle in radians
    D : angular-diameter distance in Mpc
    M_BH : black hole mass in M_sun
    i : inclination in radians

    Returns
    -------
    A_z : line-of-sight acceleration in km/s/yr
    """
    a_mag = C_a * M_BH / (r**2 * D**2)
    A_z = a_mag * jnp.cos(phi) * jnp.sin(i)
    return A_z


# -----------------------------------------------------------------------
# Marginalisation: default grid configuration
# -----------------------------------------------------------------------

DEFAULT_DELTA_R = 0.15     # mas half-width
DEFAULT_DELTA_PHI = 0.50   # rad half-width
DEFAULT_NR = 21            # odd for Simpson
DEFAULT_NPHI = 31          # odd for Simpson

SCAN_DELTA_R = 0.15        # mas half-width for scan
SCAN_DELTA_PHI = 0.50      # rad half-width for scan
SCAN_NR = 11
SCAN_NPHI = 15


def make_relative_grids(Nr=DEFAULT_NR, Nphi=DEFAULT_NPHI,
                        delta_r=DEFAULT_DELTA_R, delta_phi=DEFAULT_DELTA_PHI):
    """Build relative grid offsets and Simpson weights (computed once).

    Returns
    -------
    dr_offsets : (Nr,) offsets centred at 0
    dphi_offsets : (Nphi,) offsets centred at 0
    log_wr : (Nr,) log Simpson weights for r
    log_wphi : (Nphi,) log Simpson weights for phi
    """
    dr = jnp.linspace(-delta_r, delta_r, Nr)
    dphi = jnp.linspace(-delta_phi, delta_phi, Nphi)
    log_wr = simpson_log_weights(dr)
    log_wphi = simpson_log_weights(dphi)
    return dr, dphi, log_wr, log_wphi


# -----------------------------------------------------------------------
# Peak finder: analytical initial guess + scan-grid soft-argmax
# -----------------------------------------------------------------------

def _analytical_init(x_obs, y_obs, v_obs, a_obs, accel_measured,
                     phi_lo, phi_hi,
                     x0, y0, D, M_BH, v_sys,
                     i0, di_dr, Omega0, dOmega_dr,
                     sigma_v_sys, sigma_v_hv,
                     sigma_a_obs, sigma_a_floor):
    """Physics-based initial guess for (r, phi) per spot.

    Uses position geometry to estimate r, then scores 4 phi candidates
    (arcsin branches + 2pi wraps) against velocity + acceleration residuals.
    """
    sin_O0, cos_O0 = jnp.sin(Omega0), jnp.cos(Omega0)
    sin_i0 = jnp.sin(i0)

    dx = x_obs - x0
    dy = y_obs - y0
    u_from_pos = dx * sin_O0 + dy * cos_O0

    dv = v_obs - v_sys
    dv_safe = jnp.where(jnp.abs(dv) > 1.0, dv, jnp.sign(dv + 1e-20) * 1.0)
    r32 = C_v * jnp.sqrt(M_BH / D) * sin_i0 * u_from_pos / dv_safe
    r_init = jnp.clip(jnp.abs(r32) ** (2.0 / 3.0), 0.05, 5.0)

    sin_phi = jnp.clip(u_from_pos / r_init, -0.9999, 0.9999)
    phi_a = jnp.arcsin(sin_phi)
    phi_b = jnp.pi - phi_a

    candidates = jnp.stack([
        jnp.clip(phi_a, phi_lo, phi_hi),
        jnp.clip(phi_b, phi_lo, phi_hi),
        jnp.clip(2.0 * jnp.pi + phi_a, phi_lo, phi_hi),
        jnp.clip(2.0 * jnp.pi + phi_b - jnp.pi, phi_lo, phi_hi),
    ])  # (4, N_spots)

    i_at_r, O_at_r = warp_geometry(r_init, i0, di_dr, Omega0, dOmega_dr)

    def _score(phi_c):
        v_c = predict_velocity_los(r_init, phi_c, D, M_BH, v_sys, i_at_r, O_at_r)
        a_c = predict_acceleration_los(r_init, phi_c, D, M_BH, i_at_r)
        cos2_phi = jnp.cos(phi_c)**2
        sigma_v = jnp.sqrt(
            sigma_v_hv**2 + (sigma_v_sys**2 - sigma_v_hv**2) * cos2_phi)
        sigma_a = jnp.sqrt(sigma_a_obs**2 + sigma_a_floor**2)
        s = -0.5 * ((v_c - v_obs) / sigma_v)**2
        s += jnp.where(accel_measured, -0.5 * ((a_c - a_obs) / sigma_a)**2, 0.0)
        return s

    scores = jax.vmap(_score)(candidates)  # (4, N_spots)
    best_idx = jnp.argmax(scores, axis=0)  # (N_spots,)
    phi_init = candidates[best_idx, jnp.arange(candidates.shape[1])]

    return r_init, phi_init


def _gauss_newton_refine(r0, phi0, phi_lo, phi_hi,
                         x_obs, y_obs, v_obs, a_obs, accel_measured,
                         x0, y0, D, M_BH, v_sys,
                         i0, di_dr, Omega0, dOmega_dr,
                         sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
                         sigma_a_obs, sigma_a_floor,
                         n_iter=6):
    """Refine (r, phi) via Gauss-Newton with autodiff Jacobians.

    Minimises the weighted position + velocity residual sum-of-squares.
    Jacobians are computed via JAX autodiff instead of hand-coded
    expressions.
    """
    cos2_phi_gn = jnp.cos(phi0)**2
    sigma_v = jnp.sqrt(
        sigma_v_hv**2 + (sigma_v_sys**2 - sigma_v_hv**2) * cos2_phi_gn)

    def _predict_X(r, phi):
        i, Omega = warp_geometry(r, i0, di_dr, Omega0, dOmega_dr)
        X, _ = predict_position(r, phi, x0, y0, i, Omega)
        return X

    def _predict_Y(r, phi):
        i, Omega = warp_geometry(r, i0, di_dr, Omega0, dOmega_dr)
        _, Y = predict_position(r, phi, x0, y0, i, Omega)
        return Y

    def _predict_V(r, phi):
        i, Omega = warp_geometry(r, i0, di_dr, Omega0, dOmega_dr)
        return predict_velocity_los(r, phi, D, M_BH, v_sys, i, Omega)

    dXdr_fn = jax.grad(_predict_X, argnums=0)
    dXdp_fn = jax.grad(_predict_X, argnums=1)
    dYdr_fn = jax.grad(_predict_Y, argnums=0)
    dYdp_fn = jax.grad(_predict_Y, argnums=1)
    dVdr_fn = jax.grad(_predict_V, argnums=0)
    dVdp_fn = jax.grad(_predict_V, argnums=1)

    def _step(carry, _):
        r, phi = carry

        def _one_gn(r_k, phi_k, x_k, y_k, v_k, sv_k):
            X_pred = _predict_X(r_k, phi_k)
            Y_pred = _predict_Y(r_k, phi_k)
            V_pred = _predict_V(r_k, phi_k)

            dXdr = dXdr_fn(r_k, phi_k)
            dXdp = dXdp_fn(r_k, phi_k)
            dYdr = dYdr_fn(r_k, phi_k)
            dYdp = dYdp_fn(r_k, phi_k)
            dVdr = dVdr_fn(r_k, phi_k)
            dVdp = dVdp_fn(r_k, phi_k)

            w_x, w_y, w_v = 50.0, 20.0, 1.0 / sv_k
            fx = (X_pred - x_k) * w_x
            fy = (Y_pred - y_k) * w_y
            fv = (V_pred - v_k) * w_v
            dXdr *= w_x; dXdp *= w_x
            dYdr *= w_y; dYdp *= w_y
            dVdr *= w_v; dVdp *= w_v

            JtJ_00 = dXdr**2 + dYdr**2 + dVdr**2
            JtJ_01 = dXdr * dXdp + dYdr * dYdp + dVdr * dVdp
            JtJ_11 = dXdp**2 + dYdp**2 + dVdp**2
            Jtf_0 = dXdr * fx + dYdr * fy + dVdr * fv
            Jtf_1 = dXdp * fx + dYdp * fy + dVdp * fv

            reg = 1e-4 * (JtJ_00 + JtJ_11) + 1e-8
            JtJ_00 += reg; JtJ_11 += reg
            det = JtJ_00 * JtJ_11 - JtJ_01**2 + 1e-30

            dr = (JtJ_11 * Jtf_0 - JtJ_01 * Jtf_1) / det
            dphi = (-JtJ_01 * Jtf_0 + JtJ_00 * Jtf_1) / det
            return dr, dphi

        dr, dphi = jax.vmap(_one_gn)(r, phi, x_obs, y_obs, v_obs, sigma_v)

        step_mag = jnp.sqrt(dr**2 + dphi**2 + 1e-30)
        max_step = 0.3 * r + 0.5
        scale = jnp.where(step_mag > max_step, max_step / step_mag, 1.0)
        r_new = jnp.clip(r - scale * dr, 0.05, 5.0)
        phi_new = jnp.clip(phi - scale * dphi, phi_lo, phi_hi)
        return (r_new, phi_new), None

    (r, phi), _ = jax.lax.scan(_step, (r0, phi0), None, length=n_iter)
    return r, phi


# -----------------------------------------------------------------------
# Per-spot integrand evaluation
# -----------------------------------------------------------------------

def _spot_log_likelihood_on_grid(
        r_grid, phi_grid,
        x_obs_k, sigma_x_k, y_obs_k, sigma_y_k,
        v_obs_k, a_obs_k, sigma_a_k,
        accel_measured_k,
        x0, y0, D, M_BH, v_sys,
        i0, di_dr, Omega0, dOmega_dr,
        sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
        sigma_a_floor, A_thr, sigma_det):
    """Evaluate log-integrand on a 2D (r, phi) grid for one spot.

    Parameters
    ----------
    r_grid : (Nr,) absolute r values in mas
    phi_grid : (Nphi,) absolute phi values in rad

    Returns
    -------
    log_integrand : (Nr, Nphi)
    """
    r_2d = r_grid[:, None]
    phi_2d = phi_grid[None, :]

    i_at_r, Omega_at_r = warp_geometry(r_2d, i0, di_dr, Omega0, dOmega_dr)

    X_pred, Y_pred = predict_position(r_2d, phi_2d, x0, y0, i_at_r, Omega_at_r)
    V_pred = predict_velocity_los(r_2d, phi_2d, D, M_BH, v_sys, i_at_r, Omega_at_r)
    A_pred = predict_acceleration_los(r_2d, phi_2d, D, M_BH, i_at_r)

    var_x = sigma_x_k**2 + sigma_x_floor**2
    var_y = sigma_y_k**2 + sigma_y_floor**2
    ll_pos = (normal_logpdf_var(x_obs_k, X_pred, var_x)
              + normal_logpdf_var(y_obs_k, Y_pred, var_y))

    cos2_phi = jnp.cos(phi_2d)**2
    var_v = sigma_v_hv**2 + (sigma_v_sys**2 - sigma_v_hv**2) * cos2_phi
    ll_vel = normal_logpdf_var(v_obs_k, V_pred, var_v)

    var_a = sigma_a_k**2 + sigma_a_floor**2

    log_p_det = jax_norm.logcdf((jnp.abs(a_obs_k) - A_thr) / sigma_det)
    ll_measured = log_p_det + normal_logpdf_var(a_obs_k, A_pred, var_a)

    sigma_nondet = jnp.sqrt(sigma_det**2 + 0.25 + sigma_a_floor**2)
    ll_unmeasured = jax_norm.logcdf((A_thr - jnp.abs(A_pred)) / sigma_nondet)

    ll_acc = jnp.where(accel_measured_k, ll_measured, ll_unmeasured)

    return ll_pos + ll_vel + ll_acc


def _scan_peak(r_center, phi_center, phi_lo, phi_hi,
               x_obs, sigma_x, y_obs, sigma_y,
               v_obs, a_obs, sigma_a, accel_measured,
               x0, y0, D, M_BH, v_sys,
               i0, di_dr, Omega0, dOmega_dr,
               sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
               sigma_a_floor, A_thr, sigma_det):
    """Soft-argmax peak on a coarse grid around (r_center, phi_center).

    Evaluates the full log-likelihood on a coarse grid centered on the
    analytical init, then returns the softmax-weighted average as a
    differentiable, robust peak location.
    """
    dr_s = jnp.linspace(-SCAN_DELTA_R, SCAN_DELTA_R, SCAN_NR)
    dphi_s = jnp.linspace(-SCAN_DELTA_PHI, SCAN_DELTA_PHI, SCAN_NPHI)
    r_scan = jnp.clip(r_center[:, None] + dr_s[None, :], 0.01, 10.0)
    phi_scan = jnp.clip(phi_center[:, None] + dphi_s[None, :],
                        phi_lo[:, None], phi_hi[:, None])

    def _eval_one(rg, pg, xk, sxk, yk, syk, vk, ak, sak, amk):
        return _spot_log_likelihood_on_grid(
            rg, pg, xk, sxk, yk, syk, vk, ak, sak, amk,
            x0, y0, D, M_BH, v_sys,
            i0, di_dr, Omega0, dOmega_dr,
            sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
            sigma_a_floor, A_thr, sigma_det)

    log_scan = jax.vmap(_eval_one)(
        r_scan, phi_scan,
        x_obs, sigma_x, y_obs, sigma_y,
        v_obs, a_obs, sigma_a, accel_measured)

    N_spots = x_obs.shape[0]
    r_2d = jnp.broadcast_to(r_scan[:, :, None],
                             (N_spots, SCAN_NR, SCAN_NPHI))
    phi_2d = jnp.broadcast_to(phi_scan[:, None, :],
                               (N_spots, SCAN_NR, SCAN_NPHI))
    log_flat = log_scan.reshape(N_spots, -1)

    # Hard argmax: since stop_gradient is applied, softmax averaging
    # would only add bias without any gradient benefit.
    best_idx = jnp.argmax(log_flat, axis=-1)
    r_star = jax.lax.stop_gradient(
        r_2d.reshape(N_spots, -1)[jnp.arange(N_spots), best_idx])
    phi_star = jax.lax.stop_gradient(
        phi_2d.reshape(N_spots, -1)[jnp.arange(N_spots), best_idx])
    return r_star, phi_star


# -----------------------------------------------------------------------
# Main marginalisation routine
# -----------------------------------------------------------------------

def marginalise_spots(
        x_obs, sigma_x, y_obs, sigma_y,
        v_obs, a_obs, sigma_a,
        accel_measured,
        phi_lo, phi_hi,
        x0, y0, D, M_BH, v_sys,
        i0, di_dr, Omega0, dOmega_dr,
        sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
        sigma_a_floor, A_thr, sigma_det,
        dr_offsets, dphi_offsets, log_wr, log_wphi,
        bimodal=True):
    """Compute sum of marginalised log-likelihoods for all spots.

    Uses analytical init + autodiff G-N refinement + scan-grid argmax
    for grid centering. Mode 2 is centered on phi2 = pi - phi1
    (or 3pi - phi1 for blue spots).

    Parameters
    ----------
    x_obs, sigma_x, ... : (N_spots,) observed data
    phi_lo, phi_hi : (N_spots,) per-spot azimuthal bounds
    x0, y0, D, M_BH, ... : scalar model parameters
    dr_offsets, dphi_offsets : (Nr,), (Nphi,) relative grid offsets
    log_wr, log_wphi : (Nr,), (Nphi,) pre-computed Simpson log-weights
    bimodal : if True, integrate both modes via logaddexp

    Returns
    -------
    ll_total : scalar, sum_k log int L_k(r, phi) dr dphi
    r_star : (N_spots,) peak r values (from scan soft-argmax)
    phi_star : (N_spots,) peak phi values (from scan soft-argmax)
    """
    r_init, phi_init = _analytical_init(
        x_obs, y_obs, v_obs, a_obs, accel_measured,
        phi_lo, phi_hi,
        x0, y0, D, M_BH, v_sys,
        i0, di_dr, Omega0, dOmega_dr,
        sigma_v_sys, sigma_v_hv,
        sigma_a, sigma_a_floor)

    r_ref, phi_ref = _gauss_newton_refine(
        r_init, phi_init, phi_lo, phi_hi,
        x_obs, y_obs, v_obs, a_obs, accel_measured,
        x0, y0, D, M_BH, v_sys,
        i0, di_dr, Omega0, dOmega_dr,
        sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
        sigma_a, sigma_a_floor)

    r_star, phi_star = _scan_peak(
        r_ref, phi_ref, phi_lo, phi_hi,
        x_obs, sigma_x, y_obs, sigma_y,
        v_obs, a_obs, sigma_a, accel_measured,
        x0, y0, D, M_BH, v_sys,
        i0, di_dr, Omega0, dOmega_dr,
        sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
        sigma_a_floor, A_thr, sigma_det)

    def _integrate_mode(r_center, phi_center):
        r_grids = r_center[:, None] + dr_offsets[None, :]
        phi_grids = phi_center[:, None] + dphi_offsets[None, :]
        r_grids = jnp.clip(r_grids, 0.01, 10.0)
        phi_grids = jnp.clip(phi_grids, phi_lo[:, None], phi_hi[:, None])

        def _one_spot(r_grid_k, phi_grid_k,
                      x_k, sx_k, y_k, sy_k, v_k, a_k, sa_k,
                      am_k):
            return _spot_log_likelihood_on_grid(
                r_grid_k, phi_grid_k,
                x_k, sx_k, y_k, sy_k, v_k, a_k, sa_k, am_k,
                x0, y0, D, M_BH, v_sys,
                i0, di_dr, Omega0, dOmega_dr,
                sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
                sigma_a_floor, A_thr, sigma_det)

        log_integrand = jax.vmap(_one_spot)(
            r_grids, phi_grids,
            x_obs, sigma_x, y_obs, sigma_y,
            v_obs, a_obs, sigma_a,
            accel_measured)

        log_int_phi = ln_simpson_precomputed(log_integrand, log_wphi, axis=-1)
        return ln_simpson_precomputed(log_int_phi, log_wr, axis=-1)

    ln_I1 = _integrate_mode(r_star, phi_star)

    if bimodal:
        dx, dy = x_obs - x0, y_obs - y0
        i_m, Omega_m = warp_geometry(r_star, i0, di_dr, Omega0, dOmega_dr)
        sin_O, cos_O = jnp.sin(Omega_m), jnp.cos(Omega_m)
        cos_i = jnp.cos(i_m)

        sigma_x_tot = jnp.sqrt(sigma_x**2 + sigma_x_floor**2)
        sigma_y_tot = jnp.sqrt(sigma_y**2 + sigma_y_floor**2)
        px = 1.0 / sigma_x_tot**2
        py = 1.0 / sigma_y_tot**2
        r2, phi2 = _find_mode2(r_star, phi_star, phi_lo, dx, dy,
                               sin_O, cos_O, cos_i, px, py)
        phi2 = jnp.clip(phi2, phi_lo, phi_hi)

        ln_I2 = _integrate_mode(r2, phi2)
        ln_I = jnp.logaddexp(ln_I1, ln_I2)
    else:
        ln_I = ln_I1

    ll_total = jnp.sum(ln_I)
    return ll_total, r_star, phi_star


def build_grid_config(Nr=DEFAULT_NR, Nphi=DEFAULT_NPHI,
                      delta_r=DEFAULT_DELTA_R, delta_phi=DEFAULT_DELTA_PHI):
    """Build all grid arrays (call once at model init)."""
    dr, dphi, log_wr, log_wphi = make_relative_grids(
        Nr, Nphi, delta_r, delta_phi)
    return {
        "dr_offsets": dr,
        "dphi_offsets": dphi,
        "log_wr": log_wr,
        "log_wphi": log_wphi,
        "Nr": Nr,
        "Nphi": Nphi,
        "delta_r": delta_r,
        "delta_phi": delta_phi,
    }


def _find_mode2(r1, phi1, phi_lo, dx, dy, sin_O, cos_O, cos_i, px, py):
    """Compute second mode (r2, phi2) from the sin(phi) degeneracy.

    For red/systemic spots (phi_lo < pi): phi2 = pi - phi1.
    For blue spots (phi_lo >= pi): phi2 = 3*pi - phi1.
    """
    phi2 = jnp.where(phi_lo >= jnp.pi,
                     3 * jnp.pi - phi1,
                     jnp.pi - phi1)
    sin_phi2 = jnp.sin(phi2)
    cos_phi2 = jnp.cos(phi2)

    A2 = sin_phi2 * sin_O - cos_phi2 * cos_O * cos_i
    B2 = sin_phi2 * cos_O + cos_phi2 * sin_O * cos_i

    r2 = jnp.clip(
        (A2 * dx * px + B2 * dy * py) / (A2**2 * px + B2**2 * py + 1e-30),
        0.01, 10.0)
    return jax.lax.stop_gradient(r2), jax.lax.stop_gradient(phi2)


# -----------------------------------------------------------------------
# Model classes
# -----------------------------------------------------------------------

def _phi_bounds(spot_type, n_spots):
    """Compute per-spot phi bounds from spot type array."""
    phi_lo = np.empty(n_spots)
    phi_hi = np.empty(n_spots)
    for k in range(n_spots):
        if spot_type[k] == "r":
            phi_lo[k], phi_hi[k] = 0.0, np.pi
        elif spot_type[k] == "b":
            phi_lo[k], phi_hi[k] = np.pi, 2.0 * np.pi
        elif spot_type[k] == "s":
            phi_lo[k], phi_hi[k] = -0.5 * np.pi, 0.5 * np.pi
        else:
            raise ValueError(f"Unknown spot type '{spot_type[k]}'")
    return jnp.asarray(phi_lo), jnp.asarray(phi_hi)


class MaserDiskModel(ModelBase):
    """Megamaser disk H0 model with marginalised per-spot (r, phi)."""

    def __init__(self, config_path, data):
        super().__init__(config_path)
        fsection("Maser Disk Model")
        self._load_and_set_priors()

        self.n_spots = data["n_spots"]
        spot_type = data["spot_type"]
        self.is_systemic = jnp.asarray(data["is_systemic"])
        self.is_highvel = jnp.asarray(data["is_highvel"])
        self.accel_measured = jnp.asarray(data["accel_measured"])

        self.phi_lo, self.phi_hi = _phi_bounds(spot_type, self.n_spots)

        self._set_data_arrays(
            data, skip_keys=("spot_type", "is_systemic", "is_highvel",
                             "accel_measured", "n_spots", "galaxy_name",
                             "v_cmb_obs", "v_helio_to_cmb"))

        if "v_cmb_obs" not in data:
            raise ValueError(
                "data must contain 'v_cmb_obs' (CMB-frame recession "
                "velocity in km/s).")
        self.v_cmb_obs = float(data["v_cmb_obs"])
        self.v_helio_to_cmb = float(data.get("v_helio_to_cmb", 0.0))
        self.v_sys_obs = self.v_cmb_obs - self.v_helio_to_cmb

        D_min = float(get_nested(self.config, "model/priors/D/low", 10.0))
        D_max = float(get_nested(self.config, "model/priors/D/high", 200.0))
        self._sel_D_grid = jnp.linspace(D_min, D_max, 501)
        self._sel_dD = self._sel_D_grid[1] - self._sel_D_grid[0]
        self._sel_lp_vol = 2.0 * jnp.log(self._sel_D_grid)
        self.use_selection = get_nested(
            self.config, "model/use_selection", False)

        self.fit_di_dr = get_nested(
            self.config, "model/fit_di_dr", False)

        self.sample_accel_det = get_nested(
            self.config, "model/sample_accel_det", True)

        gc = build_grid_config()
        self._dr_offsets = gc["dr_offsets"]
        self._dphi_offsets = gc["dphi_offsets"]
        self._log_wr = gc["log_wr"]
        self._log_wphi = gc["log_wphi"]

        fprint(f"loaded {self.n_spots} maser spots "
               f"({int(self.is_systemic.sum())} systemic, "
               f"{int(self.is_highvel.sum())} high-velocity).")
        fprint(f"v_helio_to_cmb = {self.v_helio_to_cmb:.1f} km/s")
        fprint(f"use_selection = {self.use_selection}")
        fprint(f"fit_di_dr = {self.fit_di_dr}")
        fprint(f"sample_accel_det = {self.sample_accel_det}")

    def _sample_galaxy(self, H0, sigma_pec, h, suffix=""):
        """Sample all per-galaxy parameters and accumulate log-likelihood.

        Parameters
        ----------
        H0, sigma_pec : shared scalars (already sampled)
        h : H0 / 100
        suffix : appended to every numpyro site name (e.g. "_NGC4258")
        """
        D_c = rsample(f"D_c{suffix}", self.priors["D"])
        factor(f"lp_vol{suffix}", 2.0 * jnp.log(D_c))

        z_cosmo = self.distance2redshift(
            jnp.atleast_1d(D_c), h=h).squeeze()
        D_A = D_c / (1 + z_cosmo)

        M_BH = rsample(f"M_BH{suffix}", self.priors["M_BH"])
        x0 = rsample(f"x0{suffix}", self.priors["x0"])
        y0 = rsample(f"y0{suffix}", self.priors["y0"])

        i0_deg = rsample(f"i0{suffix}", self.priors["i0"])
        Omega0_deg = rsample(f"Omega0{suffix}", self.priors["Omega0"])
        dOmega_dr_deg = rsample(f"dOmega_dr{suffix}", self.priors["dOmega_dr"])

        i0 = jnp.deg2rad(i0_deg)
        Omega0 = jnp.deg2rad(Omega0_deg)
        dOmega_dr = jnp.deg2rad(dOmega_dr_deg)

        if self.fit_di_dr:
            di_dr_deg = rsample(f"di_dr{suffix}", self.priors["di_dr"])
            di_dr = jnp.deg2rad(di_dr_deg)
        else:
            di_dr = jnp.array(0.0)

        sigma_x_floor = rsample(
            f"sigma_x_floor{suffix}", self.priors["sigma_x_floor"])
        sigma_y_floor = rsample(
            f"sigma_y_floor{suffix}", self.priors["sigma_y_floor"])
        sigma_v_sys = rsample(f"sigma_v_sys{suffix}", self.priors["sigma_v_sys"])
        sigma_v_hv = rsample(f"sigma_v_hv{suffix}", self.priors["sigma_v_hv"])
        sigma_a_floor = rsample(
            f"sigma_a_floor{suffix}", self.priors["sigma_a_floor"])

        if self.sample_accel_det:
            A_thr = rsample(f"A_thr{suffix}", self.priors["A_thr"])
            sigma_det = rsample(f"sigma_det{suffix}", self.priors["sigma_det"])
        else:
            A_thr = jnp.array(0.0)
            sigma_det = jnp.array(0.1)

        ll_disk, _, _ = marginalise_spots(
            self.x, self.sigma_x, self.y, self.sigma_y,
            self.velocity, self.a, self.sigma_a,
            self.accel_measured,
            self.phi_lo, self.phi_hi,
            x0, y0, D_A, M_BH, self.v_sys_obs,
            i0, di_dr, Omega0, dOmega_dr,
            sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
            sigma_a_floor, A_thr, sigma_det,
            self._dr_offsets, self._dphi_offsets,
            self._log_wr, self._log_wphi)
        factor(f"ll_disk{suffix}", ll_disk)

        cz_cosmo = SPEED_OF_LIGHT * z_cosmo
        ll_vpec = jax_norm.logpdf(self.v_cmb_obs, cz_cosmo, sigma_pec)
        factor(f"ll_vpec{suffix}", ll_vpec)

        if self.use_selection:
            D_lim = rsample(f"D_lim{suffix}", self.priors["D_lim"])
            D_width = rsample(f"D_width{suffix}", self.priors["D_width"])

            log_sel_this = jax_norm.logcdf((D_lim - D_c) / D_width)
            factor(f"ll_sel_per_object{suffix}", log_sel_this)

            log_sel_grid = jax_norm.logcdf(
                (D_lim - self._sel_D_grid) / D_width)
            log_integrand = log_sel_grid + self._sel_lp_vol
            log_Z_sel = logsumexp(log_integrand) + jnp.log(self._sel_dD)
            factor(f"ll_sel_norm{suffix}", -log_Z_sel)

    def __call__(self):
        H0 = rsample("H0", self.priors["H0"])
        sigma_pec = rsample("sigma_pec", self.priors["sigma_pec"])
        self._sample_galaxy(H0, sigma_pec, H0 / 100.0)


class JointMaserModel(ModelBase):
    """Joint multi-galaxy megamaser H0 model.

    Shares H0 and sigma_pec across all galaxies; all other parameters are
    per-galaxy and sampled with galaxy-indexed numpyro site names via
    MaserDiskModel._sample_galaxy.
    """

    def __init__(self, config_path, data_list):
        super().__init__(config_path)
        fsection("Joint Maser Disk Model")
        self._load_and_set_priors()

        self.models = [MaserDiskModel(config_path, data) for data in data_list]
        self.galaxy_names = [d["galaxy_name"] for d in data_list]
        fprint(f"loaded {len(self.models)} galaxies: "
               f"{', '.join(self.galaxy_names)}")

    def __call__(self):
        H0 = rsample("H0", self.priors["H0"])
        sigma_pec = rsample("sigma_pec", self.priors["sigma_pec"])
        h = H0 / 100.0

        for model, gname in zip(self.models, self.galaxy_names):
            model._sample_galaxy(H0, sigma_pec, h, suffix=f"_{gname}")
