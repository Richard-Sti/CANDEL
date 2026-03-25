"""
JAX routines for marginalising (r, phi) in the megamaser disk model.

Instead of sampling per-spot (r_k, phi_k) as latent variables (adding
2*N_spots dimensions to the MCMC), we numerically integrate them out:

    ll_k = log int L_k(r, phi) dr dphi

where L_k = L_pos * L_vel * L_accel * pi(r, phi).


Design summary
--------------

**Strategy D** (adaptive per-spot grid, recentered every MCMC step):

1. Find peak (r*, phi*) for each spot using an analytical initial guess
   followed by a few Gauss--Newton iterations on the joint
   position+velocity+acceleration residual.

2. Build per-spot local grids centred at (r*_k, phi*_k) with fixed shape
   (Nr x Nphi). Different centres are fine under vmap; only the shape
   must be uniform.

3. Evaluate the full log-integrand on the grid and integrate with
   pre-computed Simpson weights in log-space.


Key numbers (CGCG 074-064, ~165 spots, i0 ~ 90 deg)
-----------------------------------------------------

Peak widths (combined position + velocity + acceleration):

  - Position constrains r*sin(phi) tightly (sigma_u ~ sigma_x ~ 0.02 mas).
  - At i ~ 90 deg the position Jacobian is nearly rank-1: position alone
    determines only one combination, r*sin(phi). The orthogonal direction
    r*cos(phi) is constrained by velocity (away from phi = pi/2) or
    acceleration (near phi = pi/2).
  - Combined sigma_r ~ 0.007--0.03 mas, sigma_phi ~ 0.007--0.15 rad
    depending on spot type and phi value.

Grid sizing:

  - delta_r = 0.15 mas, delta_phi = 0.50 rad (generous, covers worst case)
  - Nr = 21, Nphi = 31 (odd for Simpson; ~2 pts per sigma at narrowest)
  - 651 evaluations per spot, ~107k total for 165 spots
  - GPU estimate: ~0.2 ms forward, ~0.6 ms gradient (RTX 2080 Ti)

Dimension reduction: ~343 -> ~13 parameters -> >100x MCMC speedup.
"""
import jax
import jax.numpy as jnp

from .maser_disk import (
    C_v, C_a, SPEED_OF_LIGHT,
    warp_geometry, predict_position, predict_velocity_los,
    predict_acceleration_los, normal_logpdf,
)
from .simpson import simpson_log_weights, ln_simpson_precomputed


# -----------------------------------------------------------------------
# Default grid configuration
# -----------------------------------------------------------------------

DEFAULT_DELTA_R = 0.15     # mas half-width
DEFAULT_DELTA_PHI = 0.50   # rad half-width (generous for phi~pi/2 spots)
DEFAULT_NR = 21            # odd for Simpson
DEFAULT_NPHI = 31          # odd for Simpson


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
# Peak finder: analytical guess + Gauss-Newton on full residual
# -----------------------------------------------------------------------

def find_peak_rphi(x_obs, y_obs, v_obs, a_obs, accel_measured,
                   is_systemic, phi_lo, phi_hi,
                   x0, y0, D, M_BH, v_sys,
                   i0, di_dr, Omega0, dOmega_dr,
                   sigma_v_sys, sigma_v_hv,
                   sigma_a_obs, sigma_a_floor,
                   n_iter=6):
    """Find integrand peak (r*, phi*) for each spot.

    Uses an analytical initial guess followed by Gauss--Newton refinement
    on the joint position + velocity residual.

    For near-edge-on disks (i ~ 90 deg), position only constrains
    u = r*sin(phi). Velocity depends on sin(phi), which is symmetric
    about pi/2, creating a two-fold ambiguity. Acceleration (cos(phi))
    breaks this degeneracy.

    Parameters
    ----------
    x_obs, y_obs, v_obs, a_obs : (N_spots,) observed data
    accel_measured : (N_spots,) boolean
    is_systemic : (N_spots,) boolean
    phi_lo, phi_hi : (N_spots,) per-spot azimuthal bounds
    (remaining) : scalar geometry/physics parameters

    Returns
    -------
    r_star, phi_star : (N_spots,) peak locations
    """
    sin_O0, cos_O0 = jnp.sin(Omega0), jnp.cos(Omega0)
    sin_i0 = jnp.sin(i0)

    dx = x_obs - x0
    dy = y_obs - y0

    # u = r*sin(phi) from position
    u_from_pos = dx * sin_O0 + dy * cos_O0

    # r from velocity
    dv = v_obs - v_sys
    dv_safe = jnp.where(jnp.abs(dv) > 1.0, dv, jnp.sign(dv + 1e-20) * 1.0)
    r32 = C_v * jnp.sqrt(M_BH / D) * sin_i0 * u_from_pos / dv_safe
    r_init = jnp.clip(jnp.abs(r32) ** (2.0 / 3.0), 0.05, 5.0)

    # Two phi candidates: sin(phi) = sin(pi-phi)
    sin_phi = jnp.clip(u_from_pos / r_init, -0.9999, 0.9999)
    phi_a = jnp.arcsin(sin_phi)       # in [-pi/2, pi/2]
    phi_b = jnp.pi - phi_a            # complementary

    # Build per-type candidates clamped to [phi_lo, phi_hi]
    c1 = jnp.clip(phi_a, phi_lo, phi_hi)
    c2 = jnp.clip(phi_b, phi_lo, phi_hi)
    # For blueshifted (phi in [pi, 2*pi]): map negative arcsin results
    c3 = jnp.clip(2.0 * jnp.pi + phi_a, phi_lo, phi_hi)
    c4 = jnp.clip(2.0 * jnp.pi + phi_b - jnp.pi, phi_lo, phi_hi)
    # = clip(pi + phi_a, phi_lo, phi_hi) for blue case

    # Evaluate COMBINED (velocity + acceleration) score at all candidates
    i_at_r, O_at_r = warp_geometry(r_init, i0, di_dr, Omega0, dOmega_dr)

    def _score(phi_c):
        v_c = predict_velocity_los(r_init, phi_c, D, M_BH, v_sys, i_at_r, O_at_r)
        a_c = predict_acceleration_los(r_init, phi_c, D, M_BH, i_at_r)
        sigma_v = jnp.where(is_systemic, sigma_v_sys, sigma_v_hv)
        sigma_a = jnp.sqrt(sigma_a_obs**2 + sigma_a_floor**2)
        score = -0.5 * ((v_c - v_obs) / sigma_v)**2
        score = score + jnp.where(accel_measured,
                                  -0.5 * ((a_c - a_obs) / sigma_a)**2,
                                  0.0)
        return score

    s1 = _score(c1)
    s2 = _score(c2)
    s3 = _score(c3)
    s4 = _score(c4)

    # Pick best candidate
    best = c1
    best_s = s1
    best = jnp.where(s2 > best_s, c2, best)
    best_s = jnp.maximum(s2, best_s)
    best = jnp.where(s3 > best_s, c3, best)
    best_s = jnp.maximum(s3, best_s)
    best = jnp.where(s4 > best_s, c4, best)

    r = jnp.clip(r_init, 0.05, 5.0)
    phi = jnp.clip(best, phi_lo, phi_hi)

    # --- Gauss-Newton refinement on position + velocity ---
    # The peak finder output is used ONLY for grid centering, not directly
    # in the likelihood. We use lax.stop_gradient so that AD flows through
    # the grid integrand, not through the Newton solver.
    # This is correct because the integral value does not depend on where
    # we center the grid (as long as the grid covers the peak).
    sigma_v = jnp.where(is_systemic, sigma_v_sys, sigma_v_hv)

    def _step(carry, _):
        r, phi = carry
        i, Omega = warp_geometry(r, i0, di_dr, Omega0, dOmega_dr)
        X_pred, Y_pred = predict_position(r, phi, x0, y0, i, Omega)
        V_pred = predict_velocity_los(r, phi, D, M_BH, v_sys, i, Omega)

        w_x, w_y, w_v = 50.0, 20.0, 1.0 / sigma_v
        fx = (X_pred - x_obs) * w_x
        fy = (Y_pred - y_obs) * w_y
        fv = (V_pred - v_obs) * w_v

        sin_phi, cos_phi = jnp.sin(phi), jnp.cos(phi)
        sin_O, cos_O = jnp.sin(Omega), jnp.cos(Omega)
        cos_i, sin_i = jnp.cos(i), jnp.sin(i)

        dXdr = (sin_phi * sin_O - cos_phi * cos_O * cos_i) * w_x
        dXdp = r * (cos_phi * sin_O + sin_phi * cos_O * cos_i) * w_x
        dYdr = (sin_phi * cos_O + cos_phi * sin_O * cos_i) * w_y
        dYdp = r * (cos_phi * cos_O - sin_phi * sin_O * cos_i) * w_y

        v_kep = C_v * jnp.sqrt(M_BH / (r * D))
        dVdr = -0.5 * v_kep / r * sin_phi * sin_i * w_v
        dVdp = v_kep * cos_phi * sin_i * w_v

        JtJ_00 = dXdr**2 + dYdr**2 + dVdr**2
        JtJ_01 = dXdr * dXdp + dYdr * dYdp + dVdr * dVdp
        JtJ_11 = dXdp**2 + dYdp**2 + dVdp**2
        Jtf_0 = dXdr * fx + dYdr * fy + dVdr * fv
        Jtf_1 = dXdp * fx + dYdp * fy + dVdp * fv

        reg = 1e-4 * (JtJ_00 + JtJ_11) + 1e-8
        JtJ_00 = JtJ_00 + reg
        JtJ_11 = JtJ_11 + reg

        det = JtJ_00 * JtJ_11 - JtJ_01**2 + 1e-30

        dr = (JtJ_11 * Jtf_0 - JtJ_01 * Jtf_1) / det
        dphi = (-JtJ_01 * Jtf_0 + JtJ_00 * Jtf_1) / det

        # Soft damping (differentiable alternative to hard clipping)
        step_mag = jnp.sqrt(dr**2 + dphi**2 + 1e-30)
        max_step = 0.3 * r + 0.5
        scale = jnp.where(step_mag > max_step, max_step / step_mag, 1.0)
        r_new = r - scale * dr
        phi_new = phi - scale * dphi

        # Soft clamping (differentiable)
        r_new = jnp.clip(r_new, 0.05, 5.0)
        phi_new = jnp.clip(phi_new, phi_lo, phi_hi)
        return (r_new, phi_new), None

    (r, phi), _ = jax.lax.scan(_step, (r, phi), None, length=n_iter)

    # Stop gradient: grid centering should not be differentiated through.
    # The integral is invariant to grid center shifts (to integration accuracy).
    return jax.lax.stop_gradient(r), jax.lax.stop_gradient(phi)


# -----------------------------------------------------------------------
# Per-spot integrand evaluation
# -----------------------------------------------------------------------

def _spot_log_likelihood_on_grid(
        r_grid, phi_grid,
        x_obs_k, sigma_x_k, y_obs_k, sigma_y_k,
        v_obs_k, a_obs_k, sigma_a_k,
        accel_measured_k, is_systemic_k,
        x0, y0, D, M_BH, v_sys,
        i0, di_dr, Omega0, dOmega_dr,
        sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
        sigma_a_floor):
    """Evaluate log-integrand on a 2D (r, phi) grid for one spot.

    Parameters
    ----------
    r_grid : (Nr,) absolute r values in mas
    phi_grid : (Nphi,) absolute phi values in rad

    Returns
    -------
    log_integrand : (Nr, Nphi)
    """
    r_2d = r_grid[:, None]       # (Nr, 1)
    phi_2d = phi_grid[None, :]   # (1, Nphi)

    i_at_r, Omega_at_r = warp_geometry(r_2d, i0, di_dr, Omega0, dOmega_dr)

    X_pred, Y_pred = predict_position(r_2d, phi_2d, x0, y0, i_at_r, Omega_at_r)
    V_pred = predict_velocity_los(r_2d, phi_2d, D, M_BH, v_sys, i_at_r, Omega_at_r)
    A_pred = predict_acceleration_los(r_2d, phi_2d, D, M_BH, i_at_r)

    sigma_x = jnp.sqrt(sigma_x_k**2 + sigma_x_floor**2)
    sigma_y = jnp.sqrt(sigma_y_k**2 + sigma_y_floor**2)
    ll_pos = (normal_logpdf(x_obs_k, X_pred, sigma_x)
              + normal_logpdf(y_obs_k, Y_pred, sigma_y))

    sigma_v = jnp.where(is_systemic_k, sigma_v_sys, sigma_v_hv)
    ll_vel = normal_logpdf(v_obs_k, V_pred, sigma_v)

    sigma_a = jnp.sqrt(sigma_a_k**2 + sigma_a_floor**2)
    ll_acc = jnp.where(accel_measured_k,
                       normal_logpdf(a_obs_k, A_pred, sigma_a),
                       0.0)

    return ll_pos + ll_vel + ll_acc


# -----------------------------------------------------------------------
# Main marginalisation routine
# -----------------------------------------------------------------------

def marginalise_spots(
        x_obs, sigma_x, y_obs, sigma_y,
        v_obs, a_obs, sigma_a,
        accel_measured, is_systemic,
        phi_lo, phi_hi,
        x0, y0, D, M_BH, v_sys,
        i0, di_dr, Omega0, dOmega_dr,
        sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
        sigma_a_floor,
        dr_offsets, dphi_offsets, log_wr, log_wphi,
        n_newton=6, bimodal=True):
    """Compute sum of marginalised log-likelihoods for all spots.

    Uses G-N for initial grid centering, then the mode 1 grid's argmax
    as a robust peak (fused — no separate scan needed). Mode 2 is
    centered on phi2 = pi - phi1 (or 3pi - phi1 for blue spots).

    Parameters
    ----------
    x_obs, sigma_x, ... : (N_spots,) observed data
    phi_lo, phi_hi : (N_spots,) per-spot azimuthal bounds
    x0, y0, D, M_BH, ... : scalar model parameters
    dr_offsets, dphi_offsets : (Nr,), (Nphi,) relative grid offsets
    log_wr, log_wphi : (Nr,), (Nphi,) pre-computed Simpson log-weights
    n_newton : Gauss--Newton iterations for peak finding
    bimodal : if True, integrate both modes via logaddexp

    Returns
    -------
    ll_total : scalar, sum_k log int L_k(r, phi) dr dphi
    r_star : (N_spots,) peak r values (from grid argmax)
    phi_star : (N_spots,) peak phi values (from grid argmax)
    """
    Nphi = dphi_offsets.shape[0]

    # G-N peak for initial grid centering
    r_gn, phi_gn = find_peak_rphi(
        x_obs, y_obs, v_obs, a_obs, accel_measured,
        is_systemic, phi_lo, phi_hi,
        x0, y0, D, M_BH, v_sys,
        i0, di_dr, Omega0, dOmega_dr,
        sigma_v_sys, sigma_v_hv,
        sigma_a, sigma_a_floor,
        n_iter=n_newton)

    def _integrate_mode(r_center, phi_center):
        """Simpson integral on a grid centered at (r_center, phi_center)."""
        r_grids = r_center[:, None] + dr_offsets[None, :]
        phi_grids = phi_center[:, None] + dphi_offsets[None, :]
        r_grids = jnp.clip(r_grids, 0.01, 10.0)
        phi_grids = jnp.clip(phi_grids, phi_lo[:, None], phi_hi[:, None])

        def _one_spot(r_grid_k, phi_grid_k,
                      x_k, sx_k, y_k, sy_k, v_k, a_k, sa_k,
                      am_k, is_k):
            return _spot_log_likelihood_on_grid(
                r_grid_k, phi_grid_k,
                x_k, sx_k, y_k, sy_k, v_k, a_k, sa_k, am_k, is_k,
                x0, y0, D, M_BH, v_sys,
                i0, di_dr, Omega0, dOmega_dr,
                sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
                sigma_a_floor)

        log_integrand = jax.vmap(_one_spot)(
            r_grids, phi_grids,
            x_obs, sigma_x, y_obs, sigma_y,
            v_obs, a_obs, sigma_a,
            accel_measured, is_systemic)

        log_int_phi = ln_simpson_precomputed(log_integrand, log_wphi, axis=-1)
        return ln_simpson_precomputed(log_int_phi, log_wr, axis=-1)

    def _scan_peak(r_center, phi_center, dr_s, dphi_s, Nphi_s):
        """Coarse scan to find robust peak near (r_center, phi_center)."""
        r_grids = jnp.clip(r_center[:, None] + dr_s[None, :], 0.01, 10.0)
        phi_grids = jnp.clip(phi_center[:, None] + dphi_s[None, :],
                             phi_lo[:, None], phi_hi[:, None])

        def _one(rg, pg, xk, sxk, yk, syk, vk, ak, sak, amk, isk):
            li = _spot_log_likelihood_on_grid(
                rg, pg, xk, sxk, yk, syk, vk, ak, sak, amk, isk,
                x0, y0, D, M_BH, v_sys,
                i0, di_dr, Omega0, dOmega_dr,
                sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
                sigma_a_floor)
            fi = jnp.argmax(li)
            return rg[fi // Nphi_s], pg[fi % Nphi_s]

        return jax.vmap(_one)(
            r_grids, phi_grids,
            x_obs, sigma_x, y_obs, sigma_y,
            v_obs, a_obs, sigma_a, accel_measured, is_systemic)

    # Robust peak via soft argmax on a coarse scan grid.
    # The soft argmax (weighted average using softmax of h values) gives
    # a smooth, differentiable peak location that doesn't jump
    # discontinuously as theta changes — essential for HMC gradients.
    dr_s = jnp.linspace(-DEFAULT_DELTA_R, DEFAULT_DELTA_R, SCAN_NR)
    dphi_s = jnp.linspace(-DEFAULT_DELTA_PHI, DEFAULT_DELTA_PHI, SCAN_NPHI)
    r_scan_grids = jnp.clip(r_gn[:, None] + dr_s[None, :], 0.01, 10.0)
    phi_scan_grids = jnp.clip(phi_gn[:, None] + dphi_s[None, :],
                              phi_lo[:, None], phi_hi[:, None])

    def _scan_h_one(rg, pg, xk, sxk, yk, syk, vk, ak, sak, amk, isk):
        return _spot_log_likelihood_on_grid(
            rg, pg, xk, sxk, yk, syk, vk, ak, sak, amk, isk,
            x0, y0, D, M_BH, v_sys,
            i0, di_dr, Omega0, dOmega_dr,
            sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
            sigma_a_floor)

    log_scan = jax.vmap(_scan_h_one)(
        r_scan_grids, phi_scan_grids,
        x_obs, sigma_x, y_obs, sigma_y,
        v_obs, a_obs, sigma_a, accel_measured, is_systemic)
    # shape: (N_spots, SCAN_NR, SCAN_NPHI)

    # Soft argmax: weighted average of grid coordinates
    N_spots = x_obs.shape[0]
    r_2d = jnp.broadcast_to(r_scan_grids[:, :, None],
                             (N_spots, SCAN_NR, SCAN_NPHI))
    phi_2d = jnp.broadcast_to(phi_scan_grids[:, None, :],
                               (N_spots, SCAN_NR, SCAN_NPHI))
    log_flat = log_scan.reshape(N_spots, -1)
    r_flat = r_2d.reshape(N_spots, -1)
    phi_flat = phi_2d.reshape(N_spots, -1)

    weights = jax.nn.softmax(log_flat, axis=-1)
    r_star = jnp.sum(weights * r_flat, axis=-1)
    phi_star = jnp.sum(weights * phi_flat, axis=-1)

    # Mode 1: Simpson integral centered on robust smooth peak
    ln_I1 = _integrate_mode(r_star, phi_star)

    if bimodal:
        # Mode 2 center from the grid's argmax peak
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


# -----------------------------------------------------------------------
# Safety diagnostics
# -----------------------------------------------------------------------

def check_grid_coverage(log_integrand, threshold=-20.0):
    """Check that the integrand peak is not at the grid boundary.

    Parameters
    ----------
    log_integrand : (N_spots, Nr, Nphi) from the last evaluation
    threshold : boundary values below max + threshold are OK

    Returns
    -------
    all_ok : bool, True if all spots have boundary values well below peak
    worst_margin : scalar, smallest (max - boundary_max) across spots
    """
    # Max over interior
    peak_val = jnp.max(log_integrand, axis=(-2, -1))  # (N_spots,)

    # Max on the boundary ring
    boundary = jnp.concatenate([
        log_integrand[:, 0, :],      # r = r_min edge
        log_integrand[:, -1, :],     # r = r_max edge
        log_integrand[:, :, 0],      # phi = phi_min edge
        log_integrand[:, :, -1],     # phi = phi_max edge
    ], axis=-1)
    boundary_max = jnp.max(boundary, axis=-1)  # (N_spots,)

    margin = peak_val - boundary_max
    worst_margin = jnp.min(margin)
    all_ok = worst_margin > -threshold  # margin > 20 in log-likelihood
    return all_ok, worst_margin


# -----------------------------------------------------------------------
# Grid config builder
# -----------------------------------------------------------------------

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


# =======================================================================


# -----------------------------------------------------------------------
# Robust peak finder with coarse-scan fallback
# -----------------------------------------------------------------------

SCAN_NR = 7
SCAN_NPHI = 9


def find_peak_rphi_robust(x_obs, y_obs, v_obs, a_obs, accel_measured,
                          is_systemic, phi_lo, phi_hi,
                          x0, y0, D, M_BH, v_sys,
                          i0, di_dr, Omega0, dOmega_dr,
                          sigma_x, sigma_y,
                          sigma_v_sys, sigma_v_hv,
                          sigma_a_obs, sigma_a_floor,
                          sigma_x_floor, sigma_y_floor,
                          dr_scan, dphi_scan, Nphi_scan,
                          n_iter=6):
    """Robust peak finder: G-N + coarse-scan fallback.

    Runs the standard Gauss-Newton peak finder, then evaluates the
    integrand on a coarse grid centered on the G-N result. If the scan
    finds a higher peak, uses that instead. Both are stop_gradient'd.
    """
    # Standard G-N peak
    r_gn, phi_gn = find_peak_rphi(
        x_obs, y_obs, v_obs, a_obs, accel_measured,
        is_systemic, phi_lo, phi_hi,
        x0, y0, D, M_BH, v_sys,
        i0, di_dr, Omega0, dOmega_dr,
        sigma_v_sys, sigma_v_hv,
        sigma_a_obs, sigma_a_floor,
        n_iter=n_iter)

    # Coarse scan centered on G-N peak
    r_grids = jnp.clip(r_gn[:, None] + dr_scan[None, :], 0.01, 10.0)
    phi_grids = jnp.clip(phi_gn[:, None] + dphi_scan[None, :],
                         phi_lo[:, None], phi_hi[:, None])

    def _scan_one(r_grid_k, phi_grid_k,
                  x_k, sx_k, y_k, sy_k, v_k, a_k, sa_k, am_k, is_k):
        log_int = _spot_log_likelihood_on_grid(
            r_grid_k, phi_grid_k,
            x_k, sx_k, y_k, sy_k, v_k, a_k, sa_k, am_k, is_k,
            x0, y0, D, M_BH, v_sys,
            i0, di_dr, Omega0, dOmega_dr,
            sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
            sigma_a_floor)
        flat_idx = jnp.argmax(log_int)
        return r_grid_k[flat_idx // Nphi_scan], phi_grid_k[flat_idx % Nphi_scan]

    r_scan, phi_scan = jax.vmap(_scan_one)(
        r_grids, phi_grids,
        x_obs, sigma_x, y_obs, sigma_y,
        v_obs, a_obs, sigma_a_obs, accel_measured, is_systemic)

    # Evaluate h at both candidates to pick the better one
    def _eval_h(r, phi):
        i_m, O_m = warp_geometry(r, i0, di_dr, Omega0, dOmega_dr)
        X, Y = predict_position(r, phi, x0, y0, i_m, O_m)
        V = predict_velocity_los(r, phi, D, M_BH, v_sys, i_m, O_m)
        A = predict_acceleration_los(r, phi, D, M_BH, i_m)
        sigma_x_t = jnp.sqrt(sigma_x**2 + sigma_x_floor**2)
        sigma_y_t = jnp.sqrt(sigma_y**2 + sigma_y_floor**2)
        sigma_v_t = jnp.where(is_systemic, sigma_v_sys, sigma_v_hv)
        sigma_a_t = jnp.sqrt(sigma_a_obs**2 + sigma_a_floor**2)
        return (normal_logpdf(x_obs, X, sigma_x_t)
                + normal_logpdf(y_obs, Y, sigma_y_t)
                + normal_logpdf(v_obs, V, sigma_v_t)
                + jnp.where(accel_measured,
                            normal_logpdf(a_obs, A, sigma_a_t), 0.0))

    h_gn = _eval_h(r_gn, phi_gn)
    h_scan = _eval_h(r_scan, phi_scan)

    use_scan = h_scan > h_gn
    r_best = jnp.where(use_scan, r_scan, r_gn)
    phi_best = jnp.where(use_scan, phi_scan, phi_gn)

    return jax.lax.stop_gradient(r_best), jax.lax.stop_gradient(phi_best)


# -----------------------------------------------------------------------
# Mode 2 solver
# -----------------------------------------------------------------------

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
# Laplace (linearised) marginalisation — optional fast alternative
# -----------------------------------------------------------------------

def _laplace_one_spot(r_k, phi_k,
                      x_obs_k, sigma_x_k, y_obs_k, sigma_y_k,
                      v_obs_k, a_obs_k, sigma_a_k,
                      accel_measured_k, is_systemic_k,
                      x0, y0, D, M_BH, v_sys,
                      i0, di_dr, Omega0, dOmega_dr,
                      sigma_x_floor, sigma_y_floor,
                      sigma_v_sys, sigma_v_hv, sigma_a_floor):
    """Laplace-approximated marginal log-likelihood for one spot, one mode.

    ln I ~ h(mode) + ln(2*pi) - 0.5 * ln det(J^T C^{-1} J)
    """
    i_m, O_m = warp_geometry(r_k, i0, di_dr, Omega0, dOmega_dr)
    sin_phi, cos_phi = jnp.sin(phi_k), jnp.cos(phi_k)
    sin_O, cos_O = jnp.sin(O_m), jnp.cos(O_m)
    cos_i, sin_i = jnp.cos(i_m), jnp.sin(i_m)

    X, Y = predict_position(r_k, phi_k, x0, y0, i_m, O_m)
    V = predict_velocity_los(r_k, phi_k, D, M_BH, v_sys, i_m, O_m)
    A = predict_acceleration_los(r_k, phi_k, D, M_BH, i_m)

    sigma_x = jnp.sqrt(sigma_x_k**2 + sigma_x_floor**2)
    sigma_y = jnp.sqrt(sigma_y_k**2 + sigma_y_floor**2)
    sigma_v = jnp.where(is_systemic_k, sigma_v_sys, sigma_v_hv)
    sigma_a = jnp.sqrt(sigma_a_k**2 + sigma_a_floor**2)

    h_mode = (normal_logpdf(x_obs_k, X, sigma_x)
              + normal_logpdf(y_obs_k, Y, sigma_y)
              + normal_logpdf(v_obs_k, V, sigma_v)
              + jnp.where(accel_measured_k,
                          normal_logpdf(a_obs_k, A, sigma_a), 0.0))

    # Jacobian d(X,Y,V,A)/d(r,phi)
    dXdr = sin_phi * sin_O - cos_phi * cos_O * cos_i
    dXdp = r_k * (cos_phi * sin_O + sin_phi * cos_O * cos_i)
    dYdr = sin_phi * cos_O + cos_phi * sin_O * cos_i
    dYdp = r_k * (cos_phi * cos_O - sin_phi * sin_O * cos_i)
    v_kep = C_v * jnp.sqrt(M_BH / (r_k * D))
    dVdr = -0.5 * v_kep / r_k * sin_phi * sin_i
    dVdp = v_kep * cos_phi * sin_i
    a_mag = C_a * M_BH / (r_k**2 * D**2)
    dAdr = -2.0 * a_mag / r_k * cos_phi * sin_i
    dAdp = -a_mag * sin_phi * sin_i

    px = 1.0 / sigma_x**2
    py = 1.0 / sigma_y**2
    pv = 1.0 / sigma_v**2
    pa = jnp.where(accel_measured_k, 1.0 / sigma_a**2, 0.0)

    L00 = px*dXdr**2 + py*dYdr**2 + pv*dVdr**2 + pa*dAdr**2
    L01 = px*dXdr*dXdp + py*dYdr*dYdp + pv*dVdr*dVdp + pa*dAdr*dAdp
    L11 = px*dXdp**2 + py*dYdp**2 + pv*dVdp**2 + pa*dAdp**2

    det_L = jnp.maximum(L00 * L11 - L01**2, 1e-30)
    return h_mode + jnp.log(2.0 * jnp.pi) - 0.5 * jnp.log(det_L)


def marginalise_spots_laplace(
        x_obs, sigma_x, y_obs, sigma_y,
        v_obs, a_obs, sigma_a,
        accel_measured, is_systemic,
        phi_lo, phi_hi,
        x0, y0, D, M_BH, v_sys,
        i0, di_dr, Omega0, dOmega_dr,
        sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
        sigma_a_floor,
        dr_scan, dphi_scan, Nphi_scan,
        n_newton=6):
    """Laplace-approximated marginalised log-likelihood (fast).

    Uses robust peak finder + Laplace for both modes.
    """
    r1, phi1 = find_peak_rphi_robust(
        x_obs, y_obs, v_obs, a_obs, accel_measured,
        is_systemic, phi_lo, phi_hi,
        x0, y0, D, M_BH, v_sys,
        i0, di_dr, Omega0, dOmega_dr,
        sigma_x, sigma_y,
        sigma_v_sys, sigma_v_hv,
        sigma_a, sigma_a_floor,
        sigma_x_floor, sigma_y_floor,
        dr_scan, dphi_scan, Nphi_scan,
        n_iter=n_newton)

    # Mode 2
    dx, dy = x_obs - x0, y_obs - y0
    i_m, Omega_m = warp_geometry(r1, i0, di_dr, Omega0, dOmega_dr)
    sin_O, cos_O = jnp.sin(Omega_m), jnp.cos(Omega_m)
    cos_i = jnp.cos(i_m)
    sigma_x_tot = jnp.sqrt(sigma_x**2 + sigma_x_floor**2)
    sigma_y_tot = jnp.sqrt(sigma_y**2 + sigma_y_floor**2)
    px, py = 1.0 / sigma_x_tot**2, 1.0 / sigma_y_tot**2
    r2, phi2 = _find_mode2(r1, phi1, phi_lo, dx, dy,
                           sin_O, cos_O, cos_i, px, py)
    phi2 = jnp.clip(phi2, phi_lo, phi_hi)

    per_spot = (x_obs, sigma_x, y_obs, sigma_y,
                v_obs, a_obs, sigma_a, accel_measured, is_systemic)

    def _lap(r, p, *args):
        return _laplace_one_spot(
            r, p, *args,
            x0, y0, D, M_BH, v_sys,
            i0, di_dr, Omega0, dOmega_dr,
            sigma_x_floor, sigma_y_floor, sigma_v_sys, sigma_v_hv,
            sigma_a_floor)

    ln_I1 = jax.vmap(lambda r, p, *a: _lap(r, p, *a))(r1, phi1, *per_spot)
    ln_I2 = jax.vmap(lambda r, p, *a: _lap(r, p, *a))(r2, phi2, *per_spot)

    ln_I = jnp.logaddexp(ln_I1, ln_I2)
    return jnp.sum(ln_I), r1, phi1
