# Simplify Maser Peak Finder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 3-stage peak finder (analytical init + Gauss-Newton + soft-argmax scan) with a 2-stage pipeline (analytical init + scan-grid soft-argmax), eliminating ~60 lines of hand-coded Jacobian/GN machinery.

**Architecture:** The analytical init already produces a physics-informed (r, phi) guess. Instead of refining it with a hand-differentiated Gauss-Newton solver, we widen the scan grid and run the soft-argmax directly on the real log-likelihood. This is simpler and more robust — the GN uses a weighted least-squares surrogate with frozen sigma_v, while the scan uses the actual likelihood. The 4-candidate phi scoring is also simplified to a stacked argmax.

**Tech Stack:** JAX, jax.numpy, jax.vmap, jax.lax.stop_gradient

---

## File Structure

- **Modify:** `candel/model/model_H0_maser.py` — all changes are in this single file
  - `find_peak_rphi` → replaced by `_analytical_init` (~25 lines, down from ~110)
  - GN code (lines 276-330) → deleted entirely
  - Soft-argmax scan (currently embedded in `marginalise_spots` lines 460-491) → extracted into `_scan_peak` helper
  - `marginalise_spots` → simplified caller: `_analytical_init` → `_scan_peak` → integrate
  - Scan grid constants `SCAN_NR`, `SCAN_NPHI` → widened (e.g. 11x15 instead of 7x9) to compensate for no GN

No new files. No changes to the mock, the model classes, or `marginalise_spots`'s public signature.

---

### Task 1: Capture reference log-likelihood values before refactoring

Before changing anything, we need a regression anchor. Run the model at known parameters and record the log-likelihood and peak locations so we can verify the refactored code produces similar (not necessarily identical) results.

**Files:**
- Create: `scripts/test_maser_peak_refactor.py`

- [ ] **Step 1: Write the reference-capture script**

```python
#!/usr/bin/env python
"""Capture reference values from current maser peak finder for regression."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import numpy as np

from candel.mock.maser_disk_mock import gen_maser_mock_like_cgcg074
from candel.model.model_H0_maser import (
    MaserDiskModel, build_grid_config, find_peak_rphi, marginalise_spots,
    _phi_bounds,
)

data, tp = gen_maser_mock_like_cgcg074(seed=42, verbose=False)

# Build the grid config
gc = build_grid_config()

# Convert params to what marginalise_spots expects
i0 = jnp.deg2rad(tp["i0"])
Omega0 = jnp.deg2rad(tp["Omega0"])
dOmega_dr = jnp.deg2rad(tp["dOmega_dr"])
di_dr = jnp.array(0.0)
phi_lo, phi_hi = _phi_bounds(data["spot_type"], data["n_spots"])

ll, r_star, phi_star = marginalise_spots(
    jnp.asarray(data["x"]), jnp.asarray(data["sigma_x"]),
    jnp.asarray(data["y"]), jnp.asarray(data["sigma_y"]),
    jnp.asarray(data["velocity"]),
    jnp.asarray(data["a"]), jnp.asarray(data["sigma_a"]),
    jnp.asarray(data["accel_measured"]),
    phi_lo, phi_hi,
    tp["x0"], tp["y0"], tp["D"], tp["M_BH"], tp["v_sys"],
    i0, di_dr, Omega0, dOmega_dr,
    tp["sigma_x_floor"], tp["sigma_y_floor"],
    tp["sigma_v_sys"], tp["sigma_v_hv"],
    tp["sigma_a_floor"], tp["A_thr"], tp["sigma_det"],
    gc["dr_offsets"], gc["dphi_offsets"], gc["log_wr"], gc["log_wphi"],
)

print(f"ll_total = {float(ll):.4f}")
print(f"r_star  = {np.array(r_star)}")
print(f"phi_star = {np.array(phi_star)}")

np.savez("scripts/maser_peak_reference.npz",
         ll=float(ll),
         r_star=np.array(r_star),
         phi_star=np.array(phi_star))
print("Saved to scripts/maser_peak_reference.npz")
```

- [ ] **Step 2: Run the script and save reference values**

Run: `cd /mnt/users/rstiskalek/CANDEL && python scripts/test_maser_peak_refactor.py`

Record the printed `ll_total` value. It should be a finite negative number (typical range: -200 to -50).

- [ ] **Step 3: Commit**

```bash
git add scripts/test_maser_peak_refactor.py scripts/maser_peak_reference.npz
git commit -m "test: add maser peak finder regression reference values"
```

---

### Task 2: Replace `find_peak_rphi` with `_analytical_init`

Delete the GN loop entirely. Keep only the analytical r-init and phi-candidate selection, simplified with a stacked argmax.

**Files:**
- Modify: `candel/model/model_H0_maser.py:196-330`

- [ ] **Step 1: Replace `find_peak_rphi` with `_analytical_init`**

Delete the entire `find_peak_rphi` function (lines 196-330) and replace with:

```python
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
```

Key changes:
- **No GN loop** — the function just returns the analytical guess
- **Stacked argmax** replaces the manual `best`/`best_s` accumulation (lines 265-271)
- ~25 lines instead of ~110

- [ ] **Step 2: Verify it compiles**

Run: `cd /mnt/users/rstiskalek/CANDEL && python -c "from candel.model.model_H0_maser import _analytical_init; print('OK')"`

Expected: `OK` (no import errors)

- [ ] **Step 3: Commit**

```bash
git add candel/model/model_H0_maser.py
git commit -m "refactor: replace find_peak_rphi with simpler _analytical_init"
```

---

### Task 3: Extract `_scan_peak` and rewire `marginalise_spots`

Move the soft-argmax scan out of `marginalise_spots` into its own function, widen the scan grid, and wire it to use `_analytical_init` instead of the deleted `find_peak_rphi`.

**Files:**
- Modify: `candel/model/model_H0_maser.py:163-172` (constants)
- Modify: `candel/model/model_H0_maser.py:392-514` (`marginalise_spots`)

- [ ] **Step 1: Update scan grid constants**

Replace the existing constants (lines ~163-172):

```python
DEFAULT_DELTA_R = 0.15     # mas half-width
DEFAULT_DELTA_PHI = 0.50   # rad half-width
DEFAULT_NR = 21            # odd for Simpson
DEFAULT_NPHI = 31          # odd for Simpson

SCAN_NR = 7
SCAN_NPHI = 9
```

With widened scan grid (to compensate for no GN refinement):

```python
DEFAULT_DELTA_R = 0.15     # mas half-width
DEFAULT_DELTA_PHI = 0.50   # rad half-width
DEFAULT_NR = 21            # odd for Simpson
DEFAULT_NPHI = 31          # odd for Simpson

SCAN_DELTA_R = 0.4         # mas half-width for scan (wider than integration)
SCAN_DELTA_PHI = 0.8       # rad half-width for scan (wider than integration)
SCAN_NR = 11
SCAN_NPHI = 15
```

- [ ] **Step 2: Add `_scan_peak` function**

Add this function right after `_analytical_init` (before `_spot_log_likelihood_on_grid`):

```python
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
    weights = jax.nn.softmax(log_flat, axis=-1)

    r_star = jax.lax.stop_gradient(
        jnp.sum(weights * r_2d.reshape(N_spots, -1), axis=-1))
    phi_star = jax.lax.stop_gradient(
        jnp.sum(weights * phi_2d.reshape(N_spots, -1), axis=-1))
    return r_star, phi_star
```

- [ ] **Step 3: Simplify `marginalise_spots`**

Replace the body of `marginalise_spots` (everything after the docstring, from the current `r_gn, phi_gn = find_peak_rphi(...)` down to `return ll_total, r_star, phi_star`) with:

```python
    r_init, phi_init = _analytical_init(
        x_obs, y_obs, v_obs, a_obs, accel_measured,
        phi_lo, phi_hi,
        x0, y0, D, M_BH, v_sys,
        i0, di_dr, Omega0, dOmega_dr,
        sigma_v_sys, sigma_v_hv,
        sigma_a, sigma_a_floor)

    r_star, phi_star = _scan_peak(
        r_init, phi_init, phi_lo, phi_hi,
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
```

Also remove the `n_newton` parameter from `marginalise_spots`'s signature (and its `n_newton=6` default). The `bimodal` parameter stays.

- [ ] **Step 4: Remove `n_newton` from the call in `_sample_galaxy`**

In `_sample_galaxy` (line ~682), the call to `marginalise_spots` doesn't pass `n_newton` explicitly (it uses the default), so no change is needed there. But verify by reading the call site.

- [ ] **Step 5: Clean up exports**

Check if `find_peak_rphi` is imported anywhere. The reference script from Task 1 imports it — update that import to `_analytical_init`. No other files import it (verified earlier via grep).

In `scripts/test_maser_peak_refactor.py`, update the import line:
```python
from candel.model.model_H0_maser import (
    MaserDiskModel, build_grid_config, marginalise_spots,
    _phi_bounds,
)
```

(Remove `find_peak_rphi` from the import.)

- [ ] **Step 6: Verify it compiles and runs**

Run: `cd /mnt/users/rstiskalek/CANDEL && python scripts/test_maser_peak_refactor.py`

The log-likelihood should be close to (but not necessarily identical to) the reference value. Check:
- `ll_total` is finite and within ~5 of the reference (the scan grid is wider so peak locations may shift slightly, but the integral should be stable)
- `r_star` values are all positive and in [0.05, 5.0]
- `phi_star` values are within their respective `[phi_lo, phi_hi]` bounds

- [ ] **Step 7: Commit**

```bash
git add candel/model/model_H0_maser.py scripts/test_maser_peak_refactor.py
git commit -m "refactor: simplify maser peak finder — drop GN, use scan-only pipeline"
```

---

### Task 4: Run mock closure test to validate

The refactored peak finder must produce unbiased H0 inference. Run 1-2 mock closure tests to confirm.

**Files:**
- No file changes — just running `scripts/mocks/run_mock_maser_disk.py`

- [ ] **Step 1: Run a single mock closure test**

Run: `cd /mnt/users/rstiskalek/CANDEL && python scripts/mocks/run_mock_maser_disk.py --n_mocks 1 --seed 42 --num_warmup 500 --num_samples 500`

Check that:
- MCMC converges (no divergences, R-hat < 1.05)
- H0 posterior covers the true value (|bias| < 2σ)
- No NaN or Inf in the chain

- [ ] **Step 2: Run 5 mocks if single passed**

Run: `cd /mnt/users/rstiskalek/CANDEL && python scripts/mocks/run_mock_maser_disk.py --n_mocks 5 --seed 0 --num_warmup 500 --num_samples 500`

Check mean bias across mocks is < 1σ for all tracked parameters.

- [ ] **Step 3: Commit final cleanup**

If any scan grid constants needed tuning (e.g. `SCAN_DELTA_R`, `SCAN_DELTA_PHI`, `SCAN_NR`, `SCAN_NPHI`), commit the final values:

```bash
git add candel/model/model_H0_maser.py
git commit -m "refactor: finalize scan grid constants after mock validation"
```

---

### Task 5: Clean up

- [ ] **Step 1: Remove the reference file**

```bash
rm scripts/maser_peak_reference.npz
```

- [ ] **Step 2: Decide whether to keep `scripts/test_maser_peak_refactor.py`**

If the mock closure tests pass, the regression script has served its purpose. Either delete it or keep it as a lightweight smoke test. Ask the user.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: clean up maser peak refactor artifacts"
```

---

## Summary of changes

| Before | After |
|--------|-------|
| `find_peak_rphi` (~110 lines): analytical init + hand-coded GN with Jacobians, damping, regularization | `_analytical_init` (~25 lines): analytical init + stacked argmax for phi candidates |
| Soft-argmax scan embedded in `marginalise_spots` (~30 lines, unnamed) | `_scan_peak` (~35 lines): named function, wider grid, evaluates real likelihood |
| 3-stage pipeline: init → GN → scan → integrate | 2-stage pipeline: init → scan → integrate |
| Manual `best`/`best_s` accumulation for 4 phi candidates | `jnp.stack` + `jnp.argmax` |
| `SCAN_NR=7, SCAN_NPHI=9` (narrow, relied on GN accuracy) | `SCAN_NR=11, SCAN_NPHI=15` with wider half-widths (compensates for no GN) |

Net effect: ~60 fewer lines, no hand-coded Jacobians, clearer separation of concerns.
