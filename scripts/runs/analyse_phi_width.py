"""Measure phi peak width from real maser spot data.

For each spot, evaluates the 1D log-likelihood as a function of phi at
fiducial r, measures the peak width (sigma_phi), and reports statistics
for HV and systemic spots separately. Also checks bimodal separation.
"""
import sys
sys.path.insert(0, "/mnt/users/rstiskalek/CANDEL")

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")

from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.model.model_H0_maser import (
    _spot_log_likelihood_on_grid, warp_geometry, _analytical_init,
    _gauss_newton_refine, _scan_peak, _phi_bounds,
    SCAN_DELTA_R, SCAN_DELTA_PHI, SCAN_NR, SCAN_NPHI,
)


# --- Fiducial parameters (roughly Pesce+2020 posterior mean for CGCG074-064)
FIDUCIAL = dict(
    D=95.0,          # Mpc (angular diameter distance)
    M_BH=8.0e6,      # M_sun
    x0=0.0,
    y0=0.0,
    i0_deg=87.0,
    Omega0_deg=120.0,
    dOmega_dr_deg=0.0,
    di_dr_deg=0.0,
    sigma_x_floor=0.01,
    sigma_y_floor=0.01,
    sigma_v_sys=7.0,
    sigma_v_hv=5.0,
    sigma_a_floor=0.5,
    A_thr=2.0,
    sigma_det=1.0,
)

GALAXIES = {
    "CGCG074-064": dict(v_cmb_obs=7172.2, v_helio_to_cmb=263.3),
    "NGC5765b": dict(v_cmb_obs=8525.7, v_helio_to_cmb=210.1),
    "UGC3789": dict(v_cmb_obs=3325.3, v_helio_to_cmb=419.9),
    "NGC6264": dict(v_cmb_obs=10128.0, v_helio_to_cmb=-3.3),
    "NGC6323": dict(v_cmb_obs=7805.0, v_helio_to_cmb=-33.5),
}

DATA_ROOT = "data/Megamaser"


def measure_phi_widths(galaxy_name, fiducial=FIDUCIAL):
    ginfo = GALAXIES[galaxy_name]
    data = load_megamaser_spots(
        DATA_ROOT, galaxy=galaxy_name,
        v_cmb_obs=ginfo["v_cmb_obs"],
        v_helio_to_cmb=ginfo["v_helio_to_cmb"])

    n = data["n_spots"]
    x_obs = jnp.asarray(data["x"])
    sigma_x = jnp.asarray(data["sigma_x"])
    y_obs = jnp.asarray(data["y"])
    sigma_y = jnp.asarray(data["sigma_y"])
    v_obs = jnp.asarray(data["velocity"])
    a_obs = jnp.asarray(data["a"])
    sigma_a = jnp.asarray(data["sigma_a"])
    accel_measured = jnp.asarray(data["accel_measured"])
    spot_type = data["spot_type"]
    is_sys = data["is_systemic"]
    is_hv = data["is_highvel"]

    phi_lo, phi_hi = _phi_bounds(spot_type, n)

    p = fiducial
    v_sys = ginfo["v_cmb_obs"] - ginfo["v_helio_to_cmb"]
    i0 = jnp.deg2rad(p["i0_deg"])
    Omega0 = jnp.deg2rad(p["Omega0_deg"])
    dOmega_dr = jnp.deg2rad(p["dOmega_dr_deg"])
    di_dr = jnp.deg2rad(p["di_dr_deg"])

    # Find peak r for each spot via the analytical init + GN + scan pipeline
    r_init, phi_init = _analytical_init(
        x_obs, y_obs, v_obs, a_obs, accel_measured,
        phi_lo, phi_hi,
        p["x0"], p["y0"], p["D"], p["M_BH"], v_sys,
        i0, di_dr, Omega0, dOmega_dr,
        p["sigma_v_sys"], p["sigma_v_hv"],
        sigma_a, p["sigma_a_floor"])

    r_ref, phi_ref = _gauss_newton_refine(
        r_init, phi_init, phi_lo, phi_hi,
        x_obs, y_obs, v_obs, a_obs, accel_measured,
        p["x0"], p["y0"], p["D"], p["M_BH"], v_sys,
        i0, di_dr, Omega0, dOmega_dr,
        p["sigma_x_floor"], p["sigma_y_floor"],
        p["sigma_v_sys"], p["sigma_v_hv"],
        sigma_a, p["sigma_a_floor"])

    r_star, phi_star = _scan_peak(
        r_ref, phi_ref, phi_lo, phi_hi,
        x_obs, sigma_x, y_obs, sigma_y,
        v_obs, a_obs, sigma_a, accel_measured,
        p["x0"], p["y0"], p["D"], p["M_BH"], v_sys,
        i0, di_dr, Omega0, dOmega_dr,
        p["sigma_x_floor"], p["sigma_y_floor"],
        p["sigma_v_sys"], p["sigma_v_hv"],
        p["sigma_a_floor"], p["A_thr"], p["sigma_det"])

    r_star = np.asarray(r_star)
    phi_star = np.asarray(phi_star)

    # Fine 1D phi grid for each spot
    N_phi_fine = 2001
    results = []

    for k in range(n):
        lo_k = float(phi_lo[k])
        hi_k = float(phi_hi[k])
        phi_fine = jnp.linspace(lo_k, hi_k, N_phi_fine)
        r_grid_k = jnp.array([r_star[k]])  # single r value

        ll_2d = _spot_log_likelihood_on_grid(
            r_grid_k, phi_fine,
            x_obs[k], sigma_x[k], y_obs[k], sigma_y[k],
            v_obs[k], a_obs[k], sigma_a[k], accel_measured[k],
            p["x0"], p["y0"], p["D"], p["M_BH"], v_sys,
            i0, di_dr, Omega0, dOmega_dr,
            p["sigma_x_floor"], p["sigma_y_floor"],
            p["sigma_v_sys"], p["sigma_v_hv"],
            p["sigma_a_floor"], p["A_thr"], p["sigma_det"])

        ll_1d = np.asarray(ll_2d[0, :])  # shape (N_phi_fine,)
        phi_np = np.asarray(phi_fine)

        # Find peaks
        peak_idx = np.argmax(ll_1d)
        ll_max = ll_1d[peak_idx]
        phi_peak = phi_np[peak_idx]

        # Measure sigma_phi as half-width at half-maximum (ll_max - 0.5)
        threshold = ll_max - 0.5  # 1-sigma in Gaussian log-likelihood
        above = ll_1d >= threshold
        if above.sum() > 1:
            indices = np.where(above)[0]
            sigma_phi = 0.5 * (phi_np[indices[-1]] - phi_np[indices[0]])
        else:
            sigma_phi = (phi_np[1] - phi_np[0])  # unresolved

        # Find second peak (look for local max separated from first)
        ll_shifted = ll_1d - ll_max
        # Mask out region within ±0.2 rad of first peak
        mask = np.abs(phi_np - phi_peak) > 0.2
        if mask.sum() > 0:
            peak2_idx = np.argmax(ll_1d * mask + (~mask) * (-1e30))
            phi_peak2 = phi_np[peak2_idx]
            ll_peak2 = ll_1d[peak2_idx]
            delta_ll_peaks = ll_max - ll_peak2
            peak_separation = abs(phi_peak2 - phi_peak)
        else:
            phi_peak2 = np.nan
            delta_ll_peaks = np.inf
            peak_separation = 0.0

        stype = spot_type[k]
        results.append(dict(
            k=k, spot_type=stype,
            r_star=r_star[k], phi_star=phi_star[k],
            phi_peak=phi_peak, sigma_phi=sigma_phi,
            phi_peak2=phi_peak2, delta_ll_peaks=delta_ll_peaks,
            peak_separation=peak_separation,
            ll_max=ll_max,
        ))

    return results


def report(results, galaxy_name):
    print(f"\n{'='*70}")
    print(f"  {galaxy_name}: phi peak width analysis")
    print(f"{'='*70}")

    for label, mask_fn in [("HV (red+blue)", lambda r: r["spot_type"] in ("r", "b")),
                           ("Systemic", lambda r: r["spot_type"] == "s")]:
        subset = [r for r in results if mask_fn(r)]
        if not subset:
            print(f"\n  {label}: no spots")
            continue

        sigmas = np.array([r["sigma_phi"] for r in subset])
        seps = np.array([r["peak_separation"] for r in subset])
        dll = np.array([r["delta_ll_peaks"] for r in subset])

        print(f"\n  {label} ({len(subset)} spots):")
        print(f"    sigma_phi: min={np.min(sigmas):.4f} rad ({np.degrees(np.min(sigmas)):.2f} deg)")
        print(f"               median={np.median(sigmas):.4f} rad ({np.degrees(np.median(sigmas)):.2f} deg)")
        print(f"               max={np.max(sigmas):.4f} rad ({np.degrees(np.max(sigmas)):.2f} deg)")
        print(f"    Peak separation: median={np.median(seps):.3f} rad ({np.degrees(np.median(seps)):.1f} deg)")
        print(f"    Delta_ll between peaks: median={np.median(dll):.1f}")

        # How many have well-separated bimodal peaks (delta_ll < 10)?
        bimodal = dll < 10
        print(f"    Bimodal (delta_ll < 10): {bimodal.sum()}/{len(subset)}")

    # Overall recommendation
    all_sigmas = np.array([r["sigma_phi"] for r in results])
    sigma_95 = np.percentile(all_sigmas, 95)
    # Want ~5 points per sigma -> dphi = sigma_95 / 5
    # But also want at least 2 points per narrowest peak
    dphi_rec = min(np.min(all_sigmas) / 2.0, sigma_95 / 5.0)
    window = 0.5  # rad half-width
    n_phi_rec = int(np.ceil(2.0 * window / dphi_rec)) | 1  # ensure odd

    print(f"\n  Recommendation:")
    print(f"    Narrowest sigma_phi = {np.min(all_sigmas):.4f} rad ({np.degrees(np.min(all_sigmas)):.2f} deg)")
    print(f"    95th percentile sigma_phi = {sigma_95:.4f} rad ({np.degrees(sigma_95):.2f} deg)")
    print(f"    Recommended dphi = {dphi_rec:.5f} rad")
    print(f"    For ±0.5 rad window: N_phi = {n_phi_rec}")

    # Check bimodal window question
    all_seps = np.array([r["peak_separation"] for r in results])
    all_dll = np.array([r["delta_ll_peaks"] for r in results])
    bimodal_mask = all_dll < 10
    if bimodal_mask.sum() > 0:
        max_sep = np.max(all_seps[bimodal_mask])
        print(f"\n  Bimodal peaks:")
        print(f"    Max separation among bimodal spots: {max_sep:.3f} rad ({np.degrees(max_sep):.1f} deg)")
        if max_sep < 1.0:
            print(f"    A single ±0.5 rad window centered at midpoint captures both peaks.")
        else:
            print(f"    Two separate windows needed (separation > 1.0 rad).")
    else:
        print(f"\n  No bimodal spots detected (all delta_ll > 10).")

    # Print per-spot detail table
    print(f"\n  Per-spot details:")
    print(f"  {'k':>3s} {'type':>4s} {'r*':>6s} {'phi*':>7s} {'phi_pk':>7s} "
          f"{'sigma_phi':>10s} {'phi_pk2':>7s} {'dll':>6s} {'sep':>6s}")
    for r in results:
        print(f"  {r['k']:3d} {r['spot_type']:>4s} {r['r_star']:6.3f} "
              f"{np.degrees(r['phi_star']):7.1f} {np.degrees(r['phi_peak']):7.1f} "
              f"{np.degrees(r['sigma_phi']):10.2f} "
              f"{np.degrees(r['phi_peak2']):7.1f} "
              f"{r['delta_ll_peaks']:6.1f} "
              f"{np.degrees(r['peak_separation']):6.1f}")


if __name__ == "__main__":
    for galaxy in ["CGCG074-064", "NGC5765b", "UGC3789", "NGC6264", "NGC6323"]:
        try:
            results = measure_phi_widths(galaxy)
            report(results, galaxy)
        except Exception as e:
            print(f"\n{galaxy}: FAILED - {e}")
