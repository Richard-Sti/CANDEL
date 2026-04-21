# Numerical accuracy of the maser disk likelihood

This document summarises what we learned about the numerical integration
of the per-spot maser likelihood, and the choices made as a result.

## The problem

Each maser spot's likelihood is an integral over the azimuthal angle φ
(and, in Mode 2, also over the angular radius r_ang). The integrand is a
Gaussian-like peak in φ whose width is set by the ratio of position
errors to the angular radius. For the five MCP galaxies (NGC5765b,
UGC3789, CGCG074-064, NGC6264, NGC6323), position errors are ~30 μas
and radii are ~0.3–1 mas, giving φ peaks of ~2° FWHM. For NGC4258,
position errors are ~3 μas and radii are ~4–8 mas, giving φ peaks of
~0.06° — roughly 30× narrower.

This difference means:
- The MCP galaxies converge with ~250 φ grid points (arccos-spaced).
- NGC4258 needs ~30,000 uniform φ points to reach sub-nat accuracy.
- NGC4258 cannot use Mode 2 (marginalise r + φ jointly) because the
  2D grid would be prohibitively large.

## Float32 precision

Working in float32, the position χ² requires computing `(x_obs − X)² / σ²`.
When positions are stored in milli-arcseconds, NGC4258 spots have
`x_obs ≈ 0.1 mas` and `X ≈ 0.1001 mas`, so the subtraction
`0.1 − 0.1001 = −0.0001` retains only 3 significant digits. This
catastrophic cancellation corrupts the likelihood by hundreds of nats.

We addressed this in two ways:

1. **Micro-arcsecond units.** Positions are now stored in μas internally
   (×1000 on data load). The same subtraction becomes `100 − 100.1 = −0.1`,
   retaining 6 significant digits. This completely fixes the five MCP
   galaxies, where float32 convergence tests now match float64 exactly.

2. **Mixed precision for NGC4258.** Even in μas, NGC4258 positions
   (~4000 μas with ~3 μas errors) give only 4 significant digits in the
   subtraction — still marginal. The position residuals are therefore
   computed in float64 inside `_phi_integrand`. This is the only float64
   operation in the forward model; everything else (velocity χ², acceleration
   χ², logsumexp, observables) runs in float32. The overhead is negligible
   (~3 MB temporary arrays for inference-sized grids). Verified to give
   results identical to full float64 for NGC4258.

## Convergence results

### Mode 2 (5 MCP galaxies)

Tested against a 20001 × 20001 brute-force reference. Values are Δ nats
(total over all spots) relative to the reference.

**Adaptive r grid** (default φ grids, varying n_r_local):

| Galaxy | n=51 | n=101 | n=151 | n=201 | n=251 |
|--------|------|-------|-------|-------|-------|
| NGC5765b | +0.34 | +0.23 | +0.14 | default | +0.02 |
| UGC3789 | +0.94 | +0.22 | +0.10 | default | +0.04 |
| CGCG074-064 | +1.32 | +0.24 | +0.11 | default | +0.04 |
| NGC6264 | −1.63 | +0.11 | +0.05 | default | +0.02 |
| NGC6323 | +0.35 | +0.09 | +0.04 | default | +0.01 |

**φ grid** (adaptive r=201, varying G_phi_half):

| Galaxy | G=51 | G=101 | G=201 | G=251 | G=401 |
|--------|------|-------|-------|-------|-------|
| NGC5765b | −9.6 | +0.0003 | +0.04 | default | +0.04 |
| UGC3789 | −4.9 | −0.19 | +0.05 | default | +0.05 |
| CGCG074-064 | −21.3 | −0.28 | +0.07 | default | +0.06 |
| NGC6264 | −5.5 | +0.11 | +0.03 | default | +0.03 |
| NGC6323 | −0.3 | +0.02 | +0.02 | default | +0.02 |

**Defaults chosen:** `n_r_local = 201`, `G_phi_half = 251`. Worst
residual vs brute-force: 0.06 nats (CGCG). Both grids are fully
converged — doubling the resolution changes the answer by <0.001 nats.

### Mode 1 φ-marginal (uniform brute-force grids)

Tested at r_ang = 0.5×, 1.0×, 2.0× the per-spot centering estimate.
Reference: 200001 uniform points on [0, 2π].

**5 MCP galaxies:** converge by n = 5001 at all scales.

**NGC4258:**

| scale | n=1001 | n=5001 | n=10001 | n=30001 | n=50001 |
|-------|--------|--------|---------|---------|---------|
| 0.5× | −7776 | −149 | −0.37 | ~0 | 0.00 |
| 1.0× | −4306 | −42 | −0.27 | ~0 | 0.00 |
| 2.0× | −7996 | −129 | −3.4 | ~0 | 0.00 |

**Production setting:** per-type grids with restricted φ ranges:
systemic n=50001 on [-90°, 90°] (max|Δ|=0.00004),
red n=30001 on [0°, 180°] (exact),
blue n=20001 on [180°, 360°] (exact).
The trapezoidal rule converges exponentially for these Gaussian-like
integrands, so even the systemic grid is well into the converged regime.

## Why the φ peak narrows away from the correct r

The φ integrand is broadest at the correct r_ang because both position
and velocity constraints are simultaneously satisfied over a range of φ.
At the wrong r, the position constraint alone forces φ into a narrow
range (the only φ that puts the spot at the observed sky position given
the wrong radius). The velocity constraint, which is broader, becomes
irrelevant. The result is that the φ peak FWHM scales roughly as
σ_pos / r_ang — hence NGC4258's ~10× smaller errors produce ~10× narrower
peaks than the MCP galaxies.

## The k-means seeding issue

Spot classification (blue/systemic/red) uses `scipy.cluster.vq.kmeans2`
with `minit="++"`. Despite the velocity clusters being well-separated,
the random initialisation occasionally produced different classifications
across calls, causing apparent ~500 nat discrepancies in convergence tests.
Fixed by seeding with `seed=42`. The classifications are now reproducible.

## Architecture of the φ integration

All φ methods ultimately call `_phi_integrand`, which computes the raw
log-likelihood integrand at given (r, φ) points using the model's stored
spot data. Callers differ only in their grid choice and integration:

- **`_eval_marginal_phi`** (Mode 2, MCP galaxies): optimised HV reflection
  symmetry, arccos grids for HV spots, two-cluster grid for systemic spots.
  Simpson's rule for HV, trapezoidal for systemic.
- **`_eval_bruteforce_phi`** (Mode 1, NGC4258): per-type uniform grids
  (systemic [-90°, 90°], red [0°, 180°], blue [180°, 360°]),
  trapezoidal rule. Grid sizes configurable per type.

The convergence scripts (`convergence_grids.py`, `convergence_phi_marginal.py`)
also call `_phi_integrand` directly for the brute-force references,
ensuring the reference and the model use identical physics.

## 2026-04-18 GPU refactor

The φ integrand evaluation was split into two methods:

- **`_r_precompute(r_ang, idx, ...)`** gathers per-spot data and computes
  all r-only quantities (`warp_geometry`, Keplerian v_kep/γ/z_g, projection
  coefficients pA..pD, per-spot variances, log-normalisations, and ecc
  r-only precomputes).
- **`_phi_eval(r_pre, sin_phi, cos_phi)`** consumes the precomputed dict
  and computes (r, φ)-dependent observables + χ² + the returned log_f.

`_phi_integrand` remains as a thin backward-compat wrapper for
`bruteforce_ll_mode1` and Mode 0 sampling. Production Mode 1/2 paths
(`_eval_phi_marginal`) now call `_r_precompute` **once per group batch**
instead of once per sub-range, eliminating 2-3× redundant r-only work
and data gathers.

Additionally, `_eval_phi_marginal` now evaluates each group's φ marginal
over a single **concatenated** (sin_phi, cos_phi, log-weight) grid
(precomputed in `_build_phi_subranges` as `self._phi_concat`) via a
single `logsumexp`. This eliminates the staged
`logsumexp(jnp.stack(sub_lps))` combine step. Trapezoidal weights on
disjoint sub-ranges concatenate cleanly (each endpoint keeps its h/2
weight), so the integral is exactly preserved. Reduction-order change
introduces |Δ log L| ≤ 1e-12 on float64 (verified against the pinned
baseline in `scripts/megamaser/baseline_gpu_refactor.npz`).

For Mode 2 sys-without-accel spots (`_n_sys_uncons > 0`), the r grid is
now tagged `shared_r = True` in `_build_r_grids_mode2` and passes a
`(n_r,)`-shaped grid (no broadcast to `(N, n_r)`). `_phi_eval_shared_r`
evaluates X/Y/V/A at `(n_r, n_phi)` and only broadcasts to
`(N, n_r, n_phi)` at the residual step. The current production data
(CGCG, NGC5765b, NGC6264, NGC6323, UGC3789) have 0 sys-uncons spots, so
this code path is dormant — ready for future datasets.

Circular V is only evaluated when `ecc is None` (previously the circular
V computed in `_observables_from_precomputed` was always run, then
overwritten under eccentricity). The χ² term for acceleration is gated
on a static `has_any_accel` Python flag per group, computed once in
`_build_spot_indices` as `self._group_has_accel`.

**Benchmarks** (`scripts/megamaser/bench_gpu_refactor.py`, CPU, float64,
n_iter=30 × 3 trials, mean):

| Galaxy      | Mode | fwd before | fwd after | grad before | grad after |
|-------------|------|-----------:|----------:|------------:|-----------:|
| CGCG074-064 | 2    | 309 ms     | 299 ms    | 1878 ms     | 1856 ms    |
| NGC4258     | 1    | 24.1 ms    | 5.3 ms    | 127 ms      | 76 ms      |

Mode 2 CPU is neutral-to-slightly-faster (noise-dominated at this
scale). Mode 1 CPU shows a ~4.3× forward and ~1.7× gradient speedup
because NGC4258 has 3 HV sub-ranges and 2 sys sub-ranges per
evaluation — the hoisted r-precompute avoids repeating that work per
sub-range. GPU speedups are expected to be larger for Mode 2 (concat
reduces kernel-launch count and backward-pass memory traffic).

**Numerical guarantee:** all six production galaxies (5 MCP + NGC4258)
agree with the pre-refactor baseline log-L to |Δ| ≤ 1e-12 nats. See
`scripts/megamaser/test_gpu_refactor.py` for the regression harness.

## 2026-04-20 Numerical stability refactor

### Unphysical-proposal sqrt guards

Four `jnp.maximum(..., 1e-6)` guards were added to clamp denominators
before taking square roots in the relativistic factors and the eccentric
orbit denominator:

- `gamma = 1/sqrt(max(1 - beta^2, 1e-6))` in `_precompute_r_quantities`
- `one_plus_z_g = 1/sqrt(max(1 - C_g*M_BH/rD, 1e-6))` (gravitational redshift)
- `denom = max(1 + ecc*cos_d, 1e-6)` in the eccentric branches of
  `_phi_eval` and `_phi_eval_shared_r`
- `gamma_e = 1/sqrt(max(1 - beta_e2, 1e-6))` in the eccentric branch

Without these guards, an unphysical proposal (e.g. nearly relativistic
orbital speed, or eccentricity approaching 1 with periapsis facing the
observer) produces `sqrt(0) = 0 → 1/0 = +inf → V = +inf → dv = -inf →
chi2 = +inf → log_f = -inf`. JAX's `logsumexp` treats an all-(-inf) input
as `-inf` (not NaN), which poisons the NUTS gradient. The clamp threshold
1e-6 is well below the prior support for any physical disk configuration,
so it has no effect on posterior inference.

### lnorm / chi2 split in `_phi_eval`

Previously `_phi_eval` returned `lnorm - 0.5*chi2` (the full
log-integrand). It now returns only `-0.5*chi2`, and `_eval_phi_marginal`
adds `lnorm` back *after* the `logsumexp`:

```python
ps_b = lnorm_b + logsumexp(neg_half_chi2 + log_w, ...)
```

This uses the identity `logsumexp(c + a_i) = c + logsumexp(a_i)` for a
constant `c` (here `lnorm` depends only on per-spot variances and is
constant over the φ grid). The formulation is algebraically identical to
the previous one.

The precision benefit matters in float32 when chi2 is large. In the old
formulation the values entering `logsumexp` had magnitude ~lnorm ~
-chi2/2; for chi2 ~ 10^4 this is ~5000, and float32 machine epsilon gives
an absolute error of ε×5000 ≈ 6×10⁻⁴ nats per spot. In the new
formulation the values entering `logsumexp` are `-chi2/2 + log_w`, which
are bounded from above by `log_w_max ≈ -7` regardless of chi2. The
max-subtraction in logsumexp therefore operates on values near 0, giving
float32 absolute errors of ε×7 ≈ 8×10⁻⁷ nats — roughly three orders of
magnitude better for large chi2.

`_phi_integrand` (the backward-compat wrapper) still assembles the full
`log_f = lnorm + neg_half_chi2`, so Mode 0 sampling and the brute-force
convergence reference are unaffected.

These changes are purely numerical: the log-likelihood is identical to
prior results to within float32 rounding.
