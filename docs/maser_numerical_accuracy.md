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

**Production setting:** `n_phi_bruteforce = 30001`. At the physical
r_ang, this is within ~0.3 nats of the 200001-point reference.

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
- **`_eval_bruteforce_phi`** (Mode 1, NGC4258): uniform [0, 2π] grid,
  trapezoidal rule. Simple and robust, no grid tuning needed.
- **`_eval_adaptive_phi_mode1`** (Mode 1, legacy): per-spot adaptive sinh
  grids centred on the φ peak from a 2×2 position solve. Superseded by
  the brute-force approach for NGC4258.

The convergence scripts (`convergence_grids.py`, `convergence_phi_marginal.py`)
also call `_phi_integrand` directly for the brute-force references,
ensuring the reference and the model use identical physics.
