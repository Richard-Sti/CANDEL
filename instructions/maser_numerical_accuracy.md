# Megamaser Disk Model: Numerical Accuracy

Summary of grid convergence testing and numerical precision improvements
for the maser disk model's per-spot likelihood marginalisation.

## Background

The per-spot maser likelihood requires marginalising over the orbital
azimuthal angle φ (and optionally the angular radius r_ang). The integrand
is a peaked function whose width depends on position errors and the
spot's location in the disk. Accurate numerical integration requires
grids that resolve these peaks.

**Two modes:**
- **Mode 2** (MCP galaxies): marginalise both r and φ on adaptive grids.
- **Mode 1** (NGC4258): sample r_ang per spot via NUTS, marginalise φ only.

## Internal position units

Positions (x, y, σ_x, σ_y, x0, y0, σ_x_floor, σ_y_floor) are stored
internally in **micro-arcseconds (μas)**, while angular radii (r_ang,
r_ang_ref, di/dr, etc.) remain in **milli-arcseconds (mas)**. The bridge
factor `r_ang * 1e3` appears in all position predictions:

```
X_μas = x0_μas + r_ang_mas × 1e3 × (sin φ sin Ω − cos φ cos Ω cos i)
```

This eliminates catastrophic cancellation in float32 for the 5 MCP
galaxies (positions ~10–1000 μas, errors ~30 μas). NGC4258 positions
(~4000 μas, errors ~3 μas) still require mixed precision — see below.

### Mixed precision

`_phi_integrand` computes position residuals in float64:

```python
dx = x_obs.astype(float64) - X.astype(float64)
chi2_pos = dx² / var_x  # cast back to float32
```

This is the only float64 operation in the forward model. Overhead is
negligible (~2 MB temporary arrays for inference grids). Verified to give
identical results to full float64 for NGC4258.

## Mode 2 grid convergence (5 MCP galaxies)

Reference: 20001 × 20001 brute-force uniform grid.

**R convergence** (adaptive per-spot sinh grid, default φ grids):

| Galaxy | n_spots | n=51 | n=101 | n=151 | n=201 (default) | n=251 |
|--------|---------|------|-------|-------|------------------|-------|
| NGC5765b | 192 | +0.34 | +0.23 | +0.14 | — | +0.02 |
| UGC3789 | 156 | +0.94 | +0.22 | +0.10 | — | +0.04 |
| CGCG074-064 | 192 | +1.32 | +0.24 | +0.11 | — | +0.04 |
| NGC6264 | 51 | −1.63 | +0.11 | +0.05 | — | +0.02 |
| NGC6323 | 68 | +0.35 | +0.09 | +0.04 | — | +0.01 |

**φ convergence** (adaptive r=201, varying G_phi_half):

| Galaxy | G=51 | G=101 | G=201 | G=251 (default) | G=401 | G=801 |
|--------|------|-------|-------|------------------|-------|-------|
| NGC5765b | −9.6 | +0.0003 | +0.04 | — | +0.04 | +0.04 |
| UGC3789 | −4.9 | −0.19 | +0.05 | — | +0.05 | +0.05 |
| CGCG074-064 | −21.3 | −0.28 | +0.07 | — | +0.06 | +0.06 |
| NGC6264 | −5.5 | +0.11 | +0.03 | — | +0.03 | +0.03 |
| NGC6323 | −0.3 | +0.02 | +0.02 | — | +0.02 | +0.02 |

Defaults (`n_r_local=201`, `G_phi_half=251`) are well within the
converged regime. Worst residual vs brute-force: 0.06 nats (CGCG).

## Mode 1 φ-marginal convergence (uniform brute-force grids)

Reference: 200001-point uniform grid on [0, 2π]. Tested at r_ang = 0.5×,
1.0×, 2.0× the per-spot centering estimate.

**5 MCP galaxies** — converge by n=5001 (Δ=0.0000 at all scales).

**NGC4258** — converges by n=50001:

| scale | ref logL | n=1001 | n=5001 | n=10001 | n=50001 |
|-------|----------|--------|--------|---------|---------|
| 0.5× | −88.4M | −7776 | −149 | −0.37 | 0.00 |
| 1.0× | −89716 | −4306 | −42.3 | −0.27 | 0.00 |
| 2.0× | −10.3M | −7996 | −129 | −3.44 | 0.00 |

NGC4258 needs ~10× more φ points than the MCP galaxies due to ~10×
smaller position errors creating narrower φ peaks.

**Production setting:** `n_phi_bruteforce = 30001` for NGC4258. At the
physical r_ang (scale=1.0×), this is within 0.3 nats of the 200001-point
reference — negligible for 358 spots.

## Running convergence tests

```bash
# Mode 2 grids (5 MCP galaxies, ~5 min on RTX 2080 Ti)
./scripts/megamaser/submit_convergence_grids.sh

# φ-marginal (all 6 galaxies, needs ≥12 GB GPU for NGC4258)
./scripts/megamaser/submit_convergence_phi_marginal.sh

# φ-marginal NGC4258 only (needs A6000 or similar)
./scripts/megamaser/submit_convergence_phi_marginal.sh --galaxies NGC4258

# Full float64 (for reference comparison)
./scripts/megamaser/submit_convergence_phi_marginal.sh --float64
```

## Running inference

**MCP galaxies (Mode 2):** marginalise r + φ on adaptive grids.
```bash
# NGC5765b example
addqueue -q gpulong -s -m 16 --gpus 1 \
    venv_gpu_candel/bin/python -u scripts/megamaser/run_maser_disk.py \
    NGC5765b --sampler nuts
```

**NGC4258 (Mode 1):** sample r_ang, bruteforce φ marginal (30001 pts).
```bash
./scripts/megamaser/submit_n4258.sh           # defaults to optgpu
./scripts/megamaser/submit_n4258.sh gpulong   # override queue
```

NGC4258 config (`config_maser.toml`) sets:
```toml
[model.galaxies."NGC4258"]
marginalise_r = false           # Mode 1: sample r_ang per spot
phi_method = "bruteforce"       # uniform φ grid
n_phi_bruteforce = 30001        # sufficient for σ_pos ~ 3 μas
```

## Key implementation details

| Component | Description |
|-----------|-------------|
| `_phi_integrand` | Core physics: geometry → observables → χ² → log-integrand. Used by all φ methods. |
| `_eval_marginal_phi` | Mode 2: optimised HV reflection + two-cluster systemic grids. |
| `_eval_bruteforce_phi` | Mode 1 NGC4258: uniform [0, 2π] grid, calls `_phi_integrand`. |
| `_eval_adaptive_phi_mode1` | Mode 1 (legacy): per-spot adaptive sinh φ grids. |
| `phi_method` config | `"default"` / `"adaptive"` / `"bruteforce"`, per-galaxy override. |
| `marginalise_r` config | Per-galaxy override (NGC4258=false, others=true from global). |
| K-means seed | `seed=42` in `kmeans2` for reproducible spot classification. |

## φ integrand shape

The φ integrand is broadest at the correct r_ang and narrows as r moves
away from the physical value. At the correct r, both position and
velocity constraints are satisfied over a range of φ. At wrong r,
only a narrow φ range produces positions matching (x_obs, y_obs).

For NGC4258 at the physical r_ang (~6 mas), the φ peak FWHM is ~0.001 rad
(~0.06°), requiring ~10× finer grids than the MCP galaxies (~0.04 rad FWHM).

See `scripts/megamaser/plot_phi_integrand_vs_r.py` for visualisation.
