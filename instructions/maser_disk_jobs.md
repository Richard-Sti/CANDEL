# Running Maser Disk Model Jobs

## Overview

The maser disk model fits VLBI maser spot data (positions, velocities,
accelerations) to infer galaxy distances and black hole masses.

- **Runner script:** `scripts/megamaser/run_maser_disk.py`
- **Config:** `scripts/megamaser/config_maser.toml`
- **Submit scripts:** `scripts/megamaser/submit_nuts_ngc5765b.sh`, `submit_nss_ngc5765b.sh`
- **Output:** `results/Maser/<galaxy>_<sampler>_<dist_tag>.hdf5` + corner plots

## Usage

```bash
python scripts/megamaser/run_maser_disk.py <galaxy> [--sampler nuts|nss] [options]
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--sampler` | from config (`nss`) | `nuts` or `nss` |
| `--mode` | from config | `mode0` / `mode1` / `mode2` — see "Sampling modes" |
| `--seed` | 42 | RNG seed |
| `--num-warmup` | 1000 | NUTS warmup steps |
| `--num-samples` | 1000 | NUTS sample count |
| `--n-live` | 5000 | NSS live points |
| `--num-mcmc-steps` | 0 (=ndim) | NSS slice steps |
| `--num-delete` | 250 | NSS contraction batch |
| `--termination` | -3 | NSS stopping criterion |
| `--f-grid` | 1.0 | Scale every phi/r grid size (`n_phi_hv_high`, `n_phi_hv_low`, `n_phi_sys`, their `_mode1` variants, `n_r_local`, `n_r_brute`) by this factor; results rounded to nearest odd integer, min 3. Applies to global `[model]` keys and per-galaxy overrides. |
| `--no-ecc` | off | Disable eccentricity model |
| `--no-quadratic-warp` | off | Disable quadratic disk warp |
| `--save-map`, `--load-map` | — | Dump/load MAP init to TOML |

CLI args override config values.

## Sampling modes

A single config key `model/mode` (with optional per-galaxy override)
selects how per-spot angular radius `r_ang` and azimuth `φ` are handled:

| Mode | `r_ang` | `φ` | Sample sites | Typical use |
|------|---------|-----|--------------|-------------|
| `mode0` | sampled | sampled | `r_ang`, `phi_u` | full joint; useful as a reference against the analytic marginals |
| `mode1` | sampled | marginalised | `r_ang` | required when position errors are too small for a practical Mode-2 r-grid (NGC4258) |
| `mode2` | marginalised | marginalised | none per-spot | default for the five MCP galaxies; compatible with nested sampling |

**Mode 0 details.** Each spot gets two latent parameters: `r_ang_i ~ Uniform(r_lo, r_hi)` and an auxiliary `phi_u_i ~ Uniform(0, 1)` which is deterministically mapped to `φ_i` over the same allowed support used by the Mode 1/2 integrator (red: `[-outer, outer]`; blue: `[π-outer, π+outer]`; sys: `[-135°, -45°] ∪ [45°, 135°]`). This yields an exact uniform prior on the per-spot allowed φ union, consistent with the Mode 1/2 marginals (`∫ Mode 0 p(r, φ(u)) du = Mode 1 marginal / π` to float64 precision).

**Mode 1 details.** Samples `r_ang_i` per spot; φ is marginalised analytically via trapezoidal integration over the configured sub-ranges. Required for any galaxy with `forbid_marginalise_r = true`.

**Mode 2 details.** Both `r_ang_i` and φ are marginalised: per-spot adaptive sinh r-grid (HV + sys-with-accel) and log-uniform brute grid (sys-without-accel), folded with the unified φ grid. No per-spot NUTS dimensions — lowest-dimensional mode; required by nested sampling.

**Per-galaxy override.** Set `mode = "mode1"` (or other) in a
`[model.galaxies.<NAME>]` block to pin that galaxy regardless of the
global default. NGC4258 is pinned to `mode1` with `forbid_marginalise_r = true` guarding against `mode2`.

**CLI override.** `--mode mode0|mode1|mode2` overrides the global default (but *not* per-galaxy overrides).

## Available galaxies

Defined in `config_maser.toml` under `[model.galaxies]`:

| Galaxy | v_sys_obs (km/s) |
|--------|------------------|
| NGC5765b | 8327.6 |
| CGCG074-064 | 6908.9 |
| NGC4258 | 472.0 |
| NGC6264 | 10131.3 |
| NGC6323 | 7838.5 |
| UGC3789 | 2905.4 |

## Initialization: SOBOL + ADAM

Set `init_method = "sobol_adam"` in `[inference]` (default in config).

1. **Sobol survey:** 2^14 = 16,384 quasi-random points in prior volume
2. **Start selection:** Top M=10 distinct points by log-density
3. **Parallel Adam:** Optimizes all starts simultaneously (cosine LR schedule)
4. **Result:** Best MAP point used for `init_to_value()` in NUTS

Optimizer settings under `[optimise]` in config (all optional with defaults):

```toml
[optimise]
log2_N = 14           # Sobol sample count (2^N)
M = 10                # Number of Adam starts
n_steps = 5000        # Adam steps per start
lr = 0.1              # Peak learning rate
lr_end = 0.005        # Minimum learning rate
n_restarts = 3        # Cosine restart cycles
```

## Submitting to GPU queue

```bash
# NGC5765b with NUTS (sobol_adam init from config)
bash scripts/megamaser/submit_nuts_ngc5765b.sh

# NGC5765b with NSS
bash scripts/megamaser/submit_nss_ngc5765b.sh

# Custom: any galaxy, any sampler, any mode
ROOT=/mnt/users/$USER/CANDEL
PYTHON=$ROOT/venv_gpu_candel/bin/python
addqueue -q cmbgpu -s -m 16 --gpus 1 \
    $PYTHON -u $ROOT/scripts/megamaser/run_maser_disk.py NGC6264 \
    --sampler nuts --num-warmup 2000 --num-samples 2000 --mode mode2

# Mode 0 reference run (large per-spot NUTS dimension — use optgpu)
addqueue -q optgpu -s -m 16 --gpus 1 \
    $PYTHON -u $ROOT/scripts/megamaser/run_maser_disk.py NGC5765b \
    --sampler nuts --mode mode0
```

## Grid sizes and numerical accuracy

See [`docs/maser_numerical_accuracy.md`](../docs/maser_numerical_accuracy.md)
for full convergence test results and the φ-integrand analysis.

**Mode 2 globals** (MCP galaxies: NGC5765b, UGC3789, CGCG074-064,
NGC6264, NGC6323). Derived from the 2026-04-18 convergence sweep
(|Δ ll_disk| ≤ 0.12 nats across all five):

| Grid | Points | Purpose | Config key |
|------|--------|---------|------------|
| HV inner wedge (φ) | 1001 | ±45° dense band | `model/n_phi_hv_high` |
| HV outer wings (φ) | 301 / wing | ±45°→±90° per wing | `model/n_phi_hv_low` |
| Systemic (φ) | 1501 / sub-range | `[-135°,-45°]` and `[45°,135°]` | `model/n_phi_sys` |
| Radius (HV + sys-w/accel) | 251 | adaptive per-spot sinh | `model/n_r_local` |
| Radius (sys-no-accel) | 501 | log-uniform brute grid | `model/n_r_brute` |

**Mode 1 globals** (`n_phi_*_mode1` override the Mode 2 defaults when
`mode = "mode1"` so per-spot φ peaks are resolved):

| Grid | Points | Config key |
|------|--------|------------|
| HV inner wedge (φ) | 2001 | `model/n_phi_hv_high_mode1` |
| HV outer wings (φ) | 501 / wing | `model/n_phi_hv_low_mode1` |
| Systemic (φ) | 3001 / sub-range | `model/n_phi_sys_mode1` |

**NGC4258 per-galaxy override** (`[model.galaxies.NGC4258]`): Mode 1 with
`n_phi_hv_high = 10001`, `n_phi_hv_low = 2001`, `n_phi_sys = 10001`
(tighter grids needed because position errors of ~3 μas produce
sub-milliradian per-spot φ peaks).

**Mode 0** uses no φ or r grids — per-spot likelihood is evaluated
pointwise at the sampled `(r_ang_i, φ_i)`. The `n_phi_*` and `n_r_*`
keys have no effect when `mode = "mode0"`.

Internal positions are in **μas** (micro-arcseconds) for float32
stability. Angular radii remain in mas. Mixed precision (float64 position
residuals) is used for NGC4258.

## Dense-mass handling

- **Joint fit** (`galaxy = "joint"`): full dense mass across all sampled parameters.
- **Single-galaxy, `dense_mass_globals = true`**: one dense block over the global scalars (`D_c, eta, x0, y0, i0, Omega0, dOmega_dr, di_dr, σ-floors, dv_sys, ecc params, quad-warp params`). Per-spot sites (`r_ang`, `phi_u`) stay diagonal — this is what Mode 0 in particular needs for reasonable mixing.
- **Otherwise**: `dense_mass_blocks` from `[inference]`.

## Dense mass blocks (NUTS)

Correlated parameters are grouped for efficient mass matrix adaptation:

```toml
dense_mass_blocks = [
    ["D_c", "eta", "dv_sys"],
    ["i0", "di_dr", "Omega0", "dOmega_dr"],
    ["x0", "y0"],
]
```

## Output

- **HDF5:** `results/Maser/<galaxy>_<sampler>_Dflat.hdf5`
  - `samples/` group with all parameter chains
  - `D_A`, `M_BH` derived arrays
  - NSS: `log_Z`, `log_Z_err`, `n_eff` as file attributes
- **Plots:** `<galaxy>_<sampler>_Dflat_corner.png`, `_corner_raw.png`, `_spots.png`

## Monitoring

```bash
tail -f python-<JOBID>.out
squeue -u $USER
scancel <JOBID>
```
