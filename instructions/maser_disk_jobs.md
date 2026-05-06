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
| `--mode` | from config | `mode1` / `mode2` — see "Sampling modes" |
| `--seed` | 42 | RNG seed |
| `--num-warmup` | 1000 | NUTS warmup steps |
| `--num-samples` | 1000 | NUTS sample count |
| `--n-live` | 5000 | NSS live points |
| `--num-mcmc-steps` | 0 (=ndim) | NSS slice steps |
| `--num-delete` | 250 | NSS contraction batch |
| `--devices` | auto | Local-device sharding for NSS replacement chains or DE population fitness evaluations. Use `1` for the original single-device path or numeric `N` for N visible GPUs on one node. |
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
| `mode1` | sampled | marginalised | `r_ang` | when position errors are small (e.g. NGC4258) |
| `mode2` | marginalised | marginalised | none per-spot | default for the five MCP galaxies; compatible with nested sampling |

**Mode 1 details.** Samples `r_ang_i` per spot; φ is marginalised analytically via trapezoidal integration over the configured sub-ranges.

**Mode 2 details.** Both `r_ang_i` and φ are marginalised: per-spot adaptive sinh r-grid (HV + sys-with-accel) and log-uniform brute grid (sys-without-accel), folded with the unified φ grid. No per-spot NUTS dimensions — lowest-dimensional mode; required by nested sampling.

**Per-galaxy override.** Set `mode = "mode1"` in a
`[model.galaxies.<NAME>]` block to pin that galaxy regardless of the
global default. NGC4258 defaults to `mode1` but can be switched to `mode2`.

**CLI override.** `--mode mode1|mode2` overrides the global default (but *not* per-galaxy overrides).

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
PYTHON=$ROOT/venv_candel/bin/python
addqueue -q cmbgpu -s -m 16 --gpus 1 \
    $PYTHON -u $ROOT/scripts/megamaser/run_maser_disk.py NGC6264 \
    --sampler nuts --num-warmup 2000 --num-samples 2000 --mode mode2
```

### Single-node multi-GPU NSS

NSS can shard its independent replacement chains over multiple local JAX
devices. This is single-process, single-node parallelism only; do not use it
for multi-node MPI-style jobs.

On ARC, request multiple GPUs and pass the same count to the runner:

```bash
bash scripts/megamaser/submit.sh --sampler nss -q medium \
    --galaxy NGC5765b --devices 4
```

When more than one local device is used, NSS multiplies `num_delete` by the
device count. For example, `num_delete=5` on 4 GPUs deletes 20 live points per
NSS iteration and runs 5 replacement chains per GPU. If only one local device
is visible, it falls back to the original single-device behavior.

Validation plan:

1. Run a reduced single-GPU NSS job and record log output, wall time, logZ,
   `n_eff`, and any non-finite likelihood warnings.
2. Run the same reduced job on one ARC node with `--devices 2`, then
   `--devices 4` if available.
3. Confirm the log prints the scaled `num_delete` and the expected
   replacement-chain count per device.
4. Compare posterior summaries and logZ against the single-GPU run before
   launching production jobs.
5. Keep production multi-GPU runs single-node only; do not use MPI or
   multi-node JAX for this path.

### Single-node multi-GPU DE

DE can shard population-axis fitness evaluations over multiple local JAX
devices. This is also single-process, single-node parallelism only. The
configured `optimise/eval_chunk` remains a per-device batch size, so a 4-GPU
run evaluates `4 * eval_chunk` candidates per compiled fitness call.

On ARC, request multiple GPUs and pass the same count to the runner:

```bash
bash scripts/megamaser/submit.sh --sampler de -q medium \
    --galaxy NGC5765b --devices 4
```

Validation plan:

1. Run a reduced single-GPU DE job and record the MAP logP and wall time.
2. Run the same reduced job on one ARC node with `--devices 2`, then
   `--devices 4` if available.
3. Confirm the log prints population fitness sharding and the expected
   per-device `eval_chunk`.
4. Compare the final MAP parameters and logP against the single-GPU run.
5. Keep production multi-GPU runs single-node only; do not use MPI or
   multi-node JAX for this path.

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

Internal positions are in **μas** (micro-arcseconds) for float32
stability. Angular radii remain in mas. Mixed precision (float64 position
residuals) is used for NGC4258.

## Dense-mass handling

- **Joint fit** (`galaxy = "joint"`): full dense mass across all sampled parameters.
- **Single-galaxy, `dense_mass_globals = true`**: one dense block over the global scalars (`D_c, eta, x0, y0, i0, Omega0, dOmega_dr, di_dr, σ-floors, dv_sys, ecc params, quad-warp params`). Per-spot `r_ang` stays diagonal (Mode 1 only).
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

## Auto-resubmit on timeout

Add `--max-retries N` to any submit command. The script delegates to a
watcher that polls `squeue`, checks logs for a completion marker, and
resubmits with `--resume` if incomplete:

```bash
# DE MAP — resubmit up to 5 times
bash scripts/megamaser/submit.sh --sampler de -q cmbgpu --galaxy NGC5765b --max-retries 5

# DE MAP — multiple galaxies
bash scripts/megamaser/submit.sh --sampler de -q cmbgpu --galaxy NGC5765b,NGC6264 --max-retries 5

# NSS — one galaxy
bash scripts/megamaser/submit.sh --sampler nss -q cmbgpu --galaxy NGC5765b --max-retries 5

# NUTS — restarts from scratch (no checkpoint support)
bash scripts/megamaser/submit.sh --sampler nuts -q cmbgpu --galaxy NGC5765b --max-retries 3

# Custom poll interval (seconds between squeue checks, default 120)
bash scripts/megamaser/submit.sh --sampler de -q cmbgpu --galaxy NGC5765b --max-retries 10 --poll 60
```

Without `--max-retries`, the scripts behave exactly as before.
