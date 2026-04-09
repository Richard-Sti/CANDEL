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
| `--phi-prior` | off | Enable truncated Gaussian phi prior |
| `--seed` | 42 | RNG seed |
| `--num-warmup` | 1000 | NUTS warmup steps |
| `--num-samples` | 1000 | NUTS sample count |
| `--n-live` | 5000 | NSS live points |
| `--num-mcmc-steps` | 0 (=ndim) | NSS slice steps |
| `--num-delete` | 250 | NSS contraction batch |
| `--termination` | -3 | NSS stopping criterion |

CLI args override config values.

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

# Custom: any galaxy, any sampler
ROOT=/mnt/users/$USER/CANDEL
PYTHON=$ROOT/venv_gpu_candel/bin/python
addqueue -q cmbgpu -s -m 16 --gpus 1 \
    $PYTHON -u $ROOT/scripts/megamaser/run_maser_disk.py NGC6264 \
    --sampler nuts --num-warmup 2000 --num-samples 2000
```

## Grid sizes (numerical marginalisation)

The per-spot likelihood is marginalised over orbital radius and azimuth
on non-uniform grids. Defaults (set in `model_H0_maser.py`):

| Grid | Points | Spacing | Config key |
|------|--------|---------|------------|
| HV half (phi) | 102 | arccos (dense near pi/2) | `model/G_phi_half = 101` |
| Systemic (phi) | 201 | two-zone: 101 arcsin inner +/- 30 deg, 50 linear wings/side | `model/n_inner_sys`, `inner_deg_sys`, `n_wing_sys` |
| Radius | 101 | sinh in log-space | `model/n_r = 101` |

Override in config:
```toml
[model]
G_phi_half = 101
n_inner_sys = 101
inner_deg_sys = 30.0
n_wing_sys = 50
n_r = 101
```

## Dense mass blocks (NUTS)

Correlated parameters are grouped for efficient mass matrix adaptation:

```toml
dense_mass_blocks = [
    ["D_c", "eta", "dv_sys"],
    ["i0", "di_dr", "Omega0", "dOmega_dr"],
    ["x0", "y0"],
]
```

When `phi_prior = true`, a fourth block is added for the phi prior params.

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
