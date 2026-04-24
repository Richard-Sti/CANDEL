# Maser Mode 2 union-grid + AD gradient convergence — design

Date: 2026-04-25
Branch: `maswer-wip`

## Goal

Replace the current two-branch Mode 2 r-grid (per-spot sinh for
HV/sys-accel, shared brute grid for sys-no-accel) with a single
per-spot **union of local + global** r-nodes, automatically centred on
the posterior peak by a fixed-iteration optimiser. Extend the
convergence tests so they verify not only the log-posterior but also the
JAX autodiff gradient that NUTS consumes.

## Motivation

Status quo:
- HV and sys-with-accel spots get a closed-form physics seed
  (`r_vel`/`r_acc`), damped-Newton refinement, and a narrow sinh grid of
  `n_r_local` points. Captures the peak precisely.
- Sys-without-accel spots get a shared log-uniform grid of `n_r_brute`
  points over the full `[r_min, r_max]`. No centring. Covers the range
  but with poor resolution near the peak.

Problems:
1. Sys-no-accel spots can still have a resolvable r-peak from position +
   velocity; the shared grid wastes resolution on regions with no mass.
2. Two codepaths (per-spot sinh vs shared log-uniform) in the integrator
   add complexity and require separate `_phi_eval_shared_r` logic.
3. Grid convergence is currently only verified on the log-posterior.
   NUTS samples using `jax.grad`; an integrator that returns correct
   log-L but wrong gradients is indistinguishable to the current tests
   but breaks sampling.

## Design

### 1. Unified Mode 2 r-grid (per-spot union)

Every spot gets the same recipe:

1. **Seed** (closed-form where possible):
   - HV: `r_vel = M · (C_v · sin_i)² / (D · Δv²)`
   - Sys-with-accel: `r_acc = √(C_a · M · sin_i / (D² · |a|))`
   - Sys-without-accel: evaluate the φ-marginalised log-likelihood on a
     coarse shared log-uniform **scan grid** (32–64 points over
     `[r_min, r_max]`); pick each spot's argmax in log r.
2. **Refine**: existing damped-Newton on `log r`
   (`damped_newton_1d`, `n_refine_steps = 10`, `step_max = 0.5`,
   `hess_floor = 1e-3`). Produces `r_c` and `s_hess = 1 / √h_opt`.
3. **Width**:
   - HV/sys-accel: `s = max(s_hess, s_propagated)`. Per-spot
     propagated-noise width from measurement errors (existing).
   - Sys-no-accel: `s = s_hess`. No closed-form propagated-noise
     fallback since there's no closed-form seed.
4. **Local grid**: sinh-spaced `n_r_local` nodes over
   `[r_c − K_σ·s, r_c + K_σ·s]`, half-width capped per-spot so the
   local grid fits inside `[r_min, r_max]`.
5. **Global grid**: shared log-uniform `n_r_global` nodes over
   `[r_min, r_max]`, cached once per forward pass, broadcast across all
   spots.
6. **Union**: per-spot `jnp.concatenate([local_i, global_shared])`,
   sorted by `log r`; trapezoidal log-weights computed on the sorted
   union.
7. **Integrate**: `logsumexp(nhc(r_union, phi) + log_w_r + log_w_phi,
   axis=(-2, -1))` — one call per spot group, same as the current
   per-spot path.

All node positions and weights wrapped in `stop_gradient`; the Newton
loop, scan argmax, and Hessian-derived width do **not** flow gradients
into HMC.

### 2. Config changes

`scripts/megamaser/config_maser.toml`:

```toml
[model]
n_r_local = 151        # was 301
n_r_global = 101       # new; replaces n_r_brute = 501
n_r_scan = 48          # new; coarse scan for sys-no-accel seeding
# K_sigma, refine_r_center, n_refine_steps, refine_step_max,
# refine_hess_floor, mode2_spot_batch unchanged.

[mode2_mpi.galaxies.<each>]
n_r_local = 501        # unchanged
n_r_global = 201       # was n_r_brute = 2001 (but cost is now per-spot)
```

New gradient-reference blocks (Sections 3 and 4):

```toml
[convergence.mode2_gradient_reference]
n_r_ref = 8001
r_batch = 32
dtype = "float64"

[convergence.mode1_gradient_reference]
# shares [convergence.mode1_reference] n_phi/spot_batch at float64,
# with spot-axis remat to keep the reverse-mode tape bounded
```

### 3. `convergence_grids.py` / `.sh` (Mode 2) — add AD gradient check

For every row in the existing `build_test_settings` sweep (blocks
"joint", "r", "phi"), compute:

- `ll_test`, `ll_ref`, `Δll`  *(existing)*
- `grad_test = jax.grad(sum ll_test)(θ)` at config `init`, one scalar
  per parameter in the extended `_GRAD_PARAMS_BASE`
- `grad_ref = jax.grad(sum ll_ref)(θ)` against
  `[convergence.mode2_gradient_reference]` (high-res log-uniform
  `n_r_ref` × production φ grid)
- `max_abs_dgrad = max_k |grad_test[k] − grad_ref[k]|`
- `max_rel_dgrad = max_k |Δ_k| / max(|grad_test[k]|, |grad_ref[k]|,
  1e-30)`

Output table gains `max|Δ∇|`, `max rel∇` columns. Summary picker:
smallest grid with `|Δll| ≤ 0.5 nats` AND `max_rel_dgrad ≤ 1e-3`.

Memory: reference gradient via reverse-mode uses `jax.checkpoint` per
r-chunk of size `r_batch = 32`.

Parameters differentiated: `_GRAD_PARAMS_BASE`, extended per-galaxy
with `d2i_dr2/d2Omega_dr2` (quadratic warp) and ecc params when
present.

**Not changed**: `[convergence.mode2_reference]` — the log-L brute
reference stays at the existing 50000×50001 float32 grid.

### 4. `convergence_phi_marginal.py` / `.sh` (Mode 1) — add AD gradient check

For every `(grid, scale)` combination, compute:

- Summed-gradient over globals (same `_GRAD_PARAMS_BASE` as Mode 2),
  production grid vs full-2π reference
- Per-spot gradient w.r.t. `r_ang` — a length-`N_spots` vector
  (reverse-mode; cross-spot coupling is zero so one pass gives the full
  vector)

Report `max|Δ|` and `max rel` for both categories. Summary picker:
smallest grid with `|Δtotal| ≤ 0.5 nats` AND
`max_rel_dgrad_globals ≤ 1e-3` AND `max_rel_dgrad_r ≤ 1e-3` (worst
across `R_SCALES`).

Memory: reverse-mode pass on the 200001-point reference wraps the
per-spot-batch φ evaluation in `jax.checkpoint`; peak ≈ one chunk's
forward activations.

### 5. `r_ang_posteriors.py` / `.sh` — strip gradient, visualise new centring

Remove: `_GRAD_PARAMS_BASE`, `_sample_jnp`, `_jax_phys_from_sample`,
`_log_marginal_mode2`, `_log_marginal_ref`, `gradient_check`,
`write_grad_table`; CLI flags `--grad-out`, `--n-r-ref`,
`--grad-r-batch`, `--mode2-spot-batch`, `--no-grad-check`; second
galaxy-loop pass.

Keep: per-spot 1D posterior computation on a dense r-grid, r-axis
chunking, plot layout.

Change overlays: drop the dashed `r_est` and dotted `r_opt`.
Replace with:

1. **Dotted** vertical line at `r_c` (centre from the new recipe:
   seed → scan → Newton).
2. **Shaded band** `[r_c − K_σ·s, r_c + K_σ·s]` in the per-spot colour
   (alpha 0.15).
3. **Rug ticks** at each `r_global` node on the x-axis (shared per
   panel, drawn once).

The model class exposes a new helper `get_mode2_centres(sample,
type_key)` returning `(r_c, s, r_min, r_max)` so this script doesn't
re-implement the seed/scan/Newton chain.

### 6. `test_mode2_stability.py` / `.sh` — expand to include AD gradient checks

Tight, self-contained, CPU-friendly. Checks:

1. `damped_newton_1d` sanity (existing).
2. **Union-grid invariants** (new): sorted, no dups within 1e-12, total
   weight = `log(r_max − r_min)` within 1e-6 rel, all nodes ∈
   `[r_min, r_max]`.
3. `refine_r_center=True` requires `phys_args` (existing guard).
4. Finite `log-L` and finite `jax.grad` for `refine ∈ {False, True}`
   (existing, now with explicit per-parameter assertion).
5. **Mode A**: summed-gradient production vs reduced reference
   (`n_r_ref ≈ 2001`, reduced φ). Assert `max_rel ≤ --rtol` (default
   `1e-3`).
6. **Mode B**: per-spot gradient via `jax.jacfwd` over globals. Assert
   `max_rel_per_spot ≤ --rtol-per-spot` (default `1e-2`). Prints worst
   `(galaxy, spot, param)` offender.
7. FD diagnostic (opt-in `--fd`, existing).
8. NUTS smoke (opt-in `--nuts-smoke`, existing).

Default grids (small): `n_phi_hv_high = 41`, `n_phi_hv_low = 9`,
`n_phi_sys = 41`, `n_r_local = 21`, `n_r_global = 15`, `n_r_scan = 17`,
`spot_batch = 4`. Default galaxy `UGC3789`. Target runtime < 2 min on
CPU.

Reference grids: `n_phi_hv_high_ref = 401`, `n_phi_hv_low_ref = 51`,
`n_phi_sys_ref = 401`, `n_r_ref = 2001`, `r_chunk_ref = 128`.

Memory: per-rank peak ≲ 100 MB on all checks (well under the 4 GB
`MEM` default lowered from 8 GB). Reverse-mode reference via
`jax.checkpoint` per r-chunk.

## Memory strategy summary

All three reference evaluations use two levers:

- **Spot-axis batching**: existing `spot_batch` knob, used in every
  reference loop (`convergence_grids`, `convergence_phi_marginal`,
  `test_mode2_stability`).
- **r-axis chunking with `jax.checkpoint`** for reverse-mode gradients:
  peak activation ≈ one r-chunk's forward pass, not the full tape.

Forward-mode (`jax.jacfwd`) passes need no rematerialisation — one pass
per parameter, tape-free.

## Files touched

Core model:
- `candel/model/model_H0_maser.py` — union grid integrator, new
  `get_mode2_centres` helper, simplified `_eval_phi_marginal` (no
  shared-r branch).
- `candel/model/maser_convergence.py` — gradient-reference helpers,
  shared between convergence scripts.

Convergence scripts:
- `scripts/megamaser/convergence/convergence_grids.py` — add summed-
  grad columns to sweep + picker.
- `scripts/megamaser/convergence/convergence_grids.sh` — help text.
- `scripts/megamaser/convergence/convergence_phi_marginal.py` — add
  globals + r_ang grad columns.
- `scripts/megamaser/convergence/convergence_phi_marginal.sh` — help
  text.
- `scripts/megamaser/convergence/r_ang_posteriors.py` — strip gradient
  logic, swap overlay for new-centring visualisation.
- `scripts/megamaser/convergence/test_mode2_stability.py` — add Mode A
  + Mode B gradient checks, union-grid invariants, reduced default
  grids.
- `scripts/megamaser/convergence/test_mode2_stability.sh` — help text,
  lower default `MEM` to 4 GB.

Config:
- `scripts/megamaser/config_maser.toml` — add
  `n_r_global`/`n_r_scan`, replace `n_r_brute`, add
  `[convergence.mode2_gradient_reference]`,
  `[convergence.mode1_gradient_reference]` if needed.

## Out of scope

- Mode 1 r-sampling is untouched.
- `run_mode2_mpi.py` stays on the new grid names via the existing
  override path (`n_r_brute` → `n_r_global` rename).
- Selection function, joint model, prior blocks — no changes.
- NUTS sampler config — no changes.

## Acceptance criteria

1. `test_mode2_stability.py` passes on CPU in < 2 min at default args
   for `UGC3789`.
2. `convergence_grids.py` at the config default produces `|Δll| ≤ 0.5
   nats` and `max_rel_dgrad ≤ 1e-3` for all five MCP galaxies.
3. `convergence_phi_marginal.py` shows no regression vs the current
   run (same `|Δtotal|`), plus new gradient columns within tolerance.
4. `r_ang_posteriors.py` runs end-to-end and produces the centring
   overlay without error.
5. MPI runner unchanged in behaviour beyond the grid-knob rename.
