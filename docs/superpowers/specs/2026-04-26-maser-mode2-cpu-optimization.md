# Mode 2 CPU optimisation — n_r-tiled φ marginal

Date: 2026-04-26
Status: PLAN

## Context

NGC6323 ultranest run (`python-694702.out`):

- 64 ranks, 1–2 spots/rank, mode 2 with `n_phi_hv_high = n_phi_sys = 10001`,
  `n_phi_hv_low = 2001`, `n_r_local = 501`, `n_r_global = 2001`.
- Per-rank likelihood wall = **174.8 ms** at float64.
- Total 2.49 M likelihood calls in 5.2 days.

Constraints from user:

- Do not reduce grid sizes.
- Do not switch off AD in Newton refinement.
- Float32 is already on by default in `run_maser_disk.py` for nss runs
  (only NGC4258 forces f64 via `use_float64 = true`).

## Investigation

### Refactor that didn't help (lift-r-only — item "C")

- Edited `predict_velocity_los` in `candel/model/model_H0_maser.py` to
  add an explicit circular fast path that lifts `lorentz_factor` out of
  the (n_r, n_phi) broadcast.
- Verified numerically:
  - circular: max relative diff `1.8e-14` (machine eps)
  - eccentric: bit-for-bit identical
- Benchmarked at the realistic NGC6323 shape:
  - f64: orig 52.1 ms vs new 52.4 ms — **1.00× speedup**
  - f32: orig 30.7 ms vs new 30.4 ms — **1.01× speedup**
- **Conclusion**: XLA's broadcast simplifier was already lifting these
  pieces. The refactor is correct but a no-op for kernel time.
  Decision: **kept** — makes the circular vs eccentric paths explicit
  in the source, even though XLA produces the same HLO either way.

### XLA HLO/buffer evidence

Dumped optimised HLO with `XLA_FLAGS=--xla_dump_to=…` for a
representative `_phi_eval`-shaped kernel at f32, shape
`(N=1, n_r=2502, n_phi=14003)`.

- Buffer report: total temp **672 MB at f32** (≈1.34 GB at f64).
- Five separate `f32[1, 2502, 14003]` tensors materialised to RAM:
  1. `broadcast_divide_fusion.3` — χ²_x
  2. `broadcast_divide_fusion.2` — χ²_y
  3. `broadcast_divide_fusion`   — χ²_a
  4. `broadcast_divide_fusion.1` — χ²_v (already with lifted Lorentz factor)
  5. `subtract_exponential_fusion` — `exp(nhc + log_w − max)`

- The 4 χ² components are **read twice from RAM**:
  - first by `broadcast_add_fusion` (sum + max pass)
  - then by `subtract_exponential_fusion` (exp pass)
- Logsumexp itself is already cache-tiled by XLA as a tree reduction
  (`reduce-window` size=32 stride=32, repeated). Reductions are not the
  bottleneck.
- All r-only sub-kernels (`keplerian_speed`,
  `gravitational_redshift_factor`, `centripetal_acceleration`,
  `lorentz_factor`) are `f32[1, 2502]` fusions — XLA has lifted them
  out of the n_phi broadcast.

### Current batching state

- `mode2_spot_batch` is **commented out** in `config_maser.toml`. No
  galaxy override sets it. `_marginal_per_spot_r` runs single-shot.
- No batching exists along the n_r axis anywhere in the code.

### Identified bottleneck

- Memory bandwidth: ≈ **1 GB of RAM traffic per likelihood call** at f32
  just from re-reading the χ² components.
- At ~12–18 GB/s single-thread bandwidth: ≈ 30–55 ms wasted per call.

## Plan: n_r-tiled scan over the φ marginal

### Goal

Eliminate the materialise-then-reread of the 4 χ² components by
streaming the predict → χ² → logsumexp pipeline through tiles of
~32–256 r-rows so each tile's χ² lives in L2/L3 cache and is consumed
once.

### Realistic speedup

- Saved RAM traffic ≈ 4 × 134 MB = 536 MB/call (f32).
- 30–45 ms saved per call out of current ~110 ms f32 / ~175 ms f64.
- **~30 % wall-clock reduction** target.

### Implementation sketch

1. Add an internal helper `_phi_marginal_tiled` in
   `candel/model/model_H0_maser.py` that takes:
   - `r_pre` (the precomputed r-shape pytree)
   - `r_ang` shape `(N, n_r)`, `log_w_r` shape `(N, n_r)` (Mode 2)
     or `r_ang` shape `(N,)`, `log_w_r=None` (Mode 1)
   - φ grid `(sin_phi, cos_phi, log_w_phi)` for the spot type
   - tile size `n_r_tile`
   - `lnorm` per-spot

2. Use `jax.lax.scan` over chunks of `n_r_tile` rows along the n_r
   axis. Carry = `(m_running, lse_running)` per spot — running
   max and running log-sum-exp scaled to that max.

3. Inside the scan body, for one tile:
   - Slice `r_ang[:, j:j+n_r_tile]` and the corresponding warped
     angles from `r_pre`.
   - Call a refactored `_phi_eval_tile` that returns the
     `(N, n_r_tile, n_phi)` `nhc` for this tile only.
   - Add `log_w_phi[None, None, :]` and `log_w_r_tile[:, :, None]`
     (Mode 2) for the joint (r, φ) reduction.
   - Compute tile max and partial sum-exp.
   - Update the running `(m, lse)` with the standard stable
     blockwise logsumexp recurrence:
     ```
     m_new   = max(m_old, m_tile)
     lse_new = exp(m_old - m_new) * lse_old + exp(m_tile - m_new) * sumexp_tile
     ```

4. After the scan returns, `ll = lnorm + m + log(lse)`.

5. Pad `n_r` to a multiple of `n_r_tile` so all scan iterations have
   identical shapes (single XLA compile). Strip padding contribution
   by setting padded `nhc` to `-inf` (or by using a mask in the
   accumulator).

6. Add a config knob `n_r_tile` under `[model]` and per-galaxy
   `[model.galaxies.<gal>]`. Default `None` (= no tiling, current
   single-shot path). Enable per galaxy after benchmarking.

### Subtleties to handle

- **Mode 1 (per-spot r is sampled, not marginalised)**: tiling is
  unnecessary because r_ang is shape `(N,)`, not `(N, n_r)`. The
  scan path should only activate for Mode 2.
- **Newton refinement (`_refine_r_center_group`)**: currently calls
  `f_one(ell, spot)` which evaluates a φ-marginal (logsumexp over
  n_phi only, no n_r axis). Newton runs at the `n_phi`-only shape;
  no tiling needed there.
- **Mode 2 sys-uncons scan seeding (`_scan_seeds_sys_uncons`)**:
  builds an n_r_scan grid and evaluates the φ-marginal at each.
  Same shape as Mode 1's per-spot path; tiling not relevant.
- **Validation**: must reproduce the un-tiled output to within float
  epsilon on every supported galaxy. Use the existing
  `convergence_phi_marginal.py` machinery as the regression harness.

### Acceptance criteria

1. `Δ log L < 10⁻⁵` (f32) or `Δ log L < 10⁻¹²` (f64) vs the un-tiled
   baseline, evaluated at the published Pesce+2020 best-fit vector
   for at least NGC5765b, NGC6323, CGCG and UGC3789.
2. Wall-clock reduction ≥ 20 % at the production grid sizes
   `(n_phi_*, n_r_*) = (10001, 10001, 2001, 501)` on a single rank.
3. No memory-temp regression — peak temp should drop, not grow.
4. Tiling can be turned off via `n_r_tile = null` and produce
   exactly the previous single-shot graph (same buffer footprint).

### Order of work

1. Pure-Python prototype of the blockwise logsumexp recurrence on
   synthetic data — confirm numerical equivalence with one-shot
   logsumexp.
2. Wire into `_marginal_per_spot_r` behind a config flag, Mode 2
   per-spot path only. Run unit test against un-tiled.
3. Benchmark `n_r_tile ∈ {32, 64, 128, 256}` at NGC6323 production
   grids, both f32 and f64. Pick the best.
4. Validate against published distances on all five galaxies (short
   ultranest runs or DE optimisation).
5. Set `n_r_tile` per galaxy in `config_maser.toml` based on
   benchmarks.

## Things explicitly NOT on the plan

- Reducing grid sizes (user said no).
- Switching Newton from AD to FD (user said no).
- Float32 retrofit for NGC4258 mode1 (different runner; user said
  current behaviour is correct).
- Lifting more r-only quantities into `_r_precompute` — the
  HLO confirms XLA's CSE already handles these.
- Reverting the `predict_velocity_los` circular fast path — kept for
  source-level clarity even though XLA already produced the same HLO.
