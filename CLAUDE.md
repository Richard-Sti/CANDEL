Most important
--------------

1. After you complete a task, review and report the changes.

2. When there is a series of independent tasks to be completed, always deploy a series of agents.

3. After completing each major step, update the "Current state" section at the bottom of this file with: what was done, current state, and next steps. Clear out stale entries that are no longer relevant.

4. Note that the instructions below apply only to Python. But the sense of it applies to other languages too.


Current state (megamaser disk model)
------------------------------------

### What was done
- Fixed NGC5765b distance: root cause was missing inclination warp (di/dr).
  Added init_values support to avoid NUTS getting trapped in wrong mode.
- Fixed data filename (NGC5765b_Gao2016_table6_tex.dat → .dat).
- Fixed marginalise_r grid weights (now consistent with r_ang grid).
- Speed optimisations for Mode 2 (marginalise_r):
  - Precompute r-dependent quantities (v_kep, gamma, z_g, a_mag, position
    coefficients) outside the phi loop — eliminates 3 sqrts per phi point.
  - Skip acceleration computation entirely for 123/192 unmeasured spots
    (separate 3-obs chi² path, no A array allocated).
  - float32 for GPU runs.
- GPU benchmark (RTX 3070): gradient 4.3 ms (f32) vs 31.9 ms (f64).

### Current state
- NGC5765b with all optimisations: D_A = 121.7 ± 8.2 Mpc
  (Gao+2016: 126.3 ± 11.6, Pesce+2020: 112.2 +5.4/-5.1).
- GPU job running on optgpu (A6000) with marginalise_r, testing whether
  prior-median init converges to correct mode when r is marginalised.
- CPU benchmark submitted to berg node.

### Next steps
- Check GPU run results (good init vs bad init comparison).
- Run other galaxies (NGC6264, NGC6323, UGC3789) with di/dr where needed.
- Consider reparameterisation (log(M/D), log(M/D²)) to reduce leapfrog steps.


Basics
------

1. Write clean, idiomatic Python that follows PEP8 style guidelines unless explicitly told otherwise.

2. Prioritize efficiency and vectorization using NumPy, SciPy, or JAX where appropriate.

3. Minimize memory overhead by avoiding unnecessary copies and using in-place operations when safe.

4. Use informative variable names but keep them short when they're in math-heavy or array-heavy code (e.g. bf, r, z).

5. Add comments only where clarification is needed. Don't comment obvious things. Assume the reader is a domain expert.

6. Avoid boilerplate unless explicitly asked.

7. For plotting:
   - Don’t add titles or colorbars unless explicitly asked.
   - Label axes with units in LaTeX if known.

8. For shell scripting:
    - Use #!/bin/bash -l shebang for SLURM jobs.
    - Prefer awk and grep for parsing files.
    - Always print useful status info unless in silent mode.

9. When given raw array data:
    - Assume it's in physical units unless told otherwise.
    - If in cosmological context, assume comoving Mpc/h and Msun/h units.

10. When unsure of a parameter or behavior, ask for clarification instead of guessing.

11. In the function signature, do not describe the variable types. Python does not use that anyway.

12. Never use conda. Always use the local venv of the package.


Current state (megamaser disk model)
------------------------------------

### Bug fix: r_ang/sin_i argument swap (2026-04-06)
In `_r_precomp()` inside `_eval_marginal_phi`, the return order was
`(r_ang, sin_i, ...)` but `_observables_from_precomputed` expects
`(sin_i, r_ang, ...)`. Fix: swap order in `_r_precomp` and update
`_obs_3` unpacking. Code-reviewed and confirmed correct.
The previously documented "Mode 2 volume effect" was caused by this bug.

### Code changes (2026-04-06)
- `_r_precomp` return order: `(sin_i, r_ang, ...)` (was swapped).
- `_obs_3` unpacking: `(si, r_sub, ...)` to match.
- Jacobian N*log(D_A*PC) removed from both modes. Uniform r_ang prior.
- Phi grid sizes configurable: `model/G_phi_half`, `model/G_phi_sys`.
- R grid size configurable: `model/n_r`.

### Mock closure tests (2026-04-06)
- **Mode 1** (50 spots, seed=42): all params within 1σ, 0 divergences.
  D_A = 92.8 ± 23.7 (truth 88.1).
- **Mode 2** (50 spots, seed=42): all params within 2σ, 2 divergences.
  D_c = 78.6 ± 11.0 (truth 90.0).

### Real galaxy results — Mode 2 MCMC (2026-04-06)
- **NGC5765b**: D_A = 121.8 ± 7.5 Mpc, logM = 7.66 ± 0.03,
  i0 = 73.7 ± 0.8, di_dr = 11.6 ± 0.8. 0 divergences.
  (Gao+2016: 126.3 ± 11.6, Pesce+2020: 112.2 +5.4/-5.1)
- **CGCG 074-064**: D_A = 86.8 ± 7.6 Mpc, logM = 7.38 ± 0.04,
  i0 = 88.6 ± 3.9. 0 divergences.
  (Pesce+2020: 87.6 +5.8/-4.7)

### Sobol + Adam optimisation (2026-04-06)
131K Sobol (14D) + Adam (20K steps) on NGC5765b Mode 2:
all 5 Sobol starts + NUTS mean + published init converge to same MAP:
  D_A ≈ 117, logM ≈ 7.64, i0 ≈ 74.0, Omega0 ≈ 150.3, logP ≈ -401.5
Posterior is **unimodal**. No secondary modes found.

Sobol+Adam optimizer now in `candel/inference/optimise.py`.
Use `init_method = "sobol_adam"` in `[inference]` config to activate.
Settings under `[optimise]` section (all optional with defaults).

### Data-driven initial guesses (2026-04-07)
`estimate_disk_params()` in `model_H0_maser.py`:
- **Omega0**: maximise |corr(impact parameter, velocity)| for HV spots.
- **log_MBH**: per-spot Keplerian estimate M_i = dv_i² × r_i × D_A / C_v²,
  take median. Biased low by sin²(i) (~0.04 dex at i=74°).
  NGC5765b: estimate 7.61, MCMC posterior 7.66.
- **log_MBH prior**: automatically set to TruncatedNormal(estimate, 0.5, [6,9])
  in MaserDiskModel.__init__. Config value is overridden.
- **i0**: not estimated (circular with M_BH). Prior handles it.
- **D_A, D_c**: Hubble flow with second-order correction.
- **x0, y0**: median of systemic spot positions.

### Optimiser pitfalls (2026-04-06)

**potential_energy vs log_density:** numpyro's `potential_energy()`
expects UNCONSTRAINED params. Passing constrained params silently
applies a double-transform, giving wrong results. **Always use
`log_density()` for evaluating at constrained params.**

**L-BFGS-B** converges prematurely on the maser posterior. The
posterior has a long, shallow valley — L-BFGS sees a small gradient,
declares convergence, and stops hundreds of nats from the optimum.
From the same Sobol starting points, L-BFGS reaches logP≈-685 while
Adam (20K steps) reaches logP≈-401. L-BFGS from the NUTS mean also
stops immediately (0 iterations) at logP=-402, while Adam improves
it to -401.5 by shifting D_A from 122→117.
**Use Adam (or NUTS) instead of L-BFGS for this problem.**

### Notes
- Mock uses r_ang_ref = median(r_ang_true), model uses r_ang_ref = 0.
  Cosmetic reparameterization, does not affect physics.
- Mode 2 r_ang grid is fixed at D_A_est from v_sys_obs; works well
  since D_A_est ≈ D_A_true for the galaxies tested.
- Mode 2 marginalised likelihood has weak M_BH constraint at fixed
  geometry (~7 nats over 2 dex). The NUTS joint posterior constrains
  M_BH through D_c-logM correlation. This is expected, not a bug.

### Next steps
1. Run batch mock closure tests (25 mocks) for both modes.
2. Run other galaxies (NGC6264, NGC6323, UGC3789).
