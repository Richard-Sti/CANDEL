Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.


Most important
--------------

1. After you complete a task, review and report the changes.

0. The megamaser paper draft is at /mnt/users/rstiskalek/Papers/MMH0/main.tex.
   When asked, add notes there in the appropriate section in the style of the existing text.

5. The `instructions/` folder contains how-to guides for running jobs:
   - `instructions/glamdring_gpu_jobs.md` — GPU queues, addqueue syntax, monitoring
   - `instructions/maser_disk_jobs.md` — maser disk model: runner, config, submission, grid sizes

2. When there is a series of independent tasks to be completed, always deploy a series of agents.

3. After completing each major step, update the "Current state" section at the bottom of this file with: what was done, current state, and next steps. Clear out stale entries that are no longer relevant.

4. Note that the instructions below apply only to Python. But the sense of it applies to other languages too.


Current state (megamaser disk model)
------------------------------------

### What was done
- Added optional eccentricity model to MaserDiskModel (2026-04-08):
  - `_ecc_vel_factor(sin_phi, cos_phi, ecc, sin_omega, cos_omega)` pure function.
  - v_z = v_kep * (sin_phi + e*sin(2phi-omega)) / sqrt(1+e*cos(phi-omega)) * sin_i.
  - Eccentric HV block: drops the V(phi)=V(pi-phi) symmetry shortcut; computes
    V1 and V2 independently → `ll = lnorm + logaddexp(-chi2_1/2, -chi2_2/2)`.
  - Disabled by default (`model/use_ecc = false`). Enable with ecc~U(0,0.5),
    omega_disk~U(0,360) deg priors in config.
  - ecc=0 verified to reproduce circular log-density exactly.
- Added NGC4258 data loading from Argon+2007 (2026-04-08):
  - `load_NGC4258_spots` in candel/pvdata/megamaser_data.py.
  - Best-flux per (RefVel, Chan) across 18 epochs; S/N≥3 cut; 4023 spots
    (1263 red, 117 blue, 2643 systemic). No accelerations (single-epoch).
  - v_sys_obs = 472.0 km/s added to config.
  - Note: gamma uses circular-orbit Lorentz factor (error < 0.001 km/s for e<0.5).

### DE MAP optimizer (2026-04-09/10)
- `de_optimize()` in `candel/inference/optimise.py`: derivative-free MAP
  finder using Differential Evolution (evosax). Replaced PSO after testing.
- Reflection boundary handling (`_reflect_bounds`): triangle-wave folding
  instead of clipping. Clipping causes DE difference vectors to collapse
  at boundaries (UGC3789 was stuck at D_c=70 prior bound with clipping).
- Default: pop=1000, max_gen=1000, patience=100, Sobol=2^16, eval_chunk=64.
- Automatically selected by `find_MAP()` when `model.marginalise_r=True`.
  Sobol+Adam remains default for non-marginalized models.
- Hubble-flow distance seeding: Sobol D_c range narrowed to
  D_est ± 5×(σ_v/H0), configured in `[optimise]` section of config:
  `H0_for_D_estimate=70`, `D_sobol_sigma_v=500`, `D_sobol_n_sigma=5`.
- DE beats PSO on all galaxies (1-30 nats better logP, more reliable).
- `sobol_optimize` (formerly `sobol_adam`) now works in unconstrained space.
- Removed: PSO optimizer, `sobol_adam` alias, `mds.py`.

### Current state
- NGC5765b: D_A = 121.7 ± 8.2 Mpc (last run before eccentricity branch).
- NGC4258: data loaded (4023 spots), model constructs and NUTS test run passes.
  Ready for a proper inference run to reproduce Reid+2019 (D = 7.58 Mpc).
- Eccentricity model: implemented, tested, disabled by default.
- DE optimizer integrated as default init for maser r+phi mode.
- Tempered optimizer removed (was unused).
- D_c prior: `model/D_c_prior` = "uniform" (default) or "volume" (p(D_c) ∝ D_c²).
  r_ang prior is always uniform.
- Clump-averaged acceleration weighting: spots sharing a clump-averaged
  acceleration measurement (identical a, σ_a) have their acceleration
  likelihood raised to 1/N, so the clump contributes one independent
  measurement. Always on; weights computed from data in `load_megamaser_spots`.
  Affected galaxies: NGC5765b (29 spots), NGC6264 (13), NGC6323 (4).

### DE MAP results (2026-04-10, reflection + σ_v=500 + H0=70)
  CGCG074-064:  logP=-714,  D_c=85.3   (pub 87.6±7.6)
  NGC5765b:     logP=-408,  D_c=123.9  (pub 112.2±5.4)
  NGC6264:      logP=-123,  D_c=144.0  (pub 132.1±11.9)
  NGC6323:      logP=-98,   D_c=130.2  (pub 109.4±30)
  UGC3789:      logP=-293,  D_c=52.6   (pub 51.5±4.5)

### MAP init dump/load (2026-04-10)
- `dump_map_to_toml(params, fpath, galaxy)` in `candel/inference/optimise.py`:
  saves MAP dict to TOML under `[init.<galaxy>]`. Merges into existing file.
  Values rounded to 6 significant figures for clean TOML.
- `load_map_from_toml(fpath, galaxy, model=, model_kwargs=)` reads back as
  jnp arrays. Reports missing parameters (will be sampled from prior).
- CLI: `--save-map PATH` and `--load-map PATH` in `run_maser_disk.py`.
  `--load-map` skips optimization entirely; `--save-map` dumps after find_MAP.
- `run_H0_inference` supports `init_from_file` and `init_galaxy` in config.
- Workflow: `python run_maser_disk.py NGC5765b --sampler nuts --save-map results/Maser/map_init.toml`
  then `python run_maser_disk.py NGC5765b --sampler nuts --load-map results/Maser/map_init.toml`

### Quadratic disk warp (2026-04-10)
- Added optional quadratic warp terms: i(r) = i0 + di_dr*dr + d2i_dr2*dr^2,
  Omega(r) = Omega0 + dOmega_dr*dr + d2Omega_dr2*dr^2.
- Disabled by default (`model/use_quadratic_warp = false`).
- Priors: U(-30, 30) deg/mas^2 for both d2i_dr2 and d2Omega_dr2.
- Literature: Gao+2016 (NGC5765b) fitted d2i/dr2 but found it insignificant.
  Humphreys+2013 (NGC6264), Reid+2013 (UGC3789), Pesce+2020 (CGCG074-064)
  all use linear-only warps. Quadratic terms generally not needed.
- Tests (NGC5765b): d2i_dr2=0 reproduces linear logP exactly (residual=0).
  d2i_dr2=1 deg/mas^2 improves logP by ~124 nats at the reference point,
  suggesting NGC5765b may benefit from inclination curvature.
- Mock generator also updated to support quadratic warp via true_params dict.

### NGC4258 data loading cleanup (2026-04-12)
- `load_NGC4258_spots`: removed hardcoded error floors (0.02/0.03/0.3 mas),
  now returns raw e_dX, e_dY, e_Acc. Also reads e_Vlsr from the data file
  and returns it as sigma_v (was previously skipped → model defaulted to 1 km/s).
- `sigma_v_default` config option (default 0.25 km/s): used when the data
  file has no velocity errors (all galaxies except NGC4258).
- Eccentricity: renamed omega_disk → periapsis, added dperiapsis_dr for
  radial warp of the periapsis angle: omega(r) = periapsis0 + dperiapsis_dr*(r-r_ref).
- Per-galaxy model feature flags: `use_ecc` and `use_quadratic_warp` can now
  be set per-galaxy in config (NGC4258 uses both, others don't).
- Per-galaxy `r_ang_ref` override in config (NGC4258: 6.1 mas from Reid+2019).
- CLI: `--map-only` now works regardless of sampler setting (was NUTS-only).
- CLI: `--log2-N` overrides Sobol sequence size, `--init-method` overrides init.
- `toy_joint_H0.py`: fixed NSS file paths (Dvol→Dflat) and removed incorrect
  D^2 volume prior division (NSS was run with flat prior, not volume).

### Grid convergence crisis for NGC4258 (2026-04-12)
**CRITICAL:** The trapezoidal integration grids are completely inadequate for
NGC4258. Convergence test results (delta_logP vs finest grid tested):

  Phi grid (r at default 251):
    NGC4258: G=202 → −117 nats, converges ~G=600 (−0.5 nats)
    NGC5765b: G=202 → −0.001 nats (fully converged)

  R grid (phi at default):
    NGC4258: n_r=251 → −16,826 nats, n_r=601 still not converged
    NGC5765b: n_r=251 → +0.14 nats (fully converged)

  Joint scaling:
    NGC4258: 4x scale (n_r=1004) still not converged, non-monotonic behavior
    NGC5765b: 1x scale fully converged

**Root cause:** NGC4258 position errors (~0.003 mas) are ~5-10x smaller than
angular radii (~0.5-8 mas). Per-spot likelihood in r is a spike with fractional
width ~0.1-3%. A 251-point global grid cannot resolve 358 independent narrow
spikes. The other galaxies have 5-10x larger fractional errors → broader peaks
→ no grid issues.

**Possible fixes (not yet implemented):**
- Per-spot adaptive r grid: center a small local grid on each spot's projected
  radius. Each spot needs only ~20-30 points to resolve its own spike.
- Laplace approximation in r: if per-spot integrand is nearly Gaussian,
  find r_max analytically and integrate the Gaussian.
- Brute force: n_r=5000+ (expensive, O(n_spots × n_r × n_phi)).

### Next steps
- **Fix NGC4258 r-grid convergence** (blocking for NGC4258 inference).
- Run NGC4258 with NUTS/NSS to check if D_A ~ 7.58 Mpc is recovered.
- Run full NUTS inference for all galaxies using DE-initialized MAP.
- Consider reparameterisation (log(M/D), log(M/D²)) to reduce leapfrog steps.
- Test quadratic warp with NUTS on NGC5765b to see if d2i_dr2 is constrained.


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

### NSS evidence bias fix (2026-04-07)
**Root cause:** NumPyro's `Uniform.log_prob` returns `log(1/(b-a))` even
outside `[a, b]` when `validate_args=False` (the default). The blackjax NSS
slice sampler only checks `loglikelihood > threshold` for acceptance, NOT
`logprior > -inf`. This allows the sampler to explore outside the prior
support, corrupting the prior volume compression. The bias scales as ~O(d)
nats and is negative (evidence underestimated):
- d=2: 0.04, d=5: -0.26, d=10: -0.79, d=15: -1.4, d=20: -3.5 nats.
The bias is independent of n_live, num_mcmc_steps, and num_delete.

**Fix:** Added explicit support checks (`_in_support`) in `decompose_model()`
(`candel/inference/nested.py`). Both `log_prior_fn` and `log_likelihood_fn`
now return `-inf` when parameters fall outside the prior bounds. After fix:
- d=2: +0.01, d=5: +0.02, d=10: +0.02, d=15: -0.004, d=20: +0.20 nats.
All within ~2sigma of truth.

### Next steps
1. Run batch mock closure tests (25 mocks) for both modes.
2. Run other galaxies (NGC6264, NGC6323, UGC3789).
