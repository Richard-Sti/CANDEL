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

### What was done
- Fixed NGC5765b distance: root cause was missing inclination warp (di/dr).
  Added init_values support to avoid NUTS getting trapped in wrong mode.
- Fixed data filename and marginalise_r grid weights (r_ang weights on r_ang grid).
- Speed optimisations for Mode 2 (marginalise_r):
  - Precompute r-dependent quantities outside phi loop.
  - Skip acceleration computation for unmeasured spots (3-obs chi²).
  - Fused 2D logsumexp over (r, phi).
  - float32 for GPU runs.
  - Reverted log-cosh trick (introduced 4e-3/point numerical error).
- Benchmarks (gradient, RTX 3070): f32 3.84 ms, f64 24.8 ms.
- Added Jacobian D_A*PC for r_ang→R_phys change of variables.

### Mode 2 (marginalise_r) limitations — IMPORTANT
Mode 2 with a global r_ang grid is NOT reliable for distance estimation.
The marginalised-r likelihood has a volume effect: at small D, more r
values produce velocities in the observed range, so the integral grows.
This overwhelms the acceleration constraint and biases D downward.
Mock tests confirm: log_density at truth (D_A=88) is 6800 worse than
at D_A=49 even with all error floors fixed. The reference paper
(arXiv:2601.14374) avoids this by using per-spot adaptive integration
regions (±1.5° phi, narrow r range) rather than a global grid.
**Use Mode 1 (per-spot r sampled by NUTS) for all production runs.**

### Current state
- NGC5765b with Mode 1 + di/dr: D_A = 121.7 ± 8.2 Mpc (published 112-126 Mpc).
- CGCG 074-064 with Mode 1: D_A = 87-91 Mpc (published 87.6). Works without di/dr.
- Mode 2 validated for speed but NOT for accuracy. Do not use for distance.

### Next steps
- Run other galaxies (NGC6264, NGC6323, UGC3789) with Mode 1 + di/dr.
- If Mode 2 needed: implement per-spot adaptive r integration (as in 2601.14374).
- Consider per-spot r initialisation via L-BFGS for faster Mode 1 convergence.
