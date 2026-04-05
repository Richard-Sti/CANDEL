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
