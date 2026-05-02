# CANDEL

**CANDEL** (*CA*libration and *N*ormalization of the *D*istanc*E* *L*adder) is a JAX-based framework for peculiar-velocity inference, distance-ladder calibration, and megamaser disk modelling.

**Documentation:** [candel.readthedocs.io](https://candel.readthedocs.io/en/latest/)

## Overview

CANDEL forward-models distance-indicator observables (e.g. magnitude, line width, velocity dispersion) and redshift while marginalising over latent variables such as distance and absolute magnitude. Distance is either marginalised numerically via Simpson integration or sampled explicitly; latent observables are marginalised analytically where Gaussian conjugacy allows (SN stretch and colour via the Tripp relation; Fundamental Plane velocity dispersion and surface brightness), and via Gauss--Hermite quadrature otherwise. Posterior sampling uses the No-U-Turn Sampler (NUTS) from [NumPyro](https://github.com/pyro-ppl/numpyro), with JAX providing automatic differentiation and JIT compilation throughout.

When a reconstructed density and velocity field is supplied, CANDEL jointly calibrates each distance indicator and the underlying velocity field (e.g. amplitude $\beta$ and external bulk flow $\mathbf{V}_\mathrm{ext}$). The external dipole can also be inferred without any reconstructed field. For peculiar-velocity inferences, the distance prior is modelled following the phenomenological approach of [Lavaux (2016)](https://arxiv.org/abs/1512.04534), which effectively accounts for selection effects; for $H_0$ inferences, a rigorous selection function treatment is used instead (see [Stiskalek et al. 2025](https://arxiv.org/abs/2509.09665)). Model comparison is supported via BIC/AIC, Laplace evidence, and the [harmonic](https://github.com/astro-informatics/harmonic) package.

CANDEL runs locally for small samples or scales to computing clusters with GPU support (one GPU per chain). It includes cluster submission helpers and batch job generation tools for launching large parameter-grid runs from a frozen copy of the code.

### Highlights
- Forward modelling of the full distance ladder with JAX and NumPyro.
- Joint calibration of distance-indicator relations and the underlying density/velocity field.
- Analytical marginalisation of latent observables where Gaussian conjugacy allows, reducing sampler dimensionality.
- Multiple galaxy-bias models: linear ($1 + b_1 \delta$), quadratic ($1 + b_1 \delta + b_2 \delta^2$), power-law ($\rho^\alpha$), and double power-law.
- Density-dependent velocity dispersion $\sigma_v(\delta)$ via a sigmoid in log-density, allowing different dispersions in underdense and overdense regions.
- Redshift-to-real-space mapping of observed redshifts given a calibrated velocity field.
- Peculiar-velocity covariance matrices from CAMB power spectra.
- HPC-friendly tooling: batch config generation, queue submission scripts, GPU auto-detection, and precomputed line-of-sight field generation.

## Supported distance indicators and catalogues

### Peculiar-velocity models

These models work in units of $h^{-1}\,\mathrm{Mpc}$ (i.e. assume $h = 1$). Multiple catalogues can be analysed jointly via `JointPVModel`, with user-specified shared parameters across sub-models.

- **Tully--Fisher relation:** 2MTF, SFI++, CF4-TFR
- **Type Ia supernovae (SALT2):** LOSS, Foundation, Pantheon+
- **Fundamental Plane:** 6dFGS-FP, SDSS-FP

### $H_0$ inference

- **Cepheid-calibrated $H_0$:** 35 Cepheid host galaxies from SH0ES
- **TRGB-calibrated $H_0$:** Tip of the Red Giant Branch distances from CCHP and EDD, including grouped EDD hosts
- **Joint TRGB + CSP $H_0$:** CCHP TRGB calibrators combined with CSP SNe Ia *(development)*
- **2MTF-calibrated $H_0$:** Tully--Fisher distances from the EDD-2MTF sample *(experimental)*
- **Megamaser disk $H_0$:** spot-level warped disk fits for NGC 5765b, NGC 6264, NGC 6323, UGC 3789, CGCG 074-064, and NGC 4258. `JointMaserModel` fits multiple disks with a shared $H_0$; `toy_joint_H0.py` can combine saved per-galaxy distance posteriors.

## Package structure

```
candel/
  model/          Forward models for each distance indicator
  pvdata/         Data loaders for all supported catalogues
  cosmo/          Cosmography, growth rate, PV covariance matrices
  inference/      NUTS sampling, nested sampling (NSS), Sobol+Adam optimisation, evidence estimation
  field/          3D density/velocity field loading and LOS interpolation
  redshift2real/  Map observed redshift → cosmological redshift given a velocity field
  mock/           Synthetic catalogue generation for testing
  util.py         Coordinate transforms, config I/O, plotting utilities

scripts/
  runs/           PV and H0 model configs and main runner
  megamaser/      Maser disk model config and runner
  mocks/          Mock TRGB inference runs
  preprocess/     Precompute line-of-sight density/velocity data
  sync/           Cluster sync helpers
```

## Running inference

All experiments are defined in TOML configuration files that specify data paths, model parameters, priors, and output locations.

**Peculiar-velocity / $H_0$ models:**
```bash
python scripts/runs/main.py --config path/to/config.toml
```

To generate a batch of PV/$H_0$ configs from a template with a parameter grid:
```bash
python scripts/runs/generate_tasks.py
```

**Megamaser disk model:**
```bash
python scripts/megamaser/run_maser_disk.py NGC5765b --sampler nss
python scripts/megamaser/run_maser_disk.py joint --sampler nuts
```

The generic `scripts/runs/main.py` runner also supports `model.which_run = "maser_disk"` for config-driven maser jobs.

### Inference methods

- **NUTS** (default): No-U-Turn Sampler via NumPyro. Robust, gradient-based, suitable for all models.
- **Nested Slice Sampling (NSS):** Bayesian evidence computation via a self-contained reimplementation of the NSS algorithm ([Yallup et al. 2026](https://arxiv.org/abs/2601.23252)) in `candel/inference/nested.py`. No external nested-sampling dependency required.
- **Sobol + Adam MAP:** Multi-start MAP optimisation using Sobol quasi-random initialisation and Adam gradient descent. Configured via the `[optimise]` section of the TOML config.

## Publications

Here are some recent works that have used CANDEL:

1. *The Velocity Field Olympics: Assessing velocity field reconstructions with direct distance tracers*; Stiskalek et al. (2025)
  [arXiv:2502.00121](https://arxiv.org/abs/2502.00121)

2. *1.8 per cent measurement of H₀ from Cepheids alone*; Stiskalek et al. (2025)
  [arXiv:2509.09665](https://arxiv.org/abs/2509.09665)

3. *No evidence for H₀ anisotropy from Tully--Fisher or supernova distances*; Stiskalek et al. (2025)
  [arXiv:2509.14997](https://arxiv.org/abs/2509.14997)

4. *S₈ from Tully--Fisher, fundamental plane, and supernova distances agree with Planck*; Stiskalek (2025)
  [arXiv:2509.20235](https://arxiv.org/abs/2509.20235)

## Installation
```
git clone git@github.com:Richard-Sti/CANDEL.git
cd CANDEL

python -m venv venv_candel
source venv_candel/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e .
```

Nested sampling (NSS) is self-contained and ships with CANDEL; no extra nested-sampling dependency is required.

For learned harmonic-mean evidence estimates, also install [harmonic](https://github.com/astro-informatics/harmonic).

## Local configuration (`local_config.toml`)

Per-machine settings live in a `local_config.toml` file at the repository
root. This file is **not** versioned and must be created on each machine where
CANDEL is installed. It supplies machine-specific paths and Python interpreters
used by the run scripts.

A minimal `local_config.toml` looks like:

```toml
root_main    = "/path/to/CANDEL/"   # repo root (required)
root_data    = "/path/to/data/"     # optional, defaults to <root_main>/data
root_results = "/path/to/results/"  # optional, defaults to <root_main>/results

python_exec = "/path/to/venv_candel/bin/python"  # used by cluster helpers
```

Relative input paths in run-time TOML configs are resolved against `root_data`;
output paths such as `fname_output` are resolved against `root_results`.
Absolute paths are left unchanged. Cluster submission helpers may use
additional machine/module keys; see [`docs/configuration.rst`](docs/configuration.rst)
for the full configuration schema.

## Citation

If you use CANDEL, or find it useful, please cite the papers listed above.

## License

GNU General Public License v3.0 -- see [LICENSE](LICENSE) for details.
