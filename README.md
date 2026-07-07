# CANDEL

**A GPU-accelerated, hierarchical Bayesian framework for the local distance ladder.**

[![Documentation](https://readthedocs.org/projects/candel/badge/?version=latest)](https://candel.readthedocs.io/en/latest/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Powered by JAX](https://img.shields.io/badge/powered%20by-JAX-orange.svg)](https://github.com/jax-ml/jax)

CANDEL forward-models the local distance ladder end to end and infers cosmology directly from the data. A single likelihood spans Cepheid and TRGB calibrations, Type Ia supernovae, the Tully--Fisher relation, and the Fundamental Plane, and jointly constrains the Hubble constant $H_0$, the peculiar-velocity amplitude $\beta$, and external bulk flows $\mathbf{V}_\mathrm{ext}$. Inference is fully Bayesian, gradient-based, and runs on a single GPU or scales across a cluster.

**Documentation:** [candel.readthedocs.io](https://candel.readthedocs.io/en/latest/)

## Overview

CANDEL forward-models distance-indicator observables (e.g. magnitude, line width, velocity dispersion) and redshift while marginalising over latent variables such as distance and absolute magnitude. Distance is either marginalised numerically via Simpson integration or sampled explicitly; latent observables are marginalised analytically where Gaussian conjugacy allows (SN stretch and colour via the Tripp relation; Fundamental Plane velocity dispersion and surface brightness), and via Gauss--Hermite quadrature otherwise. Posterior sampling uses the No-U-Turn Sampler (NUTS) from [NumPyro](https://github.com/pyro-ppl/numpyro), with JAX providing automatic differentiation and JIT compilation throughout.

When a reconstructed density and velocity field is supplied, CANDEL jointly calibrates each distance indicator and the underlying velocity field (e.g. amplitude $\beta$ and external bulk flow $\mathbf{V}_\mathrm{ext}$). The external dipole can also be inferred without any reconstructed field. For peculiar-velocity inferences, the distance prior is modelled following the phenomenological approach of [Lavaux (2016)](https://arxiv.org/abs/1512.04534), which effectively accounts for selection effects; for $H_0$ inferences, a rigorous selection function treatment is used instead (see [Stiskalek et al. 2025](https://arxiv.org/abs/2509.09665)). Model comparison is supported via BIC/AIC, Laplace evidence, and the [harmonic](https://github.com/astro-informatics/harmonic) package.

CANDEL runs locally for small samples or scales to computing clusters with GPU support (one GPU per chain). It includes cluster submission helpers and batch job generation tools for launching large parameter-grid runs from a frozen copy of the code.

### Key capabilities
- **Full-ladder forward modelling** in JAX and NumPyro, from Cepheid and TRGB calibrations to peculiar-velocity tracers.
- **Joint field-and-calibration inference:** distance-indicator relations and the underlying density/velocity field are constrained simultaneously, rather than calibrated in separate steps.
- **Reduced sampler dimensionality:** latent observables are marginalised analytically wherever Gaussian conjugacy allows, with Gauss--Hermite quadrature otherwise.
- **Flexible galaxy-bias models:** linear ($1 + b_1 \delta$), quadratic ($1 + b_1 \delta + b_2 \delta^2$), power-law ($\rho^\alpha$), and double power-law.
- **Density-dependent velocity dispersion** $\sigma_v(\delta)$ via a sigmoid in log-density, separating underdense and overdense regions.
- **Redshift-to-real-space mapping** of observed redshifts given a calibrated velocity field.
- **Peculiar-velocity covariance matrices** from CAMB power spectra.
- **HPC-ready tooling:** batch config generation, queue submission scripts, GPU auto-detection, and precomputed line-of-sight field generation.

## Supported distance indicators and catalogues

### Peculiar-velocity models

These models work in units of $h^{-1}\,\mathrm{Mpc}$ (i.e. assume $h = 1$). Multiple catalogues can be analysed jointly via `JointPVModel`, with user-specified shared parameters across sub-models.

- **Tully--Fisher relation:** 2MTF, SFI++, CF4-TFR
- **Type Ia supernovae (SALT2):** LOSS, Foundation, Pantheon+
- **Fundamental Plane:** 6dFGS-FP, SDSS-FP

### $H_0$ inference

- **Cepheid-calibrated $H_0$:** 35 Cepheid host galaxies from SH0ES
- **Milky Way Cepheid calibration:** standalone Galactic Cepheid period-luminosity calibration via `model.which_run = "MWCepheids"`
- **TRGB-calibrated $H_0$:** Tip of the Red Giant Branch distances from CCHP and EDD, including grouped EDD hosts

## Package structure

```
candel/
  model/          Forward models for each distance indicator
    mwcepheids/   Milky Way Cepheid calibration model
  pvdata/         Data loaders for all supported catalogues
  cosmo/          Cosmography, growth rate, PV covariance matrices
  inference/      NUTS sampling, nested sampling (NSS), Sobol+Adam optimisation, evidence estimation
  field/          3D density/velocity field loading and LOS interpolation
  redshift2real/  Map observed redshift → cosmological redshift given a velocity field
  mock/           Synthetic catalogue generation for testing
  plotting/       Reusable posterior and diagnostic plotting helpers
  util.py         Coordinate transforms, config I/O, plotting utilities

scripts/
  runs/           PV and H0 model configs and main runner
  H0_convergence/ H0 selection-integral convergence checks
  BORG_fields/    Reconstruction-field product helpers
  diagnostics/    Standalone model-component diagnostics
  mocks/          Mock TRGB inference runs
  preprocess/     Precompute line-of-sight density/velocity data
  sharing/        Posterior/data sharing utilities
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
python scripts/runs/generate_tasks.py list
python scripts/runs/generate_tasks.py build test
```

### Inference methods

- **NUTS** (default for the distance-indicator models): No-U-Turn Sampler via NumPyro. Robust, gradient-based.
- **Nested Slice Sampling (NSS):** Bayesian evidence computation via a self-contained reimplementation of the NSS algorithm ([Yallup et al. 2026](https://arxiv.org/abs/2601.23252)) in `candel/inference/nested.py`. No external nested-sampling dependency required.
- **Sobol + Adam MAP:** Multi-start MAP optimisation using Sobol quasi-random initialisation and Adam gradient descent. Configured via the `[optimise]` section of the TOML config.

## Results

CANDEL underpins a series of recent analyses:

- **A 1.8 per cent measurement of $H_0$ from Cepheids alone**, using a rigorous selection-function treatment of the SH0ES calibration sample.
  Stiskalek et al. (2025), [arXiv:2509.09665](https://arxiv.org/abs/2509.09665)

- **No evidence for $H_0$ anisotropy** in Tully--Fisher or supernova distances, constraining directional departures from isotropic expansion.
  Stiskalek et al. (2025), [arXiv:2509.14997](https://arxiv.org/abs/2509.14997)

- **$S_8$ from Tully--Fisher, Fundamental Plane, and supernova distances agrees with Planck**, from joint calibration of the velocity field and distance indicators.
  Stiskalek (2025), [arXiv:2509.20235](https://arxiv.org/abs/2509.20235)

- **The Velocity Field Olympics:** a systematic comparison of velocity-field reconstructions against direct distance tracers.
  Stiskalek et al. (2025), [arXiv:2502.00121](https://arxiv.org/abs/2502.00121)

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
CANDEL is installed. Start from `example_local_config.toml`. It supplies
machine-specific paths and Python interpreters used by the run scripts.

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
