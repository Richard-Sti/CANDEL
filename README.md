# CANDEL

**CANDEL** (*CA*libration and *N*ormalization of the *D*istanc*E* *L*adder) is a JAX-based framework for peculiar-velocity inference and distance-ladder calibration.

**Documentation:** [candel.readthedocs.io](https://candel.readthedocs.io/en/latest/)

## Overview

CANDEL forward-models distance-indicator observables (e.g. magnitude, line width, velocity dispersion) and redshift while marginalising over latent variables such as distance and absolute magnitude. Distance is either marginalised numerically via Simpson integration or sampled explicitly; latent observables are marginalised analytically where Gaussian conjugacy allows (SN stretch and colour via the Tripp relation; Fundamental Plane velocity dispersion and surface brightness), and via Gauss--Hermite quadrature otherwise. Posterior sampling uses the No-U-Turn Sampler (NUTS) from [NumPyro](https://github.com/pyro-ppl/numpyro), with JAX providing automatic differentiation and JIT compilation throughout.

When a reconstructed density and velocity field is supplied, CANDEL jointly calibrates each distance indicator and the underlying velocity field (e.g. amplitude $\beta$ and external bulk flow $\mathbf{V}_\mathrm{ext}$). The external dipole can also be inferred without any reconstructed field. For peculiar-velocity inferences, the distance prior is modelled following the phenomenological approach of [Lavaux (2016)](https://arxiv.org/abs/1512.04534), which effectively accounts for selection effects; for $H_0$ inferences, a rigorous selection function treatment is used instead (see [Stiskalek et al. 2025](https://arxiv.org/abs/2509.09665)). Model comparison is supported via BIC/AIC, Laplace evidence, and the [harmonic](https://github.com/astro-informatics/harmonic) package.

CANDEL runs locally for small samples or scales to computing clusters with GPU support (one GPU per chain). It includes SLURM submission helpers and batch job generation tools for launching large parameter-grid runs from a frozen copy of the code.

### Highlights
- Forward modelling of the full distance ladder with JAX and NumPyro.
- Joint calibration of distance-indicator relations and the underlying density/velocity field.
- Analytical marginalisation of latent observables where Gaussian conjugacy allows, reducing sampler dimensionality.
- Multiple galaxy-bias models (linear, power-law, double power-law).
- Redshift-to-real-space mapping of observed redshifts given a calibrated velocity field.
- Peculiar-velocity covariance matrices from CAMB power spectra.
- HPC-friendly tooling: batch config generation, SLURM submission scripts, GPU auto-detection.

## Supported distance indicators and catalogues

### Peculiar-velocity models

These models work in units of $h^{-1}\,\mathrm{Mpc}$ (i.e. assume $h = 1$). Multiple catalogues can be analysed jointly via `JointPVModel`, with user-specified shared parameters across sub-models.

- **Tully--Fisher relation:** 2MTF, SFI++, CF4-TFR
- **Type Ia supernovae (SALT2):** LOSS, Foundation, Pantheon+
- **Fundamental Plane:** 6dFGS-FP, SDSS-FP

### $H_0$ inference

- **Cepheid-calibrated $H_0$:** 35 Cepheid host galaxies from SH0ES
- **TRGB-calibrated $H_0$:** Tip of the Red Giant Branch distances from CCHP, EDD, and SH0ES
- **2MTF-calibrated $H_0$:** Tully--Fisher distances from the 2MTF survey

## Package structure

```
candel/
  model/          Forward models for each distance indicator
  pvdata/         Data loaders for all supported catalogues
  cosmo/          Growth rate, PV covariance matrices (CAMB + Legendre)
  field/          3D density/velocity field loading and LOS interpolation
  redshift2real/  Map observed redshift → cosmological redshift given a velocity field
  mock/           Synthetic catalogue generation for testing
  inference.py    NUTS sampling, MAP initialisation, postprocessing
  evidence.py     BIC/AIC, Laplace and harmonic evidence estimation
  cosmography.py  Distance modulus ↔ comoving distance ↔ redshift
  util.py         Coordinate transforms, config I/O, plotting utilities
```

## Running inference

All experiments are defined in TOML configuration files that specify data paths, model parameters, priors, and output locations. The main entry point is:

```bash
python scripts/runs/main.py --config path/to/config.toml
```

To generate a batch of configs from a template with a parameter grid:

```bash
python scripts/runs/generate_tasks.py
```

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

For model-evidence computation, also install [harmonic](https://github.com/astro-informatics/harmonic) (note: there may be compatibility issues with recent JAX versions).

## Citation

If you use CANDEL, or find it useful, please cite the papers listed above.

## License

GNU General Public License v3.0 -- see [LICENSE](LICENSE) for details.
