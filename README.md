# CANDEL

**CANDEL** (*CA*libration and *N*ormalization of the *D*istanc*E* *L*adder) is a JAX-based framework for peculiar-velocity inference and full distance-ladder modelling.

## Overview

CANDEL is designed to jointly calibrate each distance indicator (e.g. the slope and intercept of the Tully–Fisher relation) while simultaneously calibrating the underlying density and velocity field.

Its core philosophy is to forward-model the observables (e.g. magnitudes, redshifts) while marginalising over latent variables such as distances or absolute magnitudes. Marginalisation can be performed either via numerical integration or Hamiltonian Monte Carlo (HMC) sampling. If the velocity field is provided, it also allows one to account for the peculiar velocities when forward-modelling the observed redshift. CANDEL also supports model comparison via Bayesian evidence computation using the [harmonic](https://github.com/astro-informatics/harmonic) package.

The selection function is typically modelled following the phenomenological approach of
[Lavaux (2016)](https://arxiv.org/abs/1512.04534). For precision measurements sensitive to selection effects (e.g. $H_0$), a more detailed treatment is recommended (see [Stiskalek et al. 2025](https://arxiv.org/abs/2509.09665)).

CANDEL can be run locally for small samples or scaled to computing clusters with full GPU support. It includes examples for SLURM submission and tools for batch job generation, enabling runs to be launched individually or in groups using a frozen version of the code. This setup efficiently leverages available computational resources and accelerates development.


## Supported distance indicators and catalogues
- **Tully–Fisher relation:** 2MTF, SFI++, CF4-TFR
- **Type Ia supernovae:** LOSS, Foundation, Pantheon+
- **Fundamental Plane:** 6dFGS-FP, SDSS-FP
- **Cluster scaling relations:** X-ray, SZ

## Example

CANDEL uses configuration files to set up the paths, data, and model parameters. A working example will be provided soon! In the meantime, please get in touch if you have any questions.

Here are some recent works that have used CANDEL:
1. *The Velocity Field Olympics: Assessing velocity field reconstructions with direct distance tracers*; Stiskalek et al. (2025)
  [arXiv:2502.00121](https://arxiv.org/abs/2502.00121)

2. *1.8 per cent measurement of H₀ from Cepheids alone*; Stiskalek et al. (2025)
  [arXiv:2509.09665](https://arxiv.org/abs/2509.09665)

3. *No evidence for H₀ anisotropy from Tully–Fisher or supernova distances*; Stiskalek et al. (2025)
  [arXiv:2509.14997](https://arxiv.org/abs/2509.14997)

4. *S₈ from Tully–Fisher, fundamental plane, and supernova distances agree with Planck*; Stiskalek (2025)
  [arXiv:2509.20235](https://arxiv.org/abs/2509.20235)

## Installation
```
git clone git@github.com:Richard-Sti/CANDEL.git

# Go to the cloned directory
cd CANDEL

# Create a virtual environment
python -m venv venv_candel
source venv_candel/bin/activate
python -m pip install --upgrade pip && python -m pip install --upgrade setuptools

# Finally install the cloned package
python -m pip install -e .
```

To enable model evidence computation, install the [harmonic](https://github.com/astro-informatics/harmonic) package. But there may be some compatibility issues with the latest JAX versions.

## Citation

If you use CANDEL, or find it useful, please cite the papers listed in the example section.
