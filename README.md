# CANDEL

**CANDEL** (*CA*libration and *N*ormalization of the *D*istanc*E* *L*adder) is a JAX-based framework for peculiar-velocity inference and full distance-ladder modelling.

## Overview

CANDEL is designed to **jointly calibrate** each distance indicator (e.g. the slope and intercept of the Tully–Fisher relation) while **simultaneously calibrating** the underlying density and velocity field.

Its core philosophy is to **forward-model the observables** (e.g. magnitudes, redshifts) while **marginalising over latent variables** such as distances or absolute magnitudes.
Marginalisation can be performed either via **numerical integration** or **Hamiltonian Monte Carlo (HMC)** sampling. Moreover, if the velocity field is provided, it also allows one to account for the peculiar velocities when forward-modelling the observed redshift.

The selection function is typically modelled following the phenomenological approach of
[Lavaux (2016)](https://arxiv.org/abs/1512.04534). For precision measurements sensitive to selection effects (e.g. \(H_0\)), a more detailed treatment is recommended (see [Stiskalek et al. 2025](https://arxiv.org/abs/2509.09665)).

## Supported distance indicators and catalogues
- **Tully–Fisher relation:** 2MTF, SFI++, CF4-TFR
- **Type Ia supernovae:** LOSS, Foundation, Pantheon+
- **Fundamental Plane:** 6dFGS-FP, SDSS-FP
- **Cluster scaling relations:** X-ray, SZ

---

### Citation
If you use CANDEL, please cite:

> **Stiskalek et al. (2025)**,
> *CANDEL: Calibration and Normalization of the Distance Ladder*, in prep.
> [arXiv:2509.09665](https://arxiv.org/abs/2509.09665)













# CANDEL

*CANDEL* (**CA**libration and **N**ormalization of the **D**istanc**E** **L**adder) is a JAX-based framework for peculiar velocity inferences, as well as for a complete distance ladder modelling.


The current supported distance indicators and catalogues are:
- Tully-Fisher relation (2MTF, SFI++, CF4-TFR)
- Type Ia supernovae (LOSS, Foundation, Pantheon+)
- Fundamental Plane (6dFGS-FP, SDSS-FP)
- Cluster scaling relations (X-ray, SZ)

It is designed to jointly calibrate the distance indicator (e.g. the slope and intercept of the Tully-Fisher relation) while simultaneously calibrating (or inferring) the underlying velocity field.

The philosophy of *CANDEL* inferences is to forward-model the observables (e.g. magnitudes, redshifts) while marginalizing over the latent variables (e.g. distances, absolute magnitudes). This marginalisation can either be done via numerical integration or Monte Carlo sampling with a Hamiltonian Monte Carlo (HMC) sampler. Selection function is typically modelled via the phenomenological approach of [Lavaux (2016)](https://arxiv.org/abs/1512.04534),

though for a precision measurements which are sensitive to selection effects (e.g. H0), a more careful modelling of the selection function is recommended (see e.g. [Stiskalek et al. (2025)](https://arxiv.org/abs/2509.09665)).



All forward models are built via JAX and NumPyro, which allows for automatic differentiation and GPU acceleration. The HMC sampler is provided by NumPyro, which is a lightweight probabilistic programming library built on JAX.





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

To enable model evidence computation, install the [harmonic](https://github.com/astro-informatics/harmonic) package.


If the package is not recognised in a notebook, add the project directory to the Python path manually, e.g.:
```
import sys
sys.path.insert(0, "/Users/rstiskalek/Projects/candel")
```


## TODO
