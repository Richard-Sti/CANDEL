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
- Multiple galaxy-bias models: linear ($1 + b_1 \delta$), quadratic ($1 + b_1 \delta + b_2 \delta^2$), power-law ($\rho^\alpha$), double power-law, and free-form cubic spline in $\log(1+\delta)$ with a configurable number of knots.
- Density-dependent velocity dispersion $\sigma_v(\delta)$ via a sigmoid in log-density, allowing different dispersions in underdense and overdense regions.
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
- **TRGB-calibrated $H_0$:** Tip of the Red Giant Branch distances from CCHP and EDD
- **2MTF-calibrated $H_0$:** Tully--Fisher distances from the EDD-2MTF sample *(experimental)*

## Megamaser disk model

CANDEL includes a warped Keplerian disk model (`MaserDiskModel`) for fitting VLBI water-maser spots directly, following the methodology of the Megamaser Cosmology Project ([Humphreys et al. 2013](https://arxiv.org/abs/1307.6031); [Pesce et al. 2020](https://arxiv.org/abs/2001.09213)). The model marginalises over azimuthal angle $\phi$ on a per-spot basis using log-space Simpson integration, and supports:

- **Warped geometry:** linear inclination gradient $\mathrm{d}i/\mathrm{d}r$ across the disk.
- **Eccentricity:** optional eccentric orbits with $e$ and $\omega_\mathrm{disk}$ (disabled by default).
- **Spot classification:** high-velocity (red/blue) spots constrained by Keplerian velocity, systemic spots constrained by sky position and LOS velocity near $v_\mathrm{sys}$.
- **Acceleration data:** radial acceleration constraints where available.

Supported galaxies: NGC 5765b, NGC 6264, NGC 6323, UGC 3789, CGCG 074-064, NGC 4258. In addition to the full disk model, CANDEL can also use published distance posteriors directly (e.g. from Pesce+2020) via `MegamaserModel` for a simpler analysis that bypasses spot-level fitting.

For grid convergence tests and numerical accuracy details, see [`docs/maser_numerical_accuracy.md`](docs/maser_numerical_accuracy.md). For running maser disk inference jobs, see [`instructions/maser_disk_jobs.md`](instructions/maser_disk_jobs.md).

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
python scripts/megamaser/run_maser_disk.py --config scripts/megamaser/config_maser.toml
```

### Inference methods

- **NUTS** (default): No-U-Turn Sampler via NumPyro. Robust, gradient-based, suitable for all models.
- **Nested Slice Sampling (NSS):** Bayesian evidence computation via the [blackjax](https://github.com/handley-lab/blackjax) nested sampling fork. Requires the optional `blackjax` and `nss` packages (see [Installation](#installation)).
- **Sobol + Adam MAP:** Multi-start MAP optimisation using Sobol quasi-random initialisation and Adam gradient descent. Configured via the `[optimise]` section of the TOML config.

### Job submission guides

The `instructions/` folder contains how-to guides for HPC job submission (GPU queues, batch configuration, grid sizes).

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

For GPU support (e.g. on Glamdring), see [INSTALL_GLAMDRING_GPU.md](INSTALL_GLAMDRING_GPU.md).

For nested sampling (NSS), install the [handley-lab blackjax fork](https://github.com/handley-lab/blackjax/tree/nested_sampling) and [nss](https://github.com/yallup/nss):
```bash
pip install "blackjax @ git+https://github.com/handley-lab/blackjax@nested_sampling" --no-deps
pip install git+https://github.com/yallup/nss.git
```
These are optional — CANDEL's core functionality (NUTS, optimisation) works without them.

For model-evidence computation, also install [harmonic](https://github.com/astro-informatics/harmonic) (note: there may be compatibility issues with recent JAX versions).

## Local configuration (`local_config.toml`)

Per-machine settings live in a `local_config.toml` file at the repository
root. This file is **not** versioned and must be created on each machine where
CANDEL is installed. It supplies machine-specific paths and Python interpreters
that the run scripts and submission helpers read.

A minimal `local_config.toml` looks like:

```toml
root_main    = "/path/to/CANDEL/"   # repo root (required)
root_data    = "/path/to/data/"     # optional, defaults to <root_main>/data
root_results = "/path/to/results/"  # optional, defaults to <root_main>/results

python_exec = "/path/to/venv_candel/bin/python"
machine     = "glamdring"           # "glamdring" (addqueue) or "arc" (sbatch)

modules     = ""                    # optional, space-separated module list
modules_gpu = ""                    # optional, overrides `modules` for GPU jobs
```

Keys:

- `root_main` — repository root. Required. Used as the fallback for the data
  and results roots when those are not set.
- `root_data` — base directory for input data files (catalogues, fields,
  reconstructions). Optional; defaults to `<root_main>/data`. Set this when
  the data live on a different filesystem than the code (e.g. on an HPC
  cluster with a separate scratch area).
- `root_results` — base directory for outputs (samples, plots, logs).
  Optional; defaults to `<root_main>/results`. Same rationale as
  `root_data`.
- `python_exec` — absolute path to the CANDEL Python interpreter used by
  submission scripts. A single venv is used for both CPU and GPU jobs:
  install JAX with the CUDA wheels (`pip install "jax[cuda12]"`) and it
  falls back to CPU automatically when no GPU is visible.
- `machine` — selects the cluster submission backend. Currently supported:
  `"glamdring"` (uses `addqueue`) and `"arc"` (uses `sbatch`). Also appears
  in logs and tags. An unknown value will break job submission.
- `modules` — space-separated list of environment modules to `module add`
  before a job runs. Optional; empty or missing means no modules are loaded.
- `modules_gpu` — same as `modules`, but used for GPU jobs instead of
  `modules`. Optional.
- `gpu_ld_library_path` — list of directories prepended to `LD_LIBRARY_PATH`
  on GPU jobs (typically the bundled NVIDIA libs in the venv plus system
  CUDA paths). Optional; leave empty unless JAX fails to find cuDNN/cuBLAS
  at runtime.

Path resolution: relative paths in run-time TOML configs are resolved against
the appropriate root — input data file keys against `root_data`, output keys
(`fname_output`) against `root_results`. Absolute paths are left unchanged.

## Paper notes

Implementation findings, data notes, and numerical results that arise during development should be written up in the **"Notes from code"** appendix of the megamaser paper draft:

```
/mnt/users/rstiskalek/Papers/MMH0/main.tex  →  \section{Notes from code}
```

This appendix is the canonical place for:
- Data availability findings (missing tables, non-public datasets, data quirks)
- Numerical results from convergence/grid studies
- Observations about spot classification or selection effects
- Implementation decisions with physical motivation

Keep entries terse and bolded by topic, matching the style of the existing subsections there.

## Citation

If you use CANDEL, or find it useful, please cite the papers listed above.

## License

GNU General Public License v3.0 -- see [LICENSE](LICENSE) for details.
