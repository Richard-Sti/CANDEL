# CANDEL Repository Map

This is a quick orientation map for LLM agents. It is not a replacement for
the source code, `README.md`, or the workflow guides in this folder.

## First Read

Start with these files when joining a task:

1. `/mnt/users/rstiskalek/CANDEL/AGENTS.md` for operating rules.
2. `/mnt/users/rstiskalek/CANDEL/README.md` for the scientific and package
   overview.
3. This file for code navigation.
4. A workflow guide in `/mnt/users/rstiskalek/CANDEL/instructions` before
   changing or running job generation, cluster submission, or maser workflows.

Avoid treating `/mnt/users/rstiskalek/CANDEL/data`,
`/mnt/users/rstiskalek/CANDEL/results`, and
`/mnt/users/rstiskalek/CANDEL/field_cache` as normal source directories.

## Runtime Shape

Most production runs follow this path:

```text
TOML config
  -> candel.util.load_config
  -> data loader in candel.pvdata
  -> model class in candel.model
  -> candel.inference runner
  -> HDF5 samples, diagnostics, plots under results/
```

`local_config.toml` is machine-local and not versioned. Reusable configs should
not bake in local paths, queue names, Python executables, or generated output
paths.

## Top-Level Layout

- `/mnt/users/rstiskalek/CANDEL/candel` - installable Python package.
- `/mnt/users/rstiskalek/CANDEL/scripts` - CLI runners, task generation,
  submission helpers, preprocessing, mocks, and diagnostics.
- `/mnt/users/rstiskalek/CANDEL/instructions` - short operational guides for
  agents and cluster workflows.
- `/mnt/users/rstiskalek/CANDEL/docs` - Sphinx docs plus design notes and
  numerical/debug plans.
- `/mnt/users/rstiskalek/CANDEL/notebooks` - exploratory and paper-analysis
  notebooks. Prefer package and script code for reusable behavior.
- `/mnt/users/rstiskalek/CANDEL/background_info`,
  `/mnt/users/rstiskalek/CANDEL/plots`, and
  `/mnt/users/rstiskalek/CANDEL/remote_logs` - reference/generated material;
  do not edit unless the task asks for it.

## Package Map

- `/mnt/users/rstiskalek/CANDEL/candel/__init__.py`
  exports the public convenience API used by scripts and notebooks.

- `/mnt/users/rstiskalek/CANDEL/candel/util.py`
  handles config loading and path resolution, coordinate transforms, plotting
  helpers, sample readers, labels, and local root helpers.

- `/mnt/users/rstiskalek/CANDEL/candel/pvdata/data.py`
  is the central catalogue loader module. It defines `PVDataFrame`, PV/H0 data
  loading from configs, line-of-sight loading, selection masks, field-cache
  helpers, and loaders for CF4, 2MTF, SFI, SN, SH0ES, CCHP, EDD TRGB, CSP, FP,
  and generic catalogues.

- `/mnt/users/rstiskalek/CANDEL/candel/pvdata/megamaser_data.py`
  loads spot-level megamaser data and velocity-frame conversions.

- `/mnt/users/rstiskalek/CANDEL/candel/model/base_model.py`
  contains shared model setup: priors, data arrays, cosmography, Malmquist and
  selection grids, reconstruction fields, galaxy bias, and H0 selection
  integral helpers.

- `/mnt/users/rstiskalek/CANDEL/candel/model/base_pv.py`
  contains shared PV logic, Mmiss forward terms, and `JointPVModel`.

- `/mnt/users/rstiskalek/CANDEL/candel/model/model_PV_TFR.py`,
  `/mnt/users/rstiskalek/CANDEL/candel/model/model_PV_SN.py`,
  `/mnt/users/rstiskalek/CANDEL/candel/model/model_PV_PantheonPlus.py`, and
  `/mnt/users/rstiskalek/CANDEL/candel/model/model_PV_FP.py`
  are the main PV likelihood implementations.

- `/mnt/users/rstiskalek/CANDEL/candel/model/model_H0_CH0.py` and
  `/mnt/users/rstiskalek/CANDEL/candel/model/model_H0_TRGB.py`
  implement Cepheid and TRGB H0 inference.

- `/mnt/users/rstiskalek/CANDEL/candel/model/model_H0_maser.py`
  implements the warped megamaser disk likelihood, marginalization grids,
  per-galaxy disk model, and `JointMaserModel`.

- `/mnt/users/rstiskalek/CANDEL/candel/model/dev`
  holds experimental models such as EDD-2MTF and CCHP+CSP. Check whether a
  requested change targets production or development behavior before editing.

- `/mnt/users/rstiskalek/CANDEL/candel/model/pv_utils.py`,
  `/mnt/users/rstiskalek/CANDEL/candel/model/utils.py`,
  `/mnt/users/rstiskalek/CANDEL/candel/model/integration.py`, and
  `/mnt/users/rstiskalek/CANDEL/candel/model/interp.py`
  hold shared numerical primitives: priors/distributions, external velocity
  fields, Mmiss kernels, galaxy bias, latent marginalization, log-space
  quadrature, and LOS interpolation.

- `/mnt/users/rstiskalek/CANDEL/candel/inference`
  contains inference engines and postprocessing: NumPyro NUTS in
  `inference.py`, nested slice sampling in `nested.py`, MAP optimization in
  `optimise.py`, and evidence utilities in `evidence.py`. NSS can shard
  replacement chains over multiple local devices on one node; the single-device
  path remains the fallback.

- `/mnt/users/rstiskalek/CANDEL/candel/field`
  loads density/velocity reconstructions and interpolates line-of-sight fields.

- `/mnt/users/rstiskalek/CANDEL/candel/cosmo`
  contains cosmography converters, beta-to-cosmology helpers, and PV covariance
  calculations.

- `/mnt/users/rstiskalek/CANDEL/candel/redshift2real`
  maps observed redshift to cosmological redshift given LOS fields.

- `/mnt/users/rstiskalek/CANDEL/candel/mock`
  generates synthetic catalogues and posterior predictive checks.

## Script Map

- `/mnt/users/rstiskalek/CANDEL/scripts/runs/main.py`
  is the generic config-driven inference runner. It dispatches by
  `model/which_run` for H0 and maser cases; otherwise it loads PV data with
  `candel.pvdata.load_PV_dataframes`, builds models through
  `candel.model.name2model`, and calls `run_pv_inference`.

- `/mnt/users/rstiskalek/CANDEL/scripts/runs/generate_tasks.py`
  expands named task specs into TOML configs and task lists. Read
  `/mnt/users/rstiskalek/CANDEL/instructions/inference_task_jobs.md` before
  editing it or the submission flow.

- `/mnt/users/rstiskalek/CANDEL/scripts/runs/specs_tasks.py`
  holds named parameter sweeps for `generate_tasks.py`.

- `/mnt/users/rstiskalek/CANDEL/scripts/runs/configs`
  contains reusable TOML templates for PV and H0 runs.

- `/mnt/users/rstiskalek/CANDEL/scripts/runs/submit.sh` and
  `/mnt/users/rstiskalek/CANDEL/scripts/_cluster_*.sh`
  handle cluster/local submission plumbing.

- `/mnt/users/rstiskalek/CANDEL/scripts/megamaser`
  contains maser-specific runners, config, submission scripts, mock checks,
  convergence diagnostics, and the toy joint-H0 combiner. Read
  `/mnt/users/rstiskalek/CANDEL/instructions/maser_disk_jobs.md` and
  `/mnt/users/rstiskalek/CANDEL/scripts/megamaser/README.md` before changing
  this workflow.

- `/mnt/users/rstiskalek/CANDEL/scripts/preprocess`
  precomputes line-of-sight density and velocity products and warms field
  caches.

- `/mnt/users/rstiskalek/CANDEL/scripts/H0_convergence`
  contains selection-integral and posterior subsampling diagnostics.

- `/mnt/users/rstiskalek/CANDEL/scripts/diagnostics`
  contains standalone diagnostic plotting scripts for model components.

- `/mnt/users/rstiskalek/CANDEL/scripts/mocks`
  contains TRGB mock run and merge scripts.

- `/mnt/users/rstiskalek/CANDEL/scripts/sync`
  contains machine sync helpers; avoid changing host-specific behavior without
  checking the relevant cluster guide.

## Common Edit Paths

- Adding a new catalogue: start in `candel/pvdata/data.py`, then wire config
  keys and runner behavior only if existing loaders cannot cover it.
- Adding or changing a PV model: start in `candel/model/base_pv.py` and the
  closest `model_PV_*.py` implementation; register production PV models in
  `candel/model/__init__.py`.
- Changing H0 selection behavior: start in `candel/model/base_model.py`, then
  the target H0 model file.
- Changing maser likelihood numerics: start in
  `candel/model/model_H0_maser.py`; then run the reduced convergence checks
  described in the maser guides.
- Changing inference mechanics: start in `candel/inference/inference.py`,
  `candel/inference/nested.py`, or `candel/inference/optimise.py`.
- Changing generated task behavior: read
  `/mnt/users/rstiskalek/CANDEL/instructions/inference_task_jobs.md`, then edit
  `scripts/runs/generate_tasks.py` or `scripts/runs/specs_tasks.py`.
- Changing cluster submission: read
  `/mnt/users/rstiskalek/CANDEL/instructions/glamdring_gpu_jobs.md` first.

## Verification Anchors

Use the smallest check that proves the change:

- Package syntax/import smoke:
  `/mnt/users/rstiskalek/CANDEL/venv_candel/bin/python -m compileall candel scripts`
- Package import:
  `/mnt/users/rstiskalek/CANDEL/venv_candel/bin/python -c "import candel"`
- Task generator changes:
  `generate_tasks.py list`, `show <task_index>`, then
  `build <task_index> --dry-run`.
- Maser likelihood changes:
  start with the reduced convergence checks in
  `/mnt/users/rstiskalek/CANDEL/scripts/megamaser/convergence`.
