# Installing CANDEL with GPU support on Glamdring

## 1. Check available CUDA

```bash
ls /usr/local/cuda*
nvidia-smi
```

Note the CUDA version (e.g. 11.8, 12.x). This determines which `jaxlib` wheel to install.

## 2. Create the venv

A single venv is used for both CPU and GPU jobs: JAX is installed with
the CUDA wheels and falls back to CPU automatically when no GPU is
visible.

```bash
cd /mnt/users/$USER/CANDEL
python3 -m venv venv_candel
source venv_candel/bin/activate
pip install --upgrade pip
```

## 3. Install JAX with GPU support

Match the CUDA version on glamdring. For CUDA 12:

```bash
pip install "jax[cuda12]"
```

For CUDA 11:

```bash
pip install "jax[cuda11_local]"
```

This installs both `jax` and `jaxlib` with the correct CUDA bindings.

See https://github.com/jax-ml/jax#installation for the full compatibility matrix.

## 4. Install CANDEL (and optional extras)

```bash
# Editable install pulls in all required deps from setup.py
pip install -e .

# Optional extras (not required for core inference):
pip install optax       # Adam MAP optimisation
pip install evosax      # differential-evolution MAP search
pip install harmonic    # learnt-harmonic-mean evidence
```

## 5. Verify

Run on a GPU node (not the login node):

```bash
addqueue -q cmbgpu -s -m 4 --gpus 1 --gputype rtx3090with24gb \
    /mnt/users/$USER/CANDEL/venv_candel/bin/python -c \
    "import jax; print(jax.devices()); assert jax.devices()[0].platform == 'gpu'"
```

Or interactively:

```bash
python -c "import jax; print(jax.devices())"
```

Should print something like `[GpuDevice(id=0, ...)]`.

## 6. Configure local_config.toml

Update `local_config.toml` on glamdring to point to the venv:

```toml
root_main = "/mnt/users/rstiskalek/CANDEL/"
python_exec = "/mnt/users/rstiskalek/CANDEL/venv_candel/bin/python"
machine = "glamdring"
```

## 7. Update freeze_candel.sh

Add a `glamdring` machine block (if not already present):

```bash
elif [[ "$machine" == "glamdring" ]]; then
    src_dir="/mnt/users/${USER}/CANDEL/candel"
    main_script="/mnt/users/${USER}/CANDEL/scripts/runs/main.py"
    frozen_root="/mnt/users/${USER}/frozen_candel"
```

Then freeze and submit:

```bash
cd /mnt/users/$USER/CANDEL/scripts/runs
bash freeze_candel.sh
./submit_glamdring.sh -q cmbgpu 0
```
