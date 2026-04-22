# Submitting GPU Jobs on Glamdring

## GPU Queues and Hardware

| Queue | GPU | VRAM | Nodes | Priority |
|-------|-----|------|-------|----------|
| `gpulong` | RTX 2080 Ti | 12 GB | gpu01–gpu05 (2 each) | open |
| `gpulong` | RTX 3070 | 8 GB | gpu06 (1) | open |
| `cmbgpu` | RTX 3090 | 24 GB | gpu07 (4) | CMB group |
| `optgpu` | RTX A6000 | 48 GB | gpu04 (1) | optics group |

Non-priority jobs on `cmbgpu`/`optgpu` may be killed after 6 hours.

## Checking GPU Availability

```bash
showgpus
```

Shows free/total GPUs, memory, and utilisation per node.

## Submitting Jobs

### Basic syntax

```bash
addqueue -q QUEUE -s -m MEMORY_GB --gpus 1 COMMAND [ARGS...]
```

- `-q QUEUE`: queue name (`gpulong`, `cmbgpu`, `optgpu`)
- `-s`: single node
- `-m MEMORY_GB`: CPU memory per task in GB
- `--gpus 1`: request 1 GPU

### Requesting a specific GPU type

```bash
addqueue -q gpulong -s -m 8 --gpus 1 --gputype rtx2080with12gb ./my_script.py
```

Valid `--gputype` values:
- `rtx2080with12gb`
- `rtx3070with8gb`
- `rtx3090with24gb`
- `rtxa6000with48gb`

If `--gputype` is omitted, the scheduler picks any available GPU in the queue.

### Examples

Run a Python script on any `gpulong` GPU:
```bash
addqueue -q gpulong -s -m 16 --gpus 1 \
    /mnt/users/$USER/CANDEL/venv_candel/bin/python -u my_script.py
```

Run on the A6000 (48 GB):
```bash
addqueue -q optgpu -s -m 16 --gpus 1 \
    /mnt/users/$USER/CANDEL/venv_candel/bin/python -u my_script.py
```

Run on a specific RTX 3090:
```bash
addqueue -q cmbgpu -s -m 16 --gpus 1 --gputype rtx3090with24gb \
    /mnt/users/$USER/CANDEL/venv_candel/bin/python -u my_script.py
```

## Output Files

Job output goes to `python-<JOBID>.out` in the current working directory.
The job ID is printed at submission time:

```
Sending program's output to file: python-569968.out
```

To watch output in real-time:
```bash
tail -f python-569968.out
```

## Monitoring Jobs

```bash
q                    # shorthand for squeue -u $USER (if aliased)
squeue -u $USER      # list your running/pending jobs
scancel <jobid>      # cancel a job
```

## Using with CANDEL

### Python executable

CANDEL uses a single venv (`venv_candel`) for both CPU and GPU jobs;
JAX is installed with the CUDA wheels and falls back to CPU automatically
when no GPU is visible. The interpreter lives at:
```
/mnt/users/$USER/CANDEL/venv_candel/bin/python
```

This is also set in `local_config.toml` as `python_exec`.

### Submit script for inference

The main inference submit script handles GPU queues automatically:
```bash
cd scripts/runs
./submit_glamdring.sh -q cmbgpu 0       # submit task 0 to RTX 3090
./submit_glamdring.sh -q optgpu 0       # submit task 0 to A6000
./submit_glamdring.sh -q gpulong 0      # submit task 0 to RTX 2080
```

### Submitting a one-off script

Template for submitting a standalone Python script to GPU:

```bash
#!/bin/bash -l
QUEUE=${1:-gpulong}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT_DIR/venv_candel/bin/python"

addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
    $PYTHON -u "$ROOT_DIR/scripts/my_script.py" --arg1 val1
```

### Tips

- Always use `python -u` or set `PYTHONUNBUFFERED=1` in your script to
  avoid output buffering (output appears only at the end otherwise).
- Use `nvidia-smi` inside your script to log GPU info.
- For large models (CF4 TFR, N>2000), prefer `cmbgpu` (24 GB) or
  `optgpu` (48 GB) over `gpulong` (8–12 GB).
- JAX pre-allocates 75% of GPU memory by default. Set
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` if sharing the GPU.
