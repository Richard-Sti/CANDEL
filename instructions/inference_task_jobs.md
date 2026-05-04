# Generating and Submitting Inference Tasks

## Files

- Generator: `/mnt/users/rstiskalek/CANDEL/scripts/runs/generate_tasks.py`
- Task specs: `/mnt/users/rstiskalek/CANDEL/scripts/runs/task_specs.py`
- Submitter: `/mnt/users/rstiskalek/CANDEL/scripts/runs/submit.sh`
- Generated configs: `/mnt/users/rstiskalek/CANDEL/scripts/runs/generated_configs/<task_index>/`
- Task list: `/mnt/users/rstiskalek/CANDEL/scripts/runs/tasks_<task_index>.txt`

The script is named `generate_tasks.py` in this repo.

## How `generate_tasks.py` Works

`generate_tasks.py` is a task-list generator, not the job runner. It loads a
base TOML config, applies a grid of override dictionaries, writes one generated
TOML config per grid point, and writes a `tasks_<task_index>.txt` file that
`submit.sh` can consume.

It can be run from any working directory:

```bash
/mnt/users/rstiskalek/CANDEL/venv_candel/bin/python \
    /mnt/users/rstiskalek/CANDEL/scripts/runs/generate_tasks.py <task_index>
```

The `<task_index>` must be registered in `task_specs.py`. Historical generated
outputs such as `b1_beta_variation` and `S8_production` may exist under
`generated_configs/`, but they are not buildable unless listed by
`generate_tasks.py list`. Frozen historical task files for those runs live in
`/mnt/users/rstiskalek/CANDEL/scripts/runs/used_configs/`; copy from there only
when intentionally reproducing an old run instead of generating a new spec.

The sweep is defined in `/mnt/users/rstiskalek/CANDEL/scripts/runs/task_specs.py`.
To create a new sweep, add a named entry there:

- Choose the base config via `config_path`.
- Put shared settings in `common`.
- Put per-dataset settings in `datasets`.
- Use slash-delimited keys for nested TOML values, e.g.
  `model/priors/beta` or `io/catalogue_name`.
- Scalar values are used as-is.
- List values are expanded as a Cartesian product.
- Special case: if both `inference/model` and `io/catalogue_name` are lists of
  the same length, they are kept paired for joint-model configs instead of
  Cartesian-expanded against each other.
- Use `expected_tasks` to make accidental grid-size changes fail early.

Machine-local keys such as `root_main`, `root_data`, `root_results`,
`python_exec`, `machine`, `modules`, `modules_gpu`, `use_frozen`, and GPU
library paths are intentionally not baked into generated configs. They are read
from `/mnt/users/rstiskalek/CANDEL/local_config.toml` at job runtime.

Each task-list row has this format:

```text
<task_id> <repo-relative-generated-config-path>
```

The task file also gets a comment-only provenance footer containing the
generator source, base config source, timestamp, task count, and body hash.
`submit.sh` skips comment lines.

## Autonomous Agent Workflow

1. Read this guide plus any relevant cluster guide in
   `/mnt/users/rstiskalek/CANDEL/instructions`.
2. Inspect available task specs:

```bash
/mnt/users/rstiskalek/CANDEL/venv_candel/bin/python \
    /mnt/users/rstiskalek/CANDEL/scripts/runs/generate_tasks.py list
```

3. Inspect the target spec before editing or building:

```bash
/mnt/users/rstiskalek/CANDEL/venv_candel/bin/python \
    /mnt/users/rstiskalek/CANDEL/scripts/runs/generate_tasks.py \
    show <task_index>
```

4. Edit the sweep in `/mnt/users/rstiskalek/CANDEL/scripts/runs/task_specs.py`
   if the requested task does not already exist.
5. Dry-run the generated task rows:

```bash
/mnt/users/rstiskalek/CANDEL/venv_candel/bin/python \
    /mnt/users/rstiskalek/CANDEL/scripts/runs/generate_tasks.py \
    build <task_index> --dry-run
```

6. Generate the task list:

```bash
/mnt/users/rstiskalek/CANDEL/venv_candel/bin/python \
    /mnt/users/rstiskalek/CANDEL/scripts/runs/generate_tasks.py \
    build <task_index>
```

   Use `--clean` when intentionally regenerating an existing task index and
   removing stale generated TOMLs for tasks that no longer exist.

7. Inspect the task list before submitting:

```bash
cd /mnt/users/rstiskalek/CANDEL/scripts/runs
sed -n '1,80p' tasks_<task_index>.txt
./submit.sh --status <task_index>
```

8. Do a dry run for the intended queue and task subset:

```bash
printf 'y\n' | ./submit.sh -q <queue> --tasks <ids-or-ranges> --dry <task_index>
```

9. Submit only after the dry run looks correct:

```bash
printf 'y\n' | ./submit.sh -q <queue> --tasks <ids-or-ranges> --skip-done <task_index>
```

Use `--tasks 0`, `--tasks 0-3`, or `--tasks 0,2,5-7` to limit submissions.
Use `--skip-done` for production submissions so completed outputs are not
resubmitted. Omit `--tasks` only when intentionally submitting the full file.
For active submissions, `tasks_<task_index>.txt` is authoritative; old
directories under `generated_configs/` can be stale.

For Glamdring GPU jobs, typical queues are `gpulong`, `cmbgpu`, and `optgpu`.
See `/mnt/users/rstiskalek/CANDEL/instructions/glamdring_gpu_jobs.md` for GPU
hardware and queue details.

## Useful Submit Options

```bash
./submit.sh --status <task_index>
printf 'y\n' | ./submit.sh -q cmbgpu -n 4 -m 6 --tasks 0 --dry <task_index>
printf 'y\n' | ./submit.sh -q cmbgpu -n 4 -m 6 --tasks 0 --skip-done <task_index>
printf 'y\n' | ./submit.sh -q gpulong --gputype rtx2080with12gb --tasks 0 <task_index>
```

- `-q/--queue`: required for cluster submissions.
- `-n/--ncpu`: CPU cores passed as `--host-devices`.
- `-m/--memory`: memory per CPU in GB; total request is `memory * ncpu`.
- `--tasks`: comma-separated task IDs and ranges.
- `--status`: report done/pending state without submitting.
- `--dry`: print submit commands without submitting.
- `--skip-done`: skip configs whose `io/fname_output` already exists.
- `--local`: run inline instead of submitting to a batch queue.

`submit.sh` reads `machine` and `python_exec` from
`/mnt/users/rstiskalek/CANDEL/local_config.toml`. If `machine = "local"`, queue
submission is disabled and the script runs inline.
