#!/usr/bin/env python
"""Run the CCHP TRGBH0 single-field diagnostic scripts."""

from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
TASK_DIR = ROOT / "scripts" / "runs"
RESULTS = ROOT / "results" / "TRGBH0_paper"


def run_flat(script, output_dir, prefix, *args):
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f"{prefix}_") as tmp:
        tmpdir = Path(tmp)
        command = [
            sys.executable,
            str(SCRIPT_DIR / script),
            *map(str, args),
            "--output-dir",
            str(tmpdir),
        ]
        subprocess.run(command, check=True)
        for path in sorted(tmpdir.iterdir()):
            if path.is_file():
                shutil.copyfile(path, output_dir / f"{prefix}_{path.name}")


def main():
    single_task = TASK_DIR / "tasks_TRGBH0_CCHP_single.txt"
    smoothed_task = TASK_DIR / "tasks_TRGBH0_CCHP_single_smoothed.txt"

    for likelihood in ("gaussian", "student_t"):
        run_flat(
            "plot_trgbh0_single_field_comparison.py",
            RESULTS / "cchp_single_fields" / "plots",
            f"single_field_{likelihood}",
            "--task-file", single_task,
            "--cz-likelihood", likelihood,
            "--require-complete",
            "--fail-on-unusable",
        )

        run_flat(
            "plot_trgbh0_single_smoothed_diagnostics.py",
            RESULTS / "cchp_single_fields_smoothed" / "plots",
            f"single_smoothed_{likelihood}",
            "--task-file", smoothed_task,
            "--baseline-task-file", single_task,
            "--cz-likelihood", likelihood,
            "--allow-missing",
        )

    run_flat(
        "plot_trgbh0_single_smoothed_likelihood_comparison.py",
        RESULTS / "cchp_single_fields" / "plots",
        "likelihood_R0",
        "--task-file", single_task,
        "--smooth-R", 0,
        "--mas", "PCS",
        "--allow-missing",
    )

    for smooth_R in (4, 8):
        run_flat(
            "plot_trgbh0_single_smoothed_likelihood_comparison.py",
            RESULTS / "cchp_single_fields_smoothed" / "plots",
            f"likelihood_R{smooth_R}",
            "--task-file", smoothed_task,
            "--smooth-R", smooth_R,
            "--mas", "PCS",
            "--allow-missing",
        )


if __name__ == "__main__":
    main()
