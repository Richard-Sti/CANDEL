# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Script to remove tasks from the task list if the output file already exists.
"""

import argparse
import sys
from pathlib import Path

from candel import fprint, load_config, get_nested



def exists(path):
    return Path(path).is_file()


def make_condensed_filename(task_file):
    """Append 'C' to task file stem, e.g., tasks_0.txt → tasks_0C.txt."""
    if task_file.suffix != ".txt":
        raise ValueError("Expected a .txt file")
    return task_file.with_name(task_file.stem + "C.txt")


def filter_tasks(task_file, output_file):
    with open(task_file, "r") as f:
        lines = f.readlines()

    kept_lines = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            idx, config_path = line.split(maxsplit=1)
        except ValueError:
            fprint(f"[WARNING] Malformed line skipped: {line}")
            continue

        if not exists(config_path):
            fprint(f"[WARNING] Config file not found: {config_path}")
            continue

        try:
            config = load_config(
                config_path, replace_none=False, replace_los_prior=False)
            fname_out = get_nested(config, "io/fname_output")

            if exists(fname_out):
                fprint(f"[SKIP] Output exists: {fname_out} (task {idx})")
                continue

            kept_lines.append(f"{idx} {config_path}")

        except Exception as e:
            fprint(f"[ERROR] Failed to parse or process config: {config_path}")
            fprint(f"        {e}")
            continue

    with open(output_file, "w") as f:
        for new_idx, line in enumerate(kept_lines):
            _, config_path = line.split(maxsplit=1)
            f.write(f"{new_idx} {config_path}\n")

    fprint(f"[INFO] Wrote {len(kept_lines)} tasks to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a task file.")
    parser.add_argument(
        "tasks_index", type=int, nargs="?", default=0,
        help="Index of the task list to use (e.g., 0 → tasks_0.txt)"
    )
    args = parser.parse_args()

    task_file = Path(f"tasks_{args.tasks_index}.txt")
    if not task_file.exists():
        fprint(f"[ERROR] Task file not found: {task_file}")
        sys.exit(1)

    output_file = make_condensed_filename(task_file)
    filter_tasks(task_file, output_file)
