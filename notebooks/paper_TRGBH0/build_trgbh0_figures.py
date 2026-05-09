#!/usr/bin/env python
"""Build TRGBH0 paper figures by broad category."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
TARGETS = {
    "data": SCRIPT_DIR / "plot_edd_trgb_data_summary.py",
    "model": SCRIPT_DIR / "render_trgb_forward_model_dag.py",
}
TARGET_OUTPUTS = {
    "data": [
        "edd_trgb_velocity_magnitude_histograms.pdf",
        "edd_trgb_sky_distribution.pdf",
    ],
    "model": ["trgb_forward_model_dag.pdf"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "targets", nargs="*",
        help="Figure categories to build. If omitted, build all categories.")
    parser.add_argument(
        "--paper-figdir", type=Path, default=None,
        help="Optional directory to copy generated PDFs into.")
    return parser.parse_args()


def run_target(name, args):
    command = [sys.executable, str(TARGETS[name])]
    subprocess.run(command, check=True)
    if args.paper_figdir is not None:
        args.paper_figdir.mkdir(parents=True, exist_ok=True)
        for filename in TARGET_OUTPUTS[name]:
            src = SCRIPT_DIR / "output" / filename
            shutil.copyfile(src, args.paper_figdir / filename)


def main():
    args = parse_args()
    unknown = sorted(set(args.targets) - set(TARGETS))
    if unknown:
        valid = ", ".join(sorted(TARGETS))
        raise SystemExit(f"Unknown figure category: {unknown[0]}. Choose from: {valid}.")
    targets = args.targets or sorted(TARGETS)
    for name in targets:
        run_target(name, args)


if __name__ == "__main__":
    main()
