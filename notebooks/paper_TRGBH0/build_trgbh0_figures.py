#!/usr/bin/env python
"""Build TRGBH0 paper figures by broad category."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_FIGURE_DIR = SCRIPT_DIR / "paper_figures"
TARGETS = {
    "corner": [PAPER_FIGURE_DIR / "plot_trgbh0_fiducial_corner.py"],
    "data": [
        PAPER_FIGURE_DIR / "plot_edd_trgb_magnitude_redshift_scatter.py",
        PAPER_FIGURE_DIR / "plot_edd_trgb_sky_distribution.py",
    ],
    "h0": [PAPER_FIGURE_DIR / "plot_trgbh0_h0_comparison.py"],
    "model": [PAPER_FIGURE_DIR / "render_trgb_forward_model_dag.py"],
    "student_t_beta": [PAPER_FIGURE_DIR / "plot_trgbh0_student_t_beta_h0.py"],
}
TARGET_OUTPUTS = {
    "corner": ["trgbh0_manticore_density_sigma_v_corner.pdf"],
    "data": [
        "edd_trgb_magnitude_redshift_scatter.pdf",
        "edd_trgb_sky_distribution.pdf",
    ],
    "h0": ["trgbh0_h0_comparison.pdf"],
    "model": ["trgb_forward_model_dag.pdf"],
    "student_t_beta": [
        (
            "trgbh0_student_t_beta/"
            "trgbh0_student_t_beta_free_h0_posteriors_by_lnz.pdf"
        ),
        "trgbh0_student_t_beta/trgbh0_student_t_beta_free_h0_vs_lnz.pdf",
    ],
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
    for script in TARGETS[name]:
        command = [sys.executable, str(script)]
        subprocess.run(command, check=True)
    if args.paper_figdir is not None:
        args.paper_figdir.mkdir(parents=True, exist_ok=True)
        for filename in TARGET_OUTPUTS[name]:
            src = SCRIPT_DIR / "output" / filename
            shutil.copyfile(src, args.paper_figdir / Path(filename).name)


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
