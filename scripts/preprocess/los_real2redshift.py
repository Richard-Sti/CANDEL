"""
Add redshift grid (z) to real-space LOS files.

This script converts the r grid (real-space comoving distance in Mpc) to
redshift z using Distance2Redshift. This is the same transformation used
in make_2mpp_zspace.py when creating the 2mpp_zspace LOS files.

The inverse mapping (r -> z) matches what was done in make_2mpp_zspace.py:
    dist2redshift = candel.Distance2Redshift(Om0=0.3)
    z = dist2redshift(r, h=H0_base / 100.0)
"""
from __future__ import annotations

import argparse
import os
import shutil
import numpy as np
from h5py import File

import candel


def add_redshift_to_los(
    input_path: str,
    output_path: str | None = None,
    Om0: float = 0.3,
    H0_base: float = 100.0,
    overwrite: bool = False,
):
    """
    Add redshift array z to a real-space LOS file.

    Parameters
    ----------
    input_path : str
        Path to input LOS HDF5 file (must contain 'r' dataset).
    output_path : str or None
        Path to output LOS HDF5 file. If None, modifies input in place.
    Om0 : float
        Matter density parameter for cosmology.
    H0_base : float
        Hubble constant in km/s/Mpc (used as h = H0_base / 100).
    overwrite : bool
        If True, overwrite existing 'z' dataset.
    """
    if output_path is None:
        output_path = input_path

    # If output != input, copy file first
    if output_path != input_path:
        shutil.copy2(input_path, output_path)

    with File(output_path, "r+") as f:
        if "r" not in f:
            raise ValueError(f"Input file {input_path} does not contain 'r' dataset.")

        r = f["r"][...]

        # Check if z already exists
        if "z" in f:
            if not overwrite:
                print(f"'z' dataset already exists in {output_path}. Use --overwrite to replace.")
                return
            del f["z"]

        # Compute redshift from real-space distance
        dist2redshift = candel.Distance2Redshift(Om0=Om0)
        h = H0_base / 100.0
        z = np.asarray(dist2redshift(r, h=h), dtype=np.float32)

        # Save z dataset
        f.create_dataset("z", data=z)

        # Update attributes
        f.attrs["Om0"] = Om0
        f.attrs["H0_base"] = H0_base

    print(f"Added 'z' dataset to {output_path}")
    print(f"  r range: {r[0]:.4f} to {r[-1]:.4f} Mpc")
    print(f"  z range: {z[0]:.6f} to {z[-1]:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Add redshift grid (z) to real-space LOS files."
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="+",
        help="Input LOS HDF5 file(s) to process.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (only valid with single input file). If not specified, modifies in place.",
    )
    parser.add_argument(
        "--Om0",
        type=float,
        default=0.3,
        help="Matter density parameter (default: 0.3).",
    )
    parser.add_argument(
        "--H0-base",
        type=float,
        default=100.0,
        help="Hubble constant in km/s/Mpc (default: 100.0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing 'z' dataset if present.",
    )
    args = parser.parse_args()

    if args.output is not None and len(args.input) > 1:
        raise ValueError("--output can only be used with a single input file.")

    for input_path in args.input:
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} does not exist, skipping.")
            continue

        output_path = args.output if len(args.input) == 1 else None
        add_redshift_to_los(
            input_path=input_path,
            output_path=output_path,
            Om0=args.Om0,
            H0_base=args.H0_base,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
