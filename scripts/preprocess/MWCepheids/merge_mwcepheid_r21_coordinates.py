#!/usr/bin/env python
"""Merge R21 Cepheid data with DDO Galactic coordinates."""
import argparse
import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "MWCepheids"


def normalize_name(name):
    """Normalize Cepheid name for matching."""
    # Remove leading zeros: V0386 -> V386
    name = re.sub(r"V-?0+(\d)", r"V\1", name)
    # Remove hyphen after V: V-339 -> V339
    name = re.sub(r"V-(\d)", r"V\1", name)
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ddo-file", type=Path,
        default=DATA_DIR / "ddo_cepheid_positions.csv")
    parser.add_argument(
        "--r21-file", type=Path,
        default=DATA_DIR / "Riess2021_Table1.csv")
    parser.add_argument(
        "--output", type=Path,
        default=DATA_DIR / "Riess2021_Table1_with_coords.csv")
    args = parser.parse_args()

    # Load DDO positions with multiple name variants
    ddo = {}
    with open(args.ddo_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name_upper"]
            coords = (float(row["l"]), float(row["b"]))
            ddo[name] = coords
            ddo[normalize_name(name)] = coords

    # Load R21 and merge
    with open(args.r21_file) as f:
        reader = csv.DictReader(f)
        r21_rows = list(reader)
        fieldnames = reader.fieldnames + ["ell", "b"]

    matched = 0
    for row in r21_rows:
        name = row["Cepheid"]
        norm_name = normalize_name(name)

        if name in ddo:
            ell, b = ddo[name]
        elif norm_name in ddo:
            ell, b = ddo[norm_name]
        else:
            print(f"WARNING: No match for {name}")
            ell, b = "", ""

        row["ell"] = ell
        row["b"] = b
        if ell != "":
            matched += 1

    print(f"Matched {matched}/{len(r21_rows)} stars")

    # Write merged file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(r21_rows)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
