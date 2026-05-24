#!/usr/bin/env python
"""Fetch Cepheid positions from DDO database and save l, b coordinates."""
import argparse
import csv
import re
import urllib.request
from pathlib import Path

URL = "https://www.astro.utoronto.ca/DDO/research/cepheids/table_positons.html"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "MWCepheids" / "ddo_cepheid_positions.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"Fetching DDO Cepheid table from {URL}")

    with urllib.request.urlopen(URL) as response:
        html = response.read().decode("utf-8")

    # Extract content between <pre> tags
    pre_match = re.search(r"<pre>(.*?)</pre>", html, re.IGNORECASE | re.DOTALL)
    if not pre_match:
        print("ERROR: Could not find <pre> block")
        return

    pre_content = pre_match.group(1)
    lines = pre_content.strip().split("\n")

    # Skip header line(s)
    results = []
    for line in lines:
        # Skip empty lines and header
        if not line.strip() or "ID" in line and "STAR" in line:
            continue

        # Fixed-width parsing based on observed format:
        # Cols: ID (0-5), STAR (6-14), HD (15-22), SAO (23-29), RA_h (30-34),
        #       RA_m (35-40), RA_s (41-47), Dec_d (48-53), Dec_m (54-61),
        #       l (62-69), b (70+)
        try:
            # Use split and filter - columns are whitespace separated
            parts = line.split()
            if len(parts) < 10:
                continue

            # ID is first, then star name (may be 1 or 2 tokens)
            # Look for l, b as last two numeric values
            # Find where numeric coordinate data starts

            # Star name: after ID, before HD/SAO or RA
            # The ID is a float like "1.0"
            idx = 0
            if not parts[idx].replace(".", "").isdigit():
                continue
            idx += 1

            # Collect star name tokens until we hit HD/SAO or RA hours
            name_parts = []
            while idx < len(parts):
                # HD/SAO are large integers, RA_h is small (0-23)
                token = parts[idx]
                # Check if it looks like a catalog number or coordinate
                if token.isdigit() and (len(token) >= 4 or int(token) <= 23):
                    break
                name_parts.append(token)
                idx += 1

            star = " ".join(name_parts)
            if not star:
                continue

            # l and b are the last two values
            ell = float(parts[-2])
            b = float(parts[-1])

            # Normalized name for matching
            name_upper = star.upper().replace(" ", "-")
            results.append((star, ell, b, name_upper))

        except (ValueError, IndexError):
            continue

    print(f"Parsed {len(results)} Cepheids")
    print("\nSample entries:")
    for star, ell, b, name_upper in results[:10]:
        print(f"  {star:15s}  l={ell:7.2f}  b={b:7.2f}  ({name_upper})")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["STAR", "l", "b", "name_upper"])
        writer.writerows(results)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
