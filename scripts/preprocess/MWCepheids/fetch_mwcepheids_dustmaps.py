#!/usr/bin/env python3
"""Fetch dust map data to the project data directory.

Supported maps:
  - bayestar19: Green et al. (2019), ~6GB, covers Dec > -30 deg
  - marshall: Marshall et al. (2006), ~few MB,
              inner Galaxy (|ell| < 100, |b| < 10)

Usage:
    python scripts/preprocess/MWCepheids/fetch_mwcepheids_dustmaps.py bayestar19
    python scripts/preprocess/MWCepheids/fetch_mwcepheids_dustmaps.py marshall
    python scripts/preprocess/MWCepheids/fetch_mwcepheids_dustmaps.py --list
"""
import argparse
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]

SUPPORTED_MAPS = {
    "bayestar19": {
        "module": "dustmaps.bayestar",
        "description": "Green et al. (2019), ~6GB, Dec > -30 deg",
    },
    "marshall": {
        "module": "dustmaps.marshall",
        "description": ("Marshall et al. (2006), ~few MB,"
                        " |ell| < 100, |b| < 10 deg"),
    },
}


def load_local_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch dust map data to the project data directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("map_name", nargs="?", type=str,
                        help="Name of the dust map to download")
    parser.add_argument("--list", action="store_true",
                        help="List supported dust maps")
    default_config = REPO_ROOT / "local_config.toml"
    parser.add_argument("--local-config", type=str, default=default_config,
                        help="Path to local_config.toml")
    args = parser.parse_args()

    if args.list:
        print("Supported dust maps:")
        for name, info in SUPPORTED_MAPS.items():
            print(f"  {name:12s} : {info['description']}")
        return

    if args.map_name is None:
        parser.print_help()
        sys.exit(1)

    if args.map_name not in SUPPORTED_MAPS:
        print(f"Error: Unknown map '{args.map_name}'")
        print(f"Supported maps: {', '.join(SUPPORTED_MAPS.keys())}")
        sys.exit(1)

    local = load_local_config(args.local_config)
    dustmaps_dir = local.get("paths", {}).get("dustmaps")
    if dustmaps_dir is None:
        dustmaps_dir = REPO_ROOT / "data" / "MWCepheids" / "dustmaps"
    else:
        dustmaps_dir = Path(dustmaps_dir)

    dustmaps_dir.mkdir(parents=True, exist_ok=True)

    # Configure dustmaps data directory
    try:
        from dustmaps.config import config
    except ImportError as exc:
        raise ImportError(
            "Fetching MW Cepheid dust maps requires the optional `dustmaps` "
            "package."
        ) from exc
    config["data_dir"] = str(dustmaps_dir)
    config.save()

    print(f"Downloading {args.map_name} to: {dustmaps_dir}")

    # Import and fetch the requested map
    map_info = SUPPORTED_MAPS[args.map_name]
    try:
        module = __import__(map_info["module"], fromlist=["fetch"])
    except ImportError as exc:
        raise ImportError(
            "Fetching MW Cepheid dust maps requires the optional `dustmaps` "
            f"module `{map_info['module']}`."
        ) from exc
    module.fetch()

    print("Done.")


if __name__ == "__main__":
    main()
