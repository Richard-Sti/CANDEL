#!/usr/bin/env python3
"""Read BORG field helper paths from CANDEL local_config.toml."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG_ENV = "CANDEL_LOCAL_CONFIG"
LOCAL_CONFIG = REPO_ROOT / "local_config.toml"
EXAMPLE_CONFIG = REPO_ROOT / "example_local_config.toml"
SECTION = "borg_fields"
ENV_NAMES = {
    "borg_python": "BORG_PYTHON",
    "borg_forward": "BORG_FORWARD",
    "cosmotool_sph": "COSMOTOOL_SPH",
    "plot_python": "PYTHON_PATH",
    "srun": "SRUN",
    "borg_run_dir": "BORG_RUN_DIR",
}


class LocalConfigError(RuntimeError):
    """Raised when local_config.toml is missing required BORG field paths."""


def local_config_path() -> Path:
    return Path(os.environ.get(LOCAL_CONFIG_ENV, LOCAL_CONFIG)).expanduser()


def read_local_config(path: Path | None = None) -> dict:
    path = local_config_path() if path is None else path.expanduser()
    if not path.is_file():
        raise LocalConfigError(
            f"Missing {path}. Create it from {EXAMPLE_CONFIG} and fill "
            f"the [{SECTION}] paths."
        )
    with path.open("rb") as handle:
        return tomllib.load(handle)


def borg_field_config(required: tuple[str, ...] = tuple(ENV_NAMES)) -> dict[str, str]:
    path = local_config_path()
    config = read_local_config(path)
    section = config.get(SECTION)
    if not isinstance(section, dict):
        raise LocalConfigError(
            f"Missing [{SECTION}] in {path}. See {EXAMPLE_CONFIG}."
        )

    missing = [
        key for key in required
        if not isinstance(section.get(key), str) or not section[key].strip()
    ]
    if missing:
        joined = "\n  ".join(missing)
        raise LocalConfigError(
            f"Missing required [{SECTION}] local_config.toml keys in {path}:\n"
            f"  {joined}\n"
            f"See {EXAMPLE_CONFIG}."
        )
    return {key: section[key] for key in required}


def configured_path(name: str) -> Path:
    return Path(borg_field_config((name,))[name]).expanduser()


def shell_exports(keys: tuple[str, ...]) -> str:
    values = borg_field_config(keys)
    return "\n".join(
        f"{ENV_NAMES[key]}={shlex.quote(values[key])}"
        for key in keys
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shell-env",
        nargs="+",
        choices=tuple(ENV_NAMES),
        help="Print shell assignments for the requested [borg_fields] keys.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.shell_env:
        print(shell_exports(tuple(args.shell_env)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LocalConfigError as exc:
        raise SystemExit(str(exc)) from exc
