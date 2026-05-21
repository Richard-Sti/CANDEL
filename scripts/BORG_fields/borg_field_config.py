#!/usr/bin/env python3
"""Read BORG field helper and active-chain settings from local_config.toml."""

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
CHAIN_SECTION = "chains"
ENV_NAMES = {
    "borg_python": "BORG_PYTHON",
    "borg_forward": "BORG_FORWARD",
    "cosmotool_sph": "COSMOTOOL_SPH",
    "plot_python": "PYTHON_PATH",
    "srun": "SRUN",
}
CHAIN_ENV_NAMES = {
    "chain_name": "BORG_CHAIN_NAME",
    "run_dir": "BORG_RUN_DIR",
    "schedule": "BORG_SCHEDULE",
    "field_output_dir": "BORG_FIELD_OUTPUT_DIR",
    "download_generation": "BORG_DOWNLOAD_GENERATION",
    "reference_fields_dir": "BORG_REFERENCE_FIELDS_DIR",
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
    section = borg_fields_section(config, path)

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


def borg_fields_section(config: dict, path: Path) -> dict:
    section = config.get(SECTION)
    if not isinstance(section, dict):
        raise LocalConfigError(
            f"Missing [{SECTION}] in {path}. See {EXAMPLE_CONFIG}."
        )
    return section


def active_chain_name(section: dict, path: Path) -> str:
    name = section.get("active_chain")
    if not isinstance(name, str) or not name.strip():
        raise LocalConfigError(
            f"Missing required [{SECTION}] active_chain in {path}. "
            f"See {EXAMPLE_CONFIG}."
        )
    return name


def borg_chain_config(
        required: tuple[str, ...] = ("run_dir",),
        chain_name: str | None = None) -> dict[str, str]:
    path = local_config_path()
    config = read_local_config(path)
    section = borg_fields_section(config, path)
    if chain_name is None:
        chain_name = active_chain_name(section, path)
    chains = section.get(CHAIN_SECTION)
    if not isinstance(chains, dict):
        raise LocalConfigError(
            f"Missing [{SECTION}.{CHAIN_SECTION}] in {path}. "
            f"See {EXAMPLE_CONFIG}."
        )
    chain = chains.get(chain_name)
    if not isinstance(chain, dict):
        raise LocalConfigError(
            f"Missing [{SECTION}.{CHAIN_SECTION}.{chain_name}] in {path}. "
            f"See {EXAMPLE_CONFIG}."
        )

    missing = [
        key for key in required
        if not isinstance(chain.get(key), str) or not chain[key].strip()
    ]
    if missing:
        joined = "\n  ".join(missing)
        raise LocalConfigError(
            f"Missing required [{SECTION}.{CHAIN_SECTION}.{chain_name}] "
            f"local_config.toml keys in {path}:\n"
            f"  {joined}\n"
            f"See {EXAMPLE_CONFIG}."
        )
    values = {"chain_name": chain_name}
    values.update({key: chain[key] for key in required})
    return values


def configured_path(name: str) -> Path:
    return Path(borg_field_config((name,))[name]).expanduser()


def configured_chain_value(name: str) -> str:
    return borg_chain_config((name,))[name]


def configured_chain_path(name: str) -> Path:
    return Path(configured_chain_value(name)).expanduser()


def shell_exports(keys: tuple[str, ...]) -> str:
    tool_keys = tuple(key for key in keys if key in ENV_NAMES)
    chain_keys = tuple(key for key in keys if key in CHAIN_ENV_NAMES)
    values: dict[str, str] = {}
    if tool_keys:
        values.update(borg_field_config(tool_keys))
    if chain_keys:
        required = tuple(key for key in chain_keys if key != "chain_name")
        values.update(borg_chain_config(required))
    lines = []
    for key in keys:
        env_name = ENV_NAMES[key] if key in ENV_NAMES else CHAIN_ENV_NAMES[key]
        lines.append(f"{env_name}={shlex.quote(values[key])}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shell-env",
        nargs="+",
        choices=tuple(ENV_NAMES) + tuple(CHAIN_ENV_NAMES),
        help=(
            "Print shell assignments for the requested [borg_fields] tool "
            "or active-chain keys."
        ),
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
