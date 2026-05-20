#!/usr/bin/env python3
"""Download native Manticore chain files from the public data mirror.

This is a standalone helper for the BORG field scripts. It deliberately does
not import CANDEL or use the CANDEL Python environment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from borg_field_config import configured_path

GENERATION = "2MPP_MULTIBIN_N256_DES_V2"
KEY_URL = "https://manticore.web.data-2-osu.iap.fr/public-keys.json"
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_OUTPUT_DIR = configured_path("borg_run_dir")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation",
        default=GENERATION,
        help=f"Manticore generation key. Default: {GENERATION}",
    )
    parser.add_argument(
        "--subchain",
        help="Manual mode: Manticore chain subchain.",
    )
    parser.add_argument(
        "--mcmc",
        type=int,
        help="Manual mode: MCMC sample index, e.g. 3500.",
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        help="Schedule YAML. Default: OUTPUT_DIR/schedule_final.yaml.",
    )
    parser.add_argument(
        "--steps",
        default="0:50",
        help=(
            "Schedule steps to download, inclusive. Accepts entries like "
            "'0:50', '0-50', '0,5,10:12'. Default: 0:50."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without contacting S3.",
    )
    parser.add_argument(
        "--access-json",
        type=Path,
        help=(
            "Local copy of public-keys.json. Use this when the key server "
            "is not reachable from the current machine."
        ),
    )
    parser.add_argument(
        "--save-access-json",
        type=Path,
        help="Write public-keys.json here after fetching it successfully.",
    )
    return parser.parse_args()


def require_boto3():
    try:
        import boto3
        from botocore.client import Config
    except ImportError as exc:
        raise SystemExit(
            "This script requires boto3 in the BORG environment. "
            f"Install with: {configured_path('borg_python')} -m pip install boto3"
        ) from exc
    return boto3, Config


def get_access_data(
    generation: str,
    access_json: Path | None = None,
    save_access_json: Path | None = None,
) -> dict:
    if access_json is not None:
        path = access_json.expanduser().resolve()
        with path.open("r") as handle:
            keys = json.load(handle)
    else:
        try:
            with urlopen(KEY_URL, timeout=30) as response:
                keys = json.load(response)
        except (OSError, URLError, TimeoutError) as exc:
            raise SystemExit(
                "Could not reach the Manticore key server:\n"
                f"  {KEY_URL}\n"
                f"  {exc}\n\n"
                "`--locally` only avoids queue submission; downloads still "
                "need network access. If this machine cannot reach the key "
                "server, fetch public-keys.json elsewhere and rerun with:\n"
                "  --access-json /path/to/public-keys.json"
            ) from exc

        if save_access_json is not None:
            path = save_access_json.expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as handle:
                json.dump(keys, handle)

    return keys["manticores"][generation]


def parse_steps(spec: str) -> list[int]:
    steps: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start, stop = item.split(":", 1)
        elif "-" in item:
            start, stop = item.split("-", 1)
        else:
            steps.add(int(item))
            continue

        start_i = int(start)
        stop_i = int(stop)
        if stop_i < start_i:
            raise ValueError(f"Invalid descending step range: {item}")
        steps.update(range(start_i, stop_i + 1))

    if not steps:
        raise ValueError("No schedule steps were requested.")
    return sorted(steps)


def read_schedule(path: Path) -> dict[int, tuple[str, int]]:
    """Read the simple Manticore schedule YAML without adding a dependency."""

    step_re = re.compile(r"^(\d+):\s*$")
    subchain_re = re.compile(r"^\s{2}([^:\s]+):\s*$")
    mcmc_re = re.compile(r"^\s{4}mcmc_step:\s*(\d+)\s*$")

    schedule: dict[int, tuple[str, int]] = {}
    current_step: int | None = None
    current_subchain: str | None = None

    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue

        match = step_re.match(raw_line)
        if match:
            current_step = int(match.group(1))
            current_subchain = None
            continue

        match = subchain_re.match(raw_line)
        if match and current_step is not None:
            current_subchain = match.group(1)
            continue

        match = mcmc_re.match(raw_line)
        if match and current_step is not None and current_subchain is not None:
            schedule[current_step] = (current_subchain, int(match.group(1)))
            continue

        raise ValueError(f"Could not parse {path}:{line_no}: {raw_line!r}")

    return schedule


class Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.seen = 0
        self.last_percent = -1

    def __call__(self, chunk: int) -> None:
        self.seen += chunk
        percent = int(100 * self.seen / self.total)
        if percent != self.last_percent and percent % 5 == 0:
            print(
                f"downloaded {percent:3d}% ({self.seen / 1024**2:.1f} MiB)",
                flush=True,
            )
            self.last_percent = percent


def planned_downloads(args: argparse.Namespace) -> list[tuple[int | None, str, int]]:
    manual_args = args.subchain is not None or args.mcmc is not None
    if manual_args:
        if args.subchain is None or args.mcmc is None:
            raise SystemExit("Manual mode requires both --subchain and --mcmc.")
        return [(None, args.subchain, args.mcmc)]

    output_dir = args.output_dir.expanduser().resolve()
    schedule_path = (
        args.schedule.expanduser().resolve()
        if args.schedule is not None
        else output_dir / "schedule_final.yaml"
    )
    schedule = read_schedule(schedule_path)
    steps = parse_steps(args.steps)

    missing = [step for step in steps if step not in schedule]
    if missing:
        raise SystemExit(
            f"Schedule is missing requested steps: {missing}. "
            f"Schedule: {schedule_path}"
        )

    return [(step, *schedule[step]) for step in steps]


def output_path(output_dir: Path, subchain: str, mcmc: int) -> Path:
    return output_dir / "chain" / subchain / f"mcmc_{mcmc}.h5"


def download_one(
    s3,
    bucket: str,
    subchain: str,
    mcmc: int,
    output: Path,
    overwrite: bool,
) -> None:
    key = f"chain/{subchain}/mcmc/mcmc_{mcmc}.h5"
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".part")

    if output.exists() and not overwrite:
        print(f"exists, skipping: {output}")
        return

    response = s3.get_object(Bucket=bucket, Key=key)
    total = int(response["ContentLength"])
    print(f"downloading s3://{bucket}/{key}")
    print(f"target: {output}")
    print(f"size: {total / 1024**2:.1f} MiB")

    if tmp_output.exists():
        tmp_output.unlink()
    progress = Progress(total)
    with tmp_output.open("wb") as handle:
        for chunk in response["Body"].iter_chunks(chunk_size=8 * 1024**2):
            if not chunk:
                continue
            handle.write(chunk)
            progress(len(chunk))
    tmp_output.replace(output)
    print(f"done: {output}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    targets = planned_downloads(args)
    print(f"planned downloads: {len(targets)}")
    for step, subchain, mcmc in targets:
        prefix = f"step {step}: " if step is not None else ""
        print(f"{prefix}{subchain} mcmc_{mcmc} -> {output_path(output_dir, subchain, mcmc)}")

    if args.dry_run:
        return 0

    download_targets = []
    for target in targets:
        _, subchain, mcmc = target
        output = output_path(output_dir, subchain, mcmc)
        if output.exists() and not args.overwrite:
            print(f"exists, skipping: {output}")
            continue
        download_targets.append(target)
    if not download_targets:
        return 0

    boto3, Config = require_boto3()
    access = get_access_data(
        args.generation,
        access_json=args.access_json,
        save_access_json=args.save_access_json,
    )
    s3 = boto3.client(
        "s3",
        endpoint_url=access["url"],
        aws_access_key_id=access["access_key"],
        aws_secret_access_key=access["secret_key"],
        config=Config(signature_version="s3v4"),
    )

    for _, subchain, mcmc in download_targets:
        download_one(
            s3,
            access["bucket"],
            subchain,
            mcmc,
            output_path(output_dir, subchain, mcmc),
            args.overwrite,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
