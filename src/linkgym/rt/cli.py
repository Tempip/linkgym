"""``linkgym-traces``: generate channel traces with Sionna RT (also ``python -m linkgym.rt``)."""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys

from linkgym.rt.config import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="linkgym-traces",
        description=(
            "Generate linkgym channel traces (HDF5, format version 1) from a Sionna RT scene. "
            'Needs the rt extra: pip install "linkgym[rt]", in its own environment.'
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("generate", help="solve paths along all routes and write a trace")
    p.add_argument("config", help="generator configuration (JSON)")
    p.add_argument("-o", "--output", required=True, help="trace file to write (.h5)")
    p.add_argument("--routes", nargs="+", metavar="NAME", help="only these routes")
    p.add_argument("--splits", nargs="+", metavar="SPLIT", help="only the routes of these splits")
    p.add_argument(
        "--solver",
        nargs="+",
        metavar="KEY=VALUE",
        default=[],
        help="override solver settings, e.g. samples_per_src=8000000 (recorded in the file)",
    )
    p.add_argument(
        "--max-trajectories", type=int, metavar="N", help="at most N trajectories per route"
    )
    p.add_argument(
        "--store-paths",
        action="store_true",
        help="write the paths of every anchor (trace format 2, for multi-antenna "
        "reconstruction) instead of per-PRB gains",
    )
    p.add_argument("--overwrite", action="store_true", help="replace an existing output file")

    p = commands.add_parser("check", help="check the routes against the scene, no path solves")
    p.add_argument("config", help="generator configuration (JSON)")
    p.add_argument("--json", metavar="PATH", help="also write the report as JSON")

    p = commands.add_parser(
        "accuracy", help="compare anchored channels with a path solve at every slot"
    )
    p.add_argument("config", help="generator configuration (JSON)")
    p.add_argument("--route", required=True, help="route to measure on")
    p.add_argument("--start-m", type=float, default=0.0, help="start of the stretch [m]")
    p.add_argument("--num-slots", type=int, default=1000, help="length of the stretch [slots]")
    p.add_argument(
        "--spacings", type=int, nargs="+", default=[5, 10, 20, 50], help="anchor spacings [slots]"
    )
    p.add_argument("--json", metavar="PATH", help="also write the result as JSON")

    p = commands.add_parser("plot", help="top view of the scene with the routes by split")
    p.add_argument("config", help="generator configuration (JSON)")
    p.add_argument("-o", "--output", required=True, help="image to write (.png)")
    p.add_argument("--margin", type=float, default=40.0, help="margin around the routes [m]")
    p.add_argument("--resolution", type=float, default=1.0, help="height map resolution [m]")
    p.add_argument("--labels", action="store_true", help="write the route names")

    args = parser.parse_args(argv)
    log = functools.partial(print, flush=True)  # progress also shows in redirected logs
    try:
        config = load_config(args.config)
        if args.command == "generate":
            if os.path.exists(args.output) and not args.overwrite:
                parser.error(f"{args.output} exists; pass --overwrite to replace it")
            from linkgym.rt.generate import generate

            config = config.select(args.routes).select_splits(args.splits)
            config = config.with_solver(_solver_overrides(args.solver))
            generate(
                config,
                args.output,
                max_trajectories=args.max_trajectories,
                store_paths=args.store_paths,
                log=log,
            )
        elif args.command == "check":
            from linkgym.rt.generate import check

            _dump(check(config, log=log), args.json)
        elif args.command == "accuracy":
            from linkgym.rt.generate import accuracy

            result = accuracy(
                config,
                args.route,
                start_m=args.start_m,
                num_slots=args.num_slots,
                spacings=tuple(args.spacings),
                log=log,
            )
            _dump(result, args.json)
        else:
            from linkgym.rt.generate import plot

            plot(
                config,
                args.output,
                margin_m=args.margin,
                resolution_m=args.resolution,
                labels=args.labels,
            )
    except (ConfigError, ImportError, ValueError) as e:
        print(f"linkgym-traces: error: {e}", file=sys.stderr)
        return 1
    return 0


def _solver_overrides(items: list[str]) -> dict:
    """Parse KEY=VALUE solver overrides; values are JSON (numbers, true/false)."""
    overrides = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ConfigError(f"solver override {item!r} must be KEY=VALUE")
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError as e:
            raise ConfigError(f"solver override {item!r}: value must be JSON ({e})") from e
    return overrides


def _dump(result: dict, path: str | None) -> None:
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
