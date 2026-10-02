"""Cut a small sample out of a generated trace: the first trajectory of some routes, shortened.

    python examples/rt/make_sample.py data/munich-v1.h5 tests/data/munich_sample.h5 \\
        --routes north-canyon east-avenue nw-avenue west-street --num-slots 1000

The sample keeps the attributes of the source file and records what was taken in the
attribute ``sample_of``.
"""

import argparse
import json

import h5py
import numpy as np

from linkgym.channels import inspect_trace, write_trace

FORMAT_ATTRS = {
    "format",
    "format_version",
    "gain_unit",
    "carrier_frequency_hz",
    "subcarrier_spacing_hz",
    "slot_duration_s",
    "num_prb",
    "prb_sampling",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--routes", nargs="+", required=True, help="route names, in order")
    parser.add_argument("--num-slots", type=int, default=1000)
    args = parser.parse_args()

    info = inspect_trace(args.source)
    with h5py.File(args.source, "r") as f:
        routes = json.loads(f.attrs["routes"])
        group_of = {r["name"]: r["group"] for r in routes}
        anchor_spacing = int(f.attrs["anchor_spacing_slots"])
        if args.num_slots % anchor_spacing:
            parser.error(f"--num-slots must be a multiple of {anchor_spacing}")
        picked = []
        for name in args.routes:
            matches = np.flatnonzero(info.group == group_of[name])
            if not len(matches):
                parser.error(f"route {name!r} has no trajectory in {args.source}")
            picked.append(int(matches[0]))
        n, a = args.num_slots, args.num_slots // anchor_spacing

        def take(name, length):
            # One trajectory at a time: h5py needs increasing indices for list selections
            return np.stack([f[name][t, :length] for t in picked])

        attrs = {k: v for k, v in f.attrs.items() if k not in FORMAT_ATTRS}
        attrs["sample_of"] = json.dumps(
            {"routes": args.routes, "trajectories": picked, "slots": [0, n]}
        )
        write_trace(
            args.output,
            take("gain", n),
            [info.split[t] for t in picked],
            carrier_frequency_hz=info.attrs["carrier_frequency_hz"],
            subcarrier_spacing_hz=info.attrs["subcarrier_spacing_hz"],
            slot_duration_s=info.attrs["slot_duration_s"],
            prb_sampling=info.attrs["prb_sampling"],
            group=info.group[picked],
            rx_position=take("rx_position", n),
            num_paths=take("num_paths", a),
            los=take("los", a),
            category=info.category[picked],
            attrs=attrs,
        )


if __name__ == "__main__":
    main()
