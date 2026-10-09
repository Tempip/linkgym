"""Make a small path trace (format 2) with the trajectories of a format-1 sample.

    python examples/rt/make_paths_sample.py examples/rt/munich.json \\
        tests/data/munich_sample.h5 tests/data/munich_paths_sample.h5 --num-slots 100

Solves again, with the generator's configuration, the first ``--num-slots`` slots of every
trajectory of the format-1 sample (made by make_sample.py), keeps their paths, and checks
that the gains, path counts and LoS flags equal those of the sample. Needs the rt extra and
a CUDA GPU. The output keeps the attributes of the sample and records what was taken in the
attribute ``sample_of``.
"""

import argparse
import json
from dataclasses import asdict

import h5py
import numpy as np

from linkgym.channels import inspect_trace
from linkgym.paths import _RESERVED, write_path_trace
from linkgym.rt import generate
from linkgym.rt.config import load_config

FORMAT_1_ATTRS = {"format", "format_version", "gain_unit", "prb_sampling", "sample_of"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="generator configuration of the sample's source")
    parser.add_argument("sample", help="format-1 sample (make_sample.py)")
    parser.add_argument("output")
    parser.add_argument("--num-slots", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    info = inspect_trace(args.sample)
    with h5py.File(args.sample, "r") as f:
        source = json.loads(f.attrs["generator_config"])
        if source != config.to_dict():
            parser.error(f"{args.config} is not the configuration {args.sample} was made with")
        sample_of = json.loads(f.attrs["sample_of"])
        attrs = {k: v for k, v in f.attrs.items() if k not in FORMAT_1_ATTRS | _RESERVED}
        reference = {name: f[name][()] for name in ("gain", "rx_position", "num_paths", "los")}
    k = config.anchor_spacing_slots
    n = args.num_slots
    if n % k:
        parser.error(f"--num-slots must be a multiple of {k}")

    scene = generate.load_scene(config)
    geometry = generate.Geometry(scene)
    solver = generate.Solver(scene)
    routes = {route.name: route for route in config.routes}
    trajectories, gains, columns = [], [], []
    for i, name in enumerate(sample_of["routes"]):
        layout = generate.layout_route(config, geometry, routes[name])
        # The trajectory of the route that starts where the sample's does
        start = reference["rx_position"][i, 0]
        j = int(np.flatnonzero(np.all(layout.positions[:, 0].astype(np.float32) == start, 1))[0])
        gain, num_paths, los, paths = solver.solve_slots_with_paths(
            layout.positions[j, :n], layout.directions[j, :n], k
        )
        gain = gain.astype(np.float32)
        np.testing.assert_array_equal(num_paths, reference["num_paths"][i, : n // k])
        np.testing.assert_array_equal(los, reference["los"][i, : n // k])
        np.testing.assert_array_equal(gain, reference["gain"][i, :n])
        trajectories.append(j)
        gains.append(gain)
        columns.append(paths)
        print(f"{name}: trajectory {j}, {num_paths.sum()} paths", flush=True)

    counts = np.array([[len(p["tau"]) for p in anchors] for anchors in columns])
    offsets = np.zeros((len(counts), counts.shape[1] + 1), dtype=np.int64)
    offsets[:, 1:] = np.cumsum(counts.ravel()).reshape(counts.shape)
    offsets[1:, 0] = offsets[:-1, -1]
    attrs["mitsuba_variant"] = generate.import_rt()[1].variant()
    attrs["versions"] = json.dumps(generate._versions(), sort_keys=True)
    attrs["sample_of"] = json.dumps(
        {"routes": sample_of["routes"], "route_trajectories": trajectories, "slots": [0, n]}
    )
    write_path_trace(
        args.output,
        split=info.split,
        path_offset=offsets,
        **{
            key: np.concatenate([p[key] for anchors in columns for p in anchors])
            for key in ("a", "tau", "doppler", "k_tx", "k_rx")
        },
        carrier_frequency_hz=config.carrier_frequency_hz,
        subcarrier_spacing_hz=config.subcarrier_spacing_hz,
        slot_duration_s=config.slot_duration_s,
        num_prb=config.num_prb,
        num_slots=n,
        anchor_spacing_slots=k,
        tx_antenna=asdict(config.tx_antenna),
        rx_antenna=asdict(config.rx_antenna),
        tx_orientation=np.asarray(scene.tx.orientation, dtype=np.float64).reshape(-1)[:3],
        mean_gain=np.array([g.mean(dtype=np.float64) for g in gains]),
        group=info.group,
        category=info.category,
        rx_position=reference["rx_position"][:, :n],
        los=reference["los"][:, : n // k],
        attrs=attrs,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
