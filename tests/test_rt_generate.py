"""Trace generation with Sionna RT. Needs the rt extra and a CUDA GPU; skipped otherwise."""

import copy
import hashlib
import json
from pathlib import Path

import gymnasium
import h5py
import numpy as np
import pytest

from linkgym import ENV_ID
from linkgym.channels import inspect_trace
from linkgym.rt.config import load_config, parse_config

pytestmark = pytest.mark.rt

MUNICH = load_config(Path(__file__).resolve().parents[1] / "examples" / "rt" / "munich.json")


def piece(name, start_m, length_m=2.5):
    """A straight piece of a Munich dataset route: one trajectory of 200 slots."""
    route = next(r for r in MUNICH.routes if r.name == name)
    xy = route.points(np.array([start_m, start_m + length_m]))[0]
    return xy.round(3).tolist()


# The dataset's transmitter, short trajectories of 200 slots (1.5 m) and a cheap solver
CONFIG = {
    "scene": "munich",
    "transmitter": {"position": [116.5, 80.5, 23.4]},
    "num_slots": 200,
    "solver": {"max_depth": 3, "diffraction": True},
    "routes": [
        {"name": "los", "split": "train", "waypoints": piece("north-canyon", 50.0)},
        {"name": "nlos", "split": "test", "waypoints": piece("east-avenue", 60.0)},
    ],
}


@pytest.fixture(scope="module")
def generate_module():
    try:
        import mitsuba as mi
        import sionna.rt  # noqa: F401
    except ImportError as e:
        pytest.skip(f"Sionna RT is not available: {e}")
    if not mi.variant().startswith("cuda"):
        pytest.skip(f"no CUDA GPU (Mitsuba variant {mi.variant()})")
    from linkgym.rt import generate

    return generate


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generation_is_reproducible_and_valid(generate_module, tmp_path):
    config = parse_config(CONFIG)
    first, second = tmp_path / "a.h5", tmp_path / "b.h5"
    summary = generate_module.generate(config, first, log=lambda _: None)
    generate_module.generate(config, second, log=lambda _: None)
    assert sha256(first) == sha256(second)

    info = inspect_trace(first)
    assert info.shape == (2, 200, 52)
    assert info.split.tolist() == ["train", "test"]
    assert info.group.tolist() == [0, 1]
    assert info.category.tolist() == ["los", "nlos"]
    assert summary["num_dropped"] == 0
    with h5py.File(first, "r") as f:
        assert f["num_paths"].shape == f["los"].shape == (2, 20)
        np.testing.assert_array_equal(f["los"][0], 1)
        np.testing.assert_array_equal(f["los"][1], 0)
        np.testing.assert_allclose(f["rx_position"][0, :, 2], 1.5)
        np.testing.assert_allclose(np.diff(f["rx_position"][0, :, 1]), 7.5e-3, atol=1e-5)
        attrs = dict(f.attrs)
    assert attrs["scene"] == "munich" and len(attrs["scene_sha256"]) == 64
    assert "OpenStreetMap" in attrs["attribution"]
    assert json.loads(attrs["solver"])["deterministic"] is True
    assert attrs["mitsuba_variant"].startswith("cuda")
    assert json.loads(attrs["generator_config"]) == config.to_dict()
    routes = json.loads(attrs["routes"])
    assert [r["category_source"] for r in routes] == ["los_share", "los_share"]

    # The environment reads the generated file
    for snr_mode in ("normalized", "link_budget"):
        env = gymnasium.make(
            ENV_ID,
            channel="trace",
            trace_path=first,
            episode_length=100,
            trace_splits=["train", "test"],
            snr_mode=snr_mode,
            **({"tx_power_dbm": 20.0} if snr_mode == "link_budget" else {}),
        )
        env.reset(seed=0)
        env.step(0)
        env.close()


def test_routes_inside_buildings_are_rejected(generate_module):
    data = copy.deepcopy(CONFIG)
    # Across the block east of the transmitter
    data["routes"][0]["waypoints"] = [[130.0, 120.0], [160.0, 120.0]]
    with pytest.raises(generate_module.RouteError, match="inside a building"):
        generate_module.check(parse_config(data), log=lambda _: None)


def test_trajectories_without_paths_are_dropped(generate_module, tmp_path):
    data = copy.deepcopy(CONFIG)
    # Line of sight only: the NLoS trajectory has no path at all
    data["solver"] = {"max_depth": 0, "diffraction": False}
    summary = generate_module.generate(parse_config(data), tmp_path / "t.h5", log=lambda _: None)
    assert [r["num_dropped"] for r in summary["routes"]] == [0, 1]
    assert inspect_trace(tmp_path / "t.h5").split.tolist() == ["train"]


def test_trajectories_below_the_gain_floor_are_dropped(generate_module, tmp_path):
    data = copy.deepcopy(CONFIG)
    # Between the LoS piece (about -80 dB) and the NLoS piece (about -100 dB and below)
    data["min_mean_gain_db"] = -95.0
    summary = generate_module.generate(parse_config(data), tmp_path / "t.h5", log=lambda _: None)
    los, nlos = summary["routes"]
    assert (los["num_trajectories"], los["num_dropped"]) == (1, 0)
    assert (nlos["num_dropped_low_mean_gain"], nlos["num_dropped_zero_gain"]) == (1, 0)
    assert nlos["dropped"][0]["reason"] == "low_mean_gain"
    assert nlos["dropped"][0]["mean_gain_db"] < -95.0
    with h5py.File(tmp_path / "t.h5", "r") as f:
        assert f.attrs["min_mean_gain_db"] == -95.0


def test_frequency_response_matches_sionna_cfr(generate_module):
    import mitsuba as mi

    config = parse_config(CONFIG)
    scene = generate_module.load_scene(config)
    solver = generate_module.Solver(scene)
    xy = np.asarray(CONFIG["routes"][1]["waypoints"][0])
    position = np.append(xy, generate_module.Geometry(scene).ground(xy[None])[0] + 1.5)
    solver._rx.position = mi.Point3f(*map(float, position))
    solver._rx.velocity = mi.Vector3f(15.0, 0.0, 0.0)
    paths = solver._solver(solver._scene, **solver._kwargs)
    ours = solver.frequency_response(paths, 10)
    frequencies = mi.Float(solver._frequencies.astype(np.float32))
    theirs = paths.cfr(
        frequencies,
        sampling_frequency=2000.0,
        num_time_steps=10,
        normalize_delays=False,
        out_type="numpy",
    ).reshape(10, -1)
    # Sionna computes the phases in float32
    assert np.abs(ours - theirs).max() < 1e-2 * np.abs(ours).max()
