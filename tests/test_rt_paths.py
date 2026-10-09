"""Path traces (format 2) from Sionna RT. Needs the rt extra and a CUDA GPU; skipped otherwise."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from linkgym.channels import inspect_trace
from linkgym.paths import PathWindow, inspect_path_trace, read_path_window, reconstruct_cfr
from linkgym.rt.config import load_config, parse_config

pytestmark = pytest.mark.rt

MUNICH = load_config(Path(__file__).resolve().parents[1] / "examples" / "rt" / "munich.json")


def piece(name, start_m, length_m=2.5):
    """A straight piece of a Munich dataset route: one trajectory of 200 slots."""
    route = next(r for r in MUNICH.routes if r.name == name)
    xy = route.points(np.array([start_m, start_m + length_m]))[0]
    return xy.round(3).tolist()


# As in test_rt_generate.py: the dataset's transmitter, two short trajectories, a cheap solver
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
ORIENTATIONS = [(0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (0.6, -0.2, 0.1)]


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


@pytest.fixture(scope="module")
def traces(generate_module, tmp_path_factory):
    """The same configuration generated as format 1 and twice as format 2."""
    config = parse_config(CONFIG)
    root = tmp_path_factory.mktemp("paths")
    files = {name: root / f"{name}.h5" for name in ("gains", "paths", "again")}
    generate_module.generate(config, files["gains"], log=lambda _: None)
    for name in ("paths", "again"):
        generate_module.generate(config, files[name], store_paths=True, log=lambda _: None)
    return files


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_path_traces_are_reproducible_and_valid(traces):
    assert sha256(traces["paths"]) == sha256(traces["again"])
    info = inspect_path_trace(traces["paths"], deep=True)
    assert info.shape == (2, 200, 52)
    assert info.anchor_spacing == 10
    assert info.split.tolist() == ["train", "test"]
    assert info.category.tolist() == ["los", "nlos"]
    assert json.loads(info.attrs["tx_antenna"]) == {"pattern": "iso", "polarization": "V"}


def test_path_traces_carry_the_data_of_format_1(traces):
    reference = inspect_trace(traces["gains"])
    info = inspect_path_trace(traces["paths"])
    assert info.shape == reference.shape
    np.testing.assert_array_equal(info.split, reference.split)
    np.testing.assert_array_equal(info.group, reference.group)
    np.testing.assert_array_equal(info.category, reference.category)
    with h5py.File(traces["gains"], "r") as f, h5py.File(traces["paths"], "r") as g:
        for name in ("rx_position", "los", "num_paths"):
            np.testing.assert_array_equal(g[name][()], f[name][()])
        for key in ("scene", "scene_sha256", "solver", "routes", "generator_config", "attribution"):
            assert g.attrs[key] == f.attrs[key]
        np.testing.assert_array_equal(g.attrs["tx_orientation"], f.attrs["tx_orientation"])
        assert json.loads(g.attrs["rx_antenna"]) == json.loads(f.attrs["rx_antenna"])
        # The mean gain that format 1's normalized SNR mode divides by
        for t in range(info.num_trajectories):
            assert info.mean_gain[t] == f["gain"][t].mean(dtype=np.float64)


def test_single_antenna_reconstruction_matches_format_1(traces):
    with h5py.File(traces["gains"], "r") as f:
        gains = f["gain"][()]
    for t in range(len(gains)):
        window = read_path_window(traces["paths"], t, 0, 200)
        h = reconstruct_cfr(window, dtype=torch.complex128)[..., 0].numpy()
        gain = (np.abs(h) ** 2).astype(np.float32)
        # The generator sums the paths in the same order, in float64 too
        np.testing.assert_allclose(gain, gains[t], rtol=1e-5, atol=1e-6 * gains[t].mean())
        # Windows that start between anchors give the same slots
        part = reconstruct_cfr(read_path_window(traces["paths"], t, 37, 50), dtype=torch.complex128)
        np.testing.assert_allclose(part[..., 0].numpy(), h[37:87], rtol=0, atol=1e-12)


def window_of(paths, carrier_frequency, frequencies):
    """A one-anchor, one-slot window of the sorted paths of a solve."""
    return PathWindow(
        a=paths["a"],
        tau=paths["tau"],
        doppler=paths["doppler"],
        k_tx=paths["k_tx"],
        counts=np.array([len(paths["tau"])]),
        steps=np.array([[0, 1]]),
        carrier_frequency=carrier_frequency,
        frequencies=frequencies,
        slot_duration=1e-3,
    )


@pytest.mark.parametrize("num_rows, num_cols", [(4, 8), (1, 32)])
def test_reconstruction_matches_sionna_synthetic_arrays(generate_module, num_rows, num_cols):
    import mitsuba as mi
    import sionna.rt as rt

    config = parse_config(CONFIG)
    scene = generate_module.load_scene(config)
    solver = generate_module.Solver(scene)
    geometry = generate_module.Geometry(scene)
    fc = config.carrier_frequency_hz
    frequencies = solver._frequencies
    single = scene.rt_scene.tx_array
    array = rt.PlanarArray(
        num_rows=num_rows,
        num_cols=num_cols,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    for route in CONFIG["routes"]:
        xy = np.asarray(route["waypoints"][0])
        position = np.append(xy, geometry.ground(xy[None])[0] + 1.5)
        solver._rx.position = mi.Point3f(*map(float, position))
        solver._rx.velocity = mi.Vector3f(10.0, 0.0, 0.0)
        for orientation in ORIENTATIONS:
            scene.tx.orientation = mi.Point3f(*orientation)
            # linkgym: a single antenna traced once, the array rebuilt from the stored paths
            scene.rt_scene.tx_array = single
            paths = solver.sorted_paths(solver._solver(scene.rt_scene, **solver._kwargs))
            ours = reconstruct_cfr(
                window_of(paths, fc, frequencies),
                num_rows=num_rows,
                num_cols=num_cols,
                orientation=orientation,
                dtype=torch.complex128,
            )[0].numpy()  # [F, N]

            # Sionna RT: the synthetic array traced directly
            scene.rt_scene.tx_array = array
            theirs = solver._solver(scene.rt_scene, **solver._kwargs)
            valid = np.asarray(theirs.valid.numpy()).reshape(-1).astype(bool)
            tau = np.asarray(theirs.tau.numpy(), dtype=np.float64).reshape(-1)[valid]
            assert np.array_equal(np.sort(tau), np.sort(paths["tau"].astype(np.float64)))
            re, im = (np.asarray(x.numpy(), dtype=np.float64) for x in theirs.a)
            a = (re + 1j * im).reshape(num_rows * num_cols, -1)[:, valid]  # [N, P]
            phase = np.exp(-2j * np.pi * (fc + frequencies[:, None]) * tau[None, :])  # [F, P]
            # Sionna's per-antenna coefficients, summed in float64
            expected = phase @ a.T
            scale = np.abs(expected).max()
            assert np.abs(ours - expected).max() < 1e-4 * scale, (route["name"], orientation)

            # And Sionna's own frequency response, which it computes in float32
            cfr = theirs.cfr(
                mi.Float(frequencies.astype(np.float32)),
                sampling_frequency=1000.0,
                num_time_steps=1,
                normalize_delays=False,
                out_type="numpy",
            ).reshape(num_rows * num_cols, -1)
            assert np.abs(ours - cfr.T).max() < 1e-2 * scale, (route["name"], orientation)


def test_missing_direction_vectors_are_reported(generate_module):
    with pytest.raises(RuntimeError, match="no _k_tx.*sionna-rt 2.1.0"):
        generate_module._direction_vectors(SimpleNamespace(), 4)
    wrong = SimpleNamespace(
        _k_tx=SimpleNamespace(numpy=lambda: np.zeros((1, 1, 4, 2))),
        _k_rx=SimpleNamespace(numpy=lambda: np.zeros((1, 1, 4, 3))),
    )
    with pytest.raises(RuntimeError, match=r"_k_tx has shape \(1, 1, 4, 2\)"):
        generate_module._direction_vectors(wrong, 4)
