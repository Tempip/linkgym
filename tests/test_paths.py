"""Trace format 2 (paths): writer, reader, validation, rotation and reconstruction (CPU)."""

import math

import h5py
import numpy as np
import pytest
import torch

from linkgym import beams
from linkgym.channels import TraceFormatError
from linkgym.paths import (
    PATH_TRACE_FORMAT_VERSION,
    inspect_path_trace,
    read_path_window,
    reconstruct_cfr,
    rotation_matrix,
    write_path_trace,
)

GRID = {
    "carrier_frequency_hz": 3.5e9,
    "subcarrier_spacing_hz": 30e3,
    "slot_duration_s": 0.5e-3,
    "num_prb": 52,
}


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def random_paths(num_traj=3, num_slots=40, spacing=10, seed=0, max_paths=6):
    rng = np.random.default_rng(seed)
    anchors = num_slots // spacing
    counts = rng.integers(1, max_paths, size=(num_traj, anchors))
    flat = np.concatenate([[0], np.cumsum(counts.ravel())])
    offsets = np.stack([flat[t * anchors : t * anchors + anchors + 1] for t in range(num_traj)])
    p = int(flat[-1])
    return {
        "path_offset": offsets,
        "a": (rng.normal(size=p) + 1j * rng.normal(size=p)) * 1e-5,
        "tau": rng.uniform(3e-7, 2e-6, p),
        "doppler": rng.uniform(-150, 150, p),
        "k_tx": unit(rng.normal(size=(p, 3))),
        "k_rx": unit(rng.normal(size=(p, 3))),
        "num_slots": num_slots,
        "anchor_spacing_slots": spacing,
        "split": ["train", "val", "test"][:num_traj] + ["train"] * max(0, num_traj - 3),
    }


def write(path, paths, **kwargs):
    write_path_trace(path, **GRID, **paths, **kwargs)
    return path


@pytest.fixture
def trace(tmp_path):
    paths = random_paths()
    extra = {
        "group": np.arange(3),
        "category": ["los", "nlos", "transition"],
        "rx_position": np.zeros((3, 40, 3)),
        "los": np.zeros((3, 4)),
        "attrs": {"scene": "test"},
    }
    return write(tmp_path / "paths.h5", paths, **extra), paths


def reference_cfr(paths, trajectory, offset, num_slots):
    """1x1 channel from the generator's formula (linkgym.rt.generate.Solver), in numpy."""
    spacing = paths["anchor_spacing_slots"]
    scs, nprb = GRID["subcarrier_spacing_hz"], GRID["num_prb"]
    freqs = ((np.arange(12 * nprb) - 6 * nprb) * scs).reshape(nprb, 12).mean(axis=1)
    out = []
    for slot in range(offset, offset + num_slots):
        j, n = divmod(slot, spacing)
        b, e = paths["path_offset"][trajectory, j], paths["path_offset"][trajectory, j + 1]
        a = paths["a"][b:e].astype(np.complex64).astype(np.complex128)
        tau = paths["tau"][b:e].astype(np.float32).astype(np.float64)
        fd = paths["doppler"][b:e].astype(np.float32).astype(np.float64)
        a = a * np.exp(-2j * np.pi * GRID["carrier_frequency_hz"] * tau)
        delay = np.exp(-2j * np.pi * freqs[:, None] * tau[None, :])
        t = n * GRID["slot_duration_s"]
        out.append((delay * (a * np.exp(2j * np.pi * fd * t))[None, :]).sum(axis=1))
    return np.stack(out)


def test_round_trip_and_window(trace):
    path, paths = trace
    info = inspect_path_trace(path)
    assert (info.num_trajectories, info.num_slots, info.num_prbs) == (3, 40, 52)
    assert info.shape == (3, 40, 52)
    assert info.anchor_spacing == 10 and info.num_anchors == 4
    assert info.num_paths == paths["path_offset"][-1, -1]
    assert list(info.split) == ["train", "val", "test"]
    assert list(info.category) == ["los", "nlos", "transition"]
    assert info.attrs["format_version"] == PATH_TRACE_FORMAT_VERSION
    assert info.attrs["scene"] == "test"
    with h5py.File(path, "r") as f:
        assert np.array_equal(f["num_paths"][()], np.diff(paths["path_offset"], axis=1))
    window = read_path_window(path, 1, 13, 20)  # anchors 1 and 3 partly, 2 fully
    assert window.num_slots == 20
    assert window.steps.tolist() == [[3, 7], [0, 10], [0, 3]]
    b, e = paths["path_offset"][1, 1], paths["path_offset"][1, 4]
    assert np.array_equal(window.a, paths["a"][b:e].astype(np.complex64))
    assert np.array_equal(window.k_tx, paths["k_tx"][b:e].astype(np.float32))
    assert window.counts.tolist() == np.diff(paths["path_offset"][1, 1:5]).tolist()


def test_1x1_reconstruction_is_the_generator_formula(trace):
    path, paths = trace
    for trajectory, offset, num_slots in ((0, 0, 40), (1, 13, 20), (2, 39, 1)):
        h = reconstruct_cfr(
            read_path_window(path, trajectory, offset, num_slots), dtype=torch.complex128
        )
        assert h.shape == (num_slots, 52, 1)
        expected = reference_cfr(paths, trajectory, offset, num_slots)
        assert np.allclose(h[..., 0].numpy(), expected, rtol=1e-10, atol=0)


def test_windows_are_consistent(trace):
    path, _ = trace
    full = reconstruct_cfr(read_path_window(path, 0, 0, 40), num_rows=2, num_cols=2)
    parts = [
        reconstruct_cfr(read_path_window(path, 0, o, n), num_rows=2, num_cols=2)
        for o, n in ((0, 15), (15, 25))
    ]
    assert torch.equal(full, torch.cat(parts))


def test_mean_gain(trace, tmp_path):
    path, paths = trace
    info = inspect_path_trace(path, deep=True)
    for t in range(3):
        h = reconstruct_cfr(read_path_window(path, t, 0, 40), dtype=torch.complex128)[..., 0]
        gain = h.abs().square().numpy().astype(np.float32).mean(dtype=np.float64)
        assert info.mean_gain[t] == gain
    wrong = write(tmp_path / "wrong.h5", paths, mean_gain=info.mean_gain * 2)
    inspect_path_trace(wrong)  # structure is fine
    with pytest.raises(TraceFormatError, match="mean_gain"):
        inspect_path_trace(wrong, deep=True)


def test_written_files_are_reproducible(tmp_path):
    paths = random_paths(seed=3)
    one, two = write(tmp_path / "1.h5", paths), write(tmp_path / "2.h5", paths)
    assert one.read_bytes() == two.read_bytes()


def test_plane_wave_peaks_at_the_beam_pointing_there(tmp_path):
    codebook = beams.dft_codebook(4, 8)
    b = 11
    u, v = float(codebook.u[b]), float(codebook.v[b])
    local = np.array([math.sqrt(1 - u * u - v * v), u, v])
    for orientation in ((0.0, 0.0, 0.0), (0.7, 0.0, 0.0), (0.3, -0.2, 0.1)):
        k_tx = rotation_matrix(orientation).numpy() @ local  # scene-frame direction
        paths = {
            "path_offset": np.array([[0, 1, 2]]),
            "a": np.array([1e-4 + 2e-4j, 1e-4 + 2e-4j]),
            "tau": np.array([1e-6, 1e-6]),
            "doppler": np.array([0.0, 0.0]),
            "k_tx": np.stack([k_tx, k_tx]),
            "k_rx": np.stack([k_tx, k_tx]),
            "num_slots": 20,
            "anchor_spacing_slots": 10,
            "split": ["train"],
        }
        path = write(tmp_path / "wave.h5", paths, tx_orientation=orientation)
        h = reconstruct_cfr(
            read_path_window(path, 0, 0, 20), num_rows=4, num_cols=8, orientation=orientation
        )
        assert torch.allclose(h.abs(), torch.full(h.shape, abs(1e-4 + 2e-4j)), rtol=1e-5)
        gains = beams.beam_gain(h, codebook)  # [S, F, B]
        assert torch.all(beams.best_beam(gains) == b)
        assert torch.allclose(
            gains[..., b], torch.full(gains.shape[:2], 32 * abs(1e-4 + 2e-4j) ** 2), rtol=1e-4
        )


def test_doppler_and_delay_phases(tmp_path):
    fd, tau = 120.0, 1.3e-6
    paths = {
        "path_offset": np.array([[0, 1]]),
        "a": np.array([1.0 + 0j]),
        "tau": np.array([tau]),
        "doppler": np.array([fd]),
        "k_tx": np.array([[1.0, 0.0, 0.0]]),
        "k_rx": np.array([[-1.0, 0.0, 0.0]]),
        "num_slots": 10,
        "anchor_spacing_slots": 10,
        "split": ["train"],
    }
    h = reconstruct_cfr(
        read_path_window(write(tmp_path / "one.h5", paths), 0, 0, 10), dtype=torch.complex128
    )[..., 0]
    tau32, fd32 = float(np.float32(tau)), float(np.float32(fd))
    slot_step = np.angle(h[1:, 0].numpy() / h[:-1, 0].numpy())
    assert np.allclose(slot_step, np.angle(np.exp(2j * np.pi * fd32 * 0.5e-3)))
    prb_step = np.angle(h[0, 1:].numpy() / h[0, :-1].numpy())
    assert np.allclose(prb_step, np.angle(np.exp(-2j * np.pi * 360e3 * tau32)))


def test_complex64_output_is_close_to_complex128(trace):
    path, _ = trace
    window = read_path_window(path, 2, 5, 30)
    h64 = reconstruct_cfr(window, num_rows=4, num_cols=8)
    h128 = reconstruct_cfr(window, num_rows=4, num_cols=8, dtype=torch.complex128)
    assert h64.dtype == torch.complex64
    assert torch.allclose(
        h64.to(torch.complex128), h128, rtol=1e-4, atol=1e-6 * float(h128.abs().max())
    )


def test_rotation_matrix():
    assert torch.equal(rotation_matrix((0, 0, 0)), torch.eye(3, dtype=torch.float64))
    r = rotation_matrix((0.4, -0.3, 1.1))
    assert torch.allclose(r @ r.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
    assert math.isclose(float(torch.linalg.det(r)), 1.0, rel_tol=1e-12)
    # Yaw about z, then pitch about y, then roll about x: R = Rz(a) Ry(b) Rx(c)
    a, b, c = 0.4, -0.3, 1.1
    rz = torch.tensor(
        [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]],
        dtype=torch.float64,
    )
    ry = torch.tensor(
        [[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]],
        dtype=torch.float64,
    )
    rx = torch.tensor(
        [[1, 0, 0], [0, math.cos(c), -math.sin(c)], [0, math.sin(c), math.cos(c)]],
        dtype=torch.float64,
    )
    assert torch.allclose(r, rz @ ry @ rx, atol=1e-12)
    assert torch.allclose(
        rotation_matrix((math.pi / 2, 0, 0)) @ torch.tensor([1.0, 0, 0], dtype=torch.float64),
        torch.tensor([0.0, 1, 0], dtype=torch.float64),
        atol=1e-12,
    )


def _corrupt(path, edit):
    with h5py.File(path, "r+") as f:
        edit(f)
    return path


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda f: f.attrs.__setitem__("format_version", 1), "not a paths trace"),
        (lambda f: f.attrs.__setitem__("channel_kind", "gains"), "channel_kind"),
        (lambda f: f.attrs.__setitem__("num_slots", 45), "multiple of anchor_spacing_slots"),
        (lambda f: f.attrs.__setitem__("prb_sampling", "mean12"), "prb_sampling"),
        (lambda f: f.attrs.__delitem__("tx_antenna"), "missing attributes"),
        (lambda f: f.attrs.__setitem__("rx_antenna", "iso"), "rx_antenna"),
        (lambda f: f["path_offset"].__setitem__((0, 2), 0), "never decrease"),
        (lambda f: f["num_paths"].__setitem__((0, 0), 99), "num_paths"),
        (lambda f: f["tau"].__setitem__(0, -1.0), "tau"),
        (lambda f: f["k_tx"].__setitem__(0, [1.0, 1.0, 0.0]), "unit vectors"),
        (lambda f: f["a"].__setitem__(1, complex("nan")), "NaN"),
        (lambda f: f.__delitem__("k_rx"), "missing dataset 'k_rx'"),
        (lambda f: f["mean_gain"].__setitem__(0, -1.0), "mean_gain"),
    ],
)
def test_validation_rejects(tmp_path, edit, message):
    path = _corrupt(write(tmp_path / "bad.h5", random_paths()), edit)
    with pytest.raises(TraceFormatError, match=message):
        inspect_path_trace(path)


def test_writer_rejects_reserved_attrs(tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        write(tmp_path / "x.h5", random_paths(), attrs={"num_slots": 3})


def test_window_bounds(trace):
    path, _ = trace
    with pytest.raises(ValueError, match="does not fit"):
        read_path_window(path, 0, 30, 20)
    with pytest.raises(ValueError, match="trajectory"):
        read_path_window(path, 3, 0, 10)
