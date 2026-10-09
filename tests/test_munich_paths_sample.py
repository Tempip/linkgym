"""The committed Munich path sample (tests/data, format 2) matches the format-1 sample."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from linkgym.beams import beam_gain, dft_codebook
from linkgym.channels import inspect_trace
from linkgym.paths import inspect_path_trace, read_path_window, reconstruct_cfr

DATA = Path(__file__).resolve().parent / "data"
PATHS = DATA / "munich_paths_sample.h5"
GAINS = DATA / "munich_sample.h5"
NUM_SLOTS = 100


def test_path_sample_is_small_and_valid():
    assert PATHS.stat().st_size < 1_000_000
    info = inspect_path_trace(PATHS, deep=True)
    assert info.shape == (4, NUM_SLOTS, 52)
    attrs = info.attrs
    assert attrs["scene"] == "munich"
    assert "OpenStreetMap" in attrs["attribution"] and "ODbL" in attrs["attribution"]
    assert json.loads(attrs["sample_of"])["slots"] == [0, NUM_SLOTS]


def test_path_sample_has_the_trajectories_of_the_gain_sample():
    info, reference = inspect_path_trace(PATHS), inspect_trace(GAINS)
    np.testing.assert_array_equal(info.split, reference.split)
    np.testing.assert_array_equal(info.group, reference.group)
    np.testing.assert_array_equal(info.category, reference.category)
    k = info.anchor_spacing
    with h5py.File(PATHS, "r") as f, h5py.File(GAINS, "r") as g:
        np.testing.assert_array_equal(f["rx_position"][()], g["rx_position"][:, :NUM_SLOTS])
        np.testing.assert_array_equal(f["los"][()], g["los"][:, : NUM_SLOTS // k])
        np.testing.assert_array_equal(f["num_paths"][()], g["num_paths"][:, : NUM_SLOTS // k])
        gains = g["gain"][:, :NUM_SLOTS]
        for key in ("scene_sha256", "solver", "generator_config", "routes"):
            assert f.attrs[key] == g.attrs[key]
    for t in range(info.num_trajectories):
        assert info.mean_gain[t] == gains[t].mean(dtype=np.float64)


def test_single_antenna_reconstruction_gives_the_gains_of_format_1():
    with h5py.File(GAINS, "r") as g:
        gains = g["gain"][:, :NUM_SLOTS]
    for t in range(len(gains)):
        h = reconstruct_cfr(read_path_window(PATHS, t, 0, NUM_SLOTS), dtype=torch.complex128)
        gain = (h[..., 0].abs() ** 2).numpy().astype(np.float32)
        np.testing.assert_allclose(gain, gains[t], rtol=1e-5, atol=1e-6 * gains[t].mean())


def test_best_beam_of_an_array_collects_most_of_its_power():
    # On every trajectory, the best DFT beam of a 4x8 array has an array gain above 10 dB
    # (32 at most, the number of antennas)
    codebook = dft_codebook(4, 8)
    for t in range(4):
        h = reconstruct_cfr(read_path_window(PATHS, t, 0, NUM_SLOTS), num_rows=4, num_cols=8)
        gain = beam_gain(h, codebook).mean(dim=(0, 1))  # [32 beams]
        assert 10 < (gain.max() / h.abs().square().mean()).item() <= 32
