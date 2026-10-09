"""PathTraceSource and trace_episodes on path traces, with the committed Munich path sample."""

import math
import pickle
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from linkgym.beams import beam_gain, dft_codebook
from linkgym.channels import TraceChannelSource, TraceFormatError, inspect_trace, write_trace
from linkgym.evaluation import trace_episodes
from linkgym.paths import PathTraceSource, read_path_window, reconstruct_cfr

DATA = Path(__file__).resolve().parent / "data"
PATHS = DATA / "munich_paths_sample.h5"
NUM_SLOTS = 100  # of the path sample
SCENARIO = {
    "num_prbs": 52,
    "subcarrier_spacing": 30e3,
    "carrier_frequency": 3.5e9,
    "slot_duration": 0.5e-3,
}
SPLITS = ("train", "val", "test")


def source(num_slots=50, **kwargs):
    return PathTraceSource(PATHS, **SCENARIO, num_slots=num_slots, splits=SPLITS, **kwargs)


@pytest.fixture(scope="module")
def gain_trace(tmp_path_factory):
    """The format-1 sample cut to the slots of the path sample: the same trajectories."""
    path = tmp_path_factory.mktemp("gains") / "gains.h5"
    info = inspect_trace(DATA / "munich_sample.h5")
    with h5py.File(DATA / "munich_sample.h5", "r") as f:
        gain = f["gain"][:, :NUM_SLOTS]
        routes = f.attrs["routes"]
    write_trace(
        path,
        gain,
        info.split,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30e3,
        slot_duration_s=0.5e-3,
        group=info.group,
        category=info.category,
        attrs={"routes": routes},
    )
    return path


@pytest.mark.parametrize("snr_mode", ["normalized", "link_budget"])
def test_same_draws_and_gains_as_the_gain_trace(gain_trace, snr_mode):
    power = {"tx_power_dbm": 30.0} if snr_mode == "link_budget" else {}
    paths = source(snr_mode=snr_mode, **power)
    gains = TraceChannelSource(
        gain_trace, **SCENARIO, num_slots=50, splits=SPLITS, snr_mode=snr_mode, **power
    )
    for seed in range(5):
        ours, theirs = paths.generate(50, 6, seed), gains.generate(50, 6, seed)
        assert ours.info == theirs.info
        assert ours.reference_snr_db == theirs.reference_snr_db
        assert ours.channel.shape == (6, 50, 52, 1) and ours.channel.dtype == torch.complex64
        assert ours.beam_gain is None
        power_1x1 = ours.channel[..., 0].abs().square().numpy()
        for b, gain in enumerate(theirs.gain.numpy()):
            np.testing.assert_allclose(power_1x1[b], gain, rtol=1e-5, atol=1e-6 * gain.mean())


def test_normalized_single_antenna_has_unit_mean_power():
    paths = source(num_slots=NUM_SLOTS)
    episode = paths.generate(NUM_SLOTS, 4, 0, trajectories=[0, 1, 2, 3], offsets=[0] * 4)
    mean = episode.channel.abs().square().mean(dim=(1, 2, 3)).double()
    torch.testing.assert_close(mean, torch.ones(4, dtype=torch.float64), rtol=1e-5, atol=0)


def test_episodes_are_deterministic_and_seeded():
    paths = source(num_rows=2, num_cols=4)
    first, again = paths.generate(50, 3, 7), paths.generate(50, 3, 7)
    assert torch.equal(first.channel, again.channel) and first.info == again.info
    assert first.info != paths.generate(50, 3, 8).info


def test_pinned_windows_are_the_reconstructed_windows():
    paths = source(num_rows=4, num_cols=8, orientation=(0.4, 0.0, 0.0))
    episode = paths.generate(50, 2, 0, trajectories=[2, 0], offsets=[13, 50])
    assert episode.info["trajectory"] == [2, 0] and episode.info["offset"] == [13, 50]
    for b, (t, o) in enumerate([(2, 13), (0, 50)]):
        h = reconstruct_cfr(
            read_path_window(PATHS, t, o, 50), num_rows=4, num_cols=8, orientation=(0.4, 0, 0)
        )
        scale = 1 / math.sqrt(paths.info.mean_gain[t])
        torch.testing.assert_close(episode.channel[b], h * scale)


def test_orientation_defaults_to_the_traced_transmitter():
    assert source().array["orientation"] == (0.0, 0.0, 0.0)
    pin = {"trajectories": [0], "offsets": [0]}
    default = source(num_rows=1, num_cols=8).generate(50, 1, 0, **pin).channel
    turned = source(num_rows=1, num_cols=8, orientation=(0.5, 0, 0)).generate(50, 1, 0, **pin)
    assert default.shape == turned.channel.shape == (1, 50, 52, 8)
    assert not torch.allclose(default, turned.channel)


def test_codebook_gives_beam_gains():
    codebook = dft_codebook(4, 8, oversampling=(2, 2))
    paths = source(num_rows=4, num_cols=8, codebook=codebook)
    episode = paths.generate(50, 2, 3)
    assert episode.beam_gain.shape == (2, 50, 52, codebook.num_beams)
    torch.testing.assert_close(episode.beam_gain, beam_gain(episode.channel, codebook))
    with pytest.raises(ValueError, match="codebook is for a 4x8 array"):
        source(num_rows=2, num_cols=8, codebook=codebook)
    with pytest.raises(ValueError, match="spacing"):
        source(num_rows=4, num_cols=8, spacing=(0.5, 0.7), codebook=codebook)


def test_source_can_be_pickled():
    paths = source(num_rows=2, num_cols=2)
    before = paths.generate(50, 2, 1)
    copy = pickle.loads(pickle.dumps(paths))
    assert torch.equal(copy.generate(50, 2, 1).channel, before.channel)
    paths.close()
    copy.close()


@pytest.mark.parametrize(
    "change, error, match",
    [
        ({"num_prbs": 51}, TraceFormatError, "52 PRBs"),
        ({"carrier_frequency": 28e9}, TraceFormatError, "carrier_frequency_hz"),
        ({"num_slots": 101}, TraceFormatError, "episodes need 101"),
        ({"splits": ("holdout",)}, TraceFormatError, "not in the trace"),
        ({"splits": ()}, ValueError, "must not be empty"),
        ({"snr_mode": "absolute"}, ValueError, "snr_mode"),
        ({"snr_mode": "link_budget"}, ValueError, "requires tx_power_dbm"),
        ({"num_rows": 0}, ValueError, "at least one row"),
    ],
)
def test_scenario_checks(change, error, match):
    kwargs = {**SCENARIO, "num_slots": 50, "splits": SPLITS} | change
    with pytest.raises(error, match=match):
        PathTraceSource(PATHS, **kwargs)


def test_pins_are_checked():
    paths = PathTraceSource(PATHS, **SCENARIO, num_slots=50, splits=("train",))
    with pytest.raises(ValueError, match="pin both"):
        paths.generate(50, 1, 0, trajectories=[0])
    with pytest.raises(ValueError, match="not in the splits"):
        paths.generate(50, 1, 0, trajectories=[3], offsets=[0])
    with pytest.raises(ValueError, match="outside"):
        paths.generate(50, 1, 0, trajectories=[0], offsets=[51])
    with pytest.raises(ValueError, match="one offset per link"):
        paths.generate(50, 2, 0, trajectories=[0], offsets=[0])
    with pytest.raises(ValueError, match="exceed"):
        paths.generate(101, 1, 0)


def test_trace_episodes_on_a_path_trace(gain_trace):
    for splits in (["train"], ["val", "test"]):
        episodes = trace_episodes(PATHS, splits, episode_length=50)
        assert episodes == trace_episodes(gain_trace, splits, episode_length=50)
    episodes = trace_episodes(PATHS, ["train"], episode_length=50)
    assert [e["route"] for e in episodes] == ["north-canyon"] * 2 + ["east-avenue"] * 2
    # The options pin a source's windows
    paths = PathTraceSource(PATHS, **SCENARIO, num_slots=50)
    options = [e["options"] for e in episodes]
    episode = paths.generate(
        50,
        len(options),
        0,
        trajectories=[o["trajectory"] for o in options],
        offsets=[o["offset"] for o in options],
    )
    assert episode.info["offset"] == [0, 50, 0, 50]


def test_trace_episodes_still_rejects_other_files(tmp_path):
    bad = tmp_path / "bad.h5"
    bad.write_bytes(b"not hdf5")
    with pytest.raises(TraceFormatError):
        trace_episodes(bad, ["train"])
