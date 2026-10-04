"""tx_power_for_median_snr on the sample trace."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from linkgym.channels import TraceChannelSource, inspect_trace, tx_power_for_median_snr

SAMPLE = Path(__file__).resolve().parent / "data" / "munich_sample.h5"


def train_episode(tx_power_dbm: float, noise_figure_db: float = 7.0):
    info = inspect_trace(SAMPLE)
    train = np.flatnonzero(info.split == "train")
    source = TraceChannelSource(
        SAMPLE,
        num_prbs=52,
        subcarrier_spacing=30e3,
        carrier_frequency=3.5e9,
        slot_duration=0.5e-3,
        num_slots=info.num_slots,
        splits=("train",),
        snr_mode="link_budget",
        tx_power_dbm=tx_power_dbm,
        noise_figure_db=noise_figure_db,
    )
    episode = source.generate(
        info.num_slots, len(train), seed=0, trajectories=train, offsets=[0] * len(train)
    )
    source.close()
    return train, episode


def test_median_wideband_snr_hits_the_target():
    tx = tx_power_for_median_snr(SAMPLE, ["train"], 10.0, 7.0)
    _, episode = train_episode(tx)
    wideband = episode.gain.double().mean(dim=2).numpy()  # [trajectories, slots]
    wideband_db = episode.reference_snr_db + 10 * np.log10(wideband)
    assert float(np.median(wideband_db)) == pytest.approx(10.0, abs=1e-6)


def test_power_differences_between_trajectories_are_kept():
    tx = tx_power_for_median_snr(SAMPLE, ["train"], 10.0)
    train, episode = train_episode(tx)
    with h5py.File(SAMPLE, "r") as f:
        raw = np.stack([f["gain"][t] for t in train])
    # Link-budget gains are the trace's gains: no per-trajectory normalization
    assert np.array_equal(episode.gain.numpy(), raw)
    mean_db = 10 * np.log10(raw.mean(axis=(1, 2), dtype=np.float64))
    assert abs(mean_db[0] - mean_db[1]) > 10  # the sample's two train trajectories differ


def test_power_moves_one_to_one_with_target_and_noise_figure():
    base = tx_power_for_median_snr(SAMPLE, ["train"], 10.0, 7.0)
    assert tx_power_for_median_snr(SAMPLE, ["train"], 12.5, 7.0) == pytest.approx(base + 2.5)
    assert tx_power_for_median_snr(SAMPLE, ["train"], 10.0, 9.0) == pytest.approx(base + 2.0)


def test_splits_are_validated():
    with pytest.raises(ValueError, match="not in"):
        tx_power_for_median_snr(SAMPLE, ["holdout"], 10.0)
    with pytest.raises(ValueError, match="must not be empty"):
        tx_power_for_median_snr(SAMPLE, [], 10.0)
