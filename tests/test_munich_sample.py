"""The committed Munich sample trace (tests/data) is valid and usable by the environment."""

import json
import warnings
from pathlib import Path

import gymnasium
import pytest
from gymnasium.utils.env_checker import check_env as gymnasium_check_env

from linkgym import ENV_ID
from linkgym.channels import CATEGORIES, inspect_trace

SAMPLE = Path(__file__).resolve().parent / "data" / "munich_sample.h5"


def test_sample_is_small_and_valid():
    assert SAMPLE.stat().st_size < 1_000_000
    info = inspect_trace(SAMPLE)
    assert info.shape[1:] == (1000, 52)
    assert set(info.split) == {"train", "val", "test"}
    assert set(info.category) <= set(CATEGORIES)
    attrs = info.attrs
    assert attrs["scene"] == "munich"
    assert "OpenStreetMap" in attrs["attribution"] and "ODbL" in attrs["attribution"]
    assert json.loads(attrs["solver"])["deterministic"] is True
    sample_of = json.loads(attrs["sample_of"])
    assert len(sample_of["routes"]) == info.num_trajectories


@pytest.mark.parametrize("snr_mode", ["normalized", "link_budget"])
def test_environment_runs_on_the_sample(snr_mode):
    kwargs = {"tx_power_dbm": 20.0} if snr_mode == "link_budget" else {}
    env = gymnasium.make(
        ENV_ID,
        channel="trace",
        trace_path=SAMPLE,
        trace_splits=["train", "val", "test"],
        episode_length=200,
        snr_mode=snr_mode,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gymnasium_check_env(env.unwrapped)
    env.reset(seed=0)
    channel = env.unwrapped.channel_info
    assert channel["category"] in CATEGORIES
    assert channel["split"] in {"train", "val", "test"}
    env.close()
