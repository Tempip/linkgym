"""Golden outputs of the trace channel path: TraceChannelSource, the environment, write_trace.

Recorded with linkgym 0.2.0 on the platform of tests/test_golden.py (RECORDED_ON) and
checked bit for bit only there; elsewhere the tests are skipped. ``write_trace`` file bytes
also depend on h5py and the HDF5 library, so that test additionally needs their recorded
versions. Do not update the values to make a refactor pass: a changed hash means the trace
path changed.
"""

import hashlib
from pathlib import Path

import gymnasium
import h5py
import numpy as np
import pytest
from test_golden import RECORDED_ON, digest, platform_signature, recorded_platform  # noqa: F401

from linkgym import ENV_ID
from linkgym.baselines import OLLAPolicy
from linkgym.channels import TraceChannelSource, write_trace

SAMPLE = Path(__file__).resolve().parent / "data" / "munich_sample.h5"
HDF5_RECORDED_ON = {"h5py": "3.16.0", "hdf5": "2.0.0"}

GOLDEN = {
    "source_normalized_random": "1a9547be25b4b1ad7e8d8146187208965480734899d93cff32120d0ae6b207fa",
    "source_normalized_pinned": "afbb16d649fd355aa572ce5978ac5d1ae58f8c57c538b5f2dbec01eed8eefa67",
    "source_link_budget": "e214184d2e0ba3f59964ac98b9c097e5a2b56153dbef18e037f24b682a2ea731",
    "source_link_budget_reference_snr_hex": "0x1.f1022e0a02160p+6",
    "env_olla_trajectory": "48738176b2f4f445ed3245e290a9024b927289529ef1b2e5a3ee0141b153ff78",
    "env_olla_reward_sum_hex": "0x1.77ad4e930288ep+9",
    "write_trace_sha256": "f5e69a3b5c14b4b9f86f3abb9407f38e8ccedf47641c5b9dfa00aac50e027363",
}


def source(**kwargs) -> TraceChannelSource:
    return TraceChannelSource(
        SAMPLE,
        num_prbs=52,
        subcarrier_spacing=30e3,
        carrier_frequency=3.5e9,
        slot_duration=0.5e-3,
        num_slots=200,
        **kwargs,
    )


def episode_digest(episode) -> str:
    info = [[str(v) for v in episode.info[k]] for k in sorted(episode.info)]
    return digest(episode.gain.numpy(), np.array(sorted(episode.info)), np.array(info))


def compute_source() -> dict:
    normalized = source(splits=("train", "val", "test"))
    random_episode = normalized.generate(200, 3, seed=0)
    pinned = normalized.generate(200, 2, seed=1, trajectories=[0, 3], offsets=[0, 800])
    budget = source(splits=("train",), snr_mode="link_budget", tx_power_dbm=30.0)
    budget_episode = budget.generate(200, 2, seed=2)
    return {
        "source_normalized_random": episode_digest(random_episode),
        "source_normalized_pinned": episode_digest(pinned),
        "source_link_budget": episode_digest(budget_episode),
        "source_link_budget_reference_snr_hex": float(budget_episode.reference_snr_db).hex(),
    }


def compute_env() -> dict:
    env = gymnasium.make(ENV_ID, channel="trace", trace_path=str(SAMPLE), trace_splits=("val",))
    policy = OLLAPolicy(0.1)
    obs, info = env.reset(seed=5, options={"trajectory": 2, "offset": 0})
    policy.reset()
    observations, privileged = [obs], [info["privileged"]["sinr_prb"]]
    rewards, mcs, ack, bits, tbler, report_sinr = [], [], [], [], [], []
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy.act(obs, info))
        done = terminated or truncated
        observations.append(obs)
        rewards.append(reward)
        mcs.append(info["mcs"])
        ack.append(info["ack"])
        bits.append(info["bits"])
        tbler.append(info["tbler"])
        report_sinr.append(info["report"]["sinr_wideband_db"])
        if info["privileged"] is not None:
            privileged.append(info["privileged"]["sinr_prb"])
    channel = env.unwrapped.channel_info
    env.close()
    assert len(rewards) == 1000
    trajectory = digest(
        np.stack(observations),
        np.array(rewards),
        np.array(mcs),
        np.array(ack),
        np.array(bits),
        np.array(tbler),
        np.array(report_sinr),
        np.stack(privileged),
        np.array([info["snr_db"]]),
        np.array([str(channel[k]) for k in sorted(channel)]),
    )
    return {
        "env_olla_trajectory": trajectory,
        "env_olla_reward_sum_hex": float(np.sum(rewards)).hex(),
    }


def compute_write_trace(path: Path) -> dict:
    rng = np.random.default_rng(0)
    write_trace(
        path,
        rng.exponential(size=(3, 40, 52)).astype(np.float32),
        ["train", "val", "test"],
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30e3,
        slot_duration_s=0.5e-3,
        group=np.array([0, 1, 2]),
        rx_position=rng.normal(size=(3, 40, 3)),
        num_paths=rng.integers(0, 500, size=(3, 4)),
        los=rng.integers(0, 2, size=(3, 4)),
        category=["los", "nlos", "transition"],
        attrs={"scene": "golden", "tx_position": np.array([1.0, 2.0, 3.0]), "num_dropped": 2},
    )
    return {"write_trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def recorded_hdf5(recorded_platform):  # noqa: F811
    versions = {"h5py": h5py.__version__, "hdf5": h5py.version.hdf5_version}
    if versions != HDF5_RECORDED_ON:
        pytest.skip(f"write_trace bytes recorded with {HDF5_RECORDED_ON}, this is {versions}")


def _check(computed: dict) -> None:
    for key, value in computed.items():
        assert value == GOLDEN[key], f"{key}: computed {value!r}, golden {GOLDEN[key]!r}"


def test_golden_trace_source(recorded_platform):  # noqa: F811
    _check(compute_source())


def test_golden_trace_environment(recorded_platform):  # noqa: F811
    _check(compute_env())


def test_write_trace_bytes_unchanged(recorded_hdf5, tmp_path):
    _check(compute_write_trace(tmp_path / "golden.h5"))


def test_platform_signature_matches_tdl_goldens():
    # The trace goldens share the TDL guard; recording elsewhere would need both updated
    assert set(platform_signature()) == set(RECORDED_ON)
