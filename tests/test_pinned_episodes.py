"""Pinned trace episodes (reset options), their enumeration and the cluster bootstrap."""

import gymnasium
import numpy as np
import pytest

from linkgym import ENV_ID
from linkgym.baselines import FixedMCSPolicy, OLLAPolicy
from linkgym.channels import read_trace, write_trace
from linkgym.evaluation import cluster_bootstrap, evaluate_episodes, trace_episodes

SPLITS = ["train", "train", "val", "test"]


@pytest.fixture
def trace(tmp_path):
    """4 trajectories of 120 slots, routes recorded as the generator does."""
    rng = np.random.default_rng(0)
    gain = rng.exponential(size=(4, 120, 52)).astype(np.float32) * 1e-10
    path = tmp_path / "trace.h5"
    routes = '[{"group": 0, "name": "a"}, {"group": 1, "name": "b"}, {"group": 2, "name": "c"}]'
    write_trace(
        path,
        gain,
        SPLITS,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30e3,
        slot_duration_s=0.5e-3,
        group=np.array([0, 0, 1, 2]),
        category=["los", "los", "nlos", "transition"],
        attrs={"routes": routes},
    )
    return path


def make(path, **kwargs):
    kwargs = {"channel": "trace", "trace_path": path, "episode_length": 40} | kwargs
    return gymnasium.make(ENV_ID, **kwargs)


def episode_sinr(env, seed, options=None):
    _, info = env.reset(seed=seed, options=options)
    sinr = [info["privileged"]["sinr_prb"]]
    uniforms = []
    for _ in range(39):
        *_, info = env.step(5)
        sinr.append(info["privileged"]["sinr_prb"])
        uniforms.append(env.unwrapped.last_result.ack_uniform.item())
    return np.stack(sinr), info["snr_db"], uniforms


def test_pinned_reset_uses_the_given_window(trace):
    env = make(trace, snr_db=10.0)
    sinr, _, _ = episode_sinr(env, 3, {"trajectory": 1, "offset": 60})
    channel = env.unwrapped.channel_info
    assert (channel["trajectory"], channel["offset"], channel["split"]) == (1, 60, "train")
    gain = read_trace(trace).gain.astype(np.float64)
    expected = 10.0 * gain[1, 60:100] / gain[1].mean()
    np.testing.assert_allclose(sinr, expected, rtol=1e-5)


def test_pinning_replaces_only_the_choice_of_window(trace):
    """Pinning the window a seed would draw anyway reproduces the unpinned episode."""
    env = make(trace)
    free = episode_sinr(env, 7)
    channel = env.unwrapped.channel_info
    pinned = episode_sinr(
        env, 7, {"trajectory": channel["trajectory"], "offset": channel["offset"]}
    )
    np.testing.assert_array_equal(free[0], pinned[0])
    assert free[1:] == pinned[1:]  # same SNR and ACK uniforms


def test_pinned_episodes_are_common_random_numbers(trace):
    options = {"trajectory": 0, "offset": 20}
    a = episode_sinr(make(trace), 11, options)
    b = episode_sinr(make(trace), 11, options)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1:] == b[1:]
    c = episode_sinr(make(trace), 12, options)
    assert c[1] != a[1]  # another seed: another SNR draw on the same window


@pytest.mark.parametrize(
    "options, match",
    [
        ({"trajectory": 0}, "both 'trajectory' and 'offset'"),
        ({"trajectory": 0, "offset": 0, "speed": 3}, r"unknown reset options \['speed'\]"),
        ({"trajectory": 2, "offset": 0}, r"trajectories \[2\] are not in the splits \['train'\]"),
        ({"trajectory": 0, "offset": 81}, r"offsets \[81\] outside \[0, 80\]"),
        ({"trajectory": 0, "offset": -1}, "outside"),
        ({"trajectory": 0.5, "offset": 0}, "must be integers"),
    ],
)
def test_invalid_pins_are_rejected(trace, options, match):
    env = make(trace)
    with pytest.raises(ValueError, match=match):
        env.reset(seed=0, options=options)


def test_pinning_needs_a_trace_channel():
    env = gymnasium.make(ENV_ID, episode_length=20)
    with pytest.raises(ValueError, match="need channel='trace'"):
        env.reset(seed=0, options={"trajectory": 0, "offset": 0})
    env.reset(seed=0, options={})  # empty options are fine


def test_trace_episodes(trace):
    episodes = trace_episodes(trace, ["train", "test"], episode_length=40, first_seed=100)
    assert len(episodes) == 3 * 3  # trajectories 0, 1, 3 x windows at 0, 40, 80
    assert [e["seed"] for e in episodes] == list(range(100, 109))
    assert [e["options"] for e in episodes[:3]] == [
        {"trajectory": 0, "offset": o} for o in (0, 40, 80)
    ]
    assert [(e["route"], e["route_trajectory"]) for e in episodes[::3]] == [
        ("a", 0),
        ("a", 1),
        ("c", 0),
    ]
    assert [e["category"] for e in episodes[::3]] == ["los", "los", "transition"]
    assert len(trace_episodes(trace, ["train"], episode_length=40, windows=2)) == 4
    with pytest.raises(ValueError, match="windows must be in"):
        trace_episodes(trace, ["train"], episode_length=40, windows=4)


def test_evaluate_episodes_is_paired_across_policies(trace):
    episodes = trace_episodes(trace, ["train"], episode_length=40, first_seed=5)
    kwargs = {"channel": "trace", "trace_path": trace, "episode_length": 40}
    fixed = evaluate_episodes(FixedMCSPolicy(14), kwargs, episodes)
    olla = evaluate_episodes(OLLAPolicy(0.1), kwargs, episodes)
    assert [r["snr_db"] for r in fixed] == [r["snr_db"] for r in olla]
    assert [r["route"] for r in fixed] == ["a"] * 6
    assert "options" not in fixed[0]
    assert fixed == evaluate_episodes(FixedMCSPolicy(14), kwargs, episodes)


def test_cluster_bootstrap():
    rng = np.random.default_rng(1)
    clusters = np.repeat(np.arange(10), 4)
    b = rng.uniform(20, 30, size=40)
    effect = np.repeat(rng.normal(1.0, 2.0, size=10), 4)  # shared within a cluster
    a = b + effect
    result = cluster_bootstrap(a, b, clusters, seed=0)
    assert result["mean"] == pytest.approx(np.mean(a - b))
    assert result["ci_low"] < result["mean"] < result["ci_high"]
    assert result["pct"] == pytest.approx(100 * np.mean(a - b) / np.mean(b))
    assert (result["num_clusters"], result["num_elements"]) == (10, 40)
    # Resampling the 40 episodes as if independent understates the uncertainty
    naive = cluster_bootstrap(a, b, np.arange(40), seed=0)
    assert naive["ci_high"] - naive["ci_low"] < 0.75 * (result["ci_high"] - result["ci_low"])
    # Deterministic for a seed; the mean of a alone without b
    assert cluster_bootstrap(a, b, clusters, seed=0) == result
    single = cluster_bootstrap(a, None, clusters)
    assert single["mean"] == pytest.approx(a.mean()) and "pct" not in single
    one = cluster_bootstrap(a[:4], b[:4], clusters[:4])
    assert np.isnan(one["ci_low"]) and np.isnan(one["pct_ci_high"])
