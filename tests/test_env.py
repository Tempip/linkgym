import subprocess
import sys
import warnings

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gymnasium_check_env
from gymnasium.utils.env_checker import data_equivalence

from linkgym import ENV_ID, ScenarioConfig
from linkgym.env import LinkAdaptationEnv
from linkgym.sim import MAX_MCS, MIN_MCS, tb_size_per_mcs


def make(**kwargs):
    return gymnasium.make(ENV_ID, **{"episode_length": 40, **kwargs})


def rollout(env, seed, actions):
    """Return [(obs, info)] after reset followed by (obs, reward, terminated, truncated, info)."""
    trace = [env.reset(seed=seed)]
    for action in actions:
        trace.append(env.step(action))
    return trace


def wideband_db(sinr_prb):
    return 10.0 * np.log10(np.mean(sinr_prb.astype(np.float64)))


def test_import_does_not_load_torch_or_sionna():
    code = "import sys, linkgym; print('torch' in sys.modules, 'sionna' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


def test_make_forwards_scenario_fields():
    env = gymnasium.make(ENV_ID, speed=3.0, num_reports=2, feedback_delay=2)
    config = env.unwrapped.config
    assert (config.speed, config.num_reports, config.feedback_delay) == (3.0, 2, 2)
    assert env.observation_space.shape == (6,)

    env = gymnasium.make(ENV_ID, config=ScenarioConfig(snr_db=12.0), speed=5.0)
    assert (env.unwrapped.config.snr_db, env.unwrapped.config.speed) == (12.0, 5.0)

    with pytest.raises(TypeError):
        gymnasium.make(ENV_ID, not_a_field=1)


def test_gymnasium_check_env():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gymnasium_check_env(make().unwrapped)


def test_sb3_check_env():
    env_checker = pytest.importorskip("stable_baselines3.common.env_checker")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env_checker.check_env(make().unwrapped, warn=True)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"snr_db": -20.0, "feedback_delay": 3, "num_reports": 2}, {"snr_db": 45.0}],
)
def test_spaces_and_bounds(kwargs):
    env = make(**kwargs)
    num_reports = env.unwrapped.config.num_reports
    assert env.action_space == gymnasium.spaces.Discrete(MAX_MCS - MIN_MCS + 1)
    space = env.observation_space
    assert space.shape == (3 * num_reports,)
    assert space.dtype == np.float32
    assert np.all(space.low == -1.0) and np.all(space.high == 1.0)

    env.action_space.seed(0)
    actions = [env.action_space.sample() for _ in range(40)]
    trace = rollout(env, 0, actions)
    for obs, *_ in trace:
        assert obs.dtype == np.float32
        assert obs in space
    for _, reward, *_ in trace[1:]:
        assert 0.0 <= reward <= 1.0


def test_reward_is_bits_over_max_tb_size():
    env = make(snr_db=45.0)
    max_tb_size = int(tb_size_per_mcs(env.unwrapped.num_allocated_re)[MAX_MCS])
    assert max_tb_size == int(tb_size_per_mcs(env.unwrapped.num_allocated_re).max())
    assert env.unwrapped.max_tb_size == max_tb_size

    trace = rollout(env, 0, [MAX_MCS - MIN_MCS] * 40)
    for _, reward, _, _, info in trace[1:]:
        assert reward == info["bits"] / max_tb_size
    assert any(reward == 1.0 for _, reward, *_ in trace[1:])


def test_same_seed_and_actions_are_deterministic():
    env = make()
    env.action_space.seed(1)
    actions = [env.action_space.sample() for _ in range(40)]
    first, second = rollout(make(), 3, actions), rollout(make(), 3, actions)
    for a, b in zip(first, second, strict=True):
        assert data_equivalence(a, b, exact=True)

    other = rollout(make(), 4, actions)
    assert first[0][1]["snr_db"] != other[0][1]["snr_db"]
    assert not all(np.array_equal(a[0], b[0]) for a, b in zip(first, other, strict=True))


def test_common_random_numbers_across_policies():
    """Same seed, different actions: same SNR, channel and ACK uniforms in every slot."""
    env_a, env_b = make(), make()
    env_b.action_space.seed(0)
    _, info_a = env_a.reset(seed=11)
    _, info_b = env_b.reset(seed=11)
    assert info_a["snr_db"] == info_b["snr_db"]
    for _ in range(40):
        np.testing.assert_array_equal(
            info_a["privileged"]["sinr_prb"], info_b["privileged"]["sinr_prb"]
        )
        _, _, _, _, info_a = env_a.step(5 - MIN_MCS)
        _, _, _, _, info_b = env_b.step(env_b.action_space.sample())
        assert info_a["snr_db"] == info_b["snr_db"]
        u_a = env_a.unwrapped.last_result.ack_uniform
        u_b = env_b.unwrapped.last_result.ack_uniform
        assert u_a.tolist() == u_b.tolist()


def test_truncation_at_episode_length():
    env = make(episode_length=15)
    env.reset(seed=0)
    for t in range(15):
        _, _, terminated, truncated, _ = env.step(0)
        assert terminated is False
        assert truncated is (t == 14)
    with pytest.raises(RuntimeError):
        env.step(0)


@pytest.mark.parametrize("delay", [1, 3])
def test_observation_holds_only_delayed_reports(delay):
    """The observation used to choose slot t contains reports of slots <= t - d only."""
    num_reports, length = 4, 40
    env = make(snr_db=15.0, feedback_delay=delay, num_reports=num_reports, episode_length=length)
    env.action_space.seed(2)
    obs, info = env.reset(seed=5)
    before = [(obs, info)]  # observation and info available when choosing slot t
    outcome = []  # info returned by the step that transmitted slot t
    for _ in range(length):
        obs, _, _, _, info = env.step(env.action_space.sample())
        outcome.append(info)
        before.append((obs, info))

    reports = {}
    for t in range(length):
        obs, info = before[t]
        report = info["report"]
        sinr_t = wideband_db(info["privileged"]["sinr_prb"])  # ground truth of slot t
        if t < delay:
            assert report is None
            assert not obs.any()
            continue
        assert report["slot"] == t - delay
        sinr_reported_slot = wideband_db(before[t - delay][1]["privileged"]["sinr_prb"])
        assert report["sinr_wideband_db"] == pytest.approx(sinr_reported_slot, abs=1e-4)
        assert report["sinr_wideband_db"] != pytest.approx(sinr_t, abs=1e-4)
        assert report["ack"] == outcome[t - delay]["ack"]
        assert report["mcs"] == outcome[t - delay]["mcs"]
        reports[report["slot"]] = report

        expected = np.zeros(3 * num_reports, dtype=np.float32)
        for i in range(num_reports):
            slot = t - delay - i
            if slot >= 0:
                r = reports[slot]
                sinr = (np.clip(r["sinr_wideband_db"], -10.0, 40.0) + 10.0) / 25.0 - 1.0
                harq = 1.0 if r["ack"] else -1.0
                expected[3 * i : 3 * i + 3] = (sinr, harq, (r["mcs"] - 3) / 12.5 - 1.0)
        np.testing.assert_allclose(obs, expected, atol=1e-6)


def test_snr_is_sampled_per_episode_or_fixed():
    env = make()
    snrs = [env.reset(seed=seed)[1]["snr_db"] for seed in range(10)]
    assert all(5.0 <= s <= 20.0 for s in snrs)
    assert len(set(snrs)) == len(snrs)

    env = make(snr_db=12.5)
    assert all(env.reset(seed=seed)[1]["snr_db"] == 12.5 for seed in range(3))


def test_render_ansi():
    env = make(render_mode="ansi")
    env.reset(seed=0)
    assert isinstance(env.render(), str)
    env.step(11)
    line = env.render()
    assert "slot     0" in line and "MCS 14" in line and "running TBLER" in line

    assert LinkAdaptationEnv(episode_length=5).render() is None
    # SB3's make_vec_env passes render_mode="rgb_array" by default: warn, do not fail
    with pytest.warns(UserWarning, match="not supported"):
        env = LinkAdaptationEnv(episode_length=5, render_mode="rgb_array")
    env.reset(seed=0)
    assert env.render() is None
