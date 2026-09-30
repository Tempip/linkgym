import gymnasium
import numpy as np
import pytest
from sionna.sys import PHYAbstraction

from linkgym import ENV_ID, evaluate
from linkgym.baselines import FixedMCSPolicy, ILLAPolicy, OLLAPolicy, OraclePolicy, SB3Policy
from linkgym.sim import MAX_MCS, MIN_MCS


@pytest.fixture(scope="module")
def phy():
    return PHYAbstraction(device="cpu")


POLICIES = {
    "fixed": lambda phy: FixedMCSPolicy(14),
    "illa": lambda phy: ILLAPolicy(0.1, phy_abstraction=phy),
    "olla": lambda phy: OLLAPolicy(0.1, phy_abstraction=phy),
    "oracle": lambda phy: OraclePolicy(0.1, phy_abstraction=phy),
}


def _assert_valid(result, num_seeds):
    assert len(result["per_seed"]) == num_seeds
    for s in result["per_seed"]:
        assert s["goodput_mbps"] >= 0.0
        assert 0.0 <= s["observed_tbler"] <= 1.0
        assert MIN_MCS <= s["mean_mcs"] <= MAX_MCS
    for metric in ("goodput_mbps", "observed_tbler", "mean_mcs"):
        values = [s[metric] for s in result["per_seed"]]
        assert result[metric]["mean"] == pytest.approx(np.mean(values))
        expected_std = np.std(values, ddof=1) if num_seeds > 1 else 0.0
        assert result[metric]["std"] == pytest.approx(expected_std)


@pytest.mark.parametrize("name", list(POLICIES))
def test_baseline_runs_full_episodes(phy, name):
    result = evaluate(POLICIES[name](phy), {"episode_length": 100}, seeds=[0, 1])
    _assert_valid(result, num_seeds=2)


class _DropPrivileged(gymnasium.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        del info["privileged"]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        del info["privileged"]
        return obs, reward, terminated, truncated, info


@pytest.mark.parametrize("name", ["fixed", "illa", "olla"])
def test_non_oracle_baselines_do_not_read_privileged(phy, name):
    env = _DropPrivileged(gymnasium.make(ENV_ID, episode_length=50))
    policy = POLICIES[name](phy)
    obs, info = env.reset(seed=0)
    policy.reset()
    truncated = False
    while not truncated:
        obs, _, _, truncated, info = env.step(policy.act(obs, info))


def test_olla_tracks_its_target(phy):
    result = evaluate(
        OLLAPolicy(0.1, phy_abstraction=phy), {"snr_db": 15.0, "episode_length": 2000}
    )
    assert result["observed_tbler"]["mean"] == pytest.approx(0.1, abs=0.01)


def test_olla_delta_up_reaches_sionna(phy):
    policy = OLLAPolicy(0.2, delta_up=0.25, phy_abstraction=phy)
    assert policy._olla.delta_up == 0.25
    assert policy._olla.delta_down == pytest.approx(0.25 * 0.2 / 0.8)
    assert OLLAPolicy(0.1, phy_abstraction=phy)._olla.delta_up == 1.0


def test_evaluate_accepts_any_object_with_act():
    class Constant:
        def act(self, obs, info):
            return 0

    result = evaluate(Constant(), {"episode_length": 30}, seeds=[0])
    _assert_valid(result, num_seeds=1)
    assert result["mean_mcs"]["mean"] == MIN_MCS


def test_sb3_adapter():
    sb3 = pytest.importorskip("stable_baselines3")
    model = sb3.PPO("MlpPolicy", gymnasium.make(ENV_ID, episode_length=30), seed=0, device="cpu")
    result = evaluate(SB3Policy(model), {"episode_length": 30}, seeds=[0, 1])
    _assert_valid(result, num_seeds=2)
