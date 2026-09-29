import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from linkgym import ENV_ID, evaluate
from linkgym.baselines import FixedMCSPolicy, SB3Policy
from linkgym.evaluation import paired_bootstrap
from linkgym.sim import MAX_MCS, MIN_MCS

REPO = Path(__file__).resolve().parents[1]


def _assert_valid(result, num_seeds):
    assert len(result["per_seed"]) == num_seeds
    for s in result["per_seed"]:
        assert s["goodput_mbps"] >= 0.0
        assert 0.0 <= s["observed_tbler"] <= 1.0
        assert MIN_MCS <= s["mean_mcs"] <= MAX_MCS


# make_vec_env with an env id passes render_mode="rgb_array", which only warns
@pytest.mark.filterwarnings("ignore:.*render_mode='rgb_array'")
def test_ppo_trains_saves_and_loads(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_util import make_vec_env

    env_kwargs = {"episode_length": 64}
    vec_env = make_vec_env(ENV_ID, n_envs=2, seed=0, env_kwargs=env_kwargs)
    model = sb3.PPO(
        "MlpPolicy",
        vec_env,
        n_steps=128,
        batch_size=64,
        n_epochs=2,
        gamma=0.0,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        seed=0,
        device="cpu",
    )
    model.learn(total_timesteps=2048)
    assert model.num_timesteps >= 2048
    model.save(tmp_path / "model")
    vec_env.close()

    loaded = sb3.PPO.load(tmp_path / "model.zip", device="cpu")
    obs = np.random.default_rng(0).uniform(-1, 1, size=(16, 12)).astype(np.float32)
    np.testing.assert_array_equal(
        model.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0]
    )
    result = evaluate(SB3Policy(loaded), env_kwargs, seeds=[1000, 1001])
    _assert_valid(result, num_seeds=2)


def test_evaluate_reports_every_episode():
    result = evaluate(FixedMCSPolicy(14), {"episode_length": 30}, seeds=[0, 1], episodes_per_seed=2)
    episodes = result["per_episode"]
    assert [(e["seed"], e["episode"]) for e in episodes] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for seed_result in result["per_seed"]:
        own = [e for e in episodes if e["seed"] == seed_result["seed"]]
        # Episodes have the same length, so the seed metrics are episode means
        for metric in ("goodput_mbps", "observed_tbler", "mean_mcs"):
            assert seed_result[metric] == pytest.approx(np.mean([e[metric] for e in own]))
        assert all(5.0 <= e["snr_db"] <= 20.0 for e in own)


def test_paired_bootstrap():
    b = np.random.default_rng(1).uniform(10.0, 20.0, size=50)
    shifted = paired_bootstrap(b + 1.0, b)
    assert shifted["mean"] == pytest.approx(1.0)
    assert shifted["ci_low"] == pytest.approx(1.0) and shifted["ci_high"] == pytest.approx(1.0)
    assert shifted["pct"] == pytest.approx(100.0 / b.mean())

    a = b + np.random.default_rng(2).normal(0.5, 2.0, size=50)
    first, second = paired_bootstrap(a, b, seed=7), paired_bootstrap(a, b, seed=7)
    assert first == second
    assert first["ci_low"] < first["mean"] < first["ci_high"]
    assert first["pct_ci_low"] < first["pct"] < first["pct_ci_high"]
    assert first != paired_bootstrap(a, b, seed=8)


@pytest.mark.slow
def test_train_ppo_script_end_to_end(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    out_dir = tmp_path / "run"
    cmd = [
        sys.executable,
        str(REPO / "examples" / "train_ppo.py"),
        "--gamma", "0.9",
        "--seed", "1",
        "--total-timesteps", "2048",
        "--n-envs", "2",
        "--eval-freq", "2048",
        "--eval-seeds", "500",
        "--episode-length", "64",
        "--out-dir", str(out_dir),
    ]  # fmt: skip
    subprocess.run(cmd, check=True, cwd=REPO, timeout=600)

    run = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert run["ppo"]["gamma"] == 0.9
    assert run["ppo"]["policy_kwargs"]["net_arch"] == {"pi": [64, 64], "vf": [64, 64]}
    assert run["seeds"]["train_env_seeds"] == [2, 3]
    assert set(run["versions"]) >= {"linkgym", "sionna-no-rt", "torch", "stable-baselines3"}
    assert len(run["git"]["commit"]) == 40
    assert run["timing"]["steps_per_second"] > 0
    rows = (out_dir / "eval_log.csv").read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].startswith("timesteps,goodput_mbps")
    assert len(rows) >= 3  # header, start of training, end of training
    assert list((out_dir / "tensorboard").rglob("events.out.tfevents.*"))

    model = sb3.PPO.load(out_dir / "model.zip", device="cpu")
    result = evaluate(SB3Policy(model), {"episode_length": 64}, seeds=[1000])
    _assert_valid(result, num_seeds=1)
