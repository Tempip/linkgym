"""Train PPO (Stable-Baselines3) on linkgym/LinkAdaptation-v0.

The default scenario is used for training and validation. Training envs run in
SubprocVecEnv workers (spawn), one torch thread each. A callback evaluates the
deterministic policy on fixed validation seeds every --eval-freq steps (learning curve in
eval_log.csv and TensorBoard). The output directory gets model.zip, run.json (config,
versions, git commit, timing) and tensorboard/.

Seeds: training env i of training seed s uses seed s * n_envs + i (must stay below 500),
validation uses 500-509 and the held-out test seeds are 1000 and above.

    python examples/train_ppo.py --gamma 0.9 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gymnasium
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from linkgym import ENV_ID, ScenarioConfig, evaluate
from linkgym.baselines import SB3Policy

VALIDATION_SEEDS = list(range(500, 510))
FIRST_VALIDATION_SEED = 500
FIRST_TEST_SEED = 1000
NET_ARCH = {"pi": [64, 64], "vf": [64, 64]}
PACKAGES = ("linkgym", "sionna-no-rt", "torch", "gymnasium", "stable-baselines3", "numpy")
REPO = Path(__file__).resolve().parents[1]


def make_env(**env_kwargs: Any) -> gymnasium.Env:
    """Env factory, called inside each SubprocVecEnv worker."""
    torch.set_num_threads(1)
    import linkgym  # noqa: F401  registers the env in the worker process

    return gymnasium.make(ENV_ID, **env_kwargs)


class ValidationCallback(BaseCallback):
    """Evaluates the deterministic policy on fixed seeds at the start, every ``eval_freq``
    steps and at the end; logs to TensorBoard and to a CSV file."""

    def __init__(
        self, seeds: list[int], eval_freq: int, env_kwargs: dict[str, Any], csv_path: Path
    ) -> None:
        super().__init__()
        self.seeds = seeds
        self.eval_freq = eval_freq
        self.env_kwargs = env_kwargs
        self.csv_path = csv_path
        self.eval_seconds = 0.0
        self.rows: list[dict[str, float]] = []
        self._next_eval = 0

    def _evaluate(self) -> None:
        start = time.perf_counter()
        result = evaluate(SB3Policy(self.model), self.env_kwargs, self.seeds, 1)
        seconds = time.perf_counter() - start
        self.eval_seconds += seconds
        row = {
            "timesteps": self.num_timesteps,
            "goodput_mbps": result["goodput_mbps"]["mean"],
            "goodput_mbps_std": result["goodput_mbps"]["std"],
            "observed_tbler": result["observed_tbler"]["mean"],
            "mean_mcs": result["mean_mcs"]["mean"],
            "eval_seconds": seconds,
        }
        self.rows.append(row)
        for key in ("goodput_mbps", "observed_tbler", "mean_mcs"):
            self.logger.record(f"eval/{key}", row[key])
        self.logger.dump(self.num_timesteps)
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"[eval] steps {self.num_timesteps:>8d}  goodput {row['goodput_mbps']:6.2f} Mbit/s  "
            f"TBLER {row['observed_tbler']:.4f}  mean MCS {row['mean_mcs']:5.2f}  "
            f"({seconds:.1f} s)",
            flush=True,
        )

    def _on_training_start(self) -> None:
        self._evaluate()
        self._next_eval = self.eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._evaluate()
            while self._next_eval <= self.num_timesteps:
                self._next_eval += self.eval_freq
        return True

    def _on_training_end(self) -> None:
        if self.rows[-1]["timesteps"] != self.num_timesteps:
            self._evaluate()


def git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0, help="training seed")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=VALIDATION_SEEDS)
    parser.add_argument(
        "--episode-length", type=int, default=None, help="scenario override (quick tests only)"
    )
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = REPO / "runs" / "m3" / f"gamma{args.gamma:g}_seed{args.seed}"
    return args


def main() -> None:
    args = parse_args()
    # The workers fill the physical cores; one thread is also faster for this small MLP
    torch.set_num_threads(1)

    env_kwargs = {} if args.episode_length is None else {"episode_length": args.episode_length}
    first_env_seed = args.seed * args.n_envs
    train_env_seeds = list(range(first_env_seed, first_env_seed + args.n_envs))
    if train_env_seeds[-1] >= FIRST_VALIDATION_SEED:
        raise SystemExit(f"training env seeds must stay below {FIRST_VALIDATION_SEED}")
    if not all(FIRST_VALIDATION_SEED <= s < FIRST_TEST_SEED for s in args.eval_seeds):
        raise SystemExit(f"eval seeds must be in [{FIRST_VALIDATION_SEED}, {FIRST_TEST_SEED})")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_csv = out_dir / "eval_log.csv"
    eval_csv.unlink(missing_ok=True)

    vec_env = make_vec_env(
        make_env,
        n_envs=args.n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        gamma=args.gamma,
        policy_kwargs={"net_arch": NET_ARCH},
        tensorboard_log=str(out_dir / "tensorboard"),
        seed=args.seed,
        device="cpu",
    )
    # PPO(seed=s) seeds env i with s + i (on_policy_algorithm.py:117), which overlaps across
    # training seeds; use disjoint seeds instead. They apply at the first reset in learn().
    model.get_env().seed(first_env_seed)

    callback = ValidationCallback(args.eval_seeds, args.eval_freq, env_kwargs, eval_csv)
    start = time.perf_counter()
    model.learn(total_timesteps=args.total_timesteps, callback=callback, tb_log_name="ppo")
    wall_seconds = time.perf_counter() - start
    vec_env.close()
    model.save(out_dir / "model")

    steps = model.num_timesteps
    train_seconds = wall_seconds - callback.eval_seconds
    run = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "ppo": {
            "policy": "MlpPolicy",
            "policy_kwargs": model.policy_kwargs,
            "learning_rate": model.learning_rate,
            "n_steps": model.n_steps,
            "batch_size": model.batch_size,
            "n_epochs": model.n_epochs,
            "gamma": model.gamma,
            "gae_lambda": model.gae_lambda,
            "clip_range": model.clip_range(1.0),
            "ent_coef": model.ent_coef,
            "vf_coef": model.vf_coef,
            "max_grad_norm": model.max_grad_norm,
            "normalize_advantage": model.normalize_advantage,
            "target_kl": model.target_kl,
        },
        "scenario": asdict(ScenarioConfig(**env_kwargs)),
        "seeds": {
            "training_seed": args.seed,
            "train_env_seeds": train_env_seeds,
            "validation_seeds": args.eval_seeds,
            "first_test_seed": FIRST_TEST_SEED,
        },
        "versions": {pkg: version(pkg) for pkg in PACKAGES} | {"python": platform.python_version()},
        "git": git_state(),
        "machine": {
            "processor": platform.processor(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "timing": {
            "total_timesteps": steps,
            "wall_seconds": wall_seconds,
            "validation_seconds": callback.eval_seconds,
            "steps_per_second": steps / wall_seconds,
            "training_steps_per_second": steps / train_seconds,
        },
        "final_validation": callback.rows[-1],
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(
        f"done: {steps} steps in {wall_seconds:.0f} s ({steps / wall_seconds:.0f} steps/s, "
        f"{steps / train_seconds:.0f} steps/s without validation); saved to {out_dir}"
    )


if __name__ == "__main__":
    main()
