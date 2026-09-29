"""Evaluation of link adaptation policies on ``LinkAdaptation-v0``."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import gymnasium
import numpy as np

from linkgym.config import ENV_ID

METRICS = ("goodput_mbps", "observed_tbler", "mean_mcs")


def evaluate(
    policy: Any,
    env_kwargs: dict[str, Any] | None = None,
    seeds: Iterable[int] = (0,),
    episodes_per_seed: int = 1,
) -> dict[str, Any]:
    """Run ``policy`` through ``gymnasium.make(ENV_ID, **env_kwargs)`` and summarize it.

    For each seed, the first episode is reset with that seed and the following
    ``episodes_per_seed - 1`` episodes reset without a seed, continuing its random
    stream. The metrics of a seed cover all its slots; the result reports their mean
    and sample standard deviation (ddof=1; 0 for a single seed) across seeds.

    :param policy: Any object with ``act(obs, info) -> action``; ``reset()`` is called
        at the start of every episode if the object has it
    :param env_kwargs: Scenario fields passed to ``gymnasium.make``
    :param seeds: Seeds, one set of episodes each
    :param episodes_per_seed: Episodes per seed

    :output metrics: ``{"goodput_mbps": {"mean", "std"}, "observed_tbler": {...},
        "mean_mcs": {...}, "per_seed": [{"seed", "goodput_mbps", "observed_tbler",
        "mean_mcs"}, ...]}``; goodput counts TB information bits of ACKed slots and
        observed TBLER is the fraction of NACKed slots
    """
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must not be empty")
    if episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be >= 1")
    reset_policy = getattr(policy, "reset", None)

    env = gymnasium.make(ENV_ID, **(env_kwargs or {}))
    per_seed = []
    try:
        for seed in seeds:
            bits = nacks = mcs_sum = slots = 0
            for episode in range(episodes_per_seed):
                obs, info = env.reset(seed=seed if episode == 0 else None)
                if reset_policy is not None:
                    reset_policy()
                done = False
                while not done:
                    obs, _, terminated, truncated, info = env.step(policy.act(obs, info))
                    done = terminated or truncated
                    bits += info["bits"]
                    nacks += not info["ack"]
                    mcs_sum += info["mcs"]
                    slots += 1
            per_seed.append(
                {
                    "seed": seed,
                    "goodput_mbps": bits / (slots * env.unwrapped.slot_duration) / 1e6,
                    "observed_tbler": nacks / slots,
                    "mean_mcs": mcs_sum / slots,
                }
            )
    finally:
        env.close()

    result: dict[str, Any] = {}
    for metric in METRICS:
        values = np.array([s[metric] for s in per_seed])
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result[metric] = {"mean": float(values.mean()), "std": std}
    result["per_seed"] = per_seed
    return result
