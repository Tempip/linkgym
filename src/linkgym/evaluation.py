"""Evaluation of link adaptation policies on ``LinkAdaptation-v0``."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import gymnasium
import numpy as np

from linkgym.config import ENV_ID

METRICS = ("goodput_mbps", "observed_tbler", "mean_mcs")


def _metrics(bits: int, nacks: int, mcs_sum: int, slots: int, slot_duration: float) -> dict:
    return {
        "goodput_mbps": bits / (slots * slot_duration) / 1e6,
        "observed_tbler": nacks / slots,
        "mean_mcs": mcs_sum / slots,
    }


def evaluate(
    policy: Any,
    env_kwargs: dict[str, Any] | None = None,
    seeds: Iterable[int] = (0,),
    episodes_per_seed: int = 1,
) -> dict[str, Any]:
    """Run ``policy`` through ``gymnasium.make(ENV_ID, **env_kwargs)`` and summarize it.

    For each seed, the first episode is reset with that seed and the following
    ``episodes_per_seed - 1`` episodes reset without a seed, continuing its random
    stream. With the same seeds, every policy faces the same SNR, channel and ACK draws
    in each episode. The metrics of a seed cover all its slots; the result reports their
    mean and sample standard deviation (ddof=1; 0 for a single seed) across seeds.

    :param policy: Any object with ``act(obs, info) -> action``; ``reset()`` is called
        at the start of every episode if the object has it
    :param env_kwargs: Scenario fields passed to ``gymnasium.make``
    :param seeds: Seeds, one set of episodes each
    :param episodes_per_seed: Episodes per seed

    :output metrics: ``{"goodput_mbps": {"mean", "std"}, "observed_tbler": {...},
        "mean_mcs": {...}, "per_seed": [{"seed", "goodput_mbps", "observed_tbler",
        "mean_mcs"}, ...], "per_episode": [{"seed", "episode", "snr_db",
        "goodput_mbps", "observed_tbler", "mean_mcs"}, ...]}``; goodput counts TB
        information bits of ACKed slots and observed TBLER is the fraction of NACKed slots
    """
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must not be empty")
    if episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be >= 1")
    reset_policy = getattr(policy, "reset", None)

    env = gymnasium.make(ENV_ID, **(env_kwargs or {}))
    slot_duration = env.unwrapped.slot_duration
    per_seed, per_episode = [], []
    try:
        for seed in seeds:
            seed_totals = np.zeros(4, dtype=np.int64)  # bits, nacks, mcs_sum, slots
            for episode in range(episodes_per_seed):
                obs, info = env.reset(seed=seed if episode == 0 else None)
                snr_db = info["snr_db"]
                if reset_policy is not None:
                    reset_policy()
                totals = np.zeros(4, dtype=np.int64)
                done = False
                while not done:
                    obs, _, terminated, truncated, info = env.step(policy.act(obs, info))
                    done = terminated or truncated
                    totals += (info["bits"], not info["ack"], info["mcs"], 1)
                seed_totals += totals
                per_episode.append(
                    {
                        "seed": seed,
                        "episode": episode,
                        "snr_db": snr_db,
                        **_metrics(*totals.tolist(), slot_duration),
                    }
                )
            per_seed.append({"seed": seed, **_metrics(*seed_totals.tolist(), slot_duration)})
    finally:
        env.close()

    result: dict[str, Any] = {}
    for metric in METRICS:
        values = np.array([s[metric] for s in per_seed])
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result[metric] = {"mean": float(values.mean()), "std": std}
    result["per_seed"] = per_seed
    result["per_episode"] = per_episode
    return result


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    num_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Mean paired difference ``a - b`` with a percentile bootstrap confidence interval.

    The pairs (``a[i]``, ``b[i]``) are resampled with replacement. The relative
    difference is the ratio of means, ``100 * mean(a - b) / mean(b)`` [%], with its
    interval from the same resamples.

    :output stats: ``mean``, ``ci_low``, ``ci_high`` (units of ``a``) and ``pct``,
        ``pct_ci_low``, ``pct_ci_high`` [%]
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("a and b must be non-empty 1-D sequences of equal length")
    diff = a - b
    idx = np.random.default_rng(seed).integers(0, len(diff), size=(num_resamples, len(diff)))
    boot_diff = diff[idx].mean(axis=1)
    boot_pct = 100.0 * boot_diff / b[idx].mean(axis=1)
    tails = [50.0 * (1.0 - confidence), 50.0 * (1.0 + confidence)]
    ci_low, ci_high = np.percentile(boot_diff, tails)
    pct_low, pct_high = np.percentile(boot_pct, tails)
    return {
        "mean": float(diff.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "pct": float(100.0 * diff.mean() / b.mean()),
        "pct_ci_low": float(pct_low),
        "pct_ci_high": float(pct_high),
    }
