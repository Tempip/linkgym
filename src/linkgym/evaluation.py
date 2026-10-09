"""Evaluation of link adaptation policies on ``LinkAdaptation-v0``.

Stability: stable (docs/api_stability.md).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from typing import Any

import gymnasium
import numpy as np

from linkgym.config import ENV_ID

__all__ = [
    "METRICS",
    "cluster_bootstrap",
    "evaluate",
    "evaluate_episodes",
    "paired_bootstrap",
    "trace_episodes",
]

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


def trace_episodes(
    trace_path: str | os.PathLike,
    splits: Sequence[str],
    *,
    episode_length: int = 1000,
    windows: int | None = None,
    first_seed: int = 0,
) -> list[dict[str, Any]]:
    """Every trajectory of ``splits`` in a trace, cut into non-overlapping windows.

    Episode k is trajectory t of the file (in file order) at window w, starting at slot
    ``w * episode_length``, with reset seed ``first_seed + k``. Pass ``seed`` and
    ``options`` to ``env.reset`` (see :func:`evaluate_episodes`); with the same list, every
    policy faces the same channel, SNR and ACK draws in each episode.

    :param trace_path: A gain trace (format 1) or a path trace (format 2, :mod:`linkgym.paths`;
        pass the options to its source's ``generate`` as ``trajectories`` and ``offsets``)
    :param windows: Windows per trajectory; all that fit if `None`
    :output episodes: ``[{"seed", "options": {"trajectory", "offset"}, "trajectory",
        "window", "split", "group", "route", "route_trajectory", "category"}, ...]``;
        ``route`` is the route name from the generator's ``routes`` attribute (or the group)
        and ``route_trajectory`` the trajectory's rank among the route's trajectories
    """
    # torch: not imported with linkgym
    from linkgym.channels import inspect_trace
    from linkgym.paths import _is_path_trace, inspect_path_trace

    info = (
        inspect_path_trace(trace_path) if _is_path_trace(trace_path) else inspect_trace(trace_path)
    )
    fit = info.num_slots // episode_length
    windows = fit if windows is None else windows
    if not 1 <= windows <= fit:
        raise ValueError(f"windows must be in [1, {fit}] for episodes of {episode_length} slots")
    names = {r["group"]: r["name"] for r in json.loads(info.attrs.get("routes", "[]"))}
    episodes, rank = [], {}
    for t in np.flatnonzero(np.isin(info.split, list(splits))):
        group = None if info.group is None else int(info.group[t])
        rank[group] = rank.get(group, -1) + 1
        for w in range(windows):
            episodes.append(
                {
                    "seed": first_seed + len(episodes),
                    "options": {"trajectory": int(t), "offset": w * episode_length},
                    "trajectory": int(t),
                    "window": w,
                    "split": str(info.split[t]),
                    "group": group,
                    "route": names.get(group, str(group)),
                    "route_trajectory": rank[group],
                    "category": None if info.category is None else str(info.category[t]),
                }
            )
    return episodes


def evaluate_episodes(
    policy: Any, env_kwargs: dict[str, Any], episodes: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run ``policy`` on given episodes: ``env.reset(seed=e["seed"], options=e["options"])``.

    :param episodes: E.g. from :func:`trace_episodes`; other keys are copied to the result
    :output per_episode: one dict per episode: its keys (without ``options``) and
        ``snr_db``, ``goodput_mbps``, ``observed_tbler``, ``mean_mcs``
    """
    reset_policy = getattr(policy, "reset", None)
    env = gymnasium.make(ENV_ID, **env_kwargs)
    slot_duration = env.unwrapped.slot_duration
    results = []
    try:
        for episode in episodes:
            obs, info = env.reset(seed=episode["seed"], options=episode.get("options"))
            snr_db = info["snr_db"]
            if reset_policy is not None:
                reset_policy()
            totals = np.zeros(4, dtype=np.int64)  # bits, nacks, mcs_sum, slots
            done = False
            while not done:
                obs, _, terminated, truncated, info = env.step(policy.act(obs, info))
                done = terminated or truncated
                totals += (info["bits"], not info["ack"], info["mcs"], 1)
            row = {k: v for k, v in episode.items() if k != "options"}
            results.append(row | {"snr_db": snr_db, **_metrics(*totals.tolist(), slot_duration)})
    finally:
        env.close()
    return results


def cluster_bootstrap(
    a: Sequence[float],
    b: Sequence[float] | None,
    clusters: Sequence[Any],
    *,
    num_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Mean of ``a - b`` (or of ``a``) with a cluster bootstrap confidence interval.

    Clusters (e.g. trajectories, each with several episodes) are resampled with
    replacement and keep all their elements; the statistic is the mean over the elements
    of the resample, so every element weighs the same. With ``b``, the relative
    difference is the ratio of means ``100 * mean(a - b) / mean(b)`` [%]. With a single
    cluster the interval is undefined (NaN).

    :output stats: ``mean``, ``ci_low``, ``ci_high`` and, with ``b``, ``pct``,
        ``pct_ci_low``, ``pct_ci_high``; ``num_clusters``, ``num_elements``
    """
    paired = b is not None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64) if paired else np.zeros_like(a)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0 or len(clusters) != len(a):
        raise ValueError("a, b and clusters must be non-empty 1-D sequences of equal length")
    labels, cluster_of = np.unique(np.asarray([str(c) for c in clusters]), return_inverse=True)
    n = len(labels)
    count = np.bincount(cluster_of, minlength=n).astype(np.float64)
    diff_sum = np.bincount(cluster_of, weights=a - b, minlength=n)
    b_sum = np.bincount(cluster_of, weights=b, minlength=n)
    diff = a - b
    result = {
        "mean": float(diff.mean()),
        "num_clusters": int(n),
        "num_elements": int(len(a)),
    }
    if paired:
        result["pct"] = float(100.0 * diff.mean() / b.mean())
    tails = [50.0 * (1.0 - confidence), 50.0 * (1.0 + confidence)]
    if n < 2:
        result |= {"ci_low": float("nan"), "ci_high": float("nan")}
        if paired:
            result |= {"pct_ci_low": float("nan"), "pct_ci_high": float("nan")}
        return result
    idx = np.random.default_rng(seed).integers(0, n, size=(num_resamples, n))
    size = count[idx].sum(axis=1)
    boot = diff_sum[idx].sum(axis=1) / size
    result["ci_low"], result["ci_high"] = (float(v) for v in np.percentile(boot, tails))
    if paired:
        boot_pct = 100.0 * boot / (b_sum[idx].sum(axis=1) / size)
        result["pct_ci_low"], result["pct_ci_high"] = (
            float(v) for v in np.percentile(boot_pct, tails)
        )
    return result
