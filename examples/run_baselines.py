"""Evaluate the link adaptation baselines on LinkAdaptation-v0 and print a markdown table.

Runs OLLA (TBLER targets 0.05, 0.1, 0.2), ILLA (0.1), fixed MCS 5, 14, 24 and the oracle
(0.1) on the default scenario, 5 seeds x 2 episodes each.

    python examples/run_baselines.py
"""

from __future__ import annotations

import argparse
import time

import torch
from sionna.sys import PHYAbstraction

from linkgym import ScenarioConfig, evaluate
from linkgym.baselines import FixedMCSPolicy, ILLAPolicy, OLLAPolicy, OraclePolicy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes-per-seed", type=int, default=2)
    args = parser.parse_args()

    # One thread is faster for this workload (docs/benchmarks.md)
    torch.set_num_threads(1)

    phy = PHYAbstraction(device="cpu")
    policies = [
        ("olla (target 0.05)", OLLAPolicy(0.05, phy_abstraction=phy)),
        ("olla (target 0.1)", OLLAPolicy(0.1, phy_abstraction=phy)),
        ("olla (target 0.2)", OLLAPolicy(0.2, phy_abstraction=phy)),
        ("illa (target 0.1)", ILLAPolicy(0.1, phy_abstraction=phy)),
        ("fixed (MCS 5)", FixedMCSPolicy(5)),
        ("fixed (MCS 14)", FixedMCSPolicy(14)),
        ("fixed (MCS 24)", FixedMCSPolicy(24)),
        ("oracle (target 0.1)", OraclePolicy(0.1, phy_abstraction=phy)),
    ]

    c = ScenarioConfig()
    print(
        f"Default scenario: episode_length={c.episode_length}, snr_db ~ U{c.snr_db_range} "
        f"per episode, speed={c.speed} m/s, feedback_delay={c.feedback_delay}, "
        f"TDL-{c.tdl_model}. Seeds {args.seeds}, {args.episodes_per_seed} episodes per seed; "
        f"mean ± std (ddof=1) across seeds.\n"
    )
    print("| policy | goodput Mbit/s | observed TBLER | mean MCS |")
    print("|---|---:|---:|---:|")
    start = time.perf_counter()
    for name, policy in policies:
        r = evaluate(policy, seeds=args.seeds, episodes_per_seed=args.episodes_per_seed)
        goodput, tbler, mcs = r["goodput_mbps"], r["observed_tbler"], r["mean_mcs"]
        print(
            f"| {name} | {goodput['mean']:.2f} ± {goodput['std']:.2f} | "
            f"{tbler['mean']:.4f} ± {tbler['std']:.4f} | {mcs['mean']:.2f} ± {mcs['std']:.2f} |",
            flush=True,
        )
    print(f"\nTotal time: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
