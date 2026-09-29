"""Measure how many slots per second the linkgym simulator runs on the CPU.

Workloads:
    step  LinkSimulator.step with a fixed MCS (the episode's channel is generated
          before timing starts)
    olla  full OLLA loop via run_episode: one OLLA decision plus one step per slot

Each configuration runs --slots slots once as warm-up and then --repeats times; the
table reports the median. One step advances all B links by one slot, so
link-slots/s = B * steps/s. "x real time" compares link-slots/s with the
2000 slots/s of a real link at 30 kHz subcarrier spacing (0.5 ms slots).

    python examples/benchmark.py
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time

import numpy as np
import sionna
import torch
from sionna.sys import PHYAbstraction

from linkgym.sim import LinkSimulator, run_episode

FIXED_MCS = 14


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def time_workload(sim: LinkSimulator, workload: str, repeats: int, seed: int) -> float:
    """Median wall time [s] of one pass over the episode, after one warm-up pass."""

    def one_pass() -> float:
        sim.reset(seed)
        start = time.perf_counter()
        if workload == "step":
            for _ in range(sim.num_slots):
                sim.step(FIXED_MCS)
        else:
            run_episode(sim, "olla")
        return time.perf_counter() - start

    one_pass()
    return statistics.median(one_pass() for _ in range(repeats))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--slots", type=int, default=200, help="slots per timed run")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 256])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    default_threads = torch.get_num_threads()
    print("## Machine\n")
    print(f"- CPU: {cpu_name()} ({os.cpu_count()} logical CPUs)")
    print(f"- OS: {platform.platform()}")
    print(f"- Python {platform.python_version()}, torch {torch.__version__}, ", end="")
    print(f"sionna-no-rt {sionna.__version__}, numpy {np.__version__}")
    print(f"- torch default intra-op threads: {default_threads}")
    print(
        f"- {args.slots} slots per run, median of {args.repeats} runs after 1 warm-up run, "
        f"fixed MCS {FIXED_MCS} for the step workload\n"
    )

    phy = PHYAbstraction(device="cpu")
    thread_counts = sorted({1, default_threads})
    print("| workload | B | threads | ms/step | steps/s | link-slots/s | x real time |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for threads in thread_counts:
        torch.set_num_threads(threads)
        for batch_size in args.batch_sizes:
            sim = LinkSimulator(
                args.slots, batch_size=batch_size, seed=args.seed, phy_abstraction=phy
            )
            for workload in ("step", "olla"):
                seconds = time_workload(sim, workload, args.repeats, args.seed)
                steps_per_s = args.slots / seconds
                link_slots_per_s = batch_size * steps_per_s
                real_time = link_slots_per_s * sim.slot_duration
                print(
                    f"| {workload} | {batch_size} | {threads} | {1e3 / steps_per_s:.3f} | "
                    f"{steps_per_s:.0f} | {link_slots_per_s:.0f} | {real_time:.2f} |",
                    flush=True,
                )
    torch.set_num_threads(default_threads)


if __name__ == "__main__":
    main()
