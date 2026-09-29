"""Run link adaptation baselines on the linkgym simulator and summarize the results.

Single run:
    python examples/baseline.py --policy olla --slots 2000 --seed 0
    python examples/baseline.py --policy fixed --mcs 14

Grid used in docs/benchmarks.md (prints a markdown table):
    python examples/baseline.py --grid
"""

from __future__ import annotations

import argparse
import itertools

import torch
from sionna.sys import PHYAbstraction

from linkgym.sim import MAX_MCS, MIN_MCS, POLICIES, LinkResult, LinkSimulator, run_episode

GRID_SNR_DB = (5.0, 15.0)
GRID_SPEED = (3.0, 15.0)
GRID_DELAY = (1, 4)
GRID_FIXED_MCS = (5, 14, 24)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--slots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=POLICIES, default="olla")
    parser.add_argument(
        "--mcs",
        type=int,
        choices=range(MIN_MCS, MAX_MCS + 1),
        metavar=f"{{{MIN_MCS}..{MAX_MCS}}}",
        help="MCS index for --policy fixed",
    )
    parser.add_argument("--bler-target", type=float, default=0.1, help="TBLER target")
    parser.add_argument("--snr-db", type=float, default=15.0, help="mean SNR [dB]")
    parser.add_argument("--speed", type=float, default=3.0, help="UE speed [m/s]")
    parser.add_argument("--feedback-delay", type=int, default=1, help="[slots], ILLA and OLLA")
    parser.add_argument("--tdl-model", default="A")
    parser.add_argument("--delay-spread", type=float, default=100e-9, help="[s]")
    parser.add_argument("--carrier-frequency", type=float, default=3.5e9, help="[Hz]")
    parser.add_argument(
        "--grid", action="store_true", help="run the docs/benchmarks.md grid instead"
    )
    args = parser.parse_args()
    if args.policy == "fixed" and args.mcs is None and not args.grid:
        parser.error("--policy fixed requires --mcs")
    return args


def simulate(
    args: argparse.Namespace,
    phy: PHYAbstraction,
    *,
    policy: str,
    snr_db: float,
    speed: float,
    feedback_delay: int,
    mcs: int | None = None,
) -> tuple[LinkResult, float]:
    """Run one episode; return the result and the slot duration [s]."""
    sim = LinkSimulator(
        args.slots,
        seed=args.seed,
        snr_db=snr_db,
        speed=speed,
        tdl_model=args.tdl_model,
        delay_spread=args.delay_spread,
        carrier_frequency=args.carrier_frequency,
        phy_abstraction=phy,
    )
    result = run_episode(
        sim, policy, bler_target=args.bler_target, feedback_delay=feedback_delay, fixed_mcs=mcs
    )
    return result, sim.slot_duration


def summarize(result: LinkResult, slot_duration: float) -> dict[str, float]:
    bits = result.decoded_bits.double().mean().item()
    return {
        "bits_per_slot": bits,
        "mbps": bits / slot_duration / 1e6,
        "observed_tbler": 1.0 - result.ack.double().mean().item(),
        "mean_tbler": result.tbler.double().mean().item(),
        "mean_mcs": result.mcs.double().mean().item(),
    }


def mcs_summary(mcs: torch.Tensor) -> str:
    counts = torch.bincount(mcs.flatten().long(), minlength=MAX_MCS + 1)
    total = counts.sum().item()
    top = torch.argsort(counts, descending=True, stable=True)[:5].tolist()
    most_used = ", ".join(f"{i} ({100 * counts[i] / total:.1f}%)" for i in top if counts[i] > 0)
    histogram = " ".join(f"{i}:{c}" for i, c in enumerate(counts.tolist()) if c > 0)
    return (
        f"MCS: mean {mcs.double().mean():.2f}, min {mcs.min()}, max {mcs.max()}; "
        f"most used {most_used}\n"
        f"MCS histogram (mcs:slots): {histogram}"
    )


def delay_label(policy: str, delay: int) -> str:
    return {"oracle": "0", "fixed": "-"}.get(policy, str(delay))


def run_single(args: argparse.Namespace, phy: PHYAbstraction) -> None:
    result, slot_duration = simulate(
        args,
        phy,
        policy=args.policy,
        snr_db=args.snr_db,
        speed=args.speed,
        feedback_delay=args.feedback_delay,
        mcs=args.mcs,
    )
    s = summarize(result, slot_duration)
    policy = f"fixed (MCS {args.mcs})" if args.policy == "fixed" else args.policy
    print(
        f"policy={policy} snr_db={args.snr_db} speed={args.speed} m/s "
        f"feedback_delay={delay_label(args.policy, args.feedback_delay)} "
        f"tdl={args.tdl_model} delay_spread={args.delay_spread:g} s "
        f"fc={args.carrier_frequency / 1e9:g} GHz slots={args.slots} seed={args.seed}"
    )
    print(f"goodput: {s['bits_per_slot']:.1f} bits/slot = {s['mbps']:.2f} Mbit/s")
    target = "" if args.policy == "fixed" else f", target {args.bler_target:g}"
    print(f"TBLER: observed {s['observed_tbler']:.4f} (mean tbler {s['mean_tbler']:.4f}){target}")
    print(mcs_summary(result.mcs))


def run_grid(args: argparse.Namespace, phy: PHYAbstraction) -> None:
    runs = []
    for snr_db, speed in itertools.product(GRID_SNR_DB, GRID_SPEED):
        runs.append(("oracle", snr_db, speed, 1, None))
        for delay, policy in itertools.product(GRID_DELAY, ("illa", "olla")):
            runs.append((policy, snr_db, speed, delay, None))
    for mcs in GRID_FIXED_MCS:
        runs.append(("fixed", args.snr_db, args.speed, 1, mcs))

    print(
        f"slots={args.slots} seed={args.seed} tbler_target={args.bler_target:g} "
        f"tdl={args.tdl_model} delay_spread={args.delay_spread:g} s "
        f"fc={args.carrier_frequency / 1e9:g} GHz\n"
    )
    print("| policy | snr_db | speed | delay | goodput Mbit/s | observed TBLER | mean MCS |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for policy, snr_db, speed, delay, mcs in runs:
        result, slot_duration = simulate(
            args, phy, policy=policy, snr_db=snr_db, speed=speed, feedback_delay=delay, mcs=mcs
        )
        s = summarize(result, slot_duration)
        name = f"fixed (MCS {mcs})" if policy == "fixed" else policy
        print(
            f"| {name} | {snr_db:g} | {speed:g} | {delay_label(policy, delay)} | "
            f"{s['mbps']:.2f} | {s['observed_tbler']:.4f} | {s['mean_mcs']:.2f} |",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    phy = PHYAbstraction(device="cpu")
    if args.grid:
        run_grid(args, phy)
    else:
        run_single(args, phy)


if __name__ == "__main__":
    main()
