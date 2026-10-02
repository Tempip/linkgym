"""Statistics of a generated trace: per route, per split, path gain and link-budget SNR.

    python examples/rt/dataset_stats.py data/munich-v1.h5 --json stats.json

Prints Markdown tables. Path gain is the wideband gain of a slot (mean linear gain over the
PRBs) in dB. SNR uses snr_mode="link_budget" with a 7 dB noise figure over the allocated
bandwidth; "per episode" is the realized mean SNR of 1000-slot windows (the default
episode length), taken without overlap.
"""

import argparse
import json
from collections import Counter

import h5py
import numpy as np

from linkgym.channels import inspect_trace, link_budget_snr_db

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
TX_POWERS_DBM = (20.0, 30.0)
EPISODE_SLOTS = 1000


def db(x):
    return 10 * np.log10(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace")
    parser.add_argument("--json", help="also write the statistics as JSON")
    parser.add_argument("--noise-figure-db", type=float, default=7.0)
    args = parser.parse_args()

    info = inspect_trace(args.trace)
    with h5py.File(args.trace, "r") as f:
        routes = json.loads(f.attrs["routes"])
        los = f["los"][()]
        wideband = np.stack([f["gain"][t].mean(axis=1, dtype=np.float64) for t in range(len(los))])
    bandwidth = info.num_prbs * 12 * info.attrs["subcarrier_spacing_hz"]

    stats = {"routes": [], "splits": {}, "path_gain_db": {}, "snr": {}}
    print("| route | split | category | trajectories | dropped | LoS share | mean path gain |")
    print("|---|---|---|---:|---:|---:|---:|")
    for r in routes:
        idx = np.flatnonzero(info.group == r["group"])
        entry = {
            "name": r["name"],
            "split": r["split"],
            "category": r["category"],
            "trajectories": int(len(idx)),
            "dropped": r["dropped"],
            "los_share": float(los[idx].mean()) if len(idx) else None,
            "mean_path_gain_db": float(db(wideband[idx].mean())) if len(idx) else None,
        }
        stats["routes"].append(entry)
        dropped = ", ".join(
            f"{d['reason'].replace('_', ' ')} ({d['mean_gain_db']} dB)" for d in r["dropped"]
        )
        share = "-" if entry["los_share"] is None else f"{entry['los_share']:.2f}"
        gain = "-" if entry["mean_path_gain_db"] is None else f"{entry['mean_path_gain_db']:.1f} dB"
        print(
            f"| {r['name']} | {r['split']} | {r['category']} | {len(idx)} | "
            f"{len(r['dropped'])}{': ' + dropped if dropped else ''} | {share} | {gain} |"
        )

    print("\n| split | routes | trajectories | LoS share | categories (routes) |")
    print("|---|---:|---:|---:|---|")
    for split in ("train", "val", "test"):
        idx = np.flatnonzero(info.split == split)
        split_routes = [e for e in stats["routes"] if e["split"] == split]
        mix = Counter(e["category"] for e in split_routes if e["trajectories"])
        entry = {
            "routes": len(split_routes),
            "routes_with_trajectories": sum(e["trajectories"] > 0 for e in split_routes),
            "trajectories": int(len(idx)),
            "los_share": float(los[idx].mean()),
            "categories": dict(mix),
        }
        stats["splits"][split] = entry
        mix_text = ", ".join(f"{k} {v}" for k, v in sorted(mix.items()))
        print(
            f"| {split} | {entry['routes']} | {entry['trajectories']} | "
            f"{entry['los_share']:.2f} | {mix_text} |"
        )

    gain_db = db(wideband.ravel())
    stats["path_gain_db"] = {f"p{p}": float(np.percentile(gain_db, p)) for p in PERCENTILES}
    print(
        "\nPath gain per slot [dB]: "
        + ", ".join(f"p{p} {np.percentile(gain_db, p):.1f}" for p in PERCENTILES)
    )

    windows = wideband[:, : wideband.shape[1] // EPISODE_SLOTS * EPISODE_SLOTS]
    episode_gain_db = db(windows.reshape(len(wideband), -1, EPISODE_SLOTS).mean(axis=2)).ravel()
    print(
        "\n| tx power | reference SNR | per slot p5 / p50 / p95 | per episode p5 / p50 / p95 | "
        "episodes < -5 dB | > 20 dB | > 40 dB |"
    )
    print("|---:|---:|---|---|---:|---:|---:|")
    for power in TX_POWERS_DBM:
        ref = link_budget_snr_db(power, args.noise_figure_db, bandwidth)
        slot, episode = gain_db + ref, episode_gain_db + ref
        entry = {
            "reference_snr_db": ref,
            "per_slot": {f"p{p}": float(np.percentile(slot, p)) for p in PERCENTILES},
            "per_episode": {f"p{p}": float(np.percentile(episode, p)) for p in PERCENTILES},
            "episodes_below_minus5_db": float(np.mean(episode < -5)),
            "episodes_above_20_db": float(np.mean(episode > 20)),
            "episodes_above_40_db": float(np.mean(episode > 40)),
        }
        stats["snr"][f"{power:g}_dbm"] = entry
        s5, s50, s95 = (np.percentile(slot, p) for p in (5, 50, 95))
        e5, e50, e95 = (np.percentile(episode, p) for p in (5, 50, 95))
        print(
            f"| {power:g} dBm | {ref:.1f} dB | {s5:.1f} / {s50:.1f} / {s95:.1f} dB | "
            f"{e5:.1f} / {e50:.1f} / {e95:.1f} dB | {entry['episodes_below_minus5_db']:.0%} | "
            f"{entry['episodes_above_20_db']:.0%} | {entry['episodes_above_40_db']:.0%} |"
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
