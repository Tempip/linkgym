"""Exploratory, post hoc (not in the v0.2 protocol): why OLLA misses its TBLER target on Munich.

Validation data only: the 56 pinned Munich val episodes of the protocol (seeds 5000-5055)
and the M3 TDL val seeds (500-509, one episode each). Per slot, for the tuned OLLA of the
protocol (target 0.05, delta_up 0.25 dB), one PPO-Munich model and, as a feasibility
reference, fixed MCS 3 (the lowest MCS):

- MCS, ACK and TBLER of the transmitted MCS;
- OLLA's SINR offset used for the slot;
- the slot's wideband SINR (what the next report carries) and the report the policy saw;
- the EESM effective SINR of the transmitted MCS (the simulator's) and of MCS 14 (beta
  5.66, policy independent), from the per-PRB SINR.

A slot is "infeasible" for a target t if even MCS 3 has TBLER > t there. Writes per-slot data
to results/v02/olla_analysis/ (not committed), and olla_analysis.csv, olla_bias.csv,
olla_summary.json, olla_episode.png (the val episode with the most balanced line-of-sight
share) and olla_outage_episode.png (the val episode with the longest infeasible stretch that
ends before the episode does) to docs/results/v02/.

    python examples/olla_val_analysis.py
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import gymnasium
import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import linkgym  # noqa: E402, F401  registers the env
from linkgym import ENV_ID  # noqa: E402
from linkgym.baselines import FixedMCSPolicy, OLLAPolicy, SB3Policy  # noqa: E402
from linkgym.evaluation import trace_episodes  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TRACE = REPO / "data" / "munich-v1.h5"
OUT = REPO / "results" / "v02" / "olla_analysis"
DOCS = REPO / "docs" / "results" / "v02"
TUNED = (0.05, 0.25)  # the protocol's val-selected cell
M3_TUNED = (0.1, 0.25)
BETA_MCS14 = 5.66
QUARTERS = ((0, 250), (250, 500), (500, 750), (750, 1000))
BLOCK = 100  # slots per block for the bias drift


def record(policy: Any, env: gymnasium.Env, seed: int, options: dict | None) -> dict:
    """Run one episode and return per-slot arrays."""
    obs, info = env.reset(seed=seed, options=options)
    if hasattr(policy, "reset"):
        policy.reset()
    unwrapped = env.unwrapped
    keys = ("mcs", "ack", "tbler", "offset", "wideband_db", "report_db", "eff_db", "eff14_db")
    data = {k: [] for k in keys}
    done = False
    while not done:
        slot = unwrapped._sim.slot
        sinr_prb = info["privileged"]["sinr_prb"].astype(np.float64)
        report = info["report"]
        action = policy.act(obs, info)
        offset = float(policy._olla.offset[0]) if isinstance(policy, OLLAPolicy) else np.nan
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        eff14 = -BETA_MCS14 * np.log(np.mean(np.exp(-sinr_prb / BETA_MCS14)))
        data["mcs"].append(info["mcs"])
        data["ack"].append(info["ack"])
        data["tbler"].append(info["tbler"])
        data["offset"].append(offset)
        data["wideband_db"].append(float(unwrapped._sinr_wideband_db[slot]))
        data["report_db"].append(np.nan if report is None else report["sinr_wideband_db"])
        data["eff_db"].append(10 * np.log10(float(unwrapped.last_result.sinr_eff[0])))
        data["eff14_db"].append(10 * np.log10(max(eff14, 1e-30)))
    return {k: np.asarray(v) for k, v in data.items()} | {"snr_db": info["snr_db"]}


def run(policies: dict[str, Any], env_kwargs: dict, episodes: list[dict]) -> dict[str, list]:
    env = gymnasium.make(ENV_ID, **env_kwargs)
    out = {}
    for name, policy in policies.items():
        start = time.perf_counter()
        out[name] = [record(policy, env, e["seed"], e.get("options")) for e in episodes]
        print(
            f"  {name}: {len(episodes)} episodes, {time.perf_counter() - start:.0f} s", flush=True
        )
    env.close()
    return out


def tbler(episodes: list[dict], mask=None) -> float:
    nack = np.concatenate([~e["ack"] for e in episodes])
    if mask is not None:
        m = np.concatenate(mask)
        return float(nack[m].mean()) if m.any() else float("nan")
    return float(nack.mean())


def block_means(x: np.ndarray) -> np.ndarray:
    return x[: len(x) // BLOCK * BLOCK].reshape(-1, BLOCK).mean(axis=1)


def main() -> None:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    val = trace_episodes(TRACE, ["val"], windows=4, first_seed=5000)
    munich_kwargs = {
        "channel": "trace",
        "trace_path": str(TRACE),
        "trace_splits": ("val",),
        "snr_mode": "normalized",
    }
    from stable_baselines3 import PPO

    ppo = SB3Policy(PPO.load(str(REPO / "runs/v02/munich_gamma0_seed0/model.zip"), device="cpu"))
    print("Munich val:")
    munich = run(
        {
            "OLLA tuned (0.05, 0.25 dB)": OLLAPolicy(TUNED[0], delta_up=TUNED[1]),
            "PPO-Munich seed 0": ppo,
            "fixed MCS 3": FixedMCSPolicy(3),
        },
        munich_kwargs,
        val,
    )
    tdl_episodes = [{"seed": s} for s in range(500, 510)]
    print("TDL val:")
    tdl = run(
        {
            "OLLA (0.05, 0.25 dB)": OLLAPolicy(TUNED[0], delta_up=TUNED[1]),
            "OLLA M3 tuned (0.1, 0.25 dB)": OLLAPolicy(M3_TUNED[0], delta_up=M3_TUNED[1]),
            "fixed MCS 3": FixedMCSPolicy(3),
        },
        {},
        tdl_episodes,
    )
    np.savez_compressed(
        OUT / "per_slot.npz",
        **{
            f"{scenario}|{name}|{i}|{k}": v
            for scenario, data in (("munich", munich), ("tdl", tdl))
            for name, eps in data.items()
            for i, e in enumerate(eps)
            for k, v in e.items()
        },
    )

    # 1. Observed TBLER by quarter and category; split into feasible / infeasible slots
    rows = []
    for scenario, data, meta, feasibility in (
        ("munich", munich, val, munich["fixed MCS 3"]),
        ("tdl", tdl, [{"category": "tdl"}] * len(tdl_episodes), tdl["fixed MCS 3"]),
    ):
        for name, eps in data.items():
            if name == "fixed MCS 3":
                continue
            target = 0.1 if "0.1," in name else 0.05
            for category in ("all", "los", "transition", "nlos", "tdl"):
                idx = [i for i, m in enumerate(meta) if category in ("all", m["category"])]
                if not idx:
                    continue
                sel = [eps[i] for i in idx]
                infeasible = [feasibility[i]["tbler"] > target for i in idx]
                row = {
                    "scenario": scenario,
                    "policy": name,
                    "target": target,
                    "category": category,
                    "episodes": len(idx),
                    "tbler": tbler(sel),
                }
                for q, (a, b) in enumerate(QUARTERS):
                    row[f"tbler_q{q + 1}"] = tbler([{"ack": e["ack"][a:b]} for e in sel])
                row["infeasible_slot_share"] = float(np.mean(np.concatenate(infeasible)))
                row["tbler_feasible_slots"] = tbler(sel, [~m for m in infeasible])
                for q, (a, b) in enumerate(QUARTERS):
                    row[f"tbler_feasible_q{q + 1}"] = tbler(
                        [{"ack": e["ack"][a:b]} for e in sel], [~m[a:b] for m in infeasible]
                    )
                row["tbler_infeasible_slots"] = tbler(sel, infeasible)
                nacks = np.concatenate([~e["ack"] for e in sel])
                row["nack_share_in_infeasible"] = (
                    float(np.concatenate(infeasible)[nacks].mean()) if nacks.any() else float("nan")
                )
                row["mcs3_share"] = float(np.mean(np.concatenate([e["mcs"] == 3 for e in sel])))
                rows.append(row)
    with (DOCS / "olla_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 2. Bias between the wideband SINR report and the EESM effective SINR
    bias_rows = []
    for scenario, data, meta in (
        ("munich", munich, val),
        (
            "tdl",
            tdl,
            [{"category": "tdl", "route": "tdl", "route_trajectory": i} for i in range(10)],
        ),
    ):
        olla = next(v for k, v in data.items() if k.startswith("OLLA") and "0.05" in k)
        for e, m in zip(olla, meta, strict=True):
            b14 = e["wideband_db"] - e["eff14_db"]
            bmcs = e["wideband_db"] - e["eff_db"]
            blocks = block_means(b14)
            bias_rows.append(
                {
                    "scenario": scenario,
                    "category": m["category"],
                    "route": m["route"],
                    "trajectory": m["route_trajectory"],
                    "bias_mcs14_mean_db": float(b14.mean()),
                    "bias_mcs14_block_std_db": float(blocks.std()),
                    "bias_mcs14_block_range_db": float(blocks.max() - blocks.min()),
                    "bias_mcs14_q4_minus_q1_db": float(b14[750:].mean() - b14[:250].mean()),
                    "bias_chosen_mcs_mean_db": float(bmcs.mean()),
                    "olla_offset_mean_db": float(np.nanmean(e["offset"])),
                    "wideband_block_range_db": float(
                        block_means(e["wideband_db"]).max() - block_means(e["wideband_db"]).min()
                    ),
                    "slot_to_slot_wideband_change_db": float(
                        np.mean(np.abs(np.diff(e["wideband_db"])))
                    ),
                    "tbler": float((~e["ack"]).mean()),
                }
            )
    with (DOCS / "olla_bias.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bias_rows[0]))
        writer.writeheader()
        writer.writerows(bias_rows)

    for row in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    summarize_bias(bias_rows)
    summary = outage_summary(munich, val)
    (DOCS / "olla_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    infeasible = [e["tbler"] > TUNED[0] for e in munich["fixed MCS 3"]]
    with h5py.File(TRACE, "r") as f:
        los = f["los"][()]
    los_shares = [
        los[e["options"]["trajectory"], e["options"]["offset"] // 10 :][:100].mean() for e in val
    ]
    balanced = int(np.argmin([abs(x - 0.5) if 0 < x < 1 else 9 for x in los_shares]))
    # the longest infeasible stretch that ends before its episode does (outage, then recovery)
    ended = [max((b - a for a, b in stretches(m) if b < len(m)), default=0) for m in infeasible]
    outage = int(np.argmax(ended))
    plot_episode(munich, val, balanced, infeasible[balanced], los, "olla_episode.png")
    plot_episode(munich, val, outage, infeasible[outage], los, "olla_outage_episode.png")


def stretches(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of each run of True slots."""
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def outage_summary(munich: dict, val: list[dict]) -> dict:
    """Where the infeasible slots are, and the policies after an infeasible stretch."""
    infeasible = [e["tbler"] > TUNED[0] for e in munich["fixed MCS 3"]]
    wb = np.concatenate([e["wideband_db"] for e in munich["fixed MCS 3"]])
    inf = np.concatenate(infeasible)
    runs = [b - a for m in infeasible for a, b in stretches(m)]
    by_trajectory: dict[str, int] = {}
    for m, meta in zip(infeasible, val, strict=True):
        if m.any():
            key = f"{meta['route']} trajectory {meta['route_trajectory']}"
            by_trajectory[key] = by_trajectory.get(key, 0) + int(m.sum())
    clean = [i for i, m in enumerate(infeasible) if not m.any()]
    summary = {
        "infeasible_slot_share": float(inf.mean()),
        "infeasible_slots_by_trajectory": by_trajectory,
        "episodes_with_infeasible_slots": len(val) - len(clean),
        "infeasible_stretches": len(runs),
        "infeasible_stretch_median_slots": float(np.median(runs)),
        "infeasible_stretch_max_slots": int(max(runs)),
        "infeasible_wideband_sinr_db_p10_p50_p90": [
            float(np.percentile(wb[inf], q)) for q in (10, 50, 90)
        ],
        "feasible_wideband_sinr_db_median": float(np.median(wb[~inf])),
        # ACKs needed to unwind the clamped +20 dB offset at delta_down = delta_up * t / (1 - t)
        "olla_acks_to_unwind_20_db": 20 / (TUNED[1] * TUNED[0] / (1 - TUNED[0])),
    }
    for name in ("OLLA tuned (0.05, 0.25 dB)", "PPO-Munich seed 0"):
        eps = munich[name]
        after = {"offset": [], "nack": [], "mcs": []}
        other = {"offset": [], "nack": [], "mcs": []}
        for e, m in zip(eps, infeasible, strict=True):
            if not m.any() or m.all():
                continue
            window = np.zeros(len(m), dtype=bool)
            for _, end in stretches(m):
                window[end : end + 300] = True
            window &= ~m
            for sel, target in ((window, after), (~m & ~window, other)):
                target["offset"] += list(e["offset"][sel])
                target["nack"] += list(~e["ack"][sel])
                target["mcs"] += list(e["mcs"][sel])
        summary[name] = {
            "tbler_episodes_without_infeasible_slots": tbler([eps[i] for i in clean]),
            "feasible_slots_within_300_after_infeasible": {
                "slots": len(after["mcs"]),
                **{k: float(np.mean(v)) for k, v in after.items() if not np.isnan(v).all()},
            },
            "other_feasible_slots_same_episodes": {
                "slots": len(other["mcs"]),
                **{k: float(np.mean(v)) for k, v in other.items() if not np.isnan(v).all()},
            },
        }
    return summary


def summarize_bias(bias_rows: list[dict]) -> None:
    for scenario, category in (
        ("munich", "los"),
        ("munich", "transition"),
        ("munich", "nlos"),
        ("munich", "all"),
        ("tdl", "tdl"),
    ):
        sel = [
            r for r in bias_rows if r["scenario"] == scenario and category in ("all", r["category"])
        ]

        def col(key, fn=np.mean, absolute=False):
            values = [abs(r[key]) if absolute else r[key] for r in sel]  # noqa: B023
            return fn(values)

        print(
            f"{scenario:6s} {category:10s} n={len(sel):2d} "
            f"bias14 mean {col('bias_mcs14_mean_db'):5.2f} dB "
            f"(episodes {col('bias_mcs14_mean_db', np.min):.2f}.."
            f"{col('bias_mcs14_mean_db', np.max):.2f}); "
            f"block std {col('bias_mcs14_block_std_db'):.2f}, "
            f"block range {col('bias_mcs14_block_range_db'):.2f}, "
            f"|q4-q1| {col('bias_mcs14_q4_minus_q1_db', absolute=True):.2f}; "
            f"wideband block range {col('wideband_block_range_db'):.1f}; "
            f"slot-to-slot {col('slot_to_slot_wideband_change_db'):.2f} dB"
        )


def plot_episode(munich: dict, val: list[dict], index: int, infeasible, los, name: str) -> None:
    """Wideband and effective SINR, OLLA offset, MCS and NACKs of one val episode."""
    e, meta = munich["OLLA tuned (0.05, 0.25 dB)"][index], val[index]
    t, o = meta["options"]["trajectory"], meta["options"]["offset"]
    flags = np.repeat(los[t, o // 10 : (o + 1000) // 10], 10).astype(bool)
    slots = np.arange(1000)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    ax.plot(slots, e["wideband_db"], lw=0.8, label="wideband SINR (reported one slot later)")
    ax.plot(slots, e["eff_db"], lw=0.8, label="EESM effective SINR, transmitted MCS")
    ax.set_ylabel("SINR [dB]")
    ax.legend(fontsize=8, loc="lower left")
    axes[1].plot(
        slots,
        e["wideband_db"] - e["eff_db"],
        lw=0.8,
        color="tab:purple",
        label="wideband - effective",
    )
    axes[1].plot(slots, e["offset"], lw=1.2, color="tab:red", label="OLLA offset")
    axes[1].set_ylabel("[dB]")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[2].step(slots, e["mcs"], where="post", lw=0.8, label="tuned OLLA")
    ppo = munich["PPO-Munich seed 0"][index]["mcs"]
    axes[2].step(slots, ppo, where="post", lw=0.6, alpha=0.6, label="PPO-Munich seed 0")
    axes[2].set_ylabel("MCS")
    axes[2].legend(fontsize=8, loc="lower left")
    nack = ~e["ack"]
    axes[3].vlines(slots[nack], 0, 1, color="tab:red", lw=0.6)
    axes[3].set_yticks([])
    axes[3].set_ylabel("NACK")
    axes[3].set_xlabel("slot")
    for ax in axes:
        for mask, color in ((flags, "tab:green"), (infeasible, "tab:gray")):
            for a, b in stretches(mask):
                ax.axvspan(a, b, color=color, alpha=0.12, lw=0)
    fig.suptitle(
        f"Exploratory (val): tuned OLLA on {meta['route']} trajectory {meta['route_trajectory']}, "
        f"window {meta['window']} (SNR {e['snr_db']:.1f} dB, TBLER {nack.mean():.3f})\n"
        "green: line of sight; gray: infeasible slots (even MCS 3 has TBLER > 0.05)",
        fontsize=10,
    )
    fig.savefig(DOCS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(name, json.dumps({k: meta[k] for k in ("route", "route_trajectory", "window", "seed")}))


if __name__ == "__main__":
    main()
