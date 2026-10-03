"""Exploratory, post hoc (not in the v0.2 protocol): outage stretches on the Munich test traces.

A trace-only check of the test split: no policy is evaluated and no result changes. For the
protocol's pinned test episodes (seeds 6000-6091, both ray-tracing realizations) at their drawn
SNRs, the environment is stepped with MCS 3 to get the PHY abstraction's TBLER of MCS 3 in
every slot (no ACK or goodput is recorded). A slot is "infeasible" if that TBLER is above 0.05:
no MCS can meet the tuned OLLA's target there. Per trajectory, also the spread of the normalized
wideband gain (gain over the trajectory's mean linear gain, the normalization of
snr_mode="normalized").

Both are set against the tuned OLLA's published per-episode TBLER (results/v02/test.json) and a
two-number model calibrated on val: OLLA's TBLER in the feasible and in the infeasible val slots
(docs/results/v02/olla_analysis.csv). Writes test_traces_by_trajectory.csv,
test_traces_by_group.csv and test_traces.png to docs/results/v02/.

    python examples/olla_test_traces.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import gymnasium
import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import linkgym  # noqa: E402, F401  registers the env
from linkgym import ENV_ID  # noqa: E402
from linkgym.evaluation import trace_episodes  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "results" / "v02"
TEST_FIRST_SEED = 6000  # the protocol's test seeds (examples/evaluate_v02.py)
WINDOWS = 4
TARGET = 0.05  # the tuned OLLA's TBLER target
OLLA = "OLLA tuned"
CATEGORIES = ("los", "transition", "nlos")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def val_calibration() -> tuple[float, float]:
    """Tuned OLLA's TBLER in the feasible and infeasible val slots (all categories)."""
    with (DOCS / "olla_analysis.csv").open(encoding="utf-8") as f:
        row = next(
            r
            for r in csv.DictReader(f)
            if r["scenario"] == "munich"
            and r["policy"].startswith("OLLA")
            and r["category"] == "all"
        )
    return float(row["tbler_feasible_slots"]), float(row["tbler_infeasible_slots"])


def mcs3_tbler(path: Path, episodes: list[dict], published: dict[int, dict]) -> list[dict]:
    """Per-slot TBLER of MCS 3 and wideband SINR on each pinned episode."""
    env = gymnasium.make(
        ENV_ID,
        channel="trace",
        trace_path=str(path),
        trace_splits=("test",),
        snr_mode="normalized",
    )
    out = []
    for e in episodes:
        _, info = env.reset(seed=e["seed"], options=e["options"])
        snr = info["snr_db"]
        if not np.isclose(snr, published[e["seed"]]["snr_db"], atol=1e-6):
            raise RuntimeError(f"seed {e['seed']}: SNR {snr} differs from the protocol's run")
        tbler, wideband = [], []
        unwrapped = env.unwrapped
        done = False
        while not done:
            wideband.append(float(unwrapped._sinr_wideband_db[unwrapped._sim.slot]))
            _, _, terminated, truncated, info = env.step(0)  # action 0 is MCS 3
            tbler.append(info["tbler"])
            done = terminated or truncated
        out.append(
            e | {"snr_db": snr, "tbler": np.asarray(tbler), "wideband_db": np.asarray(wideband)}
        )
    env.close()
    return out


def gain_spread(path: Path, trajectory: int) -> dict[str, float]:
    """Normalized wideband gain [dB] of one trajectory, relative to its mean linear gain."""
    with h5py.File(path, "r") as f:
        gain = f["gain"][trajectory].astype(np.float64)  # [slots, PRBs]
    rel = 10 * np.log10(gain.mean(axis=1) / gain.mean())
    return {
        "gain_median_db": float(np.median(rel)),
        "gain_p10_db": float(np.percentile(rel, 10)),
        "gain_p1_db": float(np.percentile(rel, 1)),
        "gain_max_db": float(rel.max()),
        "share_below_minus20_db": float(np.mean(rel < -20)),
    }


def main() -> None:
    torch.set_num_threads(1)
    test = json.loads((REPO / "results" / "v02" / "test.json").read_text(encoding="utf-8"))
    feasible_tbler, infeasible_tbler = val_calibration()
    print(
        f"val calibration: OLLA TBLER {feasible_tbler:.4f} feasible, "
        f"{infeasible_tbler:.4f} infeasible"
    )
    trajectory_rows, group_rows, scatter = [], [], {}
    for realization, meta in test["datasets"].items():
        path = REPO / meta["path"]
        if sha256(path) != meta["sha256"]:
            raise RuntimeError(f"{path} is not the dataset the protocol evaluated")
        published = {r["seed"]: r for r in test["results"][realization][OLLA]}
        episodes = trace_episodes(path, ["test"], windows=WINDOWS, first_seed=TEST_FIRST_SEED)
        if sorted(published) != sorted(e["seed"] for e in episodes):
            raise RuntimeError("the pinned test episodes differ from the protocol's run")
        runs = mcs3_tbler(path, episodes, published)
        for r in runs:
            r["infeasible"] = r["tbler"] > TARGET
            r["olla_tbler"] = published[r["seed"]]["observed_tbler"]
        scatter[realization] = runs
        print(f"{realization}: {len(runs)} episodes", flush=True)

        def summary(sel: list[dict]) -> dict:
            share = float(np.mean([r["infeasible"].mean() for r in sel]))
            clean = [r for r in sel if not r["infeasible"].any()]
            observed = float(np.mean([r["olla_tbler"] for r in sel]))
            predicted = feasible_tbler * (1 - share) + infeasible_tbler * share
            return {
                "episodes": len(sel),
                "episodes_with_infeasible_slots": len(sel) - len(clean),
                "infeasible_slot_share": share,
                "olla_tbler": observed,
                "olla_tbler_predicted": predicted,
                "olla_excess": observed - TARGET,
                "olla_excess_predicted": predicted - TARGET,
                "olla_tbler_episodes_without_infeasible_slots": (
                    float(np.mean([r["olla_tbler"] for r in clean])) if clean else float("nan")
                ),
            }

        for t in sorted({r["trajectory"] for r in runs}):
            sel = [r for r in runs if r["trajectory"] == t]
            infeasible = np.concatenate([r["infeasible"] for r in sel])
            wideband = np.concatenate([r["wideband_db"] for r in sel])
            trajectory_rows.append(
                {
                    "realization": realization,
                    "route": sel[0]["route"],
                    "category": sel[0]["category"],
                    "route_trajectory": sel[0]["route_trajectory"],
                    "trajectory": t,
                    **gain_spread(path, t),
                    "snr_db_min": min(r["snr_db"] for r in sel),
                    "snr_db_max": max(r["snr_db"] for r in sel),
                    "infeasible_wideband_sinr_median_db": (
                        float(np.median(wideband[infeasible])) if infeasible.any() else float("nan")
                    ),
                    **summary(sel),
                }
            )
        groups = [
            (route, [r for r in runs if r["route"] == route])
            for route in dict.fromkeys(r["route"] for r in runs)
        ]
        groups += [(f"category {c}", [r for r in runs if r["category"] == c]) for c in CATEGORIES]
        groups += [("all", runs)]
        for name, sel in groups:
            group_rows.append(
                {
                    "realization": realization,
                    "group": name,
                    "category": sel[0]["category"]
                    if not name.startswith(("category", "all"))
                    else "",
                    "trajectories": len({r["trajectory"] for r in sel}),
                    **summary(sel),
                }
            )

    for name, rows in (
        ("test_traces_by_trajectory.csv", trajectory_rows),
        ("test_traces_by_group.csv", group_rows),
    ):
        with (DOCS / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in group_rows + trajectory_rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    plot(scatter, feasible_tbler, infeasible_tbler)


def plot(scatter: dict[str, list[dict]], feasible_tbler: float, infeasible_tbler: float) -> None:
    """Per-episode infeasible share against the tuned OLLA's published TBLER."""
    colors = {"los": "tab:orange", "transition": "tab:purple", "nlos": "tab:blue"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (realization, runs) in zip(axes, scatter.items(), strict=True):
        for c in CATEGORIES:
            sel = [r for r in runs if r["category"] == c]
            ax.scatter(
                [r["infeasible"].mean() for r in sel],
                [r["olla_tbler"] for r in sel],
                s=18,
                alpha=0.7,
                color=colors[c],
                label=c,
            )
        x = np.linspace(0, 1, 2)
        ax.plot(
            x,
            feasible_tbler * (1 - x) + infeasible_tbler * x,
            color="gray",
            lw=1,
            label="val-calibrated model",
        )
        ax.axhline(TARGET, color="gray", lw=0.8, ls=":")
        ax.set_title(f"{realization} realization", fontsize=10)
        ax.set_xlabel("share of infeasible slots in the episode")
    axes[0].set_ylabel("tuned OLLA observed TBLER (published)")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Exploratory, post hoc: test episodes "
        "(trace-only check; OLLA TBLER from the protocol's run)",
        fontsize=10,
    )
    fig.savefig(DOCS / "test_traces.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
