"""Build the README header figure and the GitHub social preview.

Goodput difference against tuned OLLA [%], paired, with 95% CIs, on the TDL channel and on
the ray-traced Munich streets:

- TDL (50 test episodes, against the M3 tuned OLLA, target 0.1 and delta_up 0.25 dB): PPO-TDL
  and PPO-Munich from docs/results/v02/q4.csv; the oracle computed here from the M3
  per-episode results (results/m3/per_episode.csv, not committed) with the same paired
  percentile bootstrap and settings as the M3 results, and stored in
  docs/assets/header_tdl_oracle.csv, which is read when the per-episode file is absent;
- Munich (92 test episodes on 23 trajectories, main realization, against the tuned OLLA,
  target 0.05 and delta_up 0.25 dB): docs/results/v02/paired.csv, cluster bootstrap over
  trajectories.

Writes docs/assets/header.png and docs/assets/social_preview.png (1280 x 640).

    python docs/assets/make_assets.py
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from linkgym.evaluation import paired_bootstrap  # noqa: E402

ASSETS = Path(__file__).resolve().parent
REPO = ASSETS.parents[1]
V02 = REPO / "docs" / "results" / "v02"
M3_EPISODES = REPO / "results" / "m3" / "per_episode.csv"
TDL_ORACLE = ASSETS / "header_tdl_oracle.csv"
M3_BOOTSTRAP = {"num_resamples": 10_000, "confidence": 0.95, "seed": 2026}  # evaluate_all.py
DESCRIPTION = (
    "A Gymnasium environment for 5G NR link adaptation (MCS selection) built on NVIDIA "
    "Sionna SYS, with TDL and ray-traced channels, classical baselines and a fixed "
    "evaluation protocol."
)
ROWS = ("PPO trained on TDL", "PPO trained on Munich", "Oracle")
COLORS = ("#1f77b4", "#d62728", "#7b5ea7")
Interval = tuple[float, float, float]  # pct, pct_ci_low, pct_ci_high


def tdl_oracle() -> Interval:
    if M3_EPISODES.exists():
        with M3_EPISODES.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == "test"]

        def goodput(policy: str) -> list[float]:
            sel = sorted(
                (int(r["seed"]), int(r["episode"]), float(r["goodput_mbps"]))
                for r in rows
                if r["policy"] == policy and r["scenario"] == "default"
            )
            return [g for *_, g in sel]

        oracle, olla = goodput("oracle 0.1"), goodput("olla tuned (0.1, delta_up=0.25)")
        stats = paired_bootstrap(oracle, olla, **M3_BOOTSTRAP)
        with TDL_ORACLE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["comparison", "episodes", *stats])
            writer.writerow(["oracle 0.1 - OLLA tuned (M3)", len(oracle), *stats.values()])
    with TDL_ORACLE.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    return float(row["pct"]), float(row["pct_ci_low"]), float(row["pct_ci_high"])


def load() -> dict[str, list[Interval]]:
    with (V02 / "q4.csv").open(newline="", encoding="utf-8") as f:
        q4 = {r["policy"]: r for r in csv.DictReader(f)}
    with (V02 / "paired.csv").open(newline="", encoding="utf-8") as f:
        paired = {
            r["comparison"]: r
            for r in csv.DictReader(f)
            if r["realization"] == "main" and r["category"] == "all"
        }

    def interval(row: dict, keys: tuple[str, str, str]) -> Interval:
        return float(row[keys[0]]), float(row[keys[1]]), float(row[keys[2]])

    q4_keys = ("diff_pct", "diff_pct_ci_low", "diff_pct_ci_high")
    keys = ("pct", "pct_ci_low", "pct_ci_high")
    return {
        "tdl": [interval(q4["PPO-TDL"], q4_keys), interval(q4["PPO-Munich"], q4_keys)]
        + [tdl_oracle()],
        "munich": [
            interval(paired[f"{name} - OLLA tuned"], keys)
            for name in ("PPO-TDL", "PPO-Munich", "oracle 0.1")
        ],
    }


def draw(axes, data: dict[str, list[Interval]], small: bool) -> None:
    size = 9 if small else 10
    titles = {
        "tdl": "TDL-A channel\n50 test episodes",
        "munich": "Ray-traced Munich streets\n92 test episodes, 4 streets",
    }
    for ax, (channel, intervals) in zip(axes, data.items(), strict=True):
        ax.axvline(0, color="#444444", lw=1.2)
        for i, ((pct, low, high), color) in enumerate(zip(intervals, COLORS, strict=True)):
            y = len(ROWS) - 1 - i
            ax.errorbar(
                pct, y, xerr=[[pct - low], [high - pct]], fmt="o", color=color, ms=7, capsize=4
            )
            label = f"{pct:+.1f}%" if small else f"{pct:+.1f}% [{low:+.1f}, {high:+.1f}]"
            ax.annotate(
                label,
                (high, y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=size - 1,
            )
        ax.set_title(titles[channel], fontsize=size)
        ax.set_xlim(-15, 50)
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(labelsize=size - 1)
        ax.set_xlabel("goodput vs tuned OLLA [%]", fontsize=size)
        ax.text(0.5, -0.45, "tuned OLLA", fontsize=size - 2, color="#444444", ha="center")
    axes[0].set_yticks(range(len(ROWS)), ROWS[::-1], fontsize=size)


def header(data: dict[str, list[Interval]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    draw(axes, data, small=False)
    fig.suptitle(
        "Paired goodput difference against a validation-tuned OLLA, with 95% CIs",
        fontsize=10,
    )
    fig.text(
        0.01,
        0.01,
        "TDL: PPO from the v0.2 Q4 evaluation; oracle CI computed from the M3 per-episode "
        "results. Munich: main realization, CIs over test trajectories.",
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ASSETS / "header.png", dpi=200)
    plt.close(fig)


def social_preview(data: dict[str, list[Interval]]) -> None:
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.text(0.05, 0.72, "linkgym", fontsize=64, fontweight="bold", va="center")
    fig.text(
        0.05, 0.44, "\n".join(textwrap.wrap(DESCRIPTION, 38)), fontsize=18, va="center",
        linespacing=1.4,
    )  # fmt: skip
    fig.text(0.05, 0.1, "github.com/Tempip/linkgym", fontsize=15, color="#555555")
    axes = [fig.add_axes((0.62, 0.17, 0.17, 0.62))]
    axes.append(fig.add_axes((0.81, 0.17, 0.17, 0.62), sharey=axes[0]))
    draw(axes, data, small=True)
    axes[1].tick_params(labelleft=False)
    fig.savefig(ASSETS / "social_preview.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    data = load()
    header(data)
    social_preview(data)
    print(data)
    print(f"wrote {ASSETS / 'header.png'} and {ASSETS / 'social_preview.png'}")
