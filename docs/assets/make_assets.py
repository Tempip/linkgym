"""Build the README header figure and the GitHub social preview from the M3 results.

Reads docs/results/m3/fixed_snr.csv (test seeds, fixed SNR) and writes
docs/assets/header.png and docs/assets/social_preview.png (1280 x 640).

    python docs/assets/make_assets.py
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ASSETS = Path(__file__).resolve().parent
RESULTS = ASSETS.parent / "results" / "m3" / "fixed_snr.csv"
DESCRIPTION = (
    "A Gymnasium environment for 5G NR link adaptation (MCS selection) built on NVIDIA "
    "Sionna SYS, with classical baselines and a fixed evaluation protocol."
)
PPO_MODELS = [f"ppo gamma=0 seed={s}" for s in (0, 1, 2)]
TUNED_OLLA = "olla tuned (0.1, delta_up=0.25)"
ORACLE = "oracle 0.1"


def load() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    goodput: dict[str, dict[float, float]] = {}
    with RESULTS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            goodput.setdefault(row["policy"], {})[float(row["snr_db"])] = float(
                row["goodput_mbps_mean"]
            )
    snrs = np.array(sorted(goodput[ORACLE]))
    return snrs, {policy: np.array([v[s] for s in snrs]) for policy, v in goodput.items()}


def draw(ax: plt.Axes, snrs: np.ndarray, goodput: dict[str, np.ndarray], small: bool) -> None:
    ppo = np.array([goodput[m] for m in PPO_MODELS])
    size = 10 if small else 11
    ax.plot(snrs, goodput[ORACLE], color="#7b5ea7", linestyle="--", marker="s", label="Oracle")
    ax.plot(snrs, ppo.mean(axis=0), color="#1f77b4", marker="o", label="PPO (gamma = 0)")
    ax.fill_between(snrs, ppo.min(axis=0), ppo.max(axis=0), color="#1f77b4", alpha=0.2)
    ax.plot(snrs, goodput[TUNED_OLLA], color="#444444", marker="D", label="OLLA (tuned)")
    ax.set_xlabel("mean SNR [dB]", fontsize=size)
    ax.set_ylabel("goodput [Mbit/s]", fontsize=size)
    ax.set_xticks(snrs)
    ax.tick_params(labelsize=size - 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=size - 1, loc="upper left", frameon=False)


def header(snrs: np.ndarray, goodput: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    draw(ax, snrs, goodput, small=False)
    ax.set_title(
        "Goodput on held-out test seeds, TDL-A, 15 m/s, 52 PRB (band: 3 PPO training seeds)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "header.png", dpi=200)
    plt.close(fig)


def social_preview(snrs: np.ndarray, goodput: dict[str, np.ndarray]) -> None:
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.text(0.05, 0.72, "linkgym", fontsize=64, fontweight="bold", va="center")
    fig.text(
        0.05, 0.47, "\n".join(textwrap.wrap(DESCRIPTION, 38)), fontsize=19, va="center",
        linespacing=1.4,
    )  # fmt: skip
    fig.text(0.05, 0.12, "github.com/Tempip/linkgym", fontsize=15, color="#555555")
    ax = fig.add_axes((0.58, 0.17, 0.38, 0.68))
    draw(ax, snrs, goodput, small=True)
    fig.savefig(ASSETS / "social_preview.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    snrs, goodput = load()
    header(snrs, goodput)
    social_preview(snrs, goodput)
    print(f"wrote {ASSETS / 'header.png'} and {ASSETS / 'social_preview.png'}")
