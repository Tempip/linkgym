"""Evaluate the M3 PPO models and the baselines; write CSVs, tables and figures.

Protocol (docs/results/m3/README.md):

- Selection uses validation seeds only (500-509, 1 episode each): the PPO gamma by the
  final validation goodput in each run's eval_log.csv (mean over training seeds), the
  OLLA target by the mean validation goodput of OLLA 0.05, 0.1 and 0.2.
- Test: held-out seeds 1000-1009, 5 episodes each, default scenario. PPO models
  (deterministic), OLLA 0.05/0.1/0.2, ILLA 0.1, fixed MCS 14, oracle 0.1.
- Paired comparison of each PPO model, and of each gamma averaged over its training
  seeds, against each OLLA target: per-episode differences on the same seed and episode,
  mean with a 95% percentile bootstrap CI.
- Fixed-SNR grid (5, 10, 15, 20 dB) on the test seeds: the PPO models of the selected
  gamma, the three OLLA targets and the oracle.

Per-episode and summary CSVs go to results/m3/ (not committed); compact tables and
figures to docs/results/m3/.

    python examples/evaluate_all.py
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from linkgym import evaluate  # noqa: E402
from linkgym.evaluation import paired_bootstrap  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OLLA_TARGETS = (0.05, 0.1, 0.2)
METRICS = ("goodput_mbps", "observed_tbler", "mean_mcs")


@dataclass(frozen=True)
class Job:
    split: str  # "validation" or "test"
    scenario: str  # "default" or "snr<dB>"
    label: str
    kind: str  # ppo, olla, illa, fixed or oracle
    param: Any  # model path, TBLER target or MCS
    env_kwargs: tuple[tuple[str, Any], ...]
    seeds: tuple[int, ...]
    episodes: int


def build_policy(kind: str, param: Any) -> Any:
    if kind == "ppo":
        from stable_baselines3 import PPO

        from linkgym.baselines import SB3Policy

        return SB3Policy(PPO.load(param, device="cpu"))
    from linkgym.baselines import FixedMCSPolicy, ILLAPolicy, OLLAPolicy, OraclePolicy

    classes = {"olla": OLLAPolicy, "illa": ILLAPolicy, "oracle": OraclePolicy}
    return classes.get(kind, FixedMCSPolicy)(param)


def run_job(job: Job) -> tuple[Job, dict[str, Any], float]:
    torch.set_num_threads(1)
    start = time.perf_counter()
    policy = build_policy(job.kind, job.param)
    result = evaluate(policy, dict(job.env_kwargs), job.seeds, job.episodes)
    return job, result, time.perf_counter() - start


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for run_json in sorted(runs_dir.glob("*/run.json")):
        info = json.loads(run_json.read_text(encoding="utf-8"))
        runs.append(
            {
                "gamma": info["ppo"]["gamma"],
                "seed": info["seeds"]["training_seed"],
                "model": str(run_json.parent / "model.zip"),
                "validation_seeds": tuple(info["seeds"]["validation_seeds"]),
                "curve": read_csv(run_json.parent / "eval_log.csv"),
                "info": info,
            }
        )
    if not runs:
        raise SystemExit(f"no runs found in {runs_dir}")
    if len({r["validation_seeds"] for r in runs}) != 1:
        raise SystemExit("runs were validated on different seeds")
    return sorted(runs, key=lambda r: (r["gamma"], r["seed"]))


def ppo_label(gamma: float, seed: int | None = None) -> str:
    return f"ppo gamma={gamma:g}" + ("" if seed is None else f" seed={seed}")


def n_seeds(n: int) -> str:
    return f"{n} seed" + ("" if n == 1 else "s")


def pm(stats: dict[str, float], digits: int) -> str:
    return f"{stats['mean']:.{digits}f} ± {stats['std']:.{digits}f}"


def ci(stats: dict[str, float], digits: int, pct: bool = False) -> str:
    keys = ("pct", "pct_ci_low", "pct_ci_high") if pct else ("mean", "ci_low", "ci_high")
    mean, low, high = (stats[k] for k in keys)
    return f"{mean:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"


def mean_std(values: list[float]) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {"mean": float(values.mean()), "std": std}


def episode_metric(result: dict[str, Any], metric: str) -> np.ndarray:
    return np.array([e[metric] for e in result["per_episode"]])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs" / "m3")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results" / "m3")
    parser.add_argument("--docs-dir", type=Path, default=REPO / "docs" / "results" / "m3")
    parser.add_argument("--test-seeds", type=int, nargs="+", default=list(range(1000, 1010)))
    parser.add_argument("--test-episodes", type=int, default=5)
    parser.add_argument("--snr-grid", type=float, nargs="+", default=[5.0, 10.0, 15.0, 20.0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--episode-length", type=int, default=None, help="scenario override (quick checks only)"
    )
    args = parser.parse_args()
    if min(args.test_seeds) < 1000:
        raise SystemExit("test seeds must be >= 1000")

    runs = load_runs(args.runs_dir)
    gammas = sorted({r["gamma"] for r in runs})
    val_seeds = runs[0]["validation_seeds"]
    base_kwargs = {} if args.episode_length is None else {"episode_length": args.episode_length}

    # Gamma selection on validation seeds: final validation goodput, mean over training seeds
    val_ppo = {
        g: mean_std([r["curve"][-1]["goodput_mbps"] for r in runs if r["gamma"] == g])
        for g in gammas
    }
    best_gamma = max(gammas, key=lambda g: val_ppo[g]["mean"])

    def job(split, scenario, label, kind, param, extra=None, seeds=None, episodes=None):
        kwargs = {**base_kwargs, **(extra or {})}
        return Job(
            split,
            scenario,
            label,
            kind,
            param,
            tuple(sorted(kwargs.items())),
            tuple(seeds or args.test_seeds),
            episodes or args.test_episodes,
        )

    jobs = [
        job("validation", "default", f"olla {t:g}", "olla", t, seeds=val_seeds, episodes=1)
        for t in OLLA_TARGETS
    ]
    test_policies = [(ppo_label(r["gamma"], r["seed"]), "ppo", r["model"]) for r in runs]
    test_policies += [(f"olla {t:g}", "olla", t) for t in OLLA_TARGETS]
    test_policies += [("illa 0.1", "illa", 0.1), ("fixed 14", "fixed", 14)]
    test_policies += [("oracle 0.1", "oracle", 0.1)]
    jobs += [job("test", "default", *p) for p in test_policies]
    grid_policies = [p for p in test_policies if p[1] in ("olla", "oracle")]
    grid_policies += [
        (ppo_label(r["gamma"], r["seed"]), "ppo", r["model"])
        for r in runs
        if r["gamma"] == best_gamma
    ]
    for snr in args.snr_grid:
        jobs += [job("test", f"snr{snr:g}", *p, extra={"snr_db": snr}) for p in grid_policies]
    # Slowest policies first for better load balancing
    order = {"oracle": 0, "olla": 1, "illa": 1, "ppo": 2, "fixed": 3}
    jobs.sort(key=lambda j: (j.split != "test", order[j.kind]))

    print(
        f"{len(runs)} runs, gammas {gammas}; {len(jobs)} evaluation jobs on {args.workers} workers"
    )
    start = time.perf_counter()
    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    seconds: dict[tuple[str, str, str], float] = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for done_job, result, job_seconds in pool.map(run_job, jobs):
            key = (done_job.split, done_job.scenario, done_job.label)
            results[key], seconds[key] = result, job_seconds
    elapsed = time.perf_counter() - start
    print(f"evaluation done in {elapsed:.0f} s\n")
    job_by_key = {(j.split, j.scenario, j.label): j for j in jobs}

    # Common random numbers: every policy of a split/scenario saw the same episodes
    for split, scenario in {(j.split, j.scenario) for j in jobs}:
        episodes = [
            [(e["seed"], e["episode"], e["snr_db"]) for e in r["per_episode"]]
            for (s, sc, _), r in results.items()
            if (s, sc) == (split, scenario)
        ]
        assert all(e == episodes[0] for e in episodes), f"episodes differ in {split}/{scenario}"

    # OLLA target selection on validation seeds
    val_olla = {t: results[("validation", "default", f"olla {t:g}")] for t in OLLA_TARGETS}
    best_target = max(OLLA_TARGETS, key=lambda t: val_olla[t]["goodput_mbps"]["mean"])

    # --- Raw CSVs (results/, not committed) ---
    per_episode_rows, summary_rows = [], []
    for key, result in sorted(results.items()):
        j = job_by_key[key]
        for e in result["per_episode"]:
            per_episode_rows.append(
                {"split": j.split, "scenario": j.scenario, "policy": j.label, **e}
            )
        summary_rows.append(
            {
                "split": j.split,
                "scenario": j.scenario,
                "policy": j.label,
                **{f"{m}_{s}": result[m][s] for m in METRICS for s in ("mean", "std")},
                "seconds": seconds[key],
            }
        )
    write_csv(args.results_dir / "per_episode.csv", per_episode_rows)
    write_csv(args.results_dir / "summary.csv", summary_rows)

    # --- Validation table ---
    validation_rows = [
        {
            "policy": ppo_label(g) + f" ({n_seeds(sum(r['gamma'] == g for r in runs))})",
            "goodput_mbps_mean": val_ppo[g]["mean"],
            "goodput_mbps_std": val_ppo[g]["std"],
            "observed_tbler_mean": float(
                np.mean([r["curve"][-1]["observed_tbler"] for r in runs if r["gamma"] == g])
            ),
            "selected": g == best_gamma,
        }
        for g in gammas
    ]
    validation_rows += [
        {
            "policy": f"olla {t:g}",
            "goodput_mbps_mean": val_olla[t]["goodput_mbps"]["mean"],
            "goodput_mbps_std": val_olla[t]["goodput_mbps"]["std"],
            "observed_tbler_mean": val_olla[t]["observed_tbler"]["mean"],
            "selected": t == best_target,
        }
        for t in OLLA_TARGETS
    ]
    write_csv(args.docs_dir / "validation.csv", validation_rows)

    # --- Test summary, default scenario ---
    test = {label: results[("test", "default", label)] for label, _, _ in test_policies}
    summary = []
    for g in gammas:
        models = [test[ppo_label(r["gamma"], r["seed"])] for r in runs if r["gamma"] == g]
        summary.append(
            (
                ppo_label(g) + f" ({n_seeds(len(models))})",
                {m: mean_std([x[m]["mean"] for x in models]) for m in METRICS},
            )
        )
    summary += [(label, {m: r[m] for m in METRICS}) for label, r in test.items()]
    write_csv(
        args.docs_dir / "summary.csv",
        [
            {"policy": label, **{f"{m}_{s}": stats[m][s] for m in METRICS for s in ("mean", "std")}}
            for label, stats in summary
        ],
    )

    # --- Paired comparisons vs every OLLA target ---
    def paired_row(label: str, goodput: np.ndarray, tbler: np.ndarray, target: float) -> dict:
        olla = test[f"olla {target:g}"]
        boot = {"num_resamples": args.bootstrap_resamples, "seed": args.bootstrap_seed}
        g = paired_bootstrap(goodput, episode_metric(olla, "goodput_mbps"), **boot)
        t = paired_bootstrap(tbler, episode_metric(olla, "observed_tbler"), **boot)
        return {
            "policy": label,
            "vs": f"olla {target:g}",
            "vs_selected_on_validation": target == best_target,
            "episodes": len(goodput),
            **{f"goodput_diff_{k}": v for k, v in g.items()},
            **{f"tbler_diff_{k}": t[k] for k in ("mean", "ci_low", "ci_high")},
        }

    paired = []
    for g in gammas:
        models = [test[ppo_label(r["gamma"], r["seed"])] for r in runs if r["gamma"] == g]
        goodput = np.mean([episode_metric(m, "goodput_mbps") for m in models], axis=0)
        tbler = np.mean([episode_metric(m, "observed_tbler") for m in models], axis=0)
        label = ppo_label(g) + f" (mean of {n_seeds(len(models))})"
        paired += [paired_row(label, goodput, tbler, t) for t in OLLA_TARGETS]
    for r in runs:
        m = test[ppo_label(r["gamma"], r["seed"])]
        paired += [
            paired_row(
                ppo_label(r["gamma"], r["seed"]),
                episode_metric(m, "goodput_mbps"),
                episode_metric(m, "observed_tbler"),
                t,
            )
            for t in OLLA_TARGETS
        ]
    write_csv(args.docs_dir / "paired.csv", paired)

    # --- Fixed-SNR grid ---
    grid_labels = [p[0] for p in grid_policies]
    fixed_rows = []
    for snr in args.snr_grid:
        for label in grid_labels:
            r = results[("test", f"snr{snr:g}", label)]
            fixed_rows.append(
                {
                    "snr_db": snr,
                    "policy": label,
                    **{f"{m}_{s}": r[m][s] for m in METRICS for s in ("mean", "std")},
                }
            )
    write_csv(args.docs_dir / "fixed_snr.csv", fixed_rows)

    # --- Learning curves ---
    curve_rows = [
        {"gamma": r["gamma"], "seed": r["seed"], **row} for r in runs for row in r["curve"]
    ]
    write_csv(args.docs_dir / "learning_curves.csv", curve_rows)

    # --- Markdown tables ---
    print(
        f"Validation (seeds {val_seeds[0]}-{val_seeds[-1]}, 1 episode each; PPO rows: final "
        f"validation goodput, mean ± std across training seeds; OLLA rows: mean ± std across "
        f"validation seeds)\n"
    )
    print("| policy | goodput Mbit/s | observed TBLER | selected |")
    print("|---|---:|---:|:---:|")
    for row in validation_rows:
        print(
            f"| {row['policy']} | {row['goodput_mbps_mean']:.2f} ± {row['goodput_mbps_std']:.2f} | "
            f"{row['observed_tbler_mean']:.4f} | {'yes' if row['selected'] else ''} |"
        )
    print(
        f"\nTest, default scenario (seeds {args.test_seeds[0]}-{args.test_seeds[-1]}, "
        f"{args.test_episodes} episodes each; PPO config rows: mean ± std across training "
        f"seeds, other rows: across test seeds)\n"
    )
    print("| policy | goodput Mbit/s | observed TBLER | mean MCS |")
    print("|---|---:|---:|---:|")
    for label, stats in summary:
        print(
            f"| {label} | {pm(stats['goodput_mbps'], 2)} | {pm(stats['observed_tbler'], 4)} | "
            f"{pm(stats['mean_mcs'], 2)} |"
        )
    print("\nPaired differences vs OLLA (mean [95% bootstrap CI]; * = selected on validation)\n")
    print("| policy | vs | goodput diff Mbit/s | goodput diff % | TBLER diff |")
    print("|---|---|---:|---:|---:|")
    for row in paired:
        g = {k[len("goodput_diff_") :]: v for k, v in row.items() if k.startswith("goodput_diff_")}
        t = {k[len("tbler_diff_") :]: v for k, v in row.items() if k.startswith("tbler_diff_")}
        star = "*" if row["vs_selected_on_validation"] else ""
        cells = f"{ci(g, 2)} | {ci(g, 1, pct=True)} | {ci(t, 4)}"
        print(f"| {row['policy']} | {row['vs']}{star} | {cells} |")
    print("\nFixed SNR (test seeds; mean goodput Mbit/s / observed TBLER)\n")
    header = " | ".join(f"{snr:g} dB" for snr in args.snr_grid)
    print(f"| policy | {header} |")
    print("|---|" + "---:|" * len(args.snr_grid))
    for label in grid_labels:
        cells = []
        for snr in args.snr_grid:
            r = results[("test", f"snr{snr:g}", label)]
            cells.append(f"{r['goodput_mbps']['mean']:.2f} / {r['observed_tbler']['mean']:.3f}")
        print(f"| {label} | " + " | ".join(cells) + " |")

    plot_learning_curves(runs, gammas, val_olla, args.docs_dir / "learning_curves.png")
    plot_scatter(test, args.docs_dir / "goodput_vs_tbler.png")
    plot_snr(results, runs, best_gamma, args.snr_grid, args.docs_dir / "goodput_vs_snr.png")
    print(f"\nwrote {args.results_dir} and {args.docs_dir}")


def plot_learning_curves(runs, gammas, val_olla, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, ylabel in zip(
        axes,
        ("goodput_mbps", "observed_tbler"),
        ("goodput [Mbit/s]", "observed TBLER"),
        strict=True,
    ):
        for g in gammas:
            curves = [r["curve"] for r in runs if r["gamma"] == g]
            steps = np.array([row["timesteps"] for row in curves[0]])
            values = np.array([[row[metric] for row in c] for c in curves])
            ax.plot(steps, values.mean(axis=0), label=f"PPO gamma={g:g} (mean of {len(curves)})")
            ax.fill_between(steps, values.min(axis=0), values.max(axis=0), alpha=0.25)
        for t, style in zip(OLLA_TARGETS, (":", "--", "-."), strict=True):
            ax.axhline(
                val_olla[t][metric]["mean"], color="gray", linestyle=style, label=f"OLLA {t:g}"
            )
        ax.set_xlabel("training steps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].set_title("Validation goodput (band: min-max over seeds)")
    axes[1].set_title("Validation observed TBLER")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_scatter(test, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ppo_colors: dict[str, str] = {}
    for label, r in test.items():
        x, y = r["observed_tbler"]["mean"], r["goodput_mbps"]["mean"]
        if label.startswith("ppo"):
            # One color and legend entry per gamma; the seeds of a gamma overlap
            config = label.rsplit(" seed=", 1)[0]
            first = config not in ppo_colors
            ppo_colors.setdefault(config, f"C{len(ppo_colors)}")
            ax.scatter(
                x,
                y,
                marker="o",
                color=ppo_colors[config],
                label=f"{config} (seeds)" if first else None,
            )
        else:
            ax.scatter(x, y, marker="s", color="gray")
            ax.annotate(label, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("observed TBLER")
    ax.set_ylabel("goodput [Mbit/s]")
    ax.set_title("Test seeds, default scenario (mean over test seeds)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_snr(results, runs, best_gamma, snr_grid, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    snrs = np.array(snr_grid)
    models = [ppo_label(r["gamma"], r["seed"]) for r in runs if r["gamma"] == best_gamma]
    for ax, metric, ylabel in zip(
        axes,
        ("goodput_mbps", "observed_tbler"),
        ("goodput [Mbit/s]", "observed TBLER"),
        strict=True,
    ):
        values = np.array(
            [[results[("test", f"snr{s:g}", m)][metric]["mean"] for s in snr_grid] for m in models]
        )
        ax.plot(snrs, values.mean(axis=0), marker="o", label=f"PPO gamma={best_gamma:g} (mean)")
        ax.fill_between(snrs, values.min(axis=0), values.max(axis=0), alpha=0.25)
        for label in [f"olla {t:g}" for t in OLLA_TARGETS] + ["oracle 0.1"]:
            ys = [results[("test", f"snr{s:g}", label)][metric]["mean"] for s in snr_grid]
            ax.plot(snrs, ys, marker="s", linestyle="--", label=label)
        ax.set_xlabel("mean SNR [dB]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].set_title("Test seeds, fixed SNR (band: min-max over PPO seeds)")
    axes[1].set_title("Observed TBLER")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
