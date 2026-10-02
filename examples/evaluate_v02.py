"""v0.2 Munich experiment: validation, test (once, both realizations), Q4 and the report.

Protocol: docs/results/v02/PROTOCOL.md. Run the phases in order, after training the
Munich PPO models (examples/train_ppo.py --channel trace):

    python examples/evaluate_v02.py validation  # OLLA grid and all policies on Munich val
    python examples/evaluate_v02.py test        # once: test split, both realizations
    python examples/evaluate_v02.py q4          # Munich PPO on the TDL test seeds
    python examples/evaluate_v02.py report      # tables and figures in docs/results/v02/

Per-episode results go to results/v02/ (not committed). The test phase refuses to run if
its results exist. --smoke runs every phase on the small sample trace in tests/data (and
the smoke model in runs/smoke/munich) with outputs under results/v02-smoke/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from linkgym.evaluation import cluster_bootstrap, paired_bootstrap, trace_episodes  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATASETS = {  # realization: (path, SHA-256)
    "main": (
        "data/munich-v1.h5",
        "4b3bbb246c40f4513a685d069a9e913649d58a88e4dbdc908b3fa4e074f3cd9c",
    ),
    "alt": (
        "data/munich-v1-test-alt.h5",
        "0969ea3638aa34c2c7547d66d8de9746bc3109f59a324844c07dadb9b359c347",
    ),
}
SCENARIO = {"channel": "trace", "snr_mode": "normalized"}  # all else: v0.1 defaults
WINDOWS = 4
VAL_FIRST_SEED = 5000
TEST_FIRST_SEED = 6000
OLLA_TARGETS = (0.05, 0.1, 0.2)
OLLA_DELTA_UPS = (0.1, 0.25, 0.5, 1.0)
DEFAULT_OLLA = (0.1, 1.0)  # Sionna's default step, no selection
M3_TUNED_OLLA = (0.1, 0.25)  # the M3 cell, for Q4
TDL_TEST_SEEDS = tuple(range(1000, 1010))
TDL_EPISODES_PER_SEED = 5
SEEDS = (0, 1, 2)
BOOTSTRAP = {"num_resamples": 10_000, "confidence": 0.95, "seed": 0}
CATEGORIES = ("los", "transition", "nlos")
METRICS = ("goodput_mbps", "observed_tbler", "mean_mcs")
PRIMARY = (  # Q1 and Q2, overall; also per category (secondary)
    ("PPO-TDL", "OLLA tuned"),
    ("PPO-Munich", "OLLA tuned"),
    ("PPO-Munich", "PPO-TDL"),
)


def settings(smoke: bool) -> dict[str, Any]:
    if smoke:
        sample = "tests/data/munich_sample.h5"
        smoke_model = "runs/smoke/munich/model.zip"
        return {
            "datasets": {"main": (sample, None), "alt": (sample, None)},
            "windows": 1,
            "ppo_munich": {s: smoke_model for s in SEEDS},
            "ppo_tdl": {s: f"runs/m3/gamma0_seed{s}/model.zip" for s in SEEDS},
            "munich_runs": ["runs/smoke/munich"],
            "tdl_seeds": TDL_TEST_SEEDS[:1],
            "tdl_episodes": 1,
            "results": REPO / "results" / "v02-smoke",
            "docs": REPO / "results" / "v02-smoke" / "docs",
        }
    return {
        "datasets": DATASETS,
        "windows": WINDOWS,
        "ppo_munich": {s: f"runs/v02/munich_gamma0_seed{s}/model.zip" for s in SEEDS},
        "ppo_tdl": {s: f"runs/m3/gamma0_seed{s}/model.zip" for s in SEEDS},
        "munich_runs": [f"runs/v02/munich_gamma0_seed{s}" for s in SEEDS],
        "tdl_seeds": TDL_TEST_SEEDS,
        "tdl_episodes": TDL_EPISODES_PER_SEED,
        "results": REPO / "results" / "v02",
        "docs": REPO / "docs" / "results" / "v02",
    }


# Policies and parallel evaluation


def olla_label(target: float, delta_up: float) -> str:
    return f"OLLA ({target:g}, {delta_up:g} dB)"


def build_policy(kind: str, param: Any) -> Any:
    if kind == "ppo":
        from stable_baselines3 import PPO

        from linkgym.baselines import SB3Policy

        return SB3Policy(PPO.load(str(REPO / param), device="cpu"))
    from linkgym.baselines import FixedMCSPolicy, ILLAPolicy, OLLAPolicy, OraclePolicy

    if kind == "olla":
        return OLLAPolicy(param[0], delta_up=param[1])
    return {"illa": ILLAPolicy, "oracle": OraclePolicy, "fixed": FixedMCSPolicy}[kind](param)


def run_job(job: dict[str, Any]) -> tuple[str, list[dict[str, Any]], float]:
    """Worker: evaluate one policy on pinned episodes (trace) or seeds (TDL)."""
    import torch

    torch.set_num_threads(1)
    from linkgym import evaluate
    from linkgym.evaluation import evaluate_episodes

    start = time.perf_counter()
    policy = build_policy(job["kind"], job["param"])
    if job["episodes"] is not None:
        rows = evaluate_episodes(policy, job["env_kwargs"], job["episodes"])
    else:
        rows = evaluate(policy, {}, job["seeds"], job["episodes_per_seed"])["per_episode"]
    return job["key"], rows, time.perf_counter() - start


def run_jobs(jobs: list[dict[str, Any]], workers: int) -> dict[str, list[dict[str, Any]]]:
    results = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for key, rows, seconds in pool.map(run_job, jobs):
            results[key] = rows
            print(f"  {key}: {len(rows)} episodes, {seconds:.0f} s", flush=True)
    return results


def baseline_policies(tuned: tuple[float, float] | None) -> list[tuple[str, str, Any]]:
    policies = [
        ("fixed MCS 14", "fixed", 14),
        ("ILLA 0.1", "illa", 0.1),
        ("OLLA default", "olla", DEFAULT_OLLA),
        ("oracle 0.1", "oracle", 0.1),
    ]
    if tuned is not None:
        policies.insert(3, ("OLLA tuned", "olla", tuned))
    return policies


def ppo_policies(cfg: dict[str, Any]) -> list[tuple[str, str, Any]]:
    out = [(f"PPO-TDL seed {s}", "ppo", path) for s, path in cfg["ppo_tdl"].items()]
    out += [(f"PPO-Munich seed {s}", "ppo", path) for s, path in cfg["ppo_munich"].items()]
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_datasets(cfg: dict[str, Any]) -> dict[str, dict[str, str]]:
    checked = {}
    for name, (path, expected) in cfg["datasets"].items():
        digest = sha256(REPO / path)
        if expected is not None and digest != expected:
            raise SystemExit(f"{path}: SHA-256 {digest} differs from the protocol's {expected}")
        checked[name] = {"path": path, "sha256": digest}
    return checked


def episodes_for(cfg: dict[str, Any], realization: str, split: str) -> list[dict[str, Any]]:
    path = REPO / cfg["datasets"][realization][0]
    first = VAL_FIRST_SEED if split == "val" else TEST_FIRST_SEED
    return trace_episodes(path, [split], windows=cfg["windows"], first_seed=first)


def env_kwargs_for(cfg: dict[str, Any], realization: str, split: str) -> dict[str, Any]:
    path = str(REPO / cfg["datasets"][realization][0])
    return SCENARIO | {"trace_path": path, "trace_splits": (split,)}


def save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


# Phases


def phase_validation(cfg: dict[str, Any], workers: int) -> None:
    datasets = check_datasets(cfg)
    episodes = episodes_for(cfg, "main", "val")
    kwargs = env_kwargs_for(cfg, "main", "val")
    grid = [(t, d) for t in OLLA_TARGETS for d in OLLA_DELTA_UPS]
    policies = [(olla_label(t, d), "olla", (t, d)) for t, d in grid]
    policies += [p for p in baseline_policies(None) if p[0] != "OLLA default"]
    policies += ppo_policies(cfg)
    jobs = [
        {"key": label, "kind": kind, "param": param, "env_kwargs": kwargs, "episodes": episodes}
        for label, kind, param in policies
    ]
    print(f"validation: {len(jobs)} policies x {len(episodes)} episodes")
    results = run_jobs(jobs, workers)
    goodput = {
        cell: np.mean([r["goodput_mbps"] for r in results[olla_label(*cell)]]) for cell in grid
    }
    tuned = max(grid, key=lambda cell: goodput[cell])
    print(f"tuned OLLA on val: target {tuned[0]:g}, delta_up {tuned[1]:g} dB")
    save(
        cfg["results"] / "validation.json",
        {"datasets": datasets, "tuned_olla": tuned, "results": results},
    )


def phase_test(cfg: dict[str, Any], workers: int, force: bool) -> None:
    out = cfg["results"] / "test.json"
    if out.exists() and not force:
        raise SystemExit(f"{out} exists: the test split is evaluated once")
    datasets = check_datasets(cfg)
    validation = json.loads((cfg["results"] / "validation.json").read_text(encoding="utf-8"))
    tuned = tuple(validation["tuned_olla"])
    key = ("seed", "route", "route_trajectory", "window", "category")
    episodes = {r: episodes_for(cfg, r, "test") for r in ("main", "alt")}
    if [[e[k] for k in key] for e in episodes["main"]] != [
        [e[k] for k in key] for e in episodes["alt"]
    ]:
        raise SystemExit("the test episodes of the two realizations differ")
    policies = baseline_policies(tuned) + ppo_policies(cfg)
    jobs = [
        {
            "key": f"{realization}|{label}",
            "kind": kind,
            "param": param,
            "env_kwargs": env_kwargs_for(cfg, realization, "test"),
            "episodes": episodes[realization],
        }
        for realization in ("main", "alt")
        for label, kind, param in policies
    ]
    print(f"test: {len(policies)} policies x {len(episodes['main'])} episodes x 2 realizations")
    results = run_jobs(jobs, workers)
    by_realization = {r: {} for r in ("main", "alt")}
    for k, rows in results.items():
        realization, label = k.split("|", 1)
        by_realization[realization][label] = rows
    save(out, {"datasets": datasets, "tuned_olla": tuned, "results": by_realization})


def phase_q4(cfg: dict[str, Any], workers: int) -> None:
    policies = ppo_policies(cfg) + [("OLLA tuned (M3)", "olla", M3_TUNED_OLLA)]
    jobs = [
        {
            "key": label,
            "kind": kind,
            "param": param,
            "episodes": None,
            "seeds": list(cfg["tdl_seeds"]),
            "episodes_per_seed": cfg["tdl_episodes"],
        }
        for label, kind, param in policies
    ]
    print(f"q4: {len(jobs)} policies on the TDL test seeds")
    save(cfg["results"] / "q4.json", {"results": run_jobs(jobs, workers)})


# Report


def seed_mean(results: dict[str, list[dict[str, Any]]], family: str) -> list[dict[str, Any]]:
    """Per-episode mean over the training seeds of a PPO family."""
    runs = [results[f"{family} seed {s}"] for s in SEEDS]
    rows = []
    for episode in zip(*runs, strict=True):
        row = dict(episode[0])
        for metric in METRICS:
            row[metric] = float(np.mean([e[metric] for e in episode]))
        rows.append(row)
    return rows


def with_means(results: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return results | {f: seed_mean(results, f) for f in ("PPO-TDL", "PPO-Munich")}


def subset(rows: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    return rows if category == "all" else [r for r in rows if r["category"] == category]


def clusters(rows: list[dict[str, Any]]) -> list[str]:
    return [f"{r['route']}#{r['route_trajectory']}" for r in rows]


def paired(a_rows, b_rows, metric="goodput_mbps") -> dict[str, float]:
    a = [r[metric] for r in a_rows]
    b = [r[metric] for r in b_rows]
    return cluster_bootstrap(a, b, clusters(b_rows), **BOOTSTRAP)


def verdict(main: dict[str, float], alt: dict[str, float]) -> str:
    def sign(s):  # +1 / -1 if the CI excludes zero, else 0
        return 1 if s["ci_low"] > 0 else -1 if s["ci_high"] < 0 else 0

    m, a = sign(main), sign(alt)
    if m != 0 and m == a:
        return "holds"
    if m != 0 and a != 0 and m != a:
        return "contradicted"
    return "inconclusive"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(results, realization: str) -> list[dict[str, Any]]:
    rows = []
    for label, episodes in results.items():
        for category in ("all", *CATEGORIES):
            sel = subset(episodes, category)
            if not sel:
                continue
            ci = cluster_bootstrap(
                [r["goodput_mbps"] for r in sel], None, clusters(sel), **BOOTSTRAP
            )
            rows.append(
                {
                    "realization": realization,
                    "policy": label,
                    "category": category,
                    "episodes": len(sel),
                    "trajectories": ci["num_clusters"],
                    "goodput_mbps": ci["mean"],
                    "goodput_ci_low": ci["ci_low"],
                    "goodput_ci_high": ci["ci_high"],
                    "observed_tbler": float(np.mean([r["observed_tbler"] for r in sel])),
                    "mean_mcs": float(np.mean([r["mean_mcs"] for r in sel])),
                }
            )
    return rows


def paired_rows(results, realization: str) -> list[dict[str, Any]]:
    comparisons = [(p, "OLLA tuned") for p in results if p != "OLLA tuned"]
    comparisons.append(("PPO-Munich", "PPO-TDL"))
    rows = []
    for a, b in comparisons:
        for category in ("all", *CATEGORIES):
            sa, sb = subset(results[a], category), subset(results[b], category)
            if not sa:
                continue
            stats = paired(sa, sb)
            tbler = paired(sa, sb, "observed_tbler")
            rows.append(
                {
                    "realization": realization,
                    "comparison": f"{a} - {b}",
                    "category": category,
                    "primary": (a, b) in PRIMARY and category == "all",
                    "episodes": stats["num_elements"],
                    "trajectories": stats["num_clusters"],
                    "goodput_diff_mbps": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "pct": stats["pct"],
                    "pct_ci_low": stats["pct_ci_low"],
                    "pct_ci_high": stats["pct_ci_high"],
                    "tbler_diff": tbler["mean"],
                    "tbler_ci_low": tbler["ci_low"],
                    "tbler_ci_high": tbler["ci_high"],
                }
            )
    return rows


def route_rows(results, realization: str) -> list[dict[str, Any]]:
    rows = []
    routes = sorted({(r["route"], r["category"]) for r in results["OLLA tuned"]})
    for route, category in routes:
        base = [r for r in results["OLLA tuned"] if r["route"] == route]
        for label, episodes in results.items():
            sel = [r for r in episodes if r["route"] == route]
            stats = paired(sel, base) if label != "OLLA tuned" else None
            rows.append(
                {
                    "realization": realization,
                    "route": route,
                    "category": category,
                    "policy": label,
                    "trajectories": len({r["route_trajectory"] for r in sel}),
                    "goodput_mbps": float(np.mean([r["goodput_mbps"] for r in sel])),
                    "diff_vs_olla_tuned": None if stats is None else stats["mean"],
                    "diff_ci_low": None if stats is None else stats["ci_low"],
                    "diff_ci_high": None if stats is None else stats["ci_high"],
                    "observed_tbler": float(np.mean([r["observed_tbler"] for r in sel])),
                }
            )
    return rows


def phase_report(cfg: dict[str, Any]) -> None:
    out = cfg["docs"]
    out.mkdir(parents=True, exist_ok=True)
    validation = json.loads((cfg["results"] / "validation.json").read_text(encoding="utf-8"))
    test = json.loads((cfg["results"] / "test.json").read_text(encoding="utf-8"))
    q4 = json.loads((cfg["results"] / "q4.json").read_text(encoding="utf-8"))
    tuned = tuple(validation["tuned_olla"])

    # Validation: OLLA grid and all policies
    val = validation["results"]
    write_csv(
        out / "olla_tuning.csv",
        [
            {
                "bler_target": t,
                "delta_up_db": d,
                "goodput_mbps": float(np.mean([r["goodput_mbps"] for r in val[olla_label(t, d)]])),
                "observed_tbler": float(
                    np.mean([r["observed_tbler"] for r in val[olla_label(t, d)]])
                ),
                "mean_mcs": float(np.mean([r["mean_mcs"] for r in val[olla_label(t, d)]])),
                "selected": (t, d) == tuned,
            }
            for t in OLLA_TARGETS
            for d in OLLA_DELTA_UPS
        ],
    )
    val_named = dict(val) | {"OLLA tuned": val[olla_label(*tuned)]}
    val_named["OLLA default"] = val[olla_label(*DEFAULT_OLLA)]
    write_csv(out / "validation.csv", summary_rows(with_means(val_named), "main"))

    # Test: both realizations
    summaries, pairs, routes, robust = [], [], [], []
    tests = {r: with_means(test["results"][r]) for r in ("main", "alt")}
    for realization, results in tests.items():
        summaries += summary_rows(results, realization)
        pairs += paired_rows(results, realization)
        routes += route_rows(results, realization)
    for a, b in PRIMARY:
        for category in ("all", *CATEGORIES):
            if not subset(tests["main"][a], category):
                continue
            stats = {
                r: paired(subset(tests[r][a], category), subset(tests[r][b], category))
                for r in ("main", "alt")
            }
            robust.append(
                {
                    "comparison": f"{a} - {b}",
                    "category": category,
                    "main_mean": stats["main"]["mean"],
                    "main_ci_low": stats["main"]["ci_low"],
                    "main_ci_high": stats["main"]["ci_high"],
                    "alt_mean": stats["alt"]["mean"],
                    "alt_ci_low": stats["alt"]["ci_low"],
                    "alt_ci_high": stats["alt"]["ci_high"],
                    "verdict": verdict(stats["main"], stats["alt"]),
                }
            )
    write_csv(out / "test_summary.csv", summaries)
    write_csv(out / "paired.csv", pairs)
    write_csv(out / "per_route.csv", routes)
    write_csv(out / "robustness.csv", robust)

    # Q4: TDL test seeds, per-episode paired bootstrap as in M3
    q4_results = with_means(q4["results"])
    base = q4_results["OLLA tuned (M3)"]
    q4_rows = []
    for label, episodes in q4_results.items():
        row = {
            "policy": label,
            "goodput_mbps": float(np.mean([r["goodput_mbps"] for r in episodes])),
            "observed_tbler": float(np.mean([r["observed_tbler"] for r in episodes])),
            "mean_mcs": float(np.mean([r["mean_mcs"] for r in episodes])),
        }
        if label != "OLLA tuned (M3)":
            stats = paired_bootstrap(
                [r["goodput_mbps"] for r in episodes],
                [r["goodput_mbps"] for r in base],
                **BOOTSTRAP,
            )
            row |= {f"diff_{k}": v for k, v in stats.items()}
        q4_rows.append(row)
    write_csv(out / "q4.csv", q4_rows)

    # Learning curves on Munich val
    curves = []
    for run in cfg["munich_runs"]:
        with (REPO / run / "eval_log.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                curves.append({"run": Path(run).name} | row)
    write_csv(out / "learning_curves.csv", curves)

    plot_categories(out, tests["main"], tuned)
    plot_realizations(out, robust)
    plot_learning_curves(out, curves, val_named)
    print(f"report written to {out}")


MAIN_POLICIES = ("OLLA default", "OLLA tuned", "PPO-TDL", "PPO-Munich", "oracle 0.1")


def plot_categories(out: Path, results, tuned) -> None:
    strata = ("all", *CATEGORIES)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(MAIN_POLICIES)
    for i, label in enumerate(MAIN_POLICIES):
        means, low, high = [], [], []
        for category in strata:
            sel = subset(results[label], category)
            if not sel:
                means.append(np.nan)
                low.append(0.0)
                high.append(0.0)
                continue
            ci = cluster_bootstrap(
                [r["goodput_mbps"] for r in sel], None, clusters(sel), **BOOTSTRAP
            )
            means.append(ci["mean"])
            low.append(ci["mean"] - ci["ci_low"])
            high.append(ci["ci_high"] - ci["mean"])
        x = np.arange(len(strata)) + (i - (len(MAIN_POLICIES) - 1) / 2) * width
        name = label if label != "OLLA tuned" else f"OLLA tuned ({tuned[0]:g}, {tuned[1]:g} dB)"
        ax.bar(x, means, width, yerr=[low, high], capsize=2, label=name)
    ax.set_xticks(np.arange(len(strata)), ["all", "LoS", "transition", "NLoS"])
    ax.set_ylabel("goodput [Mbit/s]")
    ax.set_title("Munich test split (main realization): mean goodput, 95% cluster bootstrap CI")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out / "goodput_by_category.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_realizations(out: Path, robust: list[dict[str, Any]]) -> None:
    comparisons = [f"{a} - {b}" for a, b in PRIMARY]
    strata = ("all", *CATEGORIES)
    fig, axes = plt.subplots(1, len(comparisons), figsize=(13, 4), sharey=False)
    for ax, comparison in zip(axes, comparisons, strict=True):
        for j, (realization, marker) in enumerate((("main", "o"), ("alt", "s"))):
            rows = [r for r in robust if r["comparison"] == comparison]
            rows.sort(key=lambda r: strata.index(r["category"]))
            x = np.array([strata.index(r["category"]) for r in rows]) + (j - 0.5) * 0.25
            mean = np.array([r[f"{realization}_mean"] for r in rows])
            err = [
                mean - np.array([r[f"{realization}_ci_low"] for r in rows]),
                np.array([r[f"{realization}_ci_high"] for r in rows]) - mean,
            ]
            ax.errorbar(x, mean, yerr=err, fmt=marker, capsize=3, label=realization)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(np.arange(len(strata)), ["all", "LoS", "trans.", "NLoS"])
        ax.set_title(comparison, fontsize=10)
        ax.set_ylabel("goodput difference [Mbit/s]")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(title="realization", fontsize=8)
    fig.suptitle("Paired differences on the test split, 95% cluster bootstrap CI", fontsize=11)
    fig.savefig(out / "realizations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(out: Path, curves, val_named) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for run in sorted({c["run"] for c in curves}):
        rows = [c for c in curves if c["run"] == run]
        ax.plot(
            [int(r["timesteps"]) for r in rows],
            [float(r["goodput_mbps"]) for r in rows],
            marker="o",
            ms=3,
            label=run,
        )
    for label, style in (("OLLA tuned", "--"), ("OLLA default", ":")):
        level = np.mean([r["goodput_mbps"] for r in val_named[label]])
        ax.axhline(level, color="k", ls=style, lw=1, label=f"{label} (val)")
    ax.set_xlabel("training steps")
    ax.set_ylabel("val goodput [Mbit/s]")
    ax.set_title("PPO trained on Munich train streets, evaluated on the val split")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("phase", choices=("validation", "test", "q4", "report"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="sample trace and smoke model")
    parser.add_argument("--force", action="store_true", help="re-run the test phase (smoke)")
    args = parser.parse_args()
    cfg = settings(args.smoke)
    if args.force and not args.smoke:
        raise SystemExit("--force is for smoke runs: the test split is evaluated once")
    phases: dict[str, Callable[[], None]] = {
        "validation": lambda: phase_validation(cfg, args.workers),
        "test": lambda: phase_test(cfg, args.workers, args.force),
        "q4": lambda: phase_q4(cfg, args.workers),
        "report": lambda: phase_report(cfg),
    }
    start = time.perf_counter()
    phases[args.phase]()
    print(f"{args.phase}: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
