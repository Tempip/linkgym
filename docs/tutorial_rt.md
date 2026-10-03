# Tutorial: from a Sionna RT scene to a trained and evaluated agent

This tutorial builds a small ray-traced dataset on Sionna RT's `munich` scene, trains PPO
on it and compares it with Sionna's OLLA on held-out streets. The same steps work with
your own scene. Reference: [channels.md](channels.md) (trace format, generator, Munich
dataset).

Steps 1-5 need a CUDA GPU (tested on an RTX 3060, 12 GB); steps 6-9 run on a CPU. The Python
code of steps 3 and 6-8 runs in the test suite as written (`tests/test_docs.py`), on the
sample trace `tests/data/munich_sample.h5` from the repository: four trajectories of 1000
slots, two train, one val, one test. Set `TRACE` to your own file instead.

## 1. Two environments

Ray tracing and training go in separate environments: installing the `rt` extra where you
train can break `import linkgym` on machines without a CUDA GPU or LLVM (see
[channels.md](channels.md#installation)).

```bash
python -m venv .venv-rt                    # generation: needs a CUDA GPU
.venv-rt/bin/pip install "linkgym[rt]"     # Windows: .venv-rt\Scripts\pip
python -m venv .venv                       # training and evaluation: CPU is enough
.venv/bin/pip install "linkgym[train]"
```

## 2. Choose a scene

Any Mitsuba 3 XML scene that Sionna RT 2.1 can load works; give its path, relative to the
configuration file, as `scene`. The built-in Sionna RT scenes are given by name, e.g.
`"munich"` (about 1.5 x 1.2 km around the Frauenkirche). The generator records the scene's
SHA-256 (over the XML and every file it references) in the trace, so a trace can be traced
back to its scene.

Scenes built from OpenStreetMap, like `munich`, derive from data under the Open Database
License: keep the attribution (stored in every trace by default for the built-in OSM
scenes, or set with `attribution`) when you share traces or results.

## 3. Write the configuration

One transmitter, the PRB grid of the environment's default scenario (3.5 GHz, 30 kHz, 52
PRBs) and routes split by street: every route goes entirely to one split, so no street is
seen in training and in evaluation. Save this as `tutorial.json`; it uses the Munich
dataset's base station and three of its streets:

```json
{
  "description": "linkgym tutorial: three Munich streets, one per split",
  "scene": "munich",
  "carrier_frequency_hz": 3.5e9,
  "subcarrier_spacing_hz": 30e3,
  "num_prb": 52,
  "prb_sampling": "center",
  "transmitter": {"position": [116.5, 80.5, 23.4], "antenna": {"pattern": "iso", "polarization": "V"}},
  "receiver": {"height_m": 1.5, "antenna": {"pattern": "iso", "polarization": "V"}},
  "speed_mps": 15.0,
  "num_slots": 4000,
  "anchor_spacing_slots": 10,
  "min_clearance_m": 1.0,
  "min_mean_gain_db": -150.0,
  "solver": {"max_depth": 10, "los": true, "specular_reflection": true,
             "diffuse_reflection": false, "refraction": false, "diffraction": true,
             "edge_diffraction": false, "diffraction_lit_region": true,
             "samples_per_src": 4000000, "max_num_paths_per_src": 10000000, "seed": 42},
  "routes": [
    {"name": "north-canyon", "split": "train",
     "waypoints": [[113, 112], [118, 140], [120, 160], [119, 230], [120, 290], [124, 320], [130, 350], [130, 378]]},
    {"name": "north-street", "split": "val",
     "waypoints": [[17.5, 185.5], [17.5, 207.5], [32.5, 246.5], [33.5, 252.5], [39.5, 259.5], [39.5, 268.5], [46.5, 285.5], [61.5, 332.5], [61.5, 342.5], [59.5, 344.5]]},
    {"name": "long-diagonal", "split": "test",
     "waypoints": [[299.5, 111.5], [299.5, 107.5], [288.5, 87.5], [281.5, 80.5], [260.5, 52.5], [256.5, 48.5], [248.5, 48.5], [235.5, 33.5], [225.5, 10.5], [220.5, 5.5], [217.5, -0.5], [212.5, -11.5], [212.5, -14.5], [203.5, -25.5], [190.5, -49.5], [188.5, -56.5], [185.5, -59.5], [185.5, -67.5], [173.5, -80.5]]}
  ]
}
```

Each route is cut into trajectories of `num_slots` slots (4000 slots, 2 s, 30 m at 15 m/s),
with a path solve every `anchor_spacing_slots` slots. The solver settings are the Munich
dataset's: refraction off (the OSM buildings are solid blocks, see
[known limitations](channels.md#known-limitations-of-the-munich-traces)), diffraction on.
Without `category`, a route's category (`los`, `transition`, `nlos`) is derived from its
line-of-sight share at generation.

The configuration can be checked without a GPU, in either environment:

```python
from linkgym.rt.config import load_config, split_overlap

config = load_config("tutorial.json")
print([(route.name, route.split) for route in config.routes])
print(split_overlap(config))  # share of each route within 10 m of another split: want 0
```

## 4. Check the routes against the scene and generate

In the `rt` environment:

```bash
linkgym-traces check tutorial.json
linkgym-traces plot tutorial.json -o tutorial_routes.png --labels
linkgym-traces generate tutorial.json -o my_trace.h5 --max-trajectories 1   # quick test
linkgym-traces generate tutorial.json -o my_trace.h5 --overwrite
```

`check` takes seconds: it verifies that every slot position is on open ground with enough
clearance, and prints each route's length, trajectory count, line-of-sight share and
overlap with other splits. A route that fails stops everything and names the positions;
move its waypoints to the street centre. `generate` solves about 9 path sets per second on
an RTX 3060, so these 20 trajectories (8000 path solves) take about 15 minutes.
Trajectories with an anchor without any path, or with a mean gain below
`min_mean_gain_db`, are dropped, and the reason is printed and stored per route.

## 5. Check the ray budget

Ray tracing samples rays, and with these budgets the channel does not converge: more rays
find more weak paths (see [channels.md](channels.md#known-limitations-of-the-munich-traces)).
Before trusting results, generate the evaluation split a second time with another budget
and evaluate on both; a conclusion that changes between the two realizations is not
supported:

```bash
linkgym-traces generate tutorial.json -o my_trace_alt.h5 --splits test --solver samples_per_src=8000000 max_num_paths_per_src=1000000
linkgym-traces accuracy tutorial.json --route north-canyon --start-m 0 --num-slots 1000
```

`accuracy` compares the anchored channel with a path solve at every slot, for several
anchor spacings, on one stretch of a route.

## 6. Inspect the trace

From here on, use the training environment. Copy the trace over and set `TRACE` to it:

```python
from collections import Counter

import h5py
import numpy as np

from linkgym.channels import inspect_trace

TRACE = "tests/data/munich_sample.h5"  # your trace, e.g. "my_trace.h5"
info = inspect_trace(TRACE)
print(info.shape, Counter(info.split.tolist()))  # (trajectories, slots, PRBs), splits
```

With the default `snr_mode="normalized"`, each trajectory is divided by its mean linear
gain and the episode's SNR is drawn from 5-20 dB. On a trajectory with a large dynamic
range, a short strong stretch sets the mean and the rest falls far below the drawn SNR,
into stretches where not even the lowest MCS works
([channels.md](channels.md#snr-modes)). Look at the spread of each trajectory's normalized
wideband gain before training:

```python
with h5py.File(TRACE, "r") as f:
    for t in range(info.shape[0]):
        gain = f["gain"][t].astype(np.float64)
        rel = 10 * np.log10(gain.mean(axis=1) / gain.mean())  # wideband gain over the mean
        print(
            f"trajectory {t} ({info.split[t]}): median {np.median(rel):.1f} dB, "
            f"10th percentile {np.percentile(rel, 10):.1f} dB"
        )
```

In the Munich test split, no trajectory whose 10th percentile was within 5 dB of the mean
had such outage slots, and every trajectory with more than 10 % of its slots in outage had
a 10th percentile 15 dB or more below the mean. Outage slots raise every policy's observed
TBLER.

## 7. Train

A short PPO run on the train split, the same as the README's quickstart but on the trace:

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import linkgym

env_kwargs = {"channel": "trace", "trace_path": TRACE, "trace_splits": ("train",)}
vec_env = make_vec_env(
    lambda: gym.make("linkgym/LinkAdaptation-v0", **env_kwargs), n_envs=4, seed=0
)
model = PPO("MlpPolicy", vec_env, gamma=0.0, seed=0).learn(total_timesteps=8_192)
vec_env.close()
```

Each `reset()` draws a train trajectory and a 1000-slot window of it, and an SNR. For a
full run, with validation curves on pinned val episodes, use the training script (1M steps,
about 20 minutes on 8 CPU cores); it refuses to train on `val` or `test` and records the
trace's SHA-256 in `run.json`:

```bash
python examples/train_ppo.py --gamma 0 --seed 0 --channel trace --trace-path my_trace.h5 --eval-freq 100000
```

## 8. Evaluate

Evaluate on enumerated, pinned episodes: every trajectory of a split cut into
non-overlapping windows, each with a fixed reset seed, so that every policy faces the same
channels, SNRs and ACK draws. First tune OLLA's TBLER target and step on val, as a fair
baseline:

```python
from linkgym.baselines import OLLAPolicy, SB3Policy
from linkgym.evaluation import cluster_bootstrap, evaluate_episodes, trace_episodes


def mean_goodput(policy, split, episodes):
    results = evaluate_episodes(policy, env_kwargs | {"trace_splits": (split,)}, episodes)
    return np.mean([r["goodput_mbps"] for r in results]), results


val = trace_episodes(TRACE, ["val"], first_seed=5000)
grid = [(target, step) for target in (0.05, 0.1) for step in (0.25, 1.0)]
olla_val = {c: mean_goodput(OLLAPolicy(c[0], delta_up=c[1]), "val", val)[0] for c in grid}
target, step = max(olla_val, key=olla_val.get)
print(f"OLLA selected on val: target {target}, delta_up {step} dB")
```

Then evaluate once on test, PPO against the tuned OLLA, paired on the same episodes. The
episodes of a trajectory are correlated, so the confidence interval resamples
trajectories (a cluster bootstrap), not episodes:

```python
test = trace_episodes(TRACE, ["test"], first_seed=6000)
_, ppo = mean_goodput(SB3Policy(model), "test", test)
_, olla = mean_goodput(OLLAPolicy(target, delta_up=step), "test", test)
diff = cluster_bootstrap(
    [r["goodput_mbps"] for r in ppo],
    [r["goodput_mbps"] for r in olla],
    clusters=[r["trajectory"] for r in ppo],
)
print(
    f"PPO - OLLA: {diff['mean']:+.2f} Mbit/s, 95% CI [{diff['ci_low']:+.2f}, "
    f"{diff['ci_high']:+.2f}] over {diff['num_clusters']} test trajectories"
)
```

With one test trajectory, as in the sample trace, the interval is undefined (NaN); with
the trace from step 4, the test street has 7 trajectories. Each result row also carries
the episode's `route`, `category`, `snr_db` and `observed_tbler`, for results by route
category.

## 9. What to report

- **Fix the protocol before evaluating on test**, and evaluate on test once:
  [docs/results/v02/PROTOCOL.md](results/v02/PROTOCOL.md) is an example, and
  `examples/evaluate_v02.py` implements it.
- **Paired differences with cluster bootstrap CIs over trajectories**, per route category
  as well as overall. Say that the CIs do not include training-seed variance, and report
  each training seed.
- **Both ray-tracing realizations** (step 5): a conclusion that changes between them is not
  supported.
- **The share of outage slots** (step 6), since they raise every policy's observed TBLER:
  `examples/olla_test_traces.py` counts them per trajectory.
- **The scope**: one scene, one base station, a few streets per split, one ray-tracing
  configuration. The results describe these streets, not sites in general.
