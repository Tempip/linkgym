# linkgym

[![CI](https://github.com/Tempip/linkgym/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Tempip/linkgym/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Tempip/linkgym/blob/main/LICENSE)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://github.com/Tempip/linkgym/blob/main/pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/linkgym.svg)](https://pypi.org/project/linkgym/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23135621.svg)](https://doi.org/10.5281/zenodo.23135621)

linkgym is a Gymnasium environment for 5G NR link adaptation (MCS selection) built on NVIDIA
Sionna SYS, with TDL and ray-traced channels, classical baselines and a fixed evaluation
protocol.

![Paired goodput difference against tuned OLLA on the TDL channel and on ray-traced Munich streets](https://raw.githubusercontent.com/Tempip/linkgym/main/docs/assets/header.png)

*Paired goodput difference against Sionna's OLLA tuned on validation data, with 95% CIs.
Left: the default TDL-A scenario, 50 held-out test episodes. Right: 92 episodes on four
held-out ray-traced Munich streets (main ray-tracing realization; the CIs resample the 23
test trajectories). Each PPO is the mean of three training seeds; the CIs do not include
training-seed variance. The oracle knows the channel of the slot being decided; its CI on
TDL is computed from the M3 per-episode results. Details:
[TDL results](https://github.com/Tempip/linkgym/blob/main/docs/results/m3/README.md),
[Munich results](https://github.com/Tempip/linkgym/blob/main/docs/results/v02/README.md).*

On the default TDL channel, PPO trained on linkgym delivers +16.0% goodput over a
validation-tuned OLLA (95% CI [+14.9%, +17.1%], 50 held-out episodes). On ray-traced
streets in Munich (Sionna RT, four held-out streets), that advantage is gone: there is no
detectable difference from tuned OLLA, neither for PPO trained on TDL (-2.0%, CI [-4.9%,
+2.2%]) nor for PPO trained on the Munich training streets (+2.7%, CI [-0.6%, +7.4%]).
Training on the site does beat training on TDL, by +4.8% (CI [+4.0%, +5.5%]), in every
route category and on two ray-tracing realizations. Each policy does best on the kind of
channel it was trained on: PPO trained on Munich is 8.3% below tuned OLLA on TDL.

## Why

Recent RL link-adaptation studies on Sionna (see Related work) each build their own
environment. linkgym is a reusable, pip-installable Gymnasium environment,
registered as `linkgym/LinkAdaptation-v0`, with built-in baselines (fixed MCS, ILLA, OLLA,
oracle) and a fixed evaluation protocol. All policies are evaluated with common random
numbers: for a given seed, every policy faces the same channel and the same ACK draws.
Besides 3GPP TDL fading, the environment runs on channel traces: a published ray-traced
dataset of Munich streets, traces you generate from your own Sionna RT scene, or any
channel you can write as an array of gains.

## Installation

```bash
pip install linkgym           # environment and baselines
pip install "linkgym[train]"  # adds Stable-Baselines3 and TensorBoard for the PPO example
```

From source, with the training extras:

```bash
git clone https://github.com/Tempip/linkgym.git
cd linkgym
pip install -e ".[train]"
```

Requires Python >= 3.11 (tested on 3.11 and 3.12); sionna-no-rt 2.1.x is installed
automatically. A CPU-only PyTorch build is enough. To avoid downloading CUDA wheels, install
it first with `pip install torch --index-url https://download.pytorch.org/whl/cpu`. The
simulator runs faster with one torch thread; call `torch.set_num_threads(1)` in your own
script (the package does not change it).

To generate traces with Sionna RT, install `pip install "linkgym[rt]"` **in a separate
environment**: installing it where you train can break imports on machines without a CUDA
GPU or LLVM
([why](https://github.com/Tempip/linkgym/blob/main/docs/channels.md#installation)).

## Quickstart

Create the environment and evaluate Sionna's OLLA with `linkgym.evaluate`:

```python
import gymnasium as gym

import linkgym
from linkgym.baselines import OLLAPolicy

env = gym.make("linkgym/LinkAdaptation-v0", speed=3.0)  # any ScenarioConfig field
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

result = linkgym.evaluate(
    OLLAPolicy(bler_target=0.1), env_kwargs={"speed": 3.0}, seeds=[1000, 1001]
)
print(result["goodput_mbps"]["mean"], result["observed_tbler"]["mean"])
```

Train PPO with Stable-Baselines3 and evaluate it the same way. This needs the `[train]` extra
(`pip install "linkgym[train]"`); it is a short run, the results below use 1M steps:

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import linkgym
from linkgym.baselines import SB3Policy

vec_env = make_vec_env(lambda: gym.make("linkgym/LinkAdaptation-v0"), n_envs=4, seed=0)
model = PPO("MlpPolicy", vec_env, gamma=0.0, seed=0).learn(total_timesteps=8_192)
print(linkgym.evaluate(SB3Policy(model), seeds=[1000])["goodput_mbps"]["mean"])
```

A longer walk-through is in
[examples/quickstart.ipynb](https://github.com/Tempip/linkgym/blob/main/examples/quickstart.ipynb).

## Results

### TDL channel (v0.1)

Default scenario, held-out test seeds 1000-1009, 5 episodes each (50 episodes of 1000
slots). Means over the test seeds; PPO is the mean of 3 training seeds (std 0.04 Mbit/s
across them). All policies except the oracle observe exactly the same delayed feedback
(wideband SINR, HARQ ACK/NACK, MCS used).

| policy | goodput Mbit/s | observed TBLER |
|---|---:|---:|
| fixed MCS 14 | 23.34 | 0.277 |
| ILLA, no outer loop (target 0.1) | 19.45 | 0.524 |
| OLLA, default step (target 0.1, `delta_up` 1 dB) | 30.35 | 0.103 |
| OLLA, tuned on validation seeds (target 0.1, `delta_up` 0.25 dB) | 30.89 | 0.110 |
| PPO, gamma = 0 | 35.82 | 0.067 |
| oracle (ILLA on the channel of the current slot) | 38.62 | 0.026 |

PPO vs tuned OLLA, paired over the 50 test episodes: **+4.93 Mbit/s (+16.0%) goodput per
episode, 95% bootstrap CI [+4.65, +5.21] Mbit/s ([+14.9%, +17.1%])**. The protocol,
validation-based selection, the other comparisons and the figures are in
[docs/results/m3/README.md](https://github.com/Tempip/linkgym/blob/main/docs/results/m3/README.md).
These results hold for this TDL scenario only.

### Ray-traced Munich streets (v0.2)

The [Munich dataset](#munich-dataset): 15 streets around one rooftop base station, split by
street into train (8), val (3) and test (4). Test: 92 episodes of 1000 slots on 23
trajectories. OLLA was tuned on the val streets (target 0.05, `delta_up` 0.25 dB), PPO
trained on the train streets (3 training seeds, means reported), and the protocol was
committed before training. Paired differences with 95% cluster bootstrap CIs over test
trajectories; "holds" means the classification (better, worse, no detectable difference)
is the same on a second ray-tracing realization of the test streets.

| comparison | goodput difference | verdict |
|---|---|---|
| PPO trained on TDL - tuned OLLA | -0.71 Mbit/s (-2.0%), CI [-4.9%, +2.2%] | inconclusive |
| PPO trained on Munich - tuned OLLA | +0.96 Mbit/s (+2.7%), CI [-0.6%, +7.4%] | inconclusive |
| PPO trained on Munich - PPO trained on TDL | **+1.67 Mbit/s (+4.8%), CI [+4.0%, +5.5%]** | holds, in every route category |

| policy | goodput Mbit/s | observed TBLER |
|---|---:|---:|
| fixed MCS 14 | 21.79 | 0.325 |
| ILLA, no outer loop (target 0.1) | 30.64 | 0.320 |
| OLLA, default step (target 0.1, `delta_up` 1 dB) | 35.52 | 0.172 |
| OLLA, tuned on the val streets (target 0.05, `delta_up` 0.25 dB) | 35.82 | 0.129 |
| PPO trained on TDL | 35.11 | 0.125 |
| PPO trained on Munich | 36.78 | 0.113 |
| oracle | 38.77 | 0.110 |

Against tuned OLLA, by route category, PPO trained on TDL is worse on the line-of-sight
street and PPO trained on Munich better on the NLoS streets; the other differences are
inconclusive. Tuned OLLA is within 3 Mbit/s of the oracle here, which leaves less room than
on TDL. With the default SNR normalization, 9% of the test slots are outages in which even
the lowest MCS has a TBLER above 5%; a post hoc analysis shows that they account for tuned
OLLA's observed TBLER of 0.129 against its 0.05 target. Protocol, per-category and
per-route results, both realizations and the analysis:
[docs/results/v02/README.md](https://github.com/Tempip/linkgym/blob/main/docs/results/v02/README.md).

## Munich dataset

Ray-traced channel traces of 15 streets around one rooftop base station in Sionna RT's
`munich` scene: 76 trajectories of 4000 slots (2 s at 15 m/s), 52 PRBs at 3.5 GHz, split
by street into train, val and test, plus a second ray-tracing realization of the test
streets. Published on Zenodo under the Open Database License (ODbL) 1.0: version v1 has
the DOI [10.5281/zenodo.23135098](https://doi.org/10.5281/zenodo.23135098), which is the one
to cite and the one `fetch` downloads; all versions, resolving to the latest, are under
[10.5281/zenodo.23135097](https://doi.org/10.5281/zenodo.23135097). The scene derives from
OpenStreetMap data, (c) OpenStreetMap contributors.

`linkgym.datasets.fetch` downloads a file once, verifies its SHA-256 and returns its path:

```python
import gymnasium as gym

from linkgym.datasets import fetch

path = fetch("munich-v1")  # 67 MB, cached in ~/.cache/linkgym (or $LINKGYM_DATA_DIR)
env = gym.make(
    "linkgym/LinkAdaptation-v0", channel="trace", trace_path=str(path), trace_splits=["train"]
)
obs, info = env.reset(seed=0)
print(env.unwrapped.channel_info)  # trajectory, start slot, split, route category, SNR
```

How it was generated, its statistics and its known limitations (the ray tracing does not
converge within the ray budget; refraction is off) are in
[docs/channels.md](https://github.com/Tempip/linkgym/blob/main/docs/channels.md#munich-dataset).

## Bring your own channel or Sionna RT scene

Any channel you can write as linear power gains per trajectory, slot and PRB works: write
it with `write_trace`, label each trajectory with a split, and pass the file to the
environment.

```python
import gymnasium as gym
import numpy as np

from linkgym.channels import write_trace

# |h|^2 for 3 trajectories of 2000 slots and 52 PRBs: here Rayleigh block fading
gain = np.repeat(np.random.default_rng(0).exponential(size=(3, 200, 52)), 10, axis=1)
write_trace(
    "my_channel.h5",
    gain,
    split=["train", "val", "test"],
    carrier_frequency_hz=3.5e9,  # must match the scenario
    subcarrier_spacing_hz=30e3,
    slot_duration_s=0.5e-3,
)
env = gym.make("linkgym/LinkAdaptation-v0", channel="trace", trace_path="my_channel.h5")
```

For ray-traced traces, the `linkgym-traces` command (from the `rt` extra) computes them
with Sionna RT from a JSON configuration: a scene (a built-in one or your own Mitsuba XML),
a transmitter and routes split by street. The
[tutorial](https://github.com/Tempip/linkgym/blob/main/docs/tutorial_rt.md) goes from a
scene to a trained and evaluated agent, step by step;
[docs/channels.md](https://github.com/Tempip/linkgym/blob/main/docs/channels.md) documents
the trace format, the SNR modes and the generator.

## Building blocks for your own environment

Two modules can be used outside `LinkAdaptation-v0`, for example in a multi-user or
beam-management environment of your own:

- **`linkgym.phy`** (stable): the link-level functions the environment's simulator uses.
  `effective_sinr` gives the EESM effective SINR. `transmit` gives the ACK, delivered bits
  and TBLER of a transport block, from uniforms you provide, so different policies face the
  same draws. See [docs/phy.md](https://github.com/Tempip/linkgym/blob/main/docs/phy.md).
- **`linkgym.beams`** (experimental): DFT codebooks for linear and planar arrays, steering
  vectors, beam gain, RSRP and best beam, consistent with Sionna RT's arrays. See
  [docs/beams.md](https://github.com/Tempip/linkgym/blob/main/docs/beams.md).

Which parts of linkgym are stable, and what a patch or minor release may change, is set
out in [docs/api_stability.md](https://github.com/Tempip/linkgym/blob/main/docs/api_stability.md).

## Environment at a glance

| | `linkgym/LinkAdaptation-v0` |
|---|---|
| Action | `Discrete(26)`: MCS 3-28 of the 5G NR PDSCH MCS table 1 |
| Observation | `Box(-1, 1, (12,))`: the last 4 reports (wideband SINR, HARQ ACK/NACK, MCS used), delayed by the feedback delay |
| Reward | delivered bits of the slot divided by the TB size of MCS 28, in [0, 1] |
| Episode | 1000 slots, then `truncated=True`; never terminated |
| Channel | `channel="tdl"` (default): 3GPP TDL fading generated with Sionna; `channel="trace"`: an HDF5 trace, a trajectory and window drawn per episode from `trace_splits` |
| Default scenario | single cell, single SISO link, 3GPP TDL-A, 100 ns delay spread, 3.5 GHz, 15 m/s, 52 PRB at 30 kHz, mean SNR drawn per episode from U(5, 20) dB, feedback delay 1 slot |

Every scenario field can be passed to `gym.make`. Timing, observation layout and info keys
are in [docs/environment.md](https://github.com/Tempip/linkgym/blob/main/docs/environment.md),
the channel options in
[docs/channels.md](https://github.com/Tempip/linkgym/blob/main/docs/channels.md).

## Reproducing the results

TDL (v0.1), in Bash:

```bash
pip install -e ".[train]"
for gamma in 0 0.9; do
  for seed in 0 1 2; do
    python examples/train_ppo.py --gamma $gamma --seed $seed
  done
done
python examples/evaluate_all.py
```

PowerShell:

```powershell
pip install -e ".[train]"
foreach ($gamma in "0", "0.9") {
  foreach ($seed in 0, 1, 2) {
    python examples/train_ppo.py --gamma $gamma --seed $seed
  }
}
python examples/evaluate_all.py
```

Munich (v0.2), after the TDL runs above (the evaluation also uses the gamma = 0 TDL
models). The first line downloads both dataset files to `data/`. In Bash:

```bash
LINKGYM_DATA_DIR=data python -m linkgym.datasets munich-v1 munich-v1-test-alt
for seed in 0 1 2; do
  python examples/train_ppo.py --gamma 0 --seed $seed --channel trace --trace-path data/munich-v1.h5 --eval-freq 100000
done
for phase in validation test q4 report; do python examples/evaluate_v02.py $phase; done
```

PowerShell:

```powershell
$env:LINKGYM_DATA_DIR = "data"; python -m linkgym.datasets munich-v1 munich-v1-test-alt
foreach ($seed in 0, 1, 2) {
  python examples/train_ppo.py --gamma 0 --seed $seed --channel trace --trace-path data/munich-v1.h5 --eval-freq 100000
}
foreach ($phase in "validation", "test", "q4", "report") { python examples/evaluate_v02.py $phase }
```

Measured on an AMD Ryzen 7 7700X (8 cores, 16 threads), Windows 11, CPU only (torch
2.14.0+cpu, sionna-no-rt 2.1.0):

- TDL: each training run, 1,015,808 steps with 8 `SubprocVecEnv` workers, took 654-657 s;
  the six runs took 67 minutes one after another. `evaluate_all.py`, on 8 worker processes:
  42 s for the OLLA tuning grid on the validation seeds, then 535 s for all test
  evaluations. It writes the tables and figures to `docs/results/m3/`.
- Munich: each training run took 1169-1201 s, including 12 validation passes. The
  `evaluate_v02.py` phases took 347 s (validation), 583 s (test) and 103 s (q4) on 8 worker
  processes; `report` writes the tables and figures to `docs/results/v02/`. The test phase
  refuses to run again once its results exist.

Numbers can differ slightly on other hardware, operating systems or library versions; the
committed CSVs are the reference.

## Limitations

- A simplified link-level simulator: a single SISO link, a SINR-to-BLER abstraction with
  coarse BLER tables, no HARQ retransmissions, no interference, no channel estimation error.
- Few channels: one TDL scenario, and one ray-traced site with one base station, where the
  test split has four streets (one line-of-sight, one transition, two NLoS). The Munich
  results describe these streets, not sites in general.
- The ray tracing does not converge within the ray budget: the dataset is one deterministic
  realization, with refraction off. All Munich verdicts were the same on a second
  realization, which is one other ray budget, not an independent channel.
- The default SNR normalization divides each trajectory by its mean linear gain, which puts
  the weak stretches of trajectories with a large dynamic range into outage (9% of the
  Munich test slots); these slots raise every policy's observed TBLER.
- PPO uses Stable-Baselines3 default hyperparameters, 1M steps and 3 training seeds; the
  confidence intervals do not include training-seed variance.
- OLLA and PPO optimise different objectives: OLLA holds a TBLER target, PPO maximises
  goodput; the comparison is in goodput, with TBLER reported alongside.
- The oracle knows the current slot's channel and is not achievable in practice.
- A policy learned in this simulator adapts to its abstraction; the results may not transfer
  to real systems.

Full lists:
[TDL results](https://github.com/Tempip/linkgym/blob/main/docs/results/m3/README.md#limitations),
[Munich results](https://github.com/Tempip/linkgym/blob/main/docs/results/v02/README.md#interpretation-and-limitations),
[Munich traces](https://github.com/Tempip/linkgym/blob/main/docs/channels.md#known-limitations-of-the-munich-traces),
[simulator](https://github.com/Tempip/linkgym/blob/main/docs/benchmarks.md#known-limitations),
[environment](https://github.com/Tempip/linkgym/blob/main/docs/environment.md#known-limitations).

## Roadmap

Multi-antenna channels and beam management, one small release at a time, on top of the
building blocks released in 0.3 (`linkgym.phy`, `linkgym.beams`); the published results
stay unchanged.

- **0.4**: trace format 2, which stores the ray-traced paths at the anchors, so that the
  per-antenna channel can be rebuilt for any array and codebook. It comes with the
  generator option to write it, the reconstruction, an `ArrayChannelSource` interface, and
  a Munich 3.5 GHz paths dataset on Zenodo. The element pattern and polarization are fixed
  at generation and recorded in the attributes.
- **0.5**: a cache of beam gains for a chosen array and codebook, and sampling and
  evaluation of several simultaneous users from a trace.
- **0.6** (optional): an FR2 (28 GHz) dataset and/or a statistical multi-antenna channel
  source based on 3GPP CDL.

Other planned work:

- A median-based normalization option for trace channels: the mean linear gain used now
  lets a short strong stretch push the rest of a trajectory into outage
  ([channels](https://github.com/Tempip/linkgym/blob/main/docs/channels.md#snr-modes)).
- Include the agent's own pending actions in the observation when the feedback delay is
  larger than one slot.
- A variant with a continuous SINR-offset action.
- A reward variant with a TBLER constraint.
- Vectorised, batched environments for faster training.

Later:

- Ray-traced datasets of more sites and base stations, to test transfer across sites.
- PDSCH MCS table 2 (256-QAM, BLER data up to 25 dB).

## Contributing

Setup, checks and commit conventions are in
[CONTRIBUTING.md](https://github.com/Tempip/linkgym/blob/main/CONTRIBUTING.md).

## Related work

- [sionna-rl](https://github.com/tobiassugandi/sionna-rl): a reproducible study comparing PPO
  against Sionna's OLLA for single-link MCS selection.
- M. Tsampazi, N. N. Santhi, N. Perrotta, F. Dressler, T. Melodia, "ARIADNE: AI-RAN Informed
  Link Adaptation in Digital Twin Network Environments", European Wireless 2026.
  [arXiv:2605.29772](https://arxiv.org/abs/2605.29772). RL (PPO) link adaptation integrated
  with Sionna SYS over ray-traced channels, compared against OLLA and SALAD.

linkgym differs from these in that it is a reusable, pip-installable, registered Gymnasium
environment rather than a single study.

## Citation

If you use linkgym, please cite it:

```bibtex
@software{rodrigues2026linkgym,
  author  = {Rodrigues Souza, Pedro José},
  title   = {linkgym: A Gymnasium environment for 5G NR link adaptation},
  year    = {2026},
  version = {0.3.0},
  doi     = {10.5281/zenodo.23141597},
  url     = {https://github.com/Tempip/linkgym}
}
```

and, if you use the Munich traces, the dataset version you used:

```bibtex
@dataset{rodrigues2026munich,
  author    = {Rodrigues Souza, Pedro José},
  title     = {linkgym Munich ray-traced channel traces, v1},
  year      = {2026},
  version   = {v1},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23135098}
}
```

GitHub's "Cite this repository" button uses
[CITATION.cff](https://github.com/Tempip/linkgym/blob/main/CITATION.cff).

## License

The code is MIT licensed. See [LICENSE](https://github.com/Tempip/linkgym/blob/main/LICENSE).
The Munich dataset is licensed under the Open Database License (ODbL) 1.0; it derives from
OpenStreetMap data, (c) OpenStreetMap contributors. The same applies to the small sample of
it in `tests/data/munich_sample.h5`, and the route figure `docs/assets/munich_routes.png`
shows OpenStreetMap-derived buildings.

This project is not affiliated with or endorsed by NVIDIA. Sionna is a trademark of NVIDIA
Corporation.
