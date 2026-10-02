# linkgym

[![CI](https://github.com/Tempip/linkgym/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Tempip/linkgym/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Tempip/linkgym/blob/main/LICENSE)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://github.com/Tempip/linkgym/blob/main/pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/linkgym.svg)](https://pypi.org/project/linkgym/)

linkgym is a Gymnasium environment for 5G NR link adaptation (MCS selection) built on NVIDIA
Sionna SYS, with classical baselines and a fixed evaluation protocol.

![Goodput vs mean SNR for PPO, tuned OLLA and the oracle](https://raw.githubusercontent.com/Tempip/linkgym/main/docs/assets/header.png)

*Goodput vs mean SNR on held-out test seeds. PPO (gamma = 0, mean of 3 training seeds)
against Sionna's OLLA tuned on validation seeds and an oracle that knows the channel of the
slot being decided. Details in the
[M3 results](https://github.com/Tempip/linkgym/blob/main/docs/results/m3/README.md).*

In the default scenario, PPO trained on linkgym delivers +16.0% goodput over a
validation-tuned OLLA (95% CI [+14.9%, +17.1%], paired over 50 held-out episodes) at a lower
TBLER.

## Why

Recent RL link-adaptation studies on Sionna (see Related work) each build their own
environment. linkgym is a reusable, pip-installable Gymnasium environment,
registered as `linkgym/LinkAdaptation-v0`, with built-in baselines (fixed MCS, ILLA, OLLA,
oracle) and a fixed evaluation protocol. All policies are evaluated with common random
numbers: for a given seed, every policy faces the same channel and the same ACK draws.

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

## Environment at a glance

| | `linkgym/LinkAdaptation-v0` |
|---|---|
| Action | `Discrete(26)`: MCS 3-28 of the 5G NR PDSCH MCS table 1 |
| Observation | `Box(-1, 1, (12,))`: the last 4 reports (wideband SINR, HARQ ACK/NACK, MCS used), delayed by the feedback delay |
| Reward | delivered bits of the slot divided by the TB size of MCS 28, in [0, 1] |
| Episode | 1000 slots, then `truncated=True`; never terminated |
| Default scenario | single cell, single SISO link, 3GPP TDL-A, 100 ns delay spread, 3.5 GHz, 15 m/s, 52 PRB at 30 kHz, mean SNR drawn per episode from U(5, 20) dB, feedback delay 1 slot |

Every scenario field can be passed to `gym.make`. Timing, observation layout and info keys
are in [docs/environment.md](https://github.com/Tempip/linkgym/blob/main/docs/environment.md).

## Reproducing the results

Bash:

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

Measured on an AMD Ryzen 7 7700X (8 cores, 16 threads), Windows 11, CPU only (torch
2.14.0+cpu, sionna-no-rt 2.1.0):

- Each training run: 1,015,808 steps with 8 `SubprocVecEnv` workers, 654-657 s; the six runs
  took 67 minutes one after another.
- `evaluate_all.py`, on 8 worker processes: 42 s for the OLLA tuning grid on the validation
  seeds, then 535 s for all test evaluations. It writes the tables and figures to
  `docs/results/m3/`.

Numbers can differ slightly on other hardware, operating systems or library versions; the
committed CSVs are the reference.

## Limitations

- A simplified link-level simulator: a single SISO link, a SINR-to-BLER abstraction with
  coarse BLER tables, no HARQ retransmissions, no interference, no channel estimation error.
- One scenario: results for other channels, speeds or feedback delays are not known.
- PPO uses Stable-Baselines3 default hyperparameters, 1M steps and 3 training seeds.
- OLLA and PPO optimise different objectives: OLLA holds a TBLER target, PPO maximises
  goodput; the comparison is in goodput, with TBLER reported alongside.
- The oracle knows the current slot's channel and is not achievable in practice.
- A policy learned in this simulator adapts to its abstraction; the results may not transfer
  to real systems.

Full lists:
[results](https://github.com/Tempip/linkgym/blob/main/docs/results/m3/README.md#limitations),
[simulator](https://github.com/Tempip/linkgym/blob/main/docs/benchmarks.md#known-limitations),
[environment](https://github.com/Tempip/linkgym/blob/main/docs/environment.md#known-limitations).

## Roadmap (v0.2, not started)

- Include the agent's own pending actions in the observation when the feedback delay is
  larger than one slot.
- A variant with a continuous SINR-offset action.
- A reward variant with a TBLER constraint.
- Vectorised, batched environments for faster training.

Later:

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
  author  = {Rodrigues, Pedro},
  title   = {linkgym: A Gymnasium environment for 5G NR link adaptation},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Tempip/linkgym}
}
```

GitHub's "Cite this repository" button uses
[CITATION.cff](https://github.com/Tempip/linkgym/blob/main/CITATION.cff).

## License

MIT. See [LICENSE](https://github.com/Tempip/linkgym/blob/main/LICENSE).

This project is not affiliated with or endorsed by NVIDIA. Sionna is a trademark of NVIDIA
Corporation.
