# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Trace format 2 (`channel_kind = "paths"`), in the new experimental module
  `linkgym.paths`: the ray-traced paths of every anchor (complex coefficient, delay,
  Doppler shift, unit directions of departure and arrival) in a compressed-row layout,
  with the per-trajectory data of format 1 and each trajectory's mean single-antenna gain.
  `write_path_trace`, `inspect_path_trace` (strict validation, optionally recomputing the
  mean gains), `read_path_window`, and `reconstruct_cfr`, which rebuilds the channel of
  every antenna of any transmit array (geometry, spacing, orientation) as Sionna RT's
  synthetic arrays do. Format 1 and its API are unchanged.

## [0.3.0] - 2026-10-04

### Added

- **`linkgym.phy`**, a stable link-level API with the functions the environment's
  simulator uses:
  - `effective_sinr` (EESM) and `tb_size_per_mcs`;
  - `transmit(sinr_eff, mcs, u, *, num_allocated_re)`, which returns the ACK, delivered
    bits and TBLER of one transport block per link from uniforms you provide (common
    random numbers);
  - `MIN_MCS` and `MAX_MCS`.

  See `docs/phy.md`.
- **`linkgym.beams`** (experimental): DFT codebooks for linear and planar arrays, with
  oversampling and the direction of each beam; steering vectors; beam gain |w^H h|^2;
  RSRP; and the best beam. They use the element order and phase convention of Sionna RT's
  `PlanarArray` and synthetic arrays. See `docs/beams.md`.
- **`linkgym.channels.tx_power_for_median_snr`**: the transmit power that puts a trace's
  median per-slot wideband SNR at a target. With it, `snr_mode="link_budget"` stays within
  the range of the BLER tables while keeping the real power differences between
  trajectories. Compute it on the train split.
- **An API stability policy** (`docs/api_stability.md`): which parts of linkgym are
  stable, experimental or internal, and what patch and minor releases may change. Each
  module's docstring states its status, and a test checks the public names of the main
  modules.

### Changed

- The simulator's link-level code moved to `linkgym.phy`; `linkgym.sim` keeps its names.
  The outputs of the environment and the simulator are unchanged, checked by golden tests
  that now also cover the trace channel path.
- `linkgym.channels`, `linkgym.evaluation` and `linkgym.datasets` define `__all__`, so
  `from ... import *` imports only their public names.
- README: the new modules, the stability policy, and the roadmap to 0.6 (a multi-antenna
  trace format with a Munich paths dataset, beam-gain caches and multi-user sampling, an
  FR2 dataset).

## [0.2.0] - 2026-10-03

### Added

- **Trace channels.** `channel="trace"` runs the environment on an HDF5 channel trace
  (trace format version 1: the linear power gain per trajectory, slot and PRB, with a
  split label per trajectory). Each episode draws a trajectory of `trace_splits` and a
  window of it. `write_trace`, `read_trace` and `inspect_trace` write, validate and
  inspect traces; a trace is read lazily, one window per episode, and shared by
  subprocess vector environments. Optional route categories (`los`, `transition`,
  `nlos`) allow results by route type.
- **SNR modes** for traces: `normalized` (the default: scenario SNR, each trajectory
  scaled to unit mean gain) and `link_budget` (transmit power, thermal noise and noise
  figure).
- **`linkgym.channels.ChannelSource`**, a stable interface for channel models, used by the
  TDL channel and the traces.
- **Sionna RT trace generator**: the `linkgym-traces` command (`check`, `plot`, `generate`,
  `accuracy`) computes traces from a JSON configuration (scene, transmitter, PRB grid,
  routes split by street, path solver settings), with route checks against the scene
  geometry, deterministic and byte-reproducible output, and provenance attributes. It
  needs the new `rt` extra, to be installed in its own environment.
- **Munich dataset**: ray-traced traces of 15 streets around one rooftop base station in
  Sionna RT's `munich` scene, split by street into train, val and test, with a second
  ray-tracing realization of the test streets. Published on Zenodo under the ODbL 1.0;
  `linkgym.datasets.fetch("munich-v1")` downloads a file once, checks its size and SHA-256
  and caches it (`LINKGYM_DATA_DIR`), also as `python -m linkgym.datasets`.
- **Pinned evaluation episodes**: `env.reset(seed=..., options={"trajectory": i, "offset":
  o})`. `linkgym.evaluation.trace_episodes` lists every trajectory of a split in fixed
  windows, `evaluate_episodes` runs a policy on them and `cluster_bootstrap` gives
  confidence intervals over trajectories.
- `examples/train_ppo.py --channel trace`: PPO on the train split of a trace, with
  validation on pinned val episodes and the trace's SHA-256 recorded.
- **Results on ray-traced streets** (`docs/results/v02/`): PPO trained on TDL and on the
  Munich train streets against a validation-tuned OLLA on held-out streets, under a
  protocol fixed before training, by route category and on both ray-tracing realizations,
  with a post hoc analysis of OLLA's TBLER (`examples/evaluate_v02.py`,
  `examples/olla_val_analysis.py`, `examples/olla_test_traces.py`).
- `LinkAdaptationEnv.channel_info` and `LinkSimulator.channel_info`: trajectory, window,
  split, route category and realized mean SNR of the episode; `close()` on both.
- Documentation: `docs/channels.md` (channel interface, trace format, SNR modes and the
  outage stretches of `normalized`, the generator, the Munich dataset and its known
  limitations, bringing your own channel) and `docs/tutorial_rt.md` (from a Sionna RT
  scene to a trained and evaluated agent, with its CPU steps run by the tests).
- `RELEASING.md`, and `examples/rt/package_zenodo.py` to build the dataset's Zenodo
  package.

### Changed

- README: the results on TDL and on the Munich streets side by side, a new header figure,
  the dataset, and how to bring your own channel or Sionna RT scene.
- `TDLChannelGain.generate` returns a `ChannelEpisode`, and `LinkSimulator` takes an
  optional `channel_source`. The TDL outputs are unchanged, which golden tests now check.
- `linkgym.sim` raises a clearer `ImportError` when Sionna fails to import because
  sionna-rt is installed in the same environment.
- `h5py` is a dependency.
- The publish workflow uses `actions/upload-artifact@v7` and
  `actions/download-artifact@v8`, which run on Node.js 24.

### Removed

- `linkgym.sim.ChannelGainSource`, replaced by `linkgym.channels.ChannelSource`.

## [0.1.0] - 2026-09-30

### Added

- Gymnasium environment `linkgym/LinkAdaptation-v0` (`LinkAdaptationEnv`), registered on
  `import linkgym` and configured by `ScenarioConfig`: 26 MCS actions (PDSCH MCS table 1,
  MCS 3-28), the delayed wideband SINR, HARQ ACK/NACK and MCS reports as observation, a
  normalized goodput reward and a per-episode SNR draw.
- `linkgym.sim`: a single-cell, single-user downlink link simulator on Sionna SYS 2.1 (TDL
  fading, EESM, PHYAbstraction) with seeded, batched links.
- `linkgym.baselines`: fixed-MCS, ILLA, OLLA (with Sionna's `delta_up` step as a parameter)
  and oracle policies, and an adapter for Stable-Baselines3 models.
- `linkgym.evaluate`: goodput, observed TBLER and mean MCS over seeds, with per-episode
  results; `linkgym.evaluation.paired_bootstrap` for paired differences with bootstrap
  confidence intervals.
- `examples/train_ppo.py`: PPO training with Stable-Baselines3 on seeds disjoint from the
  validation and test seeds, learning curves, TensorBoard logs and a `run.json` with
  configuration, versions and git commit.
- `examples/evaluate_all.py`: OLLA tuned on validation seeds, evaluation on held-out test
  seeds, paired comparisons and a fixed-SNR grid. Results of PPO against tuned OLLA, ILLA,
  fixed MCS and the oracle on the default TDL scenario in `docs/results/m3/`.
- `examples/baseline.py`, `examples/run_baselines.py` and `examples/benchmark.py`
  (baselines and speed, `docs/benchmarks.md`), and `examples/quickstart.ipynb`.
- Documentation: README, `docs/environment.md`, `CONTRIBUTING.md` and `CITATION.cff`.
- Optional extras `train` (Stable-Baselines3, TensorBoard), `dev` and `docs`.

[Unreleased]: https://github.com/Tempip/linkgym/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Tempip/linkgym/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Tempip/linkgym/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Tempip/linkgym/releases/tag/v0.1.0
