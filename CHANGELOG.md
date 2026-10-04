# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Golden tests for the trace channel path: `TraceChannelSource` episodes (normalized and
  link budget, random and pinned), a full environment episode with OLLA on the sample
  trace, and the bytes written by `write_trace`, recorded with 0.2.0.
- `linkgym.phy`, a stable public link-level API: `tb_size_per_mcs`, `effective_sinr`
  (EESM), `transmit(sinr_eff, mcs, u, *, num_allocated_re)` returning ACK, delivered bits
  and TBLER from uniforms the caller provides, and `MIN_MCS`/`MAX_MCS`. The simulator now
  uses these functions; its outputs are unchanged (golden tests). `docs/phy.md`.
- `linkgym.beams` (experimental): `dft_codebook` for ULA and UPA arrays with oversampling
  (Sionna PHY's grids of beams, with the direction of each beam), `steering_vector`,
  `beam_gain` (|w^H h|^2), `rsrp_dbm` and `best_beam`, with the element order and phase
  convention of Sionna RT's `PlanarArray` and synthetic arrays. `docs/beams.md`.

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

[Unreleased]: https://github.com/Tempip/linkgym/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Tempip/linkgym/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Tempip/linkgym/releases/tag/v0.1.0
