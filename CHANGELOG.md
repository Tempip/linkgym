# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `linkgym.channels`: the `ChannelSource` interface and `ChannelEpisode`, a stable
  extension point for channel models (linear power gain per slot and PRB, optional
  reference SNR).
- Trace-based channels: `TraceChannelSource` reads HDF5 trace files (format version 1,
  `write_trace`/`read_trace`) with validation against the scenario, per-trajectory split
  labels and seeded selection of trajectory and start slot.
- SNR modes `normalized` (scenario SNR, unit mean gain per trajectory) and `link_budget`
  (transmit power, thermal noise and noise figure, `link_budget_snr_db`).
- Scenario fields `channel`, `trace_path`, `trace_splits`, `snr_mode`, `tx_power_dbm` and
  `noise_figure_db`; the defaults keep the v0.1 TDL scenario.
- `LinkAdaptationEnv.channel_info` and `LinkSimulator.channel_info`: trajectory, start
  slot, split and realized mean SNR of the episode.
- `docs/channels.md`: interface, trace format, SNR modes, limitations and a tested
  bring-your-own-channel example.
- Golden tests fixing the outputs of the TDL path (simulator, environment, tuned OLLA).
- `h5py` as an explicit dependency.
- Lazy trace loading: `TraceChannelSource` validates the file once without keeping its
  gains in memory, reads only each episode's window and opens one read-only handle per
  process, so subprocess vector environments share the file; `inspect_trace`,
  `LinkAdaptationEnv.close` and `LinkSimulator.close`.
- Optional trace datasets `los` (line-of-sight flag per path solve) and `category` (route
  category `los`, `nlos` or `transition`, also in `channel_info`).
- `linkgym.rt` and the `linkgym-traces` command (`generate`, `check`, `accuracy`, `plot`):
  trace generation with Sionna RT from a JSON configuration (scene, transmitter, PRB grid,
  routes with split, group and category, solver settings), with route checks against the
  scene geometry, deterministic path solves at anchors with Doppler evolution in between,
  dropping of trajectories without paths or below `min_mean_gain_db`, provenance
  attributes, and `--routes`, `--splits` and `--solver KEY=VALUE` selections and overrides.
  Optional extra `rt` (`sionna-rt==2.1.0`), to be installed in its own environment.
- Munich dataset configuration (`examples/rt/munich.json`): one rooftop base station and
  15 streets split by street into train, val and test (each with line-of-sight, NLoS and
  transition data), refraction off; the route figure
  `docs/assets/munich_routes.png`, a small sample trace in `tests/data/`, and
  `examples/rt/make_sample.py` and `examples/rt/dataset_stats.py`. The dataset itself is
  generated locally; `docs/channels.md` documents its statistics and known limitations
  (ray-tracing non-convergence, dropped trajectories, an alternative test realization).
- A clearer `ImportError` from `linkgym.sim` when Sionna fails to import because
  sionna-rt is installed in the same environment.
- Pinned trace episodes: `env.reset(seed, options={"trajectory": i, "offset": o})`, with
  `TraceChannelSource.generate(..., trajectories=, offsets=)` and
  `LinkSimulator.reset(..., channel_options=)`; `linkgym.evaluation.trace_episodes`,
  `evaluate_episodes` and `cluster_bootstrap` for enumerated evaluation.
- `examples/train_ppo.py --channel trace`: training on the train split of a trace with
  validation on pinned val episodes; the run records the dataset's SHA-256.
- The v0.2 Munich experiment: protocol (`docs/results/v02/PROTOCOL.md`),
  `examples/evaluate_v02.py` and the results (`docs/results/v02/README.md`): PPO trained on
  the Munich train streets against tuned OLLA, PPO trained on TDL and the other baselines on
  the held-out test streets, per route category and on both ray-tracing realizations.

### Changed

- `TDLChannelGain.generate` returns a `ChannelEpisode`; `LinkSimulator` takes an optional
  `channel_source`. The TDL outputs are unchanged.

### Removed

- `linkgym.sim.ChannelGainSource`, replaced by `linkgym.channels.ChannelSource`.

## [0.1.0] - 2026-09-30

### Added

- Initial project skeleton: packaging, license, CI, pre-commit and an import test.
- Project URLs in `pyproject.toml`, CI and license badges in the README, and `.gitattributes`
  enforcing LF line endings.
- `linkgym.sim`: single-cell, single-user downlink link simulator on Sionna SYS 2.1
  (TDL fading, EESM, PHYAbstraction) with seeded, batched links and oracle, ILLA, OLLA and
  fixed-MCS policies.
- `examples/baseline.py` and `examples/benchmark.py` for link adaptation baselines and CPU
  speed measurements; results in `docs/benchmarks.md`.
- Tests for reproducibility, instance isolation, Sionna RNG side effects and equivalence with
  `PHYAbstraction`'s ACK and delivered bits.
- Gymnasium environment `linkgym/LinkAdaptation-v0` (`LinkAdaptationEnv`), registered on
  `import linkgym` and configured by `ScenarioConfig`: 26 MCS actions, delayed HARQ/SINR/MCS
  reports as observation, normalized goodput reward, per-episode SNR draw.
- `linkgym.baselines`: fixed MCS, ILLA, OLLA and oracle policies acting through the
  environment, and an adapter for Stable-Baselines3 models.
- `linkgym.evaluate` for goodput, observed TBLER and mean MCS across seeds.
- `examples/run_baselines.py`, an environment speed section in `examples/benchmark.py`,
  and `docs/environment.md`.
- `examples/train_ppo.py`: PPO training on `LinkAdaptation-v0` with `SubprocVecEnv`, seeds
  disjoint from validation (500-509) and test (1000+) seeds, a validation callback for
  learning curves, TensorBoard logs and a `run.json` with config, versions and git commit.
- `examples/evaluate_all.py`: M3 evaluation protocol with selection on validation seeds,
  held-out test seeds, paired bootstrap comparisons against OLLA and a fixed-SNR grid.
- `evaluate` returns per-episode results; `linkgym.evaluation.paired_bootstrap` for paired
  differences with a percentile bootstrap confidence interval.
- `slow` pytest marker; CI runs `pytest -m "not slow"`.
- `docs/results/m3/`: M3 PPO results (gamma 0 and 0.9, 3 training seeds each) against OLLA,
  ILLA, fixed MCS and the oracle, with tables, figures, interpretation and limitations.
- `OLLAPolicy` takes a `delta_up` argument [dB], passed to Sionna's
  `OuterLoopLinkAdaptation` (default 1.0, Sionna's).
- Tuned OLLA baseline in `examples/evaluate_all.py`: TBLER target x `delta_up` grid on the
  validation seeds, the best cell evaluated on the test seeds, in the fixed-SNR grid and in
  the paired comparison with PPO. The M3 headline is now PPO vs tuned OLLA
  (`docs/results/m3/olla_tuning.csv`).
- `examples/quickstart.ipynb`: environment, rendering, an OLLA episode trace and an
  evaluation of OLLA against a random policy.
- `CONTRIBUTING.md` and `CITATION.cff`.
- `docs/assets/header.png` (README figure) and `docs/assets/social_preview.png`
  (1280 x 640), built from the M3 results by `docs/assets/make_assets.py`.
- Tests that run the README code blocks verbatim and, marked slow, the quickstart notebook.

### Changed

- README rewritten: header figure, motivation, installation, quickstart, results, environment
  summary, reproduction commands with measured times, limitations, related work, citation.
- `pyproject.toml`: singular description, keywords, classifiers, Documentation and
  Changelog URLs for the PyPI page, an explicit sdist file list, and a `docs` extra
  (`nbclient`, `ipykernel`) for the notebook test.
- `LinkSimulator.reset` takes an optional `snr_db`; the channel source now returns |h|^2 and
  the simulator applies the SNR. M1 results are bitwise unchanged.
- `LinkResult` has a new `ack_uniform` field with the uniform draw behind each ACK.
- CI installs the `train` extra so the Stable-Baselines3 environment check runs.
- Pinned ruff to 0.16.9 in the `dev` extra to match the pre-commit hook version.
- Set the author name in `pyproject.toml` to Pedro Rodrigues.
- `tensorboard` added to the `train` extra.
- `.gitignore` ignores `runs/` and `results/` only at the repository root, so
  `docs/results/` can be committed.
- `docs/benchmarks.md` describes ILLA as ILLA without outer loop, fed the raw wideband SINR
  report, whose high TBLER is the bias OLLA corrects.
- `examples/evaluate_all.py` colors the PPO models by gamma in the goodput-vs-TBLER figure
  instead of labeling each point.

### Fixed

- `LinkAdaptationEnv` no longer raises on an unsupported `render_mode`; it warns, as
  `gymnasium.make` does. Stable-Baselines3's `make_vec_env` passes
  `render_mode="rgb_array"` by default.

[Unreleased]: https://github.com/Tempip/linkgym/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Tempip/linkgym/releases/tag/v0.1.0
