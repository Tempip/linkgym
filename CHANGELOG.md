# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
