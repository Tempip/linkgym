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

### Changed

- `LinkSimulator.reset` takes an optional `snr_db`; the channel source now returns |h|^2 and
  the simulator applies the SNR. M1 results are bitwise unchanged.
- `LinkResult` has a new `ack_uniform` field with the uniform draw behind each ACK.
- CI installs the `train` extra so the Stable-Baselines3 environment check runs.
- Pinned ruff to 0.16.9 in the `dev` extra to match the pre-commit hook version.
- Set the author name in `pyproject.toml` to Pedro Rodrigues.
